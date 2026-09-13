from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Protocol
from uuid import uuid4

from .core import DeterministicFinancialCore
from .data_loader import DatasetBundle
from .decision import DecisionEngine
from .evidence.cache import EvidenceCache
from .evidence.service import EvidenceModelClient, EvidenceService
from .evidence.usage import UsageTracker
from .evaluation.submission_validator import REQUIRED_COLUMNS, assert_valid_submission
from .evaluation.usage_report import render_usage_report
from .planning import DecisionResult, format_plan_amount, validate_decision


class DecisionRunner(Protocol):
    def decide(self, request) -> DecisionResult: ...


EngineFactory = Callable[[DatasetBundle, EvidenceService], DecisionRunner]


def _default_engine(data: DatasetBundle, evidence: EvidenceService) -> DecisionRunner:
    return DecisionEngine(DeterministicFinancialCore(data, evidence=evidence))


def render_output_csv(decisions: list[DecisionResult] | tuple[DecisionResult, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in decisions:
        writer.writerow({
            "request_id": result.request_id,
            "amount_safe_to_pay": format_plan_amount(result.amount_safe_to_pay),
            "affordability_status": result.affordability_status,
            "recommended_payment_method": result.recommended_payment_method,
            "payment_plan": result.payment_plan,
            "earliest_date_for_full_payment": (
                result.earliest_date_for_full_payment.isoformat()
                if result.earliest_date_for_full_payment else ""
            ),
            "spending_changes_needed": result.spending_changes_needed,
            "decision_explanation": result.decision_explanation,
        })
    return buffer.getvalue()


def _stage_text(target: Path, content: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _replace_pair(staged: tuple[tuple[Path, Path], ...]) -> None:
    targets = [target for _, target in staged]
    if len({target.resolve() for target in targets}) != len(targets):
        raise ValueError("artifact target paths must be distinct")
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    try:
        for _, target in staged:
            if target.exists():
                backup = target.with_name(f".{target.name}.{uuid4().hex}.bak")
                os.replace(target, backup)
                backups[target] = backup
        for temporary, target in staged:
            os.replace(temporary, target)
            installed.add(target)
    except Exception as install_error:
        rollback_errors: list[str] = []
        for target in targets:
            backup = backups.get(target)
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                elif target in installed:
                    target.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "artifact installation failed and rollback was incomplete; retained backup files: "
                + "; ".join(rollback_errors)
            ) from install_error
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def publish_submission_artifacts(
    data: DatasetBundle,
    decisions: list[DecisionResult] | tuple[DecisionResult, ...],
    usage: UsageTracker,
    output_path: Path,
    report_path: Path,
) -> None:
    output_text = render_output_csv(decisions)
    output_digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    report_text = render_usage_report(usage, len(data.requests), output_digest=output_digest)
    staged_output = _stage_text(output_path, output_text)
    staged_report: Path | None = None
    try:
        assert_valid_submission(staged_output, data)
        staged_report = _stage_text(report_path, report_text)
        if not staged_report.read_text(encoding="utf-8").strip():
            raise ValueError("generated usage report is empty")
        _replace_pair(((staged_output, output_path), (staged_report, report_path)))
    finally:
        staged_output.unlink(missing_ok=True)
        if staged_report is not None:
            staged_report.unlink(missing_ok=True)


@contextmanager
def fresh_evidence_cache(requested_root: Path | None) -> Iterator[Path]:
    if requested_root is None:
        with tempfile.TemporaryDirectory(prefix="buy-or-wait-final-evidence-") as directory:
            yield Path(directory)
        return
    root = requested_root.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"final evidence cache must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    yield root


def run_submission(
    data: DatasetBundle,
    output_path: Path,
    report_path: Path,
    *,
    cache_root: Path | None = None,
    client: EvidenceModelClient | None = None,
    enable_environment_client: bool = True,
    engine_factory: EngineFactory = _default_engine,
) -> UsageTracker:
    usage = UsageTracker()
    with fresh_evidence_cache(cache_root) as isolated_cache:
        evidence = EvidenceService(
            data, client=client, usage=usage, cache=EvidenceCache(isolated_cache),
            enable_environment_client=enable_environment_client,
        )
        if enable_environment_client and client is None and evidence.client is None:
            raise RuntimeError("OPENAI_API_KEY is required for the final evidence-enabled run")
        engine = engine_factory(data, evidence)
        decisions: list[DecisionResult] = []
        for request in data.requests.values():
            try:
                result = engine.decide(request)
                errors = validate_decision(result, request)
                if errors:
                    raise ValueError("; ".join(errors))
            except Exception as exc:
                raise RuntimeError(f"submission failed for {request.request_id}: {exc}") from exc
            decisions.append(result)
        publish_submission_artifacts(data, decisions, usage, output_path, report_path)
    return usage
