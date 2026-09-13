from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from code.config import ForecastPolicy, inclusive_horizon_end
from code.data_loader import DatasetBundle
from code.evaluation.package_check import package_manifest, validate_package_manifest
from code.evaluation.submission_validator import REQUIRED_COLUMNS, _supported_change_ids, validate_submission
from code.evidence.usage import UsageTracker
from code.models import FinancialEvent, PaymentOption, Request, UserProfile
from code.planning import CandidatePlan, DecisionResult, Payment
from code.submission import (
    fresh_evidence_cache,
    publish_submission_artifacts,
    render_output_csv,
    run_submission,
)


START = date(2026, 9, 1)


def make_request(request_id: str = "request_1") -> Request:
    return Request(
        request_id, "user", START, "purchase", Decimal("100"),
        date(2026, 11, 30), True, "Can I afford this?",
    )


def make_profile() -> UserProfile:
    return UserProfile(
        "user", "USD", Decimal("500"), Decimal("100"), (), ("rent",),
        ("dining", "rent"), ("dining", "rent"),
        ("full_payment", "partial_payment", "installments"), 3,
    )


def make_event(event_id: str, month: int, category: str, *, protected: bool = False) -> FinancialEvent:
    return FinancialEvent(
        event_id, "user", "expense", "Rent" if protected else "Dining plan",
        category, "debit", Decimal("40"), "USD", date(2026, month, 5),
        date(2026, month, 5), "settled", None, "reducible_or_stoppable", Decimal("20"),
    )


def make_bundle(*request_ids: str) -> DatasetBundle:
    request_ids = request_ids or ("request_1",)
    requests = {request_id: make_request(request_id) for request_id in request_ids}
    dining = tuple(make_event(f"d{index}", month, "dining") for index, month in enumerate((6, 7, 8), 1))
    rent = tuple(make_event(f"p{index}", month, "rent", protected=True) for index, month in enumerate((6, 7, 8), 1))
    events = dining + rent
    options = {
        request_id: (PaymentOption(
            f"option_{request_id}", request_id, "installments", Decimal("50"), 2,
            date(2026, 9, 2), 30, Decimal("0"), Decimal("100"),
        ),)
        for request_id in request_ids
    }
    return DatasetBundle(
        Path("."), {"user": make_profile()}, requests, {}, {"user": events},
        {event.event_id: event for event in events}, options, {}, {}, (),
    )


def full_decision(request_id: str = "request_1") -> DecisionResult:
    candidate = CandidatePlan(
        "full_payment", (Payment(START, Decimal("100")),), Decimal("100"),
    )
    return DecisionResult(
        request_id, Decimal("100"), "affordable_now", "full_payment",
        "2026-09-01:100", START, "none", "Safe full payment.", candidate,
    )


def row_for(**updates: str) -> dict[str, str]:
    row = {
        "request_id": "request_1",
        "amount_safe_to_pay": "100",
        "affordability_status": "affordable_now",
        "recommended_payment_method": "full_payment",
        "payment_plan": "2026-09-01:100",
        "earliest_date_for_full_payment": "2026-09-01",
        "spending_changes_needed": "none",
        "decision_explanation": "Safe full payment.",
    }
    row.update(updates)
    return row


def write_rows(path: Path, rows: list[dict[str, str]], columns=REQUIRED_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SubmissionValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "output.csv"
        self.data = make_bundle()

    def tearDown(self):
        self.temporary.cleanup()

    def errors(self, row: dict[str, str]) -> tuple[str, ...]:
        write_rows(self.path, [row])
        return validate_submission(self.path, self.data)

    def test_valid_complete_synthetic_output(self):
        self.path.write_text(render_output_csv([full_decision()]), encoding="utf-8")
        self.assertEqual((), validate_submission(self.path, self.data))

    @patch("code.evaluation.submission_validator.RecurrenceDetector.infer", return_value=[])
    def test_spending_change_support_uses_shared_inclusive_horizon(self, mocked_infer):
        request = self.data.requests["request_1"]
        self.assertEqual(set(), _supported_change_ids(self.data, request))
        expected_end = inclusive_horizon_end(request.request_date, ForecastPolicy().horizon_days)
        self.assertEqual(expected_end, mocked_infer.call_args.args[3])

    def test_missing_request(self):
        data = make_bundle("request_1", "request_2")
        write_rows(self.path, [row_for()])
        self.assertTrue(any("request_2: missing prediction" in error for error in validate_submission(self.path, data)))

    def test_duplicate_request(self):
        write_rows(self.path, [row_for(), row_for()])
        self.assertTrue(any("duplicate prediction" in error for error in validate_submission(self.path, self.data)))

    def test_unexpected_request(self):
        write_rows(self.path, [row_for(), row_for(request_id="unexpected")])
        self.assertTrue(any("unexpected request_id" in error for error in validate_submission(self.path, self.data)))

    def test_invalid_column_order(self):
        columns = list(REQUIRED_COLUMNS)
        columns[0], columns[1] = columns[1], columns[0]
        write_rows(self.path, [row_for()], columns)
        self.assertTrue(any("columns must exactly match" in error for error in validate_submission(self.path, self.data)))

    def test_invalid_enum(self):
        self.assertTrue(any("invalid affordability_status" in error for error in self.errors(row_for(affordability_status="maybe"))))

    def test_safe_amount_bounds_and_nonfinite_values(self):
        for value, expected in (
            ("-1", "outside"), ("101", "outside"), ("not-a-number", "valid Decimal"),
            ("NaN", "finite"), ("Infinity", "finite"),
        ):
            with self.subTest(value=value):
                self.assertTrue(any(expected in error for error in self.errors(row_for(amount_safe_to_pay=value))))

    def test_malformed_and_nonchronological_payment_plans(self):
        malformed = self.errors(row_for(payment_plan="not-a-plan"))
        self.assertTrue(any("malformed payment_plan" in error for error in malformed))
        nonchronological = self.errors(row_for(
            recommended_payment_method="partial_payment", affordability_status="affordable_with_plan",
            amount_safe_to_pay="40", earliest_date_for_full_payment="2026-09-15",
            payment_plan="2026-09-15:60|2026-09-01:40",
        ))
        self.assertTrue(any("not chronological" in error for error in nonchronological))

    def test_invalid_installment_schedule_and_total(self):
        base = dict(
            recommended_payment_method="installments", affordability_status="affordable_with_plan",
            earliest_date_for_full_payment="", payment_plan="2026-09-03:50|2026-10-03:50",
        )
        self.assertTrue(any("does not exactly match" in error for error in self.errors(row_for(**base))))
        base["payment_plan"] = "2026-09-02:40|2026-10-02:40"
        self.assertTrue(any("does not exactly match" in error for error in self.errors(row_for(**base))))

    def test_invalid_partial_shape_and_sum(self):
        base = dict(
            recommended_payment_method="partial_payment", affordability_status="affordable_with_plan",
            amount_safe_to_pay="40", earliest_date_for_full_payment="2026-10-15",
        )
        self.assertTrue(any("exactly two" in error for error in self.errors(row_for(
            **base, payment_plan="2026-09-01:40",
        ))))
        self.assertTrue(any("sum to requested_amount" in error for error in self.errors(row_for(
            **base, payment_plan="2026-09-01:40|2026-10-15:50",
        ))))

    def test_spending_change_references_and_permissions(self):
        base = dict(
            amount_safe_to_pay="90", affordability_status="affordable_with_plan",
            spending_changes_needed="stop:missing",
        )
        self.assertTrue(any("unknown event" in error for error in self.errors(row_for(**base))))
        base["spending_changes_needed"] = "stop:p3"
        self.assertTrue(any("protected event" in error for error in self.errors(row_for(**base))))
        base["spending_changes_needed"] = "reduce_to:d3:10"
        self.assertTrue(any("below its minimum" in error for error in self.errors(row_for(**base))))
        base["spending_changes_needed"] = "stop:d3|reduce_to:d3:20"
        self.assertTrue(any("conflict or duplicate" in error for error in self.errors(row_for(**base))))


class FakeEngine:
    def __init__(self, result: DecisionResult | Exception):
        self.result = result

    def decide(self, _request):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SubmissionArtifactTests(unittest.TestCase):
    def test_no_network_cli_refuses_official_artifact_paths(self):
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(repo_root / "code" / "main.py"), "--no-network"],
            cwd=repo_root, text=True, capture_output=True,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("explicit non-official", result.stderr)

    def test_default_final_cache_is_fresh_temporary_and_not_development_cache(self):
        with fresh_evidence_cache(None) as cache_root:
            self.assertTrue(cache_root.is_dir())
            self.assertFalse(any(cache_root.iterdir()))
            self.assertNotIn("code/.cache", cache_root.as_posix())
            temporary_root = cache_root
        self.assertFalse(temporary_root.exists())

    def test_explicit_final_cache_must_be_empty_and_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "final-cache"
            cache_root.mkdir()
            marker = cache_root / "existing.json"
            marker.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                with fresh_evidence_cache(cache_root):
                    pass
            self.assertEqual("{}", marker.read_text(encoding="utf-8"))

    def test_failed_validation_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.csv"
            report = root / "usage.md"
            output.write_text("old output", encoding="utf-8")
            report.write_text("old report", encoding="utf-8")
            with self.assertRaises(Exception):
                publish_submission_artifacts(make_bundle("request_1", "request_2"), [full_decision()], UsageTracker(), output, report)
            self.assertEqual("old output", output.read_text(encoding="utf-8"))
            self.assertEqual("old report", report.read_text(encoding="utf-8"))

    def test_failed_decision_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.csv"
            report = root / "usage.md"
            output.write_text("old output", encoding="utf-8")
            report.write_text("old report", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "request_1"):
                run_submission(
                    make_bundle(), output, report, enable_environment_client=False,
                    engine_factory=lambda _data, _evidence: FakeEngine(RuntimeError("boom")),
                )
            self.assertEqual("old output", output.read_text(encoding="utf-8"))
            self.assertEqual("old report", report.read_text(encoding="utf-8"))

    def test_successful_run_creates_matching_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.csv"
            report = root / "usage.md"
            usage = run_submission(
                make_bundle(), output, report, enable_environment_client=False,
                engine_factory=lambda _data, _evidence: FakeEngine(full_decision()),
            )
            self.assertEqual((), validate_submission(output, make_bundle()))
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            report_text = report.read_text(encoding="utf-8")
            self.assertIn(digest, report_text)
            self.assertEqual(0, usage.call_count)


class PackageSafetyTests(unittest.TestCase):
    def test_package_manifest_excludes_artifacts_and_detects_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation").mkdir()
            (root / "README.md").write_text("# Solution", encoding="utf-8")
            (root / "main.py").write_text("print('ok')", encoding="utf-8")
            (root / "requirements.txt").write_text("", encoding="utf-8")
            (root / "evaluation" / "usage_report.md").write_text("report", encoding="utf-8")
            (root / ".env").write_text("excluded", encoding="utf-8")
            (root / ".cache").mkdir()
            (root / ".cache" / "evidence.json").write_text("{}", encoding="utf-8")
            (root / "final-evidence-cache").mkdir()
            (root / "final-evidence-cache" / "evidence.json").write_text("{}", encoding="utf-8")
            files = package_manifest(root)
            self.assertEqual((), validate_package_manifest(root, files))
            resolved_root = root.resolve()
            packaged = {path.resolve().relative_to(resolved_root).as_posix() for path in files}
            self.assertNotIn(".env", packaged)
            self.assertFalse(any("final-evidence-cache" in path for path in packaged))
            secret_file = root / "module.py"
            secret_file.write_text("token='" + "sk-" + ("a" * 22) + "'", encoding="utf-8")
            self.assertTrue(any("possible secret" in error for error in validate_package_manifest(root)))

    def test_package_manifest_requires_participant_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation").mkdir()
            (root / "main.py").write_text("print('ok')", encoding="utf-8")
            (root / "requirements.txt").write_text("", encoding="utf-8")
            (root / "evaluation" / "usage_report.md").write_text("report", encoding="utf-8")
            errors = validate_package_manifest(root)
            self.assertIn("required package file missing: README.md", errors)


if __name__ == "__main__":
    unittest.main()
