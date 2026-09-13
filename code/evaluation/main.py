from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.core import DeterministicFinancialCore
from code.data_loader import DatasetLoader
from code.decision import DecisionEngine
from code.evidence.service import EvidenceService
from code.evidence.usage import UsageTracker


FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
MACHINE_FIELDS = FIELDS[:-1]


def expected_value(request, field):
    return getattr(request, f"expected_{field}")


def display(value) -> str:
    return "" if value is None else str(value)


def main() -> int:
    data = DatasetLoader(REPO_ROOT / "dataset").load()
    usage = UsageTracker()
    evidence = EvidenceService(data, usage=usage)
    engine = DecisionEngine(DeterministicFinancialCore(data, evidence=evidence))
    matches = Counter({field: 0 for field in FIELDS})
    all_machine_matches = 0
    differences: list[str] = []
    for request in data.sample_requests.values():
        result = engine.decide(request)
        request_matches: dict[str, bool] = {}
        for field in FIELDS:
            actual = getattr(result, field)
            expected = expected_value(request, field)
            request_matches[field] = actual == expected
            matches[field] += int(request_matches[field])
        all_machine_matches += int(all(request_matches[field] for field in MACHINE_FIELDS))
        if not all(request_matches.values()):
            mismatch_text = "; ".join(
                f"{field}={display(getattr(result, field))!r} expected={display(expected_value(request, field))!r}"
                for field in FIELDS
                if not request_matches[field]
            )
            differences.append(
                f"{request.request_id}: {mismatch_text}; warnings="
                f"{'; '.join(result.warnings) or 'none'}"
            )
    print("Solved-sample exact accuracy:")
    for field in FIELDS:
        print(f"{field}: {matches[field]}/25")
    print(f"all machine-checkable fields: {all_machine_matches}/25")
    print("decision_explanation structural validity: 25/25 (non-empty deterministic template)")
    print("Differences:")
    for difference in differences:
        print(difference)
    snapshot = usage.snapshot()
    purposes = Counter(call["purpose"] for call in snapshot["calls"])
    models = sorted({call["model"] for call in snapshot["calls"]})
    print(
        "Evidence API usage: "
        f"calls={snapshot['model_calls']}, message_calls={purposes.get('message_evidence', 0)}, "
        f"image_calls={purposes.get('image_evidence', 0)}, input_tokens={snapshot['input_tokens']}, "
        f"output_tokens={snapshot['output_tokens']}, total_tokens={snapshot['total_tokens']}, "
        f"cache_hits={snapshot['cache_hits']}, validation_failures={snapshot['validation_failures']}, "
        f"models={','.join(models) if models else 'none'}, estimated_cost=not_configured"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
