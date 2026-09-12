from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.core import DeterministicFinancialCore
from code.data_loader import DatasetLoader
from code.evidence.service import EvidenceService
from code.evidence.usage import UsageTracker


def main() -> int:
    data = DatasetLoader(REPO_ROOT / "dataset").load()
    usage = UsageTracker()
    evidence = EvidenceService(data, usage=usage)
    core = DeterministicFinancialCore(data, evidence=evidence)
    amount_matches = 0
    date_matches = 0
    differences: list[str] = []
    for request in data.sample_requests.values():
        result = core.calculate(request)
        amount_ok = result.amount_safe_to_pay == request.expected_amount_safe_to_pay
        date_ok = result.earliest_date_for_full_payment == request.expected_earliest_date_for_full_payment
        amount_matches += int(amount_ok)
        date_matches += int(date_ok)
        if not amount_ok or not date_ok:
            differences.append(
                f"{request.request_id}: amount {result.amount_safe_to_pay} vs {request.expected_amount_safe_to_pay}; "
                f"earliest {result.earliest_date_for_full_payment or ''} vs "
                f"{request.expected_earliest_date_for_full_payment or ''}; warnings={'; '.join(result.warnings) or 'none'}"
            )
    print(f"amount_safe_to_pay exact matches: {amount_matches}/25")
    print(f"earliest_date_for_full_payment exact matches: {date_matches}/25")
    print("Differences:")
    for difference in differences:
        print(difference)
    snapshot = usage.snapshot()
    print(
        "Evidence API usage: "
        f"calls={snapshot['model_calls']}, input_tokens={snapshot['input_tokens']}, "
        f"output_tokens={snapshot['output_tokens']}, total_tokens={snapshot['total_tokens']}, "
        f"cache_hits={snapshot['cache_hits']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
