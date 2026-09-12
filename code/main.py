from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.core import DeterministicFinancialCore
from code.data_loader import DatasetLoader


def main() -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? deterministic financial core")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--samples", action="store_true", help="print core calculations for solved samples")
    args = parser.parse_args()
    data = DatasetLoader(args.dataset).load()
    if not args.samples:
        parser.error("submission generation is not implemented in this phase; use --samples")
    core = DeterministicFinancialCore(data)
    for request in data.sample_requests.values():
        result = core.calculate(request)
        earliest = result.earliest_date_for_full_payment.isoformat() if result.earliest_date_for_full_payment else ""
        print(f"{request.request_id},{result.amount_safe_to_pay},{earliest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
