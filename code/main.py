from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.core import DeterministicFinancialCore
from code.data_loader import DatasetLoader
from code.submission import run_submission


def main() -> int:
    official_output = (REPO_ROOT / "output.csv").resolve()
    official_report = (REPO_ROOT / "code" / "evaluation" / "usage_report.md").resolve()
    parser = argparse.ArgumentParser(description="Generate the Buy or Wait? submission")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--samples", action="store_true", help="print core calculations for solved samples")
    parser.add_argument("--output", type=Path, default=official_output)
    parser.add_argument(
        "--usage-report", type=Path,
        default=official_report,
    )
    parser.add_argument(
        "--evidence-cache", type=Path, default=None,
        help="fresh, empty cache directory for this final run (default: isolated temporary cache)",
    )
    parser.add_argument(
        "--no-network", action="store_true",
        help="disable the environment OpenAI client for temporary dry runs only",
    )
    args = parser.parse_args()
    if args.no_network and not args.samples and (
        args.output.resolve() == official_output or args.usage_report.resolve() == official_report
    ):
        parser.error(
            "--no-network dry runs require explicit non-official --output and --usage-report paths"
        )
    data = DatasetLoader(args.dataset).load()
    if args.samples:
        core = DeterministicFinancialCore(data)
        for request in data.sample_requests.values():
            result = core.calculate(request)
            earliest = result.earliest_date_for_full_payment.isoformat() if result.earliest_date_for_full_payment else ""
            print(f"{request.request_id},{result.amount_safe_to_pay},{earliest}")
        return 0
    try:
        usage = run_submission(
            data, args.output.resolve(), args.usage_report.resolve(),
            cache_root=args.evidence_cache,
            enable_environment_client=not args.no_network,
        )
    except Exception as exc:
        print(f"Submission generation failed: {exc}", file=sys.stderr)
        return 1
    snapshot = usage.snapshot()
    print(
        f"Generated {len(data.requests)} predictions at {args.output.resolve()} and "
        f"the matching usage report at {args.usage_report.resolve()}. "
        f"Model calls={snapshot['model_calls']}, total tokens={snapshot['total_tokens']}, "
        f"cache hits={snapshot['cache_hits']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
