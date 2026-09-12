from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data_loader import DatasetLoader
from code.evidence.cache import EvidenceCache
from code.evidence.openai_client import OpenAIEvidenceClient
from code.evidence.service import EvidenceService
from code.evidence.usage import UsageTracker


def _request(data, request_id: str):
    request = data.sample_requests.get(request_id) or data.requests.get(request_id)
    if request is None:
        raise ValueError(f"unknown request id: {request_id}")
    return request


def _assert_valid(label: str, warnings: list[str]) -> None:
    rejected = [warning for warning in warnings if "rejected model evidence" in warning]
    failed = [warning for warning in warnings if "extraction failed" in warning]
    if rejected or failed:
        raise RuntimeError(f"{label} validation failed: {rejected + failed}")


def _json_default(value):
    if isinstance(value, (Decimal, date)):
        return str(value)
    raise TypeError(f"unsupported structured value: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-call OpenAI evidence smoke test")
    parser.add_argument("--message-request-id", required=True)
    parser.add_argument("--image-request-id", required=True)
    args = parser.parse_args()

    data = DatasetLoader(REPO_ROOT / "dataset").load()
    client = OpenAIEvidenceClient.from_environment()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not present after repository .env loading")

    message_request = _request(data, args.message_request_id)
    image_request = _request(data, args.image_request_id)
    usage = UsageTracker()

    with tempfile.TemporaryDirectory(prefix="buy-or-wait-evidence-") as temporary:
        service = EvidenceService(
            data,
            client=client,
            usage=usage,
            cache=EvidenceCache(Path(temporary)),
            enable_environment_client=False,
        )
        message_first = service.resolve(message_request, 180, 90)
        _assert_valid("message", message_first.warnings)
        if usage.call_count != 1:
            raise RuntimeError(f"expected one message API call, observed {usage.call_count}")

        image_first = service.resolve(image_request, 180, 90)
        _assert_valid("image", image_first.warnings)
        if usage.call_count != 2 or not image_first.image_amounts:
            raise RuntimeError("expected one validated linked-image API call")

        calls_after_first_pass = usage.call_count
        message_second = service.resolve(message_request, 180, 90)
        image_second = service.resolve(image_request, 180, 90)
        _assert_valid("cached message", message_second.warnings)
        _assert_valid("cached image", image_second.warnings)
        if usage.call_count != calls_after_first_pass:
            raise RuntimeError("repeated evidence caused an unexpected API call")
        if usage.cache_hits != 2:
            raise RuntimeError(f"expected two application-cache hits, observed {usage.cache_hits}")

    snapshot = usage.snapshot()
    print("dotenv_key_present=True")
    print(f"model={client.model}")
    print("message_structured_validation=passed")
    print("message_facts=" + json.dumps(
        [asdict(fact) for fact in message_first.facts], default=_json_default, sort_keys=True,
    ))
    print("image_structured_validation=passed")
    print("image_facts=" + json.dumps(
        [asdict(value) for value in image_first.image_amounts.values()],
        default=_json_default, sort_keys=True,
    ))
    print(
        f"api_calls={snapshot['model_calls']} input_tokens={snapshot['input_tokens']} "
        f"output_tokens={snapshot['output_tokens']} total_tokens={snapshot['total_tokens']} "
        f"validation_failures={snapshot['validation_failures']}"
    )
    print(f"repeat_cache_hits={snapshot['cache_hits']} repeat_api_calls=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
