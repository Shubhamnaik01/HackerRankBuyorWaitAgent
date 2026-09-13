from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from code.evidence.usage import UsageTracker


PRICING_SOURCE_URL = "https://developers.openai.com/api/docs/models/gpt-5-mini"
PRICING_VERIFIED_DATE = "2026-09-13"
TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


# Standard API token rates verified from the official model page above. The
# tracker does not split cached input tokens, so all recorded input is priced
# at the standard input rate for a conservative, reproducible estimate.
MODEL_PRICES: dict[tuple[str, str], ModelPrice] = {
    ("openai", "gpt-5-mini"): ModelPrice(Decimal("0.25"), Decimal("2.00")),
    ("openai", "gpt-5-mini-2025-08-07"): ModelPrice(Decimal("0.25"), Decimal("2.00")),
}


class PricingUnavailableError(ValueError):
    pass


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int,
    prices: dict[tuple[str, str], ModelPrice] = MODEL_PRICES,
) -> Decimal:
    price = prices.get((provider, model))
    if price is None:
        raise PricingUnavailableError(
            f"no verified pricing configured for provider={provider!r}, model={model!r}"
        )
    return (
        Decimal(input_tokens) * price.input_usd_per_million
        + Decimal(output_tokens) * price.output_usd_per_million
    ) / TOKENS_PER_MILLION


def _number(value: Decimal, places: str = "0.000000") -> str:
    return format(value.quantize(Decimal(places), rounding=ROUND_HALF_UP), "f")


def render_usage_report(
    usage: UsageTracker, request_count: int, *, output_digest: str | None = None,
) -> str:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    snapshot = usage.snapshot()
    calls = snapshot["calls"]
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0, "total": 0}
    )
    purpose_counts: dict[str, int] = defaultdict(int)
    for call in calls:
        key = (call["provider"], call["model"])
        groups[key]["calls"] += 1
        groups[key]["input"] += call["input_tokens"]
        groups[key]["output"] += call["output_tokens"]
        groups[key]["total"] += call["total_tokens"]
        purpose_counts[call["purpose"]] += 1

    model_costs: dict[tuple[str, str], Decimal] = {}
    for (provider, model), totals in groups.items():
        model_costs[(provider, model)] = estimate_cost(
            provider, model, totals["input"], totals["output"],
        )
    total_cost = sum(model_costs.values(), Decimal("0"))
    requests = Decimal(request_count)
    providers = ", ".join(sorted({provider for provider, _ in groups})) or "none"
    models = ", ".join(sorted({model for _, model in groups})) or "none"
    message_calls = purpose_counts["message"] + purpose_counts["message_evidence"]
    image_calls = purpose_counts["image"] + purpose_counts["image_evidence"]

    lines = [
        "# Final Full-Dataset API Usage Report",
        "",
        "This report is generated from the run-scoped usage tracker used by the same successful run that produced `output.csv`.",
        "",
        "## Run summary",
        "",
        f"- Dataset requests processed: {request_count}",
        f"- Output CSV SHA-256: {output_digest or 'not supplied'}",
        f"- Provider(s): {providers}",
        f"- Model(s): {models}",
        f"- Total model calls: {snapshot['model_calls']}",
        f"- Message extraction calls: {message_calls}",
        f"- Image extraction calls: {image_calls}",
        f"- Input tokens: {snapshot['input_tokens']}",
        f"- Output tokens: {snapshot['output_tokens']}",
        f"- Total tokens: {snapshot['total_tokens']}",
        f"- Average model calls per request: {_number(Decimal(snapshot['model_calls']) / requests)}",
        f"- Average input tokens per request: {_number(Decimal(snapshot['input_tokens']) / requests)}",
        f"- Average output tokens per request: {_number(Decimal(snapshot['output_tokens']) / requests)}",
        f"- Average total tokens per request: {_number(Decimal(snapshot['total_tokens']) / requests)}",
        f"- Cache hits: {snapshot['cache_hits']}",
        f"- Validation failures: {snapshot['validation_failures']}",
    ]
    diagnostics = snapshot.get("validation_failure_details", [])
    if diagnostics:
        lines.extend([
            "",
            "## Evidence validation diagnostics",
            "",
            "Only source identifiers and validator reasons are recorded; message and image contents are omitted.",
            "Each row represents one rejected model response; a successful retry does not erase the failure.",
            "",
            "| Source type | Source ID | Request ID | Attempt | Validation reason |",
            "|---|---|---|---:|---|",
        ])
        for diagnostic in diagnostics:
            lines.append(
                f"| {diagnostic['source_type']} | {diagnostic['source_id']} | "
                f"{diagnostic['request_id'] or 'not supplied'} | {diagnostic['attempt']} | "
                f"{diagnostic['reason']} |"
            )
    lines.extend([
        "",
        "## Per-model breakdown",
        "",
        "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Estimated cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if groups:
        for provider, model in sorted(groups):
            totals = groups[(provider, model)]
            lines.append(
                f"| {provider} | {model} | {totals['calls']} | {totals['input']} | "
                f"{totals['output']} | {totals['total']} | {_number(model_costs[(provider, model)], '0.00000000')} |"
            )
    else:
        lines.append("| none | none | 0 | 0 | 0 | 0 | 0.00000000 |")

    lines.extend([
        "",
        "## Cost",
        "",
        f"- Estimated total API cost (USD): {_number(total_cost, '0.00000000')}",
        f"- Estimated API cost per request (USD): {_number(total_cost / requests, '0.00000000')}",
        "",
        "## Pricing basis",
        "",
        f"- Official source: {PRICING_SOURCE_URL}",
        f"- Rates verified: {PRICING_VERIFIED_DATE}",
        "- Prices are in USD per 1 million tokens.",
        "- `gpt-5-mini`: input $0.25; output $2.00.",
        "- `gpt-5-mini-2025-08-07`: input $0.25; output $2.00.",
        "- Cost formula: `(input_tokens × input_rate + output_tokens × output_rate) / 1,000,000`.",
        "- Token counts come only from completed API response usage. Cache hits add no calls or tokens.",
        "- Cached input tokens are not separately tracked, so recorded input is conservatively charged at the standard input rate.",
        "",
    ])
    return "\n".join(lines)
