from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelUsage:
    provider: str
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageTracker:
    """In-memory, run-scoped usage. Callers decide if/when to persist it."""

    def __init__(self) -> None:
        self._calls: list[ModelUsage] = []
        self.cache_hits = 0
        self.validation_failures = 0

    def record(self, model: str, purpose: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._calls.append(ModelUsage("openai", model, purpose, input_tokens, output_tokens, total_tokens))

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_validation_failure(self) -> None:
        self.validation_failures += 1

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self._calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self._calls)

    def snapshot(self) -> dict:
        return {
            "model_calls": self.call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "validation_failures": self.validation_failures,
            "calls": [asdict(call) for call in self._calls],
        }
