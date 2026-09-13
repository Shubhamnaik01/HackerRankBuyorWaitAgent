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


@dataclass(frozen=True)
class EvidenceValidationDiagnostic:
    source_type: str
    source_id: str
    request_id: str | None
    attempt: int
    reason: str


def _safe_diagnostic_text(value: str, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    text = text.replace("|", "/").replace("`", "'").replace("<", "[").replace(">", "]")
    return text if len(text) <= limit else text[: limit - 3] + "..."


class UsageTracker:
    """In-memory, run-scoped usage. Callers decide if/when to persist it."""

    def __init__(self) -> None:
        self._calls: list[ModelUsage] = []
        self._validation_diagnostics: list[EvidenceValidationDiagnostic] = []
        self.cache_hits = 0
        self.validation_failures = 0

    def record(self, model: str, purpose: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._calls.append(ModelUsage("openai", model, purpose, input_tokens, output_tokens, total_tokens))

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_validation_failure(
        self, source_type: str | None = None, source_id: str | None = None,
        request_id: str | None = None, reason: str | None = None, attempt: int = 1,
    ) -> None:
        self.validation_failures += 1
        if source_type is not None and source_id is not None and reason is not None:
            self._validation_diagnostics.append(EvidenceValidationDiagnostic(
                _safe_diagnostic_text(source_type, 20),
                _safe_diagnostic_text(source_id, 100),
                _safe_diagnostic_text(request_id, 100) if request_id is not None else None,
                attempt,
                _safe_diagnostic_text(reason),
            ))

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
            "validation_failure_details": [
                asdict(diagnostic) for diagnostic in self._validation_diagnostics
            ],
            "calls": [asdict(call) for call in self._calls],
        }
