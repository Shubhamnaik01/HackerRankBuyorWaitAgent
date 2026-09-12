from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


ALLOWED_FACT_KINDS = frozenset({
    "salary_change",
    "salary_delay",
    "employment_end",
    "confirmed_one_time_income",
    "recurring_expense_multiplier",
    "failed_debit_due",
    "exclude_from_cash",
    "internal_transfer",
})

ALLOWED_SCOPES = frozenset({"next_only", "recurring", "one_time"})


@dataclass(frozen=True)
class EvidenceFact:
    kind: str
    source_id: str
    related_event_id: str | None = None
    amount: Decimal | None = None
    currency: str | None = None
    effective_date: date | None = None
    multiplier: Decimal | None = None
    category: str | None = None
    scope: str | None = None


@dataclass(frozen=True)
class ImageAmount:
    event_id: str
    amount: Decimal
    currency: str
    evidence_id: str
    relevant_date: date | None = None


@dataclass
class EvidenceBundle:
    facts: list[EvidenceFact] = field(default_factory=list)
    image_amounts: dict[str, ImageAmount] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
