from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import ALLOWED_FACT_KINDS, ALLOWED_SCOPES, EvidenceFact, ImageAmount


class EvidenceValidationError(ValueError):
    pass


_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _decimal(value: Any, field: str) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise EvidenceValidationError(f"invalid {field}") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise EvidenceValidationError(f"{field} must be a positive finite number")
    return parsed


def _date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise EvidenceValidationError("invalid effective_date") from exc


def validate_fact(raw: dict[str, Any], *, source_id: str, allowed_event_ids: set[str]) -> EvidenceFact:
    if not isinstance(raw, dict):
        raise EvidenceValidationError("fact must be an object")
    kind = raw.get("kind")
    if kind not in ALLOWED_FACT_KINDS:
        raise EvidenceValidationError(f"unsupported fact kind: {kind!r}")
    related = raw.get("related_event_id") or None
    if related and related not in allowed_event_ids:
        raise EvidenceValidationError("fact references an event outside the user's supplied data")
    currency = raw.get("currency") or None
    if currency and not _CURRENCY.fullmatch(str(currency)):
        raise EvidenceValidationError("invalid currency")
    scope = raw.get("scope") or None
    if scope and scope not in ALLOWED_SCOPES:
        raise EvidenceValidationError(f"invalid scope: {scope!r}")
    fact = EvidenceFact(
        kind=kind,
        source_id=source_id,
        related_event_id=related,
        amount=_decimal(raw.get("amount"), "amount"),
        currency=str(currency) if currency else None,
        effective_date=_date(raw.get("effective_date")),
        multiplier=_decimal(raw.get("multiplier"), "multiplier"),
        category=str(raw.get("category")) if raw.get("category") else None,
        scope=scope,
    )
    if kind == "salary_change" and (fact.amount is None or fact.currency is None):
        raise EvidenceValidationError("salary_change requires amount and currency")
    if kind == "salary_delay" and fact.effective_date is None:
        raise EvidenceValidationError("salary_delay requires effective_date")
    if kind == "confirmed_one_time_income" and (
        fact.amount is None or fact.currency is None or fact.effective_date is None
    ):
        raise EvidenceValidationError("confirmed income requires amount, currency, and date")
    if kind == "recurring_expense_multiplier" and fact.multiplier is None:
        raise EvidenceValidationError("expense multiplier requires multiplier")
    return fact


def validate_image_amount(
    raw: dict[str, Any], *, event_id: str, evidence_id: str, expected_currency: str
) -> ImageAmount:
    if not isinstance(raw, dict):
        raise EvidenceValidationError("image result must be an object")
    amount = _decimal(raw.get("amount"), "amount")
    currency = raw.get("currency")
    if amount is None or not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
        raise EvidenceValidationError("image result requires a positive amount and ISO currency")
    if currency != expected_currency:
        raise EvidenceValidationError(
            f"image currency {currency} conflicts with structured event currency {expected_currency}"
        )
    return ImageAmount(
        event_id=event_id,
        amount=amount,
        currency=currency,
        evidence_id=evidence_id,
        relevant_date=_date(raw.get("relevant_date")),
    )
