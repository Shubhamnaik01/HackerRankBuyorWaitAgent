from __future__ import annotations

from datetime import date
from decimal import Decimal

from .exchange import ExchangeRateTable
from .models import CashFlow, FinancialEvent, UserProfile


IGNORED_STATUSES = {"cancelled", "failed", "unrealized"}


def explicit_future_flows(
    profile: UserProfile,
    events: tuple[FinancialEvent, ...],
    start: date,
    end: date,
    exchange: ExchangeRateTable,
) -> tuple[list[CashFlow], list[str]]:
    """Normalize only explicit future cash state.

    The profile balance is already a snapshot, so settled history is never
    replayed. Failed debits are ignored in this phase; message evidence that a
    failed bill remains due will be incorporated by the later evidence layer.
    """
    flows: list[CashFlow] = []
    warnings: list[str] = []
    for event in events:
        if event.status in IGNORED_STATUSES or event.status == "settled":
            continue
        if event.status == "pending" and event.direction != "debit":
            continue
        if event.status not in {"pending", "scheduled"}:
            continue
        flow_date = event.settlement_date or event.event_date
        if not start <= flow_date <= end:
            continue
        if event.amount is None:
            warnings.append(f"{event.event_id}: missing amount; linked image evidence not interpreted")
            continue
        try:
            amount = exchange.convert(event.amount, event.currency, profile.home_currency, flow_date)
        except Exception as exc:
            warnings.append(f"{event.event_id}: {exc}")
            continue
        if event.direction == "debit":
            amount = -amount
        elif event.direction != "credit":
            continue
        flows.append(CashFlow(flow_date, amount, f"explicit:{event.status}", event.event_id))
    return flows, warnings


def signed_amount(event: FinancialEvent, amount: Decimal) -> Decimal:
    return -amount if event.direction == "debit" else amount

