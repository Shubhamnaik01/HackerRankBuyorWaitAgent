from __future__ import annotations

import calendar
from datetime import date
from decimal import Decimal

from ..exchange import ExchangeRateTable
from ..models import CashFlow, FinancialEvent, UserProfile
from .models import EvidenceFact


def _add_month(value: date) -> date:
    year = value.year + (value.month // 12)
    month = 1 if value.month == 12 else value.month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _is_salary(flow: CashFlow, events: dict[str, FinancialEvent]) -> bool:
    event = events.get(flow.source_id or "")
    return flow.amount > 0 and (
        flow.source == "recurring:salary" or (event is not None and event.category == "salary")
    )


def apply_evidence_to_flows(
    flows: list[CashFlow], facts: list[EvidenceFact], profile: UserProfile,
    events: dict[str, FinancialEvent], exchange: ExchangeRateTable,
    start: date, end: date,
) -> tuple[list[CashFlow], list[str]]:
    result = list(flows)
    warnings: list[str] = []
    for fact in facts:
        if fact.kind == "exclude_from_cash":
            if fact.category == "unconfirmed_income":
                result = [flow for flow in result if flow.source != "recurring:income"]
            continue
        if fact.kind == "internal_transfer":
            continue
        if fact.kind == "employment_end":
            cutoff = fact.effective_date or start
            result = [flow for flow in result if not (_is_salary(flow, events) and flow.flow_date >= cutoff)]
            continue
        if fact.kind == "salary_delay":
            salary = sorted((flow for flow in result if _is_salary(flow, events)), key=lambda flow: flow.flow_date)
            if not salary or fact.effective_date is None:
                continue
            anchor = salary[0]
            result = [flow for flow in result if not _is_salary(flow, events)]
            when = fact.effective_date
            while when <= end:
                if when >= start:
                    result.append(CashFlow(when, anchor.amount, "evidence:salary-delay", anchor.source_id, True))
                when = _add_month(when)
            continue
        if fact.kind == "salary_change":
            assert fact.amount is not None and fact.currency is not None
            salary = sorted((flow for flow in result if _is_salary(flow, events)), key=lambda flow: flow.flow_date)
            effective = fact.effective_date or (salary[0].flow_date if salary else None)
            if effective is None:
                warnings.append(f"{fact.source_id}: salary amount has no supported future pay date")
                continue
            target_dates = [flow.flow_date for flow in salary if flow.flow_date >= effective]
            if fact.scope == "recurring":
                # An amendment's effective date changes the amount, not a
                # separately supported payroll cadence. Only synthesize a
                # cadence when no structured/inferred payday exists.
                if not target_dates:
                    when = effective
                    while when <= end:
                        if when >= start:
                            target_dates.append(when)
                        when = _add_month(when)
                else:
                    when = _add_month(target_dates[-1])
                    while when <= end:
                        target_dates.append(when)
                        when = _add_month(when)
            elif target_dates:
                target_dates = target_dates[:1]
            dates = set(target_dates)
            if fact.scope == "recurring":
                result = [flow for flow in result if not (_is_salary(flow, events) and flow.flow_date >= effective)]
            else:
                result = [flow for flow in result if not (_is_salary(flow, events) and flow.flow_date in dates)]
            for when in target_dates:
                try:
                    amount = exchange.convert(fact.amount, fact.currency, profile.home_currency, when)
                except Exception as exc:
                    warnings.append(f"{fact.source_id}: {exc}")
                    continue
                result.append(CashFlow(when, amount, "evidence:salary", fact.related_event_id or fact.source_id, True))
            continue
        if fact.kind == "confirmed_one_time_income":
            assert fact.amount is not None and fact.currency is not None and fact.effective_date is not None
            if start <= fact.effective_date <= end:
                try:
                    amount = exchange.convert(fact.amount, fact.currency, profile.home_currency, fact.effective_date)
                except Exception as exc:
                    warnings.append(f"{fact.source_id}: {exc}")
                else:
                    result.append(CashFlow(
                        fact.effective_date, amount, "evidence:confirmed-income",
                        fact.related_event_id or fact.source_id,
                    ))
            continue
        if fact.kind == "recurring_expense_multiplier":
            category = fact.category or "rent"
            changed: list[CashFlow] = []
            for flow in result:
                event = events.get(flow.source_id or "")
                if flow.inferred and flow.amount < 0 and event and event.category == category:
                    amended = (flow.amount * (fact.multiplier or Decimal("1"))).quantize(
                        exchange.policy.money_quantum, rounding=exchange.policy.rounding,
                    )
                    changed.append(CashFlow(
                        flow.flow_date, amended,
                        "evidence:expense-amendment", flow.source_id, True,
                    ))
                else:
                    changed.append(flow)
            result = changed
            continue
        if fact.kind == "failed_debit_due" and fact.related_event_id:
            failed = events.get(fact.related_event_id)
            if failed and failed.amount is not None:
                already_active = any(
                    flow.source_id == failed.event_id
                    or (events.get(flow.source_id or "") and events[flow.source_id].linked_event_id == failed.event_id)
                    for flow in result
                )
                if not already_active:
                    try:
                        amount = exchange.convert(failed.amount, failed.currency, profile.home_currency, start)
                    except Exception as exc:
                        warnings.append(f"{fact.source_id}: {exc}")
                    else:
                        result.append(CashFlow(start, -amount, "evidence:failed-debit-due", failed.event_id))
    return result, warnings
