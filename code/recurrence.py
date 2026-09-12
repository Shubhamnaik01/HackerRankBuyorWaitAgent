from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from .config import ForecastPolicy
from .exchange import ExchangeRateTable
from .models import CashFlow, FinancialEvent, UserProfile
from .normalization import signed_amount


def _add_month(value: date) -> date:
    year = value.year + (value.month // 12)
    month = 1 if value.month == 12 else value.month + 1
    if value.month == 12:
        year = value.year + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _event_cash_date(event: FinancialEvent) -> date:
    return event.settlement_date or event.event_date


def _gaps(events: list[FinancialEvent]) -> list[int]:
    dates = [_event_cash_date(event) for event in events]
    return [(right - left).days for left, right in zip(dates, dates[1:])]


class RecurrenceDetector:
    def __init__(self, policy: ForecastPolicy, exchange: ExchangeRateTable):
        self.policy = policy
        self.exchange = exchange

    def infer(self, profile: UserProfile, events: tuple[FinancialEvent, ...], start: date, end: date) -> list[CashFlow]:
        history_start = start - timedelta(days=self.policy.recurrence.history_days)
        history = [
            event for event in events
            if event.status == "settled"
            and event.amount is not None
            and event.direction in {"debit", "credit"}
            and history_start <= _event_cash_date(event) < start
        ]
        history.sort(key=_event_cash_date)
        flows: list[CashFlow] = []
        modeled_ids: set[str] = set()

        # A final payroll record is explicit evidence that older salary history
        # must not be projected past it.
        final_income = any(
            event.direction == "credit"
            and event.category == "salary"
            and "final" in event.description.lower()
            for event in history
        )

        flows.extend(self._infer_salary(profile, history, events, start, end, final_income))

        groups: dict[tuple[str, str, str, str], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            groups[(event.event_type, event.description, event.category, event.direction)].append(event)

        for key, group in groups.items():
            group.sort(key=_event_cash_date)
            if group[-1].direction == "credit":
                # Income is handled separately so description changes and
                # delayed settlement dates do not break a genuine payroll
                # series, while one-off credits are kept out.
                continue
            if len(group) < self.policy.recurrence.minimum_occurrences:
                continue
            gaps = _gaps(group)
            recent_gaps = gaps[-(self.policy.recurrence.minimum_occurrences - 1):]
            if not all(self.policy.recurrence.monthly_min_gap_days <= gap <= self.policy.recurrence.monthly_max_gap_days for gap in recent_gaps):
                continue
            last_date = _event_cash_date(group[-1])
            if (start - last_date).days > self.policy.recurrence.monthly_recency_days:
                continue
            modeled_ids.update(event.event_id for event in group)
            recent = group[-self.policy.recurrence.amount_lookback:]
            raw_amount = min(event.amount for event in recent) if group[-1].direction == "credit" else max(event.amount for event in recent)
            next_date = _add_month(last_date)
            while next_date <= end:
                if next_date >= start:
                    try:
                        amount = self.exchange.convert(raw_amount, group[-1].currency, profile.home_currency, next_date)
                    except Exception:
                        break
                    flows.append(CashFlow(next_date, signed_amount(group[-1], amount), "recurring:monthly", group[-1].event_id, True))
                next_date = _add_month(next_date)

        # Aggregate variable spending by category after stable monthly series
        # have been removed. This captures regular grocery/transport/dining
        # cadence even when merchant descriptions rotate.
        variable: dict[tuple[str, str, str], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            if event.event_id in modeled_ids or event.direction != "debit" or event.event_type not in {"expense", "subscription"}:
                continue
            variable[(event.category, event.direction, event.flexibility)].append(event)

        for group in variable.values():
            group.sort(key=_event_cash_date)
            if len(group) < self.policy.recurrence.variable_minimum_occurrences:
                continue
            recent = group[-8:]
            gaps = _gaps(recent)
            interval = int(median(gaps)) if gaps else 0
            if interval <= 0 or interval > self.policy.recurrence.variable_max_interval_days:
                continue
            last_date = _event_cash_date(group[-1])
            if (start - last_date).days > interval * self.policy.recurrence.variable_recency_multiplier:
                continue
            raw_amount = max(event.amount for event in group[-self.policy.recurrence.amount_lookback:])
            next_date = last_date + timedelta(days=interval)
            while next_date <= end:
                if next_date >= start:
                    try:
                        amount = self.exchange.convert(raw_amount, group[-1].currency, profile.home_currency, next_date)
                    except Exception:
                        break
                    flows.append(CashFlow(next_date, -amount, "recurring:variable", group[-1].event_id, True))
                next_date += timedelta(days=interval)
        return flows

    def _infer_salary(
        self,
        profile: UserProfile,
        history: list[FinancialEvent],
        all_events: tuple[FinancialEvent, ...],
        start: date,
        end: date,
        final_income: bool,
    ) -> list[CashFlow]:
        if final_income:
            return []
        excluded = ("arrears", "bonus", "commission", "reimbursement", "refund", "prize", "windfall")
        regular = [
            event for event in history
            if event.direction == "credit"
            and event.event_type == "income"
            and event.category == "salary"
            and not any(marker in event.description.lower() for marker in excluded)
            and ("salary" in event.description.lower() or "payroll" in event.description.lower())
        ]
        regular.sort(key=_event_cash_date)
        scheduled = [
            event for event in all_events
            if event.status == "scheduled"
            and event.direction == "credit"
            and event.category == "salary"
            and event.amount is not None
            and start <= _event_cash_date(event) <= end
        ]
        scheduled.sort(key=_event_cash_date)
        if len(regular) < 2 and not scheduled:
            return []

        anchor = scheduled[-1] if scheduled else regular[-1]
        # A scheduled salary supplies the amount for subsequent normal cycles;
        # otherwise use the lowest recent regular pay to avoid overstating it.
        raw_amount = anchor.amount if scheduled else min(
            event.amount for event in regular[-self.policy.recurrence.amount_lookback:]
        )
        currency = anchor.currency
        next_date = _add_month(_event_cash_date(anchor))
        if not scheduled:
            # Include the first occurrence following settled history.
            next_date = _add_month(_event_cash_date(regular[-1]))
        result: list[CashFlow] = []
        while next_date <= end:
            if next_date >= start:
                try:
                    amount = self.exchange.convert(raw_amount, currency, profile.home_currency, next_date)
                except Exception:
                    break
                result.append(CashFlow(next_date, amount, "recurring:salary", anchor.event_id, True))
            next_date = _add_month(next_date)
        return result


def remove_explicit_collisions(inferred: list[CashFlow], explicit: list[CashFlow], events_by_id: dict[str, FinancialEvent]) -> list[CashFlow]:
    explicit_keys: set[tuple[date, str, str]] = set()
    for flow in explicit:
        event = events_by_id.get(flow.source_id or "")
        if event:
            explicit_keys.add((flow.flow_date, event.category, event.direction))
    result: list[CashFlow] = []
    for flow in inferred:
        event = events_by_id.get(flow.source_id or "")
        key = (flow.flow_date, event.category, event.direction) if event else None
        if key not in explicit_keys:
            result.append(flow)
    return result
