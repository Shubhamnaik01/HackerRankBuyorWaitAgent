from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Mapping

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

    def infer(
        self, profile: UserProfile, events: tuple[FinancialEvent, ...], start: date, end: date,
        amount_overrides: Mapping[str, Decimal] | None = None,
    ) -> list[CashFlow]:
        overrides = amount_overrides or {}
        history_start = start - timedelta(days=self.policy.recurrence.history_days)
        history = [
            event for event in events
            if event.status == "settled"
            and (event.amount is not None or event.event_id in overrides)
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

        flows.extend(self._infer_salary(profile, history, events, start, end, final_income, overrides))
        flows.extend(self._infer_other_monthly_income(profile, history, start, end, overrides))

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
            raw_amount = max(overrides.get(event.event_id, event.amount) for event in recent)
            next_date = _add_month(last_date)
            while next_date <= end:
                if next_date >= start:
                    try:
                        amount = self.exchange.convert(raw_amount, group[-1].currency, profile.home_currency, next_date)
                    except Exception:
                        break
                    flows.append(CashFlow(next_date, signed_amount(group[-1], amount), "recurring:monthly", group[-1].event_id, True))
                next_date = _add_month(next_date)

        # Aggregate only essential/protected variable spending after stable
        # commitments have been removed. A median observed 28-day category
        # total is a robust baseline without pretending every merchant
        # charge repeats at the median transaction interval.
        essential_categories = set(self.policy.recurrence.essential_variable_categories)
        essential_categories.update(profile.protected_categories)
        variable: dict[tuple[str, str, str, str], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            if event.event_id in modeled_ids or event.direction != "debit" or event.event_type not in {"expense", "subscription"}:
                continue
            if event.category not in essential_categories:
                continue
            variable[(event.category, event.direction, event.flexibility, event.currency)].append(event)

        for group in variable.values():
            group.sort(key=_event_cash_date)
            if len(group) < self.policy.recurrence.variable_minimum_occurrences:
                continue
            last_date = _event_cash_date(group[-1])
            window_days = self.policy.recurrence.variable_window_days
            if (start - last_date).days > window_days:
                continue
            totals: list[Decimal] = []
            for index in range(self.policy.recurrence.variable_window_count):
                window_end = start - timedelta(days=index * window_days)
                window_start = window_end - timedelta(days=window_days)
                total = sum(
                    (overrides.get(event.event_id, event.amount) for event in group
                     if window_start <= _event_cash_date(event) < window_end),
                    Decimal("0"),
                )
                totals.append(total)
            raw_window_amount = Decimal(median(totals))
            if raw_window_amount <= 0:
                continue
            interval = self.policy.recurrence.variable_projection_interval_days
            raw_amount = raw_window_amount * Decimal(interval) / Decimal(window_days)
            next_date = start
            while next_date <= end:
                try:
                    amount = self.exchange.convert(raw_amount, group[-1].currency, profile.home_currency, next_date)
                except Exception:
                    break
                amount = amount.quantize(self.policy.money_quantum, rounding=self.policy.rounding)
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
        amount_overrides: Mapping[str, Decimal],
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
            and (event.amount is not None or event.event_id in amount_overrides)
            and start <= _event_cash_date(event) <= end
        ]
        scheduled.sort(key=_event_cash_date)
        if len(regular) < 2 and not scheduled:
            return []

        anchor = scheduled[-1] if scheduled else regular[-1]
        # The median recent regular pay is robust to a one-cycle leave deduction
        # or overtime spike. A scheduled amount anchors the cadence but should
        # not silently redefine every later cycle when supported history exists.
        raw_amount = (
            Decimal(median([
                amount_overrides.get(event.event_id, event.amount)
                for event in regular[-self.policy.recurrence.amount_lookback:]
            ]))
            if len(regular) >= 2
            else amount_overrides.get(anchor.event_id, anchor.amount)
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

    def _infer_other_monthly_income(
        self,
        profile: UserProfile,
        history: list[FinancialEvent],
        start: date,
        end: date,
        amount_overrides: Mapping[str, Decimal],
    ) -> list[CashFlow]:
        """Recognize strong non-payroll monthly cadence without merchant labels.

        Freelance retainers and contract milestones may change description each
        month. Three settled credits on the same calendar day are sufficient;
        refunds, investments, bonuses and other one-offs remain excluded.
        """
        excluded = (
            "arrears", "bonus", "commission", "reimbursement", "refund", "prize",
            "windfall", "investment", "sale proceeds", "transfer",
        )
        groups: dict[tuple[str, str, int], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            description = event.description.lower()
            if (
                event.direction != "credit" or event.event_type != "income"
                or any(marker in description for marker in excluded)
                or "salary" in description or "payroll" in description
            ):
                continue
            groups[(event.category, event.currency, _event_cash_date(event).day)].append(event)

        result: list[CashFlow] = []
        for group in groups.values():
            group.sort(key=_event_cash_date)
            if len(group) < self.policy.recurrence.minimum_occurrences:
                continue
            recent = group[-self.policy.recurrence.minimum_occurrences:]
            if not all(
                self.policy.recurrence.monthly_min_gap_days <= gap <= self.policy.recurrence.monthly_max_gap_days
                for gap in _gaps(recent)
            ):
                continue
            last_date = _event_cash_date(recent[-1])
            if (start - last_date).days > self.policy.recurrence.monthly_recency_days:
                continue
            raw_amount = min(amount_overrides.get(event.event_id, event.amount) for event in recent)
            when = _add_month(last_date)
            while when <= end:
                if when >= start:
                    try:
                        amount = self.exchange.convert(raw_amount, recent[-1].currency, profile.home_currency, when)
                    except Exception:
                        break
                    result.append(CashFlow(when, amount, "recurring:income", recent[-1].event_id, True))
                when = _add_month(when)
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
