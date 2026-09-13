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


def _month_key(value: date) -> tuple[int, int]:
    return value.year, value.month


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _month_end(value: date) -> date:
    return value.replace(day=calendar.monthrange(value.year, value.month)[1])


def _anchored_date(year: int, month: int, day: int | None) -> date:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day if day is None else min(day, last_day))


def _supported_monthly_day(events: list[FinancialEvent]) -> int | None:
    """Return a robust payday/due-day; None represents month end.

    A single delayed or image-enriched observation must not move an otherwise
    established cycle. Month-end series are recognized separately because
    their numeric day naturally varies across February and 30/31-day months.
    """
    dates = [_event_cash_date(event) for event in events]
    if sum((_month_end(value) - value).days <= 2 for value in dates) * 3 >= len(dates) * 2:
        return None
    return int(median(value.day for value in dates))


def _next_supported_month(events: list[FinancialEvent]) -> date:
    latest = max(_event_cash_date(event) for event in events)
    following = _add_month(latest.replace(day=1))
    return _anchored_date(following.year, following.month, _supported_monthly_day(events))


def _deduplicate_lifecycle_history(events: list[FinancialEvent]) -> list[FinancialEvent]:
    """Use at most one recurrence observation from a transaction lifecycle.

    Same-direction lifecycle updates contribute their terminal settled row.
    Mixed-direction lifecycles (for example, a purchase and reversal) are not
    evidence that either cash flow repeats.
    """
    by_id = {event.event_id: event for event in events}
    groups: dict[str, list[FinancialEvent]] = defaultdict(list)
    for event in events:
        root = event
        seen: set[str] = set()
        while root.linked_event_id and root.linked_event_id in by_id and root.event_id not in seen:
            seen.add(root.event_id)
            root = by_id[root.linked_event_id]
        groups[root.event_id].append(event)

    result: list[FinancialEvent] = []
    for group in groups.values():
        if len(group) == 1:
            result.extend(group)
            continue
        signatures = {(event.direction, event.event_type, event.category) for event in group}
        if len(signatures) == 1:
            result.append(max(group, key=lambda event: (_event_cash_date(event), event.event_id)))
    return result


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
            and history_start <= _event_cash_date(event) <= start
        ]
        history = _deduplicate_lifecycle_history(history)
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

        # Categories with several transactions in multiple calendar cycles are
        # budget-like spending, not several independent monthly commitments.
        category_groups: dict[tuple[str, str], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            if event.direction == "debit" and event.event_type in {"expense", "subscription"}:
                category_groups[(event.category, event.currency)].append(event)
        transaction_heavy_keys: set[tuple[str, str]] = set()
        for key, group in category_groups.items():
            cycle_counts: dict[tuple[int, int], int] = defaultdict(int)
            for event in group:
                cycle_counts[_month_key(_event_cash_date(event))] += 1
            if sum(count >= 2 for count in cycle_counts.values()) >= 2:
                transaction_heavy_keys.add(key)
        essential_categories = set(self.policy.recurrence.essential_variable_categories)
        essential_categories.update(profile.protected_categories)
        variable_keys = {
            key for key in transaction_heavy_keys if key[0] in essential_categories
        }

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
            if (group[-1].category, group[-1].currency) in transaction_heavy_keys:
                continue
            if len(group) < self.policy.recurrence.minimum_occurrences:
                continue
            # Multiple rows in one calendar cycle are ambiguous duplicates,
            # not additional proof of monthly recurrence.
            if len({_month_key(_event_cash_date(event)) for event in group}) != len(group):
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
            raw_amount = max(
                overrides.get(event.event_id, event.amount) for event in recent
            )
            next_date = _next_supported_month(group)
            while next_date <= end:
                if next_date >= start:
                    try:
                        amount = self.exchange.convert(raw_amount, group[-1].currency, profile.home_currency, next_date)
                    except Exception:
                        break
                    flows.append(CashFlow(next_date, signed_amount(group[-1], amount), "recurring:monthly", group[-1].event_id, True))
                next_date = _add_month(next_date)

        # Project transaction-heavy categories from complete calendar-cycle
        # budgets. This separates how often purchases occur from how much a
        # household normally spends in a cycle, excludes partial boundary
        # months, and makes a single anomalous month unable to dominate.
        variable: dict[tuple[str, str], list[FinancialEvent]] = defaultdict(list)
        for event in history:
            if (
                event.event_id in modeled_ids or event.direction != "debit"
                or event.event_type not in {"expense", "subscription"}
                or (event.category, event.currency) not in variable_keys
            ):
                continue
            variable[(event.category, event.currency)].append(event)
        current_variable_cycle_started = any(
            _month_key(_event_cash_date(event)) == _month_key(start)
            for group in variable.values()
            for event in group
        )

        for (category, currency), group in variable.items():
            group.sort(key=_event_cash_date)
            if len(group) < self.policy.recurrence.variable_minimum_occurrences:
                continue
            first_full_month = _month_start(history_start)
            if first_full_month < history_start:
                first_full_month = _add_month(first_full_month)
            cycle = first_full_month
            cycle_totals: list[Decimal] = []
            observed_cycles = 0
            while _month_end(cycle) < start:
                total = sum(
                    (overrides.get(event.event_id, event.amount) for event in group
                     if _month_key(_event_cash_date(event)) == _month_key(cycle)),
                    Decimal("0"),
                )
                cycle_totals.append(total)
                observed_cycles += int(total > 0)
                cycle = _add_month(cycle)
            if observed_cycles < self.policy.recurrence.minimum_occurrences or not cycle_totals:
                continue
            raw_budget = Decimal(median(cycle_totals))
            if raw_budget <= 0:
                continue
            current_spent = sum(
                (overrides.get(event.event_id, event.amount) for event in group
                 if _month_key(_event_cash_date(event)) == _month_key(start)),
                Decimal("0"),
            )
            month = _month_start(start)
            while month <= end:
                period_start = max(start, month)
                period_end = min(end, _month_end(month))
                if _month_key(month) == _month_key(start):
                    # The balance snapshot already reflects month-to-date
                    # spending. Calendar days alone are not evidence that an
                    # unstarted variable-spending cycle has consumed budget.
                    unused_budget = max(Decimal("0"), raw_budget - current_spent)
                    if current_variable_cycle_started:
                        ordinary_remainder = (
                            raw_budget * Decimal((period_end - period_start).days + 1)
                            / Decimal(_month_end(month).day)
                        )
                        remaining_budget = min(unused_budget, ordinary_remainder)
                    else:
                        remaining_budget = unused_budget
                else:
                    # A 90-day horizon commonly ends part-way through a
                    # calendar month; reserve only the covered share rather
                    # than charging a whole additional budget cycle.
                    remaining_budget = (
                        raw_budget * Decimal((period_end - period_start).days + 1)
                        / Decimal(_month_end(month).day)
                    )
                days = (period_end - period_start).days + 1
                allocated = Decimal("0")
                allocated_home = Decimal("0")
                for offset in range(days):
                    when = period_start + timedelta(days=offset)
                    raw_amount = (
                        remaining_budget - allocated
                        if offset == days - 1 else remaining_budget / Decimal(days)
                    )
                    allocated += raw_amount
                    if offset == days - 1 and currency == profile.home_currency:
                        amount = remaining_budget.quantize(
                            self.policy.money_quantum, rounding=self.policy.rounding,
                        ) - allocated_home
                    else:
                        try:
                            amount = self.exchange.convert(raw_amount, currency, profile.home_currency, when)
                        except Exception:
                            break
                        amount = amount.quantize(self.policy.money_quantum, rounding=self.policy.rounding)
                    allocated_home += amount
                    flows.append(CashFlow(
                        when, -amount, "recurring:variable", group[-1].event_id, True,
                    ))
                month = _add_month(month)
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
        next_date = (
            _add_month(_event_cash_date(anchor))
            if scheduled else _next_supported_month(regular)
        )
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
