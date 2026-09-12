from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .config import ForecastPolicy
from .models import CashFlow


class Forecast:
    def __init__(self, start: date, starting_balance: Decimal, minimum_balance: Decimal, flows: list[CashFlow], policy: ForecastPolicy):
        self.start = start
        self.end = start + timedelta(days=policy.horizon_days)
        self.starting_balance = starting_balance
        self.minimum_balance = minimum_balance
        self.policy = policy
        self.flows = sorted(flows, key=self._sort_key)

    @staticmethod
    def _sort_key(flow: CashFlow) -> tuple:
        if flow.source == "candidate-payment":
            same_day_order = 2
        else:
            same_day_order = 0 if flow.amount < 0 else 1
        return flow.flow_date, same_day_order, flow.source, flow.source_id or ""

    def balances(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> list[tuple[date, Decimal, str]]:
        entries = list(self.flows)
        if payment_date is not None and payment_amount:
            entries.append(CashFlow(payment_date, -payment_amount, "candidate-payment"))
        entries.sort(key=self._sort_key)
        balance = self.starting_balance
        path = [(self.start, balance, "opening-balance")]
        for flow in entries:
            balance += flow.amount
            path.append((flow.flow_date, balance, flow.source))
        return path

    def minimum_projected_balance(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> Decimal:
        return min(balance for _, balance, _ in self.balances(payment_date, payment_amount))

    def is_safe(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> bool:
        return self.minimum_projected_balance(payment_date, payment_amount) >= self.minimum_balance

    def maximum_safe_payment(self, payment_date: date, cap: Decimal) -> Decimal:
        if not self.is_safe():
            return Decimal("0")
        path = self.balances()
        prior = [balance for when, balance, _ in path if when < payment_date]
        if prior and min(prior) < self.minimum_balance:
            return Decimal("0")
        future = [balance for when, balance, _ in path if when >= payment_date]
        available = min(future or [self.starting_balance]) - self.minimum_balance
        return max(Decimal("0"), min(cap, available))

    def earliest_safe_full_payment(self, amount: Decimal) -> date | None:
        candidate = self.start
        while candidate <= self.end:
            if self.is_safe(candidate, amount):
                return candidate
            candidate += timedelta(days=1)
        return None
