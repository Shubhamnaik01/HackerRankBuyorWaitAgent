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
        return flow.flow_date, flow.source, flow.source_id or ""

    def balances_for_payments(
        self, payments: tuple[tuple[date, Decimal], ...] = (),
    ) -> list[tuple[date, Decimal, str]]:
        normal_by_date: dict[date, Decimal] = {}
        for flow in self.flows:
            normal_by_date[flow.flow_date] = normal_by_date.get(flow.flow_date, Decimal("0")) + flow.amount

        payments_by_date: dict[date, list[Decimal]] = {}
        for when, amount in payments:
            if amount:
                payments_by_date.setdefault(when, []).append(amount)

        balance = self.starting_balance
        path = [(self.start, balance, "opening-balance")]
        for flow_date in sorted(normal_by_date.keys() | payments_by_date.keys()):
            if flow_date in normal_by_date:
                balance += normal_by_date[flow_date]
                path.append((flow_date, balance, "normal-daily-net"))
            for amount in payments_by_date.get(flow_date, ()):
                balance -= amount
                path.append((flow_date, balance, "candidate-payment"))
        return path

    def balances(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> list[tuple[date, Decimal, str]]:
        payments = ((payment_date, payment_amount),) if payment_date is not None and payment_amount else ()
        return self.balances_for_payments(payments)

    def is_schedule_safe(self, payments: tuple[tuple[date, Decimal], ...]) -> bool:
        return min(balance for _, balance, _ in self.balances_for_payments(payments)) >= self.minimum_balance

    def minimum_projected_balance(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> Decimal:
        return min(balance for _, balance, _ in self.balances(payment_date, payment_amount))

    def is_safe(self, payment_date: date | None = None, payment_amount: Decimal = Decimal("0")) -> bool:
        return self.minimum_projected_balance(payment_date, payment_amount) >= self.minimum_balance

    def maximum_safe_payment(self, payment_date: date, cap: Decimal) -> Decimal:
        if not self.is_safe():
            return Decimal("0")
        path = self.balances()
        balance_at_payment = self.starting_balance
        for when, balance, _ in path:
            if when > payment_date:
                break
            balance_at_payment = balance
        future = [balance_at_payment]
        future.extend(balance for when, balance, _ in path if when > payment_date)
        available = min(future) - self.minimum_balance
        safe = max(Decimal("0"), min(cap, available))
        return safe.quantize(self.policy.money_quantum, rounding=self.policy.rounding)

    def earliest_safe_full_payment(self, amount: Decimal) -> date | None:
        candidate = self.start
        while candidate <= self.end:
            if self.is_safe(candidate, amount):
                return candidate
            candidate += timedelta(days=1)
        return None
