from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from code.config import ForecastPolicy
from code.exchange import ExchangeRateTable
from code.forecast import Forecast
from code.models import CashFlow, ExchangeRate, FinancialEvent, UserProfile
from code.normalization import explicit_future_flows
from code.recurrence import RecurrenceDetector


def profile(balance="1000", minimum="100", currency="USD") -> UserProfile:
    return UserProfile("u", currency, Decimal(balance), Decimal(minimum), (), (), (), (), ("full_payment",), None)


def event(event_id, event_date, amount, *, status="settled", direction="debit", description="Rent", category="rent", event_type="expense", settlement_date=None, currency="USD"):
    return FinancialEvent(
        event_id, "u", event_type, description, category, direction,
        Decimal(amount) if amount is not None else None, currency, event_date,
        settlement_date or event_date, status, None, "fixed", None,
    )


class CoreFinancialTests(unittest.TestCase):
    def setUp(self):
        self.policy = ForecastPolicy()
        self.exchange = ExchangeRateTable([], self.policy)

    def test_current_balance_is_snapshot_and_settled_history_is_not_replayed(self):
        flows, warnings = explicit_future_flows(
            profile(), (event("old", date(2026, 1, 1), "500"),),
            date(2026, 2, 1), date(2026, 5, 2), self.exchange,
        )
        self.assertEqual([], flows)
        self.assertEqual([], warnings)

    def test_pending_debit_is_reserved_but_pending_credit_is_ignored(self):
        events = (
            event("debit", date(2026, 2, 1), "80", status="pending", settlement_date=date(2026, 2, 1)),
            event("credit", date(2026, 2, 1), "500", status="pending", direction="credit", settlement_date=date(2026, 2, 1)),
        )
        flows, _ = explicit_future_flows(profile(), events, date(2026, 2, 1), date(2026, 5, 2), self.exchange)
        self.assertEqual([Decimal("-80")], [flow.amount for flow in flows])

    def test_cancelled_failed_and_unrealized_events_are_ignored(self):
        events = tuple(
            event(status, date(2026, 2, 1), "80", status=status, direction="debit" if status != "unrealized" else "non_cash")
            for status in ("cancelled", "failed", "unrealized")
        )
        flows, _ = explicit_future_flows(profile(), events, date(2026, 2, 1), date(2026, 5, 2), self.exchange)
        self.assertEqual([], flows)

    def test_dated_direct_fx_conversion_uses_decimal_rounding(self):
        table = ExchangeRateTable(
            [ExchangeRate(date(2026, 2, 15), "EUR", "USD", Decimal("1.09"))], self.policy
        )
        self.assertEqual(Decimal("10.89"), table.convert(Decimal("9.99"), "EUR", "USD", date(2026, 2, 15)))

    def test_recurring_salary_is_inferred_monthly(self):
        history = tuple(
            event(f"s{month}", date(2026, month, 15), "500", direction="credit", description="Payroll credit", category="salary", event_type="income")
            for month in (1, 2, 3)
        )
        flows = RecurrenceDetector(self.policy, self.exchange).infer(profile(), history, date(2026, 4, 1), date(2026, 6, 30))
        salary = [flow for flow in flows if flow.amount > 0]
        self.assertEqual([date(2026, 4, 15), date(2026, 5, 15), date(2026, 6, 15)], [flow.flow_date for flow in salary])
        self.assertEqual([Decimal("500")] * 3, [flow.amount for flow in salary])

    def test_recurring_fixed_expense_reserves_high_recent_amount(self):
        history = (
            event("r1", date(2026, 1, 3), "200"),
            event("r2", date(2026, 2, 3), "220"),
            event("r3", date(2026, 3, 3), "210"),
        )
        flows = RecurrenceDetector(self.policy, self.exchange).infer(profile(), history, date(2026, 4, 1), date(2026, 5, 31))
        rent = [flow for flow in flows if flow.source == "recurring:monthly"]
        self.assertEqual([Decimal("-220"), Decimal("-220")], [flow.amount for flow in rent])

    def test_same_day_confirmed_credit_and_debit_are_netted(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("200"), Decimal("100"),
            [CashFlow(date(2026, 1, 2), Decimal("100"), "salary"), CashFlow(date(2026, 1, 2), Decimal("-150"), "bill")],
            self.policy,
        )
        self.assertTrue(forecast.is_safe())
        self.assertEqual(Decimal("150"), forecast.minimum_projected_balance())

    def test_amount_safe_to_pay_uses_full_forecast_and_cap(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("1000"), Decimal("200"),
            [CashFlow(date(2026, 2, 1), Decimal("-350"), "rent")], self.policy,
        )
        self.assertEqual(Decimal("450"), forecast.maximum_safe_payment(date(2026, 1, 1), Decimal("900")))
        self.assertEqual(Decimal("300"), forecast.maximum_safe_payment(date(2026, 1, 1), Decimal("300")))

    def test_earliest_full_payment_checks_entire_horizon(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("500"), Decimal("100"),
            [CashFlow(date(2026, 1, 10), Decimal("300"), "salary"), CashFlow(date(2026, 2, 1), Decimal("-100"), "bill")],
            self.policy,
        )
        self.assertEqual(date(2026, 1, 10), forecast.earliest_safe_full_payment(Decimal("500")))

    def test_candidate_payment_follows_same_day_normal_net(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("200"), Decimal("100"),
            [
                CashFlow(date(2026, 1, 2), Decimal("100"), "confirmed-credit"),
                CashFlow(date(2026, 1, 2), Decimal("-50"), "mandatory-debit"),
            ],
            self.policy,
        )
        self.assertEqual(date(2026, 1, 1), forecast.earliest_safe_full_payment(Decimal("100")))
        self.assertEqual(
            [Decimal("250"), Decimal("150")],
            [balance for when, balance, _ in forecast.balances(date(2026, 1, 2), Decimal("100")) if when == date(2026, 1, 2)],
        )

    def test_same_day_salary_can_fund_payment_after_normal_activity(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("150"), Decimal("100"),
            [CashFlow(date(2026, 1, 1), Decimal("100"), "confirmed-salary")],
            self.policy,
        )
        self.assertEqual(Decimal("150.00"), forecast.maximum_safe_payment(date(2026, 1, 1), Decimal("150")))
        self.assertTrue(forecast.is_safe(date(2026, 1, 1), Decimal("150")))

    def test_later_credit_cannot_fund_payment_today(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("150"), Decimal("100"),
            [CashFlow(date(2026, 1, 2), Decimal("100"), "confirmed-credit")],
            self.policy,
        )
        self.assertEqual(Decimal("50.00"), forecast.maximum_safe_payment(date(2026, 1, 1), Decimal("150")))
        self.assertFalse(forecast.is_safe(date(2026, 1, 1), Decimal("150")))
        self.assertEqual(date(2026, 1, 2), forecast.earliest_safe_full_payment(Decimal("150")))

    def test_same_day_net_and_candidate_must_still_preserve_minimum(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("200"), Decimal("100"),
            [
                CashFlow(date(2026, 1, 2), Decimal("100"), "confirmed-credit"),
                CashFlow(date(2026, 1, 2), Decimal("-150"), "mandatory-debit"),
            ],
            self.policy,
        )
        self.assertFalse(forecast.is_safe(date(2026, 1, 2), Decimal("60")))
        self.assertEqual(Decimal("90"), forecast.minimum_projected_balance(date(2026, 1, 2), Decimal("60")))

    def test_safety_is_enforced_on_later_forecast_dates(self):
        forecast = Forecast(
            date(2026, 1, 1), Decimal("200"), Decimal("100"),
            [
                CashFlow(date(2026, 1, 2), Decimal("100"), "confirmed-credit"),
                CashFlow(date(2026, 1, 2), Decimal("-50"), "mandatory-debit"),
                CashFlow(date(2026, 2, 1), Decimal("-160"), "later-essential"),
            ],
            self.policy,
        )
        self.assertFalse(forecast.is_safe(date(2026, 1, 2), Decimal("40")))
        self.assertEqual(Decimal("50"), forecast.minimum_projected_balance(date(2026, 1, 2), Decimal("40")))


if __name__ == "__main__":
    unittest.main()
