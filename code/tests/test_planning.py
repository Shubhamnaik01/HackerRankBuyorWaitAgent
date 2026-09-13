from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal

from code.config import ForecastPolicy
from code.exchange import ExchangeRateTable
from code.forecast import Forecast
from code.models import CashFlow, FinancialEvent, PaymentOption, Request, UserProfile
from code.planning import (
    CandidatePlan,
    Payment,
    PaymentPlanner,
    SpendingChange,
    serialize_changes,
    serialize_payments,
    validate_decision,
)


START = date(2026, 9, 1)


def make_profile(
    methods=("full_payment",), *, protected=(), reducible=(), stoppable=(),
    max_months=None,
) -> UserProfile:
    return UserProfile(
        "user", "USD", Decimal("300"), Decimal("100"), (), protected,
        reducible, stoppable, methods, max_months,
    )


def make_request(
    amount="150", *, deadline_days=60, allows_partial=True,
) -> Request:
    return Request(
        "request", "user", START, "purchase", Decimal(amount),
        START + timedelta(days=deadline_days), allows_partial, "Test request",
    )


def make_forecast(balance="300", minimum="100", flows=()) -> Forecast:
    return Forecast(
        START, Decimal(balance), Decimal(minimum), list(flows), ForecastPolicy(),
    )


def make_option(
    option_id="option_1", *, amount="50", count=3, first_days=0,
    frequency=30, fee="0", total="150",
) -> PaymentOption:
    return PaymentOption(
        option_id, "request", "installments", Decimal(amount), count,
        START + timedelta(days=first_days), frequency, Decimal(fee), Decimal(total),
    )


def make_event(
    event_id="expense", *, flexibility="reducible", minimum="50",
    category="dining",
) -> FinancialEvent:
    return FinancialEvent(
        event_id, "user", "expense", "Recurring flexible expense", category,
        "debit", Decimal("100"), "USD", date(2026, 8, 1),
        date(2026, 8, 1), "settled", None, flexibility,
        Decimal(minimum) if minimum is not None else None,
    )


class PaymentPlanningTests(unittest.TestCase):
    def setUp(self):
        self.planner = PaymentPlanner(ExchangeRateTable([], ForecastPolicy()))

    def decide(
        self, *, request=None, profile=None, forecast=None, amount_safe="150",
        earliest=START, options=(), events=(),
    ):
        request = request or make_request()
        profile = profile or make_profile()
        forecast = forecast or make_forecast()
        return self.planner.decide(
            request, profile, forecast, Decimal(amount_safe), earliest,
            tuple(options), {event.event_id: event for event in events},
        )

    def test_safe_full_payment(self):
        result = self.decide()
        self.assertEqual("affordable_now", result.affordability_status)
        self.assertEqual("full_payment", result.recommended_payment_method)
        self.assertEqual("2026-09-01:150", result.payment_plan)
        self.assertEqual((), validate_decision(result, make_request()))

    def test_unsafe_full_payment_is_not_recommended(self):
        result = self.decide(
            forecast=make_forecast(balance="200"), amount_safe="100", earliest=None,
        )
        self.assertEqual("not_affordable", result.affordability_status)
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_wait_until_future_full_payment_date(self):
        payday = START + timedelta(days=10)
        result = self.decide(
            forecast=make_forecast(
                balance="150", flows=(CashFlow(payday, Decimal("100"), "salary"),),
            ),
            amount_safe="50", earliest=payday,
        )
        self.assertEqual("affordable_later", result.affordability_status)
        self.assertEqual("wait", result.recommended_payment_method)
        self.assertEqual("2026-09-11:150", result.payment_plan)

    def test_partial_payment_uses_exact_two_payment_rule(self):
        payday = START + timedelta(days=10)
        result = self.decide(
            profile=make_profile(methods=("partial_payment",)),
            forecast=make_forecast(
                balance="200", flows=(CashFlow(payday, Decimal("100"), "salary"),),
            ),
            amount_safe="40", earliest=payday,
        )
        self.assertEqual("affordable_with_plan", result.affordability_status)
        self.assertEqual("partial_payment", result.recommended_payment_method)
        self.assertEqual("2026-09-01:40|2026-09-11:110", result.payment_plan)

    def test_partial_payment_disallowed(self):
        result = self.decide(
            request=make_request(allows_partial=False),
            profile=make_profile(methods=("partial_payment",)),
            amount_safe="40", earliest=START + timedelta(days=10),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_installment_dates_use_exact_28_30_and_31_day_intervals(self):
        for frequency, expected_last in ((28, date(2026, 10, 27)), (30, date(2026, 10, 31)), (31, date(2026, 11, 2))):
            with self.subTest(frequency=frequency):
                result = self.decide(
                    request=make_request(deadline_days=90),
                    profile=make_profile(methods=("installments",), max_months=3),
                    forecast=make_forecast(balance="1000"),
                    options=(make_option(frequency=frequency),),
                )
                self.assertEqual(expected_last, result.selected_candidate.completion_date)

    def test_installment_deadline_violation_is_rejected(self):
        result = self.decide(
            request=make_request(deadline_days=40),
            profile=make_profile(methods=("installments",), max_months=3),
            forecast=make_forecast(balance="1000"), options=(make_option(),),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_max_installment_month_violation_is_rejected(self):
        result = self.decide(
            profile=make_profile(methods=("installments",), max_months=2),
            forecast=make_forecast(balance="1000"), options=(make_option(count=3),),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_full_payment_not_accepted_can_select_installments(self):
        result = self.decide(
            profile=make_profile(methods=("installments",), max_months=3),
            forecast=make_forecast(balance="1000"), options=(make_option(),),
        )
        self.assertEqual(Decimal("150"), result.amount_safe_to_pay)
        self.assertEqual("installments", result.recommended_payment_method)

    def test_candidate_with_future_minimum_balance_breach_is_rejected(self):
        result = self.decide(
            profile=make_profile(methods=("installments",), max_months=3),
            forecast=make_forecast(
                balance="400",
                flows=(CashFlow(START + timedelta(days=45), Decimal("-200"), "rent"),),
            ),
            options=(make_option(),),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_reduction_can_enable_plan(self):
        expense = make_event()
        result = self.decide(
            forecast=make_forecast(flows=(
                CashFlow(START + timedelta(days=5), Decimal("-100"), "recurring:monthly", "expense", True),
            )),
            amount_safe="100", earliest=None,
            profile=make_profile(reducible=("dining",)), events=(expense,),
        )
        self.assertEqual("full_payment", result.recommended_payment_method)
        self.assertEqual("affordable_with_plan", result.affordability_status)
        self.assertEqual("reduce_to:expense:50", result.spending_changes_needed)

    def test_stopping_expense_can_enable_plan(self):
        expense = make_event(flexibility="stoppable", minimum=None)
        result = self.decide(
            forecast=make_forecast(flows=(
                CashFlow(START + timedelta(days=5), Decimal("-100"), "recurring:monthly", "expense", True),
            )),
            amount_safe="100", earliest=None,
            profile=make_profile(stoppable=("dining",)), events=(expense,),
        )
        self.assertEqual("stop:expense", result.spending_changes_needed)

    def test_protected_expense_cannot_be_changed(self):
        expense = make_event()
        result = self.decide(
            forecast=make_forecast(flows=(
                CashFlow(START + timedelta(days=5), Decimal("-100"), "recurring:monthly", "expense", True),
            )),
            amount_safe="100", earliest=None,
            profile=make_profile(protected=("dining",), reducible=("dining",)),
            events=(expense,),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)

    def test_safe_no_change_plan_wins_without_generating_changes(self):
        expense = make_event()
        result = self.decide(
            forecast=make_forecast(flows=(
                CashFlow(START + timedelta(days=5), Decimal("-20"), "recurring:monthly", "expense", True),
            )),
            profile=make_profile(reducible=("dining",)), events=(expense,),
        )
        self.assertEqual("full_payment", result.recommended_payment_method)
        self.assertEqual("none", result.spending_changes_needed)

    def test_lower_total_cost_ranks_first(self):
        profile = make_profile(methods=("installments",), max_months=3)
        result = self.decide(
            request=make_request(deadline_days=90),
            profile=profile, forecast=make_forecast(balance="1000"),
            options=(
                make_option("option_1", amount="55", fee="15", total="165"),
                make_option("option_2", amount="52", fee="6", total="156"),
            ),
        )
        self.assertEqual("option_2", result.selected_candidate.payment_option_id)

    def test_earlier_start_ranks_first(self):
        profile = make_profile(methods=("installments",), max_months=3)
        result = self.decide(
            request=make_request(deadline_days=90),
            profile=profile, forecast=make_forecast(balance="1000"),
            options=(
                make_option("option_1", first_days=2),
                make_option("option_2", first_days=0),
            ),
        )
        self.assertEqual("option_2", result.selected_candidate.payment_option_id)

    def test_fewer_payments_rank_first(self):
        profile = make_profile(methods=("installments",), max_months=3)
        result = self.decide(
            profile=profile, forecast=make_forecast(balance="1000"),
            options=(
                make_option("option_1", amount="50", count=3, frequency=20),
                make_option("option_2", amount="75", count=2, frequency=40),
            ),
        )
        self.assertEqual("option_2", result.selected_candidate.payment_option_id)

    def test_option_id_is_final_tie_break(self):
        profile = make_profile(methods=("installments",), max_months=3)
        result = self.decide(
            profile=profile, forecast=make_forecast(balance="1000"),
            options=(make_option("option_9"), make_option("option_2")),
        )
        self.assertEqual("option_2", result.selected_candidate.payment_option_id)

    def test_positive_safe_amount_does_not_imply_affordable(self):
        result = self.decide(
            request=make_request(allows_partial=False),
            profile=make_profile(methods=("partial_payment",)),
            amount_safe="40", earliest=None,
        )
        self.assertEqual(Decimal("40"), result.amount_safe_to_pay)
        self.assertEqual("not_affordable", result.affordability_status)

    def test_complete_serialization(self):
        payments = (
            Payment(START, Decimal("40")),
            Payment(START + timedelta(days=30), Decimal("110.50")),
        )
        changes = (
            SpendingChange("stop", "event_2"),
            SpendingChange("reduce_to", "event_3", Decimal("20.00")),
        )
        self.assertEqual("2026-09-01:40|2026-10-01:110.50", serialize_payments(payments))
        self.assertEqual("stop:event_2|reduce_to:event_3:20.00", serialize_changes(changes))

    def test_invalid_installment_totals_are_rejected(self):
        result = self.decide(
            profile=make_profile(methods=("installments",), max_months=3),
            forecast=make_forecast(balance="1000"),
            options=(make_option(amount="49", total="150"),),
        )
        self.assertEqual("not_recommended", result.recommended_payment_method)


if __name__ == "__main__":
    unittest.main()
