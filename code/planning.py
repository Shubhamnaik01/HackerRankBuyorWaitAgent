from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations, product

from .exchange import ExchangeRateTable
from .forecast import Forecast
from .models import CashFlow, FinancialEvent, PaymentOption, Request, UserProfile


AFFORDABILITY_STATUSES = frozenset({
    "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable",
})
PAYMENT_METHODS = frozenset({
    "full_payment", "partial_payment", "installments", "wait", "not_recommended",
})


@dataclass(frozen=True)
class Payment:
    payment_date: date
    amount: Decimal


@dataclass(frozen=True)
class SpendingChange:
    action: str
    event_id: str
    new_amount: Decimal | None = None

    def serialize(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        assert self.new_amount is not None
        return f"reduce_to:{self.event_id}:{self.new_amount}"


@dataclass(frozen=True)
class CandidatePlan:
    method: str
    payments: tuple[Payment, ...]
    total_paid: Decimal
    payment_option_id: str | None = None
    changes: tuple[SpendingChange, ...] = ()

    @property
    def start_date(self) -> date:
        return self.payments[0].payment_date

    @property
    def completion_date(self) -> date:
        return self.payments[-1].payment_date


@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str
    selected_candidate: CandidatePlan | None = None
    warnings: tuple[str, ...] = ()


def format_plan_amount(amount: Decimal) -> str:
    if amount == amount.to_integral_value():
        return format(amount, ".0f")
    return format(amount.quantize(Decimal("0.01")), ".2f")


def serialize_payments(payments: tuple[Payment, ...]) -> str:
    if not payments:
        return "none"
    return "|".join(
        f"{payment.payment_date.isoformat()}:{format_plan_amount(payment.amount)}"
        for payment in payments
    )


def serialize_changes(changes: tuple[SpendingChange, ...]) -> str:
    return "|".join(change.serialize() for change in changes) if changes else "none"


def _option_sort_key(value: str | None) -> tuple[int, int, str]:
    if value is None:
        return (2, 0, "")
    match = re.search(r"(\d+)$", value)
    return (0, int(match.group(1)), value) if match else (1, 0, value)


class PaymentPlanner:
    """Generate, verify and rank plans; the forecast remains authoritative."""

    def __init__(self, exchange: ExchangeRateTable):
        self.exchange = exchange

    def decide(
        self,
        request: Request,
        profile: UserProfile,
        forecast: Forecast,
        amount_safe_to_pay: Decimal,
        earliest_date_for_full_payment: date | None,
        options: tuple[PaymentOption, ...],
        events_by_id: dict[str, FinancialEvent],
        warnings: tuple[str, ...] = (),
    ) -> DecisionResult:
        templates = self._candidate_templates(
            request, profile, amount_safe_to_pay,
            earliest_date_for_full_payment, forecast.end, options,
        )
        safe = [candidate for candidate in templates if self._is_safe(forecast, candidate)]
        if not safe:
            for changes in self._change_sets(profile, forecast, events_by_id):
                changed_forecast = self._apply_changes(forecast, changes, events_by_id, profile)
                for template in templates:
                    if template.method == "wait":
                        continue
                    candidate = replace(template, changes=changes)
                    if self._is_safe(changed_forecast, candidate):
                        safe.append(candidate)
        selected = min(safe, key=self._rank_key) if safe else None
        return self._result(
            request, profile, amount_safe_to_pay,
            earliest_date_for_full_payment, selected, warnings,
        )

    def _candidate_templates(
        self,
        request: Request,
        profile: UserProfile,
        amount_safe: Decimal,
        earliest: date | None,
        forecast_end: date,
        options: tuple[PaymentOption, ...],
    ) -> list[CandidatePlan]:
        result: list[CandidatePlan] = []
        full_option_ids = sorted(
            (option.payment_option_id for option in options if option.payment_method == "full_payment"),
            key=_option_sort_key,
        )
        full_option_id = full_option_ids[0] if full_option_ids else None
        if "full_payment" in profile.accepted_payment_methods:
            result.append(CandidatePlan(
                "full_payment", (Payment(request.request_date, request.requested_amount),),
                request.requested_amount, full_option_id,
            ))
            if (
                earliest is not None and earliest > request.request_date
                and earliest <= request.desired_completion_date
            ):
                result.append(CandidatePlan(
                    "wait", (Payment(earliest, request.requested_amount),),
                    request.requested_amount, full_option_id,
                ))
        if (
            request.allows_partial_payment
            and "partial_payment" in profile.accepted_payment_methods
            and Decimal("0") < amount_safe < request.requested_amount
            and earliest is not None
            and earliest <= request.desired_completion_date
        ):
            result.append(CandidatePlan(
                "partial_payment",
                (
                    Payment(request.request_date, amount_safe),
                    Payment(earliest, request.requested_amount - amount_safe),
                ),
                request.requested_amount,
            ))
        if (
            "installments" in profile.accepted_payment_methods
            and profile.max_installment_months is not None
        ):
            for option in options:
                if option.payment_method != "installments":
                    continue
                candidate = self._installment_candidate(
                    request, profile, forecast_end=forecast_end, option=option,
                )
                if candidate is not None:
                    result.append(candidate)
        return result

    @staticmethod
    def _installment_candidate(
        request: Request, profile: UserProfile, forecast_end: date, option: PaymentOption,
    ) -> CandidatePlan | None:
        if (
            option.number_of_payments <= 0
            or option.number_of_payments > (profile.max_installment_months or 0)
            or option.payment_frequency_days is None
            or option.payment_frequency_days <= 0
            or option.first_payment_date < request.request_date
        ):
            return None
        payments = tuple(
            Payment(
                option.first_payment_date + timedelta(days=index * option.payment_frequency_days),
                option.payment_amount,
            )
            for index in range(option.number_of_payments)
        )
        if payments[-1].payment_date > request.desired_completion_date or payments[-1].payment_date > forecast_end:
            return None
        if sum((payment.amount for payment in payments), Decimal("0")) != option.total_payable_amount:
            return None
        if option.total_payable_amount != request.requested_amount + option.financing_fee:
            return None
        return CandidatePlan(
            "installments", payments, option.total_payable_amount, option.payment_option_id,
        )

    @staticmethod
    def _is_safe(forecast: Forecast, candidate: CandidatePlan) -> bool:
        return forecast.is_schedule_safe(tuple(
            (payment.payment_date, payment.amount) for payment in candidate.payments
        ))

    @staticmethod
    def _rank_key(candidate: CandidatePlan) -> tuple:
        return (
            bool(candidate.changes),
            candidate.total_paid,
            candidate.start_date,
            len(candidate.payments),
            _option_sort_key(candidate.payment_option_id),
        )

    @staticmethod
    def _change_sets(
        profile: UserProfile, forecast: Forecast, events_by_id: dict[str, FinancialEvent],
    ) -> list[tuple[SpendingChange, ...]]:
        source_ids = sorted({
            flow.source_id for flow in forecast.flows
            if flow.inferred and flow.amount < 0 and flow.source_id is not None
        })
        choices: list[tuple[SpendingChange, ...]] = []
        for event_id in source_ids:
            event = events_by_id.get(event_id)
            if event is None or event.category in profile.protected_categories:
                continue
            event_choices: list[SpendingChange] = []
            if (
                event.category in profile.stoppable_categories
                and event.flexibility in {"stoppable", "reducible_or_stoppable"}
            ):
                event_choices.append(SpendingChange("stop", event_id))
            if (
                event.category in profile.reducible_categories
                and event.flexibility in {"reducible", "reducible_or_stoppable"}
                and event.minimum_allowed_amount is not None
                and event.amount is not None
                and Decimal("0") <= event.minimum_allowed_amount
                and event.minimum_allowed_amount < event.amount
            ):
                event_choices.append(SpendingChange(
                    "reduce_to", event_id, event.minimum_allowed_amount,
                ))
            if event_choices:
                choices.append(tuple(event_choices))

        result: list[tuple[SpendingChange, ...]] = []
        for size in range(1, min(3, len(choices)) + 1):
            for selected_events in combinations(choices, size):
                for actions in product(*selected_events):
                    result.append(tuple(sorted(actions, key=lambda change: change.event_id)))
        return result

    def _apply_changes(
        self,
        forecast: Forecast,
        changes: tuple[SpendingChange, ...],
        events_by_id: dict[str, FinancialEvent],
        profile: UserProfile,
    ) -> Forecast:
        by_id = {change.event_id: change for change in changes}
        flows: list[CashFlow] = []
        for flow in forecast.flows:
            change = by_id.get(flow.source_id or "")
            if change is None or not flow.inferred or flow.amount >= 0:
                flows.append(flow)
                continue
            if change.action == "stop":
                continue
            event = events_by_id[change.event_id]
            assert change.new_amount is not None
            amount = self.exchange.convert(
                change.new_amount, event.currency, profile.home_currency, flow.flow_date,
            )
            flows.append(replace(flow, amount=-amount))
        return Forecast(
            forecast.start, forecast.starting_balance, forecast.minimum_balance,
            flows, forecast.policy,
        )

    def _result(
        self,
        request: Request,
        profile: UserProfile,
        amount_safe: Decimal,
        earliest: date | None,
        selected: CandidatePlan | None,
        warnings: tuple[str, ...],
    ) -> DecisionResult:
        if selected is None:
            status = "not_affordable"
            method = "not_recommended"
            payments = ()
            changes = ()
        else:
            method = selected.method
            payments = selected.payments
            changes = selected.changes
            if method == "full_payment" and not changes:
                status = "affordable_now"
            elif method == "wait" and not changes:
                status = "affordable_later"
            else:
                status = "affordable_with_plan"
        explanation = self._explain(request, profile, amount_safe, selected)
        return DecisionResult(
            request.request_id, amount_safe, status, method,
            serialize_payments(payments), earliest,
            serialize_changes(changes), explanation, selected, warnings,
        )

    @staticmethod
    def _money(currency: str, amount: Decimal) -> str:
        if amount == amount.to_integral_value():
            value = f"{amount:,.0f}"
        else:
            value = f"{amount:,.2f}"
        return f"{currency} {value}"

    @classmethod
    def _explain(
        cls, request: Request, profile: UserProfile, amount_safe: Decimal,
        selected: CandidatePlan | None,
    ) -> str:
        floor = cls._money(profile.home_currency, profile.minimum_balance_to_keep)
        requested = cls._money(profile.home_currency, request.requested_amount)
        if selected is None:
            safe_today = cls._money(profile.home_currency, amount_safe)
            return (
                f"Although {safe_today} is safe today, do not proceed with the {requested} request; "
                f"no eligible plan completes it "
                f"by {request.desired_completion_date.isoformat()} while preserving the {floor} minimum."
            )
        if selected.method == "wait":
            return (
                f"Wait until {selected.completion_date.isoformat()}, then pay {requested} in full "
                f"while preserving the {floor} minimum."
            )
        if selected.method == "partial_payment":
            remainder = selected.payments[-1].amount
            return (
                f"Pay {cls._money(profile.home_currency, amount_safe)} today and "
                f"{cls._money(profile.home_currency, remainder)} on "
                f"{selected.completion_date.isoformat()} while preserving the {floor} minimum."
            )
        if selected.method == "installments":
            return (
                f"Use {len(selected.payments)} supplied installments of "
                f"{cls._money(profile.home_currency, selected.payments[0].amount)}, starting "
                f"{selected.start_date.isoformat()}, while preserving the {floor} minimum."
            )
        if selected.changes:
            actions = ", ".join(change.serialize() for change in selected.changes)
            return (
                f"Apply {actions}, then pay {requested} today while preserving the {floor} minimum."
            )
        return f"Pay {requested} today while preserving the {floor} minimum over 90 days."


def validate_decision(result: DecisionResult, request: Request) -> tuple[str, ...]:
    errors: list[str] = []
    if not Decimal("0") <= result.amount_safe_to_pay <= request.requested_amount:
        errors.append("amount_safe_to_pay is outside request bounds")
    if result.affordability_status not in AFFORDABILITY_STATUSES:
        errors.append("invalid affordability_status")
    if result.recommended_payment_method not in PAYMENT_METHODS:
        errors.append("invalid recommended_payment_method")
    if result.selected_candidate is None:
        if result.payment_plan != "none" or result.spending_changes_needed != "none":
            errors.append("fallback decision contains a plan or spending changes")
        if result.affordability_status != "not_affordable":
            errors.append("missing candidate requires not_affordable")
        if result.recommended_payment_method != "not_recommended":
            errors.append("missing candidate requires not_recommended")
    else:
        candidate = result.selected_candidate
        if candidate.method != result.recommended_payment_method:
            errors.append("selected candidate method mismatch")
        if not candidate.payments:
            errors.append("selected candidate has no payments")
        if tuple(sorted(candidate.payments, key=lambda payment: payment.payment_date)) != candidate.payments:
            errors.append("selected payments are not chronological")
        if sum((payment.amount for payment in candidate.payments), Decimal("0")) != candidate.total_paid:
            errors.append("selected payment total mismatch")
        if any(payment.amount <= 0 for payment in candidate.payments):
            errors.append("selected candidate has non-positive payment")
        if candidate.completion_date > request.desired_completion_date:
            errors.append("selected plan misses desired_completion_date")
        if serialize_payments(candidate.payments) != result.payment_plan:
            errors.append("payment_plan serialization mismatch")
        if serialize_changes(candidate.changes) != result.spending_changes_needed:
            errors.append("spending changes serialization mismatch")
        expected_status = (
            "affordable_now"
            if candidate.method == "full_payment" and not candidate.changes
            else "affordable_later"
            if candidate.method == "wait" and not candidate.changes
            else "affordable_with_plan"
        )
        if result.affordability_status != expected_status:
            errors.append("affordability status does not match selected candidate")
        if candidate.method in {"full_payment", "wait", "partial_payment"}:
            if sum((payment.amount for payment in candidate.payments), Decimal("0")) != request.requested_amount:
                errors.append("non-installment plan does not complete requested amount")
        if candidate.method == "partial_payment" and len(candidate.payments) != 2:
            errors.append("partial payment requires exactly two payments")
    if result.affordability_status == "affordable_now":
        if result.earliest_date_for_full_payment != request.request_date:
            errors.append("affordable_now requires earliest full payment on request_date")
    if not result.decision_explanation.strip():
        errors.append("decision_explanation is empty")
    return tuple(errors)
