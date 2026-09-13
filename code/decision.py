from __future__ import annotations

from .core import DeterministicFinancialCore
from .models import Request
from .planning import DecisionResult, PaymentPlanner, validate_decision


class DecisionEngine:
    """Complete deterministic recommendation layer over the financial core."""

    def __init__(self, core: DeterministicFinancialCore):
        self.core = core
        self.planner = PaymentPlanner(core.exchange)

    def decide(self, request: Request) -> DecisionResult:
        forecast, warnings = self.core.build_forecast(request)
        amount_safe = forecast.maximum_safe_payment(request.request_date, request.requested_amount)
        earliest = forecast.earliest_safe_full_payment(request.requested_amount)
        result = self.planner.decide(
            request=request,
            profile=self.core.data.profiles[request.user_id],
            forecast=forecast,
            amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=earliest,
            options=self.core.data.payment_options_by_request.get(request.request_id, ()),
            events_by_id=self.core.data.events_by_id,
            warnings=warnings,
        )
        errors = validate_decision(result, request)
        if errors:
            raise ValueError(f"invalid decision for {request.request_id}: {'; '.join(errors)}")
        return result
