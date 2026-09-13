from __future__ import annotations

from datetime import timedelta

from .config import ForecastPolicy, inclusive_horizon_end
from .data_loader import DatasetBundle
from .exchange import ExchangeRateTable
from .evidence.application import apply_evidence_to_flows
from .evidence.service import EvidenceService
from .forecast import Forecast
from .models import CoreResult, Request
from .normalization import explicit_future_flows
from .recurrence import RecurrenceDetector, remove_explicit_collisions


class DeterministicFinancialCore:
    def __init__(
        self, data: DatasetBundle, policy: ForecastPolicy | None = None,
        evidence: EvidenceService | None = None,
    ):
        self.data = data
        self.policy = policy or ForecastPolicy()
        self.exchange = ExchangeRateTable(data.exchange_rates, self.policy)
        self.recurrence = RecurrenceDetector(self.policy, self.exchange)
        self.evidence = evidence or EvidenceService(data)

    def build_forecast(self, request: Request) -> tuple[Forecast, tuple[str, ...]]:
        profile = self.data.profiles[request.user_id]
        events = self.data.events_by_user.get(request.user_id, ())
        end = inclusive_horizon_end(request.request_date, self.policy.horizon_days)
        evidence = self.evidence.resolve(
            request, self.policy.recurrence.history_days, self.policy.horizon_days,
        )
        amount_overrides = {
            event_id: value.amount for event_id, value in evidence.image_amounts.items()
        }
        explicit, warnings = explicit_future_flows(
            profile, events, request.request_date, end, self.exchange, amount_overrides,
        )
        warnings.extend(evidence.warnings)
        history_start = request.request_date - timedelta(days=self.policy.recurrence.history_days)
        warned_ids = {warning.split(":", 1)[0] for warning in warnings}
        for event in events:
            cash_date = event.settlement_date or event.event_date
            if (
                event.amount is None and event.event_id not in amount_overrides
                and history_start <= cash_date <= end and event.event_id not in warned_ids
            ):
                warnings.append(f"{event.event_id}: missing amount; linked image evidence not interpreted")
        inferred = self.recurrence.infer(
            profile, events, request.request_date, end, amount_overrides,
        )
        inferred = remove_explicit_collisions(inferred, explicit, self.data.events_by_id)
        combined, evidence_warnings = apply_evidence_to_flows(
            explicit + inferred, evidence.facts, profile, self.data.events_by_id,
            self.exchange, request.request_date, end,
        )
        warnings.extend(evidence_warnings)
        forecast = Forecast(
            request.request_date,
            profile.current_available_balance,
            profile.minimum_balance_to_keep,
            combined,
            self.policy,
        )
        return forecast, tuple(warnings)

    def calculate(self, request: Request) -> CoreResult:
        forecast, warnings = self.build_forecast(request)
        amount_safe = forecast.maximum_safe_payment(request.request_date, request.requested_amount)
        earliest = forecast.earliest_safe_full_payment(request.requested_amount)
        return CoreResult(
            request_id=request.request_id,
            amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=earliest,
            baseline_safe=forecast.is_safe(),
            minimum_projected_balance=forecast.minimum_projected_balance(),
            warnings=warnings,
        )
