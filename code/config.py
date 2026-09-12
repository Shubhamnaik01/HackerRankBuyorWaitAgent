from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP


@dataclass(frozen=True)
class RecurrencePolicy:
    history_days: int = 180
    minimum_occurrences: int = 3
    monthly_min_gap_days: int = 25
    monthly_max_gap_days: int = 35
    monthly_recency_days: int = 45
    variable_minimum_occurrences: int = 4
    variable_window_days: int = 28
    variable_window_count: int = 3
    variable_projection_interval_days: int = 7
    essential_variable_categories: tuple[str, ...] = (
        "groceries", "transport", "healthcare", "utilities",
    )
    amount_lookback: int = 3


@dataclass(frozen=True)
class ForecastPolicy:
    horizon_days: int = 90
    # Mandatory debits are applied before confirmed credits on the same date.
    # An elective candidate payment is applied after both, so confirmed money
    # settling that day can fund it, as demonstrated by the solved examples.
    debits_before_credits: bool = True
    money_quantum: Decimal = Decimal("0.01")
    rounding: str = ROUND_HALF_UP
    recurrence: RecurrencePolicy = field(default_factory=RecurrencePolicy)
