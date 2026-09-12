from __future__ import annotations

from datetime import date
from decimal import Decimal

from .config import ForecastPolicy
from .models import ExchangeRate


class ExchangeRateError(ValueError):
    pass


class ExchangeRateTable:
    def __init__(self, rates: tuple[ExchangeRate, ...] | list[ExchangeRate], policy: ForecastPolicy):
        self.policy = policy
        self._rates = {(rate.rate_date, rate.from_currency, rate.to_currency): rate.rate for rate in rates}

    def convert(self, amount: Decimal, from_currency: str, to_currency: str, settlement_date: date) -> Decimal:
        if from_currency == to_currency:
            return amount
        key = (settlement_date, from_currency, to_currency)
        if key not in self._rates:
            raise ExchangeRateError(f"missing direct exchange rate for {from_currency}->{to_currency} on {settlement_date}")
        return (amount * self._rates[key]).quantize(self.policy.money_quantum, rounding=self.policy.rounding)

