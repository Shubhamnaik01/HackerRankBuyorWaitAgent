from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.config import ForecastPolicy
from code.data_loader import DatasetBundle, DatasetLoader
from code.exchange import ExchangeRateTable
from code.models import PaymentOption, Request, UserProfile
from code.planning import AFFORDABILITY_STATUSES, PAYMENT_METHODS
from code.recurrence import RecurrenceDetector


REQUIRED_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


class SubmissionValidationError(ValueError):
    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(errors)
        super().__init__("submission validation failed:\n" + "\n".join(f"- {error}" for error in self.errors))


def _decimal(value: str, request_id: str, field: str, errors: list[str]) -> Decimal | None:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        errors.append(f"{request_id}: {field} is not a valid Decimal: {value!r}")
        return None
    if not result.is_finite():
        errors.append(f"{request_id}: {field} must be finite")
        return None
    return result


def _iso_date(value: str, request_id: str, field: str, errors: list[str]) -> date | None:
    try:
        result = date.fromisoformat(value)
    except ValueError:
        errors.append(f"{request_id}: {field} is not a valid ISO date: {value!r}")
        return None
    if result.isoformat() != value:
        errors.append(f"{request_id}: {field} must use YYYY-MM-DD format")
        return None
    return result


def _parse_plan(value: str, request_id: str, errors: list[str]) -> tuple[tuple[date, Decimal], ...] | None:
    if value == "none":
        return ()
    if not value:
        errors.append(f"{request_id}: payment_plan must be 'none' or contain payments")
        return None
    payments: list[tuple[date, Decimal]] = []
    for entry in value.split("|"):
        if entry.count(":") != 1:
            errors.append(f"{request_id}: malformed payment_plan entry: {entry!r}")
            return None
        date_text, amount_text = entry.split(":", 1)
        when = _iso_date(date_text, request_id, "payment date", errors)
        amount = _decimal(amount_text, request_id, "payment amount", errors)
        if when is None or amount is None:
            return None
        if amount <= 0:
            errors.append(f"{request_id}: payment amounts must be positive")
        payments.append((when, amount))
    if [when for when, _ in payments] != sorted(when for when, _ in payments):
        errors.append(f"{request_id}: payment_plan dates are not chronological")
    return tuple(payments)


def _matching_installment(
    payments: tuple[tuple[date, Decimal], ...], options: tuple[PaymentOption, ...],
) -> PaymentOption | None:
    for option in options:
        if option.payment_method != "installments" or option.payment_frequency_days is None:
            continue
        expected = tuple(
            (
                option.first_payment_date + timedelta(days=index * option.payment_frequency_days),
                option.payment_amount,
            )
            for index in range(option.number_of_payments)
        )
        if payments == expected and sum((amount for _, amount in payments), Decimal("0")) == option.total_payable_amount:
            return option
    return None


def _supported_change_ids(data: DatasetBundle, request: Request) -> set[str]:
    policy = ForecastPolicy()
    detector = RecurrenceDetector(policy, ExchangeRateTable(data.exchange_rates, policy))
    flows = detector.infer(
        data.profiles[request.user_id], data.events_by_user.get(request.user_id, ()),
        request.request_date, request.request_date + timedelta(days=policy.horizon_days),
    )
    return {
        flow.source_id for flow in flows
        if flow.inferred and flow.amount < 0 and flow.source_id is not None
    }


def _validate_changes(
    value: str, request: Request, profile: UserProfile, data: DatasetBundle,
    status: str, errors: list[str],
) -> None:
    if value == "none":
        return
    if not value:
        errors.append(f"{request.request_id}: spending_changes_needed must be 'none' or contain changes")
        return
    entries = value.split("|")
    if len(entries) > 3:
        errors.append(f"{request.request_id}: spending_changes_needed exceeds three changes")
    supported = _supported_change_ids(data, request)
    seen: set[str] = set()
    for entry in entries:
        parts = entry.split(":")
        if len(parts) not in {2, 3} or parts[0] not in {"stop", "reduce_to"}:
            errors.append(f"{request.request_id}: malformed spending change: {entry!r}")
            continue
        action, event_id = parts[0], parts[1]
        if (action == "stop" and len(parts) != 2) or (action == "reduce_to" and len(parts) != 3):
            errors.append(f"{request.request_id}: malformed spending change: {entry!r}")
            continue
        if event_id in seen:
            errors.append(f"{request.request_id}: stop/reduce conflict or duplicate for {event_id}")
        seen.add(event_id)
        event = data.events_by_id.get(event_id)
        if event is None:
            errors.append(f"{request.request_id}: spending change references unknown event {event_id}")
            continue
        if event.user_id != request.user_id:
            errors.append(f"{request.request_id}: spending change event {event_id} belongs to another user")
        if event_id not in supported:
            errors.append(f"{request.request_id}: event {event_id} is not a supported inferred recurring expense")
        if event.category in profile.protected_categories:
            errors.append(f"{request.request_id}: protected event {event_id} cannot be changed")
        if action == "stop":
            if event.category not in profile.stoppable_categories or event.flexibility not in {"stoppable", "reducible_or_stoppable"}:
                errors.append(f"{request.request_id}: event {event_id} is not permitted to stop")
            continue
        amount = _decimal(parts[2], request.request_id, f"reduction amount for {event_id}", errors)
        if event.category not in profile.reducible_categories or event.flexibility not in {"reducible", "reducible_or_stoppable"}:
            errors.append(f"{request.request_id}: event {event_id} is not permitted to reduce")
        if amount is None:
            continue
        if amount < 0:
            errors.append(f"{request.request_id}: reduction amount for {event_id} cannot be negative")
        if event.minimum_allowed_amount is None or amount < event.minimum_allowed_amount:
            errors.append(f"{request.request_id}: reduction amount for {event_id} is below its minimum")
        if event.amount is not None and amount >= event.amount:
            errors.append(f"{request.request_id}: reduction amount for {event_id} must be below its current amount")
    if status in {"affordable_now", "affordable_later", "not_affordable"}:
        errors.append(f"{request.request_id}: {status} must not include spending changes")


def _validate_row(row: dict[str, str], request: Request, data: DatasetBundle, errors: list[str]) -> None:
    request_id = request.request_id
    profile = data.profiles[request.user_id]
    safe_amount = _decimal(row["amount_safe_to_pay"], request_id, "amount_safe_to_pay", errors)
    if safe_amount is not None and not Decimal("0") <= safe_amount <= request.requested_amount:
        errors.append(f"{request_id}: amount_safe_to_pay is outside [0, requested_amount]")

    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    if status not in AFFORDABILITY_STATUSES:
        errors.append(f"{request_id}: invalid affordability_status {status!r}")
    if method not in PAYMENT_METHODS:
        errors.append(f"{request_id}: invalid recommended_payment_method {method!r}")

    earliest_text = row["earliest_date_for_full_payment"]
    earliest = _iso_date(earliest_text, request_id, "earliest_date_for_full_payment", errors) if earliest_text else None
    payments = _parse_plan(row["payment_plan"], request_id, errors)
    changes = row["spending_changes_needed"]
    _validate_changes(changes, request, profile, data, status, errors)

    if status == "affordable_now" and earliest != request.request_date:
        errors.append(f"{request_id}: affordable_now requires earliest date equal to request_date")
    if not row["decision_explanation"].strip():
        errors.append(f"{request_id}: decision_explanation is empty")
    if payments is None:
        return
    if method == "not_recommended":
        if payments:
            errors.append(f"{request_id}: not_recommended requires payment_plan=none")
        if changes != "none":
            errors.append(f"{request_id}: not_recommended requires spending_changes_needed=none")
        if status != "not_affordable":
            errors.append(f"{request_id}: not_recommended requires not_affordable")
        return
    if not payments:
        errors.append(f"{request_id}: {method} requires a payment plan")
        return

    total = sum((amount for _, amount in payments), Decimal("0"))
    if payments[-1][0] > request.desired_completion_date:
        errors.append(f"{request_id}: payment plan misses desired_completion_date")

    if method == "full_payment":
        if "full_payment" not in profile.accepted_payment_methods:
            errors.append(f"{request_id}: full payment is not accepted by the profile")
        if len(payments) != 1 or payments[0] != (request.request_date, request.requested_amount):
            errors.append(f"{request_id}: full_payment must pay the requested amount on request_date")
        expected_status = "affordable_with_plan" if changes != "none" else "affordable_now"
        if status != expected_status:
            errors.append(f"{request_id}: full_payment has inconsistent affordability_status")
    elif method == "wait":
        if "full_payment" not in profile.accepted_payment_methods:
            errors.append(f"{request_id}: wait/full payment is not accepted by the profile")
        if len(payments) != 1 or total != request.requested_amount or payments[0][0] != earliest:
            errors.append(f"{request_id}: wait plan must pay the full amount on earliest_date_for_full_payment")
        if earliest is None or earliest <= request.request_date:
            errors.append(f"{request_id}: wait requires a future earliest_date_for_full_payment")
        if status != "affordable_later" or changes != "none":
            errors.append(f"{request_id}: wait requires affordable_later with no spending changes")
    elif method == "partial_payment":
        if not request.allows_partial_payment:
            errors.append(f"{request_id}: request does not allow partial payment")
        if "partial_payment" not in profile.accepted_payment_methods:
            errors.append(f"{request_id}: partial payment is not accepted by the profile")
        if len(payments) != 2:
            errors.append(f"{request_id}: partial payment requires exactly two payments")
        elif safe_amount is not None:
            if payments[0] != (request.request_date, safe_amount):
                errors.append(f"{request_id}: partial first payment must equal amount_safe_to_pay on request_date")
            if payments[1][0] != earliest:
                errors.append(f"{request_id}: partial second date must equal earliest_date_for_full_payment")
            if payments[1][1] != request.requested_amount - safe_amount:
                errors.append(f"{request_id}: partial second payment must equal the remainder")
        if total != request.requested_amount:
            errors.append(f"{request_id}: partial payments must sum to requested_amount")
        if status != "affordable_with_plan":
            errors.append(f"{request_id}: partial payment requires affordable_with_plan")
    elif method == "installments":
        if "installments" not in profile.accepted_payment_methods:
            errors.append(f"{request_id}: installments are not accepted by the profile")
        option = _matching_installment(
            payments, data.payment_options_by_request.get(request_id, ()),
        )
        if option is None:
            errors.append(f"{request_id}: installment plan does not exactly match a supplied option")
        else:
            if option.total_payable_amount != request.requested_amount + option.financing_fee:
                errors.append(f"{request_id}: installment option total does not equal amount plus fee")
            if profile.max_installment_months is None or option.number_of_payments > profile.max_installment_months:
                errors.append(f"{request_id}: installment option exceeds max_installment_months")
            if payments[-1][0] > request.desired_completion_date:
                errors.append(f"{request_id}: installment option misses desired_completion_date")
        if status != "affordable_with_plan":
            errors.append(f"{request_id}: installments require affordable_with_plan")


def validate_submission(path: Path, data: DatasetBundle) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return (f"submission: cannot read {path}: {exc}",)
    try:
        parsed = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        return (f"submission: invalid CSV: {exc}",)
    if not parsed:
        return ("submission: output is empty",)
    header, raw_rows = parsed[0], parsed[1:]
    if tuple(header) != REQUIRED_COLUMNS:
        errors.append(
            "submission: columns must exactly match required order: " + ",".join(REQUIRED_COLUMNS)
        )
        return tuple(errors)
    records: list[dict[str, str]] = []
    for index, values in enumerate(raw_rows, start=2):
        if len(values) != len(REQUIRED_COLUMNS):
            errors.append(f"row {index}: expected {len(REQUIRED_COLUMNS)} columns, found {len(values)}")
            continue
        records.append(dict(zip(REQUIRED_COLUMNS, values)))

    expected_ids = list(data.requests)
    actual_ids = [record["request_id"] for record in records]
    counts = Counter(actual_ids)
    for request_id, count in counts.items():
        if count > 1:
            errors.append(f"{request_id}: duplicate prediction ({count} rows)")
    for request_id in expected_ids:
        if request_id not in counts:
            errors.append(f"{request_id}: missing prediction")
    for request_id in actual_ids:
        if request_id not in data.requests:
            errors.append(f"{request_id or '<blank>'}: unexpected request_id")
    if actual_ids != expected_ids:
        errors.append("submission: request ordering does not match dataset/requests.csv")

    validated: set[str] = set()
    for record in records:
        request_id = record["request_id"]
        if request_id in validated or request_id not in data.requests:
            continue
        validated.add(request_id)
        _validate_row(record, data.requests[request_id], data, errors)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(parsed)
    if list(csv.reader(io.StringIO(buffer.getvalue(), newline=""))) != parsed:
        errors.append("submission: CSV does not round-trip safely")
    return tuple(errors)


def assert_valid_submission(path: Path, data: DatasetBundle) -> None:
    errors = validate_submission(path, data)
    if errors:
        raise SubmissionValidationError(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Buy or Wait? submission CSV")
    parser.add_argument("output", nargs="?", type=Path, default=REPO_ROOT / "output.csv")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    args = parser.parse_args()
    errors = validate_submission(args.output.resolve(), DatasetLoader(args.dataset).load())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Valid submission: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
