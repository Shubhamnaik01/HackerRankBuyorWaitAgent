from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .models import (
    ExchangeRate,
    FinancialEvent,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
    UserProfile,
)


class DataValidationError(ValueError):
    pass


def _decimal(value: str) -> Decimal | None:
    return Decimal(value) if value != "" else None


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.split("|") if token)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class DatasetBundle:
    root: Path
    profiles: dict[str, UserProfile]
    requests: dict[str, Request]
    sample_requests: dict[str, Request]
    events_by_user: dict[str, tuple[FinancialEvent, ...]]
    events_by_id: dict[str, FinancialEvent]
    payment_options_by_request: dict[str, tuple[PaymentOption, ...]]
    messages_by_user: dict[str, tuple[Message, ...]]
    images_by_event: dict[str, ImageRecord]
    exchange_rates: tuple[ExchangeRate, ...]


class DatasetLoader:
    def __init__(self, dataset_root: Path):
        self.root = dataset_root.resolve()

    def load(self) -> DatasetBundle:
        profiles = self._profiles()
        requests = self._requests("requests.csv", solved=False)
        samples = self._requests("sample_requests.csv", solved=True)
        all_requests = {**samples, **requests}
        events = self._events()
        options = self._options()
        messages = self._messages()
        images = self._images()
        rates = self._rates()

        self._validate(profiles, all_requests, events, options, messages, images)

        events_by_user: dict[str, list[FinancialEvent]] = {}
        for event in events.values():
            events_by_user.setdefault(event.user_id, []).append(event)
        options_by_request: dict[str, list[PaymentOption]] = {}
        for option in options:
            options_by_request.setdefault(option.request_id, []).append(option)
        messages_by_user: dict[str, list[Message]] = {}
        for message in messages:
            messages_by_user.setdefault(message.user_id, []).append(message)

        return DatasetBundle(
            root=self.root,
            profiles=profiles,
            requests=requests,
            sample_requests=samples,
            events_by_user={key: tuple(value) for key, value in events_by_user.items()},
            events_by_id=events,
            payment_options_by_request={key: tuple(value) for key, value in options_by_request.items()},
            messages_by_user={key: tuple(value) for key, value in messages_by_user.items()},
            images_by_event={image.related_event_id: image for image in images},
            exchange_rates=tuple(rates),
        )

    def _profiles(self) -> dict[str, UserProfile]:
        result: dict[str, UserProfile] = {}
        for row in _rows(self.root / "financial_profiles.csv"):
            profile = UserProfile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                current_available_balance=Decimal(row["current_available_balance"]),
                minimum_balance_to_keep=Decimal(row["minimum_balance_to_keep"]),
                financial_priorities=_tokens(row["financial_priorities"]),
                protected_categories=_tokens(row["expense_categories_to_protect"]),
                reducible_categories=_tokens(row["expense_categories_user_is_willing_to_reduce"]),
                stoppable_categories=_tokens(row["expense_categories_user_is_willing_to_stop"]),
                accepted_payment_methods=_tokens(row["payment_methods_user_will_consider"]),
                max_installment_months=int(row["max_installment_months"]) if row["max_installment_months"] else None,
            )
            if profile.user_id in result:
                raise DataValidationError(f"duplicate profile: {profile.user_id}")
            result[profile.user_id] = profile
        return result

    def _requests(self, filename: str, solved: bool) -> dict[str, Request]:
        result: dict[str, Request] = {}
        for row in _rows(self.root / filename):
            request = Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=date.fromisoformat(row["request_date"]),
                request_type=row["request_type"],
                requested_amount=Decimal(row["requested_amount"]),
                desired_completion_date=date.fromisoformat(row["desired_completion_date"]),
                allows_partial_payment=row["allows_partial_payment"].lower() == "true",
                request_text=row["request_text"],
                expected_amount_safe_to_pay=_decimal(row.get("amount_safe_to_pay", "")) if solved else None,
                expected_earliest_date_for_full_payment=_date(row.get("earliest_date_for_full_payment", "")) if solved else None,
                expected_affordability_status=row.get("affordability_status") if solved else None,
                expected_recommended_payment_method=row.get("recommended_payment_method") if solved else None,
                expected_payment_plan=row.get("payment_plan") if solved else None,
                expected_spending_changes_needed=row.get("spending_changes_needed") if solved else None,
                expected_decision_explanation=row.get("decision_explanation") if solved else None,
            )
            if request.request_id in result:
                raise DataValidationError(f"duplicate request: {request.request_id}")
            result[request.request_id] = request
        return result

    def _events(self) -> dict[str, FinancialEvent]:
        result: dict[str, FinancialEvent] = {}
        for row in _rows(self.root / "financial_events.csv"):
            event = FinancialEvent(
                event_id=row["event_id"], user_id=row["user_id"], event_type=row["event_type"],
                description=row["description"], category=row["category"], direction=row["direction"],
                amount=_decimal(row["amount"]), currency=row["currency"],
                event_date=date.fromisoformat(row["event_date"]), settlement_date=_date(row["settlement_date"]),
                status=row["status"], linked_event_id=row["linked_event_id"] or None,
                flexibility=row["flexibility"], minimum_allowed_amount=_decimal(row["minimum_allowed_amount"]),
            )
            if event.event_id in result:
                raise DataValidationError(f"duplicate event: {event.event_id}")
            result[event.event_id] = event
        return result

    def _options(self) -> list[PaymentOption]:
        return [PaymentOption(
            payment_option_id=row["payment_option_id"], request_id=row["request_id"],
            payment_method=row["payment_method"], payment_amount=Decimal(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=date.fromisoformat(row["first_payment_date"]),
            payment_frequency_days=int(row["payment_frequency_days"]) if row["payment_frequency_days"] else None,
            financing_fee=Decimal(row["financing_fee"]), total_payable_amount=Decimal(row["total_payable_amount"]),
        ) for row in _rows(self.root / "request_payment_options.csv")]

    def _messages(self) -> list[Message]:
        return [Message(
            message_id=row["message_id"], user_id=row["user_id"], request_id=row["request_id"] or None,
            related_event_id=row["related_event_id"] or None,
            sent_at=datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00")),
            source_type=row["source_type"], message_text=row["message_text"],
        ) for row in _rows(self.root / "messages.csv")]

    def _images(self) -> list[ImageRecord]:
        return [ImageRecord(
            image_id=row["image_id"], user_id=row["user_id"], request_id=row["request_id"] or None,
            related_event_id=row["related_event_id"],
            path=self.root / "media" / "images" / f'{row["image_id"]}.png',
        ) for row in _rows(self.root / "images.csv")]

    def _rates(self) -> list[ExchangeRate]:
        return [ExchangeRate(
            rate_date=date.fromisoformat(row["rate_date"]), from_currency=row["from_currency"],
            to_currency=row["to_currency"], rate=Decimal(row["rate"]),
        ) for row in _rows(self.root / "exchange_rates.csv")]

    @staticmethod
    def _validate(profiles, requests, events, options, messages, images) -> None:
        errors: list[str] = []
        option_request_ids = {option.request_id for option in options}
        for request in requests.values():
            if request.user_id not in profiles:
                errors.append(f"{request.request_id}: missing profile {request.user_id}")
            if request.request_id not in option_request_ids:
                errors.append(f"{request.request_id}: no payment options")
        for event in events.values():
            if event.user_id not in profiles:
                errors.append(f"{event.event_id}: missing profile {event.user_id}")
            if event.linked_event_id and event.linked_event_id not in events:
                errors.append(f"{event.event_id}: missing linked event {event.linked_event_id}")
        for option in options:
            if option.request_id not in requests:
                errors.append(f"{option.payment_option_id}: missing request {option.request_id}")
        for message in messages:
            if message.user_id not in profiles:
                errors.append(f"{message.message_id}: missing user {message.user_id}")
            if message.request_id and message.request_id not in requests:
                errors.append(f"{message.message_id}: missing request {message.request_id}")
            if message.related_event_id and message.related_event_id not in events:
                errors.append(f"{message.message_id}: missing event {message.related_event_id}")
        for image in images:
            if image.related_event_id not in events:
                errors.append(f"{image.image_id}: missing event {image.related_event_id}")
                continue
            linked_event = events[image.related_event_id]
            if linked_event.user_id != image.user_id:
                errors.append(f"{image.image_id}: user does not match {image.related_event_id}")
            if image.request_id and image.request_id not in requests:
                errors.append(f"{image.image_id}: missing request {image.request_id}")
            if not image.path.is_file():
                errors.append(f"{image.image_id}: missing PNG {image.path}")
        if errors:
            raise DataValidationError("\n".join(errors))
