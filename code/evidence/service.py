from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from ..config import inclusive_horizon_end
from ..data_loader import DatasetBundle
from ..models import FinancialEvent, Message, Request
from .cache import EvidenceCache
from .messages import is_financially_relevant, parse_message_deterministically
from .models import EvidenceBundle
from .openai_client import ModelResult, OpenAIEvidenceClient
from .usage import UsageTracker
from .validation import EvidenceValidationError, validate_fact, validate_image_amount


class EvidenceModelClient(Protocol):
    model: str

    def extract_message(
        self, message: Message, validation_error: str | None = None,
    ) -> ModelResult: ...
    def extract_image(
        self, path: Path, event: FinancialEvent, validation_error: str | None = None,
    ) -> ModelResult: ...


class EvidenceService:
    def __init__(
        self,
        data: DatasetBundle,
        *,
        client: EvidenceModelClient | None = None,
        usage: UsageTracker | None = None,
        cache: EvidenceCache | None = None,
        enable_environment_client: bool = True,
    ):
        self.data = data
        self.usage = usage or UsageTracker()
        self.client = client
        if self.client is None and enable_environment_client:
            self.client = OpenAIEvidenceClient.from_environment()
        self.cache = cache or EvidenceCache(data.root.parent / "code" / ".cache" / "evidence")
        self._terminal_validation_failures: set[str] = set()

    def resolve(self, request: Request, history_days: int, horizon_days: int) -> EvidenceBundle:
        result = EvidenceBundle()
        events = self.data.events_by_user.get(request.user_id, ())
        allowed_event_ids = {event.event_id for event in events}

        relevant_messages = [
            message for message in self.data.messages_by_user.get(request.user_id, ())
            if message.sent_at.date() <= request.request_date
            and (message.request_id is None or message.request_id == request.request_id)
            and is_financially_relevant(message)
        ]
        # Oldest to newest makes later same-source amendments win when the
        # deterministic application layer walks facts in order.
        relevant_messages.sort(key=lambda item: (item.sent_at, item.message_id))
        for message in relevant_messages:
            facts, classified = parse_message_deterministically(message)
            result.facts.extend(facts)
            if classified or self.client is None:
                continue
            self._resolve_message_with_model(
                message, allowed_event_ids, result, request.request_id,
            )

        history_start = request.request_date - timedelta(days=history_days)
        horizon_end = inclusive_horizon_end(request.request_date, horizon_days)
        for event in events:
            cash_date = event.settlement_date or event.event_date
            image = self.data.images_by_event.get(event.event_id)
            if (
                event.amount is not None or image is None
                or not history_start <= cash_date <= horizon_end
                or (image.request_id is not None and image.request_id != request.request_id)
            ):
                continue
            if self.client is None:
                result.warnings.append(f"{event.event_id}: image amount requires OPENAI_API_KEY")
                continue
            self._resolve_image_with_model(
                image.path, image.image_id, event, result, request.request_id,
            )
        return result

    def _record(self, model_result: ModelResult, purpose: str) -> None:
        self.usage.record(
            model_result.model, purpose, model_result.input_tokens,
            model_result.output_tokens, model_result.total_tokens,
        )

    @staticmethod
    def _validate_message_payload(payload, message, allowed_event_ids):
        if not isinstance(payload, dict):
            raise EvidenceValidationError("message result must be an object")
        raw_facts = payload.get("facts")
        if not isinstance(raw_facts, list):
            raise EvidenceValidationError("facts must be an array")
        return [
            validate_fact(raw, source_id=message.message_id, allowed_event_ids=allowed_event_ids)
            for raw in raw_facts
        ]

    def _record_validation_failure(
        self, source_type: str, source_id: str, request_id: str,
        error: EvidenceValidationError, attempt: int,
    ) -> None:
        self.usage.record_validation_failure(
            source_type, source_id, request_id, str(error), attempt,
        )

    def _resolve_message_with_model(
        self, message, allowed_event_ids, bundle, request_id: str,
    ) -> None:
        assert self.client is not None
        content = message.message_text.encode("utf-8")
        key = EvidenceCache.key("message", message.message_id, self.client.model, content)
        payload = self.cache.get(key)
        if payload is not None:
            self.usage.record_cache_hit()
            try:
                bundle.facts.extend(self._validate_message_payload(payload, message, allowed_event_ids))
            except EvidenceValidationError as exc:
                self._record_validation_failure("message", message.message_id, request_id, exc, 0)
                self._terminal_validation_failures.add(key)
                bundle.warnings.append(f"{message.message_id}: rejected cached model evidence: {exc}")
            return
        if key in self._terminal_validation_failures:
            bundle.warnings.append(f"{message.message_id}: model evidence was already rejected after one retry")
            return

        validation_error: str | None = None
        for attempt in (1, 2):
            try:
                model_result = self.client.extract_message(message, validation_error)
            except Exception as exc:
                label = "retry" if validation_error is not None else "extraction"
                bundle.warnings.append(f"{message.message_id}: message {label} failed: {exc}")
                if validation_error is not None:
                    self._terminal_validation_failures.add(key)
                return
            self._record(model_result, "message")
            payload = model_result.payload
            try:
                facts = self._validate_message_payload(payload, message, allowed_event_ids)
            except EvidenceValidationError as exc:
                self._record_validation_failure("message", message.message_id, request_id, exc, attempt)
                if attempt == 1:
                    validation_error = str(exc)
                    continue
                self._terminal_validation_failures.add(key)
                bundle.warnings.append(
                    f"{message.message_id}: rejected model evidence after one retry: {exc}"
                )
                return
            bundle.facts.extend(facts)
            self.cache.put(key, payload)
            return

    def _resolve_image_with_model(
        self, path, image_id, event, bundle, request_id: str,
    ) -> None:
        assert self.client is not None
        try:
            content = path.read_bytes()
        except OSError as exc:
            bundle.warnings.append(f"{image_id}: cannot read image: {exc}")
            return
        context = json.dumps({
            "event_id": event.event_id, "description": event.description,
            "currency": event.currency, "date": str(event.settlement_date or event.event_date),
        }, sort_keys=True).encode()
        key = EvidenceCache.key("image", image_id, self.client.model, content + context)
        payload = self.cache.get(key)
        if payload is not None:
            self.usage.record_cache_hit()
            try:
                bundle.image_amounts[event.event_id] = validate_image_amount(
                    payload, event_id=event.event_id, evidence_id=image_id,
                    expected_currency=event.currency,
                )
            except EvidenceValidationError as exc:
                self._record_validation_failure("image", image_id, request_id, exc, 0)
                self._terminal_validation_failures.add(key)
                bundle.warnings.append(f"{image_id}: rejected cached model evidence: {exc}")
            return
        if key in self._terminal_validation_failures:
            bundle.warnings.append(f"{image_id}: model evidence was already rejected after one retry")
            return

        validation_error: str | None = None
        for attempt in (1, 2):
            try:
                model_result = self.client.extract_image(path, event, validation_error)
            except Exception as exc:
                label = "retry" if validation_error is not None else "extraction"
                bundle.warnings.append(f"{image_id}: image {label} failed: {exc}")
                if validation_error is not None:
                    self._terminal_validation_failures.add(key)
                return
            self._record(model_result, "image")
            payload = model_result.payload
            try:
                value = validate_image_amount(
                    payload, event_id=event.event_id, evidence_id=image_id,
                    expected_currency=event.currency,
                )
            except EvidenceValidationError as exc:
                self._record_validation_failure("image", image_id, request_id, exc, attempt)
                if attempt == 1:
                    validation_error = str(exc)
                    continue
                self._terminal_validation_failures.add(key)
                bundle.warnings.append(
                    f"{image_id}: rejected model evidence after one retry: {exc}"
                )
                return
            bundle.image_amounts[event.event_id] = value
            self.cache.put(key, payload)
            return
