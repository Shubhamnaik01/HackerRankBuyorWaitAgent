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

    def extract_message(self, message: Message) -> ModelResult: ...
    def extract_image(self, path: Path, event: FinancialEvent) -> ModelResult: ...


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
            self._resolve_message_with_model(message, allowed_event_ids, result)

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
            self._resolve_image_with_model(image.path, image.image_id, event, result)
        return result

    def _record(self, model_result: ModelResult, purpose: str) -> None:
        self.usage.record(
            model_result.model, purpose, model_result.input_tokens,
            model_result.output_tokens, model_result.total_tokens,
        )

    def _resolve_message_with_model(self, message, allowed_event_ids, bundle) -> None:
        assert self.client is not None
        content = message.message_text.encode("utf-8")
        key = EvidenceCache.key("message", message.message_id, self.client.model, content)
        payload = self.cache.get(key)
        if payload is not None:
            self.usage.record_cache_hit()
        else:
            try:
                model_result = self.client.extract_message(message)
            except Exception as exc:
                bundle.warnings.append(f"{message.message_id}: message extraction failed: {exc}")
                return
            self._record(model_result, "message")
            payload = model_result.payload
        try:
            raw_facts = payload.get("facts")
            if not isinstance(raw_facts, list):
                raise EvidenceValidationError("facts must be an array")
            facts = [
                validate_fact(raw, source_id=message.message_id, allowed_event_ids=allowed_event_ids)
                for raw in raw_facts
            ]
        except EvidenceValidationError as exc:
            self.usage.record_validation_failure()
            bundle.warnings.append(f"{message.message_id}: rejected model evidence: {exc}")
            return
        bundle.facts.extend(facts)
        if self.cache.get(key) is None:
            self.cache.put(key, payload)

    def _resolve_image_with_model(self, path, image_id, event, bundle) -> None:
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
        else:
            try:
                model_result = self.client.extract_image(path, event)
            except Exception as exc:
                bundle.warnings.append(f"{image_id}: image extraction failed: {exc}")
                return
            self._record(model_result, "image")
            payload = model_result.payload
        try:
            value = validate_image_amount(
                payload, event_id=event.event_id, evidence_id=image_id,
                expected_currency=event.currency,
            )
        except EvidenceValidationError as exc:
            self.usage.record_validation_failure()
            bundle.warnings.append(f"{image_id}: rejected model evidence: {exc}")
            return
        bundle.image_amounts[event.event_id] = value
        if self.cache.get(key) is None:
            self.cache.put(key, payload)
