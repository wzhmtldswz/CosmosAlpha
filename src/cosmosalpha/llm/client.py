"""LLM client interface and deterministic replay implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from cosmosalpha.core import CoreValidationError, JsonScalar
from cosmosalpha.llm.errors import ReplayError
from cosmosalpha.llm.prompts import PromptRenderResult


@dataclass(frozen=True, slots=True)
class LLMResponseMetadata:
    """Transport and provider evidence attached to one LLM response.

    The metadata is intentionally kept outside the persisted record schema.  The
    recorder flattens it into the existing redacted parameter map, which keeps old
    call records readable while making response classification auditable.
    """

    http_status: int | None = None
    provider_request_id: str | None = None
    provider_response_id: str | None = None
    content_type: str | None = None
    declared_content_length: int | None = None
    response_body_bytes: int = 0
    response_body_chars: int = 0
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    retry_after_seconds: float | None = None
    raw_source: str = "provider_body"

    def __post_init__(self) -> None:
        """Validate transport evidence without imposing provider-specific fields."""
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise CoreValidationError("http_status must be an HTTP status or None")
        for field_name in (
            "provider_request_id",
            "provider_response_id",
            "content_type",
            "finish_reason",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise CoreValidationError(f"{field_name} must be a non-empty string")
        if self.declared_content_length is not None and (
            type(self.declared_content_length) is not int
            or self.declared_content_length < 0
        ):
            raise CoreValidationError(
                "declared_content_length must be a non-negative integer or None"
            )
        for field_name in (
            "response_body_bytes",
            "response_body_chars",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise CoreValidationError(f"{field_name} must be non-negative")
        if self.retry_after_seconds is not None and (
            type(self.retry_after_seconds) is not float
            or self.retry_after_seconds < 0.0
            or not math.isfinite(self.retry_after_seconds)
        ):
            raise CoreValidationError(
                "retry_after_seconds must be a non-negative float or None"
            )
        if type(self.raw_source) is not str or self.raw_source not in {
            "provider_body",
            "local_diagnostic",
        }:
            raise CoreValidationError("raw_source is invalid")

    def as_audit_parameters(self) -> dict[str, JsonScalar]:
        """Flatten non-secret evidence into the existing audit parameter map."""
        parameters: dict[str, JsonScalar] = {
            "response_body_bytes": self.response_body_bytes,
            "response_body_chars": self.response_body_chars,
            "raw_response_source": self.raw_source,
        }
        optional_values: dict[str, JsonScalar | None] = {
            "http_status": self.http_status,
            "provider_request_id": self.provider_request_id,
            "provider_response_id": self.provider_response_id,
            "content_type": self.content_type,
            "declared_content_length": self.declared_content_length,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "retry_after_seconds": self.retry_after_seconds,
        }
        for key, value in optional_values.items():
            if value is not None:
                parameters[key] = value
        return parameters


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Provider-independent LLM request."""

    prompt: PromptRenderResult
    model_alias: str
    max_output_tokens: int
    response_format: str

    def __post_init__(self) -> None:
        """Validate request fields."""
        if not isinstance(self.prompt, PromptRenderResult):
            raise CoreValidationError("prompt must be PromptRenderResult")
        _require_string(self.model_alias, "model_alias")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise CoreValidationError("max_output_tokens must be positive")
        if self.response_format != "json_object":
            raise CoreValidationError("response_format must be json_object")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-independent LLM response."""

    content: str
    model_alias: str
    prompt_render_id: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    raw_response: str | None = None
    metadata: LLMResponseMetadata | None = None

    def __post_init__(self) -> None:
        """Validate response fields."""
        _require_string(self.content, "content")
        _require_string(self.model_alias, "model_alias")
        _require_string(self.prompt_render_id, "prompt_render_id")
        _require_string(self.finish_reason, "finish_reason")
        if type(self.input_tokens) is not int or self.input_tokens < 0:
            raise CoreValidationError("input_tokens must be non-negative")
        if type(self.output_tokens) is not int or self.output_tokens < 0:
            raise CoreValidationError("output_tokens must be non-negative")
        # OpenAI-compatible providers (notably DeepSeek) may emit keep-alive
        # leading newlines before the JSON body while the request waits for
        # scheduling.  Accept any non-empty raw response and keep the original
        # bytes for audit instead of rejecting framed JSON as an empty body.
        if self.raw_response is not None and (
            type(self.raw_response) is not str or not self.raw_response.strip()
        ):
            raise CoreValidationError("raw_response must be a non-empty string")
        if self.metadata is not None and not isinstance(
            self.metadata, LLMResponseMetadata
        ):
            raise CoreValidationError("metadata must be LLMResponseMetadata or None")


class LLMClient(Protocol):
    """Provider-independent LLM client contract."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one model response for a rendered prompt."""


class ReplayLLMClient:
    """Replay validated raw JSON responses without calling a provider."""

    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        model_alias: str = "replay",
    ) -> None:
        """Create a replay client.

        Response keys may be either ``prompt_render_id`` or ``prompt_id``.
        """
        self._responses = dict(responses)
        self._model_alias = model_alias
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the configured response for a request."""
        self.requests.append(request)
        content = self._responses.get(request.prompt.prompt_render_id)
        if content is None:
            content = self._responses.get(request.prompt.prompt_id)
        if content is None:
            raise ReplayError(f"REPLAY_RESPONSE_NOT_FOUND: {request.prompt.prompt_id}")
        return LLMResponse(
            content=content,
            model_alias=self._model_alias,
            prompt_render_id=request.prompt.prompt_render_id,
            finish_reason="stop",
            input_tokens=request.prompt.estimated_input_tokens,
            output_tokens=max(1, (len(content) + 3) // 4),
        )


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value
