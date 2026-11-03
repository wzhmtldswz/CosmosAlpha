"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import http.client
import json
import math
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from cosmosalpha.core import CoreValidationError, JsonValue, LLMConfig
from cosmosalpha.core.config import REASONING_EFFORTS
from cosmosalpha.llm.client import LLMRequest, LLMResponse, LLMResponseMetadata
from cosmosalpha.llm.errors import LLMClientError

_UNSET_SENTINELS = {"", "unused", "<redacted>"}


@dataclass(frozen=True, slots=True)
class JsonHttpResponse:
    """Small transport result carrying body and allowlisted response evidence."""

    status_code: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize a transport response."""
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise CoreValidationError("status_code must be an HTTP status")
        if type(self.body) is not str:
            raise CoreValidationError("body must be a string")
        if not isinstance(self.headers, Mapping) or not all(
            type(key) is str and type(value) is str
            for key, value in self.headers.items()
        ):
            raise CoreValidationError("headers must be a string mapping")
        if self.body_bytes is not None and (
            type(self.body_bytes) is not int or self.body_bytes < 0
        ):
            raise CoreValidationError("body_bytes must be a non-negative integer")
        if self.body_bytes is None:
            object.__setattr__(self, "body_bytes", len(self.body.encode("utf-8")))
        object.__setattr__(self, "headers", dict(self.headers))

    def __iter__(self) -> Iterator[int | str]:
        """Allow legacy ``status, body = transport.post_json(...)`` callers."""
        yield self.status_code
        yield self.body


class JsonHttpTransport(Protocol):
    """Minimal JSON POST transport for provider tests and stdlib runtime."""

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: int,
    ) -> JsonHttpResponse:
        """POST JSON and return the body plus response transport evidence."""


class UrllibJsonTransport:
    """Standard-library JSON HTTP transport."""

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: int,
    ) -> JsonHttpResponse:
        """POST JSON using ``urllib.request``."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - URL is explicit config input.
                request,
                timeout=timeout_seconds,
            ) as response:
                status = int(response.status)
                raw_bytes = response.read()
                raw = raw_bytes.decode("utf-8", errors="replace")
                response_result = JsonHttpResponse(
                    status_code=status,
                    body=raw,
                    headers=_header_mapping(response.headers),
                    body_bytes=len(raw_bytes),
                )
        except urllib.error.HTTPError as error:
            status = int(error.code)
            raw_bytes = error.read()
            response_result = JsonHttpResponse(
                status_code=status,
                body=raw_bytes.decode("utf-8", errors="replace"),
                headers=_header_mapping(error.headers),
                body_bytes=len(raw_bytes),
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            OSError,
        ) as error:
            partial = getattr(error, "partial", b"")
            if not isinstance(partial, bytes):
                partial = b""
            raw = partial.decode("utf-8", errors="replace")
            if _is_timeout_error(error):
                raise LLMClientError(
                    f"LLM_TIMEOUT: {error}",
                    raw_response=raw or None,
                    metadata=_transport_metadata(
                        raw,
                        raw_source="provider_body" if raw else "local_diagnostic",
                    ),
                ) from error
            if isinstance(error, http.client.IncompleteRead):
                raise LLMClientError(
                    f"LLM_RESPONSE_TRUNCATED: received_bytes={len(partial)}",
                    raw_response=raw or None,
                    metadata=_transport_metadata(
                        raw,
                        raw_source="provider_body" if raw else "local_diagnostic",
                    ),
                ) from error
            raise LLMClientError(
                f"LLM_TRANSPORT_ERROR: {error}",
                raw_response=raw or None,
                metadata=_transport_metadata(
                    raw,
                    raw_source="provider_body" if raw else "local_diagnostic",
                ),
            ) from error
        return response_result


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMClient:
    """OpenAI-compatible ``/v1/chat/completions`` client.

    The client intentionally avoids provider SDKs so the LLM boundary stays small and
    replay tests remain dependency-free.
    """

    api_key: str = ""
    api_key_env: str = ""
    base_url: str = "https://api.openai.com"
    base_url_env: str = ""
    organization: str = ""
    temperature: float = 0.0
    timeout_seconds: int = 30
    transport: JsonHttpTransport | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        """Validate static provider configuration."""
        if type(self.api_key) is not str:
            raise CoreValidationError("api_key must be a string")
        if type(self.api_key_env) is not str:
            raise CoreValidationError("api_key_env must be a string")
        _require_string(self.base_url, "base_url")
        if type(self.base_url_env) is not str:
            raise CoreValidationError("base_url_env must be a string")
        if type(self.organization) is not str:
            raise CoreValidationError("organization must be a string")
        if type(self.temperature) is not float or self.temperature < 0.0:
            raise CoreValidationError("temperature must be a non-negative float")
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise CoreValidationError("timeout_seconds must be positive")
        if self.reasoning_effort is not None:
            if (
                type(self.reasoning_effort) is not str
                or self.reasoning_effort not in REASONING_EFFORTS
            ):
                raise CoreValidationError(
                    f"reasoning_effort must be one of {REASONING_EFFORTS!r} or None"
                )

    @classmethod
    def from_config(
        cls,
        config: LLMConfig,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> OpenAICompatibleLLMClient:
        """Create a real client from strict LLM configuration."""
        if not isinstance(config, LLMConfig):
            raise CoreValidationError("config must be LLMConfig")
        if config.replay_only:
            raise LLMClientError("LLM_REAL_CLIENT_DISABLED")
        if config.provider not in {"openai", "openai-compatible"}:
            raise LLMClientError(f"LLM_PROVIDER_UNSUPPORTED: {config.provider}")
        return cls(
            api_key=config.api_key,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            base_url_env=config.base_url_env,
            organization=config.organization,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            transport=transport,
            reasoning_effort=config.reasoning_effort,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Call an OpenAI-compatible chat completion endpoint."""
        if not isinstance(request, LLMRequest):
            raise CoreValidationError("request must be LLMRequest")
        url = _chat_completions_url(_resolve_base_url(self.base_url, self.base_url_env))
        headers = {
            "Authorization": f"Bearer {_resolve_api_key(self.api_key, self.api_key_env)}",
            "Content-Type": "application/json",
        }
        organization = self.organization.strip()
        if organization and organization not in _UNSET_SENTINELS:
            headers["OpenAI-Organization"] = organization
        payload: dict[str, JsonValue] = {
            "model": request.model_alias,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.prompt.messages
            ],
            "temperature": self.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        transport = (
            self.transport if self.transport is not None else UrllibJsonTransport()
        )
        transport_result = transport.post_json(
            url,
            headers,
            payload,
            self.timeout_seconds,
        )
        response_result = _coerce_transport_response(transport_result)
        raw_response = response_result.body
        metadata = _response_metadata(response_result)
        status = response_result.status_code
        if status < 200 or status >= 300:
            raise LLMClientError(
                _http_error_message(status),
                raw_response=raw_response,
                metadata=metadata,
            )
        declared_content_length = metadata.declared_content_length
        if (
            declared_content_length is not None
            and response_result.body_bytes is not None
            and declared_content_length != response_result.body_bytes
        ):
            raise LLMClientError(
                "LLM_RESPONSE_TRUNCATED: "
                "content_length="
                f"{declared_content_length} received_bytes={response_result.body_bytes}",
                raw_response=raw_response,
                metadata=metadata,
            )
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise LLMClientError(
                "LLM_RESPONSE_EMPTY_BODY: provider response body was empty",
                raw_response=raw_response if isinstance(raw_response, str) else None,
                metadata=metadata,
            )
        parsed = _decode_response(raw_response, metadata=metadata)
        if parsed.finish_reason != "stop":
            raise LLMClientError(
                f"LLM_RESPONSE_INCOMPLETE: finish_reason={parsed.finish_reason}",
                raw_response=raw_response,
                metadata=parsed.metadata,
            )
        if not parsed.content.strip():
            raise LLMClientError(
                "LLM_RESPONSE_EMPTY_CONTENT: message.content was empty",
                raw_response=raw_response,
                metadata=parsed.metadata,
            )
        # Models may frame their JSON content with leading/trailing newlines;
        # normalize the content while keeping the raw body for audit.
        content = parsed.content.strip()
        if not content:
            raise LLMClientError(
                "LLM_RESPONSE_EMPTY_CONTENT: message.content was empty",
                raw_response=raw_response,
            )
        _validate_output_content(
            content,
            request.prompt.output_json_schema,
            raw_response,
            metadata=parsed.metadata,
        )
        return LLMResponse(
            content=content,
            model_alias=request.model_alias,
            prompt_render_id=request.prompt.prompt_render_id,
            finish_reason=parsed.finish_reason,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            raw_response=raw_response,
            metadata=parsed.metadata,
        )


@dataclass(frozen=True, slots=True)
class _ParsedChatCompletion:
    content: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    metadata: LLMResponseMetadata


def _coerce_transport_response(value: object) -> JsonHttpResponse:
    """Accept the legacy ``(status, body)`` test transport shape temporarily."""
    if isinstance(value, JsonHttpResponse):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return JsonHttpResponse(status_code=value[0], body=value[1])
        except CoreValidationError as error:
            raise LLMClientError(
                "LLM_TRANSPORT_ERROR: transport returned an invalid response"
            ) from error
    raise LLMClientError("LLM_TRANSPORT_ERROR: transport returned an invalid response")


def _response_metadata(response: JsonHttpResponse) -> LLMResponseMetadata:
    """Build bounded, non-secret evidence from one HTTP response."""
    body_bytes = response.body_bytes
    if body_bytes is None:
        body_bytes = len(response.body.encode("utf-8"))
    return LLMResponseMetadata(
        http_status=response.status_code,
        provider_request_id=_first_header(
            response.headers,
            "x-request-id",
            "request-id",
            "x-correlation-id",
        ),
        content_type=_first_header(response.headers, "content-type"),
        declared_content_length=_declared_content_length(response.headers),
        response_body_bytes=body_bytes,
        response_body_chars=len(response.body),
        retry_after_seconds=_retry_after_seconds(response.headers),
    )


def _transport_metadata(raw_response: str, *, raw_source: str) -> LLMResponseMetadata:
    """Build evidence for a transport exception and any bytes read before it."""
    return LLMResponseMetadata(
        response_body_bytes=len(raw_response.encode("utf-8")),
        response_body_chars=len(raw_response),
        raw_source=raw_source,
    )


def _header_mapping(headers: object) -> dict[str, str]:
    """Copy stdlib response headers into a plain string mapping."""
    items = getattr(headers, "items", None)
    if not callable(items):
        return {}
    return {str(key): str(value) for key, value in items()}


def _first_header(headers: Mapping[str, str], *names: str) -> str | None:
    """Return the first non-empty case-insensitive header value."""
    lowered = {key.lower(): value.strip() for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return None


def _declared_content_length(headers: Mapping[str, str]) -> int | None:
    """Return a valid Content-Length header when one was supplied."""
    value = _first_header(headers, "content-length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Parse numeric Retry-After values without guessing at HTTP date formats."""
    value = _first_header(headers, "retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0.0 and math.isfinite(parsed) else None


def _optional_string(value: object) -> str | None:
    """Return a non-empty provider identifier when present."""
    if type(value) is str and value.strip():
        return value.strip()
    return None


def _reasoning_tokens(
    details: object,
    raw_response: str,
    *,
    metadata: LLMResponseMetadata,
) -> int:
    """Extract optional reasoning-token usage from OpenAI-compatible usage data."""
    if details is None:
        return 0
    if not isinstance(details, dict):
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: completion token details invalid",
            raw_response=raw_response,
            metadata=metadata,
        )
    return _optional_non_negative_int(
        details.get("reasoning_tokens"),
        raw_response,
        metadata=metadata,
    )


def _resolve_api_key(api_key: str, api_key_env: str) -> str:
    stripped = api_key.strip()
    if stripped and stripped not in _UNSET_SENTINELS:
        return stripped
    if api_key_env.strip():
        env_value = os.environ.get(api_key_env.strip(), "").strip()
        if env_value:
            return env_value
    raise LLMClientError("LLM_API_KEY_NOT_CONFIGURED")


def _resolve_base_url(base_url: str, base_url_env: str) -> str:
    if base_url_env.strip():
        env_value = os.environ.get(base_url_env.strip(), "").strip()
        if env_value:
            return env_value
    return base_url.strip()


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _decode_response(
    raw_response: str,
    *,
    metadata: LLMResponseMetadata | None = None,
) -> _ParsedChatCompletion:
    response_metadata = metadata or LLMResponseMetadata(
        response_body_bytes=len(raw_response.encode("utf-8")),
        response_body_chars=len(raw_response),
    )
    if not raw_response.strip():
        raise LLMClientError(
            "LLM_RESPONSE_EMPTY_BODY: provider response body was empty",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise LLMClientError(
            "LLM_RESPONSE_NOT_JSON",
            raw_response=raw_response,
            metadata=response_metadata,
        ) from error
    if not isinstance(parsed, dict):
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: response must be object",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: choices missing",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: choice must be object",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: message missing",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    content = message.get("content")
    if content is None:
        content = ""
    elif type(content) is not str:
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: message.content must be string",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    finish_reason = first_choice.get("finish_reason")
    if type(finish_reason) is not str or not finish_reason:
        finish_reason = "unknown"
    usage = parsed.get("usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: usage must be object",
            raw_response=raw_response,
            metadata=response_metadata,
        )
    input_tokens = _optional_non_negative_int(
        usage.get("prompt_tokens"),
        raw_response,
        metadata=response_metadata,
    )
    output_tokens = _optional_non_negative_int(
        usage.get("completion_tokens"),
        raw_response,
        metadata=response_metadata,
    )
    reasoning_tokens = _reasoning_tokens(
        usage.get("completion_tokens_details"),
        raw_response,
        metadata=response_metadata,
    )
    completion_metadata = replace(
        response_metadata,
        provider_response_id=_optional_string(parsed.get("id")),
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    return _ParsedChatCompletion(
        content=content,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=completion_metadata,
    )


def _optional_non_negative_int(
    value: object,
    raw_response: str,
    *,
    metadata: LLMResponseMetadata,
) -> int:
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: token count invalid",
            raw_response=raw_response,
            metadata=metadata,
        )
    return value


def _validate_output_content(
    content: str,
    schema: Mapping[str, JsonValue],
    raw_response: str,
    *,
    metadata: LLMResponseMetadata,
) -> None:
    """Validate model JSON against the rendered prompt's portable schema subset."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise LLMClientError(
            "LLM_RESPONSE_SCHEMA_MISMATCH: content is not valid JSON",
            raw_response=raw_response,
            metadata=metadata,
        ) from error
    try:
        _validate_json_schema(value, schema, path="$")
    except ValueError as error:
        raise LLMClientError(
            f"LLM_RESPONSE_SCHEMA_MISMATCH: {error}",
            raw_response=raw_response,
            metadata=metadata,
        ) from error


def _validate_json_schema(
    value: object,
    schema: Mapping[str, JsonValue],
    *,
    path: str,
) -> None:
    """Validate the JSON Schema subset shared by the configured prompt templates."""
    raw_types = schema.get("type")
    if type(raw_types) is str:
        allowed_types = (raw_types,)
    elif isinstance(raw_types, list) and all(type(item) is str for item in raw_types):
        allowed_types = tuple(raw_types)
    else:
        raise ValueError(f"{path}: schema type is invalid")
    if not any(_matches_json_type(value, json_type) for json_type in allowed_types):
        expected = " or ".join(allowed_types)
        raise ValueError(f"{path}: expected {expected}")
    enum_values = schema.get("enum")
    if enum_values is not None:
        if not isinstance(enum_values, list):
            raise ValueError(f"{path}: schema enum is invalid")
        if value not in enum_values:
            raise ValueError(f"{path}: value is not one of {enum_values!r}")
    if isinstance(value, dict) and "object" in allowed_types:
        _validate_object_schema(value, schema, path=path)
    if isinstance(value, list) and "array" in allowed_types:
        _validate_array_bounds(value, schema, path=path)
        raw_items = schema.get("items")
        if raw_items is None:
            return
        if not isinstance(raw_items, Mapping):
            raise ValueError(f"{path}: schema items is invalid")
        for index, item in enumerate(value):
            _validate_json_schema(item, raw_items, path=f"{path}[{index}]")


def _validate_array_bounds(
    value: list[object],
    schema: Mapping[str, JsonValue],
    *,
    path: str,
) -> None:
    """Validate the array-size subset used by prompt output contracts."""
    for keyword, comparison, description in (
        ("minItems", lambda actual, expected: actual < expected, "at least"),
        ("maxItems", lambda actual, expected: actual > expected, "at most"),
    ):
        bound = schema.get(keyword)
        if bound is None:
            continue
        if type(bound) is not int or bound < 0:
            raise ValueError(f"{path}: schema {keyword} is invalid")
        if comparison(len(value), bound):
            raise ValueError(f"{path}: expected {description} {bound} items")


def _matches_json_type(value: object, json_type: str) -> bool:
    """Return whether a decoded JSON value has the requested JSON Schema type."""
    return (
        (json_type == "object" and isinstance(value, dict))
        or (json_type == "array" and isinstance(value, list))
        or (json_type == "string" and type(value) is str)
        or (json_type == "number" and type(value) in (int, float))
        or (json_type == "integer" and type(value) is int)
        or (json_type == "boolean" and type(value) is bool)
        or (json_type == "null" and value is None)
    )


def _validate_object_schema(
    value: dict[object, object],
    schema: Mapping[str, JsonValue],
    *,
    path: str,
) -> None:
    """Validate required object keys and declared child schemas."""
    raw_required = schema.get("required", [])
    if not isinstance(raw_required, list) or not all(
        type(item) is str for item in raw_required
    ):
        raise ValueError(f"{path}: schema required is invalid")
    raw_properties = schema.get("properties", {})
    if not isinstance(raw_properties, Mapping) or not all(
        type(key) is str and isinstance(child, Mapping)
        for key, child in raw_properties.items()
    ):
        raise ValueError(f"{path}: schema properties is invalid")
    for field_name in raw_required:
        if field_name not in value:
            raise ValueError(f"{path}: missing required field {field_name}")
    if schema.get("additionalProperties") is False:
        unknown = set(value) - set(raw_properties)
        if unknown:
            raise ValueError(f"{path}: unknown field {sorted(unknown)[0]}")
    for field_name, child_schema in raw_properties.items():
        if field_name in value:
            _validate_json_schema(
                value[field_name],
                child_schema,
                path=f"{path}.{field_name}",
            )


def _http_error_message(status: int) -> str:
    """Map provider HTTP status codes to stable execution-policy categories."""
    if status == 429:
        return "LLM_RATE_LIMITED: status=429"
    if 500 <= status <= 599:
        return f"LLM_SERVER_ERROR: status={status}"
    return f"LLM_HTTP_ERROR: status={status}"


def _is_timeout_error(error: BaseException) -> bool:
    """Return whether a stdlib transport exception represents a timeout."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(error, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    if isinstance(reason, OSError) and "timed out" in str(reason).lower():
        return True
    return "timed out" in str(error).lower()


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value
