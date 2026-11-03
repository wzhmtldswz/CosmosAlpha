"""LLM call records and raw-response artifact helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    FailureInfo,
    JsonScalar,
    stable_sha256,
)
from cosmosalpha.llm.client import LLMRequest, LLMResponse, LLMResponseMetadata
from cosmosalpha.llm.errors import LLMOutputError

LLM_CALL_RECORD_SCHEMA_VERSION = "llm-call-record/1.0"
_RAW_RESPONSE_KIND = "llm_raw_response"
_RAW_RESPONSE_MEDIA_TYPE = "application/json"
_VALID_STATUSES = {"started", "succeeded", "parse_failed", "failed"}


class ArtifactRepository(Protocol):
    """Artifact repository subset needed by LLM call recording."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist bytes and return an artifact reference."""


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """Immutable audit record for one LLM completion attempt."""

    call_id: str
    prompt_render_id: str
    model_alias: str
    redacted_parameters: dict[str, JsonScalar]
    attempt: int
    raw_response_ref: ArtifactRef
    finish_reason: str
    input_tokens: int
    output_tokens: int
    status: str
    failure: FailureInfo | None
    schema_version: str
    step_id: str = "llm"
    elapsed_milliseconds: int = 0

    def __post_init__(self) -> None:
        """Validate record fields."""
        _require_string(self.call_id, "call_id")
        _require_string(self.prompt_render_id, "prompt_render_id")
        _require_string(self.model_alias, "model_alias")
        _require_string(self.step_id, "step_id")
        if not isinstance(self.redacted_parameters, dict):
            raise CoreValidationError("redacted_parameters must be a dict")
        for key, value in self.redacted_parameters.items():
            _require_string(key, "redacted_parameters key")
            if value is None or type(value) not in (bool, int, float, str):
                raise CoreValidationError("redacted_parameters values must be scalar")
        if type(self.attempt) is not int or self.attempt < 1:
            raise CoreValidationError("attempt must be positive")
        if not isinstance(self.raw_response_ref, ArtifactRef):
            raise CoreValidationError("raw_response_ref must be ArtifactRef")
        _require_string(self.finish_reason, "finish_reason")
        if type(self.input_tokens) is not int or self.input_tokens < 0:
            raise CoreValidationError("input_tokens must be non-negative")
        if type(self.output_tokens) is not int or self.output_tokens < 0:
            raise CoreValidationError("output_tokens must be non-negative")
        if type(self.elapsed_milliseconds) is not int or self.elapsed_milliseconds < 0:
            raise CoreValidationError("elapsed_milliseconds must be non-negative")
        if self.status not in _VALID_STATUSES:
            raise CoreValidationError("status is invalid")
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        if self.status in {"started", "succeeded"} and self.failure is not None:
            raise CoreValidationError("successful calls must not carry failure")
        if self.status not in {"started", "succeeded"} and self.failure is None:
            raise CoreValidationError("failed calls must carry failure")
        if self.schema_version != LLM_CALL_RECORD_SCHEMA_VERSION:
            raise CoreValidationError("invalid LLM call record schema_version")


class LLMCallRecorder:
    """Persist raw LLM responses and build deterministic call records."""

    def __init__(self, artifact_repository: ArtifactRepository) -> None:
        """Create a recorder bound to an artifact repository."""
        self._artifact_repository = artifact_repository
        self.records: list[LLMCallRecord] = []

    def record_success(
        self,
        request: LLMRequest,
        response: LLMResponse,
        *,
        raw_response: str | None = None,
        attempt: int = 1,
        parameters: dict[str, JsonScalar] | None = None,
        step_id: str = "llm",
        elapsed_milliseconds: int = 0,
    ) -> LLMCallRecord:
        """Record a successfully parsed LLM call."""
        return self._record(
            request,
            response,
            raw_response=raw_response,
            attempt=attempt,
            parameters=parameters,
            status="succeeded",
            failure=None,
            step_id=step_id,
            elapsed_milliseconds=elapsed_milliseconds,
        )

    def record_started(
        self,
        request: LLMRequest,
        *,
        attempt: int = 1,
        parameters: dict[str, JsonScalar] | None = None,
        step_id: str = "llm",
    ) -> LLMCallRecord:
        """Persist an audit record before an external completion is issued."""
        response = LLMResponse(
            content="started",
            model_alias=request.model_alias,
            prompt_render_id=request.prompt.prompt_render_id,
            finish_reason="started",
            input_tokens=0,
            output_tokens=0,
            raw_response="started",
        )
        return self._record(
            request,
            response,
            raw_response="started",
            attempt=attempt,
            parameters=parameters,
            status="started",
            failure=None,
            step_id=step_id,
            elapsed_milliseconds=0,
        )

    def record_parse_failure(
        self,
        request: LLMRequest,
        response: LLMResponse,
        error: LLMOutputError,
        *,
        raw_response: str | None = None,
        retryable: bool = True,
        attempt: int = 1,
        parameters: dict[str, JsonScalar] | None = None,
        step_id: str = "llm",
        elapsed_milliseconds: int = 0,
    ) -> LLMCallRecord:
        """Record an LLM output that failed strict schema parsing."""
        failure = FailureInfo(
            stage="llm_parse",
            error_code="LLM_OUTPUT_SCHEMA_MISMATCH",
            message=str(error) or "LLM_OUTPUT_SCHEMA_MISMATCH",
            retryable=retryable,
        )
        return self._record(
            request,
            response,
            raw_response=raw_response,
            attempt=attempt,
            parameters=parameters,
            status="parse_failed",
            failure=failure,
            step_id=step_id,
            elapsed_milliseconds=elapsed_milliseconds,
        )

    def record_failure(
        self,
        request: LLMRequest,
        error: Exception,
        *,
        retryable: bool,
        raw_response: str | None = None,
        metadata: LLMResponseMetadata | None = None,
        attempt: int = 1,
        parameters: dict[str, JsonScalar] | None = None,
        step_id: str = "llm",
        elapsed_milliseconds: int = 0,
    ) -> LLMCallRecord:
        """Record a provider, transport, or response-format failure.

        Args:
            request: Rendered provider-independent request.
            error: Raised client error represented in the audit record.
            retryable: Whether the execution policy may issue another request.
            raw_response: Optional provider response or diagnostic text.
            metadata: Optional transport and provider evidence.
            attempt: One-based attempt sequence for this rendered prompt.
            parameters: Redacted request and runtime parameters.
            step_id: Stable workflow label associated with this LLM call.
            elapsed_milliseconds: Provider call duration rounded down to milliseconds.

        Returns:
            The persisted-response call record.
        """
        message = str(error) or error.__class__.__name__
        response_metadata = metadata or _error_metadata(error)
        if response_metadata is None:
            evidence = raw_response or ""
            response_metadata = LLMResponseMetadata(
                response_body_bytes=len(evidence.encode("utf-8")),
                response_body_chars=len(evidence),
                raw_source=(
                    "provider_body" if raw_response is not None else "local_diagnostic"
                ),
            )
        elif raw_response is None:
            response_metadata = replace(
                response_metadata,
                raw_source="local_diagnostic",
            )
        failure = FailureInfo(
            stage="llm_call",
            error_code=_error_code(message),
            message=message,
            retryable=retryable,
        )
        failure_raw = raw_response if raw_response is not None else message
        response = LLMResponse(
            content="unavailable",
            model_alias=request.model_alias,
            prompt_render_id=request.prompt.prompt_render_id,
            finish_reason=response_metadata.finish_reason or "error",
            input_tokens=response_metadata.input_tokens,
            output_tokens=response_metadata.output_tokens,
            raw_response=message,
            metadata=response_metadata,
        )
        return self._record(
            request,
            response,
            raw_response=failure_raw,
            attempt=attempt,
            parameters=parameters,
            status="failed",
            failure=failure,
            step_id=step_id,
            elapsed_milliseconds=elapsed_milliseconds,
        )

    def _record(
        self,
        request: LLMRequest,
        response: LLMResponse,
        *,
        raw_response: str | None,
        attempt: int,
        parameters: dict[str, JsonScalar] | None,
        status: str,
        failure: FailureInfo | None,
        step_id: str,
        elapsed_milliseconds: int,
    ) -> LLMCallRecord:
        if not isinstance(request, LLMRequest):
            raise CoreValidationError("request must be LLMRequest")
        if not isinstance(response, LLMResponse):
            raise CoreValidationError("response must be LLMResponse")
        raw = raw_response if raw_response is not None else response.raw_response
        if raw is None:
            raw = response.content
        raw_response_ref = self._artifact_repository.put_bytes(
            raw.encode("utf-8"),
            kind=_RAW_RESPONSE_KIND,
            media_type=_raw_response_media_type(response.metadata),
        )
        redacted_parameters = dict(
            parameters if parameters is not None else _request_parameters(request)
        )
        if response.metadata is not None:
            redacted_parameters.update(response.metadata.as_audit_parameters())
        call_id = _call_id(
            request,
            redacted_parameters,
            attempt,
            raw_response_ref,
            step_id,
        )
        record = LLMCallRecord(
            call_id=call_id,
            prompt_render_id=request.prompt.prompt_render_id,
            model_alias=request.model_alias,
            redacted_parameters=redacted_parameters,
            attempt=attempt,
            raw_response_ref=raw_response_ref,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            status=status,
            failure=failure,
            schema_version=LLM_CALL_RECORD_SCHEMA_VERSION,
            step_id=step_id,
            elapsed_milliseconds=elapsed_milliseconds,
        )
        self.records.append(record)
        return record


def parse_with_call_record[ParsedT](
    content: str,
    parse: Callable[[str], ParsedT],
    *,
    request: LLMRequest,
    response: LLMResponse,
    recorder: LLMCallRecorder,
    raw_response: str | None = None,
    attempt: int = 1,
    step_id: str = "llm",
    elapsed_milliseconds: int = 0,
) -> tuple[ParsedT, LLMCallRecord]:
    """Parse model content and record success or schema mismatch."""
    try:
        parsed = parse(content)
    except LLMOutputError as error:
        recorder.record_parse_failure(
            request,
            response,
            error,
            raw_response=raw_response,
            attempt=attempt,
            step_id=step_id,
            elapsed_milliseconds=elapsed_milliseconds,
        )
        raise
    record = recorder.record_success(
        request,
        response,
        raw_response=raw_response,
        attempt=attempt,
        step_id=step_id,
        elapsed_milliseconds=elapsed_milliseconds,
    )
    return parsed, record


def _request_parameters(request: LLMRequest) -> dict[str, JsonScalar]:
    return {
        "model_alias": request.model_alias,
        "max_output_tokens": request.max_output_tokens,
        "response_format": request.response_format,
    }


def _error_metadata(error: Exception) -> LLMResponseMetadata | None:
    """Return structured metadata attached to a provider/client exception."""
    metadata = getattr(error, "metadata", None)
    return metadata if isinstance(metadata, LLMResponseMetadata) else None


def _raw_response_media_type(metadata: LLMResponseMetadata | None) -> str:
    """Use text media type for local diagnostics and JSON for provider bodies."""
    if metadata is not None and metadata.raw_source == "local_diagnostic":
        return "text/plain"
    return _RAW_RESPONSE_MEDIA_TYPE


def _call_id(
    request: LLMRequest,
    parameters: dict[str, JsonScalar],
    attempt: int,
    raw_response_ref: ArtifactRef,
    step_id: str,
) -> str:
    stable_parameters = {
        key: value for key, value in parameters.items() if key != "retry_delay_seconds"
    }
    digest = stable_sha256(
        {
            "prompt_render_id": request.prompt.prompt_render_id,
            "model_alias": request.model_alias,
            "parameters": stable_parameters,
            "attempt": attempt,
            "raw_response_checksum": raw_response_ref.checksum,
            "step_id": step_id,
        }
    )
    return f"llm-call-{digest[:24]}"


def _error_code(message: str) -> str:
    """Extract the stable error code prefix from a client error message."""
    return message.partition(":")[0] or "LLM_CALL_FAILED"


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value
