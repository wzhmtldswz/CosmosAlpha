"""Audited, bounded-retry execution for provider-independent LLM requests."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from cosmosalpha.core import CoreValidationError, JsonScalar, ObjectRef, stable_sha256
from cosmosalpha.core.config import LLMConfig, PromptsConfig
from cosmosalpha.llm.client import LLMClient, LLMRequest, LLMResponseMetadata
from cosmosalpha.llm.errors import LLMClientError, LLMOutputError
from cosmosalpha.llm.prompts import PromptMessage
from cosmosalpha.llm.records import LLMCallRecord, LLMCallRecorder

PROMPT_RENDER_OBJECT_TYPE = "prompt_render"
LLM_CALL_RECORD_OBJECT_TYPE = "llm_call_record"


def _full_jitter(delay: float) -> float:
    """Return full jitter for a bounded retry delay."""
    return random.uniform(0.0, delay)


class LLMObjectRepository(Protocol):
    """Object repository subset required for LLM execution auditing."""

    def put_object(self, object_type: str, object_id: str, value: object) -> ObjectRef:
        """Persist a versioned object and return its reference."""


@dataclass(frozen=True, slots=True)
class LLMRetryPolicy:
    """Bounded retry settings for one live LLM execution."""

    max_timeout_retries: int
    max_rate_limit_retries: int
    max_server_retries: int
    max_format_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    max_transport_retries: int | None = None
    max_empty_retries: int | None = None
    max_incomplete_retries: int | None = None
    max_total_retries: int | None = None

    def __post_init__(self) -> None:
        """Validate bounded retry settings."""
        for field_name in (
            "max_timeout_retries",
            "max_rate_limit_retries",
            "max_server_retries",
            "max_format_retries",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise CoreValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in (
            "max_transport_retries",
            "max_empty_retries",
            "max_incomplete_retries",
            "max_total_retries",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise CoreValidationError(
                    f"{field_name} must be a non-negative integer or None"
                )
        for field_name in ("initial_backoff_seconds", "max_backoff_seconds"):
            value = getattr(self, field_name)
            if type(value) is not float or value < 0.0:
                raise CoreValidationError(f"{field_name} must be a non-negative float")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise CoreValidationError(
                "max_backoff_seconds must not be less than initial_backoff_seconds"
            )

    @classmethod
    def from_config(
        cls,
        llm: LLMConfig,
        prompts: PromptsConfig,
    ) -> LLMRetryPolicy:
        """Create one retry policy from the immutable experiment configuration."""
        # An explicit ``llm.max_empty_retries`` wins (MCTS runs rely on it);
        # otherwise the empty-body budget falls back to the server-retry budget
        # so configurations that predate the dedicated field keep working.
        max_empty_retries = (
            llm.max_empty_retries
            if llm.max_empty_retries is not None
            else llm.max_server_retries
        )
        return cls(
            max_timeout_retries=llm.max_timeout_retries,
            max_rate_limit_retries=llm.max_rate_limit_retries,
            max_server_retries=llm.max_server_retries,
            max_format_retries=prompts.max_format_retries,
            max_empty_retries=max_empty_retries,
            initial_backoff_seconds=llm.retry_initial_backoff_seconds,
            max_backoff_seconds=llm.retry_max_backoff_seconds,
            max_transport_retries=llm.max_server_retries,
            max_incomplete_retries=prompts.max_format_retries,
            max_total_retries=(
                llm.max_timeout_retries
                + llm.max_rate_limit_retries
                + llm.max_server_retries
                + prompts.max_format_retries
            ),
        )


@dataclass(slots=True)
class AuditedLLMExecutor:
    """Persist prompts and calls while applying a finite retry policy.

    This boundary is intentionally independent from the Agent DTOs.  It can therefore
    be reused by each Agent adapter without exposing provider failures to workflow
    orchestration or deterministic factor services.
    """

    client: LLMClient
    recorder: LLMCallRecorder
    object_repository: LLMObjectRepository
    retry_policy: LLMRetryPolicy
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    jitter: Callable[[float], float] = _full_jitter
    prompt_refs: dict[str, ObjectRef] = field(default_factory=dict, init=False)
    call_record_refs: list[ObjectRef] = field(default_factory=list, init=False)

    def complete_and_parse[ParsedT](
        self,
        request: LLMRequest,
        parse: Callable[[str], ParsedT],
    ) -> ParsedT:
        """Complete and strictly parse one request with persisted audit evidence.

        Args:
            request: Fully rendered provider-independent request.
            parse: Strict parser for the expected Agent draft.

        Returns:
            The parsed draft returned by ``parse``.

        Raises:
            LLMClientError: When a non-retryable client failure occurs or its retry
                budget is exhausted.
            LLMOutputError: When strict format retries are exhausted.
        """
        active_request = request
        self._persist_prompt(active_request)
        attempt = 0
        execution_id = (
            "llm-execution-"
            + stable_sha256(
                {
                    "prompt_render_id": request.prompt.prompt_render_id,
                    "model_alias": request.model_alias,
                    "max_output_tokens": request.max_output_tokens,
                    "response_format": request.response_format,
                }
            )[:24]
        )
        retries_by_kind = {
            "timeout": 0,
            "rate_limit": 0,
            "server": 0,
            "transport": 0,
            "empty": 0,
            "incomplete": 0,
            "format": 0,
        }
        total_retries = 0
        while True:
            attempt += 1
            parameters = self._parameters(active_request, execution_id)
            self._persist_record(
                self.recorder.record_started(
                    active_request,
                    attempt=attempt,
                    parameters=parameters,
                    step_id=f"llm:{active_request.prompt.prompt_id}",
                )
            )
            started_at = self.monotonic()
            try:
                response = self.client.complete(active_request)
            except LLMClientError as error:
                elapsed_milliseconds = _elapsed_milliseconds(
                    started_at,
                    self.monotonic(),
                )
                retry_kind = _retry_kind(str(error))
                retryable = retry_kind is not None and self._can_retry(
                    retry_kind,
                    retries_by_kind,
                    total_retries,
                )
                metadata = _error_metadata(error)
                retry_delay = (
                    self._backoff_delay(
                        total_retries + 1,
                        retry_after=(
                            metadata.retry_after_seconds
                            if metadata is not None
                            else None
                        ),
                    )
                    if retryable
                    else 0.0
                )
                self._persist_record(
                    self.recorder.record_failure(
                        active_request,
                        error,
                        retryable=retryable,
                        raw_response=error.raw_response,
                        metadata=metadata,
                        attempt=attempt,
                        parameters={
                            **parameters,
                            "retry_kind": retry_kind or "none",
                            "retry_delay_seconds": retry_delay,
                        },
                        step_id=f"llm:{active_request.prompt.prompt_id}",
                        elapsed_milliseconds=elapsed_milliseconds,
                    )
                )
                if not retryable:
                    if retry_kind == "format":
                        raise LLMOutputError(
                            "LLM_FORMAT_RETRY_EXHAUSTED: provider response format failed"
                        ) from error
                    raise
                retries_by_kind[retry_kind] += 1
                total_retries += 1
                if retry_kind == "format":
                    active_request = _with_format_retry_feedback(
                        active_request,
                        str(error),
                    )
                    self._persist_prompt(active_request)
                if retry_delay > 0.0:
                    self.sleep(retry_delay)
                continue
            except CoreValidationError as error:
                # A local client-contract failure (for example an invalid
                # response object) must never be relabeled as a provider empty
                # body and blindly retried.  Record it as a distinct
                # non-retryable client-validation failure and surface it.
                response_error = LLMClientError(
                    f"LLM_CLIENT_RESPONSE_INVALID: {error}",
                    metadata=LLMResponseMetadata(raw_source="local_diagnostic"),
                )
                elapsed_milliseconds = _elapsed_milliseconds(
                    started_at,
                    self.monotonic(),
                )
                self._persist_record(
                    self.recorder.record_failure(
                        active_request,
                        response_error,
                        retryable=False,
                        raw_response=None,
                        metadata=response_error.metadata,
                        attempt=attempt,
                        parameters={
                            **parameters,
                            "retry_kind": "none",
                            "retry_delay_seconds": 0.0,
                        },
                        step_id=f"llm:{active_request.prompt.prompt_id}",
                        elapsed_milliseconds=elapsed_milliseconds,
                    )
                )
                raise response_error from error

            elapsed_milliseconds = _elapsed_milliseconds(
                started_at,
                self.monotonic(),
            )

            try:
                parsed = parse(response.content)
            except LLMOutputError as error:
                retryable = self._can_retry(
                    "format",
                    retries_by_kind,
                    total_retries,
                )
                retry_delay = (
                    self._backoff_delay(total_retries + 1) if retryable else 0.0
                )
                record = self.recorder.record_parse_failure(
                    active_request,
                    response,
                    error,
                    retryable=retryable,
                    attempt=attempt,
                    parameters={
                        **parameters,
                        "retry_kind": "format",
                        "retry_delay_seconds": retry_delay,
                    },
                    step_id=f"llm:{active_request.prompt.prompt_id}",
                    elapsed_milliseconds=elapsed_milliseconds,
                )
                self._persist_record(record)
                if not retryable:
                    raise LLMOutputError(
                        "LLM_FORMAT_RETRY_EXHAUSTED: strict output parsing failed"
                    ) from error
                retries_by_kind["format"] += 1
                total_retries += 1
                active_request = _with_format_retry_feedback(
                    active_request,
                    str(error),
                )
                self._persist_prompt(active_request)
                if retry_delay > 0.0:
                    self.sleep(retry_delay)
                continue

            self._persist_record(
                self.recorder.record_success(
                    active_request,
                    response,
                    attempt=attempt,
                    parameters=parameters,
                    step_id=f"llm:{active_request.prompt.prompt_id}",
                    elapsed_milliseconds=elapsed_milliseconds,
                )
            )
            return parsed

    def _persist_prompt(self, request: LLMRequest) -> ObjectRef:
        prompt = request.prompt
        ref = self.object_repository.put_object(
            PROMPT_RENDER_OBJECT_TYPE,
            prompt.prompt_render_id,
            prompt,
        )
        self.prompt_refs[prompt.prompt_render_id] = ref
        return ref

    def _persist_record(self, record: LLMCallRecord) -> ObjectRef:
        ref = self.object_repository.put_object(
            LLM_CALL_RECORD_OBJECT_TYPE,
            record.call_id,
            record,
        )
        self.call_record_refs.append(ref)
        return ref

    def _can_retry(
        self,
        kind: str,
        retries_by_kind: dict[str, int],
        total_retries: int,
    ) -> bool:
        limits = {
            "timeout": self.retry_policy.max_timeout_retries,
            "rate_limit": self.retry_policy.max_rate_limit_retries,
            "server": self.retry_policy.max_server_retries,
            "transport": (
                self.retry_policy.max_transport_retries
                if self.retry_policy.max_transport_retries is not None
                else self.retry_policy.max_server_retries
            ),
            "empty": (
                self.retry_policy.max_empty_retries
                if self.retry_policy.max_empty_retries is not None
                else self.retry_policy.max_server_retries
            ),
            "incomplete": (
                self.retry_policy.max_incomplete_retries
                if self.retry_policy.max_incomplete_retries is not None
                else self.retry_policy.max_format_retries
            ),
            "format": self.retry_policy.max_format_retries,
        }
        total_limit = self.retry_policy.max_total_retries
        if total_limit is None:
            total_limit = (
                self.retry_policy.max_timeout_retries
                + self.retry_policy.max_rate_limit_retries
                + self.retry_policy.max_server_retries
                + self.retry_policy.max_format_retries
                + (
                    self.retry_policy.max_transport_retries
                    if self.retry_policy.max_transport_retries is not None
                    else 0
                )
                + (
                    self.retry_policy.max_empty_retries
                    if self.retry_policy.max_empty_retries is not None
                    else 0
                )
                + (
                    self.retry_policy.max_incomplete_retries
                    if self.retry_policy.max_incomplete_retries is not None
                    else 0
                )
            )
        return retries_by_kind[kind] < limits[kind] and total_retries < total_limit

    def _backoff_delay(
        self,
        retry_number: int,
        *,
        retry_after: float | None = None,
    ) -> float:
        exponential_delay = min(
            self.retry_policy.initial_backoff_seconds * (2 ** (retry_number - 1)),
            self.retry_policy.max_backoff_seconds,
        )
        upper_bound = exponential_delay
        if retry_after is not None:
            upper_bound = min(
                self.retry_policy.max_backoff_seconds,
                max(upper_bound, retry_after),
            )
        if upper_bound <= 0.0:
            return 0.0
        delay = self.jitter(upper_bound)
        if type(delay) is not float or delay < 0.0 or not math.isfinite(delay):
            raise CoreValidationError("jitter must return a non-negative float")
        delay = min(delay, self.retry_policy.max_backoff_seconds)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.retry_policy.max_backoff_seconds))
        return delay

    def _parameters(
        self,
        request: LLMRequest,
        execution_id: str,
    ) -> dict[str, JsonScalar]:
        return {
            "max_output_tokens": request.max_output_tokens,
            "model_alias": request.model_alias,
            "response_format": request.response_format,
            "execution_id": execution_id,
        }


def _with_format_retry_feedback(request: LLMRequest, error: str) -> LLMRequest:
    """Add concise deterministic validation feedback to the next retry prompt."""
    prompt = request.prompt
    last_message = prompt.messages[-1]
    correction = (
        "\n\nFormat retry correction:\n"
        f"The previous response was rejected because: {error}\n"
        "Correct this issue on the next attempt. Return only one JSON object "
        "satisfying the authoritative output contract above."
    )
    messages = (
        *prompt.messages[:-1],
        replace(last_message, content=last_message.content + correction),
    )
    prompt_render_id = stable_sha256(
        {
            "messages": messages,
            "prompt_id": prompt.prompt_id,
            "template_checksum": prompt.template_checksum,
            "template_version": prompt.template_version,
        }
    )
    rendered = replace(
        prompt,
        prompt_render_id=f"prompt-render-{prompt_render_id}",
        messages=messages,
        estimated_input_tokens=_estimate_prompt_tokens(messages),
        truncation_report="format-retry-feedback",
    )
    return replace(request, prompt=rendered)


def _estimate_prompt_tokens(messages: tuple[PromptMessage, ...]) -> int:
    """Estimate prompt tokens using the renderer's conservative heuristic."""
    total_chars = sum(len(message.role) + len(message.content) for message in messages)
    return max(1, (total_chars + 3) // 4)


def _retry_kind(message: str) -> str | None:
    """Return the retry budget selected by a stable client error code."""
    code = message.partition(":")[0]
    if code == "LLM_TIMEOUT":
        return "timeout"
    if code == "LLM_RATE_LIMITED":
        return "rate_limit"
    if code == "LLM_SERVER_ERROR":
        return "server"
    if code in {"LLM_TRANSPORT_ERROR", "LLM_RESPONSE_TRUNCATED"}:
        return "transport"
    if code in {
        "LLM_RESPONSE_EMPTY",
        "LLM_RESPONSE_EMPTY_BODY",
        "LLM_RESPONSE_EMPTY_CONTENT",
    }:
        return "empty"
    if code == "LLM_RESPONSE_INCOMPLETE":
        return "incomplete"
    if code in {
        "LLM_RESPONSE_NOT_JSON",
        "LLM_RESPONSE_SCHEMA_MISMATCH",
    }:
        return "format"
    return None


def _error_metadata(error: LLMClientError) -> LLMResponseMetadata | None:
    """Return validated response metadata attached to a client error."""
    metadata = getattr(error, "metadata", None)
    return metadata if isinstance(metadata, LLMResponseMetadata) else None


def _elapsed_milliseconds(started_at: float, finished_at: float) -> int:
    """Return a non-negative monotonic duration rounded down to milliseconds."""
    return max(0, int((finished_at - started_at) * 1000))
