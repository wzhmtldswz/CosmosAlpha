"""LLM and prompt infrastructure errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cosmosalpha.core import ObjectRef

if TYPE_CHECKING:
    from cosmosalpha.llm.client import LLMResponseMetadata


class PromptError(Exception):
    """Raised when a prompt template cannot be loaded or rendered."""


class PromptViewBuildError(PromptError):
    """Describe a deterministic failure while resolving a Prompt View."""

    def __init__(self, code: str, ref: ObjectRef, detail: str) -> None:
        """Create an error with a stable code and failed object reference."""
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.ref = ref


class LLMOutputError(Exception):
    """Raised when an LLM response does not match the expected draft schema."""


class LLMClientError(Exception):
    """Raised when a provider request or provider response is invalid."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        metadata: LLMResponseMetadata | None = None,
    ) -> None:
        """Create an error that can retain a provider response for audit storage.

        Args:
            message: Stable error code and human-readable diagnostic.
            raw_response: Optional unmodified provider response body.
            metadata: Optional transport and provider evidence.
        """
        super().__init__(message)
        self.raw_response = raw_response
        self.metadata = metadata


class ReplayError(Exception):
    """Raised when a replay client cannot resolve a deterministic response."""
