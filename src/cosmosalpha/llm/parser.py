"""Strict parsing for LLM draft DTOs."""

from __future__ import annotations

from cosmosalpha.agents import (
    FactorDesignDraftBatch,
    FactorExpressionDraft,
    FeedbackDraft,
    HypothesisDraft,
)
from cosmosalpha.core import from_json
from cosmosalpha.llm.errors import LLMOutputError


class AgentOutputParser:
    """Parse raw model JSON into strictly typed model-owned draft DTOs."""

    def parse_hypothesis(self, content: str) -> HypothesisDraft:
        """Parse a hypothesis draft."""
        return _parse(content, HypothesisDraft)

    def parse_factor_design_batch(self, content: str) -> FactorDesignDraftBatch:
        """Parse a structured factor design draft batch."""
        return _parse(content, FactorDesignDraftBatch)

    def parse_factor_expression(self, content: str) -> FactorExpressionDraft:
        """Parse one factor expression draft."""
        return _parse(content, FactorExpressionDraft)

    def parse_feedback(self, content: str) -> FeedbackDraft:
        """Parse a feedback draft."""
        return _parse(content, FeedbackDraft)


def _parse[DraftT](content: str, target_type: type[DraftT]) -> DraftT:
    try:
        return from_json(content, target_type)
    except Exception as error:
        detail = str(error) or error.__class__.__name__
        raise LLMOutputError(f"LLM_OUTPUT_SCHEMA_MISMATCH: {detail}") from error
