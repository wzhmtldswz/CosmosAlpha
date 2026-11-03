"""Deterministic feedback adapter used by Graph Search."""

from __future__ import annotations

from cosmosalpha.agents import FeedbackRequest, HypothesisVerdict, ResearchFeedback
from cosmosalpha.core import CoreValidationError, ObjectRef, stable_sha256


class GraphFeedbackStub:
    """Build persistence-compatible feedback without an LLM request.

    Graph selection and acceptance already consume deterministic evaluation
    outputs. This adapter retains the existing evidence references and any
    structured failures while deliberately making no additional research
    interpretation.
    """

    def generate(
        self,
        request: FeedbackRequest,
        *,
        cycle_id: str,
    ) -> ResearchFeedback:
        """Return stable, neutral feedback for one Graph cycle."""
        if not isinstance(request, FeedbackRequest):
            raise CoreValidationError("request must be FeedbackRequest")
        if type(cycle_id) is not str or not cycle_id:
            raise CoreValidationError("cycle_id must be a non-empty string")
        failure_analysis = tuple(
            f"{failure.stage}:{failure.error_code}" for failure in request.failures
        )
        feedback_id = (
            "graph-feedback-"
            + stable_sha256({"cycle_id": cycle_id, "request": request})[:24]
        )
        return ResearchFeedback(
            feedback_id=feedback_id,
            cycle_id=cycle_id,
            hypothesis_verdict=HypothesisVerdict.INCONCLUSIVE,
            summary=(
                "Graph deterministic feedback: LLM interpretation bypassed; "
                "deterministic evidence retained."
            ),
            supported_findings=(),
            rejected_findings=(),
            failure_analysis=failure_analysis,
            next_actions=(),
            next_hypothesis_hint=None,
            evidence_refs=_feedback_source_refs(request),
            schema_version="research-feedback/2.0",
        )


def _feedback_source_refs(request: FeedbackRequest) -> tuple[ObjectRef, ...]:
    """Return feedback evidence in the canonical lesson-builder order."""
    return (
        request.hypothesis_ref,
        *request.factor_design_refs,
        *request.factor_proposal_refs,
        *request.factor_refs,
        *request.validation_report_refs,
        *request.calculation_result_refs,
        *request.factor_preprocessing_result_refs,
        *request.evaluation_result_refs,
        *request.acceptance_decision_refs,
        *request.comparison_baseline_refs,
    )
