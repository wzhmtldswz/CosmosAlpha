"""Deterministic construction of structured experiment lessons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    ExperimentLesson,
    ExperimentLessonDesign,
    ExperimentLessonFactorOutcome,
    ExperimentLessonMetric,
    FactorCalculationResult,
    FactorDesign,
    FactorPreprocessingResult,
    FactorProposal,
    FactorSpec,
    FeedbackRequest,
    ResearchFeedback,
    ResearchHypothesis,
    ValidationReport,
)
from cosmosalpha.core import CoreValidationError, ObjectRef

EXPERIMENT_LESSON_OBJECT_TYPE = "experiment_lesson"
EXPERIMENT_LESSON_SCHEMA_VERSION = "experiment-lesson/1.0"


class LessonObjectResolver(Protocol):
    """Resolve immutable facts required to build one lesson."""

    def get_object[ObjectT](
        self,
        ref: ObjectRef,
        target_type: type[ObjectT],
    ) -> ObjectT:
        """Resolve one persisted domain object."""


class ExperimentLessonBuilder:
    """Build one immutable Lesson without asking an LLM to restate facts."""

    def __init__(self, resolver: LessonObjectResolver) -> None:
        """Create a builder backed by the object repository."""
        self._resolver = resolver

    def build(
        self,
        *,
        experiment_id: str,
        run_id: str,
        cycle_id: str,
        request: FeedbackRequest,
        request_ref: ObjectRef,
        feedback: ResearchFeedback,
        feedback_ref: ObjectRef,
    ) -> ExperimentLesson:
        """Resolve deterministic experiment facts and retain Feedback interpretation."""
        hypothesis = self._resolve(request.hypothesis_ref, ResearchHypothesis)
        designs = self._resolve_many(request.factor_design_refs, FactorDesign)
        proposals = (
            self._resolve_many(request.factor_proposal_refs, FactorProposal)
            if request.factor_design_refs
            else ()
        )
        reports = self._resolve_many(
            request.validation_report_refs,
            ValidationReport,
        )
        factors = self._resolve_many(request.factor_refs, FactorSpec)
        calculations = self._resolve_many(
            request.calculation_result_refs,
            FactorCalculationResult,
        )
        preprocessing = self._resolve_many(
            request.factor_preprocessing_result_refs,
            FactorPreprocessingResult,
        )
        evaluations = self._resolve_many(
            request.evaluation_result_refs,
            EvaluationResult,
        )
        decisions = self._resolve_many(
            request.acceptance_decision_refs,
            AcceptanceDecision,
        )
        if feedback_ref.object_id != feedback.feedback_id:
            raise CoreValidationError("feedback_ref must reference feedback")
        if feedback.evidence_refs != _feedback_source_refs(request):
            raise CoreValidationError(
                "feedback evidence_refs must match FeedbackRequest sources"
            )

        design_summaries = tuple(
            ExperimentLessonDesign(
                design_id=design.design_id,
                name=design.name,
                mechanism_mapping=design.mechanism_mapping,
                signal_definition=design.signal_definition,
                formulation=design.formulation,
                expected_direction=design.expected_direction,
                failure_conditions=design.failure_conditions,
            )
            for design in designs
        )
        factor_outcomes = tuple(
            _factor_outcome(
                design,
                proposals=proposals,
                reports=reports,
                factors=factors,
                calculations=calculations,
                preprocessing=preprocessing,
                evaluations=evaluations,
                decisions=decisions,
            )
            for design in designs
        )
        key_metrics = tuple(
            sorted(
                (
                    ExperimentLessonMetric(
                        factor_id=evaluation.factor_id,
                        name=metric.name,
                        value=metric.value,
                        direction=metric.direction,
                        sample_count=metric.sample_count,
                        calculation_version=metric.calculation_version,
                    )
                    for evaluation in evaluations
                    for metric in evaluation.metrics
                ),
                key=lambda metric: (metric.factor_id, metric.name),
            )
        )
        failure_codes = _stable_failure_codes(request, reports, decisions)
        source_refs = _ordered_unique_refs(
            (
                request_ref,
                *_feedback_source_refs(request),
                feedback_ref,
            )
        )
        direction_refs = tuple(
            ref
            for ref in hypothesis.source_refs
            if ref.object_type == "research_direction"
        )
        if len(direction_refs) > 1:
            raise CoreValidationError(
                "hypothesis must not reference multiple research directions"
            )
        return ExperimentLesson(
            lesson_id=experiment_id,
            experiment_id=experiment_id,
            run_id=run_id,
            cycle_id=cycle_id,
            research_direction_id=(
                None if not direction_refs else direction_refs[0].object_id
            ),
            hypothesis_statement=hypothesis.statement,
            hypothesis_mechanism=hypothesis.mechanism,
            design_summaries=design_summaries,
            factor_outcomes=factor_outcomes,
            key_metrics=key_metrics,
            stable_failure_codes=failure_codes,
            hypothesis_verdict=feedback.hypothesis_verdict,
            effective_lessons=_ordered_unique_strings(
                (
                    *feedback.supported_findings,
                    *feedback.rejected_findings,
                    *feedback.failure_analysis,
                )
            ),
            next_actions=feedback.next_actions,
            hypothesis_ref=request.hypothesis_ref,
            feedback_ref=feedback_ref,
            source_refs=source_refs,
            schema_version=EXPERIMENT_LESSON_SCHEMA_VERSION,
        )

    def _resolve_many[ObjectT](
        self,
        refs: tuple[ObjectRef, ...],
        target_type: type[ObjectT],
    ) -> tuple[ObjectT, ...]:
        return tuple(self._resolve(ref, target_type) for ref in refs)

    def _resolve[ObjectT](
        self,
        ref: ObjectRef,
        target_type: type[ObjectT],
    ) -> ObjectT:
        value = self._resolver.get_object(ref, target_type)
        if getattr(value, "schema_version", None) != ref.schema_version:
            raise CoreValidationError(
                "resolved lesson source schema does not match its ObjectRef"
            )
        return value


def lesson_ref_for_experiment(experiment_id: str) -> ObjectRef:
    """Return the stable one-to-one Lesson association for an experiment."""
    if type(experiment_id) is not str or not experiment_id:
        raise CoreValidationError("experiment_id must be a non-empty string")
    return ObjectRef(
        object_type=EXPERIMENT_LESSON_OBJECT_TYPE,
        object_id=experiment_id,
        schema_version=EXPERIMENT_LESSON_SCHEMA_VERSION,
    )


def _factor_outcome(
    design: FactorDesign,
    *,
    proposals: tuple[FactorProposal, ...],
    reports: tuple[ValidationReport, ...],
    factors: tuple[FactorSpec, ...],
    calculations: tuple[FactorCalculationResult, ...],
    preprocessing: tuple[FactorPreprocessingResult, ...],
    evaluations: tuple[EvaluationResult, ...],
    decisions: tuple[AcceptanceDecision, ...],
) -> ExperimentLessonFactorOutcome:
    design_proposals = tuple(
        proposal
        for proposal in proposals
        if proposal.factor_design_ref.object_id == design.design_id
    )
    proposal = (
        None
        if not design_proposals
        else max(
            design_proposals,
            key=lambda item: (item.generation_attempt, item.proposal_id),
        )
    )
    report = (
        None
        if proposal is None
        else next(
            (item for item in reports if item.proposal_id == proposal.proposal_id),
            None,
        )
    )
    factor = (
        None
        if proposal is None
        else next(
            (item for item in factors if item.proposal_id == proposal.proposal_id),
            None,
        )
    )
    if report is not None and report.is_valid != (factor is not None):
        raise CoreValidationError("lesson ValidationReport and FactorSpec disagree")
    factor_id = None if factor is None else factor.factor_id
    calculation_completed = factor_id is not None and any(
        item.factor_id == factor_id for item in calculations
    )
    preprocessing_completed = factor_id is not None and any(
        item.factor_id == factor_id for item in preprocessing
    )
    evaluation_completed = factor_id is not None and any(
        item.factor_id == factor_id for item in evaluations
    )
    decision = (
        None
        if factor_id is None
        else next(
            (item for item in decisions if item.factor_id == factor_id),
            None,
        )
    )
    return ExperimentLessonFactorOutcome(
        design_id=design.design_id,
        design_name=design.name,
        proposal_id=None if proposal is None else proposal.proposal_id,
        factor_id=factor_id,
        canonical_expression=None if factor is None else factor.expression,
        validation_passed=None if report is None else report.is_valid,
        calculation_completed=calculation_completed,
        preprocessing_completed=preprocessing_completed,
        evaluation_completed=evaluation_completed,
        accepted=None if decision is None else decision.accepted,
        acceptance_reason_codes=(() if decision is None else decision.reason_codes),
    )


def _stable_failure_codes(
    request: FeedbackRequest,
    reports: tuple[ValidationReport, ...],
    decisions: tuple[AcceptanceDecision, ...],
) -> tuple[str, ...]:
    codes = {issue.code for report in reports for issue in report.issues}
    codes.update(failure.error_code for failure in request.failures)
    codes.update(
        code
        for decision in decisions
        if not decision.accepted
        for code in decision.reason_codes
        if not code.startswith("PASS_")
    )
    return tuple(sorted(codes))


def _feedback_source_refs(request: FeedbackRequest) -> tuple[ObjectRef, ...]:
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


def _ordered_unique_refs(refs: Iterable[ObjectRef]) -> tuple[ObjectRef, ...]:
    return tuple(dict.fromkeys(refs))


def _ordered_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
