"""Strongly typed, catalog-backed views for agent prompts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import ClassVar, Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    AcceptancePolicySnapshot,
    EvaluationResult,
    ExperimentLesson,
    ExperimentRecord,
    FactorCalculationResult,
    FactorDesign,
    FactorPreprocessingResult,
    FactorProposal,
    FactorSpec,
    FeedbackRequest,
    MetricValue,
    ResearchHypothesis,
    ValidationReport,
)
from cosmosalpha.core import (
    CoreError,
    CoreValidationError,
    FailureInfo,
    ObjectRef,
    WarningInfo,
    stable_sha256,
)
from cosmosalpha.evaluation.evaluator import METRIC_DESCRIPTIONS
from cosmosalpha.factors.expression import (
    FieldCatalog,
    FieldSpec,
    OperatorCatalog,
    OperatorParameterSpec,
    OperatorScope,
)
from cosmosalpha.llm.errors import PromptViewBuildError

PROMPT_CATALOG_VIEW_SCHEMA_VERSION = "prompt-catalog-view/1.0"
HYPOTHESIS_PROMPT_VIEW_SCHEMA_VERSION = "hypothesis-prompt-view/2.0"
FACTOR_DESIGN_PROMPT_VIEW_SCHEMA_VERSION = "factor-design-prompt-view/1.0"
FACTOR_EXPRESSION_PROMPT_VIEW_SCHEMA_VERSION = "factor-expression-prompt-view/1.0"
EXPRESSION_REPAIR_PROMPT_VIEW_SCHEMA_VERSION = "expression-repair-prompt-view/1.0"
FEEDBACK_PROMPT_VIEW_SCHEMA_VERSION = "feedback-prompt-view/1.0"


class PromptObjectResolver(Protocol):
    """Resolve strict domain objects from stable references."""

    def get_object[ObjectT](
        self, ref: ObjectRef, target_type: type[ObjectT]
    ) -> ObjectT:
        """Resolve and decode one referenced object."""


@dataclass(frozen=True, slots=True)
class PromptFieldView:
    """One expression field exposed to an LLM."""

    name: str
    description: str

    def __post_init__(self) -> None:
        """Validate prompt field metadata."""
        _require_string(self.name, "name")
        _require_string(self.description, "description")


@dataclass(frozen=True, slots=True)
class PromptOperatorView:
    """One canonical operator exposed to an LLM."""

    name: str
    canonical_signature: str
    description: str
    parameter_schema: tuple[OperatorParameterSpec, ...]
    scope: OperatorScope
    supported_windows: tuple[int, ...]
    required_context_summary: str

    def __post_init__(self) -> None:
        """Validate prompt operator metadata."""
        _require_string(self.name, "name")
        _require_string(self.canonical_signature, "canonical_signature")
        _require_string(self.description, "description")
        if not isinstance(self.parameter_schema, tuple) or not all(
            isinstance(parameter, OperatorParameterSpec)
            for parameter in self.parameter_schema
        ):
            raise CoreValidationError(
                "parameter_schema must be a tuple of OperatorParameterSpec"
            )
        if not isinstance(self.scope, OperatorScope):
            raise CoreValidationError("scope must be OperatorScope")
        if not isinstance(self.supported_windows, tuple) or not all(
            type(window) is int and window > 0 for window in self.supported_windows
        ):
            raise CoreValidationError(
                "supported_windows must be a tuple of positive integers"
            )
        _require_string(self.required_context_summary, "required_context_summary")


@dataclass(frozen=True, slots=True)
class PromptCatalogView:
    """Runtime fields and operators that a prompt may truthfully advertise."""

    fields: tuple[PromptFieldView, ...]
    operators: tuple[PromptOperatorView, ...]
    field_schema_version: str
    operator_set_version: str
    frequency: str
    backend: str
    schema_version: str = PROMPT_CATALOG_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the filtered catalog view."""
        if not isinstance(self.fields, tuple) or not all(
            isinstance(field, PromptFieldView) for field in self.fields
        ):
            raise CoreValidationError("fields must be a tuple of PromptFieldView")
        if not isinstance(self.operators, tuple) or not all(
            isinstance(operator, PromptOperatorView) for operator in self.operators
        ):
            raise CoreValidationError("operators must be a tuple of PromptOperatorView")
        _require_string(self.field_schema_version, "field_schema_version")
        _require_string(self.operator_set_version, "operator_set_version")
        _require_string(self.frequency, "frequency")
        _require_string(self.backend, "backend")
        if self.schema_version != PROMPT_CATALOG_VIEW_SCHEMA_VERSION:
            raise CoreValidationError("invalid prompt catalog view schema_version")

    @property
    def expanded_operator_count(self) -> int:
        """Return the number of concrete fixed-window operator capabilities."""
        return sum(len(operator.supported_windows) or 1 for operator in self.operators)


@dataclass(frozen=True, slots=True)
class _CatalogPromptView:
    """Shared immutable envelope for role-specific Catalog Prompt Views."""

    catalog: PromptCatalogView
    source_refs: tuple[ObjectRef, ...]
    content_checksum: str
    schema_version: str

    _EXPECTED_SCHEMA_VERSION: ClassVar[str] = ""

    def __post_init__(self) -> None:
        """Validate source provenance and deterministic content checksum."""
        if not isinstance(self.catalog, PromptCatalogView):
            raise CoreValidationError("catalog must be PromptCatalogView")
        if not isinstance(self.source_refs, tuple) or not all(
            isinstance(ref, ObjectRef) for ref in self.source_refs
        ):
            raise CoreValidationError("source_refs must be a tuple of ObjectRef")
        if not self.source_refs:
            raise CoreValidationError("source_refs must not be empty")
        if self.schema_version != self._EXPECTED_SCHEMA_VERSION:
            raise CoreValidationError("invalid role Prompt View schema_version")
        expected_checksum = _content_checksum(
            self.catalog,
            self.source_refs,
            self.schema_version,
        )
        if self.content_checksum != expected_checksum:
            raise CoreValidationError("content_checksum does not match Prompt View")


@dataclass(frozen=True, slots=True)
class HypothesisPromptView:
    """Runtime capabilities and bounded structured Lessons for Hypothesis."""

    catalog: PromptCatalogView
    lessons: tuple[ExperimentLesson, ...]
    source_refs: tuple[ObjectRef, ...]
    content_checksum: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate structured history provenance and checksum."""
        if not isinstance(self.catalog, PromptCatalogView):
            raise CoreValidationError("catalog must be PromptCatalogView")
        if not isinstance(self.lessons, tuple) or not all(
            isinstance(lesson, ExperimentLesson) for lesson in self.lessons
        ):
            raise CoreValidationError("lessons must be a tuple of ExperimentLesson")
        if not isinstance(self.source_refs, tuple) or not all(
            isinstance(ref, ObjectRef) for ref in self.source_refs
        ):
            raise CoreValidationError("source_refs must be a tuple of ObjectRef")
        if len(self.source_refs) < 2:
            raise CoreValidationError("source_refs must contain both Catalog refs")
        if self.schema_version != HYPOTHESIS_PROMPT_VIEW_SCHEMA_VERSION:
            raise CoreValidationError("invalid Hypothesis Prompt View schema_version")
        expected_checksum = _hypothesis_content_checksum(
            self.catalog,
            self.lessons,
            self.source_refs,
        )
        if self.content_checksum != expected_checksum:
            raise CoreValidationError("content_checksum does not match Prompt View")


@dataclass(frozen=True, slots=True)
class FactorDesignPromptView(_CatalogPromptView):
    """Catalog foundation for a future FactorDesign prompt."""

    _EXPECTED_SCHEMA_VERSION = FACTOR_DESIGN_PROMPT_VIEW_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FactorExpressionPromptView(_CatalogPromptView):
    """Catalog foundation for a future FactorExpression prompt."""

    _EXPECTED_SCHEMA_VERSION = FACTOR_EXPRESSION_PROMPT_VIEW_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ExpressionRepairPromptView(_CatalogPromptView):
    """Catalog foundation for a future ExpressionRepair prompt."""

    _EXPECTED_SCHEMA_VERSION = EXPRESSION_REPAIR_PROMPT_VIEW_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FactorCalculationSummaryView:
    """Scalar-only calculation evidence."""

    factor_id: str
    data_snapshot_id: str
    date_range: tuple[date, date]
    sample_count: int
    missing_count: int
    eligible_mask_applied: bool
    calculation_version: str


@dataclass(frozen=True, slots=True)
class FactorPreprocessingSummaryView:
    """Scalar-only preprocessing evidence."""

    factor_id: str
    data_snapshot_id: str
    preprocessing_protocol_id: str
    preprocessing_protocol_version: str
    neutralization_applied: bool
    excluded_sample_count: int
    effective_sample_count: int
    warnings: tuple[WarningInfo, ...]


@dataclass(frozen=True, slots=True)
class MetricEvidenceView:
    """One real metric and its deterministic semantics."""

    name: str
    value: float
    meaning: str
    direction: str
    sample_count: int
    calculation_version: str


@dataclass(frozen=True, slots=True)
class MetricDeltaView:
    """Current-minus-baseline metric comparison with direction-aware improvement."""

    name: str
    current_value: float
    baseline_value: float
    raw_delta: float
    direction: str
    directional_improvement: float


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceView:
    """Scalar evaluation and immutable acceptance outcome for one factor."""

    factor_id: str
    data_snapshot_id: str
    evaluation_protocol_id: str
    preprocessing_protocol_id: str
    metrics: tuple[MetricEvidenceView, ...]
    total_date_count: int | None
    effective_date_count: int
    effective_date_ratio: float | None
    effective_sample_count: int
    coverage: float
    warnings: tuple[WarningInfo, ...]
    evaluator_version: str
    accepted: bool | None
    reason_codes: tuple[str, ...]
    policy_id: str | None
    policy_version: str | None
    baseline_deltas: tuple[MetricDeltaView, ...]


@dataclass(frozen=True, slots=True)
class ComparisonBaselineView:
    """One selected comparable historical candidate."""

    experiment_id: str
    completed_at: datetime
    factor_id: str
    factor_name: str
    expression: str
    metrics: tuple[MetricEvidenceView, ...]
    accepted: bool | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeedbackPromptView:
    """Resolved scalar evidence supplied to the v2 Feedback Agent."""

    hypothesis: ResearchHypothesis
    factor_designs: tuple[FactorDesign, ...]
    expression_attempts: tuple[FactorProposal, ...]
    validation_reports: tuple[ValidationReport, ...]
    factors: tuple[FactorSpec, ...]
    calculation_summaries: tuple[FactorCalculationSummaryView, ...]
    preprocessing_summaries: tuple[FactorPreprocessingSummaryView, ...]
    evaluations: tuple[EvaluationEvidenceView, ...]
    failures: tuple[FailureInfo, ...]
    acceptance_policy: AcceptancePolicySnapshot
    comparison_baseline: ComparisonBaselineView | None
    source_refs: tuple[ObjectRef, ...]
    content_checksum: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate provenance and deterministic content checksum."""
        if self.schema_version != FEEDBACK_PROMPT_VIEW_SCHEMA_VERSION:
            raise CoreValidationError("invalid Feedback Prompt View schema_version")
        if not self.source_refs:
            raise CoreValidationError("source_refs must not be empty")
        expected = _feedback_content_checksum(
            hypothesis=self.hypothesis,
            factor_designs=self.factor_designs,
            expression_attempts=self.expression_attempts,
            validation_reports=self.validation_reports,
            factors=self.factors,
            calculation_summaries=self.calculation_summaries,
            preprocessing_summaries=self.preprocessing_summaries,
            evaluations=self.evaluations,
            failures=self.failures,
            acceptance_policy=self.acceptance_policy,
            comparison_baseline=self.comparison_baseline,
            source_refs=self.source_refs,
        )
        if self.content_checksum != expected:
            raise CoreValidationError("content_checksum does not match Prompt View")


class PromptViewBuilder:
    """Resolve runtime Catalog references into role-specific Prompt Views."""

    def __init__(self, resolver: PromptObjectResolver) -> None:
        """Create a builder backed by a strict object resolver."""
        self._resolver = resolver

    def build_catalog(
        self,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
    ) -> PromptCatalogView:
        """Resolve and filter the Catalog content available to an LLM."""
        _require_ref_type(field_catalog_ref, "field_catalog")
        _require_ref_type(operator_catalog_ref, "operator_catalog")
        _require_string(frequency, "frequency")
        _require_string(backend, "backend")
        field_catalog = self._resolve(
            field_catalog_ref,
            FieldCatalog,
            field_catalog_ref.schema_version,
        )
        operator_catalog = self._resolve(
            operator_catalog_ref,
            OperatorCatalog,
            operator_catalog_ref.schema_version,
        )

        fields = tuple(
            sorted(
                (
                    PromptFieldView(field.name, field.description)
                    for field in field_catalog.fields
                    if field.available and not field.context_only
                ),
                key=lambda field: field.name,
            )
        )
        available_context = {
            field.name: field for field in field_catalog.fields if field.available
        }
        operators = tuple(
            sorted(
                (
                    PromptOperatorView(
                        name=operator.name,
                        canonical_signature=operator.canonical_signature,
                        description=operator.description,
                        parameter_schema=operator.parameter_schema,
                        scope=operator.scope,
                        supported_windows=operator.supported_windows,
                        required_context_summary=_context_summary(
                            operator.required_context_fields,
                            available_context,
                        ),
                    )
                    for operator in operator_catalog.operators
                    if frequency in operator.supported_frequencies
                    and backend in operator.supported_backends
                    and all(
                        context_field in available_context
                        for context_field in operator.required_context_fields
                    )
                ),
                key=lambda operator: operator.name,
            )
        )
        if not fields or not operators:
            failed_ref = field_catalog_ref if not fields else operator_catalog_ref
            raise PromptViewBuildError(
                "PROMPT_VIEW_CATALOG_EMPTY",
                failed_ref,
                "no usable runtime capabilities remain after filtering",
            )
        return PromptCatalogView(
            fields=fields,
            operators=operators,
            field_schema_version=field_catalog.schema_version,
            operator_set_version=operator_catalog.version,
            frequency=frequency,
            backend=backend,
        )

    def build_hypothesis(
        self,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
        lesson_refs: tuple[ObjectRef, ...] = (),
    ) -> HypothesisPromptView:
        """Build runtime capabilities with fully resolved structured Lessons.

        hypothesis 只需要 catalog.fields（required_fields 校验），不需要 26 个
        算子的完整定义（那是 design/expression 的输入）。砍掉 operators 后
        prompt 从 ~29KB 降到 ~8KB，处理时间显著缩短，规避 DeepSeek 服务端
        ~60s 处理阈值导致的空 body 失败。
        """
        catalog = self.build_catalog(
            field_catalog_ref,
            operator_catalog_ref,
            frequency=frequency,
            backend=backend,
        )
        # 只保留 fields：hypothesis 不选择算子，operators 对其是纯噪音
        catalog = replace(catalog, operators=())
        for ref in lesson_refs:
            _require_ref_type(ref, "experiment_lesson")
        lessons = tuple(
            self._resolve(ref, ExperimentLesson, ref.schema_version)
            for ref in lesson_refs
        )
        source_refs = (field_catalog_ref, operator_catalog_ref, *lesson_refs)
        return HypothesisPromptView(
            catalog=catalog,
            lessons=lessons,
            source_refs=source_refs,
            content_checksum=_hypothesis_content_checksum(
                catalog,
                lessons,
                source_refs,
            ),
            schema_version=HYPOTHESIS_PROMPT_VIEW_SCHEMA_VERSION,
        )

    def build_factor_design(
        self,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
    ) -> FactorDesignPromptView:
        """Build the Catalog foundation for a FactorDesign prompt."""
        return self._build_role(
            FactorDesignPromptView,
            field_catalog_ref,
            operator_catalog_ref,
            frequency=frequency,
            backend=backend,
        )

    def build_factor_expression(
        self,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
    ) -> FactorExpressionPromptView:
        """Build the Catalog foundation for a FactorExpression prompt."""
        return self._build_role(
            FactorExpressionPromptView,
            field_catalog_ref,
            operator_catalog_ref,
            frequency=frequency,
            backend=backend,
        )

    def build_expression_repair(
        self,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
    ) -> ExpressionRepairPromptView:
        """Build the Catalog foundation for an ExpressionRepair prompt."""
        return self._build_role(
            ExpressionRepairPromptView,
            field_catalog_ref,
            operator_catalog_ref,
            frequency=frequency,
            backend=backend,
        )

    def _build_role[ViewT: _CatalogPromptView](
        self,
        view_type: type[ViewT],
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        *,
        frequency: str,
        backend: str,
    ) -> ViewT:
        catalog = self.build_catalog(
            field_catalog_ref,
            operator_catalog_ref,
            frequency=frequency,
            backend=backend,
        )
        source_refs = (field_catalog_ref, operator_catalog_ref)
        schema_version = view_type._EXPECTED_SCHEMA_VERSION
        return view_type(
            catalog=catalog,
            source_refs=source_refs,
            content_checksum=_content_checksum(
                catalog,
                source_refs,
                schema_version,
            ),
            schema_version=schema_version,
        )

    def _resolve[ObjectT](
        self,
        ref: ObjectRef,
        target_type: type[ObjectT],
        expected_schema_version: str,
    ) -> ObjectT:
        try:
            value = self._resolver.get_object(ref, target_type)
        except CoreError as error:
            raise PromptViewBuildError(
                "PROMPT_VIEW_REFERENCE_UNRESOLVED",
                ref,
                str(error),
            ) from error
        actual_schema_version = getattr(value, "schema_version", None)
        if actual_schema_version != expected_schema_version:
            raise PromptViewBuildError(
                "PROMPT_VIEW_REFERENCE_VERSION_MISMATCH",
                ref,
                "resolved object schema does not match its reference",
            )
        return value


class FeedbackPromptViewBuilder:
    """Resolve a FeedbackRequest into scalar, replayable evidence."""

    def __init__(self, resolver: PromptObjectResolver) -> None:
        """Create a builder backed by the object repository."""
        self._resolver = resolver

    def build(self, request: FeedbackRequest) -> FeedbackPromptView:
        """Resolve all current and baseline references without reading artifacts."""
        hypothesis = self._resolve(request.hypothesis_ref, ResearchHypothesis)
        designs = self._resolve_many(request.factor_design_refs, FactorDesign)
        proposals = self._resolve_many(request.factor_proposal_refs, FactorProposal)
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
        baseline = self._baseline(request.comparison_baseline_refs)
        source_refs = _feedback_source_refs(request)
        calculation_summaries = tuple(
            _calculation_summary(item) for item in calculations
        )
        preprocessing_summaries = tuple(
            _preprocessing_summary(item) for item in preprocessing
        )
        evaluation_views = tuple(
            _evaluation_evidence(
                evaluation,
                _decision_for(evaluation_ref, evaluation, decisions),
                baseline,
            )
            for evaluation_ref, evaluation in zip(
                request.evaluation_result_refs,
                evaluations,
                strict=True,
            )
        )
        checksum = _feedback_content_checksum(
            hypothesis=hypothesis,
            factor_designs=designs,
            expression_attempts=proposals,
            validation_reports=reports,
            factors=factors,
            calculation_summaries=calculation_summaries,
            preprocessing_summaries=preprocessing_summaries,
            evaluations=evaluation_views,
            failures=request.failures,
            acceptance_policy=request.acceptance_policy,
            comparison_baseline=baseline,
            source_refs=source_refs,
        )
        return FeedbackPromptView(
            hypothesis=hypothesis,
            factor_designs=designs,
            expression_attempts=proposals,
            validation_reports=reports,
            factors=factors,
            calculation_summaries=calculation_summaries,
            preprocessing_summaries=preprocessing_summaries,
            evaluations=evaluation_views,
            failures=request.failures,
            acceptance_policy=request.acceptance_policy,
            comparison_baseline=baseline,
            source_refs=source_refs,
            content_checksum=checksum,
            schema_version=FEEDBACK_PROMPT_VIEW_SCHEMA_VERSION,
        )

    def _baseline(
        self,
        refs: tuple[ObjectRef, ...],
    ) -> ComparisonBaselineView | None:
        if not refs:
            return None
        by_type = {ref.object_type: ref for ref in refs}
        required = (
            "experiment_record",
            "factor_spec",
            "factor_preprocessing_result",
            "evaluation_result",
        )
        if any(object_type not in by_type for object_type in required):
            raise PromptViewBuildError(
                "PROMPT_VIEW_BASELINE_INCOMPLETE",
                refs[0],
                "baseline requires experiment, factor, and evaluation references",
            )
        record = self._resolve(by_type["experiment_record"], ExperimentRecord)
        factor = self._resolve(by_type["factor_spec"], FactorSpec)
        preprocessing = self._resolve(
            by_type["factor_preprocessing_result"],
            FactorPreprocessingResult,
        )
        evaluation = self._resolve(by_type["evaluation_result"], EvaluationResult)
        decision_ref = by_type.get("acceptance_decision")
        decision = (
            None
            if decision_ref is None
            else self._resolve(decision_ref, AcceptanceDecision)
        )
        if (
            record.completed_at is None
            or factor.factor_id != evaluation.factor_id
            or preprocessing.factor_id != factor.factor_id
            or evaluation.factor_preprocessing_result_ref
            != by_type["factor_preprocessing_result"]
        ):
            raise PromptViewBuildError(
                "PROMPT_VIEW_BASELINE_INCONSISTENT",
                refs[0],
                "baseline objects do not describe one completed factor",
            )
        return ComparisonBaselineView(
            experiment_id=record.experiment_id,
            completed_at=record.completed_at,
            factor_id=factor.factor_id,
            factor_name=factor.name,
            expression=factor.expression,
            metrics=tuple(_metric_evidence(metric) for metric in evaluation.metrics),
            accepted=None if decision is None else decision.accepted,
            reason_codes=() if decision is None else decision.reason_codes,
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
        try:
            value = self._resolver.get_object(ref, target_type)
        except CoreError as error:
            raise PromptViewBuildError(
                "PROMPT_VIEW_REFERENCE_UNRESOLVED",
                ref,
                str(error),
            ) from error
        if getattr(value, "schema_version", None) != ref.schema_version:
            raise PromptViewBuildError(
                "PROMPT_VIEW_REFERENCE_VERSION_MISMATCH",
                ref,
                "resolved object schema does not match its reference",
            )
        return value


def _calculation_summary(
    result: FactorCalculationResult,
) -> FactorCalculationSummaryView:
    return FactorCalculationSummaryView(
        factor_id=result.factor_id,
        data_snapshot_id=result.data_snapshot_id,
        date_range=result.date_range,
        sample_count=result.sample_count,
        missing_count=result.missing_count,
        eligible_mask_applied=result.eligible_mask_applied,
        calculation_version=result.calculation_version,
    )


def _preprocessing_summary(
    result: FactorPreprocessingResult,
) -> FactorPreprocessingSummaryView:
    return FactorPreprocessingSummaryView(
        factor_id=result.factor_id,
        data_snapshot_id=result.data_snapshot_id,
        preprocessing_protocol_id=result.preprocessing_protocol_id,
        preprocessing_protocol_version=result.preprocessing_protocol_version,
        neutralization_applied=result.neutralization_applied,
        excluded_sample_count=result.excluded_sample_count,
        effective_sample_count=result.effective_sample_count,
        warnings=result.warnings,
    )


def _metric_evidence(metric: MetricValue) -> MetricEvidenceView:
    return MetricEvidenceView(
        name=metric.name,
        value=metric.value,
        meaning=METRIC_DESCRIPTIONS.get(
            metric.name,
            f"Deterministic scalar evaluation metric: {metric.name}.",
        ),
        direction=metric.direction,
        sample_count=metric.sample_count,
        calculation_version=metric.calculation_version,
    )


def _evaluation_evidence(
    evaluation: EvaluationResult,
    decision: AcceptanceDecision | None,
    baseline: ComparisonBaselineView | None,
) -> EvaluationEvidenceView:
    baseline_metrics = (
        {} if baseline is None else {metric.name: metric for metric in baseline.metrics}
    )
    return EvaluationEvidenceView(
        factor_id=evaluation.factor_id,
        data_snapshot_id=evaluation.data_snapshot_id,
        evaluation_protocol_id=evaluation.evaluation_protocol_id,
        preprocessing_protocol_id=evaluation.preprocessing_protocol_id,
        metrics=tuple(_metric_evidence(metric) for metric in evaluation.metrics),
        total_date_count=evaluation.total_date_count,
        effective_date_count=evaluation.effective_date_count,
        effective_date_ratio=(
            None
            if evaluation.total_date_count in (None, 0)
            else evaluation.effective_date_count / evaluation.total_date_count
        ),
        effective_sample_count=evaluation.effective_sample_count,
        coverage=evaluation.coverage,
        warnings=evaluation.warnings,
        evaluator_version=evaluation.evaluator_version,
        accepted=None if decision is None else decision.accepted,
        reason_codes=() if decision is None else decision.reason_codes,
        policy_id=None if decision is None else decision.policy_id,
        policy_version=None if decision is None else decision.policy_version,
        baseline_deltas=tuple(
            _metric_delta(metric, baseline_metrics[metric.name])
            for metric in evaluation.metrics
            if metric.name in baseline_metrics
            and metric.direction == baseline_metrics[metric.name].direction
        ),
    )


def _metric_delta(
    current: MetricValue,
    baseline: MetricEvidenceView,
) -> MetricDeltaView:
    raw_delta = current.value - baseline.value
    return MetricDeltaView(
        name=current.name,
        current_value=current.value,
        baseline_value=baseline.value,
        raw_delta=raw_delta,
        direction=current.direction,
        directional_improvement=(
            -raw_delta if current.direction == "lower_is_better" else raw_delta
        ),
    )


def _decision_for(
    evaluation_ref: ObjectRef,
    evaluation: EvaluationResult,
    decisions: tuple[AcceptanceDecision, ...],
) -> AcceptanceDecision | None:
    matches = tuple(
        decision
        for decision in decisions
        if decision.factor_id == evaluation.factor_id
        and decision.evaluation_result_ref == evaluation_ref
    )
    if matches and any(decision != matches[0] for decision in matches[1:]):
        raise CoreValidationError("multiple acceptance decisions for one evaluation")
    return matches[0] if matches else None


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


def _feedback_content_checksum(**payload: object) -> str:
    return "sha256:" + stable_sha256(
        {
            **payload,
            "schema_version": FEEDBACK_PROMPT_VIEW_SCHEMA_VERSION,
        }
    )


def _context_summary(
    required_context_fields: tuple[str, ...],
    available_fields: dict[str, FieldSpec],
) -> str:
    if not required_context_fields:
        return "No additional context fields."
    return "Requires " + "; ".join(
        f"{name}: {available_fields[name].description}"
        for name in required_context_fields
    )


def _content_checksum(
    catalog: PromptCatalogView,
    source_refs: tuple[ObjectRef, ...],
    schema_version: str,
) -> str:
    return "sha256:" + stable_sha256(
        {
            "catalog": catalog,
            "schema_version": schema_version,
            "source_refs": source_refs,
        }
    )


def _hypothesis_content_checksum(
    catalog: PromptCatalogView,
    lessons: tuple[ExperimentLesson, ...],
    source_refs: tuple[ObjectRef, ...],
) -> str:
    return "sha256:" + stable_sha256(
        {
            "catalog": catalog,
            "lessons": lessons,
            "schema_version": HYPOTHESIS_PROMPT_VIEW_SCHEMA_VERSION,
            "source_refs": source_refs,
        }
    )


def _require_ref_type(ref: ObjectRef, object_type: str) -> None:
    if not isinstance(ref, ObjectRef):
        raise CoreValidationError("catalog reference must be ObjectRef")
    if ref.object_type != object_type:
        raise PromptViewBuildError(
            "PROMPT_VIEW_REFERENCE_TYPE_MISMATCH",
            ref,
            f"expected object_type {object_type}",
        )


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value
