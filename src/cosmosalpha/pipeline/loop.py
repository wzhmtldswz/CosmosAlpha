"""Serial CosmosAlpha research loop orchestration."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    AcceptancePolicySnapshot,
    EvaluationResult,
    ExperimentLesson,
    ExperimentRecord,
    ExperimentStatus,
    ExpressionConstraints,
    ExpressionRepairRequest,
    FactorCalculationResult,
    FactorDesign,
    FactorDesignBatch,
    FactorDesignConstraints,
    FactorDesignDraft,
    FactorDesignGenerationRequest,
    FactorExpressionRequest,
    FactorPreprocessingResult,
    FactorProposal,
    FactorSpec,
    FactorValidationResult,
    FeedbackRequest,
    ResearchContext,
    ResearchFeedback,
    ResearchHypothesis,
    ValidationReport,
)
from cosmosalpha.core import (
    ArtifactRef,
    ComponentRef,
    FailureInfo,
    ObjectRef,
    SerializationError,
    stable_sha256,
)
from cosmosalpha.core.logging import LOGGER, contextual, log_context, progress_stage
from cosmosalpha.data import DataSnapshot
from cosmosalpha.evaluation import (
    AcceptancePolicy,
    FactorEvaluationError,
    FactorEvaluator,
    ForwardReturnCalculator,
    ForwardReturnResult,
    GplearnNeutralizedPreprocessor,
    PreprocessingError,
    ReferenceFactorValuesProvider,
    ReturnCalculationError,
)
from cosmosalpha.factors import (
    FieldCatalog,
    OperatorCatalog,
    build_base_field_catalog,
    build_base_operator_catalog,
    validate_factor_proposal,
)
from cosmosalpha.factors.execution import FactorCalculator, FactorExecutionError
from cosmosalpha.llm.errors import (
    LLMClientError,
    LLMOutputError,
    PromptError,
    ReplayError,
)
from cosmosalpha.pipeline.agents import (
    ExpressionRepairAgent,
    FactorDesignAgent,
    FactorExpressionAgent,
    FeedbackAgent,
    HypothesisAgent,
)
from cosmosalpha.pipeline.feedback import select_comparison_baseline_refs
from cosmosalpha.pipeline.lesson import (
    EXPERIMENT_LESSON_OBJECT_TYPE,
    ExperimentLessonBuilder,
    lesson_ref_for_experiment,
)
from cosmosalpha.storage import (
    FactorLibraryEntry,
    FactorLibraryRepository,
    RecoveryIssue,
    RecoveryReport,
    RepositoryError,
    WorkflowCycle,
    WorkflowCycleStatus,
    WorkflowMetadataError,
    WorkflowMetadataRepository,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStepAttempt,
    WorkflowStepStatus,
)

HYPOTHESIS_OBJECT_TYPE = "research_hypothesis"
FACTOR_PROPOSAL_OBJECT_TYPE = "factor_proposal"
FACTOR_DESIGN_OBJECT_TYPE = "factor_design"
FACTOR_DESIGN_BATCH_OBJECT_TYPE = "factor_design_batch"
FIELD_CATALOG_OBJECT_TYPE = "field_catalog"
OPERATOR_CATALOG_OBJECT_TYPE = "operator_catalog"
VALIDATION_REPORT_OBJECT_TYPE = "validation_report"
FACTOR_VALIDATION_RESULT_OBJECT_TYPE = "factor_validation_result"
FACTOR_SPEC_OBJECT_TYPE = "factor_spec"
FACTOR_CALCULATION_RESULT_OBJECT_TYPE = "factor_calculation_result"
FORWARD_RETURN_RESULT_OBJECT_TYPE = "forward_return_result"
FACTOR_PREPROCESSING_RESULT_OBJECT_TYPE = "factor_preprocessing_result"
EVALUATION_RESULT_OBJECT_TYPE = "evaluation_result"
ACCEPTANCE_DECISION_OBJECT_TYPE = "acceptance_decision"
FACTOR_LIBRARY_ENTRY_OBJECT_TYPE = "factor_library_entry"
RESEARCH_FEEDBACK_OBJECT_TYPE = "research_feedback"
FEEDBACK_REQUEST_OBJECT_TYPE = "feedback_request"
RESEARCH_CONTEXT_OBJECT_TYPE = "research_context"
DATA_SNAPSHOT_OBJECT_TYPE = "data_snapshot"


_REPAIRABLE_VALIDATION_CODES = frozenset(
    {
        "SYNTAX_ERROR",
        "UNKNOWN_FIELD",
        "UNKNOWN_OPERATOR",
        "INVALID_ARITY",
        "INVALID_PARAMETER_TYPE",
        "UNSUPPORTED_WINDOW",
    }
)


class CosmosAlphaLoopError(Exception):
    """Raised when the serial loop cannot complete a required batch step."""


class _CandidateBoundaryFailure(Exception):
    """Carry a classified step failure across the candidate boundary."""

    def __init__(self, failure: FailureInfo) -> None:
        """Create a boundary exception from immutable failure evidence."""
        super().__init__(failure.message)
        self.failure = failure


def classify_attempt_failure(
    error: BaseException,
    *,
    stage: str | None = None,
) -> str:
    """Classify a Loop failure without allowing classification to fail."""
    try:
        if isinstance(
            error,
            (LLMClientError, LLMOutputError, PromptError, ReplayError),
        ):
            return "LLM_FAILED"
        if isinstance(error, WorkflowMetadataError):
            return "WORKFLOW_METADATA_FAILED"
        if isinstance(error, SerializationError):
            return "PERSISTENCE_FAILED"
        if isinstance(error, (RepositoryError, OSError)):
            return "STORAGE_FAILED"
        if isinstance(error, (FactorExecutionError, ReturnCalculationError)):
            return "CALCULATION_FAILED"
        if isinstance(error, PreprocessingError):
            return "PREPROCESSING_FAILED"
        if isinstance(error, FactorEvaluationError):
            return "EVALUATION_FAILED"
        step = stage.rsplit(":", 1)[-1] if isinstance(stage, str) else ""
        if step == "calculation":
            return "CALCULATION_FAILED"
        if step == "preprocessing":
            return "PREPROCESSING_FAILED"
        if step == "evaluation":
            return "EVALUATION_FAILED"
        if step == "acceptance":
            return "ACCEPTANCE_FAILED"
        return "UNEXPECTED_ATTEMPT_FAILURE"
    except Exception:  # noqa: BLE001 - the classifier must be total
        return "UNEXPECTED_ATTEMPT_FAILURE"


def _exception_message(error: BaseException) -> str:
    """Return a non-empty diagnostic even for an unusual exception object."""
    try:
        message = str(error)
        return message or error.__class__.__name__
    except Exception:  # noqa: BLE001 - failure reporting must not mask the error
        return type(error).__name__


def _failure_info(
    *,
    stage: str,
    error: BaseException,
    input_refs: tuple[ObjectRef, ...],
    artifact_refs: tuple[ArtifactRef, ...],
) -> FailureInfo:
    """Build stable failure evidence for one failed Loop step."""
    return FailureInfo(
        stage=stage,
        error_code=classify_attempt_failure(error, stage=stage),
        message=_exception_message(error),
        retryable=False,
        input_refs=input_refs,
        artifact_refs=artifact_refs,
    )


class ArtifactRepository(Protocol):
    """Artifact repository methods required by default loop components."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist bytes and return an artifact reference."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read artifact bytes."""


class ExperimentRepository(Protocol):
    """Object repository methods required by the loop."""

    def put_object(self, object_type: str, object_id: str, value: object) -> ObjectRef:
        """Persist a strict JSON object."""

    def put_experiment(self, record: ExperimentRecord) -> ObjectRef:
        """Persist an experiment record."""

    def get_experiment(self, ref: ObjectRef) -> ExperimentRecord:
        """Resolve an experiment record."""

    def latest_completed(self, run_id: str | None = None) -> ExperimentRecord | None:
        """Return the latest completed record."""

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        """Return persisted experiment records."""

    def recover(self) -> RecoveryReport:
        """Scan persisted objects and report recovery issues."""

    def get_object[ObjectT](
        self, ref: ObjectRef, target_type: type[ObjectT]
    ) -> ObjectT:
        """Resolve a persisted protocol object."""


class FactorCalculatorLike(Protocol):
    """Factor calculator interface used by the loop."""

    def calculate(
        self,
        factor_spec: FactorSpec,
        data_snapshot: DataSnapshot,
        *,
        validation_report_ref: ObjectRef,
    ) -> FactorCalculationResult:
        """Calculate one factor."""


class ForwardReturnCalculatorLike(Protocol):
    """Forward return calculator interface used by the loop."""

    def calculate(self, data_snapshot: DataSnapshot) -> ForwardReturnResult:
        """Calculate forward returns."""


class PreprocessorLike(Protocol):
    """Factor preprocessing interface used by the loop."""

    def preprocess(
        self,
        factor_result: FactorCalculationResult,
        forward_return_result: ForwardReturnResult,
        data_snapshot: DataSnapshot,
    ) -> FactorPreprocessingResult:
        """Preprocess one factor result."""


class EvaluatorLike(Protocol):
    """Factor evaluator interface used by the loop."""

    def evaluate(
        self,
        preprocessing_result: FactorPreprocessingResult,
        preprocessing_result_ref: ObjectRef,
        data_snapshot: DataSnapshot,
    ) -> EvaluationResult:
        """Evaluate one preprocessed factor."""


class AcceptancePolicyLike(Protocol):
    """Acceptance policy interface used by the loop."""

    def decide(
        self,
        evaluation_result: EvaluationResult,
        evaluation_result_ref: ObjectRef,
    ) -> AcceptanceDecision:
        """Create an acceptance decision."""

    def snapshot(self) -> AcceptancePolicySnapshot:
        """Return replayable policy evidence."""


@dataclass(frozen=True, slots=True)
class CandidateCalculationDecision:
    """Decision returned after final validation and before factor calculation."""

    proceed: bool
    reason_code: str | None = None

    def __post_init__(self) -> None:
        """Require a stable reason whenever calculation is skipped."""
        if type(self.proceed) is not bool:
            raise CosmosAlphaLoopError("proceed must be a boolean")
        if self.reason_code is not None and (
            type(self.reason_code) is not str or not self.reason_code
        ):
            raise CosmosAlphaLoopError("reason_code must be a non-empty string")
        if not self.proceed and self.reason_code is None:
            raise CosmosAlphaLoopError(
                "reason_code is required when calculation is skipped"
            )


class CandidateCalculationGate(Protocol):
    """Optional staged gate invoked before calculation for a valid candidate."""

    def decide(
        self,
        *,
        proposal: FactorProposal,
        factor_spec: FactorSpec,
        factor_ref: ObjectRef,
        validation_report_ref: ObjectRef,
    ) -> CandidateCalculationDecision:
        """Allow calculation or return a structured pre-calculation outcome."""


@dataclass(frozen=True, slots=True)
class CandidateCalculationInput:
    """One candidate exposed after final validation and before calculation."""

    candidate_index: int
    proposal: FactorProposal
    factor_spec: FactorSpec
    factor_ref: ObjectRef
    validation_report_ref: ObjectRef

    def __post_init__(self) -> None:
        """Validate the immutable staged candidate boundary."""
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise CosmosAlphaLoopError("candidate_index must be non-negative")
        if not isinstance(self.proposal, FactorProposal):
            raise CosmosAlphaLoopError("proposal must be a factor proposal")
        if not isinstance(self.factor_spec, FactorSpec):
            raise CosmosAlphaLoopError("factor_spec must be FactorSpec")
        if not isinstance(self.factor_ref, ObjectRef):
            raise CosmosAlphaLoopError("factor_ref must be ObjectRef")
        if not isinstance(self.validation_report_ref, ObjectRef):
            raise CosmosAlphaLoopError("validation_report_ref must be ObjectRef")


class CandidateCalculationBatchGate(Protocol):
    """Decide the calculation fate of a complete validated candidate batch."""

    def decide(
        self,
        candidates: tuple[CandidateCalculationInput, ...],
    ) -> tuple[CandidateCalculationDecision, ...]:
        """Return one deterministic decision for every staged candidate."""


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """Persisted references and status for one independent factor candidate."""

    proposal_ref: ObjectRef
    validation_report_ref: ObjectRef
    status: ExperimentStatus
    factor_ref: ObjectRef | None = None
    calculation_result_ref: ObjectRef | None = None
    preprocessing_result_ref: ObjectRef | None = None
    evaluation_result_ref: ObjectRef | None = None
    acceptance_decision_ref: ObjectRef | None = None
    failure: FailureInfo | None = None
    proposal_attempt_refs: tuple[ObjectRef, ...] = ()
    validation_report_attempt_refs: tuple[ObjectRef, ...] = ()
    pre_calculation_gate_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CandidateValidation:
    """Persisted outputs needed to continue or recover one candidate."""

    proposal_ref: ObjectRef
    validation_report_ref: ObjectRef
    validation_result: FactorValidationResult
    factor_ref: ObjectRef | None


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    """Validated candidate plus repair audit refs ready for a calculation gate."""

    candidate_index: int
    proposal: FactorProposal
    validation: _CandidateValidation
    proposal_attempt_refs: tuple[ObjectRef, ...]
    validation_report_attempt_refs: tuple[ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class CosmosAlphaLoopResult:
    """Result returned by a serial CosmosAlpha loop run."""

    run_id: str
    cycle_id: str
    experiment_ref: ObjectRef | None
    experiment_record: ExperimentRecord | None
    hypothesis_ref: ObjectRef | None
    factor_design_batch_ref: ObjectRef | None
    feedback_ref: ObjectRef | None
    candidate_results: tuple[CandidateRunResult, ...]
    recovery_issues: tuple[RecoveryIssue, ...]
    resumed: bool
    factor_library_entry_refs: tuple[ObjectRef, ...] = ()
    expression_failures: tuple[FailureInfo, ...] = ()
    lesson_ref: ObjectRef | None = None


class CosmosAlphaLoop:
    """Complete serial CosmosAlpha chain with candidate-level failure isolation."""

    def __init__(
        self,
        *,
        hypothesis_agent: HypothesisAgent,
        factor_design_agent: FactorDesignAgent,
        factor_expression_agent: FactorExpressionAgent,
        expression_repair_agent: ExpressionRepairAgent | None = None,
        feedback_agent: FeedbackAgent,
        artifact_repository: ArtifactRepository,
        experiment_repository: ExperimentRepository,
        workflow_metadata_repository: WorkflowMetadataRepository,
        field_catalog: FieldCatalog | None = None,
        operator_catalog: OperatorCatalog | None = None,
        factor_calculator: FactorCalculatorLike | None = None,
        forward_return_calculator: ForwardReturnCalculatorLike | None = None,
        preprocessor: PreprocessorLike | None = None,
        evaluator: EvaluatorLike | None = None,
        acceptance_policy: AcceptancePolicyLike | None = None,
        factor_library: FactorLibraryRepository | None = None,
        generation_constraints: FactorDesignConstraints | None = None,
        max_validation_repair_attempts: int = 0,
        clock: Clock | None = None,
        publish_min_rank_ic: float | None = None,
        publish_max_corr: float | None = None,
        expression_max_depth: int = 8,
    ) -> None:
        """Create a serial loop from deterministic services and agent adapters."""
        self._hypothesis_agent = hypothesis_agent
        self._factor_design_agent = factor_design_agent
        self._factor_expression_agent = factor_expression_agent
        self._expression_repair_agent = expression_repair_agent
        if factor_design_agent is None or factor_expression_agent is None:
            raise CosmosAlphaLoopError(
                "factor design and expression agents are required"
            )
        if generation_constraints is not None and not isinstance(
            generation_constraints, FactorDesignConstraints
        ):
            raise CosmosAlphaLoopError(
                "generation_constraints must be FactorDesignConstraints"
            )
        if (
            type(max_validation_repair_attempts) is not int
            or max_validation_repair_attempts < 0
            or max_validation_repair_attempts > 2
        ):
            raise CosmosAlphaLoopError(
                "max_validation_repair_attempts must be between 0 and 2"
            )
        if max_validation_repair_attempts and expression_repair_agent is None:
            raise CosmosAlphaLoopError(
                "ExpressionRepairAgent is required when repair is enabled"
            )
        if type(expression_max_depth) is not int or expression_max_depth < 1:
            raise CosmosAlphaLoopError(
                "expression_max_depth must be a positive integer"
            )
        self._expression_max_depth = expression_max_depth
        self._feedback_agent = feedback_agent
        self._artifacts = artifact_repository
        self._objects = experiment_repository
        self._lesson_builder = ExperimentLessonBuilder(experiment_repository)
        self._metadata = workflow_metadata_repository
        self._field_catalog = field_catalog or build_base_field_catalog()
        self._operator_catalog = operator_catalog or build_base_operator_catalog()
        self._factor_calculator = factor_calculator or FactorCalculator(
            artifact_repository
        )
        self._forward_return_calculator = (
            forward_return_calculator or ForwardReturnCalculator(artifact_repository)
        )
        self._preprocessor = preprocessor or GplearnNeutralizedPreprocessor(
            artifact_repository
        )
        self._factor_library = factor_library
        self._generation_constraints = (
            generation_constraints or FactorDesignConstraints(max_designs=2)
        )
        self._max_validation_repair_attempts = max_validation_repair_attempts
        self._evaluator = evaluator or FactorEvaluator(
            artifact_repository,
            reference_factor_values_provider=factor_library,
        )
        self._acceptance_policy = acceptance_policy or AcceptancePolicy()
        self._clock = clock or _utc_now
        # MCTS 模式：acceptance 阈值被设为 -1.0（保留弱节点给 UCT 探索），
        # 但发布到因子库必须应用真实质量门槛，避免弱因子污染交付库。
        if publish_min_rank_ic is not None and (
            type(publish_min_rank_ic) not in (int, float)
            or not math.isfinite(float(publish_min_rank_ic))
        ):
            raise CosmosAlphaLoopError(
                "publish_min_rank_ic must be a finite number or None"
            )
        self._publish_min_rank_ic = (
            float(publish_min_rank_ic) if publish_min_rank_ic is not None else None
        )
        # 发布时值相关门槛：evaluator 相对已入库因子计算逐日横截面
        # correlation 的有符号时间均值，再取绝对值。达到阈值即跳过入库
        # （树/FZoo 节点不受影响）。修复 v15m 中 F02/F03 (|r|≈0.92) 等值相关
        # 因子同时入库的问题 —— 文本/结构去重抓不住“同值不同构”。默认 None 关闭。
        if publish_max_corr is not None and (
            type(publish_max_corr) not in (int, float)
            or not 0.0 <= float(publish_max_corr) <= 1.0
        ):
            raise CosmosAlphaLoopError("publish_max_corr must be in [0, 1] or None")
        self._publish_max_corr = (
            float(publish_max_corr) if publish_max_corr is not None else None
        )

    def set_reference_factor_values_provider(
        self,
        provider: ReferenceFactorValuesProvider,
    ) -> None:
        """Set the immutable peer-factor view for the next Loop execution."""
        setter = getattr(self._evaluator, "set_reference_factor_values_provider", None)
        if not callable(setter):
            raise CosmosAlphaLoopError(
                "configured evaluator cannot accept a reference-factor provider"
            )
        setter(provider)

    @contextual
    def run(
        self,
        *,
        run_id: str,
        cycle_id: str,
        context: ResearchContext,
        data_snapshot: DataSnapshot,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        parent_ids: tuple[str, ...] = (),
        existing_factor_refs: tuple[ObjectRef, ...] = (),
        graph_parent_ref: ObjectRef | None = None,
        history_lesson_refs: tuple[ObjectRef, ...] = (),
        pre_calculation_gate: CandidateCalculationGate | None = None,
        pre_calculation_batch_gate: CandidateCalculationBatchGate | None = None,
        run_context: ResearchContext | None = None,
        complete_run: bool = True,
        resume: bool = True,
        fixed_factor_design_drafts: tuple[FactorDesignDraft, ...] | None = None,
        skip_hypothesis: bool = False,
        skip_feedback: bool = False,
    ) -> CosmosAlphaLoopResult:
        """Run one complete serial cycle or resume its verified atomic steps.

        Args:
            run_id: Stable workflow run identifier.
            cycle_id: Stable identifier of this serial research cycle.
            context: Cycle-specific, bounded research context passed to Agents.
            data_snapshot: Immutable market-data input for deterministic services.
            config_snapshot_ref: Redacted configuration snapshot artifact reference.
            code_version: Version identifier of the executing CosmosAlpha code.
            parent_ids: Immediate parent experiment identifiers for lineage.
            existing_factor_refs: Historical factors exposed to the factor Agent for
                duplicate-aware proposal generation.
            graph_parent_ref: Optional Graph parent whose sole factor reference is
                exposed to Graph FactorDesign and FactorExpression requests.
            history_lesson_refs: Structured parent/history Lessons exposed to the
                Hypothesis Agent.
            pre_calculation_gate: Optional staged gate invoked after final
                validation/repair and before factor calculation.
            pre_calculation_batch_gate: Optional gate invoked once after all
                candidates have completed final validation/repair and before any
                factor calculation.
            run_context: Stable root context identity for multi-cycle run metadata.
            complete_run: Whether this cycle is the final requested cycle for the run.
            resume: Whether verified successful attempts may be reused.
            fixed_factor_design_drafts: Fixed designs whose formulation is supplied DSL. When set,
                the normal LLM factor-design/expression generation stages are
                bypassed along with expression repair; validation, calculation, acceptance,
                feedback, and audit persistence remain unchanged.
            skip_hypothesis: Skip the Hypothesis Agent and use a synthetic hypothesis.
                Mutually exclusive with ``fixed_factor_design_drafts``.
            skip_feedback: Skip the Feedback Agent for this cycle.
        """
        if not isinstance(context, ResearchContext):
            raise CosmosAlphaLoopError("context must be ResearchContext")
        if run_context is not None and not isinstance(run_context, ResearchContext):
            raise CosmosAlphaLoopError("run_context must be ResearchContext or None")
        if not isinstance(existing_factor_refs, tuple) or not all(
            isinstance(ref, ObjectRef) for ref in existing_factor_refs
        ):
            raise CosmosAlphaLoopError(
                "existing_factor_refs must be a tuple of ObjectRef"
            )
        if graph_parent_ref is not None and not isinstance(graph_parent_ref, ObjectRef):
            raise CosmosAlphaLoopError("graph_parent_ref must be ObjectRef or None")
        if graph_parent_ref is not None:
            if existing_factor_refs and existing_factor_refs != (graph_parent_ref,):
                raise CosmosAlphaLoopError(
                    "Graph requests may expose only the current parent factor"
                )
            existing_factor_refs = (graph_parent_ref,)
        if pre_calculation_gate is not None and pre_calculation_batch_gate is not None:
            raise CosmosAlphaLoopError(
                "configure either pre_calculation_gate or " "pre_calculation_batch_gate"
            )
        if not isinstance(history_lesson_refs, tuple) or not all(
            isinstance(ref, ObjectRef) for ref in history_lesson_refs
        ):
            raise CosmosAlphaLoopError(
                "history_lesson_refs must be a tuple of ObjectRef"
            )
        if fixed_factor_design_drafts is not None and (
            not isinstance(fixed_factor_design_drafts, tuple)
            or not all(
                isinstance(proposal, FactorDesignDraft)
                for proposal in fixed_factor_design_drafts
            )
            or not fixed_factor_design_drafts
        ):
            raise CosmosAlphaLoopError(
                "fixed_factor_design_drafts must be a non-empty tuple of FactorDesignDraft"
            )
        if fixed_factor_design_drafts is not None and skip_hypothesis:
            raise CosmosAlphaLoopError(
                "fixed_factor_design_drafts and skip_hypothesis are mutually exclusive"
            )
        if type(complete_run) is not bool:
            raise CosmosAlphaLoopError("complete_run must be a boolean")
        self._reused_step = False
        self._recovered_step = False
        self._resume_enabled = resume
        recovery_report = self._objects.recover()
        if resume:
            self._recovered_step = bool(
                self._metadata.recover_interrupted(
                    completed_at=self._clock(),
                    run_id=run_id,
                    cycle_id=cycle_id,
                )
            )
        run_context_value = context if run_context is None else run_context
        run_context_ref = self._objects.put_object(
            RESEARCH_CONTEXT_OBJECT_TYPE,
            "context-" + stable_sha256(run_context_value)[:24],
            run_context_value,
        )
        context_ref = self._objects.put_object(
            RESEARCH_CONTEXT_OBJECT_TYPE,
            "context-" + stable_sha256(context)[:24],
            context,
        )
        data_snapshot_ref = self._objects.put_object(
            DATA_SNAPSHOT_OBJECT_TYPE,
            data_snapshot.snapshot_id,
            data_snapshot,
        )
        self._metadata.create_run(
            WorkflowRun(
                run_id=run_id,
                context_ref=run_context_ref,
                data_snapshot_ref=data_snapshot_ref,
                config_snapshot_ref=config_snapshot_ref,
                code_version=code_version,
                status=WorkflowRunStatus.ACTIVE,
                created_at=self._clock(),
                completed_at=None,
            )
        )
        self._metadata.create_cycle(
            WorkflowCycle(
                run_id=run_id,
                cycle_id=cycle_id,
                parent_ids=parent_ids,
                status=WorkflowCycleStatus.ACTIVE,
                created_at=self._clock(),
                completed_at=None,
            )
        )
        completed_attempt = self._metadata.latest_successful_attempt(
            run_id, cycle_id, "experiment_publish"
        )
        if resume and completed_attempt is not None and self._factor_library is None:
            LOGGER.debug("Cycle restored from completed experiment")
            experiment_ref = _only_object_ref(completed_attempt)
            experiment = self._objects.get_experiment(experiment_ref)
            if complete_run:
                self._metadata.complete_run(run_id, completed_at=self._clock())
            return CosmosAlphaLoopResult(
                run_id=run_id,
                cycle_id=cycle_id,
                experiment_ref=experiment_ref,
                experiment_record=experiment,
                hypothesis_ref=experiment.hypothesis_ref,
                factor_design_batch_ref=None,
                feedback_ref=experiment.feedback_ref,
                candidate_results=(),
                recovery_issues=recovery_report.issues,
                resumed=True,
            )

        field_catalog_ref = self._objects.put_object(
            FIELD_CATALOG_OBJECT_TYPE,
            "fields-" + stable_sha256(self._field_catalog)[:24],
            self._field_catalog,
        )
        operator_catalog_ref = self._objects.put_object(
            OPERATOR_CATALOG_OBJECT_TYPE,
            context.operator_catalog_ref.object_id,
            self._operator_catalog,
        )
        if operator_catalog_ref != context.operator_catalog_ref:
            raise CosmosAlphaLoopError(
                "context operator_catalog_ref does not match runtime OperatorCatalog"
            )

        def hypothesis_operation() -> (
            tuple[ResearchHypothesis, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]
        ):
            if fixed_factor_design_drafts is not None:
                return self._persist_seed_hypothesis(
                    context,
                    field_catalog_ref,
                    operator_catalog_ref,
                    fixed_factor_design_drafts,
                    run_id=run_id,
                    cycle_id=cycle_id,
                )
            return self._persist_hypothesis(
                context,
                field_catalog_ref,
                history_lesson_refs,
            )

        if skip_hypothesis:
            # 轻量 refine：跳过 HypothesisAgent，用合成 hypothesis 携带当前
            # 展开上下文（Parent factor / Focus dimension / 分数 / HARD
            # CONSTRAINTS），保证 FactorDesignAgent 仍能看到父因子信息
            # （省 1 次 LLM）。给定种子因子时种子优先，该分支不会被触发。
            hypothesis = self._synthetic_refine_hypothesis(
                context, field_catalog_ref, history_lesson_refs
            )
            # 合成假设也需持久化：feedback/lesson 步骤通过 hypothesis_ref
            # 从仓库读取对象，不持久化会导致 object not found。
            self._objects.put_object(
                HYPOTHESIS_OBJECT_TYPE, hypothesis.hypothesis_id, hypothesis
            )
        else:
            hypothesis = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id="hypothesis",
                input_object_refs=(
                    context_ref,
                    field_catalog_ref,
                    operator_catalog_ref,
                    *history_lesson_refs,
                ),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("hypothesis-agent", "1.0"),
                operation=hypothesis_operation,
                load=lambda attempt: self._objects.get_object(
                    _only_object_ref(attempt), ResearchHypothesis
                ),
            )
        hypothesis_ref = ObjectRef(
            HYPOTHESIS_OBJECT_TYPE,
            hypothesis.hypothesis_id,
            hypothesis.schema_version,
        )
        factor_design_batch_ref: ObjectRef | None = None
        factor_design_refs: tuple[ObjectRef, ...] = ()
        expression_failures: tuple[FailureInfo, ...] = ()
        if fixed_factor_design_drafts is not None:
            designs = tuple(
                FactorDesign.from_draft(
                    draft,
                    hypothesis_ref=hypothesis_ref,
                    frequency=context.frequency,
                    field_catalog_ref=field_catalog_ref,
                    operator_catalog_ref=operator_catalog_ref,
                    index=index,
                )
                for index, draft in enumerate(fixed_factor_design_drafts)
            )
            factor_design_refs = tuple(
                self._objects.put_object(
                    FACTOR_DESIGN_OBJECT_TYPE, design.design_id, design
                )
                for design in designs
            )
            proposals = [
                FactorProposal.from_design(design, design.formulation)
                for design in designs
            ]
            for proposal in proposals:
                self._objects.put_object(
                    FACTOR_PROPOSAL_OBJECT_TYPE, proposal.proposal_id, proposal
                )
        else:
            design_request = self._design_generation_request(
                hypothesis=hypothesis,
                hypothesis_ref=hypothesis_ref,
                field_catalog_ref=field_catalog_ref,
                operator_catalog_ref=operator_catalog_ref,
                context=context,
                existing_factor_refs=existing_factor_refs,
            )
            design_batch = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id="factor_design_generation",
                input_object_refs=(
                    context_ref,
                    hypothesis_ref,
                    field_catalog_ref,
                    operator_catalog_ref,
                ),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("factor-design-agent", "1.0"),
                operation=lambda: self._persist_design_batch(design_request),
                load=self._load_design_batch,
            )
            factor_design_batch_ref = ObjectRef(
                FACTOR_DESIGN_BATCH_OBJECT_TYPE,
                _design_batch_id(design_batch),
                design_batch.schema_version,
            )
            factor_design_refs = tuple(
                ObjectRef(
                    FACTOR_DESIGN_OBJECT_TYPE,
                    design.design_id,
                    design.schema_version,
                )
                for design in design_batch.designs
            )
            proposals: list[FactorProposal] = []
            failures: list[FailureInfo] = []
            for design in design_batch.designs:
                proposal, failure = self._generate_expression(
                    design,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    factor_design_batch_ref=factor_design_batch_ref,
                    config_snapshot_ref=config_snapshot_ref,
                    existing_factor_refs=existing_factor_refs,
                )
                if proposal is not None:
                    proposals.append(proposal)
                if failure is not None:
                    failures.append(failure)
            expression_failures = tuple(failures)
        forward_return_result: ForwardReturnResult | None = None

        def get_forward_return_result() -> ForwardReturnResult:
            nonlocal forward_return_result
            if forward_return_result is None:
                forward_return_result = self._execute_step(
                    run_id=run_id,
                    cycle_id=cycle_id,
                    step_id="forward_return",
                    input_object_refs=(data_snapshot_ref,),
                    input_artifact_refs=(config_snapshot_ref,),
                    component=ComponentRef("forward-return-calculator", "1.0"),
                    operation=lambda: self._persist_forward_return(data_snapshot),
                    load=lambda attempt: self._objects.get_object(
                        _only_object_ref(attempt), ForwardReturnResult
                    ),
                )
            return forward_return_result

        if pre_calculation_batch_gate is None:
            candidate_results = tuple(
                self._run_candidate(
                    proposal,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    proposal_source_ref=proposal.factor_design_ref,
                    data_snapshot_ref=data_snapshot_ref,
                    config_snapshot_ref=config_snapshot_ref,
                    data_snapshot=data_snapshot,
                    get_forward_return_result=get_forward_return_result,
                    pre_calculation_gate=pre_calculation_gate,
                    allow_repair=fixed_factor_design_drafts is None,
                )
                for proposal in proposals
            )
        else:
            candidate_results = self._run_candidate_batch(
                proposals,
                run_id=run_id,
                cycle_id=cycle_id,
                data_snapshot_ref=data_snapshot_ref,
                config_snapshot_ref=config_snapshot_ref,
                data_snapshot=data_snapshot,
                get_forward_return_result=get_forward_return_result,
                batch_gate=pre_calculation_batch_gate,
                allow_repair=fixed_factor_design_drafts is None,
            )
        for candidate in candidate_results:
            if candidate.failure is not None or candidate.pre_calculation_gate_reason:
                LOGGER.log(
                    logging.WARNING if candidate.failure is not None else logging.INFO,
                    "Candidate %s id=%s reason=%s",
                    candidate.status.value,
                    (candidate.factor_ref or candidate.proposal_ref).object_id,
                    (
                        candidate.failure.error_code
                        if candidate.failure is not None
                        else candidate.pre_calculation_gate_reason
                    ),
                )
        for failure in expression_failures:
            LOGGER.warning(
                "Expression candidate failed stage=%s reason=%s",
                failure.stage,
                failure.error_code,
            )
        feedback_ref: ObjectRef | None = None
        lesson_ref: ObjectRef | None = None
        if not skip_feedback:
            # 轻量 refine：跳过 FeedbackAgent + Lesson（MCTS 用 IC 回传评价，省 1 次 LLM）
            acceptance_policy_snapshot = self._acceptance_policy.snapshot()
            comparison_baseline_refs = select_comparison_baseline_refs(
                self._objects,
                parent_ids=parent_ids,
                current_evaluation_refs=_existing_refs(
                    result.evaluation_result_ref for result in candidate_results
                ),
                acceptance_metric=acceptance_policy_snapshot.acceptance_metric,
            )
            feedback_request = _feedback_request(
                hypothesis_ref,
                factor_design_refs,
                candidate_results,
                comparison_baseline_refs=comparison_baseline_refs,
                expression_failures=expression_failures,
                acceptance_policy=acceptance_policy_snapshot,
            )
            feedback_request_ref = self._objects.put_object(
                FEEDBACK_REQUEST_OBJECT_TYPE,
                "feedback-request-" + stable_sha256(feedback_request)[:24],
                feedback_request,
            )
            feedback = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id="feedback",
                input_object_refs=(
                    context_ref,
                    feedback_request_ref,
                    *_feedback_request_refs(feedback_request),
                ),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("feedback-agent", "1.0"),
                operation=lambda: self._persist_feedback(feedback_request, cycle_id),
                load=lambda attempt: self._objects.get_object(
                    _only_object_ref(attempt), ResearchFeedback
                ),
            )
            feedback_ref = ObjectRef(
                RESEARCH_FEEDBACK_OBJECT_TYPE,
                feedback.feedback_id,
                feedback.schema_version,
            )
            experiment_id = _experiment_id(run_id, cycle_id, hypothesis_ref)
            lesson = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id="experiment_lesson",
                input_object_refs=(
                    feedback_request_ref,
                    feedback_ref,
                    *_feedback_request_refs(feedback_request),
                ),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("experiment-lesson-builder", "1.0"),
                operation=lambda: self._persist_lesson(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    request=feedback_request,
                    request_ref=feedback_request_ref,
                    feedback=feedback,
                    feedback_ref=feedback_ref,
                ),
                load=lambda attempt: self._objects.get_object(
                    _only_object_ref(attempt),
                    ExperimentLesson,
                ),
            )
            lesson_ref = lesson_ref_for_experiment(lesson.experiment_id)
        experiment_record = self._execute_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id="experiment_publish",
            input_object_refs=(
                *_candidate_result_refs(candidate_results),
                hypothesis_ref,
                *(ref for ref in (feedback_ref, lesson_ref) if ref is not None),
            ),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("experiment-publisher", "1.0"),
            operation=lambda: self._persist_experiment(
                run_id=run_id,
                cycle_id=cycle_id,
                parent_ids=parent_ids,
                hypothesis_ref=hypothesis_ref,
                candidate_results=candidate_results,
                feedback_ref=feedback_ref,
                config_snapshot_ref=config_snapshot_ref,
                code_version=code_version,
                data_snapshot=data_snapshot,
            ),
            load=lambda attempt: self._objects.get_experiment(
                _only_object_ref(attempt)
            ),
        )
        experiment_ref = ObjectRef(
            "experiment_record",
            experiment_record.experiment_id,
            experiment_record.schema_version,
        )
        factor_library_entry_refs = self._publish_accepted_to_library(
            run_id=run_id,
            cycle_id=cycle_id,
            experiment_record=experiment_record,
            experiment_ref=experiment_ref,
            candidate_results=candidate_results,
            config_snapshot_ref=config_snapshot_ref,
            min_rank_ic=self._publish_min_rank_ic,
        )
        self._metadata.complete_cycle(run_id, cycle_id, completed_at=self._clock())
        if complete_run:
            self._metadata.complete_run(run_id, completed_at=self._clock())
        LOGGER.info(
            "Cycle completed accepted=%d rejected=%d failed=%d resumed=%s",
            sum(c.status is ExperimentStatus.ACCEPTED for c in candidate_results),
            sum(c.status is ExperimentStatus.REJECTED for c in candidate_results),
            sum(c.status is ExperimentStatus.FAILED for c in candidate_results)
            + len(expression_failures),
            self._reused_step or self._recovered_step,
        )
        return CosmosAlphaLoopResult(
            run_id=run_id,
            cycle_id=cycle_id,
            experiment_ref=experiment_ref,
            experiment_record=experiment_record,
            hypothesis_ref=hypothesis_ref,
            factor_design_batch_ref=factor_design_batch_ref,
            feedback_ref=feedback_ref,
            candidate_results=candidate_results,
            recovery_issues=recovery_report.issues,
            resumed=self._reused_step or self._recovered_step,
            factor_library_entry_refs=factor_library_entry_refs,
            expression_failures=expression_failures,
            lesson_ref=lesson_ref,
        )

    def _persist_hypothesis(
        self,
        context: ResearchContext,
        field_catalog_ref: ObjectRef,
        history_lesson_refs: tuple[ObjectRef, ...],
    ) -> tuple[ResearchHypothesis, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        hypothesis = self._hypothesis_agent.propose(
            context,
            field_catalog_ref=field_catalog_ref,
            history_lesson_refs=history_lesson_refs,
        )
        ref = self._objects.put_object(
            HYPOTHESIS_OBJECT_TYPE, hypothesis.hypothesis_id, hypothesis
        )
        return hypothesis, (ref,), ()

    def _persist_seed_hypothesis(
        self,
        context: ResearchContext,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        drafts: tuple[FactorDesignDraft, ...],
        *,
        run_id: str,
        cycle_id: str,
    ) -> tuple[ResearchHypothesis, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        """Persist deterministic provenance for supplied fixed designs."""
        hypothesis_id = (
            "seed-hypothesis-"
            + stable_sha256({"run_id": run_id, "cycle_id": cycle_id, "drafts": drafts})[
                :24
            ]
        )
        hypothesis = ResearchHypothesis.from_fixed_designs(
            drafts,
            hypothesis_id=hypothesis_id,
            frequency=context.frequency,
            source="the configured Graph seed table",
            source_refs=(
                field_catalog_ref,
                operator_catalog_ref,
            ),
        )
        ref = self._objects.put_object(
            HYPOTHESIS_OBJECT_TYPE, hypothesis_id, hypothesis
        )
        return hypothesis, (ref,), ()

    def _synthetic_refine_hypothesis(
        self,
        context: ResearchContext,
        field_catalog_ref: ObjectRef,
        history_lesson_refs: tuple[ObjectRef, ...],
    ) -> ResearchHypothesis:
        """轻量 refine：从当前 context 构造无 LLM 的合成 hypothesis.

        把展开约束（Parent factor / Focus dimension / 分数 / HARD
        CONSTRAINTS）嵌入 statement，使下游 FactorDesignAgent 仍能看到
        父因子信息（省 1 次 HypothesisAgent LLM 调用）。
        """
        sections: list[str] = []
        if context.constraints:
            sections.append("MCTS constraints:\n" + "\n".join(context.constraints))
        if context.history_summaries:
            sections.append("MCTS history:\n" + "\n".join(context.history_summaries))
        context_text = "\n".join(sections) or "MCTS refinement"
        hid = stable_sha256({"context": context_text})
        return ResearchHypothesis(
            hypothesis_id=f"mcts-h-{hid[:16]}",
            statement=f"MCTS-guided factor refinement: {context_text}",
            rationale=(
                "Auto-generated from MCTS expand context; HypothesisAgent "
                "bypassed for cost."
            ),
            mechanism="MCTS UCT-guided refinement of the parent factor.",
            testable_prediction=(
                "Factor will improve the target dimension relative to parent."
            ),
            expected_direction=(
                "positive" if "effectiveness" in context_text.lower() else "any"
            ),
            required_fields=tuple(context.available_fields),
            prediction_horizon=context.frequency,
            assumptions=(),
            source_refs=(
                (context.operator_catalog_ref,)
                + ((field_catalog_ref,) if field_catalog_ref else ())
                + history_lesson_refs
            ),
            schema_version="research-hypothesis/1.0",
        )

    def _persist_design_batch(
        self, request: FactorDesignGenerationRequest
    ) -> tuple[FactorDesignBatch, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        if self._factor_design_agent is None:
            raise CosmosAlphaLoopError("FactorDesignAgent is not configured")
        batch = self._factor_design_agent.generate(request)
        design_refs = tuple(
            self._objects.put_object(
                FACTOR_DESIGN_OBJECT_TYPE,
                design.design_id,
                design,
            )
            for design in batch.designs
        )
        batch_ref = self._objects.put_object(
            FACTOR_DESIGN_BATCH_OBJECT_TYPE,
            _design_batch_id(batch),
            batch,
        )
        return batch, (*design_refs, batch_ref), ()

    def _load_design_batch(self, attempt: WorkflowStepAttempt) -> FactorDesignBatch:
        ref = _object_ref_by_type(
            attempt,
            FACTOR_DESIGN_BATCH_OBJECT_TYPE,
        )
        return self._objects.get_object(ref, FactorDesignBatch)

    def _persist_expression(
        self, request: FactorExpressionRequest
    ) -> tuple[FactorProposal, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        if self._factor_expression_agent is None:
            raise CosmosAlphaLoopError("FactorExpressionAgent is not configured")
        proposal = self._factor_expression_agent.generate(request)
        ref = self._objects.put_object(
            FACTOR_PROPOSAL_OBJECT_TYPE,
            proposal.proposal_id,
            proposal,
        )
        return proposal, (ref,), ()

    def _persist_repair(
        self, request: ExpressionRepairRequest
    ) -> tuple[FactorProposal, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        if self._expression_repair_agent is None:
            raise CosmosAlphaLoopError("ExpressionRepairAgent is not configured")
        proposal = self._expression_repair_agent.repair(request)
        ref = self._objects.put_object(
            FACTOR_PROPOSAL_OBJECT_TYPE,
            proposal.proposal_id,
            proposal,
        )
        return proposal, (ref,), ()

    def _persist_forward_return(
        self, data_snapshot: DataSnapshot
    ) -> tuple[ForwardReturnResult, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        result = self._forward_return_calculator.calculate(data_snapshot)
        ref = self._objects.put_object(
            FORWARD_RETURN_RESULT_OBJECT_TYPE,
            _forward_return_result_id(data_snapshot),
            result,
        )
        return result, (ref,), ()

    def _persist_feedback(
        self, request: FeedbackRequest, cycle_id: str
    ) -> tuple[ResearchFeedback, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        feedback = self._feedback_agent.generate(request, cycle_id=cycle_id)
        ref = self._objects.put_object(
            RESEARCH_FEEDBACK_OBJECT_TYPE, feedback.feedback_id, feedback
        )
        return feedback, (ref,), ()

    def _persist_lesson(
        self,
        **kwargs: object,
    ) -> tuple[ExperimentLesson, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        lesson = self._lesson_builder.build(**kwargs)  # type: ignore[arg-type]
        ref = self._objects.put_object(
            EXPERIMENT_LESSON_OBJECT_TYPE,
            lesson.experiment_id,
            lesson,
        )
        return lesson, (ref,), ()

    def _persist_experiment(
        self, **kwargs: object
    ) -> tuple[ExperimentRecord, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        record = self._experiment_record(**kwargs)  # type: ignore[arg-type]
        ref = self._objects.put_experiment(record)
        return record, (ref,), ()

    def _execute_step[OutputT](
        self,
        *,
        run_id: str,
        cycle_id: str,
        step_id: str,
        input_object_refs: tuple[ObjectRef, ...],
        input_artifact_refs: tuple[ArtifactRef, ...],
        component: ComponentRef,
        operation: Callable[
            [], tuple[OutputT, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]
        ],
        load: Callable[[WorkflowStepAttempt], OutputT],
    ) -> OutputT:
        attempt = self._metadata.begin_attempt(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=step_id,
            input_object_refs=input_object_refs,
            input_artifact_refs=input_artifact_refs,
            component=component,
            started_at=self._clock(),
            reuse_success=self._resume_enabled,
        )
        if attempt.status is WorkflowStepStatus.SUCCEEDED:
            self._reused_step = True
            LOGGER.debug("Step restored step=%s", step_id)
            return load(attempt)
        step_kind = "repair" if ":repair:" in step_id else step_id.rsplit(":", 1)[-1]
        labels = {
            "hypothesis": "Hypothesis generation",
            "factor_design_generation": "Factor design",
            "expression": "Expression generation",
            "repair": "Expression repair",
            "forward_return": "Forward return calculation",
            "calculation": "Factor calculation",
            "preprocessing": "Factor neutralization",
            "evaluation": "Factor evaluation",
            "feedback": "Research feedback",
        }
        level = logging.INFO if step_kind in labels else logging.DEBUG
        try:
            with (
                log_context(step=step_id),
                progress_stage(
                    labels.get(step_kind, step_kind),
                    level=level,
                    report_completion=step_kind != "evaluation",
                ),
            ):
                output, object_refs, artifact_refs = operation()
        except Exception as error:
            self._metadata.fail_attempt(
                attempt,
                failure=_failure_info(
                    stage=step_id,
                    error=error,
                    input_refs=input_object_refs,
                    artifact_refs=input_artifact_refs,
                ),
                completed_at=self._clock(),
            )
            raise
        self._metadata.succeed_attempt(
            attempt,
            output_object_refs=object_refs,
            output_artifact_refs=artifact_refs,
            completed_at=self._clock(),
        )
        return output

    def _execute_candidate_step[OutputT](
        self,
        *,
        run_id: str,
        cycle_id: str,
        step_id: str,
        failure_stage: str,
        input_object_refs: tuple[ObjectRef, ...],
        input_artifact_refs: tuple[ArtifactRef, ...],
        component: ComponentRef,
        operation: Callable[
            [], tuple[OutputT, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]
        ],
        load: Callable[[WorkflowStepAttempt], OutputT],
    ) -> OutputT:
        """Run a candidate step and preserve its failure at the candidate boundary."""
        try:
            return self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id=step_id,
                input_object_refs=input_object_refs,
                input_artifact_refs=input_artifact_refs,
                component=component,
                operation=operation,
                load=load,
            )
        except Exception as error:
            raise _CandidateBoundaryFailure(
                _failure_info(
                    stage=failure_stage,
                    error=error,
                    input_refs=input_object_refs,
                    artifact_refs=input_artifact_refs,
                )
            ) from error

    def _design_generation_request(
        self,
        *,
        hypothesis: ResearchHypothesis,
        hypothesis_ref: ObjectRef,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        context: ResearchContext,
        existing_factor_refs: tuple[ObjectRef, ...],
    ) -> FactorDesignGenerationRequest:
        parent_expression, focus_dimension, parent_scores = _parse_mcts_parent_guidance(
            context
        )
        return FactorDesignGenerationRequest(
            hypothesis_ref=hypothesis_ref,
            hypothesis=hypothesis,
            field_catalog_ref=field_catalog_ref,
            operator_catalog_ref=operator_catalog_ref,
            frequency=context.frequency,
            universe=context.universe,
            existing_factor_refs=existing_factor_refs,
            generation_constraints=self._generation_constraints,
            schema_version="factor-design-generation-request/1.1",
            parent_expression=parent_expression,
            focus_dimension=focus_dimension,
            parent_scores=parent_scores,
        )

    def _generate_expression(
        self,
        design: FactorDesign,
        *,
        run_id: str,
        cycle_id: str,
        factor_design_batch_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
        existing_factor_refs: tuple[ObjectRef, ...],
    ) -> tuple[FactorProposal | None, FailureInfo | None]:
        design_ref = ObjectRef(
            FACTOR_DESIGN_OBJECT_TYPE,
            design.design_id,
            design.schema_version,
        )
        request = FactorExpressionRequest(
            factor_design_ref=design_ref,
            factor_design=design,
            field_catalog_ref=design.field_catalog_ref,
            operator_catalog_ref=design.operator_catalog_ref,
            existing_factor_refs=existing_factor_refs,
            expression_constraints=ExpressionConstraints(
                max_characters=2048,
                max_depth=self._expression_max_depth,
                max_operator_nodes=32,
                max_fields=6,
            ),
            frequency=design.frequency,
            schema_version="factor-expression-request/1.0",
        )
        step_id = f"candidate:{design.design_id}:expression"
        try:
            proposal = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id=step_id,
                input_object_refs=(factor_design_batch_ref, design_ref),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("factor-expression-agent", "1.0"),
                operation=lambda: self._persist_expression(request),
                load=lambda attempt: self._objects.get_object(
                    _only_object_ref(attempt), FactorProposal
                ),
            )
        except Exception as error:
            return None, _failure_info(
                stage=step_id,
                error=error,
                input_refs=(factor_design_batch_ref, design_ref),
                artifact_refs=(config_snapshot_ref,),
            )
        return proposal, None

    def _run_candidate(
        self,
        proposal: FactorProposal,
        *,
        run_id: str,
        cycle_id: str,
        proposal_source_ref: ObjectRef,
        data_snapshot_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
        data_snapshot: DataSnapshot,
        get_forward_return_result: Callable[[], ForwardReturnResult],
        pre_calculation_gate: CandidateCalculationGate | None,
        allow_repair: bool = True,
    ) -> CandidateRunResult:
        validation = self._validate_candidate(
            proposal,
            run_id=run_id,
            cycle_id=cycle_id,
            proposal_source_ref=proposal_source_ref,
            config_snapshot_ref=config_snapshot_ref,
        )
        proposal_attempt_refs = [validation.proposal_ref]
        validation_attempt_refs = [validation.validation_report_ref]

        while (
            allow_repair
            and self._expression_repair_agent is not None
            and len(proposal_attempt_refs) <= self._max_validation_repair_attempts
            and _is_repairable(validation.validation_result.report)
        ):
            try:
                proposal = self._repair_expression(
                    proposal,
                    validation,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    config_snapshot_ref=config_snapshot_ref,
                )
                proposal_attempt_refs.append(_proposal_ref(proposal))
                validation = self._validate_candidate(
                    proposal,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    proposal_source_ref=_proposal_ref(proposal),
                    config_snapshot_ref=config_snapshot_ref,
                )
            except Exception as error:
                return CandidateRunResult(
                    proposal_ref=validation.proposal_ref,
                    validation_report_ref=validation.validation_report_ref,
                    status=ExperimentStatus.FAILED,
                    failure=_failure_info(
                        stage="expression_repair",
                        error=error,
                        input_refs=(
                            validation.proposal_ref,
                            validation.validation_report_ref,
                        ),
                        artifact_refs=(config_snapshot_ref,),
                    ),
                    proposal_attempt_refs=tuple(proposal_attempt_refs),
                    validation_report_attempt_refs=tuple(validation_attempt_refs),
                )
            validation_attempt_refs.append(validation.validation_report_ref)

        if (
            not validation.validation_result.report.is_valid
            or validation.validation_result.factor_spec is None
        ):
            return CandidateRunResult(
                proposal_ref=validation.proposal_ref,
                validation_report_ref=validation.validation_report_ref,
                status=ExperimentStatus.FAILED,
                failure=_validation_failure(
                    validation.validation_result, validation.proposal_ref
                ),
                proposal_attempt_refs=tuple(proposal_attempt_refs),
                validation_report_attempt_refs=tuple(validation_attempt_refs),
            )

        try:
            if pre_calculation_gate is not None:
                gate_decision = pre_calculation_gate.decide(
                    proposal=proposal,
                    factor_spec=validation.validation_result.factor_spec,
                    factor_ref=validation.factor_ref,
                    validation_report_ref=validation.validation_report_ref,
                )
                if not isinstance(gate_decision, CandidateCalculationDecision):
                    raise CosmosAlphaLoopError(
                        "pre_calculation_gate must return CandidateCalculationDecision"
                    )
                if not gate_decision.proceed:
                    return CandidateRunResult(
                        proposal_ref=validation.proposal_ref,
                        validation_report_ref=validation.validation_report_ref,
                        status=ExperimentStatus.REJECTED,
                        factor_ref=validation.factor_ref,
                        pre_calculation_gate_reason=gate_decision.reason_code,
                        proposal_attempt_refs=tuple(proposal_attempt_refs),
                        validation_report_attempt_refs=tuple(validation_attempt_refs),
                    )
            return self._run_valid_candidate(
                run_id=run_id,
                cycle_id=cycle_id,
                proposal_ref=validation.proposal_ref,
                validation_report_ref=validation.validation_report_ref,
                factor_ref=validation.factor_ref,
                factor_spec=validation.validation_result.factor_spec,
                data_snapshot_ref=data_snapshot_ref,
                config_snapshot_ref=config_snapshot_ref,
                data_snapshot=data_snapshot,
                forward_return_result=get_forward_return_result(),
                proposal_attempt_refs=tuple(proposal_attempt_refs),
                validation_report_attempt_refs=tuple(validation_attempt_refs),
            )
        except Exception as error:
            failure = (
                error.failure
                if isinstance(error, _CandidateBoundaryFailure)
                else _failure_info(
                    stage="candidate",
                    error=error,
                    input_refs=(
                        validation.proposal_ref,
                        validation.validation_report_ref,
                    ),
                    artifact_refs=(),
                )
            )
            return CandidateRunResult(
                proposal_ref=validation.proposal_ref,
                validation_report_ref=validation.validation_report_ref,
                status=ExperimentStatus.FAILED,
                factor_ref=validation.factor_ref,
                failure=failure,
                proposal_attempt_refs=tuple(proposal_attempt_refs),
                validation_report_attempt_refs=tuple(validation_attempt_refs),
            )

    def _run_candidate_batch(
        self,
        proposals: list[FactorProposal],
        *,
        run_id: str,
        cycle_id: str,
        data_snapshot_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
        data_snapshot: DataSnapshot,
        get_forward_return_result: Callable[[], ForwardReturnResult],
        batch_gate: CandidateCalculationBatchGate,
        allow_repair: bool = True,
    ) -> tuple[CandidateRunResult, ...]:
        """Run final validation as a batch before any candidate calculation."""
        prepared: list[_PreparedCandidate] = []
        failures: dict[int, CandidateRunResult] = {}
        for candidate_index, proposal in enumerate(proposals):
            result = self._prepare_candidate(
                proposal,
                candidate_index=candidate_index,
                run_id=run_id,
                cycle_id=cycle_id,
                proposal_source_ref=proposal.factor_design_ref,
                allow_repair=allow_repair,
                config_snapshot_ref=config_snapshot_ref,
            )
            if isinstance(result, _PreparedCandidate):
                prepared.append(result)
            else:
                failures[candidate_index] = result

        decisions = batch_gate.decide(
            tuple(
                CandidateCalculationInput(
                    candidate_index=item.candidate_index,
                    proposal=item.proposal,
                    factor_spec=item.validation.validation_result.factor_spec,  # type: ignore[arg-type]
                    factor_ref=item.validation.factor_ref,  # type: ignore[arg-type]
                    validation_report_ref=item.validation.validation_report_ref,
                )
                for item in prepared
            )
        )
        if not isinstance(decisions, tuple) or len(decisions) != len(prepared):
            raise CosmosAlphaLoopError(
                "pre_calculation_batch_gate must return one decision per candidate"
            )
        results: dict[int, CandidateRunResult] = dict(failures)
        for item, decision in zip(prepared, decisions, strict=True):
            if not isinstance(decision, CandidateCalculationDecision):
                raise CosmosAlphaLoopError(
                    "pre_calculation_batch_gate returned an invalid decision"
                )
            validation = item.validation
            factor_spec = validation.validation_result.factor_spec
            factor_ref = validation.factor_ref
            if factor_spec is None or factor_ref is None:
                raise CosmosAlphaLoopError("prepared candidate is missing FactorSpec")
            if not decision.proceed:
                results[item.candidate_index] = CandidateRunResult(
                    proposal_ref=validation.proposal_ref,
                    validation_report_ref=validation.validation_report_ref,
                    status=ExperimentStatus.REJECTED,
                    factor_ref=factor_ref,
                    pre_calculation_gate_reason=decision.reason_code,
                    proposal_attempt_refs=item.proposal_attempt_refs,
                    validation_report_attempt_refs=item.validation_report_attempt_refs,
                )
                continue
            try:
                results[item.candidate_index] = self._run_valid_candidate(
                    run_id=run_id,
                    cycle_id=cycle_id,
                    proposal_ref=validation.proposal_ref,
                    validation_report_ref=validation.validation_report_ref,
                    factor_ref=factor_ref,
                    factor_spec=factor_spec,
                    data_snapshot_ref=data_snapshot_ref,
                    config_snapshot_ref=config_snapshot_ref,
                    data_snapshot=data_snapshot,
                    forward_return_result=get_forward_return_result(),
                    proposal_attempt_refs=item.proposal_attempt_refs,
                    validation_report_attempt_refs=item.validation_report_attempt_refs,
                )
            except Exception as error:
                failure = (
                    error.failure
                    if isinstance(error, _CandidateBoundaryFailure)
                    else _failure_info(
                        stage="candidate",
                        error=error,
                        input_refs=(
                            validation.proposal_ref,
                            validation.validation_report_ref,
                        ),
                        artifact_refs=(),
                    )
                )
                results[item.candidate_index] = CandidateRunResult(
                    proposal_ref=validation.proposal_ref,
                    validation_report_ref=validation.validation_report_ref,
                    status=ExperimentStatus.FAILED,
                    factor_ref=factor_ref,
                    failure=failure,
                    proposal_attempt_refs=item.proposal_attempt_refs,
                    validation_report_attempt_refs=item.validation_report_attempt_refs,
                )
        return tuple(results[index] for index in range(len(proposals)))

    def _prepare_candidate(
        self,
        proposal: FactorProposal,
        *,
        candidate_index: int,
        run_id: str,
        cycle_id: str,
        proposal_source_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
        allow_repair: bool = True,
    ) -> _PreparedCandidate | CandidateRunResult:
        """Validate one candidate, including repair, without calculating it."""
        validation = self._validate_candidate(
            proposal,
            run_id=run_id,
            cycle_id=cycle_id,
            proposal_source_ref=proposal_source_ref,
            config_snapshot_ref=config_snapshot_ref,
        )
        proposal_attempt_refs = [validation.proposal_ref]
        validation_attempt_refs = [validation.validation_report_ref]
        while (
            allow_repair
            and self._expression_repair_agent is not None
            and len(proposal_attempt_refs) <= self._max_validation_repair_attempts
            and _is_repairable(validation.validation_result.report)
        ):
            try:
                proposal = self._repair_expression(
                    proposal,
                    validation,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    config_snapshot_ref=config_snapshot_ref,
                )
                proposal_attempt_refs.append(_proposal_ref(proposal))
                validation = self._validate_candidate(
                    proposal,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    proposal_source_ref=_proposal_ref(proposal),
                    config_snapshot_ref=config_snapshot_ref,
                )
                validation_attempt_refs.append(validation.validation_report_ref)
            except Exception as error:
                return CandidateRunResult(
                    proposal_ref=validation.proposal_ref,
                    validation_report_ref=validation.validation_report_ref,
                    status=ExperimentStatus.FAILED,
                    failure=_failure_info(
                        stage="expression_repair",
                        error=error,
                        input_refs=(
                            validation.proposal_ref,
                            validation.validation_report_ref,
                        ),
                        artifact_refs=(config_snapshot_ref,),
                    ),
                    proposal_attempt_refs=tuple(proposal_attempt_refs),
                    validation_report_attempt_refs=tuple(validation_attempt_refs),
                )
        if (
            not validation.validation_result.report.is_valid
            or validation.validation_result.factor_spec is None
            or validation.factor_ref is None
        ):
            return CandidateRunResult(
                proposal_ref=validation.proposal_ref,
                validation_report_ref=validation.validation_report_ref,
                status=ExperimentStatus.FAILED,
                failure=_validation_failure(
                    validation.validation_result, validation.proposal_ref
                ),
                proposal_attempt_refs=tuple(proposal_attempt_refs),
                validation_report_attempt_refs=tuple(validation_attempt_refs),
            )
        return _PreparedCandidate(
            candidate_index=candidate_index,
            proposal=proposal,
            validation=validation,
            proposal_attempt_refs=tuple(proposal_attempt_refs),
            validation_report_attempt_refs=tuple(validation_attempt_refs),
        )

    def _validate_candidate(
        self,
        proposal: FactorProposal,
        *,
        run_id: str,
        cycle_id: str,
        proposal_source_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
    ) -> _CandidateValidation:
        return self._execute_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=f"candidate:{proposal.proposal_id}:validation",
            input_object_refs=(proposal_source_ref,),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("expression-validator", "1.0"),
            operation=lambda: self._persist_candidate_validation(proposal),
            load=self._load_candidate_validation,
        )

    def _repair_expression(
        self,
        proposal: FactorProposal,
        validation: _CandidateValidation,
        *,
        run_id: str,
        cycle_id: str,
        config_snapshot_ref: ArtifactRef,
    ) -> FactorProposal:
        if self._expression_repair_agent is None:
            raise CosmosAlphaLoopError("ExpressionRepairAgent is not configured")
        design = self._objects.get_object(
            proposal.factor_design_ref,
            FactorDesign,
        )
        request = ExpressionRepairRequest(
            factor_design_ref=proposal.factor_design_ref,
            factor_design=design,
            factor_proposal_ref=validation.proposal_ref,
            factor_proposal=proposal,
            validation_report_ref=validation.validation_report_ref,
            validation_report=validation.validation_result.report,
            field_catalog_ref=design.field_catalog_ref,
            operator_catalog_ref=design.operator_catalog_ref,
            generation_attempt=proposal.generation_attempt + 1,
            max_validation_repair_attempts=(self._max_validation_repair_attempts),
            schema_version="expression-repair-request/1.0",
        )
        step_id = f"candidate:{design.design_id}:repair:{request.generation_attempt}"
        return self._execute_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=step_id,
            input_object_refs=(
                request.factor_design_ref,
                request.factor_proposal_ref,
                request.validation_report_ref,
                request.field_catalog_ref,
                request.operator_catalog_ref,
            ),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("expression-repair-agent", "1.0"),
            operation=lambda: self._persist_repair(request),
            load=lambda attempt: self._objects.get_object(
                _only_object_ref(attempt), FactorProposal
            ),
        )

    def _run_valid_candidate(
        self,
        *,
        run_id: str,
        cycle_id: str,
        proposal_ref: ObjectRef,
        validation_report_ref: ObjectRef,
        factor_ref: ObjectRef | None,
        factor_spec: FactorSpec,
        data_snapshot_ref: ObjectRef,
        config_snapshot_ref: ArtifactRef,
        data_snapshot: DataSnapshot,
        forward_return_result: ForwardReturnResult,
        proposal_attempt_refs: tuple[ObjectRef, ...],
        validation_report_attempt_refs: tuple[ObjectRef, ...],
    ) -> CandidateRunResult:
        if factor_ref is None:
            raise CosmosAlphaLoopError("validated factor has no persisted FactorSpec")
        calculation_result = self._execute_candidate_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=f"candidate:{factor_ref.object_id}:calculation",
            failure_stage="calculation",
            input_object_refs=(factor_ref, validation_report_ref, data_snapshot_ref),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("factor-calculator", "1.0"),
            operation=lambda: self._persist_calculation(
                factor_spec, data_snapshot, validation_report_ref
            ),
            load=lambda attempt: self._objects.get_object(
                _only_object_ref(attempt), FactorCalculationResult
            ),
        )
        calculation_result_ref = ObjectRef(
            FACTOR_CALCULATION_RESULT_OBJECT_TYPE,
            _content_scoped_id("calculation", calculation_result),
            calculation_result.schema_version,
        )
        forward_ref = ObjectRef(
            FORWARD_RETURN_RESULT_OBJECT_TYPE,
            _forward_return_result_id(data_snapshot),
            forward_return_result.schema_version,
        )
        preprocessing_result = self._execute_candidate_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=f"candidate:{factor_ref.object_id}:preprocessing",
            failure_stage="preprocessing",
            input_object_refs=(calculation_result_ref, forward_ref, data_snapshot_ref),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("factor-preprocessor", "1.0"),
            operation=lambda: self._persist_preprocessing(
                calculation_result, forward_return_result, data_snapshot
            ),
            load=lambda attempt: self._objects.get_object(
                _only_object_ref(attempt), FactorPreprocessingResult
            ),
        )
        preprocessing_result_ref = ObjectRef(
            FACTOR_PREPROCESSING_RESULT_OBJECT_TYPE,
            _content_scoped_id("preprocessing", preprocessing_result),
            preprocessing_result.schema_version,
        )
        evaluation_started = perf_counter()
        evaluation_result = self._execute_candidate_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=f"candidate:{factor_ref.object_id}:evaluation",
            failure_stage="evaluation",
            input_object_refs=(preprocessing_result_ref, data_snapshot_ref),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("factor-evaluator", "1.0"),
            operation=lambda: self._persist_evaluation(
                preprocessing_result, preprocessing_result_ref, data_snapshot
            ),
            load=lambda attempt: self._objects.get_object(
                _only_object_ref(attempt), EvaluationResult
            ),
        )
        evaluation_result_ref = ObjectRef(
            EVALUATION_RESULT_OBJECT_TYPE,
            _content_scoped_id("evaluation", evaluation_result),
            evaluation_result.schema_version,
        )
        decision = self._execute_candidate_step(
            run_id=run_id,
            cycle_id=cycle_id,
            step_id=f"candidate:{factor_ref.object_id}:acceptance",
            failure_stage="acceptance",
            input_object_refs=(evaluation_result_ref,),
            input_artifact_refs=(config_snapshot_ref,),
            component=ComponentRef("acceptance-policy", "1.0"),
            operation=lambda: self._persist_decision(
                evaluation_result, evaluation_result_ref
            ),
            load=lambda attempt: self._objects.get_object(
                _only_object_ref(attempt), AcceptanceDecision
            ),
        )
        decision_ref = ObjectRef(
            ACCEPTANCE_DECISION_OBJECT_TYPE,
            _content_scoped_id("decision", decision),
            decision.schema_version,
        )
        rank_ic = next(
            (m.value for m in evaluation_result.metrics if m.name == "rank_ic_mean"),
            None,
        )
        LOGGER.debug(
            "Factor identity id=%s object=%s",
            factor_spec.factor_id,
            factor_ref.object_id,
        )
        LOGGER.info(
            "Factor evaluation completed factor=%s rank_ic=%s status=%s reason=%s elapsed=%.2fs",
            factor_spec.factor_id[:12],
            "unavailable" if rank_ic is None else f"{rank_ic:.5f}",
            "accepted" if decision.accepted else "rejected",
            ",".join(decision.reason_codes) or "none",
            perf_counter() - evaluation_started,
        )
        return CandidateRunResult(
            proposal_ref=proposal_ref,
            validation_report_ref=validation_report_ref,
            status=(
                ExperimentStatus.ACCEPTED
                if decision.accepted
                else ExperimentStatus.REJECTED
            ),
            factor_ref=factor_ref,
            calculation_result_ref=calculation_result_ref,
            preprocessing_result_ref=preprocessing_result_ref,
            evaluation_result_ref=evaluation_result_ref,
            acceptance_decision_ref=decision_ref,
            proposal_attempt_refs=proposal_attempt_refs,
            validation_report_attempt_refs=validation_report_attempt_refs,
        )

    def _persist_candidate_validation(
        self, proposal: FactorProposal
    ) -> tuple[_CandidateValidation, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        proposal_ref = self._objects.put_object(
            FACTOR_PROPOSAL_OBJECT_TYPE, proposal.proposal_id, proposal
        )
        result = validate_factor_proposal(
            proposal,
            self._field_catalog,
            self._operator_catalog,
            max_depth=self._expression_max_depth,
        )
        report_ref = self._objects.put_object(
            VALIDATION_REPORT_OBJECT_TYPE,
            _validation_report_id(proposal),
            result.report,
        )
        result_ref = self._objects.put_object(
            FACTOR_VALIDATION_RESULT_OBJECT_TYPE,
            _validation_result_id(proposal),
            result,
        )
        refs: list[ObjectRef] = [proposal_ref, report_ref, result_ref]
        factor_ref: ObjectRef | None = None
        if result.factor_spec is not None:
            factor_ref = self._objects.put_object(
                FACTOR_SPEC_OBJECT_TYPE,
                _content_scoped_id("factor", result.factor_spec),
                result.factor_spec,
            )
            refs.append(factor_ref)
        return (
            _CandidateValidation(proposal_ref, report_ref, result, factor_ref),
            tuple(refs),
            (),
        )

    def _load_candidate_validation(
        self, attempt: WorkflowStepAttempt
    ) -> _CandidateValidation:
        if len(attempt.output_object_refs) not in (3, 4):
            raise CosmosAlphaLoopError("invalid validation step output references")
        proposal_ref, report_ref, result_ref, *factor_refs = attempt.output_object_refs
        result = self._objects.get_object(result_ref, FactorValidationResult)
        return _CandidateValidation(
            proposal_ref=proposal_ref,
            validation_report_ref=report_ref,
            validation_result=result,
            factor_ref=factor_refs[0] if factor_refs else None,
        )

    def _persist_calculation(
        self,
        factor_spec: FactorSpec,
        data_snapshot: DataSnapshot,
        validation_report_ref: ObjectRef,
    ) -> tuple[FactorCalculationResult, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        result = self._factor_calculator.calculate(
            factor_spec, data_snapshot, validation_report_ref=validation_report_ref
        )
        ref = self._objects.put_object(
            FACTOR_CALCULATION_RESULT_OBJECT_TYPE,
            _content_scoped_id("calculation", result),
            result,
        )
        return result, (ref,), ()

    def _persist_preprocessing(
        self,
        calculation: FactorCalculationResult,
        forward_return: ForwardReturnResult,
        data_snapshot: DataSnapshot,
    ) -> tuple[
        FactorPreprocessingResult, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]
    ]:
        result = self._preprocessor.preprocess(
            calculation, forward_return, data_snapshot
        )
        ref = self._objects.put_object(
            FACTOR_PREPROCESSING_RESULT_OBJECT_TYPE,
            _content_scoped_id("preprocessing", result),
            result,
        )
        return result, (ref,), ()

    def _persist_evaluation(
        self,
        preprocessing: FactorPreprocessingResult,
        preprocessing_ref: ObjectRef,
        data_snapshot: DataSnapshot,
    ) -> tuple[EvaluationResult, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        result = self._evaluator.evaluate(
            preprocessing, preprocessing_ref, data_snapshot
        )
        ref = self._objects.put_object(
            EVALUATION_RESULT_OBJECT_TYPE,
            _content_scoped_id("evaluation", result),
            result,
        )
        return result, (ref,), ()

    def _persist_decision(
        self, evaluation: EvaluationResult, evaluation_ref: ObjectRef
    ) -> tuple[AcceptanceDecision, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        decision = self._acceptance_policy.decide(evaluation, evaluation_ref)
        ref = self._objects.put_object(
            ACCEPTANCE_DECISION_OBJECT_TYPE,
            _content_scoped_id("decision", decision),
            decision,
        )
        return decision, (ref,), ()

    def _publish_accepted_to_library(
        self,
        *,
        run_id: str,
        cycle_id: str,
        experiment_record: ExperimentRecord,
        experiment_ref: ObjectRef,
        candidate_results: tuple[CandidateRunResult, ...],
        config_snapshot_ref: ArtifactRef,
        min_rank_ic: float | None = None,
    ) -> tuple[ObjectRef, ...]:
        """Publish only accepted deterministic candidates to the configured library.

        min_rank_ic (v12): 发布到因子库的最低 |rank_ic_mean|。MCTS 模式的
        acceptance 阈值为 -1.0（保留弱节点给 UCT），但因子库只应收录真实
        达标因子，弱因子（如 |IC| < 0.02）跳过入库（树节点不受影响）。
        """
        if self._factor_library is None:
            return ()
        refs: list[ObjectRef] = []
        for candidate in candidate_results:
            if candidate.status is not ExperimentStatus.ACCEPTED:
                continue
            if (
                candidate.factor_ref is None
                or candidate.evaluation_result_ref is None
                or candidate.acceptance_decision_ref is None
            ):
                raise CosmosAlphaLoopError("accepted candidate lacks library inputs")
            factor = self._objects.get_object(candidate.factor_ref, FactorSpec)
            evaluation = self._objects.get_object(
                candidate.evaluation_result_ref,
                EvaluationResult,
            )
            # 质量门槛：|rank_ic_mean| < min_rank_ic 的弱因子跳过入库
            # （acceptance 阈值 -1.0 只为保留树节点，不代表因子库准入）。
            if min_rank_ic is not None:
                rank_ic = next(
                    (m.value for m in evaluation.metrics if m.name == "rank_ic_mean"),
                    None,
                )
                if rank_ic is None or abs(rank_ic) < min_rank_ic:
                    continue
            # v15n: 值相关入库门（与 FZoo 0.75 对齐）。
            # Graph-style daily correlation 由 evaluator 相对已入库因子计算；
            # 达到 publish_max_corr 表示值重复，跳过发布。
            # 说明：这一步只影响“交付库”去重；树节点/评分/FZoo 不受影响。
            if self._publish_max_corr is not None:
                corr_metric = next(
                    (
                        m.value
                        for m in evaluation.metrics
                        if m.name == "factor_max_abs_daily_correlation"
                    ),
                    None,
                )
                if (
                    corr_metric is not None
                    and float(corr_metric) >= self._publish_max_corr
                ):
                    continue
            decision = self._objects.get_object(
                candidate.acceptance_decision_ref,
                AcceptanceDecision,
            )

            def publish_entry(
                *,
                factor: FactorSpec = factor,
                factor_ref: ObjectRef = candidate.factor_ref,
                evaluation: EvaluationResult = evaluation,
                evaluation_ref: ObjectRef = candidate.evaluation_result_ref,
                decision: AcceptanceDecision = decision,
                decision_ref: ObjectRef = candidate.acceptance_decision_ref,
            ) -> tuple[
                FactorLibraryEntry,
                tuple[ObjectRef, ...],
                tuple[ArtifactRef, ...],
            ]:
                return self._persist_library_entry(
                    factor=factor,
                    factor_ref=factor_ref,
                    evaluation=evaluation,
                    evaluation_ref=evaluation_ref,
                    decision=decision,
                    decision_ref=decision_ref,
                    source_experiment_id=experiment_record.experiment_id,
                )

            entry = self._execute_step(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id=(f"candidate:{candidate.factor_ref.object_id}:factor_library"),
                input_object_refs=(
                    candidate.factor_ref,
                    candidate.evaluation_result_ref,
                    candidate.acceptance_decision_ref,
                    experiment_ref,
                ),
                input_artifact_refs=(config_snapshot_ref,),
                component=ComponentRef("factor-library", "1.0"),
                operation=publish_entry,
                load=lambda attempt: self._objects.get_object(
                    _only_object_ref(attempt),
                    FactorLibraryEntry,
                ),
            )
            entry_ref = ObjectRef(
                FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
                entry.factor_id,
                entry.schema_version,
            )
            if entry_ref not in refs:
                refs.append(entry_ref)
        return tuple(refs)

    def _persist_library_entry(
        self,
        *,
        factor: FactorSpec,
        factor_ref: ObjectRef,
        evaluation: EvaluationResult,
        evaluation_ref: ObjectRef,
        decision: AcceptanceDecision,
        decision_ref: ObjectRef,
        source_experiment_id: str,
    ) -> tuple[FactorLibraryEntry, tuple[ObjectRef, ...], tuple[ArtifactRef, ...]]:
        """Write an accepted factor through the repository and persist its DTO mirror."""
        if self._factor_library is None:
            raise CosmosAlphaLoopError("factor library is not configured")
        entry = self._factor_library.get_entry(factor.factor_id)
        if entry is None:
            entry = self._factor_library.publish_accepted(
                factor_spec=factor,
                factor_ref=factor_ref,
                evaluation_result=evaluation,
                evaluation_result_ref=evaluation_ref,
                decision=decision,
                decision_ref=decision_ref,
                source_experiment_id=source_experiment_id,
                accepted_at=self._clock(),
            )
        entry_ref = self._objects.put_object(
            FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
            entry.factor_id,
            entry,
        )
        return entry, (entry_ref,), ()

    def _experiment_record(
        self,
        *,
        run_id: str,
        cycle_id: str,
        parent_ids: tuple[str, ...],
        hypothesis_ref: ObjectRef,
        candidate_results: tuple[CandidateRunResult, ...],
        feedback_ref: ObjectRef | None,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        data_snapshot: DataSnapshot,
    ) -> ExperimentRecord:
        created_at = self._clock()
        return ExperimentRecord(
            experiment_id=_experiment_id(run_id, cycle_id, hypothesis_ref),
            run_id=run_id,
            cycle_id=cycle_id,
            status=_batch_status(candidate_results),
            parent_ids=parent_ids,
            hypothesis_ref=hypothesis_ref,
            factor_proposal_refs=tuple(
                ref
                for result in candidate_results
                for ref in _proposal_attempt_refs(result)
            ),
            factor_refs=_existing_refs(
                result.factor_ref for result in candidate_results
            ),
            validation_report_refs=tuple(
                ref
                for result in candidate_results
                for ref in _validation_attempt_refs(result)
            ),
            calculation_result_refs=_existing_refs(
                result.calculation_result_ref for result in candidate_results
            ),
            factor_preprocessing_result_refs=_existing_refs(
                result.preprocessing_result_ref for result in candidate_results
            ),
            evaluation_result_refs=_existing_refs(
                result.evaluation_result_ref for result in candidate_results
            ),
            acceptance_decision_refs=_existing_refs(
                result.acceptance_decision_ref for result in candidate_results
            ),
            feedback_ref=feedback_ref,
            config_snapshot_ref=config_snapshot_ref,
            code_version=code_version,
            data_version=data_snapshot.snapshot_id,
            created_at=created_at,
            completed_at=created_at,
            schema_version="experiment-record/1.0",
        )


type Clock = Callable[[], datetime]


def _feedback_request(
    hypothesis_ref: ObjectRef,
    factor_design_refs: tuple[ObjectRef, ...],
    candidate_results: tuple[CandidateRunResult, ...],
    *,
    comparison_baseline_refs: tuple[ObjectRef, ...],
    expression_failures: tuple[FailureInfo, ...],
    acceptance_policy: AcceptancePolicySnapshot,
) -> FeedbackRequest:
    return FeedbackRequest(
        hypothesis_ref=hypothesis_ref,
        factor_design_refs=factor_design_refs,
        factor_proposal_refs=tuple(
            ref
            for result in candidate_results
            for ref in _proposal_attempt_refs(result)
        ),
        factor_refs=_existing_refs(result.factor_ref for result in candidate_results),
        validation_report_refs=tuple(
            ref
            for result in candidate_results
            for ref in _validation_attempt_refs(result)
        ),
        calculation_result_refs=_existing_refs(
            result.calculation_result_ref for result in candidate_results
        ),
        factor_preprocessing_result_refs=_existing_refs(
            result.preprocessing_result_ref for result in candidate_results
        ),
        evaluation_result_refs=_existing_refs(
            result.evaluation_result_ref for result in candidate_results
        ),
        acceptance_decision_refs=_existing_refs(
            result.acceptance_decision_ref for result in candidate_results
        ),
        comparison_baseline_refs=comparison_baseline_refs,
        failures=(
            *expression_failures,
            *(
                result.failure
                for result in candidate_results
                if result.failure is not None
            ),
        ),
        acceptance_policy=acceptance_policy,
        schema_version="feedback-request/2.0",
    )


def _only_object_ref(attempt: WorkflowStepAttempt) -> ObjectRef:
    if len(attempt.output_object_refs) != 1:
        raise CosmosAlphaLoopError("step must have exactly one output object reference")
    return attempt.output_object_refs[0]


def _object_ref_by_type(
    attempt: WorkflowStepAttempt,
    object_type: str,
) -> ObjectRef:
    matches = tuple(
        ref for ref in attempt.output_object_refs if ref.object_type == object_type
    )
    if len(matches) != 1:
        raise CosmosAlphaLoopError(
            f"step must have exactly one {object_type} output reference"
        )
    return matches[0]


def _candidate_result_refs(
    results: tuple[CandidateRunResult, ...],
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []
    for result in results:
        refs.extend(_proposal_attempt_refs(result))
        refs.extend(_validation_attempt_refs(result))
        refs.extend(
            ref
            for ref in (
                result.factor_ref,
                result.calculation_result_ref,
                result.preprocessing_result_ref,
                result.evaluation_result_ref,
                result.acceptance_decision_ref,
            )
            if ref is not None
        )
    return tuple(refs)


def _feedback_request_refs(request: FeedbackRequest) -> tuple[ObjectRef, ...]:
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


def _existing_refs(refs: Iterable[ObjectRef | None]) -> tuple[ObjectRef, ...]:
    return tuple(ref for ref in refs if ref is not None)


def _validation_failure(
    validation_result: FactorValidationResult,
    proposal_ref: ObjectRef,
) -> FailureInfo:
    if validation_result.report.issues:
        first_issue = validation_result.report.issues[0]
        error_code = first_issue.code
        message = first_issue.message
    else:
        error_code = "VALIDATION_FAILED"
        message = "factor proposal did not pass validation"
    return FailureInfo(
        stage="validation",
        error_code=error_code,
        message=message,
        retryable=False,
        input_refs=(proposal_ref,),
        artifact_refs=(),
    )


def _batch_status(
    candidate_results: tuple[CandidateRunResult, ...],
) -> ExperimentStatus:
    if any(result.status is ExperimentStatus.ACCEPTED for result in candidate_results):
        return ExperimentStatus.ACCEPTED
    if any(result.status is ExperimentStatus.REJECTED for result in candidate_results):
        return ExperimentStatus.REJECTED
    return ExperimentStatus.FAILED


_MCTS_PARENT_GUIDANCE_PREFIX = "MCTS_PARENT_GUIDANCE_JSON="


def _parse_mcts_parent_guidance(
    context: ResearchContext,
) -> tuple[str | None, str | None, tuple[tuple[str, float], ...]]:
    """Parse an optional MCTS parent-guidance constraint into design fields.

    MCTS expand 注入 ``MCTS_PARENT_GUIDANCE_JSON=<json>`` constraint（对齐
    Graph 的 GRAPH_GUIDANCE_JSON 做法），让 FactorDesign 直接拿到结构化的
    父表达式 / 聚焦维度 / 父指标，而无需从假设散文推断。
    """
    for constraint in getattr(context, "constraints", ()):
        if not isinstance(constraint, str):
            continue
        if not constraint.startswith(_MCTS_PARENT_GUIDANCE_PREFIX):
            continue
        try:
            data = json.loads(constraint[len(_MCTS_PARENT_GUIDANCE_PREFIX) :])
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        expression = data.get("parent_expression")
        dimension = data.get("focus_dimension")
        scores_raw = data.get("parent_scores")
        scores: list[tuple[str, float]] = []
        if isinstance(scores_raw, dict):
            for name, value in scores_raw.items():
                if isinstance(name, str) and isinstance(value, (int, float)):
                    scores.append((name, float(value)))
        return (
            expression if isinstance(expression, str) and expression else None,
            dimension if isinstance(dimension, str) and dimension else None,
            tuple(scores),
        )
    return None, None, ()


def _design_batch_id(design_batch: FactorDesignBatch) -> str:
    return _content_scoped_id("design-batch", design_batch)


def _proposal_ref(proposal: FactorProposal) -> ObjectRef:
    return ObjectRef(
        FACTOR_PROPOSAL_OBJECT_TYPE,
        proposal.proposal_id,
        proposal.schema_version,
    )


def _proposal_attempt_refs(result: CandidateRunResult) -> tuple[ObjectRef, ...]:
    return result.proposal_attempt_refs or (result.proposal_ref,)


def _validation_attempt_refs(
    result: CandidateRunResult,
) -> tuple[ObjectRef, ...]:
    return result.validation_report_attempt_refs or (result.validation_report_ref,)


def _is_repairable(report: object) -> bool:
    if not isinstance(report, ValidationReport) or report.is_valid or not report.issues:
        return False
    return all(issue.code in _REPAIRABLE_VALIDATION_CODES for issue in report.issues)


def _validation_report_id(proposal: FactorProposal) -> str:
    return _proposal_scoped_id("validation-report", proposal.proposal_id)


def _validation_result_id(proposal: FactorProposal) -> str:
    return _proposal_scoped_id("validation-result", proposal.proposal_id)


def _proposal_scoped_id(prefix: str, proposal_id: str) -> str:
    return f"{prefix}-{stable_sha256({'proposal_id': proposal_id})[:24]}"


def _content_scoped_id(prefix: str, value: object) -> str:
    """Scope storage identity to the complete immutable object content."""
    return f"{prefix}-{stable_sha256(value)[:24]}"


def _forward_return_result_id(data_snapshot: DataSnapshot) -> str:
    return (
        "forward-return-"
        + stable_sha256({"data_snapshot_id": data_snapshot.snapshot_id})[:24]
    )


def _experiment_id(run_id: str, cycle_id: str, hypothesis_ref: ObjectRef) -> str:
    return (
        "experiment-"
        + stable_sha256(
            {
                "cycle_id": cycle_id,
                "hypothesis_ref": hypothesis_ref,
                "run_id": run_id,
            }
        )[:24]
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
