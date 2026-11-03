"""Configuration-driven formal runtime assembly for CosmosAlpha chains."""

from __future__ import annotations

import math
import multiprocessing
import sqlite3
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from cosmosalpha import __version__
from cosmosalpha.agents import (
    AcceptanceDecision,
    BranchOutcome,
    BranchStatus,
    EvaluationResult,
    ExperimentRecord,
    FactorDesignConstraints,
    FactorSpec,
    ResearchContext,
    ResearchDirection,
)
from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    ExperimentConfig,
    FailureInfo,
    ObjectRef,
    QlibSourceConfig,
    create_config_snapshot,
    decode_record_artifact,
    from_json,
    from_json_value,
    stable_sha256,
    to_json,
)
from cosmosalpha.data import (
    MARKET_DATA_FIELDS,
    DataSnapshot,
    FixedSampleMarketDataProvider,
    MarketDataRequest,
    QlibFeatureSource,
    QlibFieldMapping,
    QlibFrameLoader,
    QlibMarketDataProvider,
)
from cosmosalpha.evaluation import (
    EVALUATION_PROTOCOL_ID,
    GPLEARN_NEUTRALIZED_PROTOCOL_ID,
    AcceptancePolicy,
    EvaluationInputCache,
    FactorEvaluator,
    ForwardReturnCalculator,
    GplearnNeutralizedPreprocessor,
    GraphCorrelationEngine,
    GraphCorrelationProtocol,
    ReferenceFactorValuesProvider,
)
from cosmosalpha.factors import (
    FactorCalculationCache,
    FactorCalculator,
    build_base_field_catalog,
    build_base_operator_catalog,
)
from cosmosalpha.llm import (
    AuditedLLMExecutor,
    FeedbackPromptViewBuilder,
    LLMCallRecorder,
    LLMClient,
    LLMRetryPolicy,
    OpenAICompatibleLLMClient,
    PromptTemplateRegistry,
    PromptViewBuilder,
    ReplayLLMClient,
)
from cosmosalpha.pipeline.agents import (
    ExpressionRepairAgent,
    FactorDesignAgent,
    FactorExpressionAgent,
    FeedbackAgent,
    HypothesisAgent,
)
from cosmosalpha.pipeline.architecture import (
    ARCHITECTURE_REGISTRY,
    ChainArchitecture,
    GraphArchitecture,
    MCTSArchitecture,
    ParallelBranchesArchitecture,
)
from cosmosalpha.pipeline.context import ContextBuilder
from cosmosalpha.pipeline.graph import (
    GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_OBJECT_TYPE,
    GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION,
    EmptyExistingHighQualityFactorPool,
    ExistingHighQualityFactorPool,
    FactorLibraryCanonicalKeySnapshot,
    GraphAttemptRequest,
    GraphConfig,
    GraphController,
    GraphDiagnostics,
    GraphFactorLibraryReferenceSnapshot,
    GraphFeedbackStub,
    GraphPublicationOutcomeStatus,
    GraphTelemetryRecorder,
    LocalLoopAttemptExecutor,
    configured_seed_candidates,
    graph_attempt_id,
    graph_factor_library_reference_snapshot_id,
    load_graph_candidate_outcomes,
    load_graph_publication_outcomes,
    load_graph_selector_audits,
    load_graph_telemetry,
    normalize_quality_point,
    quality_point_from_node,
    reconcile_graph_candidate_outcomes,
    reconcile_graph_publications,
    write_graph_candidate_outcome_report,
    write_graph_publication_report,
    write_graph_selector_audit_report,
    write_graph_telemetry_report,
)
from cosmosalpha.pipeline.graph.persistence import graph_initial_cycle_index
from cosmosalpha.pipeline.loop import (
    DATA_SNAPSHOT_OBJECT_TYPE,
    FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
    RESEARCH_CONTEXT_OBJECT_TYPE,
    CosmosAlphaLoop,
    CosmosAlphaLoopResult,
)
from cosmosalpha.pipeline.mcts.controller import MCTSController
from cosmosalpha.pipeline.mcts.stub_agents import MCTSFeedbackStub, MCTSHypothesisStub
from cosmosalpha.pipeline.multicycle import MultiCycleController
from cosmosalpha.pipeline.preflight import run_preflight
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    RunStorageLayout,
    SqliteFactorLibrary,
    SqliteWorkflowMetadataRepository,
    WorkflowCycle,
    WorkflowCycleStatus,
    WorkflowMetadataError,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStepAttempt,
    WorkflowStepStatus,
    write_run_reports,
)

_RUNTIME_REPLAY_INSTRUMENTS = (
    "sh600000",
    "sz000001",
    "sz000002",
    "sh600519",
    "sz000003",
    "sh600001",
    "sz000004",
    "sh600002",
)
_MCTS_SEED_EXPRESSION = "rank(close - delay(close, 5))"


class FormalRuntimeError(Exception):
    """Raised when formal runtime assembly or execution cannot proceed."""


@dataclass(frozen=True, slots=True)
class _InitialFactorLibrarySnapshot:
    """Stable peer-factor view captured before parallel workers start."""

    data_snapshot_id: str
    entries: tuple[tuple[str, ArtifactRef], ...]

    def reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Return only peers from the run-start library snapshot."""
        if data_snapshot_id != self.data_snapshot_id:
            return ()
        return tuple(
            ref for factor_id, ref in self.entries if factor_id != exclude_factor_id
        )


class _GraphWorkflowLifecycle:
    """Close Graph-owned workflow cycles without changing shared Loop behavior."""

    def __init__(
        self,
        metadata: SqliteWorkflowMetadataRepository,
        *,
        database_path: str | Path,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind the Graph lifecycle adapter to one run's metadata database."""
        self._metadata = metadata
        self._database_path = Path(database_path)
        self._clock = clock

    def prepare_cycle(self, request: GraphAttemptRequest) -> None:
        """Create an attempt cycle before any Graph-local work can fail."""
        parent_ids = () if request.parent_factor_ref is None else (request.parent_id,)
        self._metadata.create_cycle(
            WorkflowCycle(
                run_id=request.run_id,
                cycle_id=request.attempt_id,
                parent_ids=parent_ids,
                status=WorkflowCycleStatus.ACTIVE,
                created_at=self._clock(),
                completed_at=None,
            )
        )

    def fail_cycle(self, request: GraphAttemptRequest, error: Exception) -> None:
        """Persist a terminal failed cycle for the original attempt exception."""
        del error  # The Loop step failure retains the structured root cause.
        self._metadata.recover_interrupted(
            completed_at=self._clock(),
            run_id=request.run_id,
            cycle_id=request.attempt_id,
        )
        self._metadata.fail_cycle(
            request.run_id,
            request.attempt_id,
            completed_at=self._clock(),
        )

    def complete_run(
        self,
        run_id: str,
        *,
        allow_failed_recovery: bool = False,
    ) -> None:
        """Complete a Graph run only after every associated cycle is terminal."""
        if type(allow_failed_recovery) is not bool:
            raise CoreValidationError("allow_failed_recovery must be a boolean")
        self._metadata.recover_interrupted(
            completed_at=self._clock(),
            run_id=run_id,
        )
        active_attempts = tuple(
            attempt
            for attempt in _read_graph_workflow_attempts(self._database_path, run_id)
            if attempt.status is WorkflowStepStatus.STARTED
        )
        if active_attempts:
            self._metadata.fail_run(run_id, completed_at=self._clock())
            step_ids = ",".join(attempt.step_id for attempt in active_attempts)
            raise FormalRuntimeError("GRAPH_ACTIVE_STEP_ATTEMPTS: " + step_ids)
        cycles = _read_graph_workflow_cycles(self._database_path, run_id)
        active_cycles = tuple(
            cycle for cycle in cycles if cycle.status is WorkflowCycleStatus.ACTIVE
        )
        if active_cycles:
            for cycle in active_cycles:
                self._metadata.fail_cycle(
                    run_id,
                    cycle.cycle_id,
                    completed_at=self._clock(),
                )
            remaining_active = tuple(
                cycle
                for cycle in _read_graph_workflow_cycles(self._database_path, run_id)
                if cycle.status is WorkflowCycleStatus.ACTIVE
            )
            if remaining_active:
                cycle_ids = ",".join(cycle.cycle_id for cycle in remaining_active)
                raise FormalRuntimeError("GRAPH_ACTIVE_CYCLES_UNRESOLVED: " + cycle_ids)
            self._metadata.fail_run(run_id, completed_at=self._clock())
            cycle_ids = ",".join(cycle.cycle_id for cycle in active_cycles)
            raise FormalRuntimeError("GRAPH_ACTIVE_CYCLES: " + cycle_ids)
        run = _read_graph_workflow_run(self._database_path, run_id)
        if run.status is WorkflowRunStatus.FAILED and not allow_failed_recovery:
            raise FormalRuntimeError("GRAPH_WORKFLOW_RUN_ALREADY_FAILED")
        self._metadata.complete_run(run_id, completed_at=self._clock())


def _read_graph_workflow_cycles(
    database_path: str | Path,
    run_id: str,
) -> tuple[WorkflowCycle, ...]:
    """Read Graph run cycles for the finalization guard.

    The workflow repository intentionally exposes mutation methods only.  This
    Graph adapter uses a read-only query at the terminal boundary so it can verify
    the existing workflow schema without changing that shared repository API.
    """
    try:
        with closing(sqlite3.connect(database_path, timeout=30.0)) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            rows = connection.execute(
                "SELECT payload FROM cycles WHERE run_id = ? ORDER BY cycle_id",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_READ_FAILED") from error
    try:
        return tuple(from_json(row[0], WorkflowCycle) for row in rows)
    except Exception as error:  # noqa: BLE001 - preserve malformed metadata cause
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_INVALID") from error


def _read_graph_workflow_attempts(
    database_path: str | Path,
    run_id: str,
) -> tuple[WorkflowStepAttempt, ...]:
    """Read Graph step attempts in their persisted candidate order."""
    try:
        with closing(sqlite3.connect(database_path, timeout=30.0)) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            rows = connection.execute(
                "SELECT payload FROM step_attempts WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_READ_FAILED") from error
    try:
        return tuple(from_json(row[0], WorkflowStepAttempt) for row in rows)
    except Exception as error:  # noqa: BLE001 - preserve malformed metadata cause
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_INVALID") from error


def _read_graph_workflow_run(
    database_path: str | Path,
    run_id: str,
) -> WorkflowRun:
    """Read the Graph run status for an idempotent completion guard."""
    try:
        with closing(sqlite3.connect(database_path, timeout=30.0)) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            row = connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
    except sqlite3.Error as error:
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_READ_FAILED") from error
    if row is None:
        raise FormalRuntimeError("GRAPH_WORKFLOW_RUN_NOT_FOUND")
    try:
        return from_json(row[0], WorkflowRun)
    except Exception as error:  # noqa: BLE001 - preserve malformed metadata cause
        raise FormalRuntimeError("GRAPH_WORKFLOW_METADATA_INVALID") from error


def _graph_attempt_identities(
    run_id: str,
    cycles: tuple[WorkflowCycle, ...],
    *,
    max_rounds: int,
) -> dict[str, tuple[int, str]]:
    """Map recoverable Graph cycles back to deterministic request identity."""
    identities: dict[str, tuple[int, str]] = {}
    for cycle in cycles:
        if graph_initial_cycle_index(run_id, cycle.cycle_id) is not None:
            identities[cycle.cycle_id] = (0, "graph-initial")
            continue
        if cycle.status is not WorkflowCycleStatus.FAILED:
            continue
        if len(cycle.parent_ids) != 1:
            continue
        parent_id = cycle.parent_ids[0]
        for round_id in range(max_rounds):
            if graph_attempt_id(run_id, round_id, parent_id) == cycle.cycle_id:
                identities[cycle.cycle_id] = (round_id, parent_id)
                break
    return identities


def _capture_graph_reference_factor_values(
    request: GraphAttemptRequest,
    *,
    objects: LocalExperimentRepository,
    factor_library: SqliteFactorLibrary,
) -> GraphFactorLibraryReferenceSnapshot:
    """Load or persist the accepted-factor value view for one Graph round."""
    snapshot_ref = ObjectRef(
        GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_OBJECT_TYPE,
        graph_factor_library_reference_snapshot_id(
            request.run_id,
            request.round_id,
        ),
        GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION,
    )
    if objects.exists(snapshot_ref):
        snapshot = objects.get_object(
            snapshot_ref,
            GraphFactorLibraryReferenceSnapshot,
        )
    else:
        entries = tuple(
            sorted(
                (
                    (entry.factor_id, entry.evaluated_factor_values_ref)
                    for entry in factor_library.list_entries()
                    if entry.data_snapshot_id == request.data_snapshot.snapshot_id
                ),
                key=lambda item: item[0],
            )
        )
        snapshot = GraphFactorLibraryReferenceSnapshot(
            run_id=request.run_id,
            round_id=request.round_id,
            data_snapshot_id=request.data_snapshot.snapshot_id,
            entries=entries,
        )
        stored_ref = objects.put_object(
            GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_OBJECT_TYPE,
            snapshot_ref.object_id,
            snapshot,
        )
        if stored_ref != snapshot_ref:
            raise FormalRuntimeError("GRAPH_REFERENCE_SNAPSHOT_REFERENCE_MISMATCH")
    if (
        snapshot.run_id != request.run_id
        or snapshot.round_id != request.round_id
        or snapshot.data_snapshot_id != request.data_snapshot.snapshot_id
    ):
        raise FormalRuntimeError("GRAPH_REFERENCE_SNAPSHOT_IDENTITY_MISMATCH")
    return snapshot


class MarketDataProvider(Protocol):
    """Configured market-data provider required by a formal runtime."""

    def load_snapshot(self, request: MarketDataRequest) -> DataSnapshot:
        """Load one immutable snapshot for the configured request."""


type MarketDataProviderFactory = Callable[
    [LocalArtifactRepository, ExperimentConfig], MarketDataProvider
]


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Optional test-only replacements for configured external boundaries."""

    market_data_provider_factory: MarketDataProviderFactory | None = None
    llm_client: LLMClient | None = None
    qlib_frame_loader: QlibFrameLoader | None = None
    clock: Callable[[], datetime] | None = None
    graph_config: GraphConfig | None = None
    high_quality_factor_pool: ExistingHighQualityFactorPool | None = None


@dataclass(frozen=True, slots=True)
class _BranchTask:
    """Pickle-safe inputs passed to one spawned branch worker."""

    config: ExperimentConfig
    run_id: str
    branch_index: int
    direction: ResearchDirection
    direction_ref: ObjectRef
    root_context: ResearchContext
    data_snapshot: DataSnapshot
    reference_factor_values: tuple[tuple[str, ArtifactRef], ...]
    config_snapshot_ref: ArtifactRef
    resume: bool


@dataclass(frozen=True, slots=True)
class FormalRuntimeResult:
    """Auditable result of one formal single- or multi-cycle invocation."""

    run_id: str
    config_snapshot_ref: ArtifactRef
    data_snapshot_id: str
    data_snapshot_ref: ObjectRef
    architecture_name: str
    cycle_results: tuple[CosmosAlphaLoopResult, ...]
    branch_outcome_refs: tuple[ObjectRef, ...]
    branch_outcomes: tuple[BranchOutcome, ...]
    resumed: bool
    run_directory: str
    cycle_result_paths: tuple[str, ...]
    factor_library_csv_path: str
    workflow_metadata_path: str
    factor_library_path: str
    experiment_record_refs: tuple[ObjectRef, ...]
    hypothesis_refs: tuple[ObjectRef, ...]
    factor_design_batch_refs: tuple[ObjectRef, ...]
    feedback_refs: tuple[ObjectRef, ...]
    evaluation_result_refs: tuple[ObjectRef, ...]
    acceptance_decision_refs: tuple[ObjectRef, ...]
    factor_library_entry_refs: tuple[ObjectRef, ...]
    prompt_render_refs: tuple[ObjectRef, ...]
    llm_call_record_refs: tuple[ObjectRef, ...]
    llm_raw_response_refs: tuple[ArtifactRef, ...]
    mcts_tree_ref: ObjectRef | None = None
    graph_result_ref: ObjectRef | None = None
    graph_diagnostics: GraphDiagnostics | None = None

    def summary_json(self) -> str:
        """Return a compact, non-secret formal-run CLI summary."""
        records = tuple(result.experiment_record for result in self.cycle_results)
        return to_json(
            {
                "config_snapshot_ref": self.config_snapshot_ref,
                "architecture": self.architecture_name,
                "branch_outcome_refs": self.branch_outcome_refs,
                "branch_statuses": tuple(
                    {
                        "branch_id": outcome.branch_id,
                        "status": outcome.status.value,
                    }
                    for outcome in self.branch_outcomes
                ),
                "cycle_count": len(self.cycle_results),
                "cycle_ids": tuple(result.cycle_id for result in self.cycle_results),
                "data_snapshot_ref": self.data_snapshot_ref,
                "data_snapshot_id": self.data_snapshot_id,
                "experiment_record_refs": self.experiment_record_refs,
                "experiment_ids": tuple(
                    None if record is None else record.experiment_id
                    for record in records
                ),
                "hypothesis_refs": self.hypothesis_refs,
                "factor_design_batch_refs": self.factor_design_batch_refs,
                "feedback_refs": self.feedback_refs,
                "evaluation_result_refs": self.evaluation_result_refs,
                "acceptance_decision_refs": self.acceptance_decision_refs,
                "factor_library_entry_refs": self.factor_library_entry_refs,
                "prompt_render_refs": self.prompt_render_refs,
                "llm_call_record_refs": self.llm_call_record_refs,
                "llm_raw_response_refs": self.llm_raw_response_refs,
                "mcts_tree_ref": self.mcts_tree_ref,
                "graph_result_ref": self.graph_result_ref,
                "graph_diagnostics": self.graph_diagnostics,
                "resumed": self.resumed,
                "run_id": self.run_id,
                "statuses": tuple(
                    None if record is None else record.status.value
                    for record in records
                ),
                "run_directory": self.run_directory,
                "cycle_result_paths": self.cycle_result_paths,
                "factor_library_csv_path": self.factor_library_csv_path,
                "workflow_metadata_path": self.workflow_metadata_path,
                "factor_library_path": self.factor_library_path,
            }
        )


class CosmosAlphaRuntime:
    """Fully assembled formal runtime with no CLI concerns."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        run_id: str,
        artifacts: LocalArtifactRepository,
        objects: LocalExperimentRepository,
        layout: RunStorageLayout,
        factor_library: SqliteFactorLibrary,
        data_provider: MarketDataProvider,
        loop: CosmosAlphaLoop | None,
        graph_config: GraphConfig | None,
        graph_loop_factory: Callable[[], CosmosAlphaLoop] | None,
        graph_initial_loop_factory: Callable[[], CosmosAlphaLoop] | None,
        high_quality_factor_pool: ExistingHighQualityFactorPool | None,
        config_snapshot_ref: ArtifactRef,
        llm_executor: AuditedLLMExecutor | None,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind already-validated configured components for one run identity."""
        self._config = config
        self._run_id = run_id
        self._artifacts = artifacts
        self._objects = objects
        self._layout = layout
        self._factor_library = factor_library
        self._data_provider = data_provider
        self._loop = loop
        self._graph_config = graph_config
        self._graph_loop_factory = graph_loop_factory
        self._graph_initial_loop_factory = graph_initial_loop_factory
        self._high_quality_factor_pool = high_quality_factor_pool
        self._config_snapshot_ref = config_snapshot_ref
        self._llm_executor = llm_executor
        self._clock = clock

    def run(self, *, resume: bool) -> FormalRuntimeResult:
        """Load one snapshot and execute the selected top-level architecture."""
        request = _market_data_request(self._config)
        snapshot = self._data_provider.load_snapshot(request)
        base_context = _base_context(self._config, snapshot)
        architecture = ARCHITECTURE_REGISTRY.create(self._config.architecture.name)
        if isinstance(architecture, ChainArchitecture):
            return self._run_chain(snapshot, base_context, resume=resume)
        if isinstance(architecture, MCTSArchitecture):
            return self._run_mcts(snapshot, base_context, resume=resume)
        if isinstance(architecture, GraphArchitecture):
            return self._run_graph(snapshot, base_context, resume=resume)
        if isinstance(architecture, ParallelBranchesArchitecture):
            return self._run_parallel_branches(
                snapshot,
                base_context,
                resume=resume,
            )
        raise FormalRuntimeError("RUNTIME_ARCHITECTURE_UNSUPPORTED")

    def _run_chain(
        self,
        snapshot: DataSnapshot,
        base_context: ResearchContext,
        *,
        resume: bool,
    ) -> FormalRuntimeResult:
        """Execute the existing serial chain unchanged."""
        if self._loop is None:
            raise FormalRuntimeError("CHAIN_LOOP_NOT_CONFIGURED")
        if self._config.workflow.max_cycles == 1:
            cycle_result = self._loop.run(
                run_id=self._run_id,
                cycle_id="cycle-001",
                context=base_context,
                data_snapshot=snapshot,
                config_snapshot_ref=self._config_snapshot_ref,
                code_version=__version__,
                resume=resume,
            )
            cycle_results = (cycle_result,)
        else:
            multi_result = MultiCycleController(
                self._loop,
                ContextBuilder(
                    self._objects,
                ),
                workflow_config=self._config.workflow,
                prompts_config=self._config.prompts,
            ).run(
                run_id=self._run_id,
                cycle_id_prefix="cycle",
                base_context=base_context,
                data_snapshot=snapshot,
                config_snapshot_ref=self._config_snapshot_ref,
                code_version=__version__,
                resume=resume,
            )
            cycle_results = multi_result.cycle_results
        data_snapshot_ref = ObjectRef(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot.schema_version,
        )
        candidate_results = tuple(
            candidate
            for result in cycle_results
            for candidate in result.candidate_results
        )
        records = tuple(
            result.experiment_record
            for result in cycle_results
            if result.experiment_record is not None
        )
        cycle_result_paths, factor_library_csv_path = write_run_reports(
            self._layout,
            records,
            objects=self._objects,
        )
        executor = self._llm_executor
        return FormalRuntimeResult(
            run_id=self._run_id,
            config_snapshot_ref=self._config_snapshot_ref,
            data_snapshot_id=snapshot.snapshot_id,
            data_snapshot_ref=data_snapshot_ref,
            architecture_name="chain",
            cycle_results=cycle_results,
            branch_outcome_refs=(),
            branch_outcomes=(),
            resumed=any(result.resumed for result in cycle_results),
            run_directory=str(self._layout.run_root),
            cycle_result_paths=cycle_result_paths,
            factor_library_csv_path=factor_library_csv_path,
            workflow_metadata_path=str(self._layout.workflow_db_path),
            factor_library_path=str(self._layout.factor_library_db_path),
            experiment_record_refs=tuple(
                result.experiment_ref
                for result in cycle_results
                if result.experiment_ref is not None
            ),
            hypothesis_refs=tuple(
                result.hypothesis_ref
                for result in cycle_results
                if result.hypothesis_ref is not None
            ),
            factor_design_batch_refs=tuple(
                result.factor_design_batch_ref
                for result in cycle_results
                if result.factor_design_batch_ref is not None
            ),
            feedback_refs=tuple(
                result.feedback_ref
                for result in cycle_results
                if result.feedback_ref is not None
            ),
            evaluation_result_refs=tuple(
                candidate.evaluation_result_ref
                for candidate in candidate_results
                if candidate.evaluation_result_ref is not None
            ),
            acceptance_decision_refs=tuple(
                candidate.acceptance_decision_ref
                for candidate in candidate_results
                if candidate.acceptance_decision_ref is not None
            ),
            factor_library_entry_refs=tuple(
                ref
                for result in cycle_results
                for ref in result.factor_library_entry_refs
            ),
            prompt_render_refs=(
                () if executor is None else tuple(executor.prompt_refs.values())
            ),
            llm_call_record_refs=(
                () if executor is None else tuple(executor.call_record_refs)
            ),
            llm_raw_response_refs=(
                ()
                if executor is None
                else tuple(
                    record.raw_response_ref for record in executor.recorder.records
                )
            ),
        )

    def _run_mcts(
        self,
        snapshot: DataSnapshot,
        base_context: ResearchContext,
        *,
        resume: bool,
    ) -> FormalRuntimeResult:
        """Execute MCTS with the shared v2 factor pipeline."""
        del resume
        if self._loop is None:
            raise FormalRuntimeError("MCTS_LOOP_NOT_CONFIGURED")
        settings = self._config.mcts
        if settings is None:
            raise FormalRuntimeError("MCTS_CONFIG_REQUIRED")
        self._objects.put_object(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot,
        )

        controller = MCTSController(
            loop=self._loop,
            experiment_repository=self._objects,
            artifact_repository=self._artifacts,
            llm_client=(
                None if self._llm_executor is None else self._llm_executor.client
            ),
            field_catalog=build_base_field_catalog(),
            operator_catalog=build_base_operator_catalog(),
            exploration_weight=settings.exploration_weight,
            alpha_threshold=self._config.evaluation.acceptance_threshold,
            model_alias=self._config.llm.model_alias,
            correlation_min_cross_section_samples=_cross_section_minimum(self._config),
            correlation_min_daily_effective_stock_ratio=(
                self._config.evaluation.min_daily_effective_stock_ratio
            ),
            correlation_min_pair_window_ratio=(
                self._config.evaluation.min_effective_date_ratio
            ),
        )
        controller.run(
            seed_expression=_MCTS_SEED_EXPRESSION,
            data_snapshot=snapshot,
            base_context=base_context,
            config_snapshot_ref=self._config_snapshot_ref,
            code_version=__version__,
            run_id=self._run_id,
            persist_id="mcts-tree",
            max_iterations=settings.max_iterations,
            initial_budget=settings.initial_budget,
            budget_increment=settings.budget_increment,
            breakthrough_relax=settings.breakthrough_relax,
        )

        records = _run_experiment_records(self._objects, self._run_id)
        cycle_result_paths, factor_library_csv_path = write_run_reports(
            self._layout,
            records,
            objects=self._objects,
        )
        data_snapshot_ref = ObjectRef(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot.schema_version,
        )
        executor = self._llm_executor
        return FormalRuntimeResult(
            run_id=self._run_id,
            config_snapshot_ref=self._config_snapshot_ref,
            data_snapshot_id=snapshot.snapshot_id,
            data_snapshot_ref=data_snapshot_ref,
            architecture_name="mcts",
            cycle_results=(),
            branch_outcome_refs=(),
            branch_outcomes=(),
            resumed=False,
            run_directory=str(self._layout.run_root),
            cycle_result_paths=cycle_result_paths,
            factor_library_csv_path=factor_library_csv_path,
            workflow_metadata_path=str(self._layout.workflow_db_path),
            factor_library_path=str(self._layout.factor_library_db_path),
            experiment_record_refs=tuple(
                ObjectRef(
                    "experiment_record",
                    record.experiment_id,
                    record.schema_version,
                )
                for record in records
            ),
            hypothesis_refs=tuple(record.hypothesis_ref for record in records),
            factor_design_batch_refs=(),
            feedback_refs=tuple(
                record.feedback_ref
                for record in records
                if record.feedback_ref is not None
            ),
            evaluation_result_refs=tuple(
                ref for record in records for ref in record.evaluation_result_refs
            ),
            acceptance_decision_refs=tuple(
                ref for record in records for ref in record.acceptance_decision_refs
            ),
            factor_library_entry_refs=(),
            prompt_render_refs=(
                () if executor is None else tuple(executor.prompt_refs.values())
            ),
            llm_call_record_refs=(
                () if executor is None else tuple(executor.call_record_refs)
            ),
            llm_raw_response_refs=(
                ()
                if executor is None
                else tuple(
                    record.raw_response_ref for record in executor.recorder.records
                )
            ),
            mcts_tree_ref=ObjectRef("mcts_tree", "mcts-tree", "mcts-tree/1.0"),
        )

    def _run_graph(
        self,
        snapshot: DataSnapshot,
        base_context: ResearchContext,
        *,
        resume: bool,
    ) -> FormalRuntimeResult:
        """Execute Graph Search v1 with committed snapshot resume."""
        if self._graph_config is None or self._graph_loop_factory is None:
            raise FormalRuntimeError("GRAPH_RUNTIME_NOT_CONFIGURED")
        data_snapshot_ref = ObjectRef(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot.schema_version,
        )
        self._objects.put_object(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot,
        )
        metadata = SqliteWorkflowMetadataRepository.initialize(
            self._layout.workflow_db_path,
            object_verifier=self._objects,
            artifact_verifier=self._artifacts,
        )
        root_context_ref = self._objects.put_object(
            RESEARCH_CONTEXT_OBJECT_TYPE,
            "context-" + stable_sha256(base_context)[:24],
            base_context,
        )
        metadata.create_run(
            WorkflowRun(
                run_id=self._run_id,
                context_ref=root_context_ref,
                data_snapshot_ref=data_snapshot_ref,
                config_snapshot_ref=self._config_snapshot_ref,
                code_version=__version__,
                status=WorkflowRunStatus.ACTIVE,
                created_at=self._clock(),
                completed_at=None,
            )
        )
        run_before_graph = _read_graph_workflow_run(
            self._layout.workflow_db_path,
            self._run_id,
        )
        lifecycle = _GraphWorkflowLifecycle(
            metadata,
            database_path=self._layout.workflow_db_path,
            clock=self._clock,
        )
        evaluation_dates = _snapshot_evaluation_dates(self._artifacts, snapshot)
        protocol = _graph_correlation_protocol(
            self._config,
            snapshot,
            evaluation_dates,
        )
        telemetry = GraphTelemetryRecorder(self._objects, run_id=self._run_id)
        correlation_engine = GraphCorrelationEngine(
            self._artifacts,
            protocol,
            evaluation_dates,
        )
        library_snapshot_object_id = "graph-factor-library-" + self._run_id
        library_snapshot_ref = ObjectRef(
            "graph_factor_library_key_snapshot",
            library_snapshot_object_id,
            "graph-library-key-snapshot/1.0",
        )
        if resume and self._objects.exists(library_snapshot_ref):
            library_key_snapshot = self._objects.get_object(
                library_snapshot_ref,
                FactorLibraryCanonicalKeySnapshot,
            )
        else:
            library_entries = self._factor_library.list_entries()
            library_specs = tuple(
                self._objects.get_object(entry.factor_ref, FactorSpec)
                for entry in library_entries
            )
            library_key_snapshot = FactorLibraryCanonicalKeySnapshot.from_factor_specs(
                library_specs,
                source_revision=stable_sha256(
                    tuple(
                        (entry.factor_id, entry.schema_version, entry.factor_ref)
                        for entry in library_entries
                    )
                ),
            )
            self._objects.put_object(
                "graph_factor_library_key_snapshot",
                library_snapshot_object_id,
                library_key_snapshot,
            )
        workflow_cycles = _read_graph_workflow_cycles(
            self._layout.workflow_db_path,
            self._run_id,
        )
        workflow_attempts = _read_graph_workflow_attempts(
            self._layout.workflow_db_path,
            self._run_id,
        )
        recovered_candidate_outcomes = reconcile_graph_candidate_outcomes(
            self._objects,
            workflow_attempts,
            run_id=self._run_id,
            attempt_identities=_graph_attempt_identities(
                self._run_id,
                workflow_cycles,
                max_rounds=self._graph_config.search.max_expansion_attempts,
            ),
        )
        if resume:
            reconcile_graph_publications(
                self._objects,
                self._factor_library,
                run_id=self._run_id,
                clock=self._clock,
                candidate_outcomes=recovered_candidate_outcomes,
                workflow_attempts=workflow_attempts,
            )
        attempt_executor = LocalLoopAttemptExecutor(
            self._graph_loop_factory,
            max_workers=self._config.runtime.max_workers,
            initial_loop_factory=self._graph_initial_loop_factory,
            initial_run_context=base_context,
            guidance_persister=lambda guidance: self._objects.put_object(
                "graph_expansion_guidance",
                "guidance-" + stable_sha256(guidance)[:24],
                guidance,
            ),
            cycle_preparer=lifecycle.prepare_cycle,
            cycle_failure_finalizer=lifecycle.fail_cycle,
            reference_factor_values_provider_factory=lambda request: (
                _capture_graph_reference_factor_values(
                    request,
                    objects=self._objects,
                    factor_library=self._factor_library,
                )
            ),
        )

        def refresh_graph_reports() -> None:
            write_run_reports(
                self._layout,
                _run_experiment_records(self._objects, self._run_id),
                objects=self._objects,
                include_branch_cycle=False,
            )

        refresh_graph_reports()
        controller = GraphController(
            config=self._graph_config,
            attempt_executor=attempt_executor,
            artifacts=self._artifacts,
            objects=self._objects,
            correlation_engine=correlation_engine,
            correlation_protocol=protocol,
            min_effective_date_ratio=self._config.evaluation.min_effective_date_ratio,
            high_quality_pool=(
                self._high_quality_factor_pool or EmptyExistingHighQualityFactorPool()
            ),
            seed_candidates=(
                configured_seed_candidates(self._graph_config.initialization.seed_pool)
                if self._graph_config.initialization.seed_pool
                in {"huatai", "huatai_v1"}
                else ()
            ),
            canonical_key_snapshot=library_key_snapshot,
            progress_callback=refresh_graph_reports,
            telemetry=telemetry,
            initial_cycle_statuses={
                cycle.cycle_id: cycle.status.value
                for cycle in workflow_cycles
                if graph_initial_cycle_index(self._run_id, cycle.cycle_id) is not None
            },
        )

        def reconcile_graph_audits() -> tuple[GraphPublicationOutcomeStatus, ...]:
            """Reconcile accepted decisions before allowing Graph run completion."""
            workflow_attempts = _read_graph_workflow_attempts(
                self._layout.workflow_db_path,
                self._run_id,
            )
            candidate_outcomes = reconcile_graph_candidate_outcomes(
                self._objects,
                workflow_attempts,
                run_id=self._run_id,
                attempt_identities=_graph_attempt_identities(
                    self._run_id,
                    _read_graph_workflow_cycles(
                        self._layout.workflow_db_path,
                        self._run_id,
                    ),
                    max_rounds=self._graph_config.search.max_expansion_attempts,
                ),
            )
            reconcile_graph_publications(
                self._objects,
                self._factor_library,
                run_id=self._run_id,
                clock=self._clock,
                candidate_outcomes=candidate_outcomes,
                workflow_attempts=workflow_attempts,
            )
            latest_publications = {}
            for outcome in load_graph_publication_outcomes(
                self._objects,
                run_id=self._run_id,
            ):
                latest_publications[outcome.acceptance_decision_ref] = outcome
            return tuple(outcome.status for outcome in latest_publications.values())

        def flush_graph_audit_reports() -> None:
            """Write derived Graph reports without changing lifecycle semantics."""
            try:
                write_graph_publication_report(
                    self._layout.run_root,
                    load_graph_publication_outcomes(
                        self._objects,
                        run_id=self._run_id,
                    ),
                    run_id=self._run_id,
                )
            except Exception:
                # Immutable publication objects remain authoritative.
                pass
            try:
                write_graph_candidate_outcome_report(
                    self._layout.run_root,
                    load_graph_candidate_outcomes(
                        self._objects,
                        run_id=self._run_id,
                    ),
                    run_id=self._run_id,
                )
            except Exception:
                # Immutable candidate objects remain authoritative.
                pass

        def flush_graph_telemetry() -> None:
            """Persist observational Graph timings without affecting run outcome."""
            try:
                telemetry.persist()
                write_graph_telemetry_report(
                    self._layout.run_root,
                    load_graph_telemetry(self._objects, run_id=self._run_id),
                    run_id=self._run_id,
                )
            except Exception:
                # Telemetry is diagnostic only; search and lifecycle semantics win.
                return

        def flush_graph_selector_audit_report() -> None:
            """Write selector audit rows without changing Graph lifecycle state."""
            try:
                write_graph_selector_audit_report(
                    self._layout.run_root,
                    load_graph_selector_audits(self._objects, run_id=self._run_id),
                    run_id=self._run_id,
                )
            except Exception:
                # Immutable audit objects remain authoritative if report writing fails.
                return

        try:
            graph_result = controller.run(
                run_id=self._run_id,
                context=base_context,
                data_snapshot=snapshot,
                config_snapshot_ref=self._config_snapshot_ref,
                code_version=__version__,
                resume=resume,
            )
        except Exception:
            flush_graph_telemetry()
            flush_graph_selector_audit_report()
            try:
                publication_statuses = reconcile_graph_audits()
            except Exception:
                metadata.fail_run(self._run_id, completed_at=self._clock())
                raise
            flush_graph_audit_reports()
            if GraphPublicationOutcomeStatus.PUBLICATION_FAILED in publication_statuses:
                metadata.fail_run(self._run_id, completed_at=self._clock())
            raise
        try:
            publication_statuses = reconcile_graph_audits()
        except Exception:
            flush_graph_telemetry()
            flush_graph_selector_audit_report()
            metadata.fail_run(self._run_id, completed_at=self._clock())
            raise
        flush_graph_audit_reports()
        if GraphPublicationOutcomeStatus.PUBLICATION_FAILED in publication_statuses:
            flush_graph_telemetry()
            flush_graph_selector_audit_report()
            metadata.fail_run(self._run_id, completed_at=self._clock())
            raise FormalRuntimeError("GRAPH_PUBLICATION_FAILED")
        lifecycle.complete_run(
            self._run_id,
            allow_failed_recovery=(
                resume and run_before_graph.status is WorkflowRunStatus.FAILED
            ),
        )
        flush_graph_telemetry()
        flush_graph_selector_audit_report()
        graph_result_ref = ObjectRef(
            "graph_result",
            "graph-" + stable_sha256(graph_result)[:24],
            graph_result.schema_version,
        )
        records = _run_experiment_records(self._objects, self._run_id)
        library_refs = tuple(
            ObjectRef(
                FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
                entry.factor_id,
                entry.schema_version,
            )
            for entry in self._factor_library.list_entries()
        )
        cycle_result_paths, factor_library_csv_path = write_run_reports(
            self._layout,
            records,
            objects=self._objects,
            include_branch_cycle=False,
            graph_normalized_metrics_by_factor_id={
                node.factor_id: dict(
                    zip(
                        (
                            "normalized_rank_ic",
                            "normalized_rank_ic_std",
                            "normalized_top20_mean_return",
                        ),
                        normalize_quality_point(
                            quality_point_from_node(node),
                            self._graph_config.quality,
                        ),
                        strict=True,
                    )
                )
                for node in graph_result.graph.nodes.values()
            },
        )
        return FormalRuntimeResult(
            run_id=self._run_id,
            config_snapshot_ref=self._config_snapshot_ref,
            data_snapshot_id=snapshot.snapshot_id,
            data_snapshot_ref=data_snapshot_ref,
            architecture_name="graph",
            cycle_results=(),
            branch_outcome_refs=(),
            branch_outcomes=(),
            resumed=resume,
            run_directory=str(self._layout.run_root),
            cycle_result_paths=cycle_result_paths,
            factor_library_csv_path=factor_library_csv_path,
            workflow_metadata_path=str(self._layout.workflow_db_path),
            factor_library_path=str(self._layout.factor_library_db_path),
            experiment_record_refs=tuple(
                ObjectRef(
                    "experiment_record", record.experiment_id, record.schema_version
                )
                for record in records
            ),
            hypothesis_refs=tuple(record.hypothesis_ref for record in records),
            factor_design_batch_refs=(),
            feedback_refs=tuple(
                record.feedback_ref
                for record in records
                if record.feedback_ref is not None
            ),
            evaluation_result_refs=tuple(
                ref for record in records for ref in record.evaluation_result_refs
            ),
            acceptance_decision_refs=tuple(
                ref for record in records for ref in record.acceptance_decision_refs
            ),
            factor_library_entry_refs=library_refs,
            prompt_render_refs=(),
            llm_call_record_refs=(),
            llm_raw_response_refs=(),
            graph_result_ref=graph_result_ref,
            graph_diagnostics=graph_result.diagnostics,
        )

    def _run_parallel_branches(
        self,
        snapshot: DataSnapshot,
        base_context: ResearchContext,
        *,
        resume: bool,
    ) -> FormalRuntimeResult:
        """Execute configured directions concurrently and merge deterministically."""
        data_snapshot_ref = self._objects.put_object(
            DATA_SNAPSHOT_OBJECT_TYPE,
            snapshot.snapshot_id,
            snapshot,
        )
        root_context_ref = self._objects.put_object(
            "research_context",
            "context-" + stable_sha256(base_context)[:24],
            base_context,
        )
        metadata = SqliteWorkflowMetadataRepository.initialize(
            self._layout.workflow_db_path,
            object_verifier=self._objects,
            artifact_verifier=self._artifacts,
        )
        metadata.create_run(
            WorkflowRun(
                run_id=self._run_id,
                context_ref=root_context_ref,
                data_snapshot_ref=data_snapshot_ref,
                config_snapshot_ref=self._config_snapshot_ref,
                code_version=__version__,
                status=WorkflowRunStatus.ACTIVE,
                created_at=self._clock(),
                completed_at=None,
            )
        )
        reference_factor_values = tuple(
            (entry.factor_id, entry.evaluated_factor_values_ref)
            for entry in self._factor_library.list_entries()
            if entry.data_snapshot_id == snapshot.snapshot_id
        )
        tasks = tuple(
            self._branch_task(
                index,
                description,
                base_context,
                snapshot,
                reference_factor_values,
                resume,
            )
            for index, description in enumerate(
                self._config.architecture.directions, start=1
            )
        )
        refs: list[ObjectRef] = []
        published_experiment_ids: set[str] = set()
        published_factor_ids: set[str] = set()
        library_refs: list[ObjectRef] = []
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=min(self._config.runtime.max_workers, len(tasks)),
                mp_context=context,
            ) as pool:
                futures = {
                    pool.submit(_run_parallel_branch_worker, task): task
                    for task in tasks
                }
                pending = set(futures)
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )
                    new_refs, new_records = _publish_completed_parallel_cycles(
                        self._run_id,
                        objects=self._objects,
                        library=self._factor_library,
                        clock=self._clock,
                        published_experiment_ids=published_experiment_ids,
                        published_factor_ids=published_factor_ids,
                    )
                    library_refs.extend(new_refs)
                    if new_records:
                        write_run_reports(
                            self._layout,
                            _run_experiment_records(self._objects, self._run_id),
                            objects=self._objects,
                        )
                    for future in done:
                        task = futures[future]
                        try:
                            refs.append(future.result())
                        except Exception as error:
                            refs.append(self._persist_worker_crash(task, error))
                new_refs, new_records = _publish_completed_parallel_cycles(
                    self._run_id,
                    objects=self._objects,
                    library=self._factor_library,
                    clock=self._clock,
                    published_experiment_ids=published_experiment_ids,
                    published_factor_ids=published_factor_ids,
                )
                library_refs.extend(new_refs)
                if new_records:
                    write_run_reports(
                        self._layout,
                        _run_experiment_records(self._objects, self._run_id),
                        objects=self._objects,
                    )
        except Exception:
            metadata.fail_run(self._run_id, completed_at=self._clock())
            raise
        outcomes = tuple(
            sorted(
                (self._objects.get_object(ref, BranchOutcome) for ref in refs),
                key=lambda outcome: outcome.branch_index,
            )
        )
        outcome_refs = tuple(_branch_outcome_ref(outcome) for outcome in outcomes)
        successful = tuple(
            outcome for outcome in outcomes if outcome.status is BranchStatus.COMPLETED
        )
        if not successful:
            metadata.fail_run(self._run_id, completed_at=self._clock())
            raise FormalRuntimeError("PARALLEL_BRANCHES_ALL_FAILED")
        metadata.complete_run(self._run_id, completed_at=self._clock())
        experiment_refs = tuple(
            ref for outcome in outcomes for ref in outcome.experiment_refs
        )
        records = tuple(self._objects.get_experiment(ref) for ref in experiment_refs)
        cycle_result_paths, factor_library_csv_path = write_run_reports(
            self._layout,
            records,
            objects=self._objects,
        )
        return FormalRuntimeResult(
            run_id=self._run_id,
            config_snapshot_ref=self._config_snapshot_ref,
            data_snapshot_id=snapshot.snapshot_id,
            data_snapshot_ref=data_snapshot_ref,
            architecture_name="parallel_branches",
            cycle_results=(),
            branch_outcome_refs=outcome_refs,
            branch_outcomes=outcomes,
            resumed=resume,
            run_directory=str(self._layout.run_root),
            cycle_result_paths=cycle_result_paths,
            factor_library_csv_path=factor_library_csv_path,
            workflow_metadata_path=str(self._layout.workflow_db_path),
            factor_library_path=str(self._layout.factor_library_db_path),
            experiment_record_refs=experiment_refs,
            hypothesis_refs=tuple(record.hypothesis_ref for record in records),
            factor_design_batch_refs=(),
            feedback_refs=tuple(
                record.feedback_ref
                for record in records
                if record.feedback_ref is not None
            ),
            evaluation_result_refs=tuple(
                ref for record in records for ref in record.evaluation_result_refs
            ),
            acceptance_decision_refs=tuple(
                ref for record in records for ref in record.acceptance_decision_refs
            ),
            factor_library_entry_refs=tuple(
                sorted(
                    library_refs,
                    key=lambda ref: (ref.object_type, ref.object_id),
                )
            ),
            prompt_render_refs=tuple(
                ref for outcome in outcomes for ref in outcome.prompt_render_refs
            ),
            llm_call_record_refs=tuple(
                ref for outcome in outcomes for ref in outcome.llm_call_record_refs
            ),
            llm_raw_response_refs=tuple(
                ref for outcome in outcomes for ref in outcome.llm_raw_response_refs
            ),
        )

    def _branch_task(
        self,
        index: int,
        description: str,
        base_context: ResearchContext,
        snapshot: DataSnapshot,
        reference_factor_values: tuple[tuple[str, ArtifactRef], ...],
        resume: bool,
    ) -> _BranchTask:
        direction = ResearchDirection(
            direction_id=(
                "direction-" + stable_sha256({"description": description})[:24]
            ),
            description=description,
            schema_version="research-direction/1.0",
        )
        direction_ref = self._objects.put_object(
            "research_direction",
            direction.direction_id,
            direction,
        )
        return _BranchTask(
            config=self._config,
            run_id=self._run_id,
            branch_index=index,
            direction=direction,
            direction_ref=direction_ref,
            root_context=base_context,
            data_snapshot=snapshot,
            reference_factor_values=reference_factor_values,
            config_snapshot_ref=self._config_snapshot_ref,
            resume=resume,
        )

    def _persist_worker_crash(self, task: _BranchTask, error: Exception) -> ObjectRef:
        outcome = _failed_branch_outcome(
            task,
            (),
            FailureInfo(
                stage="parallel_branch",
                error_code="BRANCH_PROCESS_FAILED",
                message=str(error) or type(error).__name__,
                retryable=True,
                input_refs=(task.direction_ref,),
                artifact_refs=(task.config_snapshot_ref,),
            ),
        )
        return self._objects.put_object(
            "branch_outcome",
            _branch_outcome_id(outcome),
            outcome,
        )


class RuntimeBuilder:
    """Construct formal runtime components from immutable configuration only."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        run_id: str,
        dependencies: RuntimeDependencies | None = None,
    ) -> None:
        """Create a builder without creating files or contacting services."""
        if not isinstance(config, ExperimentConfig):
            raise CoreValidationError("config must be ExperimentConfig")
        if type(run_id) is not str or not run_id:
            raise CoreValidationError("run_id must be a non-empty string")
        self._config = config
        self._run_id = run_id
        self._dependencies = dependencies or RuntimeDependencies()

    def build(self, *, resume: bool) -> CosmosAlphaRuntime:
        """Initialize configured local repositories and assemble the chain.

        Callers must run preflight first. This method creates only configured local
        storage; provider loading and LLM requests remain deferred to ``run``.
        """
        report = run_preflight(self._config, run_id=self._run_id, resume=resume)
        if not report.ready:
            codes = ", ".join(issue.code for issue in report.issues)
            raise FormalRuntimeError(f"RUNTIME_PREFLIGHT_FAILED: {codes}")
        if (
            self._config.architecture.name == "parallel_branches"
            and self._dependencies.llm_client is not None
        ):
            raise FormalRuntimeError("PARALLEL_BRANCHES_LLM_OVERRIDE_UNSUPPORTED")
        clock = self._dependencies.clock or _utc_now
        layout = RunStorageLayout.resolve(
            self._config.storage.root,
            self._run_id,
        )
        artifacts = LocalArtifactRepository(
            layout.run_root,
            clock=clock,
        )
        objects = LocalExperimentRepository(
            layout.run_root,
            clock=clock,
        )
        metadata = SqliteWorkflowMetadataRepository.initialize(
            layout.workflow_db_path,
            object_verifier=objects,
            artifact_verifier=artifacts,
        )
        factor_library = SqliteFactorLibrary.initialize(
            layout.factor_library_db_path,
            object_verifier=objects,
            artifact_verifier=artifacts,
        )
        config_snapshot = create_config_snapshot(self._config)
        objects.put_object(
            "config_snapshot",
            config_snapshot.snapshot_id,
            config_snapshot,
        )
        config_snapshot_ref = artifacts.put_bytes(
            config_snapshot.redacted_json.encode("utf-8"),
            kind="config_snapshot",
            media_type="application/json",
        )
        graph_config: GraphConfig | None = None
        graph_loop_factory: Callable[[], CosmosAlphaLoop] | None = None
        graph_initial_loop_factory: Callable[[], CosmosAlphaLoop] | None = None
        if self._config.architecture.name in {"chain", "mcts"}:
            loop, executor = _build_loop(
                self._config,
                artifacts=artifacts,
                objects=objects,
                metadata=metadata,
                factor_library=factor_library,
                publish_to_library=True,
                llm_override=self._dependencies.llm_client,
                clock=clock,
                mcts_mode=self._config.architecture.name == "mcts",
            )
        elif self._config.architecture.name == "graph":
            graph_config = self._dependencies.graph_config
            if graph_config is None:
                if self._config.graph is None:
                    raise FormalRuntimeError("GRAPH_CONFIG_REQUIRED")
                try:
                    graph_config = from_json_value(self._config.graph, GraphConfig)
                except Exception as error:
                    raise FormalRuntimeError("GRAPH_CONFIG_INVALID") from error
            if (
                self._config.evaluation.acceptance_metric != "rank_ic_mean"
                or not self._config.evaluation.acceptance_use_absolute_value
                or not math.isclose(
                    self._config.evaluation.acceptance_threshold,
                    graph_config.average_distance.minimum_abs_rank_ic,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise FormalRuntimeError("GRAPH_ACCEPTANCE_GATE_MISMATCH")

            factor_calculation_cache = FactorCalculationCache()
            forward_return_calculator = ForwardReturnCalculator(artifacts)
            evaluation_input_cache = EvaluationInputCache()

            def build_graph_loop(
                *,
                parent_grounded: bool,
                generation_max_proposals: int | None = None,
            ) -> CosmosAlphaLoop:
                attempt_artifacts = LocalArtifactRepository(
                    layout.run_root,
                    clock=clock,
                )
                attempt_objects = LocalExperimentRepository(
                    layout.run_root,
                    clock=clock,
                )
                attempt_metadata = SqliteWorkflowMetadataRepository.initialize(
                    layout.workflow_db_path,
                    object_verifier=attempt_objects,
                    artifact_verifier=attempt_artifacts,
                )
                attempt_library = SqliteFactorLibrary.initialize(
                    layout.factor_library_db_path,
                    object_verifier=attempt_objects,
                    artifact_verifier=attempt_artifacts,
                )
                attempt_loop, _ = _build_loop(
                    self._config,
                    artifacts=attempt_artifacts,
                    objects=attempt_objects,
                    metadata=attempt_metadata,
                    factor_library=attempt_library,
                    publish_to_library=True,
                    llm_override=self._dependencies.llm_client,
                    clock=clock,
                    graph_mode=parent_grounded,
                    graph_feedback=True,
                    generation_max_proposals=generation_max_proposals,
                    factor_calculation_cache=factor_calculation_cache,
                    forward_return_calculator=forward_return_calculator,
                    evaluation_input_cache=evaluation_input_cache,
                )
                return attempt_loop

            def graph_loop_factory() -> CosmosAlphaLoop:
                return build_graph_loop(parent_grounded=True)

            def graph_initial_loop_factory() -> CosmosAlphaLoop:
                return build_graph_loop(
                    parent_grounded=False,
                    generation_max_proposals=1,
                )

            loop, executor = None, None
        else:
            loop, executor = None, None
        provider = self._create_data_provider(artifacts)
        return CosmosAlphaRuntime(
            config=self._config,
            run_id=self._run_id,
            artifacts=artifacts,
            objects=objects,
            layout=layout,
            factor_library=factor_library,
            data_provider=provider,
            loop=loop,
            graph_config=graph_config,
            graph_loop_factory=graph_loop_factory,
            graph_initial_loop_factory=graph_initial_loop_factory,
            high_quality_factor_pool=self._dependencies.high_quality_factor_pool,
            config_snapshot_ref=config_snapshot_ref,
            llm_executor=executor,
            clock=self._dependencies.clock or _utc_now,
        )

    def _create_data_provider(
        self, artifacts: LocalArtifactRepository
    ) -> MarketDataProvider:
        if self._dependencies.market_data_provider_factory is not None:
            return self._dependencies.market_data_provider_factory(
                artifacts, self._config
            )
        if self._config.data.provider == "mock":
            return FixedSampleMarketDataProvider(
                artifacts,
                provider_alias="runtime_replay",
                provider_uri_fingerprint="runtime-replay-v1",
                trading_calendar_version=self._config.data.trading_calendar_version,
                instruments=_RUNTIME_REPLAY_INSTRUMENTS,
            )
        if self._config.data.provider == "qlib":
            return QlibMarketDataProvider(
                artifacts,
                provider_uri=self._config.qlib.market.provider_uri,
                trading_calendar_version=self._config.data.trading_calendar_version,
                exposure_source=_qlib_source(self._config.qlib.exposure),
                mask_source=_qlib_source(self._config.qlib.mask),
                instruments=self._config.qlib.instruments or None,
                region=self._config.qlib.region,
                frame_loader=self._dependencies.qlib_frame_loader,
            )
        raise FormalRuntimeError(
            f"RUNTIME_DATA_PROVIDER_UNSUPPORTED: {self._config.data.provider}"
        )


def _build_loop(
    config: ExperimentConfig,
    *,
    artifacts: LocalArtifactRepository,
    objects: LocalExperimentRepository,
    metadata: SqliteWorkflowMetadataRepository,
    factor_library: SqliteFactorLibrary,
    publish_to_library: bool,
    llm_override: LLMClient | None,
    clock: Callable[[], datetime],
    reference_factor_values_provider: ReferenceFactorValuesProvider | None = None,
    mcts_mode: bool = False,
    graph_mode: bool = False,
    graph_feedback: bool = False,
    generation_max_proposals: int | None = None,
    factor_calculation_cache: FactorCalculationCache | None = None,
    forward_return_calculator: ForwardReturnCalculator | None = None,
    evaluation_input_cache: EvaluationInputCache | None = None,
) -> tuple[CosmosAlphaLoop, AuditedLLMExecutor | None]:
    """Build one process-local workflow and its optional audited LLM executor."""
    shared_evaluation_inputs = evaluation_input_cache or EvaluationInputCache()
    client, executor = _llm_runtime(
        config,
        artifacts=artifacts,
        objects=objects,
        override=llm_override,
    )
    registry = PromptTemplateRegistry.from_manifest(
        config.prompts.manifest,
    )
    prompt_view_builder = PromptViewBuilder(objects)
    if mcts_mode and graph_mode:
        raise FormalRuntimeError("MCTS_AND_GRAPH_PROMPT_MODES_CONFLICT")
    if mcts_mode and graph_feedback:
        raise FormalRuntimeError("MCTS_AND_GRAPH_FEEDBACK_MODES_CONFLICT")
    factor_design_prompt_id = (
        "mcts_factor_design"
        if mcts_mode
        else "graph_factor_design" if graph_mode else "factor_design"
    )
    # Graph parent-grounded design has a stage-specific budget in the prompt
    # manifest; all other pipelines remain bounded by the global LLM setting.
    factor_design_max_output_tokens = (
        max(
            config.llm.max_output_tokens,
            registry.get(factor_design_prompt_id).max_output_tokens,
        )
        if graph_mode
        else config.llm.max_output_tokens
    )
    loop = CosmosAlphaLoop(
        hypothesis_agent=(
            MCTSHypothesisStub()
            if mcts_mode
            else HypothesisAgent(
                client,
                registry=registry,
                model_alias=config.llm.model_alias,
                max_input_tokens=config.prompts.max_input_tokens,
                max_output_tokens=config.llm.max_output_tokens,
                executor=executor,
                prompt_view_builder=prompt_view_builder,
            )
        ),
        factor_design_agent=FactorDesignAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=factor_design_max_output_tokens,
            executor=executor,
            prompt_id=factor_design_prompt_id,
        ),
        factor_expression_agent=FactorExpressionAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=executor,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=executor,
        ),
        feedback_agent=(
            MCTSFeedbackStub()
            if mcts_mode
            else (
                GraphFeedbackStub()
                if graph_feedback
                else FeedbackAgent(
                    client,
                    prompt_view_builder=FeedbackPromptViewBuilder(objects),
                    registry=registry,
                    model_alias=config.llm.model_alias,
                    max_input_tokens=config.prompts.max_input_tokens,
                    max_output_tokens=config.llm.max_output_tokens,
                    executor=executor,
                )
            )
        ),
        artifact_repository=artifacts,
        experiment_repository=objects,
        workflow_metadata_repository=metadata,
        factor_calculator=FactorCalculator(
            artifacts,
            calculation_cache=factor_calculation_cache,
        ),
        forward_return_calculator=forward_return_calculator,
        preprocessor=GplearnNeutralizedPreprocessor(
            artifacts,
            min_cross_section_samples=_cross_section_minimum(config),
            input_cache=shared_evaluation_inputs,
        ),
        evaluator=FactorEvaluator(
            artifacts,
            min_cross_section_samples=_cross_section_minimum(config),
            min_daily_effective_stock_ratio=(
                config.evaluation.min_daily_effective_stock_ratio
            ),
            reference_factor_values_provider=(
                factor_library
                if reference_factor_values_provider is None
                else reference_factor_values_provider
            ),
            input_cache=shared_evaluation_inputs,
        ),
        acceptance_policy=AcceptancePolicy(
            metric_name=config.evaluation.acceptance_metric,
            threshold=config.evaluation.acceptance_threshold,
            min_effective_sample_count=(config.evaluation.min_effective_sample_count),
            min_effective_date_ratio=config.evaluation.min_effective_date_ratio,
            use_absolute_value=config.evaluation.acceptance_use_absolute_value,
        ),
        factor_library=factor_library if publish_to_library else None,
        generation_constraints=FactorDesignConstraints(
            max_designs=(
                config.factor_generation.max_proposals
                if generation_max_proposals is None
                else generation_max_proposals
            )
        ),
        max_validation_repair_attempts=(
            config.factor_generation.max_validation_repair_attempts
        ),
        clock=clock,
    )
    return loop, executor


def _run_parallel_branch_worker(task: _BranchTask) -> ObjectRef:
    """Run one direction in a fresh spawned process and return only its result ref."""
    layout = RunStorageLayout.resolve(task.config.storage.root, task.run_id)
    artifacts = LocalArtifactRepository(layout.run_root)
    objects = LocalExperimentRepository(layout.run_root)
    metadata = SqliteWorkflowMetadataRepository.initialize(
        layout.workflow_db_path,
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    factor_library = SqliteFactorLibrary.initialize(
        layout.factor_library_db_path,
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    loop, executor = _build_loop(
        task.config,
        artifacts=artifacts,
        objects=objects,
        metadata=metadata,
        factor_library=factor_library,
        publish_to_library=False,
        reference_factor_values_provider=_InitialFactorLibrarySnapshot(
            data_snapshot_id=task.data_snapshot.snapshot_id,
            entries=task.reference_factor_values,
        ),
        llm_override=None,
        clock=_utc_now,
    )
    branch_context = replace(
        task.root_context,
        research_direction=task.direction,
        research_direction_ref=task.direction_ref,
        schema_version="research-context/1.2",
    )
    branch_id = _branch_id(task.branch_index)
    try:
        result = MultiCycleController(
            loop,
            ContextBuilder(
                objects,
            ),
            workflow_config=task.config.workflow,
            prompts_config=task.config.prompts,
        ).run(
            run_id=task.run_id,
            cycle_id_prefix=f"{branch_id}-cycle",
            base_context=branch_context,
            run_context=task.root_context,
            data_snapshot=task.data_snapshot,
            config_snapshot_ref=task.config_snapshot_ref,
            code_version=__version__,
            complete_run=False,
            resume=task.resume,
        )
        experiment_refs = tuple(
            cycle.experiment_ref
            for cycle in result.cycle_results
            if cycle.experiment_ref is not None
        )
        outcome = BranchOutcome(
            run_id=task.run_id,
            branch_id=branch_id,
            branch_index=task.branch_index,
            direction_ref=task.direction_ref,
            status=BranchStatus.COMPLETED,
            cycle_ids=tuple(cycle.cycle_id for cycle in result.cycle_results),
            experiment_refs=experiment_refs,
            terminal_experiment_ref=experiment_refs[-1],
            failure=None,
            prompt_render_refs=_executor_prompt_refs(executor),
            llm_call_record_refs=_executor_call_refs(executor),
            llm_raw_response_refs=_executor_raw_refs(executor),
            schema_version="branch-outcome/1.0",
        )
    except Exception as error:
        experiment_refs = _branch_experiment_refs(objects, task)
        failed_cycle_index = min(
            len(experiment_refs) + 1,
            task.config.workflow.max_cycles,
        )
        failed_cycle_id = f"{branch_id}-cycle-{failed_cycle_index:03d}"
        metadata_failure: str | None = None
        try:
            metadata.fail_cycle(
                task.run_id,
                failed_cycle_id,
                completed_at=_utc_now(),
            )
        except WorkflowMetadataError as metadata_error:
            if "WORKFLOW_CYCLE_NOT_FOUND" not in str(metadata_error):
                metadata_failure = str(metadata_error)
        message = str(error) or type(error).__name__
        if metadata_failure is not None:
            message += f"; cycle failure persistence failed: {metadata_failure}"
        outcome = _failed_branch_outcome(
            task,
            experiment_refs,
            FailureInfo(
                stage=failed_cycle_id,
                error_code="BRANCH_EXECUTION_FAILED",
                message=message,
                retryable=False,
                input_refs=(task.direction_ref,),
                artifact_refs=(task.config_snapshot_ref,),
            ),
            executor=executor,
        )
    return objects.put_object(
        "branch_outcome",
        _branch_outcome_id(outcome),
        outcome,
    )


def _branch_experiment_refs(
    objects: LocalExperimentRepository,
    task: _BranchTask,
) -> tuple[ObjectRef, ...]:
    prefix = f"{_branch_id(task.branch_index)}-cycle-"
    records = sorted(
        (
            record
            for record in objects.list_experiments()
            if record.run_id == task.run_id and record.cycle_id.startswith(prefix)
        ),
        key=lambda record: record.cycle_id,
    )
    return tuple(
        ObjectRef("experiment_record", record.experiment_id, record.schema_version)
        for record in records
    )


def _failed_branch_outcome(
    task: _BranchTask,
    experiment_refs: tuple[ObjectRef, ...],
    failure: FailureInfo,
    *,
    executor: AuditedLLMExecutor | None = None,
) -> BranchOutcome:
    return BranchOutcome(
        run_id=task.run_id,
        branch_id=_branch_id(task.branch_index),
        branch_index=task.branch_index,
        direction_ref=task.direction_ref,
        status=BranchStatus.FAILED,
        cycle_ids=tuple(
            f"{_branch_id(task.branch_index)}-cycle-{index:03d}"
            for index in range(1, len(experiment_refs) + 1)
        ),
        experiment_refs=experiment_refs,
        terminal_experiment_ref=None,
        failure=failure,
        prompt_render_refs=_executor_prompt_refs(executor),
        llm_call_record_refs=_executor_call_refs(executor),
        llm_raw_response_refs=_executor_raw_refs(executor),
        schema_version="branch-outcome/1.0",
    )


def _executor_prompt_refs(
    executor: AuditedLLMExecutor | None,
) -> tuple[ObjectRef, ...]:
    if executor is None:
        return ()
    return tuple(
        sorted(
            executor.prompt_refs.values(),
            key=lambda ref: (ref.object_type, ref.object_id),
        )
    )


def _executor_call_refs(
    executor: AuditedLLMExecutor | None,
) -> tuple[ObjectRef, ...]:
    return () if executor is None else tuple(executor.call_record_refs)


def _executor_raw_refs(
    executor: AuditedLLMExecutor | None,
) -> tuple[ArtifactRef, ...]:
    if executor is None:
        return ()
    return tuple(record.raw_response_ref for record in executor.recorder.records)


def _run_experiment_records(
    objects: LocalExperimentRepository,
    run_id: str,
) -> tuple[ExperimentRecord, ...]:
    """Return all completed experiment records for one run in cycle order."""
    return tuple(
        sorted(
            (
                record
                for record in objects.list_experiments()
                if record.run_id == run_id and record.completed_at is not None
            ),
            key=lambda record: record.cycle_id,
        )
    )


def _publish_completed_parallel_cycles(
    run_id: str,
    *,
    objects: LocalExperimentRepository,
    library: SqliteFactorLibrary,
    clock: Callable[[], datetime],
    published_experiment_ids: set[str],
    published_factor_ids: set[str],
) -> tuple[tuple[ObjectRef, ...], tuple[ExperimentRecord, ...]]:
    """Publish newly completed cycles and return their library object refs."""
    records = _run_experiment_records(objects, run_id)
    new_records = tuple(
        record
        for record in records
        if record.experiment_id not in published_experiment_ids
    )
    if not new_records:
        return (), ()
    refs = _publish_accepted_experiments(
        tuple(
            (_branch_index_from_cycle_id(record.cycle_id), record)
            for record in new_records
        ),
        objects=objects,
        library=library,
        clock=clock,
        published_factor_ids=published_factor_ids,
    )
    published_experiment_ids.update(record.experiment_id for record in new_records)
    return refs, new_records


def _publish_accepted_experiments(
    records: tuple[tuple[int, ExperimentRecord], ...],
    *,
    objects: LocalExperimentRepository,
    library: SqliteFactorLibrary,
    clock: Callable[[], datetime],
    published_factor_ids: set[str],
) -> tuple[ObjectRef, ...]:
    """Publish accepted candidates from completed records in stable order."""
    candidates: list[
        tuple[
            str,
            int,
            str,
            ObjectRef,
            FactorSpec,
            ObjectRef,
            EvaluationResult,
            ObjectRef,
            AcceptanceDecision,
            str,
        ]
    ] = []
    for branch_index, record in records:
        factors = tuple(
            (ref, objects.get_object(ref, FactorSpec)) for ref in record.factor_refs
        )
        for decision_ref in record.acceptance_decision_refs:
            decision = objects.get_object(decision_ref, AcceptanceDecision)
            if not decision.accepted:
                continue
            evaluation_ref = decision.evaluation_result_ref
            evaluation = objects.get_object(evaluation_ref, EvaluationResult)
            matching = tuple(
                (ref, factor)
                for ref, factor in factors
                if factor.factor_id == decision.factor_id
            )
            if not matching:
                raise FormalRuntimeError("PARENT_PUBLICATION_FACTOR_REFERENCE_MISSING")
            factor_ref, factor = min(
                matching,
                key=lambda item: item[0].object_id,
            )
            candidates.append(
                (
                    factor.factor_id,
                    branch_index,
                    record.cycle_id,
                    factor_ref,
                    factor,
                    evaluation_ref,
                    evaluation,
                    decision_ref,
                    decision,
                    record.experiment_id,
                )
            )
    published: list[ObjectRef] = []
    for (
        factor_id,
        _branch_index,
        _cycle_id,
        factor_ref,
        factor,
        evaluation_ref,
        evaluation,
        decision_ref,
        decision,
        source_experiment_id,
    ) in sorted(
        candidates,
        key=lambda item: (item[1], item[2], item[0], item[3].object_id),
    ):
        if factor_id in published_factor_ids:
            continue
        entry = library.get_entry(factor_id)
        if entry is None:
            entry = library.publish_accepted(
                factor_spec=factor,
                factor_ref=factor_ref,
                evaluation_result=evaluation,
                evaluation_result_ref=evaluation_ref,
                decision=decision,
                decision_ref=decision_ref,
                source_experiment_id=source_experiment_id,
                accepted_at=clock(),
            )
        published.append(
            objects.put_object(
                FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
                entry.factor_id,
                entry,
            )
        )
        published_factor_ids.add(factor_id)
    return tuple(published)


def _branch_index_from_cycle_id(cycle_id: str) -> int:
    """Extract the stable numeric branch index from a parallel cycle ID."""
    prefix, separator, _cycle_number = cycle_id.partition("-cycle-")
    if not separator or not prefix.startswith("branch-"):
        raise FormalRuntimeError("PARALLEL_CYCLE_ID_INVALID")
    try:
        return int(prefix.removeprefix("branch-"))
    except ValueError as error:
        raise FormalRuntimeError("PARALLEL_CYCLE_ID_INVALID") from error


def _branch_id(index: int) -> str:
    return f"branch-{index:03d}"


def _branch_outcome_id(outcome: BranchOutcome) -> str:
    return "branch-outcome-" + stable_sha256(outcome)[:24]


def _branch_outcome_ref(outcome: BranchOutcome) -> ObjectRef:
    return ObjectRef(
        "branch_outcome",
        _branch_outcome_id(outcome),
        outcome.schema_version,
    )


def run_formal_runtime(
    config: ExperimentConfig,
    *,
    run_id: str,
    resume: bool,
    dependencies: RuntimeDependencies | None = None,
) -> FormalRuntimeResult:
    """Build and execute one formal runtime invocation from strict configuration."""
    return (
        RuntimeBuilder(
            config,
            run_id=run_id,
            dependencies=dependencies,
        )
        .build(resume=resume)
        .run(resume=resume)
    )


def _market_data_request(config: ExperimentConfig) -> MarketDataRequest:
    if config.data.provider == "mock":
        return MarketDataRequest(
            market=config.data.market,
            universe=config.universe.name,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            warmup_start_date=config.data.warmup_start_date,
            frequency=config.data.frequency,
            fields=MARKET_DATA_FIELDS,
            dataset_version=config.data.dataset_version,
            price_adjustment=config.data.price_adjustment,
            schema_version="market-data/1.0",
        )
    return MarketDataRequest(
        market=config.data.market,
        universe=config.universe.name,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        warmup_start_date=config.data.warmup_start_date,
        frequency=config.data.frequency,
        fields=tuple(field.field_name for field in config.qlib.market.fields),
        dataset_version=config.data.dataset_version,
        price_adjustment=config.data.price_adjustment,
        schema_version="market-data/1.0",
    )


def _base_context(config: ExperimentConfig, snapshot: DataSnapshot) -> ResearchContext:
    return ResearchContext(
        research_goal=(
            "Discover one deterministic daily price-volume factor under the "
            "configured evaluation protocol."
        ),
        market=snapshot.market,
        universe=snapshot.request.universe,
        frequency=snapshot.frequency,
        available_fields=snapshot.market_data.fields,
        operator_catalog_ref=ObjectRef(
            "operator_catalog",
            config.operators.operator_set.lower(),
            "operator-catalog/1.0",
        ),
        evaluation_protocol_ref=ObjectRef(
            "evaluation_protocol",
            config.evaluation.protocol,
            "evaluation-protocol/1.0",
        ),
        history_summaries=(),
        constraints=(
            "Use only supplied fields and registered operators.",
            "Do not claim deterministic validation or acceptance in the draft.",
        ),
        research_direction=None,
        research_direction_ref=None,
        schema_version="research-context/1.2",
    )


def _llm_runtime(
    config: ExperimentConfig,
    *,
    artifacts: LocalArtifactRepository,
    objects: LocalExperimentRepository,
    override: LLMClient | None,
) -> tuple[LLMClient, AuditedLLMExecutor | None]:
    if config.llm.replay_only:
        if override is not None:
            return override, None
        return (
            ReplayLLMClient(_replay_responses(), model_alias=config.llm.model_alias),
            None,
        )
    client = override or OpenAICompatibleLLMClient.from_config(config.llm)
    executor = AuditedLLMExecutor(
        client=client,
        recorder=LLMCallRecorder(artifacts),
        object_repository=objects,
        retry_policy=LLMRetryPolicy.from_config(config.llm, config.prompts),
    )
    return client, executor


def _snapshot_evaluation_dates(
    artifacts: LocalArtifactRepository,
    snapshot: DataSnapshot,
) -> tuple[str, ...]:
    """Read the deterministic evaluation calendar from the market artifact."""
    records = decode_record_artifact(artifacts.get_bytes(snapshot.market_data.data_ref))
    dates = tuple(
        sorted(
            {
                record["datetime"]
                for record in records
                if type(record.get("datetime")) is str
            }
        )
    )
    if not dates:
        raise FormalRuntimeError("GRAPH_EVALUATION_CALENDAR_EMPTY")
    return dates


def _graph_correlation_protocol(
    config: ExperimentConfig,
    snapshot: DataSnapshot,
    evaluation_dates: tuple[str, ...],
) -> GraphCorrelationProtocol:
    """Derive the run-bound correlation protocol from explicit core settings."""
    minimum_ratio = config.evaluation.min_effective_date_ratio
    return GraphCorrelationProtocol(
        protocol_id="graph-correlation/daily-cross-sectional-pearson/v1",
        data_snapshot_id=snapshot.snapshot_id,
        preprocessing_protocol_id=GPLEARN_NEUTRALIZED_PROTOCOL_ID,
        evaluation_protocol_id=EVALUATION_PROTOCOL_ID,
        min_cross_section_samples=config.evaluation.min_cross_section_samples,
        min_daily_effective_stock_ratio=(
            config.evaluation.min_daily_effective_stock_ratio
        ),
        min_pair_window_ratio=minimum_ratio,
        min_valid_date_count=max(1, math.ceil(len(evaluation_dates) * minimum_ratio)),
        min_valid_date_ratio=minimum_ratio,
        use_pairwise_finite=True,
        daily_weighting="equal",
        similarity_use_abs=True,
        exposure_source="evaluated_factor_values",
        calculation_dtype="float64",
        store_dtype="float32",
    )


def _qlib_source(source: QlibSourceConfig) -> QlibFeatureSource:
    return QlibFeatureSource(
        provider_uri=source.provider_uri,
        fields=tuple(
            QlibFieldMapping(field.field_name, field.expression)
            for field in source.fields
        ),
    )


def _cross_section_minimum(config: ExperimentConfig) -> int:
    return config.evaluation.min_cross_section_samples


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _replay_responses() -> dict[str, str]:
    responses = {
        "hypothesis": to_json(
            {
                "assumptions": ["The configured replay snapshot is deterministic."],
                "expected_direction": "positive",
                "mechanism": "Cross-sectional volume-to-price pressure varies daily.",
                "prediction_horizon": "1d",
                "rationale": "Use a bounded deterministic replay proposal.",
                "required_fields": ["close", "volume"],
                "statement": "Relative volume per unit price may predict returns.",
                "testable_prediction": "rank(div(volume, close)) has finite metrics.",
            }
        ),
        "factor": to_json(
            {
                "proposals": [
                    {
                        "description": "Cross-sectional rank of volume divided by close.",
                        "expected_direction": "positive",
                        "name": "runtime_ranked_volume_price",
                        "proposed_expression": "rank(div(volume, close))",
                    }
                ]
            }
        ),
        "factor_design": to_json(
            {
                "designs": [
                    {
                        "name": "runtime_ranked_volume_price",
                        "description": (
                            "Cross-sectional rank of volume divided by close."
                        ),
                        "mechanism_mapping": (
                            "High activity per price unit proxies trading pressure."
                        ),
                        "signal_definition": (
                            "Rank volume divided by close across instruments."
                        ),
                        "variables": [
                            {"name": "volume", "role": "activity numerator"},
                            {"name": "close", "role": "price denominator"},
                        ],
                        "formulation": "cross-sectional rank(volume / close)",
                        "window_choices": [],
                        "expected_direction": "positive",
                        "failure_conditions": [
                            "The signal is dominated by price-level effects."
                        ],
                    }
                ]
            }
        ),
        "factor_expression": to_json(
            {"proposed_expression": "rank(div(volume, close))"}
        ),
        "expression_repair": to_json(
            {"proposed_expression": "rank(div(volume, close))"}
        ),
        "feedback": to_json(
            {
                "failure_analysis": [],
                "hypothesis_verdict": "supported",
                "next_actions": ["Use the persisted evaluation as the next baseline."],
                "next_hypothesis_hint": "Test a distinct registered operator next.",
                "rejected_findings": [],
                "summary": "The configured runtime chain completed.",
                "supported_findings": ["The runtime published deterministic evidence."],
            }
        ),
    }
    responses["mcts_factor_design"] = responses["factor_design"]
    responses["graph_factor_design"] = responses["factor_design"]
    return responses
