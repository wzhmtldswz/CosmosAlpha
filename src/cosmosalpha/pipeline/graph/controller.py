"""Graph Search v1 orchestration over process-local CosmosAlpha loops."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import Condition
from time import perf_counter
from typing import Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    ExperimentStatus,
    FactorDesignDraft,
    FactorProposal,
    FactorSpec,
    ResearchContext,
)
from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    FailureInfo,
    ObjectRef,
    stable_sha256,
    to_json,
)
from cosmosalpha.data import DataSnapshot
from cosmosalpha.evaluation import (
    EVALUATOR_VERSION,
    CorrelationFactorRef,
    CorrelationStore,
    GraphCorrelationEngine,
    GraphCorrelationProtocol,
    GraphNodeEligibilityGate,
    ReferenceFactorValuesProvider,
    factor_orientation,
)
from cosmosalpha.pipeline.graph.audit import (
    GraphCandidateOutcome,
    GraphCandidateOutcomeStatus,
    load_graph_candidate_outcomes,
    persist_graph_candidate_outcome,
)
from cosmosalpha.pipeline.graph.config import GraphConfig
from cosmosalpha.pipeline.graph.core import (
    AverageDistanceReferenceMode,
    AverageDistanceReferenceSnapshot,
    ExpansionAttempt,
    ExpansionAttemptStatus,
    ExternalReferenceFactor,
    ExternalReferencePoolSnapshot,
    FactorGraph,
    GenerationEvent,
    GraphNode,
    QualityParetoArchive,
)
from cosmosalpha.pipeline.graph.diagnostics import (
    GraphDiagnostics,
    build_graph_diagnostics,
)
from cosmosalpha.pipeline.graph.flow import GraphFlowResult, run_graph_flow
from cosmosalpha.pipeline.graph.generation import build_generation_graph
from cosmosalpha.pipeline.graph.identity import (
    DuplicateIndex,
    FactorLibraryCanonicalKeySnapshot,
    GraphDuplicateKey,
)
from cosmosalpha.pipeline.graph.persistence import (
    GraphPersistence,
    graph_attempt_id,
    graph_initial_cycle_id,
    graph_initial_cycle_index,
)
from cosmosalpha.pipeline.graph.reward import (
    attempt_fertility_outcome,
    average_distance_from_config,
    batch_archive_commit,
    batch_birth_time_delta_hv,
    fertility_from_attempts,
    pareto_archive,
    quality_point_from_node,
    self_quality_from_node,
)
from cosmosalpha.pipeline.graph.seeds import (
    GraphSeedCandidate,
)
from cosmosalpha.pipeline.graph.selector import (
    FlowGuidedGraphSelector,
    ParentSelectionResult,
    crowding_distances,
    pareto_fronts,
)
from cosmosalpha.pipeline.graph.selector_audit import (
    GraphSelectorAudit,
    persist_graph_selector_audit,
)
from cosmosalpha.pipeline.graph.structure import (
    StructureCanonicalForm,
    build_structure_fact_store,
    project_structure_neighborhood,
)
from cosmosalpha.pipeline.graph.telemetry import GraphTelemetryRecorder
from cosmosalpha.pipeline.loop import (
    CandidateCalculationBatchGate,
    CandidateCalculationDecision,
    CandidateCalculationInput,
    CosmosAlphaLoop,
    CosmosAlphaLoopResult,
    classify_attempt_failure,
)

GRAPH_RESULT_SCHEMA_VERSION = "graph-result/2.0"
GRAPH_GUIDANCE_SCHEMA_VERSION = "graph-expansion-guidance/2.0"
GRAPH_GUIDANCE_OBJECT_TYPE = "graph_expansion_guidance"
GRAPH_RESULT_OBJECT_TYPE = "graph_result"
GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION = (
    "graph-factor-library-reference-snapshot/1.0"
)
GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_OBJECT_TYPE = (
    "graph_factor_library_reference_snapshot"
)


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_finite(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CoreValidationError(f"{field_name} must be finite")
    return float(value)


def graph_factor_library_reference_snapshot_id(run_id: str, round_id: int) -> str:
    """Return the deterministic object ID for one Graph round reference view."""
    _require_text(run_id, "run_id")
    if type(round_id) is not int or round_id < 0:
        raise CoreValidationError("round_id must be non-negative")
    return (
        "graph-factor-library-reference-"
        + stable_sha256({"run_id": run_id, "round_id": round_id})[:24]
    )


@dataclass(frozen=True, slots=True)
class GraphFactorLibraryReferenceSnapshot:
    """Immutable accepted-factor value references captured for one Graph round."""

    run_id: str
    round_id: int
    data_snapshot_id: str
    entries: tuple[tuple[str, ArtifactRef], ...]
    schema_version: str = GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the compact, deterministically ordered reference payload."""
        _require_text(self.run_id, "run_id")
        if type(self.round_id) is not int or self.round_id < 0:
            raise CoreValidationError("round_id must be non-negative")
        _require_text(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.entries, tuple):
            raise CoreValidationError("entries must be a tuple")
        factor_ids: list[str] = []
        for entry in self.entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise CoreValidationError(
                    "entries must contain factor_id/ArtifactRef pairs"
                )
            factor_id, factor_values_ref = entry
            _require_text(factor_id, "entries.factor_id")
            if not isinstance(factor_values_ref, ArtifactRef):
                raise CoreValidationError(
                    "entries.factor_values_ref must be ArtifactRef"
                )
            factor_ids.append(factor_id)
        if len(factor_ids) != len(set(factor_ids)):
            raise CoreValidationError("entries must contain unique factor IDs")
        if tuple(factor_ids) != tuple(sorted(factor_ids)):
            raise CoreValidationError("entries must be sorted by factor ID")
        if (
            self.schema_version
            != GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION
        ):
            raise CoreValidationError(
                "invalid GraphFactorLibraryReferenceSnapshot schema_version"
            )

    def reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Return the captured peer references, never reading live library state."""
        if data_snapshot_id != self.data_snapshot_id:
            return ()
        return tuple(
            factor_values_ref
            for factor_id, factor_values_ref in self.entries
            if factor_id != exclude_factor_id
        )


@dataclass(frozen=True, slots=True)
class GraphExpansionGuidance:
    """Auditable, parent-grounded mutation guidance for one attempt."""

    parent_id: str
    parent_expression: str
    parent_metrics: tuple[tuple[str, float], ...]
    target_objectives: tuple[str, ...]
    pareto_gaps: tuple[tuple[str, float], ...]
    forbidden_structures: tuple[str, ...]
    correlation_constraints: tuple[tuple[str, float], ...]
    allowed_operations: tuple[str, ...]
    schema_version: str = GRAPH_GUIDANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and deterministically order guidance metadata."""
        _require_text(self.parent_id, "parent_id")
        _require_text(self.parent_expression, "parent_expression")
        for field_name in (
            "target_objectives",
            "forbidden_structures",
            "allowed_operations",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not all(
                type(item) is str and item for item in values
            ):
                raise CoreValidationError(f"{field_name} must be a tuple of strings")
        for field_name in (
            "parent_metrics",
            "pareto_gaps",
            "correlation_constraints",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise CoreValidationError(f"{field_name} must be a tuple")
            for key, value in values:
                _require_text(key, f"{field_name}.key")
                _require_finite(value, f"{field_name}.{key}")
        _require_text(self.schema_version, "schema_version")
        if self.schema_version != GRAPH_GUIDANCE_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphExpansionGuidance schema_version")


@dataclass(frozen=True, slots=True)
class GraphSeed:
    """A protocol-compatible seed and its already-derived duplicate key."""

    node: GraphNode
    duplicate_key: GraphDuplicateKey
    factor_ref: ObjectRef | None = None

    def __post_init__(self) -> None:
        """Validate neutral seed-pool data without importing adapters."""
        if not isinstance(self.node, GraphNode):
            raise CoreValidationError("seed.node must be GraphNode")
        if not isinstance(self.duplicate_key, GraphDuplicateKey):
            raise CoreValidationError("seed.duplicate_key must be GraphDuplicateKey")
        if self.factor_ref is not None and not isinstance(self.factor_ref, ObjectRef):
            raise CoreValidationError("seed.factor_ref must be ObjectRef or None")


class ExistingHighQualityFactorPool(Protocol):
    """Neutral seed source; adapters are intentionally outside Graph v1."""

    def candidates(
        self,
        *,
        data_snapshot_id: str,
        evaluation_protocol_id: str,
        preprocessing_protocol_id: str,
    ) -> tuple[GraphSeed, ...]:
        """Return protocol-compatible candidates without mutating the pool."""


class EmptyExistingHighQualityFactorPool:
    """Explicit empty pool used before historical-factor adapters are frozen."""

    def candidates(
        self,
        *,
        data_snapshot_id: str,
        evaluation_protocol_id: str,
        preprocessing_protocol_id: str,
    ) -> tuple[GraphSeed, ...]:
        """Return no seed candidates."""
        _require_text(data_snapshot_id, "data_snapshot_id")
        _require_text(evaluation_protocol_id, "evaluation_protocol_id")
        _require_text(preprocessing_protocol_id, "preprocessing_protocol_id")
        return ()


@dataclass(frozen=True, slots=True)
class GraphAttemptRequest:
    """Stable input for one complete Graph-to-Loop attempt."""

    attempt_id: str
    round_id: int
    parent_id: str
    parent_factor_ref: ObjectRef | None
    parent_selection_rank: int
    guidance: GraphExpansionGuidance | None
    context: ResearchContext
    data_snapshot: DataSnapshot
    config_snapshot_ref: ArtifactRef
    code_version: str
    run_id: str
    guidance_ref: ObjectRef | None = None

    def __post_init__(self) -> None:
        """Validate deterministic attempt identity and parent boundary."""
        for field_name in ("attempt_id", "parent_id", "code_version", "run_id"):
            _require_text(getattr(self, field_name), field_name)
        if type(self.round_id) is not int or self.round_id < 0:
            raise CoreValidationError("round_id must be non-negative")
        if (
            type(self.parent_selection_rank) is not int
            or self.parent_selection_rank < 0
        ):
            raise CoreValidationError("parent_selection_rank must be non-negative")
        if self.parent_factor_ref is not None and not isinstance(
            self.parent_factor_ref, ObjectRef
        ):
            raise CoreValidationError("parent_factor_ref must be ObjectRef or None")
        if self.parent_factor_ref is None and self.guidance is not None:
            raise CoreValidationError("initial attempt cannot have guidance")
        if self.guidance_ref is not None and not isinstance(
            self.guidance_ref, ObjectRef
        ):
            raise CoreValidationError("guidance_ref must be ObjectRef or None")
        if self.guidance is not None and self.guidance.parent_id != self.parent_id:
            raise CoreValidationError("guidance parent_id must match parent_id")
        if not isinstance(self.context, ResearchContext):
            raise CoreValidationError("context must be ResearchContext")
        if not isinstance(self.data_snapshot, DataSnapshot):
            raise CoreValidationError("data_snapshot must be DataSnapshot")
        if not isinstance(self.config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")


@dataclass(frozen=True, slots=True)
class GraphAttemptExecution:
    """Result of an attempt executor, retaining deterministic request identity."""

    request: GraphAttemptRequest
    loop_result: CosmosAlphaLoopResult | None
    error_code: str | None = None

    def __post_init__(self) -> None:
        """Require an error code only when Loop execution failed."""
        if not isinstance(self.request, GraphAttemptRequest):
            raise CoreValidationError("request must be GraphAttemptRequest")
        if self.loop_result is None:
            _require_text(self.error_code, "error_code")
        elif self.error_code is not None:
            raise CoreValidationError("successful execution cannot have error_code")


@dataclass(slots=True)
class _GraphInitializationState:
    """Mutable state shared by the small initialization fill loops."""

    request: GraphAttemptRequest
    target: int
    nodes: dict[str, GraphNode]
    factor_refs: dict[str, ObjectRef]
    duplicate_index: DuplicateIndex
    cycle_statuses: dict[int, str]
    last_error_code: str | None = None


class GraphAttemptExecutor(Protocol):
    """Execution boundary allowing runtime-specific process orchestration."""

    def run_initial(
        self,
        request: GraphAttemptRequest,
        *,
        committed_index: DuplicateIndex | None = None,
        fixed_factor_design_drafts: tuple[FactorDesignDraft, ...] = (),
    ) -> GraphAttemptExecution:
        """Run an unparented cold start or configured seed evaluation."""

    def run_batch(
        self,
        requests: tuple[GraphAttemptRequest, ...],
        *,
        committed_index: DuplicateIndex,
    ) -> tuple[GraphAttemptExecution, ...]:
        """Run one parent-grounded attempt per request with a batch gate."""


class _BatchDuplicateCoordinator(CandidateCalculationBatchGate):
    """Coordinate Loop gates and make duplicate winners in parent-rank order."""

    def __init__(
        self,
        requests: tuple[GraphAttemptRequest, ...],
        committed_index: DuplicateIndex,
    ) -> None:
        self._attempt_ids = tuple(request.attempt_id for request in requests)
        self._ranks = {
            request.attempt_id: request.parent_selection_rank for request in requests
        }
        self._index = committed_index
        self._condition = Condition()
        self._candidates: dict[str, tuple[CandidateCalculationInput, ...]] = {}
        self._decisions: dict[str, tuple[CandidateCalculationDecision, ...]] = {}
        self._error: Exception | None = None

    def fail(self, attempt_id: str, error: Exception) -> None:
        """Let other attempts proceed when one attempt cannot reach the barrier."""
        del error
        if attempt_id not in self._attempt_ids:
            raise CoreValidationError("unknown Graph attempt id")
        with self._condition:
            self._candidates.setdefault(attempt_id, ())
            if len(self._candidates) == len(self._attempt_ids):
                try:
                    self._decisions = self._compute()
                except Exception as barrier_error:  # noqa: BLE001
                    self._error = barrier_error
            self._condition.notify_all()

    def decide_for(
        self,
        attempt_id: str,
        candidates: tuple[CandidateCalculationInput, ...],
    ) -> tuple[CandidateCalculationDecision, ...]:
        """Wait for all attempts, then apply committed and batch duplicate rules."""
        if attempt_id not in self._attempt_ids:
            raise CoreValidationError("unknown Graph attempt id")
        with self._condition:
            self._candidates[attempt_id] = candidates
            if len(self._candidates) == len(self._attempt_ids):
                try:
                    self._decisions = self._compute()
                except Exception as barrier_error:  # noqa: BLE001
                    self._error = barrier_error
                self._condition.notify_all()
            else:
                self._condition.wait_for(
                    lambda: len(self._candidates) == len(self._attempt_ids)
                    or self._error is not None
                )
            if self._error is not None:
                raise self._error
            return self._decisions[attempt_id]

    def decide(
        self,
        candidates: tuple[CandidateCalculationInput, ...],
    ) -> tuple[CandidateCalculationDecision, ...]:
        """Reject use without an explicit attempt identity."""
        del candidates
        raise CoreValidationError("Graph coordinator requires decide_for")

    def _compute(self) -> dict[str, tuple[CandidateCalculationDecision, ...]]:
        occurrences: dict[
            GraphDuplicateKey, list[tuple[str, CandidateCalculationInput]]
        ] = {}
        decisions: dict[str, list[CandidateCalculationDecision]] = {
            attempt_id: [] for attempt_id in self._attempt_ids
        }
        for attempt_id in self._attempt_ids:
            for candidate in self._candidates[attempt_id]:
                key = GraphDuplicateKey.from_factor_spec(candidate.factor_spec)
                occurrences.setdefault(key, []).append((attempt_id, candidate))
        for attempt_id in self._attempt_ids:
            for candidate in self._candidates[attempt_id]:
                key = GraphDuplicateKey.from_factor_spec(candidate.factor_spec)
                if self._index.contains(key):
                    decision = CandidateCalculationDecision(False, "DUPLICATE_ATTEMPT")
                else:
                    winner = min(
                        occurrences[key],
                        key=lambda item: (
                            self._ranks[item[0]],
                            item[1].candidate_index,
                            item[0],
                        ),
                    )
                    decision = CandidateCalculationDecision(
                        winner == (attempt_id, candidate)
                        or (
                            winner[0] == attempt_id
                            and winner[1].candidate_index == candidate.candidate_index
                        ),
                        None,
                    )
                    if not decision.proceed:
                        decision = CandidateCalculationDecision(
                            False, "DUPLICATE_ATTEMPT"
                        )
                decisions[attempt_id].append(decision)
        return {attempt_id: tuple(values) for attempt_id, values in decisions.items()}


class LocalLoopAttemptExecutor:
    """Run independent Loop instances concurrently in the current process."""

    def __init__(
        self,
        loop_factory: Callable[[], CosmosAlphaLoop],
        *,
        max_workers: int,
        initial_loop_factory: Callable[[], CosmosAlphaLoop] | None = None,
        guidance_persister: Callable[[GraphExpansionGuidance], ObjectRef] | None = None,
        cycle_preparer: Callable[[GraphAttemptRequest], None] | None = None,
        cycle_failure_finalizer: (
            Callable[[GraphAttemptRequest, Exception], None] | None
        ) = None,
        reference_factor_values_provider_factory: (
            Callable[[GraphAttemptRequest], ReferenceFactorValuesProvider] | None
        ) = None,
        initial_run_context: ResearchContext | None = None,
    ) -> None:
        """Bind a factory; no mutable Loop instance is shared between attempts.

        The optional lifecycle callbacks are Graph-runtime adapters.  They do not
        alter the public ``GraphAttemptExecutor`` protocol; they ensure that a
        cycle exists before an attempt starts and that captured exceptions close
        that cycle without replacing the original failure classification.
        """
        if not callable(loop_factory):
            raise CoreValidationError("loop_factory must be callable")
        if type(max_workers) is not int or max_workers < 1:
            raise CoreValidationError("max_workers must be positive")
        self._loop_factory = loop_factory
        self._initial_loop_factory = initial_loop_factory or loop_factory
        self._max_workers = max_workers
        self._guidance_persister = guidance_persister
        self._cycle_preparer = cycle_preparer
        self._cycle_failure_finalizer = cycle_failure_finalizer
        if cycle_preparer is not None and not callable(cycle_preparer):
            raise CoreValidationError("cycle_preparer must be callable or None")
        if cycle_failure_finalizer is not None and not callable(
            cycle_failure_finalizer
        ):
            raise CoreValidationError(
                "cycle_failure_finalizer must be callable or None"
            )
        self._reference_factor_values_provider_factory = (
            reference_factor_values_provider_factory
        )
        if reference_factor_values_provider_factory is not None and not callable(
            reference_factor_values_provider_factory
        ):
            raise CoreValidationError(
                "reference_factor_values_provider_factory must be callable"
            )
        if initial_run_context is not None and not isinstance(
            initial_run_context, ResearchContext
        ):
            raise CoreValidationError(
                "initial_run_context must be ResearchContext or None"
            )
        self._initial_run_context = initial_run_context

    def _prepare_cycle(self, request: GraphAttemptRequest) -> None:
        """Create the workflow cycle before any attempt-local work can fail."""
        if self._cycle_preparer is not None:
            self._cycle_preparer(request)

    def _finalize_cycle_failure(
        self,
        request: GraphAttemptRequest,
        error: Exception,
    ) -> None:
        """Close a failed cycle while preserving the attempt exception as cause."""
        if self._cycle_failure_finalizer is None:
            return
        try:
            self._cycle_failure_finalizer(request, error)
        except Exception as finalizer_error:  # noqa: BLE001 - preserve root cause
            raise finalizer_error from error

    def run_initial(
        self,
        request: GraphAttemptRequest,
        *,
        committed_index: DuplicateIndex | None = None,
        fixed_factor_design_drafts: tuple[FactorDesignDraft, ...] = (),
    ) -> GraphAttemptExecution:
        """Run an initial Loop without exposing a parent factor."""
        try:
            self._prepare_cycle(request)
            loop = self._initial_loop_factory()

            class Gate(CandidateCalculationBatchGate):
                def decide(
                    _self,
                    candidates: tuple[CandidateCalculationInput, ...],
                ) -> tuple[CandidateCalculationDecision, ...]:
                    winner_by_key: dict[GraphDuplicateKey, int] = {}
                    for candidate in candidates:
                        key = GraphDuplicateKey.from_factor_spec(candidate.factor_spec)
                        winner_by_key[key] = min(
                            winner_by_key.get(key, candidate.candidate_index),
                            candidate.candidate_index,
                        )
                    decisions = []
                    for candidate in candidates:
                        key = GraphDuplicateKey.from_factor_spec(candidate.factor_spec)
                        proceed = winner_by_key[key] == candidate.candidate_index
                        if committed_index is not None and committed_index.contains(
                            key
                        ):
                            proceed = False
                        decisions.append(
                            CandidateCalculationDecision(
                                proceed=proceed,
                                reason_code=None if proceed else "DUPLICATE_ATTEMPT",
                            )
                        )
                    return tuple(decisions)

            result = loop.run(
                run_id=request.run_id,
                cycle_id=request.attempt_id,
                context=request.context,
                data_snapshot=request.data_snapshot,
                config_snapshot_ref=request.config_snapshot_ref,
                code_version=request.code_version,
                parent_ids=(),
                run_context=self._initial_run_context or request.context,
                complete_run=False,
                pre_calculation_batch_gate=Gate(),
                resume=True,
                # The immutable Factor Library snapshot is part of the initial
                # duplicate barrier as well as every later Graph round.
                fixed_factor_design_drafts=(fixed_factor_design_drafts or None),
            )
            return GraphAttemptExecution(request, result)
        except Exception as error:  # noqa: BLE001 - structured attempt failure
            error_code = classify_attempt_failure(error)
            self._finalize_cycle_failure(request, error)
            return GraphAttemptExecution(request, None, error_code)

    def run_batch(
        self,
        requests: tuple[GraphAttemptRequest, ...],
        *,
        committed_index: DuplicateIndex,
    ) -> tuple[GraphAttemptExecution, ...]:
        """Execute independent Loops and coordinate their pre-calculation gates."""
        if not requests:
            return ()
        reference_factor_values_provider = None
        if self._reference_factor_values_provider_factory is not None:
            reference_factor_values_provider = (
                self._reference_factor_values_provider_factory(requests[0])
            )
        coordinator = _BatchDuplicateCoordinator(requests, committed_index)

        def execute(request: GraphAttemptRequest) -> GraphAttemptExecution:
            try:
                self._prepare_cycle(request)
                if (
                    request.guidance is not None
                    and self._guidance_persister is not None
                ):
                    self._guidance_persister(request.guidance)
                loop = self._loop_factory()
                if reference_factor_values_provider is not None:
                    loop.set_reference_factor_values_provider(
                        reference_factor_values_provider
                    )
                context = request.context
                if request.guidance is not None:
                    context = replace(
                        context,
                        constraints=(
                            *context.constraints,
                            "GRAPH_GUIDANCE_JSON=" + to_json(request.guidance),
                        ),
                    )

                class Gate(CandidateCalculationBatchGate):
                    def decide(
                        _self,
                        candidates: tuple[CandidateCalculationInput, ...],
                    ) -> tuple[CandidateCalculationDecision, ...]:
                        return coordinator.decide_for(request.attempt_id, candidates)

                result = loop.run(
                    run_id=request.run_id,
                    cycle_id=request.attempt_id,
                    context=context,
                    data_snapshot=request.data_snapshot,
                    config_snapshot_ref=request.config_snapshot_ref,
                    code_version=request.code_version,
                    parent_ids=(request.parent_id,),
                    graph_parent_ref=request.parent_factor_ref,
                    run_context=request.context,
                    complete_run=False,
                    pre_calculation_batch_gate=Gate(),
                    resume=True,
                )
                return GraphAttemptExecution(request, result)
            except Exception as error:  # noqa: BLE001 - structured attempt failure
                try:
                    error_code = classify_attempt_failure(error)
                except Exception:  # noqa: BLE001 - defensive barrier release
                    error_code = "UNEXPECTED_ATTEMPT_FAILURE"
                try:
                    coordinator.fail(request.attempt_id, error)
                except Exception as release_error:  # noqa: BLE001
                    raise release_error from error
                self._finalize_cycle_failure(request, error)
                return GraphAttemptExecution(request, None, error_code)

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(requests))
        ) as pool:
            futures = tuple(pool.submit(execute, request) for request in requests)
            results = tuple(future.result() for future in futures)
        return tuple(
            sorted(results, key=lambda item: item.request.parent_selection_rank)
        )


@dataclass(frozen=True, slots=True)
class GraphResult:
    """Persistable terminal state returned by one Graph Search invocation."""

    run_id: str
    graph: FactorGraph
    attempts: tuple[ExpansionAttempt, ...]
    archive: QualityParetoArchive
    average_distance_reference: AverageDistanceReferenceSnapshot
    flow_result: GraphFlowResult
    diagnostics: GraphDiagnostics
    attempts_used: int
    remaining_budget: int
    termination_reason: str
    schema_version: str = GRAPH_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate terminal Graph state and stable schema identity."""
        _require_text(self.run_id, "run_id")
        if not isinstance(self.graph, FactorGraph):
            raise CoreValidationError("graph must be FactorGraph")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(attempt, ExpansionAttempt) for attempt in self.attempts
        ):
            raise CoreValidationError("attempts must contain ExpansionAttempt values")
        if not isinstance(self.archive, QualityParetoArchive):
            raise CoreValidationError("archive must be QualityParetoArchive")
        if not isinstance(
            self.average_distance_reference, AverageDistanceReferenceSnapshot
        ):
            raise CoreValidationError(
                "average_distance_reference must be AverageDistanceReferenceSnapshot"
            )
        if not isinstance(self.flow_result, GraphFlowResult):
            raise CoreValidationError("flow_result must be GraphFlowResult")
        if not isinstance(self.diagnostics, GraphDiagnostics):
            raise CoreValidationError("diagnostics must be GraphDiagnostics")
        for field_name in ("attempts_used", "remaining_budget"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise CoreValidationError(f"{field_name} must be non-negative")
        _require_text(self.termination_reason, "termination_reason")
        if self.schema_version != GRAPH_RESULT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphResult schema_version")


class GraphController:
    """Coordinate frozen Graph rounds while delegating factor work to Loop."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        attempt_executor: GraphAttemptExecutor,
        artifacts: object,
        objects: object,
        correlation_engine: GraphCorrelationEngine | None,
        correlation_protocol: GraphCorrelationProtocol,
        min_effective_date_ratio: float,
        high_quality_pool: ExistingHighQualityFactorPool | None = None,
        seed_candidates: tuple[GraphSeedCandidate, ...] = (),
        canonical_key_snapshot: FactorLibraryCanonicalKeySnapshot | None = None,
        persistence_fault_hook: Callable[[str], None] | None = None,
        progress_callback: Callable[[], None] | None = None,
        telemetry: GraphTelemetryRecorder | None = None,
        initial_cycle_statuses: Mapping[str, str] | None = None,
    ) -> None:
        """Create a controller with explicit, already-validated dependencies."""
        if not isinstance(config, GraphConfig):
            raise CoreValidationError("config must be GraphConfig")
        if not hasattr(attempt_executor, "run_batch") or not hasattr(
            attempt_executor, "run_initial"
        ):
            raise CoreValidationError(
                "attempt_executor does not implement GraphAttemptExecutor"
            )
        if not hasattr(artifacts, "get_bytes"):
            raise CoreValidationError("artifacts must provide get_bytes")
        if not hasattr(objects, "put_object") or not hasattr(objects, "get_object"):
            raise CoreValidationError("objects must provide object repository methods")
        if correlation_engine is not None and not all(
            hasattr(correlation_engine, method) for method in ("build", "update")
        ):
            raise CoreValidationError(
                "correlation_engine must implement build and update"
            )
        if not isinstance(correlation_protocol, GraphCorrelationProtocol):
            raise CoreValidationError(
                "correlation_protocol must be GraphCorrelationProtocol"
            )
        if (
            type(min_effective_date_ratio) not in (int, float)
            or not math.isfinite(float(min_effective_date_ratio))
            or not 0.0 <= float(min_effective_date_ratio) <= 1.0
        ):
            raise CoreValidationError("min_effective_date_ratio must be in [0, 1]")
        self._config = config
        self._executor = attempt_executor
        self._artifacts = artifacts
        self._objects = objects
        self._correlation_engine = correlation_engine
        self._correlation_protocol = correlation_protocol
        self._eligibility_gate = GraphNodeEligibilityGate(
            min_effective_date_ratio=float(min_effective_date_ratio)
        )
        self._pool = high_quality_pool or EmptyExistingHighQualityFactorPool()
        if not isinstance(seed_candidates, tuple) or not all(
            isinstance(candidate, GraphSeedCandidate) for candidate in seed_candidates
        ):
            raise CoreValidationError(
                "seed_candidates must be a tuple of GraphSeedCandidate"
            )
        self._seed_candidates = seed_candidates
        if canonical_key_snapshot is not None and not isinstance(
            canonical_key_snapshot, FactorLibraryCanonicalKeySnapshot
        ):
            raise CoreValidationError(
                "canonical_key_snapshot must be FactorLibraryCanonicalKeySnapshot"
            )
        self._canonical_key_snapshot = canonical_key_snapshot
        if progress_callback is not None and not callable(progress_callback):
            raise CoreValidationError("progress_callback must be callable")
        if telemetry is not None and not isinstance(telemetry, GraphTelemetryRecorder):
            raise CoreValidationError(
                "telemetry must be GraphTelemetryRecorder or None"
            )
        self._progress_callback = progress_callback
        self._telemetry = telemetry
        if initial_cycle_statuses is not None and (
            not isinstance(initial_cycle_statuses, Mapping)
            or not all(
                type(cycle_id) is str
                and type(status) is str
                and status in {"active", "completed", "failed"}
                for cycle_id, status in initial_cycle_statuses.items()
            )
        ):
            raise CoreValidationError(
                "initial_cycle_statuses must map cycle IDs to workflow statuses"
            )
        self._initial_cycle_statuses = dict(initial_cycle_statuses or {})
        self._persistence = GraphPersistence(
            artifacts=artifacts,
            objects=objects,
            correlation_engine=correlation_engine,
            fault_hook=persistence_fault_hook,
        )

    def run(
        self,
        *,
        run_id: str,
        context: ResearchContext,
        data_snapshot: DataSnapshot,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        resume: bool = False,
    ) -> GraphResult:
        """Run initialization and all budgeted Graph rounds."""
        _require_text(run_id, "run_id")
        if not isinstance(context, ResearchContext):
            raise CoreValidationError("context must be ResearchContext")
        if not isinstance(data_snapshot, DataSnapshot):
            raise CoreValidationError("data_snapshot must be DataSnapshot")
        if not isinstance(config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        if type(resume) is not bool:
            raise CoreValidationError("resume must be a boolean")
        self._child_delta_hv = {}
        if hasattr(self, "_average_distance_reference"):
            del self._average_distance_reference
        resumed = (
            self._persistence.load_latest(
                run_id=run_id,
                data_snapshot_id=data_snapshot.snapshot_id,
                config_snapshot_ref=config_snapshot_ref,
                expected_protocol=self._correlation_protocol,
                random_seed=self._config.initialization.random_seed,
            )
            if resume
            else None
        )
        if resumed is not None:
            snapshot = resumed.snapshot
            expected_library_snapshot_id = (
                ""
                if self._canonical_key_snapshot is None
                else self._canonical_key_snapshot.snapshot_id
            )
            if (
                snapshot.duplicate_index.external_snapshot_id
                != expected_library_snapshot_id
            ):
                raise CoreValidationError(
                    "Graph resume Factor Library canonical-key snapshot mismatch"
                )
            nodes = dict(snapshot.graph.nodes)
            graph = snapshot.graph
            duplicate_index = DuplicateIndex.from_snapshot(snapshot.duplicate_index)
            attempts = list(snapshot.attempts)
            events = list(graph.generation_events)
            archive = snapshot.archive
            factor_refs = dict(snapshot.factor_refs)
            self._child_delta_hv = dict(snapshot.child_delta_hv)
            self._external_reference_pool = snapshot.external_reference_pool
            self._average_distance_reference = snapshot.average_distance_reference
            correlation_store = resumed.correlation_store
            attempts_used = snapshot.attempts_used
            next_round_index = snapshot.next_round_index
        else:
            resumed = None
            next_round_index = 0
            attempts_used = 0
            seed_mode = self._config.initialization.seed_mode
            pool_candidates = self._pool.candidates(
                data_snapshot_id=data_snapshot.snapshot_id,
                evaluation_protocol_id=self._correlation_protocol.evaluation_protocol_id,
                preprocessing_protocol_id=self._correlation_protocol.preprocessing_protocol_id,
            )
            pool_candidates = tuple(
                seed
                for seed in pool_candidates
                if seed.node.data_snapshot_id == data_snapshot.snapshot_id
                and seed.node.evaluation_protocol_id
                == self._correlation_protocol.evaluation_protocol_id
                and seed.node.preprocessing_protocol_id
                == self._correlation_protocol.preprocessing_protocol_id
                and seed.node.rank_ic_std_calculation_version == EVALUATOR_VERSION
            )
            seeds = () if seed_mode == "cold" else pool_candidates
            selected_seeds = self._select_seeds(seeds)
            selected_seed_ids = {seed.node.node_id for seed in selected_seeds}
            self._external_reference_pool = self._build_external_reference_pool(
                tuple(
                    seed
                    for seed in pool_candidates
                    if seed.node.node_id not in selected_seed_ids
                )
            )
            nodes = {
                seed.node.node_id: self._refresh_node_quality(seed.node)
                for seed in selected_seeds
            }
            factor_refs = {
                seed.node.node_id: seed.factor_ref
                for seed in selected_seeds
                if seed.factor_ref is not None
            }
            if len(factor_refs) != len(selected_seeds):
                raise CoreValidationError(
                    "selected seed candidates must provide factor_ref for Graph mutation"
                )
            duplicate_index = self._new_duplicate_index(
                admitted_keys=tuple(seed.duplicate_key for seed in selected_seeds)
            )
            for seed in selected_seeds:
                duplicate_index.add_node(seed.duplicate_key, seed.node.node_id)
            target_node_count = self._config.initialization.initial_node_count
            state = _GraphInitializationState(
                request=GraphAttemptRequest(
                    attempt_id=graph_initial_cycle_id(run_id),
                    round_id=0,
                    parent_id="graph-initial",
                    parent_factor_ref=None,
                    parent_selection_rank=0,
                    guidance=None,
                    guidance_ref=None,
                    context=context,
                    data_snapshot=data_snapshot,
                    config_snapshot_ref=config_snapshot_ref,
                    code_version=code_version,
                    run_id=run_id,
                ),
                target=target_node_count,
                nodes=nodes,
                factor_refs=factor_refs,
                duplicate_index=duplicate_index,
                cycle_statuses={
                    batch_index: status
                    for cycle_id, status in self._initial_cycle_statuses.items()
                    if (batch_index := graph_initial_cycle_index(run_id, cycle_id))
                    is not None
                },
            )
            if len(nodes) < target_node_count:
                if seed_mode == "warm":
                    self._fill_initial_from_seeds(state)
                else:
                    self._fill_initial_from_llm(state)
            if not nodes:
                raise CoreValidationError(
                    "GRAPH_WARM_START_NO_SEED_CANDIDATES"
                    if seed_mode == "warm"
                    and not any(
                        candidate.frequency == context.frequency
                        for candidate in self._seed_candidates
                    )
                    else state.last_error_code
                    or "GRAPH_INITIALIZATION_NO_ELIGIBLE_NODE"
                )
            attempts = []
            events = []
            correlation_store = self._build_correlation_store(
                nodes.values(), data_snapshot, round_id=0
            )

            archive = pareto_archive(
                tuple(quality_point_from_node(node) for node in nodes.values())
            )
            self._average_distance_reference = self._build_reference_snapshot(
                archive,
                round_index=0,
            )
            graph = self._build_graph(
                nodes,
                events,
                correlation_store,
                telemetry_round_id=0,
            )
            nodes, graph = self._refresh_average_distances(
                graph,
                reference=self._average_distance_reference,
                correlation_store=correlation_store,
                telemetry_round_id=0,
            )
            self._persist_round(
                run_id=run_id,
                round_index=0,
                next_round_index=0,
                data_snapshot=data_snapshot,
                config_snapshot_ref=config_snapshot_ref,
                graph=graph,
                duplicate_index=duplicate_index,
                attempts=attempts,
                archive=archive,
                factor_refs=factor_refs,
                attempts_used=attempts_used,
                correlation_store=correlation_store,
            )
        selector = FlowGuidedGraphSelector(self._config)
        flow_result = self._run_flow(
            graph,
            {node_id: self._reward(node) for node_id, node in graph.nodes.items()},
            round_id=next_round_index,
        )
        termination_reason = "GLOBAL_ATTEMPT_BUDGET_EXHAUSTED"
        for round_id in range(
            next_round_index, self._config.search.max_expansion_attempts
        ):
            pre_archive = archive
            round_reference = self._build_reference_snapshot(
                pre_archive,
                round_index=round_id,
            )
            nodes, graph = self._refresh_average_distances(
                graph,
                reference=round_reference,
                correlation_store=correlation_store,
                telemetry_round_id=round_id,
            )
            self._average_distance_reference = round_reference
            selection_kwargs: dict[str, object] = {
                "attempts_used": attempts_used,
                "run_seed": self._config.initialization.random_seed,
                "round_index": round_id,
            }
            if self._telemetry is not None:
                selection_kwargs["flow_observer"] = (
                    lambda elapsed_seconds, result, round_id=round_id: self._record_flow_timing(
                        round_id,
                        elapsed_seconds,
                        result,
                    )
                )
            selection = selector.select(
                graph,
                **selection_kwargs,
            )
            self._persist_selector_audit(
                run_id=run_id,
                round_id=round_id,
                quality_archive=pre_archive,
                selection=selection,
            )
            flow_result = selection.flow_result
            if not selection.selected_parent_ids:
                termination_reason = selection.termination_reason.value
                break
            requests = tuple(
                self._attempt_request(
                    parent_id=parent_id,
                    selection_rank=rank,
                    round_id=round_id,
                    graph=graph,
                    archive=archive,
                    attempts=tuple(attempts),
                    factor_refs=factor_refs,
                    context=context,
                    data_snapshot=data_snapshot,
                    config_snapshot_ref=config_snapshot_ref,
                    code_version=code_version,
                    run_id=run_id,
                )
                for rank, parent_id in enumerate(selection.selected_parent_ids)
            )
            executions = self._executor.run_batch(
                requests,
                committed_index=duplicate_index,
            )
            if self._progress_callback is not None:
                self._progress_callback()
            expected_attempt_ids = tuple(request.attempt_id for request in requests)
            actual_attempt_ids = tuple(
                execution.request.attempt_id for execution in executions
            )
            if len(actual_attempt_ids) != len(set(actual_attempt_ids)) or set(
                actual_attempt_ids
            ) != set(expected_attempt_ids):
                raise CoreValidationError(
                    "Graph executor must return exactly one result per attempt"
                )
            new_nodes, new_keys, new_events, round_attempts = self._consume_batch(
                executions,
                graph=graph,
                duplicate_index=duplicate_index,
                pre_archive=pre_archive,
                run_id=run_id,
                data_snapshot=data_snapshot,
            )
            if new_nodes:
                delta_hv = dict(
                    batch_birth_time_delta_hv(
                        pre_archive,
                        tuple(
                            quality_point_from_node(node) for node in new_nodes.values()
                        ),
                        self._config.quality,
                    )
                )
            else:
                delta_hv = {}
            attempts.extend(round_attempts)
            events.extend(new_events)
            for key, node_id in new_keys.items():
                duplicate_index.add_node(key, node_id)
            factor_refs.update(self._last_new_factor_refs)
            all_delta_hv = getattr(self, "_child_delta_hv", {})
            all_delta_hv.update(delta_hv)
            self._child_delta_hv = all_delta_hv
            round_attempts = tuple(
                replace(
                    attempt,
                    fertility_outcome=(
                        attempt_fertility_outcome(attempt, all_delta_hv)
                        if attempt.status is ExpansionAttemptStatus.SUCCESS
                        else 0.0
                    ),
                )
                for attempt in round_attempts
            )
            attempts[-len(round_attempts) :] = round_attempts
            nodes.update(new_nodes)
            for parent_id in selection.selected_parent_ids:
                parent = nodes[parent_id]
                parent_attempts = tuple(
                    attempt for attempt in attempts if attempt.parent_id == parent_id
                )
                fertility = fertility_from_attempts(
                    parent_id, parent_attempts, all_delta_hv
                )
                nodes[parent_id] = replace(
                    parent,
                    expansion_attempt_count=parent.expansion_attempt_count + 1,
                    valid_child_count=parent.valid_child_count
                    + sum(
                        1
                        for attempt in round_attempts
                        if attempt.parent_id == parent_id
                        for _child in attempt.child_ids
                    ),
                    fertility=fertility,
                )
            archive = batch_archive_commit(
                pre_archive,
                tuple(quality_point_from_node(node) for node in new_nodes.values()),
            )
            correlation_store = self._update_correlation(
                correlation_store,
                tuple(new_nodes.values()),
                data_snapshot,
                round_id=round_id + 1,
            )
            graph = self._build_graph(
                nodes,
                events,
                correlation_store,
                telemetry_round_id=round_id + 1,
            )
            nodes, graph = self._refresh_average_distances(
                graph,
                reference=round_reference,
                correlation_store=correlation_store,
                telemetry_round_id=round_id + 1,
            )
            self._average_distance_reference = round_reference
            flow_result = self._run_flow(
                graph,
                {node_id: self._reward(node) for node_id, node in graph.nodes.items()},
                round_id=round_id + 1,
            )
            attempts_used += len(requests)
            self._persist_round(
                run_id=run_id,
                round_index=round_id + 1,
                next_round_index=round_id + 1,
                data_snapshot=data_snapshot,
                config_snapshot_ref=config_snapshot_ref,
                graph=graph,
                duplicate_index=duplicate_index,
                attempts=attempts,
                archive=archive,
                factor_refs=factor_refs,
                attempts_used=attempts_used,
                correlation_store=correlation_store,
            )
            if attempts_used >= self._config.search.max_expansion_attempts:
                termination_reason = "GLOBAL_ATTEMPT_BUDGET_EXHAUSTED"
                break
        else:
            termination_reason = "GLOBAL_ATTEMPT_BUDGET_EXHAUSTED"
        result = GraphResult(
            run_id=run_id,
            graph=graph,
            attempts=tuple(attempts),
            archive=archive,
            average_distance_reference=self._average_distance_reference,
            flow_result=flow_result,
            diagnostics=build_graph_diagnostics(
                graph,
                flow_result,
                structure_neighbor_cap=self._config.structure.max_neighbors,
                correlation_neighbor_cap=self._config.correlation.max_neighbors,
            ),
            attempts_used=attempts_used,
            remaining_budget=self._config.search.max_expansion_attempts - attempts_used,
            termination_reason=termination_reason,
        )
        self._objects.put_object(
            GRAPH_RESULT_OBJECT_TYPE,
            "graph-" + stable_sha256(result)[:24],
            result,
        )
        return result

    def _select_seeds(self, seeds: tuple[GraphSeed, ...]) -> tuple[GraphSeed, ...]:
        unique: dict[GraphDuplicateKey, GraphSeed] = {}
        for seed in sorted(seeds, key=lambda item: item.node.node_id):
            unique.setdefault(seed.duplicate_key, seed)
        ordered = tuple(unique.values())
        if len(ordered) <= self._config.initialization.initial_node_count:
            return ordered
        randomizer = random.Random(self._config.initialization.random_seed)
        return tuple(
            sorted(
                randomizer.sample(
                    ordered, self._config.initialization.initial_node_count
                ),
                key=lambda seed: seed.node.node_id,
            )
        )

    def _fill_initial_from_llm(self, state: _GraphInitializationState) -> None:
        """Fill cold or auto roots with bounded single-candidate LLM cycles."""
        existing_last = max(state.cycle_statuses, default=-1)
        avoid_expressions: list[str] = []

        def run_batch(batch_index: int) -> None:
            batch_context = replace(
                state.request.context,
                constraints=(
                    *state.request.context.constraints,
                    "GRAPH_INITIALIZATION_DIVERSITY_JSON="
                    + to_json(
                        {
                            "batch_index": batch_index,
                            "target_node_count": state.target,
                            "current_node_count": len(state.nodes),
                            "remaining_node_count": state.target - len(state.nodes),
                            "avoid_expressions": tuple(avoid_expressions),
                            "instruction": (
                                "Propose one factor that differs from earlier batches "
                                "in economic mechanism, primary fields, operator structure, "
                                "or window structure; renaming and cosmetic changes do not count."
                            ),
                        }
                    ),
                ),
            )
            for expression in self._run_initial_cycle(
                state, batch_index, context=batch_context
            ):
                if expression not in avoid_expressions:
                    avoid_expressions.append(expression)

        for batch_index in range(existing_last + 1):
            status = state.cycle_statuses.get(batch_index)
            if status is None:
                continue
            if status in {"completed", "failed"}:
                expressions = self._recover_initial_cycle(
                    state,
                    graph_initial_cycle_id(state.request.run_id, batch_index),
                )
                if status == "completed":
                    for expression in expressions:
                        if expression not in avoid_expressions:
                            avoid_expressions.append(expression)
            else:
                run_batch(batch_index)
            if len(state.nodes) >= state.target:
                return

        first_new_batch = existing_last + 1
        batch_limit = 2 * (state.target - len(state.nodes))
        for batch_index in range(first_new_batch, first_new_batch + batch_limit):
            run_batch(batch_index)
            if len(state.nodes) >= state.target:
                return

    def _fill_initial_from_seeds(self, state: _GraphInitializationState) -> None:
        """Fill warm roots from configured proposals without calling an LLM."""
        candidates = self._ordered_seed_candidates(
            state.request.context,
            first_batch_size=state.target - len(state.nodes),
        )
        candidate_offset = 0
        batch_index = 0
        while len(state.nodes) < state.target and candidate_offset < len(candidates):
            batch_size = state.target - len(state.nodes)
            batch = candidates[candidate_offset : candidate_offset + batch_size]
            candidate_offset += len(batch)
            status = state.cycle_statuses.get(batch_index)
            if status in {"completed", "failed"}:
                self._recover_initial_cycle(
                    state,
                    graph_initial_cycle_id(state.request.run_id, batch_index),
                )
            else:
                self._run_initial_cycle(
                    state,
                    batch_index,
                    fixed_factor_design_drafts=self._seed_design_drafts(
                        frequency=state.request.context.frequency,
                        candidates=batch,
                    ),
                )
            batch_index += 1

    def _run_initial_cycle(
        self,
        state: _GraphInitializationState,
        batch_index: int,
        *,
        context: ResearchContext | None = None,
        fixed_factor_design_drafts: tuple[FactorDesignDraft, ...] = (),
    ) -> tuple[str, ...]:
        """Run and merge one initialization cycle, retaining failures as audit."""
        request = replace(
            state.request,
            attempt_id=graph_initial_cycle_id(state.request.run_id, batch_index),
            context=state.request.context if context is None else context,
        )
        execution = self._executor.run_initial(
            request,
            committed_index=state.duplicate_index,
            fixed_factor_design_drafts=fixed_factor_design_drafts,
        )
        if self._progress_callback is not None:
            self._progress_callback()
        if execution.loop_result is None:
            state.last_error_code = (
                execution.error_code or "GRAPH_INITIALIZATION_FAILED"
            )
            self._persist_attempt_failure_outcome(
                request,
                state.last_error_code,
            )
            return ()
        result = execution.loop_result
        expressions = (
            ()
            if context is None
            else tuple(
                self._objects.get_object(
                    candidate.proposal_ref, FactorProposal
                ).proposed_expression
                for candidate in result.candidate_results
            )
        )
        batch_nodes, batch_index_values = self._nodes_from_loop_result(
            result,
            request=request,
            data_snapshot=request.data_snapshot,
        )
        for key in batch_index_values.snapshot().invalid_keys:
            if not state.duplicate_index.contains(key):
                state.duplicate_index.mark_invalid(key)
        for entry in batch_index_values.snapshot().entries:
            if state.duplicate_index.contains(entry.key):
                continue
            state.duplicate_index.add_node(entry.key, entry.node_id)
            state.nodes[entry.node_id] = batch_nodes[entry.node_id]
            state.factor_refs[entry.node_id] = self._last_factor_refs[entry.node_id]
        return expressions

    def _recover_initial_cycle(
        self,
        state: _GraphInitializationState,
        cycle_id: str,
    ) -> tuple[str, ...]:
        """Recover completed initialization facts without re-running their Loop."""
        outcomes = tuple(
            outcome
            for outcome in load_graph_candidate_outcomes(
                self._objects,
                run_id=state.request.run_id,
            )
            if outcome.attempt_id == cycle_id
        )
        expressions = tuple(
            outcome.expression for outcome in outcomes if outcome.expression is not None
        )
        records = tuple(
            record
            for record in self._objects.list_experiments()
            if record.run_id == state.request.run_id and record.cycle_id == cycle_id
        )
        experiment_ref = (
            None
            if not records
            else ObjectRef(
                "experiment_record",
                records[-1].experiment_id,
                records[-1].schema_version,
            )
        )
        for outcome in outcomes:
            if (
                outcome.factor_ref is None
                or outcome.evaluation_result_ref is None
                or outcome.candidate_status is None
            ):
                continue
            node = self._node_from_refs(
                factor_ref=outcome.factor_ref,
                evaluation_ref=outcome.evaluation_result_ref,
                status=outcome.candidate_status,
                experiment_ref=experiment_ref,
                data_snapshot=state.request.data_snapshot,
            )
            factor = self._objects.get_object(outcome.factor_ref, FactorSpec)
            key = GraphDuplicateKey.from_factor_spec(factor)
            if node is None:
                if not state.duplicate_index.contains(key):
                    state.duplicate_index.mark_invalid(key)
                continue
            if state.duplicate_index.contains(key):
                continue
            state.duplicate_index.add_node(key, node.node_id)
            state.nodes[node.node_id] = node
            state.factor_refs[node.node_id] = outcome.factor_ref
        return expressions

    def _ordered_seed_candidates(
        self,
        context: ResearchContext,
        *,
        first_batch_size: int,
    ) -> tuple[GraphSeedCandidate, ...]:
        """Order warm seed candidates with the existing first-batch sampling rule."""
        candidates = tuple(
            candidate
            for candidate in self._seed_candidates
            if candidate.frequency == context.frequency
        )
        if len(candidates) <= first_batch_size:
            return candidates
        randomizer = random.Random(self._config.initialization.random_seed)
        first_batch = tuple(
            sorted(
                randomizer.sample(candidates, first_batch_size),
                key=lambda candidate: candidate.candidate_id,
            )
        )
        selected_ids = {candidate.candidate_id for candidate in first_batch}
        remaining = [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in selected_ids
        ]
        randomizer.shuffle(remaining)
        return (*first_batch, *remaining)

    def _build_external_reference_pool(
        self,
        seeds: tuple[GraphSeed, ...],
    ) -> ExternalReferencePoolSnapshot:
        """Freeze compatible historical candidates before applying the RankIC gate."""
        seeds = tuple(
            seed
            for seed in seeds
            if seed.node.rank_ic_std_calculation_version == EVALUATOR_VERSION
        )
        unique: dict[GraphDuplicateKey, GraphSeed] = {}
        for seed in sorted(seeds, key=lambda item: item.node.node_id):
            unique.setdefault(seed.duplicate_key, seed)
        candidates = tuple(unique.values())
        by_node_id = {seed.node.node_id: seed for seed in candidates}
        if len(by_node_id) != len(candidates):
            raise CoreValidationError(
                "external reference candidates must have unique node IDs"
            )
        rewards = {
            node_id: quality_point_from_node(seed.node).objectives
            for node_id, seed in by_node_id.items()
        }
        ordered_ids: list[str] = []
        for front in pareto_fronts(rewards) if rewards else ():
            crowding = crowding_distances(
                front,
                rewards,
                epsilon=self._config.selector.crowding_epsilon,
            )
            ordered_ids.extend(
                sorted(
                    front,
                    key=lambda node_id: (
                        -crowding[node_id],
                        by_node_id[node_id].node.factor_id,
                        node_id,
                    ),
                )
            )
        threshold = self._config.average_distance.minimum_abs_rank_ic
        external_candidates = {
            node_id: ExternalReferenceFactor.from_node(seed.node)
            for node_id, seed in by_node_id.items()
        }
        selected = tuple(
            external_candidates[node_id]
            for node_id in ordered_ids
            if abs(by_node_id[node_id].node.rank_ic_mean) >= threshold
        )[: self._config.average_distance.external_pool_capacity]
        return ExternalReferencePoolSnapshot(
            candidate_node_ids=tuple(ordered_ids),
            candidate_content_digest=stable_sha256(
                tuple(external_candidates[node_id] for node_id in sorted(ordered_ids))
            ),
            references=selected,
            minimum_abs_rank_ic=threshold,
            capacity=self._config.average_distance.external_pool_capacity,
        )

    def _build_reference_snapshot(
        self,
        archive: QualityParetoArchive,
        *,
        round_index: int,
    ) -> AverageDistanceReferenceSnapshot:
        """Freeze one round's strict or bootstrap AverageDistance references."""
        threshold = self._config.average_distance.minimum_abs_rank_ic
        qualified_internal = tuple(
            sorted(
                point.node_id
                for point in archive.points
                if point.objectives[0] >= threshold
            )
        )
        external_ids = tuple(
            sorted(value.node_id for value in self._external_reference_pool.references)
        )
        previous_reference = getattr(self, "_average_distance_reference", None)
        if (
            isinstance(previous_reference, AverageDistanceReferenceSnapshot)
            and previous_reference.mode is AverageDistanceReferenceMode.STRICT
            and not qualified_internal
            and not external_ids
        ):
            raise CoreValidationError(
                "AverageDistance strict mode cannot revert to bootstrap"
            )
        if qualified_internal or external_ids:
            mode = AverageDistanceReferenceMode.STRICT
            internal_ids = qualified_internal
        else:
            mode = AverageDistanceReferenceMode.BOOTSTRAP
            internal_ids = tuple(sorted(point.node_id for point in archive.points))
            external_ids = ()
        return AverageDistanceReferenceSnapshot(
            round_index=round_index,
            archive_snapshot_id=("quality-archive-" + stable_sha256(archive)[:24]),
            external_pool_snapshot_id=self._external_reference_pool.snapshot_id,
            mode=mode,
            internal_node_ids=internal_ids,
            external_node_ids=external_ids,
            reference_node_ids=tuple(sorted(set(internal_ids) | set(external_ids))),
        )

    def _new_duplicate_index(
        self,
        *,
        admitted_keys: tuple[GraphDuplicateKey, ...] = (),
        invalid_keys: tuple[GraphDuplicateKey, ...] = (),
    ) -> DuplicateIndex:
        """Create a run index while reserving the immutable library key set."""
        external_keys: tuple[GraphDuplicateKey, ...] = ()
        snapshot_id = ""
        if self._canonical_key_snapshot is not None:
            admitted = set(admitted_keys)
            external_keys = tuple(
                key for key in self._canonical_key_snapshot.keys if key not in admitted
            )
            snapshot_id = self._canonical_key_snapshot.snapshot_id
        index = DuplicateIndex(
            external_keys=external_keys,
            external_snapshot_id=snapshot_id,
        )
        for key in invalid_keys:
            if not index.contains(key):
                index.mark_invalid(key)
        return index

    def _seed_design_drafts(
        self,
        *,
        frequency: str,
        candidates: tuple[GraphSeedCandidate, ...],
    ) -> tuple[FactorDesignDraft, ...]:
        """Convert one warm-start batch without generating expressions."""
        if any(candidate.frequency != frequency for candidate in candidates):
            raise CoreValidationError("seed frequency must match the research context")
        return tuple(candidate.to_design_draft() for candidate in candidates)

    def _attempt_request(
        self,
        *,
        parent_id: str,
        selection_rank: int,
        round_id: int,
        graph: FactorGraph,
        archive: QualityParetoArchive,
        attempts: tuple[ExpansionAttempt, ...],
        factor_refs: Mapping[str, ObjectRef],
        context: ResearchContext,
        data_snapshot: DataSnapshot,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        run_id: str,
    ) -> GraphAttemptRequest:
        parent = graph.nodes[parent_id]
        parent_factor_ref = factor_refs.get(parent_id)
        if parent_factor_ref is None:
            raise CoreValidationError(
                "Graph parent is missing its FactorSpec reference"
            )
        parent_quality = quality_point_from_node(parent).objectives
        archive_best = tuple(
            max(
                (point.objectives[index] for point in archive.points),
                default=parent_quality[index],
            )
            for index in range(3)
        )
        parent_attempts = tuple(
            attempt for attempt in attempts if attempt.parent_id == parent_id
        )
        guidance = GraphExpansionGuidance(
            parent_id=parent_id,
            parent_expression=parent.expression,
            parent_metrics=(
                ("rank_ic_mean", parent.rank_ic_mean),
                ("rank_ic_std", parent.rank_ic_std),
                ("top20_mean_return", parent.top20_return),
                ("self_quality", parent.self_quality),
                ("fertility", parent.fertility),
                ("average_distance", parent.average_distance),
                ("expansion_attempt_count", float(len(parent_attempts))),
                ("valid_child_count", float(parent.valid_child_count)),
                (
                    "failed_attempt_count",
                    float(
                        sum(
                            attempt.status is not ExpansionAttemptStatus.SUCCESS
                            for attempt in parent_attempts
                        )
                    ),
                ),
            ),
            target_objectives=("SelfQuality", "Fertility", "AverageDistance"),
            pareto_gaps=tuple(
                (name, max(0.0, archive_best[index] - parent_quality[index]))
                for index, name in enumerate(
                    (
                        "rank_ic_strength",
                        "rank_ic_std_reduction",
                        "top20_mean_return",
                    )
                )
            ),
            forbidden_structures=(),
            correlation_constraints=(
                ("edge_threshold", self._config.correlation.edge_threshold),
                ("max_neighbors", float(self._config.correlation.max_neighbors)),
                (
                    "min_valid_date_count",
                    float(self._correlation_protocol.min_valid_date_count),
                ),
                (
                    "min_valid_date_ratio",
                    self._correlation_protocol.min_valid_date_ratio,
                ),
            ),
            allowed_operations=(),
        )
        return GraphAttemptRequest(
            attempt_id=graph_attempt_id(run_id, round_id, parent_id),
            round_id=round_id,
            parent_id=parent_id,
            parent_factor_ref=parent_factor_ref,
            parent_selection_rank=selection_rank,
            guidance=guidance,
            guidance_ref=self._objects.put_object(
                GRAPH_GUIDANCE_OBJECT_TYPE,
                "guidance-" + stable_sha256(guidance)[:24],
                guidance,
            ),
            context=context,
            data_snapshot=data_snapshot,
            config_snapshot_ref=config_snapshot_ref,
            code_version=code_version,
            run_id=run_id,
        )

    def _nodes_from_loop_result(
        self,
        result: CosmosAlphaLoopResult,
        *,
        request: GraphAttemptRequest,
        data_snapshot: DataSnapshot,
    ) -> tuple[dict[str, GraphNode], DuplicateIndex]:
        nodes: dict[str, GraphNode] = {}
        index = DuplicateIndex()
        self._last_factor_refs: dict[str, ObjectRef] = {}
        self._persist_expression_failures(request, result)
        for candidate_index, candidate in enumerate(result.candidate_results):
            node = self._node_from_candidate(
                candidate,
                result=result,
                data_snapshot=data_snapshot,
            )
            if node is None:
                self._persist_candidate_outcome(
                    self._candidate_outcome(
                        request,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        outcome_override=(
                            GraphCandidateOutcomeStatus.EVALUATION_FAILURE
                            if getattr(candidate, "evaluation_result_ref", None)
                            is not None
                            and getattr(candidate, "failure", None) is None
                            else None
                        ),
                    )
                )
                self._mark_technically_invalid(candidate, index)
                continue
            factor = self._objects.get_object(candidate.factor_ref, FactorSpec)  # type: ignore[arg-type]
            key = GraphDuplicateKey.from_factor_spec(factor)
            if index.contains(key):
                self._persist_candidate_outcome(
                    self._candidate_outcome(
                        request,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        outcome_override=GraphCandidateOutcomeStatus.DUPLICATE,
                    )
                )
                continue
            self._persist_candidate_outcome(
                self._candidate_outcome(
                    request,
                    candidate_index=candidate_index,
                    candidate=candidate,
                )
            )
            nodes[node.node_id] = node
            self._last_factor_refs[node.node_id] = candidate.factor_ref  # type: ignore[assignment]
            index.add_node(key, node.node_id)
        return nodes, index

    def _consume_batch(
        self,
        executions: tuple[GraphAttemptExecution, ...],
        *,
        graph: FactorGraph,
        duplicate_index: DuplicateIndex,
        pre_archive: QualityParetoArchive,
        run_id: str,
        data_snapshot: DataSnapshot,
    ) -> tuple[
        dict[str, GraphNode],
        dict[GraphDuplicateKey, str],
        tuple[GenerationEvent, ...],
        tuple[ExpansionAttempt, ...],
    ]:
        new_nodes: dict[str, GraphNode] = {}
        new_keys: dict[GraphDuplicateKey, str] = {}
        events: list[GenerationEvent] = []
        attempts: list[ExpansionAttempt] = []
        self._last_new_factor_refs: dict[str, ObjectRef] = {}
        for execution in sorted(
            executions,
            key=lambda item: (
                item.request.parent_selection_rank,
                item.request.attempt_id,
            ),
        ):
            request = execution.request
            child_ids: list[str] = []
            outcome_codes: list[str] = []
            if execution.loop_result is None:
                outcome_codes.append(execution.error_code or "LLM_FAILED")
                self._persist_attempt_failure_outcome(
                    request,
                    execution.error_code or "LLM_FAILED",
                )
            else:
                self._persist_expression_failures(request, execution.loop_result)
                outcome_codes.extend(
                    _expression_failure_code(failure)
                    for failure in execution.loop_result.expression_failures
                )
                for candidate_index, candidate in enumerate(
                    execution.loop_result.candidate_results
                ):
                    if candidate.pre_calculation_gate_reason is not None:
                        outcome_codes.append(candidate.pre_calculation_gate_reason)
                        self._persist_candidate_outcome(
                            self._candidate_outcome(
                                request,
                                candidate_index=candidate_index,
                                candidate=candidate,
                            )
                        )
                        continue
                    node = self._node_from_candidate(
                        candidate,
                        result=execution.loop_result,
                        data_snapshot=data_snapshot,
                    )
                    if node is None:
                        self._persist_candidate_outcome(
                            self._candidate_outcome(
                                request,
                                candidate_index=candidate_index,
                                candidate=candidate,
                                outcome_override=(
                                    GraphCandidateOutcomeStatus.EVALUATION_FAILURE
                                    if getattr(candidate, "evaluation_result_ref", None)
                                    is not None
                                    and getattr(candidate, "failure", None) is None
                                    else None
                                ),
                            )
                        )
                        if candidate.failure is not None:
                            outcome_codes.append(
                                _candidate_failure_code(candidate.failure)
                            )
                        else:
                            outcome_codes.append("EVALUATION_FAILED")
                        self._mark_technically_invalid(candidate, duplicate_index)
                        continue
                    factor = self._objects.get_object(candidate.factor_ref, FactorSpec)  # type: ignore[arg-type]
                    key = GraphDuplicateKey.from_factor_spec(factor)
                    if duplicate_index.contains(key) or key in new_keys:
                        outcome_codes.append("DUPLICATE_ATTEMPT")
                        self._persist_candidate_outcome(
                            self._candidate_outcome(
                                request,
                                candidate_index=candidate_index,
                                candidate=candidate,
                                outcome_override=GraphCandidateOutcomeStatus.DUPLICATE,
                            )
                        )
                        continue
                    self._persist_candidate_outcome(
                        self._candidate_outcome(
                            request,
                            candidate_index=candidate_index,
                            candidate=candidate,
                        )
                    )
                    new_nodes[node.node_id] = node
                    new_keys[key] = node.node_id
                    self._last_new_factor_refs[node.node_id] = candidate.factor_ref  # type: ignore[assignment]
                    child_ids.append(node.node_id)
                    events.append(
                        GenerationEvent(
                            event_id="generation-"
                            + stable_sha256(
                                {
                                    "attempt_id": request.attempt_id,
                                    "child_id": node.node_id,
                                }
                            )[:24],
                            parent_ids=(request.parent_id,),
                            child_id=node.node_id,
                            operation="mutation",
                            attempt_id=request.attempt_id,
                            guidance_ref=request.guidance_ref,
                            experiment_ref=execution.loop_result.experiment_ref,
                            created_at=_event_created_at(execution.loop_result),
                        )
                    )
            status = _attempt_status(child_ids, outcome_codes)
            attempts.append(
                ExpansionAttempt(
                    attempt_id=request.attempt_id,
                    parent_id=request.parent_id,
                    round_id=request.round_id,
                    status=status,
                    child_ids=tuple(child_ids),
                    outcome_codes=tuple(outcome_codes),
                    fertility_outcome=0.0,
                    guidance_ref=request.guidance_ref,
                )
            )
        return new_nodes, new_keys, tuple(events), tuple(attempts)

    def _persist_expression_failures(
        self,
        request: GraphAttemptRequest,
        result: CosmosAlphaLoopResult,
    ) -> None:
        """Persist expression-generation failures as candidate audit outcomes."""
        for offset, failure in enumerate(
            result.expression_failures,
            start=len(result.candidate_results),
        ):
            outcome = GraphCandidateOutcome(
                run_id=request.run_id,
                round_id=request.round_id,
                attempt_id=request.attempt_id,
                parent_id=request.parent_id,
                candidate_index=offset,
                outcome=_failure_outcome_status(failure),
                candidate_status=ExperimentStatus.FAILED,
                proposal_ref=None,
                validation_report_ref=None,
                factor_ref=None,
                calculation_result_ref=None,
                preprocessing_result_ref=None,
                evaluation_result_ref=None,
                acceptance_decision_ref=None,
                reason_codes=(failure.error_code,),
                failure=failure,
            )
            self._persist_candidate_outcome(outcome)

    def _persist_attempt_failure_outcome(
        self,
        request: GraphAttemptRequest,
        error_code: str,
    ) -> None:
        """Persist a structured audit row when no candidate result exists."""
        outcome = GraphCandidateOutcome(
            run_id=request.run_id,
            round_id=request.round_id,
            attempt_id=request.attempt_id,
            parent_id=request.parent_id,
            candidate_index=0,
            outcome=GraphCandidateOutcomeStatus.OTHER_FAILURE,
            candidate_status=None,
            proposal_ref=None,
            validation_report_ref=None,
            factor_ref=None,
            calculation_result_ref=None,
            preprocessing_result_ref=None,
            evaluation_result_ref=None,
            acceptance_decision_ref=None,
            reason_codes=(error_code,),
        )
        self._persist_candidate_outcome(outcome)

    def _candidate_outcome(
        self,
        request: GraphAttemptRequest,
        *,
        candidate_index: int,
        candidate: object,
        outcome_override: GraphCandidateOutcomeStatus | None = None,
    ) -> GraphCandidateOutcome:
        """Convert one Loop candidate result into a Graph audit object."""
        candidate_status = getattr(candidate, "status", None)
        if not isinstance(candidate_status, ExperimentStatus):
            candidate_status = None
        failure = getattr(candidate, "failure", None)
        if not isinstance(failure, FailureInfo):
            failure = None
        proposal_ref = _optional_object_ref(candidate, "proposal_ref")
        validation_report_ref = _optional_object_ref(
            candidate,
            "validation_report_ref",
        )
        factor_ref = _optional_object_ref(candidate, "factor_ref")
        calculation_result_ref = _optional_object_ref(
            candidate,
            "calculation_result_ref",
        )
        preprocessing_result_ref = _optional_object_ref(
            candidate,
            "preprocessing_result_ref",
        )
        evaluation_result_ref = _optional_object_ref(
            candidate,
            "evaluation_result_ref",
        )
        acceptance_decision_ref = _optional_object_ref(
            candidate,
            "acceptance_decision_ref",
        )
        gate_reason = getattr(candidate, "pre_calculation_gate_reason", None)
        if not isinstance(gate_reason, str):
            gate_reason = None
        if outcome_override is not None:
            outcome = outcome_override
        elif gate_reason == "DUPLICATE_ATTEMPT":
            outcome = GraphCandidateOutcomeStatus.DUPLICATE
        elif candidate_status is ExperimentStatus.ACCEPTED:
            outcome = GraphCandidateOutcomeStatus.ACCEPTED
        elif candidate_status is ExperimentStatus.REJECTED:
            outcome = GraphCandidateOutcomeStatus.REJECTED
        elif candidate_status is ExperimentStatus.FAILED:
            outcome = _failure_outcome_status(
                failure,
                has_evaluation_result=evaluation_result_ref is not None,
            )
        else:
            outcome = GraphCandidateOutcomeStatus.OTHER_FAILURE
        reason_codes = _candidate_reason_codes(
            gate_reason=gate_reason,
            failure=failure,
            acceptance_decision_ref=acceptance_decision_ref,
            objects=self._objects,
        )
        if outcome is GraphCandidateOutcomeStatus.DUPLICATE and not reason_codes:
            reason_codes = ("DUPLICATE_ATTEMPT",)
        expression = None
        if factor_ref is not None:
            expression = self._objects.get_object(factor_ref, FactorSpec).expression
        return GraphCandidateOutcome(
            run_id=request.run_id,
            round_id=request.round_id,
            attempt_id=request.attempt_id,
            parent_id=request.parent_id,
            candidate_index=candidate_index,
            outcome=outcome,
            candidate_status=candidate_status,
            proposal_ref=proposal_ref,
            validation_report_ref=validation_report_ref,
            factor_ref=factor_ref,
            calculation_result_ref=calculation_result_ref,
            preprocessing_result_ref=preprocessing_result_ref,
            evaluation_result_ref=evaluation_result_ref,
            acceptance_decision_ref=acceptance_decision_ref,
            pre_calculation_gate_reason=gate_reason,
            reason_codes=reason_codes,
            failure=failure,
            expression=expression,
        )

    def _persist_candidate_outcome(self, outcome: GraphCandidateOutcome) -> None:
        """Publish one immutable candidate audit object idempotently."""
        persist_graph_candidate_outcome(self._objects, outcome)

    def _node_from_candidate(
        self,
        candidate: object,
        *,
        result: CosmosAlphaLoopResult,
        data_snapshot: DataSnapshot,
    ) -> GraphNode | None:
        if not hasattr(candidate, "factor_ref"):
            return None
        factor_ref = candidate.factor_ref
        evaluation_ref = candidate.evaluation_result_ref
        if not isinstance(factor_ref, ObjectRef) or not isinstance(
            evaluation_ref, ObjectRef
        ):
            return None
        return self._node_from_refs(
            factor_ref=factor_ref,
            evaluation_ref=evaluation_ref,
            status=candidate.status,
            experiment_ref=result.experiment_ref,
            data_snapshot=data_snapshot,
        )

    def _node_from_refs(
        self,
        *,
        factor_ref: ObjectRef,
        evaluation_ref: ObjectRef,
        status: ExperimentStatus,
        experiment_ref: ObjectRef | None,
        data_snapshot: DataSnapshot,
    ) -> GraphNode | None:
        """Build a Graph node from one persisted, fully evaluated candidate."""
        if status not in (
            ExperimentStatus.ACCEPTED,
            ExperimentStatus.REJECTED,
        ):
            return None
        factor = self._objects.get_object(factor_ref, FactorSpec)
        evaluation = self._objects.get_object(evaluation_ref, EvaluationResult)
        if evaluation.data_snapshot_id != data_snapshot.snapshot_id:
            raise CoreValidationError(
                "Graph candidate evaluation data snapshot does not match the run"
            )
        if (
            evaluation.evaluation_protocol_id
            != self._correlation_protocol.evaluation_protocol_id
            or evaluation.preprocessing_protocol_id
            != self._correlation_protocol.preprocessing_protocol_id
        ):
            raise CoreValidationError(
                "Graph candidate evaluation protocols do not match the run"
            )
        eligibility = self._eligibility_gate.evaluate(evaluation)
        if not eligibility.eligible:
            return None
        metrics_by_name = {metric.name: metric for metric in evaluation.metrics}
        required = ("rank_ic_mean", "rank_ic_std", "top20_mean_return")
        if any(
            name not in metrics_by_name
            or not math.isfinite(metrics_by_name[name].value)
            for name in required
        ):
            return None
        if (
            evaluation.evaluator_version != EVALUATOR_VERSION
            or metrics_by_name["rank_ic_std"].calculation_version != EVALUATOR_VERSION
        ):
            return None
        metrics = {name: metric.value for name, metric in metrics_by_name.items()}
        orientation = factor_orientation(metrics["rank_ic_mean"])
        key = GraphDuplicateKey.from_factor_spec(factor)
        node_id = "graph-node-" + key.digest[:24]
        node = GraphNode(
            node_id=node_id,
            factor_id=factor.factor_id,
            expression=factor.expression,
            expression_hash=key.digest,
            experiment_ref=experiment_ref,
            evaluation_result_ref=evaluation_ref,
            evaluated_factor_values_ref=evaluation.evaluated_factor_values_ref,
            data_snapshot_id=evaluation.data_snapshot_id,
            evaluation_protocol_id=evaluation.evaluation_protocol_id,
            preprocessing_protocol_id=evaluation.preprocessing_protocol_id,
            rank_ic_mean=float(metrics["rank_ic_mean"]),
            rank_ic_std=float(metrics["rank_ic_std"]),
            rank_ic_std_calculation_version=metrics_by_name[
                "rank_ic_std"
            ].calculation_version,
            top20_return=float(metrics["top20_mean_return"]),
            orientation=orientation,
            self_quality=0.0,
            fertility=0.0,
            average_distance=0.0,
        )
        return replace(
            node, self_quality=self_quality_from_node(node, self._config.quality)
        )

    def _refresh_node_quality(self, node: GraphNode) -> GraphNode:
        return replace(
            node, self_quality=self_quality_from_node(node, self._config.quality)
        )

    def _run_flow(
        self,
        graph: FactorGraph,
        rewards: Mapping[str, tuple[float, float, float]],
        *,
        round_id: int,
    ) -> GraphFlowResult:
        """Run Graph Flow and optionally record its observational timing."""
        started_at = perf_counter()
        result = run_graph_flow(graph, rewards, self._config.flow)
        self._record_flow_timing(round_id, perf_counter() - started_at, result)
        return result

    def _persist_selector_audit(
        self,
        *,
        run_id: str,
        round_id: int,
        quality_archive: QualityParetoArchive,
        selection: object,
    ) -> None:
        """Persist one selector result without adding it to Graph state."""
        if not isinstance(selection, ParentSelectionResult):
            raise CoreValidationError("selection must be ParentSelectionResult")
        audit = GraphSelectorAudit.from_selection(
            run_id=run_id,
            round_id=round_id,
            quality_archive=quality_archive,
            selection=selection,
        )
        try:
            persist_graph_selector_audit(self._objects, audit)
        except Exception:
            # Selector audit is observational and cannot change parent selection.
            return

    def _record_correlation_timing(self, round_id: int) -> None:
        """Forward engine timing to the optional Graph telemetry recorder."""
        if self._telemetry is not None and self._correlation_engine is not None:
            try:
                self._telemetry.record_correlation(
                    round_id,
                    self._correlation_engine.last_timing,
                )
            except Exception:
                # Telemetry is observational only and cannot fail Graph search.
                return

    def _record_flow_timing(
        self,
        round_id: int,
        elapsed_seconds: float,
        result: GraphFlowResult,
    ) -> None:
        """Forward one Flow timing sample without affecting Graph state."""
        if self._telemetry is not None:
            try:
                self._telemetry.record_flow(round_id, elapsed_seconds, result)
            except Exception:
                # Telemetry is observational only and cannot fail Graph search.
                return

    def _mark_technically_invalid(
        self,
        candidate: object,
        duplicate_index: DuplicateIndex,
    ) -> None:
        """Remember evaluated but ineligible expressions for this Graph run."""
        factor_ref = getattr(candidate, "factor_ref", None)
        evaluation_ref = getattr(candidate, "evaluation_result_ref", None)
        if not isinstance(factor_ref, ObjectRef) or not isinstance(
            evaluation_ref, ObjectRef
        ):
            return
        factor = self._objects.get_object(factor_ref, FactorSpec)
        key = GraphDuplicateKey.from_factor_spec(factor)
        if not duplicate_index.contains(key):
            duplicate_index.mark_invalid(key)

    def _refresh_average_distances(
        self,
        graph: FactorGraph,
        *,
        reference: AverageDistanceReferenceSnapshot,
        correlation_store: CorrelationStore,
        telemetry_round_id: int = 0,
    ) -> tuple[dict[str, GraphNode], FactorGraph]:
        """Refresh novelty rewards against one committed reference set."""
        nodes = {
            node_id: replace(
                node,
                average_distance=self._average_distance(
                    node_id,
                    reference=reference,
                    graph=graph,
                    correlation_store=correlation_store,
                ),
            )
            for node_id, node in graph.nodes.items()
        }
        return nodes, self._build_graph(
            nodes,
            graph.generation_events,
            correlation_store,
            telemetry_round_id=telemetry_round_id,
        )

    def _build_correlation_store(
        self,
        nodes: Sequence[GraphNode],
        data_snapshot: DataSnapshot,
        *,
        round_id: int = 0,
    ) -> CorrelationStore:
        refs = self._correlation_refs(
            tuple(nodes) + self._external_reference_pool.references,
            data_snapshot,
        )
        if self._correlation_engine is None:
            return CorrelationStore.empty(self._correlation_protocol)
        store = self._correlation_engine.build(refs)
        self._record_correlation_timing(round_id)
        return store

    def _update_correlation(
        self,
        store: CorrelationStore,
        new_nodes: Sequence[GraphNode],
        data_snapshot: DataSnapshot,
        *,
        round_id: int = 0,
    ) -> CorrelationStore:
        if self._correlation_engine is None or not new_nodes:
            return store
        existing_factor_ids = set(store.factor_ids)
        new_refs = tuple(
            ref
            for ref in self._correlation_refs(new_nodes, data_snapshot)
            if ref.factor_id not in existing_factor_ids
        )
        if not new_refs:
            return store
        all_refs = store.factor_refs + new_refs
        updated = self._correlation_engine.update(
            store=store,
            new_factors=new_refs,
            all_existing_factors=all_refs,
        ).store
        self._record_correlation_timing(round_id)
        return updated

    def _correlation_refs(
        self,
        nodes: Sequence[GraphNode | ExternalReferenceFactor],
        data_snapshot: DataSnapshot,
    ) -> tuple[CorrelationFactorRef, ...]:
        refs_by_factor_id = {
            node.factor_id: CorrelationFactorRef(
                factor_id=node.factor_id,
                data_snapshot_id=node.data_snapshot_id,
                preprocessing_protocol_id=node.preprocessing_protocol_id,
                evaluation_protocol_id=node.evaluation_protocol_id,
                evaluated_factor_values_ref=node.evaluated_factor_values_ref,
                mask_data_ref=data_snapshot.mask_data.mask_data_ref,
            )
            for node in nodes
        }
        return tuple(refs_by_factor_id[key] for key in sorted(refs_by_factor_id))

    def _build_graph(
        self,
        nodes: Mapping[str, GraphNode],
        events: Sequence[GenerationEvent],
        correlation_store: CorrelationStore,
        *,
        telemetry_round_id: int = 0,
    ) -> FactorGraph:
        started_at = perf_counter()
        ordered_ids = tuple(sorted(nodes))
        forms = {
            node_id: StructureCanonicalForm.from_expression(nodes[node_id].expression)
            for node_id in ordered_ids
        }
        structure_facts = build_structure_fact_store(forms)
        w_structure = project_structure_neighborhood(
            structure_facts,
            edge_threshold=self._config.structure.edge_threshold,
            max_neighbors=self._config.structure.max_neighbors,
        )
        generation = build_generation_graph(ordered_ids, tuple(events))
        w_correlation = self._correlation_weights(correlation_store, ordered_ids, nodes)
        graph = FactorGraph(
            nodes=dict(nodes),
            generation_events=tuple(events),
            w_structure=w_structure,
            w_correlation=w_correlation,
            w_generation=generation.w_generation,
            node_index={node_id: index for index, node_id in enumerate(ordered_ids)},
        )
        if self._telemetry is not None:
            try:
                self._telemetry.record_w_construction(
                    telemetry_round_id,
                    perf_counter() - started_at,
                )
            except Exception:
                # Telemetry is observational only and cannot fail Graph search.
                return graph
        return graph

    def _correlation_weights(
        self,
        store: CorrelationStore,
        node_ids: tuple[str, ...],
        nodes: Mapping[str, GraphNode],
    ) -> tuple[tuple[float, ...], ...]:
        if not store.factor_refs:
            return tuple(tuple(0.0 for _ in node_ids) for _ in node_ids)
        projected = store.build_neighborhood(
            edge_threshold=self._config.correlation.edge_threshold,
            max_neighbors=self._config.correlation.max_neighbors,
        )
        factor_index = {
            factor_id: index for index, factor_id in enumerate(store.factor_ids)
        }
        return tuple(
            tuple(
                (
                    float(
                        projected[
                            factor_index[nodes[left].factor_id],
                            factor_index[nodes[right].factor_id],
                        ]
                    )
                    if nodes[left].factor_id in factor_index
                    and nodes[right].factor_id in factor_index
                    else 0.0
                )
                for right in node_ids
            )
            for left in node_ids
        )

    def _average_distance(
        self,
        node_id: str,
        *,
        reference: AverageDistanceReferenceSnapshot,
        graph: FactorGraph,
        correlation_store: CorrelationStore,
    ) -> float:
        forms = {
            key: StructureCanonicalForm.from_expression(node.expression)
            for key, node in graph.nodes.items()
        }
        for external in self._external_reference_pool.references:
            forms.setdefault(
                external.node_id,
                StructureCanonicalForm.from_expression(external.expression),
            )
        facts = build_structure_fact_store(forms)
        node_to_factor_id = {key: node.factor_id for key, node in graph.nodes.items()}
        for external in self._external_reference_pool.references:
            node_to_factor_id.setdefault(external.node_id, external.factor_id)
        return average_distance_from_config(
            node_id,
            reference_node_ids=reference.reference_node_ids,
            structure_facts=facts,
            correlation_store=correlation_store,
            node_to_factor_id=node_to_factor_id,
            config=self._config.average_distance,
        )

    @staticmethod
    def _reward(node: GraphNode) -> tuple[float, float, float]:
        return (node.self_quality, node.fertility, node.average_distance)

    def _persist_round(
        self,
        *,
        run_id: str,
        round_index: int,
        next_round_index: int,
        data_snapshot: DataSnapshot,
        config_snapshot_ref: ArtifactRef,
        graph: FactorGraph,
        duplicate_index: DuplicateIndex,
        attempts: Sequence[ExpansionAttempt],
        archive: QualityParetoArchive,
        factor_refs: Mapping[str, ObjectRef],
        attempts_used: int,
        correlation_store: CorrelationStore,
    ) -> None:
        """Persist one complete Graph round before publishing its commit marker."""
        self._persistence.persist_round(
            run_id=run_id,
            round_index=round_index,
            next_round_index=next_round_index,
            data_snapshot_id=data_snapshot.snapshot_id,
            config_snapshot_ref=config_snapshot_ref,
            graph=graph,
            duplicate_index=duplicate_index.snapshot(),
            attempts=tuple(attempts),
            archive=archive,
            child_delta_hv=self._child_delta_hv,
            factor_refs=factor_refs,
            external_reference_pool=self._external_reference_pool,
            average_distance_reference=self._average_distance_reference,
            attempts_used=attempts_used,
            random_seed=self._config.initialization.random_seed,
            correlation_store=correlation_store,
        )


def _attempt_status(
    child_ids: Sequence[str], outcome_codes: Sequence[str]
) -> ExpansionAttemptStatus:
    if child_ids:
        return ExpansionAttemptStatus.SUCCESS
    codes = set(outcome_codes)
    if codes == {"DUPLICATE_ATTEMPT"}:
        return ExpansionAttemptStatus.DUPLICATE
    if codes == {"INVALID_EXPRESSION"} or (
        bool(codes) and codes <= _VALIDATION_FAILURE_CODES
    ):
        return ExpansionAttemptStatus.INVALID_EXPRESSION
    if codes == {"LLM_FAILED"}:
        return ExpansionAttemptStatus.LLM_FAILED
    if codes == {"EVALUATION_FAILED"}:
        return ExpansionAttemptStatus.EVALUATION_FAILED
    return ExpansionAttemptStatus.NO_CHILD


def _failure_outcome_status(
    failure: FailureInfo | None,
    *,
    has_evaluation_result: bool = False,
) -> GraphCandidateOutcomeStatus:
    """Classify an existing structured failure without changing its code."""
    if failure is not None:
        stage = failure.stage.rsplit(":", 1)[-1]
        return {
            "calculation": GraphCandidateOutcomeStatus.CALCULATION_FAILURE,
            "preprocessing": GraphCandidateOutcomeStatus.PREPROCESSING_FAILURE,
            "evaluation": GraphCandidateOutcomeStatus.EVALUATION_FAILURE,
        }.get(stage, GraphCandidateOutcomeStatus.OTHER_FAILURE)
    if has_evaluation_result:
        return GraphCandidateOutcomeStatus.EVALUATION_FAILURE
    return GraphCandidateOutcomeStatus.OTHER_FAILURE


def _optional_object_ref(candidate: object, field_name: str) -> ObjectRef | None:
    """Read an optional candidate reference without widening the audit schema."""
    value = getattr(candidate, field_name, None)
    return value if isinstance(value, ObjectRef) else None


def _candidate_reason_codes(
    *,
    gate_reason: str | None,
    failure: FailureInfo | None,
    acceptance_decision_ref: ObjectRef | None,
    objects: object,
) -> tuple[str, ...]:
    """Collect reason codes from existing gate, failure, or decision facts."""
    if gate_reason is not None:
        return (gate_reason,)
    if failure is not None:
        return (failure.error_code,)
    if acceptance_decision_ref is None:
        return ()
    decision = objects.get_object(acceptance_decision_ref, AcceptanceDecision)
    return tuple(decision.reason_codes)


def _candidate_failure_code(failure: object) -> str:
    """Return the exact FailureInfo code for one candidate failure."""
    return _failure_info_code(failure, "UNEXPECTED_ATTEMPT_FAILURE")


def _expression_failure_code(failure: object) -> str:
    """Return the exact FailureInfo code for expression generation failure."""
    return _failure_info_code(failure, "LLM_FAILED")


_VALIDATION_FAILURE_CODES = frozenset(
    {
        "SYNTAX_ERROR",
        "UNKNOWN_FIELD",
        "UNKNOWN_OPERATOR",
        "INVALID_ARITY",
        "INVALID_PARAMETER_TYPE",
        "UNSUPPORTED_WINDOW",
        "UNSUPPORTED_FREQUENCY",
        "UNSUPPORTED_BACKEND",
        "COMPLEXITY_EXCEEDED",
    }
)


def _failure_info_code(failure: object, fallback: str) -> str:
    """Read a validated FailureInfo code without masking an attempt outcome."""
    try:
        code = getattr(failure, "error_code", None)
        return code if isinstance(code, str) and code else fallback
    except Exception:  # noqa: BLE001 - outcome reporting must be total
        return fallback


def _event_created_at(result: CosmosAlphaLoopResult) -> str:
    """Reuse the immutable Loop timestamp so interrupted rounds rebuild identically."""
    if result.experiment_record is not None:
        timestamp = (
            result.experiment_record.completed_at or result.experiment_record.created_at
        )
        return timestamp.isoformat().replace("+00:00", "Z")
    return datetime(1970, 1, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "EmptyExistingHighQualityFactorPool",
    "ExistingHighQualityFactorPool",
    "GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_OBJECT_TYPE",
    "GRAPH_FACTOR_LIBRARY_REFERENCE_SNAPSHOT_SCHEMA_VERSION",
    "GRAPH_GUIDANCE_OBJECT_TYPE",
    "GRAPH_GUIDANCE_SCHEMA_VERSION",
    "GRAPH_RESULT_OBJECT_TYPE",
    "GRAPH_RESULT_SCHEMA_VERSION",
    "GraphAttemptExecution",
    "GraphAttemptExecutor",
    "GraphAttemptRequest",
    "GraphController",
    "GraphExpansionGuidance",
    "GraphFactorLibraryReferenceSnapshot",
    "GraphResult",
    "GraphSeed",
    "LocalLoopAttemptExecutor",
    "graph_factor_library_reference_snapshot_id",
]
