"""Versioned Graph round snapshots and atomic commit markers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    ObjectRef,
    from_json,
    stable_sha256,
)
from cosmosalpha.evaluation.correlation import (
    CorrelationProtocolMismatch,
    CorrelationSnapshot,
    CorrelationStore,
    GraphCorrelationEngine,
    GraphCorrelationProtocol,
)
from cosmosalpha.pipeline.graph.core import (
    AverageDistanceReferenceSnapshot,
    ExpansionAttempt,
    ExternalReferencePoolSnapshot,
    FactorGraph,
    QualityParetoArchive,
)
from cosmosalpha.pipeline.graph.identity import DuplicateIndexSnapshot

GRAPH_SNAPSHOT_SCHEMA_VERSION = "graph-snapshot/2.0"
GRAPH_ROUND_COMMIT_SCHEMA_VERSION = "graph-round-commit/2.0"
GRAPH_SNAPSHOT_OBJECT_TYPE = "graph_snapshot"
GRAPH_ROUND_COMMIT_OBJECT_TYPE = "graph_round_commit"
GRAPH_ATTEMPT_OBJECT_TYPE = "graph_expansion_attempt"
GRAPH_EVENT_OBJECT_TYPE = "graph_generation_event"


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise CoreValidationError(f"{field_name} must be a non-negative integer")


def _require_finite_non_negative(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CoreValidationError(f"{field_name} must be finite")
    if float(value) < 0.0:
        raise CoreValidationError(f"{field_name} must be non-negative")
    return float(value)


def graph_round_id(run_id: str, round_index: int) -> str:
    """Return the deterministic identity of one committed Graph round."""
    _require_text(run_id, "run_id")
    _require_non_negative_int(round_index, "round_index")
    return (
        "graph-round-"
        + stable_sha256({"run_id": run_id, "round_index": round_index})[:24]
    )


def graph_attempt_id(run_id: str, round_index: int, parent_id: str) -> str:
    """Return the deterministic attempt and Loop cycle identity."""
    _require_text(run_id, "run_id")
    _require_non_negative_int(round_index, "round_index")
    _require_text(parent_id, "parent_id")
    return (
        f"graph-round-{round_index:04d}-attempt-"
        + stable_sha256(
            {"run_id": run_id, "round": round_index, "parent_id": parent_id}
        )[:16]
    )


def graph_initial_cycle_id(run_id: str, batch_index: int = 0) -> str:
    """Return the deterministic identity of one initialization Loop cycle."""
    _require_text(run_id, "run_id")
    _require_non_negative_int(batch_index, "batch_index")
    if batch_index == 0:
        return "graph-initial-" + stable_sha256(run_id)[:16]
    return (
        f"graph-initial-{batch_index:04d}-"
        + stable_sha256({"run_id": run_id, "batch_index": batch_index})[:16]
    )


def graph_initial_cycle_index(run_id: str, cycle_id: str) -> int | None:
    """Return a verified initialization batch index, or ``None`` for other IDs."""
    _require_text(run_id, "run_id")
    _require_text(cycle_id, "cycle_id")
    if cycle_id == graph_initial_cycle_id(run_id):
        return 0
    prefix = "graph-initial-"
    if not cycle_id.startswith(prefix):
        return None
    index_text, separator, _digest = cycle_id.removeprefix(prefix).partition("-")
    if not separator or not index_text.isdecimal():
        return None
    batch_index = int(index_text)
    if batch_index == 0 or cycle_id != graph_initial_cycle_id(run_id, batch_index):
        return None
    return batch_index


def _canonical_graph(graph: FactorGraph) -> FactorGraph:
    nodes = {node_id: graph.nodes[node_id] for node_id in sorted(graph.nodes)}
    events = tuple(sorted(graph.generation_events, key=lambda event: event.event_id))
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    return replace(graph, nodes=nodes, generation_events=events, node_index=node_index)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Complete immutable Graph state for one committed round version."""

    run_id: str
    round_id: str
    round_index: int
    next_round_index: int
    data_snapshot_id: str
    config_snapshot_ref: ArtifactRef
    correlation_snapshot_ref: ArtifactRef
    graph: FactorGraph
    duplicate_index: DuplicateIndexSnapshot
    attempts: tuple[ExpansionAttempt, ...]
    archive: QualityParetoArchive
    child_delta_hv: tuple[tuple[str, float], ...]
    factor_refs: tuple[tuple[str, ObjectRef], ...]
    external_reference_pool: ExternalReferencePoolSnapshot
    average_distance_reference: AverageDistanceReferenceSnapshot
    attempts_used: int
    random_seed: int
    schema_version: str = GRAPH_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and canonicalize all persisted Graph state."""
        _require_text(self.run_id, "run_id")
        _require_text(self.round_id, "round_id")
        if self.round_id != graph_round_id(self.run_id, self.round_index):
            raise CoreValidationError("round_id does not match run_id and round_index")
        _require_non_negative_int(self.round_index, "round_index")
        _require_non_negative_int(self.next_round_index, "next_round_index")
        if self.next_round_index < self.round_index:
            raise CoreValidationError("next_round_index must not precede round_index")
        _require_text(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        if not isinstance(self.correlation_snapshot_ref, ArtifactRef):
            raise CoreValidationError("correlation_snapshot_ref must be ArtifactRef")
        if not isinstance(self.graph, FactorGraph):
            raise CoreValidationError("graph must be FactorGraph")
        if not isinstance(self.duplicate_index, DuplicateIndexSnapshot):
            raise CoreValidationError("duplicate_index must be DuplicateIndexSnapshot")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(attempt, ExpansionAttempt) for attempt in self.attempts
        ):
            raise CoreValidationError("attempts must contain ExpansionAttempt values")
        if not isinstance(self.archive, QualityParetoArchive):
            raise CoreValidationError("archive must be QualityParetoArchive")
        if not isinstance(self.child_delta_hv, tuple):
            raise CoreValidationError("child_delta_hv must be a tuple")
        delta_ids = tuple(node_id for node_id, _ in self.child_delta_hv)
        if len(delta_ids) != len(set(delta_ids)):
            raise CoreValidationError("child_delta_hv must have unique node ids")
        for node_id, delta_hv in self.child_delta_hv:
            _require_text(node_id, "child_delta_hv.node_id")
            _require_finite_non_negative(delta_hv, "child_delta_hv.value")
        if not isinstance(self.factor_refs, tuple):
            raise CoreValidationError("factor_refs must be a tuple")
        factor_ids = tuple(node_id for node_id, _ in self.factor_refs)
        if set(factor_ids) != set(self.graph.nodes) or len(factor_ids) != len(
            set(factor_ids)
        ):
            raise CoreValidationError("factor_refs must cover every Graph node once")
        if not all(
            type(node_id) is str and node_id and isinstance(ref, ObjectRef)
            for node_id, ref in self.factor_refs
        ):
            raise CoreValidationError("factor_refs must contain node/ObjectRef pairs")
        if not isinstance(self.external_reference_pool, ExternalReferencePoolSnapshot):
            raise CoreValidationError(
                "external_reference_pool must be ExternalReferencePoolSnapshot"
            )
        if not isinstance(
            self.average_distance_reference, AverageDistanceReferenceSnapshot
        ):
            raise CoreValidationError(
                "average_distance_reference must be AverageDistanceReferenceSnapshot"
            )
        if (
            self.average_distance_reference.external_pool_snapshot_id
            != self.external_reference_pool.snapshot_id
        ):
            raise CoreValidationError(
                "AverageDistance reference uses a different external pool"
            )
        if not set(self.average_distance_reference.internal_node_ids).issubset(
            self.graph.nodes
        ):
            raise CoreValidationError(
                "internal AverageDistance references must be Graph nodes"
            )
        external_ids = {
            value.node_id for value in self.external_reference_pool.references
        }
        if not set(self.average_distance_reference.external_node_ids).issubset(
            external_ids
        ):
            raise CoreValidationError(
                "external AverageDistance references must be in the frozen pool"
            )
        expected_external_ids = tuple(sorted(external_ids))
        if self.average_distance_reference.mode.value == "strict":
            if (
                self.average_distance_reference.external_node_ids
                != expected_external_ids
            ):
                raise CoreValidationError(
                    "strict AverageDistance references do not match frozen inputs"
                )
        elif self.average_distance_reference.external_node_ids or expected_external_ids:
            raise CoreValidationError(
                "bootstrap AverageDistance references cannot use external inputs"
            )
        if self.average_distance_reference.round_index > self.round_index:
            raise CoreValidationError(
                "AverageDistance reference round cannot follow the Graph snapshot"
            )
        _require_non_negative_int(self.attempts_used, "attempts_used")
        _require_non_negative_int(self.random_seed, "random_seed")
        if self.schema_version != GRAPH_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphSnapshot schema_version")
        object.__setattr__(self, "graph", _canonical_graph(self.graph))
        object.__setattr__(
            self,
            "attempts",
            tuple(sorted(self.attempts, key=lambda attempt: attempt.attempt_id)),
        )
        object.__setattr__(
            self,
            "child_delta_hv",
            tuple(sorted(self.child_delta_hv, key=lambda item: item[0])),
        )
        object.__setattr__(
            self,
            "factor_refs",
            tuple(sorted(self.factor_refs, key=lambda item: item[0])),
        )


@dataclass(frozen=True, slots=True)
class GraphRoundCommit:
    """Atomic marker declaring one Graph and correlation pair committed."""

    run_id: str
    round_id: str
    round_index: int
    data_snapshot_id: str
    config_snapshot_ref: ArtifactRef
    graph_snapshot_ref: ObjectRef
    correlation_snapshot_ref: ArtifactRef
    schema_version: str = GRAPH_ROUND_COMMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the commit marker's immutable cross-snapshot references."""
        _require_text(self.run_id, "run_id")
        _require_text(self.round_id, "round_id")
        _require_non_negative_int(self.round_index, "round_index")
        if self.round_id != graph_round_id(self.run_id, self.round_index):
            raise CoreValidationError("round_id does not match commit identity")
        _require_text(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        if not isinstance(self.graph_snapshot_ref, ObjectRef):
            raise CoreValidationError("graph_snapshot_ref must be ObjectRef")
        if self.graph_snapshot_ref.object_type != GRAPH_SNAPSHOT_OBJECT_TYPE:
            raise CoreValidationError("graph_snapshot_ref has an invalid object type")
        if self.graph_snapshot_ref.object_id != self.round_id:
            raise CoreValidationError("graph_snapshot_ref has an invalid object id")
        if self.graph_snapshot_ref.schema_version != GRAPH_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError("graph_snapshot_ref has an invalid schema")
        if not isinstance(self.correlation_snapshot_ref, ArtifactRef):
            raise CoreValidationError("correlation_snapshot_ref must be ArtifactRef")
        if self.schema_version != GRAPH_ROUND_COMMIT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphRoundCommit schema_version")


@dataclass(frozen=True, slots=True)
class GraphResumeState:
    """Validated state loaded only through an existing commit marker."""

    commit: GraphRoundCommit
    snapshot: GraphSnapshot
    correlation_snapshot: CorrelationSnapshot
    correlation_store: CorrelationStore


class GraphPersistence:
    """Persist immutable Graph state and discover only committed rounds."""

    def __init__(
        self,
        *,
        artifacts: object,
        objects: object,
        correlation_engine: GraphCorrelationEngine | None,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Bind local repositories and an optional test-only fault hook."""
        if not hasattr(artifacts, "put_bytes") or not hasattr(artifacts, "get_bytes"):
            raise CoreValidationError("artifacts must provide byte storage")
        if not all(
            hasattr(objects, method)
            for method in ("put_object", "get_object", "exists", "list_object_ids")
        ):
            raise CoreValidationError("objects must provide Graph persistence methods")
        if fault_hook is not None and not callable(fault_hook):
            raise CoreValidationError("fault_hook must be callable or None")
        self._artifacts = artifacts
        self._objects = objects
        self._correlation_engine = correlation_engine
        self._fault_hook = fault_hook

    def persist_round(
        self,
        *,
        run_id: str,
        round_index: int,
        next_round_index: int,
        data_snapshot_id: str,
        config_snapshot_ref: ArtifactRef,
        graph: FactorGraph,
        duplicate_index: DuplicateIndexSnapshot,
        attempts: tuple[ExpansionAttempt, ...],
        archive: QualityParetoArchive,
        child_delta_hv: Mapping[str, float],
        factor_refs: Mapping[str, ObjectRef],
        external_reference_pool: ExternalReferencePoolSnapshot,
        average_distance_reference: AverageDistanceReferenceSnapshot,
        attempts_used: int,
        random_seed: int,
        correlation_store: CorrelationStore,
    ) -> GraphRoundCommit:
        """Write snapshots, then atomically publish the round commit marker."""
        round_id = graph_round_id(run_id, round_index)
        for attempt in attempts:
            _put_immutable(
                self._objects,
                GRAPH_ATTEMPT_OBJECT_TYPE,
                attempt.attempt_id,
                attempt,
            )
        self._fault("after_evaluation")
        for event in graph.generation_events:
            _put_immutable(
                self._objects,
                GRAPH_EVENT_OBJECT_TYPE,
                event.event_id,
                event,
            )
        self._fault("after_generation")

        correlation_snapshot = correlation_store.persist(
            self._artifacts,
            snapshot_version=round_id,
        )
        if correlation_snapshot.metadata_ref is None:
            raise CoreValidationError("correlation snapshot must have metadata_ref")
        self._fault("after_correlation_snapshot")
        snapshot = GraphSnapshot(
            run_id=run_id,
            round_id=round_id,
            round_index=round_index,
            next_round_index=next_round_index,
            data_snapshot_id=data_snapshot_id,
            config_snapshot_ref=config_snapshot_ref,
            correlation_snapshot_ref=correlation_snapshot.metadata_ref,
            graph=graph,
            duplicate_index=duplicate_index,
            attempts=attempts,
            archive=archive,
            child_delta_hv=tuple(child_delta_hv.items()),
            factor_refs=tuple(factor_refs.items()),
            external_reference_pool=external_reference_pool,
            average_distance_reference=average_distance_reference,
            attempts_used=attempts_used,
            random_seed=random_seed,
        )
        graph_snapshot_ref = _put_immutable(
            self._objects,
            GRAPH_SNAPSHOT_OBJECT_TYPE,
            round_id,
            snapshot,
        )
        self._fault("after_graph_snapshot")
        commit = GraphRoundCommit(
            run_id=run_id,
            round_id=round_id,
            round_index=round_index,
            data_snapshot_id=data_snapshot_id,
            config_snapshot_ref=config_snapshot_ref,
            graph_snapshot_ref=graph_snapshot_ref,
            correlation_snapshot_ref=correlation_snapshot.metadata_ref,
        )
        self._fault("before_commit")
        _put_immutable(
            self._objects,
            GRAPH_ROUND_COMMIT_OBJECT_TYPE,
            round_id,
            commit,
        )
        return commit

    def load_latest(
        self,
        *,
        run_id: str,
        data_snapshot_id: str,
        config_snapshot_ref: ArtifactRef,
        expected_protocol: GraphCorrelationProtocol,
        random_seed: int,
    ) -> GraphResumeState | None:
        """Load the newest committed round and reject incompatible state."""
        _require_text(run_id, "run_id")
        _require_text(data_snapshot_id, "data_snapshot_id")
        if not isinstance(config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        if not isinstance(expected_protocol, GraphCorrelationProtocol):
            raise CoreValidationError(
                "expected_protocol must be GraphCorrelationProtocol"
            )
        _require_non_negative_int(random_seed, "random_seed")
        commits = []
        for object_id in self._objects.list_object_ids(GRAPH_ROUND_COMMIT_OBJECT_TYPE):
            commit_ref = ObjectRef(
                GRAPH_ROUND_COMMIT_OBJECT_TYPE,
                object_id,
                GRAPH_ROUND_COMMIT_SCHEMA_VERSION,
            )
            commit = self._objects.get_object(commit_ref, GraphRoundCommit)
            if commit.run_id == run_id:
                commits.append(commit)
        if not commits:
            return None
        commit = max(commits, key=lambda item: item.round_index)
        if commit.data_snapshot_id != data_snapshot_id:
            raise CoreValidationError("Graph resume data snapshot mismatch")
        if commit.config_snapshot_ref != config_snapshot_ref:
            raise CoreValidationError("Graph resume config snapshot mismatch")
        snapshot = self._objects.get_object(commit.graph_snapshot_ref, GraphSnapshot)
        if snapshot.run_id != run_id:
            raise CoreValidationError("Graph snapshot run mismatch")
        if snapshot.data_snapshot_id != data_snapshot_id:
            raise CoreValidationError("Graph snapshot data snapshot mismatch")
        if snapshot.config_snapshot_ref != config_snapshot_ref:
            raise CoreValidationError("Graph snapshot config snapshot mismatch")
        if snapshot.round_id != commit.round_id:
            raise CoreValidationError("Graph snapshot and commit round mismatch")
        if snapshot.correlation_snapshot_ref != commit.correlation_snapshot_ref:
            raise CoreValidationError(
                "Graph snapshot and commit correlation reference mismatch"
            )
        if snapshot.random_seed != random_seed:
            raise CoreValidationError("Graph resume random seed mismatch")
        correlation_snapshot = from_json(
            self._artifacts.get_bytes(commit.correlation_snapshot_ref),
            CorrelationSnapshot,
        )
        if correlation_snapshot.snapshot_version != commit.round_id:
            raise CoreValidationError("correlation snapshot and commit round mismatch")
        if correlation_snapshot.protocol != expected_protocol:
            raise CoreValidationError("Graph resume protocol mismatch")
        try:
            if self._correlation_engine is None or not hasattr(
                self._correlation_engine, "load"
            ):
                correlation_store = CorrelationStore.load(
                    self._artifacts,
                    correlation_snapshot,
                    expected_protocol=expected_protocol,
                )
            else:
                correlation_store = self._correlation_engine.load(correlation_snapshot)
        except CorrelationProtocolMismatch as exc:
            raise CoreValidationError("Graph resume protocol mismatch") from exc
        return GraphResumeState(
            commit=commit,
            snapshot=snapshot,
            correlation_snapshot=correlation_snapshot,
            correlation_store=correlation_store,
        )

    def _fault(self, stage: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage)


def _put_immutable(
    objects: object,
    object_type: str,
    object_id: str,
    value: object,
) -> ObjectRef:
    """Put one object idempotently while rejecting conflicting rewrites."""
    schema_version = getattr(value, "schema_version", None)
    if type(schema_version) is not str:
        raise CoreValidationError("persisted value must expose schema_version")
    ref = ObjectRef(object_type, object_id, schema_version)
    if objects.exists(ref):
        existing = objects.get_object(ref, type(value))
        if existing != value:
            raise CoreValidationError(
                f"immutable object conflict: {object_type}/{object_id}"
            )
        return ref
    stored = objects.put_object(object_type, object_id, value)
    if stored != ref:
        raise CoreValidationError("object repository returned an unexpected reference")
    return ref


__all__ = [
    "GRAPH_ATTEMPT_OBJECT_TYPE",
    "GRAPH_EVENT_OBJECT_TYPE",
    "GRAPH_ROUND_COMMIT_OBJECT_TYPE",
    "GRAPH_ROUND_COMMIT_SCHEMA_VERSION",
    "GRAPH_SNAPSHOT_OBJECT_TYPE",
    "GRAPH_SNAPSHOT_SCHEMA_VERSION",
    "GraphPersistence",
    "GraphResumeState",
    "GraphRoundCommit",
    "GraphSnapshot",
    "graph_attempt_id",
    "graph_initial_cycle_id",
    "graph_round_id",
]
