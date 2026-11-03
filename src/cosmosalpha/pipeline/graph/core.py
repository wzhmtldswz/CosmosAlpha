"""Immutable domain objects for Graph Search v1."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from cosmosalpha.core import ArtifactRef, CoreValidationError, ObjectRef, stable_sha256

GRAPH_NODE_SCHEMA_VERSION = "graph-node/2.0"
GENERATION_EVENT_SCHEMA_VERSION = "generation-event/1.0"
EXPANSION_ATTEMPT_SCHEMA_VERSION = "expansion-attempt/1.0"
QUALITY_POINT_SCHEMA_VERSION = "quality-point/2.0"
QUALITY_ARCHIVE_SCHEMA_VERSION = "quality-archive/2.0"
FACTOR_GRAPH_SCHEMA_VERSION = "factor-graph/1.0"
DUPLICATE_INDEX_ENTRY_SCHEMA_VERSION = "duplicate-index-entry/1.0"
DUPLICATE_INDEX_SNAPSHOT_SCHEMA_VERSION = "duplicate-index-snapshot/1.0"
EXTERNAL_REFERENCE_FACTOR_SCHEMA_VERSION = "external-reference-factor/1.0"
EXTERNAL_REFERENCE_POOL_SCHEMA_VERSION = "external-reference-pool/1.0"
AVERAGE_DISTANCE_REFERENCE_SCHEMA_VERSION = "average-distance-reference/1.0"

type DenseMatrix = tuple[tuple[float, ...], ...]


class ExpansionAttemptStatus(StrEnum):
    """Machine-readable outcome of one Graph expansion attempt."""

    SUCCESS = "success"
    NO_CHILD = "no_child"
    DUPLICATE = "duplicate"
    INVALID_EXPRESSION = "invalid_expression"
    EVALUATION_FAILED = "evaluation_failed"
    LLM_FAILED = "llm_failed"


class AverageDistanceReferenceMode(StrEnum):
    """Whether a round uses strict quality anchors or the cold-start fallback."""

    BOOTSTRAP = "bootstrap"
    STRICT = "strict"


_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be a non-empty string without surrounding whitespace"
        )


def _require_schema_version(value: object, field_name: str) -> None:
    _require_text(value, field_name)
    if (
        re.fullmatch(r"[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value)
        is None
    ):
        raise CoreValidationError(
            f"{field_name} must match '<schema-name>/<major>.<minor>'"
        )


def _require_finite_float(value: object, field_name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise CoreValidationError(f"{field_name} must be a finite float")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise CoreValidationError(f"{field_name} must be a non-negative integer")


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_text(item, f"{field_name}[{index}]")


def _validate_matrix(
    value: object,
    field_name: str,
    *,
    size: int,
) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise CoreValidationError(f"{field_name} must be a {size}x{size} tuple matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != size:
            raise CoreValidationError(
                f"{field_name}[{row_index}] must contain {size} values"
            )
        for column_index, item in enumerate(row):
            _require_finite_float(item, f"{field_name}[{row_index}][{column_index}]")


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One unique, technically valid and evaluated factor in the Graph."""

    node_id: str
    factor_id: str
    expression: str
    expression_hash: str
    experiment_ref: ObjectRef | None
    evaluation_result_ref: ObjectRef
    evaluated_factor_values_ref: ArtifactRef
    data_snapshot_id: str
    evaluation_protocol_id: str
    preprocessing_protocol_id: str
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_std_calculation_version: str
    top20_return: float
    orientation: int
    self_quality: float
    fertility: float
    average_distance: float
    expansion_attempt_count: int = 0
    valid_child_count: int = 0
    schema_version: str = GRAPH_NODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate node identity, lineage references, metrics and counters."""
        for field_name in (
            "node_id",
            "factor_id",
            "expression",
            "data_snapshot_id",
            "evaluation_protocol_id",
            "preprocessing_protocol_id",
            "rank_ic_std_calculation_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        if not _HEX_DIGEST_PATTERN.fullmatch(self.expression_hash):
            raise CoreValidationError(
                "expression_hash must be a lowercase SHA-256 digest"
            )
        if self.experiment_ref is not None and not isinstance(
            self.experiment_ref, ObjectRef
        ):
            raise CoreValidationError("experiment_ref must be ObjectRef or None")
        if not isinstance(self.evaluation_result_ref, ObjectRef):
            raise CoreValidationError("evaluation_result_ref must be ObjectRef")
        if not isinstance(self.evaluated_factor_values_ref, ArtifactRef):
            raise CoreValidationError("evaluated_factor_values_ref must be ArtifactRef")
        for field_name in (
            "rank_ic_mean",
            "rank_ic_std",
            "top20_return",
            "self_quality",
            "fertility",
            "average_distance",
        ):
            _require_finite_float(getattr(self, field_name), field_name)
        if self.rank_ic_std < 0.0:
            raise CoreValidationError("rank_ic_std must be non-negative")
        if type(self.orientation) is not int or self.orientation not in (-1, 1):
            raise CoreValidationError("orientation must be either -1 or 1")
        _require_non_negative_int(
            self.expansion_attempt_count, "expansion_attempt_count"
        )
        _require_non_negative_int(self.valid_child_count, "valid_child_count")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != GRAPH_NODE_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphNode schema_version")


@dataclass(frozen=True, slots=True)
class ExternalReferenceFactor:
    """One run-frozen historical factor used only by AverageDistance."""

    node_id: str
    factor_id: str
    expression: str
    evaluation_result_ref: ObjectRef
    evaluated_factor_values_ref: ArtifactRef
    data_snapshot_id: str
    evaluation_protocol_id: str
    preprocessing_protocol_id: str
    rank_ic_mean: float
    rank_ic_std: float
    top20_return: float
    rank_ic_std_calculation_version: str
    schema_version: str = EXTERNAL_REFERENCE_FACTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identity, protocol compatibility fields and quality values."""
        for field_name in (
            "node_id",
            "factor_id",
            "expression",
            "data_snapshot_id",
            "evaluation_protocol_id",
            "preprocessing_protocol_id",
            "rank_ic_std_calculation_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        if not isinstance(self.evaluation_result_ref, ObjectRef):
            raise CoreValidationError("evaluation_result_ref must be ObjectRef")
        if not isinstance(self.evaluated_factor_values_ref, ArtifactRef):
            raise CoreValidationError("evaluated_factor_values_ref must be ArtifactRef")
        for field_name in ("rank_ic_mean", "rank_ic_std", "top20_return"):
            _require_finite_float(getattr(self, field_name), field_name)
        if self.rank_ic_std < 0.0:
            raise CoreValidationError("rank_ic_std must be non-negative")
        if self.schema_version != EXTERNAL_REFERENCE_FACTOR_SCHEMA_VERSION:
            raise CoreValidationError("invalid ExternalReferenceFactor schema_version")

    @classmethod
    def from_node(cls, node: GraphNode) -> ExternalReferenceFactor:
        """Copy the distance-relevant immutable fields from a GraphNode v2 seed."""
        if not isinstance(node, GraphNode):
            raise CoreValidationError("node must be GraphNode")
        return cls(
            node_id=node.node_id,
            factor_id=node.factor_id,
            expression=node.expression,
            evaluation_result_ref=node.evaluation_result_ref,
            evaluated_factor_values_ref=node.evaluated_factor_values_ref,
            data_snapshot_id=node.data_snapshot_id,
            evaluation_protocol_id=node.evaluation_protocol_id,
            preprocessing_protocol_id=node.preprocessing_protocol_id,
            rank_ic_mean=node.rank_ic_mean,
            rank_ic_std=node.rank_ic_std,
            top20_return=node.top20_return,
            rank_ic_std_calculation_version=node.rank_ic_std_calculation_version,
        )


@dataclass(frozen=True, slots=True)
class ExternalReferencePoolSnapshot:
    """Run-start external candidates and the bounded selected anchor subset."""

    candidate_node_ids: tuple[str, ...]
    candidate_content_digest: str
    references: tuple[ExternalReferenceFactor, ...]
    minimum_abs_rank_ic: float
    capacity: int
    schema_version: str = EXTERNAL_REFERENCE_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate deterministic ordering, admission gate and capacity."""
        _require_string_tuple(self.candidate_node_ids, "candidate_node_ids")
        if len(self.candidate_node_ids) != len(set(self.candidate_node_ids)):
            raise CoreValidationError("candidate_node_ids must be unique")
        if not _HEX_DIGEST_PATTERN.fullmatch(self.candidate_content_digest):
            raise CoreValidationError(
                "candidate_content_digest must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.references, tuple) or not all(
            isinstance(value, ExternalReferenceFactor) for value in self.references
        ):
            raise CoreValidationError(
                "references must contain ExternalReferenceFactor values"
            )
        reference_ids = tuple(value.node_id for value in self.references)
        if len(reference_ids) != len(set(reference_ids)):
            raise CoreValidationError("external references must be unique")
        if not set(reference_ids).issubset(self.candidate_node_ids):
            raise CoreValidationError("external references must be candidates")
        _require_finite_float(self.minimum_abs_rank_ic, "minimum_abs_rank_ic")
        if self.minimum_abs_rank_ic < 0.0:
            raise CoreValidationError("minimum_abs_rank_ic must be non-negative")
        if type(self.capacity) is not int or self.capacity < 1:
            raise CoreValidationError("capacity must be a positive integer")
        if len(self.references) > self.capacity:
            raise CoreValidationError("external references exceed capacity")
        if any(
            abs(value.rank_ic_mean) < self.minimum_abs_rank_ic
            for value in self.references
        ):
            raise CoreValidationError("external reference failed the RankIC gate")
        if self.schema_version != EXTERNAL_REFERENCE_POOL_SCHEMA_VERSION:
            raise CoreValidationError(
                "invalid ExternalReferencePoolSnapshot schema_version"
            )

    @property
    def snapshot_id(self) -> str:
        """Return a content-derived identity for resume and audit."""
        return "external-reference-pool-" + stable_sha256(self)[:24]


@dataclass(frozen=True, slots=True)
class AverageDistanceReferenceSnapshot:
    """Exact reference identities used by every GraphNode in one round state."""

    round_index: int
    archive_snapshot_id: str
    external_pool_snapshot_id: str
    mode: AverageDistanceReferenceMode
    internal_node_ids: tuple[str, ...]
    external_node_ids: tuple[str, ...]
    reference_node_ids: tuple[str, ...]
    schema_version: str = AVERAGE_DISTANCE_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the partition and stable reference ordering."""
        _require_non_negative_int(self.round_index, "round_index")
        _require_text(self.archive_snapshot_id, "archive_snapshot_id")
        _require_text(self.external_pool_snapshot_id, "external_pool_snapshot_id")
        if isinstance(self.mode, str) and not isinstance(
            self.mode, AverageDistanceReferenceMode
        ):
            try:
                object.__setattr__(
                    self, "mode", AverageDistanceReferenceMode(self.mode)
                )
            except ValueError as error:
                raise CoreValidationError("unsupported reference mode") from error
        if not isinstance(self.mode, AverageDistanceReferenceMode):
            raise CoreValidationError("mode must be AverageDistanceReferenceMode")
        for field_name in (
            "internal_node_ids",
            "external_node_ids",
            "reference_node_ids",
        ):
            values = getattr(self, field_name)
            _require_string_tuple(values, field_name)
            if len(values) != len(set(values)) or values != tuple(sorted(values)):
                raise CoreValidationError(f"{field_name} must be unique and sorted")
        expected = tuple(
            sorted(set(self.internal_node_ids) | set(self.external_node_ids))
        )
        if self.reference_node_ids != expected:
            raise CoreValidationError(
                "reference_node_ids must equal the internal/external union"
            )
        if (
            self.mode is AverageDistanceReferenceMode.BOOTSTRAP
            and self.external_node_ids
        ):
            raise CoreValidationError(
                "bootstrap references cannot include external IDs"
            )
        if self.schema_version != AVERAGE_DISTANCE_REFERENCE_SCHEMA_VERSION:
            raise CoreValidationError(
                "invalid AverageDistanceReferenceSnapshot schema_version"
            )

    @property
    def snapshot_id(self) -> str:
        """Return a content-derived identity for persisted Graph state."""
        return "average-distance-reference-" + stable_sha256(self)[:24]


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """A persisted parent-to-new-child mutation provenance event."""

    event_id: str
    parent_ids: tuple[str, ...]
    child_id: str
    operation: str
    attempt_id: str
    guidance_ref: ObjectRef | None
    experiment_ref: ObjectRef | None
    created_at: str
    schema_version: str = GENERATION_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate v1 single-parent generation provenance."""
        for field_name in (
            "event_id",
            "child_id",
            "operation",
            "attempt_id",
            "created_at",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_string_tuple(self.parent_ids, "parent_ids")
        if len(self.parent_ids) != 1:
            raise CoreValidationError(
                "Graph v1 GenerationEvent requires exactly one parent"
            )
        if self.operation != "mutation":
            raise CoreValidationError(
                "Graph v1 GenerationEvent operation must be mutation"
            )
        for field_name in ("guidance_ref", "experiment_ref"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ObjectRef):
                raise CoreValidationError(f"{field_name} must be ObjectRef or None")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != GENERATION_EVENT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GenerationEvent schema_version")


@dataclass(frozen=True, slots=True)
class ExpansionAttempt:
    """One complete logical Graph-to-Loop expansion attempt."""

    attempt_id: str
    parent_id: str
    round_id: int
    status: ExpansionAttemptStatus
    child_ids: tuple[str, ...]
    outcome_codes: tuple[str, ...]
    fertility_outcome: float
    guidance_ref: ObjectRef | None
    schema_version: str = EXPANSION_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate attempt budget identity and outcome invariants."""
        for field_name in ("attempt_id", "parent_id"):
            _require_text(getattr(self, field_name), field_name)
        _require_non_negative_int(self.round_id, "round_id")
        if isinstance(self.status, str) and not isinstance(
            self.status, ExpansionAttemptStatus
        ):
            try:
                object.__setattr__(self, "status", ExpansionAttemptStatus(self.status))
            except ValueError as error:
                raise CoreValidationError(
                    f"unsupported expansion attempt status: {self.status!r}"
                ) from error
        if not isinstance(self.status, ExpansionAttemptStatus):
            raise CoreValidationError(
                f"unsupported expansion attempt status: {self.status!r}"
            )
        _require_string_tuple(self.child_ids, "child_ids")
        if len(self.child_ids) != len(set(self.child_ids)):
            raise CoreValidationError("child_ids must be unique")
        _require_string_tuple(self.outcome_codes, "outcome_codes")
        _require_finite_float(self.fertility_outcome, "fertility_outcome")
        if self.fertility_outcome < 0.0:
            raise CoreValidationError("fertility_outcome must be non-negative")
        if self.status is ExpansionAttemptStatus.SUCCESS and not self.child_ids:
            raise CoreValidationError("success attempt must contain at least one child")
        if self.status is not ExpansionAttemptStatus.SUCCESS and self.child_ids:
            raise CoreValidationError("non-success attempt must not contain child_ids")
        if self.guidance_ref is not None and not isinstance(
            self.guidance_ref, ObjectRef
        ):
            raise CoreValidationError("guidance_ref must be ObjectRef or None")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != EXPANSION_ATTEMPT_SCHEMA_VERSION:
            raise CoreValidationError("invalid ExpansionAttempt schema_version")


@dataclass(frozen=True, slots=True)
class QualityPoint:
    """One factor's three higher-is-better intrinsic quality objectives."""

    node_id: str
    objectives: tuple[float, float, float]
    schema_version: str = QUALITY_POINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the fixed three-objective quality vector."""
        _require_text(self.node_id, "node_id")
        if not isinstance(self.objectives, tuple) or len(self.objectives) != 3:
            raise CoreValidationError("objectives must be a three-item tuple")
        for index, value in enumerate(self.objectives):
            _require_finite_float(value, f"objectives[{index}]")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != QUALITY_POINT_SCHEMA_VERSION:
            raise CoreValidationError("invalid QualityPoint schema_version")


@dataclass(frozen=True, slots=True)
class QualityParetoArchive:
    """Current Graph-run Pareto points, without graph edges or search rewards."""

    points: tuple[QualityPoint, ...]
    schema_version: str = QUALITY_ARCHIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate unique archive node identities and stable point ordering."""
        if not isinstance(self.points, tuple) or not all(
            isinstance(point, QualityPoint) for point in self.points
        ):
            raise CoreValidationError("points must be a tuple of QualityPoint")
        node_ids = tuple(point.node_id for point in self.points)
        if len(node_ids) != len(set(node_ids)):
            raise CoreValidationError("archive points must have unique node_id values")
        object.__setattr__(
            self,
            "points",
            tuple(sorted(self.points, key=lambda point: point.node_id)),
        )
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != QUALITY_ARCHIVE_SCHEMA_VERSION:
            raise CoreValidationError("invalid QualityParetoArchive schema_version")


@dataclass(frozen=True, slots=True)
class FactorGraph:
    """Graph nodes, provenance and JSON-serializable dense neighborhood facts."""

    nodes: dict[str, GraphNode]
    generation_events: tuple[GenerationEvent, ...]
    w_structure: DenseMatrix
    w_correlation: DenseMatrix
    w_generation: DenseMatrix
    node_index: dict[str, int]
    schema_version: str = FACTOR_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate graph identity/index alignment and dense matrix shapes."""
        if not isinstance(self.nodes, dict) or not all(
            isinstance(node, GraphNode) for node in self.nodes.values()
        ):
            raise CoreValidationError("nodes must be a dict of GraphNode")
        if set(self.nodes) != {node.node_id for node in self.nodes.values()}:
            raise CoreValidationError("nodes keys must match GraphNode.node_id values")
        if not isinstance(self.generation_events, tuple) or not all(
            isinstance(event, GenerationEvent) for event in self.generation_events
        ):
            raise CoreValidationError(
                "generation_events must be a tuple of GenerationEvent"
            )
        if not isinstance(self.node_index, dict):
            raise CoreValidationError("node_index must be a dict")
        if set(self.node_index) != set(self.nodes):
            raise CoreValidationError("node_index must contain exactly all node ids")
        indices = tuple(self.node_index.values())
        if any(type(index) is not int or index < 0 for index in indices):
            raise CoreValidationError("node_index values must be non-negative integers")
        if set(indices) != set(range(len(self.nodes))):
            raise CoreValidationError(
                "node_index must be a contiguous zero-based index"
            )
        size = len(self.nodes)
        for field_name in ("w_structure", "w_correlation", "w_generation"):
            _validate_matrix(getattr(self, field_name), field_name, size=size)
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != FACTOR_GRAPH_SCHEMA_VERSION:
            raise CoreValidationError("invalid FactorGraph schema_version")


__all__ = [
    "AVERAGE_DISTANCE_REFERENCE_SCHEMA_VERSION",
    "EXTERNAL_REFERENCE_FACTOR_SCHEMA_VERSION",
    "EXTERNAL_REFERENCE_POOL_SCHEMA_VERSION",
    "AverageDistanceReferenceMode",
    "AverageDistanceReferenceSnapshot",
    "DenseMatrix",
    "ExpansionAttempt",
    "ExpansionAttemptStatus",
    "ExternalReferenceFactor",
    "ExternalReferencePoolSnapshot",
    "FactorGraph",
    "GenerationEvent",
    "GraphNode",
    "QualityParetoArchive",
    "QualityPoint",
]
