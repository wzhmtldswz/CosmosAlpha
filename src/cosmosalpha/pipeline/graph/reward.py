"""Static quality, Pareto, fertility, and novelty semantics for Graph v1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cosmosalpha.core import CoreValidationError
from cosmosalpha.evaluation.correlation import CorrelationStore
from cosmosalpha.pipeline.graph.config import (
    GraphAverageDistanceConfig,
    GraphNormalizationAnchor,
    GraphQualityConfig,
)
from cosmosalpha.pipeline.graph.core import (
    ExpansionAttempt,
    ExpansionAttemptStatus,
    GraphNode,
    QualityParetoArchive,
    QualityPoint,
)
from cosmosalpha.pipeline.graph.structure import StructureFactStore

type QualityVector = tuple[float, float, float]


def _require_finite(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CoreValidationError(f"{field_name} must be finite")
    return float(value)


def _require_quality_vector(value: object, field_name: str) -> QualityVector:
    if not isinstance(value, tuple) or len(value) != 3:
        raise CoreValidationError(f"{field_name} must contain three values")
    return tuple(
        _require_finite(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def oriented_quality_objectives(
    rank_ic_mean: float,
    rank_ic_std: float,
    top20_return: float,
) -> QualityVector:
    """Return raw Quality v2 objectives in higher-is-better form."""
    mean = _require_finite(rank_ic_mean, "rank_ic_mean")
    standard_deviation = _require_finite(rank_ic_std, "rank_ic_std")
    top_return = _require_finite(top20_return, "top20_return")
    if standard_deviation < 0.0:
        raise CoreValidationError("rank_ic_std must be non-negative")
    return (abs(mean), -standard_deviation, top_return)


def quality_point_from_node(node: GraphNode) -> QualityPoint:
    """Build a raw quality point from one evaluated GraphNode.

    ``GraphNode.top20_return`` is already direction-aligned. The signless
    sample standard deviation is negated so all internal objectives remain
    higher-is-better.
    """
    if not isinstance(node, GraphNode):
        raise CoreValidationError("node must be GraphNode")
    return QualityPoint(
        node_id=node.node_id,
        objectives=(
            abs(_require_finite(node.rank_ic_mean, "rank_ic_mean")),
            -_require_finite(node.rank_ic_std, "rank_ic_std"),
            _require_finite(node.top20_return, "top20_return"),
        ),
    )


def clip_normalize(value: float, anchor: GraphNormalizationAnchor) -> float:
    """Map one finite value to ``[0, 1]`` using fixed clipping anchors."""
    if not isinstance(anchor, GraphNormalizationAnchor):
        raise CoreValidationError("anchor must be GraphNormalizationAnchor")
    numeric = _require_finite(value, "value")
    return min(1.0, max(0.0, (numeric - anchor.lower) / (anchor.upper - anchor.lower)))


def normalize_quality_point(
    point: QualityPoint, config: GraphQualityConfig
) -> QualityVector:
    """Normalize raw quality objectives without using batch statistics."""
    if not isinstance(point, QualityPoint):
        raise CoreValidationError("point must be QualityPoint")
    if not isinstance(config, GraphQualityConfig):
        raise CoreValidationError("config must be GraphQualityConfig")
    return (
        clip_normalize(point.objectives[0], config.rank_ic),
        1.0 - clip_normalize(-point.objectives[1], config.rank_ic_std),
        clip_normalize(point.objectives[2], config.top20_return),
    )


def self_quality(point: QualityPoint, config: GraphQualityConfig) -> float:
    """Return the equal-weighted normalized intrinsic quality score."""
    return sum(normalize_quality_point(point, config)) / 3.0


def self_quality_from_node(node: GraphNode, config: GraphQualityConfig) -> float:
    """Return SelfQuality for one GraphNode using the fixed quality anchors."""
    return self_quality(quality_point_from_node(node), config)


def pareto_dominates(left: QualityPoint, right: QualityPoint) -> bool:
    """Return whether raw higher-is-better ``left`` dominates ``right``."""
    if not isinstance(left, QualityPoint) or not isinstance(right, QualityPoint):
        raise CoreValidationError("Pareto inputs must be QualityPoint")
    return all(
        left_value >= right_value
        for left_value, right_value in zip(
            left.objectives, right.objectives, strict=True
        )
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(
            left.objectives, right.objectives, strict=True
        )
    )


def pareto_archive(points: Sequence[QualityPoint]) -> QualityParetoArchive:
    """Return the deterministic raw-objective Pareto frontier."""
    if not isinstance(points, Sequence):
        raise CoreValidationError("points must be a sequence")
    values = tuple(points)
    if not all(isinstance(point, QualityPoint) for point in values):
        raise CoreValidationError("every point must be QualityPoint")
    node_ids = tuple(point.node_id for point in values)
    if len(node_ids) != len(set(node_ids)):
        raise CoreValidationError("Pareto points must have unique node_id values")
    frontier = tuple(
        point
        for point in values
        if not any(
            other.node_id != point.node_id and pareto_dominates(other, point)
            for other in values
        )
    )
    return QualityParetoArchive(points=frontier)


def batch_archive_commit(
    pre_archive: QualityParetoArchive,
    batch_points: Sequence[QualityPoint],
) -> QualityParetoArchive:
    """Commit a whole child batch after all birth-time rewards are computed."""
    if not isinstance(pre_archive, QualityParetoArchive):
        raise CoreValidationError("pre_archive must be QualityParetoArchive")
    if not isinstance(batch_points, Sequence):
        raise CoreValidationError("batch_points must be a sequence")
    points = tuple(pre_archive.points) + tuple(batch_points)
    return pareto_archive(points)


def hypervolume_3d(points: Sequence[QualityVector]) -> float:
    """Calculate exact dominated volume in the normalized unit cube."""
    if not isinstance(points, Sequence):
        raise CoreValidationError("points must be a sequence")
    values = tuple(_require_quality_vector(point, "point") for point in points)
    if any(value < 0.0 or value > 1.0 for point in values for value in point):
        raise CoreValidationError("normalized hypervolume points must be in [0, 1]")
    if not values:
        return 0.0

    x_boundaries = tuple(sorted({0.0, *(point[0] for point in values)}))
    volume = 0.0
    for left, right in zip(x_boundaries, x_boundaries[1:], strict=False):
        if right <= left:
            continue
        active = tuple(point for point in values if point[0] >= right)
        if not active:
            continue
        y_boundaries = tuple(sorted({0.0, *(point[1] for point in active)}))
        area = 0.0
        for lower, upper in zip(y_boundaries, y_boundaries[1:], strict=False):
            if upper <= lower:
                continue
            z_max = max(
                (point[2] for point in active if point[1] >= upper),
                default=0.0,
            )
            area += (upper - lower) * z_max
        volume += (right - left) * area
    return volume


def quality_hypervolume(
    points: Sequence[QualityPoint], config: GraphQualityConfig
) -> float:
    """Calculate Hypervolume after fixed-anchor normalization."""
    if not isinstance(config, GraphQualityConfig):
        raise CoreValidationError("config must be GraphQualityConfig")
    return hypervolume_3d(
        tuple(normalize_quality_point(point, config) for point in points)
    )


def birth_time_delta_hv(
    pre_archive: QualityParetoArchive,
    child: QualityPoint,
    config: GraphQualityConfig,
) -> float:
    """Calculate one child's independent birth-time Hypervolume gain."""
    if not isinstance(pre_archive, QualityParetoArchive):
        raise CoreValidationError("pre_archive must be QualityParetoArchive")
    if not isinstance(child, QualityPoint):
        raise CoreValidationError("child must be QualityPoint")
    pre_hv = quality_hypervolume(pre_archive.points, config)
    post_hv = quality_hypervolume(tuple(pre_archive.points) + (child,), config)
    delta_hv = post_hv - pre_hv
    if math.isclose(post_hv, pre_hv, rel_tol=1e-12, abs_tol=1e-15):
        return 0.0
    if delta_hv >= 0.0:
        return delta_hv
    raise CoreValidationError("birth-time DeltaHV decreased beyond numeric tolerance")


def batch_birth_time_delta_hv(
    pre_archive: QualityParetoArchive,
    children: Sequence[QualityPoint],
    config: GraphQualityConfig,
) -> tuple[tuple[str, float], ...]:
    """Calculate all child gains against one frozen archive, order-independently."""
    if not isinstance(children, Sequence):
        raise CoreValidationError("children must be a sequence")
    values = tuple(children)
    if not all(isinstance(child, QualityPoint) for child in values):
        raise CoreValidationError("children must contain QualityPoint values")
    if len({child.node_id for child in values}) != len(values):
        raise CoreValidationError("children must have unique node_id values")
    result = {
        child.node_id: birth_time_delta_hv(pre_archive, child, config)
        for child in values
    }
    return tuple(sorted(result.items()))


def top3_mean(values: Sequence[float]) -> float:
    """Return Top3Mean with missing observations represented by zero padding."""
    if not isinstance(values, Sequence):
        raise CoreValidationError("values must be a sequence")
    finite_values = tuple(
        _require_finite(value, "fertility outcome") for value in values
    )
    if any(value < 0.0 for value in finite_values):
        raise CoreValidationError("fertility outcomes must be non-negative")
    return sum(sorted(finite_values, reverse=True)[:3]) / 3.0


def attempt_fertility_outcome(
    attempt: ExpansionAttempt,
    child_delta_hv: Mapping[str, float],
) -> float:
    """Return one attempt's max-child ΔHV, or zero for failed attempts."""
    if not isinstance(attempt, ExpansionAttempt):
        raise CoreValidationError("attempt must be ExpansionAttempt")
    if not isinstance(child_delta_hv, Mapping):
        raise CoreValidationError("child_delta_hv must be a mapping")
    if attempt.status is not ExpansionAttemptStatus.SUCCESS:
        return 0.0
    values: list[float] = []
    for child_id in attempt.child_ids:
        if child_id not in child_delta_hv:
            raise CoreValidationError(f"missing DeltaHV for child {child_id}")
        delta_hv = _require_finite(child_delta_hv[child_id], child_id)
        if delta_hv < 0.0:
            raise CoreValidationError(f"{child_id} DeltaHV must be non-negative")
        values.append(delta_hv)
    return max(values, default=0.0)


def fertility_from_attempts(
    parent_id: str,
    attempts: Sequence[ExpansionAttempt],
    child_delta_hv: Mapping[str, float],
) -> float:
    """Return a parent's attempt-level Top3Mean fertility."""
    if type(parent_id) is not str or not parent_id:
        raise CoreValidationError("parent_id must be a non-empty string")
    if not isinstance(attempts, Sequence):
        raise CoreValidationError("attempts must be a sequence")
    values = [
        attempt_fertility_outcome(attempt, child_delta_hv)
        for attempt in attempts
        if attempt.parent_id == parent_id
    ]
    return top3_mean(values)


def _knn_mean(distances: Sequence[tuple[float, str]], knn_k: int) -> float:
    """Average the closest available deterministic neighbors."""
    ordered = sorted(distances, key=lambda item: (item[0], item[1]))[:knn_k]
    return 0.0 if not ordered else sum(item[0] for item in ordered) / len(ordered)


def average_distance(
    node_id: str,
    *,
    reference_node_ids: Sequence[str],
    structure_facts: StructureFactStore,
    correlation_store: CorrelationStore,
    node_to_factor_id: Mapping[str, str],
    knn_k: int,
) -> float:
    """Return mean structure/correlation KNN distance to a reference set."""
    if type(node_id) is not str or not node_id:
        raise CoreValidationError("node_id must be a non-empty string")
    if not isinstance(structure_facts, StructureFactStore):
        raise CoreValidationError("structure_facts must be StructureFactStore")
    if not isinstance(correlation_store, CorrelationStore):
        raise CoreValidationError("correlation_store must be CorrelationStore")
    if not isinstance(node_to_factor_id, Mapping):
        raise CoreValidationError("node_to_factor_id must be a mapping")
    if not isinstance(reference_node_ids, Sequence):
        raise CoreValidationError("reference_node_ids must be a sequence")
    if type(knn_k) is not int or knn_k < 1:
        raise CoreValidationError("knn_k must be at least 1")
    raw_refs = tuple(reference_node_ids)
    if not all(type(ref) is str and ref for ref in raw_refs):
        raise CoreValidationError("reference_node_ids must contain strings")
    refs = tuple(sorted(set(raw_refs) - {node_id}))

    structure_index = {
        value: index for index, value in enumerate(structure_facts.node_ids)
    }
    structure_distances: list[tuple[float, str]] = []
    if node_id in structure_index:
        source_index = structure_index[node_id]
        for ref in refs:
            if ref in structure_index:
                structure_distances.append(
                    (structure_facts.distance[source_index][structure_index[ref]], ref)
                )

    source_factor_id = node_to_factor_id.get(node_id, node_id)
    correlation_distances: list[tuple[float, str]] = []
    if correlation_store.has_factor(source_factor_id):
        for ref in refs:
            factor_id = node_to_factor_id.get(ref, ref)
            if not correlation_store.has_factor(factor_id):
                continue
            correlation = correlation_store.correlation(source_factor_id, factor_id)
            if math.isfinite(correlation):
                correlation_distances.append((max(0.0, 1.0 - abs(correlation)), ref))

    return (
        _knn_mean(structure_distances, knn_k) + _knn_mean(correlation_distances, knn_k)
    ) / 2.0


def average_distance_from_config(
    node_id: str,
    *,
    reference_node_ids: Sequence[str],
    structure_facts: StructureFactStore,
    correlation_store: CorrelationStore,
    node_to_factor_id: Mapping[str, str],
    config: GraphAverageDistanceConfig,
) -> float:
    """Calculate AverageDistance with all K/threshold settings from config."""
    if not isinstance(config, GraphAverageDistanceConfig):
        raise CoreValidationError("config must be GraphAverageDistanceConfig")
    return average_distance(
        node_id,
        reference_node_ids=reference_node_ids,
        structure_facts=structure_facts,
        correlation_store=correlation_store,
        node_to_factor_id=node_to_factor_id,
        knn_k=config.knn_k,
    )


def selection_reward_vector(node: GraphNode) -> QualityVector:
    """Return the higher-is-better ``(Self, Fertility, AverageDistance)`` vector."""
    if not isinstance(node, GraphNode):
        raise CoreValidationError("node must be GraphNode")
    return (node.self_quality, node.fertility, node.average_distance)


__all__ = [
    "QualityVector",
    "attempt_fertility_outcome",
    "average_distance",
    "average_distance_from_config",
    "batch_archive_commit",
    "batch_birth_time_delta_hv",
    "birth_time_delta_hv",
    "clip_normalize",
    "fertility_from_attempts",
    "hypervolume_3d",
    "normalize_quality_point",
    "oriented_quality_objectives",
    "pareto_archive",
    "pareto_dominates",
    "quality_hypervolume",
    "quality_point_from_node",
    "selection_reward_vector",
    "self_quality",
    "self_quality_from_node",
    "top3_mean",
]
