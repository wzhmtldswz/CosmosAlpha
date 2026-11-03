"""Lightweight, deterministic diagnostics for Graph Search v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cosmosalpha.core import CoreValidationError
from cosmosalpha.pipeline.graph.core import DenseMatrix, FactorGraph
from cosmosalpha.pipeline.graph.flow import GraphFlowResult

GRAPH_NEIGHBORHOOD_DIAGNOSTICS_SCHEMA_VERSION = "graph-neighborhood-diagnostics/1.0"
GRAPH_DIAGNOSTICS_SCHEMA_VERSION = "graph-diagnostics/1.0"


def _require_finite(value: object, field_name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise CoreValidationError(f"{field_name} must be a finite float")


@dataclass(frozen=True, slots=True)
class GraphNeighborhoodDiagnostics:
    """Compact topology summary for one Graph neighborhood modality."""

    node_count: int
    edge_count: int
    mean_degree: float
    median_degree: float
    zero_degree_ratio: float
    neighbor_cap: int | None
    k_hit_ratio: float | None
    edge_weight_quantiles: tuple[float, float, float, float, float]
    component_count: int
    largest_component_ratio: float
    schema_version: str = GRAPH_NEIGHBORHOOD_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate counts, ratios, quantiles and version identity."""
        for field_name in ("node_count", "edge_count", "component_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise CoreValidationError(f"{field_name} must be non-negative")
        if self.component_count > self.node_count:
            raise CoreValidationError("component_count cannot exceed node_count")
        for field_name in (
            "mean_degree",
            "median_degree",
            "zero_degree_ratio",
            "largest_component_ratio",
        ):
            _require_finite(getattr(self, field_name), field_name)
        for field_name in ("zero_degree_ratio", "largest_component_ratio"):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise CoreValidationError(f"{field_name} must be in [0, 1]")
        if self.neighbor_cap is None:
            if self.k_hit_ratio is not None:
                raise CoreValidationError("k_hit_ratio requires neighbor_cap")
        else:
            if type(self.neighbor_cap) is not int or self.neighbor_cap < 1:
                raise CoreValidationError("neighbor_cap must be positive")
            _require_finite(self.k_hit_ratio, "k_hit_ratio")
            if not 0.0 <= self.k_hit_ratio <= 1.0:  # type: ignore[operator]
                raise CoreValidationError("k_hit_ratio must be in [0, 1]")
        if (
            not isinstance(self.edge_weight_quantiles, tuple)
            or len(self.edge_weight_quantiles) != 5
        ):
            raise CoreValidationError("edge_weight_quantiles must contain five values")
        for index, value in enumerate(self.edge_weight_quantiles):
            _require_finite(value, f"edge_weight_quantiles[{index}]")
        if tuple(sorted(self.edge_weight_quantiles)) != self.edge_weight_quantiles:
            raise CoreValidationError("edge_weight_quantiles must be ordered")
        if self.schema_version != GRAPH_NEIGHBORHOOD_DIAGNOSTICS_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphNeighborhoodDiagnostics version")


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    """Topology and reward-spread diagnostics emitted with a Graph result."""

    structure: GraphNeighborhoodDiagnostics
    correlation: GraphNeighborhoodDiagnostics
    generation: GraphNeighborhoodDiagnostics
    raw_reward_variance: tuple[float, float, float]
    flow_reward_variance: tuple[float, float, float]
    schema_version: str = GRAPH_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate modality summaries and the two reward variance vectors."""
        for field_name in ("structure", "correlation", "generation"):
            if not isinstance(getattr(self, field_name), GraphNeighborhoodDiagnostics):
                raise CoreValidationError(
                    f"{field_name} must be GraphNeighborhoodDiagnostics"
                )
        for field_name in ("raw_reward_variance", "flow_reward_variance"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or len(values) != 3:
                raise CoreValidationError(f"{field_name} must contain three values")
            for index, value in enumerate(values):
                _require_finite(value, f"{field_name}[{index}]")
                if value < 0.0:
                    raise CoreValidationError(
                        f"{field_name}[{index}] must be non-negative"
                    )
        if self.schema_version != GRAPH_DIAGNOSTICS_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphDiagnostics schema_version")


def build_graph_diagnostics(
    graph: FactorGraph,
    flow_result: GraphFlowResult,
    *,
    structure_neighbor_cap: int,
    correlation_neighbor_cap: int,
) -> GraphDiagnostics:
    """Build lightweight diagnostics from committed graph and Flow facts."""
    if not isinstance(graph, FactorGraph):
        raise CoreValidationError("graph must be FactorGraph")
    if not isinstance(flow_result, GraphFlowResult):
        raise CoreValidationError("flow_result must be GraphFlowResult")
    if set(flow_result.node_ids) != set(graph.nodes):
        raise CoreValidationError("Flow result must contain exactly all Graph nodes")
    return GraphDiagnostics(
        structure=_neighborhood_diagnostics(
            graph.w_structure, neighbor_cap=structure_neighbor_cap
        ),
        correlation=_neighborhood_diagnostics(
            graph.w_correlation, neighbor_cap=correlation_neighbor_cap
        ),
        generation=_neighborhood_diagnostics(graph.w_generation, neighbor_cap=None),
        raw_reward_variance=_reward_variance(flow_result.raw_rewards),
        flow_reward_variance=_reward_variance(flow_result.propagated_rewards),
    )


def _neighborhood_diagnostics(
    matrix: DenseMatrix,
    *,
    neighbor_cap: int | None,
) -> GraphNeighborhoodDiagnostics:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise CoreValidationError("diagnostic matrix must be square")
    if neighbor_cap is not None and (type(neighbor_cap) is not int or neighbor_cap < 1):
        raise CoreValidationError("neighbor_cap must be positive")
    degrees = []
    for row_index, row in enumerate(matrix):
        for value in row:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise CoreValidationError(
                    "diagnostic matrix values must be finite non-negative floats"
                )
        degrees.append(
            sum(
                1
                for column_index, value in enumerate(row)
                if column_index != row_index and value > 0.0
            )
        )
    edge_weights = tuple(
        max(matrix[left][right], matrix[right][left])
        for left in range(size)
        for right in range(left + 1, size)
        if matrix[left][right] > 0.0 or matrix[right][left] > 0.0
    )
    components = _component_sizes(matrix)
    return GraphNeighborhoodDiagnostics(
        node_count=size,
        edge_count=len(edge_weights),
        mean_degree=(sum(degrees) / size if size else 0.0),
        median_degree=_percentile(tuple(float(value) for value in degrees), 0.5),
        zero_degree_ratio=(
            sum(value == 0 for value in degrees) / size if size else 0.0
        ),
        neighbor_cap=neighbor_cap,
        k_hit_ratio=(
            None
            if neighbor_cap is None
            else (
                sum(value >= neighbor_cap for value in degrees) / size if size else 0.0
            )
        ),
        edge_weight_quantiles=tuple(
            _percentile(edge_weights, quantile)
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        ),  # type: ignore[arg-type]
        component_count=len(components),
        largest_component_ratio=(max(components, default=0) / size if size else 0.0),
    )


def _component_sizes(matrix: DenseMatrix) -> tuple[int, ...]:
    unseen = set(range(len(matrix)))
    sizes: list[int] = []
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            neighbors = tuple(
                index
                for index in sorted(unseen)
                if matrix[node][index] > 0.0 or matrix[index][node] > 0.0
            )
            for neighbor in neighbors:
                unseen.remove(neighbor)
                pending.append(neighbor)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = tuple(sorted(values))
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _reward_variance(rewards: DenseMatrix) -> tuple[float, float, float]:
    if not rewards:
        return (0.0, 0.0, 0.0)
    result = []
    for objective in range(3):
        values = tuple(row[objective] for row in rewards)
        mean = sum(values) / len(values)
        result.append(sum((value - mean) ** 2 for value in values) / len(values))
    return tuple(result)  # type: ignore[return-value]


__all__ = [
    "GRAPH_DIAGNOSTICS_SCHEMA_VERSION",
    "GRAPH_NEIGHBORHOOD_DIAGNOSTICS_SCHEMA_VERSION",
    "GraphDiagnostics",
    "GraphNeighborhoodDiagnostics",
    "build_graph_diagnostics",
]
