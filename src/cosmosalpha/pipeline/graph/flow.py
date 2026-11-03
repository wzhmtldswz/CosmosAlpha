"""Deterministic vector reward diffusion for Graph Search v1."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cosmosalpha.core import CoreValidationError
from cosmosalpha.pipeline.graph.config import GraphFlowConfig, GraphModalityWeights
from cosmosalpha.pipeline.graph.core import DenseMatrix, FactorGraph
from cosmosalpha.pipeline.graph.reward import QualityVector

FLOW_RESULT_SCHEMA_VERSION = "graph-flow-result/1.0"
_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


def _require_finite(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CoreValidationError(f"{field_name} must be finite")
    return float(value)


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be a non-empty string without surrounding whitespace"
        )


def _require_schema_version(value: object, field_name: str) -> None:
    _require_text(value, field_name)
    if _SCHEMA_VERSION_PATTERN.fullmatch(value) is None:
        raise CoreValidationError(
            f"{field_name} must match '<schema-name>/<major>.<minor>'"
        )


def _validate_square_matrix(
    value: object,
    field_name: str,
    *,
    size: int | None = None,
    non_negative: bool = False,
) -> DenseMatrix:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple matrix")
    matrix_size = len(value)
    if size is not None and matrix_size != size:
        raise CoreValidationError(f"{field_name} must be a {size}x{size} matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != matrix_size:
            raise CoreValidationError(
                f"{field_name}[{row_index}] must contain {matrix_size} values"
            )
        for column_index, item in enumerate(row):
            numeric = _require_finite(
                item, f"{field_name}[{row_index}][{column_index}]"
            )
            if non_negative and numeric < 0.0:
                raise CoreValidationError(
                    f"{field_name}[{row_index}][{column_index}] must be non-negative"
                )
    return value


def _validate_reward_matrix(value: object, *, size: int) -> DenseMatrix:
    if not isinstance(value, tuple) or len(value) != size:
        raise CoreValidationError("reward matrix must have one row per node")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != 3:
            raise CoreValidationError(
                f"reward matrix row {row_index} must contain three values"
            )
        for column_index, item in enumerate(row):
            _require_finite(item, f"reward[{row_index}][{column_index}]")
    return value


def _validate_node_ids(node_ids: object) -> tuple[str, ...]:
    if not isinstance(node_ids, tuple):
        raise CoreValidationError("node_ids must be a tuple")
    if not all(type(node_id) is str and node_id for node_id in node_ids):
        raise CoreValidationError("node_ids must contain non-empty strings")
    if len(node_ids) != len(set(node_ids)):
        raise CoreValidationError("node_ids must be unique")
    return node_ids


@dataclass(frozen=True, slots=True)
class GraphFlowResult:
    """Persistable result of one deterministic Graph Flow propagation."""

    node_ids: tuple[str, ...]
    transition_matrix: DenseMatrix
    raw_rewards: DenseMatrix
    propagated_rewards: DenseMatrix
    iterations_run: int
    converged: bool
    schema_version: str = FLOW_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate Flow matrices, reward vectors and convergence metadata."""
        node_ids = _validate_node_ids(self.node_ids)
        size = len(node_ids)
        transition = _validate_square_matrix(
            self.transition_matrix,
            "transition_matrix",
            size=size,
            non_negative=True,
        )
        for row_index, row in enumerate(transition):
            if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise CoreValidationError(
                    f"transition_matrix row {row_index} must sum to one"
                )
        _validate_reward_matrix(self.raw_rewards, size=size)
        _validate_reward_matrix(self.propagated_rewards, size=size)
        if type(self.iterations_run) is not int or self.iterations_run < 0:
            raise CoreValidationError("iterations_run must be non-negative")
        if type(self.converged) is not bool:
            raise CoreValidationError("converged must be a boolean")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != FLOW_RESULT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphFlowResult schema_version")


def row_normalize_matrix(matrix: DenseMatrix) -> DenseMatrix:
    """Row-normalize non-negative weights, using a self-loop for zero rows."""
    values = _validate_square_matrix(matrix, "matrix", non_negative=True)
    size = len(values)
    normalized: list[tuple[float, ...]] = []
    for row_index, row in enumerate(values):
        total = sum(row)
        if total == 0.0:
            normalized.append(
                tuple(1.0 if column == row_index else 0.0 for column in range(size))
            )
            continue
        normalized.append(tuple(value / total for value in row))
    return tuple(normalized)


def combine_modality_matrices(
    structure: DenseMatrix,
    correlation: DenseMatrix,
    generation: DenseMatrix,
    weights: GraphModalityWeights,
) -> DenseMatrix:
    """Combine three independently normalized modalities by configured weights."""
    if not isinstance(weights, GraphModalityWeights):
        raise CoreValidationError("weights must be GraphModalityWeights")
    structure_values = _validate_square_matrix(
        structure, "structure", non_negative=True
    )
    size = len(structure_values)
    correlation_values = _validate_square_matrix(
        correlation, "correlation", size=size, non_negative=True
    )
    generation_values = _validate_square_matrix(
        generation, "generation", size=size, non_negative=True
    )
    normalized = tuple(
        row_normalize_matrix(matrix)
        for matrix in (structure_values, correlation_values, generation_values)
    )
    return tuple(
        tuple(
            weights.structure * normalized[0][row][column]
            + weights.correlation * normalized[1][row][column]
            + weights.generation * normalized[2][row][column]
            for column in range(size)
        )
        for row in range(size)
    )


def _ordered_graph_node_ids(graph: FactorGraph) -> tuple[str, ...]:
    if not isinstance(graph, FactorGraph):
        raise CoreValidationError("graph must be FactorGraph")
    return tuple(
        node_id
        for node_id, _index in sorted(
            graph.node_index.items(), key=lambda item: item[1]
        )
    )


def build_flow_transition_matrix(
    graph: FactorGraph,
    config: GraphFlowConfig,
) -> DenseMatrix:
    """Build the row-stochastic Flow matrix from a FactorGraph."""
    if not isinstance(config, GraphFlowConfig):
        raise CoreValidationError("config must be GraphFlowConfig")
    return combine_modality_matrices(
        graph.w_structure,
        graph.w_correlation,
        graph.w_generation,
        config.modality_weights,
    )


def _ordered_reward_matrix(
    node_ids: tuple[str, ...],
    rewards: Mapping[str, QualityVector],
) -> DenseMatrix:
    if not isinstance(rewards, Mapping):
        raise CoreValidationError("rewards must be a mapping")
    if set(rewards) != set(node_ids):
        raise CoreValidationError("rewards must contain exactly all graph node ids")
    rows: list[tuple[float, ...]] = []
    for node_id in node_ids:
        value = rewards[node_id]
        if not isinstance(value, tuple) or len(value) != 3:
            raise CoreValidationError(f"reward for {node_id} must contain three values")
        rows.append(
            tuple(
                _require_finite(item, f"reward[{node_id}][{index}]")
                for index, item in enumerate(value)
            )
        )
    return tuple(rows)


def iterative_flow(
    node_ids: tuple[str, ...],
    transition_matrix: DenseMatrix,
    raw_rewards: DenseMatrix,
    config: GraphFlowConfig,
) -> GraphFlowResult:
    """Solve the frozen Flow fixed point with a bounded iteration loop."""
    if not isinstance(config, GraphFlowConfig):
        raise CoreValidationError("config must be GraphFlowConfig")
    ordered_ids = _validate_node_ids(node_ids)
    size = len(ordered_ids)
    transition = _validate_square_matrix(
        transition_matrix,
        "transition_matrix",
        size=size,
        non_negative=True,
    )
    for row_index, row in enumerate(transition):
        if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise CoreValidationError(
                f"transition_matrix row {row_index} must sum to one"
            )
    rewards = _validate_reward_matrix(raw_rewards, size=size)
    current = rewards
    converged = size == 0
    iterations_run = 0
    for iteration in range(1, config.iterations + 1):
        next_rows = []
        for row_index in range(size):
            propagated = tuple(
                (1.0 - config.gamma) * rewards[row_index][objective]
                + config.gamma
                * sum(
                    transition[row_index][column] * current[column][objective]
                    for column in range(size)
                )
                for objective in range(3)
            )
            if not all(math.isfinite(value) for value in propagated):
                raise CoreValidationError("Flow produced a non-finite reward")
            next_rows.append(propagated)
        next_values = tuple(next_rows)
        iterations_run = iteration
        max_delta = max(
            (
                abs(next_values[row][objective] - current[row][objective])
                for row in range(size)
                for objective in range(3)
            ),
            default=0.0,
        )
        current = next_values
        if max_delta <= config.convergence_tolerance:
            converged = True
            break
    return GraphFlowResult(
        node_ids=ordered_ids,
        transition_matrix=transition,
        raw_rewards=rewards,
        propagated_rewards=current,
        iterations_run=iterations_run,
        converged=converged,
    )


def run_graph_flow(
    graph: FactorGraph,
    rewards: Mapping[str, QualityVector],
    config: GraphFlowConfig,
) -> GraphFlowResult:
    """Run Flow for graph nodes using their stable matrix ordering."""
    node_ids = _ordered_graph_node_ids(graph)
    raw_rewards = _ordered_reward_matrix(node_ids, rewards)
    return iterative_flow(
        node_ids,
        build_flow_transition_matrix(graph, config),
        raw_rewards,
        config,
    )


__all__ = [
    "FLOW_RESULT_SCHEMA_VERSION",
    "GraphFlowResult",
    "build_flow_transition_matrix",
    "combine_modality_matrices",
    "iterative_flow",
    "row_normalize_matrix",
    "run_graph_flow",
]
