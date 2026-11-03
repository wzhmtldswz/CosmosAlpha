"""Generation provenance graph and deterministic generation neighborhoods."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from cosmosalpha.core import CoreValidationError
from cosmosalpha.pipeline.graph.core import GenerationEvent

GENERATION_GRAPH_SCHEMA_VERSION = "generation-graph/1.0"
type DistanceMatrix = tuple[tuple[int | None, ...], ...]

_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


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


def _validate_dense_binary_matrix(
    value: object,
    field_name: str,
    *,
    size: int,
) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise CoreValidationError(f"{field_name} must be a {size}x{size} matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != size:
            raise CoreValidationError(
                f"{field_name}[{row_index}] must contain {size} values"
            )
        for column_index, item in enumerate(row):
            if (
                type(item) is not float
                or not math.isfinite(item)
                or item
                not in (
                    0.0,
                    1.0,
                )
            ):
                raise CoreValidationError(
                    f"{field_name}[{row_index}][{column_index}] must be 0.0 or 1.0"
                )


def _validate_distance_matrix(value: object, *, size: int) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise CoreValidationError(f"d_gen must be a {size}x{size} matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != size:
            raise CoreValidationError(f"d_gen[{row_index}] must contain {size} values")
        for column_index, item in enumerate(row):
            if item is not None and (type(item) is not int or item < 0):
                raise CoreValidationError(
                    f"d_gen[{row_index}][{column_index}] must be a non-negative "
                    "integer or None"
                )


@dataclass(frozen=True, slots=True)
class GenerationGraph:
    """Dense generation facts derived from successful mutation events."""

    node_ids: tuple[str, ...]
    directed_adjacency: tuple[tuple[float, ...], ...]
    w_generation: tuple[tuple[float, ...], ...]
    d_gen: DistanceMatrix
    schema_version: str = GENERATION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate node ordering and generation matrix invariants."""
        if not isinstance(self.node_ids, tuple) or not all(
            type(node_id) is str and node_id for node_id in self.node_ids
        ):
            raise CoreValidationError("node_ids must be a tuple of non-empty strings")
        if self.node_ids != tuple(sorted(self.node_ids)):
            raise CoreValidationError("node_ids must be sorted deterministically")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise CoreValidationError("node_ids must be unique")
        size = len(self.node_ids)
        _validate_dense_binary_matrix(
            self.directed_adjacency,
            "directed_adjacency",
            size=size,
        )
        _validate_dense_binary_matrix(
            self.w_generation,
            "w_generation",
            size=size,
        )
        _validate_distance_matrix(self.d_gen, size=size)
        for index in range(size):
            if self.directed_adjacency[index][index] != 0.0:
                raise CoreValidationError("directed_adjacency diagonal must be zero")
            if self.w_generation[index][index] != 0.0:
                raise CoreValidationError("w_generation diagonal must be zero")
            for other in range(size):
                if self.w_generation[index][other] != self.w_generation[other][index]:
                    raise CoreValidationError("w_generation must be symmetric")
                expected_edge = (
                    self.directed_adjacency[index][other]
                    or self.directed_adjacency[other][index]
                )
                if self.w_generation[index][other] != float(bool(expected_edge)):
                    raise CoreValidationError(
                        "w_generation must be the symmetric direct adjacency"
                    )
                distance = self.d_gen[index][other]
                if index == other and distance != 0:
                    raise CoreValidationError("d_gen diagonal must be zero")
                if index != other and self.w_generation[index][other] and distance != 1:
                    raise CoreValidationError(
                        "direct generation neighbors must have d_gen equal to 1"
                    )
                if distance != self.d_gen[other][index]:
                    raise CoreValidationError("d_gen must be symmetric")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != GENERATION_GRAPH_SCHEMA_VERSION:
            raise CoreValidationError("invalid GenerationGraph schema_version")

    @property
    def node_index(self) -> dict[str, int]:
        """Return the stable node-to-matrix index mapping."""
        return {node_id: index for index, node_id in enumerate(self.node_ids)}


def build_generation_graph(
    node_ids: Iterable[str],
    generation_events: Iterable[GenerationEvent],
) -> GenerationGraph:
    """Build direct and shortest-path generation facts from provenance events."""
    if not isinstance(node_ids, tuple):
        node_ids = tuple(node_ids)
    if not isinstance(generation_events, tuple):
        generation_events = tuple(generation_events)
    if not all(type(node_id) is str and node_id for node_id in node_ids):
        raise CoreValidationError("node_ids must contain non-empty strings")
    ordered_node_ids = tuple(sorted(node_ids))
    if len(set(ordered_node_ids)) != len(ordered_node_ids):
        raise CoreValidationError("node_ids must be unique")
    known_nodes = set(ordered_node_ids)
    node_index = {node_id: index for index, node_id in enumerate(ordered_node_ids)}
    event_ids: set[str] = set()
    parent_by_child: dict[str, str] = {}
    direct = [[0.0 for _column in ordered_node_ids] for _row in ordered_node_ids]
    if not all(isinstance(event, GenerationEvent) for event in generation_events):
        raise CoreValidationError("generation_events must contain GenerationEvent")
    for event in sorted(generation_events, key=lambda item: item.event_id):
        if event.event_id in event_ids:
            raise CoreValidationError("generation event ids must be unique")
        event_ids.add(event.event_id)
        parent_id = event.parent_ids[0]
        if parent_id not in known_nodes or event.child_id not in known_nodes:
            raise CoreValidationError("generation event references an unknown node")
        if parent_id == event.child_id:
            raise CoreValidationError("generation event cannot be a self-loop")
        previous_parent = parent_by_child.setdefault(event.child_id, parent_id)
        if previous_parent != parent_id:
            raise CoreValidationError(
                "Graph v1 generation child cannot have multiple parents"
            )
        direct[node_index[parent_id]][node_index[event.child_id]] = 1.0

    size = len(ordered_node_ids)
    undirected = [row[:] for row in direct]
    for left in range(size):
        for right in range(size):
            if direct[left][right] or direct[right][left]:
                undirected[left][right] = 1.0
                undirected[right][left] = 1.0
    distances: list[list[int | None]] = [[None] * size for _row in range(size)]
    for source in range(size):
        distances[source][source] = 0
        queue = [source]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            current_distance = distances[source][current]
            if current_distance is None:
                raise CoreValidationError("internal generation distance error")
            for neighbor in range(size):
                if (
                    not undirected[current][neighbor]
                    or distances[source][neighbor] is not None
                ):
                    continue
                distances[source][neighbor] = current_distance + 1
                queue.append(neighbor)
    return GenerationGraph(
        node_ids=ordered_node_ids,
        directed_adjacency=tuple(tuple(row) for row in direct),
        w_generation=tuple(tuple(row) for row in undirected),
        d_gen=tuple(tuple(row) for row in distances),
    )


__all__ = [
    "DistanceMatrix",
    "GENERATION_GRAPH_SCHEMA_VERSION",
    "GenerationGraph",
    "build_generation_graph",
]
