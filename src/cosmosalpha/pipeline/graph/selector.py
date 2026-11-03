"""Pareto-guided parent selection for Graph Search v1."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from cosmosalpha.core import CoreValidationError, stable_sha256
from cosmosalpha.pipeline.graph.config import GraphConfig
from cosmosalpha.pipeline.graph.core import FactorGraph
from cosmosalpha.pipeline.graph.flow import GraphFlowResult, run_graph_flow
from cosmosalpha.pipeline.graph.reward import QualityVector, selection_reward_vector

GRAPH_SELECTION_RESULT_SCHEMA_VERSION = "graph-selection-result/1.0"
_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


class SelectionTerminationReason(StrEnum):
    """Normal v1 reasons for returning no further parent batch."""

    BATCH_SELECTED = "BATCH_SELECTED"
    GLOBAL_ATTEMPT_BUDGET_EXHAUSTED = "GLOBAL_ATTEMPT_BUDGET_EXHAUSTED"
    ALL_NODES_EXPANSION_CAPPED = "ALL_NODES_EXPANSION_CAPPED"


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


def _validate_reward(value: object, field_name: str) -> QualityVector:
    if not isinstance(value, tuple) or len(value) != 3:
        raise CoreValidationError(f"{field_name} must contain three values")
    return tuple(
        _require_finite(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _validate_reward_mapping(
    rewards: Mapping[str, QualityVector],
) -> dict[str, QualityVector]:
    if not isinstance(rewards, Mapping):
        raise CoreValidationError("rewards must be a mapping")
    if not all(type(node_id) is str and node_id for node_id in rewards):
        raise CoreValidationError("reward keys must be non-empty strings")
    return {
        node_id: _validate_reward(value, f"reward[{node_id}]")
        for node_id, value in rewards.items()
    }


def pareto_fronts(rewards: Mapping[str, QualityVector]) -> tuple[tuple[str, ...], ...]:
    """Return deterministic non-dominated fronts for higher-is-better vectors."""
    values = _validate_reward_mapping(rewards)
    remaining = set(values)
    fronts: list[tuple[str, ...]] = []
    while remaining:
        front = tuple(
            sorted(
                node_id
                for node_id in remaining
                if not any(
                    other != node_id and _dominates(values[other], values[node_id])
                    for other in remaining
                )
            )
        )
        if not front:
            raise CoreValidationError("Pareto sorting failed to produce a front")
        fronts.append(front)
        remaining.difference_update(front)
    return tuple(fronts)


def _dominates(left: QualityVector, right: QualityVector) -> bool:
    return all(left[index] >= right[index] for index in range(3)) and any(
        left[index] > right[index] for index in range(3)
    )


def crowding_distances(
    front: Sequence[str],
    rewards: Mapping[str, QualityVector],
    *,
    epsilon: float,
) -> dict[str, float]:
    """Return standard NSGA-II crowding distances for one front."""
    values = _validate_reward_mapping(rewards)
    if not isinstance(front, Sequence) or isinstance(front, (str, bytes)):
        raise CoreValidationError("front must be a sequence of node ids")
    front_ids = tuple(front)
    if not front_ids or len(front_ids) != len(set(front_ids)):
        raise CoreValidationError("front must contain unique node ids")
    if not all(type(node_id) is str and node_id in values for node_id in front_ids):
        raise CoreValidationError("front must reference reward node ids")
    threshold = _require_finite(epsilon, "crowding epsilon")
    if threshold < 0.0:
        raise CoreValidationError("crowding epsilon must be non-negative")

    distances = {node_id: 0.0 for node_id in front_ids}
    for objective in range(3):
        ordered = sorted(
            front_ids,
            key=lambda node_id: (values[node_id][objective], node_id),
        )
        lower = values[ordered[0]][objective]
        upper = values[ordered[-1]][objective]
        span = upper - lower
        if span <= threshold:
            continue
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        for index in range(1, len(ordered) - 1):
            if math.isinf(distances[ordered[index]]):
                continue
            previous_value = values[ordered[index - 1]][objective]
            next_value = values[ordered[index + 1]][objective]
            distances[ordered[index]] += (next_value - previous_value) / span
    return distances


def deterministic_tie_key(
    *,
    run_seed: int,
    round_index: int,
    factor_id: str,
) -> str:
    """Return the frozen deterministic pseudo-random tie-break key."""
    if type(run_seed) is not int or run_seed < 0:
        raise CoreValidationError("run_seed must be a non-negative integer")
    if type(round_index) is not int or round_index < 0:
        raise CoreValidationError("round_index must be a non-negative integer")
    _require_text(factor_id, "factor_id")
    return stable_sha256(
        {
            "run_seed": run_seed,
            "round_index": round_index,
            "factor_id": factor_id,
        }
    )


@dataclass(frozen=True, slots=True)
class ParentSelectionResult:
    """Persistable output of one Flow-guided parent selection round."""

    selected_parent_ids: tuple[str, ...]
    eligible_parent_ids: tuple[str, ...]
    pareto_fronts: tuple[tuple[str, ...], ...]
    flow_result: GraphFlowResult
    remaining_budget: int
    termination_reason: SelectionTerminationReason
    schema_version: str = GRAPH_SELECTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate deterministic selection output and budget metadata."""
        for field_name in ("selected_parent_ids", "eligible_parent_ids"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                type(node_id) is str and node_id for node_id in value
            ):
                raise CoreValidationError(f"{field_name} must be a tuple of node ids")
            if len(value) != len(set(value)):
                raise CoreValidationError(f"{field_name} must contain unique node ids")
        if not set(self.selected_parent_ids).issubset(self.eligible_parent_ids):
            raise CoreValidationError("selected parents must be eligible")
        if not isinstance(self.pareto_fronts, tuple):
            raise CoreValidationError("pareto_fronts must be a tuple")
        flattened = []
        for front in self.pareto_fronts:
            if not isinstance(front, tuple) or not all(
                type(node_id) is str and node_id for node_id in front
            ):
                raise CoreValidationError("Pareto fronts must contain node-id tuples")
            if len(front) != len(set(front)):
                raise CoreValidationError("Pareto fronts must contain unique node ids")
            flattened.extend(front)
        if tuple(sorted(flattened)) != tuple(sorted(self.eligible_parent_ids)):
            raise CoreValidationError("Pareto fronts must partition eligible parents")
        if not isinstance(self.flow_result, GraphFlowResult):
            raise CoreValidationError("flow_result must be GraphFlowResult")
        if not set(self.eligible_parent_ids).issubset(self.flow_result.node_ids):
            raise CoreValidationError("eligible parents must be present in flow_result")
        if type(self.remaining_budget) is not int or self.remaining_budget < 0:
            raise CoreValidationError("remaining_budget must be non-negative")
        if isinstance(self.termination_reason, str) and not isinstance(
            self.termination_reason, SelectionTerminationReason
        ):
            try:
                object.__setattr__(
                    self,
                    "termination_reason",
                    SelectionTerminationReason(self.termination_reason),
                )
            except ValueError as error:
                raise CoreValidationError(
                    f"unsupported selection termination reason: {self.termination_reason!r}"
                ) from error
        if not isinstance(self.termination_reason, SelectionTerminationReason):
            raise CoreValidationError("termination_reason must be a known v1 reason")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != GRAPH_SELECTION_RESULT_SCHEMA_VERSION:
            raise CoreValidationError("invalid ParentSelectionResult schema_version")


class FlowGuidedGraphSelector:
    """Select uncapped Graph parents from propagated vector rewards."""

    def __init__(self, config: GraphConfig) -> None:
        """Create a selector from one explicit Graph v1 configuration."""
        if not isinstance(config, GraphConfig):
            raise CoreValidationError("config must be GraphConfig")
        if config.selector.name != "flow_guided_graph":
            raise CoreValidationError("config does not select flow_guided_graph")
        self._config = config

    def select(
        self,
        graph: FactorGraph,
        *,
        attempts_used: int,
        run_seed: int,
        round_index: int,
        flow_observer: Callable[[float, GraphFlowResult], None] | None = None,
    ) -> ParentSelectionResult:
        """Run Flow and select one without-replacement parent batch."""
        if not isinstance(graph, FactorGraph):
            raise CoreValidationError("graph must be FactorGraph")
        if type(attempts_used) is not int or attempts_used < 0:
            raise CoreValidationError("attempts_used must be a non-negative integer")
        if attempts_used > self._config.search.max_expansion_attempts:
            raise CoreValidationError("attempts_used exceeds global attempt budget")
        if type(run_seed) is not int or run_seed < 0:
            raise CoreValidationError("run_seed must be a non-negative integer")
        if type(round_index) is not int or round_index < 0:
            raise CoreValidationError("round_index must be a non-negative integer")
        if flow_observer is not None and not callable(flow_observer):
            raise CoreValidationError("flow_observer must be callable or None")

        rewards = {
            node_id: selection_reward_vector(node)
            for node_id, node in graph.nodes.items()
        }
        flow_started_at = perf_counter()
        flow_result = run_graph_flow(graph, rewards, self._config.flow)
        if flow_observer is not None:
            flow_observer(perf_counter() - flow_started_at, flow_result)
        flow_rewards = {
            node_id: flow_result.propagated_rewards[index]
            for index, node_id in enumerate(flow_result.node_ids)
        }
        eligible = tuple(
            sorted(
                node_id
                for node_id, node in graph.nodes.items()
                if node.expansion_attempt_count
                < self._config.search.max_expansions_per_node
            )
        )
        fronts = pareto_fronts({node_id: flow_rewards[node_id] for node_id in eligible})
        remaining_budget = self._config.search.max_expansion_attempts - attempts_used
        if not eligible:
            selected: tuple[str, ...] = ()
            reason = SelectionTerminationReason.ALL_NODES_EXPANSION_CAPPED
        elif remaining_budget == 0:
            selected = ()
            reason = SelectionTerminationReason.GLOBAL_ATTEMPT_BUDGET_EXHAUSTED
        else:
            target = min(
                self._config.search.parents_per_round,
                len(eligible),
                remaining_budget,
            )
            selected_list: list[str] = []
            for front in fronts:
                distances = crowding_distances(
                    front,
                    flow_rewards,
                    epsilon=self._config.selector.crowding_epsilon,
                )
                ordered = tuple(
                    sorted(
                        front,
                        key=lambda node_id: (
                            -distances[node_id],
                            deterministic_tie_key(
                                run_seed=run_seed,
                                round_index=round_index,
                                factor_id=graph.nodes[node_id].factor_id,
                            ),
                            node_id,
                        ),
                    )
                )
                selected_list.extend(ordered[: target - len(selected_list)])
                if len(selected_list) >= target:
                    break
            selected = tuple(selected_list)
            reason = SelectionTerminationReason.BATCH_SELECTED
        return ParentSelectionResult(
            selected_parent_ids=selected,
            eligible_parent_ids=eligible,
            pareto_fronts=fronts,
            flow_result=flow_result,
            remaining_budget=remaining_budget,
            termination_reason=reason,
        )


__all__ = [
    "GRAPH_SELECTION_RESULT_SCHEMA_VERSION",
    "FlowGuidedGraphSelector",
    "ParentSelectionResult",
    "SelectionTerminationReason",
    "crowding_distances",
    "deterministic_tie_key",
    "pareto_fronts",
]
