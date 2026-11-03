"""Strict, explicit configuration contracts for Graph Search v1."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cosmosalpha.core import CoreValidationError

GRAPH_CONFIG_SCHEMA_VERSION = "graph-config/2.0"
GRAPH_QUALITY_NORMALIZATION_VERSION = "graph-quality-normalization/2.0"
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


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise CoreValidationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise CoreValidationError(f"{field_name} must be a finite float")
    if minimum is not None and value < minimum:
        raise CoreValidationError(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise CoreValidationError(f"{field_name} must be at most {maximum}")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise CoreValidationError(f"{field_name} must be a boolean")


@dataclass(frozen=True, slots=True)
class GraphInitializationConfig:
    """Required seed selection settings; source adapters are intentionally absent."""

    seed_pool: str
    initial_node_count: int
    random_seed: int
    seed_mode: str

    def __post_init__(self) -> None:
        """Validate initial Graph seed settings."""
        _require_text(self.seed_pool, "graph.initialization.seed_pool")
        _require_int(
            self.initial_node_count,
            "graph.initialization.initial_node_count",
            minimum=1,
        )
        _require_int(self.random_seed, "graph.initialization.random_seed")
        if self.seed_mode not in {"auto", "cold", "warm"}:
            raise CoreValidationError(
                "graph.initialization.seed_mode must be 'auto', 'cold', or 'warm'"
            )


@dataclass(frozen=True, slots=True)
class GraphSearchConfig:
    """Required parent and expansion budget settings."""

    parents_per_round: int
    max_expansion_attempts: int
    max_expansions_per_node: int

    def __post_init__(self) -> None:
        """Validate search budget settings."""
        for field_name in (
            "parents_per_round",
            "max_expansion_attempts",
            "max_expansions_per_node",
        ):
            _require_int(
                getattr(self, field_name),
                f"graph.search.{field_name}",
                minimum=1,
            )


@dataclass(frozen=True, slots=True)
class GraphNeighborhoodConfig:
    """Threshold and local Top-K settings for one neighborhood modality."""

    edge_threshold: float
    max_neighbors: int

    def __post_init__(self) -> None:
        """Validate local-neighborhood settings."""
        _require_float(
            self.edge_threshold,
            "graph.neighborhood.edge_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        _require_int(self.max_neighbors, "graph.neighborhood.max_neighbors", minimum=1)


@dataclass(frozen=True, slots=True)
class GraphAverageDistanceConfig:
    """Required novelty reference-pool settings."""

    knn_k: int
    minimum_abs_rank_ic: float
    external_pool_capacity: int

    def __post_init__(self) -> None:
        """Validate AverageDistance settings without selecting a production rule."""
        _require_int(self.knn_k, "graph.average_distance.knn_k", minimum=1)
        _require_float(
            self.minimum_abs_rank_ic,
            "graph.average_distance.minimum_abs_rank_ic",
            minimum=0.0,
        )
        _require_int(
            self.external_pool_capacity,
            "graph.average_distance.external_pool_capacity",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class GraphNormalizationAnchor:
    """One fixed lower/upper clipping anchor pair."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate a finite, non-degenerate normalization interval."""
        _require_float(self.lower, "graph.quality.anchor.lower")
        _require_float(self.upper, "graph.quality.anchor.upper")
        if self.upper <= self.lower:
            raise CoreValidationError("normalization anchor upper must exceed lower")


@dataclass(frozen=True, slots=True)
class GraphQualityConfig:
    """Required fixed-anchor configuration for intrinsic quality metrics."""

    normalization_version: str
    rank_ic: GraphNormalizationAnchor
    rank_ic_std: GraphNormalizationAnchor
    top20_return: GraphNormalizationAnchor

    def __post_init__(self) -> None:
        """Validate all quality anchors without selecting experimental values."""
        _require_text(
            self.normalization_version,
            "graph.quality.normalization_version",
        )
        if self.normalization_version != GRAPH_QUALITY_NORMALIZATION_VERSION:
            raise CoreValidationError("invalid graph.quality.normalization_version")
        for field_name in ("rank_ic", "rank_ic_std", "top20_return"):
            if not isinstance(getattr(self, field_name), GraphNormalizationAnchor):
                raise CoreValidationError(
                    f"graph.quality.{field_name} must be GraphNormalizationAnchor"
                )
        expected_anchors = {
            "rank_ic": (0.0, 0.05),
            "rank_ic_std": (0.05, 0.09),
            "top20_return": (0.0, 8e-4),
        }
        for field_name, (lower, upper) in expected_anchors.items():
            anchor = getattr(self, field_name)
            if anchor.lower != lower or anchor.upper != upper:
                raise CoreValidationError(
                    f"graph.quality.{field_name} must use the frozen v2 anchors"
                )


@dataclass(frozen=True, slots=True)
class GraphFertilityConfig:
    """Frozen fertility semantics represented as explicit required fields."""

    top_q: int
    use_birth_time_delta_hv: bool

    def __post_init__(self) -> None:
        """Validate the v1 Top-3 birth-time fertility contract."""
        _require_int(self.top_q, "graph.fertility.top_q", minimum=1)
        if self.top_q != 3:
            raise CoreValidationError(
                "graph.fertility.top_q is frozen to 3 in Graph v1"
            )
        _require_bool(
            self.use_birth_time_delta_hv,
            "graph.fertility.use_birth_time_delta_hv",
        )
        if not self.use_birth_time_delta_hv:
            raise CoreValidationError(
                "graph.fertility.use_birth_time_delta_hv must be true in Graph v1"
            )


@dataclass(frozen=True, slots=True)
class GraphModalityWeights:
    """Equal weights for the three Graph neighborhood modalities."""

    structure: float
    correlation: float
    generation: float

    def __post_init__(self) -> None:
        """Validate frozen equal weighting semantics."""
        values = (self.structure, self.correlation, self.generation)
        for field_name, value in zip(
            ("structure", "correlation", "generation"), values, strict=True
        ):
            _require_float(
                value, f"graph.flow.modality_weights.{field_name}", minimum=0.0
            )
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise CoreValidationError("Graph modality weights must sum to 1")
        if not (
            math.isclose(self.structure, self.correlation, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                self.structure, self.generation, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise CoreValidationError(
                "Graph v1 requires equal structure/correlation/generation weights"
            )


@dataclass(frozen=True, slots=True)
class GraphFlowConfig:
    """Required Flow propagation settings."""

    gamma: float
    iterations: int
    convergence_tolerance: float
    modality_weights: GraphModalityWeights

    def __post_init__(self) -> None:
        """Validate Flow settings without choosing experimental values."""
        _require_float(self.gamma, "graph.flow.gamma", minimum=0.0, maximum=1.0)
        if self.gamma >= 1.0:
            raise CoreValidationError("graph.flow.gamma must be less than 1")
        _require_int(self.iterations, "graph.flow.iterations", minimum=1)
        _require_float(
            self.convergence_tolerance,
            "graph.flow.convergence_tolerance",
            minimum=0.0,
        )
        if not isinstance(self.modality_weights, GraphModalityWeights):
            raise CoreValidationError(
                "graph.flow.modality_weights must be GraphModalityWeights"
            )


@dataclass(frozen=True, slots=True)
class GraphSelectorConfig:
    """Frozen selector implementation identity and policy."""

    name: str
    use_pareto_fronts: bool
    tie_breaker: str
    crowding_epsilon: float

    def __post_init__(self) -> None:
        """Validate the v1 selector contract."""
        _require_text(self.name, "graph.selector.name")
        if self.name != "flow_guided_graph":
            raise CoreValidationError(
                "graph.selector.name must be 'flow_guided_graph' in Graph v1"
            )
        _require_bool(self.use_pareto_fronts, "graph.selector.use_pareto_fronts")
        if not self.use_pareto_fronts:
            raise CoreValidationError(
                "graph.selector.use_pareto_fronts must be true in Graph v1"
            )
        _require_text(self.tie_breaker, "graph.selector.tie_breaker")
        if self.tie_breaker != "crowding_distance":
            raise CoreValidationError(
                "graph.selector.tie_breaker must be 'crowding_distance' in Graph v1"
            )
        _require_float(
            self.crowding_epsilon,
            "graph.selector.crowding_epsilon",
            minimum=0.0,
        )


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Complete explicit Graph Search v1 configuration contract."""

    initialization: GraphInitializationConfig
    search: GraphSearchConfig
    structure: GraphNeighborhoodConfig
    correlation: GraphNeighborhoodConfig
    average_distance: GraphAverageDistanceConfig
    quality: GraphQualityConfig
    fertility: GraphFertilityConfig
    flow: GraphFlowConfig
    selector: GraphSelectorConfig
    schema_version: str

    def __post_init__(self) -> None:
        """Validate all Graph configuration sections and schema identity."""
        for field_name, field_type in (
            ("initialization", GraphInitializationConfig),
            ("search", GraphSearchConfig),
            ("structure", GraphNeighborhoodConfig),
            ("correlation", GraphNeighborhoodConfig),
            ("average_distance", GraphAverageDistanceConfig),
            ("quality", GraphQualityConfig),
            ("fertility", GraphFertilityConfig),
            ("flow", GraphFlowConfig),
            ("selector", GraphSelectorConfig),
        ):
            if not isinstance(getattr(self, field_name), field_type):
                raise CoreValidationError(
                    f"graph.{field_name} must be {field_type.__name__}"
                )
        _require_schema_version(self.schema_version, "graph.schema_version")
        if self.schema_version != GRAPH_CONFIG_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphConfig schema_version")


__all__ = [
    "GRAPH_CONFIG_SCHEMA_VERSION",
    "GRAPH_QUALITY_NORMALIZATION_VERSION",
    "GraphAverageDistanceConfig",
    "GraphConfig",
    "GraphFertilityConfig",
    "GraphFlowConfig",
    "GraphInitializationConfig",
    "GraphModalityWeights",
    "GraphNeighborhoodConfig",
    "GraphNormalizationAnchor",
    "GraphQualityConfig",
    "GraphSearchConfig",
    "GraphSelectorConfig",
]
