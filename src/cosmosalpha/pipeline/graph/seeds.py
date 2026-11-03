"""Configured Graph seed candidates for warm-start initialization."""

from __future__ import annotations

from dataclasses import dataclass

from cosmosalpha.agents import FactorDesignDraft
from cosmosalpha.core import CoreValidationError
from cosmosalpha.factors import (
    build_base_operator_catalog,
    build_fixed_factor_design_draft,
    parse_expression,
)

GRAPH_SEED_CANDIDATE_SCHEMA_VERSION = "graph-seed-candidate/1.0"


@dataclass(frozen=True, slots=True)
class GraphSeedCandidate:
    """One expression candidate that can be evaluated during warm start."""

    candidate_id: str
    name: str
    expression: str
    source: str
    source_factor: str
    expected_direction: str
    frequency: str = "1d"
    schema_version: str = GRAPH_SEED_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate a configured candidate before it reaches the Loop."""
        for field_name in (
            "candidate_id",
            "name",
            "expression",
            "source",
            "source_factor",
            "expected_direction",
            "frequency",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value != value.strip():
                raise CoreValidationError(f"{field_name} must be non-empty text")
        try:
            parse_expression(self.expression)
        except Exception as error:  # noqa: BLE001 - table validation boundary
            raise CoreValidationError(
                f"seed candidate expression is invalid: {self.candidate_id}"
            ) from error
        if self.schema_version != GRAPH_SEED_CANDIDATE_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphSeedCandidate schema_version")

    def to_design_draft(self) -> FactorDesignDraft:
        """Describe the seed while preserving its supplied DSL and provenance."""
        return build_fixed_factor_design_draft(
            self.expression,
            name=self.name,
            description=f"Configured warm-start seed from {self.source_factor}.",
            expected_direction=self.expected_direction,
            source=f"{self.source}: {self.source_factor} ({self.candidate_id})",
            operator_catalog=build_base_operator_catalog(),
        )


HUATAI_SEED_CANDIDATES: tuple[GraphSeedCandidate, ...] = (
    GraphSeedCandidate(
        "huatai-01",
        "价量背离_开盘量秩相关",
        "sub(0, corr(rank(open), rank(volume), 10))",
        "huatai",
        "Alpha3 variant",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-02",
        "价量背离_高价量秩相关",
        "sub(0, corr(high, rank(volume), 5))",
        "huatai",
        "Alpha44 variant",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-03",
        "价量背离_量价秩相关极值",
        "sub(0, ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5))",
        "huatai",
        "Alpha50 variant",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-04",
        "趋势_低价秩时序排名",
        "sub(0, ts_rank(rank(low), 10))",
        "huatai",
        "Alpha4 adapted",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-05",
        "趋势_收盘动量乘开盘比",
        "sub(0, mul(ts_rank(close, 10), rank(div(close, open))))",
        "huatai",
        "Alpha38 variant",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-06",
        "反转_开量时序相关",
        "sub(0, corr(open, volume, 10))",
        "huatai",
        "Alpha6",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-07",
        "波动_高价波动乘量价相关",
        "sub(0, mul(rank(ts_stddev(high, 10)), corr(high, volume, 10)))",
        "huatai",
        "Alpha40 variant",
        "positive",
    ),
    GraphSeedCandidate(
        "huatai-08",
        "价差_VWAP收盘偏离比",
        "rank_div(rank(sub(vwap, close)), rank(add(vwap, close)))",
        "huatai",
        "Alpha42 variant",
        "positive",
    ),
)


def configured_seed_candidates(seed_pool: str) -> tuple[GraphSeedCandidate, ...]:
    """Return the explicit warm-start table selected by configuration."""
    if seed_pool in {"huatai", "huatai_v1"}:
        return HUATAI_SEED_CANDIDATES
    if seed_pool in {"none", "empty"}:
        return ()
    raise CoreValidationError(f"unsupported Graph seed_pool: {seed_pool}")


__all__ = [
    "GRAPH_SEED_CANDIDATE_SCHEMA_VERSION",
    "GraphSeedCandidate",
    "HUATAI_SEED_CANDIDATES",
    "configured_seed_candidates",
]
