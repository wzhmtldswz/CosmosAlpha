"""Shared fixed samples; no data files, credentials or network access."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from cosmosalpha.agents import FactorSpec
from cosmosalpha.core import (
    ArtifactRef,
    ExperimentConfig,
    ObjectRef,
    load_experiment_config,
    load_experiment_config_values,
    to_json_value,
)
from cosmosalpha.factors import CalculationFrame
from cosmosalpha.modeling import LightGBMIntegrationConfig, ModelFactorInput
from cosmosalpha.pipeline.graph import (
    GraphAverageDistanceConfig,
    GraphConfig,
    GraphFertilityConfig,
    GraphFlowConfig,
    GraphInitializationConfig,
    GraphModalityWeights,
    GraphNeighborhoodConfig,
    GraphNormalizationAnchor,
    GraphQualityConfig,
    GraphSearchConfig,
    GraphSelectorConfig,
)

MOCK_VALUES: dict[str, object] = {
    "schema_version": "experiment-config/1.6",
    "architecture": {"name": "chain", "directions": []},
    "data": {
        "provider": "mock",
        "dataset_version": "mock-daily-v1",
        "price_adjustment": "post_adjusted",
        "start_date": "2020-01-02",
        "end_date": "2020-01-04",
        "warmup_start_date": "2020-01-01",
        "market": "cn",
        "frequency": "1d",
        "trading_calendar_version": "mock-calendar-v1",
    },
    "universe": {
        "name": "all_a_share",
        "new_stock_min_age_years": 1,
    },
    "returns": {
        "single_factor": "close_t1_over_close_t",
        "suspension_compression": "clip_10_percent",
        "horizon_days": 1,
    },
    "preprocessing": {
        "protocol": "gplearn_neutralized_v1",
        "missing_policy": "field_specific",
        "rolling_min_fraction": 0.6666666667,
        "industry_field": "industry_id",
        "continuous_exposures": ["log_mkt_cap", "ret20", "turn20", "vol20"],
        "mad_multiplier": 5.0,
        "rolling_std_ddof": 1,
    },
    "operators": {
        "operator_set": "OPERATOR_SET_BASE",
        "registry_version": "operator-registry/1.0",
        "allowed_windows": [5, 10, 20],
    },
    "factor_generation": {
        "max_proposals": 2,
        "max_expression_depth": 5,
        "max_validation_repair_attempts": 2,
        "duplicate_policy": "reject_existing_expression",
    },
    "evaluation": {
        "protocol": "rankic_neutralized_v1",
        "metrics": ["rank_ic_mean", "rank_ic_ir"],
        "min_effective_sample_count": 1,
        "min_cross_section_samples": 4,
        "acceptance_metric": "rank_ic_mean",
        "acceptance_threshold": 0.02,
        "min_effective_date_ratio": 0.0,
        "acceptance_use_absolute_value": True,
        "min_daily_effective_stock_ratio": 0.8,
    },
    "workflow": {"max_cycles": 1, "max_retries": 1},
    "prompts": {
        "manifest": "builtin:v2",
        "max_input_tokens": 16000,
        "max_history_items": 6,
        "max_proposals_per_call": 2,
        "max_format_retries": 2,
    },
    "llm": {
        "provider": "mock",
        "model_alias": "mock-replay",
        "base_url": "https://api.openai.com",
        "base_url_env": "COSMOSALPHA_LLM_BASE_URL",
        "organization": "unused",
        "api_key": "unused",
        "api_key_env": "COSMOSALPHA_LLM_API_KEY",
        "timeout_seconds": 30,
        "temperature": 0.0,
        "max_output_tokens": 3000,
        "max_timeout_retries": 1,
        "max_rate_limit_retries": 2,
        "max_server_retries": 2,
        "max_empty_retries": 1,
        "retry_initial_backoff_seconds": 1.0,
        "retry_max_backoff_seconds": 8.0,
        "replay_only": True,
    },
    "storage": {"backend": "local_mock", "root": "runs"},
    "runtime": {
        "max_workers": 1,
        "random_seed": 0,
    },
    "qlib": {
        "enabled": False,
        "region": "cn",
        "instruments": [],
        "market": {
            "provider_uri": "replace-with-qlib-market-uri",
            "fields": [{"field_name": "close", "expression": "$close"}],
        },
        "exposure": {
            "provider_uri": "replace-with-qlib-exposure-uri",
            "fields": [{"field_name": "industry_id", "expression": "$industry_id"}],
        },
        "mask": {
            "provider_uri": "replace-with-qlib-mask-uri",
            "fields": [{"field_name": "eligible_mask", "expression": "$eligible_mask"}],
        },
    },
}


def mock_config(root: Path) -> ExperimentConfig:
    """Return an independent fixed-data Replay configuration."""
    config = load_experiment_config_values(copy.deepcopy(MOCK_VALUES))
    return replace(config, storage=replace(config.storage, root=str(root)))


def graph_runtime_config(root: Path, threshold: float = 2.0) -> ExperimentConfig:
    """Build one bounded cold-start Graph run with deterministic responses."""
    config = mock_config(root)
    graph = graph_config()
    graph = replace(
        graph,
        initialization=replace(
            graph.initialization, seed_mode="cold", seed_pool="empty"
        ),
        average_distance=replace(graph.average_distance, minimum_abs_rank_ic=threshold),
    )
    return replace(
        config,
        architecture=replace(config.architecture, name="graph"),
        graph=to_json_value(graph),
        evaluation=replace(config.evaluation, acceptance_threshold=threshold),
        factor_generation=replace(config.factor_generation, max_proposals=1),
        prompts=replace(config.prompts, max_proposals_per_call=1),
    )


def candidate(factor_id: str, status: str = "accepted") -> ModelFactorInput:
    """Create a candidate whose artifacts are unused by array-only tests."""
    return ModelFactorInput(
        factor_id=factor_id,
        expression=factor_id,
        factor_ref=ObjectRef("factor_spec", factor_id, "factor-spec/1.0"),
        evaluation_result_ref=ObjectRef(
            "evaluation_result", factor_id, "evaluation-result/1.1"
        ),
        acceptance_decision_ref=ObjectRef(
            "acceptance_decision", factor_id, "acceptance-decision/1.0"
        ),
        source_evaluated_values_ref=ArtifactRef(
            artifact_id=factor_id,
            kind="evaluated_factor_values",
            uri=f"artifact://test/{factor_id}",
            media_type="application/octet-stream",
            checksum="b" * 64,
            size_bytes=1,
            schema_version="artifact-ref/1.0",
        ),
        source_data_snapshot_id="source-snapshot",
        source_experiment_id="experiment",
        source_status=status,
    )


def make_spec(
    expression: str,
    *,
    required_fields: tuple[str, ...] = ("close",),
    factor_id: str = "factor-001",
) -> FactorSpec:
    """Create a minimal factor spec for execution tests."""
    return FactorSpec(
        factor_id=factor_id,
        proposal_id="proposal-001",
        hypothesis_id="hypothesis-001",
        name="test_factor",
        expression=expression,
        description="A test factor.",
        frequency="1d",
        required_fields=required_fields,
        expected_direction="positive",
        operator_set_version="operator-set-base/1.0",
        field_schema_version="field-catalog/1.0",
        expression_schema_version="factor-expression/1.0",
        schema_version="factor-spec/1.0",
    )


def make_frame(
    *,
    datetimes: tuple[str, ...],
    instruments: tuple[str, ...],
    fields: dict[str, tuple[float, ...]],
    industry: tuple[float, ...] | None = None,
    eligible: tuple[bool, ...] | None = None,
) -> CalculationFrame:
    """Create a calculation frame from compact tuples."""
    n_samples = len(datetimes)
    return CalculationFrame(
        datetimes=datetimes,
        instruments=instruments,
        fields={name: np.array(values, dtype=float) for name, values in fields.items()},
        context_fields={
            "industry_id": np.array(
                industry if industry is not None else (math.nan,) * n_samples,
                dtype=float,
            )
        },
        eligible_mask=np.array(
            eligible if eligible is not None else (True,) * n_samples,
            dtype=bool,
        ),
    )


def graph_config() -> GraphConfig:
    """Build a complete explicit test-only Graph configuration."""
    return GraphConfig(
        initialization=GraphInitializationConfig("fixture", 1, 7, "auto"),
        search=GraphSearchConfig(1, 1, 1),
        structure=GraphNeighborhoodConfig(0.2, 10),
        correlation=GraphNeighborhoodConfig(0.2, 10),
        average_distance=GraphAverageDistanceConfig(2, 0.02, 30),
        quality=GraphQualityConfig(
            "graph-quality-normalization/2.0",
            GraphNormalizationAnchor(0.0, 0.05),
            GraphNormalizationAnchor(0.05, 0.09),
            GraphNormalizationAnchor(0.0, 8e-4),
        ),
        fertility=GraphFertilityConfig(3, True),
        flow=GraphFlowConfig(
            0.5,
            20,
            1e-12,
            GraphModalityWeights(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
        selector=GraphSelectorConfig(
            "flow_guided_graph", True, "crowding_distance", 1e-12
        ),
        schema_version="graph-config/2.0",
    )


def qlib_market_config(root: Path) -> ExperimentConfig:
    """Write tiny Qlib providers without requiring market metadata files."""

    import json

    config = mock_config(root)
    public = load_experiment_config()
    calendar = b"2020-01-01\n2020-01-02\n2020-01-03\n2020-01-04\n"
    universe_file = f"instruments/{config.universe.name}.txt"
    for name in ("market", "exposure", "mask"):
        provider = root / name
        (provider / "calendars").mkdir(parents=True, exist_ok=True)
        (provider / "instruments").mkdir(exist_ok=True)
        (provider / "calendars/day.txt").write_bytes(calendar)
        (provider / universe_file).write_bytes(b"SH600000\t2020-01-01\t2020-01-04\n")
    (root / "exposure/cosmosalpha_industry_manifest.json").write_text(
        json.dumps(
            {
                "build_status": "complete",
                "point_in_time_certified": False,
                "usable_for_industry_calculations": True,
            }
        ),
        encoding="utf-8",
    )
    market = root / "market"
    feature_dir = market / "features/sh600000"
    feature_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(sorted(item.field_name for item in public.qlib.market.fields))
    for name in names:
        (feature_dir / f"{name}.day.bin").write_bytes(
            np.array([0, 1, 2, 3, 4], dtype="<f4").tobytes()
        )
    return replace(
        config,
        data=replace(config.data, provider="qlib"),
        qlib=replace(
            public.qlib,
            market=replace(
                public.qlib.market,
                provider_uri=str(market),
            ),
            exposure=replace(
                public.qlib.exposure,
                provider_uri=str(root / "exposure"),
            ),
            mask=replace(
                public.qlib.mask,
                provider_uri=str(root / "mask"),
            ),
        ),
    )


def model_config(root: Path) -> LightGBMIntegrationConfig:
    """Return a tiny daily model config aligned with the fixed source domain."""
    from cosmosalpha.modeling import DateSegmentConfig, load_lightgbm_integration_config

    public = load_lightgbm_integration_config(
        "configs/lightgbm_integration.example.toml"
    )
    source = qlib_market_config(root)
    return replace(
        public,
        factor_source=replace(
            public.factor_source, storage_root=str(root), run_id="source"
        ),
        segments=replace(
            public.segments,
            train=DateSegmentConfig("2020-01-02", "2020-01-02"),
            valid=DateSegmentConfig("2020-01-03", "2020-01-03"),
            test=DateSegmentConfig("2020-01-04", "2020-01-04"),
        ),
        data=replace(
            public.data,
            provider=source.data.provider,
            dataset_version=source.data.dataset_version,
            trading_calendar_version=source.data.trading_calendar_version,
            start_date="2020-01-02",
            end_date="2020-01-04",
            warmup_start_date="2020-01-01",
        ),
        universe=source.universe,
        qlib=source.qlib,
        feature_preprocessing=replace(
            public.feature_preprocessing, min_cross_section_samples=4
        ),
        output=replace(public.output, root=str(root)),
    )


def write_factor_source(
    root: Path, statuses: tuple[str, ...] = ("accepted", "rejected", "failed")
) -> None:
    """Write a native Graph source with self-consistent status and lineage."""
    import csv

    from cosmosalpha.agents import AcceptanceDecision, EvaluationResult, MetricValue
    from cosmosalpha.core import create_config_snapshot
    from cosmosalpha.storage import (
        LocalArtifactRepository,
        LocalExperimentRepository,
        SqliteFactorLibrary,
    )

    layout = root / "source"
    artifacts = LocalArtifactRepository(layout)
    objects = LocalExperimentRepository(layout)
    config = graph_runtime_config(root)
    market = qlib_market_config(root)
    config = replace(
        config,
        data=replace(config.data, provider="qlib", end_date="2020-01-02"),
        qlib=market.qlib,
    )
    snapshot = create_config_snapshot(config)
    objects.put_object("config_snapshot", snapshot.snapshot_id, snapshot)
    SqliteFactorLibrary.initialize(
        layout / "factor-library.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    rows: list[dict[str, str]] = []
    for index, status in enumerate(statuses):
        factor_id = f"factor-{index}"
        values = artifacts.put_bytes(
            f"historical values {index}".encode(),
            kind="evaluated_factor_values",
            media_type="application/octet-stream",
        )
        spec = make_spec(
            "rank(volume)", required_fields=("volume",), factor_id=factor_id
        )
        objects.put_object("factor_spec", factor_id, spec)
        evaluation = EvaluationResult(
            factor_id=factor_id,
            data_snapshot_id="fixed-source",
            raw_factor_values_ref=values,
            evaluated_factor_values_ref=values,
            factor_preprocessing_result_ref=ObjectRef(
                "factor_preprocessing_result",
                f"pre-{index}",
                "factor-preprocessing-result/1.0",
            ),
            forward_return_ref=values,
            evaluation_protocol_id="single_factor_ic_v2",
            preprocessing_protocol_id="gplearn_neutralized_v1",
            neutralization_applied=True,
            metrics=(
                MetricValue("rank_ic_mean", 0.0, "higher_is_better", 8, "fixed/1.0"),
            ),
            effective_date_count=1,
            effective_sample_count=8,
            coverage=1.0,
            warnings=(),
            evaluator_version="fixed/1.0",
            schema_version="evaluation-result/1.1",
            total_date_count=1,
        )
        evaluation_ref = objects.put_object(
            "evaluation_result", f"eval-{index}", evaluation
        )
        decision = AcceptanceDecision(
            factor_id=factor_id,
            evaluation_result_ref=evaluation_ref,
            accepted=status == "accepted",
            reason_codes=(
                (
                    "PASS_RANKIC_THRESHOLD"
                    if status == "accepted"
                    else "FAIL_RANKIC_THRESHOLD"
                ),
            ),
            policy_id="fixed",
            policy_version="1.0.0",
            schema_version="acceptance-decision/1.0",
        )
        objects.put_object("acceptance_decision", f"decision-{index}", decision)
        rows.append(
            {
                "status": status,
                "expression": spec.expression,
                "factor_id": factor_id,
                "frequency": spec.frequency,
                "experiment_id": f"experiment-{index}",
                "data_snapshot_id": "fixed-source",
                "evaluated_factor_values_artifact_id": values.artifact_id,
            }
        )
    csv_path = layout / "factor_library/factors.csv"
    csv_path.parent.mkdir()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
