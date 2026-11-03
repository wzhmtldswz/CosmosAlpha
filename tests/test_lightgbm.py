"""Daily all-factor training, zero costs and source-lineage checks."""

from __future__ import annotations

import importlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cosmosalpha.modeling.lightgbm as modeling
from cosmosalpha.core import CoreValidationError, stable_sha256, to_json, to_json_value
from cosmosalpha.data import FixedSampleMarketDataProvider, MarketDataRequest
from cosmosalpha.modeling import (
    DateSegmentConfig,
    FactorExclusion,
    FactorSelectionResult,
    LightGBMIntegrationConfig,
    LightGBMPreflightReport,
    ModelFactorInput,
    ModelRuntimeVersions,
    PortfolioMetrics,
    SignalMetrics,
    load_lightgbm_integration_config,
    load_lightgbm_integration_config_values,
    select_factors_greedily,
)
from cosmosalpha.storage import LocalArtifactRepository, LocalExperimentRepository
from tests.fixtures import candidate, model_config, write_factor_source


def select_all(
    features: dict[str, np.ndarray],
    *,
    candidates: tuple[ModelFactorInput, ...] | None = None,
    returns: np.ndarray | None = None,
    pre_exclusions: tuple[FactorExclusion, ...] = (),
) -> FactorSelectionResult:
    """Use strict IC coverage settings that must not screen unfiltered features."""
    return select_factors_greedily(
        (
            tuple(candidate(name) for name in features)
            if candidates is None
            else candidates
        ),
        features,
        np.full(8, np.nan) if returns is None else returns,
        datetimes=("2020-01-02",) * 4 + ("2022-01-03",) * 4,
        eligible=np.ones(8, dtype=bool),
        train_segment=DateSegmentConfig("2020-01-02", "2020-01-02"),
        enabled=False,
        threshold=0.8,
        min_cross_section_samples=100,
        min_daily_effective_stock_ratio=1.0,
        min_effective_date_ratio=1.0,
        pre_exclusions=pre_exclusions,
    )


def test_daily_example_and_omitted_cost_defaults_are_zero() -> None:
    config = load_lightgbm_integration_config(
        "configs/lightgbm_integration.example.toml"
    )
    values = to_json_value(config)
    assert isinstance(values, dict)
    values.pop("backtest")
    values.pop("correlation_filter")
    values.pop("factor_source")
    values["factor_source"] = {"storage_root": "runs", "run_id": "source"}
    defaults = load_lightgbm_integration_config_values(values)
    for checked in (config, defaults):
        assert checked.factor_source.include_rejected_factors
        assert not checked.correlation_filter.enabled
        assert checked.label.horizon_days == 1
        assert checked.label.return_definition == "close_t1_over_close_t"
        assert checked.label.normalization == "qlib_cs_rank_norm_v1"
        assert checked.feature_preprocessing.neutralization_enabled
        assert checked.feature_preprocessing.protocol == "model_feature_neutralized_v1"
        assert checked.feature_preprocessing.industry_field == "industry_id"
        assert checked.feature_preprocessing.continuous_exposures == (
            "log_mkt_cap",
            "ret20",
            "turn20",
            "vol20",
        )
        assert checked.model.num_threads == 4
        assert checked.parallel.factor_workers == 1
        assert checked.parallel.native_threads == 1
        assert (
            checked.backtest.open_cost,
            checked.backtest.close_cost,
            checked.backtest.min_cost,
            checked.backtest.impact_cost,
        ) == (0, 0, 0, 0)
        assert (checked.backtest.topk, checked.backtest.n_drop) == (50, 5)
        assert checked.backtest.return_mode == "absolute"
        assert checked.backtest.benchmark == ""
        assert checked.backtest.method_sell == "bottom"
        assert checked.backtest.method_buy == "top"
        assert checked.backtest.deal_price == "open"


@pytest.mark.parametrize("enabled", [True, False])
def test_neutralization_switch_resolves_protocol_and_rejects_conflicts(
    enabled: bool,
) -> None:
    with Path("configs/lightgbm_integration.example.toml").open("rb") as config_file:
        values = tomllib.load(config_file)
    preprocessing = values["feature_preprocessing"]
    preprocessing["neutralization_enabled"] = enabled
    expected = (
        "model_feature_neutralized_v1" if enabled else "model_feature_mad_zscore_v1"
    )
    resolved = load_lightgbm_integration_config_values(values)
    assert resolved.feature_preprocessing.protocol == expected
    assert resolved.feature_preprocessing.neutralization_enabled is enabled
    preprocessing["protocol"] = (
        "model_feature_mad_zscore_v1" if enabled else "model_feature_neutralized_v1"
    )
    with pytest.raises(CoreValidationError, match="must agree"):
        load_lightgbm_integration_config_values(values)


def test_unfiltered_features_ignore_rank_ic_correlations_and_status() -> None:
    values = np.arange(8, dtype=float)
    features = {
        "positive": values,
        "negative": -values,
        "duplicate": values.copy(),
        "constant": np.ones(8),
        "sparse": np.array([1, np.nan, np.nan, np.nan] * 2),
    }
    first_candidates = tuple(candidate(name, "accepted") for name in features)
    second_candidates = tuple(candidate(name, "rejected") for name in features)
    first = select_all(features, candidates=first_candidates)
    second = select_all(features, candidates=second_candidates, returns=-values)
    assert (
        first.selected_factor_ids
        == second.selected_factor_ids
        == tuple(sorted(features))
    )
    assert all(
        d.rank_ic_mean is None and d.blocker_factor_id is None for d in first.decisions
    )
    assert first.included_statuses == ("accepted", "rejected")
    assert first.schema_version == "lightgbm-factor-selection/1.1"
    assert '"rank_ic_mean":null' in modeling.to_json(first)


def test_only_training_feature_availability_controls_admission() -> None:
    features = {"a": np.arange(8, dtype=float)}
    first = select_all(features, returns=np.arange(8, dtype=float))
    changed_labels = np.array([0, 1, 2, 3, 1e9, -1e9, np.nan, 0], dtype=float)
    second = select_all(features, returns=changed_labels)
    assert first == second
    future_missing = {"a": np.array([0, 1, 2, 3, np.nan, np.nan, np.nan, np.nan])}
    assert select_all(future_missing).selected_factor_ids == ("a",)


def test_missing_failed_and_training_empty_features_are_audited() -> None:
    candidates = tuple(
        candidate(name) for name in ("good", "missing", "error", "future")
    )
    features = {
        "good": np.arange(8, dtype=float),
        "future": np.array([np.nan] * 4 + [1.0] * 4),
    }
    failure = FactorExclusion(
        "error",
        "rejected",
        "feature_calculation",
        "CALCULATION_FAILED",
        "fixed failure",
    )
    result = select_all(features, candidates=candidates, pre_exclusions=(failure,))
    assert result.selected_factor_ids == ("good",)
    reasons = {item.factor_id: item.error_code for item in result.excluded_factors}
    assert reasons == {
        "error": "CALCULATION_FAILED",
        "missing": "FEATURE_NOT_AVAILABLE",
        "future": "NO_FINITE_TRAINING_FEATURE_VALUES",
    }
    with pytest.raises(
        modeling.LightGBMIntegrationError, match="EMPTY_AFTER_EXCLUSIONS"
    ):
        select_all({"empty": np.full(8, np.nan)})
    with pytest.raises(modeling.LightGBMIntegrationError, match="UNIQUE_CANDIDATES"):
        select_all({})
    with pytest.raises(modeling.LightGBMIntegrationError, match="INDEX_MISMATCH"):
        select_all({"bad": np.ones(3)})


def test_optional_correlation_filter_still_works() -> None:
    values = np.array([1, 2, 3, 4] * 2, dtype=float)
    result = select_factors_greedily(
        (candidate("a"), candidate("b")),
        {"a": values, "b": -values},
        values,
        datetimes=("2020-01-02",) * 4 + ("2020-01-03",) * 4,
        eligible=np.ones(8, dtype=bool),
        train_segment=DateSegmentConfig("2020-01-02", "2020-01-03"),
        enabled=True,
        threshold=0.8,
        min_cross_section_samples=4,
        min_daily_effective_stock_ratio=1.0,
        min_effective_date_ratio=1.0,
    )
    assert len(result.selected_factor_ids) == 1
    assert sum(not item.selected for item in result.decisions) == 1


def test_legacy_graph_source_snapshot_remains_readable(tmp_path: Path) -> None:
    write_factor_source(tmp_path)
    (snapshot_path,) = (tmp_path / "source/objects/config_snapshot").glob("*.json")
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot["config"].pop("mcts")
    snapshot["config"]["schema_version"] = "experiment-config/1.5"
    snapshot["redacted_json"] = to_json(snapshot["config"])
    snapshot["snapshot_id"] = stable_sha256(snapshot["config"])
    legacy_path = snapshot_path.with_name(f'{snapshot["snapshot_id"]}.json')
    legacy_bytes = to_json(snapshot).encode()
    snapshot_path.unlink()
    legacy_path.write_bytes(legacy_bytes)
    config = model_config(tmp_path)
    source = modeling._load_source_bundle(config)
    modeling._validate_source_compatibility(config, source.source_config)
    assert source.source_config.snapshot_id == snapshot["snapshot_id"]
    assert source.source_config.config.schema_version == "experiment-config/1.5"
    assert source.source_config.config.mcts is None
    assert legacy_path.read_bytes() == legacy_bytes


def test_native_source_includes_both_statuses_and_excludes_failed(
    tmp_path: Path,
) -> None:
    write_factor_source(tmp_path)
    config = model_config(tmp_path)
    source = modeling._load_source_bundle(config)
    assert tuple(c.source_status for c in source.candidates) == ("accepted", "rejected")
    assert source.excluded_status_counts == (("failed", 1),)
    modeling._validate_source_compatibility(config, source.source_config)
    invalid = replace(
        source.source_config,
        config=replace(
            source.source_config.config,
            architecture=replace(
                source.source_config.config.architecture, name="chain"
            ),
        ),
    )
    with pytest.raises(modeling.LightGBMIntegrationError, match="MUST_BE_GRAPH"):
        modeling._validate_source_compatibility(config, invalid)
    overlapping = replace(
        source.source_config,
        config=replace(
            source.source_config.config,
            data=replace(source.source_config.config.data, end_date="2020-01-04"),
        ),
    )
    with pytest.raises(modeling.LightGBMIntegrationError, match="SOURCE_"):
        modeling._validate_source_compatibility(config, overlapping)
    csv_path = tmp_path / "source/factor_library/factors.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("rejected", "accepted"),
        encoding="utf-8",
    )
    with pytest.raises(modeling.LightGBMIntegrationError, match="LINEAGE_MISMATCH"):
        modeling._load_source_bundle(config)


def test_model_source_identity_checks_allow_relocation(tmp_path: Path) -> None:
    write_factor_source(tmp_path)
    config = model_config(tmp_path)
    source = modeling._load_source_bundle(config).source_config
    for name in ("market", "exposure", "mask"):
        provider = getattr(config.qlib, name)
        relocated = replace(
            config,
            qlib=replace(
                config.qlib,
                **{name: replace(provider, provider_uri="different-directory")},
            ),
        )
        modeling._validate_source_compatibility(relocated, source)
        changed_fields = replace(
            provider,
            fields=(replace(provider.fields[0], expression="$high"),),
        )
        invalid = replace(config, qlib=replace(config.qlib, **{name: changed_fields}))
        with pytest.raises(
            modeling.LightGBMIntegrationError, match="DATA_IDENTITY_MISMATCH"
        ):
            modeling._validate_source_compatibility(invalid, source)


def test_model_preflight_needs_no_market_manifest(tmp_path: Path) -> None:
    write_factor_source(tmp_path)
    config = model_config(tmp_path)
    assert not (
        Path(config.qlib.market.provider_uri) / "cosmosalpha_market_manifest.json"
    ).exists()
    report = modeling.preflight_lightgbm_integration(config, "valid-model")
    assert report.candidate_count == 2
    assert {issue.code for issue in report.issues} <= {"MODEL_DEPENDENCY_MISSING"}
    assert not (tmp_path / "valid-model").exists()


@pytest.mark.parametrize("source_name", ("market", "exposure", "mask"))
def test_model_preflight_checks_source_without_outputs(
    tmp_path: Path, source_name: str
) -> None:
    write_factor_source(tmp_path)
    config = model_config(tmp_path)
    provider = getattr(config.qlib, source_name)
    changed = replace(
        provider, fields=(replace(provider.fields[0], expression="$high"),)
    )
    config = replace(config, qlib=replace(config.qlib, **{source_name: changed}))
    report = modeling.preflight_lightgbm_integration(config, "invalid-model")
    assert not report.ready
    issues = [
        issue for issue in report.issues if issue.code == "SOURCE_OR_CONFIG_INVALID"
    ]
    assert len(issues) == 1
    assert "DATA_IDENTITY_MISMATCH" in issues[0].message
    assert not (tmp_path / "invalid-model").exists()


def test_model_assembly_passes_all_features_and_saves_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_factor_source(tmp_path)
    config = model_config(tmp_path)
    observed: dict[str, object] = {}

    class FakeBooster:
        best_iteration = 2

        def model_to_string(self) -> str:
            return "fixed model"

    class FakeModel:
        model = FakeBooster()

        def predict(self, dataset: object, *, segment: str) -> object:
            observed.setdefault("predicted", []).append(segment)
            return segment

    def fixed_snapshot(
        _config: LightGBMIntegrationConfig,
        _specs: object,
        artifacts: LocalArtifactRepository,
    ) -> object:
        provider = FixedSampleMarketDataProvider(
            artifacts,
            instruments=tuple(f"sh{600000 + index}" for index in range(12)),
        )
        return provider.load_snapshot(
            MarketDataRequest(
                market="cn",
                universe="all_a_share",
                start_date="2020-01-02",
                end_date="2020-01-04",
                warmup_start_date="2020-01-01",
                frequency="1d",
                fields=("close", "volume"),
                dataset_version="fixed-model",
                price_adjustment="post_adjusted",
                schema_version="market-data/1.0",
            )
        )

    def fixed_dataset(
        _config: object, selected: tuple[str, ...], *_args: object
    ) -> tuple[object, object]:
        observed["selected"] = selected
        return object(), object()

    monkeypatch.setattr(
        modeling,
        "preflight_lightgbm_integration",
        lambda *_: LightGBMPreflightReport("model", True, 2, ()),
    )
    monkeypatch.setattr(modeling, "_load_model_snapshot", fixed_snapshot)
    monkeypatch.setattr(modeling, "_build_qlib_dataset", fixed_dataset)
    monkeypatch.setattr(
        modeling,
        "_fit_qlib_model",
        lambda *_: (
            FakeModel(),
            {"valid": {"l2": [1.0]}},
            ModelRuntimeVersions("3.12", "mock", "mock", "mock"),
        ),
    )
    monkeypatch.setattr(
        modeling,
        "_evaluate_predictions",
        lambda *_args, **_kwargs: (
            ({"datetime": "2020-01-04", "prediction": 0.1},),
            ({"datetime": "2020-01-04", "ic": 0.2},),
            SignalMetrics(0.2, 0.1, 2.0, 0.3, 0.1, 3.0, 12, 1),
        ),
    )
    monkeypatch.setattr(modeling, "_concat_prediction_segments", lambda *_: object())
    monkeypatch.setattr(
        modeling,
        "_run_qlib_backtest",
        lambda *_: (
            PortfolioMetrics(0.1, 1.0, -0.1, 1.0),
            ({"datetime": "2020-01-04", "return": 0.01},),
            (),
        ),
    )
    result = modeling.run_lightgbm_integration(config, "model")
    assert observed["selected"] == ("factor-0", "factor-1")
    assert observed["predicted"] == ["valid", "test"]
    assert (result.candidate_factor_count, result.selected_factor_count) == (2, 2)
    artifacts = LocalArtifactRepository(tmp_path / "model")
    for ref in (
        result.model_ref,
        result.predictions_ref,
        result.training_curve_ref,
        result.portfolio_report_ref,
    ):
        assert artifacts.verify(ref)
    objects = LocalExperimentRepository(tmp_path / "model")
    selection = objects.get_object(result.factor_selection_ref, FactorSelectionResult)
    assert selection.included_statuses == ("accepted", "rejected")
    assert not selection.excluded_factors


def test_daily_backtest_passes_zero_costs_and_daily_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qlib = pytest.importorskip("qlib")
    pd = pytest.importorskip("pandas")
    backtest_module = importlib.import_module("qlib.backtest")
    evaluate = importlib.import_module("qlib.contrib.evaluate")
    from types import SimpleNamespace

    data_module = importlib.import_module("qlib.data")
    data = SimpleNamespace(
        instruments=lambda _: "all_a_share",
        list_instruments=lambda *_args, **_kwargs: ["sh600000"],
    )
    monkeypatch.setattr(data_module, "D", data)
    config = model_config(tmp_path)
    observed: dict[str, object] = {}
    report = pd.DataFrame(
        {"return": [0.01, -0.02], "cost": [0.0, 0.0]},
        index=pd.to_datetime(["2020-01-04", "2020-01-05"]),
    )

    def backtest(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        observed.update(kwargs)
        return {"1day": (report, {})}, {}

    monkeypatch.setattr(qlib, "init", lambda **_: None)
    monkeypatch.setattr(backtest_module, "backtest", backtest)
    monkeypatch.setattr(
        evaluate,
        "risk_analysis",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"risk": [0.25, 1.5, -0.15]},
            index=["annualized_return", "information_ratio", "max_drawdown"],
        ),
    )
    metrics, _, _ = modeling._run_qlib_backtest(config, object())
    assert observed["executor"]["kwargs"]["time_per_step"] == "day"
    fees = observed["exchange_kwargs"]
    assert all(
        fees[name] == 0
        for name in ("open_cost", "close_cost", "min_cost", "impact_cost")
    )
    assert observed["strategy"]["kwargs"]["topk"] == 50
    assert observed["strategy"]["kwargs"]["n_drop"] == 5
    assert metrics.annualized_return == 0.25
