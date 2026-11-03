"""Independent Qlib LightGBM training over usable CosmosAlpha Graph factors."""

from __future__ import annotations

import csv
import importlib.util
import math
import multiprocessing
import os
import platform
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from cosmosalpha.agents import AcceptanceDecision, EvaluationResult, FactorSpec
from cosmosalpha.core import (
    COLUMNAR_RECORDS_MEDIA_TYPE,
    ArtifactRef,
    ConfigSnapshot,
    CoreError,
    ObjectRef,
    QlibSourceConfig,
    decode_record_artifact,
    encode_columnar_records,
    stable_sha256,
    to_json,
)
from cosmosalpha.data import (
    MARKET_DATA_FIELDS,
    DataSnapshot,
    MarketDataRequest,
    QlibFeatureSource,
    QlibFieldMapping,
    QlibMarketDataProvider,
)
from cosmosalpha.data.qlib_mask import read_mask_new_stock_min_age_years
from cosmosalpha.evaluation import ForwardReturnCalculator, preprocess_feature_values
from cosmosalpha.evaluation._statistics import safe_pearson
from cosmosalpha.factors import CalculationFrame, FactorCalculator, evaluate_factor_spec
from cosmosalpha.modeling.config import (
    DateSegmentConfig,
    LightGBMIntegrationConfig,
    QlibLightGBMConfig,
    create_lightgbm_config_snapshot,
)
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    RunStorageLayout,
    SqliteFactorLibrary,
)

LIGHTGBM_SELECTION_SCHEMA_VERSION = "lightgbm-factor-selection/1.1"
LIGHTGBM_RUN_SCHEMA_VERSION = "lightgbm-run-result/1.0"
LIGHTGBM_PREFLIGHT_SCHEMA_VERSION = "lightgbm-preflight/1.0"
LIGHTGBM_RUNTIME_VERSIONS_SCHEMA_VERSION = "lightgbm-runtime-versions/1.0"
LIGHTGBM_SIGNAL_METRICS_SCHEMA_VERSION = "lightgbm-signal-metrics/1.0"
MODEL_CORRELATION_PROTOCOL = "model-correlation/daily-cross-sectional-pearson/v1"


class LightGBMIntegrationError(Exception):
    """Report a deterministic model integration failure."""


@dataclass(frozen=True, slots=True)
class LightGBMPreflightIssue:
    """One machine-readable preflight issue."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class LightGBMPreflightReport:
    """Lightweight source, configuration, and dependency validation result."""

    run_id: str
    ready: bool
    candidate_count: int
    issues: tuple[LightGBMPreflightIssue, ...]
    excluded_status_counts: tuple[tuple[str, int], ...] = ()
    schema_version: str = LIGHTGBM_PREFLIGHT_SCHEMA_VERSION

    def summary_json(self) -> str:
        """Return the complete preflight report as strict JSON."""
        return to_json(self)


@dataclass(frozen=True, slots=True)
class ModelFactorInput:
    """Auditable Graph factor used as a model candidate."""

    factor_id: str
    expression: str
    factor_ref: ObjectRef
    evaluation_result_ref: ObjectRef
    acceptance_decision_ref: ObjectRef
    source_evaluated_values_ref: ArtifactRef
    source_data_snapshot_id: str
    source_experiment_id: str
    source_status: str = "accepted"


@dataclass(frozen=True, slots=True)
class FactorExclusion:
    """One factor excluded during model-side calculation or selection."""

    factor_id: str
    source_status: str
    stage: str
    error_code: str
    message: str
    valid_date_count: int | None = None
    required_date_count: int | None = None


@dataclass(frozen=True, slots=True)
class FactorSelectionDecision:
    """One deterministic greedy selection decision."""

    factor_id: str
    selected: bool
    rank_ic_mean: float | None
    blocker_factor_id: str | None
    signed_correlation: float | None
    blocker_rank_ic_mean: float | None


@dataclass(frozen=True, slots=True)
class FactorSelectionResult:
    """Complete model feature ordering and correlation audit."""

    enabled: bool
    threshold: float
    correlation_protocol: str
    candidates: tuple[ModelFactorInput, ...]
    selected_factor_ids: tuple[str, ...]
    decisions: tuple[FactorSelectionDecision, ...]
    candidate_source: str = "factor_library.sqlite3"
    included_statuses: tuple[str, ...] = ("accepted", "rejected")
    excluded_statuses: tuple[str, ...] = ("failed",)
    excluded_status_counts: tuple[tuple[str, int], ...] = ()
    usable_factor_ids: tuple[str, ...] = ()
    excluded_factors: tuple[FactorExclusion, ...] = ()
    schema_version: str = LIGHTGBM_SELECTION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """Aggregate test-period signal statistics against raw forward returns."""

    ic_mean: float
    ic_std: float
    ic_ir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    effective_sample_count: int
    valid_date_count: int
    schema_version: str = LIGHTGBM_SIGNAL_METRICS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Qlib risk metrics on cost-adjusted absolute portfolio returns."""

    annualized_return: float
    information_ratio: float
    max_drawdown: float
    calmar_ratio: float
    return_mode: str = "absolute"
    schema_version: str = "lightgbm-portfolio-metrics/1.0"


@dataclass(frozen=True, slots=True)
class ModelRuntimeVersions:
    """Installed package versions that affect model reproduction."""

    python: str
    qlib: str
    lightgbm: str
    pandas: str
    schema_version: str = LIGHTGBM_RUNTIME_VERSIONS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LightGBMRunResult:
    """Immutable references and metrics for one completed model run."""

    run_id: str
    source_run_id: str
    source_config_snapshot_ref: ObjectRef
    config_snapshot_ref: ObjectRef
    data_snapshot_ref: ObjectRef
    factor_selection_ref: ObjectRef
    model_ref: ArtifactRef
    training_curve_ref: ArtifactRef
    predictions_ref: ArtifactRef
    daily_signal_metrics_ref: ArtifactRef
    portfolio_report_ref: ArtifactRef
    portfolio_positions_ref: ArtifactRef
    portfolio_metrics: PortfolioMetrics
    metrics: SignalMetrics
    runtime_versions: ModelRuntimeVersions
    candidate_factor_count: int
    selected_factor_count: int
    removed_factor_count: int
    best_iteration: int
    skipped_factor_count: int = 0
    schema_version: str = LIGHTGBM_RUN_SCHEMA_VERSION

    def summary_json(self) -> str:
        """Return a compact command-line result."""
        return to_json(
            {
                "run_id": self.run_id,
                "source_run_id": self.source_run_id,
                "source_config_snapshot_ref": self.source_config_snapshot_ref,
                "candidate_factor_count": self.candidate_factor_count,
                "selected_factor_count": self.selected_factor_count,
                "removed_factor_count": self.removed_factor_count,
                "skipped_factor_count": self.skipped_factor_count,
                "best_iteration": self.best_iteration,
                "metrics": self.metrics,
                "runtime_versions": self.runtime_versions,
                "config_snapshot_ref": self.config_snapshot_ref,
                "data_snapshot_ref": self.data_snapshot_ref,
                "factor_selection_ref": self.factor_selection_ref,
                "model_ref": self.model_ref,
                "training_curve_ref": self.training_curve_ref,
                "predictions_ref": self.predictions_ref,
                "daily_signal_metrics_ref": self.daily_signal_metrics_ref,
                "portfolio_report_ref": self.portfolio_report_ref,
                "portfolio_positions_ref": self.portfolio_positions_ref,
                "portfolio_metrics": self.portfolio_metrics,
                "display_metrics": {
                    "IC_percent": self.metrics.ic_mean * 100.0,
                    "ICIR_percent": self.metrics.ic_ir * 100.0,
                    "RankIC_percent": self.metrics.rank_ic_mean * 100.0,
                    "RankICIR_percent": self.metrics.rank_ic_ir * 100.0,
                    "ARR_percent": self.portfolio_metrics.annualized_return * 100.0,
                    "IR": self.portfolio_metrics.information_ratio,
                    "MDD_percent": self.portfolio_metrics.max_drawdown * 100.0,
                    "CR": self.portfolio_metrics.calmar_ratio,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class _SourceBundle:
    candidates: tuple[ModelFactorInput, ...]
    specs: tuple[FactorSpec, ...]
    source_config: ConfigSnapshot
    source_config_ref: ObjectRef
    candidate_source: str
    included_statuses: tuple[str, ...]
    excluded_statuses: tuple[str, ...]
    excluded_status_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _FeatureWorkerInit:
    """Pickle-safe inputs used to initialize one feature worker."""

    artifact_root: str
    snapshot: DataSnapshot
    required_fields: tuple[str, ...]
    industry_field: str
    continuous_exposures: tuple[str, ...]
    mad_multiplier: float
    min_cross_section_samples: int
    neutralize: bool


@dataclass(frozen=True, slots=True)
class _FeatureWorkerState:
    """Read-only feature calculation state local to one worker process."""

    frame: CalculationFrame
    industry: np.ndarray
    continuous: np.ndarray
    mad_multiplier: float
    min_cross_section_samples: int
    neutralize: bool


_FEATURE_WORKER_STATE: _FeatureWorkerState | None = None


def preflight_lightgbm_integration(
    config: LightGBMIntegrationConfig,
    run_id: str,
) -> LightGBMPreflightReport:
    """Validate a model run without loading market data or training."""
    issues: list[LightGBMPreflightIssue] = []
    candidate_count = 0
    excluded_status_counts: tuple[tuple[str, int], ...] = ()
    issues.extend(_lightgbm_named_universe_issues(config))
    mask_policy_issue = _lightgbm_mask_policy_issue(config)
    if mask_policy_issue is not None:
        issues.append(mask_policy_issue)
    try:
        output_layout = RunStorageLayout.resolve(config.output.root, run_id)
        if output_layout.run_root.exists():
            issues.append(
                LightGBMPreflightIssue(
                    "OUTPUT_RUN_EXISTS",
                    f"output run already exists: {output_layout.run_root}",
                )
            )
        source = _load_source_bundle(config)
        candidate_count = len(source.candidates)
        excluded_status_counts = source.excluded_status_counts
        _validate_source_compatibility(config, source.source_config)
    except (
        CoreError,
        OSError,
        ValueError,
        LightGBMIntegrationError,
    ) as error:
        issues.append(LightGBMPreflightIssue("SOURCE_OR_CONFIG_INVALID", str(error)))

    missing = tuple(
        name
        for name in ("pandas", "qlib", "lightgbm")
        if importlib.util.find_spec(name) is None
    )
    if missing:
        issues.append(
            LightGBMPreflightIssue(
                "MODEL_DEPENDENCY_MISSING",
                "missing model dependencies: " + ", ".join(missing),
            )
        )
    return LightGBMPreflightReport(
        run_id=run_id,
        ready=not issues,
        candidate_count=candidate_count,
        issues=tuple(issues),
        excluded_status_counts=excluded_status_counts,
    )


def _lightgbm_named_universe_issues(
    config: LightGBMIntegrationConfig,
) -> tuple[LightGBMPreflightIssue, ...]:
    """Require identical local universe files for the main-board model run."""
    if config.universe.name != "cn_main_board" or config.qlib.instruments:
        return ()

    issues: list[LightGBMPreflightIssue] = []
    contents: list[bytes] = []
    for source_name, source in (
        ("market", config.qlib.market),
        ("exposure", config.qlib.exposure),
        ("mask", config.qlib.mask),
    ):
        if "://" in source.provider_uri or source.provider_uri.startswith(
            "replace-with-"
        ):
            continue
        universe_path = (
            Path(source.provider_uri).expanduser()
            / "instruments"
            / f"{config.universe.name}.txt"
        )
        if not universe_path.is_file():
            issues.append(
                LightGBMPreflightIssue(
                    "QLIB_UNIVERSE_FILE_MISSING",
                    f"{source_name} source is missing named universe: {universe_path}",
                )
            )
            continue
        try:
            contents.append(universe_path.read_bytes())
        except OSError as error:
            issues.append(
                LightGBMPreflightIssue("QLIB_UNIVERSE_FILE_UNREADABLE", str(error))
            )
    if len(set(contents)) > 1:
        issues.append(
            LightGBMPreflightIssue(
                "QLIB_UNIVERSE_FILE_MISMATCH",
                "cn_main_board files differ across Qlib sources",
            )
        )
    return tuple(issues)


def _lightgbm_mask_policy_issue(
    config: LightGBMIntegrationConfig,
) -> LightGBMPreflightIssue | None:
    """Validate the optional versioned new-stock policy of a local Mask source."""
    mask_uri = config.qlib.mask.provider_uri
    if "://" in mask_uri or mask_uri.startswith("replace-with-"):
        return None
    try:
        actual = read_mask_new_stock_min_age_years(Path(mask_uri))
    except ValueError as error:
        return LightGBMPreflightIssue("QLIB_NEW_STOCK_POLICY_INVALID", str(error))
    if actual is None:
        if config.universe.name == "cn_main_board":
            return LightGBMPreflightIssue(
                "QLIB_NEW_STOCK_POLICY_MISSING",
                "cn_main_board requires a versioned Mask new-stock policy",
            )
        return None
    expected = config.universe.new_stock_min_age_years
    if actual != expected:
        return LightGBMPreflightIssue(
            "QLIB_NEW_STOCK_POLICY_MISMATCH",
            f"Mask new-stock age is {actual}, configuration requires {expected}",
        )
    return None


def select_factors_greedily(
    candidates: tuple[ModelFactorInput, ...],
    features: Mapping[str, np.ndarray],
    raw_returns: np.ndarray,
    *,
    datetimes: tuple[str, ...],
    eligible: np.ndarray,
    train_segment: DateSegmentConfig,
    enabled: bool,
    threshold: float,
    min_cross_section_samples: int,
    min_daily_effective_stock_ratio: float,
    min_effective_date_ratio: float,
    candidate_source: str = "factor_library.sqlite3",
    included_statuses: tuple[str, ...] = ("accepted", "rejected"),
    excluded_statuses: tuple[str, ...] = ("failed",),
    excluded_status_counts: tuple[tuple[str, int], ...] = (),
    pre_exclusions: tuple[FactorExclusion, ...] = (),
) -> FactorSelectionResult:
    """Keep usable features, optionally filtering by training-only correlation.

    With filtering disabled, labels and RankIC never determine admission.
    A feature only needs one finite value on an eligible training row.
    """
    candidate_ids = tuple(candidate.factor_id for candidate in candidates)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise LightGBMIntegrationError("FACTOR_SELECTION_REQUIRES_UNIQUE_CANDIDATES")
    if not set(features).issubset(candidate_ids):
        raise LightGBMIntegrationError("FACTOR_SELECTION_FEATURE_SET_MISMATCH")
    if raw_returns.shape != (len(datetimes),) or eligible.shape != (len(datetimes),):
        raise LightGBMIntegrationError("FACTOR_SELECTION_INDEX_MISMATCH")

    inputs_by_id = {item.factor_id: item for item in candidates}
    exclusions = list(pre_exclusions)
    pre_excluded_ids = {item.factor_id for item in pre_exclusions}
    scores: dict[str, float] = {}
    usable_ids: set[str] = set()
    training_rows = (
        _segment_mask(np.asarray(datetimes, dtype=str), train_segment) & eligible
    )
    for factor_id in candidate_ids:
        if factor_id in pre_excluded_ids:
            continue
        if factor_id not in features:
            exclusions.append(
                FactorExclusion(
                    factor_id=factor_id,
                    source_status=inputs_by_id[factor_id].source_status,
                    stage="feature_calculation",
                    error_code="FEATURE_NOT_AVAILABLE",
                    message="factor feature was not produced",
                )
            )
            continue
        values = features[factor_id]
        if values.shape != raw_returns.shape:
            raise LightGBMIntegrationError("FACTOR_SELECTION_INDEX_MISMATCH")
        if not enabled:
            if np.any(np.isfinite(values[training_rows])):
                usable_ids.add(factor_id)
            else:
                exclusions.append(
                    FactorExclusion(
                        factor_id=factor_id,
                        source_status=inputs_by_id[factor_id].source_status,
                        stage="feature_calculation",
                        error_code="NO_FINITE_TRAINING_FEATURE_VALUES",
                        message="factor has no finite feature values on eligible training rows",
                    )
                )
            continue
        try:
            scores[factor_id] = _mean_daily_correlation(
                features[factor_id],
                raw_returns,
                datetimes=datetimes,
                eligible=eligible,
                segment=train_segment,
                rank=True,
                min_cross_section_samples=min_cross_section_samples,
                min_daily_effective_stock_ratio=min_daily_effective_stock_ratio,
                min_effective_date_ratio=min_effective_date_ratio,
                label=f"RankIC for {factor_id}",
            )
        except LightGBMIntegrationError as error:
            exclusions.append(
                _factor_exclusion_from_error(inputs_by_id[factor_id], "rank_ic", error)
            )
    if not (scores if enabled else usable_ids):
        raise LightGBMIntegrationError("FACTOR_SELECTION_EMPTY_AFTER_EXCLUSIONS")
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.factor_id))
    if not enabled:
        return FactorSelectionResult(
            enabled=False,
            threshold=threshold,
            correlation_protocol=MODEL_CORRELATION_PROTOCOL,
            candidates=ordered_candidates,
            selected_factor_ids=tuple(
                item.factor_id
                for item in ordered_candidates
                if item.factor_id in usable_ids
            ),
            decisions=tuple(
                FactorSelectionDecision(
                    factor_id=item.factor_id,
                    selected=True,
                    rank_ic_mean=None,
                    blocker_factor_id=None,
                    signed_correlation=None,
                    blocker_rank_ic_mean=None,
                )
                for item in ordered_candidates
                if item.factor_id in usable_ids
            ),
            candidate_source=candidate_source,
            included_statuses=included_statuses,
            excluded_statuses=excluded_statuses,
            excluded_status_counts=excluded_status_counts,
            usable_factor_ids=tuple(sorted(usable_ids)),
            excluded_factors=tuple(exclusions),
        )
    priority = tuple(
        sorted(scores, key=lambda factor_id: (-abs(scores[factor_id]), factor_id))
    )
    selected: list[str] = []
    decisions: list[FactorSelectionDecision] = []
    correlation_excluded: set[str] = set()
    for factor_id in priority:
        blocker: str | None = None
        blocker_correlation: float | None = None
        try:
            for selected_id in selected:
                correlation = _mean_daily_correlation(
                    features[factor_id],
                    features[selected_id],
                    datetimes=datetimes,
                    eligible=eligible,
                    segment=train_segment,
                    rank=False,
                    min_cross_section_samples=min_cross_section_samples,
                    min_daily_effective_stock_ratio=min_daily_effective_stock_ratio,
                    min_effective_date_ratio=min_effective_date_ratio,
                    label=f"correlation for {factor_id}/{selected_id}",
                )
                if abs(correlation) >= threshold:
                    blocker = selected_id
                    blocker_correlation = correlation
                    break
        except LightGBMIntegrationError as error:
            exclusions.append(
                _factor_exclusion_from_error(
                    inputs_by_id[factor_id], "correlation", error
                )
            )
            correlation_excluded.add(factor_id)
            continue
        keep = blocker is None
        if keep:
            selected.append(factor_id)
        decisions.append(
            FactorSelectionDecision(
                factor_id=factor_id,
                selected=keep,
                rank_ic_mean=scores[factor_id],
                blocker_factor_id=blocker,
                signed_correlation=blocker_correlation,
                blocker_rank_ic_mean=scores[blocker] if blocker is not None else None,
            )
        )
    if not selected:
        raise LightGBMIntegrationError("FACTOR_SELECTION_EMPTY")
    return FactorSelectionResult(
        enabled=True,
        threshold=threshold,
        correlation_protocol=MODEL_CORRELATION_PROTOCOL,
        candidates=ordered_candidates,
        selected_factor_ids=tuple(selected),
        decisions=tuple(decisions),
        candidate_source=candidate_source,
        included_statuses=included_statuses,
        excluded_statuses=excluded_statuses,
        excluded_status_counts=excluded_status_counts,
        usable_factor_ids=tuple(
            factor_id for factor_id in priority if factor_id not in correlation_excluded
        ),
        excluded_factors=tuple(exclusions),
    )


def run_lightgbm_integration(
    config: LightGBMIntegrationConfig,
    run_id: str,
) -> LightGBMRunResult:
    """Recompute usable Graph factors, train Qlib LightGBM, and persist results."""
    report = preflight_lightgbm_integration(config, run_id)
    if not report.ready:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        raise LightGBMIntegrationError(details)
    try:
        return _run_lightgbm_integration(config, run_id)
    except LightGBMIntegrationError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize the external model boundary
        raise LightGBMIntegrationError(str(error)) from error


def _run_lightgbm_integration(
    config: LightGBMIntegrationConfig,
    run_id: str,
) -> LightGBMRunResult:
    source = _load_source_bundle(config)
    _validate_source_compatibility(config, source.source_config)
    layout = RunStorageLayout.resolve(config.output.root, run_id)
    artifacts = LocalArtifactRepository(layout.run_root)
    objects = LocalExperimentRepository(layout.run_root)

    config_snapshot = create_lightgbm_config_snapshot(config)
    config_snapshot_ref = objects.put_object(
        "lightgbm_config_snapshot",
        config_snapshot.snapshot_id,
        config_snapshot,
    )
    snapshot = _load_model_snapshot(config, source.specs, artifacts)
    data_snapshot_ref = objects.put_object(
        "data_snapshot", snapshot.snapshot_id, snapshot
    )
    required_fields = _required_market_fields(source.specs)
    frame = FactorCalculator(artifacts).load_frame(snapshot, required_fields)
    forward_return_result = ForwardReturnCalculator(artifacts).calculate(snapshot)
    raw_returns = _aligned_artifact_values(
        artifacts,
        forward_return_result.return_values_ref,
        frame.datetimes,
        frame.instruments,
        "forward_return",
    )
    features, feature_exclusions = _calculate_model_features(
        config,
        source.candidates,
        source.specs,
        frame,
        artifact_root=str(layout.run_root),
        snapshot=snapshot,
        required_fields=required_fields,
    )
    candidates = source.candidates
    selection = select_factors_greedily(
        candidates,
        features,
        raw_returns,
        datetimes=frame.datetimes,
        eligible=frame.eligible_mask,
        train_segment=config.segments.train,
        enabled=config.correlation_filter.enabled,
        threshold=config.correlation_filter.threshold,
        min_cross_section_samples=(
            config.feature_preprocessing.min_cross_section_samples
        ),
        min_daily_effective_stock_ratio=(
            config.correlation_filter.min_daily_effective_stock_ratio
        ),
        min_effective_date_ratio=config.correlation_filter.min_effective_date_ratio,
        candidate_source=source.candidate_source,
        included_statuses=source.included_statuses,
        excluded_statuses=source.excluded_statuses,
        excluded_status_counts=source.excluded_status_counts,
        pre_exclusions=feature_exclusions,
    )
    selection_ref = objects.put_object(
        "lightgbm_factor_selection",
        stable_sha256(selection),
        selection,
    )

    dataset, raw_label_series = _build_qlib_dataset(
        config,
        selection.selected_factor_ids,
        features,
        raw_returns,
        frame.datetimes,
        frame.instruments,
    )
    model, evaluation_history, versions = _fit_qlib_model(config.model, dataset)
    validation_predictions = model.predict(dataset, segment="valid")
    predictions = model.predict(dataset, segment="test")
    prediction_records, daily_records, metrics = _evaluate_predictions(
        predictions,
        raw_label_series,
        min_cross_section_samples=(
            config.feature_preprocessing.min_cross_section_samples
        ),
    )
    backtest_signal = _concat_prediction_segments(
        validation_predictions,
        predictions,
    )
    portfolio_metrics, portfolio_records, position_records = _run_qlib_backtest(
        config,
        backtest_signal,
    )

    booster = getattr(model, "model", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        raise LightGBMIntegrationError("QLIB_LIGHTGBM_MODEL_NOT_AVAILABLE")
    model_ref = artifacts.put_bytes(
        booster.model_to_string().encode("utf-8"),
        kind="lightgbm_booster",
        media_type="text/plain",
    )
    training_curve_ref = artifacts.put_bytes(
        to_json(evaluation_history).encode("utf-8"),
        kind="lightgbm_training_curve",
        media_type="application/json",
    )
    predictions_ref = artifacts.put_bytes(
        encode_columnar_records(prediction_records),
        kind="lightgbm_predictions",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    daily_signal_metrics_ref = artifacts.put_bytes(
        encode_columnar_records(daily_records),
        kind="lightgbm_daily_signal_metrics",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    portfolio_report_ref = artifacts.put_bytes(
        encode_columnar_records(portfolio_records),
        kind="qlib_portfolio_report",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    portfolio_positions_ref = artifacts.put_bytes(
        encode_columnar_records(position_records),
        kind="qlib_portfolio_positions",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    best_iteration = int(getattr(booster, "best_iteration", 0) or 0)
    if best_iteration <= 0:
        best_iteration = config.model.num_boost_round
    result = LightGBMRunResult(
        run_id=run_id,
        source_run_id=config.factor_source.run_id,
        source_config_snapshot_ref=source.source_config_ref,
        config_snapshot_ref=config_snapshot_ref,
        data_snapshot_ref=data_snapshot_ref,
        factor_selection_ref=selection_ref,
        model_ref=model_ref,
        training_curve_ref=training_curve_ref,
        predictions_ref=predictions_ref,
        daily_signal_metrics_ref=daily_signal_metrics_ref,
        portfolio_report_ref=portfolio_report_ref,
        portfolio_positions_ref=portfolio_positions_ref,
        portfolio_metrics=portfolio_metrics,
        metrics=metrics,
        runtime_versions=versions,
        candidate_factor_count=len(candidates),
        selected_factor_count=len(selection.selected_factor_ids),
        removed_factor_count=len(candidates) - len(selection.selected_factor_ids),
        best_iteration=best_iteration,
        skipped_factor_count=len(selection.excluded_factors),
    )
    objects.put_object("lightgbm_run_result", run_id, result)
    return result


def _load_source_bundle(config: LightGBMIntegrationConfig) -> _SourceBundle:
    layout = RunStorageLayout.resolve(
        config.factor_source.storage_root,
        config.factor_source.run_id,
    )
    required_paths = (
        layout.run_root,
        layout.run_root / "artifacts",
        layout.run_root / "objects",
        layout.factor_library_db_path,
    )
    if config.factor_source.include_rejected_factors:
        required_paths += (layout.factor_library_csv_path,)
    if not all(path.exists() for path in required_paths):
        raise LightGBMIntegrationError(f"FACTOR_SOURCE_NOT_FOUND: {layout.run_root}")
    artifacts = LocalArtifactRepository(layout.run_root)
    objects = LocalExperimentRepository(layout.run_root)
    config_ids = objects.list_object_ids("config_snapshot")
    if len(config_ids) != 1:
        raise LightGBMIntegrationError("FACTOR_SOURCE_REQUIRES_ONE_CONFIG_SNAPSHOT")
    source_config_ref = ObjectRef(
        "config_snapshot", config_ids[0], "config-snapshot/1.0"
    )
    source_config = objects.get_object(source_config_ref, ConfigSnapshot)
    if config.factor_source.include_rejected_factors:
        return _load_csv_source_bundle(
            layout,
            source_config,
            source_config_ref,
            artifacts,
            objects,
        )
    library = SqliteFactorLibrary(
        layout.factor_library_db_path,
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    entries = library.list_entries()
    if not entries:
        raise LightGBMIntegrationError("FACTOR_SOURCE_HAS_NO_ACCEPTED_FACTORS")
    if len({entry.data_snapshot_id for entry in entries}) != 1:
        raise LightGBMIntegrationError("FACTOR_SOURCE_MIXES_DATA_SNAPSHOTS")

    specs: list[FactorSpec] = []
    for entry in entries:
        if not all(
            objects.exists(ref)
            for ref in (
                entry.factor_ref,
                entry.evaluation_result_ref,
                entry.acceptance_decision_ref,
            )
        ) or not artifacts.verify(entry.evaluated_factor_values_ref):
            raise LightGBMIntegrationError(
                f"FACTOR_SOURCE_REFERENCE_MISSING: {entry.factor_id}"
            )
        try:
            spec = objects.get_object(entry.factor_ref, FactorSpec)
            evaluation = objects.get_object(
                entry.evaluation_result_ref,
                EvaluationResult,
            )
            decision = objects.get_object(
                entry.acceptance_decision_ref,
                AcceptanceDecision,
            )
        except (CoreError, OSError, ValueError) as error:
            raise LightGBMIntegrationError(
                f"FACTOR_SOURCE_REFERENCE_INVALID: {entry.factor_id}: {error}"
            ) from error
        if (
            spec.factor_id != entry.factor_id
            or spec.expression != entry.expression
            or spec.frequency != entry.frequency
            or spec.frequency != source_config.config.data.frequency
            or evaluation.factor_id != entry.factor_id
            or evaluation.data_snapshot_id != entry.data_snapshot_id
            or evaluation.evaluated_factor_values_ref
            != entry.evaluated_factor_values_ref
            or decision.factor_id != entry.factor_id
            or decision.evaluation_result_ref != entry.evaluation_result_ref
            or not decision.accepted
        ):
            raise LightGBMIntegrationError(
                f"FACTOR_SOURCE_LINEAGE_MISMATCH: {entry.factor_id}"
            )
        specs.append(spec)
    ordered = tuple(
        sorted(zip(entries, specs, strict=True), key=lambda item: item[0].factor_id)
    )
    return _SourceBundle(
        candidates=tuple(
            ModelFactorInput(
                factor_id=entry.factor_id,
                expression=entry.expression,
                factor_ref=entry.factor_ref,
                evaluation_result_ref=entry.evaluation_result_ref,
                acceptance_decision_ref=entry.acceptance_decision_ref,
                source_evaluated_values_ref=entry.evaluated_factor_values_ref,
                source_data_snapshot_id=entry.data_snapshot_id,
                source_experiment_id=entry.source_experiment_id,
                source_status="accepted",
            )
            for entry, _spec in ordered
        ),
        specs=tuple(item[1] for item in ordered),
        source_config=source_config,
        source_config_ref=source_config_ref,
        candidate_source="factor-library.sqlite3",
        included_statuses=("accepted",),
        excluded_statuses=("rejected", "failed"),
        excluded_status_counts=(),
    )


def _load_csv_source_bundle(
    layout: RunStorageLayout,
    source_config: ConfigSnapshot,
    source_config_ref: ObjectRef,
    artifacts: LocalArtifactRepository,
    objects: LocalExperimentRepository,
) -> _SourceBundle:
    """Load accepted and rejected factor rows from the Graph CSV manifest."""
    specs_by_id: dict[str, list[tuple[FactorSpec, ObjectRef]]] = {}
    for object_id in objects.list_object_ids("factor_spec"):
        probe = ObjectRef("factor_spec", object_id, "factor-spec/1.0")
        spec = objects.get_object(probe, FactorSpec)
        ref = ObjectRef("factor_spec", object_id, spec.schema_version)
        specs_by_id.setdefault(spec.factor_id, []).append((spec, ref))

    evaluations: dict[tuple[str, str, str], tuple[EvaluationResult, ObjectRef]] = {}
    for object_id in objects.list_object_ids("evaluation_result"):
        probe = ObjectRef("evaluation_result", object_id, "evaluation-result/1.1")
        evaluation = objects.get_object(probe, EvaluationResult)
        key = (
            evaluation.factor_id,
            evaluation.data_snapshot_id,
            evaluation.evaluated_factor_values_ref.artifact_id,
        )
        if key in evaluations:
            raise LightGBMIntegrationError(
                f"FACTOR_SOURCE_DUPLICATE_EVALUATION: {evaluation.factor_id}"
            )
        evaluations[key] = (
            evaluation,
            ObjectRef("evaluation_result", object_id, evaluation.schema_version),
        )

    decisions: dict[tuple[str, str], tuple[AcceptanceDecision, ObjectRef]] = {}
    for object_id in objects.list_object_ids("acceptance_decision"):
        probe = ObjectRef("acceptance_decision", object_id, "acceptance-decision/1.0")
        decision = objects.get_object(probe, AcceptanceDecision)
        key = (decision.factor_id, decision.evaluation_result_ref.object_id)
        if key in decisions:
            raise LightGBMIntegrationError(
                f"FACTOR_SOURCE_DUPLICATE_DECISION: {decision.factor_id}"
            )
        decisions[key] = (
            decision,
            ObjectRef("acceptance_decision", object_id, decision.schema_version),
        )

    candidates: list[ModelFactorInput] = []
    specs: list[FactorSpec] = []
    excluded: dict[str, int] = {}
    with layout.factor_library_csv_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "status",
            "expression",
            "factor_id",
            "frequency",
            "experiment_id",
            "data_snapshot_id",
            "evaluated_factor_values_artifact_id",
        }
        if not required_columns.issubset(reader.fieldnames or ()):
            missing = sorted(required_columns - set(reader.fieldnames or ()))
            raise LightGBMIntegrationError(
                "FACTOR_SOURCE_CSV_MISSING_COLUMNS: " + ", ".join(missing)
            )
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            status = row["status"]
            if status == "failed":
                excluded[status] = excluded.get(status, 0) + 1
                continue
            if status not in {"accepted", "rejected"}:
                raise LightGBMIntegrationError(
                    f"FACTOR_SOURCE_CSV_UNKNOWN_STATUS: row {row_number}: {status}"
                )
            factor_id = row["factor_id"]
            if factor_id in seen_ids:
                raise LightGBMIntegrationError(
                    f"FACTOR_SOURCE_CSV_DUPLICATE_FACTOR: {factor_id}"
                )
            seen_ids.add(factor_id)
            try:
                spec_matches = [
                    item
                    for item in specs_by_id[factor_id]
                    if item[0].expression == row["expression"]
                    and item[0].frequency == row["frequency"]
                ]
                if not spec_matches:
                    raise KeyError("factor specification fields differ")
                spec, factor_ref = spec_matches[0]
                if spec.frequency != source_config.config.data.frequency:
                    raise KeyError("factor frequency differs from source config")
                evaluation, evaluation_ref = evaluations[
                    (
                        factor_id,
                        row["data_snapshot_id"],
                        row["evaluated_factor_values_artifact_id"],
                    )
                ]
                decision, decision_ref = decisions[
                    (factor_id, evaluation_ref.object_id)
                ]
            except KeyError as error:
                raise LightGBMIntegrationError(
                    f"FACTOR_SOURCE_LINEAGE_MISMATCH: {factor_id}: {error}"
                ) from error
            if (
                evaluation.factor_id != factor_id
                or evaluation.data_snapshot_id != row["data_snapshot_id"]
                or decision.factor_id != factor_id
                or decision.evaluation_result_ref.object_id != evaluation_ref.object_id
                or decision.accepted != (status == "accepted")
                or not artifacts.verify(evaluation.evaluated_factor_values_ref)
            ):
                raise LightGBMIntegrationError(
                    f"FACTOR_SOURCE_LINEAGE_MISMATCH: {factor_id}"
                )
            candidates.append(
                ModelFactorInput(
                    factor_id=factor_id,
                    expression=spec.expression,
                    factor_ref=factor_ref,
                    evaluation_result_ref=evaluation_ref,
                    acceptance_decision_ref=decision_ref,
                    source_evaluated_values_ref=evaluation.evaluated_factor_values_ref,
                    source_data_snapshot_id=evaluation.data_snapshot_id,
                    source_experiment_id=row["experiment_id"],
                    source_status=status,
                )
            )
            specs.append(spec)
    if not candidates:
        raise LightGBMIntegrationError("FACTOR_SOURCE_HAS_NO_USABLE_FACTORS")
    ordered = tuple(
        sorted(zip(candidates, specs, strict=True), key=lambda item: item[0].factor_id)
    )
    return _SourceBundle(
        candidates=tuple(item[0] for item in ordered),
        specs=tuple(item[1] for item in ordered),
        source_config=source_config,
        source_config_ref=source_config_ref,
        candidate_source="factor_library/factors.csv",
        included_statuses=("accepted", "rejected"),
        excluded_statuses=("failed",),
        excluded_status_counts=(("failed", excluded.get("failed", 0)),),
    )


def _validate_source_compatibility(
    config: LightGBMIntegrationConfig,
    source_snapshot: ConfigSnapshot,
) -> None:
    source = source_snapshot.config
    if source.architecture.name != "graph":
        raise LightGBMIntegrationError("FACTOR_SOURCE_MUST_BE_GRAPH_RUN")
    if source.data.end_date > config.segments.valid.end_date:
        raise LightGBMIntegrationError("SOURCE_ENDS_AFTER_VALIDATION")
    if source.data.end_date >= config.segments.test.start_date:
        raise LightGBMIntegrationError("SOURCE_OVERLAPS_TEST")
    if (
        source.data.provider != config.data.provider
        or source.data.market != config.data.market
        or source.data.frequency != config.data.frequency
        or source.data.dataset_version != config.data.dataset_version
        or source.data.price_adjustment != config.data.price_adjustment
        or source.data.trading_calendar_version != config.data.trading_calendar_version
        or source.universe != config.universe
    ):
        raise LightGBMIntegrationError("SOURCE_MODEL_RESEARCH_DOMAIN_MISMATCH")
    for name in ("market", "exposure", "mask"):
        source_provider = getattr(source.qlib, name)
        model_provider = getattr(config.qlib, name)
        if source_provider.fields != model_provider.fields:
            raise LightGBMIntegrationError(
                f"SOURCE_MODEL_DATA_IDENTITY_MISMATCH: {name}"
            )
    if (
        source.qlib.region != config.qlib.region
        or source.qlib.instruments != config.qlib.instruments
    ):
        raise LightGBMIntegrationError("SOURCE_MODEL_DATA_IDENTITY_MISMATCH: universe")


def _required_market_fields(specs: tuple[FactorSpec, ...]) -> tuple[str, ...]:
    required = {"close"}
    for spec in specs:
        required.update(spec.required_fields)
    unsupported = required - set(MARKET_DATA_FIELDS)
    if unsupported:
        raise LightGBMIntegrationError(
            "UNSUPPORTED_MODEL_FIELDS: " + ", ".join(sorted(unsupported))
        )
    return tuple(field for field in MARKET_DATA_FIELDS if field in required)


def _load_model_snapshot(
    config: LightGBMIntegrationConfig,
    specs: tuple[FactorSpec, ...],
    artifacts: LocalArtifactRepository,
) -> DataSnapshot:
    provider = QlibMarketDataProvider(
        artifacts,
        provider_uri=config.qlib.market.provider_uri,
        trading_calendar_version=config.data.trading_calendar_version,
        exposure_source=_qlib_source(config.qlib.exposure),
        mask_source=_qlib_source(config.qlib.mask),
        instruments=config.qlib.instruments or None,
        region=config.qlib.region,
    )
    request = MarketDataRequest(
        market=config.data.market,
        universe=config.universe.name,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        warmup_start_date=config.data.warmup_start_date,
        frequency=config.data.frequency,
        fields=_required_market_fields(specs),
        dataset_version=config.data.dataset_version,
        price_adjustment=config.data.price_adjustment,
        schema_version="market-data/1.0",
    )
    return provider.load_snapshot(request)


def _qlib_source(source: QlibSourceConfig) -> QlibFeatureSource:
    return QlibFeatureSource(
        provider_uri=source.provider_uri,
        fields=tuple(
            QlibFieldMapping(item.field_name, item.expression) for item in source.fields
        ),
    )


def _calculate_model_features(
    config: LightGBMIntegrationConfig,
    candidates: tuple[ModelFactorInput, ...],
    specs: tuple[FactorSpec, ...],
    frame: CalculationFrame,
    *,
    artifact_root: str,
    snapshot: DataSnapshot,
    required_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], tuple[FactorExclusion, ...]]:
    """Calculate model features sequentially or in isolated spawn workers."""
    if config.parallel.factor_workers <= 1 or len(specs) <= 1:
        return _calculate_model_features_sequential(config, candidates, specs, frame)

    worker_init = _FeatureWorkerInit(
        artifact_root=artifact_root,
        snapshot=snapshot,
        required_fields=required_fields,
        industry_field=config.feature_preprocessing.industry_field,
        continuous_exposures=config.feature_preprocessing.continuous_exposures,
        mad_multiplier=config.feature_preprocessing.mad_multiplier,
        min_cross_section_samples=(
            config.feature_preprocessing.min_cross_section_samples
        ),
        neutralize=config.feature_preprocessing.neutralization_enabled,
    )
    context = multiprocessing.get_context("spawn")
    results: dict[str, tuple[np.ndarray | None, str | None]] = {}
    with ProcessPoolExecutor(
        max_workers=min(config.parallel.factor_workers, len(specs)),
        mp_context=context,
        initializer=_initialize_feature_worker,
        initargs=(worker_init,),
    ) as executor:
        futures = {
            executor.submit(_calculate_one_model_feature, spec): spec for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                values, error_message = future.result()
            except Exception as error:  # noqa: BLE001 - isolate one worker failure
                values = None
                error_message = f"{type(error).__name__}: {error}"
            results[spec.factor_id] = (values, error_message)

    candidates_by_id = {candidate.factor_id: candidate for candidate in candidates}
    features: dict[str, np.ndarray] = {}
    exclusions: list[FactorExclusion] = []
    for spec in specs:
        values, error_message = results.get(
            spec.factor_id,
            (None, "feature worker returned no result"),
        )
        if values is not None:
            features[spec.factor_id] = values
            continue
        candidate = candidates_by_id[spec.factor_id]
        exclusions.append(
            FactorExclusion(
                factor_id=spec.factor_id,
                source_status=candidate.source_status,
                stage="feature_calculation",
                error_code="FACTOR_RECALCULATION_FAILED",
                message=error_message or "worker returned no feature values",
            )
        )
    return features, tuple(exclusions)


def _calculate_model_features_sequential(
    config: LightGBMIntegrationConfig,
    candidates: tuple[ModelFactorInput, ...],
    specs: tuple[FactorSpec, ...],
    frame: CalculationFrame,
) -> tuple[dict[str, np.ndarray], tuple[FactorExclusion, ...]]:
    """Calculate model features in the current process."""
    industry = frame.context_fields[config.feature_preprocessing.industry_field]
    continuous = np.column_stack(
        [
            frame.context_fields[field]
            for field in config.feature_preprocessing.continuous_exposures
        ]
    )
    candidates_by_id = {candidate.factor_id: candidate for candidate in candidates}
    features: dict[str, np.ndarray] = {}
    exclusions: list[FactorExclusion] = []
    for spec in specs:
        try:
            raw = evaluate_factor_spec(spec, frame)
            features[spec.factor_id] = preprocess_feature_values(
                raw,
                datetimes=frame.datetimes,
                industry=industry,
                continuous=continuous,
                eligible=frame.eligible_mask,
                mad_multiplier=config.feature_preprocessing.mad_multiplier,
                min_cross_section_samples=(
                    config.feature_preprocessing.min_cross_section_samples
                ),
                neutralize=config.feature_preprocessing.neutralization_enabled,
            )
        except Exception as error:  # noqa: BLE001 - attach the factor identity
            candidate = candidates_by_id[spec.factor_id]
            exclusions.append(
                FactorExclusion(
                    factor_id=spec.factor_id,
                    source_status=candidate.source_status,
                    stage="feature_calculation",
                    error_code="FACTOR_RECALCULATION_FAILED",
                    message=str(error),
                )
            )
    return features, tuple(exclusions)


def _initialize_feature_worker(worker_init: _FeatureWorkerInit) -> None:
    """Load one read-only calculation frame inside a spawned worker."""
    global _FEATURE_WORKER_STATE
    artifacts = LocalArtifactRepository(worker_init.artifact_root)
    frame = FactorCalculator(artifacts).load_frame(
        worker_init.snapshot,
        worker_init.required_fields,
    )
    _FEATURE_WORKER_STATE = _FeatureWorkerState(
        frame=frame,
        industry=frame.context_fields[worker_init.industry_field],
        continuous=np.column_stack(
            [frame.context_fields[field] for field in worker_init.continuous_exposures]
        ),
        mad_multiplier=worker_init.mad_multiplier,
        min_cross_section_samples=worker_init.min_cross_section_samples,
        neutralize=worker_init.neutralize,
    )


def _calculate_one_model_feature(
    spec: FactorSpec,
) -> tuple[np.ndarray | None, str | None]:
    """Calculate one feature using the worker-local calculation frame."""
    state = _FEATURE_WORKER_STATE
    if state is None:
        return None, "feature worker was not initialized"
    try:
        raw = evaluate_factor_spec(spec, state.frame)
        values = preprocess_feature_values(
            raw,
            datetimes=state.frame.datetimes,
            industry=state.industry,
            continuous=state.continuous,
            eligible=state.frame.eligible_mask,
            mad_multiplier=state.mad_multiplier,
            min_cross_section_samples=state.min_cross_section_samples,
            neutralize=state.neutralize,
        )
    except Exception as error:  # noqa: BLE001 - return factor-local failure
        return None, str(error)
    return values, None


def _aligned_artifact_values(
    artifacts: LocalArtifactRepository,
    ref: ArtifactRef,
    datetimes: tuple[str, ...],
    instruments: tuple[str, ...],
    field: str,
) -> np.ndarray:
    records = decode_record_artifact(artifacts.get_bytes(ref))
    expected_keys = tuple(zip(datetimes, instruments, strict=True))
    actual_keys = tuple(
        (record.get("datetime"), record.get("instrument")) for record in records
    )
    if actual_keys != expected_keys:
        raise LightGBMIntegrationError("MODEL_ARTIFACT_INDEX_MISMATCH")
    values = np.full(len(records), np.nan, dtype=float)
    for index, record in enumerate(records):
        value = record.get(field)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise LightGBMIntegrationError(f"INVALID_MODEL_ARTIFACT_VALUE: {field}")
        values[index] = float(value)
    return values


def _mean_daily_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    datetimes: tuple[str, ...],
    eligible: np.ndarray,
    segment: DateSegmentConfig,
    rank: bool,
    min_cross_section_samples: int,
    min_daily_effective_stock_ratio: float,
    min_effective_date_ratio: float,
    label: str,
) -> float:
    if left.shape != right.shape or left.shape != (len(datetimes),):
        raise LightGBMIntegrationError(f"{label}: value lengths differ")
    groups: dict[str, list[int]] = {}
    for index, date_value in enumerate(datetimes):
        if segment.start_date <= date_value <= segment.end_date:
            groups.setdefault(date_value, []).append(index)
    if not groups:
        raise LightGBMIntegrationError(f"{label}: segment has no observations")

    daily: list[float] = []
    for positions in groups.values():
        indices = np.asarray(positions, dtype=np.int64)
        eligible_count = int(np.count_nonzero(eligible[indices]))
        keep = (
            eligible[indices] & np.isfinite(left[indices]) & np.isfinite(right[indices])
        )
        common_count = int(np.count_nonzero(keep))
        if (
            eligible_count <= 0
            or common_count < min_cross_section_samples
            or common_count / eligible_count < min_daily_effective_stock_ratio
        ):
            continue
        left_values = left[indices][keep]
        right_values = right[indices][keep]
        if rank:
            left_values = _average_ranks(left_values)
            right_values = _average_ranks(right_values)
        correlation = safe_pearson(left_values, right_values)
        if math.isfinite(correlation):
            daily.append(correlation)
    required_dates = max(1, math.ceil(len(groups) * min_effective_date_ratio))
    if len(daily) < required_dates:
        raise LightGBMIntegrationError(
            f"{label}: {len(daily)} valid dates, requires {required_dates}"
        )
    return float(np.mean(np.asarray(daily, dtype=np.float64)))


def _factor_exclusion_from_error(
    candidate: ModelFactorInput,
    stage: str,
    error: LightGBMIntegrationError,
) -> FactorExclusion:
    """Convert a factor-local selection error into an audit record."""
    message = str(error)
    match = re.search(r"(\d+) valid dates, requires (\d+)", message)
    valid_date_count = int(match.group(1)) if match else None
    required_date_count = int(match.group(2)) if match else None
    error_code = "INSUFFICIENT_EFFECTIVE_DATES" if match else "FACTOR_SELECTION_ERROR"
    return FactorExclusion(
        factor_id=candidate.factor_id,
        source_status=candidate.source_status,
        stage=stage,
        error_code=error_code,
        message=message,
        valid_date_count=valid_date_count,
        required_date_count=required_date_count,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _build_qlib_dataset(
    config: LightGBMIntegrationConfig,
    factor_ids: tuple[str, ...],
    features: Mapping[str, np.ndarray],
    raw_returns: np.ndarray,
    datetimes: tuple[str, ...],
    instruments: tuple[str, ...],
) -> tuple[object, object]:
    import pandas as pd
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import CSRankNorm

    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(datetimes), instruments],
        names=("datetime", "instrument"),
    )
    feature_frame = pd.DataFrame(
        np.column_stack([features[factor_id] for factor_id in factor_ids]),
        index=index,
        columns=factor_ids,
    )
    raw_label = pd.Series(raw_returns, index=index, name="LABEL0")
    dataset_frame = pd.concat(
        {"feature": feature_frame, "label": raw_label.to_frame()}, axis=1
    )
    dataset_frame = CSRankNorm(fields_group="label")(dataset_frame)
    date_values = np.asarray(datetimes, dtype=str)
    train_or_valid = _segment_mask(date_values, config.segments.train) | _segment_mask(
        date_values, config.segments.valid
    )
    test = _segment_mask(date_values, config.segments.test)
    any_feature = np.any(np.isfinite(feature_frame.to_numpy(dtype=float)), axis=1)
    normalized_label = dataset_frame[("label", "LABEL0")].to_numpy(dtype=float)
    keep = (train_or_valid & any_feature & np.isfinite(normalized_label)) | (
        test & any_feature
    )
    dataset_frame = dataset_frame.loc[keep]
    raw_label = raw_label.loc[keep]
    if dataset_frame.empty:
        raise LightGBMIntegrationError("MODEL_DATASET_EMPTY")
    dataset = DatasetH(
        handler=DataHandlerLP.from_df(dataset_frame),
        segments={
            "train": (
                config.segments.train.start_date,
                config.segments.train.end_date,
            ),
            "valid": (
                config.segments.valid.start_date,
                config.segments.valid.end_date,
            ),
            "test": (
                config.segments.test.start_date,
                config.segments.test.end_date,
            ),
        },
    )
    for segment in ("train", "valid", "test"):
        if dataset.prepare(segment).empty:
            raise LightGBMIntegrationError(f"MODEL_SEGMENT_EMPTY: {segment}")
    return dataset, raw_label


def _segment_mask(values: np.ndarray, segment: DateSegmentConfig) -> np.ndarray:
    return (values >= segment.start_date) & (values <= segment.end_date)


def _fit_qlib_model(
    config: QlibLightGBMConfig,
    dataset: object,
) -> tuple[object, dict[str, dict[str, list[float]]], ModelRuntimeVersions]:
    from unittest.mock import patch

    # NumPy/BLAS remain single-threaded; LightGBM is the only native threaded
    # stage and receives its explicit OpenMP budget through num_threads.
    os.environ["OMP_NUM_THREADS"] = str(config.num_threads)
    import lightgbm
    import pandas
    import qlib
    import qlib.contrib.model.gbdt as gbdt

    model = gbdt.LGBModel(
        loss=config.loss,
        colsample_bytree=config.colsample_bytree,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        lambda_l1=config.lambda_l1,
        lambda_l2=config.lambda_l2,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        num_threads=config.num_threads,
        early_stopping_rounds=config.early_stopping_rounds,
        num_boost_round=config.num_boost_round,
    )
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    # Qlib 0.9.7 logs every epoch through its Recorder even when no recorder is
    # requested. Keep this standalone workflow recorder-free and use evals_result.
    recorder_stub = SimpleNamespace(log_metrics=lambda **_metrics: None)
    with patch.object(gbdt, "R", recorder_stub):
        model.fit(dataset, evals_result=evaluation_history)
    normalized_history = cast(
        dict[str, dict[str, list[float]]],
        _normalize_json_scalars(evaluation_history),
    )
    return (
        model,
        normalized_history,
        ModelRuntimeVersions(
            python=platform.python_version(),
            qlib=str(qlib.__version__),
            lightgbm=str(lightgbm.__version__),
            pandas=str(pandas.__version__),
        ),
    )


def _normalize_json_scalars(value: object) -> object:
    """Convert Qlib and NumPy scalar containers to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_scalars(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_json_scalars(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_predictions(
    predictions: object,
    raw_labels: object,
    *,
    min_cross_section_samples: int,
) -> tuple[
    tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], SignalMetrics
]:
    aligned_labels = raw_labels.reindex(predictions.index)
    prediction_records: list[Mapping[str, object]] = []
    by_date: dict[str, list[tuple[float, float]]] = {}
    for key, score in predictions.items():
        datetime_value = key[predictions.index.names.index("datetime")]
        instrument = key[predictions.index.names.index("instrument")]
        date_value = datetime_value.strftime("%Y-%m-%d")
        raw_label = aligned_labels.loc[key]
        label_value = (
            float(raw_label)
            if type(raw_label) in (int, float, np.float32, np.float64)
            and math.isfinite(float(raw_label))
            else None
        )
        numeric_score = float(score)
        score_value = numeric_score if math.isfinite(numeric_score) else None
        prediction_records.append(
            {
                "datetime": date_value,
                "instrument": str(instrument),
                "score": score_value,
                "raw_forward_return": label_value,
            }
        )
        if label_value is not None and score_value is not None:
            by_date.setdefault(date_value, []).append((score_value, label_value))

    daily_records: list[Mapping[str, object]] = []
    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    effective_sample_count = 0
    for date_value, pairs in sorted(by_date.items()):
        if len(pairs) < min_cross_section_samples:
            continue
        values = np.asarray(pairs, dtype=np.float64)
        ic = safe_pearson(values[:, 0], values[:, 1])
        rank_ic = safe_pearson(
            _average_ranks(values[:, 0]), _average_ranks(values[:, 1])
        )
        if not math.isfinite(ic) or not math.isfinite(rank_ic):
            continue
        daily_ic.append(ic)
        daily_rank_ic.append(rank_ic)
        effective_sample_count += len(pairs)
        daily_records.append(
            {
                "datetime": date_value,
                "ic": ic,
                "rank_ic": rank_ic,
                "sample_count": len(pairs),
            }
        )
    if not daily_ic:
        raise LightGBMIntegrationError("NO_VALID_TEST_SIGNAL_DATES")
    ic_values = np.asarray(daily_ic, dtype=float)
    rank_ic_values = np.asarray(daily_rank_ic, dtype=float)
    metrics = SignalMetrics(
        ic_mean=float(np.mean(ic_values)),
        ic_std=_sample_std(ic_values),
        ic_ir=_information_ratio(ic_values),
        rank_ic_mean=float(np.mean(rank_ic_values)),
        rank_ic_std=_sample_std(rank_ic_values),
        rank_ic_ir=_information_ratio(rank_ic_values),
        effective_sample_count=effective_sample_count,
        valid_date_count=len(daily_ic),
    )
    return tuple(prediction_records), tuple(daily_records), metrics


def _concat_prediction_segments(validation: object, test: object) -> object:
    """Join validation and test predictions for the first test trading day."""
    import pandas as pd

    combined = pd.concat((validation, test))
    if combined.index.has_duplicates:
        raise LightGBMIntegrationError("DUPLICATE_BACKTEST_SIGNAL_INDEX")
    return combined.sort_index()


def _run_qlib_backtest(
    config: LightGBMIntegrationConfig,
    signal: object,
) -> tuple[
    PortfolioMetrics,
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    """Run the fixed Qlib TopkDropout backtest and serialize its daily outputs."""
    import pandas as pd

    try:
        import qlib
        from qlib.backtest import backtest
        from qlib.contrib.evaluate import risk_analysis
        from qlib.data import D

        qlib.init(
            provider_uri=os.path.expanduser(config.qlib.market.provider_uri),
            region=config.qlib.region,
            dataset_cache=None,
        )
        test = config.segments.test
        instruments = D.instruments(config.universe.name)
        codes = D.list_instruments(
            instruments,
            start_time=test.start_date,
            end_time=test.end_date,
            as_list=True,
        )
        if not codes:
            raise LightGBMIntegrationError("PORTFOLIO_BACKTEST_EMPTY_UNIVERSE")
        settings = config.backtest
        # Qlib 0.9.7 turns ``benchmark=None`` into its SH000300 default inside
        # Account.  A zero series avoids that provider lookup; it is retained
        # only for Qlib's report schema and is never used in our metrics.
        zero_benchmark = pd.Series(
            0.0,
            index=pd.to_datetime([test.start_date, test.end_date]),
            name="absolute_return_baseline",
        )
        portfolio_metric_dict, _indicator_dict = backtest(
            start_time=test.start_date,
            end_time=test.end_date,
            strategy={
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "signal": signal,
                    "topk": settings.topk,
                    "n_drop": settings.n_drop,
                    "method_sell": settings.method_sell,
                    "method_buy": settings.method_buy,
                    "hold_thresh": settings.hold_thresh,
                    "only_tradable": settings.only_tradable,
                    "forbid_all_trade_at_limit": settings.forbid_all_trade_at_limit,
                    "risk_degree": settings.risk_degree,
                },
            },
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    "verbose": False,
                    "track_data": False,
                    "trade_type": "serial",
                    "indicator_config": {"show_indicator": False},
                },
            },
            benchmark=zero_benchmark,
            account=settings.account,
            exchange_kwargs={
                "codes": codes,
                "limit_threshold": settings.limit_threshold,
                "deal_price": settings.deal_price,
                "open_cost": settings.open_cost,
                "close_cost": settings.close_cost,
                "min_cost": settings.min_cost,
                "impact_cost": settings.impact_cost,
                "trade_unit": settings.trade_unit,
            },
        )
    except LightGBMIntegrationError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize Qlib boundary
        raise LightGBMIntegrationError(f"PORTFOLIO_BACKTEST_FAILED: {error}") from error

    if not portfolio_metric_dict or "1day" not in portfolio_metric_dict:
        raise LightGBMIntegrationError("PORTFOLIO_BACKTEST_NO_DAILY_REPORT")
    report, positions = portfolio_metric_dict["1day"]
    if not isinstance(report, pd.DataFrame) or "return" not in report:
        raise LightGBMIntegrationError("PORTFOLIO_BACKTEST_INVALID_REPORT")

    portfolio_return = report["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cost = (
        report["cost"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "cost" in report
        else pd.Series(0.0, index=report.index)
    )
    net_return = (portfolio_return - cost).dropna()
    if net_return.empty:
        raise LightGBMIntegrationError("PORTFOLIO_BACKTEST_NO_VALID_ABSOLUTE_RETURNS")
    analysis = risk_analysis(net_return, freq="day", mode="sum")
    risk = analysis["risk"] if isinstance(analysis, pd.DataFrame) else analysis
    annualized_return = float(risk["annualized_return"])
    information_ratio = float(risk["information_ratio"])
    max_drawdown = float(risk["max_drawdown"])
    if not all(
        math.isfinite(value)
        for value in (annualized_return, information_ratio, max_drawdown)
    ):
        raise LightGBMIntegrationError("PORTFOLIO_BACKTEST_NONFINITE_METRICS")
    calmar_ratio = (
        annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else 0.0
    )
    return (
        PortfolioMetrics(
            annualized_return=annualized_return,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            return_mode=settings.return_mode,
        ),
        _dataframe_records(report),
        _position_records(positions),
    )


def _dataframe_records(frame: object) -> tuple[Mapping[str, object], ...]:
    """Convert a Qlib report frame to stable scalar records."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        return ()
    rows = frame.reset_index().to_dict(orient="records")
    return tuple(
        {str(key): _scalar_value(value) for key, value in row.items()} for row in rows
    )


def _position_records(positions: object) -> tuple[Mapping[str, object], ...]:
    """Flatten Qlib historical Position objects into audit-friendly rows."""
    records: list[Mapping[str, object]] = []
    for datetime_value, position in getattr(positions, "items", lambda: ())():
        values = getattr(position, "position", position)
        if not isinstance(values, Mapping):
            continue
        for instrument, details in values.items():
            if isinstance(details, Mapping):
                row = {
                    "datetime": _scalar_value(datetime_value),
                    "instrument": str(instrument),
                    **{
                        str(key): _scalar_value(value) for key, value in details.items()
                    },
                }
            else:
                row = {
                    "datetime": _scalar_value(datetime_value),
                    "instrument": str(instrument),
                    "value": _scalar_value(details),
                }
            records.append(row)
    return tuple(records)


def _scalar_value(value: object) -> object:
    """Convert pandas and NumPy scalars into columnar-compatible values."""
    import pandas as pd

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _scalar_value(value.item())
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return str(value)


def _sample_std(values: np.ndarray) -> float:
    return 0.0 if values.size < 2 else float(np.std(values, ddof=1))


def _information_ratio(values: np.ndarray) -> float:
    std = _sample_std(values)
    if not math.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(np.mean(values) / std)
