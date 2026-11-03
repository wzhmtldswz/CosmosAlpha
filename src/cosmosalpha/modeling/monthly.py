"""Monthly month-end sampling, training, and backtesting for LightGBM."""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import cast

import numpy as np

from cosmosalpha.agents import FactorSpec
from cosmosalpha.core import (
    COLUMNAR_RECORDS_MEDIA_TYPE,
    ArtifactRef,
    ObjectRef,
    encode_columnar_records,
    stable_sha256,
    to_json,
)
from cosmosalpha.data import DataSnapshot
from cosmosalpha.evaluation import preprocess_feature_values
from cosmosalpha.factors import CalculationFrame, FactorCalculator, evaluate_factor_spec
from cosmosalpha.modeling.config import (
    LightGBMIntegrationConfig,
    LightGBMMonthlyIntegrationConfig,
    create_lightgbm_monthly_config_snapshot,
)
from cosmosalpha.modeling.lightgbm import (
    FactorExclusion,
    LightGBMIntegrationError,
    LightGBMPreflightIssue,
    LightGBMPreflightReport,
    ModelFactorInput,
    ModelRuntimeVersions,
    PortfolioMetrics,
    SignalMetrics,
    _build_qlib_dataset,
    _concat_prediction_segments,
    _evaluate_predictions,
    _fit_qlib_model,
    _load_model_snapshot,
    _load_source_bundle,
    _required_market_fields,
    _run_qlib_backtest,
    _validate_source_compatibility,
    select_factors_greedily,
)
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    RunStorageLayout,
)

LIGHTGBM_MONTHLY_RUN_SCHEMA_VERSION = "lightgbm-monthly-run-result/1.0"
MONTHLY_SAMPLING_ARTIFACT_KIND = "lightgbm_monthly_sampling_calendar"
MONTHLY_LABEL_ARTIFACT_KIND = "lightgbm_monthly_labels"


@dataclass(frozen=True, slots=True)
class LightGBMMonthlyRunResult:
    """Immutable references and metrics for one monthly model run."""

    run_id: str
    source_run_id: str
    source_config_snapshot_ref: ObjectRef
    config_snapshot_ref: ObjectRef
    data_snapshot_ref: ObjectRef
    factor_selection_ref: ObjectRef
    sampling_calendar_ref: ArtifactRef
    labels_ref: ArtifactRef
    model_ref: ArtifactRef
    training_curve_ref: ArtifactRef
    predictions_ref: ArtifactRef
    signal_metrics_ref: ArtifactRef
    portfolio_report_ref: ArtifactRef
    portfolio_positions_ref: ArtifactRef
    portfolio_metrics: PortfolioMetrics
    metrics: SignalMetrics
    runtime_versions: ModelRuntimeVersions
    candidate_factor_count: int
    selected_factor_count: int
    removed_factor_count: int
    best_iteration: int
    skipped_factor_count: int
    schema_version: str = LIGHTGBM_MONTHLY_RUN_SCHEMA_VERSION

    def summary_json(self) -> str:
        """Return the structured monthly run summary."""
        return to_json(
            {
                "run_id": self.run_id,
                "source_run_id": self.source_run_id,
                "candidate_factor_count": self.candidate_factor_count,
                "selected_factor_count": self.selected_factor_count,
                "removed_factor_count": self.removed_factor_count,
                "skipped_factor_count": self.skipped_factor_count,
                "best_iteration": self.best_iteration,
                "metrics": self.metrics,
                "portfolio_metrics": self.portfolio_metrics,
                "runtime_versions": self.runtime_versions,
                "display_metrics": {
                    "IC": self.metrics.ic_mean,
                    "ICIR": self.metrics.ic_ir,
                    "RankIC": self.metrics.rank_ic_mean,
                    "RankICIR": self.metrics.rank_ic_ir,
                    "ARR_percent": self.portfolio_metrics.annualized_return * 100.0,
                    "IR": self.portfolio_metrics.information_ratio,
                    "MDD_percent": self.portfolio_metrics.max_drawdown * 100.0,
                    "CR": self.portfolio_metrics.calmar_ratio,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class _RawFeatureWorkerInit:
    """Pickle-safe inputs for one raw factor worker."""

    artifact_root: str
    snapshot: DataSnapshot
    required_fields: tuple[str, ...]


_RAW_FEATURE_FRAME: CalculationFrame | None = None


def preflight_lightgbm_monthly_integration(
    config: LightGBMMonthlyIntegrationConfig,
    run_id: str,
) -> LightGBMPreflightReport:
    """Validate monthly sources and output isolation without loading market data."""
    issues: list[LightGBMPreflightIssue] = []
    candidate_count = 0
    excluded_status_counts: tuple[tuple[str, int], ...] = ()
    try:
        layout = RunStorageLayout.resolve(config.output.root, run_id)
        if layout.run_root.exists():
            issues.append(
                LightGBMPreflightIssue(
                    "OUTPUT_RUN_EXISTS",
                    f"output run already exists: {layout.run_root}",
                )
            )
        source = _load_source_bundle(cast(LightGBMIntegrationConfig, config))
        candidate_count = len(source.candidates)
        excluded_status_counts = source.excluded_status_counts
        _validate_source_compatibility(
            cast(LightGBMIntegrationConfig, config), source.source_config
        )
    except Exception as error:  # noqa: BLE001 - preflight is a structured boundary
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


def run_lightgbm_monthly_integration(
    config: LightGBMMonthlyIntegrationConfig,
    run_id: str,
) -> LightGBMMonthlyRunResult:
    """Recompute factors, sample month ends, train, and backtest monthly signals."""
    report = preflight_lightgbm_monthly_integration(config, run_id)
    if not report.ready:
        details = "; ".join(f"{item.code}: {item.message}" for item in report.issues)
        raise LightGBMIntegrationError(details)
    try:
        return _run_lightgbm_monthly_integration(config, run_id)
    except LightGBMIntegrationError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize external model boundary
        raise LightGBMIntegrationError(str(error)) from error


def _run_lightgbm_monthly_integration(
    config: LightGBMMonthlyIntegrationConfig,
    run_id: str,
) -> LightGBMMonthlyRunResult:
    source = _load_source_bundle(cast(LightGBMIntegrationConfig, config))
    _validate_source_compatibility(
        cast(LightGBMIntegrationConfig, config), source.source_config
    )
    layout = RunStorageLayout.resolve(config.output.root, run_id)
    artifacts = LocalArtifactRepository(layout.run_root)
    objects = LocalExperimentRepository(layout.run_root)

    config_snapshot = create_lightgbm_monthly_config_snapshot(config)
    config_snapshot_ref = objects.put_object(
        "lightgbm_monthly_config_snapshot",
        config_snapshot.snapshot_id,
        config_snapshot,
    )
    snapshot = _load_model_snapshot(
        cast(LightGBMIntegrationConfig, config), source.specs, artifacts
    )
    data_snapshot_ref = objects.put_object(
        "data_snapshot", snapshot.snapshot_id, snapshot
    )
    required_fields = _required_market_fields(source.specs)
    frame = FactorCalculator(artifacts).load_frame(snapshot, required_fields)
    sampled_rows, sampled_dates = select_month_end_rows(
        frame.datetimes,
        start_date=config.segments.train.start_date,
        end_date=config.data.end_date,
        horizon=config.sampling.label_horizon_trading_days,
    )
    if not sampled_rows:
        raise LightGBMIntegrationError("MONTHLY_SAMPLING_EMPTY")
    sample_positions = np.asarray(sampled_rows, dtype=np.int64)
    raw_returns = forward_close_returns(
        frame,
        sample_positions,
        horizon=config.sampling.label_horizon_trading_days,
    )
    sampled_datetimes = tuple(frame.datetimes[index] for index in sampled_rows)
    sampled_instruments = tuple(frame.instruments[index] for index in sampled_rows)
    features, feature_exclusions = _calculate_monthly_features(
        config,
        source.candidates,
        source.specs,
        frame,
        sample_positions,
        artifact_root=str(layout.run_root),
        snapshot=snapshot,
        required_fields=required_fields,
    )
    selection = select_factors_greedily(
        source.candidates,
        features,
        raw_returns,
        datetimes=sampled_datetimes,
        eligible=frame.eligible_mask[sample_positions],
        train_segment=config.segments.train,
        enabled=False,
        threshold=config.correlation_filter.threshold,
        min_cross_section_samples=config.feature_preprocessing.min_cross_section_samples,
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
        "lightgbm_monthly_factor_selection",
        stable_sha256(selection),
        selection,
    )
    dataset, raw_label_series = _build_qlib_dataset(
        cast(LightGBMIntegrationConfig, config),
        selection.selected_factor_ids,
        features,
        raw_returns,
        sampled_datetimes,
        sampled_instruments,
    )
    model, evaluation_history, versions = _fit_qlib_model(config.model, dataset)
    validation_predictions = model.predict(dataset, segment="valid")
    predictions = model.predict(dataset, segment="test")
    prediction_records, signal_records, metrics = _evaluate_predictions(
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
        cast(LightGBMIntegrationConfig, config), backtest_signal
    )

    booster = getattr(model, "model", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        raise LightGBMIntegrationError("QLIB_LIGHTGBM_MODEL_NOT_AVAILABLE")
    model_ref = artifacts.put_bytes(
        booster.model_to_string().encode("utf-8"),
        kind="lightgbm_monthly_booster",
        media_type="text/plain",
    )
    training_curve_ref = artifacts.put_bytes(
        to_json(evaluation_history).encode("utf-8"),
        kind="lightgbm_monthly_training_curve",
        media_type="application/json",
    )
    predictions_ref = artifacts.put_bytes(
        encode_columnar_records(prediction_records),
        kind="lightgbm_monthly_predictions",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    signal_metrics_ref = artifacts.put_bytes(
        encode_columnar_records(signal_records),
        kind="lightgbm_monthly_signal_metrics",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    portfolio_report_ref = artifacts.put_bytes(
        encode_columnar_records(portfolio_records),
        kind="qlib_monthly_portfolio_report",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    portfolio_positions_ref = artifacts.put_bytes(
        encode_columnar_records(position_records),
        kind="qlib_monthly_portfolio_positions",
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    sampling_calendar_ref = artifacts.put_bytes(
        encode_columnar_records(
            tuple(
                {
                    "month_end": date_value,
                    "horizon_trading_days": config.sampling.label_horizon_trading_days,
                }
                for date_value in sampled_dates
            )
        ),
        kind=MONTHLY_SAMPLING_ARTIFACT_KIND,
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    labels_ref = artifacts.put_bytes(
        encode_columnar_records(
            tuple(
                {
                    "datetime": datetime_value,
                    "instrument": instrument,
                    "raw_forward_return_20d": (
                        float(raw_returns[index])
                        if math.isfinite(float(raw_returns[index]))
                        else None
                    ),
                }
                for index, (datetime_value, instrument) in enumerate(
                    zip(sampled_datetimes, sampled_instruments, strict=True)
                )
            )
        ),
        kind=MONTHLY_LABEL_ARTIFACT_KIND,
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )
    best_iteration = int(getattr(booster, "best_iteration", 0) or 0)
    if best_iteration <= 0:
        best_iteration = config.model.num_boost_round
    result = LightGBMMonthlyRunResult(
        run_id=run_id,
        source_run_id=config.factor_source.run_id,
        source_config_snapshot_ref=source.source_config_ref,
        config_snapshot_ref=config_snapshot_ref,
        data_snapshot_ref=data_snapshot_ref,
        factor_selection_ref=selection_ref,
        sampling_calendar_ref=sampling_calendar_ref,
        labels_ref=labels_ref,
        model_ref=model_ref,
        training_curve_ref=training_curve_ref,
        predictions_ref=predictions_ref,
        signal_metrics_ref=signal_metrics_ref,
        portfolio_report_ref=portfolio_report_ref,
        portfolio_positions_ref=portfolio_positions_ref,
        portfolio_metrics=portfolio_metrics,
        metrics=metrics,
        runtime_versions=versions,
        candidate_factor_count=len(source.candidates),
        selected_factor_count=len(selection.selected_factor_ids),
        removed_factor_count=len(source.candidates)
        - len(selection.selected_factor_ids),
        best_iteration=best_iteration,
        skipped_factor_count=len(selection.excluded_factors),
    )
    objects.put_object("lightgbm_monthly_run_result", run_id, result)
    return result


def select_month_end_rows(
    datetimes: tuple[str, ...],
    *,
    start_date: str,
    end_date: str,
    horizon: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return rows on the last available trading day of each eligible month."""
    date_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, date_value in enumerate(datetimes):
        if start_date <= date_value <= end_date:
            date_to_rows[date_value].append(index)
    calendar = sorted(date_to_rows)
    last_by_month: dict[str, str] = {}
    for date_value in calendar:
        last_by_month[date_value[:7]] = date_value
    eligible_dates = tuple(
        date_value
        for position, date_value in enumerate(calendar)
        if last_by_month[date_value[:7]] == date_value
        and position + horizon < len(calendar)
        and calendar[position + horizon] <= end_date
    )
    rows = tuple(
        index for date_value in eligible_dates for index in date_to_rows[date_value]
    )
    return rows, eligible_dates


def forward_close_returns(
    frame: CalculationFrame,
    sampled_rows: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Calculate uncompressed close-to-close returns ``horizon`` rows forward."""
    close = frame.fields.get("close")
    if close is None:
        raise LightGBMIntegrationError("MONTHLY_LABEL_CLOSE_FIELD_MISSING")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, instrument in enumerate(frame.instruments):
        groups[instrument].append(index)
    positions = {
        index: offset for group in groups.values() for offset, index in enumerate(group)
    }
    values = np.full(sampled_rows.shape, np.nan, dtype=float)
    for output_index, row_index in enumerate(sampled_rows):
        group = groups[frame.instruments[int(row_index)]]
        offset = positions[int(row_index)]
        future_offset = offset + horizon
        if future_offset >= len(group):
            continue
        current = close[int(row_index)]
        future = close[group[future_offset]]
        if np.isfinite(current) and np.isfinite(future) and current > 0.0:
            values[output_index] = future / current - 1.0
    return values


def _calculate_monthly_features(
    config: LightGBMMonthlyIntegrationConfig,
    candidates: tuple[ModelFactorInput, ...],
    specs: tuple[FactorSpec, ...],
    frame: CalculationFrame,
    sampled_rows: np.ndarray,
    *,
    artifact_root: str,
    snapshot: DataSnapshot,
    required_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], tuple[FactorExclusion, ...]]:
    """Calculate raw daily factors in spawned workers, then preprocess month ends."""
    results: dict[str, tuple[np.ndarray | None, str | None]] = {}
    worker_count = min(config.parallel.factor_workers, len(specs))
    if worker_count <= 1 or len(specs) <= 1:
        for spec in specs:
            try:
                results[spec.factor_id] = (evaluate_factor_spec(spec, frame), None)
            except Exception as error:  # noqa: BLE001 - factor-local exclusion
                results[spec.factor_id] = (None, str(error))
    else:
        worker_init = _RawFeatureWorkerInit(
            artifact_root=artifact_root,
            snapshot=snapshot,
            required_fields=required_fields,
        )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_raw_feature_worker,
            initargs=(worker_init,),
        ) as executor:
            futures = {
                executor.submit(_calculate_one_raw_feature, spec): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results[spec.factor_id] = future.result()
                except Exception as error:  # noqa: BLE001 - isolate worker failure
                    results[spec.factor_id] = (None, f"{type(error).__name__}: {error}")

    industry = frame.context_fields[config.feature_preprocessing.industry_field][
        sampled_rows
    ]
    continuous = np.column_stack(
        [
            frame.context_fields[field][sampled_rows]
            for field in config.feature_preprocessing.continuous_exposures
        ]
    )
    datetimes = tuple(frame.datetimes[index] for index in sampled_rows)
    eligible = frame.eligible_mask[sampled_rows]
    candidates_by_id = {candidate.factor_id: candidate for candidate in candidates}
    features: dict[str, np.ndarray] = {}
    exclusions: list[FactorExclusion] = []
    for spec in specs:
        raw, error_message = results.get(
            spec.factor_id,
            (None, "feature worker returned no result"),
        )
        if raw is None:
            candidate = candidates_by_id[spec.factor_id]
            exclusions.append(
                FactorExclusion(
                    factor_id=spec.factor_id,
                    source_status=candidate.source_status,
                    stage="feature_calculation",
                    error_code="FACTOR_RECALCULATION_FAILED",
                    message=error_message or "raw feature was not produced",
                )
            )
            continue
        try:
            features[spec.factor_id] = preprocess_feature_values(
                raw[sampled_rows],
                datetimes=datetimes,
                industry=industry,
                continuous=continuous,
                eligible=eligible,
                mad_multiplier=config.feature_preprocessing.mad_multiplier,
                min_cross_section_samples=(
                    config.feature_preprocessing.min_cross_section_samples
                ),
                neutralize=config.feature_preprocessing.neutralization_enabled,
            )
        except Exception as error:  # noqa: BLE001 - factor-local exclusion
            candidate = candidates_by_id[spec.factor_id]
            exclusions.append(
                FactorExclusion(
                    factor_id=spec.factor_id,
                    source_status=candidate.source_status,
                    stage="monthly_preprocessing",
                    error_code="MONTHLY_PREPROCESSING_FAILED",
                    message=str(error),
                )
            )
    return features, tuple(exclusions)


def _initialize_raw_feature_worker(worker_init: _RawFeatureWorkerInit) -> None:
    """Load the shared daily frame inside a spawned worker."""
    global _RAW_FEATURE_FRAME
    artifacts = LocalArtifactRepository(worker_init.artifact_root)
    _RAW_FEATURE_FRAME = FactorCalculator(artifacts).load_frame(
        worker_init.snapshot,
        worker_init.required_fields,
    )


def _calculate_one_raw_feature(
    spec: FactorSpec,
) -> tuple[np.ndarray | None, str | None]:
    """Calculate one raw factor in a worker-local frame."""
    if _RAW_FEATURE_FRAME is None:
        return None, "feature worker was not initialized"
    try:
        return evaluate_factor_spec(spec, _RAW_FEATURE_FRAME), None
    except Exception as error:  # noqa: BLE001 - return factor-local failure
        return None, str(error)
