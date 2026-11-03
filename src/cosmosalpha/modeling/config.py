"""Strict configuration for post-mining LightGBM integration."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from cosmosalpha.core import (
    CoreValidationError,
    DataConfig,
    JsonValue,
    QlibConfig,
    SerializationError,
    UniverseConfig,
    from_json_value,
    stable_sha256,
    to_json_value,
)

LIGHTGBM_CONFIG_SCHEMA_VERSION = "lightgbm-integration-config/1.2"
LIGHTGBM_CONFIG_SNAPSHOT_SCHEMA_VERSION = "lightgbm-config-snapshot/1.0"
LIGHTGBM_MONTHLY_CONFIG_SCHEMA_VERSION = "lightgbm-monthly-integration-config/1.2"
LIGHTGBM_MONTHLY_CONFIG_SNAPSHOT_SCHEMA_VERSION = "lightgbm-monthly-config-snapshot/1.0"
MODEL_FEATURE_PREPROCESSING_PROTOCOL = "model_feature_mad_zscore_v1"
MODEL_FEATURE_NEUTRALIZED_PROTOCOL = "model_feature_neutralized_v1"
MODEL_LABEL_PROTOCOL = "qlib_cs_rank_norm_v1"

QLIB_ALPHA158_MODEL_DEFAULTS: dict[str, JsonValue] = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 4,
    "early_stopping_rounds": 50,
    "num_boost_round": 1000,
}


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _date(value: object, field_name: str) -> date:
    raw = _text(value, field_name)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise CoreValidationError(
            f"{field_name} must use YYYY-MM-DD date format"
        ) from error
    if parsed.isoformat() != raw:
        raise CoreValidationError(f"{field_name} must use canonical YYYY-MM-DD")
    return parsed


def _ratio(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CoreValidationError(f"{field_name} must be a float in [0, 1]")
    return value


def _correlation_threshold(value: object) -> float:
    threshold = _ratio(value, "correlation_filter.threshold")
    if threshold <= 0.0:
        raise CoreValidationError("correlation_filter.threshold must be in (0, 1]")
    return threshold


def _positive_int(value: object, field_name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise CoreValidationError(f"{field_name} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class FactorSourceConfig:
    """One Graph factor source used as model input."""

    storage_root: str
    run_id: str
    include_rejected_factors: bool

    def __post_init__(self) -> None:
        """Validate source location."""
        _text(self.storage_root, "factor_source.storage_root")
        _text(self.run_id, "factor_source.run_id")
        if type(self.include_rejected_factors) is not bool:
            raise CoreValidationError(
                "factor_source.include_rejected_factors must be a boolean"
            )


@dataclass(frozen=True, slots=True)
class DateSegmentConfig:
    """One explicit inclusive model date segment."""

    start_date: str
    end_date: str

    def __post_init__(self) -> None:
        """Validate the inclusive date range."""
        start = _date(self.start_date, "segment.start_date")
        end = _date(self.end_date, "segment.end_date")
        if start > end:
            raise CoreValidationError("segment.start_date must not be after end_date")


@dataclass(frozen=True, slots=True)
class ModelSegmentsConfig:
    """Chronological train, validation, and test ranges."""

    train: DateSegmentConfig
    valid: DateSegmentConfig
    test: DateSegmentConfig

    def __post_init__(self) -> None:
        """Require ordered, non-overlapping ranges."""
        if not (
            isinstance(self.train, DateSegmentConfig)
            and isinstance(self.valid, DateSegmentConfig)
            and isinstance(self.test, DateSegmentConfig)
        ):
            raise CoreValidationError("segments must contain train, valid, and test")
        if _date(self.train.end_date, "segments.train.end_date") >= _date(
            self.valid.start_date, "segments.valid.start_date"
        ):
            raise CoreValidationError("train must end before valid starts")
        if _date(self.valid.end_date, "segments.valid.end_date") >= _date(
            self.test.start_date, "segments.test.start_date"
        ):
            raise CoreValidationError("valid must end before test starts")


@dataclass(frozen=True, slots=True)
class FeaturePreprocessingConfig:
    """Leakage-safe factor preprocessing settings."""

    protocol: str
    neutralization_enabled: bool
    industry_field: str
    continuous_exposures: tuple[str, ...]
    mad_multiplier: float
    min_cross_section_samples: int

    def __post_init__(self) -> None:
        """Validate the frozen model feature protocol."""
        if self.protocol not in (
            MODEL_FEATURE_PREPROCESSING_PROTOCOL,
            MODEL_FEATURE_NEUTRALIZED_PROTOCOL,
        ):
            raise CoreValidationError(
                "feature_preprocessing.protocol must be one of "
                f"{MODEL_FEATURE_PREPROCESSING_PROTOCOL!r} or "
                f"{MODEL_FEATURE_NEUTRALIZED_PROTOCOL!r}"
            )
        if type(self.neutralization_enabled) is not bool:
            raise CoreValidationError(
                "feature_preprocessing.neutralization_enabled must be a boolean"
            )
        expected_neutralization = self.protocol == MODEL_FEATURE_NEUTRALIZED_PROTOCOL
        if self.neutralization_enabled != expected_neutralization:
            raise CoreValidationError(
                "feature_preprocessing.protocol and neutralization_enabled must agree"
            )
        if self.industry_field != "industry_id":
            raise CoreValidationError(
                "feature_preprocessing.industry_field must be 'industry_id'"
            )
        expected = ("log_mkt_cap", "ret20", "turn20", "vol20")
        if self.continuous_exposures != expected:
            raise CoreValidationError(
                "feature_preprocessing.continuous_exposures must match " f"{expected!r}"
            )
        if (
            type(self.mad_multiplier) is not float
            or not math.isfinite(self.mad_multiplier)
            or self.mad_multiplier <= 0.0
        ):
            raise CoreValidationError(
                "feature_preprocessing.mad_multiplier must be positive"
            )
        _positive_int(
            self.min_cross_section_samples,
            "feature_preprocessing.min_cross_section_samples",
            minimum=2,
        )


@dataclass(frozen=True, slots=True)
class CorrelationFilterConfig:
    """Training-only greedy correlation filter settings."""

    enabled: bool
    threshold: float
    min_daily_effective_stock_ratio: float
    min_effective_date_ratio: float

    def __post_init__(self) -> None:
        """Validate filter thresholds."""
        if type(self.enabled) is not bool:
            raise CoreValidationError("correlation_filter.enabled must be a boolean")
        _correlation_threshold(self.threshold)
        _ratio(
            self.min_daily_effective_stock_ratio,
            "correlation_filter.min_daily_effective_stock_ratio",
        )
        _ratio(
            self.min_effective_date_ratio,
            "correlation_filter.min_effective_date_ratio",
        )


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """One fixed next-day return target and learning transform."""

    return_definition: str
    horizon_days: int
    normalization: str

    def __post_init__(self) -> None:
        """Reject label semantics outside the v1 contract."""
        if self.return_definition != "close_t1_over_close_t":
            raise CoreValidationError(
                "label.return_definition must be 'close_t1_over_close_t'"
            )
        if self.horizon_days != 1:
            raise CoreValidationError("label.horizon_days must be 1")
        if self.normalization != MODEL_LABEL_PROTOCOL:
            raise CoreValidationError(
                f"label.normalization must be {MODEL_LABEL_PROTOCOL!r}"
            )


@dataclass(frozen=True, slots=True)
class QlibLightGBMConfig:
    """Typed Qlib Alpha158 LightGBM benchmark parameters."""

    loss: str
    colsample_bytree: float
    learning_rate: float
    subsample: float
    lambda_l1: float
    lambda_l2: float
    max_depth: int
    num_leaves: int
    num_threads: int
    early_stopping_rounds: int
    num_boost_round: int

    def __post_init__(self) -> None:
        """Validate model parameters without allowing arbitrary kwargs."""
        if self.loss != "mse":
            raise CoreValidationError("model.loss must be 'mse'")
        for field_name in (
            "colsample_bytree",
            "learning_rate",
            "subsample",
        ):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise CoreValidationError(f"model.{field_name} must be positive")
        for field_name in ("lambda_l1", "lambda_l2"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise CoreValidationError(f"model.{field_name} must be non-negative")
        if self.colsample_bytree > 1.0 or self.subsample > 1.0:
            raise CoreValidationError("model sampling ratios must be <= 1")
        _positive_int(self.max_depth, "model.max_depth")
        _positive_int(self.num_leaves, "model.num_leaves", minimum=2)
        _positive_int(self.num_threads, "model.num_threads")
        _positive_int(self.early_stopping_rounds, "model.early_stopping_rounds")
        _positive_int(self.num_boost_round, "model.num_boost_round")


@dataclass(frozen=True, slots=True)
class LightGBMBacktestConfig:
    """Fixed Qlib TopkDropout absolute portfolio evaluation settings."""

    account: int
    return_mode: str
    benchmark: str
    topk: int
    n_drop: int
    method_sell: str
    method_buy: str
    hold_thresh: int
    only_tradable: bool
    forbid_all_trade_at_limit: bool
    risk_degree: float
    deal_price: str
    limit_threshold: float
    open_cost: float
    close_cost: float
    min_cost: int
    impact_cost: float
    trade_unit: int

    def __post_init__(self) -> None:
        """Validate the reproducible absolute-return backtest."""
        _positive_int(self.account, "backtest.account")
        if self.return_mode != "absolute":
            raise CoreValidationError("backtest.return_mode must be 'absolute'")
        if self.benchmark != "":
            raise CoreValidationError(
                "backtest.benchmark must be empty when return_mode is 'absolute'"
            )
        _positive_int(self.topk, "backtest.topk")
        _positive_int(self.n_drop, "backtest.n_drop")
        if self.method_sell != "bottom":
            raise CoreValidationError("backtest.method_sell must be 'bottom'")
        if self.method_buy != "top":
            raise CoreValidationError("backtest.method_buy must be 'top'")
        _positive_int(self.hold_thresh, "backtest.hold_thresh")
        if type(self.only_tradable) is not bool:
            raise CoreValidationError("backtest.only_tradable must be a boolean")
        if type(self.forbid_all_trade_at_limit) is not bool:
            raise CoreValidationError(
                "backtest.forbid_all_trade_at_limit must be a boolean"
            )
        if (
            type(self.risk_degree) is not float
            or not math.isfinite(self.risk_degree)
            or not 0.0 < self.risk_degree <= 1.0
        ):
            raise CoreValidationError("backtest.risk_degree must be in (0, 1]")
        if self.deal_price != "open":
            raise CoreValidationError("backtest.deal_price must be 'open'")
        _ratio(self.limit_threshold, "backtest.limit_threshold")
        for field_name in ("open_cost", "close_cost", "impact_cost"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise CoreValidationError(f"backtest.{field_name} must be non-negative")
        if type(self.min_cost) is not int or self.min_cost < 0:
            raise CoreValidationError(
                "backtest.min_cost must be a non-negative integer"
            )
        _positive_int(self.trade_unit, "backtest.trade_unit")


@dataclass(frozen=True, slots=True)
class ModelOutputConfig:
    """Local root for independent model runs."""

    root: str

    def __post_init__(self) -> None:
        """Validate output root."""
        _text(self.root, "output.root")


@dataclass(frozen=True, slots=True)
class LightGBMParallelConfig:
    """Cross-platform factor worker settings."""

    factor_workers: int
    native_threads: int

    def __post_init__(self) -> None:
        """Require one native numerical thread per factor worker."""
        _positive_int(self.factor_workers, "parallel.factor_workers")
        if self.native_threads != 1:
            raise CoreValidationError("parallel.native_threads must be 1")


@dataclass(frozen=True, slots=True)
class LightGBMIntegrationConfig:
    """Complete, immutable LightGBM integration configuration."""

    factor_source: FactorSourceConfig
    segments: ModelSegmentsConfig
    data: DataConfig
    universe: UniverseConfig
    feature_preprocessing: FeaturePreprocessingConfig
    correlation_filter: CorrelationFilterConfig
    label: LabelConfig
    model: QlibLightGBMConfig
    backtest: LightGBMBacktestConfig
    parallel: LightGBMParallelConfig
    output: ModelOutputConfig
    qlib: QlibConfig
    schema_version: str

    def __post_init__(self) -> None:
        """Validate cross-section and model-data invariants."""
        if self.schema_version != LIGHTGBM_CONFIG_SCHEMA_VERSION:
            raise CoreValidationError(
                f"schema_version must be {LIGHTGBM_CONFIG_SCHEMA_VERSION!r}"
            )
        if self.data.provider != "qlib" or not self.qlib.enabled:
            raise CoreValidationError("LightGBM integration requires enabled Qlib data")
        if self.data.market != "cn" or self.data.frequency != "1d":
            raise CoreValidationError("LightGBM v1 supports only cn daily data")
        if self.data.start_date != self.segments.train.start_date:
            raise CoreValidationError(
                "data.start_date must equal segments.train.start_date"
            )
        if self.data.end_date != self.segments.test.end_date:
            raise CoreValidationError("data.end_date must equal segments.test.end_date")
        if _date(self.data.warmup_start_date, "data.warmup_start_date") > _date(
            self.segments.train.start_date, "segments.train.start_date"
        ):
            raise CoreValidationError("data warmup must not start after training")


@dataclass(frozen=True, slots=True)
class MonthlySamplingConfig:
    """Monthly sampling and forward-label contract."""

    protocol: str
    label_horizon_trading_days: int
    tail_policy: str

    def __post_init__(self) -> None:
        """Validate the fixed month-end sampling protocol."""
        if self.protocol != "month_end_cross_section_v1":
            raise CoreValidationError(
                "sampling.protocol must be 'month_end_cross_section_v1'"
            )
        _positive_int(
            self.label_horizon_trading_days,
            "sampling.label_horizon_trading_days",
        )
        if self.label_horizon_trading_days != 20:
            raise CoreValidationError("sampling.label_horizon_trading_days must be 20")
        if self.tail_policy != "truncate_incomplete_forward_label":
            raise CoreValidationError(
                "sampling.tail_policy must be 'truncate_incomplete_forward_label'"
            )


@dataclass(frozen=True, slots=True)
class MonthlyLabelConfig:
    """Fixed 20-trading-day close-to-close model label."""

    return_definition: str
    horizon_days: int
    normalization: str

    def __post_init__(self) -> None:
        """Validate monthly label semantics."""
        if self.return_definition != "close_t20_over_close_t":
            raise CoreValidationError(
                "label.return_definition must be 'close_t20_over_close_t'"
            )
        if self.horizon_days != 20:
            raise CoreValidationError("label.horizon_days must be 20")
        if self.normalization != MODEL_LABEL_PROTOCOL:
            raise CoreValidationError(
                f"label.normalization must be {MODEL_LABEL_PROTOCOL!r}"
            )


@dataclass(frozen=True, slots=True)
class LightGBMMonthlyIntegrationConfig:
    """Complete standalone configuration for monthly LightGBM integration."""

    factor_source: FactorSourceConfig
    segments: ModelSegmentsConfig
    data: DataConfig
    universe: UniverseConfig
    feature_preprocessing: FeaturePreprocessingConfig
    correlation_filter: CorrelationFilterConfig
    label: MonthlyLabelConfig
    sampling: MonthlySamplingConfig
    model: QlibLightGBMConfig
    backtest: LightGBMBacktestConfig
    parallel: LightGBMParallelConfig
    output: ModelOutputConfig
    qlib: QlibConfig
    schema_version: str

    def __post_init__(self) -> None:
        """Validate monthly data and evaluation invariants."""
        if self.schema_version != LIGHTGBM_MONTHLY_CONFIG_SCHEMA_VERSION:
            raise CoreValidationError(
                "schema_version must be " f"{LIGHTGBM_MONTHLY_CONFIG_SCHEMA_VERSION!r}"
            )
        if self.data.provider != "qlib" or not self.qlib.enabled:
            raise CoreValidationError("monthly integration requires enabled Qlib data")
        if self.data.market != "cn" or self.data.frequency != "1d":
            raise CoreValidationError(
                "monthly integration uses cn daily source data and month-end sampling"
            )
        if self.universe.name != "csi300":
            raise CoreValidationError(
                "monthly integration requires the csi300 universe"
            )
        if self.data.start_date != self.segments.train.start_date:
            raise CoreValidationError(
                "data.start_date must equal segments.train.start_date"
            )
        if self.data.end_date != self.segments.test.end_date:
            raise CoreValidationError("data.end_date must equal segments.test.end_date")
        if _date(self.data.warmup_start_date, "data.warmup_start_date") > _date(
            self.segments.train.start_date, "segments.train.start_date"
        ):
            raise CoreValidationError("data warmup must not start after training")
        if self.correlation_filter.enabled:
            raise CoreValidationError(
                "monthly correlation_filter.enabled must be false"
            )


@dataclass(frozen=True, slots=True)
class LightGBMMonthlyConfigSnapshot:
    """Content-identified monthly model configuration snapshot."""

    snapshot_id: str
    config: LightGBMMonthlyIntegrationConfig
    schema_version: str = LIGHTGBM_MONTHLY_CONFIG_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate monthly snapshot identity."""
        if self.schema_version != LIGHTGBM_MONTHLY_CONFIG_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError("invalid monthly LightGBM config snapshot schema")
        if self.snapshot_id != stable_sha256(to_json_value(self.config)):
            raise CoreValidationError(
                "monthly LightGBM config snapshot identity mismatch"
            )


@dataclass(frozen=True, slots=True)
class LightGBMConfigSnapshot:
    """Content-identified resolved model configuration."""

    snapshot_id: str
    config: LightGBMIntegrationConfig
    schema_version: str = LIGHTGBM_CONFIG_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate snapshot identity."""
        if self.schema_version != LIGHTGBM_CONFIG_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError("invalid LightGBM config snapshot schema")
        if self.snapshot_id != stable_sha256(to_json_value(self.config)):
            raise CoreValidationError("LightGBM config snapshot identity mismatch")


def decode_lightgbm_config_snapshot(value: JsonValue) -> LightGBMConfigSnapshot:
    """Decode a snapshot while migrating the pre-source-mode snapshot shape.

    The original LightGBM snapshot did not contain
    ``factor_source.include_rejected_factors``.  Snapshots are immutable, so
    migration normalizes the in-memory value and recomputes its content ID;
    it never rewrites the stored object.
    """
    if not isinstance(value, dict):
        return from_json_value(value, LightGBMConfigSnapshot)
    normalized = dict(value)
    raw_config = normalized.get("config")
    if isinstance(raw_config, dict):
        config = dict(raw_config)
        raw_source = config.get("factor_source")
        if isinstance(raw_source, dict):
            source = dict(raw_source)
            source.setdefault("include_rejected_factors", False)
            config["factor_source"] = source
        raw_preprocessing = config.get("feature_preprocessing")
        if isinstance(raw_preprocessing, dict):
            preprocessing = dict(raw_preprocessing)
            protocol = preprocessing.get("protocol")
            preprocessing.setdefault(
                "neutralization_enabled",
                protocol == MODEL_FEATURE_NEUTRALIZED_PROTOCOL,
            )
            config["feature_preprocessing"] = preprocessing
        raw_parallel = config.get("parallel")
        if raw_parallel is None:
            config["parallel"] = {"factor_workers": 1, "native_threads": 1}
        elif isinstance(raw_parallel, dict):
            parallel = dict(raw_parallel)
            parallel.setdefault("factor_workers", 1)
            parallel.setdefault("native_threads", 1)
            config["parallel"] = parallel
        normalized["config"] = config
        normalized["snapshot_id"] = stable_sha256(cast(JsonValue, config))
    return from_json_value(cast(JsonValue, normalized), LightGBMConfigSnapshot)


def load_lightgbm_integration_config(
    path: str | Path,
) -> LightGBMIntegrationConfig:
    """Load one strict standalone LightGBM TOML configuration."""
    try:
        with Path(path).open("rb") as config_file:
            values = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise SerializationError(f"invalid TOML config: {error}") from error
    return load_lightgbm_integration_config_values(values)


def load_lightgbm_integration_config_values(
    values: Mapping[str, object],
) -> LightGBMIntegrationConfig:
    """Decode parsed values after applying documented model defaults."""
    if not isinstance(values, Mapping):
        raise CoreValidationError("configuration values must be a mapping")
    normalized = dict(values)
    _merge_defaults(
        normalized,
        "factor_source",
        {"include_rejected_factors": True},
    )
    preprocessing = normalized.get("feature_preprocessing")
    default_feature_protocol = MODEL_FEATURE_NEUTRALIZED_PROTOCOL
    if (
        isinstance(preprocessing, Mapping)
        and preprocessing.get("neutralization_enabled") is False
    ):
        default_feature_protocol = MODEL_FEATURE_PREPROCESSING_PROTOCOL
    _merge_defaults(
        normalized,
        "feature_preprocessing",
        {
            "protocol": default_feature_protocol,
            "neutralization_enabled": True,
            "industry_field": "industry_id",
            "continuous_exposures": ["log_mkt_cap", "ret20", "turn20", "vol20"],
            "mad_multiplier": 5.0,
            "min_cross_section_samples": 100,
        },
    )
    _merge_defaults(
        normalized,
        "correlation_filter",
        {
            "enabled": False,
            "threshold": 0.8,
            "min_daily_effective_stock_ratio": 0.8,
            "min_effective_date_ratio": 0.8,
        },
    )
    _merge_defaults(
        normalized,
        "label",
        {
            "return_definition": "close_t1_over_close_t",
            "horizon_days": 1,
            "normalization": MODEL_LABEL_PROTOCOL,
        },
    )
    _merge_defaults(normalized, "model", QLIB_ALPHA158_MODEL_DEFAULTS)
    _merge_defaults(
        normalized,
        "parallel",
        {
            "factor_workers": 1,
            "native_threads": 1,
        },
    )
    _merge_defaults(
        normalized,
        "backtest",
        {
            "account": 100000000,
            "return_mode": "absolute",
            "benchmark": "",
            "topk": 50,
            "n_drop": 5,
            "method_sell": "bottom",
            "method_buy": "top",
            "hold_thresh": 1,
            "only_tradable": False,
            "forbid_all_trade_at_limit": True,
            "risk_degree": 0.95,
            "deal_price": "open",
            "limit_threshold": 0.095,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0,
            "impact_cost": 0.0,
            "trade_unit": 100,
        },
    )
    return from_json_value(cast(JsonValue, normalized), LightGBMIntegrationConfig)


def create_lightgbm_config_snapshot(
    config: LightGBMIntegrationConfig,
) -> LightGBMConfigSnapshot:
    """Create a deterministic resolved configuration snapshot."""
    return LightGBMConfigSnapshot(
        snapshot_id=stable_sha256(to_json_value(config)),
        config=config,
    )


def create_lightgbm_monthly_config_snapshot(
    config: LightGBMMonthlyIntegrationConfig,
) -> LightGBMMonthlyConfigSnapshot:
    """Create a deterministic monthly configuration snapshot."""
    return LightGBMMonthlyConfigSnapshot(
        snapshot_id=stable_sha256(to_json_value(config)),
        config=config,
    )


def load_lightgbm_monthly_integration_config(
    path: str | Path,
) -> LightGBMMonthlyIntegrationConfig:
    """Load one strict standalone monthly LightGBM TOML configuration."""
    try:
        with Path(path).open("rb") as config_file:
            values = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise SerializationError(f"invalid TOML config: {error}") from error
    return load_lightgbm_monthly_integration_config_values(values)


def load_lightgbm_monthly_integration_config_values(
    values: Mapping[str, object],
) -> LightGBMMonthlyIntegrationConfig:
    """Decode monthly configuration values after applying documented defaults."""
    if not isinstance(values, Mapping):
        raise CoreValidationError("configuration values must be a mapping")
    normalized = dict(values)
    _merge_defaults(normalized, "factor_source", {"include_rejected_factors": False})
    _merge_defaults(
        normalized,
        "feature_preprocessing",
        {
            "protocol": MODEL_FEATURE_NEUTRALIZED_PROTOCOL,
            "neutralization_enabled": True,
            "industry_field": "industry_id",
            "continuous_exposures": ["log_mkt_cap", "ret20", "turn20", "vol20"],
            "mad_multiplier": 5.0,
            "min_cross_section_samples": 100,
        },
    )
    _merge_defaults(
        normalized,
        "correlation_filter",
        {
            "enabled": False,
            "threshold": 0.8,
            "min_daily_effective_stock_ratio": 0.8,
            "min_effective_date_ratio": 0.8,
        },
    )
    _merge_defaults(
        normalized,
        "label",
        {
            "return_definition": "close_t20_over_close_t",
            "horizon_days": 20,
            "normalization": MODEL_LABEL_PROTOCOL,
        },
    )
    _merge_defaults(
        normalized,
        "sampling",
        {
            "protocol": "month_end_cross_section_v1",
            "label_horizon_trading_days": 20,
            "tail_policy": "truncate_incomplete_forward_label",
        },
    )
    model_defaults = dict(QLIB_ALPHA158_MODEL_DEFAULTS)
    model_defaults["num_threads"] = 4
    _merge_defaults(normalized, "model", model_defaults)
    _merge_defaults(normalized, "parallel", {"factor_workers": 24, "native_threads": 1})
    _merge_defaults(
        normalized,
        "backtest",
        {
            "account": 100000000,
            "return_mode": "absolute",
            "benchmark": "",
            "topk": 50,
            "n_drop": 5,
            "method_sell": "bottom",
            "method_buy": "top",
            "hold_thresh": 1,
            "only_tradable": False,
            "forbid_all_trade_at_limit": True,
            "risk_degree": 0.95,
            "deal_price": "open",
            "limit_threshold": 0.095,
            "open_cost": 0.0015,
            "close_cost": 0.0025,
            "min_cost": 5,
            "impact_cost": 0.0,
            "trade_unit": 100,
        },
    )
    return from_json_value(
        cast(JsonValue, normalized), LightGBMMonthlyIntegrationConfig
    )


def _merge_defaults(
    values: dict[str, object], section: str, defaults: Mapping[str, JsonValue]
) -> None:
    raw = values.get(section)
    if raw is None:
        merged: dict[str, object] = {}
    elif isinstance(raw, Mapping):
        merged = dict(raw)
    else:
        return
    for key, value in defaults.items():
        merged.setdefault(key, value)
    values[section] = merged
