"""Strict experiment configuration loading and snapshotting."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from cosmosalpha.core.errors import CoreValidationError, SerializationError
from cosmosalpha.core.identifiers import stable_sha256
from cosmosalpha.core.serialization import (
    JsonValue,
    _parse_json,
    from_json_value,
    to_json,
    to_json_value,
)

DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/graph_experiment.toml")
_CONFIG_SCHEMA_VERSION = "experiment-config/1.6"
_CONFIG_SNAPSHOT_SCHEMA_VERSION = "config-snapshot/1.0"
_REDACTED_VALUE = "<redacted>"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CoreValidationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_int_tuple(value: object, field_name: str, *, minimum: int = 0) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_int(item, f"{field_name}[{index}]", minimum=minimum)


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise CoreValidationError(f"{field_name} must be a boolean")
    return value


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise CoreValidationError(f"{field_name} must be a finite float")
    if minimum is not None and value < minimum:
        raise CoreValidationError(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise CoreValidationError(f"{field_name} must be at most {maximum}")
    return value


def _parse_config_date(value: object, field_name: str) -> date:
    raw = _require_string(value, field_name)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise CoreValidationError(
            f"{field_name} must use YYYY-MM-DD date format"
        ) from error
    if parsed.isoformat() != raw:
        raise CoreValidationError(f"{field_name} must use canonical YYYY-MM-DD")
    return parsed


def _require_schema_version(value: object, field_name: str) -> str:
    version = _require_string(value, field_name)
    if version not in {_CONFIG_SCHEMA_VERSION, "experiment-config/1.5"}:
        raise CoreValidationError(f"{field_name} must be {_CONFIG_SCHEMA_VERSION!r}")
    return version


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Market data selection and versioning configuration."""

    provider: str
    dataset_version: str
    price_adjustment: str
    start_date: str
    end_date: str
    warmup_start_date: str
    market: str
    frequency: str
    trading_calendar_version: str

    def __post_init__(self) -> None:
        """Validate data configuration."""
        _require_string(self.provider, "data.provider")
        _require_string(self.dataset_version, "data.dataset_version")
        _require_string(self.price_adjustment, "data.price_adjustment")
        start = _parse_config_date(self.start_date, "data.start_date")
        end = _parse_config_date(self.end_date, "data.end_date")
        warmup = _parse_config_date(self.warmup_start_date, "data.warmup_start_date")
        if warmup > start:
            raise CoreValidationError(
                "data.warmup_start_date must not be after data.start_date"
            )
        if start > end:
            raise CoreValidationError("data.start_date must not be after data.end_date")
        _require_string(self.market, "data.market")
        _require_string(self.frequency, "data.frequency")
        _require_string(
            self.trading_calendar_version,
            "data.trading_calendar_version",
        )


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Universe construction configuration."""

    name: str
    new_stock_min_age_years: int

    def __post_init__(self) -> None:
        """Validate universe configuration."""
        _require_string(self.name, "universe.name")
        _require_int(
            self.new_stock_min_age_years,
            "universe.new_stock_min_age_years",
        )


@dataclass(frozen=True, slots=True)
class ReturnsConfig:
    """Forward return calculation configuration."""

    single_factor: str
    suspension_compression: str
    horizon_days: int

    def __post_init__(self) -> None:
        """Validate return calculation configuration."""
        _require_string(self.single_factor, "returns.single_factor")
        _require_string(
            self.suspension_compression,
            "returns.suspension_compression",
        )
        _require_int(self.horizon_days, "returns.horizon_days", minimum=1)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Factor preprocessing and neutralization configuration."""

    protocol: str
    missing_policy: str
    rolling_min_fraction: float
    industry_field: str
    continuous_exposures: tuple[str, ...]
    mad_multiplier: float
    rolling_std_ddof: int

    def __post_init__(self) -> None:
        """Validate preprocessing configuration."""
        _require_string(self.protocol, "preprocessing.protocol")
        _require_string(self.missing_policy, "preprocessing.missing_policy")
        _require_float(
            self.rolling_min_fraction,
            "preprocessing.rolling_min_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_string(self.industry_field, "preprocessing.industry_field")
        _require_string_tuple(
            self.continuous_exposures,
            "preprocessing.continuous_exposures",
        )
        _require_float(
            self.mad_multiplier,
            "preprocessing.mad_multiplier",
            minimum=0.0,
        )
        _require_int(self.rolling_std_ddof, "preprocessing.rolling_std_ddof")


@dataclass(frozen=True, slots=True)
class OperatorsConfig:
    """Operator catalog selection configuration."""

    operator_set: str
    registry_version: str
    allowed_windows: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate operator configuration."""
        _require_string(self.operator_set, "operators.operator_set")
        _require_string(self.registry_version, "operators.registry_version")
        _require_int_tuple(self.allowed_windows, "operators.allowed_windows", minimum=1)


@dataclass(frozen=True, slots=True)
class FactorGenerationConfig:
    """Factor proposal generation limits."""

    max_proposals: int
    max_expression_depth: int
    max_validation_repair_attempts: int
    duplicate_policy: str

    def __post_init__(self) -> None:
        """Validate factor generation configuration."""
        if type(self.max_proposals) is not int or self.max_proposals not in (1, 2):
            raise CoreValidationError("factor_generation.max_proposals must be 1 or 2")
        _require_int(
            self.max_expression_depth,
            "factor_generation.max_expression_depth",
            minimum=1,
        )
        if (
            type(self.max_validation_repair_attempts) is not int
            or self.max_validation_repair_attempts < 0
            or self.max_validation_repair_attempts > 2
        ):
            raise CoreValidationError(
                "factor_generation.max_validation_repair_attempts "
                "must be between 0 and 2"
            )
        _require_string(self.duplicate_policy, "factor_generation.duplicate_policy")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Factor evaluation and acceptance threshold configuration."""

    protocol: str
    metrics: tuple[str, ...]
    min_effective_sample_count: int
    min_cross_section_samples: int
    acceptance_metric: str
    acceptance_threshold: float
    min_effective_date_ratio: float = 0.8
    acceptance_use_absolute_value: bool = True
    min_daily_effective_stock_ratio: float = 0.8

    def __post_init__(self) -> None:
        """Validate evaluation configuration."""
        _require_string(self.protocol, "evaluation.protocol")
        _require_string_tuple(self.metrics, "evaluation.metrics")
        _require_int(
            self.min_effective_sample_count,
            "evaluation.min_effective_sample_count",
            minimum=1,
        )
        _require_int(
            self.min_cross_section_samples,
            "evaluation.min_cross_section_samples",
            minimum=2,
        )
        _require_string(self.acceptance_metric, "evaluation.acceptance_metric")
        _require_float(self.acceptance_threshold, "evaluation.acceptance_threshold")
        _require_float(
            self.min_effective_date_ratio,
            "evaluation.min_effective_date_ratio",
        )
        if not 0.0 <= self.min_effective_date_ratio <= 1.0:
            raise CoreValidationError(
                "evaluation.min_effective_date_ratio must be between 0.0 and 1.0"
            )
        _require_bool(
            self.acceptance_use_absolute_value,
            "evaluation.acceptance_use_absolute_value",
        )
        _require_float(
            self.min_daily_effective_stock_ratio,
            "evaluation.min_daily_effective_stock_ratio",
        )
        if not 0.0 <= self.min_daily_effective_stock_ratio <= 1.0:
            raise CoreValidationError(
                "evaluation.min_daily_effective_stock_ratio must be between 0.0 and 1.0"
            )


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Workflow retry and cycle control configuration."""

    max_cycles: int
    max_retries: int

    def __post_init__(self) -> None:
        """Validate workflow configuration."""
        _require_int(self.max_cycles, "workflow.max_cycles", minimum=1)
        _require_int(self.max_retries, "workflow.max_retries")


@dataclass(frozen=True, slots=True)
class PromptsConfig:
    """Prompt rendering budget configuration."""

    manifest: str
    max_input_tokens: int
    max_history_items: int
    max_proposals_per_call: int
    max_format_retries: int

    def __post_init__(self) -> None:
        """Validate prompt configuration."""
        _require_string(self.manifest, "prompts.manifest")
        _require_int(self.max_input_tokens, "prompts.max_input_tokens", minimum=1)
        _require_int(self.max_history_items, "prompts.max_history_items")
        if type(
            self.max_proposals_per_call
        ) is not int or self.max_proposals_per_call not in (1, 2):
            raise CoreValidationError("prompts.max_proposals_per_call must be 1 or 2")
        _require_int(self.max_format_retries, "prompts.max_format_retries")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM adapter configuration without exposing provider SDK types."""

    provider: str
    model_alias: str
    base_url: str
    base_url_env: str
    organization: str
    api_key: str
    api_key_env: str
    timeout_seconds: int
    temperature: float
    max_output_tokens: int
    max_timeout_retries: int
    max_rate_limit_retries: int
    max_server_retries: int
    retry_initial_backoff_seconds: float
    retry_max_backoff_seconds: float
    replay_only: bool
    reasoning_effort: str | None = None
    max_empty_retries: int | None = None

    def __post_init__(self) -> None:
        """Validate LLM configuration."""
        _require_string(self.provider, "llm.provider")
        _require_string(self.model_alias, "llm.model_alias")
        _require_string(self.base_url, "llm.base_url")
        _require_string(self.base_url_env, "llm.base_url_env")
        _require_string(self.organization, "llm.organization")
        _require_string(self.api_key, "llm.api_key")
        _require_string(self.api_key_env, "llm.api_key_env")
        _require_int(self.timeout_seconds, "llm.timeout_seconds", minimum=1)
        _require_float(self.temperature, "llm.temperature", minimum=0.0)
        _require_int(self.max_output_tokens, "llm.max_output_tokens", minimum=1)
        _require_int(self.max_timeout_retries, "llm.max_timeout_retries")
        _require_int(self.max_rate_limit_retries, "llm.max_rate_limit_retries")
        _require_int(self.max_server_retries, "llm.max_server_retries")
        if self.max_empty_retries is not None:
            _require_int(self.max_empty_retries, "llm.max_empty_retries")
        _require_float(
            self.retry_initial_backoff_seconds,
            "llm.retry_initial_backoff_seconds",
            minimum=0.0,
        )
        _require_float(
            self.retry_max_backoff_seconds,
            "llm.retry_max_backoff_seconds",
            minimum=0.0,
            maximum=60.0,
        )
        if self.retry_max_backoff_seconds < self.retry_initial_backoff_seconds:
            raise CoreValidationError(
                "llm.retry_max_backoff_seconds must not be less than "
                "llm.retry_initial_backoff_seconds"
            )
        _require_bool(self.replay_only, "llm.replay_only")
        if self.reasoning_effort is not None:
            _require_string(self.reasoning_effort, "llm.reasoning_effort")
            if self.reasoning_effort not in REASONING_EFFORTS:
                raise CoreValidationError(
                    "llm.reasoning_effort must be one of "
                    f"{REASONING_EFFORTS!r} or None"
                )


@dataclass(frozen=True, slots=True)
class QlibSourceFieldConfig:
    """One Qlib source field mapping used by startup validation."""

    field_name: str
    expression: str

    def __post_init__(self) -> None:
        """Validate source field mapping."""
        _require_string(self.field_name, "field_name")
        _require_string(self.expression, "expression")


@dataclass(frozen=True, slots=True)
class QlibSourceConfig:
    """One Qlib market, exposure, or mask source."""

    provider_uri: str
    fields: tuple[QlibSourceFieldConfig, ...]

    def __post_init__(self) -> None:
        """Validate source identity and field mappings."""
        _require_string(self.provider_uri, "provider_uri")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise CoreValidationError("fields must be a non-empty tuple")
        if not all(isinstance(item, QlibSourceFieldConfig) for item in self.fields):
            raise CoreValidationError("fields must contain QlibSourceFieldConfig")
        names = tuple(item.field_name for item in self.fields)
        if len(names) != len(set(names)):
            raise CoreValidationError("Qlib source field names must be unique")


@dataclass(frozen=True, slots=True)
class QlibConfig:
    """Qlib sources plus an optional explicit instrument-list override."""

    enabled: bool
    region: str
    instruments: tuple[str, ...]
    market: QlibSourceConfig
    exposure: QlibSourceConfig
    mask: QlibSourceConfig

    def __post_init__(self) -> None:
        """Validate Qlib source selection without importing Qlib."""
        _require_bool(self.enabled, "qlib.enabled")
        _require_string(self.region, "qlib.region")
        _require_string_tuple(self.instruments, "qlib.instruments")
        if not isinstance(self.market, QlibSourceConfig):
            raise CoreValidationError("qlib.market must be QlibSourceConfig")
        if not isinstance(self.exposure, QlibSourceConfig):
            raise CoreValidationError("qlib.exposure must be QlibSourceConfig")
        if not isinstance(self.mask, QlibSourceConfig):
            raise CoreValidationError("qlib.mask must be QlibSourceConfig")


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Storage backend and root directory for run-scoped outputs."""

    backend: str
    root: str

    def __post_init__(self) -> None:
        """Validate storage configuration."""
        _require_string(self.backend, "storage.backend")
        _require_string(self.root, "storage.root")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime resource configuration."""

    max_workers: int
    random_seed: int

    def __post_init__(self) -> None:
        """Validate runtime configuration."""
        _require_int(self.max_workers, "runtime.max_workers", minimum=1)
        _require_int(self.random_seed, "runtime.random_seed")


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Top-level mining architecture selection."""

    name: str
    directions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate architecture research inputs."""
        _require_string(self.name, "architecture.name")
        _require_string_tuple(self.directions, "architecture.directions")
        if self.name not in {"chain", "mcts", "graph", "parallel_branches"}:
            raise CoreValidationError(
                "architecture.name must be 'chain', 'mcts', 'graph', or "
                "'parallel_branches'"
            )
        if len(self.directions) != len(set(self.directions)):
            raise CoreValidationError("architecture.directions must be unique")
        if self.name in {"chain", "mcts", "graph"} and self.directions:
            raise CoreValidationError(
                f"{self.name} architecture must not define directions"
            )
        if self.name == "parallel_branches" and len(self.directions) < 2:
            raise CoreValidationError(
                "parallel_branches architecture requires at least two directions"
            )


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Search controls for single-tree MCTS through the formal run entry point."""

    max_iterations: int = 50
    exploration_weight: float = 1.0
    initial_budget: int = 3
    budget_increment: int = 3
    breakthrough_relax: float = 0.95

    def __post_init__(self) -> None:
        """Validate finite search bounds and dynamic-budget controls."""
        _require_int(self.max_iterations, "mcts.max_iterations", minimum=1)
        _require_float(self.exploration_weight, "mcts.exploration_weight", minimum=0.0)
        _require_int(self.initial_budget, "mcts.initial_budget", minimum=1)
        _require_int(self.budget_increment, "mcts.budget_increment")
        _require_float(
            self.breakthrough_relax,
            "mcts.breakthrough_relax",
            minimum=0.0,
            maximum=1.0,
        )
        if self.breakthrough_relax == 0.0:
            raise CoreValidationError("mcts.breakthrough_relax must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Strict, immutable configuration for one experiment run."""

    data: DataConfig
    universe: UniverseConfig
    returns: ReturnsConfig
    preprocessing: PreprocessingConfig
    operators: OperatorsConfig
    factor_generation: FactorGenerationConfig
    evaluation: EvaluationConfig
    workflow: WorkflowConfig
    prompts: PromptsConfig
    llm: LLMConfig
    storage: StorageConfig
    runtime: RuntimeConfig
    architecture: ArchitectureConfig
    qlib: QlibConfig
    schema_version: str
    graph: dict[str, JsonValue] | None = None
    mcts: MCTSConfig | None = None

    def __post_init__(self) -> None:
        """Validate top-level experiment configuration sections."""
        for field_name, field_type in (
            ("data", DataConfig),
            ("universe", UniverseConfig),
            ("returns", ReturnsConfig),
            ("preprocessing", PreprocessingConfig),
            ("operators", OperatorsConfig),
            ("factor_generation", FactorGenerationConfig),
            ("evaluation", EvaluationConfig),
            ("workflow", WorkflowConfig),
            ("prompts", PromptsConfig),
            ("llm", LLMConfig),
            ("storage", StorageConfig),
            ("runtime", RuntimeConfig),
            ("architecture", ArchitectureConfig),
            ("qlib", QlibConfig),
        ):
            if not isinstance(getattr(self, field_name), field_type):
                raise CoreValidationError(f"{field_name} must be {field_type.__name__}")
        if self.graph is not None:
            if not isinstance(self.graph, dict) or not all(
                type(key) is str for key in self.graph
            ):
                raise CoreValidationError(
                    "graph must be a string-keyed mapping or None"
                )
        if self.architecture.name == "graph" and not self.graph:
            raise CoreValidationError(
                "graph configuration is required when architecture.name is 'graph'"
            )
        if self.mcts is not None and not isinstance(self.mcts, MCTSConfig):
            raise CoreValidationError("mcts must be MCTSConfig or None")
        if self.architecture.name == "mcts" and self.mcts is None:
            raise CoreValidationError(
                "mcts configuration is required when architecture.name is 'mcts'"
            )
        _require_schema_version(self.schema_version, "schema_version")
        if self.prompts.max_proposals_per_call > self.factor_generation.max_proposals:
            raise CoreValidationError(
                "prompts.max_proposals_per_call must not exceed "
                "factor_generation.max_proposals"
            )


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """Redacted, deterministic snapshot of an experiment configuration."""

    snapshot_id: str
    config: ExperimentConfig
    redacted_json: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate configuration snapshot metadata."""
        _require_string(self.snapshot_id, "snapshot_id")
        if not isinstance(self.config, ExperimentConfig):
            raise CoreValidationError("config must be ExperimentConfig")
        _require_string(self.redacted_json, "redacted_json")
        if self.schema_version != _CONFIG_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError(
                f"schema_version must be {_CONFIG_SNAPSHOT_SCHEMA_VERSION!r}"
            )


def load_experiment_config(
    path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> ExperimentConfig:
    """Load a strict experiment configuration from a TOML file.

    Args:
        path: TOML configuration path.

    Returns:
        Parsed immutable experiment configuration.
    """
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            values = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise SerializationError(f"invalid TOML config: {error}") from error
    return load_experiment_config_values(values)


def decode_config_snapshot(data: str | bytes | bytearray) -> ConfigSnapshot:
    """Read legacy Graph snapshots without rewriting their contents or IDs."""
    value = _parse_json(data)
    if isinstance(value, dict):
        config = value.get("config")
        if isinstance(config, dict) and config.get("schema_version") == (
            "experiment-config/1.5"
        ):
            normalized = dict(config)
            normalized.setdefault("mcts", None)
            value = {**value, "config": normalized}
    return from_json_value(value, ConfigSnapshot)


def load_experiment_config_values(values: Mapping[str, object]) -> ExperimentConfig:
    """Decode parsed configuration values using the strict typed protocol codec.

    Args:
        values: Parsed mapping, typically produced by ``tomllib``.

    Returns:
        Parsed immutable experiment configuration.
    """
    if not isinstance(values, Mapping):
        raise CoreValidationError("configuration values must be a mapping")
    normalized_values = dict(values)
    if normalized_values.get("schema_version") == "experiment-config/1.5":
        normalized_values["schema_version"] = _CONFIG_SCHEMA_VERSION
    normalized_values.setdefault("graph", None)
    normalized_values.setdefault("mcts", None)
    architecture_values = normalized_values.get("architecture")
    if (
        isinstance(architecture_values, Mapping)
        and architecture_values.get("name") == "mcts"
        and normalized_values["mcts"] is None
    ):
        normalized_values["mcts"] = {}
    mcts_values = normalized_values["mcts"]
    if isinstance(mcts_values, Mapping):
        defaults = cast(dict[str, JsonValue], to_json_value(MCTSConfig()))
        normalized_values["mcts"] = {**defaults, **mcts_values}
    llm_values = normalized_values.get("llm")
    if isinstance(llm_values, Mapping):
        normalized_llm = dict(llm_values)
        # ``max_empty_retries`` is optional: when a configuration omits it the
        # retry policy falls back to ``max_server_retries``.  Configurations
        # that pass it keep their explicit empty-body budget.
        normalized_llm.setdefault("max_empty_retries", None)
        if "reasoning_effort" not in normalized_llm:
            normalized_llm["reasoning_effort"] = None
        normalized_values["llm"] = normalized_llm
    return from_json_value(cast(JsonValue, normalized_values), ExperimentConfig)


def create_config_snapshot(config: ExperimentConfig) -> ConfigSnapshot:
    """Create a redacted deterministic snapshot for a parsed configuration.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Snapshot whose identifier is the SHA-256 of redacted canonical JSON.
    """
    if not isinstance(config, ExperimentConfig):
        raise CoreValidationError("config must be ExperimentConfig")
    redacted_values = _redact_config_value(to_json_value(config))
    redacted_config = from_json_value(redacted_values, ExperimentConfig)
    redacted_json = to_json(redacted_values)
    return ConfigSnapshot(
        snapshot_id=stable_sha256(redacted_values),
        config=redacted_config,
        redacted_json=redacted_json,
        schema_version=_CONFIG_SNAPSHOT_SCHEMA_VERSION,
    )


def _redact_config_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_redact_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                _REDACTED_VALUE
                if _is_sensitive_key(key)
                else _redact_config_value(item)
            )
            for key, item in value.items()
        }
    return value


def _is_sensitive_key(key: str) -> bool:
    return (
        key in {"api_key", "api_key_env", "base_url_env", "organization"}
        or key.startswith("secret")
        or key.endswith("_env")
    )
