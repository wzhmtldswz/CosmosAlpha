"""Explicit small-sample smoke entry point for real Qlib snapshots."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cosmosalpha.core import CoreValidationError, JsonValue, to_json
from cosmosalpha.data.models import MarketDataRequest
from cosmosalpha.data.providers import (
    QlibFeatureSource,
    QlibFieldMapping,
    QlibMarketDataProvider,
)
from cosmosalpha.storage import LocalArtifactRepository


@dataclass(frozen=True, slots=True)
class QlibSnapshotSmokeConfig:
    """Configuration for an explicit, small real-Qlib snapshot smoke run."""

    request: MarketDataRequest
    instruments: tuple[str, ...]
    region: str
    market_provider_uri: str
    trading_calendar_version: str
    exposure_source: QlibFeatureSource
    mask_source: QlibFeatureSource

    def __post_init__(self) -> None:
        """Validate smoke scope and infrastructure source identities."""
        if not isinstance(self.request, MarketDataRequest):
            raise CoreValidationError("request must be MarketDataRequest")
        if not isinstance(self.instruments, tuple):
            raise CoreValidationError("instruments must be a tuple")
        if not all(
            type(instrument) is str and instrument for instrument in self.instruments
        ):
            raise CoreValidationError("instruments must contain non-empty strings")
        for field_name in (
            "region",
            "market_provider_uri",
            "trading_calendar_version",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise CoreValidationError(f"{field_name} must be a non-empty string")
        if not isinstance(self.exposure_source, QlibFeatureSource):
            raise CoreValidationError("exposure_source must be QlibFeatureSource")
        if not isinstance(self.mask_source, QlibFeatureSource):
            raise CoreValidationError("mask_source must be QlibFeatureSource")


def load_qlib_snapshot_smoke_config(path: str | Path) -> QlibSnapshotSmokeConfig:
    """Load a strict TOML configuration for the real-Qlib snapshot smoke entry.

    Args:
        path: Explicit TOML configuration file describing all Qlib sources.

    Returns:
        A validated small-sample Qlib snapshot configuration.
    """
    config_path = Path(path)
    try:
        with config_path.open("rb") as file:
            values = cast(dict[str, JsonValue], tomllib.load(file))
    except FileNotFoundError as error:
        raise CoreValidationError(
            f"Qlib smoke config not found: {config_path}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise CoreValidationError(f"invalid Qlib smoke TOML: {error}") from error
    expected = {"request", "sources", "instruments"}
    if set(values) != expected:
        raise CoreValidationError(
            "Qlib smoke config must contain request, sources, instruments"
        )
    request_values = _mapping(values["request"], "request")
    request = MarketDataRequest(
        market=_string(request_values, "market"),
        universe=_string(request_values, "universe"),
        start_date=_string(request_values, "start_date"),
        end_date=_string(request_values, "end_date"),
        warmup_start_date=_string(request_values, "warmup_start_date"),
        frequency=_string(request_values, "frequency"),
        fields=_string_tuple(request_values, "fields"),
        dataset_version=_string(request_values, "dataset_version"),
        price_adjustment=_string(request_values, "price_adjustment"),
        schema_version="market-data/1.0",
    )
    sources = _mapping(values["sources"], "sources")
    if set(sources) != {"market", "exposure", "mask"}:
        raise CoreValidationError("sources must contain market, exposure, and mask")
    market = _mapping(sources["market"], "sources.market")
    exposure = _source_from_values(
        _mapping(sources["exposure"], "sources.exposure"),
        required_fields=("industry_id", "log_mkt_cap"),
    )
    mask = _source_from_values(
        _mapping(sources["mask"], "sources.mask"),
        required_fields=(
            "universe_mask",
            "eligible_mask",
            "entry_tradable_mask",
            "exit_tradable_mask",
        ),
    )
    return QlibSnapshotSmokeConfig(
        request=request,
        instruments=_string_tuple(values, "instruments", allow_empty=True),
        region=_string(market, "region"),
        market_provider_uri=_string(market, "provider_uri"),
        trading_calendar_version=_string(market, "trading_calendar_version"),
        exposure_source=exposure,
        mask_source=mask,
    )


def run_qlib_snapshot_smoke(config_path: str | Path, storage_root: str | Path) -> str:
    """Load and summarize a real Qlib snapshot without invoking LLMs or evaluation.

    Args:
        config_path: Explicit Qlib smoke configuration.
        storage_root: Local artifact root for the generated snapshot artifacts.

    Returns:
        Canonical JSON summary of the persisted snapshot.
    """
    config = load_qlib_snapshot_smoke_config(config_path)
    artifacts = LocalArtifactRepository(storage_root)
    provider = QlibMarketDataProvider(
        artifacts,
        provider_uri=config.market_provider_uri,
        trading_calendar_version=config.trading_calendar_version,
        exposure_source=config.exposure_source,
        mask_source=config.mask_source,
        instruments=config.instruments or None,
        region=config.region,
    )
    snapshot = provider.load_snapshot(config.request)
    return to_json(
        {
            "date_count": snapshot.market_data.date_count,
            "excluded_sample_count": snapshot.mask_data.excluded_sample_count,
            "instrument_count": snapshot.market_data.instrument_count,
            "sample_count": snapshot.market_data.sample_count,
            "snapshot_id": snapshot.snapshot_id,
            "storage_root": str(storage_root),
        }
    )


def _source_from_values(
    values: dict[str, JsonValue], *, required_fields: tuple[str, ...]
) -> QlibFeatureSource:
    """Decode one strict sidecar source table into a Qlib feature source."""
    expected = {"provider_uri", "fields"}
    if set(values) != expected:
        raise CoreValidationError("Qlib sidecar source has unexpected fields")
    fields = _mapping(values["fields"], "source.fields")
    if tuple(sorted(fields)) != tuple(sorted(required_fields)):
        raise CoreValidationError("Qlib sidecar field mapping is incomplete")
    return QlibFeatureSource(
        provider_uri=_string(values, "provider_uri"),
        fields=tuple(
            QlibFieldMapping(field_name, _string(fields, field_name))
            for field_name in required_fields
        ),
    )


def _mapping(value: JsonValue, field_name: str) -> dict[str, JsonValue]:
    """Require one TOML table represented as a JSON-compatible mapping."""
    if not isinstance(value, dict):
        raise CoreValidationError(f"{field_name} must be a table")
    return value


def _string(values: dict[str, JsonValue], field_name: str) -> str:
    """Read one required non-empty TOML string field."""
    value = values.get(field_name)
    if type(value) is not str or not value:
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(
    values: dict[str, JsonValue], field_name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    """Read one required TOML string array."""
    value = values.get(field_name)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise CoreValidationError(f"{field_name} must be {qualifier}")
    if not all(type(item) is str and item for item in value):
        raise CoreValidationError(f"{field_name} must contain non-empty strings")
    return tuple(value)
