"""Immutable market data contracts shared across providers and evaluators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from cosmosalpha.core import ArtifactRef, CoreValidationError, stable_sha256

MARKET_DATA_SCHEMA_VERSION = "market-data/1.0"
MARKET_DATA_BATCH_SCHEMA_VERSION = "market-data-batch/1.0"
EXPOSURE_DATA_BATCH_SCHEMA_VERSION = "exposure-data-batch/1.0"
UNIVERSE_MASK_BATCH_SCHEMA_VERSION = "universe-mask-batch/1.0"
DATA_QUALITY_REPORT_SCHEMA_VERSION = "data-quality-report/1.0"
DATA_SNAPSHOT_SCHEMA_VERSION = "data-snapshot/1.0"

FIELD_SCHEMA_VERSION = "field-schema/1.0"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_MARKET = "cn"
DEFAULT_FREQUENCY = "1d"
DEFAULT_PRICE_ADJUSTMENT = "post_adjusted"

MARKET_DATA_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "turn",
    "free_turn",
    "returns",
)
EXPOSURE_FIELDS = ("industry_id", "log_mkt_cap", "ret20", "turn20", "vol20")
MASK_FIELDS = (
    "universe_mask",
    "eligible_mask",
    "entry_tradable_mask",
    "exit_tradable_mask",
)

_FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_INSTRUMENT_PATTERN = re.compile(r"^(?:sh|sz|bj)[0-9]{6}$")


def canonical_instrument_code(value: object) -> str:
    """Return the canonical lower-case A-share instrument code.

    Args:
        value: A Qlib or Tushare-derived instrument code.

    Returns:
        A lower-case exchange prefix followed by six digits.

    Raises:
        CoreValidationError: If the value is not a canonicalizable A-share code.
    """
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(
            "instrument must be a non-empty string without surrounding whitespace"
        )
    code = value.lower()
    if _INSTRUMENT_PATTERN.fullmatch(code) is None:
        raise CoreValidationError(
            "instrument must use a two-letter exchange prefix and six digits"
        )
    return code


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _require_date(value: object, field_name: str) -> date:
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


def _require_code(value: object, field_name: str) -> str:
    code = _require_string(value, field_name)
    if _CODE_PATTERN.fullmatch(code) is None:
        raise CoreValidationError(
            f"{field_name} must contain only uppercase letters, digits, and underscores"
        )
    return code


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


def _require_field_tuple(
    value: object,
    field_name: str,
    *,
    allowed_fields: tuple[str, ...],
) -> None:
    _require_string_tuple(value, field_name)
    seen: set[str] = set()
    allowed = set(allowed_fields)
    for index, item in enumerate(value):
        if _FIELD_PATTERN.fullmatch(item) is None:
            raise CoreValidationError(
                f"{field_name}[{index}] must be a lower-case field name"
            )
        if item in seen:
            raise CoreValidationError(f"{field_name} must not contain duplicates")
        if item not in allowed:
            raise CoreValidationError(f"{field_name}[{index}] is not supported")
        seen.add(item)


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CoreValidationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_artifact_ref(value: object, field_name: str) -> None:
    if not isinstance(value, ArtifactRef):
        raise CoreValidationError(f"{field_name} must be ArtifactRef")


def _require_artifact_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_artifact_ref(item, f"{field_name}[{index}]")


def _require_schema(
    value: object, expected: str, field_name: str = "schema_version"
) -> None:
    if value != expected:
        raise CoreValidationError(f"{field_name} must be {expected!r}")


@dataclass(frozen=True, slots=True)
class MarketDataRequest:
    """Request a versioned market data slice."""

    market: str
    universe: str
    start_date: str
    end_date: str
    warmup_start_date: str
    frequency: str
    fields: tuple[str, ...]
    dataset_version: str
    price_adjustment: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate request metadata and supported v1 values."""
        if self.market != DEFAULT_MARKET:
            raise CoreValidationError(f"market must be {DEFAULT_MARKET!r}")
        _require_string(self.universe, "universe")
        start = _require_date(self.start_date, "start_date")
        end = _require_date(self.end_date, "end_date")
        warmup = _require_date(self.warmup_start_date, "warmup_start_date")
        if warmup > start:
            raise CoreValidationError("warmup_start_date must not be after start_date")
        if start > end:
            raise CoreValidationError("start_date must not be after end_date")
        if self.frequency != DEFAULT_FREQUENCY:
            raise CoreValidationError(f"frequency must be {DEFAULT_FREQUENCY!r}")
        if not self.fields:
            raise CoreValidationError("fields must not be empty")
        _require_field_tuple(self.fields, "fields", allowed_fields=MARKET_DATA_FIELDS)
        _require_string(self.dataset_version, "dataset_version")
        if self.price_adjustment != DEFAULT_PRICE_ADJUSTMENT:
            raise CoreValidationError(
                f"price_adjustment must be {DEFAULT_PRICE_ADJUSTMENT!r}"
            )
        _require_schema(self.schema_version, MARKET_DATA_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class MarketDataBatch:
    """Reference a market data artifact and its sample metadata."""

    data_ref: ArtifactRef
    fields: tuple[str, ...]
    sample_count: int
    date_count: int
    instrument_count: int
    source_start_date: str
    source_end_date: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate market data batch metadata."""
        _require_artifact_ref(self.data_ref, "data_ref")
        if not self.fields:
            raise CoreValidationError("fields must not be empty")
        _require_field_tuple(self.fields, "fields", allowed_fields=MARKET_DATA_FIELDS)
        _require_int(self.sample_count, "sample_count")
        _require_int(self.date_count, "date_count")
        _require_int(self.instrument_count, "instrument_count")
        start = _require_date(self.source_start_date, "source_start_date")
        end = _require_date(self.source_end_date, "source_end_date")
        if start > end:
            raise CoreValidationError(
                "source_start_date must not be after source_end_date"
            )
        _require_schema(self.schema_version, MARKET_DATA_BATCH_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class ExposureDataBatch:
    """Reference an exposure artifact and its sample metadata."""

    data_ref: ArtifactRef
    fields: tuple[str, ...]
    sample_count: int
    date_count: int
    instrument_count: int
    schema_version: str

    def __post_init__(self) -> None:
        """Validate exposure batch metadata."""
        _require_artifact_ref(self.data_ref, "data_ref")
        if not self.fields:
            raise CoreValidationError("fields must not be empty")
        _require_field_tuple(self.fields, "fields", allowed_fields=EXPOSURE_FIELDS)
        _require_int(self.sample_count, "sample_count")
        _require_int(self.date_count, "date_count")
        _require_int(self.instrument_count, "instrument_count")
        _require_schema(self.schema_version, EXPOSURE_DATA_BATCH_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class UniverseMaskBatch:
    """Reference universe, eligibility, and tradability mask artifacts."""

    mask_data_ref: ArtifactRef
    reason_data_ref: ArtifactRef | None
    fields: tuple[str, ...]
    sample_count: int
    excluded_sample_count: int
    unknown_sample_count: int
    schema_version: str

    def __post_init__(self) -> None:
        """Validate mask batch metadata."""
        _require_artifact_ref(self.mask_data_ref, "mask_data_ref")
        if self.reason_data_ref is not None:
            _require_artifact_ref(self.reason_data_ref, "reason_data_ref")
        if not self.fields:
            raise CoreValidationError("fields must not be empty")
        _require_field_tuple(self.fields, "fields", allowed_fields=MASK_FIELDS)
        _require_int(self.sample_count, "sample_count")
        _require_int(self.excluded_sample_count, "excluded_sample_count")
        _require_int(self.unknown_sample_count, "unknown_sample_count")
        _require_schema(self.schema_version, UNIVERSE_MASK_BATCH_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    """Describe one machine-readable market data quality issue."""

    code: str
    message: str
    severity: str
    sample_count: int

    def __post_init__(self) -> None:
        """Validate data quality issue fields."""
        _require_code(self.code, "code")
        _require_string(self.message, "message")
        if self.severity not in {"warning", "error"}:
            raise CoreValidationError("severity must be 'warning' or 'error'")
        _require_int(self.sample_count, "sample_count")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Summarize data quality checks for one snapshot."""

    issues: tuple[DataQualityIssue, ...]
    nonfinite_value_count: int
    invalid_price_count: int
    invalid_ohlc_count: int
    schema_version: str

    def __post_init__(self) -> None:
        """Validate data quality report fields."""
        if not isinstance(self.issues, tuple):
            raise CoreValidationError("issues must be a tuple")
        if not all(isinstance(issue, DataQualityIssue) for issue in self.issues):
            raise CoreValidationError("every issue must be DataQualityIssue")
        _require_int(self.nonfinite_value_count, "nonfinite_value_count")
        _require_int(self.invalid_price_count, "invalid_price_count")
        _require_int(self.invalid_ohlc_count, "invalid_ohlc_count")
        _require_schema(self.schema_version, DATA_QUALITY_REPORT_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """Versioned collection of market, exposure, mask, and quality artifacts."""

    snapshot_id: str
    request: MarketDataRequest
    market_data: MarketDataBatch
    exposure_data: ExposureDataBatch
    mask_data: UniverseMaskBatch
    quality_report: DataQualityReport
    provider_alias: str
    provider_uri_fingerprint: str
    dataset_version: str
    trading_calendar_version: str
    field_schema_version: str
    market: str
    timezone: str
    frequency: str
    price_adjustment: str
    source_date_range: tuple[str, str]
    artifact_refs: tuple[ArtifactRef, ...]
    schema_version: str

    def __post_init__(self) -> None:
        """Validate snapshot metadata and stable identifier."""
        if not isinstance(self.request, MarketDataRequest):
            raise CoreValidationError("request must be MarketDataRequest")
        if not isinstance(self.market_data, MarketDataBatch):
            raise CoreValidationError("market_data must be MarketDataBatch")
        if not isinstance(self.exposure_data, ExposureDataBatch):
            raise CoreValidationError("exposure_data must be ExposureDataBatch")
        if not isinstance(self.mask_data, UniverseMaskBatch):
            raise CoreValidationError("mask_data must be UniverseMaskBatch")
        if not isinstance(self.quality_report, DataQualityReport):
            raise CoreValidationError("quality_report must be DataQualityReport")
        _require_string(self.snapshot_id, "snapshot_id")
        _require_string(self.provider_alias, "provider_alias")
        _require_string(self.provider_uri_fingerprint, "provider_uri_fingerprint")
        _require_string(self.dataset_version, "dataset_version")
        _require_string(self.trading_calendar_version, "trading_calendar_version")
        _require_string(self.field_schema_version, "field_schema_version")
        _require_schema(
            self.field_schema_version, FIELD_SCHEMA_VERSION, "field_schema_version"
        )
        if self.market != self.request.market:
            raise CoreValidationError("market must match request.market")
        if self.frequency != self.request.frequency:
            raise CoreValidationError("frequency must match request.frequency")
        if self.price_adjustment != self.request.price_adjustment:
            raise CoreValidationError(
                "price_adjustment must match request.price_adjustment"
            )
        if self.dataset_version != self.request.dataset_version:
            raise CoreValidationError(
                "dataset_version must match request.dataset_version"
            )
        if self.timezone != DEFAULT_TIMEZONE:
            raise CoreValidationError(f"timezone must be {DEFAULT_TIMEZONE!r}")
        if len(self.source_date_range) != 2:
            raise CoreValidationError("source_date_range must contain two dates")
        source_start = _require_date(self.source_date_range[0], "source_date_range[0]")
        source_end = _require_date(self.source_date_range[1], "source_date_range[1]")
        if source_start > source_end:
            raise CoreValidationError("source_date_range start must not be after end")
        _require_artifact_tuple(self.artifact_refs, "artifact_refs")
        expected_refs = (
            self.market_data.data_ref,
            self.exposure_data.data_ref,
            self.mask_data.mask_data_ref,
            *(
                ()
                if self.mask_data.reason_data_ref is None
                else (self.mask_data.reason_data_ref,)
            ),
        )
        for ref in expected_refs:
            if ref not in self.artifact_refs:
                raise CoreValidationError(
                    "artifact_refs must include snapshot artifacts"
                )
        if self.market_data.sample_count != self.exposure_data.sample_count:
            raise CoreValidationError("market and exposure sample counts must match")
        if self.market_data.sample_count != self.mask_data.sample_count:
            raise CoreValidationError("market and mask sample counts must match")
        _require_schema(self.schema_version, DATA_SNAPSHOT_SCHEMA_VERSION)
        expected_id = compute_data_snapshot_id(
            request=self.request,
            market_data=self.market_data,
            exposure_data=self.exposure_data,
            mask_data=self.mask_data,
            provider_alias=self.provider_alias,
            provider_uri_fingerprint=self.provider_uri_fingerprint,
            trading_calendar_version=self.trading_calendar_version,
            field_schema_version=self.field_schema_version,
            source_date_range=self.source_date_range,
        )
        if self.snapshot_id != expected_id:
            raise CoreValidationError("snapshot_id does not match snapshot content")


def compute_data_snapshot_id(
    *,
    request: MarketDataRequest,
    market_data: MarketDataBatch,
    exposure_data: ExposureDataBatch,
    mask_data: UniverseMaskBatch,
    provider_alias: str,
    provider_uri_fingerprint: str,
    trading_calendar_version: str,
    field_schema_version: str,
    source_date_range: tuple[str, str],
) -> str:
    """Compute a path-independent stable snapshot identifier."""
    return stable_sha256(
        {
            "schema_version": DATA_SNAPSHOT_SCHEMA_VERSION,
            "request": {
                "market": request.market,
                "universe": request.universe,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "warmup_start_date": request.warmup_start_date,
                "frequency": request.frequency,
                "fields": sorted(request.fields),
                "dataset_version": request.dataset_version,
                "price_adjustment": request.price_adjustment,
                "schema_version": request.schema_version,
            },
            "provider_alias": provider_alias,
            "provider_uri_fingerprint": provider_uri_fingerprint,
            "trading_calendar_version": trading_calendar_version,
            "field_schema_version": field_schema_version,
            "source_date_range": tuple(source_date_range),
            "artifacts": {
                "market_data": market_data.data_ref.checksum,
                "exposure_data": exposure_data.data_ref.checksum,
                "mask_data": mask_data.mask_data_ref.checksum,
                "mask_reasons": (
                    None
                    if mask_data.reason_data_ref is None
                    else mask_data.reason_data_ref.checksum
                ),
            },
        }
    )


def empty_quality_report() -> DataQualityReport:
    """Return an empty data quality report for deterministic samples."""
    return DataQualityReport(
        issues=(),
        nonfinite_value_count=0,
        invalid_price_count=0,
        invalid_ohlc_count=0,
        schema_version=DATA_QUALITY_REPORT_SCHEMA_VERSION,
    )
