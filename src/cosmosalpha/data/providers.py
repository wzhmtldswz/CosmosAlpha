"""Market data providers and provider-side validation helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

from cosmosalpha.core import (
    COLUMNAR_RECORDS_MEDIA_TYPE,
    ArtifactRef,
    CoreValidationError,
    JsonValue,
    encode_columnar_records,
)
from cosmosalpha.data.errors import DataProviderError, DataValidationError
from cosmosalpha.data.models import (
    DATA_SNAPSHOT_SCHEMA_VERSION,
    DEFAULT_TIMEZONE,
    EXPOSURE_DATA_BATCH_SCHEMA_VERSION,
    EXPOSURE_FIELDS,
    FIELD_SCHEMA_VERSION,
    MARKET_DATA_BATCH_SCHEMA_VERSION,
    MARKET_DATA_FIELDS,
    MASK_FIELDS,
    UNIVERSE_MASK_BATCH_SCHEMA_VERSION,
    DataQualityIssue,
    DataSnapshot,
    ExposureDataBatch,
    MarketDataBatch,
    MarketDataRequest,
    UniverseMaskBatch,
    canonical_instrument_code,
    compute_data_snapshot_id,
    empty_quality_report,
)
from cosmosalpha.data.qlib_industry import inspect_industry_provider

QLIB_FIELD_MAP = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
    "vwap": "$vwap",
    "turn": "$turn",
    "free_turn": "$free_turn",
    "returns": "$returns",
}

_DERIVED_EXPOSURE_MARKET_FIELDS = ("close", "turn", "returns")


class ArtifactRepository(Protocol):
    """Minimal artifact repository interface required by data providers."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist bytes and return an artifact reference."""


type SampleRow = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class QlibFieldMapping:
    """Map one configured Qlib expression to a CosmosAlpha field."""

    field_name: str
    expression: str

    def __post_init__(self) -> None:
        """Validate a non-empty infrastructure field mapping."""
        if type(self.field_name) is not str or not self.field_name:
            raise CoreValidationError("field_name must be a non-empty string")
        if type(self.expression) is not str or not self.expression:
            raise CoreValidationError("expression must be a non-empty string")


@dataclass(frozen=True, slots=True)
class QlibFeatureSource:
    """Qlib source configuration for one aligned sidecar dataset."""

    provider_uri: str
    fields: tuple[QlibFieldMapping, ...]

    def __post_init__(self) -> None:
        """Validate source identity and unique output field mappings."""
        if type(self.provider_uri) is not str or not self.provider_uri:
            raise CoreValidationError("provider_uri must be a non-empty string")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise CoreValidationError("fields must be a non-empty tuple")
        if not all(isinstance(field, QlibFieldMapping) for field in self.fields):
            raise CoreValidationError("fields must contain QlibFieldMapping values")
        names = tuple(field.field_name for field in self.fields)
        if len(names) != len(set(names)):
            raise CoreValidationError("source fields must not contain duplicates")


class QlibFrameLoader(Protocol):
    """Load one Qlib feature frame from an explicitly configured source."""

    def load(
        self,
        source: QlibFeatureSource,
        *,
        region: str,
        instruments: object,
        start_time: str,
        end_time: str,
    ) -> object:
        """Return a pandas-compatible feature frame for the configured source."""


def default_market_data_request() -> MarketDataRequest:
    """Return the deterministic request used by the fixed sample provider."""
    return MarketDataRequest(
        market="cn",
        universe="all_a_share",
        start_date="2020-01-02",
        end_date="2020-01-04",
        warmup_start_date="2020-01-01",
        frequency="1d",
        fields=MARKET_DATA_FIELDS,
        dataset_version="fixed-sample-v1",
        price_adjustment="post_adjusted",
        schema_version="market-data/1.0",
    )


class FixedSampleMarketDataProvider:
    """Provider that returns a tiny deterministic daily A-share sample."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        provider_alias: str = "fixed_sample",
        provider_uri_fingerprint: str = "fixed-sample",
        trading_calendar_version: str = "fixed-calendar/1.0",
        instruments: tuple[str, ...] = ("sh600000", "sz000001"),
    ) -> None:
        """Create a deterministic fixed sample provider."""
        if not instruments or not all(
            type(instrument) is str and instrument for instrument in instruments
        ):
            raise CoreValidationError(
                "instruments must be a non-empty tuple of strings"
            )
        self._artifacts = artifact_repository
        self._provider_alias = provider_alias
        self._provider_uri_fingerprint = provider_uri_fingerprint
        self._trading_calendar_version = trading_calendar_version
        self._instruments = tuple(sorted(instruments))

    def load_snapshot(self, request: MarketDataRequest | None = None) -> DataSnapshot:
        """Generate and persist a deterministic fixed sample snapshot."""
        request = request or default_market_data_request()
        records = _filter_records(_fixed_market_records(self._instruments), request)
        if not records:
            raise DataProviderError("EMPTY_DATASET: fixed sample returned no rows")
        market_records = _select_market_fields(records, request.fields)
        exposure_records = _fixed_exposure_records(records)
        mask_records = _fixed_mask_records(records)
        return _build_snapshot(
            artifact_repository=self._artifacts,
            request=request,
            market_records=market_records,
            exposure_records=exposure_records,
            mask_records=mask_records,
            provider_alias=self._provider_alias,
            provider_uri_fingerprint=self._provider_uri_fingerprint,
            trading_calendar_version=self._trading_calendar_version,
            source_date_range=(request.warmup_start_date, request.end_date),
        )


class QlibMarketDataProvider:
    """Qlib adapter that requires explicit exposure and Mask sidecar sources."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        provider_uri: str,
        trading_calendar_version: str,
        exposure_source: QlibFeatureSource | None = None,
        mask_source: QlibFeatureSource | None = None,
        instruments: tuple[str, ...] | None = None,
        region: str = "cn",
        data_api: object | None = None,
        qlib_initializer: Callable[[str, str], None] | None = None,
        frame_loader: QlibFrameLoader | None = None,
    ) -> None:
        """Create a Qlib provider with required point-in-time sidecar sources."""
        self._artifacts = artifact_repository
        self._provider_uri = provider_uri
        self._trading_calendar_version = trading_calendar_version
        self._region = region
        self._data_api = data_api
        self._qlib_initializer = qlib_initializer
        self._exposure_source = exposure_source
        self._mask_source = mask_source
        if instruments:
            if not all(
                type(instrument) is str and instrument for instrument in instruments
            ):
                raise CoreValidationError("instruments must contain non-empty strings")
            try:
                normalized_instruments = tuple(
                    canonical_instrument_code(instrument) for instrument in instruments
                )
            except CoreValidationError as error:
                raise CoreValidationError(
                    "instruments must contain canonicalizable A-share codes"
                ) from error
            if len(normalized_instruments) != len(set(normalized_instruments)):
                raise CoreValidationError(
                    "instruments must not contain case-insensitive duplicates"
                )
            self._instruments = normalized_instruments
        else:
            self._instruments = None
        self._frame_loader = frame_loader

    def load_snapshot(self, request: MarketDataRequest) -> DataSnapshot:
        """Load market data from Qlib and persist CosmosAlpha artifacts."""
        exposure_source = self._require_source(
            self._exposure_source, "EXPOSURE_SOURCE_NOT_CONFIGURED"
        )
        mask_source = self._require_source(
            self._mask_source, "MASK_SOURCE_NOT_CONFIGURED"
        )
        try:
            industry_manifest = inspect_industry_provider(exposure_source.provider_uri)
        except ValueError as error:
            raise DataProviderError(str(error)) from error
        source_issues = ()
        if (
            industry_manifest is not None
            and not industry_manifest["point_in_time_certified"]
        ):
            source_issues = (
                DataQualityIssue(
                    code="INDUSTRY_NOT_PIT_CERTIFIED",
                    message=(
                        f"Industry source {industry_manifest.get('classification')} "
                        f"({industry_manifest.get('industry_mode', 'historical_reconstruction')}) "
                        "is not certified as historically available; coverage is not a PIT guarantee."
                    ),
                    severity="warning",
                    sample_count=0,
                ),
            )
        required_fields = tuple(
            dict.fromkeys((*request.fields, *_DERIVED_EXPOSURE_MARKET_FIELDS))
        )
        market_source = QlibFeatureSource(
            provider_uri=self._provider_uri,
            fields=tuple(
                QlibFieldMapping(field, QLIB_FIELD_MAP[field])
                for field in required_fields
            ),
        )
        instruments = self._resolve_instruments(request)
        market_frame = self._load_frame(market_source, instruments, request)
        source_market_records = _frame_to_source_records(market_frame, market_source)
        _validate_market_records(source_market_records, request.fields)
        mask_frame = self._load_frame(mask_source, instruments, request)
        source_mask_records = _frame_to_source_records(mask_frame, mask_source)
        mask_records = _build_qlib_mask_records(
            source_market_records, source_mask_records
        )
        exposure_frame = self._load_frame(exposure_source, instruments, request)
        source_exposure_records = _frame_to_source_records(
            exposure_frame, exposure_source
        )
        exposure_records = _build_qlib_exposure_records(
            source_market_records, source_exposure_records
        )
        market_records = _select_market_fields(source_market_records, request.fields)
        reason_records = _mask_reason_records(mask_records)
        return _build_snapshot(
            artifact_repository=self._artifacts,
            request=request,
            market_records=market_records,
            exposure_records=exposure_records,
            mask_records=mask_records,
            provider_alias="qlib",
            provider_uri_fingerprint=request.dataset_version,
            trading_calendar_version=self._trading_calendar_version,
            source_date_range=(request.warmup_start_date, request.end_date),
            reason_records=reason_records,
            source_issues=source_issues,
        )

    def _require_source(
        self, source: QlibFeatureSource | None, error_code: str
    ) -> QlibFeatureSource:
        """Return one required sidecar source or raise a stable provider error."""
        if source is None:
            raise DataProviderError(error_code)
        return source

    def _resolve_instruments(self, request: MarketDataRequest) -> object:
        if self._instruments is not None:
            return self._instruments
        data_api = self._data_api
        if data_api is None:
            data_api, initializer = _load_qlib_backend()
            initializer(os.path.expanduser(self._provider_uri), self._region)
        return _resolve_qlib_instruments(data_api, request)

    def _load_frame(
        self,
        source: QlibFeatureSource,
        instruments: object,
        request: MarketDataRequest,
    ) -> object:
        if self._frame_loader is not None:
            return self._frame_loader.load(
                source,
                region=self._region,
                instruments=instruments,
                start_time=request.warmup_start_date,
                end_time=request.end_date,
            )
        data_api = self._data_api
        if data_api is None:
            return _QlibRuntimeFrameLoader().load(
                source,
                region=self._region,
                instruments=instruments,
                start_time=request.warmup_start_date,
                end_time=request.end_date,
            )
        if self._qlib_initializer is not None:
            self._qlib_initializer(
                os.path.expanduser(source.provider_uri), self._region
            )
        return data_api.features(
            instruments=instruments,
            fields=[field.expression for field in source.fields],
            start_time=request.warmup_start_date,
            end_time=request.end_date,
            freq="day",
        )


def _load_qlib_backend() -> tuple[object, Callable[[str, str], None]]:
    try:
        import qlib
        from qlib.data import D
    except ModuleNotFoundError as error:
        raise DataProviderError(
            "Qlib provider requires qlib to be installed"
        ) from error

    def _initialize(provider_uri: str, region: str) -> None:
        qlib.init(provider_uri=provider_uri, region=region)

    return D, _initialize


class _QlibRuntimeFrameLoader:
    """Load source frames through Qlib only when a real provider is requested."""

    def load(
        self,
        source: QlibFeatureSource,
        *,
        region: str,
        instruments: object,
        start_time: str,
        end_time: str,
    ) -> object:
        """Initialize the configured source and return its requested feature frame."""
        data_api, initializer = _load_qlib_backend()
        initializer(os.path.expanduser(source.provider_uri), region)
        return data_api.features(
            instruments=instruments,
            fields=[field.expression for field in source.fields],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )


def _build_snapshot(
    *,
    artifact_repository: ArtifactRepository,
    request: MarketDataRequest,
    market_records: tuple[SampleRow, ...],
    exposure_records: tuple[SampleRow, ...],
    mask_records: tuple[SampleRow, ...],
    provider_alias: str,
    provider_uri_fingerprint: str,
    trading_calendar_version: str,
    source_date_range: tuple[str, str],
    reason_records: tuple[SampleRow, ...] = (),
    source_issues: tuple[DataQualityIssue, ...] = (),
) -> DataSnapshot:
    if not market_records:
        raise DataProviderError("EMPTY_DATASET: provider returned no rows")
    _validate_alignment(market_records, exposure_records, "EXPOSURE_INDEX_MISMATCH")
    _validate_alignment(market_records, mask_records, "MASK_INDEX_MISMATCH")
    _validate_market_records(market_records, request.fields)
    missing_industry_count = _validate_exposure_records(exposure_records, mask_records)
    _validate_mask_records(mask_records)
    quality_report = replace(empty_quality_report(), issues=source_issues)
    if missing_industry_count:
        quality_report = replace(
            quality_report,
            issues=(
                *source_issues,
                DataQualityIssue(
                    code="MISSING_INDUSTRY_VALUE",
                    message=(
                        "Unknown industry is retained as null; industry-dependent "
                        "operators and neutralization exclude these samples."
                    ),
                    severity="warning",
                    sample_count=missing_industry_count,
                ),
            ),
        )

    market_ref = _put_records(
        artifact_repository,
        market_records,
        kind="market_data",
    )
    exposure_ref = _put_records(
        artifact_repository,
        exposure_records,
        kind="exposure_data",
    )
    mask_ref = _put_records(artifact_repository, mask_records, kind="mask_data")
    reason_ref = (
        _put_records(artifact_repository, reason_records, kind="mask_reasons")
        if reason_records
        else None
    )
    date_count, instrument_count = _count_index_levels(market_records)
    market_batch = MarketDataBatch(
        data_ref=market_ref,
        fields=tuple(sorted(request.fields)),
        sample_count=len(market_records),
        date_count=date_count,
        instrument_count=instrument_count,
        source_start_date=source_date_range[0],
        source_end_date=source_date_range[1],
        schema_version=MARKET_DATA_BATCH_SCHEMA_VERSION,
    )
    exposure_batch = ExposureDataBatch(
        data_ref=exposure_ref,
        fields=EXPOSURE_FIELDS,
        sample_count=len(exposure_records),
        date_count=date_count,
        instrument_count=instrument_count,
        schema_version=EXPOSURE_DATA_BATCH_SCHEMA_VERSION,
    )
    mask_batch = UniverseMaskBatch(
        mask_data_ref=mask_ref,
        reason_data_ref=reason_ref,
        fields=MASK_FIELDS,
        sample_count=len(mask_records),
        excluded_sample_count=_count_excluded_masks(mask_records),
        unknown_sample_count=_count_unknown_masks(mask_records),
        schema_version=UNIVERSE_MASK_BATCH_SCHEMA_VERSION,
    )
    snapshot_id = compute_data_snapshot_id(
        request=request,
        market_data=market_batch,
        exposure_data=exposure_batch,
        mask_data=mask_batch,
        provider_alias=provider_alias,
        provider_uri_fingerprint=provider_uri_fingerprint,
        trading_calendar_version=trading_calendar_version,
        field_schema_version=FIELD_SCHEMA_VERSION,
        source_date_range=source_date_range,
    )
    return DataSnapshot(
        snapshot_id=snapshot_id,
        request=request,
        market_data=market_batch,
        exposure_data=exposure_batch,
        mask_data=mask_batch,
        quality_report=quality_report,
        provider_alias=provider_alias,
        provider_uri_fingerprint=provider_uri_fingerprint,
        dataset_version=request.dataset_version,
        trading_calendar_version=trading_calendar_version,
        field_schema_version=FIELD_SCHEMA_VERSION,
        market=request.market,
        timezone=DEFAULT_TIMEZONE,
        frequency=request.frequency,
        price_adjustment=request.price_adjustment,
        source_date_range=source_date_range,
        artifact_refs=tuple(
            ref for ref in (market_ref, exposure_ref, mask_ref, reason_ref) if ref
        ),
        schema_version=DATA_SNAPSHOT_SCHEMA_VERSION,
    )


def _put_records(
    artifact_repository: ArtifactRepository,
    records: tuple[SampleRow, ...],
    *,
    kind: str,
) -> ArtifactRef:
    return artifact_repository.put_bytes(
        encode_columnar_records(records),
        kind=kind,
        media_type=COLUMNAR_RECORDS_MEDIA_TYPE,
    )


def _fixed_market_records(instruments: tuple[str, ...]) -> tuple[SampleRow, ...]:
    rows: list[SampleRow] = []
    dates = ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04")
    for date_offset, current_date in enumerate(dates):
        for inst_offset, instrument in enumerate(instruments):
            base = 10.0 + date_offset * 0.35 + inst_offset * 0.7
            close = base + ((inst_offset + date_offset) % 3) * 0.09
            rows.append(
                {
                    "datetime": current_date,
                    "instrument": instrument,
                    "open": base,
                    "high": close + 0.25,
                    "low": base - 0.25,
                    "close": close,
                    "volume": 1000.0
                    + inst_offset * 37.0
                    + date_offset * 53.0
                    + ((inst_offset + 2 * date_offset) % 5) * 11.0,
                    "vwap": base + 0.1,
                    "turn": 0.01
                    + date_offset * 0.001
                    + inst_offset * 0.002
                    + ((inst_offset + date_offset) % 3) * 0.0005,
                    "free_turn": 0.02
                    + date_offset * 0.001
                    + inst_offset * 0.002
                    + ((inst_offset + 2 * date_offset) % 4) * 0.0005,
                    "returns": (
                        0.0
                        if date_offset == 0
                        else (inst_offset - (len(instruments) - 1) / 2.0) * 0.002
                        + date_offset * 0.001
                    ),
                }
            )
    return tuple(rows)


def _fixed_exposure_records(
    market_records: Iterable[SampleRow],
) -> tuple[SampleRow, ...]:
    records = tuple(market_records)
    instrument_count = len({str(row["instrument"]) for row in records})
    rows: list[SampleRow] = []
    for index, row in enumerate(records):
        instrument = cast(str, row["instrument"])
        close = cast(float, row["close"])
        inst_offset = index % instrument_count
        date_offset = index // instrument_count
        rows.append(
            {
                "datetime": row["datetime"],
                "instrument": instrument,
                "industry_id": 1 if instrument.startswith("sh") else 2,
                "log_mkt_cap": math.log(close * 1_000_000.0),
                "ret20": cast(float, row["returns"]),
                "turn20": cast(float, row["turn"]),
                "vol20": 0.03 + ((inst_offset + date_offset * 2) % 5) * 0.004,
            }
        )
    return tuple(rows)


def _fixed_mask_records(market_records: Iterable[SampleRow]) -> tuple[SampleRow, ...]:
    return tuple(
        {
            "datetime": row["datetime"],
            "instrument": row["instrument"],
            "universe_mask": True,
            "eligible_mask": True,
            "entry_tradable_mask": True,
            "exit_tradable_mask": True,
        }
        for row in market_records
    )


def _filter_records(
    records: Iterable[SampleRow],
    request: MarketDataRequest,
) -> tuple[SampleRow, ...]:
    return tuple(
        row
        for row in records
        if request.warmup_start_date <= cast(str, row["datetime"]) <= request.end_date
    )


def _select_market_fields(
    records: Iterable[SampleRow],
    fields: tuple[str, ...],
) -> tuple[SampleRow, ...]:
    sorted_fields = tuple(sorted(fields))
    return tuple(
        {
            "datetime": row["datetime"],
            "instrument": row["instrument"],
            **{field: row[field] for field in sorted_fields},
        }
        for row in records
    )


def _resolve_qlib_instruments(data_api: object, request: MarketDataRequest) -> object:
    instruments = getattr(data_api, "instruments", None)
    if callable(instruments):
        return instruments(market=request.universe)
    return request.universe


def _frame_to_source_records(
    frame: object,
    source: QlibFeatureSource,
) -> tuple[SampleRow, ...]:
    """Convert one Qlib frame to records using its explicit source mapping."""
    frame = _normalize_frame_index(frame)
    qlib_fields = tuple(field.expression for field in source.fields)
    output_fields = tuple(field.field_name for field in source.fields)
    columns = tuple(str(column) for column in getattr(frame, "columns", ()))
    if columns != qlib_fields:
        raise DataProviderError(
            "MISSING_REQUIRED_FIELD: Qlib returned columns do not match request"
        )
    index = getattr(frame, "index", None)
    _validate_frame_index(index)
    split = frame.to_dict(orient="split")
    index_values = split["index"]
    data_values = split["data"]
    records: list[SampleRow] = []
    for index_item, values in zip(index_values, data_values, strict=True):
        dt_value, instrument = _split_index_item(index_item)
        try:
            canonical_instrument = canonical_instrument_code(instrument)
        except CoreValidationError as error:
            raise DataProviderError(
                "INVALID_INDEX_SCHEMA: instrument is not canonicalizable"
            ) from error
        record: SampleRow = {
            "datetime": _format_date_value(dt_value),
            "instrument": canonical_instrument,
        }
        for field, value in zip(output_fields, values, strict=True):
            record[field] = _json_number(value)
        records.append(record)
    return tuple(records)


def _normalize_frame_index(frame: object) -> object:
    """Normalize Qlib's native instrument/datetime index to project order."""
    index = getattr(frame, "index", None)
    names = tuple(getattr(index, "names", ()))
    if names == ("instrument", "datetime"):
        reorder_levels = getattr(frame, "reorder_levels", None)
        if not callable(reorder_levels):
            raise DataProviderError(
                "INVALID_INDEX_SCHEMA: Qlib frame cannot reorder its index"
            )
        try:
            normalized = reorder_levels(["datetime", "instrument"])
            return normalized.sort_index()
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise DataProviderError(
                "INVALID_INDEX_SCHEMA: Qlib frame index normalization failed"
            ) from error
    return frame


def _validate_frame_index(index: object) -> None:
    if index is None:
        raise DataProviderError("INVALID_INDEX_SCHEMA: Qlib frame has no index")
    if tuple(getattr(index, "names", ())) != ("datetime", "instrument"):
        raise DataProviderError(
            "INVALID_INDEX_SCHEMA: expected datetime/instrument index"
        )
    if getattr(index, "is_unique", True) is False:
        raise DataProviderError("DUPLICATE_SAMPLE_INDEX: duplicate sample index")
    if getattr(index, "is_monotonic_increasing", True) is False:
        raise DataProviderError("UNSORTED_SAMPLE_INDEX: sample index is not sorted")


def _split_index_item(index_item: object) -> tuple[object, object]:
    if isinstance(index_item, (list, tuple)) and len(index_item) == 2:
        return index_item[0], index_item[1]
    raise DataProviderError("INVALID_INDEX_SCHEMA: sample index items must be pairs")


def _format_date_value(value: object) -> str:
    if hasattr(value, "date") and callable(value.date):
        value = value.date()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return str(value.isoformat())
    raw = str(value)
    return raw[:10] if len(raw) >= 10 else raw


def _json_number(value: object) -> JsonValue:
    if value is None:
        return None
    if type(value) in (int, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    return cast(JsonValue, value)


def _validate_market_records(
    records: tuple[SampleRow, ...],
    fields: tuple[str, ...],
) -> None:
    seen: set[tuple[str, str]] = set()
    previous_key: tuple[str, str] | None = None
    for row in records:
        key = _row_key(row)
        if key in seen:
            raise DataValidationError("DUPLICATE_SAMPLE_INDEX: duplicate sample index")
        if previous_key is not None and key < previous_key:
            raise DataValidationError(
                "UNSORTED_SAMPLE_INDEX: sample index is not sorted"
            )
        seen.add(key)
        previous_key = key
        for field in fields:
            if field not in row:
                raise DataValidationError(f"MISSING_REQUIRED_FIELD: {field}")
            value = row[field]
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise DataValidationError(f"NONFINITE_VALUE: {field}")
        _validate_prices(row)


def _validate_prices(row: SampleRow) -> None:
    for field in ("open", "high", "low", "close"):
        if field in row and row[field] is not None and cast(float, row[field]) <= 0:
            raise DataValidationError(f"INVALID_PRICE: {field}")
    if "volume" in row and row["volume"] is not None and cast(float, row["volume"]) < 0:
        raise DataValidationError("INVALID_PRICE: volume")
    required = {"open", "high", "low", "close"}
    if not required.issubset(row):
        return
    if any(row[field] is None for field in required):
        return
    open_value = cast(float, row["open"])
    high_value = cast(float, row["high"])
    low_value = cast(float, row["low"])
    close_value = cast(float, row["close"])
    if high_value < max(open_value, close_value, low_value):
        raise DataValidationError("INVALID_OHLC_RELATION: high is too low")
    if low_value > min(open_value, close_value, high_value):
        raise DataValidationError("INVALID_OHLC_RELATION: low is too high")


def _build_qlib_exposure_records(
    market_records: tuple[SampleRow, ...],
    source_records: tuple[SampleRow, ...],
) -> tuple[SampleRow, ...]:
    """Join source exposures with backward-looking market derivatives."""
    _validate_alignment(market_records, source_records, "EXPOSURE_INDEX_MISMATCH")
    _require_source_fields(source_records, ("industry_id", "log_mkt_cap"), "exposure")
    derived = _derived_exposures(market_records)
    rows: list[SampleRow] = []
    for market, source, values in zip(
        market_records, source_records, derived, strict=True
    ):
        industry = source["industry_id"]
        log_mkt_cap = source["log_mkt_cap"]
        rows.append(
            {
                "datetime": market["datetime"],
                "instrument": market["instrument"],
                "industry_id": industry,
                "log_mkt_cap": log_mkt_cap,
                **values,
            }
        )
    return tuple(rows)


def _derived_exposures(
    market_records: tuple[SampleRow, ...],
) -> tuple[dict[str, JsonValue], ...]:
    """Compute point-in-time 20-day exposure features by instrument."""
    output: list[dict[str, JsonValue]] = [
        {"ret20": None, "turn20": None, "vol20": None} for _ in market_records
    ]
    by_instrument: dict[str, list[int]] = {}
    for index, row in enumerate(market_records):
        by_instrument.setdefault(cast(str, row["instrument"]), []).append(index)
    for indices in by_instrument.values():
        closes = [_finite_float(market_records[index]["close"]) for index in indices]
        turns = [_finite_float(market_records[index]["turn"]) for index in indices]
        returns = [_finite_float(market_records[index]["returns"]) for index in indices]
        for position, record_index in enumerate(indices):
            start = max(0, position - 19)
            turn_window = [
                value for value in turns[start : position + 1] if value is not None
            ]
            return_window = [
                value for value in returns[start : position + 1] if value is not None
            ]
            if len(turn_window) >= 14:
                output[record_index]["turn20"] = float(
                    sum(turn_window) / len(turn_window)
                )
            if len(return_window) >= 14:
                mean = sum(return_window) / len(return_window)
                variance = sum((value - mean) ** 2 for value in return_window) / (
                    len(return_window) - 1
                )
                output[record_index]["vol20"] = float(math.sqrt(variance))
            if position >= 20:
                current = closes[position]
                baseline = closes[position - 20]
                if current is not None and baseline is not None and baseline > 0.0:
                    output[record_index]["ret20"] = current / baseline - 1.0
    return tuple(output)


def _build_qlib_mask_records(
    market_records: tuple[SampleRow, ...],
    source_records: tuple[SampleRow, ...],
) -> tuple[SampleRow, ...]:
    """Join configured eligibility and tradability Masks without defaults."""
    _validate_alignment(market_records, source_records, "MASK_INDEX_MISMATCH")
    _require_source_fields(source_records, MASK_FIELDS, "mask")
    rows: list[SampleRow] = []
    for market, source in zip(market_records, source_records, strict=True):
        rows.append(
            {
                "datetime": market["datetime"],
                "instrument": market["instrument"],
                **{field: _mask_value(source[field], field) for field in MASK_FIELDS},
            }
        )
    return tuple(rows)


def _mask_reason_records(mask_records: tuple[SampleRow, ...]) -> tuple[SampleRow, ...]:
    """Persist generic source-state reasons for excluded and unknown Masks."""
    reasons: list[SampleRow] = []
    for row in mask_records:
        for field in MASK_FIELDS:
            value = row[field]
            if value is True:
                continue
            reasons.append(
                {
                    "datetime": row["datetime"],
                    "instrument": row["instrument"],
                    "reason_code": (
                        f"{field.upper()}_UNKNOWN"
                        if value is None
                        else f"{field.upper()}_FALSE"
                    ),
                }
            )
    return tuple(reasons)


def _require_source_fields(
    records: tuple[SampleRow, ...], fields: tuple[str, ...], source_name: str
) -> None:
    if not records:
        raise DataProviderError(f"EMPTY_{source_name.upper()}_SOURCE")
    missing = sorted(
        field for field in fields if any(field not in record for record in records)
    )
    if missing:
        raise DataProviderError(
            f"MISSING_{source_name.upper()}_FIELD: " + ", ".join(missing)
        )


def _finite_float(value: JsonValue) -> float | None:
    if type(value) not in (int, float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mask_value(value: JsonValue, field_name: str) -> bool | None:
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        numeric = float(value)
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
    raise DataProviderError(f"INVALID_MASK_VALUE: {field_name}")


def _validate_exposure_records(
    records: tuple[SampleRow, ...], mask_records: tuple[SampleRow, ...]
) -> int:
    _require_source_fields(records, EXPOSURE_FIELDS, "exposure")
    _validate_alignment(records, mask_records, "MASK_INDEX_MISMATCH")
    missing_industry_count = 0
    for row, mask in zip(records, mask_records, strict=True):
        if mask["eligible_mask"] is False:
            continue
        if _finite_float(row["log_mkt_cap"]) is None:
            raise DataProviderError("MISSING_EXPOSURE_VALUE: log_mkt_cap")
        industry = _finite_float(row["industry_id"])
        if industry is None:
            missing_industry_count += 1
        elif not industry.is_integer():
            raise DataProviderError("INVALID_INDUSTRY_ID: industry must be an integer")
    return missing_industry_count


def _validate_mask_records(records: tuple[SampleRow, ...]) -> None:
    _require_source_fields(records, MASK_FIELDS, "mask")
    for row in records:
        for field in MASK_FIELDS:
            _mask_value(row[field], field)


def _validate_alignment(
    left: Sequence[SampleRow],
    right: Sequence[SampleRow],
    error_code: str,
) -> None:
    left_keys = tuple(_row_key(row) for row in left)
    right_keys = tuple(_row_key(row) for row in right)
    if left_keys != right_keys:
        raise DataValidationError(f"{error_code}: artifact indexes differ")


def _row_key(row: SampleRow) -> tuple[str, str]:
    datetime_value = row.get("datetime")
    instrument_value = row.get("instrument")
    if type(datetime_value) is not str or type(instrument_value) is not str:
        raise DataValidationError("INVALID_INDEX_SCHEMA: invalid row index")
    return datetime_value, instrument_value


def _count_index_levels(records: Sequence[SampleRow]) -> tuple[int, int]:
    dates = {cast(str, row["datetime"]) for row in records}
    instruments = {cast(str, row["instrument"]) for row in records}
    return len(dates), len(instruments)


def _count_excluded_masks(records: Sequence[SampleRow]) -> int:
    count = 0
    for row in records:
        values = tuple(row[field] for field in MASK_FIELDS if field in row)
        if any(value is False for value in values):
            count += 1
    return count


def _count_unknown_masks(records: Sequence[SampleRow]) -> int:
    count = 0
    for row in records:
        values = tuple(row[field] for field in MASK_FIELDS if field in row)
        if any(value is None for value in values):
            count += 1
    return count
