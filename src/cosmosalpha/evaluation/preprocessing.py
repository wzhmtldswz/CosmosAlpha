"""Forward returns and gplearn_neutralized_v1 factor preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np

from cosmosalpha.agents import FactorCalculationResult, FactorPreprocessingResult
from cosmosalpha.core import (
    COLUMNAR_RECORDS_MEDIA_TYPE,
    ArtifactRef,
    ColumnarRecordsError,
    ColumnarTable,
    CoreValidationError,
    WarningInfo,
    decode_record_artifact_table,
    encode_columnar_columns,
)
from cosmosalpha.data import DataSnapshot

FORWARD_RETURN_VERSION = "forward-return-calculator/1.0"
FORWARD_RETURN_SCHEMA_VERSION = "forward-return-result/1.0"
GPLEARN_NEUTRALIZED_PROTOCOL_ID = "gplearn_neutralized_v1"
GPLEARN_NEUTRALIZED_PROTOCOL_VERSION = "1.0.0"
FACTOR_VALUE_KIND = "factor_values"
FORWARD_RETURN_KIND = "forward_returns"
ARTIFACT_COLUMNAR_MEDIA_TYPE = COLUMNAR_RECORDS_MEDIA_TYPE
CONTINUOUS_EXPOSURES = ("log_mkt_cap", "ret20", "turn20", "vol20")


class ReturnCalculationError(Exception):
    """Raised when forward returns cannot be calculated."""


class PreprocessingError(Exception):
    """Raised when factor preprocessing cannot be performed."""


class ArtifactReader(Protocol):
    """Minimal artifact reader used by cached evaluation inputs."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read artifact bytes."""


class ArtifactRepository(ArtifactReader, Protocol):
    """Artifact reader and writer used by preprocessing components."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist artifact bytes."""


@dataclass(frozen=True, slots=True)
class ForwardReturnTiming:
    """Non-persisted timing from the latest forward-return calculation."""

    total_seconds: float
    input_seconds: float
    calculation_seconds: float
    persistence_seconds: float


@dataclass(frozen=True, slots=True)
class PreprocessingTiming:
    """Non-persisted timing from the latest preprocessing call."""

    total_seconds: float
    input_seconds: float
    calculation_seconds: float
    persistence_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    """Immutable return, mask, and index arrays shared across factor attempts."""

    cache_key: tuple[str, str, str]
    datetimes: np.ndarray
    instruments: np.ndarray
    forward_returns: np.ndarray
    eligible: np.ndarray
    date_slices: Mapping[str, slice]


class EvaluationInputCache:
    """Cache immutable snapshot inputs and evaluated factors for one run."""

    def __init__(self) -> None:
        """Create an unbounded run-local cache for immutable artifact content."""
        self._evaluation_inputs: dict[tuple[str, str, str], EvaluationInputs] = {}
        self._exposures: dict[
            tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._factor_values: dict[tuple[tuple[str, str, str], str], np.ndarray] = {}
        self._lock = Lock()
        self._factor_lock = Lock()

    @property
    def evaluation_entry_count(self) -> int:
        """Return the number of cached return/mask input sets."""
        with self._lock:
            return len(self._evaluation_inputs)

    @property
    def factor_entry_count(self) -> int:
        """Return the number of cached evaluated-factor arrays."""
        with self._factor_lock:
            return len(self._factor_values)

    def get_evaluation_inputs(
        self,
        key: tuple[str, str, str],
        loader: Callable[[], EvaluationInputs],
    ) -> EvaluationInputs:
        """Return aligned static inputs, loading one cache miss at most once."""
        with self._lock:
            cached = self._evaluation_inputs.get(key)
            if cached is None:
                cached = loader()
                if cached.cache_key != key:
                    raise PreprocessingError("evaluation input cache key mismatch")
                self._evaluation_inputs[key] = cached
            return cached

    def get_exposures(
        self,
        key: tuple[str, str, str, str],
        loader: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return point-in-time exposures shared by factor preprocessing."""
        with self._lock:
            cached = self._exposures.get(key)
            if cached is None:
                cached = loader()
                _make_read_only(*cached)
                self._exposures[key] = cached
            return cached

    def get_factor_values(
        self,
        inputs: EvaluationInputs,
        ref: ArtifactRef,
        loader: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Return one evaluated factor array aligned to cached snapshot inputs."""
        key = (inputs.cache_key, ref.artifact_id)
        with self._factor_lock:
            cached = self._factor_values.get(key)
            if cached is None:
                cached = loader()
                _validate_cached_factor_values(cached, inputs)
                self._factor_values[key] = cached
            return cached

    def get_factor_values_batch(
        self,
        inputs: EvaluationInputs,
        refs: tuple[ArtifactRef, ...],
        loader: Callable[[ArtifactRef], np.ndarray],
        *,
        max_workers: int,
    ) -> tuple[np.ndarray, ...]:
        """Return aligned factor arrays, loading cache misses concurrently."""
        if type(max_workers) is not int or max_workers < 1:
            raise PreprocessingError("max_workers must be a positive integer")
        keys = tuple((inputs.cache_key, ref.artifact_id) for ref in refs)
        with self._factor_lock:
            missing_by_id: dict[str, ArtifactRef] = {}
            for key, ref in zip(keys, refs, strict=True):
                if key not in self._factor_values:
                    missing_by_id.setdefault(ref.artifact_id, ref)
            missing_refs = tuple(missing_by_id.values())
            if len(missing_refs) == 1:
                loaded = (loader(missing_refs[0]),)
            elif missing_refs:
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(missing_refs))
                ) as executor:
                    loaded = tuple(executor.map(loader, missing_refs))
            else:
                loaded = ()
            for ref, values in zip(missing_refs, loaded, strict=True):
                _validate_cached_factor_values(values, inputs)
                self._factor_values[(inputs.cache_key, ref.artifact_id)] = values
            return tuple(self._factor_values[key] for key in keys)

    def store_factor_values(
        self,
        inputs: EvaluationInputs,
        ref: ArtifactRef,
        values: np.ndarray,
    ) -> None:
        """Register a freshly persisted evaluated array without decoding it again."""
        _validate_cached_factor_values(values, inputs)
        key = (inputs.cache_key, ref.artifact_id)
        with self._factor_lock:
            self._factor_values.setdefault(key, values)


@dataclass(frozen=True, slots=True)
class ForwardReturnResult:
    """Metadata for deterministic forward return artifacts."""

    return_definition: str
    holding_period: int
    entry_time: str
    exit_time: str
    factor_timestamp: str
    data_snapshot_id: str
    return_values_ref: ArtifactRef
    valid_sample_count: int
    missing_sample_count: int
    calculation_version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate forward return result metadata."""
        _require_string(self.return_definition, "return_definition")
        _require_int(self.holding_period, "holding_period", minimum=1)
        _require_string(self.entry_time, "entry_time")
        _require_string(self.exit_time, "exit_time")
        _require_string(self.factor_timestamp, "factor_timestamp")
        _require_string(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.return_values_ref, ArtifactRef):
            raise CoreValidationError("return_values_ref must be ArtifactRef")
        _require_int(self.valid_sample_count, "valid_sample_count")
        _require_int(self.missing_sample_count, "missing_sample_count")
        _require_string(self.calculation_version, "calculation_version")
        if self.schema_version != FORWARD_RETURN_SCHEMA_VERSION:
            raise CoreValidationError(
                f"schema_version must be {FORWARD_RETURN_SCHEMA_VERSION!r}"
            )


class ForwardReturnCalculator:
    """Calculate close-to-next-close forward returns from a data snapshot."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        compression_limit: float = 0.10,
    ) -> None:
        """Create a forward return calculator."""
        self._artifacts = artifact_repository
        self._compression_limit = compression_limit
        self._last_timing = ForwardReturnTiming(0.0, 0.0, 0.0, 0.0)
        self._cache: dict[tuple[str, str], ForwardReturnResult] = {}
        self._cache_lock = Lock()

    @property
    def last_timing(self) -> ForwardReturnTiming:
        """Return timing from the latest successful calculation."""
        return self._last_timing

    def calculate(self, data_snapshot: DataSnapshot) -> ForwardReturnResult:
        """Calculate forward returns and persist a return artifact."""
        started_at = perf_counter()
        if not isinstance(data_snapshot, DataSnapshot):
            raise ReturnCalculationError("data_snapshot must be DataSnapshot")
        cache_key = (
            data_snapshot.snapshot_id,
            data_snapshot.market_data.data_ref.artifact_id,
        )
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._last_timing = ForwardReturnTiming(
                    total_seconds=perf_counter() - started_at,
                    input_seconds=0.0,
                    calculation_seconds=0.0,
                    persistence_seconds=0.0,
                )
                return cached
            result = self._calculate_uncached(data_snapshot, started_at)
            self._cache[cache_key] = result
            return result

    def _calculate_uncached(
        self,
        data_snapshot: DataSnapshot,
        started_at: float,
    ) -> ForwardReturnResult:
        """Calculate and persist a cache miss while the caller owns the lock."""
        phase_started_at = perf_counter()
        market_table = _read_table(
            self._artifacts.get_bytes(data_snapshot.market_data.data_ref),
            "market_data",
            ReturnCalculationError,
        )
        datetimes, instruments = _validate_sorted_unique_table(
            market_table,
            ReturnCalculationError,
        )
        closes = _table_numeric_array(
            market_table,
            "close",
            ReturnCalculationError,
        )
        input_seconds = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        returns = np.full(market_table.row_count, np.nan, dtype=float)
        for indices in _group_indices(instruments):
            returns[indices] = _forward_returns_for_group(
                closes[indices],
                compression_limit=self._compression_limit,
            )
        calculation_seconds = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        output_ref = self._artifacts.put_bytes(
            encode_columnar_columns(
                {
                    "datetime": datetimes,
                    "forward_return": returns,
                    "instrument": instruments,
                }
            ),
            kind=FORWARD_RETURN_KIND,
            media_type=ARTIFACT_COLUMNAR_MEDIA_TYPE,
        )
        persistence_seconds = perf_counter() - phase_started_at
        valid_count = int(np.count_nonzero(np.isfinite(returns)))
        result = ForwardReturnResult(
            return_definition="close_t1_over_close_t",
            holding_period=1,
            entry_time="close_t",
            exit_time="close_t1",
            factor_timestamp="close_t",
            data_snapshot_id=data_snapshot.snapshot_id,
            return_values_ref=output_ref,
            valid_sample_count=valid_count,
            missing_sample_count=market_table.row_count - valid_count,
            calculation_version=FORWARD_RETURN_VERSION,
            schema_version=FORWARD_RETURN_SCHEMA_VERSION,
        )
        self._last_timing = ForwardReturnTiming(
            total_seconds=perf_counter() - started_at,
            input_seconds=input_seconds,
            calculation_seconds=calculation_seconds,
            persistence_seconds=persistence_seconds,
        )
        return result


class GplearnNeutralizedPreprocessor:
    """Apply the gplearn_neutralized_v1 deterministic preprocessing protocol."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        mad_multiplier: float = 5.0,
        min_cross_section_samples: int = 20,
        input_cache: EvaluationInputCache | None = None,
    ) -> None:
        """Create the preprocessor."""
        self._artifacts = artifact_repository
        self._mad_multiplier = mad_multiplier
        self._min_samples = min_cross_section_samples
        self._input_cache = input_cache or EvaluationInputCache()
        self._last_timing = PreprocessingTiming(0.0, 0.0, 0.0, 0.0)

    @property
    def last_timing(self) -> PreprocessingTiming:
        """Return timing from the latest successful preprocessing call."""
        return self._last_timing

    def preprocess(
        self,
        factor_result: FactorCalculationResult,
        forward_return_result: ForwardReturnResult,
        data_snapshot: DataSnapshot,
    ) -> FactorPreprocessingResult:
        """Preprocess raw factor values for single-factor evaluation."""
        started_at = perf_counter()
        if factor_result.data_snapshot_id != data_snapshot.snapshot_id:
            raise PreprocessingError("factor result data_snapshot_id mismatch")
        if forward_return_result.data_snapshot_id != data_snapshot.snapshot_id:
            raise PreprocessingError("forward return data_snapshot_id mismatch")

        phase_started_at = perf_counter()
        inputs = self._evaluation_inputs(
            forward_return_result,
            data_snapshot,
            PreprocessingError,
        )
        factor_table = _read_table(
            self._artifacts.get_bytes(factor_result.factor_values_ref),
            "factor_values",
            PreprocessingError,
        )
        _validate_table_alignment(
            factor_table,
            inputs,
            PreprocessingError,
        )
        raw_factor = _table_numeric_array(
            factor_table,
            "factor_value",
            PreprocessingError,
            missing_ok=True,
        )
        exposure_key = (
            *inputs.cache_key,
            data_snapshot.exposure_data.data_ref.artifact_id,
        )
        industry, continuous = self._input_cache.get_exposures(
            exposure_key,
            lambda: self._load_exposures(data_snapshot, inputs),
        )
        input_seconds = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        arrays = _PreprocessedArrays.empty(raw_factor.size)
        warnings: list[WarningInfo] = []
        for date_key, indices in inputs.date_slices.items():
            warning = self._preprocess_one_day(
                indices=indices,
                raw_factor=raw_factor,
                forward_return=inputs.forward_returns,
                industry=industry,
                continuous=continuous,
                eligible=inputs.eligible,
                arrays=arrays,
            )
            if warning is not None:
                warnings.append(
                    WarningInfo(
                        "INSUFFICIENT_SAMPLES",
                        f"{date_key}: {warning}",
                    )
                )
        calculation_seconds = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        eligible_ref = self._put_factor_records(
            inputs.datetimes,
            inputs.instruments,
            arrays.eligible,
            "eligible_factor_values",
        )
        winsorized_ref = self._put_factor_records(
            inputs.datetimes,
            inputs.instruments,
            arrays.winsorized,
            "winsorized_factor_values",
        )
        neutralized_ref = self._put_factor_records(
            inputs.datetimes,
            inputs.instruments,
            arrays.neutralized,
            "neutralized_factor_values",
        )
        evaluated_ref = self._put_factor_records(
            inputs.datetimes,
            inputs.instruments,
            arrays.evaluated,
            "evaluated_factor_values",
        )
        self._input_cache.store_factor_values(inputs, evaluated_ref, arrays.evaluated)
        persistence_seconds = perf_counter() - phase_started_at
        effective = int(np.count_nonzero(np.isfinite(arrays.evaluated)))
        result = FactorPreprocessingResult(
            factor_id=factor_result.factor_id,
            data_snapshot_id=data_snapshot.snapshot_id,
            input_factor_values_ref=factor_result.factor_values_ref,
            eligible_factor_values_ref=eligible_ref,
            winsorized_factor_values_ref=winsorized_ref,
            neutralized_factor_values_ref=neutralized_ref,
            evaluated_factor_values_ref=evaluated_ref,
            forward_return_ref=forward_return_result.return_values_ref,
            exposure_data_ref=data_snapshot.exposure_data.data_ref,
            mask_data_ref=data_snapshot.mask_data.mask_data_ref,
            preprocessing_protocol_id=GPLEARN_NEUTRALIZED_PROTOCOL_ID,
            preprocessing_protocol_version=GPLEARN_NEUTRALIZED_PROTOCOL_VERSION,
            neutralization_applied=True,
            excluded_sample_count=raw_factor.size - effective,
            effective_sample_count=effective,
            warnings=tuple(warnings),
            schema_version="factor-preprocessing-result/1.0",
        )
        self._last_timing = PreprocessingTiming(
            total_seconds=perf_counter() - started_at,
            input_seconds=input_seconds,
            calculation_seconds=calculation_seconds,
            persistence_seconds=persistence_seconds,
        )
        return result

    def _preprocess_one_day(
        self,
        *,
        indices: slice,
        raw_factor: np.ndarray,
        forward_return: np.ndarray,
        industry: np.ndarray,
        continuous: np.ndarray,
        eligible: np.ndarray,
        arrays: _PreprocessedArrays,
    ) -> str | None:
        day_arrays, warning = _preprocess_cross_section(
            raw_factor[indices],
            industry[indices],
            continuous[indices],
            eligible[indices] & np.isfinite(forward_return[indices]),
            mad_multiplier=self._mad_multiplier,
            min_cross_section_samples=self._min_samples,
            neutralize=True,
        )
        arrays.eligible[indices] = day_arrays.eligible
        arrays.winsorized[indices] = day_arrays.winsorized
        arrays.neutralized[indices] = day_arrays.neutralized
        arrays.evaluated[indices] = day_arrays.evaluated
        return warning

    def _evaluation_inputs(
        self,
        forward_return_result: ForwardReturnResult,
        data_snapshot: DataSnapshot,
        error_type: type[Exception],
    ) -> EvaluationInputs:
        key = (
            data_snapshot.snapshot_id,
            forward_return_result.return_values_ref.artifact_id,
            data_snapshot.mask_data.mask_data_ref.artifact_id,
        )
        return self._input_cache.get_evaluation_inputs(
            key,
            lambda: _load_evaluation_inputs(
                self._artifacts,
                data_snapshot.snapshot_id,
                forward_return_result.return_values_ref,
                data_snapshot.mask_data.mask_data_ref,
                error_type,
            ),
        )

    def _load_exposures(
        self,
        data_snapshot: DataSnapshot,
        inputs: EvaluationInputs,
    ) -> tuple[np.ndarray, np.ndarray]:
        table = _read_table(
            self._artifacts.get_bytes(data_snapshot.exposure_data.data_ref),
            "exposure_data",
            PreprocessingError,
        )
        _validate_table_alignment(table, inputs, PreprocessingError)
        industry = _table_numeric_array(
            table,
            "industry_id",
            PreprocessingError,
        )
        continuous = np.column_stack(
            [
                _table_numeric_array(table, field, PreprocessingError)
                for field in CONTINUOUS_EXPOSURES
            ]
        )
        return industry, continuous

    def _put_factor_records(
        self,
        datetimes: np.ndarray,
        instruments: np.ndarray,
        values: np.ndarray,
        kind: str,
    ) -> ArtifactRef:
        return self._artifacts.put_bytes(
            encode_columnar_columns(
                {
                    "datetime": datetimes,
                    "factor_value": values,
                    "instrument": instruments,
                }
            ),
            kind=kind,
            media_type=ARTIFACT_COLUMNAR_MEDIA_TYPE,
        )


@dataclass(frozen=True, slots=True)
class _PreprocessedArrays:
    eligible: np.ndarray
    winsorized: np.ndarray
    neutralized: np.ndarray
    evaluated: np.ndarray

    @classmethod
    def empty(cls, size: int) -> _PreprocessedArrays:
        return cls(
            eligible=np.full(size, np.nan, dtype=float),
            winsorized=np.full(size, np.nan, dtype=float),
            neutralized=np.full(size, np.nan, dtype=float),
            evaluated=np.full(size, np.nan, dtype=float),
        )


def preprocess_feature_values(
    raw_factor: np.ndarray,
    *,
    datetimes: tuple[str, ...],
    industry: np.ndarray,
    continuous: np.ndarray,
    eligible: np.ndarray,
    mad_multiplier: float,
    min_cross_section_samples: int,
    neutralize: bool = True,
) -> np.ndarray:
    """Winsorize, optionally neutralize, and standardize feature values.

    Args:
        raw_factor: Raw expression values aligned to the sample index.
        datetimes: Date key for every sample.
        industry: Source-versioned industry code; unavailable values remain NaN.
        continuous: Point-in-time continuous exposure matrix.
        eligible: Current-date research eligibility mask.
        mad_multiplier: Cross-sectional MAD winsorization multiplier.
        min_cross_section_samples: Minimum usable samples per date.
        neutralize: Whether to remove industry and continuous exposures.

    Returns:
        Cross-sectional feature values; historical availability follows the inputs.
    """
    size = len(datetimes)
    if raw_factor.shape != (size,):
        raise PreprocessingError("raw_factor length does not match datetimes")
    if industry.shape != (size,):
        raise PreprocessingError("industry length does not match datetimes")
    if continuous.ndim != 2 or continuous.shape[0] != size:
        raise PreprocessingError("continuous exposures do not match datetimes")
    if eligible.shape != (size,) or eligible.dtype != np.bool_:
        raise PreprocessingError("eligible mask does not match datetimes")
    if type(mad_multiplier) is not float or mad_multiplier <= 0.0:
        raise PreprocessingError("mad_multiplier must be positive")
    if type(min_cross_section_samples) is not int or min_cross_section_samples < 2:
        raise PreprocessingError("min_cross_section_samples must be >= 2")
    if type(neutralize) is not bool:
        raise PreprocessingError("neutralize must be a boolean")

    result = np.full(size, np.nan, dtype=float)
    groups: dict[str, list[int]] = {}
    for index, date_value in enumerate(datetimes):
        groups.setdefault(date_value, []).append(index)
    for positions in groups.values():
        indices = np.asarray(positions, dtype=np.int64)
        day_arrays, _warning = _preprocess_cross_section(
            raw_factor[indices],
            industry[indices],
            continuous[indices],
            eligible[indices],
            mad_multiplier=mad_multiplier,
            min_cross_section_samples=min_cross_section_samples,
            neutralize=neutralize,
        )
        result[indices] = day_arrays.evaluated
    return result


def _preprocess_cross_section(
    raw_factor: np.ndarray,
    industry: np.ndarray,
    continuous: np.ndarray,
    available: np.ndarray,
    *,
    mad_multiplier: float,
    min_cross_section_samples: int,
    neutralize: bool,
) -> tuple[_PreprocessedArrays, str | None]:
    arrays = _PreprocessedArrays.empty(raw_factor.size)
    base_keep = available & np.isfinite(raw_factor)
    if int(np.count_nonzero(base_keep)) < min_cross_section_samples:
        return arrays, "not enough eligible factor/return pairs"

    base_indices = np.flatnonzero(base_keep)
    factor_w = _mad_winsorize(raw_factor[base_indices], mad_multiplier)
    arrays.eligible[base_indices] = raw_factor[base_indices]
    arrays.winsorized[base_indices] = factor_w

    if not neutralize:
        evaluated = _zscore(factor_w)
        finite = np.isfinite(evaluated)
        if int(np.count_nonzero(finite)) < min_cross_section_samples:
            return arrays, "not enough standardized samples"
        output_indices = base_indices[finite]
        arrays.neutralized[output_indices] = factor_w[finite]
        arrays.evaluated[output_indices] = evaluated[finite]
        return arrays, None

    exposure_keep = np.isfinite(industry[base_indices])
    exposure_keep &= np.all(np.isfinite(continuous[base_indices]), axis=1)
    if int(np.count_nonzero(exposure_keep)) < min_cross_section_samples:
        return arrays, "not enough samples with complete exposures"

    regression_indices = base_indices[exposure_keep]
    y = factor_w[exposure_keep]
    industry_values = np.rint(industry[regression_indices]).astype(
        np.int64,
        copy=False,
    )
    continuous_values = continuous[regression_indices]
    continuous_z = np.column_stack(
        [
            _zscore(_mad_winsorize(continuous_values[:, column], mad_multiplier))
            for column in range(continuous_values.shape[1])
        ]
    )
    final_keep = np.isfinite(y) & np.all(np.isfinite(continuous_z), axis=1)
    if int(np.count_nonzero(final_keep)) < min_cross_section_samples:
        return arrays, "not enough samples after exposure standardization"

    regression_indices = regression_indices[final_keep]
    y = y[final_keep]
    industry_values = industry_values[final_keep]
    continuous_z = continuous_z[final_keep]
    industry_labels, industry_inverse = np.unique(
        industry_values,
        return_inverse=True,
    )
    if y.size <= industry_labels.size + continuous_z.shape[1]:
        return (
            arrays,
            "neutralization design matrix has insufficient degrees of freedom",
        )

    residual = _fixed_effects_residual(y, continuous_z, industry_inverse)
    evaluated = _zscore(residual)
    finite_final = np.isfinite(evaluated)
    if int(np.count_nonzero(finite_final)) < min_cross_section_samples:
        return arrays, "not enough standardized residual samples"

    output_indices = regression_indices[finite_final]
    arrays.neutralized[output_indices] = residual[finite_final]
    arrays.evaluated[output_indices] = evaluated[finite_final]
    return arrays, None


def _forward_returns_for_group(
    closes: np.ndarray, *, compression_limit: float
) -> np.ndarray:
    output = np.full(closes.shape, np.nan, dtype=float)
    for index, current in enumerate(closes[:-1]):
        if not np.isfinite(current) or current <= 0:
            continue
        next_index = index + 1
        while next_index < closes.size and not np.isfinite(closes[next_index]):
            next_index += 1
        if next_index >= closes.size:
            continue
        next_close = closes[next_index]
        if not np.isfinite(next_close) or next_close <= 0:
            continue
        value = next_close / current - 1.0
        if next_index > index + 1:
            value = float(np.clip(value, -compression_limit, compression_limit))
        output[index] = value
    return output


def _mad_winsorize(values: np.ndarray, multiplier: float) -> np.ndarray:
    output = values.astype(float, copy=True)
    finite = np.isfinite(output)
    if not finite.any():
        return output
    median = float(np.median(output[finite]))
    mad = float(np.median(np.abs(output[finite] - median)))
    if not np.isfinite(mad) or mad == 0:
        output[finite] = median
        return output
    lower = median - multiplier * mad
    upper = median + multiplier * mad
    output[finite] = np.clip(output[finite], lower, upper)
    return output


def _zscore(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite], ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return output
    output[finite] = (values[finite] - mean) / std
    return output


def _fixed_effects_residual(
    y: np.ndarray,
    continuous_z: np.ndarray,
    industry_inverse: np.ndarray,
) -> np.ndarray:
    """Remove industry fixed effects without materializing dummy columns.

    The Frisch-Waugh-Lovell equivalence lets us demean the factor and continuous
    exposures within industry, then solve only the small continuous regression.
    """
    industry_count = int(np.max(industry_inverse)) + 1
    counts = np.bincount(industry_inverse, minlength=industry_count).astype(float)
    y_means = (
        np.bincount(
            industry_inverse,
            weights=y,
            minlength=industry_count,
        )
        / counts
    )
    y_centered = y - y_means[industry_inverse]

    continuous_centered = continuous_z.copy()
    for column in range(continuous_centered.shape[1]):
        means = (
            np.bincount(
                industry_inverse,
                weights=continuous_centered[:, column],
                minlength=industry_count,
            )
            / counts
        )
        continuous_centered[:, column] -= means[industry_inverse]
    try:
        beta, *_ = np.linalg.lstsq(continuous_centered, y_centered, rcond=None)
    except np.linalg.LinAlgError as error:
        raise PreprocessingError("neutralization lstsq failed") from error
    return y_centered - continuous_centered @ beta


def _load_evaluation_inputs(
    artifacts: ArtifactReader,
    data_snapshot_id: str,
    forward_return_ref: ArtifactRef,
    mask_data_ref: ArtifactRef,
    error_type: type[Exception],
) -> EvaluationInputs:
    """Load aligned return/mask arrays without constructing row dictionaries."""
    return_table = _read_table(
        artifacts.get_bytes(forward_return_ref),
        "forward_returns",
        error_type,
    )
    mask_table = _read_table(
        artifacts.get_bytes(mask_data_ref),
        "mask_data",
        error_type,
    )
    datetimes, instruments = _validate_sorted_unique_table(mask_table, error_type)
    _validate_index_arrays(return_table, datetimes, instruments, error_type)
    forward_returns = _table_numeric_array(
        return_table,
        "forward_return",
        error_type,
        missing_ok=True,
    )
    eligible = _table_true_array(mask_table, "eligible_mask")
    _make_read_only(datetimes, instruments, forward_returns, eligible)
    key = (
        data_snapshot_id,
        forward_return_ref.artifact_id,
        mask_data_ref.artifact_id,
    )
    return EvaluationInputs(
        cache_key=key,
        datetimes=datetimes,
        instruments=instruments,
        forward_returns=forward_returns,
        eligible=eligible,
        date_slices=_date_slices(datetimes),
    )


def _read_table(
    data: bytes,
    label: str,
    error_type: type[Exception],
) -> ColumnarTable:
    """Decode a record artifact into typed columns with a domain error."""
    try:
        return decode_record_artifact_table(data)
    except ColumnarRecordsError as error:
        raise error_type(f"{label}: invalid record artifact") from error


def _validate_sorted_unique_table(
    table: ColumnarTable,
    error_type: type[Exception],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the canonical datetime/instrument order and return its columns."""
    if table.row_count == 0:
        raise error_type("EMPTY_DATASET: records are empty")
    datetimes = _table_string_array(table, "datetime", error_type)
    instruments = _table_string_array(table, "instrument", error_type)
    same_date = datetimes[1:] == datetimes[:-1]
    same_instrument = instruments[1:] == instruments[:-1]
    if np.any(same_date & same_instrument):
        raise error_type("DUPLICATE_SAMPLE_INDEX: duplicate samples")
    earlier_date = np.char.less(datetimes[1:], datetimes[:-1])
    earlier_instrument = np.char.less(instruments[1:], instruments[:-1])
    if np.any(earlier_date | (same_date & earlier_instrument)):
        raise error_type("UNSORTED_SAMPLE_INDEX: records are not sorted")
    return datetimes, instruments


def _validate_table_alignment(
    table: ColumnarTable,
    inputs: EvaluationInputs,
    error_type: type[Exception],
) -> None:
    """Require a table to use the cached canonical sample index."""
    _validate_index_arrays(
        table,
        inputs.datetimes,
        inputs.instruments,
        error_type,
    )


def _validate_index_arrays(
    table: ColumnarTable,
    datetimes: np.ndarray,
    instruments: np.ndarray,
    error_type: type[Exception],
) -> None:
    candidate_dates = _table_string_array(table, "datetime", error_type)
    candidate_instruments = _table_string_array(table, "instrument", error_type)
    if not np.array_equal(candidate_dates, datetimes) or not np.array_equal(
        candidate_instruments,
        instruments,
    ):
        raise error_type("INDEX_MISMATCH: artifacts are not aligned")


def _table_string_array(
    table: ColumnarTable,
    field: str,
    error_type: type[Exception],
) -> np.ndarray:
    if field not in table.columns:
        raise error_type(f"MISSING_REQUIRED_FIELD: {field}")
    if table.column_types[field] != "str" or not np.all(table.validity[field]):
        raise error_type("INVALID_INDEX_SCHEMA: invalid record index")
    values = table.columns[field].astype(str, copy=False)
    _make_read_only(values)
    return values


def _table_numeric_array(
    table: ColumnarTable,
    field: str,
    error_type: type[Exception],
    *,
    missing_ok: bool = False,
) -> np.ndarray:
    if field not in table.columns:
        if missing_ok:
            values = np.full(table.row_count, np.nan, dtype=float)
            _make_read_only(values)
            return values
        raise error_type(f"MISSING_REQUIRED_FIELD: {field}")
    if table.column_types[field] not in {"float", "int"}:
        raise error_type(f"NONFINITE_VALUE: {field}")
    values = table.columns[field].astype(np.float64, copy=False)
    valid = table.validity[field]
    if np.any(~valid):
        values = values.copy()
        values[~valid] = np.nan
    if np.any(~np.isfinite(values[valid])):
        raise error_type(f"NONFINITE_VALUE: {field}")
    _make_read_only(values)
    return values


def _table_true_array(table: ColumnarTable, field: str) -> np.ndarray:
    """Match legacy ``value is True`` mask semantics for a typed column."""
    if field not in table.columns or table.column_types[field] != "bool":
        values = np.zeros(table.row_count, dtype=bool)
    else:
        values = table.columns[field] & table.validity[field]
    _make_read_only(values)
    return values


def _date_slices(datetimes: np.ndarray) -> Mapping[str, slice]:
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(datetimes[1:] != datetimes[:-1]) + 1,
            np.array([datetimes.size], dtype=np.int64),
        )
    )
    groups = {
        str(datetimes[start]): slice(int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    }
    return MappingProxyType(groups)


def _group_indices(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Group equal labels with one stable sort instead of Python row lists."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    return tuple(np.split(order, boundaries))


def _make_read_only(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)


def _validate_cached_factor_values(
    values: np.ndarray,
    inputs: EvaluationInputs,
) -> None:
    """Validate and freeze one factor array before caching it."""
    if values.shape != inputs.forward_returns.shape:
        raise PreprocessingError("factor value cache length mismatch")
    _make_read_only(values)


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CoreValidationError(f"{field_name} must be >= {minimum}")
    return value
