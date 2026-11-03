"""Exact, incremental, and persistable Graph correlation calculations."""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np

from cosmosalpha.core import (
    ArtifactRef,
    ColumnarTable,
    CoreValidationError,
    canonical_json_bytes,
    decode_record_artifact_table,
    from_json,
)
from cosmosalpha.evaluation._statistics import mean_daily_cross_sectional_pearson
from cosmosalpha.evaluation.correlation.models import (
    CorrelationFactorRef,
    CorrelationSnapshot,
    CorrelationStore,
    GraphCorrelationProtocol,
    _snapshot_fingerprint,
)

CORRELATION_MATRIX_MEDIA_TYPE = "application/vnd.cosmosalpha.graph-correlation-npz"
_CORRELATION_MATRIX_KIND = "graph_correlation_snapshot"
_CORRELATION_SNAPSHOT_KIND = "graph_correlation_metadata"


class CorrelationError(ValueError):
    """Base class for invalid correlation input or state."""


class CorrelationDataError(CorrelationError):
    """Report malformed or misaligned evaluated exposure artifacts."""


class CorrelationProtocolMismatch(CorrelationError):
    """Report an attempt to mix incompatible correlation inputs or snapshots."""


class CorrelationArtifactRepository(Protocol):
    """Minimal artifact repository needed by the engine."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read and integrity-check an artifact."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist bytes and return an artifact reference."""


@dataclass(frozen=True, slots=True)
class CorrelationUpdateResult:
    """Report an incremental update and its resulting complete store."""

    store: CorrelationStore
    computed_pair_count: int

    def __post_init__(self) -> None:
        """Validate update accounting."""
        if not isinstance(self.store, CorrelationStore):
            raise CoreValidationError("store must be CorrelationStore")
        if type(self.computed_pair_count) is not int or self.computed_pair_count < 0:
            raise CoreValidationError("computed_pair_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CorrelationTiming:
    """Non-persisted timing facts from one correlation build or update."""

    total_seconds: float
    exposure_decode_seconds: float
    pairwise_correlation_seconds: float
    computed_pair_count: int

    def __post_init__(self) -> None:
        """Validate timing values without making them part of correlation identity."""
        for field_name in (
            "total_seconds",
            "exposure_decode_seconds",
            "pairwise_correlation_seconds",
        ):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise CoreValidationError(f"{field_name} must be finite")
            if float(value) < 0.0:
                raise CoreValidationError(f"{field_name} must be non-negative")
        if type(self.computed_pair_count) is not int or self.computed_pair_count < 0:
            raise CoreValidationError("computed_pair_count must be non-negative")


@dataclass(frozen=True, slots=True)
class _DailyExposure:
    """One factor's values and mask for a single date."""

    instruments: np.ndarray
    values: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True, slots=True)
class _Exposure:
    """Decoded factor exposure kept only for the active pair calculation."""

    dates: Mapping[str, _DailyExposure]


@dataclass(frozen=True, slots=True)
class _DailyMask:
    """One date's canonical instrument index and eligibility values."""

    instruments: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True, slots=True)
class _MaskData:
    """Decoded daily eligible mask shared by compatible factor refs."""

    dates: Mapping[str, _DailyMask]
    eligible_counts: Mapping[str, int]


def _require_factor_sequence(
    factors: Sequence[CorrelationFactorRef], field_name: str
) -> tuple[CorrelationFactorRef, ...]:
    """Validate a factor sequence and preserve its explicit matrix order."""
    if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)):
        raise CorrelationDataError(f"{field_name} must be a sequence")
    values = tuple(factors)
    if not all(isinstance(factor, CorrelationFactorRef) for factor in values):
        raise CorrelationDataError(
            f"every item in {field_name} must be CorrelationFactorRef"
        )
    ids = tuple(factor.factor_id for factor in values)
    if len(set(ids)) != len(ids):
        raise CorrelationDataError(f"{field_name} contains duplicate factor_id values")
    return values


def _validate_dates(evaluation_dates: Sequence[str]) -> tuple[str, ...]:
    """Validate and deterministically order the run evaluation calendar."""
    if not isinstance(evaluation_dates, Sequence) or isinstance(
        evaluation_dates, (str, bytes)
    ):
        raise CorrelationDataError("evaluation_dates must be a sequence")
    dates = tuple(evaluation_dates)
    if not dates or not all(type(value) is str and value for value in dates):
        raise CorrelationDataError("evaluation_dates must contain non-empty strings")
    if len(set(dates)) != len(dates):
        raise CorrelationDataError("evaluation_dates must be unique")
    return tuple(sorted(dates))


def _read_table(
    artifacts: CorrelationArtifactRepository, ref: ArtifactRef, label: str
) -> ColumnarTable:
    """Decode current columnar or legacy artifacts without row dictionaries."""
    try:
        return decode_record_artifact_table(artifacts.get_bytes(ref))
    except Exception as error:  # noqa: BLE001 - normalize repository/codec failures
        if isinstance(error, CorrelationError):
            raise
        raise CorrelationDataError(f"{label}: invalid record artifact") from error


def _table_index(
    table: ColumnarTable,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a validated sorted, unique datetime/instrument index."""
    if table.row_count == 0:
        raise CorrelationDataError(f"{label}: records are empty")
    for field in ("datetime", "instrument"):
        if (
            field not in table.columns
            or table.column_types[field] != "str"
            or not np.all(table.validity[field])
        ):
            raise CorrelationDataError(f"{label}: invalid {field} key")
    datetimes = table.columns["datetime"].astype(str, copy=False)
    instruments = table.columns["instrument"].astype(str, copy=False)
    same_date = datetimes[1:] == datetimes[:-1]
    same_instrument = instruments[1:] == instruments[:-1]
    if np.any(same_date & same_instrument):
        raise CorrelationDataError(f"{label}: duplicate sample index")
    if np.any(
        np.char.less(datetimes[1:], datetimes[:-1])
        | (same_date & np.char.less(instruments[1:], instruments[:-1]))
    ):
        raise CorrelationDataError(f"{label}: records are not sorted")
    datetimes.setflags(write=False)
    instruments.setflags(write=False)
    return datetimes, instruments


def _date_slices(datetimes: np.ndarray) -> tuple[tuple[str, slice], ...]:
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(datetimes[1:] != datetimes[:-1]) + 1,
            np.array([datetimes.size], dtype=np.int64),
        )
    )
    return tuple(
        (str(datetimes[start]), slice(int(start), int(end)))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def _read_mask(artifacts: CorrelationArtifactRepository, ref: ArtifactRef) -> _MaskData:
    """Decode an eligibility mask and its daily denominators."""
    table = _read_table(artifacts, ref, "mask_data")
    datetimes, instruments = _table_index(table, "mask_data")
    if "eligible_mask" not in table.columns:
        eligible = np.zeros(table.row_count, dtype=np.bool_)
    elif table.column_types["eligible_mask"] != "bool":
        raise CorrelationDataError("mask_data: eligible_mask must be boolean or null")
    else:
        eligible = table.columns["eligible_mask"] & table.validity["eligible_mask"]
    eligible.setflags(write=False)
    dates: dict[str, _DailyMask] = {}
    eligible_counts: dict[str, int] = {}
    for date_value, indices in _date_slices(datetimes):
        day = _DailyMask(
            instruments=instruments[indices],
            eligible=eligible[indices],
        )
        dates[date_value] = day
        eligible_counts[date_value] = int(np.count_nonzero(day.eligible))
    return _MaskData(
        dates=MappingProxyType(dates),
        eligible_counts=MappingProxyType(eligible_counts),
    )


def _read_exposure(
    artifacts: CorrelationArtifactRepository,
    factor: CorrelationFactorRef,
    mask: _MaskData,
) -> _Exposure:
    """Decode one factor into date blocks aligned by explicit keys."""
    label = f"factor {factor.factor_id}"
    table = _read_table(
        artifacts,
        factor.evaluated_factor_values_ref,
        label,
    )
    datetimes, instruments = _table_index(table, label)
    if "factor_value" not in table.columns:
        values = np.full(table.row_count, np.nan, dtype=np.float64)
    elif table.column_types["factor_value"] not in {"float", "int"}:
        raise CorrelationDataError(
            f"factor {factor.factor_id}: factor_value must be finite or null"
        )
    else:
        values = table.columns["factor_value"].astype(np.float64, copy=False)
        valid = table.validity["factor_value"]
        if np.any(~valid):
            values = values.copy()
            values[~valid] = np.nan
    values.setflags(write=False)
    dates: dict[str, _DailyExposure] = {}
    for date_value, indices in _date_slices(datetimes):
        if date_value not in mask.dates:
            raise CorrelationDataError(
                f"factor {factor.factor_id}: key is missing from mask_data"
            )
        day_instruments = instruments[indices]
        mask_day = mask.dates[date_value]
        if np.array_equal(day_instruments, mask_day.instruments):
            day_eligible = mask_day.eligible
        else:
            positions = np.searchsorted(mask_day.instruments, day_instruments)
            in_bounds = positions < mask_day.instruments.size
            if not np.all(in_bounds) or not np.array_equal(
                mask_day.instruments[positions],
                day_instruments,
            ):
                raise CorrelationDataError(
                    f"factor {factor.factor_id}: key is missing from mask_data"
                )
            day_eligible = mask_day.eligible[positions]
        dates[date_value] = _DailyExposure(
            instruments=day_instruments,
            values=values[indices],
            eligible=day_eligible,
        )
    return _Exposure(dates=MappingProxyType(dates))


def _pair_correlation(
    left: _Exposure,
    right: _Exposure,
    mask: _MaskData,
    evaluation_dates: tuple[str, ...],
    protocol: GraphCorrelationProtocol,
) -> tuple[float, int, int]:
    """Calculate one signed fact and its date counters."""
    overlap_dates = tuple(
        date_value
        for date_value in evaluation_dates
        if date_value in left.dates and date_value in right.dates
    )
    overlap_count = len(overlap_dates)
    if (
        not overlap_count
        or overlap_count / len(evaluation_dates) < protocol.min_pair_window_ratio
    ):
        return math.nan, 0, overlap_count

    daily_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    for date_value in overlap_dates:
        left_day = left.dates[date_value]
        right_day = right.dates[date_value]
        if np.array_equal(left_day.instruments, right_day.instruments):
            left_values = left_day.values
            right_values = right_day.values
            pairwise_eligible = left_day.eligible & right_day.eligible
        else:
            _, left_indices, right_indices = np.intersect1d(
                left_day.instruments,
                right_day.instruments,
                assume_unique=True,
                return_indices=True,
            )
            if left_indices.size == 0:
                continue
            left_values = left_day.values[left_indices]
            right_values = right_day.values[right_indices]
            pairwise_eligible = (
                left_day.eligible[left_indices] & right_day.eligible[right_indices]
            )
        if not np.any(pairwise_eligible):
            continue
        daily_pairs.append(
            (
                left_values[pairwise_eligible],
                right_values[pairwise_eligible],
                mask.eligible_counts.get(date_value, 0),
            )
        )
    correlation, valid_count = mean_daily_cross_sectional_pearson(
        daily_pairs,
        min_cross_section_samples=protocol.min_cross_section_samples,
        min_daily_effective_stock_ratio=(protocol.min_daily_effective_stock_ratio),
    )
    return correlation, valid_count, overlap_count


def _validate_protocol_factor(
    factor: CorrelationFactorRef, protocol: GraphCorrelationProtocol
) -> None:
    """Reject factor artifacts from a different evaluation protocol."""
    if factor.data_snapshot_id != protocol.data_snapshot_id:
        raise CorrelationProtocolMismatch(
            f"factor {factor.factor_id} has a different data snapshot"
        )
    if factor.preprocessing_protocol_id != protocol.preprocessing_protocol_id:
        raise CorrelationProtocolMismatch(
            f"factor {factor.factor_id} has a different preprocessing protocol"
        )
    if factor.evaluation_protocol_id != protocol.evaluation_protocol_id:
        raise CorrelationProtocolMismatch(
            f"factor {factor.factor_id} has a different evaluation protocol"
        )


class GraphCorrelationEngine:
    """Calculate exact Graph correlation facts using disk-backed artifacts."""

    def __init__(
        self,
        artifacts: CorrelationArtifactRepository,
        protocol: GraphCorrelationProtocol,
        evaluation_dates: Sequence[str],
        *,
        max_cached_exposures: int = 1024,
    ) -> None:
        """Create an engine for one run calendar and correlation protocol."""
        if not isinstance(protocol, GraphCorrelationProtocol):
            raise CoreValidationError("protocol must be GraphCorrelationProtocol")
        if not hasattr(artifacts, "get_bytes"):
            raise CoreValidationError("artifacts must provide get_bytes")
        if type(max_cached_exposures) is not int or max_cached_exposures < 1:
            raise CoreValidationError("max_cached_exposures must be positive")
        self._artifacts = artifacts
        self.protocol = protocol
        self.evaluation_dates = _validate_dates(evaluation_dates)
        self._mask_cache: dict[str, _MaskData] = {}
        self._exposure_cache: OrderedDict[tuple[str, str], _Exposure] = OrderedDict()
        self._max_cached_exposures = max_cached_exposures
        self._last_timing = CorrelationTiming(0.0, 0.0, 0.0, 0)

    @property
    def last_timing(self) -> CorrelationTiming:
        """Return the latest non-persisted timing facts."""
        return self._last_timing

    def build(self, factors: Sequence[CorrelationFactorRef]) -> CorrelationStore:
        """Build a complete store from an initial factor sequence."""
        refs = _require_factor_sequence(factors, "factors")
        return self.update(
            store=CorrelationStore.empty(self.protocol),
            new_factors=refs,
            all_existing_factors=refs,
        ).store

    def update(
        self,
        *,
        store: CorrelationStore,
        new_factors: Sequence[CorrelationFactorRef],
        all_existing_factors: Sequence[CorrelationFactorRef],
    ) -> CorrelationUpdateResult:
        """Append only new-old and new-new pairs to an existing store."""
        started_at = perf_counter()
        decode_seconds = 0.0
        pairwise_seconds = 0.0
        if not isinstance(store, CorrelationStore):
            raise CoreValidationError("store must be CorrelationStore")
        if store.protocol != self.protocol:
            raise CorrelationProtocolMismatch("store protocol does not match engine")
        new_refs = _require_factor_sequence(new_factors, "new_factors")
        all_refs = _require_factor_sequence(
            all_existing_factors, "all_existing_factors"
        )
        self._validate_factor_sequence(new_refs)
        self._validate_factor_sequence(all_refs)
        old_ids = store.factor_ids
        all_ids = tuple(ref.factor_id for ref in all_refs)
        if all_ids == old_ids:
            all_refs = store.factor_refs + new_refs
            all_ids = tuple(ref.factor_id for ref in all_refs)
        else:
            if old_ids != all_ids[: len(old_ids)]:
                raise CorrelationDataError(
                    "all_existing_factors must retain the store matrix-index prefix"
                )
            if tuple(ref.factor_id for ref in new_refs) != all_ids[len(old_ids) :]:
                raise CorrelationDataError(
                    "new_factors must be the suffix of all_existing_factors"
                )
        if any(factor.factor_id in old_ids for factor in new_refs):
            raise CorrelationDataError("new_factors contains a factor already in store")
        for old_ref, all_ref in zip(
            store.factor_refs, all_refs[: len(old_ids)], strict=True
        ):
            if old_ref != all_ref:
                raise CorrelationDataError(
                    f"metadata for existing factor {old_ref.factor_id} changed"
                )

        if not new_refs:
            self._last_timing = CorrelationTiming(
                total_seconds=perf_counter() - started_at,
                exposure_decode_seconds=0.0,
                pairwise_correlation_seconds=0.0,
                computed_pair_count=0,
            )
            return CorrelationUpdateResult(store=store, computed_pair_count=0)

        def load(factor: CorrelationFactorRef) -> _Exposure:
            nonlocal decode_seconds
            exposure_key = (
                factor.evaluated_factor_values_ref.artifact_id,
                factor.mask_data_ref.artifact_id,
            )
            cached = self._exposure_cache.get(exposure_key)
            if cached is not None:
                self._exposure_cache.move_to_end(exposure_key)
                return cached
            decode_started = perf_counter()
            mask_key = factor.mask_data_ref.artifact_id
            if mask_key not in self._mask_cache:
                self._mask_cache[mask_key] = _read_mask(
                    self._artifacts,
                    factor.mask_data_ref,
                )
            exposure = _read_exposure(
                self._artifacts,
                factor,
                self._mask_cache[mask_key],
            )
            decode_seconds += perf_counter() - decode_started
            self._exposure_cache[exposure_key] = exposure
            if len(self._exposure_cache) > self._max_cached_exposures:
                self._exposure_cache.popitem(last=False)
            return exposure

        new_exposures = {factor.factor_id: load(factor) for factor in new_refs}
        reference_mask = self._mask_cache[new_refs[0].mask_data_ref.artifact_id]
        for factor in new_refs:
            if factor.mask_data_ref != new_refs[0].mask_data_ref:
                raise CorrelationProtocolMismatch(
                    "all factors must use the same evaluation mask artifact"
                )

        old_count = len(old_ids)
        new_count = len(new_refs)
        new_correlation = np.full(
            (new_count, old_count + new_count), np.nan, dtype=np.float32
        )
        new_valid = np.zeros((new_count, old_count + new_count), dtype=np.int64)
        new_overlap = np.zeros((new_count, old_count + new_count), dtype=np.int64)
        computed_pairs = 0

        old_ref_by_id = {ref.factor_id: ref for ref in store.factor_refs}
        for old_index, old_id in enumerate(old_ids):
            old_exposure = load(old_ref_by_id[old_id])
            for new_index, new_ref in enumerate(new_refs):
                new_exposure = new_exposures[new_ref.factor_id]
                pair_started = perf_counter()
                value, valid_count, overlap_count = _pair_correlation(
                    new_exposure,
                    old_exposure,
                    reference_mask,
                    self.evaluation_dates,
                    self.protocol,
                )
                pairwise_seconds += perf_counter() - pair_started
                new_correlation[new_index, old_index] = np.float32(value)
                new_valid[new_index, old_index] = valid_count
                new_overlap[new_index, old_index] = overlap_count
                computed_pairs += 1

        for new_index, new_ref in enumerate(new_refs):
            new_exposure = new_exposures[new_ref.factor_id]
            for later_index in range(new_index + 1, new_count):
                later_ref = new_refs[later_index]
                pair_started = perf_counter()
                value, valid_count, overlap_count = _pair_correlation(
                    new_exposure,
                    new_exposures[later_ref.factor_id],
                    reference_mask,
                    self.evaluation_dates,
                    self.protocol,
                )
                pairwise_seconds += perf_counter() - pair_started
                new_correlation[new_index, old_count + later_index] = np.float32(value)
                new_valid[new_index, old_count + later_index] = valid_count
                new_overlap[new_index, old_count + later_index] = overlap_count
                computed_pairs += 1
            pair_started = perf_counter()
            self_value, self_valid, self_overlap = _pair_correlation(
                new_exposure,
                new_exposure,
                reference_mask,
                self.evaluation_dates,
                self.protocol,
            )
            pairwise_seconds += perf_counter() - pair_started
            new_correlation[new_index, old_count + new_index] = np.float32(
                1.0 if math.isfinite(self_value) else math.nan
            )
            new_valid[new_index, old_count + new_index] = self_valid
            new_overlap[new_index, old_count + new_index] = self_overlap

        combined_size = old_count + new_count
        combined_correlation = np.full(
            (combined_size, combined_size), np.nan, dtype=np.float32
        )
        combined_valid = np.zeros((combined_size, combined_size), dtype=np.int64)
        combined_overlap = np.zeros((combined_size, combined_size), dtype=np.int64)
        if old_count:
            combined_correlation[:old_count, :old_count] = store.correlation_matrix
            combined_valid[:old_count, :old_count] = store.valid_date_count_matrix
            combined_overlap[:old_count, :old_count] = store.overlap_date_count_matrix
            combined_correlation[old_count:, :old_count] = new_correlation[
                :, :old_count
            ]
            combined_valid[old_count:, :old_count] = new_valid[:, :old_count]
            combined_overlap[old_count:, :old_count] = new_overlap[:, :old_count]
            combined_correlation[:old_count, old_count:] = new_correlation[
                :, :old_count
            ].T
            combined_valid[:old_count, old_count:] = new_valid[:, :old_count].T
            combined_overlap[:old_count, old_count:] = new_overlap[:, :old_count].T

        new_correlation_block = new_correlation[:, old_count:].copy()
        new_valid_block = new_valid[:, old_count:].copy()
        new_overlap_block = new_overlap[:, old_count:].copy()
        lower_rows, lower_columns = np.tril_indices(new_count, k=-1)
        new_correlation_block[lower_rows, lower_columns] = new_correlation_block[
            lower_columns, lower_rows
        ]
        new_valid_block[lower_rows, lower_columns] = new_valid_block[
            lower_columns, lower_rows
        ]
        new_overlap_block[lower_rows, lower_columns] = new_overlap_block[
            lower_columns, lower_rows
        ]
        combined_correlation[old_count:, old_count:] = new_correlation_block
        combined_valid[old_count:, old_count:] = new_valid_block
        combined_overlap[old_count:, old_count:] = new_overlap_block
        result_store = CorrelationStore(
            protocol=self.protocol,
            factor_refs=all_refs,
            correlation_matrix=combined_correlation,
            valid_date_count_matrix=combined_valid,
            overlap_date_count_matrix=combined_overlap,
        )
        self._last_timing = CorrelationTiming(
            total_seconds=perf_counter() - started_at,
            exposure_decode_seconds=decode_seconds,
            pairwise_correlation_seconds=pairwise_seconds,
            computed_pair_count=computed_pairs,
        )
        return CorrelationUpdateResult(
            store=result_store,
            computed_pair_count=computed_pairs,
        )

    def load(self, snapshot: CorrelationSnapshot) -> CorrelationStore:
        """Load a versioned snapshot after checking this engine's protocol."""
        return CorrelationStore.load(
            self._artifacts,
            snapshot,
            expected_protocol=self.protocol,
        )

    def load_snapshot(self, metadata_ref: ArtifactRef) -> CorrelationStore:
        """Load a snapshot through its metadata artifact reference."""
        try:
            snapshot = from_json(
                self._artifacts.get_bytes(metadata_ref), CorrelationSnapshot
            )
        except Exception as error:  # noqa: BLE001 - normalize metadata failures
            if isinstance(error, CorrelationError):
                raise
            raise CorrelationDataError(
                "invalid correlation snapshot metadata"
            ) from error
        return self.load(snapshot)

    def _validate_factor_sequence(
        self, factors: tuple[CorrelationFactorRef, ...]
    ) -> None:
        """Validate provenance and common mask identity for a factor sequence."""
        mask_ref: ArtifactRef | None = None
        for factor in factors:
            _validate_protocol_factor(factor, self.protocol)
            if mask_ref is None:
                mask_ref = factor.mask_data_ref
            elif factor.mask_data_ref != mask_ref:
                raise CorrelationProtocolMismatch(
                    "all factors must use the same evaluation mask artifact"
                )


class CorrelationNeighborhoodBuilder:
    """Project complete signed facts into deterministic local ``W_corr``."""

    def build(
        self,
        correlation_matrix: np.ndarray,
        valid_date_count_matrix: np.ndarray,
        overlap_date_count_matrix: np.ndarray,
        *,
        edge_threshold: float,
        max_neighbors: int,
        min_valid_date_count: int,
        min_valid_date_ratio: float,
    ) -> np.ndarray:
        """Apply reliability, threshold, Top-K, and union symmetrization."""
        if not isinstance(correlation_matrix, np.ndarray):
            raise CoreValidationError("correlation_matrix must be a NumPy array")
        if (
            correlation_matrix.ndim != 2
            or correlation_matrix.shape[0] != correlation_matrix.shape[1]
        ):
            raise CoreValidationError("correlation_matrix must be square")
        shape = correlation_matrix.shape
        if (
            valid_date_count_matrix.shape != shape
            or overlap_date_count_matrix.shape != shape
        ):
            raise CoreValidationError(
                "date count matrices must match correlation shape"
            )
        if (
            type(edge_threshold) is not float
            or not math.isfinite(edge_threshold)
            or not 0.0 <= edge_threshold <= 1.0
        ):
            raise CoreValidationError("edge_threshold must be a float in [0, 1]")
        if type(max_neighbors) is not int or max_neighbors <= 0:
            raise CoreValidationError("max_neighbors must be a positive integer")
        if type(min_valid_date_count) is not int or min_valid_date_count <= 0:
            raise CoreValidationError("min_valid_date_count must be positive")
        if (
            type(min_valid_date_ratio) is not float
            or not math.isfinite(min_valid_date_ratio)
            or not 0.0 <= min_valid_date_ratio <= 1.0
        ):
            raise CoreValidationError("min_valid_date_ratio must be a float in [0, 1]")

        scores = correlation_similarity(correlation_matrix)
        with np.errstate(divide="ignore", invalid="ignore"):
            valid_ratio = np.divide(
                valid_date_count_matrix,
                overlap_date_count_matrix,
                out=np.zeros(shape, dtype=np.float64),
                where=overlap_date_count_matrix > 0,
            )
        eligible = (
            np.isfinite(scores)
            & (valid_date_count_matrix >= min_valid_date_count)
            & (valid_ratio >= min_valid_date_ratio)
            & (scores >= edge_threshold)
        )
        np.fill_diagonal(eligible, False)
        selected = np.zeros(shape, dtype=np.bool_)
        for row_index in range(shape[0]):
            candidates = np.flatnonzero(eligible[row_index])
            ordered = sorted(
                (int(index) for index in candidates),
                key=lambda index: (-float(scores[row_index, index]), index),
            )
            selected[row_index, ordered[:max_neighbors]] = True
        selected |= selected.T
        return np.where(selected & eligible, scores, 0.0)


def correlation_similarity(correlation_matrix: np.ndarray) -> np.ndarray:
    """Return ``abs(C)`` while leaving missing facts as ``nan``."""
    if not isinstance(correlation_matrix, np.ndarray):
        raise CoreValidationError("correlation_matrix must be a NumPy array")
    similarity = np.abs(correlation_matrix.astype(np.float64, copy=False))
    if np.any(np.isinf(similarity)) or np.any(similarity > 1.0 + 1e-12):
        raise CoreValidationError("correlation values must be within [-1, 1]")
    return np.minimum(similarity, 1.0)


def correlation_distance(correlation_matrix: np.ndarray) -> np.ndarray:
    """Return ``1 - abs(C)`` while preserving missing facts as ``nan``."""
    return 1.0 - correlation_similarity(correlation_matrix)


def _encode_store_matrices(store: CorrelationStore) -> bytes:
    """Encode dense matrices without object arrays or pickle."""
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        correlation=store.correlation_matrix.astype(np.float32, copy=False),
        valid_date_count=store.valid_date_count_matrix.astype(np.int64, copy=False),
        overlap_date_count=store.overlap_date_count_matrix.astype(np.int64, copy=False),
    )
    return stream.getvalue()


def persist_store(
    store: CorrelationStore,
    artifacts: CorrelationArtifactRepository,
    *,
    snapshot_version: str,
) -> CorrelationSnapshot:
    """Persist a store's dense arrays and immutable metadata manifest."""
    if not isinstance(store, CorrelationStore):
        raise CoreValidationError("store must be CorrelationStore")
    if type(snapshot_version) is not str or not snapshot_version.strip():
        raise CoreValidationError("snapshot_version must be a non-empty string")
    matrix_ref = artifacts.put_bytes(
        _encode_store_matrices(store),
        kind=_CORRELATION_MATRIX_KIND,
        media_type=CORRELATION_MATRIX_MEDIA_TYPE,
    )
    shape = store.correlation_matrix.shape
    snapshot_id = _snapshot_fingerprint(
        snapshot_version=snapshot_version,
        protocol=store.protocol,
        factor_refs=store.factor_refs,
        matrix_ref=matrix_ref,
        matrix_shape=(shape[0], shape[1]),
        correlation_dtype="float32",
        count_dtype="int64",
    )
    snapshot_without_metadata_ref = CorrelationSnapshot(
        snapshot_version=snapshot_version,
        snapshot_id=snapshot_id,
        protocol=store.protocol,
        factor_refs=store.factor_refs,
        matrix_ref=matrix_ref,
        matrix_shape=(shape[0], shape[1]),
        correlation_dtype="float32",
        count_dtype="int64",
        metadata_ref=None,
    )
    metadata_ref = artifacts.put_bytes(
        canonical_json_bytes(snapshot_without_metadata_ref),
        kind=_CORRELATION_SNAPSHOT_KIND,
        media_type="application/json",
    )
    return CorrelationSnapshot(
        snapshot_version=snapshot_version,
        snapshot_id=snapshot_id,
        protocol=store.protocol,
        factor_refs=store.factor_refs,
        matrix_ref=matrix_ref,
        matrix_shape=(shape[0], shape[1]),
        correlation_dtype="float32",
        count_dtype="int64",
        metadata_ref=metadata_ref,
    )


def _read_store_matrices(
    artifacts: CorrelationArtifactRepository, snapshot: CorrelationSnapshot
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and strictly validate the dense matrix payload."""
    try:
        with np.load(
            io.BytesIO(artifacts.get_bytes(snapshot.matrix_ref)), allow_pickle=False
        ) as archive:
            expected = {"correlation", "valid_date_count", "overlap_date_count"}
            if set(archive.files) != expected:
                raise CorrelationDataError(
                    "correlation matrix artifact has invalid arrays"
                )
            correlation = np.asarray(archive["correlation"])
            valid = np.asarray(archive["valid_date_count"])
            overlap = np.asarray(archive["overlap_date_count"])
    except CorrelationError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize NumPy/repository failures
        raise CorrelationDataError("invalid correlation matrix artifact") from error
    if correlation.dtype != np.dtype("float32"):
        raise CorrelationDataError("persisted correlation matrix must use float32")
    if valid.dtype != np.dtype("int64") or overlap.dtype != np.dtype("int64"):
        raise CorrelationDataError("persisted date counts must use int64")
    shape = snapshot.matrix_shape
    if correlation.shape != shape or valid.shape != shape or overlap.shape != shape:
        raise CorrelationDataError("persisted matrix shape does not match metadata")
    if np.any(valid < 0) or np.any(overlap < 0) or np.any(valid > overlap):
        raise CorrelationDataError("persisted date counts are invalid")
    finite_correlation = np.isfinite(correlation)
    if np.any(np.isinf(correlation)) or np.any(
        finite_correlation & ((correlation < -1.0) | (correlation > 1.0))
    ):
        raise CorrelationDataError("persisted correlations are outside [-1, 1]")
    return correlation, valid, overlap


def load_store(
    artifacts: CorrelationArtifactRepository,
    snapshot: CorrelationSnapshot,
    *,
    expected_protocol: GraphCorrelationProtocol,
) -> CorrelationStore:
    """Load a snapshot only when all protocol metadata agrees."""
    if not isinstance(snapshot, CorrelationSnapshot):
        raise CoreValidationError("snapshot must be CorrelationSnapshot")
    if snapshot.protocol != expected_protocol:
        raise CorrelationProtocolMismatch("correlation snapshot protocol mismatch")
    factor_refs = _require_factor_sequence(snapshot.factor_refs, "snapshot.factor_refs")
    for factor in factor_refs:
        _validate_protocol_factor(factor, expected_protocol)
    if factor_refs and any(
        factor.mask_data_ref != factor_refs[0].mask_data_ref for factor in factor_refs
    ):
        raise CorrelationProtocolMismatch(
            "correlation snapshot factors use different mask artifacts"
        )
    correlation, valid, overlap = _read_store_matrices(artifacts, snapshot)
    return CorrelationStore(
        protocol=expected_protocol,
        factor_refs=snapshot.factor_refs,
        correlation_matrix=correlation,
        valid_date_count_matrix=valid,
        overlap_date_count_matrix=overlap,
        snapshot_version=snapshot.snapshot_version,
    )
