"""Factor expression execution and deterministic factor calculation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cosmosalpha.agents import FactorCalculationResult, FactorSpec
from cosmosalpha.core import (
    COLUMNAR_RECORDS_MEDIA_TYPE,
    ArtifactRef,
    ColumnarRecordsError,
    ColumnarTable,
    ObjectRef,
    decode_record_artifact,
    decode_record_artifact_table,
    encode_columnar_columns,
)
from cosmosalpha.data import DataSnapshot
from cosmosalpha.factors.expression import (
    DEFAULT_TIME_WINDOWS,
    CallNode,
    ExpressionNode,
    FieldNode,
    NumberNode,
    normalize_expression,
    parse_expression,
)

CALCULATION_VERSION = "factor-calculator/1.1"
FACTOR_VALUE_ARTIFACT_KIND = "factor_values"
FACTOR_VALUE_MEDIA_TYPE = COLUMNAR_RECORDS_MEDIA_TYPE


class FactorExecutionError(Exception):
    """Raised when a factor expression cannot be evaluated."""


class FactorCalculationError(FactorExecutionError):
    """Raised when FactorCalculator cannot produce a valid result."""


class ArtifactRepository(Protocol):
    """Minimal artifact repository required by FactorCalculator."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Return artifact bytes."""

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist artifact bytes and return a reference."""


@dataclass(frozen=True, slots=True)
class FactorCalculationTiming:
    """Non-persisted timing from the latest factor calculation."""

    total_seconds: float
    frame_load_seconds: float
    expression_seconds: float
    persistence_seconds: float


@dataclass(frozen=True, slots=True)
class OperatorContext:
    """Precomputed grouping metadata for factor operators."""

    date_codes: np.ndarray
    instrument_codes: np.ndarray
    instrument_indices: Mapping[int, np.ndarray]
    date_indices: Mapping[int, np.ndarray]
    industry_codes: np.ndarray | None
    eligible_mask: np.ndarray
    n_samples: int

    @classmethod
    def from_frame(cls, frame: CalculationFrame) -> OperatorContext:
        """Build context arrays from a validated calculation frame."""
        date_codes, _ = _factorize_preserving_order(frame.datetimes)
        instrument_codes, _ = _factorize_preserving_order(frame.instruments)
        industry = frame.context_fields.get("industry_id")
        return cls(
            date_codes=date_codes,
            instrument_codes=instrument_codes,
            instrument_indices=_build_index_map(instrument_codes),
            date_indices=_build_index_map(date_codes),
            industry_codes=industry,
            eligible_mask=frame.eligible_mask.copy(),
            n_samples=frame.n_samples,
        )


@dataclass(frozen=True, slots=True)
class CalculationFrame:
    """Read-only in-process view used by expression execution."""

    datetimes: Sequence[str]
    instruments: Sequence[str]
    fields: Mapping[str, np.ndarray]
    context_fields: Mapping[str, np.ndarray]
    eligible_mask: np.ndarray

    def __post_init__(self) -> None:
        """Validate frame dimensions and sorted unique index."""
        if len(self.datetimes) == 0:
            raise FactorCalculationError("EMPTY_DATASET: calculation frame is empty")
        if len(self.datetimes) != len(self.instruments):
            raise FactorCalculationError("INVALID_INDEX_SCHEMA: index length mismatch")
        datetimes = np.asarray(self.datetimes).astype(str, copy=False)
        instruments = np.asarray(self.instruments).astype(str, copy=False)
        same_date = datetimes[1:] == datetimes[:-1]
        same_instrument = instruments[1:] == instruments[:-1]
        if np.any(same_date & same_instrument):
            raise FactorCalculationError("DUPLICATE_SAMPLE_INDEX: duplicate samples")
        if np.any(
            np.char.less(datetimes[1:], datetimes[:-1])
            | (same_date & np.char.less(instruments[1:], instruments[:-1]))
        ):
            raise FactorCalculationError("UNSORTED_SAMPLE_INDEX: index is not sorted")
        if self.eligible_mask.shape != (len(datetimes),):
            raise FactorCalculationError("MASK_INDEX_MISMATCH: eligible mask mismatch")
        for name, values in (*self.fields.items(), *self.context_fields.items()):
            if values.shape != (len(datetimes),):
                raise FactorCalculationError(f"{name} length does not match frame")

    @property
    def n_samples(self) -> int:
        """Return sample row count."""
        return len(self.datetimes)

    @classmethod
    def from_records(
        cls,
        *,
        market_records: Sequence[Mapping[str, object]],
        exposure_records: Sequence[Mapping[str, object]],
        mask_records: Sequence[Mapping[str, object]],
        required_fields: tuple[str, ...],
    ) -> CalculationFrame:
        """Build a calculation frame from aligned JSON records."""
        if not market_records:
            raise FactorCalculationError("EMPTY_DATASET: market records are empty")
        market_keys = tuple(_record_key(row) for row in market_records)
        exposure_keys = tuple(_record_key(row) for row in exposure_records)
        mask_keys = tuple(_record_key(row) for row in mask_records)
        if market_keys != exposure_keys:
            raise FactorCalculationError("EXPOSURE_INDEX_MISMATCH: indexes differ")
        if market_keys != mask_keys:
            raise FactorCalculationError("MASK_INDEX_MISMATCH: indexes differ")
        datetimes = tuple(key[0] for key in market_keys)
        instruments = tuple(key[1] for key in market_keys)
        fields = {
            field: _numeric_array(market_records, field)
            for field in sorted(set(required_fields))
        }
        context_fields = {
            field: _numeric_array(exposure_records, field, missing_ok=True)
            for field in ("industry_id", "log_mkt_cap", "ret20", "turn20", "vol20")
            if any(field in row for row in exposure_records)
        }
        eligible = np.array(
            [row.get("eligible_mask") is True for row in mask_records],
            dtype=bool,
        )
        return cls(
            datetimes=datetimes,
            instruments=instruments,
            fields=fields,
            context_fields=context_fields,
            eligible_mask=eligible,
        )


@dataclass(frozen=True, slots=True)
class _CachedCalculationFrame:
    """One immutable snapshot frame and its reusable operator grouping."""

    frame: CalculationFrame
    context: OperatorContext
    datetimes: np.ndarray
    instruments: np.ndarray


class FactorCalculationCache:
    """Share immutable calculation frames across factors and Graph workers."""

    def __init__(self) -> None:
        """Create an unbounded run-local cache keyed by immutable artifacts."""
        self._entries: dict[tuple[str, str, str, str], _CachedCalculationFrame] = {}
        self._lock = Lock()

    @property
    def entry_count(self) -> int:
        """Return the number of cached snapshot frames."""
        with self._lock:
            return len(self._entries)

    def get_or_create(
        self,
        key: tuple[str, str, str, str],
        loader: Callable[[], CalculationFrame],
    ) -> _CachedCalculationFrame:
        """Return one frame, allowing only one thread to build a cache miss."""
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached
            frame = loader()
            datetimes = np.asarray(frame.datetimes)
            instruments = np.asarray(frame.instruments)
            datetimes.setflags(write=False)
            instruments.setflags(write=False)
            cached = _CachedCalculationFrame(
                frame=frame,
                context=OperatorContext.from_frame(frame),
                datetimes=datetimes,
                instruments=instruments,
            )
            self._entries[key] = cached
            return cached


class FactorCalculator:
    """Evaluate validated factor specifications against data snapshot artifacts."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        calculation_cache: FactorCalculationCache | None = None,
    ) -> None:
        """Create a factor calculator."""
        self._artifacts = artifact_repository
        self._cache = calculation_cache or FactorCalculationCache()
        self._last_timing = FactorCalculationTiming(0.0, 0.0, 0.0, 0.0)

    @property
    def last_timing(self) -> FactorCalculationTiming:
        """Return timing from the latest successful calculation."""
        return self._last_timing

    def calculate(
        self,
        factor_spec: FactorSpec,
        data_snapshot: DataSnapshot,
        *,
        validation_report_ref: ObjectRef,
    ) -> FactorCalculationResult:
        """Calculate factor values and persist the output artifact."""
        started_at = perf_counter()
        if not isinstance(factor_spec, FactorSpec):
            raise FactorCalculationError("factor_spec must be FactorSpec")
        if not isinstance(data_snapshot, DataSnapshot):
            raise FactorCalculationError("data_snapshot must be DataSnapshot")
        if not isinstance(validation_report_ref, ObjectRef):
            raise FactorCalculationError("validation_report_ref must be ObjectRef")
        if not factor_spec.factor_id:
            raise FactorCalculationError("factor_spec must be validated")

        phase_started_at = perf_counter()
        cached_frame = self._load_cached_frame(
            data_snapshot,
            factor_spec.required_fields,
        )
        frame = cached_frame.frame
        frame_load_seconds = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        raw_values = evaluate_factor_spec(
            factor_spec,
            frame,
            context=cached_frame.context,
        )
        values = raw_values.copy()
        values[~frame.eligible_mask] = np.nan
        expression_seconds = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        output_ref = self._artifacts.put_bytes(
            encode_columnar_columns(
                {
                    "datetime": cached_frame.datetimes,
                    "factor_value": values,
                    "instrument": cached_frame.instruments,
                }
            ),
            kind=FACTOR_VALUE_ARTIFACT_KIND,
            media_type=FACTOR_VALUE_MEDIA_TYPE,
        )
        persistence_seconds = perf_counter() - phase_started_at
        missing_count = int(np.count_nonzero(~np.isfinite(values)))
        result = FactorCalculationResult(
            factor_id=factor_spec.factor_id,
            validation_report_ref=validation_report_ref,
            data_snapshot_id=data_snapshot.snapshot_id,
            mask_data_ref=data_snapshot.mask_data.mask_data_ref,
            factor_values_ref=output_ref,
            date_range=(
                date.fromisoformat(data_snapshot.request.start_date),
                date.fromisoformat(data_snapshot.request.end_date),
            ),
            sample_count=frame.n_samples,
            missing_count=missing_count,
            eligible_mask_applied=True,
            calculation_version=CALCULATION_VERSION,
            schema_version="factor-calculation-result/1.0",
        )
        self._last_timing = FactorCalculationTiming(
            total_seconds=perf_counter() - started_at,
            frame_load_seconds=frame_load_seconds,
            expression_seconds=expression_seconds,
            persistence_seconds=persistence_seconds,
        )
        return result

    def load_frame(
        self, data_snapshot: DataSnapshot, required_fields: tuple[str, ...]
    ) -> CalculationFrame:
        """Load one shared calculation frame without evaluating a factor."""
        return self._load_cached_frame(data_snapshot, required_fields).frame

    def _load_cached_frame(
        self,
        data_snapshot: DataSnapshot,
        required_fields: tuple[str, ...],
    ) -> _CachedCalculationFrame:
        """Resolve and validate one full-snapshot cached calculation frame."""
        if not isinstance(data_snapshot, DataSnapshot):
            raise FactorCalculationError("data_snapshot must be DataSnapshot")
        if not isinstance(required_fields, tuple) or not all(
            type(field) is str and field for field in required_fields
        ):
            raise FactorCalculationError(
                "required_fields must be a tuple of non-empty strings"
            )
        missing = sorted(set(required_fields) - set(data_snapshot.market_data.fields))
        if missing:
            raise FactorCalculationError(
                "MISSING_REQUIRED_FIELD: " + ", ".join(missing)
            )
        key = (
            data_snapshot.snapshot_id,
            data_snapshot.market_data.data_ref.artifact_id,
            data_snapshot.exposure_data.data_ref.artifact_id,
            data_snapshot.mask_data.mask_data_ref.artifact_id,
        )
        return self._cache.get_or_create(
            key,
            lambda: self._load_frame_uncached(data_snapshot),
        )

    def _load_frame_uncached(self, data_snapshot: DataSnapshot) -> CalculationFrame:
        """Decode all reusable snapshot columns on a cache miss."""
        market_table = _read_table(
            self._artifacts.get_bytes(data_snapshot.market_data.data_ref),
            "market_data",
        )
        exposure_table = _read_table(
            self._artifacts.get_bytes(data_snapshot.exposure_data.data_ref),
            "exposure_data",
        )
        mask_table = _read_table(
            self._artifacts.get_bytes(data_snapshot.mask_data.mask_data_ref),
            "mask_data",
        )
        datetimes, instruments = _table_index(market_table, "market_data")
        _validate_table_alignment(
            exposure_table,
            datetimes,
            instruments,
            "exposure_data",
        )
        _validate_table_alignment(
            mask_table,
            datetimes,
            instruments,
            "mask_data",
        )
        fields = {
            field: _table_numeric_array(market_table, field)
            for field in data_snapshot.market_data.fields
        }
        context_fields = {
            field: _table_numeric_array(exposure_table, field, missing_ok=True)
            for field in ("industry_id", "log_mkt_cap", "ret20", "turn20", "vol20")
            if field in exposure_table.columns
        }
        eligible = _table_true_array(mask_table, "eligible_mask")
        return CalculationFrame(
            datetimes=datetimes,
            instruments=instruments,
            fields=fields,
            context_fields=context_fields,
            eligible_mask=eligible,
        )


def evaluate_factor_spec(
    factor_spec: FactorSpec,
    frame: CalculationFrame,
    *,
    context: OperatorContext | None = None,
) -> np.ndarray:
    """Evaluate a validated factor spec against a calculation frame."""
    node = parse_expression(factor_spec.expression)
    if normalize_expression(node) != factor_spec.expression:
        raise FactorCalculationError("factor_spec.expression must be canonical")
    missing = sorted(set(factor_spec.required_fields) - set(frame.fields))
    if missing:
        raise FactorCalculationError("MISSING_REQUIRED_FIELD: " + ", ".join(missing))
    result = _evaluate_node(node, frame, context or OperatorContext.from_frame(frame))
    if result.shape != (frame.n_samples,):
        raise FactorCalculationError("operator output shape mismatch")
    return result


def _evaluate_node(
    node: ExpressionNode, frame: CalculationFrame, context: OperatorContext
) -> np.ndarray:
    if isinstance(node, FieldNode):
        try:
            return frame.fields[node.name].copy()
        except KeyError as error:
            raise FactorCalculationError(
                f"MISSING_REQUIRED_FIELD: {node.name}"
            ) from error
    if isinstance(node, NumberNode):
        return np.full(frame.n_samples, float(node.value), dtype=float)
    if isinstance(node, CallNode):
        return _evaluate_call(node, frame, context)
    raise FactorCalculationError("unsupported expression node")


def _evaluate_call(
    node: CallNode, frame: CalculationFrame, context: OperatorContext
) -> np.ndarray:
    name = node.function
    args = node.arguments
    window = _window_arg(args) if _is_window_operator(name) else None
    value_args = args[:-1] if window is not None else args
    values = tuple(_evaluate_node(arg, frame, context) for arg in value_args)

    if name in {"add", "sub", "mul", "div"}:
        _require_value_count(name, values, 2)
        return _elementwise_binary(name, values[0], values[1])
    if name in {"abs", "sqrt", "log", "inv", "sigmoid"}:
        _require_value_count(name, values, 1)
        return _elementwise_unary(name, values[0])
    if name == "rank":
        _require_value_count(name, values, 1)
        return _cross_sectional_rank(values[0], context)
    if name == "rank_sub":
        _require_value_count(name, values, 2)
        return _cross_sectional_rank(values[0], context) - _cross_sectional_rank(
            values[1], context
        )
    if name == "rank_div":
        _require_value_count(name, values, 2)
        numerator = _cross_sectional_rank(values[0], context)
        denominator = _cross_sectional_rank(values[1], context)
        return _safe_divide(numerator, denominator)
    if name == "indneutralize":
        _require_value_count(name, values, 1)
        return _indneutralize(values[0], context)
    if window is None:
        raise FactorCalculationError(f"UNKNOWN_OPERATOR: {name}")
    if name == "delay":
        _require_value_count(name, values, 1)
        return _delay(values[0], context, window)
    if name == "delta":
        _require_value_count(name, values, 1)
        delayed = _delay(values[0], context, window)
        return _finite_binary(values[0], delayed, lambda x, y: x - y)
    if name == "corr":
        _require_value_count(name, values, 2)
        return _rolling_binary(values[0], values[1], context, window, _window_corr)
    if name == "cov":
        _require_value_count(name, values, 2)
        return _rolling_binary(values[0], values[1], context, window, _window_cov)
    if name == "decay_linear":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_decay_linear)
    if name == "ts_min":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_min)
    if name == "ts_max":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_max)
    if name == "ts_argmin":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_argmin)
    if name == "ts_argmax":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_argmax)
    if name == "ts_rank":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_rank_last)
    if name == "ts_sum":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_sum)
    if name == "ts_stddev":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_std)
    if name == "ts_zscore":
        _require_value_count(name, values, 1)
        return _rolling_unary(values[0], context, window, _window_zscore)
    raise FactorCalculationError(f"UNKNOWN_OPERATOR: {name}")


def _require_value_count(
    operator: str, values: tuple[np.ndarray, ...], expected: int
) -> None:
    if len(values) != expected:
        raise FactorCalculationError(f"INVALID_ARITY: {operator} expects {expected}")


def _is_window_operator(name: str) -> bool:
    return name in {
        "delay",
        "delta",
        "corr",
        "cov",
        "decay_linear",
        "ts_min",
        "ts_max",
        "ts_argmin",
        "ts_argmax",
        "ts_rank",
        "ts_sum",
        "ts_stddev",
        "ts_zscore",
    }


def _window_arg(args: tuple[ExpressionNode, ...]) -> int:
    if (
        not args
        or not isinstance(args[-1], NumberNode)
        or type(args[-1].value) is not int
    ):
        raise FactorCalculationError("INVALID_PARAMETER_TYPE: window must be int")
    window = args[-1].value
    if window <= 0:
        raise FactorCalculationError("UNSUPPORTED_WINDOW: window must be positive")
    if window not in DEFAULT_TIME_WINDOWS:
        raise FactorCalculationError(f"UNSUPPORTED_WINDOW: {window}")
    return window


def _elementwise_binary(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if name == "add":
        return _finite_binary(x, y, lambda left, right: left + right)
    if name == "sub":
        return _finite_binary(x, y, lambda left, right: left - right)
    if name == "mul":
        return _finite_binary(x, y, lambda left, right: left * right)
    if name == "div":
        return _safe_divide(x, y)
    raise FactorCalculationError(f"UNKNOWN_OPERATOR: {name}")


def _elementwise_unary(name: str, x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x)
    if name == "abs":
        out[valid] = np.abs(x[valid])
        return out
    if name == "sqrt":
        valid &= x >= 0
        out[valid] = np.sqrt(x[valid])
        return out
    if name == "log":
        valid &= x > 0
        out[valid] = np.log(x[valid])
        return out
    if name == "inv":
        valid &= np.abs(x) > 1e-12
        out[valid] = 1.0 / x[valid]
        return out
    if name == "sigmoid":
        out[valid] = 1.0 / (1.0 + np.exp(-x[valid]))
        return out
    raise FactorCalculationError(f"UNKNOWN_OPERATOR: {name}")


def _finite_binary(
    x: np.ndarray,
    y: np.ndarray,
    operation: object,
) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        out[valid] = operation(x[valid], y[valid])  # type: ignore[operator]
    return out


def _safe_divide(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 1e-12)
    if valid.any():
        out[valid] = x[valid] / y[valid]
    return out


def _cross_sectional_rank(x: np.ndarray, context: OperatorContext) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    for indices in context.date_indices.values():
        eligible = context.eligible_mask[indices] & np.isfinite(x[indices])
        if not eligible.any():
            continue
        local_indices = indices[eligible]
        out[local_indices] = _average_rank_pct(x[local_indices])
    return out


def _average_rank_pct(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    finite_values = values[finite]
    order = np.argsort(finite_values, kind="stable")
    sorted_values = finite_values[order]
    ranks = np.empty(finite_values.size, dtype=float)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([finite_values.size], dtype=np.int64)))
    average_ranks = (starts + 1 + ends) / 2.0 / finite_values.size
    ranks[order] = np.repeat(average_ranks, ends - starts)
    out[finite] = ranks
    return out


def _indneutralize(x: np.ndarray, context: OperatorContext) -> np.ndarray:
    if context.industry_codes is None:
        raise FactorCalculationError("MISSING_OPERATOR_CONTEXT: industry_id")
    out = np.full(x.shape, np.nan, dtype=float)
    industry = context.industry_codes
    for indices in context.date_indices.values():
        valid = context.eligible_mask[indices] & np.isfinite(x[indices])
        valid &= np.isfinite(industry[indices])
        if not valid.any():
            continue
        local_indices = indices[valid]
        values = x[local_indices]
        industries = np.rint(industry[local_indices]).astype(np.int64, copy=False)
        unique, inverse = np.unique(industries, return_inverse=True)
        sums = np.bincount(inverse, weights=values, minlength=unique.size)
        counts = np.bincount(inverse, minlength=unique.size)
        means = sums / counts
        out[local_indices] = values - means[inverse]
    return out


def _delay(x: np.ndarray, context: OperatorContext, window: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    for indices in context.instrument_indices.values():
        if indices.size <= window:
            continue
        source = x[indices[:-window]]
        target_indices = indices[window:]
        valid = np.isfinite(source)
        out[target_indices[valid]] = source[valid]
    return out


def _rolling_unary(
    x: np.ndarray,
    context: OperatorContext,
    window: int,
    reducer: object,
) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    for indices in context.instrument_indices.values():
        windows = _windows_for_group(x[indices], window)
        values = reducer(windows, window)  # type: ignore[operator]
        out[indices] = values
    return out


def _rolling_binary(
    x: np.ndarray,
    y: np.ndarray,
    context: OperatorContext,
    window: int,
    reducer: object,
) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    for indices in context.instrument_indices.values():
        x_windows = _windows_for_group(x[indices], window)
        y_windows = _windows_for_group(y[indices], window)
        values = reducer(x_windows, y_windows, window)  # type: ignore[operator]
        out[indices] = values
    return out


def _windows_for_group(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.empty(values.size + window - 1, dtype=float)
    padded[: window - 1] = np.nan
    padded[window - 1 :] = values
    return sliding_window_view(padded, window)


def _min_periods(window: int) -> int:
    return int(math.ceil(window * 2.0 / 3.0))


def _valid_rows(
    windows: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(windows)
    count = finite.sum(axis=1)
    valid = count >= _min_periods(window)
    return finite, count, valid


def _window_min(windows: np.ndarray, window: int) -> np.ndarray:
    finite, _count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    out[valid] = np.min(np.where(finite[valid], windows[valid], np.inf), axis=1)
    return out


def _window_max(windows: np.ndarray, window: int) -> np.ndarray:
    finite, _count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    out[valid] = np.max(np.where(finite[valid], windows[valid], -np.inf), axis=1)
    return out


def _window_sum(windows: np.ndarray, window: int) -> np.ndarray:
    finite, _count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    out[valid] = np.sum(windows[valid], axis=1, where=finite[valid], initial=0.0)
    return out


def _window_decay_linear(windows: np.ndarray, window: int) -> np.ndarray:
    finite, _count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    if valid.any():
        weights = np.arange(1, window + 1, dtype=float)
        active = finite[valid]
        denom = np.sum(np.where(active, weights[None, :], 0.0), axis=1)
        numer = np.sum(np.where(active, windows[valid] * weights[None, :], 0.0), axis=1)
        keep = denom > 0
        out[np.flatnonzero(valid)[keep]] = numer[keep] / denom[keep]
    return out


def _window_argmin(windows: np.ndarray, window: int) -> np.ndarray:
    return _window_arg_extreme(windows, window, argmax=False)


def _window_argmax(windows: np.ndarray, window: int) -> np.ndarray:
    return _window_arg_extreme(windows, window, argmax=True)


def _window_arg_extreme(
    windows: np.ndarray, window: int, *, argmax: bool
) -> np.ndarray:
    finite, _count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    if not valid.any():
        return out
    active = windows[valid]
    active_finite = finite[valid]
    masked = np.where(active_finite, active, -np.inf if argmax else np.inf)
    positions = np.argmax(masked, axis=1) if argmax else np.argmin(masked, axis=1)
    out[valid] = positions.astype(float) + 1.0
    return out


def _window_rank_last(windows: np.ndarray, window: int) -> np.ndarray:
    """Rank each trailing value using the vectorized GPLEARN formulation.

    Adapted from
    ``GPLEARN gp_mining/custom_functions/rolling.py::
    _nan_window_rank_of_last`` while retaining CosmosAlpha's minimum-period
    rule.
    """
    finite, count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    last = windows[:, -1]
    valid &= np.isfinite(last)
    if valid.any():
        active = windows[valid]
        active_finite = finite[valid]
        targets = last[valid]
        less = ((active < targets[:, None]) & active_finite).sum(axis=1)
        equal = ((active == targets[:, None]) & active_finite).sum(axis=1)
        out[np.flatnonzero(valid)] = (less + (equal + 1.0) / 2.0) / count[valid]
    return out


def _window_std(windows: np.ndarray, window: int) -> np.ndarray:
    finite, count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    valid &= count >= 2
    if valid.any():
        out[valid] = _sample_std(
            windows[valid],
            finite[valid],
            count=count[valid],
        )
        out[out < 1e-12] = np.nan
    return out


def _window_zscore(windows: np.ndarray, window: int) -> np.ndarray:
    finite, count, valid = _valid_rows(windows, window)
    out = np.full(windows.shape[0], np.nan, dtype=float)
    last = windows[:, -1]
    valid &= count >= 2
    valid &= np.isfinite(last)
    if valid.any():
        active_count = count[valid]
        means = _finite_mean(
            windows[valid],
            finite[valid],
            count=active_count,
        )
        stds = _sample_std(
            windows[valid],
            finite[valid],
            count=active_count,
            means=means,
        )
        keep = np.isfinite(stds) & (stds >= 1e-12)
        out[np.flatnonzero(valid)[keep]] = (last[valid][keep] - means[keep]) / stds[
            keep
        ]
    return out


def _window_cov(
    x_windows: np.ndarray, y_windows: np.ndarray, window: int
) -> np.ndarray:
    finite = np.isfinite(x_windows) & np.isfinite(y_windows)
    count = finite.sum(axis=1)
    valid = count >= max(_min_periods(window), 2)
    out = np.full(x_windows.shape[0], np.nan, dtype=float)
    if not valid.any():
        return out
    rows = np.flatnonzero(valid)
    x_values = x_windows[valid]
    y_values = y_windows[valid]
    active = finite[valid]
    active_count = count[valid]
    x_mean = _finite_mean(x_values, active, count=active_count)
    y_mean = _finite_mean(y_values, active, count=active_count)
    centered_x = np.where(active, x_values - x_mean[:, None], 0.0)
    centered_y = np.where(active, y_values - y_mean[:, None], 0.0)
    out[rows] = np.sum(centered_x * centered_y, axis=1) / (count[valid] - 1)
    return out


def _window_corr(
    x_windows: np.ndarray, y_windows: np.ndarray, window: int
) -> np.ndarray:
    finite = np.isfinite(x_windows) & np.isfinite(y_windows)
    count = finite.sum(axis=1)
    valid = count >= max(_min_periods(window), 2)
    out = np.full(x_windows.shape[0], np.nan, dtype=float)
    if not valid.any():
        return out
    rows = np.flatnonzero(valid)
    x_values = x_windows[valid]
    y_values = y_windows[valid]
    active = finite[valid]
    active_count = count[valid]
    x_mean = _finite_mean(x_values, active, count=active_count)
    y_mean = _finite_mean(y_values, active, count=active_count)
    centered_x = np.where(active, x_values - x_mean[:, None], 0.0)
    centered_y = np.where(active, y_values - y_mean[:, None], 0.0)
    cov_num = np.sum(centered_x * centered_y, axis=1)
    x_var_num = np.sum(centered_x * centered_x, axis=1)
    y_var_num = np.sum(centered_y * centered_y, axis=1)
    denom = np.sqrt(x_var_num * y_var_num)
    keep = np.isfinite(denom) & (denom >= 1e-12)
    out[rows[keep]] = cov_num[keep] / denom[keep]
    return out


def _finite_mean(
    values: np.ndarray,
    finite: np.ndarray,
    *,
    count: np.ndarray | None = None,
) -> np.ndarray:
    if count is None:
        count = finite.sum(axis=1)
    sums = np.sum(values, axis=1, where=finite, initial=0.0)
    return sums / count


def _sample_std(
    values: np.ndarray,
    finite: np.ndarray,
    *,
    count: np.ndarray | None = None,
    means: np.ndarray | None = None,
) -> np.ndarray:
    if count is None:
        count = finite.sum(axis=1)
    if means is None:
        means = _finite_mean(values, finite, count=count)
    centered = np.where(finite, values - means[:, None], 0.0)
    variance = np.sum(centered * centered, axis=1) / (count - 1)
    return np.sqrt(np.maximum(variance, 0.0))


def _factorize_preserving_order(
    values: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    array = np.asarray(values).astype(str, copy=False)
    unique, first_indices, inverse = np.unique(
        array,
        return_index=True,
        return_inverse=True,
    )
    first_seen_order = np.argsort(first_indices, kind="stable")
    remap = np.empty(unique.size, dtype=np.int64)
    remap[first_seen_order] = np.arange(unique.size, dtype=np.int64)
    labels = tuple(str(value) for value in unique[first_seen_order])
    return remap[inverse], labels


def _build_index_map(codes: np.ndarray) -> Mapping[int, np.ndarray]:
    if codes.size == 0:
        return {}
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_codes)) + 1)
    )
    ends = np.concatenate((starts[1:], np.array([codes.size], dtype=np.int64)))
    return {
        int(sorted_codes[start]): order[start:end]
        for start, end in zip(starts, ends, strict=True)
    }


def _read_table(data: bytes, label: str) -> ColumnarTable:
    try:
        return decode_record_artifact_table(data)
    except ColumnarRecordsError as error:
        raise FactorCalculationError(f"{label}: invalid record artifact") from error


def _table_index(
    table: ColumnarTable,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if table.row_count == 0:
        raise FactorCalculationError("EMPTY_DATASET: calculation frame is empty")
    for field in ("datetime", "instrument"):
        if (
            field not in table.columns
            or table.column_types[field] != "str"
            or not np.all(table.validity[field])
        ):
            raise FactorCalculationError(
                f"{label}: INVALID_INDEX_SCHEMA: invalid row index"
            )
    datetimes = table.columns["datetime"].astype(str, copy=False)
    instruments = table.columns["instrument"].astype(str, copy=False)
    datetimes.setflags(write=False)
    instruments.setflags(write=False)
    return datetimes, instruments


def _validate_table_alignment(
    table: ColumnarTable,
    datetimes: np.ndarray,
    instruments: np.ndarray,
    label: str,
) -> None:
    candidate_dates, candidate_instruments = _table_index(table, label)
    if not np.array_equal(candidate_dates, datetimes) or not np.array_equal(
        candidate_instruments,
        instruments,
    ):
        raise FactorCalculationError(f"{label}: indexes differ")


def _table_numeric_array(
    table: ColumnarTable,
    field: str,
    *,
    missing_ok: bool = False,
) -> np.ndarray:
    if field not in table.columns:
        if missing_ok:
            values = np.full(table.row_count, np.nan, dtype=float)
            values.setflags(write=False)
            return values
        raise FactorCalculationError(f"MISSING_REQUIRED_FIELD: {field}")
    if table.column_types[field] not in {"float", "int"}:
        raise FactorCalculationError(f"NONFINITE_VALUE: {field}")
    values = table.columns[field].astype(np.float64, copy=False)
    valid = table.validity[field]
    if np.any(~valid):
        values = values.copy()
        values[~valid] = np.nan
    values.setflags(write=False)
    return values


def _table_true_array(table: ColumnarTable, field: str) -> np.ndarray:
    if field not in table.columns or table.column_types[field] != "bool":
        values = np.zeros(table.row_count, dtype=bool)
    else:
        values = table.columns[field] & table.validity[field]
    values.setflags(write=False)
    return values


def _record_key(row: Mapping[str, object]) -> tuple[str, str]:
    datetime_value = row.get("datetime")
    instrument_value = row.get("instrument")
    if type(datetime_value) is not str or type(instrument_value) is not str:
        raise FactorCalculationError("INVALID_INDEX_SCHEMA: invalid row index")
    return datetime_value, instrument_value


def _numeric_array(
    records: Sequence[Mapping[str, object]],
    field: str,
    *,
    missing_ok: bool = False,
) -> np.ndarray:
    values = np.full(len(records), np.nan, dtype=float)
    for index, row in enumerate(records):
        if field not in row:
            if missing_ok:
                continue
            raise FactorCalculationError(f"MISSING_REQUIRED_FIELD: {field}")
        value = row[field]
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise FactorCalculationError(f"NONFINITE_VALUE: {field}")
        values[index] = float(value)
    return values


def _read_records(data: bytes, label: str) -> tuple[Mapping[str, object], ...]:
    try:
        return decode_record_artifact(data)
    except ColumnarRecordsError as error:
        raise FactorCalculationError(f"{label}: invalid record artifact") from error
