"""Pure utilities for Qlib point-in-time instrument universe files."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cosmosalpha.core import CoreValidationError
from cosmosalpha.data.instrument_history import canonical_instrument_from_tushare
from cosmosalpha.data.models import canonical_instrument_code


@dataclass(frozen=True, slots=True, order=True)
class QlibInstrumentRange:
    """One inclusive point-in-time membership interval in Qlib text format."""

    symbol: str
    start: str
    end: str

    def __post_init__(self) -> None:
        """Validate the normalized symbol and inclusive ISO date interval."""
        try:
            canonical = canonical_instrument_code(self.symbol)
        except CoreValidationError as error:
            raise ValueError(
                "symbol must be a canonical lower-case Qlib symbol"
            ) from error
        if canonical != self.symbol:
            raise ValueError("symbol must be a canonical lower-case Qlib symbol")
        _parse_date(self.start)
        _parse_date(self.end)
        if self.start > self.end:
            raise ValueError("instrument range start must not be after end")


@dataclass(frozen=True, slots=True)
class UniverseStatistics:
    """Compact statistics for one point-in-time instrument universe."""

    universe_name: str
    requested_start: str
    requested_end: str
    first_active_date: str | None
    last_active_date: str | None
    interval_count: int
    symbol_count: int
    active_day_count: int
    minimum_daily_members: int
    maximum_daily_members: int
    mean_daily_members: float


def read_qlib_calendar(path: Path) -> tuple[str, ...]:
    """Read and validate a daily Qlib calendar file."""
    values = tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
    )
    if not values or any(not value for value in values):
        raise ValueError(f"Qlib calendar is empty or malformed: {path}")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"Qlib calendar must contain sorted unique dates: {path}")
    for value in values:
        _parse_date(value)
    return values


def read_qlib_instrument_ranges(path: Path) -> tuple[QlibInstrumentRange, ...]:
    """Read Qlib ``SYMBOL START END`` instrument membership records."""
    ranges: list[QlibInstrumentRange] = []
    for line_number, raw_line in enumerate(
        path.expanduser().read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(
                f"Malformed Qlib instrument line {line_number} in {path}: {raw_line!r}"
            )
        try:
            symbol = canonical_instrument_code(fields[0])
        except CoreValidationError as error:
            raise ValueError(
                f"Invalid instrument symbol on line {line_number}: {fields[0]!r}"
            ) from error
        ranges.append(QlibInstrumentRange(symbol, fields[1], fields[2]))
    if not ranges:
        raise ValueError(f"Qlib instrument file is empty: {path}")
    return tuple(ranges)


def write_qlib_instrument_ranges(
    path: Path, ranges: Iterable[QlibInstrumentRange]
) -> tuple[QlibInstrumentRange, ...]:
    """Write normalized ranges in the Qlib three-column instrument format."""
    normalized = merge_qlib_instrument_ranges(ranges)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        f"{item.symbol}\t{item.start}\t{item.end}" for item in normalized
    )
    path.write_text(f"{content}\n", encoding="utf-8")
    return normalized


def merge_qlib_instrument_ranges(
    ranges: Iterable[QlibInstrumentRange],
) -> tuple[QlibInstrumentRange, ...]:
    """Merge overlapping or adjacent ranges for every symbol."""
    grouped: dict[str, list[QlibInstrumentRange]] = defaultdict(list)
    for item in ranges:
        grouped[item.symbol].append(item)

    merged: list[QlibInstrumentRange] = []
    for symbol in sorted(grouped):
        pending = sorted(grouped[symbol], key=lambda item: (item.start, item.end))
        current = pending[0]
        for item in pending[1:]:
            if _next_calendar_day(current.end) >= item.start:
                current = QlibInstrumentRange(
                    symbol,
                    current.start,
                    max(current.end, item.end),
                )
            else:
                merged.append(current)
                current = item
        merged.append(current)
    return tuple(merged)


def intersect_qlib_instrument_ranges(
    ranges: Iterable[QlibInstrumentRange],
    *constraints: Iterable[QlibInstrumentRange],
) -> tuple[QlibInstrumentRange, ...]:
    """Intersect membership ranges with one or more Qlib availability sources."""
    result = merge_qlib_instrument_ranges(ranges)
    for constraint in constraints:
        result = _intersect_two_range_sets(
            result, merge_qlib_instrument_ranges(constraint)
        )
    return result


def snapshot_memberships_to_ranges(
    snapshots: Mapping[str, Iterable[str]],
    calendar: Sequence[str],
    *,
    requested_start: str,
    requested_end: str,
) -> tuple[QlibInstrumentRange, ...]:
    """Convert dated constituent snapshots into conservative Qlib intervals.

    A snapshot is considered effective from its own date through the trading day
    before the next available snapshot.  This intentionally avoids backfilling
    constituent knowledge before Tushare reports it.
    """
    _validate_calendar(calendar)
    _parse_date(requested_start)
    _parse_date(requested_end)
    if requested_start > requested_end:
        raise ValueError("requested_start must not be after requested_end")

    dates = tuple(sorted(date_value for date_value in snapshots if date_value))
    for date_value in dates:
        _parse_date(date_value)
    output: list[QlibInstrumentRange] = []
    for index, snapshot_date in enumerate(dates):
        start = _first_calendar_on_or_after(calendar, snapshot_date)
        if start is None:
            continue
        next_snapshot = dates[index + 1] if index + 1 < len(dates) else None
        end = (
            _last_calendar_before(calendar, next_snapshot)
            if next_snapshot is not None
            else calendar[-1]
        )
        if end is None or start > end:
            continue
        start = max(start, requested_start)
        end = min(end, requested_end)
        if start > end:
            continue
        for symbol in sorted(
            {canonical_instrument_code(value) for value in snapshots[snapshot_date]}
        ):
            output.append(QlibInstrumentRange(symbol, start, end))
    return merge_qlib_instrument_ranges(output)


def clip_ranges_to_calendar(
    ranges: Iterable[QlibInstrumentRange], calendar: Sequence[str]
) -> tuple[QlibInstrumentRange, ...]:
    """Clip intervals to dates present in a Qlib calendar."""
    _validate_calendar(calendar)
    clipped: list[QlibInstrumentRange] = []
    for item in ranges:
        start = _first_calendar_on_or_after(calendar, item.start)
        end = _last_calendar_on_or_before(calendar, item.end)
        if start is not None and end is not None and start <= end:
            clipped.append(QlibInstrumentRange(item.symbol, start, end))
    return merge_qlib_instrument_ranges(clipped)


def summarize_qlib_universe(
    universe_name: str,
    ranges: Iterable[QlibInstrumentRange],
    calendar: Sequence[str],
    *,
    requested_start: str,
    requested_end: str,
) -> tuple[UniverseStatistics, tuple[tuple[str, int], ...]]:
    """Calculate daily constituent counts and compact universe statistics."""
    _validate_calendar(calendar)
    active_calendar = tuple(
        value for value in calendar if requested_start <= value <= requested_end
    )
    normalized = clip_ranges_to_calendar(ranges, active_calendar)
    difference = [0] * (len(active_calendar) + 1)
    for item in normalized:
        start = bisect_left(active_calendar, item.start)
        end = bisect_right(active_calendar, item.end)
        difference[start] += 1
        difference[end] -= 1

    counts: list[tuple[str, int]] = []
    active = 0
    for index, current_date in enumerate(active_calendar):
        active += difference[index]
        counts.append((current_date, active))
    nonzero = tuple(count for _, count in counts if count > 0)
    statistics = UniverseStatistics(
        universe_name=universe_name,
        requested_start=requested_start,
        requested_end=requested_end,
        first_active_date=next((value for value, count in counts if count > 0), None),
        last_active_date=next(
            (value for value, count in reversed(counts) if count > 0), None
        ),
        interval_count=len(normalized),
        symbol_count=len({item.symbol for item in normalized}),
        active_day_count=len(nonzero),
        minimum_daily_members=min(nonzero, default=0),
        maximum_daily_members=max(nonzero, default=0),
        mean_daily_members=(sum(nonzero) / len(nonzero)) if nonzero else 0.0,
    )
    return statistics, tuple(counts)


def tushare_code_to_qlib_symbol(ts_code: str) -> str:
    """Map a Tushare historical or current code to its stable Qlib symbol."""
    return canonical_instrument_from_tushare(ts_code)


def _intersect_two_range_sets(
    left: Iterable[QlibInstrumentRange], right: Iterable[QlibInstrumentRange]
) -> tuple[QlibInstrumentRange, ...]:
    left_grouped = _group_ranges(left)
    right_grouped = _group_ranges(right)
    output: list[QlibInstrumentRange] = []
    for symbol in sorted(left_grouped.keys() & right_grouped.keys()):
        for left_item in left_grouped[symbol]:
            for right_item in right_grouped[symbol]:
                start = max(left_item.start, right_item.start)
                end = min(left_item.end, right_item.end)
                if start <= end:
                    output.append(QlibInstrumentRange(symbol, start, end))
    return merge_qlib_instrument_ranges(output)


def _group_ranges(
    ranges: Iterable[QlibInstrumentRange],
) -> dict[str, tuple[QlibInstrumentRange, ...]]:
    grouped: dict[str, list[QlibInstrumentRange]] = defaultdict(list)
    for item in ranges:
        grouped[item.symbol].append(item)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


def _first_calendar_on_or_after(calendar: Sequence[str], value: str) -> str | None:
    index = bisect_left(calendar, value)
    return calendar[index] if index < len(calendar) else None


def _last_calendar_before(calendar: Sequence[str], value: str) -> str | None:
    index = bisect_left(calendar, value) - 1
    return calendar[index] if index >= 0 else None


def _last_calendar_on_or_before(calendar: Sequence[str], value: str) -> str | None:
    index = bisect_right(calendar, value) - 1
    return calendar[index] if index >= 0 else None


def _validate_calendar(calendar: Sequence[str]) -> None:
    if not calendar:
        raise ValueError("calendar must not be empty")
    values = tuple(calendar)
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("calendar must contain sorted unique dates")
    for value in values:
        _parse_date(value)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected ISO date, got {value!r}") from error


def _next_calendar_day(value: str) -> str:
    return (
        date.fromisoformat(value)
        .fromordinal(date.fromisoformat(value).toordinal() + 1)
        .isoformat()
    )
