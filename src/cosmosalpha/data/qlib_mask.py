"""Utilities for reading and writing Qlib feature binaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmosalpha.core import CoreValidationError
from cosmosalpha.data.models import canonical_instrument_code

MASK_BIN_FIELDS: tuple[str, ...] = (
    "universe_mask",
    "eligible_mask",
    "entry_tradable_mask",
    "exit_tradable_mask",
)
MASK_NEW_STOCK_AGE_POLICY = "first_trading_day_on_or_after_calendar_anniversary"


@dataclass(frozen=True)
class InstrumentRange:
    """A Qlib instrument and its inclusive calendar range."""

    symbol: str
    start: str
    end: str

    def __post_init__(self) -> None:
        """Normalize the stored symbol to the project canonical form."""
        try:
            symbol = canonical_instrument_code(self.symbol)
        except CoreValidationError as error:
            raise ValueError("symbol must be a canonical A-share code") from error
        object.__setattr__(self, "symbol", symbol)


def read_calendar(path: Path) -> tuple[str, ...]:
    """Read a Qlib calendar file."""
    dates = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    if not dates or any(not date for date in dates):
        raise ValueError(f"Qlib calendar is empty or malformed: {path}")
    return dates


def read_instrument_ranges(path: Path) -> tuple[InstrumentRange, ...]:
    """Read Qlib's tab-separated instrument range file."""
    ranges: list[InstrumentRange] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Malformed instrument line {line_number}: {raw_line!r}")
        ranges.append(InstrumentRange(parts[0], parts[1], parts[2]))
    if not ranges:
        raise ValueError(f"Qlib instrument file is empty: {path}")
    return tuple(sorted(ranges, key=lambda item: item.symbol))


def read_qlib_field(
    provider_uri: Path,
    symbol: str,
    field: str,
    calendar_length: int,
    *,
    freq: str = "day",
) -> np.ndarray:
    """Read one Qlib field and align it to the complete calendar."""
    path = provider_uri / "features" / symbol.lower() / f"{field}.{freq}.bin"
    aligned = np.full(calendar_length, np.nan, dtype=np.float32)
    if not path.exists():
        return aligned

    raw = np.fromfile(path, dtype="<f4")
    if raw.size <= 1:
        return aligned
    start_index = int(raw[0])
    if start_index < 0 or start_index >= calendar_length:
        raise ValueError(f"Invalid Qlib field header in {path}: {start_index}")
    values = raw[1 : 1 + calendar_length - start_index]
    aligned[start_index : start_index + values.size] = values
    return aligned


def market_validity(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Return a conservative market-data validity mask."""
    required = ("open", "high", "low", "close", "volume", "vwap", "turn")
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(f"Missing market fields: {', '.join(missing)}")

    open_value = fields["open"]
    high_value = fields["high"]
    low_value = fields["low"]
    close_value = fields["close"]
    valid = np.ones(open_value.shape, dtype=bool)
    for field in required:
        valid &= np.isfinite(fields[field])
    valid &= open_value > 0
    valid &= high_value > 0
    valid &= low_value > 0
    valid &= close_value > 0
    valid &= fields["volume"] >= 0
    valid &= high_value >= np.maximum.reduce((open_value, low_value, close_value))
    valid &= low_value <= np.minimum.reduce((open_value, high_value, close_value))
    return valid


def write_qlib_field(
    provider_uri: Path,
    symbol: str,
    field: str,
    start_index: int,
    values: np.ndarray,
    *,
    freq: str = "day",
) -> None:
    """Write one Qlib field with its calendar start index header."""
    if values.ndim != 1:
        raise ValueError("Qlib field values must be one-dimensional")
    if start_index < 0:
        raise ValueError("Qlib field start index must be non-negative")

    output_dir = provider_uri / "features" / symbol.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    header = np.asarray([start_index], dtype="<f4")
    payload = np.asarray(values, dtype="<f4")
    np.concatenate((header, payload)).tofile(output_dir / f"{field}.{freq}.bin")


def mask_state_to_float(state: np.ndarray) -> np.ndarray:
    """Convert tri-state values 1/0/-1 to Qlib float values 1/0/NaN."""
    if state.dtype.kind not in "biu":
        raise ValueError("Mask state must be an integer or boolean array")
    output = np.full(state.shape, np.nan, dtype=np.float32)
    output[state == 0] = 0.0
    output[state == 1] = 1.0
    invalid = ~np.isin(state, (-1, 0, 1))
    if invalid.any():
        raise ValueError("Mask state contains values other than -1, 0, and 1")
    return output


def read_mask_new_stock_min_age_years(provider_uri: Path) -> int | None:
    """Read the optional new-stock eligibility policy from a Mask manifest."""
    manifest_path = provider_uri.expanduser() / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Mask manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid Mask manifest object: {manifest_path}")
    value = manifest.get("new_stock_min_age_years")
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid new_stock_min_age_years in Mask manifest: {value!r}")
    policy = manifest.get("new_stock_age_policy")
    if policy != MASK_NEW_STOCK_AGE_POLICY:
        raise ValueError(f"Unsupported Mask new-stock age policy: {policy!r}")
    return value
