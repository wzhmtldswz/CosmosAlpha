"""Shared deterministic statistical primitives for factor evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Return a float64 Pearson coefficient bounded to its mathematical range."""
    left_values = left.astype(np.float64, copy=False)
    right_values = right.astype(np.float64, copy=False)
    left_dm = left_values - float(np.mean(left_values))
    right_dm = right_values - float(np.mean(right_values))
    denominator = float(
        np.sqrt(np.sum(left_dm * left_dm) * np.sum(right_dm * right_dm))
    )
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return math.nan
    coefficient = float(np.sum(left_dm * right_dm) / denominator)
    if not math.isfinite(coefficient):
        return math.nan
    return min(1.0, max(-1.0, coefficient))


def mean_daily_cross_sectional_pearson(
    daily_pairs: Iterable[tuple[np.ndarray, np.ndarray, int]],
    *,
    min_cross_section_samples: int,
    min_daily_effective_stock_ratio: float,
) -> tuple[float, int]:
    """Return the equal-date mean of valid daily cross-sectional correlations.

    Each pair contains already aligned, eligible left/right values and the
    day's full eligible-stock count. Pairwise non-finite values are removed
    before applying the shared Graph coverage gates.
    """
    correlations: list[float] = []
    for left, right, eligible_count in daily_pairs:
        if left.shape != right.shape:
            raise ValueError("daily correlation arrays must have equal shape")
        finite = np.isfinite(left) & np.isfinite(right)
        common_count = int(np.count_nonzero(finite))
        if (
            eligible_count <= 0
            or common_count < min_cross_section_samples
            or common_count / eligible_count < min_daily_effective_stock_ratio
        ):
            continue
        correlation = safe_pearson(left[finite], right[finite])
        if math.isfinite(correlation):
            correlations.append(correlation)
    if not correlations:
        return math.nan, 0
    return float(np.mean(np.asarray(correlations, dtype=np.float64))), len(correlations)


__all__ = ["mean_daily_cross_sectional_pearson", "safe_pearson"]
