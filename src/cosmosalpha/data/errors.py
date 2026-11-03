"""Errors raised by market data contracts and providers."""

from __future__ import annotations


class DataError(Exception):
    """Base class for market data errors."""


class DataValidationError(DataError):
    """Raised when market data contracts or payload metadata are invalid."""


class DataProviderError(DataError):
    """Raised when a market data provider cannot produce a valid snapshot."""
