"""Error types for CosmosAlpha core protocols."""

from __future__ import annotations


class CoreError(Exception):
    """Base class for errors raised by the core protocol layer."""


class CoreValidationError(CoreError, ValueError):
    """Report that a core value violates its domain invariants."""


class SerializationError(CoreError, ValueError):
    """Report that a value cannot cross the strict JSON boundary."""


class UnknownFieldError(SerializationError):
    """Report an unexpected field in a serialized object."""


class MissingFieldError(SerializationError):
    """Report a required field missing from a serialized object."""


class TypeMismatchError(SerializationError):
    """Report a serialized value whose type does not match its annotation."""
