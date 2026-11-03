"""Errors raised by storage repositories."""

from cosmosalpha.core import CoreError


class RepositoryError(CoreError):
    """Base class for repository failures."""


class ObjectNotFoundError(RepositoryError):
    """Raised when an object reference cannot be resolved."""


class ObjectConflictError(RepositoryError):
    """Raised when an object identity is reused with different content."""


class ArtifactNotFoundError(RepositoryError):
    """Raised when an artifact reference cannot be resolved."""


class IntegrityError(RepositoryError):
    """Raised when stored bytes do not match declared integrity metadata."""


class RecoveryError(RepositoryError):
    """Raised when repository recovery cannot proceed."""
