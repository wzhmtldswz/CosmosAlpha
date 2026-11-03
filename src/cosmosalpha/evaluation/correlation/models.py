"""Stable models for the Graph correlation protocol and dense store."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cosmosalpha.core import ArtifactRef, CoreValidationError, stable_sha256

if TYPE_CHECKING:
    from cosmosalpha.evaluation.correlation.engine import CorrelationArtifactRepository

CORRELATION_PROTOCOL_SCHEMA_VERSION = "graph-correlation-protocol/1.0"
CORRELATION_FACTOR_REF_SCHEMA_VERSION = "correlation-factor-ref/1.0"
CORRELATION_SNAPSHOT_SCHEMA_VERSION = "correlation-snapshot/1.0"

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _require_schema(value: object, field_name: str, expected: str) -> None:
    _require_text(value, field_name)
    if value != expected:
        raise CoreValidationError(f"{field_name} must be {expected!r}")


def _require_ratio(value: object, field_name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CoreValidationError(f"{field_name} must be a float in [0, 1]")


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise CoreValidationError(f"{field_name} must be a positive integer")


def _require_artifact(value: object, field_name: str) -> None:
    if not isinstance(value, ArtifactRef):
        raise CoreValidationError(f"{field_name} must be ArtifactRef")


@dataclass(frozen=True, slots=True)
class GraphCorrelationProtocol:
    """Define every semantic choice that affects Graph correlation facts."""

    protocol_id: str
    data_snapshot_id: str
    preprocessing_protocol_id: str
    evaluation_protocol_id: str
    min_cross_section_samples: int
    min_daily_effective_stock_ratio: float
    min_pair_window_ratio: float
    min_valid_date_count: int
    min_valid_date_ratio: float
    use_pairwise_finite: bool
    daily_weighting: str
    similarity_use_abs: bool
    exposure_source: str
    calculation_dtype: str
    store_dtype: str
    schema_version: str = CORRELATION_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject protocol objects whose semantics are not frozen for v1."""
        for name in (
            "protocol_id",
            "data_snapshot_id",
            "preprocessing_protocol_id",
            "evaluation_protocol_id",
        ):
            _require_text(getattr(self, name), name)
        _require_positive_int(
            self.min_cross_section_samples, "min_cross_section_samples"
        )
        _require_positive_int(self.min_valid_date_count, "min_valid_date_count")
        for name in (
            "min_daily_effective_stock_ratio",
            "min_pair_window_ratio",
            "min_valid_date_ratio",
        ):
            _require_ratio(getattr(self, name), name)
        if type(self.use_pairwise_finite) is not bool or not self.use_pairwise_finite:
            raise CoreValidationError("use_pairwise_finite must be true")
        if self.daily_weighting != "equal":
            raise CoreValidationError("daily_weighting must be 'equal'")
        if type(self.similarity_use_abs) is not bool or not self.similarity_use_abs:
            raise CoreValidationError("similarity_use_abs must be true")
        if self.exposure_source != "evaluated_factor_values":
            raise CoreValidationError(
                "exposure_source must be 'evaluated_factor_values'"
            )
        if self.calculation_dtype != "float64":
            raise CoreValidationError("calculation_dtype must be 'float64'")
        if self.store_dtype != "float32":
            raise CoreValidationError("store_dtype must be 'float32'")
        _require_schema(
            self.schema_version,
            "schema_version",
            CORRELATION_PROTOCOL_SCHEMA_VERSION,
        )


@dataclass(frozen=True, slots=True)
class CorrelationFactorRef:
    """Identify one evaluated factor and its aligned eligibility artifact."""

    factor_id: str
    data_snapshot_id: str
    preprocessing_protocol_id: str
    evaluation_protocol_id: str
    evaluated_factor_values_ref: ArtifactRef
    mask_data_ref: ArtifactRef
    schema_version: str = CORRELATION_FACTOR_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate factor provenance without importing Graph domain types."""
        for name in (
            "factor_id",
            "data_snapshot_id",
            "preprocessing_protocol_id",
            "evaluation_protocol_id",
        ):
            _require_text(getattr(self, name), name)
        _require_artifact(
            self.evaluated_factor_values_ref, "evaluated_factor_values_ref"
        )
        _require_artifact(self.mask_data_ref, "mask_data_ref")
        _require_schema(
            self.schema_version,
            "schema_version",
            CORRELATION_FACTOR_REF_SCHEMA_VERSION,
        )


def _snapshot_fingerprint(
    *,
    snapshot_version: str,
    protocol: GraphCorrelationProtocol,
    factor_refs: tuple[CorrelationFactorRef, ...],
    matrix_ref: ArtifactRef,
    matrix_shape: tuple[int, int],
    correlation_dtype: str,
    count_dtype: str,
) -> str:
    """Create the stable identity of a matrix snapshot."""
    return stable_sha256(
        {
            "schema_version": CORRELATION_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_version": snapshot_version,
            "protocol": protocol,
            "factor_refs": factor_refs,
            "matrix_ref": matrix_ref,
            "matrix_shape": matrix_shape,
            "correlation_dtype": correlation_dtype,
            "count_dtype": count_dtype,
        }
    )


@dataclass(frozen=True, slots=True)
class CorrelationSnapshot:
    """Reference a self-consistent, versioned dense correlation payload."""

    snapshot_version: str
    snapshot_id: str
    protocol: GraphCorrelationProtocol
    factor_refs: tuple[CorrelationFactorRef, ...]
    matrix_ref: ArtifactRef
    matrix_shape: tuple[int, int]
    correlation_dtype: str
    count_dtype: str
    schema_version: str = CORRELATION_SNAPSHOT_SCHEMA_VERSION
    metadata_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        """Validate snapshot metadata and its deterministic identity."""
        _require_text(self.snapshot_version, "snapshot_version")
        if type(self.snapshot_id) is not str or not _DIGEST_PATTERN.fullmatch(
            self.snapshot_id
        ):
            raise CoreValidationError("snapshot_id must be a lowercase SHA-256 digest")
        if not isinstance(self.protocol, GraphCorrelationProtocol):
            raise CoreValidationError("protocol must be GraphCorrelationProtocol")
        if not isinstance(self.factor_refs, tuple):
            raise CoreValidationError("factor_refs must be a tuple")
        if not all(isinstance(ref, CorrelationFactorRef) for ref in self.factor_refs):
            raise CoreValidationError(
                "every factor_refs item must be CorrelationFactorRef"
            )
        factor_ids = tuple(ref.factor_id for ref in self.factor_refs)
        if len(set(factor_ids)) != len(factor_ids):
            raise CoreValidationError(
                "factor_refs must contain unique factor_id values"
            )
        _require_artifact(self.matrix_ref, "matrix_ref")
        if (
            not isinstance(self.matrix_shape, tuple)
            or len(self.matrix_shape) != 2
            or not all(type(size) is int and size >= 0 for size in self.matrix_shape)
            or self.matrix_shape[0] != self.matrix_shape[1]
            or self.matrix_shape[0] != len(self.factor_refs)
        ):
            raise CoreValidationError(
                "matrix_shape must be a square shape matching factor_refs"
            )
        if self.correlation_dtype != "float32":
            raise CoreValidationError("correlation_dtype must be 'float32'")
        if self.count_dtype != "int64":
            raise CoreValidationError("count_dtype must be 'int64'")
        _require_schema(
            self.schema_version,
            "schema_version",
            CORRELATION_SNAPSHOT_SCHEMA_VERSION,
        )
        if self.metadata_ref is not None:
            _require_artifact(self.metadata_ref, "metadata_ref")
        expected_id = _snapshot_fingerprint(
            snapshot_version=self.snapshot_version,
            protocol=self.protocol,
            factor_refs=self.factor_refs,
            matrix_ref=self.matrix_ref,
            matrix_shape=self.matrix_shape,
            correlation_dtype=self.correlation_dtype,
            count_dtype=self.count_dtype,
        )
        if self.snapshot_id != expected_id:
            raise CoreValidationError("snapshot_id does not match snapshot content")


@dataclass(slots=True)
class CorrelationStore:
    """Hold complete signed facts and factor metadata for one protocol."""

    protocol: GraphCorrelationProtocol
    factor_refs: tuple[CorrelationFactorRef, ...]
    correlation_matrix: np.ndarray
    valid_date_count_matrix: np.ndarray
    overlap_date_count_matrix: np.ndarray
    snapshot_version: str | None = None

    def __post_init__(self) -> None:
        """Validate and detach dense arrays from caller-owned memory."""
        if not isinstance(self.protocol, GraphCorrelationProtocol):
            raise CoreValidationError("protocol must be GraphCorrelationProtocol")
        if not isinstance(self.factor_refs, tuple):
            raise CoreValidationError("factor_refs must be a tuple")
        if not all(isinstance(ref, CorrelationFactorRef) for ref in self.factor_refs):
            raise CoreValidationError(
                "every factor_refs item must be CorrelationFactorRef"
            )
        ids = tuple(ref.factor_id for ref in self.factor_refs)
        if len(set(ids)) != len(ids):
            raise CoreValidationError(
                "factor_refs must contain unique factor_id values"
            )
        size = len(self.factor_refs)
        if not isinstance(self.correlation_matrix, np.ndarray):
            raise CoreValidationError("correlation_matrix must be a NumPy array")
        if not isinstance(self.valid_date_count_matrix, np.ndarray):
            raise CoreValidationError("valid_date_count_matrix must be a NumPy array")
        if not isinstance(self.overlap_date_count_matrix, np.ndarray):
            raise CoreValidationError("overlap_date_count_matrix must be a NumPy array")
        expected_shape = (size, size)
        if (
            self.correlation_matrix.shape != expected_shape
            or self.valid_date_count_matrix.shape != expected_shape
            or self.overlap_date_count_matrix.shape != expected_shape
        ):
            raise CoreValidationError("all correlation matrices must be square")
        if self.correlation_matrix.dtype != np.dtype("float32"):
            raise CoreValidationError("correlation_matrix must use float32")
        if self.valid_date_count_matrix.dtype != np.dtype("int64"):
            raise CoreValidationError("valid_date_count_matrix must use int64")
        if self.overlap_date_count_matrix.dtype != np.dtype("int64"):
            raise CoreValidationError("overlap_date_count_matrix must use int64")
        if np.any(self.valid_date_count_matrix < 0) or np.any(
            self.overlap_date_count_matrix < 0
        ):
            raise CoreValidationError("date count matrices must be non-negative")
        if np.any(self.valid_date_count_matrix > self.overlap_date_count_matrix):
            raise CoreValidationError(
                "valid_date_count_matrix cannot exceed overlap_date_count_matrix"
            )
        finite_correlation = np.isfinite(self.correlation_matrix)
        if np.any(np.isinf(self.correlation_matrix)) or np.any(
            finite_correlation
            & ((self.correlation_matrix < -1.0) | (self.correlation_matrix > 1.0))
        ):
            raise CoreValidationError(
                "correlation_matrix values must be in [-1, 1] or NaN"
            )
        if self.snapshot_version is not None:
            _require_text(self.snapshot_version, "snapshot_version")
        self.correlation_matrix = np.array(self.correlation_matrix, copy=True)
        self.valid_date_count_matrix = np.array(self.valid_date_count_matrix, copy=True)
        self.overlap_date_count_matrix = np.array(
            self.overlap_date_count_matrix, copy=True
        )

    @classmethod
    def empty(cls, protocol: GraphCorrelationProtocol) -> CorrelationStore:
        """Create an empty store for a validated protocol."""
        if not isinstance(protocol, GraphCorrelationProtocol):
            raise CoreValidationError("protocol must be GraphCorrelationProtocol")
        empty_float = np.empty((0, 0), dtype=np.float32)
        empty_count = np.empty((0, 0), dtype=np.int64)
        return cls(protocol, (), empty_float, empty_count, empty_count)

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Return factor IDs in stable matrix-index order."""
        return tuple(ref.factor_id for ref in self.factor_refs)

    def has_factor(self, factor_id: str) -> bool:
        """Return whether the store has a factor identity."""
        return factor_id in self.factor_ids

    def factor_index(self, factor_id: str) -> int:
        """Return the matrix index for a factor or raise ``KeyError``."""
        for index, known_id in enumerate(self.factor_ids):
            if known_id == factor_id:
                return index
        raise KeyError(factor_id)

    def correlation(self, left_id: str, right_id: str) -> float:
        """Return a signed correlation fact, possibly ``nan``."""
        return float(
            self.correlation_matrix[
                self.factor_index(left_id), self.factor_index(right_id)
            ]
        )

    def get(self, left_id: str, right_id: str) -> float:
        """Return a signed correlation fact through the store protocol name."""
        return self.correlation(left_id, right_id)

    def get_row(self, factor_id: str) -> np.ndarray:
        """Return a detached signed-correlation row in matrix-index order."""
        return np.array(
            self.correlation_matrix[self.factor_index(factor_id)], copy=True
        )

    def get_matrix(self) -> np.ndarray:
        """Return a detached copy of the complete signed matrix."""
        return np.array(self.correlation_matrix, copy=True)

    def valid_date_count(self, left_id: str, right_id: str) -> int:
        """Return the number of valid daily Pearson observations."""
        return int(
            self.valid_date_count_matrix[
                self.factor_index(left_id), self.factor_index(right_id)
            ]
        )

    def overlap_date_count(self, left_id: str, right_id: str) -> int:
        """Return the number of dates in the pair evaluation window."""
        return int(
            self.overlap_date_count_matrix[
                self.factor_index(left_id), self.factor_index(right_id)
            ]
        )

    def valid_date_ratio(self, left_id: str, right_id: str) -> float:
        """Return valid-date coverage, using zero for an empty overlap."""
        valid = self.valid_date_count(left_id, right_id)
        overlap = self.overlap_date_count(left_id, right_id)
        return 0.0 if overlap == 0 else valid / overlap

    def build_neighborhood(
        self,
        *,
        edge_threshold: float,
        max_neighbors: int,
    ) -> np.ndarray:
        """Build the local ``W_corr`` projection from complete facts."""
        from cosmosalpha.evaluation.correlation.engine import (
            CorrelationNeighborhoodBuilder,
        )

        return CorrelationNeighborhoodBuilder().build(
            self.correlation_matrix,
            self.valid_date_count_matrix,
            self.overlap_date_count_matrix,
            edge_threshold=edge_threshold,
            max_neighbors=max_neighbors,
            min_valid_date_count=self.protocol.min_valid_date_count,
            min_valid_date_ratio=self.protocol.min_valid_date_ratio,
        )

    def persist(
        self,
        artifacts: CorrelationArtifactRepository,
        *,
        snapshot_version: str,
    ) -> CorrelationSnapshot:
        """Persist dense matrices and metadata as a versioned snapshot."""
        from cosmosalpha.evaluation.correlation.engine import persist_store

        return persist_store(self, artifacts, snapshot_version=snapshot_version)

    @classmethod
    def load(
        cls,
        artifacts: CorrelationArtifactRepository,
        snapshot: CorrelationSnapshot,
        *,
        expected_protocol: GraphCorrelationProtocol,
    ) -> CorrelationStore:
        """Load a snapshot after checking protocol and matrix metadata."""
        from cosmosalpha.evaluation.correlation.engine import load_store

        return load_store(
            artifacts,
            snapshot,
            expected_protocol=expected_protocol,
        )
