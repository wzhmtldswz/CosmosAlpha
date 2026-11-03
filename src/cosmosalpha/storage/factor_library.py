"""SQLite-backed accepted-factor library metadata and correlation references."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    FactorSpec,
    MetricValue,
)
from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    ObjectRef,
    from_json,
    to_json,
)
from cosmosalpha.storage.errors import RepositoryError

FACTOR_LIBRARY_ENTRY_SCHEMA_VERSION = "factor-library-entry/1.0"


class FactorLibraryError(RepositoryError):
    """Raised when accepted factor metadata cannot be safely published."""


class ObjectReferenceVerifier(Protocol):
    """Verify persisted object references before SQL factor-library publication."""

    def exists(self, ref: ObjectRef) -> bool:
        """Return whether an object reference resolves."""


class ArtifactReferenceVerifier(Protocol):
    """Verify persisted artifact references before SQL factor-library publication."""

    def verify(self, ref: ArtifactRef) -> bool:
        """Return whether an artifact reference resolves and validates."""


class FactorLibraryRepository(Protocol):
    """Accepted-factor persistence and peer-value lookup boundary."""

    def publish_accepted(
        self,
        *,
        factor_spec: FactorSpec,
        factor_ref: ObjectRef,
        evaluation_result: EvaluationResult,
        evaluation_result_ref: ObjectRef,
        decision: AcceptanceDecision,
        decision_ref: ObjectRef,
        source_experiment_id: str,
        accepted_at: datetime,
    ) -> FactorLibraryEntry:
        """Publish one deterministically accepted factor or reuse an identical entry."""

    def reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Return accepted peer factor values for comparable correlation checks."""

    def get_entry(self, factor_id: str) -> FactorLibraryEntry | None:
        """Return one accepted factor entry when it already exists."""


@dataclass(frozen=True, slots=True)
class FactorLibraryEntry:
    """Immutable accepted-factor metadata stored in the factor library."""

    factor_id: str
    expression: str
    frequency: str
    factor_ref: ObjectRef
    evaluation_result_ref: ObjectRef
    acceptance_decision_ref: ObjectRef
    evaluated_factor_values_ref: ArtifactRef
    data_snapshot_id: str
    evaluation_protocol_id: str
    metrics: tuple[MetricValue, ...]
    source_experiment_id: str
    accepted_at: datetime
    schema_version: str = FACTOR_LIBRARY_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate immutable factor library metadata and lineage references."""
        for field_name in (
            "factor_id",
            "expression",
            "frequency",
            "data_snapshot_id",
            "evaluation_protocol_id",
            "source_experiment_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        for field_name in (
            "factor_ref",
            "evaluation_result_ref",
            "acceptance_decision_ref",
        ):
            if not isinstance(getattr(self, field_name), ObjectRef):
                raise CoreValidationError(f"{field_name} must be ObjectRef")
        if not isinstance(self.evaluated_factor_values_ref, ArtifactRef):
            raise CoreValidationError("evaluated_factor_values_ref must be ArtifactRef")
        if not isinstance(self.metrics, tuple) or not all(
            isinstance(metric, MetricValue) for metric in self.metrics
        ):
            raise CoreValidationError("metrics must be a tuple of MetricValue")
        if (
            not isinstance(self.accepted_at, datetime)
            or self.accepted_at.tzinfo is None
        ):
            raise CoreValidationError("accepted_at must be a UTC datetime")
        if self.accepted_at.utcoffset() != UTC.utcoffset(self.accepted_at):
            raise CoreValidationError("accepted_at must be a UTC datetime")
        if self.schema_version != FACTOR_LIBRARY_ENTRY_SCHEMA_VERSION:
            raise CoreValidationError("invalid factor library entry schema_version")


class SqliteFactorLibrary:
    """Explicitly initialized SQLite implementation of accepted-factor metadata."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        object_verifier: ObjectReferenceVerifier,
        artifact_verifier: ArtifactReferenceVerifier,
    ) -> None:
        """Bind an initialized database and reference verifiers."""
        self._database_path = Path(database_path)
        self._objects = object_verifier
        self._artifacts = artifact_verifier

    @classmethod
    def initialize(
        cls,
        database_path: str | Path,
        *,
        object_verifier: ObjectReferenceVerifier,
        artifact_verifier: ArtifactReferenceVerifier,
    ) -> SqliteFactorLibrary:
        """Explicitly create the local factor-library schema."""
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=30.0)) as connection, connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                raise FactorLibraryError("unsupported factor library schema")
            if version == 0:
                connection.executescript(_SCHEMA_SQL)
                connection.execute("PRAGMA user_version = 1")
        return cls(
            path,
            object_verifier=object_verifier,
            artifact_verifier=artifact_verifier,
        )

    def publish_accepted(
        self,
        *,
        factor_spec: FactorSpec,
        factor_ref: ObjectRef,
        evaluation_result: EvaluationResult,
        evaluation_result_ref: ObjectRef,
        decision: AcceptanceDecision,
        decision_ref: ObjectRef,
        source_experiment_id: str,
        accepted_at: datetime,
    ) -> FactorLibraryEntry:
        """Publish one accepted factor idempotently after verifying all references."""
        _validate_publication_inputs(
            factor_spec,
            factor_ref,
            evaluation_result,
            evaluation_result_ref,
            decision,
            decision_ref,
            source_experiment_id,
            accepted_at,
        )
        self._verify_refs(
            (factor_ref, evaluation_result_ref, decision_ref),
            (evaluation_result.evaluated_factor_values_ref,),
        )
        entry = FactorLibraryEntry(
            factor_id=factor_spec.factor_id,
            expression=factor_spec.expression,
            frequency=factor_spec.frequency,
            factor_ref=factor_ref,
            evaluation_result_ref=evaluation_result_ref,
            acceptance_decision_ref=decision_ref,
            evaluated_factor_values_ref=evaluation_result.evaluated_factor_values_ref,
            data_snapshot_id=evaluation_result.data_snapshot_id,
            evaluation_protocol_id=evaluation_result.evaluation_protocol_id,
            metrics=evaluation_result.metrics,
            source_experiment_id=source_experiment_id,
            accepted_at=accepted_at,
        )
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM factor_library WHERE factor_id = ?",
                (entry.factor_id,),
            ).fetchone()
            if row is not None:
                stored = from_json(row[0], FactorLibraryEntry)
                if not _same_publication(stored, entry):
                    raise FactorLibraryError("FACTOR_LIBRARY_CONFLICT")
                return stored
            connection.execute(
                """
                INSERT INTO factor_library(
                    factor_id, data_snapshot_id, evaluation_protocol_id, payload
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    entry.factor_id,
                    entry.data_snapshot_id,
                    entry.evaluation_protocol_id,
                    to_json(entry),
                ),
            )
        return entry

    def reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Return accepted peer values in stable factor-id order for one snapshot."""
        _require_text(data_snapshot_id, "data_snapshot_id")
        if exclude_factor_id is not None:
            _require_text(exclude_factor_id, "exclude_factor_id")
        with self._connection() as connection:
            if exclude_factor_id is None:
                rows = connection.execute(
                    "SELECT payload FROM factor_library WHERE data_snapshot_id = ? ORDER BY factor_id",
                    (data_snapshot_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT payload FROM factor_library
                    WHERE data_snapshot_id = ? AND factor_id <> ?
                    ORDER BY factor_id
                    """,
                    (data_snapshot_id, exclude_factor_id),
                ).fetchall()
        return tuple(
            from_json(row[0], FactorLibraryEntry).evaluated_factor_values_ref
            for row in rows
        )

    def list_entries(self) -> tuple[FactorLibraryEntry, ...]:
        """Return all accepted entries in stable factor-id order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM factor_library ORDER BY factor_id"
            ).fetchall()
        return tuple(from_json(row[0], FactorLibraryEntry) for row in rows)

    def get_entry(self, factor_id: str) -> FactorLibraryEntry | None:
        """Return one accepted factor entry by semantic factor identity."""
        _require_text(factor_id, "factor_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM factor_library WHERE factor_id = ?",
                (factor_id,),
            ).fetchone()
        return None if row is None else from_json(row[0], FactorLibraryEntry)

    def _verify_refs(
        self,
        object_refs: tuple[ObjectRef, ...],
        artifact_refs: tuple[ArtifactRef, ...],
    ) -> None:
        if not all(self._objects.exists(ref) for ref in object_refs):
            raise FactorLibraryError("FACTOR_LIBRARY_OBJECT_REFERENCE_INVALID")
        if not all(self._artifacts.verify(ref) for ref in artifact_refs):
            raise FactorLibraryError("FACTOR_LIBRARY_ARTIFACT_REFERENCE_INVALID")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        try:
            with connection:
                connection.execute("PRAGMA busy_timeout = 30000")
                connection.execute("PRAGMA foreign_keys = ON")
                yield connection
        finally:
            connection.close()


def _validate_publication_inputs(
    factor_spec: FactorSpec,
    factor_ref: ObjectRef,
    evaluation_result: EvaluationResult,
    evaluation_result_ref: ObjectRef,
    decision: AcceptanceDecision,
    decision_ref: ObjectRef,
    source_experiment_id: str,
    accepted_at: datetime,
) -> None:
    """Reject any non-accepted or incompatible factor-library publication."""
    if not isinstance(factor_spec, FactorSpec):
        raise FactorLibraryError("factor_spec must be FactorSpec")
    if not isinstance(evaluation_result, EvaluationResult):
        raise FactorLibraryError("evaluation_result must be EvaluationResult")
    if not isinstance(decision, AcceptanceDecision) or not decision.accepted:
        raise FactorLibraryError("FACTOR_LIBRARY_REQUIRES_ACCEPTED_DECISION")
    for ref, field_name in (
        (factor_ref, "factor_ref"),
        (evaluation_result_ref, "evaluation_result_ref"),
        (decision_ref, "decision_ref"),
    ):
        if not isinstance(ref, ObjectRef):
            raise FactorLibraryError(f"{field_name} must be ObjectRef")
    _require_text(source_experiment_id, "source_experiment_id")
    if not isinstance(accepted_at, datetime) or accepted_at.tzinfo is None:
        raise FactorLibraryError("accepted_at must be a UTC datetime")
    if accepted_at.utcoffset() != UTC.utcoffset(accepted_at):
        raise FactorLibraryError("accepted_at must be a UTC datetime")
    if (
        factor_spec.factor_id != evaluation_result.factor_id
        or factor_spec.factor_id != decision.factor_id
        or decision.evaluation_result_ref != evaluation_result_ref
    ):
        raise FactorLibraryError("FACTOR_LIBRARY_LINEAGE_MISMATCH")


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _same_publication(
    left: FactorLibraryEntry,
    right: FactorLibraryEntry,
) -> bool:
    """Compare idempotency inputs while preserving the first publication timestamp."""
    return (
        left.factor_id == right.factor_id
        and left.expression == right.expression
        and left.frequency == right.frequency
        and left.factor_ref == right.factor_ref
        and left.evaluation_result_ref == right.evaluation_result_ref
        and left.acceptance_decision_ref == right.acceptance_decision_ref
        and left.evaluated_factor_values_ref == right.evaluated_factor_values_ref
        and left.data_snapshot_id == right.data_snapshot_id
        and left.evaluation_protocol_id == right.evaluation_protocol_id
        and left.metrics == right.metrics
        and left.schema_version == right.schema_version
    )


_SCHEMA_SQL = """
CREATE TABLE factor_library (
    factor_id TEXT PRIMARY KEY,
    data_snapshot_id TEXT NOT NULL,
    evaluation_protocol_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX factor_library_snapshot_idx
ON factor_library(data_snapshot_id, factor_id);
"""
