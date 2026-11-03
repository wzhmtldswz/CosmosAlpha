"""SQLite-backed, recoverable workflow metadata repositories."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from cosmosalpha.core import (
    ArtifactRef,
    ComponentRef,
    CoreValidationError,
    FailureInfo,
    ObjectRef,
    from_json,
    stable_sha256,
    to_json,
)
from cosmosalpha.storage.errors import RecoveryError, RepositoryError

WORKFLOW_RUN_SCHEMA_VERSION = "workflow-run/1.0"
WORKFLOW_CYCLE_SCHEMA_VERSION = "workflow-cycle/1.0"
WORKFLOW_STEP_ATTEMPT_SCHEMA_VERSION = "workflow-step-attempt/1.0"


class WorkflowRunStatus(StrEnum):
    """Stable workflow run lifecycle states."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowCycleStatus(StrEnum):
    """Stable workflow cycle lifecycle states."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStepStatus(StrEnum):
    """Stable states for one persisted logical step attempt."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CoreValidationError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        raise CoreValidationError(f"{field_name} must be a UTC datetime")
    return value


def _require_refs(value: object, ref_type: type[object], field_name: str) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(ref, ref_type) for ref in value
    ):
        raise CoreValidationError(
            f"{field_name} must be a tuple of {ref_type.__name__}"
        )


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    """Metadata for one immutable workflow input set."""

    run_id: str
    context_ref: ObjectRef
    data_snapshot_ref: ObjectRef
    config_snapshot_ref: ArtifactRef
    code_version: str
    status: WorkflowRunStatus
    created_at: datetime
    completed_at: datetime | None
    schema_version: str = WORKFLOW_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate run identity, lifecycle, and root references."""
        _require_text(self.run_id, "run_id")
        if not isinstance(self.context_ref, ObjectRef):
            raise CoreValidationError("context_ref must be ObjectRef")
        if not isinstance(self.data_snapshot_ref, ObjectRef):
            raise CoreValidationError("data_snapshot_ref must be ObjectRef")
        if not isinstance(self.config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        _require_text(self.code_version, "code_version")
        if not isinstance(self.status, WorkflowRunStatus):
            raise CoreValidationError("status must be WorkflowRunStatus")
        created_at = _require_utc(self.created_at, "created_at")
        if self.completed_at is not None:
            completed_at = _require_utc(self.completed_at, "completed_at")
            if completed_at < created_at:
                raise CoreValidationError("completed_at must not precede created_at")
        if self.schema_version != WORKFLOW_RUN_SCHEMA_VERSION:
            raise CoreValidationError("invalid workflow run schema_version")


@dataclass(frozen=True, slots=True)
class WorkflowCycle:
    """Metadata for one cycle within a workflow run."""

    run_id: str
    cycle_id: str
    parent_ids: tuple[str, ...]
    status: WorkflowCycleStatus
    created_at: datetime
    completed_at: datetime | None
    schema_version: str = WORKFLOW_CYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate cycle identity, ancestry, and lifecycle timestamps."""
        _require_text(self.run_id, "run_id")
        _require_text(self.cycle_id, "cycle_id")
        if not isinstance(self.parent_ids, tuple):
            raise CoreValidationError("parent_ids must be a tuple")
        for index, parent_id in enumerate(self.parent_ids):
            _require_text(parent_id, f"parent_ids[{index}]")
        if not isinstance(self.status, WorkflowCycleStatus):
            raise CoreValidationError("status must be WorkflowCycleStatus")
        created_at = _require_utc(self.created_at, "created_at")
        if self.completed_at is not None:
            completed_at = _require_utc(self.completed_at, "completed_at")
            if completed_at < created_at:
                raise CoreValidationError("completed_at must not precede created_at")
        if self.schema_version != WORKFLOW_CYCLE_SCHEMA_VERSION:
            raise CoreValidationError("invalid workflow cycle schema_version")


@dataclass(frozen=True, slots=True)
class WorkflowStepAttempt:
    """One immutable execution attempt for a logical workflow step."""

    run_id: str
    cycle_id: str
    step_id: str
    attempt: int
    input_fingerprint: str
    input_object_refs: tuple[ObjectRef, ...]
    input_artifact_refs: tuple[ArtifactRef, ...]
    output_object_refs: tuple[ObjectRef, ...]
    output_artifact_refs: tuple[ArtifactRef, ...]
    component: ComponentRef
    status: WorkflowStepStatus
    failure: FailureInfo | None
    started_at: datetime
    completed_at: datetime | None
    schema_version: str = WORKFLOW_STEP_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate attempt inputs, terminal state, and output invariants."""
        _require_text(self.run_id, "run_id")
        _require_text(self.cycle_id, "cycle_id")
        _require_text(self.step_id, "step_id")
        if type(self.attempt) is not int or self.attempt < 1:
            raise CoreValidationError("attempt must be a positive integer")
        _require_text(self.input_fingerprint, "input_fingerprint")
        _require_refs(self.input_object_refs, ObjectRef, "input_object_refs")
        _require_refs(self.input_artifact_refs, ArtifactRef, "input_artifact_refs")
        _require_refs(self.output_object_refs, ObjectRef, "output_object_refs")
        _require_refs(self.output_artifact_refs, ArtifactRef, "output_artifact_refs")
        if not isinstance(self.component, ComponentRef):
            raise CoreValidationError("component must be ComponentRef")
        if not isinstance(self.status, WorkflowStepStatus):
            raise CoreValidationError("status must be WorkflowStepStatus")
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        started_at = _require_utc(self.started_at, "started_at")
        if self.completed_at is not None:
            completed_at = _require_utc(self.completed_at, "completed_at")
            if completed_at < started_at:
                raise CoreValidationError("completed_at must not precede started_at")
        if self.status is WorkflowStepStatus.STARTED:
            valid = (
                self.failure is None
                and self.completed_at is None
                and not self.output_object_refs
                and not self.output_artifact_refs
            )
        elif self.status is WorkflowStepStatus.SUCCEEDED:
            valid = self.failure is None and self.completed_at is not None
        else:
            valid = self.failure is not None and self.completed_at is not None
        if not valid:
            raise CoreValidationError("invalid workflow step attempt state")
        if self.schema_version != WORKFLOW_STEP_ATTEMPT_SCHEMA_VERSION:
            raise CoreValidationError("invalid workflow step attempt schema_version")


class ObjectReferenceVerifier(Protocol):
    """Verify persisted domain object references before SQL publication."""

    def exists(self, ref: ObjectRef) -> bool:
        """Return whether an object reference resolves."""


class ArtifactReferenceVerifier(Protocol):
    """Verify persisted artifact references before SQL publication."""

    def verify(self, ref: ArtifactRef) -> bool:
        """Return whether an artifact reference resolves and validates."""


class WorkflowMetadataRepository(Protocol):
    """Persistence boundary for recoverable workflow metadata."""

    def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create or return an identical workflow run."""

    def create_cycle(self, cycle: WorkflowCycle) -> WorkflowCycle:
        """Create or return an identical workflow cycle."""

    def complete_cycle(
        self, run_id: str, cycle_id: str, *, completed_at: datetime
    ) -> WorkflowCycle:
        """Mark one cycle complete after its final experiment is published."""

    def complete_run(self, run_id: str, *, completed_at: datetime) -> WorkflowRun:
        """Mark one run complete after its requested cycle is published."""

    def fail_cycle(
        self, run_id: str, cycle_id: str, *, completed_at: datetime
    ) -> WorkflowCycle:
        """Mark one cycle failed after a terminal branch error."""

    def fail_run(self, run_id: str, *, completed_at: datetime) -> WorkflowRun:
        """Mark one run failed after all requested branches fail."""

    def begin_attempt(
        self,
        *,
        run_id: str,
        cycle_id: str,
        step_id: str,
        input_object_refs: tuple[ObjectRef, ...],
        input_artifact_refs: tuple[ArtifactRef, ...],
        component: ComponentRef,
        started_at: datetime,
        reuse_success: bool = True,
    ) -> WorkflowStepAttempt:
        """Start an attempt or return a reusable successful attempt."""

    def succeed_attempt(
        self,
        attempt: WorkflowStepAttempt,
        *,
        output_object_refs: tuple[ObjectRef, ...],
        output_artifact_refs: tuple[ArtifactRef, ...],
        completed_at: datetime,
    ) -> WorkflowStepAttempt:
        """Atomically publish verified output references for an attempt."""

    def fail_attempt(
        self,
        attempt: WorkflowStepAttempt,
        *,
        failure: FailureInfo,
        completed_at: datetime,
    ) -> WorkflowStepAttempt:
        """Persist a structured terminal failure for an attempt."""

    def latest_successful_attempt(
        self, run_id: str, cycle_id: str, step_id: str
    ) -> WorkflowStepAttempt | None:
        """Return the latest successful attempt for one logical step."""

    def recover_interrupted(
        self,
        *,
        completed_at: datetime,
        run_id: str | None = None,
        cycle_id: str | None = None,
    ) -> tuple[WorkflowStepAttempt, ...]:
        """Mark matching unfinished attempts as interrupted failures."""


class WorkflowMetadataError(RepositoryError):
    """Raised when workflow metadata cannot be safely published or reused."""


class SqliteWorkflowMetadataRepository:
    """SQLite implementation of recoverable workflow metadata storage."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        object_verifier: ObjectReferenceVerifier,
        artifact_verifier: ArtifactReferenceVerifier,
    ) -> None:
        """Bind a repository to an already initialized SQLite database."""
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
    ) -> SqliteWorkflowMetadataRepository:
        """Explicitly create or migrate the local metadata schema."""
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30.0) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                raise WorkflowMetadataError("unsupported workflow metadata schema")
            if version == 0:
                connection.executescript(_SCHEMA_SQL)
                connection.execute("PRAGMA user_version = 1")
        return cls(
            path,
            object_verifier=object_verifier,
            artifact_verifier=artifact_verifier,
        )

    def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create a run, rejecting a conflicting reuse of its stable identifier."""
        self._verify_refs(
            (run.context_ref, run.data_snapshot_ref), (run.config_snapshot_ref,)
        )
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run.run_id,)
            ).fetchone()
            if existing is not None:
                stored = from_json(existing[0], WorkflowRun)
                if not _same_run_inputs(stored, run):
                    raise WorkflowMetadataError("RUN_INPUT_MISMATCH")
                return stored
            connection.execute(
                "INSERT INTO runs(run_id, payload) VALUES (?, ?)",
                (run.run_id, to_json(run)),
            )
        return run

    def create_cycle(self, cycle: WorkflowCycle) -> WorkflowCycle:
        """Create a cycle, rejecting a conflicting reuse of its stable identifier."""
        with self._connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (cycle.run_id,)
                ).fetchone()
                is None
            ):
                raise WorkflowMetadataError("WORKFLOW_RUN_NOT_FOUND")
            existing = connection.execute(
                "SELECT payload FROM cycles WHERE run_id = ? AND cycle_id = ?",
                (cycle.run_id, cycle.cycle_id),
            ).fetchone()
            if existing is not None:
                stored = from_json(existing[0], WorkflowCycle)
                if not _same_cycle_inputs(stored, cycle):
                    raise WorkflowMetadataError("CYCLE_INPUT_MISMATCH")
                return stored
            connection.execute(
                "INSERT INTO cycles(run_id, cycle_id, payload) VALUES (?, ?, ?)",
                (cycle.run_id, cycle.cycle_id, to_json(cycle)),
            )
        return cycle

    def complete_cycle(
        self, run_id: str, cycle_id: str, *, completed_at: datetime
    ) -> WorkflowCycle:
        """Mark one active cycle completed without mutating its input identity."""
        _require_utc(completed_at, "completed_at")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM cycles WHERE run_id = ? AND cycle_id = ?",
                (run_id, cycle_id),
            ).fetchone()
            if row is None:
                raise WorkflowMetadataError("WORKFLOW_CYCLE_NOT_FOUND")
            cycle = from_json(row[0], WorkflowCycle)
            if cycle.status is WorkflowCycleStatus.COMPLETED:
                return cycle
            completed = WorkflowCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                parent_ids=cycle.parent_ids,
                status=WorkflowCycleStatus.COMPLETED,
                created_at=cycle.created_at,
                completed_at=completed_at,
            )
            connection.execute(
                "UPDATE cycles SET payload = ? WHERE run_id = ? AND cycle_id = ?",
                (to_json(completed), run_id, cycle_id),
            )
        return completed

    def complete_run(self, run_id: str, *, completed_at: datetime) -> WorkflowRun:
        """Mark one active run completed without mutating its input identity."""
        _require_utc(completed_at, "completed_at")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise WorkflowMetadataError("WORKFLOW_RUN_NOT_FOUND")
            run = from_json(row[0], WorkflowRun)
            if run.status is WorkflowRunStatus.COMPLETED:
                return run
            completed = WorkflowRun(
                run_id=run.run_id,
                context_ref=run.context_ref,
                data_snapshot_ref=run.data_snapshot_ref,
                config_snapshot_ref=run.config_snapshot_ref,
                code_version=run.code_version,
                status=WorkflowRunStatus.COMPLETED,
                created_at=run.created_at,
                completed_at=completed_at,
            )
            connection.execute(
                "UPDATE runs SET payload = ? WHERE run_id = ?",
                (to_json(completed), run_id),
            )
        return completed

    def fail_cycle(
        self, run_id: str, cycle_id: str, *, completed_at: datetime
    ) -> WorkflowCycle:
        """Mark one active cycle failed without mutating its input identity."""
        _require_utc(completed_at, "completed_at")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM cycles WHERE run_id = ? AND cycle_id = ?",
                (run_id, cycle_id),
            ).fetchone()
            if row is None:
                raise WorkflowMetadataError("WORKFLOW_CYCLE_NOT_FOUND")
            cycle = from_json(row[0], WorkflowCycle)
            if cycle.status is not WorkflowCycleStatus.ACTIVE:
                return cycle
            failed = WorkflowCycle(
                run_id=cycle.run_id,
                cycle_id=cycle.cycle_id,
                parent_ids=cycle.parent_ids,
                status=WorkflowCycleStatus.FAILED,
                created_at=cycle.created_at,
                completed_at=completed_at,
            )
            connection.execute(
                "UPDATE cycles SET payload = ? WHERE run_id = ? AND cycle_id = ?",
                (to_json(failed), run_id, cycle_id),
            )
        return failed

    def fail_run(self, run_id: str, *, completed_at: datetime) -> WorkflowRun:
        """Mark one active run failed without mutating its input identity."""
        _require_utc(completed_at, "completed_at")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise WorkflowMetadataError("WORKFLOW_RUN_NOT_FOUND")
            run = from_json(row[0], WorkflowRun)
            if run.status is not WorkflowRunStatus.ACTIVE:
                return run
            failed = WorkflowRun(
                run_id=run.run_id,
                context_ref=run.context_ref,
                data_snapshot_ref=run.data_snapshot_ref,
                config_snapshot_ref=run.config_snapshot_ref,
                code_version=run.code_version,
                status=WorkflowRunStatus.FAILED,
                created_at=run.created_at,
                completed_at=completed_at,
            )
            connection.execute(
                "UPDATE runs SET payload = ? WHERE run_id = ?",
                (to_json(failed), run_id),
            )
        return failed

    def begin_attempt(
        self,
        *,
        run_id: str,
        cycle_id: str,
        step_id: str,
        input_object_refs: tuple[ObjectRef, ...],
        input_artifact_refs: tuple[ArtifactRef, ...],
        component: ComponentRef,
        started_at: datetime,
        reuse_success: bool = True,
    ) -> WorkflowStepAttempt:
        """Create a new attempt or reuse a verified successful output."""
        _require_text(run_id, "run_id")
        _require_text(cycle_id, "cycle_id")
        _require_text(step_id, "step_id")
        _require_refs(input_object_refs, ObjectRef, "input_object_refs")
        _require_refs(input_artifact_refs, ArtifactRef, "input_artifact_refs")
        _require_utc(started_at, "started_at")
        fingerprint = _input_fingerprint(input_object_refs, input_artifact_refs)
        with self._connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM cycles WHERE run_id = ? AND cycle_id = ?",
                    (run_id, cycle_id),
                ).fetchone()
                is None
            ):
                raise WorkflowMetadataError("WORKFLOW_CYCLE_NOT_FOUND")
            rows = connection.execute(
                "SELECT payload FROM step_attempts "
                "WHERE run_id = ? AND cycle_id = ? AND step_id = ? "
                "ORDER BY attempt DESC",
                (run_id, cycle_id, step_id),
            ).fetchall()
            attempts = tuple(from_json(row[0], WorkflowStepAttempt) for row in rows)
            for stored in attempts:
                if stored.status is WorkflowStepStatus.SUCCEEDED:
                    if stored.input_fingerprint != fingerprint:
                        raise WorkflowMetadataError("STEP_INPUT_MISMATCH")
                    if reuse_success:
                        if not self._outputs_resolve(stored):
                            raise WorkflowMetadataError("STEP_OUTPUT_REFERENCE_INVALID")
                        return stored
            if attempts and attempts[0].status is WorkflowStepStatus.STARTED:
                raise WorkflowMetadataError("STEP_ATTEMPT_IN_PROGRESS")
            next_attempt = 1 if not attempts else attempts[0].attempt + 1
            created = WorkflowStepAttempt(
                run_id=run_id,
                cycle_id=cycle_id,
                step_id=step_id,
                attempt=next_attempt,
                input_fingerprint=fingerprint,
                input_object_refs=input_object_refs,
                input_artifact_refs=input_artifact_refs,
                output_object_refs=(),
                output_artifact_refs=(),
                component=component,
                status=WorkflowStepStatus.STARTED,
                failure=None,
                started_at=started_at,
                completed_at=None,
            )
            connection.execute(
                "INSERT INTO step_attempts("
                "run_id, cycle_id, step_id, attempt, status, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    cycle_id,
                    step_id,
                    next_attempt,
                    WorkflowStepStatus.STARTED.value,
                    to_json(created),
                ),
            )
        return created

    def succeed_attempt(
        self,
        attempt: WorkflowStepAttempt,
        *,
        output_object_refs: tuple[ObjectRef, ...],
        output_artifact_refs: tuple[ArtifactRef, ...],
        completed_at: datetime,
    ) -> WorkflowStepAttempt:
        """Publish a successful attempt after verifying all referenced outputs."""
        self._require_started(attempt)
        self._verify_refs(output_object_refs, output_artifact_refs)
        _require_utc(completed_at, "completed_at")
        completed = WorkflowStepAttempt(
            run_id=attempt.run_id,
            cycle_id=attempt.cycle_id,
            step_id=attempt.step_id,
            attempt=attempt.attempt,
            input_fingerprint=attempt.input_fingerprint,
            input_object_refs=attempt.input_object_refs,
            input_artifact_refs=attempt.input_artifact_refs,
            output_object_refs=output_object_refs,
            output_artifact_refs=output_artifact_refs,
            component=attempt.component,
            status=WorkflowStepStatus.SUCCEEDED,
            failure=None,
            started_at=attempt.started_at,
            completed_at=completed_at,
        )
        self._replace_started(attempt, completed)
        return completed

    def fail_attempt(
        self,
        attempt: WorkflowStepAttempt,
        *,
        failure: FailureInfo,
        completed_at: datetime,
    ) -> WorkflowStepAttempt:
        """Publish a structured failed attempt."""
        self._require_started(attempt)
        if not isinstance(failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo")
        _require_utc(completed_at, "completed_at")
        completed = WorkflowStepAttempt(
            run_id=attempt.run_id,
            cycle_id=attempt.cycle_id,
            step_id=attempt.step_id,
            attempt=attempt.attempt,
            input_fingerprint=attempt.input_fingerprint,
            input_object_refs=attempt.input_object_refs,
            input_artifact_refs=attempt.input_artifact_refs,
            output_object_refs=(),
            output_artifact_refs=(),
            component=attempt.component,
            status=WorkflowStepStatus.FAILED,
            failure=failure,
            started_at=attempt.started_at,
            completed_at=completed_at,
        )
        self._replace_started(attempt, completed)
        return completed

    def latest_successful_attempt(
        self, run_id: str, cycle_id: str, step_id: str
    ) -> WorkflowStepAttempt | None:
        """Resolve the latest successful attempt for a logical step."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM step_attempts "
                "WHERE run_id = ? AND cycle_id = ? AND step_id = ? "
                "AND status = ? ORDER BY attempt DESC LIMIT 1",
                (run_id, cycle_id, step_id, WorkflowStepStatus.SUCCEEDED.value),
            ).fetchone()
        if row is None:
            return None
        stored = from_json(row[0], WorkflowStepAttempt)
        return stored if self._outputs_resolve(stored) else None

    def recover_interrupted(
        self,
        *,
        completed_at: datetime,
        run_id: str | None = None,
        cycle_id: str | None = None,
    ) -> tuple[WorkflowStepAttempt, ...]:
        """Convert unfinished attempts into retryable interrupted failures."""
        _require_utc(completed_at, "completed_at")
        if run_id is not None:
            _require_text(run_id, "run_id")
        if cycle_id is not None:
            _require_text(cycle_id, "cycle_id")
            if run_id is None:
                raise CoreValidationError("cycle_id requires run_id")
        clauses = ["status = ?"]
        parameters: list[str] = [WorkflowStepStatus.STARTED.value]
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            parameters.append(cycle_id)
        recovered: list[WorkflowStepAttempt] = []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM step_attempts WHERE " + " AND ".join(clauses),
                tuple(parameters),
            ).fetchall()
            for row in rows:
                attempt = from_json(row[0], WorkflowStepAttempt)
                failure = FailureInfo(
                    stage=attempt.step_id,
                    error_code="RECOVERY_INTERRUPTED",
                    message="workflow process ended before the step was published",
                    retryable=True,
                    input_refs=attempt.input_object_refs,
                    artifact_refs=attempt.input_artifact_refs,
                )
                interrupted = WorkflowStepAttempt(
                    run_id=attempt.run_id,
                    cycle_id=attempt.cycle_id,
                    step_id=attempt.step_id,
                    attempt=attempt.attempt,
                    input_fingerprint=attempt.input_fingerprint,
                    input_object_refs=attempt.input_object_refs,
                    input_artifact_refs=attempt.input_artifact_refs,
                    output_object_refs=(),
                    output_artifact_refs=(),
                    component=attempt.component,
                    status=WorkflowStepStatus.INTERRUPTED,
                    failure=failure,
                    started_at=attempt.started_at,
                    completed_at=completed_at,
                )
                connection.execute(
                    "UPDATE step_attempts SET status = ?, payload = ? "
                    "WHERE run_id = ? AND cycle_id = ? AND step_id = ? AND attempt = ? "
                    "AND status = ?",
                    (
                        WorkflowStepStatus.INTERRUPTED.value,
                        to_json(interrupted),
                        attempt.run_id,
                        attempt.cycle_id,
                        attempt.step_id,
                        attempt.attempt,
                        WorkflowStepStatus.STARTED.value,
                    ),
                )
                recovered.append(interrupted)
        return tuple(recovered)

    def _replace_started(
        self, previous: WorkflowStepAttempt, updated: WorkflowStepAttempt
    ) -> None:
        with self._connection() as connection:
            result = connection.execute(
                "UPDATE step_attempts SET status = ?, payload = ? "
                "WHERE run_id = ? AND cycle_id = ? AND step_id = ? AND attempt = ? "
                "AND status = ?",
                (
                    updated.status.value,
                    to_json(updated),
                    previous.run_id,
                    previous.cycle_id,
                    previous.step_id,
                    previous.attempt,
                    WorkflowStepStatus.STARTED.value,
                ),
            )
            if result.rowcount != 1:
                raise WorkflowMetadataError("STEP_ATTEMPT_NOT_STARTED")

    def _require_started(self, attempt: WorkflowStepAttempt) -> None:
        if not isinstance(attempt, WorkflowStepAttempt):
            raise CoreValidationError("attempt must be WorkflowStepAttempt")
        if attempt.status is not WorkflowStepStatus.STARTED:
            raise WorkflowMetadataError("STEP_ATTEMPT_NOT_STARTED")

    def _verify_refs(
        self,
        object_refs: Iterable[ObjectRef],
        artifact_refs: Iterable[ArtifactRef],
    ) -> None:
        for ref in object_refs:
            if not self._objects.exists(ref):
                raise WorkflowMetadataError("OBJECT_REFERENCE_INVALID")
        for ref in artifact_refs:
            if not self._artifacts.verify(ref):
                raise WorkflowMetadataError("ARTIFACT_REFERENCE_INVALID")

    def _outputs_resolve(self, attempt: WorkflowStepAttempt) -> bool:
        try:
            self._verify_refs(attempt.output_object_refs, attempt.output_artifact_refs)
        except WorkflowMetadataError:
            return False
        return True

    def _connection(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._database_path, timeout=30.0)
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise RecoveryError(
                f"workflow metadata database unavailable: {error}"
            ) from error


def _input_fingerprint(
    object_refs: tuple[ObjectRef, ...], artifact_refs: tuple[ArtifactRef, ...]
) -> str:
    return stable_sha256({"object_refs": object_refs, "artifact_refs": artifact_refs})


def _same_run_inputs(left: WorkflowRun, right: WorkflowRun) -> bool:
    return (
        left.run_id,
        left.context_ref,
        left.data_snapshot_ref,
        left.config_snapshot_ref,
        left.code_version,
        left.schema_version,
    ) == (
        right.run_id,
        right.context_ref,
        right.data_snapshot_ref,
        right.config_snapshot_ref,
        right.code_version,
        right.schema_version,
    )


def _same_cycle_inputs(left: WorkflowCycle, right: WorkflowCycle) -> bool:
    return (
        left.run_id,
        left.cycle_id,
        left.parent_ids,
        left.schema_version,
    ) == (
        right.run_id,
        right.cycle_id,
        right.parent_ids,
        right.schema_version,
    )


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cycles (
    run_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, cycle_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS step_attempts (
    run_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, cycle_id, step_id, attempt),
    FOREIGN KEY (run_id, cycle_id) REFERENCES cycles(run_id, cycle_id)
);
CREATE INDEX IF NOT EXISTS step_attempt_lookup
    ON step_attempts(run_id, cycle_id, step_id, status, attempt DESC);
"""
