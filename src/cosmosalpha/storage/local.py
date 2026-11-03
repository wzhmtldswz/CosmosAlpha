"""Local filesystem artifact and experiment repositories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cosmosalpha.agents import ExperimentRecord
from cosmosalpha.core import (
    ArtifactRef,
    ConfigSnapshot,
    CoreValidationError,
    JsonValue,
    ObjectRef,
    SerializationError,
    decode_config_snapshot,
    from_json,
    to_json,
)
from cosmosalpha.storage.errors import (
    ArtifactNotFoundError,
    IntegrityError,
    ObjectConflictError,
    ObjectNotFoundError,
    RecoveryError,
    RepositoryError,
)

_SAFE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ARTIFACT_SCHEMA_VERSION = "artifact-ref/1.0"
_EXPERIMENT_OBJECT_TYPE = "experiment_record"
_STORAGE_NAME_SCHEMA_VERSION = "storage-name/1.0"
_STORAGE_NAME_PREFIX = "_names"


@dataclass(frozen=True, slots=True)
class RecoveryIssue:
    """Describe one invalid object discovered during recovery scanning."""

    path: str
    message: str

    def __post_init__(self) -> None:
        """Validate recovery issue fields."""
        if type(self.path) is not str or not self.path:
            raise CoreValidationError("path must be a non-empty string")
        if type(self.message) is not str or not self.message:
            raise CoreValidationError("message must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Report produced by a recovery scan."""

    issues: tuple[RecoveryIssue, ...]

    def __post_init__(self) -> None:
        """Validate recovery report fields."""
        if not isinstance(self.issues, tuple):
            raise CoreValidationError("issues must be a tuple")
        if not all(isinstance(issue, RecoveryIssue) for issue in self.issues):
            raise CoreValidationError("every recovery issue must be RecoveryIssue")


@dataclass(frozen=True, slots=True)
class StorageNameRecord:
    """Human-readable timestamp index entry for a stored resource."""

    resource_kind: str
    resource_type: str
    resource_id: str
    display_name: str
    created_at: datetime
    schema_version: str = _STORAGE_NAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate a timestamped storage name record."""
        for field_name in (
            "resource_kind",
            "resource_type",
            "resource_id",
            "display_name",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value != value.strip():
                raise CoreValidationError(
                    f"{field_name} must be a non-empty string without surrounding whitespace"
                )
        _require_utc(self.created_at, "created_at")
        if self.schema_version != _STORAGE_NAME_SCHEMA_VERSION:
            raise CoreValidationError("invalid storage name schema_version")


@dataclass(frozen=True, slots=True)
class RunStorageLayout:
    """Resolved local paths for one self-contained experiment run."""

    run_root: Path
    cycles_root: Path
    workflow_db_path: Path
    factor_library_db_path: Path
    factor_library_csv_path: Path

    @classmethod
    def resolve(cls, storage_root: str | Path, run_id: str) -> RunStorageLayout:
        """Resolve a safe ``<storage_root>/<run_id>`` output layout."""
        if type(run_id) is not str or _SAFE_RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise CoreValidationError("run_id must be a safe storage directory name")
        run_root = Path(storage_root).expanduser() / run_id
        return cls(
            run_root=run_root,
            cycles_root=run_root / "cycles",
            workflow_db_path=run_root / "workflow.sqlite3",
            factor_library_db_path=run_root / "factor-library.sqlite3",
            factor_library_csv_path=run_root / "factor_library" / "factors.csv",
        )


class LocalArtifactRepository:
    """Content-addressed artifact repository backed by local files."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a local artifact repository.

        Args:
            root: Repository root directory.
            clock: Optional UTC clock used for human-readable storage names.
        """
        self.root = Path(root)
        self.artifacts_root = self.root / "artifacts"
        self.names_root = self.root / _STORAGE_NAME_PREFIX
        self._clock = clock or _utc_now
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.names_root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, *, kind: str, media_type: str) -> ArtifactRef:
        """Persist bytes and return a stable artifact reference."""
        if type(data) is not bytes:
            raise RepositoryError("data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        path = self._artifact_path(digest)
        if not path.exists():
            _atomic_write_bytes(path, data)
        else:
            self._verify_bytes(path.read_bytes(), digest, len(data))
        created_at = self._clock()
        _ensure_storage_name(
            self.names_root,
            StorageNameRecord(
                resource_kind="artifact",
                resource_type=kind,
                resource_id=digest,
                display_name=_storage_display_name(
                    created_at, "artifact", kind, digest
                ),
                created_at=created_at,
            ),
        )
        return ArtifactRef(
            artifact_id=digest,
            kind=kind,
            uri=f"artifact://local/{digest}",
            media_type=media_type,
            checksum=f"sha256:{digest}",
            size_bytes=len(data),
            schema_version=_ARTIFACT_SCHEMA_VERSION,
        )

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read artifact bytes after verifying size and checksum."""
        if not isinstance(ref, ArtifactRef):
            raise RepositoryError("ref must be ArtifactRef")
        path = self._artifact_path(ref.artifact_id)
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact not found: {ref.artifact_id}")
        data = path.read_bytes()
        self._verify_ref(data, ref)
        return data

    def exists(self, ref: ArtifactRef) -> bool:
        """Return whether an artifact file exists for the reference."""
        if not isinstance(ref, ArtifactRef):
            raise RepositoryError("ref must be ArtifactRef")
        return self._artifact_path(ref.artifact_id).exists()

    def verify(self, ref: ArtifactRef) -> bool:
        """Return whether an artifact exists and matches its integrity metadata."""
        try:
            self.get_bytes(ref)
        except (ArtifactNotFoundError, IntegrityError):
            return False
        return True

    def _artifact_path(self, artifact_id: str) -> Path:
        _require_safe_segment(artifact_id, "artifact_id")
        return self.artifacts_root / artifact_id[:2] / artifact_id

    def _verify_ref(self, data: bytes, ref: ArtifactRef) -> None:
        expected_checksum = f"sha256:{ref.artifact_id}"
        if ref.uri != f"artifact://local/{ref.artifact_id}":
            raise IntegrityError("artifact URI does not belong to this repository")
        if ref.checksum != expected_checksum:
            raise IntegrityError("artifact checksum does not match artifact id")
        self._verify_bytes(data, ref.artifact_id, ref.size_bytes)

    def _verify_bytes(self, data: bytes, digest: str, size_bytes: int) -> None:
        if len(data) != size_bytes:
            raise IntegrityError("artifact size does not match reference")
        actual_digest = hashlib.sha256(data).hexdigest()
        if actual_digest != digest:
            raise IntegrityError("artifact checksum mismatch")


class LocalExperimentRepository:
    """Strict JSON object repository backed by local files."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a local experiment object repository."""
        self.root = Path(root)
        self.objects_root = self.root / "objects"
        self.names_root = self.root / _STORAGE_NAME_PREFIX
        self._clock = clock or _utc_now
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.names_root.mkdir(parents=True, exist_ok=True)

    def put_object(self, object_type: str, object_id: str, value: object) -> ObjectRef:
        """Persist a strict JSON object and return its object reference."""
        _require_safe_segment(object_type, "object_type")
        _require_safe_segment(object_id, "object_id")
        schema_version = getattr(value, "schema_version", None)
        if type(schema_version) is not str or not schema_version:
            raise RepositoryError("stored object must expose a schema_version string")
        data = to_json(value).encode("utf-8")
        path = self._object_path(object_type, object_id)
        if not _atomic_create_bytes(path, data):
            try:
                existing = path.read_bytes()
            except FileNotFoundError as error:
                raise RepositoryError(
                    f"object disappeared during immutable publish: "
                    f"{object_type}/{object_id}"
                ) from error
            if existing != data:
                raise ObjectConflictError(
                    f"OBJECT_CONTENT_CONFLICT: {object_type}/{object_id}"
                )
        created_at = self._clock()
        _ensure_storage_name(
            self.names_root,
            StorageNameRecord(
                resource_kind="object",
                resource_type=object_type,
                resource_id=object_id,
                display_name=_storage_display_name(
                    created_at, "object", object_type, object_id
                ),
                created_at=created_at,
            ),
        )
        return ObjectRef(
            object_type=object_type,
            object_id=object_id,
            schema_version=schema_version,
        )

    def get_object[ObjectT](
        self, ref: ObjectRef, target_type: type[ObjectT]
    ) -> ObjectT:
        """Resolve an object reference and strictly decode it."""
        if not isinstance(ref, ObjectRef):
            raise RepositoryError("ref must be ObjectRef")
        path = self._object_path(ref.object_type, ref.object_id)
        if not path.exists():
            raise ObjectNotFoundError(f"object not found: {ref.object_id}")
        data = path.read_bytes()
        if target_type is ConfigSnapshot:
            return cast(ObjectT, decode_config_snapshot(data))
        return from_json(data, target_type)

    def exists(self, ref: ObjectRef) -> bool:
        """Return whether an object file exists for a reference."""
        if not isinstance(ref, ObjectRef):
            raise RepositoryError("ref must be ObjectRef")
        return self._object_path(ref.object_type, ref.object_id).exists()

    def list_object_ids(self, object_type: str) -> tuple[str, ...]:
        """Return persisted object IDs for one safe object type."""
        _require_safe_segment(object_type, "object_type")
        root = self.objects_root / object_type
        if not root.exists():
            return ()
        return tuple(sorted(path.stem for path in root.glob("*.json")))

    def put_experiment(self, record: ExperimentRecord) -> ObjectRef:
        """Persist an experiment record."""
        if not isinstance(record, ExperimentRecord):
            raise RepositoryError("record must be ExperimentRecord")
        return self.put_object(
            _EXPERIMENT_OBJECT_TYPE,
            record.experiment_id,
            record,
        )

    def get_experiment(self, ref: ObjectRef) -> ExperimentRecord:
        """Resolve an experiment record reference."""
        if ref.object_type != _EXPERIMENT_OBJECT_TYPE:
            raise RepositoryError("ref must point to an experiment_record object")
        return self.get_object(ref, ExperimentRecord)

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        """Return all readable experiment records sorted by experiment id."""
        records: list[ExperimentRecord] = []
        root = self.objects_root / _EXPERIMENT_OBJECT_TYPE
        if not root.exists():
            return ()
        for path in sorted(root.glob("*.json")):
            try:
                records.append(from_json(path.read_bytes(), ExperimentRecord))
            except Exception:
                continue
        return tuple(sorted(records, key=lambda record: record.experiment_id))

    def latest_completed(self, run_id: str | None = None) -> ExperimentRecord | None:
        """Return the latest completed experiment record."""
        candidates = [
            record
            for record in self.list_experiments()
            if record.completed_at is not None
            and (run_id is None or record.run_id == run_id)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.completed_at)

    def recover(self) -> RecoveryReport:
        """Scan complete object files and report unreadable JSON objects."""
        issues: list[RecoveryIssue] = []
        if not self.objects_root.exists():
            return RecoveryReport(issues=())
        for path in sorted(self.objects_root.rglob("*")):
            if path.is_dir() or path.suffix == ".tmp":
                continue
            if path.suffix != ".json":
                continue
            try:
                _strict_json_object(path.read_bytes())
            except Exception as error:
                issues.append(
                    RecoveryIssue(
                        path=str(path.relative_to(self.root)),
                        message=str(error),
                    )
                )
        return RecoveryReport(issues=tuple(issues))

    def _object_path(self, object_type: str, object_id: str) -> Path:
        _require_safe_segment(object_type, "object_type")
        _require_safe_segment(object_id, "object_id")
        return self.objects_root / object_type / f"{object_id}.json"


def _require_safe_segment(value: object, field_name: str) -> str:
    if type(value) is not str or _SAFE_SEGMENT_PATTERN.fullmatch(value) is None:
        raise CoreValidationError(f"{field_name} must be a safe path segment")
    return value


def _require_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CoreValidationError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        raise CoreValidationError(f"{field_name} must be a UTC datetime")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _storage_display_name(
    created_at: datetime,
    resource_kind: str,
    resource_type: str,
    resource_id: str,
) -> str:
    """Build a sortable, readable name without changing resource identity."""
    _require_utc(created_at, "created_at")
    return "__".join(
        (
            created_at.strftime("%Y%m%dT%H%M%S%fZ"),
            _safe_name_segment(resource_kind),
            _safe_name_segment(resource_type),
            _safe_name_segment(resource_id[:16]),
        )
    )


def _safe_name_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "unknown"


def _ensure_storage_name(names_root: Path, record: StorageNameRecord) -> None:
    """Persist one first-write timestamp index entry for a resource."""
    lookup_path = (
        names_root
        / "by_id"
        / record.resource_kind
        / _safe_name_segment(record.resource_type)
        / f"{_safe_name_segment(record.resource_id)}.json"
    )
    data = to_json(record).encode("utf-8")
    if _atomic_create_bytes(lookup_path, data):
        stored_record = record
    else:
        try:
            stored_record = from_json(lookup_path.read_bytes(), StorageNameRecord)
        except FileNotFoundError as error:
            raise RepositoryError(
                f"storage name lookup disappeared during publish: {lookup_path}"
            ) from error

    display_path = names_root / f"{stored_record.display_name}.json"
    if _atomic_create_bytes(
        display_path,
        to_json(stored_record).encode("utf-8"),
    ):
        return

    try:
        existing = from_json(display_path.read_bytes(), StorageNameRecord)
    except FileNotFoundError:
        if _atomic_create_bytes(
            display_path,
            to_json(stored_record).encode("utf-8"),
        ):
            return
        existing = from_json(display_path.read_bytes(), StorageNameRecord)
    if existing == stored_record:
        return

    suffix = 2
    while True:
        candidate = names_root / f"{stored_record.display_name}__{suffix:02d}.json"
        if _atomic_create_bytes(candidate, to_json(stored_record).encode("utf-8")):
            return
        try:
            existing = from_json(candidate.read_bytes(), StorageNameRecord)
        except FileNotFoundError:
            continue
        if existing == stored_record:
            return
        suffix += 1


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        tmp_path.unlink(missing_ok=True)


def _atomic_create_bytes(path: Path, data: bytes) -> bool:
    """Publish bytes once without replacing an existing path.

    Returns:
        True when this invocation created ``path``; False when another writer
        already created it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if descriptor != -1:
            os.close(descriptor)
        tmp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json_object(data: bytes) -> dict[str, JsonValue]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, JsonValue]],
    ) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise SerializationError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SerializationError(f"non-finite JSON number is forbidden: {value}")

    try:
        parsed = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RecoveryError(f"invalid JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise RecoveryError("stored object JSON must be an object")
    if type(parsed.get("schema_version")) is not str:
        raise RecoveryError("stored object must contain schema_version")
    return parsed
