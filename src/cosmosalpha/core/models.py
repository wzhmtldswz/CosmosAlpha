"""Immutable models shared by CosmosAlpha protocol boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from cosmosalpha.core.errors import CoreValidationError

_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _require_schema_version(value: object) -> str:
    version = _require_string(value, "schema_version")
    if _SCHEMA_VERSION_PATTERN.fullmatch(version) is None:
        raise CoreValidationError(
            "schema_version must match '<schema-name>/<major>.<minor>'"
        )
    return version


def _require_code(value: object, field_name: str) -> str:
    code = _require_string(value, field_name)
    if _CODE_PATTERN.fullmatch(code) is None:
        raise CoreValidationError(
            f"{field_name} must contain only uppercase letters, digits, and underscores"
        )
    return code


def _require_tuple_of(
    value: object,
    item_type: type[object],
    field_name: str,
) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    if not all(isinstance(item, item_type) for item in value):
        raise CoreValidationError(
            f"every item in {field_name} must be {item_type.__name__}"
        )


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Reference a persisted domain object by stable identity."""

    object_type: str
    object_id: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate the reference fields."""
        _require_string(self.object_type, "object_type")
        _require_string(self.object_id, "object_id")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference a persisted large payload without embedding its contents."""

    artifact_id: str
    kind: str
    uri: str
    media_type: str
    checksum: str
    size_bytes: int
    schema_version: str

    def __post_init__(self) -> None:
        """Validate artifact metadata and reject machine-local paths."""
        _require_string(self.artifact_id, "artifact_id")
        _require_string(self.kind, "kind")
        uri = _require_string(self.uri, "uri")
        _require_string(self.media_type, "media_type")
        _require_string(self.checksum, "checksum")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise CoreValidationError("size_bytes must be a non-negative integer")
        _require_schema_version(self.schema_version)

        parsed_uri = urlsplit(uri)
        if (
            parsed_uri.scheme.lower() == "file"
            or PurePosixPath(uri).is_absolute()
            or PureWindowsPath(uri).is_absolute()
        ):
            raise CoreValidationError(
                "uri must not use an absolute machine-local filesystem path"
            )


@dataclass(frozen=True, slots=True)
class ComponentRef:
    """Identify a component through its stable alias and implementation version."""

    alias: str
    version: str

    def __post_init__(self) -> None:
        """Validate the component identity."""
        _require_string(self.alias, "alias")
        _require_string(self.version, "version")


@dataclass(frozen=True, slots=True)
class WarningInfo:
    """Describe a non-fatal, machine-readable protocol warning."""

    code: str
    message: str

    def __post_init__(self) -> None:
        """Validate warning fields."""
        _require_code(self.code, "code")
        _require_string(self.message, "message")


@dataclass(frozen=True, slots=True)
class FailureInfo:
    """Describe an expected workflow failure without embedding an exception."""

    stage: str
    error_code: str
    message: str
    retryable: bool
    input_refs: tuple[ObjectRef, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate failure metadata and referenced evidence."""
        _require_string(self.stage, "stage")
        _require_code(self.error_code, "error_code")
        _require_string(self.message, "message")
        if type(self.retryable) is not bool:
            raise CoreValidationError("retryable must be a boolean")
        _require_tuple_of(self.input_refs, ObjectRef, "input_refs")
        _require_tuple_of(self.artifact_refs, ArtifactRef, "artifact_refs")


@dataclass(frozen=True, slots=True)
class SkipInfo:
    """Describe why a workflow step was intentionally not executed."""

    stage: str
    reason_code: str
    message: str
    input_refs: tuple[ObjectRef, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate skip metadata and referenced evidence."""
        _require_string(self.stage, "stage")
        _require_code(self.reason_code, "reason_code")
        _require_string(self.message, "message")
        _require_tuple_of(self.input_refs, ObjectRef, "input_refs")
        _require_tuple_of(self.artifact_refs, ArtifactRef, "artifact_refs")


class StepStatus(StrEnum):
    """Stable serialized states for a workflow step."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class StepOutcome[OutputT]:
    """Represent exactly one successful, failed, or skipped step result."""

    status: StepStatus
    output: OutputT | None = None
    failure: FailureInfo | None = None
    skip: SkipInfo | None = None
    warnings: tuple[WarningInfo, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        """Enforce state-specific output invariants."""
        if not isinstance(self.status, StepStatus):
            raise CoreValidationError("status must be a StepStatus")
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        if self.skip is not None and not isinstance(self.skip, SkipInfo):
            raise CoreValidationError("skip must be SkipInfo or None")
        _require_tuple_of(self.warnings, WarningInfo, "warnings")
        _require_tuple_of(self.artifact_refs, ArtifactRef, "artifact_refs")

        if self.status is StepStatus.SUCCESS:
            valid = (
                self.output is not None and self.failure is None and self.skip is None
            )
        elif self.status is StepStatus.FAILED:
            valid = (
                self.output is None and self.failure is not None and self.skip is None
            )
        else:
            valid = (
                self.output is None and self.failure is None and self.skip is not None
            )
        if not valid:
            raise CoreValidationError(
                f"invalid field combination for step status {self.status.value!r}"
            )

    @classmethod
    def success(
        cls,
        output: OutputT,
        *,
        warnings: tuple[WarningInfo, ...] = (),
        artifact_refs: tuple[ArtifactRef, ...] = (),
    ) -> StepOutcome[OutputT]:
        """Create a successful outcome containing a real output."""
        return cls(
            status=StepStatus.SUCCESS,
            output=output,
            warnings=warnings,
            artifact_refs=artifact_refs,
        )

    @classmethod
    def failed(
        cls,
        failure: FailureInfo,
        *,
        warnings: tuple[WarningInfo, ...] = (),
        artifact_refs: tuple[ArtifactRef, ...] = (),
    ) -> StepOutcome[OutputT]:
        """Create a failed outcome containing structured failure details."""
        return cls(
            status=StepStatus.FAILED,
            failure=failure,
            warnings=warnings,
            artifact_refs=artifact_refs,
        )

    @classmethod
    def skipped(
        cls,
        skip: SkipInfo,
        *,
        warnings: tuple[WarningInfo, ...] = (),
        artifact_refs: tuple[ArtifactRef, ...] = (),
    ) -> StepOutcome[OutputT]:
        """Create a skipped outcome containing a structured reason."""
        return cls(
            status=StepStatus.SKIPPED,
            skip=skip,
            warnings=warnings,
            artifact_refs=artifact_refs,
        )
