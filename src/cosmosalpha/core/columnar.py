"""Versioned NumPy columnar codec for large tabular artifacts.

The codec deliberately keeps the public artifact contract as ``ArtifactRef``.
It stores each logical field in a separate typed NumPy array and does not use
pickle, making it suitable for market, mask, factor-value, and return tables.
Legacy JSON record artifacts remain readable during the migration.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from cosmosalpha.core.serialization import JsonValue, to_json

COLUMNAR_RECORDS_MEDIA_TYPE = "application/vnd.cosmosalpha.columnar-npz"
COLUMNAR_RECORDS_SCHEMA_VERSION = "columnar-records/1.0"
_MANIFEST_KEY = "__cosmosalpha_manifest__"
_COLUMN_TYPES = {"bool", "float", "int", "str"}
_COMPRESSION_LEVEL = 1


class ColumnarRecordsError(ValueError):
    """Raised when a columnar record artifact is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class ColumnarTable:
    """Read-only NumPy columns decoded without materializing row dictionaries."""

    row_count: int
    columns: Mapping[str, np.ndarray]
    validity: Mapping[str, np.ndarray]
    column_types: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate aligned one-dimensional columns and validity arrays."""
        names = set(self.columns)
        if names != set(self.validity) or names != set(self.column_types):
            raise ColumnarRecordsError("columnar table mappings must share keys")
        for name in names:
            values = self.columns[name]
            valid = self.validity[name]
            if values.shape != (self.row_count,) or valid.shape != (self.row_count,):
                raise ColumnarRecordsError(
                    f"columnar table column {name!r} has invalid length"
                )
            if valid.dtype != np.bool_:
                raise ColumnarRecordsError(
                    f"columnar table validity {name!r} must be boolean"
                )


def encode_columnar_records(records: Sequence[Mapping[str, object]]) -> bytes:
    """Encode scalar records into a compressed, typed columnar artifact.

    Missing values are represented by per-column validity arrays.  The format
    stores no object arrays and is always decoded with ``allow_pickle=False``.

    Args:
        records: Rows consisting only of JSON scalar values or ``None``.

    Returns:
        Compressed NumPy archive bytes with a versioned manifest.

    Raises:
        ColumnarRecordsError: If rows are not scalar mappings or field types
            cannot be represented by the stable columnar schema.
    """
    rows = tuple(records)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ColumnarRecordsError("every record must be a mapping")
        if any(type(key) is not str or not key for key in row):
            raise ColumnarRecordsError("record field names must be non-empty strings")

    names = tuple(sorted({name for row in rows for name in row}))
    arrays: dict[str, np.ndarray] = {}
    columns: list[dict[str, JsonValue]] = []
    for index, name in enumerate(names):
        values = tuple(row.get(name) for row in rows)
        column_type = _infer_column_type(values, name)
        data, valid = _encode_column(values, column_type)
        data_key = f"column_{index}"
        valid_key = f"valid_{index}"
        arrays[data_key] = data
        arrays[valid_key] = valid
        columns.append(
            {
                "name": name,
                "type": column_type,
                "data_key": data_key,
                "valid_key": valid_key,
            }
        )

    manifest = {
        "schema_version": COLUMNAR_RECORDS_SCHEMA_VERSION,
        "row_count": len(rows),
        "columns": columns,
    }
    arrays[_MANIFEST_KEY] = np.array(to_json(manifest))
    return _encode_archive(arrays)


def encode_columnar_columns(
    columns: Mapping[str, np.ndarray],
    *,
    validity: Mapping[str, np.ndarray] | None = None,
) -> bytes:
    """Encode aligned NumPy columns using the existing columnar-record schema."""
    if not isinstance(columns, Mapping) or not columns:
        raise ColumnarRecordsError("columns must be a non-empty mapping")
    names = tuple(sorted(columns))
    if any(type(name) is not str or not name for name in names):
        raise ColumnarRecordsError("column names must be non-empty strings")
    row_count: int | None = None
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, JsonValue]] = []
    for index, name in enumerate(names):
        raw = columns[name]
        if not isinstance(raw, np.ndarray) or raw.ndim != 1:
            raise ColumnarRecordsError(f"column {name!r} must be a 1D NumPy array")
        if row_count is None:
            row_count = int(raw.size)
        elif raw.size != row_count:
            raise ColumnarRecordsError("column lengths must match")
        column_type = _array_column_type(raw, name)
        valid = _column_validity(raw, name, column_type, validity)
        data = _normalize_column_array(raw, valid, column_type)
        data_key = f"column_{index}"
        valid_key = f"valid_{index}"
        arrays[data_key] = data
        arrays[valid_key] = valid
        metadata.append(
            {
                "name": name,
                "type": column_type,
                "data_key": data_key,
                "valid_key": valid_key,
            }
        )
    if validity is not None and set(validity) - set(names):
        raise ColumnarRecordsError("validity contains unknown columns")
    manifest = {
        "schema_version": COLUMNAR_RECORDS_SCHEMA_VERSION,
        "row_count": row_count or 0,
        "columns": metadata,
    }
    arrays[_MANIFEST_KEY] = np.array(to_json(manifest))
    return _encode_archive(arrays)


def _encode_archive(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Write a compatible NPZ archive with low-overhead DEFLATE compression."""
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=_COMPRESSION_LEVEL,
        allowZip64=True,
    ) as archive:
        for name, values in arrays.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member,
                    np.asanyarray(values),
                    allow_pickle=False,
                )
    return stream.getvalue()


def decode_columnar_table(data: bytes) -> ColumnarTable:
    """Decode typed NumPy columns without constructing per-row dictionaries."""
    if type(data) is not bytes:
        raise ColumnarRecordsError("columnar artifact must be bytes")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if _MANIFEST_KEY not in archive.files:
                raise ColumnarRecordsError("columnar artifact is missing manifest")
            manifest = _decode_manifest(archive[_MANIFEST_KEY])
            row_count, metadata = _validate_manifest(manifest)
            columns: dict[str, np.ndarray] = {}
            validity: dict[str, np.ndarray] = {}
            column_types: dict[str, str] = {}
            for column in metadata:
                name = cast(str, column["name"])
                values, valid = _decode_column_arrays(archive, column, row_count)
                values.setflags(write=False)
                valid.setflags(write=False)
                columns[name] = values
                validity[name] = valid
                column_types[name] = cast(str, column["type"])
    except (OSError, ValueError, KeyError, UnicodeDecodeError) as error:
        if isinstance(error, ColumnarRecordsError):
            raise
        raise ColumnarRecordsError("invalid columnar artifact") from error
    return ColumnarTable(
        row_count=row_count,
        columns=MappingProxyType(columns),
        validity=MappingProxyType(validity),
        column_types=MappingProxyType(column_types),
    )


def decode_record_artifact_table(data: bytes) -> ColumnarTable:
    """Decode current columnar bytes or convert a legacy JSON record artifact."""
    if data.startswith(b"PK"):
        return decode_columnar_table(data)
    records = decode_record_artifact(data)
    if not records:
        raise ColumnarRecordsError("record artifact must not be empty")
    names = tuple(sorted({name for record in records for name in record}))
    columns: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    column_types: dict[str, str] = {}
    for name in names:
        values = tuple(record.get(name) for record in records)
        column_type = _infer_column_type(values, name)
        data_column, valid = _encode_column(values, column_type)
        data_column.setflags(write=False)
        valid.setflags(write=False)
        columns[name] = data_column
        validity[name] = valid
        column_types[name] = column_type
    return ColumnarTable(
        row_count=len(records),
        columns=MappingProxyType(columns),
        validity=MappingProxyType(validity),
        column_types=MappingProxyType(column_types),
    )


def decode_columnar_records(data: bytes) -> tuple[Mapping[str, object], ...]:
    """Decode one versioned columnar record artifact.

    Args:
        data: Bytes written by :func:`encode_columnar_records`.

    Returns:
        Immutable sequence of decoded record mappings.

    Raises:
        ColumnarRecordsError: If the bytes do not form a valid supported table.
    """
    if type(data) is not bytes:
        raise ColumnarRecordsError("columnar artifact must be bytes")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if _MANIFEST_KEY not in archive.files:
                raise ColumnarRecordsError("columnar artifact is missing manifest")
            manifest = _decode_manifest(archive[_MANIFEST_KEY])
            row_count, columns = _validate_manifest(manifest)
            decoded_columns = [
                _decode_column(archive, column, row_count) for column in columns
            ]
    except (OSError, ValueError, KeyError, UnicodeDecodeError) as error:
        if isinstance(error, ColumnarRecordsError):
            raise
        raise ColumnarRecordsError("invalid columnar artifact") from error

    records: list[Mapping[str, object]] = []
    for row_index in range(row_count):
        records.append(
            {
                str(column["name"]): values[row_index]
                for column, values in zip(columns, decoded_columns, strict=True)
            }
        )
    return tuple(records)


def decode_record_artifact(data: bytes) -> tuple[Mapping[str, object], ...]:
    """Decode a columnar artifact or the legacy JSON ``{"records": ...}`` form.

    This compatibility reader lets completed JSON-backed attempts remain
    recoverable while all newly published large tables use columnar storage.
    """
    if data.startswith(b"PK"):
        return decode_columnar_records(data)
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ColumnarRecordsError("invalid JSON record artifact") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ColumnarRecordsError("record artifact must contain records")
    records: list[Mapping[str, object]] = []
    for row in payload["records"]:
        if not isinstance(row, dict):
            raise ColumnarRecordsError("every record must be an object")
        records.append(cast(Mapping[str, object], row))
    return tuple(records)


def is_columnar_record_artifact(data: bytes) -> bool:
    """Return whether bytes use the CosmosAlpha columnar archive envelope."""
    return type(data) is bytes and data.startswith(b"PK")


def records_to_aligned_values(
    records: Sequence[Mapping[str, object]],
    *,
    preserve_missing: bool = False,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Records → (factor value array, instrument@datetime alignment keys).

    Falls back to ``row{idx}@datetime`` alignment when no instrument field is
    present. When ``preserve_missing`` is true, null and non-finite rows remain
    as NaN so daily overlap follows the Graph correlation definition. Returns
    ``(None, ())`` when fewer than 2 valid values remain.

    This conversion provides consistent factor alignment for correlation checks.
    """
    if not records:
        return None, ()
    has_instrument = any(r.get("instrument") is not None for r in records)
    values: list[float] = []
    keys: list[str] = []
    valid_count = 0
    for idx, record in enumerate(records):
        value = record.get("factor_value", np.nan)
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            float_value = math.nan
        dt = str(record.get("datetime", ""))
        inst = str(record.get("instrument", ""))
        if has_instrument and inst:
            key = f"{inst}@{dt}"
        else:
            key = f"row{idx}@{dt}"
        if not np.isfinite(float_value):
            if preserve_missing:
                values.append(math.nan)
                keys.append(key)
            continue
        valid_count += 1
        values.append(float_value)
        keys.append(key)
    if valid_count < 2:
        return None, ()
    return np.asarray(values, dtype=float), tuple(keys)


def _infer_column_type(values: Sequence[object], name: str) -> str:
    non_null = tuple(value for value in values if value is not None)
    if not non_null:
        return "float"
    if all(type(value) is bool for value in non_null):
        return "bool"
    if all(type(value) is int for value in non_null):
        return "int"
    if all(type(value) in (int, float) for value in non_null):
        for value in non_null:
            if type(value) is float and not math.isfinite(value):
                raise ColumnarRecordsError(f"{name}: non-finite floats are not allowed")
        return "float"
    if all(type(value) is str for value in non_null):
        return "str"
    raise ColumnarRecordsError(f"{name}: values must have one scalar column type")


def _encode_column(
    values: Sequence[object], column_type: str
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray([value is not None for value in values], dtype=np.bool_)
    if column_type == "bool":
        return (
            np.asarray(
                [bool(value) if value is not None else False for value in values]
            ),
            valid,
        )
    if column_type == "int":
        return (
            np.asarray([int(value) if value is not None else 0 for value in values]),
            valid,
        )
    if column_type == "float":
        return (
            np.asarray(
                [float(value) if value is not None else np.nan for value in values],
                dtype=np.float64,
            ),
            valid,
        )
    if column_type == "str":
        return (
            np.asarray([str(value) if value is not None else "" for value in values]),
            valid,
        )
    raise ColumnarRecordsError(f"unsupported column type: {column_type}")


def _array_column_type(values: np.ndarray, name: str) -> str:
    if values.dtype == np.bool_:
        return "bool"
    if values.dtype.kind in {"i", "u"}:
        return "int"
    if values.dtype.kind == "f":
        return "float"
    if values.dtype.kind in {"U", "S"}:
        return "str"
    raise ColumnarRecordsError(f"column {name!r} has unsupported dtype")


def _column_validity(
    values: np.ndarray,
    name: str,
    column_type: str,
    validity: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    supplied = None if validity is None else validity.get(name)
    if supplied is None:
        valid = (
            np.isfinite(values)
            if column_type == "float"
            else np.ones(values.size, dtype=np.bool_)
        )
    else:
        if (
            not isinstance(supplied, np.ndarray)
            or supplied.shape != values.shape
            or supplied.dtype != np.bool_
        ):
            raise ColumnarRecordsError(
                f"validity for column {name!r} must be an aligned boolean array"
            )
        valid = supplied.copy()
    if column_type == "float" and np.any(~np.isfinite(values[valid])):
        raise ColumnarRecordsError(f"column {name!r} contains non-finite valid values")
    return valid.astype(np.bool_, copy=False)


def _normalize_column_array(
    values: np.ndarray,
    valid: np.ndarray,
    column_type: str,
) -> np.ndarray:
    if column_type == "bool":
        result = values.astype(np.bool_, copy=True)
        result[~valid] = False
        return result
    if column_type == "int":
        result = values.astype(np.int64, copy=True)
        result[~valid] = 0
        return result
    if column_type == "float":
        result = values.astype(np.float64, copy=True)
        result[~valid] = np.nan
        return result
    if column_type == "str":
        result = values.astype(str, copy=True)
        result[~valid] = ""
        return result
    raise ColumnarRecordsError(f"unsupported column type: {column_type}")


def _decode_manifest(value: np.ndarray) -> object:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ColumnarRecordsError("columnar manifest has invalid shape or dtype")
    try:
        return json.loads(str(value.item()))
    except json.JSONDecodeError as error:
        raise ColumnarRecordsError("columnar manifest is invalid JSON") from error


def _validate_manifest(
    manifest: object,
) -> tuple[int, tuple[Mapping[str, object], ...]]:
    if not isinstance(manifest, dict):
        raise ColumnarRecordsError("columnar manifest must be an object")
    if manifest.get("schema_version") != COLUMNAR_RECORDS_SCHEMA_VERSION:
        raise ColumnarRecordsError("unsupported columnar schema version")
    row_count = manifest.get("row_count")
    columns = manifest.get("columns")
    if type(row_count) is not int or row_count < 0:
        raise ColumnarRecordsError("columnar row_count must be non-negative")
    if not isinstance(columns, list):
        raise ColumnarRecordsError("columnar columns must be a list")
    names: set[str] = set()
    validated: list[Mapping[str, object]] = []
    for column in columns:
        if not isinstance(column, dict):
            raise ColumnarRecordsError("columnar column metadata must be an object")
        name = column.get("name")
        column_type = column.get("type")
        if type(name) is not str or not name or name in names:
            raise ColumnarRecordsError("columnar field names must be unique strings")
        if column_type not in _COLUMN_TYPES:
            raise ColumnarRecordsError("columnar field type is unsupported")
        if not all(type(column.get(key)) is str for key in ("data_key", "valid_key")):
            raise ColumnarRecordsError("columnar column keys must be strings")
        names.add(name)
        validated.append(cast(Mapping[str, object], column))
    return row_count, tuple(validated)


def _decode_column(
    archive: object,
    column: Mapping[str, object],
    row_count: int,
) -> tuple[object, ...]:
    data_key = cast(str, column["data_key"])
    valid_key = cast(str, column["valid_key"])
    data = archive[data_key]  # type: ignore[index]
    valid = archive[valid_key]  # type: ignore[index]
    if data.shape != (row_count,) or valid.shape != (row_count,):
        raise ColumnarRecordsError("columnar column length does not match row_count")
    if valid.dtype != np.bool_:
        raise ColumnarRecordsError("columnar validity array must be boolean")
    column_type = cast(str, column["type"])
    if column_type == "bool":
        if data.dtype != np.bool_:
            raise ColumnarRecordsError("boolean column has invalid dtype")
        return tuple(
            bool(value) if keep else None
            for value, keep in zip(data, valid, strict=True)
        )
    if column_type == "int":
        if data.dtype.kind not in {"i", "u"}:
            raise ColumnarRecordsError("integer column has invalid dtype")
        return tuple(
            int(value) if keep else None
            for value, keep in zip(data, valid, strict=True)
        )
    if column_type == "float":
        if data.dtype.kind != "f" or np.any(~np.isfinite(data[valid])):
            raise ColumnarRecordsError("float column contains invalid values")
        return tuple(
            float(value) if keep else None
            for value, keep in zip(data, valid, strict=True)
        )
    if column_type == "str":
        if data.dtype.kind not in {"U", "S"}:
            raise ColumnarRecordsError("string column has invalid dtype")
        return tuple(
            str(value) if keep else None
            for value, keep in zip(data, valid, strict=True)
        )
    raise ColumnarRecordsError("unsupported column type")


def _decode_column_arrays(
    archive: object,
    column: Mapping[str, object],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    data_key = cast(str, column["data_key"])
    valid_key = cast(str, column["valid_key"])
    data = np.asarray(archive[data_key]).copy()  # type: ignore[index]
    valid = np.asarray(archive[valid_key]).copy()  # type: ignore[index]
    if data.shape != (row_count,) or valid.shape != (row_count,):
        raise ColumnarRecordsError("columnar column length does not match row_count")
    if valid.dtype != np.bool_:
        raise ColumnarRecordsError("columnar validity array must be boolean")
    column_type = cast(str, column["type"])
    if column_type == "bool" and data.dtype != np.bool_:
        raise ColumnarRecordsError("boolean column has invalid dtype")
    if column_type == "int" and data.dtype.kind not in {"i", "u"}:
        raise ColumnarRecordsError("integer column has invalid dtype")
    if column_type == "float" and (
        data.dtype.kind != "f" or np.any(~np.isfinite(data[valid]))
    ):
        raise ColumnarRecordsError("float column contains invalid values")
    if column_type == "str" and data.dtype.kind not in {"U", "S"}:
        raise ColumnarRecordsError("string column has invalid dtype")
    return data, valid
