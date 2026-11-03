"""Strict, deterministic JSON serialization for core protocol values."""

from __future__ import annotations

import dataclasses
import json
import math
import types
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from enum import Enum
from types import NoneType
from typing import (
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from cosmosalpha.core.errors import (
    CoreValidationError,
    MissingFieldError,
    SerializationError,
    TypeMismatchError,
    UnknownFieldError,
)

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _path_field(path: str, field_name: str) -> str:
    return f"{path}.{field_name}"


def _type_name(target_type: object) -> str:
    return getattr(target_type, "__name__", repr(target_type))


def _type_error(path: str, expected: object, value: object) -> TypeMismatchError:
    return TypeMismatchError(
        f"{path}: expected {_type_name(expected)}, got {type(value).__name__}"
    )


def _encode_datetime(value: datetime) -> str:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None or offset != timedelta(0):
        raise SerializationError("datetime values must use UTC timezone information")
    return value.isoformat().replace("+00:00", "Z")


def to_json_value(value: object) -> JsonValue:
    """Convert a supported Python value into a strict JSON-compatible value.

    Args:
        value: Value composed of supported scalar, collection, enum, date, datetime,
            or dataclass types.

    Returns:
        A JSON-compatible representation.

    Raises:
        SerializationError: If the value is unsupported or violates JSON rules.
    """
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SerializationError("non-finite floating-point values are forbidden")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, datetime):
        return _encode_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SerializationError("JSON object keys must be strings")
            result[key] = to_json_value(item)
        return result
    raise SerializationError(f"unsupported JSON value type: {type(value).__name__}")


def to_json(value: object) -> str:
    """Serialize a supported value to canonical UTF-8 JSON text.

    Args:
        value: Supported protocol or JSON value.

    Returns:
        Canonical JSON with sorted keys and no insignificant whitespace.
    """
    return json.dumps(
        to_json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical UTF-8 bytes used for checksums and stable IDs.

    Args:
        value: Supported protocol or JSON value.

    Returns:
        Deterministic UTF-8 encoded JSON.
    """
    return to_json(value).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SerializationError(f"non-finite JSON number is forbidden: {value}")


def _parse_json(data: str | bytes | bytearray) -> JsonValue:
    if not isinstance(data, (str, bytes, bytearray)):
        raise TypeMismatchError("JSON input must be str, bytes, or bytearray")
    try:
        parsed = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except SerializationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SerializationError(f"invalid JSON: {error}") from error
    return parsed


def _substitute_typevars(
    target_type: object,
    typevars: Mapping[TypeVar, object],
) -> object:
    if isinstance(target_type, TypeVar):
        return typevars.get(target_type, target_type)
    origin = get_origin(target_type)
    if origin is None:
        return target_type
    args = tuple(_substitute_typevars(arg, typevars) for arg in get_args(target_type))
    if origin in (Union, types.UnionType):
        result = args[0]
        for arg in args[1:]:
            result = result | arg
        return result
    try:
        return origin[args]
    except TypeError:
        return target_type


def _decode_union(
    value: JsonValue,
    target_type: object,
    path: str,
    typevars: Mapping[TypeVar, object],
) -> object:
    args = get_args(target_type)
    if value is None:
        if NoneType in args:
            return None
        raise _type_error(path, target_type, value)

    candidates = tuple(arg for arg in args if arg is not NoneType)
    errors: list[SerializationError] = []
    for candidate in candidates:
        try:
            return _decode(value, candidate, path, typevars)
        except SerializationError as error:
            errors.append(error)
    if errors:
        raise TypeMismatchError(
            f"{path}: value does not match any member of {_type_name(target_type)}"
        ) from errors[0]
    raise _type_error(path, target_type, value)


def _decode_tuple(
    value: JsonValue,
    target_type: object,
    path: str,
    typevars: Mapping[TypeVar, object],
) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise _type_error(path, target_type, value)
    args = get_args(target_type)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _decode(item, args[0], f"{path}[{index}]", typevars)
            for index, item in enumerate(value)
        )
    if len(value) != len(args):
        raise TypeMismatchError(
            f"{path}: expected tuple with {len(args)} items, got {len(value)}"
        )
    return tuple(
        _decode(item, item_type, f"{path}[{index}]", typevars)
        for index, (item, item_type) in enumerate(zip(value, args, strict=True))
    )


def _decode_dataclass(
    value: JsonValue,
    target_type: object,
    path: str,
    typevars: Mapping[TypeVar, object],
) -> object:
    origin = get_origin(target_type) or target_type
    if not isinstance(origin, type) or not dataclasses.is_dataclass(origin):
        raise SerializationError(
            f"{path}: unsupported target type {_type_name(target_type)}"
        )
    if not isinstance(value, dict):
        raise _type_error(path, target_type, value)

    local_typevars = dict(typevars)
    parameters = getattr(origin, "__parameters__", ())
    arguments = get_args(target_type)
    if parameters:
        if len(parameters) != len(arguments):
            raise SerializationError(
                f"{path}: generic target {_type_name(target_type)} is not fully specialized"
            )
        local_typevars.update(zip(parameters, arguments, strict=True))

    fields = {field.name: field for field in dataclasses.fields(origin)}
    unknown = value.keys() - fields.keys()
    if unknown:
        names = ", ".join(sorted(unknown))
        raise UnknownFieldError(f"{path}: unknown field(s): {names}")
    missing = fields.keys() - value.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise MissingFieldError(f"{path}: missing field(s): {names}")

    hints = get_type_hints(origin)
    kwargs = {
        name: _decode(
            value[name],
            _substitute_typevars(hints[name], local_typevars),
            _path_field(path, name),
            local_typevars,
        )
        for name in fields
    }
    try:
        return origin(**kwargs)
    except CoreValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise SerializationError(
            f"{path}: invalid {_type_name(origin)}: {error}"
        ) from error


def _decode(
    value: JsonValue,
    target_type: object,
    path: str,
    typevars: Mapping[TypeVar, object],
) -> object:
    target_type = _substitute_typevars(target_type, typevars)
    if isinstance(target_type, TypeAliasType):
        target_type = target_type.__value__
    if isinstance(target_type, TypeVar):
        raise SerializationError(
            f"{path}: unresolved generic type variable {target_type.__name__}"
        )

    origin = get_origin(target_type)
    if origin in (Union, types.UnionType):
        return _decode_union(value, target_type, path, typevars)
    if origin is tuple:
        return _decode_tuple(value, target_type, path, typevars)
    if origin is list:
        if not isinstance(value, list):
            raise _type_error(path, target_type, value)
        (item_type,) = get_args(target_type)
        return [
            _decode(item, item_type, f"{path}[{index}]", typevars)
            for index, item in enumerate(value)
        ]
    if origin in (dict, Mapping):
        if not isinstance(value, dict):
            raise _type_error(path, target_type, value)
        key_type, item_type = get_args(target_type)
        if key_type is not str:
            raise SerializationError(f"{path}: JSON mappings require string keys")
        return {
            key: _decode(item, item_type, _path_field(path, key), typevars)
            for key, item in value.items()
        }

    dataclass_origin = origin or target_type
    if isinstance(dataclass_origin, type) and dataclasses.is_dataclass(
        dataclass_origin
    ):
        return _decode_dataclass(value, target_type, path, typevars)
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        expected_type = type(next(iter(target_type)).value)
        if type(value) is not expected_type:
            raise _type_error(path, target_type, value)
        try:
            return target_type(value)
        except ValueError as error:
            raise SerializationError(
                f"{path}: unknown {_type_name(target_type)} value {value!r}"
            ) from error
    if target_type is datetime:
        if type(value) is not str:
            raise _type_error(path, target_type, value)
        if not value.endswith("Z"):
            raise SerializationError(f"{path}: datetime must use canonical UTC 'Z'")
        try:
            decoded_datetime = datetime.fromisoformat(f"{value[:-1]}+00:00")
        except ValueError as error:
            raise SerializationError(f"{path}: invalid datetime {value!r}") from error
        if _encode_datetime(decoded_datetime) != value:
            raise SerializationError(f"{path}: datetime is not in canonical form")
        return decoded_datetime
    if target_type is date:
        if type(value) is not str:
            raise _type_error(path, target_type, value)
        try:
            decoded_date = date.fromisoformat(value)
        except ValueError as error:
            raise SerializationError(f"{path}: invalid date {value!r}") from error
        if decoded_date.isoformat() != value:
            raise SerializationError(f"{path}: date is not in canonical form")
        return decoded_date
    if target_type is NoneType:
        if value is not None:
            raise _type_error(path, target_type, value)
        return None
    if target_type in (bool, int, float, str):
        if type(value) is not target_type:
            raise _type_error(path, target_type, value)
        if target_type is float and not math.isfinite(value):
            raise SerializationError(f"{path}: non-finite float is forbidden")
        return value
    raise SerializationError(
        f"{path}: unsupported target type {_type_name(target_type)}"
    )


def from_json_value[DecodedT](
    value: JsonValue, target_type: type[DecodedT]
) -> DecodedT:
    """Decode a JSON-compatible value using strict runtime annotations.

    Args:
        value: Parsed JSON-compatible value.
        target_type: Concrete type or fully specialized generic alias.

    Returns:
        A value of the requested target type.

    Raises:
        SerializationError: If fields, types, or values violate the target schema.
    """
    return cast(DecodedT, _decode(value, target_type, "$", {}))


def from_json[DecodedT](
    data: str | bytes | bytearray, target_type: type[DecodedT]
) -> DecodedT:
    """Deserialize strict JSON text into a typed protocol value.

    Args:
        data: UTF-8 JSON text or bytes.
        target_type: Concrete type or fully specialized generic alias.

    Returns:
        A value of the requested target type.

    Raises:
        SerializationError: If JSON syntax or the typed payload is invalid.
    """
    return from_json_value(_parse_json(data), target_type)
