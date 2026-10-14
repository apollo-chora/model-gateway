"""Shared parsing vocabulary for the canonical domain types (PRD sections 5, 9.1).

Every ``from_dict`` in this package reads its input through these helpers, so one
rule holds across the whole package: a key the canonical type does not define is
an error rather than something to ignore, and a value of the wrong JSON type is
an error rather than something to coerce. That is PRD section 9.1 applied at the
boundary — a feature the gateway cannot represent fails loudly here instead of
disappearing between the caller and the provider.

The helpers are shared by the canonical types and are not part of the gateway's
public API.
"""

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from model_gateway.domain.json_value import FrozenJsonObject, freeze_json

__all__ = [
    "as_boolean",
    "as_enumeration",
    "as_int",
    "as_json_object",
    "as_object",
    "as_optional_boolean",
    "as_optional_float",
    "as_optional_int",
    "as_optional_text",
    "as_required_text",
    "as_sequence",
    "as_text",
    "as_text_tuple",
    "describe",
    "reject_unknown_keys",
]


def describe(value: object) -> str:
    """Name the JSON type of ``value``, for an error message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, Mapping):
        return "a JSON object"
    if isinstance(value, Sequence):
        return "a JSON array"
    return f"a {type(value).__name__}"


def as_object(where: str, value: object) -> Mapping[str, object]:
    """A JSON object, or a ``ValueError`` naming what arrived instead."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object, got {describe(value)}")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(
                f"{where} has the non-string key {key!r}; a JSON object's keys are strings"
            )
    return value


def reject_unknown_keys(where: str, data: Mapping[str, object], allowed: frozenset[str]) -> None:
    """Refuse a key the canonical type does not define (PRD section 9.1)."""
    unsupported = sorted(set(data) - allowed)
    if unsupported:
        supported = ", ".join(repr(key) for key in sorted(allowed))
        raise ValueError(
            f"{where} has unsupported key {unsupported[0]!r}; supported keys are {supported}. "
            f"An unrecognised field is a feature the gateway cannot represent, so it fails "
            f"explicitly rather than being dropped (PRD section 9.1)"
        )


def as_text(where: str, value: object) -> str:
    """A string, which may be empty."""
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string, got {describe(value)}")
    return value


def as_required_text(where: str, value: object) -> str:
    """A string that carries something."""
    parsed = as_text(where, value)
    if not parsed:
        raise ValueError(f"{where} must not be empty")
    return parsed


def as_optional_text(where: str, value: object) -> str | None:
    """A non-empty string, or ``None`` when nothing was supplied."""
    if value is None:
        return None
    return as_required_text(where, value)


def as_optional_int(where: str, value: object, *, minimum: int | None = None) -> int | None:
    """An integer, or ``None``. A boolean is not an integer here."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {describe(value)}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be at least {minimum}, got {value}")
    return value


def as_int(where: str, value: object, *, minimum: int | None = None) -> int:
    """An integer that must be present."""
    parsed = as_optional_int(where, value, minimum=minimum)
    if parsed is None:
        raise ValueError(f"{where} must be an integer, got null")
    return parsed


def as_optional_float(
    where: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """A finite number, or ``None``. A boolean is not a number here."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} must be a number, got {describe(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be a finite number, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{where} must be at least {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{where} must be at most {maximum}, got {number}")
    return number


def as_boolean(where: str, value: object) -> bool:
    """A boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean, got {describe(value)}")
    return value


def as_optional_boolean(where: str, value: object, default: bool) -> bool:
    """A boolean, or ``default`` when nothing was supplied."""
    if value is None:
        return default
    return as_boolean(where, value)


def as_sequence(where: str, value: object) -> tuple[object, ...]:
    """A JSON array as a tuple; an absent or null array is an empty one."""
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{where} must be a JSON array, got {describe(value)}")
    return tuple(value)


def as_text_tuple(where: str, value: object) -> tuple[str, ...]:
    """A JSON array of non-empty strings as a tuple.

    A bare string is refused rather than treated as a one-element array, because
    the two are different values and guessing between them would change what the
    provider receives.
    """
    if isinstance(value, str):
        raise ValueError(
            f"{where} must be a JSON array of strings, got a string; a single string is not "
            f"accepted as a one-element array"
        )
    return tuple(
        as_required_text(f"{where}[{index}]", item)
        for index, item in enumerate(as_sequence(where, value))
    )


def as_json_object(where: str, value: object) -> FrozenJsonObject:
    """Arbitrary JSON as an immutable object; absent or null is an empty one."""
    if value is None:
        return FrozenJsonObject()
    frozen = freeze_json(where, value)
    if not isinstance(frozen, FrozenJsonObject):
        raise ValueError(f"{where} must be a JSON object, got {describe(value)}")
    return frozen


def as_enumeration[T: StrEnum](where: str, value: object, member: type[T]) -> T:
    """A member of ``member``, accepting its string value as well as the member itself."""
    if isinstance(value, member):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string, got {describe(value)}")
    try:
        return member(value)
    except ValueError as error:
        supported = ", ".join(repr(item.value) for item in member)
        raise ValueError(f"{where} must be one of {supported}; got {value!r}") from error
