"""Immutable, hashable JSON values for the canonical domain types (PRD sections 5, 9.3).

Tool schemas, tool arguments and request metadata are arbitrary JSON. The
canonical types are frozen dataclasses, but a frozen dataclass holding a plain
``dict`` is only shallowly frozen: the attribute cannot be reassigned while the
object it points at can still be mutated in place, so a value that passed
validation can change afterwards. The route model already made that mistake once,
when a plain dict inside a frozen dataclass allowed in-place mutation.

``FrozenJsonObject`` is therefore the only object type this package stores. It
refuses every mutator, and its values are scalars, tuples and further
``FrozenJsonObject`` instances, so a whole tree is immutable and hashable — which
is what lets the dataclasses holding one be hashable too.

Anything JSON cannot carry — a set, a ``bytes`` payload, a ``datetime``, a
non-string key, a NaN — is rejected rather than coerced. The canonical
representation has no way to hold it, and a silent coercion would be exactly the
loss PRD section 9.3 forbids.
"""

import math
from collections.abc import Iterator, Mapping
from typing import NoReturn, Self

__all__ = ["FrozenJsonObject", "JsonDocument", "JsonValue", "freeze_json", "thaw_json"]

type JsonValue = str | int | float | bool | None | tuple[JsonValue, ...] | Mapping[str, JsonValue]

type JsonDocument = str | int | float | bool | None | list[JsonDocument] | dict[str, JsonDocument]


class FrozenJsonObject(dict[str, "JsonValue"]):
    """A JSON object that cannot be mutated once it is built.

    Every ordinary mutator raises ``TypeError``, which is the same strength of
    freeze the enclosing frozen dataclass provides. Subclassing ``dict`` keeps
    equality and ``json.dumps`` working; ``__reduce__`` keeps the frozen type
    across copying and pickling, and ``__hash__`` is order-independent so that
    two equal objects built in different key orders hash alike.
    """

    __slots__ = ()

    def __setitem__(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("a canonical JSON object is immutable")

    __delitem__ = __setitem__
    __ior__ = __setitem__
    clear = __setitem__
    pop = __setitem__
    popitem = __setitem__
    setdefault = __setitem__
    update = __setitem__

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[Self], tuple[dict[str, JsonValue]]]:
        return (type(self), (dict(self),))


def freeze_json(where: str, value: object) -> JsonValue:
    """Return ``value`` as an immutable, hashable JSON value.

    ``where`` names the location the value came from, so a rejection says which
    field carried something the canonical representation cannot hold.
    """
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must be a finite number; JSON cannot carry {value!r}")
        return value
    if isinstance(value, FrozenJsonObject):
        return value
    if isinstance(value, Mapping):
        return FrozenJsonObject(
            {key: freeze_json(f"{where}.{key}", item) for key, item in _items(where, value)}
        )
    if isinstance(value, tuple | list):
        return tuple(freeze_json(f"{where}[{index}]", item) for index, item in enumerate(value))
    raise ValueError(
        f"{where} is a {type(value).__name__}, which the canonical JSON representation cannot "
        f"carry; encode it as JSON before it reaches the domain layer (PRD section 9.3)"
    )


def thaw_json(value: JsonValue) -> JsonDocument:
    """Return a frozen JSON value as plain, JSON-serialisable Python data."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _items(where: str, value: Mapping[object, object]) -> Iterator[tuple[str, object]]:
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{where} has the non-string key {key!r}; a JSON object's keys are strings"
            )
        yield key, item
