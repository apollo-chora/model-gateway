"""Canonical embeddings request and response (PRD sections 4.7, 6.3, 9.6 and 13).

An embeddings invocation is the third operation the gateway serves, and its
canonical types follow the same rules as the text operation's: the request names
a public model ID and carries the caller's parameters, the response reports the
public ID beside the upstream model that served it, and both round-trip through
``to_dict`` and ``from_dict`` with every field always emitted (PRD sections 4.7
and 5.2).

``EmbeddingRequest.input`` is the texts to embed, in order, and
``EmbeddingResponse.data`` is the vectors in the same order: PRD section 4.7
requires the gateway to preserve input ordering, so the position of a vector in
``data`` is the position of its text in ``input`` and ``Embedding.index`` states
it explicitly rather than leaving a consumer to infer it. A response whose
vectors cannot be matched to their texts has already lost the ordering by the
time a caller sees it.

``EmbeddingResponse.usage`` is the token accounting of the call (PRD section 13).
Embedding requests carry no generation parameters, so the only limits the
gateway enforces here are the ones the canonical representation itself cannot
hold: a non-empty input, a positive ``dimensions`` when one is asked for, and an
``encoding_format`` from the documented set. A value outside that set is refused
rather than dropped, because a parameter that quietly disappears between the
caller and the provider is worse than a rejected request (PRD section 9.1).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from model_gateway.domain.generation import Usage, as_usage
from model_gateway.domain.json_value import FrozenJsonObject, JsonDocument, thaw_json
from model_gateway.domain.serialization import (
    as_int,
    as_json_object,
    as_object,
    as_optional_float,
    as_optional_int,
    as_optional_text,
    as_required_text,
    as_sequence,
    as_text_tuple,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "ENCODING_FORMATS",
    "Embedding",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "as_embeddings",
]

_EMBEDDING_KEYS = frozenset({"index", "values"})
_REQUEST_KEYS = frozenset(
    {"model", "input", "dimensions", "encoding_format", "user", "metadata"}
)
_RESPONSE_KEYS = frozenset({"model", "upstream_model", "data", "usage", "invocation_id"})

ENCODING_FORMATS: frozenset[str] = frozenset({"float", "base64"})
"""The wire encodings an embeddings request may ask for (PRD section 4.7).

The gateway's embeddings endpoint is OpenAI-compatible, and that API documents
exactly two encodings: ``float``, the numbers themselves, and ``base64``, the
same numbers as a base64 string. A request naming anything else is refused
rather than silently answered in the gateway's own default, because a caller
told its request was honoured while the gateway ignored the encoding it asked
for cannot tell the two apart (PRD section 9.1). The canonical
:class:`EmbeddingResponse` carries the float form, which is the representation
the gateway's own wire response uses (PRD section 4.7).
"""


@dataclass(frozen=True)
class Embedding:
    """One embedding vector, positioned in the caller's input (PRD section 4.7).

    ``index`` is the position of the text this vector embeds, so a consumer can
    match a vector to its text without relying on array order alone. ``values``
    is the vector itself, always non-empty and always finite: a vector with no
    components is not a vector, and a component that is not a finite number is
    a value the canonical representation cannot carry (PRD section 9.1).
    """

    index: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", as_int("embedding 'index'", self.index, minimum=0)
        )
        object.__setattr__(self, "values", _as_values("embedding 'values'", self.values))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {"index": self.index, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "embedding"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _EMBEDDING_KEYS)

        return cls(
            index=as_int(f"{where} 'index'", payload.get("index"), minimum=0),
            values=_as_values(f"{where} 'values'", payload.get("values")),
        )


@dataclass(frozen=True)
class EmbeddingRequest:
    """One canonical embeddings invocation (PRD sections 4.7 and 6.3).

    ``model`` is the public model ID, never an upstream model name, for the same
    reason as an invocation request: the route resolver decides which upstream
    serves it and a caller-supplied name must not reach an unconfigured endpoint
    (PRD section 10).

    ``input`` is the texts to embed, in order, and must carry at least one of
    them: an embedding request with nothing to embed has no canonical
    representation. No entry may be blank, because a blank string is not text
    the gateway can ask a model to embed and an empty entry would silently
    shift every later index.

    ``dimensions`` asks the route for vectors of that length and is honoured
    only when the route enables it (PRD section 6.3); it is positive when set,
    because a vector of zero or negative length is not a request any provider
    could honour. ``encoding_format`` is the wire encoding the caller asks for,
    one of :data:`ENCODING_FORMATS`. ``user`` is the caller's stable end-user
    identifier, and ``metadata`` is non-sensitive request metadata held as an
    immutable JSON object, never forwarded to the model or the provider
    (PRD section 5.1).
    """

    model: str
    input: tuple[str, ...]
    dimensions: int | None = None
    encoding_format: str | None = None
    user: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", as_required_text("model", self.model))
        object.__setattr__(self, "input", _as_input("input", self.input))
        object.__setattr__(
            self,
            "dimensions",
            as_optional_int("dimensions", self.dimensions, minimum=1),
        )
        object.__setattr__(
            self,
            "encoding_format",
            _as_encoding_format("encoding_format", self.encoding_format),
        )
        object.__setattr__(self, "user", as_optional_text("user", self.user))
        object.__setattr__(self, "metadata", _as_metadata("metadata", self.metadata))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "model": self.model,
            "input": list(self.input),
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            "user": self.user,
            "metadata": None if self.metadata is None else thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "embedding request"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _REQUEST_KEYS)

        return cls(
            model=as_required_text(f"{where} 'model'", payload.get("model")),
            input=_as_input(f"{where} 'input'", payload.get("input")),
            dimensions=as_optional_int(
                f"{where} 'dimensions'", payload.get("dimensions"), minimum=1
            ),
            encoding_format=_as_encoding_format(
                f"{where} 'encoding_format'", payload.get("encoding_format")
            ),
            user=as_optional_text(f"{where} 'user'", payload.get("user")),
            metadata=_as_metadata(f"{where} 'metadata'", payload.get("metadata")),
        )


@dataclass(frozen=True)
class EmbeddingResponse:
    """The canonical result of one embeddings invocation (PRD sections 4.7 and 13).

    ``model`` is the public model ID the caller asked for and ``upstream_model``
    the route's own model ID, reported separately because a fallback may have
    served it (PRD section 5.2). ``data`` holds one vector per input text, in
    input order, and ``Embedding.index`` states that position explicitly.

    ``usage`` is the token accounting of the call, and ``invocation_id`` the
    identifier of the invocation this response belongs to, so a caller can
    reconcile the response with the request it sent (PRD section 13).
    """

    model: str
    upstream_model: str
    data: tuple[Embedding, ...]
    usage: Usage
    invocation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", as_required_text("model", self.model))
        object.__setattr__(
            self, "upstream_model", as_required_text("upstream_model", self.upstream_model)
        )
        object.__setattr__(self, "data", as_embeddings("data", self.data))
        object.__setattr__(self, "usage", as_usage("usage", self.usage))
        object.__setattr__(
            self, "invocation_id", as_required_text("invocation_id", self.invocation_id)
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "model": self.model,
            "upstream_model": self.upstream_model,
            "data": [embedding.to_dict() for embedding in self.data],
            "usage": self.usage.to_dict(),
            "invocation_id": self.invocation_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "embedding response"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESPONSE_KEYS)

        return cls(
            model=as_required_text(f"{where} 'model'", payload.get("model")),
            upstream_model=as_required_text(
                f"{where} 'upstream_model'", payload.get("upstream_model")
            ),
            data=tuple(
                Embedding.from_dict(item)
                for item in as_sequence(f"{where} 'data'", payload.get("data"))
            ),
            usage=Usage.from_dict(payload.get("usage")),
            invocation_id=as_required_text(
                f"{where} 'invocation_id'", payload.get("invocation_id")
            ),
        )


def as_embeddings(where: str, value: object) -> tuple[Embedding, ...]:
    """Coerce a sequence of embeddings, refusing anything else."""
    items = as_sequence(where, value)
    for index, item in enumerate(items):
        if not isinstance(item, Embedding):
            raise ValueError(
                f"{where}[{index}] must be an Embedding, got {describe(item)}"
            )
    return items


def _as_values(where: str, value: object) -> tuple[float, ...]:
    """A non-empty tuple of finite numbers."""
    items = as_sequence(where, value)
    if not items:
        raise ValueError(
            f"{where} must carry at least one number; an embedding with no values is not a "
            f"vector the gateway can return (PRD section 4.7)"
        )
    return tuple(_as_value(f"{where}[{index}]", item) for index, item in enumerate(items))


def _as_value(where: str, value: object) -> float:
    """One embedding value: a finite number."""
    number = as_optional_float(where, value)
    if number is None:
        raise ValueError(f"{where} must be a number, got null")
    return number


def _as_input(where: str, value: object) -> tuple[str, ...]:
    """A non-empty tuple of non-blank strings, in the order the caller gave them."""
    items = as_text_tuple(where, value)
    if not items:
        raise ValueError(
            f"{where} must carry at least one string; an embedding request with nothing to "
            f"embed has no canonical representation (PRD section 4.7)"
        )
    for index, item in enumerate(items):
        if not item.strip():
            raise ValueError(
                f"{where}[{index}] is blank; a blank string is not text the gateway can ask "
                f"a model to embed, and dropping it would shift every later index"
            )
    return items


def _as_encoding_format(where: str, value: object) -> str | None:
    """One of the documented wire encodings, or ``None`` when the caller stated none."""
    encoding = as_optional_text(where, value)
    if encoding is None:
        return None
    if encoding not in ENCODING_FORMATS:
        supported = ", ".join(repr(item) for item in sorted(ENCODING_FORMATS))
        raise ValueError(
            f"{where} must be one of {supported}; got {encoding!r}. An encoding outside the "
            f"documented set is refused rather than answered in the gateway's own default, "
            f"because a caller told its request was honoured while the gateway ignored the "
            f"encoding it asked for cannot tell the two apart (PRD section 9.1)"
        )
    return encoding


def _as_metadata(where: str, value: object) -> FrozenJsonObject | None:
    """Arbitrary JSON as an immutable object; ``None`` when the caller supplied none.

    ``None`` is kept as ``None`` rather than folded into an empty object, so a
    request that carried no metadata round-trips to one that still carries none.
    """
    if value is None:
        return None
    return as_json_object(where, value)
