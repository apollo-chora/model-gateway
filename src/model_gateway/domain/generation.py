"""Canonical generation parameters, output shape, finish reasons and usage (PRD 4.4, 5.1, 5.2, 13).

These are the things an invocation says about a model's behaviour rather than
about its content: how it was asked to sample, what shape its answer must take,
why it stopped, and what it cost in tokens. None of them names a vendor.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.json_value import JsonDocument, JsonValue, thaw_json
from model_gateway.domain.serialization import (
    as_enumeration,
    as_json_object,
    as_object,
    as_optional_boolean,
    as_optional_float,
    as_optional_int,
    as_optional_text,
    as_text_tuple,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "FinishReason",
    "GenerationConfig",
    "ResponseFormat",
    "ResponseFormatMode",
    "Usage",
    "UsageSource",
    "as_generation_config",
    "as_response_format",
    "as_usage",
]

_CONFIG_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_output_tokens",
        "stop_sequences",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
    }
)
_RESPONSE_FORMAT_KEYS = frozenset({"mode", "schema", "name", "strict"})
_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "estimated_fields"}
)
_USAGE_COUNTS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")


class FinishReason(StrEnum):
    """Why generation stopped (PRD sections 4.4 and 5.2).

    The vocabulary is the gateway's own, not any provider's. A reason the gateway
    does not recognise — including a provider that reports none at all — becomes
    ``UNKNOWN``: the response always states a finish reason, and ``UNKNOWN``
    states plainly that the gateway could not classify it. Dropping the field
    instead would leave a caller unable to tell "the model finished" from "the
    gateway lost the reason" (PRD section 9.1).

    ``UNKNOWN`` is a classification, not a record of absence. The provider's own
    reason is kept beside it as ``InvocationResponse.raw_finish_reason``, so
    ``UNKNOWN`` can never mean that the gateway deleted what upstream told it.
    """

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: str | None) -> "FinishReason":
        """Map a raw reason onto this vocabulary (PRD section 5.2).

        Matching is exact and case-sensitive, because inventing a mapping the
        gateway was never given is the provider translation this layer does not
        do. Anything else — a reason from a provider the gateway has not been
        taught, or no reason at all — becomes ``UNKNOWN`` rather than being
        dropped or guessed at.
        """
        if value is None:
            return cls.UNKNOWN
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


class ResponseFormatMode(StrEnum):
    """The output shape a caller requires (PRD section 9.5).

    * ``TEXT`` — ordinary text; the provider's own default.
    * ``JSON_OBJECT`` — JSON, with no schema the provider must satisfy.
    * ``JSON_SCHEMA`` — JSON conforming to ``ResponseFormat.schema``.

    ``JSON_OBJECT`` and ``JSON_SCHEMA`` are separate modes because they ask for
    different guarantees: the first promises valid JSON, the second promises a
    particular shape, and a provider may support one without the other. Folding
    them together would let the gateway claim a guarantee the provider never
    gave.
    """

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True)
class ResponseFormat:
    """A structured-output requirement (PRD sections 5.1 and 9.5).

    PRD section 9.5 names structured-output requirements among the semantics the
    gateway must never discard, so the canonical request has to be able to state
    one. Without this type a caller's schema could not be represented at all, and
    the capability error that promises to refuse an unsupported structured-output
    request could never be raised, because nothing would know the request
    contained one.

    ``schema`` is the JSON Schema the output must satisfy, kept exactly as the
    caller supplied it and never rewritten: a schema the gateway had filled in or
    normalised would describe a different requirement. It is required by
    ``JSON_SCHEMA`` and refused by every other mode, because a mode that never
    consults a schema carrying one is a contradiction rather than a request the
    gateway can send. A JSON Schema that is not an object — the boolean schemas
    the specification permits — is refused explicitly rather than coerced.

    ``name`` labels the schema for providers that require a name, and ``strict``
    asks the provider to enforce the schema rather than merely accept it. Both
    belong to ``JSON_SCHEMA`` alone; naming or enforcing a schema in a mode that
    has none is not a request any provider could honour (PRD section 9.5).
    """

    mode: ResponseFormatMode = ResponseFormatMode.TEXT
    schema: Mapping[str, JsonValue] | None = None
    name: str | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        mode = as_enumeration("response format 'mode'", self.mode, ResponseFormatMode)
        schema = (
            None
            if self.schema is None
            else as_json_object("response format 'schema'", self.schema)
        )
        name = as_optional_text("response format 'name'", self.name)
        strict = as_optional_boolean("response format 'strict'", self.strict, False)

        if mode is ResponseFormatMode.JSON_SCHEMA:
            if schema is None:
                raise ValueError(
                    "a 'json_schema' response format must carry the 'schema' the output has to "
                    "satisfy; without one there is no requirement to state, and the gateway must "
                    "not claim a guarantee it cannot express"
                )
        else:
            if schema is not None:
                raise ValueError(
                    f"a {mode.value!r} response format must not carry a 'schema'; only "
                    f"'json_schema' consults one, and accepting it here would suggest a "
                    f"guarantee this mode does not make"
                )
            if name is not None:
                raise ValueError(
                    f"a {mode.value!r} response format must not carry a 'name'; only "
                    f"'json_schema' names the schema it applies"
                )
            if strict:
                raise ValueError(
                    f"a {mode.value!r} response format must not be strict; only 'json_schema' "
                    f"can be enforced against a schema"
                )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "strict", strict)

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "mode": self.mode.value,
            "schema": None if self.schema is None else thaw_json(self.schema),
            "name": self.name,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "response format"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESPONSE_FORMAT_KEYS)
        schema = payload.get("schema")
        return cls(
            mode=as_enumeration(
                f"{where} 'mode'", payload.get("mode", ResponseFormatMode.TEXT), ResponseFormatMode
            ),
            schema=None if schema is None else as_json_object(f"{where} 'schema'", schema),
            name=as_optional_text(f"{where} 'name'", payload.get("name")),
            strict=as_optional_boolean(f"{where} 'strict'", payload.get("strict"), False),
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters for one invocation (PRD sections 4.4 and 5.1).

    Every field defaults to "not specified", and "not specified" means the
    gateway sends no value so that the provider's own default applies:

    * ``temperature`` — unset; the provider's own sampling temperature applies.
    * ``top_p`` — unset; the provider's own nucleus threshold applies.
    * ``max_output_tokens`` — unset; the provider's own output ceiling applies,
      which is why the gateway does not invent one from a route's advertised
      context window.
    * ``stop_sequences`` — empty; no stop sequence is sent.
    * ``seed`` — unset; sampling is not pinned and a repeat may differ.
    * ``presence_penalty`` and ``frequency_penalty`` — unset; the provider's own
      defaults apply.
    * ``response_format`` — unset; the provider's own output shape applies. It is
      ``None`` rather than a text-mode :class:`ResponseFormat`, because requiring
      text is a request in its own right and a caller that wants it asks for it.

    The gateway deliberately substitutes no vendor default, such as a temperature
    of 1.0, because doing so would encode one vendor's behaviour in the
    vendor-neutral core (PRD section 23). "Unset" and "zero" are different
    requests, and a caller that wants a temperature of 0 asks for it.

    Only provider-independent invariants are enforced here: a token ceiling of at
    least one, a non-negative temperature, a ``top_p`` between 0 and 1, finite
    numbers, and a structured-output requirement that is internally consistent.
    Provider-specific bounds belong to the route, not to this type.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: ResponseFormat | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "temperature", as_optional_float("temperature", self.temperature, minimum=0.0)
        )
        object.__setattr__(
            self,
            "top_p",
            as_optional_float("top_p", self.top_p, minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            as_optional_int("max_output_tokens", self.max_output_tokens, minimum=1),
        )
        object.__setattr__(
            self, "stop_sequences", as_text_tuple("stop_sequences", self.stop_sequences)
        )
        object.__setattr__(self, "seed", as_optional_int("seed", self.seed))
        object.__setattr__(
            self,
            "presence_penalty",
            as_optional_float("presence_penalty", self.presence_penalty),
        )
        object.__setattr__(
            self,
            "frequency_penalty",
            as_optional_float("frequency_penalty", self.frequency_penalty),
        )
        object.__setattr__(
            self,
            "response_format",
            as_response_format("response_format", self.response_format),
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "stop_sequences": list(self.stop_sequences),
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": (
                None if self.response_format is None else self.response_format.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "generation config"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _CONFIG_KEYS)
        return cls(
            temperature=as_optional_float(
                f"{where} 'temperature'", payload.get("temperature"), minimum=0.0
            ),
            top_p=as_optional_float(
                f"{where} 'top_p'", payload.get("top_p"), minimum=0.0, maximum=1.0
            ),
            max_output_tokens=as_optional_int(
                f"{where} 'max_output_tokens'", payload.get("max_output_tokens"), minimum=1
            ),
            stop_sequences=as_text_tuple(
                f"{where} 'stop_sequences'", payload.get("stop_sequences")
            ),
            seed=as_optional_int(f"{where} 'seed'", payload.get("seed")),
            presence_penalty=as_optional_float(
                f"{where} 'presence_penalty'", payload.get("presence_penalty")
            ),
            frequency_penalty=as_optional_float(
                f"{where} 'frequency_penalty'", payload.get("frequency_penalty")
            ),
            response_format=_optional_response_format(payload.get("response_format")),
        )


class UsageSource(StrEnum):
    """Whether one token count was reported by the provider or derived (PRD section 13).

    PRD section 13 requires the distinction because the two are not
    interchangeable: a provider-reported count is a fact about the upstream call,
    while an estimate is the gateway's own arithmetic and may be wrong.

    The source belongs to each count rather than to the record, because a record
    can mix them. A provider that reports input and output tokens but no total
    leaves the gateway to derive one, and only the total is then an estimate; a
    single record-wide flag would present the reported counts as estimates too,
    which is the opposite error and just as unauditable.
    """

    PROVIDER = "provider"
    ESTIMATE = "estimate"


@dataclass(frozen=True)
class Usage:
    """Token counts for one invocation (PRD sections 4.4, 5.2 and 13).

    Every count is optional, and ``None`` means "not known", which is not the
    same as zero: a provider that does not report cached tokens has told the
    gateway nothing, and recording 0 would assert that no tokens were cached.

    ``estimated_fields`` names the counts the gateway derived, so provenance is
    per count rather than per record. A provider that reported input and output
    tokens but no total leaves ``total_tokens`` in this set and the other two out
    of it, and a consumer can tell that the first two are facts; an empty set
    means every recorded count was reported. A field named here must carry a
    value, because an estimate is arithmetic the gateway performed, while an
    unknown count is an absence rather than an estimate. ``source_of`` answers
    the question for one count, and ``to_dict`` emits the names sorted so the
    serialised form is stable.

    ``total_tokens`` is recorded as supplied and never recomputed from the other
    two: a provider's own total is the reported fact, and a total the gateway
    computed is an estimate and is named in ``estimated_fields`` as one.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    estimated_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in _USAGE_COUNTS:
            object.__setattr__(self, name, as_optional_int(name, getattr(self, name), minimum=0))
        estimated = _as_estimated_fields("estimated_fields", self.estimated_fields)
        for name in sorted(estimated):
            if getattr(self, name) is None:
                raise ValueError(
                    f"estimated_fields names {name!r}, but that count is unknown; an estimate is "
                    f"a value the gateway derived, so a count it did not derive cannot be an "
                    f"estimate of one"
                )
        object.__setattr__(self, "estimated_fields", estimated)

    @property
    def is_estimated(self) -> bool:
        """Whether any count in this record was derived rather than reported."""
        return bool(self.estimated_fields)

    def source_of(self, field: str) -> UsageSource:
        """Whether one count was reported by the provider or derived by the gateway."""
        if field not in _USAGE_COUNTS:
            supported = ", ".join(repr(name) for name in _USAGE_COUNTS)
            raise ValueError(
                f"{field!r} is not a token count this record defines; supported counts are "
                f"{supported}"
            )
        return UsageSource.ESTIMATE if field in self.estimated_fields else UsageSource.PROVIDER

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated_fields": sorted(self.estimated_fields),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "usage"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _USAGE_KEYS)
        return cls(
            input_tokens=as_optional_int(
                f"{where} 'input_tokens'", payload.get("input_tokens"), minimum=0
            ),
            output_tokens=as_optional_int(
                f"{where} 'output_tokens'", payload.get("output_tokens"), minimum=0
            ),
            total_tokens=as_optional_int(
                f"{where} 'total_tokens'", payload.get("total_tokens"), minimum=0
            ),
            cached_tokens=as_optional_int(
                f"{where} 'cached_tokens'", payload.get("cached_tokens"), minimum=0
            ),
            estimated_fields=_as_estimated_fields(
                f"{where} 'estimated_fields'", payload.get("estimated_fields")
            ),
        )


def as_generation_config(where: str, value: object) -> GenerationConfig:
    """A generation config, refusing anything else."""
    if not isinstance(value, GenerationConfig):
        raise ValueError(f"{where} must be a GenerationConfig, got {describe(value)}")
    return value


def as_response_format(where: str, value: object) -> ResponseFormat | None:
    """A response format, or ``None`` when the caller required no particular shape."""
    if value is None:
        return None
    if not isinstance(value, ResponseFormat):
        raise ValueError(f"{where} must be a ResponseFormat, got {describe(value)}")
    return value


def as_usage(where: str, value: object) -> Usage:
    """A usage record, refusing anything else."""
    if not isinstance(value, Usage):
        raise ValueError(f"{where} must be a Usage, got {describe(value)}")
    return value


def _as_estimated_fields(where: str, value: object) -> frozenset[str]:
    """The names of the derived counts, refusing a name that is not a count."""
    names = as_text_tuple(
        where, tuple(value) if isinstance(value, set | frozenset) else value
    )
    unknown = sorted(set(names) - set(_USAGE_COUNTS))
    if unknown:
        supported = ", ".join(repr(name) for name in _USAGE_COUNTS)
        raise ValueError(
            f"{where} names {unknown[0]!r}, which is not a token count this record defines; "
            f"supported counts are {supported}"
        )
    return frozenset(names)


def _optional_response_format(value: object) -> ResponseFormat | None:
    """Parse an optional response format."""
    if value is None:
        return None
    return ResponseFormat.from_dict(value)
