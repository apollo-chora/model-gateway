"""Canonical generation parameters, finish reasons and token usage (PRD 4.4, 5.1, 5.2, 13).

These are the three things an invocation says about a model's behaviour rather
than about its content: how it was asked to sample, why it stopped, and what it
cost in tokens. None of them names a vendor.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.serialization import (
    as_enumeration,
    as_object,
    as_optional_float,
    as_optional_int,
    as_text_tuple,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "FinishReason",
    "GenerationConfig",
    "Usage",
    "UsageSource",
    "as_generation_config",
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
    }
)
_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "source"}
)
_USAGE_COUNTS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")


class FinishReason(StrEnum):
    """Why generation stopped (PRD sections 4.4 and 5.2).

    The vocabulary is the gateway's own, not any provider's. A reason the gateway
    does not recognise — including a provider that reports none at all — becomes
    ``UNKNOWN``: the response always states a finish reason, and ``UNKNOWN``
    states plainly that the gateway could not classify it. Dropping the field
    instead would leave a caller unable to tell "the model finished" from "the
    gateway lost the reason" (PRD section 9.3).
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

    The gateway deliberately substitutes no vendor default, such as a temperature
    of 1.0, because doing so would encode one vendor's behaviour in the
    vendor-neutral core (PRD section 23). "Unset" and "zero" are different
    requests, and a caller that wants a temperature of 0 asks for it.

    Only provider-independent invariants are enforced here: a token ceiling of at
    least one, a non-negative temperature, a ``top_p`` between 0 and 1, and
    finite numbers. Provider-specific bounds belong to the route, not to this
    type.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

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

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "stop_sequences": list(self.stop_sequences),
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
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
        )


class UsageSource(StrEnum):
    """Whether a usage record's counts were reported by the provider or derived.

    PRD section 13 requires the distinction because the two are not
    interchangeable: a provider-reported count is a fact about the upstream call,
    while an estimate is the gateway's own arithmetic and may be wrong.
    Accounting that mixes them without saying so cannot be audited.
    """

    PROVIDER = "provider"
    ESTIMATE = "estimate"


@dataclass(frozen=True)
class Usage:
    """Token counts for one invocation (PRD sections 4.4, 5.2 and 13).

    Every count is optional, and ``None`` means "not known", which is not the
    same as zero: a provider that does not report cached tokens has told the
    gateway nothing, and recording 0 would assert that no tokens were cached.

    ``source`` describes the record as a whole and is ``ESTIMATE`` whenever the
    gateway derived any count rather than receiving it, because a record mixing
    reported and estimated numbers must not be presented as reported.
    ``total_tokens`` is recorded as supplied and never recomputed from the other
    two: a provider's own total is the reported fact, and a derived one would
    have to be marked as an estimate.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    source: UsageSource = UsageSource.PROVIDER

    def __post_init__(self) -> None:
        for name in _USAGE_COUNTS:
            object.__setattr__(self, name, as_optional_int(name, getattr(self, name), minimum=0))
        object.__setattr__(self, "source", as_enumeration("source", self.source, UsageSource))

    @property
    def is_estimated(self) -> bool:
        """Whether any count in this record was derived rather than reported."""
        return self.source is UsageSource.ESTIMATE

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "source": self.source.value,
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
            source=as_enumeration(
                f"{where} 'source'", payload.get("source", UsageSource.PROVIDER), UsageSource
            ),
        )


def as_generation_config(where: str, value: object) -> GenerationConfig:
    """A generation config, refusing anything else."""
    if not isinstance(value, GenerationConfig):
        raise ValueError(f"{where} must be a GenerationConfig, got {describe(value)}")
    return value


def as_usage(where: str, value: object) -> Usage:
    """A usage record, refusing anything else."""
    if not isinstance(value, Usage):
        raise ValueError(f"{where} must be a Usage, got {describe(value)}")
    return value
