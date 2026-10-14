"""Canonical streaming events: the gateway's own representation of a streamed answer (PRD 5.3, 16).

The canonical request and response describe an invocation that has finished. A
streamed invocation also needs a type for the pieces that arrive before it has:
a text fragment, a fragment of a tool call, a usage report, the reason
generation stopped, and the failure that ends a stream which has already
produced output. This module is that type, and nothing else — it knows no
provider wire format and no vendor vocabulary, so translating a provider's
streaming chunks into these events is the LiteLLM binding's work (PRD section 9).

The same three properties hold here as in the rest of the canonical package.

**Immutable.** Frozen dataclasses with no mutable field: a validated event
cannot change into a different one afterwards.

**Serialisable.** Every type round-trips through ``to_dict`` and ``from_dict``,
and ``to_dict`` always emits every field, so the shape is pinned rather than
implied.

**Explicit about what it cannot represent (PRD section 9.1).** An event whose
kind and payload do not agree — a ``content`` event without text, a ``finish``
event without a reason — is refused rather than silently truncated, for the
same reason the rest of the package refuses an unrepresentable state.

**Sequence numbers are per invocation.** ``sequence`` is 0-based and strictly
increasing within one invocation: event 0 is the first piece of the answer, and
each later event carries the next number, so a consumer can reorder a stream,
detect a gap, and tell the first event from a retry. The numbers are the
gateway's own — the gateway numbers the events it emits and never takes a
sequence number from upstream, because a provider's chunk ordering is not a
numbering the gateway can trust. One invocation therefore owns one numbering
from 0, and a second invocation numbers from 0 again.

**A stream ends in one of two terminal events, or in neither.** A stream that
has delivered its first event ends in exactly one of two ways, and each has an
event of its own.

* ``StreamEventKind.FINISH`` ends a stream that **completed**. It is the only
  event carrying a finish reason, and it reports why generation stopped (PRD
  section 5.3).
* ``StreamEventKind.ERROR`` ends a stream that **failed** after it had already
  produced output. It carries the gateway's own stable reason and its sanitised
  message, and no finish reason, because the provider did not finish (PRD
  sections 5.4 and 16).

Nothing follows either one: a terminal event is the last event of the stream it
ends, so a stream contains at most one. A stream that reaches neither simply
stops, and a consumer reads that as **truncated** — the client went away, the
transport was lost, or the upstream ended without ever reporting why generation
stopped — which is deliberately not the same as a completed answer.

**A failure is its own kind, not a finish reason.** The obvious alternative —
emit ``FINISH`` with ``FinishReason.UNKNOWN`` and put the failure's reason in
``raw_finish_reason`` — is refused here, and the reason is the point of the
split. ``raw_finish_reason`` means *the provider's own finish reason, verbatim*
(PRD section 5.2), so parking a gateway reason such as ``provider_unavailable``
in it would overload a field that belongs to a different vocabulary with a value
from a namespace the provider never used. It would also present a gateway
failure as a finish reason to every consumer of the OpenAI-compatible shape the
HTTP boundary renders, where the value would be a reason the provider never
reported and the specification does not define. ``error_reason`` and
``error_message`` are that failure's own fields, and construction refuses to let
either kind carry the other's payload.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.generation import FinishReason, Usage
from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.serialization import (
    as_enumeration,
    as_int,
    as_object,
    as_optional_text,
    as_required_text,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "StreamEvent",
    "StreamEventKind",
    "ToolCallDelta",
]


class StreamEventKind(StrEnum):
    """What one streaming event reports (PRD sections 5.3 and 16).

    The vocabulary is the gateway's own. A provider's chunk shape does not
    decide it: a chunk carrying nothing reportable produces no event at all, and
    a chunk carrying two different things produces two events — the kind names
    the one payload the event carries, and is never widened to hold more.

    * ``CONTENT`` — a fragment of generated text.
    * ``TOOL_CALL`` — a fragment of a tool call: its identifier, its name, or a
      piece of its arguments (PRD section 16).
    * ``USAGE`` — a token report, on the chunk the provider chose to send it.
    * ``FINISH`` — the terminal event of a stream that completed. The reason
      generation stopped (PRD section 5.3).
    * ``ERROR`` — the terminal event of a stream that failed after it had
      produced output. The gateway's own stable reason and its sanitised
      message (PRD sections 5.4 and 16).

    ``FINISH`` and ``ERROR`` are terminal in the ordering sense as well: each is
    the last event of the stream it ends, because nothing the provider says after
    it belongs to an answer the caller was already told was complete, or was
    already told had failed. A stream that reaches neither ends without a
    terminal event, which is what a consumer reads as truncated.
    """

    CONTENT = "content"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    FINISH = "finish"
    ERROR = "error"


@dataclass(frozen=True)
class ToolCallDelta:
    """One fragment of a tool call as it arrives (PRD sections 5.3, 9.3 and 16).

    A tool call does not arrive whole. The first chunk that mentions it carries
    its identifier and its name; the ones after it carry fragments of its
    arguments; the canonical response assembles them. This type is one step of
    that assembly rather than the call itself, so it is deliberately partial:
    ``id``, ``name`` and ``arguments_delta`` are each ``None`` when this chunk
    said nothing about them, and a consumer accumulates them by ``index``.

    ``index`` is the call's position among the calls the answer contains, not a
    position within the text. It is what lets a consumer tell two interleaved
    calls apart, and match a later fragment to the call it belongs to. It is
    required, because a fragment with no index cannot be attributed to any call.

    Exactly as the canonical tool call keeps the gateway out of the provider's
    business, this delta keeps the arguments as text: the fragments are the
    provider's own bytes, accumulated verbatim, and the consumer decides when
    they form a complete JSON object — the gateway does not repair or parse a
    half-received argument string.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", as_int("tool call delta 'index'", self.index, minimum=0)
        )
        object.__setattr__(self, "id", as_optional_text("tool call delta 'id'", self.id))
        object.__setattr__(self, "name", as_optional_text("tool call delta 'name'", self.name))
        object.__setattr__(
            self,
            "arguments_delta",
            as_optional_text("tool call delta 'arguments_delta'", self.arguments_delta),
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "index": self.index,
            "id": self.id,
            "name": self.name,
            "arguments_delta": self.arguments_delta,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "tool call delta"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, frozenset({"index", "id", "name", "arguments_delta"}))
        return cls(
            index=as_int(f"{where} 'index'", payload.get("index"), minimum=0),
            id=as_optional_text(f"{where} 'id'", payload.get("id")),
            name=as_optional_text(f"{where} 'name'", payload.get("name")),
            arguments_delta=as_optional_text(
                f"{where} 'arguments_delta'", payload.get("arguments_delta")
            ),
        )


_EVENT_KEYS = frozenset(
    {
        "kind",
        "sequence",
        "invocation_id",
        "text",
        "tool_call",
        "usage",
        "finish_reason",
        "raw_finish_reason",
        "error_reason",
        "error_message",
    }
)

# The one payload each kind reports, and the auxiliary fields that may accompany
# it. The rule the rest of the package applies — a mode and its payload must
# agree, a ``text`` response format must not carry a schema — holds here too:
# the kind names the payload, and a second one is a different event rather than
# a larger one. ``raw_finish_reason`` alone is auxiliary beside ``finish_reason``,
# because the provider's own reason is only meaningful beside the classification
# of it (PRD section 9.1); ``error_message`` is auxiliary beside ``error_reason``,
# because the reason is what a caller switches on and the message only explains
# it. The two terminal kinds allow nothing else, and in particular neither
# allows the other's payload: a finish reason classifies *why generation
# stopped*, and an ``error`` event says generation did not stop but failed, so a
# terminal event carrying both would assert two contradictory things at once.
_EVENT_PAYLOADS: dict[StreamEventKind, tuple[frozenset[str], frozenset[str]]] = {
    StreamEventKind.CONTENT: (frozenset({"text"}), frozenset()),
    StreamEventKind.TOOL_CALL: (frozenset({"tool_call"}), frozenset()),
    StreamEventKind.USAGE: (frozenset({"usage"}), frozenset()),
    StreamEventKind.FINISH: (frozenset({"finish_reason"}), frozenset({"raw_finish_reason"})),
    StreamEventKind.ERROR: (frozenset({"error_reason"}), frozenset({"error_message"})),
}

_PAYLOAD_FIELDS = (
    "text",
    "tool_call",
    "usage",
    "finish_reason",
    "raw_finish_reason",
    "error_reason",
    "error_message",
)


@dataclass(frozen=True)
class StreamEvent:
    """One event of a streamed invocation (PRD sections 5.3 and 16).

    ``kind`` says what the event reports, and the event carries exactly that
    payload: a ``content`` event its ``text``, a ``tool_call`` event its
    :class:`ToolCallDelta`, a ``usage`` event its :class:`~model_gateway.domain.Usage`,
    and a ``finish`` event its ``finish_reason``. Any other combination is a
    state the canonical stream cannot represent honestly — a ``content`` event
    that also reports usage is two events wearing one name — and is refused at
    construction rather than silently truncated.

    ``finish_reason`` is the gateway's own classification; ``raw_finish_reason``
    is the provider's own reason verbatim, kept beside it exactly as the
    canonical response keeps the two apart, so an unrecognised reason states
    plainly that the gateway did not classify it without the gateway having
    deleted what upstream said (PRD sections 5.2 and 9.1). Both belong to a
    ``finish`` event alone.

    ``error_reason`` and ``error_message`` belong to an ``error`` event alone,
    and they are the failure's own vocabulary rather than a finish reason's
    (PRD sections 5.4 and 16). ``error_reason`` is the gateway's stable
    machine-readable reason — the value of
    :class:`~model_gateway.errors.ErrorReason` for the failure that ended the
    stream — and is what a consumer switches on. ``error_message`` is the same
    sanitised message the failure carries, gateway-authored and never an
    upstream message, URL or credential, so the reason and the text agree with
    the error the gateway reports beside the event. A ``finish`` event may not
    carry either, and an ``error`` event may not carry a finish reason: a
    terminal event that carried both would say generation stopped and that it
    never stopped.
    """

    kind: StreamEventKind
    sequence: int
    invocation_id: str
    text: str | None = None
    tool_call: ToolCallDelta | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    raw_finish_reason: str | None = None
    error_reason: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        kind = as_enumeration("stream event 'kind'", self.kind, StreamEventKind)
        sequence = as_int("stream event 'sequence'", self.sequence, minimum=0)
        invocation_id = as_required_text("stream event 'invocation_id'", self.invocation_id)
        text = as_optional_text("stream event 'text'", self.text)
        tool_call = _as_tool_call_delta("stream event 'tool_call'", self.tool_call)
        usage = _as_usage("stream event 'usage'", self.usage)
        finish_reason = _as_finish_reason("stream event 'finish_reason'", self.finish_reason)
        raw_finish_reason = as_optional_text(
            "stream event 'raw_finish_reason'", self.raw_finish_reason
        )
        error_reason = as_optional_text("stream event 'error_reason'", self.error_reason)
        error_message = as_optional_text("stream event 'error_message'", self.error_message)
        _check_payload(
            kind,
            text,
            tool_call,
            usage,
            finish_reason,
            raw_finish_reason,
            error_reason,
            error_message,
        )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "invocation_id", invocation_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "tool_call", tool_call)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "raw_finish_reason", raw_finish_reason)
        object.__setattr__(self, "error_reason", error_reason)
        object.__setattr__(self, "error_message", error_message)

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "kind": self.kind.value,
            "sequence": self.sequence,
            "invocation_id": self.invocation_id,
            "text": self.text,
            "tool_call": None if self.tool_call is None else self.tool_call.to_dict(),
            "usage": None if self.usage is None else self.usage.to_dict(),
            "finish_reason": None if self.finish_reason is None else self.finish_reason.value,
            "raw_finish_reason": self.raw_finish_reason,
            "error_reason": self.error_reason,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "stream event"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _EVENT_KEYS)
        raw_finish_reason = payload.get("finish_reason")
        return cls(
            kind=as_enumeration(f"{where} 'kind'", payload.get("kind"), StreamEventKind),
            sequence=as_int(f"{where} 'sequence'", payload.get("sequence"), minimum=0),
            invocation_id=as_required_text(
                f"{where} 'invocation_id'", payload.get("invocation_id")
            ),
            text=as_optional_text(f"{where} 'text'", payload.get("text")),
            tool_call=_optional_tool_call_delta(payload.get("tool_call")),
            usage=_optional_usage(payload.get("usage")),
            finish_reason=_optional_finish_reason(raw_finish_reason),
            raw_finish_reason=as_optional_text(
                f"{where} 'raw_finish_reason'", payload.get("raw_finish_reason")
            ),
            error_reason=as_optional_text(f"{where} 'error_reason'", payload.get("error_reason")),
            error_message=as_optional_text(
                f"{where} 'error_message'", payload.get("error_message")
            ),
        )


def _as_tool_call_delta(where: str, value: object) -> ToolCallDelta | None:
    """A tool call delta, or ``None`` when the event carries none."""
    if value is None:
        return None
    if not isinstance(value, ToolCallDelta):
        raise ValueError(f"{where} must be a ToolCallDelta, got {describe(value)}")
    return value


def _as_usage(where: str, value: object) -> Usage | None:
    """A usage record, or ``None`` when the event carries none."""
    if value is None:
        return None
    if not isinstance(value, Usage):
        raise ValueError(f"{where} must be a Usage, got {describe(value)}")
    return value


def _as_finish_reason(where: str, value: object) -> FinishReason | None:
    """A finish reason, or ``None`` when the event carries none."""
    if value is None:
        return None
    return as_enumeration(where, value, FinishReason)


def _optional_tool_call_delta(value: object) -> ToolCallDelta | None:
    """Parse an optional tool call delta."""
    if value is None:
        return None
    return ToolCallDelta.from_dict(value)


def _optional_usage(value: object) -> Usage | None:
    """Parse an optional usage record."""
    if value is None:
        return None
    return Usage.from_dict(value)


def _optional_finish_reason(value: object) -> FinishReason | None:
    """Parse an optional finish reason."""
    if value is None:
        return None
    return as_enumeration("stream event 'finish_reason'", value, FinishReason)


def _check_payload(
    kind: StreamEventKind,
    text: str | None,
    tool_call: ToolCallDelta | None,
    usage: Usage | None,
    finish_reason: FinishReason | None,
    raw_finish_reason: str | None,
    error_reason: str | None,
    error_message: str | None,
) -> None:
    """Refuse a kind that does not agree with the payload it carries (PRD section 9.1).

    The payload the kind names must be present, because a ``content`` event with
    no text reports nothing while claiming to report content, and a terminal
    event with nothing to terminate on does not terminate anything: a ``finish``
    event with no reason does not say why generation stopped, and an ``error``
    event with no reason does not say what failed.

    No other field may be present, because the kind names the one payload the
    event carries: a second one is a different event, and accepting it here
    would let one event stand for two facts the gateway must be able to tell
    apart. That rule is what keeps the two terminal kinds disjoint — a
    ``finish`` event may not carry an error reason or an error message, and an
    ``error`` event may not carry a finish reason or the provider's raw reason —
    so a consumer never has to decide which of two contradictory statements a
    terminal event was making.
    """
    payload, auxiliary = _EVENT_PAYLOADS[kind]
    carried = {
        "text": text,
        "tool_call": tool_call,
        "usage": usage,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "error_reason": error_reason,
        "error_message": error_message,
    }
    missing = sorted(name for name in payload if carried[name] is None)
    if missing:
        raise ValueError(
            f"a {kind.value!r} stream event carries none of {missing}; the kind names the one "
            f"payload it reports, so an event of that kind without it asserts nothing"
        )
    extra = sorted(
        name
        for name in _PAYLOAD_FIELDS
        if carried[name] is not None and name not in payload and name not in auxiliary
    )
    if extra:
        raise ValueError(
            f"a {kind.value!r} stream event carries {extra[0]!r}, which that kind does not "
            f"report; one event carries the one payload its kind names, and a second payload "
            f"is a different event rather than a larger one"
        )
