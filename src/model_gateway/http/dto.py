"""OpenAI-compatible request and response DTOs (PRD sections 4.3, 4.4, 9.1 and 9.5).

The HTTP surface speaks the OpenAI wire format and the gateway core speaks the
canonical one; this module is the whole of the translation between them, in both
directions.

**Inbound, every field is mapped or refused.** Nothing the caller sent is
dropped. PRD section 9.5 forbids silently discarding a parameter, and a caller
told its request succeeded while a parameter was discarded has been misled about
what the model was asked. A parameter the gateway recognises but cannot honour —
``n`` above one, ``logprobs``, the legacy ``functions`` family, ``audio`` — is
answered with a capability error naming it, a recognised parameter whose value
carries the wrong JSON type is answered with an invalid-request error naming
the parameter and the type supplied, and a key the OpenAI format does not
define is answered with an invalid-request error naming that key. That is a
deliberately strict reading of "OpenAI-compatible": the response shapes are the
compatible part, and the request shape is validated rather than waved through.
Only a parameter whose OpenAI default is *also* a promise this gateway can keep is
accepted, and the promise is about what the gateway does rather than about what
the value spells. ``store: false`` and ``n: 1`` qualify, because the gateway
stores nothing and answers one choice. ``parallel_tool_calls: true`` does not:
nothing in this slice implements parallel tool calls or preserves the distinction
between parallel and sequential ones, so accepting the flag — in either
direction — would advertise a capability the gateway does not have, and the
parameter is refused whatever its value.

**Inbound images must be inline.** An ``image_url`` part is accepted only as a
base64 data URI. A remote ``http`` or ``https`` URL is refused with a capability
error naming ``remote_image_fetch``. The canonical image type documents a URL as
one the URL policy has already admitted; PRD section 9.4 requires the gateway to
screen it before the fetch, and section 14.1 requires the fetch itself to run
under address pinning and redirect evaluation. None of that exists on this
endpoint, and the screening cannot simply be added here: it resolves DNS, and a
blocking resolution inside an async request handler stalls the event loop for
every other request. Refusing the URL is the honest answer until the safe
asynchronous validation and transport path exists (PRD section 9.5). Any other
scheme, ``file`` included, is malformed input rather than a missing feature.

**Outbound, every canonical item has a home or the response is refused.** Text,
tool calls and refusals each have a field in the OpenAI completion message. An
image does not: the chat completions shape has nowhere to put one, so a response
carrying one is refused with a capability error rather than reported with the
image missing. A caller must be able to tell that the answer it received is not
the answer the model gave (PRD section 9.1).

**Outbound, the order the model emitted must survive the translation.** Canonical
output is an ordered sequence in which text, refusals and tool calls interleave —
a model may explain itself, call a tool and then explain the result — while the
OpenAI message separates ``content`` from ``tool_calls`` and so records no order
between them. Joining the text and hoisting the calls would quietly turn "said,
did, said" into "said said, did", which is not the answer the model gave, so a
response whose items do not group — every content item before every tool call —
is refused with a capability error naming ``interleaved_output`` rather than
flattened. The rule is about the ordering the shape cannot carry, not about how
many items there are: an ordinary answer, a tool-call-only answer and the common
"text then tool calls" answer all remain representable and are returned normally.

Two canonical distinctions survive the translation because the OpenAI shape has
room for them. A finish reason the gateway could not classify is reported as the
provider's own raw reason rather than as an invented OpenAI value, and a token
count the provider never reported is reported as zero only because the OpenAI
usage object types its counts as integers — the canonical record keeps the
difference between "none were used" and "nobody counted" (PRD section 13).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Self
from urllib.parse import urlsplit

import orjson

from model_gateway.capabilities import Capability
from model_gateway.domain import (
    FinishReason,
    FrozenJsonObject,
    GenerationConfig,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    JsonValue,
    Message,
    ResponseFormat,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    Usage,
    freeze_json,
    thaw_json,
)
from model_gateway.errors import CapabilityError, InvalidArgumentError

if TYPE_CHECKING:
    from model_gateway.registry import ModelInfo
    from model_gateway.security.identity import Identity

__all__ = [
    "ChatCompletion",
    "ChatMessage",
    "ChatRequest",
    "ModelCard",
    "model_list_payload",
]

_OBJECT_MODEL = "model"
_OBJECT_LIST = "list"
_OBJECT_CHAT_COMPLETION = "chat.completion"
_OWNED_BY = "gateway"
_MODEL_CREATED = 0
_ASSISTANT = "assistant"

_CHAT_REQUEST_KEYS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
        "tools",
        "tool_choice",
        "stream",
        "user",
        "metadata",
        "tenant_id",
    }
)

_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_calls", "tool_call_id"})
_TEXT_PART_KEYS = frozenset({"type", "text"})
_IMAGE_PART_KEYS = frozenset({"type", "image_url"})
_IMAGE_URL_KEYS = frozenset({"url"})
_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
_FUNCTION_CALL_KEYS = frozenset({"name", "arguments"})
_TOOL_KEYS = frozenset({"type", "function"})
_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
_NAMED_CHOICE_KEYS = frozenset({"type", "function"})
_NAMED_CHOICE_FUNCTION_KEYS = frozenset({"name"})
_RESPONSE_FORMAT_KEYS = frozenset({"type", "json_schema"})
_JSON_SCHEMA_KEYS = frozenset({"name", "schema", "strict"})

_DATA_URI_PREFIX = "data:"
_BASE64_MARKER = ";base64,"
_SECURE_SCHEMES = frozenset({"http", "https"})


def _is_int(value: object) -> bool:
    # bool is a subclass of int, and JSON true is not an integer
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _is_object(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_array(value: object) -> bool:
    return isinstance(value, list)


def _is_array_or_object(value: object) -> bool:
    return isinstance(value, (list, Mapping))


def _nullable(check: Callable[[object], bool]) -> Callable[[object], bool]:
    def check_or_null(value: object) -> bool:
        return value is None or check(value)

    return check_or_null


def _is_default_of(default: object) -> Callable[[object], bool]:
    def is_default(value: object) -> bool:
        return value is None if default is None else value == default

    return is_default


def _is_empty(value: object) -> bool:
    return not value


class _Feature(NamedTuple):
    """An OpenAI parameter the gateway recognises and cannot honour.

    ``capability`` names what the parameter asks for, so the refusal says which
    feature is missing rather than only which key was sent. ``wire`` states the
    JSON type the value must carry and ``typed`` enforces it on the raw body:
    a value of another type is malformed input — the caller sent something the
    format does not define for that key — and is refused as a 400 naming the
    parameter and the type supplied, before the capability question is reached.
    ``default`` decides whether a correctly typed value actually asks for that
    feature: a parameter whose value is the OpenAI default asks for nothing, and
    refusing it would reject a request the gateway can serve in full.
    """

    capability: str
    wire: str
    typed: Callable[[object], bool]
    default: Callable[[object], bool]


_UNSUPPORTED_FEATURES: Mapping[str, _Feature] = {
    "n": _Feature("multiple_choices", "an integer", _is_int, _is_default_of(1)),
    "logprobs": _Feature("log_probabilities", "a boolean", _is_bool, _is_default_of(False)),
    "top_logprobs": _Feature(
        "log_probabilities", "an integer or null", _nullable(_is_int), _is_default_of(None)
    ),
    "logit_bias": _Feature("logit_bias", "an object", _is_object, _is_default_of({})),
    "store": _Feature("response_storage", "a boolean", _is_bool, _is_default_of(False)),
    "stream_options": _Feature("streaming", "an object", _is_object, _is_default_of({})),
    # Refused whatever its value, unlike the defaults above. ``true`` asks for
    # parallel tool calls and ``false`` asks for a guarantee of sequential ones,
    # and this slice implements neither and models no distinction between them,
    # so accepting the parameter would advertise a capability the gateway does
    # not have (PRD section 9.5). It is refused rather than mapped onto a
    # canonical field, because inventing that field is a contract change and not
    # a translation.
    "parallel_tool_calls": _Feature(
        "parallel_tool_calls", "a boolean or null", _nullable(_is_bool), _is_default_of(None)
    ),
    "modalities": _Feature("non_text_output", "an array", _is_array, _is_default_of(["text"])),
    "audio": _Feature(
        "audio_output", "an object or null", _nullable(_is_object), _is_default_of(None)
    ),
    "prediction": _Feature(
        "predicted_output", "an object or null", _nullable(_is_object), _is_default_of(None)
    ),
    "reasoning_effort": _Feature(
        "reasoning_effort", "a string or null", _nullable(_is_text), _is_default_of(None)
    ),
    "service_tier": _Feature(
        "service_tier", "a string or null", _nullable(_is_text), _is_default_of(None)
    ),
    "web_search_options": _Feature(
        "grounded_search", "an object or null", _nullable(_is_object), _is_default_of(None)
    ),
    "functions": _Feature(
        "legacy_functions", "an array or object", _is_array_or_object, _is_empty
    ),
    "function_call": _Feature(
        "legacy_functions", "a string or null", _nullable(_is_text), _is_default_of(None)
    ),
}

_ROLE_NAMES = ", ".join(repr(member.value) for member in Role)

_FINISH_REASONS: Mapping[FinishReason, str] = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.TOOL_CALLS: "tool_calls",
    FinishReason.CONTENT_FILTER: "content_filter",
}

_IMAGE_OUTPUT_CAPABILITY = "image_output"
_REMOTE_IMAGE_CAPABILITY = "remote_image_fetch"
_INTERLEAVED_OUTPUT_CAPABILITY = "interleaved_output"


@dataclass(frozen=True)
class ModelCard:
    """One entry of the ``GET /v1/models`` list (PRD section 4.3).

    A card carries the public model ID and the capability metadata a caller needs
    to choose a model, and nothing else. It is built from a
    :class:`~model_gateway.registry.ModelInfo`, which holds no route, endpoint or
    credential, so upstream configuration cannot reach a response through this
    type. That is what makes acceptance criterion 12 structural rather than a
    matter of remembering to strip fields: there is no field here to strip.

    ``context_window``, ``max_output_tokens`` and ``dimensions`` are emitted only
    when the route states them. A field the deployment never configured is left
    out rather than reported as zero, because zero is a claim about the model
    that no configuration made.
    """

    id: str
    capabilities: frozenset[Capability]
    context_window: int | None = None
    max_output_tokens: int | None = None
    dimensions: int | None = None
    created: int = _MODEL_CREATED
    owned_by: str = _OWNED_BY

    @classmethod
    def from_model_info(cls, info: ModelInfo) -> Self:
        """Build the public card for one configured model."""
        return cls(
            id=info.id,
            capabilities=frozenset(info.capabilities),
            context_window=info.context_window,
            max_output_tokens=info.max_output_tokens,
            dimensions=info.dimensions,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "object": _OBJECT_MODEL,
            "created": self.created,
            "owned_by": self.owned_by,
            "capabilities": sorted(capability.value for capability in self.capabilities),
        }
        optional: Mapping[str, int | None] = {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "dimensions": self.dimensions,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
        return payload


def model_list_payload(cards: Sequence[ModelCard]) -> dict[str, object]:
    """The ``GET /v1/models`` envelope: an OpenAI list of model objects."""
    return {"object": _OBJECT_LIST, "data": [card.to_dict() for card in cards]}


@dataclass(frozen=True)
class ChatRequest:
    """A parsed ``POST /v1/chat/completions`` request (PRD section 4.4).

    The parse is the gateway's, not the wire format's: by the time a caller has
    one of these, every field has a canonical home and every message is a
    canonical :class:`~model_gateway.domain.Message`. Two OpenAI fields map onto
    canonical fields that are not their namesakes — ``user`` is the caller's
    stable end-user identifier and becomes the invocation's ``actor_id``, and
    ``metadata`` becomes the invocation's non-sensitive request metadata, which
    the gateway never forwards upstream (PRD section 5.1).

    ``tenant_id`` maps onto the invocation's tenant, but is never trusted as
    one: the handler validates it against the tenant bound to the presented
    credential — and discards it when the caller is unauthenticated — before
    the invocation is built (PRD section 5.1).

    The canonical request is built during the parse rather than afterwards, so
    the rules it enforces — a conversation with at least one message, metadata
    that does not name a canonical field, a tool choice the declared tools can
    satisfy — are applied before any provider is called. A request the gateway
    cannot represent is refused as malformed input, not discovered to be
    unrepresentable halfway through an invocation.

    ``stream`` is carried rather than acted on. The endpoint refuses streaming
    today (PRD section 16), and the refusal belongs to the endpoint, which knows
    what it can serve; this type records what the caller asked for so that the
    refusal can name it.
    """

    invocation: InvocationRequest
    stream: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        """Parse a decoded JSON request body, refusing anything it cannot carry.

        A ``ValueError`` from the canonical types is a validation failure of this
        request and becomes an invalid-request error; its message is the domain's
        own, which is gateway-authored and names the field that failed.
        """
        where = "the request body"
        body = _as_object(where, payload)
        _reject_unsupported_parameters(body)
        try:
            invocation = InvocationRequest(
                model=_as_required_text(f"{where} 'model'", body.get("model")),
                messages=_as_messages(f"{where} 'messages'", body.get("messages")),
                actor_id=_as_optional_text(f"{where} 'user'", body.get("user")),
                tenant_id=_as_optional_text(f"{where} 'tenant_id'", body.get("tenant_id")),
                generation_config=_generation_config(body),
                tools=_as_tools(f"{where} 'tools'", body.get("tools")),
                tool_choice=_as_tool_choice(f"{where} 'tool_choice'", body.get("tool_choice")),
                metadata=_as_metadata(f"{where} 'metadata'", body.get("metadata")),
            )
        except ValueError as error:
            raise InvalidArgumentError(f"invalid chat completion request: {error}") from error
        return cls(invocation=invocation, stream=_as_flag(f"{where} 'stream'", body.get("stream")))

    def to_invocation(
        self, *, invocation_id: str, identity: Identity | None = None
    ) -> InvocationRequest:
        """The canonical invocation, carrying the id minted for this request.

        ``response_modality`` stays text: this endpoint produces chat text, tool
        calls and refusals, and an image request has no representation in its
        response shape.

        ``identity`` is the trusted identity assigned by the authentication
        middleware. The invocation's ``tenant_id`` is that identity's tenant —
        never the request body's. A body ``tenant_id`` has already been checked
        against the trusted tenant by the handler (PRD section 5.1), so the
        body's value is discarded here rather than trusted: when there is no
        trusted identity, no tenant is propagated at all.
        """
        tenant = identity.tenant if identity is not None and identity.authenticated else None
        return replace(self.invocation, invocation_id=invocation_id, tenant_id=tenant)


@dataclass(frozen=True)
class ChatMessage:
    """The assistant message of an OpenAI chat completion (PRD section 4.4).

    ``content`` is the concatenated text of the model's output and is ``None``
    when the answer was only tool calls, which is how OpenAI spells that case.
    Joining the text parts is faithful only because a response that interleaves
    text and tool calls never reaches this type: :class:`ChatCompletion` refuses
    it (see :func:`_reject_unrepresentable_order`). ``refusal`` carries the
    provider's own refusal text, kept apart from ordinary text because a provider
    declining on safety grounds is a different event from a model answering
    (PRD section 5.2).
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    refusal: str | None = None
    role: str = _ASSISTANT

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.refusal is not None:
            payload["refusal"] = self.refusal
        if self.tool_calls:
            payload["tool_calls"] = [_tool_call_payload(call) for call in self.tool_calls]
        return payload


@dataclass(frozen=True)
class ChatCompletion:
    """An OpenAI chat completion response (PRD section 4.4).

    The gateway's own accounting — upstream model, attempt count, fallback use,
    idempotency status, cost — is deliberately not in this payload: the OpenAI
    chat completions shape has no field for it, and adding one would be a
    contract the gateway invented. That accounting belongs to the gRPC contract
    and to usage records (PRD sections 5.2 and 13).
    """

    id: str
    model: str
    message: ChatMessage
    usage: Usage
    created: int
    finish_reason: str | None = None

    @classmethod
    def from_invocation(
        cls,
        response: InvocationResponse,
        *,
        completion_id: str,
        created: int,
    ) -> Self:
        """Translate a canonical response into the OpenAI completion shape.

        Two canonical responses have no representation here and are refused
        rather than approximated: one carrying an image, because the shape has
        nowhere to put it, and one whose text and tool calls interleave, because
        the shape cannot record their order. Both refusals are capability errors,
        so a caller learns which feature is missing instead of receiving an
        answer that is not the one the model gave (PRD sections 9.1 and 9.5).
        """
        if response.images:
            raise CapabilityError(
                "the chat completions response shape cannot carry image output, and the gateway "
                "does not return a partial answer; the model produced an image",
                details={"capability": _IMAGE_OUTPUT_CAPABILITY},
            )
        _reject_unrepresentable_order(response)
        return cls(
            id=completion_id,
            model=response.model,
            message=ChatMessage(
                content=_message_content(response),
                tool_calls=response.tool_calls,
                refusal=_refusal_text(response),
            ),
            usage=response.usage,
            created=created,
            finish_reason=_finish_reason(response),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "object": _OBJECT_CHAT_COMPLETION,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": self.message.to_dict(),
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": _usage_payload(self.usage),
        }


def _reject_unrepresentable_order(response: InvocationResponse) -> None:
    """Refuse output whose order the OpenAI message shape cannot carry.

    The shape holds one ``content`` field, one ``refusal`` field and one
    ``tool_calls`` array, so it records the order of the text parts among
    themselves and of the tool calls among themselves, and nothing about the
    order between the two groups. Canonical output is one sequence in which they
    interleave, and a response that interleaves them — text, a call, more text —
    therefore has no representation here: joining the text and hoisting the calls
    would report "said did said" as "said said, did", which is a different answer
    from the one the model gave (PRD section 9.1). Every content item before
    every tool call is the one grouping that survives, and anything else is a
    capability the response shape does not have (PRD section 9.5).
    """
    seen_tool_call = False
    for item in response.content:
        if isinstance(item, ToolCall):
            seen_tool_call = True
        elif seen_tool_call:
            raise CapabilityError(
                "the chat completions response shape cannot carry a response whose text and tool "
                "calls interleave, and the gateway does not flatten model output into a different "
                "answer; the model emitted content after a tool call",
                details={"capability": _INTERLEAVED_OUTPUT_CAPABILITY},
            )


def _message_content(response: InvocationResponse) -> str | None:
    """The assistant content, or ``None`` for a tool-call-only answer.

    The text parts are joined because the OpenAI shape types ``content`` as one
    string; :func:`_reject_unrepresentable_order` has already established that no
    tool call falls between them, so nothing about their order is lost by it.
    """
    text = response.text
    if not text and response.tool_calls:
        return None
    return text


def _refusal_text(response: InvocationResponse) -> str | None:
    """The provider's refusal text, or ``None`` when it did not decline."""
    refusals = response.refusals
    if not refusals:
        return None
    return "".join(refusal.text for refusal in refusals)


def _finish_reason(response: InvocationResponse) -> str | None:
    """The OpenAI finish reason, or the provider's own reason when unclassified.

    A reason the gateway could not classify is reported verbatim rather than
    replaced by an OpenAI value the provider never gave. ``UNKNOWN`` beside a raw
    reason means the gateway did not recognise it, and the raw reason is what
    upstream actually said (PRD sections 5.2 and 9.1). ``None`` means the
    provider reported no reason at all.
    """
    if response.finish_reason is FinishReason.UNKNOWN:
        return response.raw_finish_reason
    return _FINISH_REASONS[response.finish_reason]


def _usage_payload(usage: Usage) -> dict[str, int]:
    """The OpenAI usage object.

    Its three counts are typed as integers, so a count the provider never
    reported is presented as zero here. That presentation is the boundary's, not
    the record's: the canonical :class:`~model_gateway.domain.Usage` keeps an
    unreported count apart from a zero one, and marks a count the gateway derived
    rather than received. A total the provider omitted is summed from the two
    counts it did report, which is arithmetic the caller can repeat.
    """
    prompt = 0 if usage.input_tokens is None else usage.input_tokens
    completion = 0 if usage.output_tokens is None else usage.output_tokens
    total = usage.total_tokens
    if total is None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _tool_call_payload(call: ToolCall) -> dict[str, object]:
    """One tool call in the OpenAI shape: arguments are a JSON-encoded string.

    Canonical arguments are the parsed object when the provider's arguments
    parsed, the raw string when they did not, and ``None`` when the provider
    reported none. All three are reported as the string OpenAI expects, so an
    argument payload the gateway could not parse is returned to the caller
    unchanged rather than repaired into an empty object (PRD section 9.3).
    """
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": _arguments_payload(call.arguments)},
    }


def _arguments_payload(arguments: Mapping[str, JsonValue] | str | None) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return orjson.dumps(thaw_json(arguments)).decode("utf-8")


def _reject_unsupported_parameters(body: Mapping[str, object]) -> None:
    """Refuse a parameter the gateway cannot honour, naming what it asked for.

    A key the OpenAI format does not define is malformed input, and so is a key
    it does define whose value carries a type that format does not define for
    it: a wrong-typed value is refused as a 400 naming the parameter and the
    type supplied, which is a different error from the unsupported parameter.
    A correctly typed value that asks for something this gateway has no
    representation for is a capability the route cannot provide — the request is
    well formed and the refusal is about the feature (PRD section 9.5).
    """
    for name in sorted(set(body) - _CHAT_REQUEST_KEYS):
        feature = _UNSUPPORTED_FEATURES.get(name)
        if feature is None:
            supported = ", ".join(repr(key) for key in sorted(_CHAT_REQUEST_KEYS))
            raise InvalidArgumentError(
                f"the request body has unsupported parameter {name!r}; supported parameters are "
                f"{supported}"
            )
        value = body[name]
        if not feature.typed(value):
            raise InvalidArgumentError(
                f"the request body parameter {name!r} must be {feature.wire}, got "
                f"{_json_type(value)}"
            )
        if feature.default(value):
            continue
        raise CapabilityError(
            f"the gateway cannot honour {name!r}; it would request {feature.capability!r}, which "
            f"no configured route is asked to provide and the gateway will not silently discard",
            details={"capability": feature.capability, "parameter": name},
        )


def _as_messages(where: str, value: object) -> tuple[Message, ...]:
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be an array of messages, got {_json_type(value)}"
        )
    return tuple(
        _message_from(f"{where}[{index}]", item) for index, item in enumerate(value)
    )


def _message_from(where: str, payload: object) -> Message:
    body = _as_object(where, payload)
    _reject_unknown_keys(where, body, _MESSAGE_KEYS)
    try:
        return Message(
            role=_role(f"{where} 'role'", body.get("role")),
            content=_content_from(f"{where} 'content'", body.get("content")),
            name=_as_optional_text(f"{where} 'name'", body.get("name")),
            tool_calls=_as_tool_calls(f"{where} 'tool_calls'", body.get("tool_calls")),
            tool_call_id=_as_optional_text(f"{where} 'tool_call_id'", body.get("tool_call_id")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid message: {error}") from error


def _role(where: str, value: object) -> Role:
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {_json_type(value)}")
    try:
        return Role(value)
    except ValueError as error:
        raise InvalidArgumentError(
            f"{where} must be one of {_ROLE_NAMES}; got {value!r}"
        ) from error


def _content_from(where: str, value: object) -> tuple[TextPart | ImagePart, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (TextPart(value),)
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be a string or an array of content parts, got {_json_type(value)}"
        )
    return tuple(
        _content_part_from(f"{where}[{index}]", item) for index, item in enumerate(value)
    )


def _content_part_from(where: str, payload: object) -> TextPart | ImagePart:
    body = _as_object(where, payload)
    kind = body.get("type")
    if kind == "text":
        _reject_unknown_keys(where, body, _TEXT_PART_KEYS)
        text = body.get("text")
        if not isinstance(text, str):
            raise InvalidArgumentError(
                f"{where} 'text' must be a string, got {_json_type(text)}"
            )
        return TextPart(text)
    if kind == "image_url":
        _reject_unknown_keys(where, body, _IMAGE_PART_KEYS)
        image = _as_object(f"{where} 'image_url'", body.get("image_url"))
        _reject_unknown_keys(f"{where} 'image_url'", image, _IMAGE_URL_KEYS)
        return _image_part_from(f"{where} 'image_url' 'url'", image.get("url"))
    if not isinstance(kind, str):
        raise InvalidArgumentError(
            f"{where} must carry a string 'type' discriminator, got {_json_type(kind)}"
        )
    raise InvalidArgumentError(
        f"{where} has type {kind!r}; this endpoint carries 'text' and 'image_url' content parts"
    )


def _image_part_from(where: str, value: object) -> ImagePart:
    """An image part from an OpenAI ``image_url`` value.

    A data URI carries the payload inline, so it becomes base64 content with its
    media type and needs no fetch, no policy decision and no network. A remote
    ``http`` or ``https`` URL is refused with a capability error: the canonical
    type documents its URL as one the URL policy has admitted, and admitting it
    means resolving the hostname and rejecting loopback, private, link-local and
    metadata addresses — a blocking DNS lookup this async handler must not
    perform, and one that would only screen the URL, not pin the connection or
    evaluate redirects as PRD section 14.1 requires. Remote image fetching
    arrives with that safe asynchronous validation and transport path; until then
    only inline content is accepted. Any other scheme, ``file`` included, is
    malformed input rather than a missing feature.
    """
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(
            f"{where} must be a non-empty string, got {_json_type(value)}"
        )
    if value.startswith(_DATA_URI_PREFIX):
        media_type, _, payload = value[len(_DATA_URI_PREFIX) :].partition(_BASE64_MARKER)
        if not payload:
            raise InvalidArgumentError(
                f"{where} is a data URI without a base64 payload; supply "
                f"'data:<media type>;base64,<payload>'"
            )
        return ImagePart(data=payload, media_type=media_type or None)
    scheme = urlsplit(value).scheme.lower()
    if scheme in _SECURE_SCHEMES:
        raise CapabilityError(
            "the gateway cannot fetch a remote image URL on this endpoint: it screens an image "
            "URL against its URL policy and fetches it under address pinning and redirect "
            "evaluation, and neither is implemented yet, so only an inline base64 data URI is "
            "accepted; remote image fetching arrives with that safe asynchronous path",
            details={"capability": _REMOTE_IMAGE_CAPABILITY},
        )
    raise InvalidArgumentError(
        f"{where} must be a base64 data URI, got scheme {scheme!r}; supply "
        f"'data:<media type>;base64,<payload>'"
    )


def _as_tool_calls(where: str, value: object) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be an array of tool calls, got {_json_type(value)}"
        )
    return tuple(
        _tool_call_from(f"{where}[{index}]", item) for index, item in enumerate(value)
    )


def _tool_call_from(where: str, payload: object) -> ToolCall:
    body = _as_object(where, payload)
    _reject_unknown_keys(where, body, _TOOL_CALL_KEYS)
    kind = body.get("type", "function")
    if kind != "function":
        raise InvalidArgumentError(
            f"{where} 'type' must be 'function', got {_json_type(kind)}"
        )
    function = _as_object(f"{where} 'function'", body.get("function"))
    _reject_unknown_keys(f"{where} 'function'", function, _FUNCTION_CALL_KEYS)
    return ToolCall(
        id=_as_required_text(f"{where} 'id'", body.get("id")),
        name=_as_required_text(f"{where} 'function' 'name'", function.get("name")),
        arguments=_arguments_from(f"{where} 'function' 'arguments'", function.get("arguments")),
    )


def _arguments_from(where: str, value: object) -> Mapping[str, JsonValue] | str | None:
    """Tool arguments, parsed when they parse and kept verbatim when they do not.

    OpenAI encodes arguments as a JSON string. Storing the parsed object when it
    is one keeps the canonical representation useful, and keeping the raw string
    when it is not means a malformed argument payload is relayed rather than
    repaired into an empty object the caller never sent (PRD section 9.3).
    """
    if value is None or isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise InvalidArgumentError(
            f"{where} must be a JSON-encoded string, got {_json_type(value)}"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, Mapping) else value


def _as_tools(where: str, value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be an array of tool declarations, got {_json_type(value)}"
        )
    return tuple(_tool_from(f"{where}[{index}]", item) for index, item in enumerate(value))


def _tool_from(where: str, payload: object) -> ToolDefinition:
    body = _as_object(where, payload)
    _reject_unknown_keys(where, body, _TOOL_KEYS)
    kind = body.get("type", "function")
    if kind != "function":
        raise InvalidArgumentError(
            f"{where} 'type' must be 'function'; this gateway declares function tools only, "
            f"got {_json_type(kind)}"
        )
    function = _as_object(f"{where} 'function'", body.get("function"))
    _reject_unknown_keys(f"{where} 'function'", function, _FUNCTION_KEYS)
    parameters = function.get("parameters")
    if parameters is None:
        parameters = {}
    return ToolDefinition(
        name=_as_required_text(f"{where} 'function' 'name'", function.get("name")),
        description=_as_text(f"{where} 'function' 'description'", function.get("description")),
        parameters=_as_object(f"{where} 'function' 'parameters'", parameters),
    )


def _as_tool_choice(where: str, value: object) -> ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return ToolChoice(mode=ToolChoiceMode(value))
        except ValueError as error:
            raise InvalidArgumentError(
                f"{where} must be 'none', 'auto', 'required' or a named function choice; "
                f"got {value!r}"
            ) from error
    body = _as_object(where, value)
    _reject_unknown_keys(where, body, _NAMED_CHOICE_KEYS)
    if body.get("type") != "function":
        raise InvalidArgumentError(
            f"{where} 'type' must be 'function', got {_json_type(body.get('type'))}"
        )
    function = _as_object(f"{where} 'function'", body.get("function"))
    _reject_unknown_keys(f"{where} 'function'", function, _NAMED_CHOICE_FUNCTION_KEYS)
    name = _as_required_text(f"{where} 'function' 'name'", function.get("name"))
    return ToolChoice.specific(name)


def _generation_config(body: Mapping[str, object]) -> GenerationConfig:
    """Map the OpenAI sampling parameters onto the canonical generation config.

    ``max_tokens`` becomes ``max_output_tokens``: the OpenAI name counts the
    model's output, which is exactly what the canonical field bounds. Every
    parameter the caller omitted stays unset, so the provider's own default
    applies rather than a value the gateway invented (PRD section 5.1).
    """
    return GenerationConfig(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_output_tokens=body.get("max_tokens"),
        stop_sequences=_as_stop_sequences("the request body 'stop'", body.get("stop")),
        seed=body.get("seed"),
        presence_penalty=body.get("presence_penalty"),
        frequency_penalty=body.get("frequency_penalty"),
        response_format=_as_response_format(
            "the request body 'response_format'", body.get("response_format")
        ),
    )


def _as_stop_sequences(where: str, value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidArgumentError(
            f"{where} must be a string or an array of strings, got {_json_type(value)}"
        )
    return tuple(value)


def _as_response_format(where: str, value: object) -> ResponseFormat | None:
    if value is None:
        return None
    body = _as_object(where, value)
    _reject_unknown_keys(where, body, _RESPONSE_FORMAT_KEYS)
    kind = body.get("type")
    if kind == "text":
        return ResponseFormat(mode=ResponseFormatMode.TEXT)
    if kind == "json_object":
        return ResponseFormat(mode=ResponseFormatMode.JSON_OBJECT)
    if kind == "json_schema":
        schema = _as_object(f"{where} 'json_schema'", body.get("json_schema"))
        _reject_unknown_keys(f"{where} 'json_schema'", schema, _JSON_SCHEMA_KEYS)
        return ResponseFormat(
            mode=ResponseFormatMode.JSON_SCHEMA,
            schema=_schema_document(f"{where} 'json_schema' 'schema'", schema.get("schema")),
            name=_as_optional_text(f"{where} 'json_schema' 'name'", schema.get("name")),
            strict=_as_flag(f"{where} 'json_schema' 'strict'", schema.get("strict")),
        )
    if not isinstance(kind, str):
        raise InvalidArgumentError(
            f"{where} must carry a string 'type' discriminator, got {_json_type(kind)}"
        )
    raise InvalidArgumentError(
        f"{where} has type {kind!r}; supported types are 'text', 'json_object' and 'json_schema'"
    )


def _schema_document(where: str, value: object) -> Mapping[str, JsonValue] | bool:
    """A JSON Schema: an object, or one of the two boolean schemas.

    The canonical type refuses a schema a mode cannot consult, so a
    ``json_schema`` response format without one is an invalid request rather than
    a requirement the gateway quietly drops (PRD section 9.5).
    """
    if isinstance(value, bool):
        return value
    return _as_object(where, value)


def _as_metadata(where: str, value: object) -> Mapping[str, JsonValue]:
    if value is None:
        return FrozenJsonObject()
    frozen = freeze_json(where, value)
    if not isinstance(frozen, Mapping):
        raise InvalidArgumentError(f"{where} must be a JSON object, got {_json_type(value)}")
    return frozen


def _as_flag(where: str, value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise InvalidArgumentError(f"{where} must be a boolean, got {_json_type(value)}")
    return value


def _as_text(where: str, value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {_json_type(value)}")
    return value


def _as_required_text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(
            f"{where} must be a non-empty string, got {_json_type(value)}"
        )
    return value


def _as_optional_text(where: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {_json_type(value)}")
    return value or None


def _as_object(where: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidArgumentError(f"{where} must be a JSON object, got {_json_type(value)}")
    return value


def _reject_unknown_keys(
    where: str, body: Mapping[str, object], supported: frozenset[str]
) -> None:
    unknown = sorted(set(body) - supported)
    if unknown:
        names = ", ".join(repr(key) for key in sorted(supported))
        raise InvalidArgumentError(
            f"{where} has unsupported key {unknown[0]!r}; supported keys are {names}"
        )


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, Mapping):
        return "an object"
    return type(value).__name__
