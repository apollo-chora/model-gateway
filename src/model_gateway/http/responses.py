"""The Responses endpoint: ``POST /v1/responses`` (PRD sections 4.5, 9.1, 9.5 and 16).

The handler is thin on purpose, exactly as the other endpoints' are. It decodes
the body, normalises it into the canonical invocation, calls the inference
service and renders the canonical answer back into the Responses representation.
It resolves no route, retries nothing and counts nothing: routing, fallback,
retry, the deadline and the capability contract are the gateway core's (PRD
sections 9.2, 9.5 and 11.2), and a second copy of that logic on the HTTP path is
how the two transports drift apart.

**A Responses-compatible upstream is not required** (PRD section 4.5). This
endpoint is served by the ordinary chat provider: the canonical invocation is
the one ``POST /v1/chat/completions`` builds and the canonical answer is the one
the chat binding produces. The Responses representation is the *caller-facing*
one, and rendering the canonical answer into it is this module's work rather
than the provider's, which is what lets a provider that has never heard of the
Responses API serve the endpoint without lying about what it returned.

**The accepted request is a documented subset, and nothing else is ignored.**
The body carries ``model``, ``input``, ``instructions``, ``tools``,
``temperature``, ``top_p``, ``max_output_tokens``, ``stream``,
``response_format``, ``tool_choice`` and ``metadata``, and that set is closed
(PRD section 4.5). A key the gateway does not understand is refused with an
invalid-request error naming it; a key the Responses format defines and this
endpoint does not carry — ``previous_response_id``, ``store``, ``text`` and
the rest — is refused with a capability error naming the feature it would
request. Neither is dropped, so a client SDK that serialises a parameter
receives an error it can act on instead of a false assurance that its request
was honoured (PRD sections 4.5 and 9.5). ``stream`` is accepted only as
``false``, which is the default and asks for nothing; ``store: false`` is
likewise the one value that matches what the gateway does, because it stores
nothing. ``idempotency_key`` is the caller's own deduplication key, mapped onto
the invocation's so a repeat of the same request is answered from the gateway's
record of the first answer rather than by asking the provider again; the
response carries ``idempotency_status`` beside ``id``, ``object``, ``created_at``,
``model`` and ``status``, so a caller can tell whether the provider was reached
(PRD section 12). A recognised parameter is held to its wire type before
anything else: a value of the wrong JSON type — ``{"store": 0}``, ``{"background": ""}``,
``{"include": "text"}`` — is an invalid-request error naming the parameter
and the type actually supplied, because the test that decides whether a
recognised parameter was actually requested runs on a validated value rather
than on whatever the caller sent.

**The input is normalised into the canonical conversation.** A string ``input``
is one user message. An array carries message-like objects, each of which is a
role, an ordered content list and nothing else, beside the two items a tool turn
needs: ``function_call``, which becomes an assistant message proposing the call,
and ``function_call_output``, which becomes the tool message answering it. A
content part is ``input_text``, ``output_text`` or ``input_image``: the first
two become canonical text and the third a canonical image, which the route
must declare the ``vision`` capability to carry. The Responses input item and
the canonical message are not the same shape, so the mapping is written out
rather than guessed at, and an item type this endpoint does not carry is
refused with a capability error naming the type.

**The answer is rendered content-first, then tool calls.** The Responses
``output`` array is ordered and mixes ``message`` items with ``function_call``
items, but the renderer emits every content part of the assistant turn first
and then every tool call, so the order it can express is ``(ContentPart)*
(ToolCall)*`` and an arbitrary interleaving of content and tool calls is not
representable. One assistant turn's content parts become one ``message`` item's
ordered content list, and each tool call becomes a ``function_call`` item after
every content part. Two canonical answers are still refused rather than
approximated: one carrying an image, which no ``message`` content variant can
hold, and one whose content continues after a tool call. The second is refused
because a Responses ``output`` is *one assistant turn*, and a turn that calls a
function ends there — splitting the content into a second ``message`` item
would present one turn as two, and joining it would report "said did said" as
"said said, did", which is not the answer the model gave (PRD section 9.1).

**Streaming is refused, not faked.** ``stream: true`` is answered with a
capability error naming ``streaming``, before any provider is called. The
Responses streaming protocol is an *item-lifecycle* protocol: it opens with
``response.created`` and ``response.in_progress``, brackets every item with
``response.output_item.added`` and ``response.output_item.done``, brackets every
content part with ``response.content_part.added`` and ``response.content_part.done``,
streams text as ``response.output_text.delta`` and closes with
``response.completed`` carrying the whole final response object. Every event
carries an item id, an output index, a content index and a monotonic sequence
number, and the ``added``/``done`` events carry the complete item. The canonical
stream (PRD section 16) has none of that: its five kinds report a text fragment,
a tool-call fragment, a usage snapshot, a terminal finish and a terminal error,
and it carries no item identity and no item boundary. Emitting the lifecycle
events would therefore mean the transport inventing item ids and item boundaries
the canonical vocabulary deliberately does not carry, and inventing event types
is exactly what the gateway must not do. The gateway will not answer a streaming
request with a complete, non-streamed body either, because that would tell the
caller it received tokens as they were produced when it did not (PRD section 16),
so the request is refused. This is a documented subset limitation: the endpoint
serves the non-streaming Responses representation only.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Self
from urllib.parse import urlsplit
from uuid import uuid4

import orjson
from blacksheep import HTTPException, Request, Response
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.domain import (
    GenerationConfig,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    JsonValue,
    Message,
    OutputItem,
    RefusalPart,
    ResponseFormat,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    Usage,
    thaw_json,
)
from model_gateway.domain.serialization import (
    as_object,
    as_optional_text,
    as_required_text,
    describe,
)
from model_gateway.errors import CapabilityError, InvalidArgumentError
from model_gateway.http.auth import get_identity
from model_gateway.security.identity import caller_namespace, caller_tenant

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService
    from model_gateway.security.identity import Identity

__all__ = [
    "ResponseObject",
    "ResponsesRequest",
    "create_response",
    "register_responses_routes",
]

ResponsesHandler = Callable[[Request], Awaitable[Response]]

_RESPONSE_ID_PREFIX = "resp_"
_MESSAGE_ID_PREFIX = "msg_"
_FUNCTION_CALL_ID_PREFIX = "fc_"
_OBJECT_RESPONSE = "response"
_OBJECT_MESSAGE = "message"
_OBJECT_OUTPUT_TEXT = "output_text"
_OBJECT_REFUSAL = "refusal"
_OBJECT_FUNCTION_CALL = "function_call"
_ROLE_ASSISTANT = "assistant"
_STATUS_COMPLETED = "completed"
_TYPE_FUNCTION = "function"
_INPUT_MESSAGE = "message"
_INPUT_FUNCTION_CALL = "function_call"
_INPUT_FUNCTION_CALL_OUTPUT = "function_call_output"
_INPUT_TEXT = "input_text"
_INPUT_OUTPUT_TEXT = "output_text"
_INPUT_IMAGE = "input_image"

_STREAMING_CAPABILITY = "streaming"
_IMAGE_OUTPUT_CAPABILITY = "image_output"
_INTERLEAVED_OUTPUT_CAPABILITY = "interleaved_output"
_ANNOTATIONS_CAPABILITY = "annotations"
_STRICT_TOOLS_CAPABILITY = "strict_tool_schemas"
_REMOTE_IMAGE_CAPABILITY = "remote_image_fetch"

_DATA_URI_PREFIX = "data:"
_BASE64_MARKER = ";base64,"
_SECURE_SCHEMES = frozenset({"http", "https"})

_REQUEST_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "temperature",
        "top_p",
        "max_output_tokens",
        "stream",
        "response_format",
        "tool_choice",
        "metadata",
        "idempotency_key",
    }
)
"""The request fields ``POST /v1/responses`` accepts (PRD section 4.5).

The set is closed on purpose, exactly as the other endpoints' are: PRD section
4.5 makes the gateway's OpenAI compatibility a supported subset rather than
drop-in compatibility, so a parameter the gateway does not understand is refused
rather than ignored.
"""

_MESSAGE_KEYS = frozenset({"type", "role", "content", "id", "status"})
_FUNCTION_CALL_KEYS = frozenset({"type", "call_id", "name", "arguments", "id", "status"})
_FUNCTION_CALL_OUTPUT_KEYS = frozenset({"type", "call_id", "output", "id", "status"})
_INPUT_TEXT_PART_KEYS = frozenset({"type", "text"})
_OUTPUT_TEXT_PART_KEYS = frozenset({"type", "text", "annotations"})
_INPUT_IMAGE_PART_KEYS = frozenset({"type", "image_url"})
_NAMED_CHOICE_KEYS = frozenset({"type", "function"})
_NAMED_CHOICE_FUNCTION_KEYS = frozenset({"name"})
_TOOL_KEYS = frozenset({"type", "name", "description", "parameters", "strict"})
_RESPONSE_FORMAT_KEYS = frozenset({"type", "json_schema"})
_JSON_SCHEMA_KEYS = frozenset({"name", "schema", "strict"})

_NESTED_FEATURES: Mapping[str, str] = {
    "detail": "image_detail",
    "file_id": "file_input",
    "logprobs": "log_probabilities",
    "phase": "message_phase",
    "prompt_cache_breakpoint": "prompt_caching",
}
"""Nested keys the Responses format defines and this endpoint cannot carry.

A nested object's key set is otherwise closed and an unlisted key is malformed
input, exactly as the chat endpoint's nested objects are. These five are listed
separately because they are not malformed: each names a feature the Responses
format defines, so refusing one is a capability refusal rather than a complaint
about the request (PRD section 9.5).
"""

_IGNORED_ITEM_KEYS = ("id", "status")
"""Keys on an input item that are read and deliberately ignored.

Both are the Responses format's own item bookkeeping rather than answer
semantics: ``id`` is the upstream's item identity, and the gateway stamps its own
identity on the items it returns (PRD section 5.2), while ``status`` describes
the lifecycle of an item whose content is already complete. They are accepted so
that a caller can feed an answer's ``output`` array back as the next request's
``input``, which is the pattern the Responses API is built around, and neither
changes what the gateway sends upstream (PRD section 9.1).
"""


_WIRE_BOOLEAN = "boolean"
_WIRE_INTEGER = "integer"
_WIRE_NUMBER = "number"
_WIRE_STRING = "string"
_WIRE_OBJECT = "object"
_WIRE_ARRAY = "array"
_WIRE_NULL = "null"

_WIRE_TYPE_PHRASES = {
    _WIRE_BOOLEAN: "a boolean",
    _WIRE_INTEGER: "an integer",
    _WIRE_NUMBER: "a number",
    _WIRE_STRING: "a string",
    _WIRE_OBJECT: "a JSON object",
    _WIRE_ARRAY: "a JSON array",
    _WIRE_NULL: "null",
}
_WIRE_TYPE_ORDER = (
    _WIRE_BOOLEAN,
    _WIRE_INTEGER,
    _WIRE_NUMBER,
    _WIRE_STRING,
    _WIRE_OBJECT,
    _WIRE_ARRAY,
    _WIRE_NULL,
)


class _Wire(NamedTuple):
    """The wire contract of one recognised parameter.

    ``accepts`` names the JSON types the value may carry, and ``default`` is
    the one correctly-typed value that asks for nothing, because it is the one
    this gateway already behaves like. A value of any other type is malformed
    input; a correctly-typed value that is not the default asks for the
    feature. The contract is stated on the parameter before the default-
    equivalence test runs, so the test never sees a value of the wrong type —
    a boolean is not an integer because Python's ``True == 1`` says it is, and
    an empty string is not the empty object a falsiness test would accept.
    """

    accepts: frozenset[str]
    default: object


class _Feature(NamedTuple):
    """A Responses parameter the gateway recognises and cannot honour.

    ``capability`` names what the parameter asks for, so the refusal says which
    feature is missing rather than only which key was sent. ``wire`` states the
    parameter's wire contract: which JSON types its value may carry, and which
    correctly-typed value is the one this gateway already behaves like, so
    refusing that value would reject a request the gateway can serve in full.
    """

    capability: str
    wire: _Wire


_UNSUPPORTED_FEATURES: Mapping[str, _Feature] = {
    "background": _Feature(
        "background_responses", _Wire(frozenset({_WIRE_BOOLEAN}), False)
    ),
    "context_management": _Feature(
        "context_management", _Wire(frozenset({_WIRE_ARRAY, _WIRE_NULL}), None)
    ),
    "conversation": _Feature(
        "conversation_state",
        _Wire(frozenset({_WIRE_STRING, _WIRE_OBJECT, _WIRE_NULL}), None),
    ),
    "include": _Feature("included_response_data", _Wire(frozenset({_WIRE_ARRAY}), [])),
    "max_tool_calls": _Feature(
        "tool_call_limit", _Wire(frozenset({_WIRE_INTEGER, _WIRE_NULL}), None)
    ),
    "moderation": _Feature(
        "output_moderation", _Wire(frozenset({_WIRE_OBJECT, _WIRE_NULL}), None)
    ),
    # Refused whatever its value, unlike the defaults above. ``true`` asks for
    # parallel tool calls and ``false`` asks for a guarantee of sequential ones,
    # and this slice implements neither and models no distinction between them,
    # so accepting the parameter would advertise a capability the gateway does
    # not have (PRD section 9.5). The chat endpoint refuses it for the same
    # reason, and the two endpoints must agree about it.
    "parallel_tool_calls": _Feature(
        "parallel_tool_calls", _Wire(frozenset({_WIRE_BOOLEAN, _WIRE_NULL}), None)
    ),
    "previous_response_id": _Feature(
        "response_chaining", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    "prompt": _Feature(
        "prompt_templates", _Wire(frozenset({_WIRE_OBJECT, _WIRE_NULL}), None)
    ),
    "prompt_cache_key": _Feature(
        "prompt_caching", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    "prompt_cache_options": _Feature(
        "prompt_caching", _Wire(frozenset({_WIRE_OBJECT, _WIRE_NULL}), None)
    ),
    "prompt_cache_retention": _Feature(
        "prompt_caching", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    "reasoning": _Feature(
        "reasoning_effort", _Wire(frozenset({_WIRE_OBJECT, _WIRE_NULL}), None)
    ),
    "safety_identifier": _Feature(
        "safety_identifier", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    "service_tier": _Feature(
        "service_tier", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    # ``store: false`` is accepted because it is the one value that agrees with
    # what the gateway does — it stores nothing — while ``true`` asks for the
    # Responses format's own default of a retained response, which the gateway
    # cannot provide (PRD section 9.5).
    "store": _Feature("response_storage", _Wire(frozenset({_WIRE_BOOLEAN}), False)),
    "stream_options": _Feature("streaming", _Wire(frozenset({_WIRE_OBJECT}), {})),
    "text": _Feature(
        "text_configuration", _Wire(frozenset({_WIRE_OBJECT, _WIRE_NULL}), None)
    ),
    "top_logprobs": _Feature(
        "log_probabilities", _Wire(frozenset({_WIRE_INTEGER, _WIRE_NULL}), None)
    ),
    "truncation": _Feature(
        "context_truncation", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)
    ),
    "user": _Feature("actor_identity", _Wire(frozenset({_WIRE_STRING, _WIRE_NULL}), None)),
}
"""Responses request parameters this endpoint does not carry.

Every entry is a parameter the Responses format defines, beside the documented
subset this endpoint accepts, whose feature the canonical invocation has no way
to ask for, or no way to ask for *here*. None of them is dropped, because a
caller told its request succeeded while a parameter was discarded has been
misled about what the model was asked (PRD section 9.5). Each entry's wire
contract states the JSON types the value may carry and the one correctly-typed
value that asks for nothing; a value of any other type is malformed input,
refused as an invalid request naming the parameter and the type supplied.
"""


def _wire_kind(value: object) -> str:
    """The JSON type of ``value``, as the wire contracts name them.

    ``bool`` is a subclass of ``int``, so a boolean is named before an integer
    is tested for: the two are different JSON types, and a wire contract that
    accepts one does not accept the other whatever Python's ``True == 1`` says.
    """
    if value is None:
        return _WIRE_NULL
    if isinstance(value, bool):
        return _WIRE_BOOLEAN
    if isinstance(value, int):
        return _WIRE_INTEGER
    if isinstance(value, float):
        return _WIRE_NUMBER
    if isinstance(value, str):
        return _WIRE_STRING
    if isinstance(value, Mapping):
        return _WIRE_OBJECT
    return _WIRE_ARRAY


def _wire_prose(accepts: frozenset[str]) -> str:
    """The accepted types of a wire contract as prose: ``a string or null``."""
    names = [_WIRE_TYPE_PHRASES[kind] for kind in _WIRE_TYPE_ORDER if kind in accepts]
    return names[0] if len(names) == 1 else " or ".join(names)


def _is_default(value: object, default: object) -> bool:
    """Whether a correctly-typed value is the one that asks for nothing.

    The wire contract has already established the value's type, so equality
    against the default is the whole test; ``True == 1`` cannot arise, because
    no wire contract accepts a boolean where the default is an integer.
    """
    if default is None:
        return value is None
    return value == default


@dataclass(frozen=True)
class ResponsesRequest:
    """A parsed ``POST /v1/responses`` request (PRD section 4.5).

    The parse is the gateway's, not the wire format's: by the time a caller has
    one of these, every field has a canonical home and every rule the canonical
    types enforce has already been applied, so a request the gateway cannot
    represent is refused before any provider is called.

    ``stream`` is carried rather than acted on. This endpoint refuses streaming
    (see the module docstring), and the refusal belongs to the endpoint, which
    knows what it can serve; this type records what the caller asked for so that
    the refusal can name it.

    The invocation carries no tenant from the body. This endpoint's documented
    subset has no tenant field, so the tenant is the one bound to the presented
    credential and nothing else (PRD section 5.1).
    """

    invocation: InvocationRequest
    stream: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        """Parse a decoded JSON request body, refusing anything it cannot carry.

        A ``ValueError`` from the canonical types is a validation failure of this
        request and becomes an invalid-request error carrying the domain's own
        gateway-authored message, which names the field that failed.
        """
        where = "the request body"
        try:
            body = as_object(where, payload)
        except ValueError as error:
            raise InvalidArgumentError(f"invalid responses request: {error}") from error
        _reject_unsupported_parameters(body)
        try:
            invocation = InvocationRequest(
                model=as_required_text(f"{where} 'model'", body.get("model")),
                messages=_as_messages(f"{where} 'input'", body.get("input")),
                system_instruction=as_optional_text(
                    f"{where} 'instructions'", body.get("instructions")
                ),
                generation_config=_generation_config(where, body),
                tools=_as_tools(f"{where} 'tools'", body.get("tools")),
                tool_choice=_tool_choice_from(f"{where} 'tool_choice'", body.get("tool_choice")),
                metadata=_metadata_from(f"{where} 'metadata'", body.get("metadata")),
                idempotency_key=_as_idempotency_key(
                    f"{where} 'idempotency_key'", body.get("idempotency_key")
                ),
            )
        except ValueError as error:
            raise InvalidArgumentError(f"invalid responses request: {error}") from error
        return cls(invocation=invocation, stream=_as_flag(f"{where} 'stream'", body.get("stream")))

    def to_invocation(
        self, *, invocation_id: str, identity: Identity | None = None
    ) -> InvocationRequest:
        """The canonical invocation, carrying the id minted for this request.

        ``response_modality`` stays text: this endpoint produces chat text, tool
        calls and refusals, and an image request has no representation in its
        response shape.

        ``identity`` is the trusted identity assigned by the authentication
        middleware, and the invocation's ``tenant_id`` is that identity's tenant.
        There is no body tenant to reconcile against, because the documented
        subset carries no tenant field: when there is no trusted identity, no
        tenant is propagated at all (PRD section 5.1).
        """
        tenant = identity.tenant if identity is not None and identity.authenticated else None
        return replace(self.invocation, invocation_id=invocation_id, tenant_id=tenant)


@dataclass(frozen=True)
class ResponseObject:
    """An OpenAI Responses object (PRD section 4.5).

    ``output`` is the ordered array of items the answer produced, rendered from
    the canonical output sequence content-first: one ``message`` item carrying
    the turn's content parts in the order the model emitted them, then one
    ``function_call`` item per tool call, so the order the representation can
    express is ``(ContentPart)* (ToolCall)*`` and an arbitrary interleaving of
    content and tool calls is not representable.

    The payload is the documented subset of the Responses object this endpoint
    can state truthfully: its identity, its creation time, the public model ID,
    its status, its output and its usage. The gateway's own accounting — the
    upstream model, the attempt count, fallback use, cost — is deliberately not
    in it, because the Responses object has no field for it and adding one would
    be a contract the gateway invented (PRD sections 5.2 and 13).
    ``idempotency_status`` is the one piece of it the shape has a home for: it is
    rendered beside ``status`` so a caller reading the answer can tell whether
    the provider was reached (PRD section 12). The fields the
    Responses object carries that describe the *request* — ``instructions``,
    ``tools``, ``temperature`` — are left out for the same reason the chat
    endpoint leaves out its accounting: echoing a request back is not part of
    answering it, and a field the gateway did not read from the provider is not
    something to present as the provider's answer.
    """

    id: str
    model: str
    idempotency_status: str
    content: tuple[OutputItem, ...]
    usage: Usage
    created_at: int
    status: str = _STATUS_COMPLETED
    object: str = _OBJECT_RESPONSE

    @classmethod
    def from_invocation(
        cls,
        response: InvocationResponse,
        *,
        response_id: str,
        created_at: int,
    ) -> Self:
        """Translate a canonical response into the Responses object.

        Two canonical responses have no representation here and are refused
        rather than approximated: one carrying an image, because no ``message``
        content variant can hold one, and one whose content continues after a
        tool call, because a Responses ``output`` is one assistant turn and a
        turn that calls a function ends there. Both refusals are capability
        errors, so a caller learns which feature is missing instead of receiving
        an answer that is not the one the model gave (PRD sections 9.1 and 9.5).
        """
        _reject_unrepresentable_output(response)
        return cls(
            id=response_id,
            model=response.model,
            idempotency_status=response.idempotency_status.value,
            content=response.content,
            usage=response.usage,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "model": self.model,
            "status": self.status,
            "idempotency_status": self.idempotency_status,
            "output": _output_items(self.content, response_id=self.id),
            "usage": _usage_payload(self.usage),
        }


def create_response(service: InferenceService) -> ResponsesHandler:
    """Build the handler for ``POST /v1/responses`` (PRD section 4.5)."""

    async def responses(request: Request) -> Response:
        parsed = ResponsesRequest.from_payload(await _read_json_body(request))
        if parsed.stream:
            raise _streaming_refusal()

        identity = get_identity(request)
        invocation_id = uuid4().hex
        invocation = parsed.to_invocation(invocation_id=invocation_id, identity=identity)

        answer = await service.invoke(
            invocation,
            namespace=caller_namespace(identity),
            tenant=caller_tenant(identity),
        )
        return json(
            ResponseObject.from_invocation(
                answer,
                response_id=f"{_RESPONSE_ID_PREFIX}{answer.invocation_id or invocation_id}",
                created_at=int(time.time()),
            ).to_dict()
        )

    return responses


def register_responses_routes(router: Router, service: InferenceService) -> None:
    """Add the Responses endpoint to ``router``."""
    router.add_post("/v1/responses", create_response(service))


def _streaming_refusal() -> CapabilityError:
    """The refusal for ``stream: true`` (PRD sections 4.5 and 16).

    The refusal names ``streaming`` and says which protocol is missing, because
    the feature the caller asked for is the Responses streaming protocol rather
    than streaming as such: the chat endpoint streams. It is raised before the
    model is resolved and before any provider is called, because it is decided
    from the request alone — it is a property of this endpoint's own contract,
    not of any provider.
    """
    return CapabilityError(
        "this endpoint does not serve the Responses streaming event protocol: it is an "
        "item-lifecycle protocol whose events carry item ids, output and content indices and "
        "whole-item boundaries, and the canonical stream carries none of them, so the events "
        "could only be produced by inventing event types; the gateway will not answer a "
        "streaming request with a complete, non-streamed body either, so 'stream: true' is "
        "refused and only the non-streaming Responses representation is served",
        details={"capability": _STREAMING_CAPABILITY, "parameter": "stream"},
    )


async def _read_json_body(request: Request) -> object:
    """Decode the request body, refusing anything that is not a JSON document.

    BlackSheep raises its own ``BadRequestFormat`` when the declared content type
    is JSON and the body is not, and returns ``None`` when the content type never
    declared JSON at all. Both are malformed input for this endpoint, and both
    become an invalid-request error carrying the taxonomy's reason rather than
    the framework's plain-text body, so every failure on this path answers in one
    shape (PRD section 5.4).
    """
    try:
        payload = await request.json()
    except HTTPException as error:
        raise InvalidArgumentError(
            "the request body must be valid JSON and the Content-Type header must declare it"
        ) from error
    if payload is None:
        raise InvalidArgumentError(
            "the request body must be a JSON object sent with Content-Type: application/json"
        )
    return payload


def _reject_unsupported_parameters(body: Mapping[str, object]) -> None:
    """Refuse a parameter the gateway cannot honour, naming what it asked for.

    A key the Responses format does not define is malformed input. A key it does
    define is held to its wire contract before anything else: a value of the
    wrong JSON type is an invalid-request error naming the parameter and the
    type actually supplied, because the default-equivalence test that decides
    whether the parameter was actually requested runs on a validated value. A
    correctly-typed value that is not the one this gateway already behaves like
    asks for something this endpoint has no representation for, and is a
    capability the route cannot provide — the request is well formed and the
    refusal is about the feature (PRD section 9.5).
    """
    for name in sorted(set(body) - _REQUEST_KEYS):
        feature = _UNSUPPORTED_FEATURES.get(name)
        if feature is None:
            supported = ", ".join(repr(key) for key in sorted(_REQUEST_KEYS))
            raise InvalidArgumentError(
                f"the request body has unsupported parameter {name!r}; supported parameters are "
                f"{supported}"
            )
        value = body[name]
        if _wire_kind(value) not in feature.wire.accepts:
            raise InvalidArgumentError(
                f"the request body parameter {name!r} must be {_wire_prose(feature.wire.accepts)}, "
                f"got {describe(value)}"
            )
        if _is_default(value, feature.wire.default):
            continue
        raise CapabilityError(
            f"the gateway cannot honour {name!r}; it would request {feature.capability!r}, which "
            f"this endpoint's documented subset does not carry and the gateway will not silently "
            f"discard",
            details={"capability": feature.capability, "parameter": name},
        )


def _reject_nested_keys(where: str, body: Mapping[str, object], supported: frozenset[str]) -> None:
    """Refuse a nested key, as a capability gap when the key names a feature.

    A nested object's key set is closed, exactly as the chat endpoint's are. A
    key the Responses format defines is refused as a capability the endpoint does
    not have rather than as malformed input, because the request is well formed
    and the refusal is about the feature (PRD section 9.5).
    """
    for key in sorted(set(body) - supported):
        feature = _NESTED_FEATURES.get(key)
        if feature is not None:
            raise CapabilityError(
                f"{where} carries {key!r}; it would request {feature!r}, which the gateway will "
                f"not silently discard",
                details={"capability": feature, "parameter": key},
            )
        names = ", ".join(repr(name) for name in sorted(supported))
        raise InvalidArgumentError(
            f"{where} has unsupported key {key!r}; supported keys are {names}"
        )


def _as_object(where: str, value: object) -> Mapping[str, object]:
    """A JSON object, as an invalid-request error when it is not one."""
    try:
        return as_object(where, value)
    except ValueError as error:
        raise InvalidArgumentError(str(error)) from error


def _as_flag(where: str, value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise InvalidArgumentError(f"{where} must be a boolean, got {describe(value)}")
    return value


def _as_messages(where: str, value: object) -> tuple[Message, ...]:
    """The canonical conversation from a Responses ``input`` (PRD section 4.5).

    A string is the one user message it spells. An array is an ordered sequence
    of input items, and their order is the conversation's order, so the items are
    mapped in the order they were sent rather than grouped by kind.
    """
    if isinstance(value, str):
        return (Message(role=Role.USER, content=(TextPart(value),)),)
    if value is None:
        raise InvalidArgumentError(
            f"{where} is required: this endpoint carries no response chaining, so every "
            f"invocation states the input it is about"
        )
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be a string or an array of input items, got {describe(value)}"
        )
    messages: list[Message] = []
    for index, item in enumerate(value):
        messages.extend(_messages_from_item(f"{where}[{index}]", item))
    if not messages:
        raise InvalidArgumentError(f"{where} must carry at least one input item")
    return tuple(messages)


def _messages_from_item(where: str, payload: object) -> tuple[Message, ...]:
    """One ``input`` array entry as canonical messages.

    A message-like object is a message. The two tool-turn items are messages too:
    a ``function_call`` is an assistant message proposing the call, and a
    ``function_call_output`` is the tool message answering it. Any other item
    type is a feature this endpoint does not carry and is refused naming it,
    rather than skipped, because a dropped item changes the conversation the
    model sees (PRD sections 4.5 and 9.1).
    """
    body = _as_object(where, payload)
    kind = body.get("type")
    if kind is None or kind == _INPUT_MESSAGE:
        _reject_nested_keys(where, body, _MESSAGE_KEYS)
        return (_message_from(where, body),)
    if kind == _INPUT_FUNCTION_CALL:
        _reject_nested_keys(where, body, _FUNCTION_CALL_KEYS)
        return (_function_call_message(where, body),)
    if kind == _INPUT_FUNCTION_CALL_OUTPUT:
        _reject_nested_keys(where, body, _FUNCTION_CALL_OUTPUT_KEYS)
        return (_function_call_output_message(where, body),)
    if not isinstance(kind, str):
        raise InvalidArgumentError(
            f"{where} must carry a string 'type' discriminator, got {describe(kind)}"
        )
    raise CapabilityError(
        f"the gateway cannot carry an input item of type {kind!r}: it maps the Responses input "
        f"message, 'function_call' and 'function_call_output' items onto the canonical "
        f"conversation and has no representation for anything else, so the item is refused "
        f"rather than dropped",
        details={"capability": kind, "parameter": "input[].type"},
    )


def _message_from(where: str, body: Mapping[str, object]) -> Message:
    try:
        return Message(
            role=_role(f"{where} 'role'", body.get("role")),
            content=_content_from(f"{where} 'content'", body.get("content")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid responses input message: {error}") from error


def _role(where: str, value: object) -> Role:
    """The role of an input message, refusing the one the shape cannot carry.

    A ``tool`` role is refused with its own explanation rather than the canonical
    type's: the Responses input message carries no tool-call identifier, so a
    tool result belongs to a ``function_call_output`` item, which names the call
    it answers. A message the gateway cannot match to a call is not a
    conversation it can send (PRD section 9.3).
    """
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {describe(value)}")
    try:
        role = Role(value)
    except ValueError as error:
        names = ", ".join(repr(member.value) for member in Role)
        raise InvalidArgumentError(f"{where} must be one of {names}; got {value!r}") from error
    if role is Role.TOOL:
        raise InvalidArgumentError(
            f"{where} is {value!r}; a tool result is carried by a 'function_call_output' input "
            f"item, which names the call it answers, and an input message carries no tool-call "
            f"identifier"
        )
    return role


def _content_from(where: str, value: object) -> tuple[TextPart | ImagePart, ...]:
    if isinstance(value, str):
        return (TextPart(value),)
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be a string or an array of content parts, got {describe(value)}"
        )
    return tuple(_content_part_from(f"{where}[{index}]", item) for index, item in enumerate(value))


def _content_part_from(where: str, payload: object) -> TextPart | ImagePart:
    """One input content part as canonical content.

    ``input_text`` is what a caller writes and ``output_text`` is what an
    assistant turn the caller is replaying contained; both are text and both
    become a canonical text part. ``input_image`` is a caller-supplied image
    and becomes a canonical image part, which the route must declare the
    ``vision`` capability to carry. Anything else is refused rather than
    approximated: a part type this endpoint does not carry is a feature the
    gateway has no representation for, and it is named so the caller learns
    what is missing instead of sending a message the model never saw (PRD
    sections 9.1 and 9.5).
    """
    body = _as_object(where, payload)
    kind = body.get("type")
    if kind == _INPUT_TEXT:
        _reject_nested_keys(where, body, _INPUT_TEXT_PART_KEYS)
        return _text_part_from(f"{where} 'text'", body.get("text"))
    if kind == _INPUT_OUTPUT_TEXT:
        _reject_nested_keys(where, body, _OUTPUT_TEXT_PART_KEYS)
        _reject_annotations(where, body.get("annotations"))
        return _text_part_from(f"{where} 'text'", body.get("text"))
    if kind == _INPUT_IMAGE:
        _reject_nested_keys(where, body, _INPUT_IMAGE_PART_KEYS)
        return _image_part_from(f"{where} 'image_url'", body.get("image_url"))
    if not isinstance(kind, str):
        raise InvalidArgumentError(
            f"{where} must carry a string 'type' discriminator, got {describe(kind)}"
        )
    raise CapabilityError(
        f"{where} has type {kind!r}; this endpoint carries 'input_text', 'output_text' and "
        f"'input_image' content parts and has no representation for {kind!r}, so the part is "
        f"refused rather than dropped",
        details={"capability": kind, "parameter": "input[].content[].type"},
    )


def _image_part_from(where: str, value: object) -> ImagePart:
    """An ``input_image`` part's ``image_url`` as a canonical image part.

    A data URI carries the base64 payload inline, so it becomes image content
    with its media type and needs no fetch, no policy decision and no network —
    exactly as the chat endpoint's ``image_url`` part carries it. A remote
    ``http`` or ``https`` URL is refused with a capability error: the canonical
    type documents its URL as one the URL policy has admitted, and admitting it
    means resolving the hostname and rejecting loopback, private, link-local and
    metadata addresses — a blocking DNS lookup this async handler must not
    perform, and one that would only screen the URL, not pin the connection or
    evaluate redirects as PRD section 14.1 requires. Remote image fetching
    arrives with that safe asynchronous validation and transport path; until
    then only inline content is accepted. Any other scheme, ``file`` included,
    is malformed input rather than a missing feature.
    """
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(f"{where} must be a non-empty string, got {describe(value)}")
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


def _text_part_from(where: str, value: object) -> TextPart:
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {describe(value)}")
    return TextPart(value)


def _reject_annotations(where: str, value: object) -> None:
    """Refuse a non-empty ``annotations`` list on a replayed output text part.

    An empty list is the absence of annotations, which is what this gateway's own
    output carries, so a caller can feed an answer back as the next request's
    input. A populated one carries citations — a URL the provider consulted, a
    file it read — which the canonical content representation has no field for,
    and a text part returned without them would report the model's words while
    losing what they pointed at (PRD section 9.1).
    """
    if not value:
        return
    raise CapabilityError(
        f"{where} carries annotations; the canonical content representation has no field for a "
        f"citation, so a replayed output text part carrying one cannot be forwarded and the "
        f"gateway will not silently discard it",
        details={"capability": _ANNOTATIONS_CAPABILITY, "parameter": "annotations"},
    )


def _function_call_message(where: str, body: Mapping[str, object]) -> Message:
    """A ``function_call`` item as the assistant message that proposed it."""
    try:
        call = ToolCall(
            id=as_required_text(f"{where} 'call_id'", body.get("call_id")),
            name=as_required_text(f"{where} 'name'", body.get("name")),
            arguments=_arguments_from(f"{where} 'arguments'", body.get("arguments")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid responses input item: {error}") from error
    return Message(role=Role.ASSISTANT, tool_calls=(call,))


def _function_call_output_message(where: str, body: Mapping[str, object]) -> Message:
    """A ``function_call_output`` item as the tool message that answers it."""
    call_id = _required_text(f"{where} 'call_id'", body.get("call_id"))
    output = body.get("output")
    if output is None:
        raise InvalidArgumentError(
            f"{where} 'output' is required: a tool result that carries nothing cannot be told "
            f"from a tool result that was never sent"
        )
    return Message(
        role=Role.TOOL,
        content=_content_from(f"{where} 'output'", output),
        tool_call_id=call_id,
    )


def _required_text(where: str, value: object) -> str:
    try:
        return as_required_text(where, value)
    except ValueError as error:
        raise InvalidArgumentError(str(error)) from error


def _as_idempotency_key(where: str, value: object) -> str | None:
    """The caller's idempotency key, or ``None`` when the request carries none.

    ``None`` is an optional parameter's wire default and asks for nothing. Any
    other value must be a non-empty string: the format defines the key as one, so
    a value of another type is malformed input naming the type that arrived, and
    an empty string is no key at all rather than a key that happens to be empty
    (PRD section 12).
    """
    if value is None:
        return None
    return as_required_text(where, value)


def _arguments_from(where: str, value: object) -> Mapping[str, JsonValue] | str | None:
    """Tool arguments, parsed when they parse and kept verbatim when they do not.

    The Responses format encodes arguments as a JSON string, exactly as the chat
    format does. Storing the parsed object when it is one keeps the canonical
    representation useful, and keeping the raw string when it is not means a
    malformed argument payload is relayed rather than repaired into an empty
    object the caller never sent (PRD section 9.3).
    """
    if value is None or isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a JSON-encoded string, got {describe(value)}")
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, Mapping) else value


def _as_tools(where: str, value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidArgumentError(
            f"{where} must be an array of tool declarations, got {describe(value)}"
        )
    return tuple(_tool_from(f"{where}[{index}]", item) for index, item in enumerate(value))


def _tool_from(where: str, payload: object) -> ToolDefinition:
    """One Responses function tool as a canonical tool definition.

    The Responses tool is flat — ``type``, ``name``, ``description``,
    ``parameters`` — rather than the chat format's nested ``function`` object,
    and the canonical definition is built from the flat form. A hosted tool is
    refused naming its type, because the gateway relays function tools and hosts
    none (PRD section 9.3), and ``strict`` is refused when it asks for schema
    enforcement the canonical declaration cannot request.
    """
    body = _as_object(where, payload)
    kind = body.get("type")
    if kind != _TYPE_FUNCTION:
        if not isinstance(kind, str):
            raise InvalidArgumentError(f"{where} 'type' must be 'function', got {describe(kind)}")
        raise CapabilityError(
            f"the gateway cannot honour a tool of type {kind!r}: it relays function tools to the "
            f"provider and hosts no tool of its own, so a hosted tool has no representation to "
            f"be forwarded in and is refused rather than dropped",
            details={"capability": kind, "parameter": "tools[].type"},
        )
    _reject_nested_keys(where, body, _TOOL_KEYS)
    if body.get("strict"):
        raise CapabilityError(
            f"{where} asks for 'strict'; the canonical tool declaration carries no field for "
            f"schema enforcement, so the guarantee cannot be forwarded and the gateway will not "
            f"report a tool as enforced when nothing asked the provider to enforce it",
            details={"capability": _STRICT_TOOLS_CAPABILITY, "parameter": "strict"},
        )
    description = body.get("description")
    parameters = body.get("parameters")
    try:
        return ToolDefinition(
            name=as_required_text(f"{where} 'name'", body.get("name")),
            description="" if description is None else description,
            parameters={} if parameters is None else parameters,
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid responses tool declaration: {error}") from error


def _tool_choice_from(where: str, value: object) -> ToolChoice | None:
    """A Responses ``tool_choice`` as the canonical tool-choice policy.

    The Responses format carries the choice as a bare string naming a mode or
    as an object naming one function, and both shapes map onto the canonical
    policy: the chat endpoint accepts the same two and the binding forwards the
    policy upstream, so accepting it here keeps the two endpoints in agreement
    about what a caller may ask the model to do with its tools (PRD section
    9.3).
    """
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
    _reject_nested_keys(where, body, _NAMED_CHOICE_KEYS)
    if body.get("type") != _TYPE_FUNCTION:
        raise InvalidArgumentError(
            f"{where} 'type' must be 'function', got {describe(body.get('type'))}"
        )
    function = _as_object(f"{where} 'function'", body.get("function"))
    _reject_nested_keys(f"{where} 'function'", function, _NAMED_CHOICE_FUNCTION_KEYS)
    return ToolChoice.specific(
        _required_text(f"{where} 'function' 'name'", function.get("name"))
    )


def _metadata_from(where: str, value: object) -> Mapping[str, JsonValue]:
    """A Responses ``metadata`` object as the canonical request metadata.

    PRD section 5.1 defines ``metadata`` as request metadata, and the canonical
    request carries it as arbitrary JSON held as an immutable mapping, so the
    value is forwarded to the canonical type and nothing is invented here. The
    canonical type's own rules bound it and are enforced there rather than
    restated: a key naming a canonical invocation field or a generation
    parameter is refused, because metadata is never forwarded to a model or a
    provider and a model parameter parked there would be a semantic the gateway
    neither represents nor sends (PRD section 9.1).
    """
    if value is None:
        return {}
    return _as_object(where, value)


def _generation_config(where: str, body: Mapping[str, object]) -> GenerationConfig:
    """Map the Responses sampling parameters onto the canonical generation config.

    ``max_output_tokens`` is the canonical field's own name, and ``temperature``
    and ``top_p`` map onto themselves. Every parameter the caller omitted stays
    unset, so the provider's own default applies rather than a value the
    gateway invented (PRD section 5.1).
    """
    try:
        return GenerationConfig(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_output_tokens=body.get("max_output_tokens"),
            response_format=_as_response_format(
                f"{where} 'response_format'", body.get("response_format")
            ),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid responses request: {error}") from error


def _as_response_format(where: str, value: object) -> ResponseFormat | None:
    """A structured-output requirement in the shape the chat endpoint accepts.

    PRD section 4.5 gives this endpoint ``response_format``, and the shape it
    takes is the one ``POST /v1/chat/completions`` takes, so the gateway's
    structured-output contract reads the same on both endpoints. The requirement
    itself is enforced at the gateway core against the primary route's advertised
    capabilities (PRD section 9.5), never here.
    """
    if value is None:
        return None
    body = _as_object(where, value)
    _reject_nested_keys(where, body, _RESPONSE_FORMAT_KEYS)
    kind = body.get("type")
    if kind == ResponseFormatMode.TEXT.value:
        return ResponseFormat(mode=ResponseFormatMode.TEXT)
    if kind == ResponseFormatMode.JSON_OBJECT.value:
        return ResponseFormat(mode=ResponseFormatMode.JSON_OBJECT)
    if kind == ResponseFormatMode.JSON_SCHEMA.value:
        schema = _as_object(f"{where} 'json_schema'", body.get("json_schema"))
        _reject_nested_keys(f"{where} 'json_schema'", schema, _JSON_SCHEMA_KEYS)
        try:
            return ResponseFormat(
                mode=ResponseFormatMode.JSON_SCHEMA,
                schema=_schema_document(f"{where} 'json_schema' 'schema'", schema.get("schema")),
                name=as_optional_text(f"{where} 'json_schema' 'name'", schema.get("name")),
                strict=_as_flag(f"{where} 'json_schema' 'strict'", schema.get("strict")),
            )
        except ValueError as error:
            raise InvalidArgumentError(f"invalid responses request: {error}") from error
    if not isinstance(kind, str):
        raise InvalidArgumentError(
            f"{where} must carry a string 'type' discriminator, got {describe(kind)}"
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


def _reject_unrepresentable_output(response: InvocationResponse) -> None:
    """Refuse output the Responses object cannot carry (PRD sections 9.1 and 9.5).

    Two cases, and they are refused for different reasons. An image has no home
    at all: a Responses ``message`` item's content holds output text and
    refusals, and nothing else. Content that continues after a tool call has no
    home *as one turn*: the Responses ``output`` array is one assistant turn's
    items, and a turn that calls a function ends there, so the content that
    followed the call would have to become a second ``message`` item — presenting
    one turn as two — or be joined to the first, which would report "said did
    said" as "said said, did". Neither is the answer the model gave.
    """
    if response.images:
        raise CapabilityError(
            "the Responses representation cannot carry image output: a 'message' item's content "
            "holds output text and refusals, and the gateway does not return a partial answer; "
            "the model produced an image",
            details={"capability": _IMAGE_OUTPUT_CAPABILITY},
        )
    seen_tool_call = False
    for item in response.content:
        if isinstance(item, ToolCall):
            seen_tool_call = True
        elif seen_tool_call:
            raise CapabilityError(
                "the Responses representation cannot carry a response whose content continues "
                "after a tool call: one 'output' array is one assistant turn and a turn that "
                "calls a function ends there, so the answer would have to be split into two "
                "messages or flattened into a different answer; the model emitted content after "
                "a tool call",
                details={"capability": _INTERLEAVED_OUTPUT_CAPABILITY},
            )


def _output_items(content: tuple[OutputItem, ...], *, response_id: str) -> list[dict[str, object]]:
    """The ordered ``output`` array for one canonical answer (PRD section 4.5).

    The rendering is content-first: every content part becomes the content list
    of one ``message`` item, in the order the model emitted them, and then every
    tool call becomes a ``function_call`` item. The order the representation
    can express is therefore ``(ContentPart)* (ToolCall)*``, and an arbitrary
    interleaving of content and tool calls is not representable — the one
    interleaving that would break the partition, content after a tool call, is
    refused by :func:`_reject_unrepresentable_output` before this runs, so the
    partition loses no ordering the representation could have carried. A
    tool-call-only answer has no ``message`` item at all, exactly as the chat
    endpoint answers it with no content.
    """
    content_parts = tuple(item for item in content if not isinstance(item, ToolCall))
    calls = tuple(item for item in content if isinstance(item, ToolCall))
    items: list[dict[str, object]] = []
    if content_parts:
        items.append(_message_item(content_parts, response_id=response_id))
    items.extend(
        _function_call_item(call, response_id=response_id, index=index)
        for index, call in enumerate(calls)
    )
    return items


def _message_item(
    parts: tuple[TextPart | RefusalPart | ImagePart, ...], *, response_id: str
) -> dict[str, object]:
    """One ``message`` item: the assistant turn's content, in order.

    A refusal keeps its own variant rather than being folded into text, because a
    provider declining on safety grounds is a different event from a model
    answering (PRD section 5.2), and the Responses content union has a variant
    for each.
    """
    return {
        "id": f"{_MESSAGE_ID_PREFIX}{response_id}",
        "type": _OBJECT_MESSAGE,
        "role": _ROLE_ASSISTANT,
        "status": _STATUS_COMPLETED,
        "content": [_content_payload(part) for part in parts],
    }


def _content_payload(part: TextPart | RefusalPart | ImagePart) -> dict[str, object]:
    """One content part in the Responses shape.

    ``annotations`` is emitted as an empty list because the Responses output text
    variant types it as a list and this gateway attaches no citation to the text
    it returns; the empty list is the absence of annotations rather than a claim
    that none were found, and an empty list is what this endpoint accepts back on
    a replayed ``output_text`` part.
    """
    if isinstance(part, RefusalPart):
        return {"type": _OBJECT_REFUSAL, "refusal": part.text}
    return {"type": _OBJECT_OUTPUT_TEXT, "text": part.text, "annotations": []}


def _function_call_item(call: ToolCall, *, response_id: str, index: int) -> dict[str, object]:
    """One ``function_call`` item.

    ``call_id`` is the canonical call's own identifier, which is what a caller
    matches a ``function_call_output`` against; ``id`` is the item's identity,
    which the gateway mints because the canonical call carries none and the
    Responses representation is the gateway's to stamp (PRD section 5.2).
    """
    return {
        "id": f"{_FUNCTION_CALL_ID_PREFIX}{response_id}_{index}",
        "type": _OBJECT_FUNCTION_CALL,
        "call_id": call.id,
        "name": call.name,
        "arguments": _arguments_payload(call.arguments),
        "status": _STATUS_COMPLETED,
    }


def _arguments_payload(arguments: Mapping[str, JsonValue] | str | None) -> str:
    """Tool arguments as the JSON-encoded string the Responses shape types.

    Canonical arguments are the parsed object when the provider's arguments
    parsed, the raw string when they did not, and ``None`` when the provider
    reported none. All three are reported as the string the shape expects, so an
    argument payload the gateway could not parse is returned to the caller
    unchanged rather than repaired into an empty object (PRD section 9.3).
    """
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return orjson.dumps(thaw_json(arguments)).decode("utf-8")


def _usage_payload(usage: Usage) -> dict[str, object]:
    """The Responses usage object, with its own field names.

    The three counts are the Responses names for the canonical counts. A count
    the provider never reported is presented as zero, because the Responses
    usage object types its counts as integers and has no null: the wire shape
    cannot preserve the canonical record's difference between "none were used"
    and "nobody counted", so the gateway reports zero rather than inventing an
    omission the shape has no field for, and a caller cannot recover from the
    response alone which of the two a zero is (PRD section 13). A total the
    provider omitted is summed from the two counts it did report, which is
    arithmetic the caller can repeat.

    The two detail objects are the categories the Responses usage shape carries.
    ``cached_tokens`` is the canonical record's own cached count. The other two —
    cache-write tokens and reasoning tokens — are categories the canonical usage
    model does not track at all, so the gateway has never observed one and
    reports zero for them rather than omitting a field the shape requires; a
    binding that met either would refuse the upstream response as unrepresentable
    rather than pass it on (PRD section 9.1).
    """
    prompt = 0 if usage.input_tokens is None else usage.input_tokens
    completion = 0 if usage.output_tokens is None else usage.output_tokens
    total = usage.total_tokens
    if total is None:
        total = prompt + completion
    cached = 0 if usage.cached_tokens is None else usage.cached_tokens
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
