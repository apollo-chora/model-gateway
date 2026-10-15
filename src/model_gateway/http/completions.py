"""The legacy completions endpoint: ``POST /v1/completions`` (PRD sections 4 and 16).

The handler is thin on purpose, exactly as the other endpoints' are. It decodes
the body, translates it into the canonical invocation, calls the inference
service and renders the canonical response back into the OpenAI shape. It
resolves no route, retries nothing and counts nothing: routing, fallback, retry
and accounting are the gateway core's (PRD sections 9.2 and 11.3), and a second
copy of that logic on the HTTP path is how the two transports drift apart.

**The accepted request is the documented subset, and nothing else is ignored.**
A legacy completion has no roles, so the prompt becomes a single canonical user
message; the sampling parameters (``max_tokens``, ``temperature``, ``top_p``,
``stop``) map onto the canonical generation config. A key the gateway does not
understand is refused with an invalid-request error naming it rather than
dropped, and a key it recognises but cannot honour is refused with a capability
error naming the feature (PRD section 4's compatibility boundary, applied to
this endpoint's own documented subset):

* ``n`` above one asks for several completions, and this endpoint answers one;
  ``n: 1`` is the gateway's own default and is accepted.
* ``logprobs`` asks for token log-probabilities, which the canonical usage record
  does not carry and the legacy response shape has no field for; ``logprobs:
  false`` is the upstream default and is accepted.
* ``echo`` asks for the prompt to be echoed back in the completion text, which
  the gateway never does; ``echo: false`` is the upstream default and is accepted.
* ``best_of`` above one asks for several samples with the best returned, which
  the gateway does not do; ``best_of: 1`` is the upstream default and is accepted.
* ``suffix`` asks the model to continue a suffix after the prompt, which the
  canonical request has no field for; ``suffix: null`` states nothing and is
  accepted, and a stated suffix is refused.
* ``idempotency_key`` is the caller's own deduplication key, mapped onto the
  invocation's so a repeat of the same request is answered from the gateway's
  record of the first answer rather than by asking the provider again. It is a
  plain string and optional; the response carries ``idempotency_status`` beside
  ``id`` and ``model`` so a caller can tell whether the provider was reached
  (PRD section 12).

**A recognised parameter's wire type is checked before its value is judged.**
Each parameter in that subset has one JSON type on the wire, and a value of
another type is malformed input rather than a feature request: ``{"n": true}``
is not ``n: 1`` — JSON's ``true`` is a boolean and not an integer, however
Python's ``bool`` subclasses ``int`` — and ``{"echo": 0}`` is not ``echo:
false``. Neither value is read as asking for nothing, and neither is refused as
a missing capability: a capability refusal would tell the caller the request was
understood when it was not. An ill-typed value is an invalid-request error
naming the parameter and the type that arrived (PRD sections 4 and 9.5).

**An array prompt is refused, not narrowed.** The OpenAI format accepts a
``prompt`` that is a string or an array of strings, and an array asks for one
completion *per element*. The canonical invocation carries one conversation and
the legacy shape answers one completion, so an array prompt has no
representation here: the gateway does not silently use the first element (which
would answer a request it never received) and does not join the elements (which
would send the model one prompt the caller never sent). The refusal is a
capability error naming ``multiple_prompts``, so a caller learns the shape
cannot carry what it asked for (PRD section 9.1).

**The response is the documented shape, and the whole answer must fit it.** The
OpenAI legacy completion carries one ``text`` per choice, so the answer is the
canonical output's text. A canonical answer can also carry tool calls, refusals
or images, and the legacy shape has no field for any of them: a caller told the
answer was delivered when part of it was dropped has been misled, so an answer
carrying any of those is refused with a capability error naming it rather than
returned without it (PRD sections 9.1 and 9.5). The gateway's own accounting —
the invocation identifier, the upstream model, the attempt count — is
deliberately not in this payload, because the legacy shape has no field for it
(PRD sections 5.2 and 13). ``idempotency_status`` is the one piece of it the
shape has a home for, and it is rendered beside ``id`` and ``model``.

**Streaming is served as Server-Sent Events in the legacy chunk shape.** A
request carrying ``stream: true`` is answered with an ``text/event-stream``
response whose chunks carry the fragment the event reports under
``choices[0].text``, and the stream is terminated by ``[DONE]`` only when it
ended with a ``finish`` event: a stream that ended with an ``error`` event, or
that ended with no terminal event at all, is closed without it, because the
answer did not complete (PRD section 16). The events come from the inference
service's own stream, so routing, retry, fallback, the deadline and the
no-fallback-after-first-output rule are the gateway core's, not duplicated
here. The Server-Sent Events response itself is the gateway's shared builder,
imported from the chat endpoint rather than reimplemented, so the two
transports' stream teardown — a client disconnect stops the upstream work —
cannot drift apart. A provider that cannot stream is refused with a capability
error naming ``streaming`` before the response starts, because answering a
streaming request with a complete, non-streamed body would tell the caller it
received something it did not (PRD sections 16 and 9.5).

**A streamed tool call ends the stream rather than being dropped.** The legacy
chunk shape has no place for a tool call, exactly as the non-streaming shape has
none, so the streaming path applies the same rule: a tool-call event ends the
stream with a terminal ``error`` event carrying the gateway's own reason and a
message naming ``tool_calls``, and no ``[DONE]`` follows, because the answer did
not complete. Before the first event has been delivered the stream has produced
nothing the caller could see, so a leading tool call is refused as an ordinary
error response rather than a stream, exactly as a pre-stream capability gap is.

The invocation id is minted here, per request, and carried into the canonical
request. It is what the response's ``id`` is built from and what correlates the
caller's completion with the gateway's own logs and usage records.

The request shape carries no tenant field — the legacy completions format has
none — so there is no claimed tenant to reconcile; the invocation's tenant comes
from the authenticated identity alone, exactly as on the other endpoints
(PRD section 5.1).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Self
from uuid import uuid4

import orjson
from blacksheep import HTTPException, Request, Response
from blacksheep.contents import ServerSentEvent, TextServerSentEvent
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.domain import (
    FinishReason,
    GenerationConfig,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    StreamEvent,
    StreamEventKind,
    TextPart,
    Usage,
)
from model_gateway.domain.serialization import as_required_text, describe
from model_gateway.errors import CapabilityError, ErrorReason, GatewayError, InvalidArgumentError
from model_gateway.http.auth import get_identity
from model_gateway.http.chat import _gateway_sse_response
from model_gateway.security.identity import caller_namespace, caller_tenant

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService
    from model_gateway.security.identity import Identity

__all__ = [
    "CompletionRequest",
    "TextCompletion",
    "TextCompletionChoice",
    "completion_request_from_payload",
    "create_completions",
    "register_completions_routes",
]

CompletionsHandler = Callable[[Request], Awaitable[Response]]

_OBJECT_TEXT_COMPLETION = "text_completion"
_COMPLETION_ID_PREFIX = "cmpl-"
_DONE = "[DONE]"

_REQUEST_KEYS = frozenset(
    {
        "model",
        "prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "n",
        "stream",
        "echo",
        "suffix",
        "best_of",
        "logprobs",
        "idempotency_key",
    }
)
"""The request fields ``POST /v1/completions`` recognises.

The set is closed on purpose, exactly as the other endpoints': PRD section 4
makes the gateway's OpenAI compatibility a supported subset rather than drop-in
compatibility, so a parameter the gateway does not understand is refused rather
than ignored. The set is this endpoint's documented subset — a legacy
completion carries no roles, no tools, no metadata and no tenant — and each
parameter in it is either honoured or refused with a capability error naming
the feature, never silently dropped.
"""

_FINISH_REASONS: Mapping[FinishReason, str] = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.TOOL_CALLS: "tool_calls",
    FinishReason.CONTENT_FILTER: "content_filter",
}

_IMAGE_OUTPUT_CAPABILITY = "image_output"
_TOOL_CALLS_CAPABILITY = "tool_calls"
_REFUSAL_CAPABILITY = "refusal"
_MULTIPLE_PROMPTS_CAPABILITY = "multiple_prompts"
_MULTIPLE_CHOICES_CAPABILITY = "multiple_choices"
_LOG_PROBABILITIES_CAPABILITY = "log_probabilities"
_ECHO_CAPABILITY = "echo"
_SUFFIX_CAPABILITY = "suffix"
_BEST_OF_CAPABILITY = "best_of"

_TOOL_CALL_REFUSAL_MESSAGE = (
    "the gateway cannot return an answer carrying a tool call on this endpoint: the legacy "
    "text-completion shape has no field for one, and the gateway does not silently drop "
    "model output; the model produced a tool call"
)
"""The message a tool-carrying answer is refused with, on both paths.

One message serves the non-streaming capability error and the streaming terminal
error event, so a caller learns the same fact whichever way it asked for the
answer.
"""


class _Wire(NamedTuple):
    """The JSON type a recognised parameter's value must have on the wire.

    ``name`` is how the type is named in a malformed-input message, so a caller
    learns which type was expected as well as which one arrived. ``accepts``
    tests a value against that type as the wire format defines it rather than as
    Python does: a JSON ``true`` is a boolean and not the integer ``1``, even
    though Python's ``bool`` is an ``int``, so ``{"n": true}`` is not ``n: 1``.
    """

    name: str
    accepts: Callable[[object], bool]


def _is_integer(value: object) -> bool:
    """A JSON integer. A JSON ``true`` is a boolean, not the integer ``1``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_boolean(value: object) -> bool:
    return isinstance(value, bool)


def _is_string_or_null(value: object) -> bool:
    return value is None or isinstance(value, str)


def _is_default_one(value: object) -> bool:
    return value == 1


def _is_default_false(value: object) -> bool:
    return value is False


def _is_default_none(value: object) -> bool:
    return value is None


_INTEGER = _Wire("an integer", _is_integer)
_BOOLEAN = _Wire("a boolean", _is_boolean)
_STRING_OR_NULL = _Wire("a string or null", _is_string_or_null)


class _Feature(NamedTuple):
    """An OpenAI parameter the gateway recognises and cannot honour.

    ``capability`` names what the parameter asks for, so the refusal says which
    feature is missing rather than only which key was sent. ``wire`` is the JSON
    type its value must have, and ``default`` decides whether a value of that
    type is the OpenAI default, which asks for nothing: refusing one would reject
    a request the gateway can serve in full. A value whose wire type is wrong
    asks for no feature and is not a capability the shape lacks — it is malformed
    input, refused as an invalid request naming the type that arrived.
    """

    capability: str
    wire: _Wire
    default: Callable[[object], bool]


_UNSUPPORTED_FEATURES: Mapping[str, _Feature] = {
    "n": _Feature(_MULTIPLE_CHOICES_CAPABILITY, _INTEGER, _is_default_one),
    "logprobs": _Feature(_LOG_PROBABILITIES_CAPABILITY, _BOOLEAN, _is_default_false),
    "echo": _Feature(_ECHO_CAPABILITY, _BOOLEAN, _is_default_false),
    "suffix": _Feature(_SUFFIX_CAPABILITY, _STRING_OR_NULL, _is_default_none),
    "best_of": _Feature(_BEST_OF_CAPABILITY, _INTEGER, _is_default_one),
}
"""The recognised parameters this endpoint cannot honour, and what they ask for.

Each carries the wire type its value must have and the OpenAI default that asks
for nothing: ``n: 1`` and ``best_of: 1`` ask for the one completion and the one
sample the gateway already produces, ``logprobs: false`` and ``echo: false`` ask
for the absence of something the gateway never adds, and ``suffix: null`` states
nothing at all. A value of the right type that is not that default asks for the
feature, and is refused with a capability error naming it; a value of the wrong
type is malformed input, and is refused as an invalid request (PRD section 9.5).
"""


@dataclass(frozen=True)
class CompletionRequest:
    """A parsed ``POST /v1/completions`` request.

    The parse is the gateway's, not the wire format's: by the time a caller has
    one of these, every recognised field has a canonical home and the prompt is
    a single canonical user message. Two wire fields map onto canonical fields
    that are not their namesakes — ``max_tokens`` becomes ``max_output_tokens``,
    because the OpenAI name counts the model's output, which is exactly what the
    canonical field bounds — and the prompt's array form is refused during the
    parse rather than narrowed to its first element (see the module docstring).

    The canonical request is built during the parse rather than afterwards, so
    the rules it enforces are applied before any provider is called. A request
    the gateway cannot represent is refused as malformed input, not discovered to
    be unrepresentable halfway through an invocation.

    ``stream`` is carried rather than acted on here. What the caller asked for is
    recorded so the handler can serve it as Server-Sent Events or refuse it with
    a capability error naming ``streaming`` when the bound provider cannot
    stream; the decision belongs to the endpoint, which knows what it can serve.
    """

    invocation: InvocationRequest
    stream: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        """Parse a decoded JSON request body, refusing anything it cannot carry.

        A ``ValueError`` from the canonical types is a validation failure of this
        request and becomes an invalid-request error; its message is the domain's
        own, which is gateway-authored and names the field that failed. A
        recognised parameter this endpoint cannot honour is refused with a
        capability error naming the feature, and a key the endpoint does not
        recognise is an invalid-request error naming that key.
        """
        where = "the request body"
        body = _as_object(where, payload)
        _reject_unsupported_parameters(body)
        try:
            invocation = InvocationRequest(
                model=as_required_text(f"{where} 'model'", body.get("model")),
                messages=(_user_message(f"{where} 'prompt'", body.get("prompt")),),
                generation_config=_generation_config(body),
                idempotency_key=_as_idempotency_key(
                    f"{where} 'idempotency_key'", body.get("idempotency_key")
                ),
            )
        except ValueError as error:
            raise InvalidArgumentError(f"invalid completion request: {error}") from error
        return cls(invocation=invocation, stream=_as_flag(f"{where} 'stream'", body.get("stream")))

    def to_invocation(
        self, *, invocation_id: str, identity: Identity | None = None
    ) -> InvocationRequest:
        """The canonical invocation, carrying the id minted for this request.

        ``identity`` is the trusted identity assigned by the authentication
        middleware. The invocation's ``tenant_id`` is that identity's tenant;
        the request shape carries no tenant claim of its own, so there is nothing
        to reconcile and the body's absence of a tenant is the whole of that rule
        here (PRD section 5.1).
        """
        tenant = identity.tenant if identity is not None and identity.authenticated else None
        return replace(self.invocation, invocation_id=invocation_id, tenant_id=tenant)


@dataclass(frozen=True)
class TextCompletionChoice:
    """One choice of an OpenAI legacy text completion.

    The legacy shape carries the generated ``text`` and the ``finish_reason``,
    and nothing else: there is no field for a role, a tool call or a refusal, so
    an answer carrying any of those never reaches this type —
    :class:`TextCompletion` refuses it (see :func:`_reject_unrepresentable_answer`).
    """

    text: str
    index: int
    finish_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "index": self.index, "finish_reason": self.finish_reason}


@dataclass(frozen=True)
class TextCompletion:
    """An OpenAI legacy text completion response.

    The gateway's own accounting — upstream model, attempt count, fallback use,
    cost — is deliberately not in this payload: the legacy completions shape has
    no field for it, and adding one would be a contract the gateway invented.
    That accounting belongs to the gRPC contract and to usage records (PRD
    sections 5.2 and 13). ``idempotency_status`` is the one piece of it the shape
    has a home for: it is rendered beside ``id`` and ``model`` so a caller
    reading the answer can tell whether the provider was reached (PRD section 12).
    """

    id: str
    model: str
    idempotency_status: str
    choices: tuple[TextCompletionChoice, ...]
    usage: Usage
    created: int

    @classmethod
    def from_invocation(
        cls,
        response: InvocationResponse,
        *,
        completion_id: str,
        created: int,
    ) -> Self:
        """Translate a canonical response into the OpenAI legacy completion shape.

        A canonical answer carrying a tool call, a refusal or an image has no
        representation here and is refused rather than approximated, because the
        legacy choice shape has nowhere to put any of them and the gateway does
        not return a partial answer. The refusal is a capability error, so a
        caller learns which feature is missing instead of receiving an answer
        that is not the one the model gave (PRD sections 9.1 and 9.5).
        """
        _reject_unrepresentable_answer(response)
        return cls(
            id=completion_id,
            model=response.model,
            idempotency_status=response.idempotency_status.value,
            choices=(
                TextCompletionChoice(
                    text=response.text,
                    index=0,
                    finish_reason=_finish_reason(response),
                ),
            ),
            usage=response.usage,
            created=created,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "object": _OBJECT_TEXT_COMPLETION,
            "created": self.created,
            "model": self.model,
            "idempotency_status": self.idempotency_status,
            "choices": [choice.to_dict() for choice in self.choices],
            "usage": _usage_payload(self.usage),
        }


def completion_request_from_payload(payload: object) -> CompletionRequest:
    """Parse a decoded JSON body into a :class:`CompletionRequest`.

    The parse is the gateway's, not the wire format's: by the time a caller has
    one of these, every recognised field has a canonical home and the prompt is
    a single canonical user message, so an array prompt, an unimplementable
    feature value or a malformed field is refused here rather than discovered
    halfway through an invocation.

    A ``ValueError`` from the canonical types or from the serialisation helpers
    is a validation failure of this request and becomes an invalid-request error
    carrying the domain's own gateway-authored message, which names the field
    that failed. A recognised parameter this endpoint cannot honour is refused
    with a capability error naming the feature, and a key the endpoint does not
    recognise is an invalid-request error naming that key (PRD sections 4 and
    9.5).
    """
    return CompletionRequest.from_payload(payload)


def create_completions(service: InferenceService) -> CompletionsHandler:
    """Build the handler for ``POST /v1/completions``."""

    async def completions(request: Request) -> Response:
        completion_request = completion_request_from_payload(await _read_json_body(request))

        identity = get_identity(request)
        invocation_id = uuid4().hex
        invocation = completion_request.to_invocation(
            invocation_id=invocation_id, identity=identity
        )

        if not completion_request.stream:
            response = await service.invoke(
                invocation,
                namespace=caller_namespace(identity),
                tenant=caller_tenant(identity),
            )
            completion = TextCompletion.from_invocation(
                response,
                completion_id=f"{_COMPLETION_ID_PREFIX}{response.invocation_id or invocation_id}",
                created=int(time.time()),
            )
            return json(completion.to_dict())

        return await _stream_completion(
            service,
            invocation,
            completion_id=f"{_COMPLETION_ID_PREFIX}{invocation_id}",
            namespace=caller_namespace(identity),
            tenant=caller_tenant(identity),
        )

    return completions


def register_completions_routes(router: Router, service: InferenceService) -> None:
    """Add the legacy completions endpoint to ``router``."""
    router.add_post("/v1/completions", create_completions(service))


async def _stream_completion(
    service: InferenceService,
    invocation: InvocationRequest,
    *,
    completion_id: str,
    namespace: str,
    tenant: str | None,
) -> Response:
    """Serve one streaming invocation as Server-Sent Events (PRD section 16).

    The response is an ``text/event-stream`` whose chunks follow the OpenAI
    legacy streaming shape: each chunk carries the fragment the event reports
    under ``choices[0].text``, and the terminal finish chunk carries the reason
    generation stopped with an empty text. The events come from the inference
    service's own stream, so routing, retry, fallback, the deadline and the
    no-fallback-after-first-output rule are the gateway core's, not duplicated
    here.

    The handler reflects the service's three terminal states. A stream whose last
    event is a ``finish`` completed, and it is the only one that ends with
    ``[DONE]``. A stream whose last event is an ``error`` failed after its answer
    had begun: the error chunk is the last frame and no ``[DONE]`` follows it,
    because the answer did not complete. A stream that ends with no terminal
    event at all — the upstream went away, or the caller did — gets no ``[DONE]``
    either, so a caller never reads a finished answer from a stream that simply
    stopped.

    A failure before the first event is handled differently: the invocation has
    produced nothing the caller could see, so a refusal — an unknown model, a
    capability gap, a deadline — is returned as an ordinary error response with
    its own status, before the SSE response commits to a 200 the caller cannot be
    given back. The same holds for a leading tool-call event, which the legacy
    chunk shape cannot carry: it is refused as an ordinary capability error rather
    than streamed.
    """
    event_iter = service.invoke_stream(invocation, namespace=namespace, tenant=tenant)

    # Pull the first event to surface a pre-stream refusal — a GatewayError or a
    # tool call the shape cannot carry — before the SSE response commits to a 200
    # the caller cannot be given back.
    try:
        first_event: StreamEvent | None = await event_iter.__anext__()
    except StopAsyncIteration:
        first_event = None
    except GatewayError as error:
        return json(error.to_payload(), status=error.http_status)

    created = int(time.time())
    model = invocation.model

    if first_event is not None and first_event.kind is StreamEventKind.TOOL_CALL:
        # A leading tool call is a capability gap discovered before anything was
        # delivered, so it is refused as an ordinary error response rather than
        # a stream (PRD section 16).
        await event_iter.aclose()
        error = _tool_call_capability_error()
        return json(error.to_payload(), status=error.http_status)

    def sse_event(event: StreamEvent) -> TextServerSentEvent:
        return TextServerSentEvent(
            orjson.dumps(
                _chunk_from_event(event, completion_id=completion_id, created=created, model=model)
            ).decode("utf-8")
        )

    async def events_provider() -> AsyncIterator[ServerSentEvent]:
        completed = False
        try:
            if first_event is not None:
                completed = first_event.kind is StreamEventKind.FINISH
                yield sse_event(first_event)
            try:
                async for event in event_iter:
                    if event.kind is StreamEventKind.TOOL_CALL:
                        # A tool call has no place in the legacy chunk shape. End
                        # the stream with the terminal error event and no
                        # [DONE], because the answer did not complete.
                        yield _tool_call_error_event(
                            completion_id=completion_id, created=created, model=model
                        )
                        return
                    completed = event.kind is StreamEventKind.FINISH
                    yield sse_event(event)
            except GatewayError:
                # A failure after the first event: the service has already
                # emitted the terminal ERROR event, which the caller has seen.
                # End the stream without [DONE], because the answer did not
                # complete.
                return
            if completed:
                # [DONE] closes a stream that ended with a finish, and only that
                # stream: a stream that stopped without a terminal event is not
                # a completed answer, so it is closed without one.
                yield TextServerSentEvent(_DONE)
        finally:
            # Close the service's stream when this generator ends — normally, or
            # because the consumer went away — so a client disconnect stops the
            # upstream work rather than leaving it running.
            await event_iter.aclose()

    return _gateway_sse_response(events_provider)


def _chunk_envelope(completion_id: str, created: int, model: str) -> dict[str, object]:
    """The fields every gateway streaming chunk carries, before its payload.

    The OpenAI legacy streaming shape keeps ``id``, ``object``, ``created`` and
    ``model`` on every frame, so a consumer that multiplexes streams attributes
    each one and correlates it with the request that produced it. What a chunk
    reports beyond that — its ``choices`` and, for a terminal error, its
    ``error`` — is added by the caller.
    """
    return {
        "id": completion_id,
        "object": _OBJECT_TEXT_COMPLETION,
        "created": created,
        "model": model,
    }


def _chunk_from_event(
    event: StreamEvent, *, completion_id: str, created: int, model: str
) -> dict[str, object]:
    """One stream event as an OpenAI legacy streaming chunk (PRD section 16).

    The chunk shape is the OpenAI-compatible one: an ``id``, ``object``,
    ``created`` and ``model`` carried on every chunk, and a single choice whose
    ``text`` carries the fragment the event reported. A content event carries
    text, a usage event carries the usage report on a chunk with no choice, a
    finish event carries the reason generation stopped with an empty text, and a
    terminal error event carries the gateway's reason and sanitised message
    under ``error`` on a chunk with no choice. A tool-call event never reaches
    this function: the events provider ends the stream with an error event
    instead, because the legacy chunk shape has no place for it.
    """
    chunk = _chunk_envelope(completion_id, created, model)
    if event.kind is StreamEventKind.CONTENT:
        chunk["choices"] = [{"index": 0, "text": event.text, "finish_reason": None}]
    elif event.kind is StreamEventKind.USAGE:
        chunk["choices"] = []
        chunk["usage"] = _usage_payload(event.usage)
    elif event.kind is StreamEventKind.FINISH:
        chunk["choices"] = [
            {"index": 0, "text": "", "finish_reason": _finish_reason_chunk(event)}
        ]
    elif event.kind is StreamEventKind.ERROR:
        chunk["choices"] = []
        chunk["error"] = {"reason": event.error_reason, "message": event.error_message}
    return chunk


def _tool_call_error_fields() -> dict[str, str]:
    """The reason and message a tool-carrying answer is refused with.

    The same refusal the non-streaming path raises, so a caller learns the same
    fact from either path: the legacy shape has no field for a tool call, and
    the gateway does not silently drop model output (PRD sections 9.1 and 9.5).
    """
    return {
        "reason": ErrorReason.UNSUPPORTED_CAPABILITY.value,
        "message": _TOOL_CALL_REFUSAL_MESSAGE,
    }


def _tool_call_error_event(
    *, completion_id: str, created: int, model: str
) -> TextServerSentEvent:
    """The terminal error event that ends a stream carrying a tool call.

    A tool call has no place in the legacy chunk shape, exactly as it has none
    in the non-streaming response, so the stream cannot honestly continue. It
    ends with the gateway's own stable reason — the capability refusal a
    tool-carrying answer is refused with on the non-streaming path — and the
    same message, on a chunk with no choice, so the caller is not told the
    answer completed (PRD sections 9.1 and 16).
    """
    chunk = _chunk_envelope(completion_id, created, model)
    chunk["choices"] = []
    chunk["error"] = _tool_call_error_fields()
    return TextServerSentEvent(orjson.dumps(chunk).decode("utf-8"))


def _tool_call_capability_error() -> CapabilityError:
    """The refusal for an answer the legacy shape cannot carry a tool call in."""
    return CapabilityError(
        _TOOL_CALL_REFUSAL_MESSAGE,
        details={"capability": _TOOL_CALLS_CAPABILITY},
    )


def _finish_reason_chunk(event: StreamEvent) -> str | None:
    """The OpenAI finish reason for a finish event.

    A finish event whose reason the gateway could not classify reports the raw
    reason verbatim rather than an invented OpenAI value, exactly as the
    non-streaming response does (PRD sections 5.2 and 9.1). A failure after
    output does not reach here: it ends the stream with an ``error`` event, not
    a ``finish`` event, so the gateway's own reason is never rendered as a
    finish reason.
    """
    if event.finish_reason is FinishReason.UNKNOWN:
        return event.raw_finish_reason
    return _FINISH_REASONS[event.finish_reason]


def _reject_unrepresentable_answer(response: InvocationResponse) -> None:
    """Refuse an answer the legacy choice shape cannot carry.

    The legacy choice holds one ``text`` and nothing else, so a canonical answer
    carrying a tool call, a refusal or an image has no representation here.
    Returning it without the unrepresentable part would tell the caller the
    answer was delivered when part of it was dropped, which PRD section 9.5
    forbids; the refusal names the feature so the caller learns what the shape
    is missing (PRD section 9.1).
    """
    if response.tool_calls:
        raise _tool_call_capability_error()
    if response.refusals:
        raise CapabilityError(
            "the gateway cannot return an answer carrying a refusal on this endpoint: the "
            "legacy text-completion shape has no field for one, and the gateway does not "
            "silently drop model output; the model produced a refusal",
            details={"capability": _REFUSAL_CAPABILITY},
        )
    if response.images:
        raise CapabilityError(
            "the gateway cannot return an answer carrying an image on this endpoint: the "
            "legacy text-completion shape has no field for one, and the gateway does not "
            "silently drop model output; the model produced an image",
            details={"capability": _IMAGE_OUTPUT_CAPABILITY},
        )


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
    """The OpenAI usage object, as the non-streaming response and the stream carry it.

    A count the provider never reported is presented as zero because the OpenAI
    usage object types its counts as integers; the canonical record keeps the
    difference between "none were used" and "nobody counted" (PRD section 13).
    On a streaming chunk the same counts appear on a chunk with no choice,
    because a usage report is a fact about the answer, not a fragment of it
    (PRD section 16).
    """
    prompt = 0 if usage.input_tokens is None else usage.input_tokens
    completion = 0 if usage.output_tokens is None else usage.output_tokens
    total = usage.total_tokens
    if total is None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _reject_unsupported_parameters(body: Mapping[str, object]) -> None:
    """Refuse a parameter the gateway cannot honour, naming what it asked for.

    A key this endpoint does not recognise is malformed input. A key it does
    recognise is judged on its wire type first: a value of the wrong type is
    malformed input too, refused as an invalid request naming the parameter and
    the type that arrived, because ``{"echo": 0}`` is ill-typed rather than a
    request for the echo feature. A value of the right type that asks for
    something this gateway has no representation for is a capability the shape
    cannot provide — the request is well formed and the refusal is about the
    feature (PRD section 9.5) — and a value that is the OpenAI default is
    accepted, because refusing it would reject a request the gateway can serve in
    full.
    """
    unknown = sorted(set(body) - _REQUEST_KEYS)
    if unknown:
        supported = ", ".join(repr(key) for key in sorted(_REQUEST_KEYS))
        raise InvalidArgumentError(
            f"the request body has unsupported parameter {unknown[0]!r}; supported parameters "
            f"are {supported}"
        )
    for name in sorted(_UNSUPPORTED_FEATURES):
        if name not in body:
            continue
        feature = _UNSUPPORTED_FEATURES[name]
        value = body[name]
        if not feature.wire.accepts(value):
            raise InvalidArgumentError(
                f"the request body {name!r} must be {feature.wire.name}, got {describe(value)}"
            )
        if feature.default(value):
            continue
        raise CapabilityError(
            f"the gateway cannot honour {name!r}; it would request {feature.capability!r}, which "
            f"this endpoint's shape cannot carry and the gateway will not silently discard",
            details={"capability": feature.capability, "parameter": name},
        )


def _user_message(where: str, value: object) -> Message:
    """The prompt as a single canonical user message.

    A legacy completion has no roles, so the prompt becomes one user message
    carrying its text. An array prompt is refused rather than narrowed: it asks
    for one completion per element, and the canonical invocation carries one
    conversation while the legacy shape answers one completion, so the gateway
    neither silently uses the first element nor joins the elements into a prompt
    the caller never sent (see the module docstring).
    """
    if isinstance(value, list):
        raise CapabilityError(
            "the gateway cannot honour a 'prompt' array: it asks for one completion per element, "
            "and the canonical invocation carries one conversation while this endpoint's shape "
            "answers one completion, so the gateway does not silently use the first element",
            details={"capability": _MULTIPLE_PROMPTS_CAPABILITY, "parameter": "prompt"},
        )
    text = _as_prompt_text(where, value)
    return Message(role=Role.USER, content=(TextPart(text),))


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


def _as_prompt_text(where: str, value: object) -> str:
    """The prompt text, which may be empty but must be a string.

    An empty prompt is honoured rather than refused: the canonical user message
    may carry empty text, and the gateway forwards exactly what the caller sent
    rather than inventing a prompt of its own. A missing prompt, or one that is
    neither a string nor an array, is malformed input.
    """
    if value is None:
        raise InvalidArgumentError(f"{where} is required and must be a string")
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{where} must be a string, got {describe(value)}")
    return value


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
    )


def _as_stop_sequences(where: str, value: object) -> tuple[str, ...]:
    """A stop sequence or an array of them, the two forms the wire format accepts."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidArgumentError(
            f"{where} must be a string or an array of strings, got {describe(value)}"
        )
    return tuple(value)


def _as_flag(where: str, value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise InvalidArgumentError(f"{where} must be a boolean, got {describe(value)}")
    return value


def _as_object(where: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidArgumentError(f"{where} must be a JSON object, got {describe(value)}")
    return value


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
