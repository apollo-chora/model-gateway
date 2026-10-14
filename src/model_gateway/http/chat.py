"""The chat endpoint: ``POST /v1/chat/completions`` (PRD sections 4.4 and 16).

The handler is thin on purpose. It decodes the body, translates it into the
canonical invocation, calls the inference service and renders the canonical
response back into the OpenAI shape. It resolves no route, retries nothing and
counts nothing: routing, fallback, retry and accounting are the gateway core's
(PRD sections 9.2 and 11.3), and a second copy of that logic on the HTTP path is
how the two transports drift apart.

**Streaming is served as Server-Sent Events.** A request carrying ``stream:
true`` is answered with an ``text/event-stream`` response whose chunks follow
the OpenAI streaming shape, terminated by ``[DONE]`` (PRD section 16). The
gateway does not maintain per-provider streaming adapters: it consumes the
stream the binding produces and re-normalises each chunk at the boundary. A
provider that cannot stream is refused with a capability error naming
``streaming`` before the response starts, because answering a streaming request
with a complete, non-streamed body would tell the caller it received something
it did not (PRD sections 16 and 9.5).

A failure is handled differently depending on whether the response has started.
Before the first event, the invocation has produced nothing the caller could
see, so a refusal — an unknown model, a capability gap, a deadline — is returned
as an ordinary error response with its own status. After the first event, the
response has begun and its status can no longer change; the inference service
ends the stream with a terminal error event and the handler closes the stream
without ``[DONE]``, so the caller sees the error rather than a stream that
silently stopped.

The invocation id is minted here, per request, and carried into the canonical
request. It is what the response's ``id`` is built from and what correlates the
caller's completion with the gateway's own logs and usage records.

The trusted identity is the one the authentication middleware assigned to the
request. The handler reconciles the body's ``tenant_id`` against it with
:func:`~model_gateway.security.authorization.reconcile_identity` — the
transport-independent policy — and hands the identity to
:meth:`~model_gateway.http.dto.ChatRequest.to_invocation`, so the invocation
carries the credential's tenant and never the body's (PRD section 5.1).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import uuid4

import orjson
from blacksheep import HTTPException, Request, Response
from blacksheep.contents import ServerSentEvent, StreamedContent, TextServerSentEvent
from blacksheep.scribe import write_sse
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    StreamEvent,
    StreamEventKind,
    Usage,
)
from model_gateway.domain.streaming import ToolCallDelta
from model_gateway.errors import GatewayError, InvalidArgumentError
from model_gateway.http.auth import get_identity
from model_gateway.http.dto import ChatCompletion, ChatRequest
from model_gateway.security.authorization import reconcile_identity

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService

__all__ = ["create_chat_completion", "register_chat_routes"]

ChatHandler = Callable[[Request], Awaitable[Response]]

_STREAMING_CAPABILITY = "streaming"
_COMPLETION_ID_PREFIX = "chatcmpl-"
_DONE = "[DONE]"
_OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
_STREAM_CONTENT_TYPE = b"text/event-stream"
_SSE_HEADERS = [(b"Cache-Control", b"no-cache"), (b"Connection", b"Keep-Alive")]

_FINISH_REASONS: dict[FinishReason, str] = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.TOOL_CALLS: "tool_calls",
    FinishReason.CONTENT_FILTER: "content_filter",
}


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


def create_chat_completion(service: InferenceService) -> ChatHandler:
    """Build the handler for ``POST /v1/chat/completions`` (PRD section 4.4)."""

    async def chat_completions(request: Request) -> Response:
        chat_request = ChatRequest.from_payload(await _read_json_body(request))

        identity = get_identity(request)
        reconcile_identity(identity, chat_request.invocation.tenant_id)

        invocation_id = uuid4().hex
        invocation = chat_request.to_invocation(invocation_id=invocation_id, identity=identity)

        if not chat_request.stream:
            response = await service.invoke(invocation)
            completion = ChatCompletion.from_invocation(
                response,
                completion_id=f"{_COMPLETION_ID_PREFIX}{response.invocation_id or invocation_id}",
                created=int(time.time()),
            )
            return json(completion.to_dict())

        return await _stream_chat_completion(
            service, invocation, completion_id=f"{_COMPLETION_ID_PREFIX}{invocation_id}"
        )

    return chat_completions


async def _stream_chat_completion(
    service: InferenceService,
    invocation: InvocationRequest,
    *,
    completion_id: str,
) -> Response:
    """Serve one streaming invocation as Server-Sent Events (PRD section 16).

    The response is an ``text/event-stream`` whose chunks follow the OpenAI
    streaming shape, terminated by ``[DONE]``. The events come from the
    inference service's own stream, so routing, retry, fallback, the deadline
    and the no-fallback-after-first-output rule are the gateway core's, not
    duplicated here.

    A failure is handled differently depending on whether the response has
    started. Before the first event, the invocation has produced nothing the
    caller could see, so a refusal — an unknown model, a capability gap, a
    deadline — is returned as an ordinary error response with its own status.
    After the first event, the response has begun and its status can no longer
    change; the service ends the stream with a terminal error event and the
    stream is closed without ``[DONE]``, so the caller sees the error rather
    than a stream that silently stopped.
    """
    event_iter = service.invoke_stream(invocation)

    # Pull the first event to surface a pre-stream refusal before the SSE
    # response commits to a 200 the caller cannot be given back.
    try:
        first_event: StreamEvent | None = await event_iter.__anext__()
    except StopAsyncIteration:
        first_event = None
    except GatewayError as error:
        return json(error.to_payload(), status=error.http_status)

    created = int(time.time())
    model = invocation.model

    def sse_event(event: StreamEvent) -> TextServerSentEvent:
        return TextServerSentEvent(
            orjson.dumps(
                _chunk_from_event(event, completion_id=completion_id, created=created, model=model)
            ).decode("utf-8")
        )

    async def events_provider() -> AsyncIterator[ServerSentEvent]:
        try:
            if first_event is not None:
                yield sse_event(first_event)
            try:
                async for event in event_iter:
                    yield sse_event(event)
            except GatewayError:
                # A failure after the first event: the service has already emitted
                # the terminal error event. End the stream without [DONE], because
                # the answer did not complete.
                return
            yield TextServerSentEvent(_DONE)
        finally:
            # Close the service's stream when this generator ends — normally, or
            # because the consumer went away — so a client disconnect stops the
            # upstream work rather than leaving it running.
            await event_iter.aclose()

    return _gateway_sse_response(events_provider)


class _GatewaySSEContent(StreamedContent):
    """Server-Sent Events content that closes the gateway's event provider.

    The gateway's event provider wraps the inference service's stream, which
    wraps the provider's upstream call. BlackSheep's own SSE content does not
    close the provider it is handed — it releases it, leaving the upstream
    work to be stopped by garbage collection, which a client disconnect cannot
    wait for. This content closes the provider when the stream ends, whether
    it ends normally or because the consumer went away, so a disconnect stops
    the upstream work deterministically (PRD section 16).
    """

    def __init__(self, events_provider: Callable[[], AsyncIterator[ServerSentEvent]]) -> None:
        super().__init__(_STREAM_CONTENT_TYPE, self._write_events(events_provider))

    @staticmethod
    def _write_events(
        events_provider: Callable[[], AsyncIterator[ServerSentEvent]],
    ) -> Callable[[], AsyncIterator[bytes]]:
        async def write_events() -> AsyncIterator[bytes]:
            provider = events_provider()
            try:
                async for event in provider:
                    yield write_sse(event)
            finally:
                await provider.aclose()

        return write_events


def _gateway_sse_response(
    events_provider: Callable[[], AsyncIterator[ServerSentEvent]],
) -> Response:
    """The SSE response for one streaming invocation (PRD section 16).

    The headers are the ones Server-Sent Events calls for: a no-cache
    directive, because a stream is a live answer rather than a cacheable
    document, and a keep-alive connection, because the stream stays open until
    the answer is complete.
    """
    return Response(200, _SSE_HEADERS, _GatewaySSEContent(events_provider))


def _chunk_from_event(
    event: StreamEvent, *, completion_id: str, created: int, model: str
) -> dict[str, object]:
    """One stream event as an OpenAI streaming chunk (PRD section 16).

    The chunk shape is the OpenAI-compatible one: an ``id``, ``object``,
    ``created`` and ``model`` carried on every chunk, and a single choice whose
    ``delta`` carries the fragment the event reports. A content event carries
    text, a tool-call event carries a tool-call fragment, a usage event carries
    the usage report on a chunk with no choice, and a finish event carries the
    reason generation stopped with an empty delta.
    """
    chunk: dict[str, object] = {
        "id": completion_id,
        "object": _OBJECT_CHAT_COMPLETION_CHUNK,
        "created": created,
        "model": model,
    }
    if event.kind is StreamEventKind.CONTENT:
        chunk["choices"] = [
            {"index": 0, "delta": {"content": event.text}, "finish_reason": None}
        ]
    elif event.kind is StreamEventKind.TOOL_CALL:
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"tool_calls": [_tool_call_chunk(event.tool_call)]},
                "finish_reason": None,
            }
        ]
    elif event.kind is StreamEventKind.USAGE:
        chunk["choices"] = []
        chunk["usage"] = _usage_chunk(event.usage)
    elif event.kind is StreamEventKind.FINISH:
        chunk["choices"] = [
            {"index": 0, "delta": {}, "finish_reason": _finish_reason_chunk(event)}
        ]
    return chunk


def _tool_call_chunk(delta: ToolCallDelta) -> dict[str, object]:
    """One tool-call delta as an OpenAI streaming tool-call fragment.

    The fragment carries only the fields this event reported: the identifier,
    the name and the argument fragment are each present only when the event
    carried them, because a tool call arrives piece by piece and a consumer
    accumulates the pieces by ``index`` (PRD sections 5.3 and 16).
    """
    call: dict[str, object] = {"index": delta.index, "type": "function", "function": {}}
    function: dict[str, object] = call["function"]  # type: ignore[index]
    if delta.id is not None:
        call["id"] = delta.id
    if delta.name is not None:
        function["name"] = delta.name
    if delta.arguments_delta is not None:
        function["arguments"] = delta.arguments_delta
    return call


def _usage_chunk(usage: Usage) -> dict[str, int]:
    """The OpenAI usage object, shaped as the non-streaming response shapes it.

    A count the provider never reported is presented as zero because the OpenAI
    usage object types its counts as integers; the canonical record keeps the
    difference between "none were used" and "nobody counted" (PRD section 13).
    """
    prompt = 0 if usage.input_tokens is None else usage.input_tokens
    completion = 0 if usage.output_tokens is None else usage.output_tokens
    total = usage.total_tokens
    if total is None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _finish_reason_chunk(event: StreamEvent) -> str | None:
    """The OpenAI finish reason for a finish event.

    A finish event whose reason the gateway could not classify — which includes
    the terminal event that reports an error — reports the raw reason verbatim
    rather than an invented OpenAI value, exactly as the non-streaming response
    does (PRD sections 5.2 and 9.1).
    """
    if event.finish_reason is FinishReason.UNKNOWN:
        return event.raw_finish_reason
    return _FINISH_REASONS[event.finish_reason]


def register_chat_routes(router: Router, service: InferenceService) -> None:
    """Add the chat completions endpoint to ``router``."""
    router.add_post("/v1/chat/completions", create_chat_completion(service))
