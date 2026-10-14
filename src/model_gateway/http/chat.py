"""The chat endpoint: ``POST /v1/chat/completions`` (PRD sections 4.4 and 16).

The handler is thin on purpose. It decodes the body, translates it into the
canonical invocation, calls the inference service and renders the canonical
response back into the OpenAI shape. It resolves no route, retries nothing and
counts nothing: routing, fallback, retry and accounting are the gateway core's
(PRD sections 9.2 and 11.3), and a second copy of that logic on the HTTP path is
how the two transports drift apart.

**Streaming is refused, not faked.** A request carrying ``stream: true`` is
answered with a capability error. PRD section 16 permits a gateway to reject a
streaming request it cannot serve, and forbids the alternative: returning a
complete, non-streamed body in answer to a request that asked for tokens as they
were produced would tell the caller it received something it did not (PRD
sections 16 and 9.5). The refusal names the feature, so the caller learns what is
missing rather than guessing why the response arrived at once.

The invocation id is minted here, per request, and carried into the canonical
request. It is what the response's ``id`` is built from and what correlates the
caller's completion with the gateway's own logs and usage records.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from blacksheep import HTTPException, Request, Response
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.errors import CapabilityError, InvalidArgumentError
from model_gateway.http.dto import ChatCompletion, ChatRequest

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService

__all__ = ["create_chat_completion", "register_chat_routes"]

ChatHandler = Callable[[Request], Awaitable[Response]]

_STREAMING_CAPABILITY = "streaming"
_COMPLETION_ID_PREFIX = "chatcmpl-"


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


def _reject_streaming(chat_request: ChatRequest) -> None:
    """Refuse a streaming request rather than answering it with a whole body."""
    if not chat_request.stream:
        return
    raise CapabilityError(
        "this endpoint cannot stream a chat completion; the gateway does not answer a streaming "
        "request with a non-streamed body, so repeat the request with 'stream': false",
        details={"capability": _STREAMING_CAPABILITY},
    )


def create_chat_completion(service: InferenceService) -> ChatHandler:
    """Build the handler for ``POST /v1/chat/completions`` (PRD section 4.4)."""

    async def chat_completions(request: Request) -> Response:
        chat_request = ChatRequest.from_payload(await _read_json_body(request))
        _reject_streaming(chat_request)

        invocation_id = uuid4().hex
        response = await service.invoke(chat_request.to_invocation(invocation_id=invocation_id))

        completion = ChatCompletion.from_invocation(
            response,
            completion_id=f"{_COMPLETION_ID_PREFIX}{response.invocation_id or invocation_id}",
            created=int(time.time()),
        )
        return json(completion.to_dict())

    return chat_completions


def register_chat_routes(router: Router, service: InferenceService) -> None:
    """Add the chat completions endpoint to ``router``."""
    router.add_post("/v1/chat/completions", create_chat_completion(service))
