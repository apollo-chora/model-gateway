"""Conformance: the public surface against the real binding (PRD sections 20 and 21).

Every other public-surface suite in this tree injects a fake *provider* at the
binding boundary, so the real binding — its translation, its client injection,
its error mapping — is never on the path a public endpoint takes. That is the
gap this module closes, and it is the one PRD section 20 states directly: the
stand-in belongs to the upstream, never to the gateway.

**What is real here.** Configuration comes entirely from an environment mapping:
``Settings.from_env`` and ``ModelRegistry.from_env`` parse the same variables a
deployment sets, so the routes under test are the routes the parser builds. The
binding is the production ``LiteLLMProvider``, the service is the production
``InferenceService``, and the surface is the production application built by
``create_app``. A request through ``/v1/chat/completions`` therefore runs the
whole gateway: routing, capability enforcement, the deadline, the binding's
parameter translation, the pinned transport, the library's own response
construction, and the canonical translation back out.

**What is not real, and why.** The socket is not. The one thing injected is the
binding's ``inner_factory`` — the transport that performs I/O beneath the pinned
transport — and the resolver is pinned to a public address so the URL policy's
own screen passes. No test here opens a connection, which is what "no real
provider account" has to mean.

**Why not a mock upstream server.** PRD section 20 asks for "deterministic mock
upstream servers". Section 14.1 forbids the gateway from contacting any address
class a same-host server would occupy — loopback, private, link-local, metadata,
unique-local — and the prohibition is enforced twice, at configuration time and
again in ``pin_request`` before the socket is opened, with no opt-out. A mock
server on this machine is therefore unreachable by construction, and weakening
that control to make the tests easier would trade a security guarantee for test
ergonomics. The stand-in is instead an ``httpx`` transport that speaks the
provider's wire protocol — the same shape the embeddings and image suites already
use — and it is placed where section 20 says the stand-in belongs.
"""

import json
from typing import Any

import httpx
import pytest
from blacksheep import JSONContent
from blacksheep.testing import TestClient

from model_gateway.app import create_app
from model_gateway.config import Settings
from model_gateway.inference import InferenceService
from model_gateway.providers.litellm_provider import LiteLLMProvider
from model_gateway.registry import ModelRegistry

UPSTREAM_ADDRESS = "93.184.216.34"
"""A public address, so the URL policy's screen passes and the transport pins it."""

UPSTREAM_ENDPOINT = f"https://{UPSTREAM_ADDRESS}/v1"
"""The endpoint as a literal public address.

A literal rather than a hostname because ``ModelRegistry.from_env`` screens the
endpoint host at configuration time through the system resolver, and a test that
needs DNS is not deterministic. The address is public, so the screen accepts it;
the binding's injected resolver then pins it, and the wire beneath never opens a
socket either way. The parser is still exercised — scheme, address class and path
joining are all its work.
"""
UPSTREAM_SECRET = "sk-upstream-secret-key"
"""The provider credential, which no public response may carry."""

GATEWAY_KEY = "sk-gateway-key"
CHAT_TEXT = "hello from the upstream"


def _public_resolver(host: str) -> tuple[str, ...]:
    """Resolve every hostname to one public address, so no test touches DNS."""
    return (UPSTREAM_ADDRESS,)


# --------------------------------------------------------------------------- #
# The upstream: a transport that speaks the openai dialect's wire protocol
# --------------------------------------------------------------------------- #


def _completion(text: str) -> dict[str, Any]:
    """An OpenAI chat-completion body, in the shape the library parses."""
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "upstream-text",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _completion_chunks(text: str) -> bytes:
    """The same answer as a server-sent-event stream, terminated as the format requires."""

    def chunk(delta: dict[str, Any], finish: str | None) -> bytes:
        body = {
            "id": "chatcmpl-conformance",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "upstream-text",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    events = [
        chunk({"role": "assistant", "content": ""}, None),
        chunk({"content": text}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    ]
    return b"".join(events)


def _text_completion(text: str) -> dict[str, Any]:
    """An OpenAI legacy-completion body."""
    return {
        "id": "cmpl-conformance",
        "object": "text_completion",
        "created": 1_700_000_000,
        "model": "upstream-text",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


def _embeddings() -> dict[str, Any]:
    """An OpenAI embeddings body."""
    return {
        "object": "list",
        "model": "upstream-embed",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 0.75]}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


def _image() -> dict[str, Any]:
    """An OpenAI images body carrying inline base64 rather than a URL."""
    return {
        "created": 1_700_000_000,
        "data": [{"b64_json": "aGVsbG8="}],
        "output_format": "webp",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


class UpstreamWire(httpx.AsyncBaseTransport):
    """The upstream, speaking the wire protocol the ``openai`` dialect sends.

    It records every request it is asked to send, so a test can assert both that
    the request reached the upstream and what it carried. The host is ignored —
    the pinned transport has already rewritten it to the validated address — and
    the operation is read off the path, which is what a real upstream does with
    the same information.
    """

    def __init__(self, *, status: int | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # The SDK streams the request body when it is asked for a streamed answer,
        # so the body has to be read before it can be inspected: accessing
        # ``request.content`` on an unread streaming body raises rather than
        # returning it, and a transport that raised there would look to the
        # library like a connection failure.
        await request.aread()
        if self.status is not None:
            return httpx.Response(
                self.status,
                json={
                    "error": {
                        "message": "the upstream refused this request",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
                request=request,
            )
        path = request.url.path
        if path.endswith("/chat/completions"):
            if self._wants_stream(request):
                return httpx.Response(
                    200,
                    content=_completion_chunks(CHAT_TEXT),
                    headers={"content-type": "text/event-stream"},
                    request=request,
                )
            return httpx.Response(200, json=_completion(CHAT_TEXT), request=request)
        if path.endswith("/completions"):
            return httpx.Response(200, json=_text_completion(CHAT_TEXT), request=request)
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=_embeddings(), request=request)
        if path.endswith("/images/generations"):
            return httpx.Response(200, json=_image(), request=request)
        return httpx.Response(
            404,
            json={"error": {"message": f"the wire has no route for {path}"}},
            request=request,
        )

    @staticmethod
    def _wants_stream(request: httpx.Request) -> bool:
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:
            return False
        return bool(body.get("stream"))

    def paths(self) -> list[str]:
        """The path of every request the upstream was asked to serve, in order."""
        return [request.url.path for request in self.requests]

    def last_body(self) -> dict[str, Any]:
        """The JSON body of the last request, as the upstream received it."""
        return json.loads(self.requests[-1].content or b"{}")


# --------------------------------------------------------------------------- #
# The gateway, built from environment variables alone
# --------------------------------------------------------------------------- #

ENVIRONMENT = {
    "CHORA_ENVIRONMENT": "development",
    "TEXT_LLM_ENDPOINT": UPSTREAM_ENDPOINT,
    "TEXT_LLM_MODEL_ID": "upstream-text",
    "TEXT_LLM_API_KEY": UPSTREAM_SECRET,
    "TEXT_LLM_FORMAT": "openai",
    "TEXT_LLM_CAPABILITIES": "chat,tools,vision",
    "EMBEDDINGS_ENDPOINT": UPSTREAM_ENDPOINT,
    "EMBEDDINGS_MODEL_ID": "upstream-embed",
    "EMBEDDINGS_API_KEY": UPSTREAM_SECRET,
    "EMBEDDINGS_FORMAT": "openai",
    "IMAGE_GEN_ENDPOINT": UPSTREAM_ENDPOINT,
    "IMAGE_GEN_MODEL_ID": "upstream-image",
    "IMAGE_GEN_API_KEY": UPSTREAM_SECRET,
    "IMAGE_GEN_FORMAT": "openai",
    "IMAGE_GEN_CAPABILITIES": "generation",
    "GATEWAY_AUTH_ENABLED": "true",
    "GATEWAY_API_KEYS": f"{GATEWAY_KEY}:tenant-a",
}
"""The environment a deployment sets, and the only source of configuration here."""

AUTHORIZATION = {"Authorization": f"Bearer {GATEWAY_KEY}"}


def gateway(wire: UpstreamWire):
    """The production application over the real binding and the given upstream.

    The registry and the settings are parsed from ``ENVIRONMENT`` exactly as a
    deployment's are, so the routes under test are the ones the parser builds.
    Only the binding's I/O transport is supplied, because a socket is the one
    thing a conformance test may not open.
    """
    settings = Settings.from_env(dict(ENVIRONMENT))
    registry = ModelRegistry.from_env(dict(ENVIRONMENT))
    binding = LiteLLMProvider(resolver=_public_resolver, inner_factory=lambda: wire)
    service = InferenceService(registry, binding)
    return create_app(settings, registry=registry, service=service)


class _Running:
    """A started application, driven by a test client, stopped on exit.

    The client is built after ``start()`` because that is when the application's
    router is live; building it first yields a simulator over an unstarted app.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __aenter__(self) -> TestClient:
        await self.app.start()
        return TestClient(self.app)

    async def __aexit__(self, *exc_info: object) -> None:
        await self.app.stop()


def _running(wire: UpstreamWire) -> _Running:
    return _Running(gateway(wire))


async def _json(response) -> Any:
    return json.loads(await response.text())


# --------------------------------------------------------------------------- #
# Every public endpoint, over the real binding
# --------------------------------------------------------------------------- #


async def test_the_listing_is_served_from_environment_configuration():
    """No upstream is contacted: the listing is the gateway's own configured routes."""
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.get("/v1/models", headers=AUTHORIZATION)

    assert response.status == 200
    assert sorted(model["id"] for model in (await _json(response))["data"]) == [
        "embedding-default",
        "image-default",
        "text-default",
    ]
    assert wire.paths() == []


async def test_health_and_readiness_are_served_without_a_credential():
    wire = UpstreamWire()

    async with _running(wire) as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")

    assert health.status == 200
    assert ready.status == 200
    assert wire.paths() == []


async def test_metrics_are_published_and_require_a_credential():
    """Section 17's endpoint is authenticated like every other path."""
    wire = UpstreamWire()

    async with _running(wire) as client:
        refused = await client.get("/metrics")
        allowed = await client.get("/metrics", headers=AUTHORIZATION)

    assert refused.status == 401
    assert allowed.status == 200
    assert "gateway_requests_total" in await allowed.text()


async def test_a_chat_completion_is_served_by_the_real_binding():
    """The whole gateway: routing, the deadline, the binding, the library, back out.

    The assertion on the recorded body is the part no fake-provider suite can
    make: it is the binding's own parameter translation, on the wire.
    """
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent(
                {
                    "model": "text-default",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    body = await _json(response)
    assert body["choices"][0]["message"]["content"] == CHAT_TEXT
    assert body["model"] == "text-default"

    assert wire.paths() == ["/v1/chat/completions"]
    assert wire.last_body()["model"] == "upstream-text"
    assert [message["role"] for message in wire.last_body()["messages"]] == ["user"]


async def test_a_streamed_chat_completion_is_served_by_the_real_binding():
    """The streamed path, over the library's own SSE parsing and the gateway's contract."""
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent(
                {
                    "model": "text-default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            ),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    text = await response.text()
    assert text.rstrip().endswith("data: [DONE]")
    assert CHAT_TEXT in text
    assert wire.paths() == ["/v1/chat/completions"]
    assert wire.last_body()["stream"] is True


async def test_a_legacy_completion_is_served_by_the_real_binding():
    """The legacy endpoint is served by the binding's chat call, not a second one.

    ``/v1/completions`` is a compatibility surface over the same operation, so the
    request the upstream receives is the chat one — which is worth pinning, because
    a caller reading only the endpoint name would expect ``/completions``.
    """
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/completions",
            content=JSONContent({"model": "text-default", "prompt": "hi"}),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    assert (await _json(response))["choices"][0]["text"] == CHAT_TEXT
    assert wire.paths() == ["/v1/chat/completions"]


async def test_a_responses_call_is_served_by_the_real_binding():
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/responses",
            content=JSONContent({"model": "text-default", "input": "hi"}),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    assert wire.paths() == ["/v1/chat/completions"]


async def test_an_embedding_is_served_by_the_real_binding():
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/embeddings",
            content=JSONContent({"model": "embedding-default", "input": "hi"}),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    body = await _json(response)
    assert body["data"][0]["embedding"] == [0.25, -0.5, 0.75]
    assert wire.paths() == ["/v1/embeddings"]
    assert wire.last_body()["model"] == "upstream-embed"


async def test_an_image_generation_is_served_by_the_real_binding():
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/images/generations",
            content=JSONContent({"model": "image-default", "prompt": "a red fox"}),
            headers=AUTHORIZATION,
        )

    assert response.status == 200
    assert wire.paths() == ["/v1/images/generations"]
    assert wire.last_body()["prompt"] == "a red fox"


# --------------------------------------------------------------------------- #
# What the public surface must never carry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/completions", {"model": "text-default", "prompt": "hi"}),
        ("/v1/responses", {"model": "text-default", "input": "hi"}),
        ("/v1/embeddings", {"model": "embedding-default", "input": "hi"}),
        ("/v1/images/generations", {"model": "image-default", "prompt": "a red fox"}),
    ],
)
async def test_no_answer_carries_the_provider_credential_or_the_upstream_model(path, payload):
    """Acceptance criterion 12, on the wire the caller actually reads."""
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(path, content=JSONContent(payload), headers=AUTHORIZATION)

    assert response.status == 200
    text = await response.text()
    assert UPSTREAM_SECRET not in text
    assert UPSTREAM_ENDPOINT not in text
    assert "upstream-text" not in text


@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [
        # The gateway composed a request the upstream rejected: the caller did
        # nothing wrong, so this is the gateway's own failure, not a 400.
        (400, 500),
        # The gateway's configured credential was rejected: an operator
        # misconfiguration, and the same reasoning applies.
        (401, 500),
        # The upstream's rate limit is resource exhaustion, and reported as such.
        (429, 429),
        # The upstream failed: unavailable, and retryable by the caller.
        (500, 503),
        (503, 503),
    ],
)
async def test_an_upstream_failure_becomes_the_gateways_own_error_contract(
    upstream_status, expected_status
):
    """A provider refusal is mapped, sanitised and answered with a status, not raised.

    The upstream's own message and code are not what the caller reads: the
    gateway's taxonomy decides both the status and the payload (PRD section 5.4).
    """
    wire = UpstreamWire(status=upstream_status)

    async with _running(wire) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent(
                {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
            ),
            headers=AUTHORIZATION,
        )

    assert response.status == expected_status
    body = await _json(response)
    assert body["reason"]
    assert "message" in body
    text = await response.text()
    assert UPSTREAM_SECRET not in text
    assert "the upstream refused this request" not in text
    assert "invalid_api_key" not in text


async def test_an_unknown_model_reaches_no_upstream():
    """Acceptance criterion 13, on the public path."""
    wire = UpstreamWire()

    async with _running(wire) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent(
                {"model": "not-configured", "messages": [{"role": "user", "content": "hi"}]}
            ),
            headers=AUTHORIZATION,
        )

    assert response.status == 404
    assert wire.paths() == []
