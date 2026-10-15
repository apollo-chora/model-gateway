"""The inbound body-size limit (PRD section 14).

Section 14 requires request-body size limits, and this is the suite for the one the
HTTP transport applies. Three things are worth proving separately: that the bound
is configuration and a deployment cannot configure it away, that a request which
*states* an oversized length is refused without the gateway buffering a byte of it,
and that a request which states no length is measured rather than waved through —
because a bound a caller evades by declining to state a size is not a bound.

**What this suite proves and what it does not.** It proves the limit's decisions
once a request is available to a middleware: an oversized declared length is
refused without the body being read, a body that declares no length is measured,
and the gRPC listener refuses an oversized message. It does **not** establish when
the HTTP server begins buffering a real network body, because the requests here are
constructed rather than received: a middleware is given a body the server has
already read, so what the limit bounds is what the gateway accepts and processes,
not the cost of receiving it. Framing guarantees — where a message ends, whether a
chunked body is well formed, and whether a proxy in front agrees — are the server's
and the deployment's, and no middleware can repair a disagreement about them after
the fact.
"""

import grpc
import pytest
from blacksheep import JSONContent, Request, Response
from blacksheep.contents import Content
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.config import ConfigurationError, Settings
from model_gateway.errors import ErrorReason, PayloadTooLargeError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.server import GrpcServer
from model_gateway.http.body_limit import create_body_limit_middleware
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

BOUND = 1024
"""The bound these tests configure, small enough to exceed with a readable payload."""

ENVIRONMENT = {
    "TEXT_LLM_ENDPOINT": "https://93.184.216.34/v1",
    "TEXT_LLM_MODEL_ID": "upstream-text",
    "TEXT_LLM_API_KEY": "sk-upstream-secret",
    "TEXT_LLM_FORMAT": "openai",
    "TEXT_LLM_CAPABILITIES": "chat",
    "GATEWAY_MAX_REQUEST_BODY_BYTES": str(BOUND),
}
"""A literal public address, because the registry screens the endpoint host at
configuration time through the system resolver and a test that needs DNS is not
deterministic. The bound is what this suite is about; the route only has to parse."""


# --------------------------------------------------------------------------- #
# The bound is configuration, and it cannot be configured away
# --------------------------------------------------------------------------- #


def test_the_deployment_that_configures_nothing_still_gets_a_bound():
    """Section 14's rule that production must not silently fall back to insecure.

    A body limit that only exists when someone remembers to set it is not a limit:
    the deployment that never thought about it is the one that needs it, and it is
    the one a default protects.
    """
    assert Settings.from_env({}).max_request_body_bytes > 0


def test_a_configured_bound_is_the_bound_that_applies():
    assert Settings.from_env(ENVIRONMENT).max_request_body_bytes == BOUND


@pytest.mark.parametrize("value", ["0", "-1", "not a number", "", "1.5"])
def test_a_bound_that_is_not_a_positive_whole_number_is_refused_at_startup(value):
    """Zero is refused rather than read as "no limit".

    A limit of zero bytes refuses every request and a limit that is absent is the
    behaviour section 14 requires the gateway not to have, so neither is a
    configuration a deployment may state: the setting tunes the bound rather than
    removing it.
    """
    with pytest.raises(ConfigurationError):
        Settings.from_env({**ENVIRONMENT, "GATEWAY_MAX_REQUEST_BODY_BYTES": value})


# --------------------------------------------------------------------------- #
# The middleware, on its own
# --------------------------------------------------------------------------- #


class _BodyThatMustNotBeRead(Content):
    """A body whose read raises, so a test can prove the body was never read."""

    def __init__(self) -> None:
        super().__init__(b"application/json", b"")

    async def read(self) -> bytes:
        raise AssertionError("the body was read, and an oversized request must not be")

    async def stream(self):
        raise AssertionError("the body was streamed, and an oversized request must not be")


async def _handler(request: Request) -> Response:
    """A handler that reports whether it ran, and what it was handed."""
    body = await request.read()
    return Response(200, None, Content(b"text/plain", b"handled:" + str(len(body or b"")).encode()))


def _middleware(maximum_bytes: int = BOUND):
    return create_body_limit_middleware(maximum_bytes)


def _post(headers, body: bytes | None = None) -> Request:
    request = Request("POST", b"/v1/chat/completions", headers)
    if body is not None:
        request.content = Content(b"application/json", body)
    return request


async def test_a_declared_length_over_the_bound_is_refused_without_reading_the_body():
    """The cheapest refusal is the one that never buffers what it refuses."""
    request = _post([(b"content-length", str(BOUND + 1).encode())])
    request.content = _BodyThatMustNotBeRead()

    with pytest.raises(PayloadTooLargeError):
        await _middleware()(request, _handler)


async def test_a_declared_length_within_the_bound_reaches_the_handler():
    request = _post([(b"content-length", str(BOUND).encode())], b"x" * BOUND)

    response = await _middleware()(request, _handler)

    assert response.status == 200


async def test_a_body_that_declares_no_length_is_measured():
    """A chunked body states no size, so the only way to bound it is to measure it."""
    request = _post([], b"x" * (BOUND + 1))

    with pytest.raises(PayloadTooLargeError) as refusal:
        await _middleware()(request, _handler)

    assert "received" in refusal.value.details["observed"]


async def test_a_body_that_declares_no_length_and_fits_reaches_the_handler():
    request = _post([], b"x" * BOUND)

    response = await _middleware()(request, _handler)

    assert response.status == 200
    assert await response.text() == f"handled:{BOUND}"


async def test_a_declared_length_the_gateway_cannot_parse_is_measured_rather_than_trusted():
    """An unreadable length is not evidence of a small body."""
    request = _post([(b"content-length", b"not a number")], b"x" * (BOUND + 1))

    with pytest.raises(PayloadTooLargeError):
        await _middleware()(request, _handler)


async def test_the_refusal_names_the_bound_and_carries_no_part_of_the_body():
    """A refusal is the gateway's own words: the body is what it refused to read."""
    secret = "sk-caller-secret-in-the-body"
    request = _post([], (secret + "x" * BOUND).encode())

    with pytest.raises(PayloadTooLargeError) as refusal:
        await _middleware()(request, _handler)

    error = refusal.value
    assert error.reason is ErrorReason.PAYLOAD_TOO_LARGE
    assert error.http_status == 413
    assert error.details["limit_bytes"] == str(BOUND)
    assert secret not in error.message
    assert secret not in str(error.to_payload(include_details=True))


# --------------------------------------------------------------------------- #
# Through the application the factory builds
# --------------------------------------------------------------------------- #


def _app():
    from model_gateway.app import create_app

    settings = Settings.from_env(ENVIRONMENT)
    registry = ModelRegistry.from_env(dict(ENVIRONMENT))
    service = InferenceService(registry, MockProvider())
    # Its own router: the framework keys applications by their router's identity,
    # so an application built on the framework's default router shares it with
    # every other such application in the process and the second one built resets
    # the first's routes. A suite that builds more than one gives each its own.
    return create_app(
        settings, registry=registry, service=service, router=Router()
    )


def _chat(content: str) -> dict[str, object]:
    return {"model": "text-default", "messages": [{"role": "user", "content": content}]}


async def test_a_request_over_the_bound_is_answered_with_its_own_status():
    app = _app()
    await app.start()
    try:
        response = await TestClient(app).post(
            "/v1/chat/completions",
            content=JSONContent(_chat("x" * (BOUND * 4))),
        )
    finally:
        await app.stop()

    assert response.status == 413
    body = await response.json()
    assert body["reason"] == ErrorReason.PAYLOAD_TOO_LARGE.value


async def test_a_request_within_the_bound_is_served():
    app = _app()
    await app.start()
    try:
        response = await TestClient(app).post(
            "/v1/chat/completions",
            content=JSONContent(_chat("hello")),
        )
    finally:
        await app.stop()

    assert response.status == 200


# --------------------------------------------------------------------------- #
# The same bound on the gRPC transport
# --------------------------------------------------------------------------- #


async def test_a_grpc_message_over_the_bound_is_refused_by_the_listener():
    """One configured value bounds both transports, in each one's own unit.

    gRPC enforces its own receive bound and reports the condition as
    ``RESOURCE_EXHAUSTED``, which is the status the taxonomy maps for a payload the
    gateway will not carry — so the two transports agree on what happened even
    though each enforces the bound itself.
    """
    settings = Settings.from_env(ENVIRONMENT)
    registry = ModelRegistry.from_env(dict(ENVIRONMENT))
    service = InferenceService(registry, MockProvider())
    server = GrpcServer(
        service=service,
        registry=registry,
        settings=settings,
        api_keys=ApiKeySet.from_settings(settings),
    )
    request = pb.InvokeRequest(
        model="text-default",
        messages=(pb.Message(role=pb.ROLE_USER, content=()),),
        response_modality=pb.RESPONSE_MODALITY_TEXT,
    )
    request.messages[0].content.add().text.text = "x" * (BOUND * 4)

    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            stub = gateway_pb2_grpc.ModelGatewayServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as refusal:
                await stub.Invoke(request)
    finally:
        await server.stop()

    assert refusal.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
