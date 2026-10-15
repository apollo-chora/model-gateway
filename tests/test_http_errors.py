"""Error-mapping tests: the taxonomy onto HTTP, and the sanitising backstop."""

import pytest
from blacksheep import Application, Content
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    FailedPreconditionError,
    GatewayError,
    InternalError,
    InvalidArgumentError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.http.errors import register_error_handlers

SECRET = "sk-live-do-not-publish"
UPSTREAM = "https://upstream.example/v1"


def errors() -> dict[str, GatewayError]:
    """A fresh instance per request, so one test cannot carry state into another."""
    return {
        "invalid_argument": InvalidArgumentError("the request body must be a JSON object"),
        "authentication_required": UnauthenticatedError("no credential was presented"),
        "invalid_credentials": UnauthenticatedError(
            "the credential was rejected", reason="invalid_credentials"
        ),
        "permission_denied": PermissionDeniedError("this tenant may not use that model"),
        "model_not_found": ModelNotFoundError("the gateway does not expose that model"),
        "resource_not_found": NotFoundError("no such resource"),
        "rate_limited": ResourceExhaustedError("the request rate limit was reached"),
        "concurrency": ResourceExhaustedError(
            "too many requests are in flight", reason="concurrency_limit_exceeded"
        ),
        "failed_precondition": FailedPreconditionError("the gateway is not configured"),
        "deadline": DeadlineExceededError("the invocation deadline expired"),
        "unavailable": UnavailableError("no configured route could serve the request"),
        "internal": InternalError("the gateway could not complete the request"),
        "capability": CapabilityError(
            "this endpoint cannot stream a chat completion",
            details={"capability": "streaming"},
        ),
    }


def build_app() -> Application:
    """An application carrying only the error handlers and the routes these tests probe.

    Each application gets its own router: BlackSheep's default router is shared
    process-wide and reset whenever another ``Application`` is built on it, which
    would let one test's routes disappear under another's.
    """
    app = Application(router=Router())
    register_error_handlers(app)

    @app.router.get("/error/{name}")
    async def raise_named(request, name: str):
        raise errors()[name]

    @app.router.get("/leaky")
    async def leaky(request):
        raise InvalidArgumentError(f"upstream {UPSTREAM} rejected api_key={SECRET}")

    @app.router.get("/boom")
    async def boom(request):
        raise RuntimeError(f"provider rejected the call to {UPSTREAM} with api_key={SECRET}")

    @app.router.post("/echo-json")
    async def echo_json(request):
        return await request.json()

    return app


@pytest.fixture
async def client():
    app = build_app()
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("invalid_argument", 400, "invalid_request"),
        ("authentication_required", 401, "authentication_required"),
        ("invalid_credentials", 401, "invalid_credentials"),
        ("permission_denied", 403, "permission_denied"),
        ("model_not_found", 404, "model_not_found"),
        ("resource_not_found", 404, "resource_not_found"),
        ("rate_limited", 429, "rate_limited"),
        ("concurrency", 429, "concurrency_limit_exceeded"),
        ("failed_precondition", 412, "failed_precondition"),
        ("deadline", 504, "deadline_exceeded"),
        ("unavailable", 503, "provider_unavailable"),
        ("internal", 500, "internal_error"),
        ("capability", 422, "unsupported_capability"),
    ],
)
async def test_every_category_maps_to_its_status_and_reason(client, name, status, reason):
    response = await client.get(f"/error/{name}")

    assert response.status == status
    body = await response.json()
    assert body["reason"] == reason
    assert isinstance(body["message"], str)
    assert body["message"]


async def test_the_error_body_is_json(client):
    response = await client.get("/error/model_not_found")

    assert response.content_type() == b"application/json"


async def test_the_public_payload_does_not_carry_details_by_default(client):
    response = await client.get("/error/capability")

    assert "details" not in await response.json()


async def test_an_unexpected_exception_becomes_a_sanitised_internal_error(client):
    response = await client.get("/boom")

    assert response.status == 500
    body = await response.json()
    assert body["reason"] == "internal_error"
    assert body["message"] == "the gateway could not complete the request"


async def test_an_unexpected_exception_leaks_nothing_about_the_upstream(client):
    body = await (await client.get("/boom")).text()

    assert SECRET not in body
    assert UPSTREAM not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body
    assert "api_key" not in body


async def test_a_gateway_error_message_is_sanitised_at_construction(client):
    response = await client.get("/leaky")

    body = await response.text()
    assert response.status == 400
    assert SECRET not in body
    assert UPSTREAM not in body


async def test_the_framework_still_owns_its_own_http_exceptions(client):
    response = await client.get("/no-such-route")

    assert response.status == 404


async def test_a_body_the_framework_cannot_parse_stays_a_400(client):
    response = await client.post(
        "/echo-json",
        content=Content(b"application/json", b"{not json"),
    )

    assert response.status == 400
