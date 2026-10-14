"""Tests for the wired HTTP surface: ``create_app`` builds a working gateway (PRD 4, 19).

These tests exercise the application factory rather than the handlers, so they
cover what the two things the handlers cannot: that the routes and the error
handler are installed on the application the server actually runs, and that the
health endpoints keep working beside them.
"""

import pytest
from blacksheep import Content, JSONContent
from blacksheep.testing import TestClient
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import FinishReason, InvocationResponse, TextPart, Usage

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
EXTRA_HEADER_VALUE = "tenant-header-value"


class FakeTextProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, route: Route, request) -> InvocationResponse:
        self.calls += 1
        return InvocationResponse(
            model="text-default",
            upstream_model="upstream-text",
            content=(TextPart("Recursion is a technique..."),),
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=Usage(input_tokens=25, output_tokens=42, total_tokens=67),
        )


def registry() -> ModelRegistry:
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=UPSTREAM_ENDPOINT,
                base_url=UPSTREAM_ENDPOINT,
                model_id="upstream-text",
                format="openai",
                api_key=SECRET_KEY,
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={"x-tenant": EXTRA_HEADER_VALUE},
                allow_absolute_path_url=False,
            ),
        )
    )


@pytest.fixture
async def client():
    models = registry()
    app = create_app(
        Settings.from_env({}),
        registry=models,
        service=InferenceService(models, FakeTextProvider()),
    )
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


async def test_health_still_answers_beside_the_inference_routes(client):
    response = await client.get("/healthz")

    assert response.status == 200
    assert await response.text() == "ok"


async def test_readiness_still_answers_beside_the_inference_routes(client):
    response = await client.get("/readyz")

    assert response.status == 200
    assert await response.json() == {"status": "ready", "checks": {}}


async def test_the_model_endpoints_are_installed(client):
    response = await client.get("/v1/models")

    assert response.status == 200
    assert [entry["id"] for entry in (await response.json())["data"]] == ["text-default"]


async def test_the_chat_endpoint_is_installed(client):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent(
            {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
        ),
    )

    assert response.status == 200
    assert (await response.json())["object"] == "chat.completion"


async def test_the_error_handler_is_installed(client):
    response = await client.get("/v1/models/no-such-model")

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"


async def test_no_public_response_carries_upstream_configuration(client):
    responses = [
        await client.get("/v1/models"),
        await client.get("/v1/models/text-default"),
        await client.get("/v1/models/no-such-model"),
        await client.post(
            "/v1/chat/completions",
            content=JSONContent(
                {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
            ),
        ),
        await client.post(
            "/v1/chat/completions",
            content=JSONContent({"model": "text-default", "messages": [], "stream": True}),
        ),
        await client.post(
            "/v1/chat/completions",
            content=Content(b"application/json", b"{not json"),
        ),
    ]

    for response in responses:
        body = await response.text()
        assert SECRET_KEY not in body
        assert UPSTREAM_ENDPOINT not in body
        assert EXTRA_HEADER_VALUE not in body
        assert "upstream-text" not in body


async def test_a_default_application_starts_and_serves_health():
    app = create_app(Settings.from_env({}))
    await app.start()
    try:
        client = TestClient(app)
        assert (await client.get("/healthz")).status == 200
        assert (await client.get("/readyz")).status == 200
    finally:
        await app.stop()
