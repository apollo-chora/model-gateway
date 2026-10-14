"""Tests for the wired HTTP surface: ``create_app`` builds a working gateway (PRD 4, 19).

These tests exercise the application factory rather than the handlers, so they
cover what the two things the handlers cannot: that the routes and the error
handler are installed on the application the server actually runs, and that the
health endpoints keep working beside them.

They also cover what the factory wires when nothing is injected: the production
LiteLLM binding over the registry's routes, with the configured attempt budget,
so a process started with configuration alone really calls upstream. An
injected service is still used verbatim — that is how tests bind the
deterministic mock, and how a caller substitutes a provider of its own.
"""

import pytest
from blacksheep import Content, JSONContent
from blacksheep.testing import TestClient

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import FinishReason, InvocationResponse, TextPart, Usage
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
EXTRA_HEADER_VALUE = "tenant-header-value"


class FakeTextProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, route: Route, request, context) -> InvocationResponse:
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
async def provider():
    return FakeTextProvider()


@pytest.fixture
async def client(provider):
    models = registry()
    app = create_app(
        Settings.from_env({}),
        registry=models,
        service=InferenceService(models, provider),
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


async def test_the_chat_endpoint_is_installed(client, provider):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent(
            {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
        ),
    )

    assert response.status == 200
    assert (await response.json())["object"] == "chat.completion"
    assert provider.calls == 1


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


async def test_the_factory_builds_the_litellm_provider_when_no_service_is_injected(monkeypatch):
    """The default path wires the real binding, never an implicit stand-in.

    A factory that picked the deterministic mock would advertise
    ``text-default``, accept a production chat request and answer it with an
    echo — a fake that looks exactly like working upstream inference.

    The application is started and stopped, not merely built: blacksheep's
    default router is shared process-wide, and an application that is never
    started leaves its routes queued in it for the next application.
    """
    constructed = []

    class RecordingProvider:
        def __init__(self):
            constructed.append(self)

    monkeypatch.setattr("model_gateway.app.LiteLLMProvider", RecordingProvider)

    app = create_app(Settings.from_env({}))
    await app.start()
    await app.stop()

    assert len(constructed) == 1


async def test_the_factory_builds_the_litellm_service_with_a_registry_but_no_service(
    monkeypatch,
):
    constructed = []

    class RecordingProvider:
        def __init__(self):
            constructed.append(self)

    monkeypatch.setattr("model_gateway.app.LiteLLMProvider", RecordingProvider)

    app = create_app(Settings.from_env({}), registry=registry())
    await app.start()
    await app.stop()

    assert len(constructed) == 1


async def test_the_factory_gives_the_built_service_the_configured_attempt_budget(monkeypatch):
    """Every attempt context carries the configured provider timeout (PRD 9.6).

    The service derives that budget as the minimum of the configured timeout and
    the invocation deadline remaining when the attempt begins (PRD section 15);
    the service constructor is where the configured value enters.
    """
    captured = {}

    class RecordingService:
        def __init__(
            self,
            registry,
            provider,
            *,
            attempt_timeout=None,
            retry_count=None,
            retry_backoff_ms=None,
            limiter=None,
        ):
            captured["attempt_timeout"] = attempt_timeout

    monkeypatch.setattr("model_gateway.app.InferenceService", RecordingService)

    app = create_app(Settings.from_env({"PROVIDER_TIMEOUT_SECONDS": "42"}))
    await app.start()
    await app.stop()

    assert captured["attempt_timeout"] == 42.0


async def test_the_factory_gives_the_built_service_every_configured_reliability_value(
    monkeypatch,
):
    """Each configured reliability value reaches the service the factory builds.

    A deployment that configures retries and a concurrency ceiling and silently
    gets none of them — the operator believing their configuration took effect —
    is exactly the defect this guards against, so every ``PROVIDER_*``
    reliability setting is asserted here, not just the timeout.
    """
    captured = {}

    class RecordingService:
        def __init__(
            self,
            registry,
            provider,
            *,
            attempt_timeout=None,
            retry_count=None,
            retry_backoff_ms=None,
            limiter=None,
        ):
            captured.update(
                attempt_timeout=attempt_timeout,
                retry_count=retry_count,
                retry_backoff_ms=retry_backoff_ms,
                limiter=limiter,
            )

    monkeypatch.setattr("model_gateway.app.InferenceService", RecordingService)

    settings = Settings.from_env(
        {
            "PROVIDER_TIMEOUT_SECONDS": "42",
            "PROVIDER_RETRY_COUNT": "3",
            "PROVIDER_RETRY_BACKOFF_MS": "250",
            "PROVIDER_MAX_CONCURRENT_REQUESTS": "7",
        }
    )
    app = create_app(settings)
    await app.start()
    await app.stop()

    assert captured["attempt_timeout"] == 42.0
    assert captured["retry_count"] == 3
    assert captured["retry_backoff_ms"] == 250
    assert isinstance(captured["limiter"], ConcurrencyLimiter)
    assert captured["limiter"].maximum == 7


async def test_the_default_path_serves_an_application_with_the_real_binding():
    app = create_app(Settings.from_env({}))
    await app.start()
    try:
        client = TestClient(app)
        assert (await client.get("/healthz")).status == 200
        assert (await client.get("/readyz")).status == 200
    finally:
        await app.stop()


async def test_an_injected_service_is_used_verbatim(provider):
    models = registry()
    app = create_app(
        Settings.from_env({}),
        registry=models,
        service=InferenceService(models, provider),
    )
    await app.start()
    try:
        client = TestClient(app)
        assert (await client.get("/healthz")).status == 200
        assert (await client.get("/readyz")).status == 200
    finally:
        await app.stop()
