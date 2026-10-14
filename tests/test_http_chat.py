"""Tests for the chat completions endpoint (PRD sections 4.4, 9.5 and 16)."""

import pytest
from blacksheep import Application, Content, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationResponse,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import CapabilityError, UnavailableError
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"


def make_registry() -> ModelRegistry:
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
                capabilities=frozenset({Capability.CHAT, Capability.TOOLS}),
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


def completion(**overrides) -> InvocationResponse:
    fields = {
        "model": "text-default",
        "upstream_model": "upstream-text",
        "content": (TextPart("Recursion is a technique..."),),
        "finish_reason": FinishReason.STOP,
        "raw_finish_reason": "stop",
        "usage": Usage(input_tokens=25, output_tokens=42, total_tokens=67),
    }
    fields.update(overrides)
    return InvocationResponse(**fields)


class FakeTextProvider:
    """A ``TextProvider`` double: records the calls and returns a canned response.

    The gateway's HTTP surface is what these tests are about, so the provider
    stands in for the upstream at the binding boundary and no socket is opened.
    """

    def __init__(self, result: InvocationResponse | None = None, error: Exception | None = None):
        self.calls: list[tuple[Route, object]] = []
        self.result = result if result is not None else completion()
        self.error = error

    async def complete(self, route: Route, request, context) -> InvocationResponse:
        self.calls.append((route, request))
        if self.error is not None:
            raise self.error
        return self.result


def build_app(provider: FakeTextProvider) -> Application:
    registry = make_registry()
    app = Application(router=Router())
    register_error_handlers(app)
    register_chat_routes(app.router, InferenceService(registry, provider))
    return app


@pytest.fixture
async def provider():
    return FakeTextProvider()


@pytest.fixture
async def client(provider):
    app = build_app(provider)
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


def payload(**overrides):
    body = {
        "model": "text-default",
        "messages": [{"role": "user", "content": "Explain recursion."}],
    }
    body.update(overrides)
    return body


async def post(client, body):
    return await client.post("/v1/chat/completions", content=JSONContent(body))


async def test_happy_path_returns_the_documented_response_shape(client):
    response = await post(client, payload(temperature=0.7, max_tokens=2048, stream=False))

    assert response.status == 200
    body = await response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "text-default"
    assert isinstance(body["created"], int)
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Recursion is a technique..."},
            "finish_reason": "stop",
        }
    ]
    assert body["usage"] == {"prompt_tokens": 25, "completion_tokens": 42, "total_tokens": 67}


async def test_happy_path_response_is_json(client):
    response = await post(client, payload())

    assert response.content_type() == b"application/json"


async def test_every_message_role_reaches_the_provider_as_canonical_content(client, provider):
    body = payload(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "developer", "content": "Answer briefly."},
            {"role": "user", "content": "Explain recursion."},
            {"role": "assistant", "content": "Recursion is a technique..."},
            {"role": "tool", "content": "42", "tool_call_id": "call_1"},
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert [message.role.value for message in invocation.messages] == [
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    ]
    assert invocation.messages[4].tool_call_id == "call_1"


async def test_the_invocation_carries_the_generation_parameters(client, provider):
    await post(client, payload(temperature=0.7, top_p=0.9, max_tokens=2048, stop=["END"]))

    _, invocation = provider.calls[0]
    assert invocation.generation_config.temperature == 0.7
    assert invocation.generation_config.top_p == 0.9
    assert invocation.generation_config.max_output_tokens == 2048
    assert invocation.generation_config.stop_sequences == ("END",)


async def test_tool_declarations_reach_the_provider(client, provider):
    body = payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="required",
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.tools[0].name == "get_weather"
    assert invocation.tool_choice.mode.value == "required"


async def test_a_tool_call_answer_is_returned_with_null_content(client):
    provider = FakeTextProvider(
        completion(
            content=(ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    choice = (await response.json())["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert choice["finish_reason"] == "tool_calls"


async def test_an_unknown_model_is_a_model_not_found_error(client, provider):
    response = await post(client, payload(model="no-such-model"))

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"
    assert provider.calls == []


async def test_a_streaming_request_is_refused_as_an_unsupported_capability(client, provider):
    response = await post(client, payload(stream=True))

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "stream" in body["message"]
    assert provider.calls == []


async def test_a_streaming_request_never_returns_a_whole_body(client):
    body = await (await post(client, payload(stream=True))).text()

    assert "chat.completion" not in body
    assert "choices" not in body


async def test_a_remote_image_url_is_refused_before_the_provider_is_called(client, provider):
    body = payload(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://169.254.169.254/latest/meta-data/"},
                    },
                ],
            }
        ]
    )

    response = await post(client, body)

    assert response.status == 422
    assert (await response.json())["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_an_interleaved_provider_answer_is_refused_rather_than_flattened(client):
    provider = FakeTextProvider(
        completion(
            content=(
                TextPart("Checking the weather."),
                ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
                TextPart("It is sunny."),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    assert response.status == 422
    body = await response.text()
    assert (await response.json())["reason"] == "unsupported_capability"
    assert "It is sunny." not in body


async def test_an_explicit_parallel_tool_calls_flag_is_refused(client, provider):
    response = await post(client, payload(parallel_tool_calls=True))

    assert response.status == 422
    assert (await response.json())["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_a_body_that_is_not_json_is_refused(client):
    response = await client.post(
        "/v1/chat/completions",
        content=Content(b"application/json", b"{not json"),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_a_content_type_is_refused(client):
    response = await client.post(
        "/v1/chat/completions",
        content=Content(b"text/plain", b'{"model": "text-default", "messages": []}'),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_that_is_not_an_object_is_refused(client):
    response = await client.post("/v1/chat/completions", content=JSONContent([1, 2, 3]))

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_messages_is_refused(client, provider):
    response = await post(client, {"model": "text-default"})

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"
    assert provider.calls == []


async def test_a_body_with_an_unknown_parameter_is_refused(client, provider):
    response = await post(client, payload(verbosity="high"))

    assert response.status == 400
    assert "verbosity" in (await response.json())["message"]
    assert provider.calls == []


async def test_a_capability_error_from_the_service_reaches_the_caller(client):
    provider = FakeTextProvider(error=CapabilityError("the route cannot honour tools"))
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    assert response.status == 422
    assert (await response.json())["reason"] == "unsupported_capability"


async def test_a_provider_outage_is_reported_as_unavailable(client):
    provider = FakeTextProvider(error=UnavailableError("no configured route could serve this"))
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    assert response.status == 503
    assert (await response.json())["reason"] == "provider_unavailable"


async def test_an_unexpected_provider_failure_is_sanitised(client):
    provider = FakeTextProvider(
        error=RuntimeError(f"upstream {UPSTREAM_ENDPOINT} rejected api_key={SECRET_KEY}")
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    assert response.status == 500
    body = await response.text()
    assert SECRET_KEY not in body
    assert UPSTREAM_ENDPOINT not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body


async def test_no_response_body_carries_upstream_configuration(client):
    bodies = [
        await (await post(client, payload())).text(),
        await (await post(client, payload(model="no-such-model"))).text(),
        await (await post(client, payload(stream=True))).text(),
        await (await post(client, {"model": "text-default"})).text(),
    ]

    for body in bodies:
        assert SECRET_KEY not in body
        assert UPSTREAM_ENDPOINT not in body
        assert "upstream-text" not in body
