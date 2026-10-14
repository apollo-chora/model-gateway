"""Tests for the chat completions endpoint (PRD sections 4.4, 9.5 and 16)."""

import asyncio
import json

import pytest
from blacksheep import Application, Content, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationResponse,
    StreamEvent,
    StreamEventKind,
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


class FakeStreamingProvider:
    """A provider double that can both complete and stream (PRD 16).

    ``complete`` behaves like :class:`FakeTextProvider`. ``stream`` yields one
    content event per scripted delta and then a finish event, or raises the
    scripted error — before any event if ``stream_error`` is set, or after the
    deltas if an exception appears among them. Setting ``finish`` to false omits
    the trailing finish event, so the stream ends without a terminal event. The
    invocation id on every event is the one the handler minted for the request.
    """

    def __init__(
        self,
        *,
        deltas: list[object] | None = None,
        stream_error: Exception | None = None,
        error: Exception | None = None,
        result: InvocationResponse | None = None,
        finish: bool = True,
    ):
        self.calls: list[tuple[Route, object]] = []
        self.result = result if result is not None else completion()
        self.error = error
        self.deltas = deltas
        self.stream_error = stream_error
        self.finish = finish

    async def complete(self, route: Route, request, context) -> InvocationResponse:
        self.calls.append((route, request))
        if self.error is not None:
            raise self.error
        return self.result

    async def stream(self, route: Route, request, context):
        self.calls.append((route, request))
        if self.stream_error is not None:
            raise self.stream_error
        invocation_id = request.invocation_id or ""
        items = self.deltas
        if items is None:
            text = self.result.text
            items = [text] if text else []
        sequence = 0
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=sequence,
                invocation_id=invocation_id,
                text=item,
            )
            sequence += 1
        if not self.finish:
            return
        yield StreamEvent(
            kind=StreamEventKind.FINISH,
            sequence=sequence,
            invocation_id=invocation_id,
            finish_reason=self.result.finish_reason,
            raw_finish_reason=self.result.raw_finish_reason,
        )


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


async def test_a_streaming_request_is_served_as_server_sent_events():
    """A streaming request is answered with SSE in the OpenAI chunk shape, ended by [DONE]."""
    provider = FakeStreamingProvider()
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 200
    assert response.content_type() == b"text/event-stream"
    frames = _sse_data_frames(await response.text())
    # The stream is terminated by [DONE], because it ended with a finish event.
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks, "a streaming answer carries at least one chunk"
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "text-default"
        assert chunk["id"].startswith("chatcmpl-")
    # The conventional assistant-role frame leads the stream.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    content_chunks = [chunk for chunk in chunks if "content" in chunk["choices"][0]["delta"]]
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "Recursion is a technique..."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_the_first_frame_of_a_stream_is_the_conventional_assistant_role_delta():
    """OpenAI-compatible clients expect the assistant role on the first frame."""
    provider = FakeStreamingProvider()
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 200
    frames = _sse_data_frames(await response.text())
    first = json.loads(frames[0])
    assert first["choices"] == [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
    ]
    # The role frame carries neither an error nor usage.
    assert "error" not in first
    assert "usage" not in first


async def test_a_streaming_request_never_returns_a_whole_body():
    """A streaming request is answered with chunks, not a single complete body."""
    provider = FakeStreamingProvider()
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 200
    body = await response.text()
    assert "chat.completion\n" not in body
    assert '"object": "chat.completion"' not in body


async def test_a_streaming_request_with_a_provider_that_cannot_stream_is_refused(client, provider):
    """A provider that cannot stream is refused, not faked (PRD 16)."""
    response = await post(client, payload(stream=True))

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_a_streaming_request_with_an_unknown_model_is_refused_before_the_stream_starts():
    """A pre-stream refusal is an ordinary error response, not an SSE stream."""
    provider = FakeStreamingProvider()
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(model="no-such-model", stream=True))
    finally:
        await app.stop()

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"
    assert provider.calls == []


async def test_an_unknown_model_is_a_model_not_found_error_even_without_a_streaming_provider():
    """The model is resolved before streaming eligibility, so the refusal is about the model.

    A provider that cannot stream is refused only for a model that exists; a
    request naming a model the gateway does not expose is a model-not-found
    error, because that is the more accurate answer to the request.
    """
    provider = FakeTextProvider()  # not a streaming provider
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(model="no-such-model", stream=True))
    finally:
        await app.stop()

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"
    assert provider.calls == []


async def test_a_streaming_request_with_a_capability_gap_is_refused_before_the_stream_starts():
    """A capability gap refuses the request before the SSE response starts."""
    provider = FakeStreamingProvider()
    app = build_app(provider)
    await app.start()
    try:
        response = await post(
            TestClient(app),
            payload(stream=True, response_format={"type": "json_object"}),
        )
    finally:
        await app.stop()

    assert response.status == 422
    assert (await response.json())["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_a_mid_stream_failure_ends_the_stream_without_done():
    """A failure after the first chunk ends the stream with an error, not [DONE]."""
    provider = FakeStreamingProvider(
        deltas=["Recursion is a technique...", UnavailableError("the connection dropped")]
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 200
    frames = _sse_data_frames(await response.text())
    # The stream ended without [DONE], because the answer did not complete.
    assert "[DONE]" not in frames
    # The last chunk is the terminal ERROR event: the gateway's own reason and
    # sanitised message, on a chunk with no choice and no finish reason.
    last_chunk = json.loads(frames[-1])
    assert last_chunk["choices"] == []
    assert last_chunk["error"] == {
        "reason": "provider_unavailable",
        "message": "the connection dropped",
    }
    # Nothing follows the terminal event: it is the last frame of the stream.
    assert frames[-1] != "[DONE]"


async def test_a_stream_that_ends_without_a_finish_event_emits_no_done():
    """A stream with no terminal event is not a completed answer, so it is closed plainly."""
    provider = FakeStreamingProvider(deltas=["Recursion is a technique..."], finish=False)
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 200
    frames = _sse_data_frames(await response.text())
    # The upstream never reported a finish, so the client must not be told the
    # answer completed.
    assert "[DONE]" not in frames
    assert frames[-1] != "[DONE]"
    assert json.loads(frames[-1])["choices"][0]["delta"]["content"] == "Recursion is a technique..."


async def test_a_client_disconnect_stops_upstream_work():
    """Closing the SSE stream mid-answer stops the provider's upstream work."""
    provider = _BlockingStreamProvider()
    app = build_app(provider)
    await app.start()
    try:
        client = TestClient(app)
        response = await client.post(
            "/v1/chat/completions", content=JSONContent(payload(stream=True))
        )
        assert response.status == 200

        stream_iter = response.stream().__aiter__()
        chunk = await stream_iter.__anext__()
        assert chunk.startswith(b"data:")
        await provider.started.wait()
        # A client disconnect closes the stream iterator.
        await stream_iter.aclose()
        for _ in range(50):
            await asyncio.sleep(0)
        # The stream was closed mid-stream: the upstream work was stopped.
        assert provider.closed
        assert not provider.completed
    finally:
        await app.stop()


class _BlockingStreamProvider:
    """A streaming provider that yields one event and then blocks until closed.

    ``closed`` records that the stream was closed — normally, or because the
    consumer went away — and ``completed`` records that it ran to its finish
    event. A client disconnect closes the stream before it completes, so the
    upstream work is stopped mid-stream.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False
        self.completed = False

    async def complete(self, route, request, context) -> InvocationResponse:
        return completion()

    async def stream(self, route: Route, request, context):
        self.started.set()
        invocation_id = request.invocation_id or ""
        try:
            yield StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=invocation_id,
                text="Recursion is a technique...",
            )
            await asyncio.Event().wait()
            yield StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=1,
                invocation_id=invocation_id,
                finish_reason=FinishReason.STOP,
                raw_finish_reason="stop",
            )
            self.completed = True
        finally:
            self.closed = True


def _sse_data_frames(body: str) -> list[str]:
    """Extract the ``data`` payloads from a Server-Sent Events body, in order."""
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


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
