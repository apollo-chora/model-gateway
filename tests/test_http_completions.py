"""Tests for the legacy completions endpoint."""

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
    IdempotencyStatus,
    InvocationResponse,
    RefusalPart,
    Role,
    StreamEvent,
    StreamEventKind,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.domain.streaming import ToolCallDelta
from model_gateway.errors import CapabilityError, UnavailableError
from model_gateway.http.completions import TextCompletion, register_completions_routes
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
    """A provider double that can both complete and stream.

    ``complete`` behaves like :class:`FakeTextProvider`. ``stream`` yields one
    event per scripted delta — a ``str`` becomes a content event, a
    :class:`ToolCallDelta` becomes a tool-call event, and an exception is
    raised in its place — then a finish event, or raises the scripted error
    before any event if ``stream_error`` is set. Setting ``finish`` to false
    omits the trailing finish event, so the stream ends without a terminal
    event. The invocation id on every event is the one the handler minted for
    the request.
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
            if isinstance(item, ToolCallDelta):
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL,
                    sequence=sequence,
                    invocation_id=invocation_id,
                    tool_call=item,
                )
            else:
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
    register_completions_routes(app.router, InferenceService(registry, provider))
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
    body = {"model": "text-default", "prompt": "Once upon a time"}
    body.update(overrides)
    return body


async def post(client, body):
    return await client.post("/v1/completions", content=JSONContent(body))


def _sse_data_frames(body: str) -> list[str]:
    """Extract the ``data`` payloads from a Server-Sent Events body, in order."""
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


async def test_happy_path_returns_the_documented_response_shape(client):
    response = await post(client, payload(max_tokens=2048, temperature=0.7))

    assert response.status == 200
    body = await response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "text-default"
    assert isinstance(body["created"], int)
    assert body["id"].startswith("cmpl-")
    assert body["choices"] == [
        {"text": "Recursion is a technique...", "index": 0, "finish_reason": "stop"}
    ]
    assert body["usage"] == {"prompt_tokens": 25, "completion_tokens": 42, "total_tokens": 67}


async def test_happy_path_response_is_json(client):
    response = await post(client, payload())

    assert response.content_type() == b"application/json"


async def test_the_prompt_reaches_the_provider_as_one_user_message(client, provider):
    await post(client, payload(prompt="Explain recursion."))

    _, invocation = provider.calls[0]
    assert len(invocation.messages) == 1
    assert invocation.messages[0].role is Role.USER
    assert isinstance(invocation.messages[0].content[0], TextPart)
    assert invocation.messages[0].content[0].text == "Explain recursion."


async def test_the_invocation_carries_the_generation_parameters(client, provider):
    await post(client, payload(max_tokens=2048, temperature=0.7, top_p=0.9, stop=["END"]))

    _, invocation = provider.calls[0]
    assert invocation.generation_config.max_output_tokens == 2048
    assert invocation.generation_config.temperature == 0.7
    assert invocation.generation_config.top_p == 0.9
    assert invocation.generation_config.stop_sequences == ("END",)


async def test_an_array_prompt_is_refused_rather_than_narrowed(client, provider):
    """An array prompt asks for one completion per element, which this shape cannot carry."""
    response = await post(client, payload(prompt=["First prompt.", "Second prompt."]))

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "prompt" in body["message"]
    assert provider.calls == []


async def test_an_unknown_model_is_a_model_not_found_error(client, provider):
    response = await post(client, payload(model="no-such-model"))

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"
    assert provider.calls == []


async def test_a_tool_call_answer_is_refused(client):
    """A tool call has no place in the legacy choice shape, so the answer is refused."""
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

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "tool call" in body["message"]
    assert "get_weather" not in await response.text()


async def test_a_refusal_answer_is_refused(client):
    """A refusal has no place in the legacy choice shape, so the answer is refused."""
    provider = FakeTextProvider(
        completion(
            content=(RefusalPart("I cannot help with that."),),
            finish_reason=FinishReason.CONTENT_FILTER,
            raw_finish_reason="content_filter",
        )
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload())
    finally:
        await app.stop()

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "refusal" in body["message"]
    assert "I cannot help with that." not in await response.text()


@pytest.mark.parametrize(
    ("overrides", "feature"),
    [
        ({"n": 2}, "multiple_choices"),
        ({"logprobs": True}, "log_probabilities"),
        ({"echo": True}, "echo"),
        ({"suffix": " the end"}, "suffix"),
        ({"best_of": 4}, "best_of"),
    ],
)
async def test_an_unsupported_parameter_is_refused_with_a_capability_error(
    client, provider, overrides, feature
):
    response = await post(client, payload(**overrides))

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert feature in body["message"]
    assert provider.calls == []


@pytest.mark.parametrize(
    "overrides",
    [{"n": 1}, {"logprobs": False}, {"echo": False}, {"best_of": 1}, {"suffix": None}],
)
async def test_a_default_value_that_matches_gateway_behaviour_is_accepted(
    client, provider, overrides
):
    response = await post(client, payload(**overrides))

    assert response.status == 200
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("overrides", "json_type"),
    [
        ({"n": True}, "a boolean"),
        ({"n": None}, "null"),
        ({"n": "1"}, "a string"),
        ({"best_of": True}, "a boolean"),
        ({"logprobs": 0}, "a number"),
        ({"echo": 0}, "a number"),
        ({"echo": ""}, "a string"),
        ({"echo": []}, "a JSON array"),
        ({"echo": {}}, "a JSON object"),
        ({"suffix": 5}, "a number"),
        ({"suffix": True}, "a boolean"),
    ],
)
async def test_an_ill_typed_parameter_is_malformed_not_a_capability_refusal(
    client, provider, overrides, json_type
):
    """A value of the wrong wire type asks for no feature, so it is malformed input.

    ``{"n": true}`` is the case Python reads wrongly — ``True == 1`` — and the
    falsy values are the ones generic emptiness reads as asking for nothing.
    """
    response = await post(client, payload(**overrides))

    assert response.status == 400
    body = await response.json()
    assert body["reason"] == "invalid_request"
    assert next(iter(overrides)) in body["message"]
    assert json_type in body["message"]
    assert provider.calls == []


async def test_the_wire_type_refusal_is_not_the_unknown_parameter_refusal(client):
    """An ill-typed recognised parameter is a different error from an unknown key."""
    response = await post(client, payload(echo=0))

    body = await response.json()
    assert response.status == 400
    assert body["reason"] == "invalid_request"
    assert "unsupported parameter" not in body["message"]


async def test_a_body_with_an_unknown_parameter_is_refused(client, provider):
    response = await post(client, payload(verbosity="high"))

    assert response.status == 400
    assert "verbosity" in (await response.json())["message"]
    assert provider.calls == []


async def test_an_idempotency_key_reaches_the_provider(client, provider):
    """The key the caller chose is the invocation's, so a repeat can be matched."""
    response = await post(client, payload(idempotency_key="once-upon-a-time"))

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.idempotency_key == "once-upon-a-time"


@pytest.mark.parametrize("overrides", [{}, {"idempotency_key": None}])
async def test_a_request_without_an_idempotency_key_is_answered_not_applicable(
    client, overrides
):
    """No key means deduplication was never in play, and the answer says so."""
    response = await post(client, payload(**overrides))

    assert response.status == 200
    assert (await response.json())["idempotency_status"] == "not_applicable"


async def test_a_repeated_idempotency_key_is_answered_from_the_gateway_record(
    client, provider
):
    """A repeat is answered from the record, so the provider is asked once."""
    first = await post(client, payload(idempotency_key="once-upon-a-time"))
    second = await post(client, payload(idempotency_key="once-upon-a-time"))

    assert first.status == second.status == 200
    assert (await first.json())["idempotency_status"] == "new"
    assert (await second.json())["idempotency_status"] == "replayed"
    assert len(provider.calls) == 1


@pytest.mark.parametrize("key", [7, True, ["once-upon-a-time"], {"key": "once"}])
async def test_a_non_string_idempotency_key_is_refused_naming_the_field(
    client, provider, key
):
    """The format defines the key as a string, so anything else is malformed input."""
    response = await post(client, payload(idempotency_key=key))

    assert response.status == 400
    body = await response.json()
    assert body["reason"] == "invalid_request"
    assert "idempotency_key" in body["message"]
    assert provider.calls == []


async def test_an_empty_idempotency_key_is_refused_rather_than_read_as_absent(
    client, provider
):
    """An empty string is no key at all, not a key that happens to be empty."""
    response = await post(client, payload(idempotency_key=""))

    assert response.status == 400
    body = await response.json()
    assert body["reason"] == "invalid_request"
    assert "idempotency_key" in body["message"]
    assert provider.calls == []


@pytest.mark.parametrize(
    ("status", "rendered"),
    [
        (IdempotencyStatus.NOT_APPLICABLE, "not_applicable"),
        (IdempotencyStatus.NEW, "new"),
        (IdempotencyStatus.REPLAYED, "replayed"),
        (IdempotencyStatus.INDETERMINATE, "indeterminate"),
    ],
)
def test_the_idempotency_status_is_rendered_as_the_openai_string(status, rendered):
    """The four canonical statuses reach the caller as snake_case strings."""
    body = TextCompletion.from_invocation(
        completion(idempotency_status=status), completion_id="cmpl-1", created=0
    ).to_dict()

    assert body["idempotency_status"] == rendered


async def test_a_streaming_request_is_served_as_server_sent_events():
    """A streaming request is answered with SSE in the legacy chunk shape, ended by [DONE]."""
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
        assert chunk["object"] == "text_completion"
        assert chunk["model"] == "text-default"
        assert chunk["id"].startswith("cmpl-")
    # A content chunk carries the fragment under choices[0].text.
    content_chunks = [chunk for chunk in chunks if chunk["choices"][0]["text"]]
    assert content_chunks[0]["choices"][0]["text"] == "Recursion is a technique..."
    assert content_chunks[0]["choices"][0]["finish_reason"] is None
    # The terminal finish chunk carries the reason with an empty text.
    assert chunks[-1]["choices"][0]["text"] == ""
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


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
    assert "text_completion\n" not in body
    assert '"object": "text_completion"' not in body


async def test_a_streaming_request_with_a_provider_that_cannot_stream_is_refused(client, provider):
    """A provider that cannot stream is refused, not faked."""
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
    assert json.loads(frames[-1])["choices"][0]["text"] == "Recursion is a technique..."


async def test_a_leading_tool_call_is_refused_before_the_stream_starts():
    """A tool call before any output is a capability gap, refused as an ordinary error."""
    provider = FakeStreamingProvider(
        deltas=[ToolCallDelta(index=0, id="call_1", name="get_weather")]
    )
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(stream=True))
    finally:
        await app.stop()

    assert response.status == 422
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "tool call" in body["message"]


async def test_a_mid_stream_tool_call_ends_the_stream_without_done():
    """A tool call after output has begun ends the stream with an error, not [DONE]."""
    provider = FakeStreamingProvider(
        deltas=[
            "Checking the weather.",
            ToolCallDelta(index=0, id="call_1", name="get_weather", arguments_delta='{"city"'),
        ]
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
    # The content the caller already saw stands, and the last chunk is the
    # terminal error event naming the feature the shape cannot carry.
    first_chunk = json.loads(frames[0])
    assert first_chunk["choices"][0]["text"] == "Checking the weather."
    last_chunk = json.loads(frames[-1])
    assert last_chunk["choices"] == []
    assert last_chunk["error"]["reason"] == "unsupported_capability"
    assert "tool call" in last_chunk["error"]["message"]


async def test_a_client_disconnect_stops_upstream_work():
    """Closing the SSE stream mid-answer stops the provider's upstream work."""
    provider = _BlockingStreamProvider()
    app = build_app(provider)
    await app.start()
    try:
        client = TestClient(app)
        response = await client.post(
            "/v1/completions", content=JSONContent(payload(stream=True))
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


async def test_a_body_that_is_not_json_is_refused(client):
    response = await client.post(
        "/v1/completions",
        content=Content(b"application/json", b"{not json"),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_a_content_type_is_refused(client):
    response = await client.post(
        "/v1/completions",
        content=Content(b"text/plain", b'{"model": "text-default", "prompt": "Hi"}'),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_that_is_not_an_object_is_refused(client):
    response = await client.post("/v1/completions", content=JSONContent([1, 2, 3]))

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_a_model_is_refused(client, provider):
    response = await post(client, {"prompt": "Once upon a time"})

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"
    assert provider.calls == []


async def test_a_body_without_a_prompt_is_refused(client, provider):
    response = await post(client, {"model": "text-default"})

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"
    assert provider.calls == []


async def test_a_capability_error_from_the_service_reaches_the_caller(client):
    provider = FakeTextProvider(error=CapabilityError("the route cannot honour this"))
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
