"""Tests for the Responses endpoint (PRD sections 4.5, 9.1, 9.5 and 16)."""

import pytest
from blacksheep import Application, Content, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    ImagePart,
    InvocationResponse,
    RefusalPart,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import CapabilityError, UnavailableError
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.responses import register_responses_routes
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"

_TEXT_CAPABILITIES = frozenset({Capability.CHAT, Capability.TOOLS, Capability.STRUCTURED_OUTPUT})


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
                capabilities=_TEXT_CAPABILITIES,
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


def answer(**overrides) -> InvocationResponse:
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


def image_part() -> ImagePart:
    return ImagePart(data="aGVsbG8=", media_type="image/png")


class FakeTextProvider:
    """A ``TextProvider`` double: records the calls and returns a canned response.

    The gateway's HTTP surface is what these tests are about, so the provider
    stands in for the upstream at the binding boundary and no socket is opened.
    """

    def __init__(
        self, result: InvocationResponse | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[tuple[Route, object]] = []
        self.result = result if result is not None else answer()
        self.error = error

    async def complete(self, route: Route, request, context) -> InvocationResponse:
        self.calls.append((route, request))
        if self.error is not None:
            raise self.error
        return self.result


def build_app(provider: FakeTextProvider) -> Application:
    app = Application(router=Router())
    register_error_handlers(app)
    register_responses_routes(app.router, InferenceService(make_registry(), provider))
    return app


async def call(provider: FakeTextProvider, body) -> tuple[int, object]:
    """Send one request through a freshly built application, without a socket."""
    app = build_app(provider)
    await app.start()
    try:
        response = await TestClient(app).post("/v1/responses", content=JSONContent(body))
        return response.status, await response.json()
    finally:
        await app.stop()


async def call_text(provider: FakeTextProvider, body) -> tuple[int, str]:
    app = build_app(provider)
    await app.start()
    try:
        response = await TestClient(app).post("/v1/responses", content=JSONContent(body))
        return response.status, await response.text()
    finally:
        await app.stop()


@pytest.fixture
async def provider() -> FakeTextProvider:
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
    body = {"model": "text-default", "input": "Explain recursion."}
    body.update(overrides)
    return body


async def post(client, body):
    return await client.post("/v1/responses", content=JSONContent(body))


async def test_happy_path_returns_the_documented_response_shape(client):
    response = await post(client, payload(temperature=0.7, max_output_tokens=2048))

    assert response.status == 200
    body = await response.json()
    assert body["object"] == "response"
    assert body["model"] == "text-default"
    assert body["status"] == "completed"
    assert isinstance(body["created_at"], int)
    assert body["id"].startswith("resp_")
    assert body["output"] == [
        {
            "id": body["output"][0]["id"],
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Recursion is a technique...", "annotations": []}
            ],
        }
    ]
    assert body["output"][0]["id"].startswith("msg_")


async def test_happy_path_response_is_json(client):
    response = await post(client, payload())

    assert response.content_type() == b"application/json"


async def test_usage_uses_the_responses_field_names(client):
    response = await post(client, payload())

    body = await response.json()
    assert body["usage"] == {
        "input_tokens": 25,
        "output_tokens": 42,
        "total_tokens": 67,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


async def test_a_cached_count_the_provider_reported_is_carried(client, provider):
    provider.result = answer(
        usage=Usage(input_tokens=25, output_tokens=42, total_tokens=67, cached_tokens=10)
    )

    response = await post(client, payload())

    body = await response.json()
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 10


async def test_an_unreported_total_is_summed_from_the_counts_that_were_reported(client, provider):
    provider.result = answer(usage=Usage(input_tokens=25, output_tokens=42))

    response = await post(client, payload())

    body = await response.json()
    assert body["usage"]["total_tokens"] == 67


async def test_a_string_input_becomes_one_user_message(client, provider):
    await post(client, payload(input="Explain recursion."))

    _, invocation = provider.calls[0]
    assert len(invocation.messages) == 1
    message = invocation.messages[0]
    assert message.role is Role.USER
    assert message.content == (TextPart("Explain recursion."),)


async def test_an_array_input_becomes_the_ordered_conversation(client, provider):
    body = payload(
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "developer", "content": "Answer briefly."},
            {"role": "user", "content": "Explain recursion."},
            {"role": "assistant", "content": "Recursion is a technique..."},
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
    ]
    assert invocation.messages[2].content == (TextPart("Explain recursion."),)


async def test_an_input_message_may_carry_content_parts(client, provider):
    body = payload(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Explain "},
                    {"type": "input_text", "text": "recursion."},
                ],
            }
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.messages[0].content == (TextPart("Explain "), TextPart("recursion."))


async def test_a_replayed_output_text_part_is_accepted_with_empty_annotations(client, provider):
    body = payload(
        input=[
            {"role": "user", "content": "Explain recursion."},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Recursion is a technique...",
                        "annotations": [],
                    }
                ],
            },
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.messages[1].content == (TextPart("Recursion is a technique..."),)


async def test_an_annotated_output_text_part_is_refused(client, provider):
    body = payload(
        input=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "See this.",
                        "annotations": [{"type": "url_citation", "url": "https://example.com"}],
                    }
                ],
            }
        ]
    )

    status, body = await call(provider, body)

    assert status == 422
    assert body["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_a_function_call_item_becomes_an_assistant_message(client, provider):
    body = payload(
        input=[
            {"role": "user", "content": "What is the weather?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Rome"}',
                "id": "fc_1",
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    proposed = invocation.messages[1]
    assert proposed.role is Role.ASSISTANT
    assert proposed.tool_calls == (
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
    )
    answered = invocation.messages[2]
    assert answered.role is Role.TOOL
    assert answered.tool_call_id == "call_1"
    assert answered.content == (TextPart("sunny"),)


async def test_an_unparsable_function_call_argument_is_relayed_verbatim(client, provider):
    body = payload(
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{not json",
            }
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.messages[0].tool_calls[0].arguments == "{not json"


async def test_instructions_map_onto_the_invocation_level_instruction(client, provider):
    response = await post(client, payload(instructions="Answer in one sentence."))

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.system_instruction == "Answer in one sentence."
    # The instruction is a separate element: it is not folded into the conversation.
    assert [message.role.value for message in invocation.messages] == ["user"]


async def test_the_generation_parameters_reach_the_provider(client, provider):
    await post(client, payload(temperature=0.7, max_output_tokens=2048))

    _, invocation = provider.calls[0]
    assert invocation.generation_config.temperature == 0.7
    assert invocation.generation_config.max_output_tokens == 2048


async def test_tool_declarations_reach_the_provider(client, provider):
    body = payload(
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object"},
            }
        ]
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    declaration = invocation.tools[0]
    assert declaration.name == "get_weather"
    assert declaration.description == "Look up the weather."
    assert declaration.parameters == {"type": "object"}


async def test_a_tool_without_parameters_declares_an_empty_schema(client, provider):
    await post(client, payload(tools=[{"type": "function", "name": "ping"}]))

    _, invocation = provider.calls[0]
    assert invocation.tools[0].parameters == {}


async def test_a_hosted_tool_is_refused_naming_its_type(client, provider):
    status, body = await call(provider, payload(tools=[{"type": "web_search"}]))

    assert status == 422
    assert body["reason"] == "unsupported_capability"
    assert "web_search" in body["message"]
    assert provider.calls == []


async def test_a_strict_tool_declaration_is_refused(client, provider):
    body = payload(
        tools=[{"type": "function", "name": "ping", "strict": True}],
    )

    status, response = await call(provider, body)

    assert status == 422
    assert response["reason"] == "unsupported_capability"
    assert provider.calls == []


async def test_a_false_strict_flag_is_accepted(client, provider):
    body = payload(tools=[{"type": "function", "name": "ping", "strict": False}])

    response = await post(client, body)

    assert response.status == 200


async def test_response_format_maps_onto_the_canonical_requirement(client, provider):
    await post(client, payload(response_format={"type": "json_object"}))

    _, invocation = provider.calls[0]
    assert invocation.generation_config.response_format.mode is ResponseFormatMode.JSON_OBJECT


async def test_a_json_schema_response_format_reaches_the_provider(client, provider):
    body = payload(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }
    )

    response = await post(client, body)

    assert response.status == 200
    _, invocation = provider.calls[0]
    requirement = invocation.generation_config.response_format
    assert requirement.mode is ResponseFormatMode.JSON_SCHEMA
    assert requirement.name == "answer"
    assert requirement.strict is True


async def test_a_text_response_format_asks_for_text(client, provider):
    await post(client, payload(response_format={"type": "text"}))

    _, invocation = provider.calls[0]
    assert invocation.generation_config.response_format.mode is ResponseFormatMode.TEXT


async def test_an_unknown_response_format_type_is_refused(client, provider):
    status, body = await call(provider, payload(response_format={"type": "xml"}))

    assert status == 400
    assert body["reason"] == "invalid_request"
    assert provider.calls == []


async def test_a_structured_output_requirement_the_route_cannot_honour_is_refused(client):
    """The capability is enforced at the gateway core, not by this endpoint."""
    provider = FakeTextProvider(error=CapabilityError("the route cannot honour json_object"))
    app = build_app(provider)
    await app.start()
    try:
        response = await post(TestClient(app), payload(response_format={"type": "json_object"}))
    finally:
        await app.stop()

    assert response.status == 422
    assert (await response.json())["reason"] == "unsupported_capability"


async def test_a_tool_call_answer_renders_a_function_call_item_after_the_message(client):
    provider = FakeTextProvider(
        answer(
            content=(
                TextPart("Checking the weather."),
                ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )

    status, body = await call(provider, payload())

    assert status == 200
    output = body["output"]
    assert [item["type"] for item in output] == ["message", "function_call"]
    assert output[0]["content"] == [
        {"type": "output_text", "text": "Checking the weather.", "annotations": []}
    ]
    assert output[1]["call_id"] == "call_1"
    assert output[1]["name"] == "get_weather"
    assert output[1]["arguments"] == '{"city":"Rome"}'
    assert output[1]["status"] == "completed"
    assert output[1]["id"].startswith("fc_")


async def test_a_tool_call_only_answer_carries_no_message_item(client):
    provider = FakeTextProvider(
        answer(
            content=(ToolCall(id="call_1", name="get_weather", arguments=None),),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )

    status, body = await call(provider, payload())

    assert status == 200
    assert [item["type"] for item in body["output"]] == ["function_call"]
    # Arguments the provider never reported are relayed as the empty string the
    # Responses shape types, not invented as an empty object.
    assert body["output"][0]["arguments"] == ""


async def test_several_tool_calls_keep_their_order(client):
    provider = FakeTextProvider(
        answer(
            content=(
                ToolCall(id="call_1", name="first", arguments={}),
                ToolCall(id="call_2", name="second", arguments={}),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )

    status, body = await call(provider, payload())

    assert status == 200
    assert [item["call_id"] for item in body["output"]] == ["call_1", "call_2"]


async def test_a_refusal_keeps_its_own_variant(client):
    provider = FakeTextProvider(
        answer(
            content=(RefusalPart("I cannot help with that."),),
            finish_reason=FinishReason.CONTENT_FILTER,
            raw_finish_reason="content_filter",
        )
    )

    status, body = await call(provider, payload())

    assert status == 200
    assert body["output"][0]["content"] == [
        {"type": "refusal", "refusal": "I cannot help with that."}
    ]


async def test_an_interleaved_answer_is_refused_rather_than_flattened(client):
    provider = FakeTextProvider(
        answer(
            content=(
                TextPart("Checking the weather."),
                ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
                TextPart("It is sunny."),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            raw_finish_reason="tool_calls",
        )
    )

    status, body = await call_text(provider, payload())

    assert status == 422
    assert "It is sunny." not in body
    assert "unsupported_capability" in body


async def test_an_image_answer_is_refused(client):
    provider = FakeTextProvider(answer(content=(TextPart("Here it is."), image_part())))

    status, body = await call(provider, payload())

    assert status == 422
    assert body["reason"] == "unsupported_capability"


async def test_an_unknown_model_is_a_model_not_found_error(client, provider):
    response = await post(client, payload(model="no-such-model"))

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"
    assert provider.calls == []


UNSUPPORTED_PARAMETERS = [
    {"top_p": 0.9},
    {"tool_choice": "auto"},
    {"previous_response_id": "resp_1"},
    {"store": True},
    {"background": True},
    {"conversation": "conv_1"},
    {"include": ["reasoning.encrypted_content"]},
    {"max_tool_calls": 2},
    {"metadata": {"trace": "abc"}},
    {"moderation": {"model": "omni-moderation-latest"}},
    {"parallel_tool_calls": True},
    {"parallel_tool_calls": False},
    {"prompt": {"id": "pmpt_1"}},
    {"prompt_cache_key": "key"},
    {"prompt_cache_retention": "24h"},
    {"reasoning": {"effort": "high"}},
    {"safety_identifier": "user-1"},
    {"service_tier": "auto"},
    {"stream_options": {"include_usage": True}},
    {"text": {"format": {"type": "text"}}},
    {"top_logprobs": 3},
    {"truncation": "auto"},
    {"user": "user-1"},
    {"context_management": [{"type": "compaction"}]},
]


@pytest.mark.parametrize("extra", UNSUPPORTED_PARAMETERS, ids=lambda extra: next(iter(extra)))
async def test_every_unsupported_parameter_is_refused_naming_it(provider, extra):
    status, body = await call(provider, payload(**extra))

    assert status == 422
    assert body["reason"] == "unsupported_capability"
    assert next(iter(extra)) in body["message"]
    assert provider.calls == []


async def test_store_false_is_accepted_because_the_gateway_stores_nothing(client, provider):
    response = await post(client, payload(store=False))

    assert response.status == 200
    assert provider.calls != []


async def test_an_unknown_parameter_is_refused_naming_the_key(client, provider):
    response = await post(client, payload(verbosity="high"))

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"
    assert "verbosity" in (await response.json())["message"]
    assert provider.calls == []


async def test_an_input_item_type_this_endpoint_does_not_carry_is_refused(client, provider):
    status, body = await call(provider, payload(input=[{"type": "mcp_call"}]))

    assert status == 422
    assert body["reason"] == "unsupported_capability"
    assert "mcp_call" in body["message"]
    assert provider.calls == []


async def test_an_input_image_part_is_refused_naming_it(client, provider):
    body = payload(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
                ],
            }
        ]
    )

    status, response = await call(provider, body)

    assert status == 422
    assert response["reason"] == "unsupported_capability"
    assert "input_image" in response["message"]
    assert provider.calls == []


async def test_an_input_file_part_is_refused_naming_it(client, provider):
    body = payload(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": "file_1"}],
            }
        ]
    )

    status, response = await call(provider, body)

    assert status == 422
    assert response["reason"] == "unsupported_capability"
    assert "input_file" in response["message"]


async def test_a_tool_role_message_is_refused_with_its_own_explanation(client, provider):
    body = payload(input=[{"role": "tool", "content": "42"}])

    status, response = await call(provider, body)

    assert status == 400
    assert response["reason"] == "invalid_request"
    assert "function_call_output" in response["message"]


async def test_an_unknown_message_role_is_refused(client, provider):
    status, body = await call(provider, payload(input=[{"role": "critic", "content": "hi"}]))

    assert status == 400
    assert body["reason"] == "invalid_request"


async def test_an_unknown_message_key_is_refused(client, provider):
    body = payload(input=[{"role": "user", "content": "hi", "reasoning": {}}])

    status, response = await call(provider, body)

    assert status == 400
    assert "reasoning" in response["message"]


async def test_a_message_phase_is_refused_as_a_capability_gap(client, provider):
    body = payload(input=[{"role": "assistant", "content": "hi", "phase": "final_answer"}])

    status, response = await call(provider, body)

    assert status == 422
    assert response["reason"] == "unsupported_capability"


async def test_a_streaming_request_is_refused_with_a_capability_error(client, provider):
    """The Responses event protocol is not served; a non-streamed body is not returned."""
    response = await post(client, payload(stream=True))

    assert response.status == 422
    assert response.content_type() == b"application/json"
    body = await response.json()
    assert body["reason"] == "unsupported_capability"
    assert "stream" in body["message"]
    assert provider.calls == []


async def test_a_streaming_request_does_not_return_a_non_streamed_body(client, provider):
    response = await post(client, payload(stream=True))

    body = await response.text()
    assert '"object": "response"' not in body
    assert "output_text" not in body


async def test_a_streaming_request_never_returns_server_sent_events(client, provider):
    response = await post(client, payload(stream=True))

    assert response.content_type() != b"text/event-stream"


async def test_stream_false_is_accepted(client, provider):
    response = await post(client, payload(stream=False))

    assert response.status == 200
    assert provider.calls != []


async def test_a_body_that_is_not_json_is_refused(client):
    response = await client.post(
        "/v1/responses",
        content=Content(b"application/json", b"{not json"),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_a_content_type_is_refused(client):
    response = await client.post(
        "/v1/responses",
        content=Content(b"text/plain", b'{"model": "text-default", "input": "hi"}'),
    )

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_that_is_not_an_object_is_refused(client):
    response = await client.post("/v1/responses", content=JSONContent([1, 2, 3]))

    assert response.status == 400
    assert (await response.json())["reason"] == "invalid_request"


async def test_a_body_without_input_is_refused(client, provider):
    status, body = await call(provider, {"model": "text-default"})

    assert status == 400
    assert body["reason"] == "invalid_request"
    assert provider.calls == []


async def test_a_body_without_a_model_is_refused(client, provider):
    status, body = await call(provider, {"input": "hi"})

    assert status == 400
    assert body["reason"] == "invalid_request"
    assert provider.calls == []


async def test_an_empty_input_array_is_refused(client, provider):
    status, body = await call(provider, payload(input=[]))

    assert status == 400
    assert body["reason"] == "invalid_request"


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
        await (await post(client, payload(verbosity="high"))).text(),
    ]

    for body in bodies:
        assert SECRET_KEY not in body
        assert UPSTREAM_ENDPOINT not in body
        assert "upstream-text" not in body
