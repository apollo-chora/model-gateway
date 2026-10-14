"""The deterministic provider: translation and execution without a network (PRD 9, 20).

The mock's contract is that its answer is a pure function of its scripted spec,
the route it is handed and the request it is given. These tests pin that: the
round trip is echoed (the public model ID, the route's upstream model and the
last user message), tool calls and finish reasons come back as scripted, a
configured failure raises, and nothing about the environment or the clock can
change the answer.

No resolver, endpoint or credential here is contacted: every route is a
standalone value and every call is a coroutine over in-memory values.
"""

import inspect

import pytest

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import UnavailableError
from model_gateway.providers.base import TextProvider
from model_gateway.providers.mock import MockProvider, MockSpec


def route(model_id: str = "upstream-model", **overrides) -> Route:
    """A standalone text route; nothing about it is contacted."""
    values = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.example/v1",
        "base_url": "https://upstream.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT, Capability.TOOLS}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def request(**overrides) -> InvocationRequest:
    """A standalone canonical request."""
    values = {
        "model": "text-default",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
    }
    values.update(overrides)
    return InvocationRequest(**values)


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


async def test_the_mock_returns_a_well_formed_response():
    response = await MockProvider().complete(route(), request())

    assert isinstance(response, InvocationResponse)
    assert response.model == "text-default"
    assert response.upstream_model == "upstream-model"
    assert response.attempt_count == 1
    assert response.fallback_used is False
    assert response.idempotency_status.value == "not_applicable"


async def test_the_mock_echoes_the_public_model_id():
    response = await MockProvider().complete(route(), request(model="embedding-default"))

    assert response.model == "embedding-default"


async def test_the_mock_echoes_the_upstream_model_id_from_the_route():
    first = await MockProvider().complete(route("first-upstream"), request())
    second = await MockProvider().complete(route("second-upstream"), request())

    assert first.upstream_model == "first-upstream"
    assert second.upstream_model == "second-upstream"


async def test_the_mock_echoes_the_last_user_message():
    conversation = request(
        messages=(
            Message(role=Role.USER, content=(TextPart(text="earlier question"),)),
            Message(role=Role.ASSISTANT, content=(TextPart(text="an answer"),)),
            Message(role=Role.USER, content=(TextPart(text="the latest question"),)),
        )
    )

    response = await MockProvider().complete(route(), conversation)

    assert response.text == "the latest question"
    assert response.content == (TextPart(text="the latest question"),)


async def test_the_mock_echoes_a_user_message_that_carries_an_image():
    conversation = request(
        messages=(
            Message(
                role=Role.USER,
                content=(TextPart(text="what is in this image?"), TextPart(text=" and this?")),
            ),
        )
    )

    response = await MockProvider().complete(route(), conversation)

    assert response.text == "what is in this image? and this?"


async def test_the_mock_echoes_nothing_when_the_conversation_has_no_user_message():
    conversation = request(
        messages=(Message(role=Role.SYSTEM, content=(TextPart(text="answer concisely"),)),)
    )

    response = await MockProvider().complete(route(), conversation)

    assert response.text == ""


async def test_the_mock_echoes_the_text_it_was_scripted_with():
    provider = MockProvider(MockSpec(text="Recursion is a technique..."))

    response = await provider.complete(route(), request())

    assert response.text == "Recursion is a technique..."


async def test_the_mock_records_every_call_it_is_given():
    provider = MockProvider()

    await provider.complete(route("first"), request())
    await provider.complete(route("second"), request())

    assert [call[0].model_id for call in provider.calls] == ["first", "second"]
    assert all(call[1].model == "text-default" for call in provider.calls)


async def test_the_mock_answers_a_different_request_differently():
    provider = MockProvider()

    first = await provider.complete(route(), request())
    second = await provider.complete(route(), request(model="image-default"))

    assert first.model != second.model


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


async def test_the_mock_is_a_pure_function_of_its_inputs():
    provider = MockProvider()

    first = await provider.complete(route(), request())
    second = await provider.complete(route(), request())

    assert first == second


async def test_two_mocks_scripted_alike_answer_alike():
    first = await MockProvider(MockSpec(text="same")).complete(route(), request())
    second = await MockProvider(MockSpec(text="same")).complete(route(), request())

    assert first == second


async def test_the_mock_is_unaffected_by_the_environment(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_CONTEXT_WINDOW", "1048576")
    monkeypatch.setenv("PROVIDER_RETRY_COUNT", "2")
    provider = MockProvider()

    response = await provider.complete(route(), request())

    assert response.text == "What is the weather in Paris?"


# ---------------------------------------------------------------------------
# Tool calls and finish reasons
# ---------------------------------------------------------------------------


async def test_the_mock_returns_tool_calls():
    provider = MockProvider(
        MockSpec(
            tool_calls=(
                ToolCall(
                    id="call_01",
                    name="get_weather",
                    arguments={"city": "Paris", "units": "metric"},
                ),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    response = await provider.complete(route(), request())

    assert response.tool_calls == (
        ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris", "units": "metric"}),
    )
    assert response.content[1:] == response.tool_calls


async def test_the_mock_keeps_the_providers_own_raw_finish_reason():
    provider = MockProvider(
        MockSpec(finish_reason=FinishReason.LENGTH, raw_finish_reason="max_tokens")
    )

    response = await provider.complete(route(), request())

    assert response.finish_reason is FinishReason.LENGTH
    assert response.raw_finish_reason == "max_tokens"


async def test_the_default_raw_reason_is_the_classification_it_reports():
    response = await MockProvider().complete(route(), request())

    assert response.finish_reason is FinishReason.STOP
    assert response.raw_finish_reason == "stop"


async def test_the_mock_reports_the_usage_it_was_scripted_with():
    usage = Usage(input_tokens=25, output_tokens=42, total_tokens=67)
    provider = MockProvider(MockSpec(usage=usage))

    response = await provider.complete(route(), request())

    assert response.usage == usage
    assert response.usage.input_tokens == 25


async def test_an_unreported_usage_count_is_not_reported_as_zero():
    response = await MockProvider().complete(route(), request())

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None


# ---------------------------------------------------------------------------
# Failure on demand
# ---------------------------------------------------------------------------


async def test_the_mock_raises_the_error_it_was_scripted_with():
    failure = UnavailableError("the upstream provider is unavailable")
    provider = MockProvider(MockSpec(error=failure))

    with pytest.raises(UnavailableError) as error:
        await provider.complete(route(), request())

    assert error.value is failure
    assert error.value.reason.value == "provider_unavailable"


async def test_the_mock_raises_before_it_returns_a_response():
    provider = MockProvider(MockSpec(error=UnavailableError("no provider could serve this")))

    with pytest.raises(UnavailableError):
        await provider.complete(route(), request())

    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Scripted sequences
# ---------------------------------------------------------------------------


async def test_a_sequence_of_specs_serves_one_response_per_call():
    provider = MockProvider(
        (
            MockSpec(error=UnavailableError("the first attempt failed")),
            MockSpec(text="recovered"),
        )
    )

    with pytest.raises(UnavailableError):
        await provider.complete(route(), request())

    response = await provider.complete(route(), request())

    assert response.text == "recovered"


async def test_the_last_spec_repeats_once_the_script_is_exhausted():
    provider = MockProvider((MockSpec(text="once"), MockSpec(text="twice")))

    assert (await provider.complete(route(), request())).text == "once"
    assert (await provider.complete(route(), request())).text == "twice"
    assert (await provider.complete(route(), request())).text == "twice"


async def test_a_mock_provider_needs_at_least_one_response_to_return():
    with pytest.raises(ValueError, match="at least one response"):
        MockProvider(())


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


async def test_a_mock_provider_satisfies_the_text_provider_protocol():
    provider = MockProvider()

    assert inspect.iscoroutinefunction(provider.complete)
    assert list(inspect.signature(provider.complete).parameters) == ["route", "request"]


def test_the_protocol_is_the_providers_whole_contract():
    """The protocol's whole surface is one coroutine: a complete call."""
    assert callable(TextProvider.complete)
    assert inspect.iscoroutinefunction(TextProvider.complete)
    assert list(inspect.signature(TextProvider.complete).parameters) == ["self", "route", "request"]


async def test_a_provider_can_be_used_through_the_protocol_type():
    provider: TextProvider = MockProvider()

    assert (await provider.complete(route(), request())).model == "text-default"
