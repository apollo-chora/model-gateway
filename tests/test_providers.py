"""The deterministic provider: translation and execution without a network.

The mock's contract is that its answer is a pure function of its scripted spec,
the route it is handed, the request it is given and the attempt context beside
them. These tests pin that: the round trip is echoed (the public model ID, the
route's upstream model and the last user message), tool calls and finish reasons
come back as scripted, a configured failure raises, and nothing about the
environment or the clock can change the answer.

The attempt context is pinned here too, because it is the seam the gateway's
deadline travels through: it carries the facts of one attempt
and refuses a shape no attempt could have.

No resolver, endpoint or credential here is contacted: every route is a
standalone value and every call is a coroutine over in-memory values.
"""

import dataclasses
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
from model_gateway.providers.base import AttemptContext, TextProvider
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


def context(**overrides) -> AttemptContext:
    """A standalone attempt context; nothing about it is contacted."""
    values = {"invocation_id": "inv-1", "attempt_number": 1, "timeout": 30.0}
    values.update(overrides)
    return AttemptContext(**values)


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


async def test_the_mock_returns_a_well_formed_response():
    response = await MockProvider().complete(route(), request(), context())

    assert isinstance(response, InvocationResponse)
    assert response.model == "text-default"
    assert response.upstream_model == "upstream-model"
    assert response.attempt_count == 1
    assert response.fallback_used is False
    assert response.idempotency_status.value == "not_applicable"


async def test_the_mock_stamps_the_operations_public_model_id():
    response = await MockProvider().complete(route(), request(model="embedding-default"), context())

    assert response.model == "text-default"


async def test_the_mock_echoes_the_upstream_model_id_from_the_route():
    first = await MockProvider().complete(route("first-upstream"), request(), context())
    second = await MockProvider().complete(route("second-upstream"), request(), context())

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

    response = await MockProvider().complete(route(), conversation, context())

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

    response = await MockProvider().complete(route(), conversation, context())

    assert response.text == "what is in this image? and this?"


async def test_the_mock_echoes_nothing_when_the_conversation_has_no_user_message():
    conversation = request(
        messages=(Message(role=Role.SYSTEM, content=(TextPart(text="answer concisely"),)),)
    )

    response = await MockProvider().complete(route(), conversation, context())

    assert response.text == ""


async def test_the_mock_echoes_the_text_it_was_scripted_with():
    provider = MockProvider(MockSpec(text="Recursion is a technique..."))

    response = await provider.complete(route(), request(), context())

    assert response.text == "Recursion is a technique..."


async def test_the_mock_records_every_call_it_is_given():
    provider = MockProvider()

    await provider.complete(route("first"), request(), context())
    await provider.complete(route("second"), request(), context())

    assert [call[0].model_id for call in provider.calls] == ["first", "second"]
    assert all(call[1].model == "text-default" for call in provider.calls)


async def test_the_mock_answers_a_different_request_differently():
    provider = MockProvider()

    first = await provider.complete(route(), request(), context())
    second = await provider.complete(
        route(),
        request(
            messages=(
                Message(role=Role.USER, content=(TextPart(text="What is the weather in Rome?"),)),
            )
        ),
        context(),
    )

    assert first.text == "What is the weather in Paris?"
    assert second.text == "What is the weather in Rome?"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


async def test_the_mock_is_a_pure_function_of_its_inputs():
    provider = MockProvider()

    first = await provider.complete(route(), request(), context())
    second = await provider.complete(route(), request(), context())

    assert first == second


async def test_two_mocks_scripted_alike_answer_alike():
    first = await MockProvider(MockSpec(text="same")).complete(route(), request(), context())
    second = await MockProvider(MockSpec(text="same")).complete(route(), request(), context())

    assert first == second


async def test_the_mock_is_unaffected_by_the_environment(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_CONTEXT_WINDOW", "1048576")
    monkeypatch.setenv("PROVIDER_RETRY_COUNT", "2")
    provider = MockProvider()

    response = await provider.complete(route(), request(), context())

    assert response.text == "What is the weather in Paris?"


async def test_the_attempt_context_changes_nothing_about_the_answer():
    """The attempt's own facts are not an input to the translation."""
    provider = MockProvider()

    first = await provider.complete(route(), request(), context(attempt_number=1, timeout=30.0))
    second = await provider.complete(route(), request(), context(attempt_number=4, timeout=1.0))

    assert first == second


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

    response = await provider.complete(route(), request(), context())

    assert response.tool_calls == (
        ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris", "units": "metric"}),
    )
    assert response.content[1:] == response.tool_calls


async def test_the_mock_keeps_the_providers_own_raw_finish_reason():
    provider = MockProvider(
        MockSpec(finish_reason=FinishReason.LENGTH, raw_finish_reason="max_tokens")
    )

    response = await provider.complete(route(), request(), context())

    assert response.finish_reason is FinishReason.LENGTH
    assert response.raw_finish_reason == "max_tokens"


async def test_the_default_raw_reason_is_the_classification_it_reports():
    response = await MockProvider().complete(route(), request(), context())

    assert response.finish_reason is FinishReason.STOP
    assert response.raw_finish_reason == "stop"


async def test_the_mock_reports_the_usage_it_was_scripted_with():
    usage = Usage(input_tokens=25, output_tokens=42, total_tokens=67)
    provider = MockProvider(MockSpec(usage=usage))

    response = await provider.complete(route(), request(), context())

    assert response.usage == usage
    assert response.usage.input_tokens == 25


async def test_an_unreported_usage_count_is_not_reported_as_zero():
    response = await MockProvider().complete(route(), request(), context())

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None


# ---------------------------------------------------------------------------
# Failure on demand
# ---------------------------------------------------------------------------


async def test_the_mock_raises_the_error_it_was_scripted_with():
    failure = UnavailableError("the upstream provider is unavailable")
    provider = MockProvider(MockSpec(error=failure))

    with pytest.raises(UnavailableError) as error:
        await provider.complete(route(), request(), context())

    assert error.value is failure
    assert error.value.reason.value == "provider_unavailable"


async def test_the_mock_raises_before_it_returns_a_response():
    provider = MockProvider(MockSpec(error=UnavailableError("no provider could serve this")))

    with pytest.raises(UnavailableError):
        await provider.complete(route(), request(), context())

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
        await provider.complete(route(), request(), context())

    response = await provider.complete(route(), request(), context())

    assert response.text == "recovered"


async def test_the_last_spec_repeats_once_the_script_is_exhausted():
    provider = MockProvider((MockSpec(text="once"), MockSpec(text="twice")))

    assert (await provider.complete(route(), request(), context())).text == "once"
    assert (await provider.complete(route(), request(), context())).text == "twice"
    assert (await provider.complete(route(), request(), context())).text == "twice"


async def test_a_mock_provider_needs_at_least_one_response_to_return():
    with pytest.raises(ValueError, match="at least one response"):
        MockProvider(())


# ---------------------------------------------------------------------------
# The attempt context
# ---------------------------------------------------------------------------


def test_the_attempt_context_carries_the_attempts_own_facts():
    attempt = context(invocation_id="inv-7", attempt_number=3, timeout=12.5)

    assert (attempt.invocation_id, attempt.attempt_number, attempt.timeout) == ("inv-7", 3, 12.5)


def test_an_attempt_context_may_carry_no_invocation_id():
    """An invocation is not required to carry an identifier of its own."""
    assert context(invocation_id=None).invocation_id is None


def test_no_budget_is_not_a_budget_of_zero():
    assert context(timeout=None).timeout is None


def test_the_attempt_context_is_frozen_and_hashable():
    attempt = context()

    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.timeout = 1.0
    assert len({attempt}) == 1


@pytest.mark.parametrize("number", [0, -1])
def test_an_attempt_number_below_one_is_refused(number):
    with pytest.raises(ValueError, match="attempt_number must be at least 1"):
        context(attempt_number=number)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan")])
def test_a_budget_no_attempt_could_use_is_refused(budget):
    with pytest.raises(ValueError, match="timeout must be a positive number of seconds"):
        context(timeout=budget)


async def test_the_mock_records_the_attempt_context_it_was_given():
    provider = MockProvider()
    attempt = context(invocation_id="inv-9", attempt_number=2, timeout=5.0)

    await provider.complete(route(), request(), attempt)

    recorded_route, recorded_request, recorded_attempt = provider.calls[0]
    assert recorded_route.model_id == "upstream-model"
    assert recorded_request.model == "text-default"
    assert recorded_attempt is attempt


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


async def test_a_mock_provider_satisfies_the_text_provider_protocol():
    provider = MockProvider()

    assert inspect.iscoroutinefunction(provider.complete)
    assert list(inspect.signature(provider.complete).parameters) == [
        "route",
        "request",
        "context",
    ]


def test_the_protocol_is_the_providers_whole_contract():
    """The protocol's whole surface is one coroutine: a complete call."""
    assert callable(TextProvider.complete)
    assert inspect.iscoroutinefunction(TextProvider.complete)
    assert list(inspect.signature(TextProvider.complete).parameters) == [
        "self",
        "route",
        "request",
        "context",
    ]


async def test_a_provider_can_be_used_through_the_protocol_type():
    provider: TextProvider = MockProvider()

    assert (await provider.complete(route(), request(), context())).model == "text-default"
