"""Tests for the LiteLLM provider: one gateway attempt through the pinned binding.

No test reaches the network: ``litellm.acompletion`` is monkeypatched with a
recording stand-in, so the real per-dialect client is built and closed around a
transport that is never asked for a request, and the response fixtures are the
real LiteLLM types — built through LiteLLM's own converter from a realistic
upstream payload — so what runs is exactly what the installed binding does with
a real provider's answer.

The failure paths construct LiteLLM's and the OpenAI SDK's exception classes
in memory, which is where the failure mapping lives. The tests that
pin the ``.env`` guard run a binding module's import in a subprocess with a
``.env`` in its working directory: that side effect only exists at import time,
and only a fresh process can show whether the seam suppressed it.
"""

import asyncio
import inspect
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path

import httpx
import litellm
import openai
import pytest
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.types.utils import Usage as LiteLLMUsage

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    RefusalPart,
    Role,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    InternalError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.providers.base import AttemptContext, ImageProvider, TextProvider
from model_gateway.providers.litellm_provider import SUPPORTED_DIALECTS, LiteLLMProvider
from model_gateway.security.transport import PinnedAddressTransport

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "gpt-4o"
PUBLIC_ADDRESS = "93.184.216.34"

ALL_CAPABILITIES = frozenset(
    {
        Capability.CHAT,
        Capability.TOOLS,
        Capability.VISION,
        Capability.STRUCTURED_OUTPUT,
    }
)


def make_route(**overrides: object) -> Route:
    """A resolved text route; override any field to model a different configuration."""
    fields: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": "openai",
        "api_key": SECRET_KEY,
        "capabilities": ALL_CAPABILITIES,
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request(**overrides: object) -> InvocationRequest:
    """A canonical invocation; override any field to model a different request."""
    fields: dict[str, object] = {
        "model": "text-default",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
        "invocation_id": "inv-7f3c1a90",
    }
    fields.update(overrides)
    return InvocationRequest(**fields)


def make_context(**overrides: object) -> AttemptContext:
    """A standalone attempt context; the budget is the provider's only deadline input."""
    fields: dict[str, object] = {
        "invocation_id": "inv-7f3c1a90",
        "attempt_number": 1,
        "timeout": 30.0,
    }
    fields.update(overrides)
    return AttemptContext(**fields)


def http_error(status: int) -> httpx.Response:
    """An HTTP error response with a request attached, as LiteLLM exceptions carry."""
    return httpx.Response(
        status_code=status,
        request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions"),
    )


def completion_payload(**overrides: object) -> dict[str, object]:
    """A realistic OpenAI-shaped upstream completion payload."""
    payload: dict[str, object] = {
        "id": "chatcmpl-9x2f",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recursion is a technique...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 42,
            "total_tokens": 67,
        },
    }
    payload.update(overrides)
    return payload


def response_from(payload: dict[str, object]) -> ModelResponse:
    """Build a real ``ModelResponse`` by parsing a payload through LiteLLM's own converter.

    This is the shape a real provider response has once LiteLLM has decoded it,
    including the fields LiteLLM itself adds or normalises.
    """
    response = ModelResponse()
    convert_to_model_response_object(response_object=payload, model_response_object=response)
    return response


def pinned_transport_of(client: object) -> PinnedAddressTransport:
    """The pinned transport ``client`` connects through, whichever shape it is.

    Both shapes keep the httpx client that owns the connection in one
    attribute; the OpenAI SDK's is private, which is why this is a helper
    rather than an assertion in each test.
    """
    http_client = (
        client._client  # pyright: ignore[reportPrivateUsage]  # the SDK keeps its httpx client here
        if isinstance(client, openai.AsyncOpenAI)
        else client.client  # pyright: ignore[reportAttributeAccessIssue]  # AsyncHTTPHandler property
    )
    transport = http_client._transport  # pyright: ignore[reportPrivateUsage]  # httpx keeps its transport here
    assert isinstance(transport, PinnedAddressTransport)
    return transport


async def collect(agen) -> list:
    """Drain an async generator into a list."""
    return [event async for event in agen]


def recording_aclose(monkeypatch: pytest.MonkeyPatch) -> list[PinnedAddressTransport]:
    """Record every pinned transport closed while the test runs."""
    closed: list[PinnedAddressTransport] = []
    original = PinnedAddressTransport.aclose

    async def record(self: PinnedAddressTransport) -> None:
        closed.append(self)
        await original(self)

    monkeypatch.setattr(PinnedAddressTransport, "aclose", record)
    return closed


class RecordingCompletion:
    """A stand-in for ``litellm.acompletion`` that records each call and answers from a script."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcome = outcome

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class RecordingStream:
    """A stand-in for ``litellm.acompletion`` that records each call and streams chunks.

    The outcome is either an :exc:`BaseException``, raised as the call is made,
    or an async iterable of :class:`~litellm.types.utils.ModelResponseStream`
    chunks, which is what the real library returns for ``stream=True``.
    """

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcome = outcome

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def stream_of(*chunks: ModelResponseStream, error: BaseException | None = None):
    """An async iterable of chunks, the shape ``litellm.acompletion`` returns for a stream.

    ``error``, when given, is raised after the last chunk — the upstream that
    produced output and then failed before terminating, which is the case the
    provider's terminal-event rule has to survive.
    """

    async def generate():
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    return generate()


def content_chunk(text: str) -> ModelResponseStream:
    """A chunk carrying a text delta and the role marker that opens the answer."""
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text, role="assistant"))]
    )


def finish_chunk(finish_reason: str) -> ModelResponseStream:
    """A chunk carrying a finish reason."""
    return ModelResponseStream(choices=[StreamingChoices(index=0, finish_reason=finish_reason)])


def usage_chunk(
    prompt_tokens: int, completion_tokens: int, total_tokens: int
) -> ModelResponseStream:
    """A trailing usage chunk, the shape the pinned library emits after a finish reason.

    It carries a choice with no delta and no finish reason, which is exactly what
    ``CustomStreamWrapper`` produces at end of stream when usage reporting is on;
    :class:`TestTheLibrariesOwnChunkOrdering` pins that against the library
    itself. ``total_tokens`` is passed as well, because that is what the
    upstream reports beside the two parts.
    """
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta())],
        usage=LiteLLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def upstream_failure() -> litellm.APIConnectionError:
    """The upstream failing mid-stream, as the library reports it."""
    return litellm.APIConnectionError(
        message="the upstream connection failed",
        llm_provider="openai",
        model=UPSTREAM_MODEL,
    )


def tool_call_chunk(
    *,
    index: int = 0,
    identifier: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> ModelResponseStream:
    """A chunk carrying one tool-call delta, as OpenAI sends one across several."""
    call = ChatCompletionDeltaToolCall(
        index=index,
        id=identifier,
        type="function",
        function=Function(arguments=arguments, name=name),
    )
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=None, tool_calls=[call]))]
    )


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Wire a provider to a scripted completion; the real client is built, never connected."""

    def wire_outcome(
        outcome: object,
    ) -> tuple[LiteLLMProvider, RecordingCompletion]:
        completion = RecordingCompletion(outcome)
        monkeypatch.setattr(litellm, "acompletion", completion)
        return LiteLLMProvider(), completion

    return wire_outcome


@pytest.fixture
def wire_stream(monkeypatch: pytest.MonkeyPatch):
    """Wire a provider to a scripted stream; the real client is built, never connected."""

    def wire_outcome(
        outcome: object,
    ) -> tuple[LiteLLMProvider, RecordingStream]:
        stream = RecordingStream(outcome)
        monkeypatch.setattr(litellm, "acompletion", stream)
        return LiteLLMProvider(), stream

    return wire_outcome


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestTheCall:
    """What one attempt hands to the pinned binding, and what comes back."""

    async def test_the_parameters_are_the_translation_plus_the_attempts_own_controls(
        self, wire
    ) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert call["model"] == "openai/gpt-4o"
        assert call["api_base"] == ENDPOINT
        assert call["api_key"] == SECRET_KEY
        assert call["extra_headers"] == {}
        assert call["messages"] == [{"role": "user", "content": "What is the weather in Paris?"}]
        assert call["num_retries"] == 0
        assert call["timeout"] == 30.0

    async def test_the_injected_client_is_the_sdk_client_over_the_pinned_transport(
        self, wire
    ) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        client = call["client"]
        assert isinstance(client, openai.AsyncOpenAI)
        # The SDK reads these off the object it is handed, which is why a bare
        # httpx client fails this path outright rather than losing the controls.
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        assert client.max_retries == 0
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)

    async def test_the_client_is_closed_when_the_attempt_ends(self, wire, monkeypatch) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert closed == [pinned_transport_of(call["client"])]

    async def test_the_attempt_budget_is_the_calls_timeout(self, wire) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=12.5))

        assert completion.calls[0]["timeout"] == 12.5

    async def test_no_budget_is_passed_through_as_none(self, wire) -> None:
        """The gateway set no budget; the binding invents none of its own."""
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=None))

        assert completion.calls[0]["timeout"] is None

    async def test_each_attempt_gets_a_fresh_client_with_its_own_budget(
        self, wire, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=7.0))
        await provider.complete(make_route(), make_request(), make_context(timeout=7.0))

        first, second = (call["client"] for call in completion.calls)
        assert first is not second
        assert closed == [pinned_transport_of(first), pinned_transport_of(second)]
        assert first.timeout == httpx.Timeout(7.0)
        assert second.timeout == httpx.Timeout(7.0)

    async def test_the_call_is_made_exactly_once_per_attempt(self, wire) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        assert len(completion.calls) == 1

    async def test_the_instruction_is_forwarded_as_its_own_system_message(self, wire) -> None:
        """The invocation-level instruction and a conversation system message stay unmerged."""
        provider, completion = wire(response_from(completion_payload()))
        request = make_request(
            system_instruction="answer concisely",
            messages=(
                Message(role=Role.SYSTEM, content=(TextPart(text="be terse"),)),
                Message(role=Role.USER, content=(TextPart(text="hi"),)),
            ),
        )

        await provider.complete(make_route(), request, make_context())

        assert completion.calls[0]["messages"] == [
            {"role": "system", "content": "answer concisely"},
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    async def test_the_answer_is_normalised_into_the_canonical_response(self, wire) -> None:
        provider, _ = wire(response_from(completion_payload()))

        response = await provider.complete(make_route(), make_request(), make_context())

        assert isinstance(response, InvocationResponse)
        assert response.model == "text-default"
        assert response.upstream_model == UPSTREAM_MODEL
        assert response.invocation_id == "inv-7f3c1a90"
        assert response.text == "Recursion is a technique..."
        assert response.finish_reason is FinishReason.STOP
        assert response.raw_finish_reason == "stop"
        assert response.usage == Usage(input_tokens=25, output_tokens=42, total_tokens=67)

    async def test_a_tool_call_survives_the_round_trip(self, wire) -> None:
        payload = completion_payload(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_01",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )
        provider, _ = wire(response_from(payload))

        response = await provider.complete(make_route(), make_request(), make_context())

        assert response.tool_calls == (
            ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris"}),
        )
        assert response.finish_reason is FinishReason.TOOL_CALLS


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


class TestTheStream:
    """The streaming path: one gateway attempt with ``stream=True``.

    The stream is the same single attempt under the same controls as
    :meth:`LiteLLMProvider.complete` — the per-dialect client, ``num_retries=0``,
    the attempt timeout, the error mapping — and the chunks are translated by
    the binding's own chunk translation. No test reaches the network: the real
    client is built and closed around a transport that is never asked for a
    request.
    """

    async def test_the_parameters_are_the_stream_translation_plus_the_attempts_own_controls(
        self, wire_stream
    ) -> None:
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context()))

        (call,) = stream.calls
        assert call["model"] == "openai/gpt-4o"
        assert call["api_base"] == ENDPOINT
        assert call["api_key"] == SECRET_KEY
        assert call["num_retries"] == 0
        assert call["stream"] is True
        assert call["timeout"] == 30.0

    async def test_the_injected_client_is_the_same_per_dialect_construction(
        self, wire_stream
    ) -> None:
        """The stream reaches the wire through the same client as the non-streaming call."""
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context()))

        (call,) = stream.calls
        client = call["client"]
        assert isinstance(client, openai.AsyncOpenAI)
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        assert client.max_retries == 0
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)

    async def test_the_client_is_closed_when_the_stream_ends(
        self, wire_stream, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context()))

        (call,) = stream.calls
        assert closed == [pinned_transport_of(call["client"])]

    async def test_each_attempt_gets_a_fresh_client_with_its_own_budget(
        self, wire_stream, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context(timeout=7.0)))
        await collect(provider.stream(make_route(), make_request(), make_context(timeout=7.0)))

        first, second = (call["client"] for call in stream.calls)
        assert first is not second
        assert closed == [pinned_transport_of(first), pinned_transport_of(second)]

    async def test_the_attempt_budget_is_the_calls_timeout(self, wire_stream) -> None:
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context(timeout=12.5)))

        assert stream.calls[0]["timeout"] == 12.5

    async def test_no_budget_is_passed_through_as_none(self, wire_stream) -> None:
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context(timeout=None)))

        assert stream.calls[0]["timeout"] is None

    async def test_the_call_is_made_exactly_once_per_attempt(self, wire_stream) -> None:
        provider, stream = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        await collect(provider.stream(make_route(), make_request(), make_context()))

        assert len(stream.calls) == 1

    async def test_each_chunk_is_translated_into_a_canonical_event(self, wire_stream) -> None:
        provider, _ = wire_stream(
            stream_of(
                content_chunk("Recursion"),
                content_chunk(" is a technique"),
                finish_chunk("stop"),
            )
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.kind for event in events] == [
            StreamEventKind.CONTENT,
            StreamEventKind.CONTENT,
            StreamEventKind.FINISH,
        ]
        assert [event.text for event in events if event.kind is StreamEventKind.CONTENT] == [
            "Recursion",
            " is a technique",
        ]
        assert events[-1].finish_reason is FinishReason.STOP

    async def test_the_sequence_numbers_start_at_zero_and_increase(self, wire_stream) -> None:
        provider, _ = wire_stream(
            stream_of(content_chunk("Hello"), content_chunk(" world"), finish_chunk("stop"))
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.sequence for event in events] == [0, 1, 2]

    async def test_every_event_carries_the_invocation_id(self, wire_stream) -> None:
        provider, _ = wire_stream(stream_of(content_chunk("Hello"), finish_chunk("stop")))

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert all(event.invocation_id == "inv-7f3c1a90" for event in events)

    async def test_a_tool_call_delta_is_translated(self, wire_stream) -> None:
        provider, _ = wire_stream(
            stream_of(
                tool_call_chunk(index=0, identifier="call_01", name="get_weather"),
                tool_call_chunk(index=0, arguments='{"city":'),
                tool_call_chunk(index=0, arguments=' "Paris"}'),
                finish_chunk("tool_calls"),
            )
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        tool_events = [event for event in events if event.kind is StreamEventKind.TOOL_CALL]
        assert [event.tool_call.id for event in tool_events] == ["call_01", None, None]
        assert [event.tool_call.name for event in tool_events] == ["get_weather", None, None]
        assert [event.tool_call.arguments_delta for event in tool_events] == [
            None,
            '{"city":',
            ' "Paris"}',
        ]
        assert events[-1].finish_reason is FinishReason.TOOL_CALLS

    async def test_a_chunk_carrying_nothing_reportable_produces_no_event(self, wire_stream) -> None:
        """A keep-alive chunk produces no event, and the counter does not advance."""
        role_only = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant"))]
        )
        provider, _ = wire_stream(
            stream_of(role_only, content_chunk("Hello"), finish_chunk("stop"))
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert [event.sequence for event in events] == [0, 1]

    async def test_usage_arriving_after_the_finish_reason_precedes_the_finish(
        self, wire_stream
    ) -> None:
        """The trailing usage report belongs to the answer, so the finish still comes last.

        This is the order the pinned library produces for the OpenAI dialect when
        usage reporting is on, and the finish reason is buffered so the usage is
        not stranded behind a terminal event that was already emitted.
        """
        provider, _ = wire_stream(
            stream_of(content_chunk("Hello"), finish_chunk("stop"), usage_chunk(25, 42, 67))
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.kind for event in events] == [
            StreamEventKind.CONTENT,
            StreamEventKind.USAGE,
            StreamEventKind.FINISH,
        ]
        assert events[1].usage == Usage(input_tokens=25, output_tokens=42, total_tokens=67)
        assert events[-1].finish_reason is FinishReason.STOP

    async def test_the_buffered_finish_is_renumbered_so_the_sequence_stays_consecutive(
        self, wire_stream
    ) -> None:
        """The usage chunk takes the number the finish would have had, and the finish follows."""
        provider, _ = wire_stream(
            stream_of(
                content_chunk("Hello"),
                finish_chunk("stop"),
                usage_chunk(25, 42, 67),
                usage_chunk(25, 42, 67),
            )
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.kind for event in events] == [
            StreamEventKind.CONTENT,
            StreamEventKind.USAGE,
            StreamEventKind.USAGE,
            StreamEventKind.FINISH,
        ]
        assert [event.sequence for event in events] == [0, 1, 2, 3]

    async def test_a_stream_that_never_reports_a_finish_reason_has_no_terminal_event(
        self, wire_stream
    ) -> None:
        """A clean end is not a completion: only a reported reason ends the stream.

        A clean TCP close and a deliberate completion look alike from here, so
        the finish reason is what tells them apart. Without one the stream is
        truncated, which the contract says ends without a terminal event rather
        than with one claiming a completion that never happened.
        """
        provider, _ = wire_stream(
            stream_of(content_chunk("Hello"), content_chunk(" world"), usage_chunk(25, 42, 67))
        )

        events = await collect(provider.stream(make_route(), make_request(), make_context()))

        assert [event.kind for event in events] == [
            StreamEventKind.CONTENT,
            StreamEventKind.CONTENT,
            StreamEventKind.USAGE,
        ]

    async def test_a_failure_after_the_finish_reason_emits_no_terminal_event(
        self, wire_stream
    ) -> None:
        """The stream did not complete, so the buffered finish is discarded, not emitted.

        A finish reason the upstream reported and then failed to terminate is not
        a completed answer. Emitting the finish would say it was one; emitting it
        and then an error would put two terminal events in one stream. The
        attempt fails, and the gateway core reports that failure as the stream's
        single terminal ``ERROR`` event.
        """
        provider, _ = wire_stream(
            stream_of(content_chunk("Hello"), finish_chunk("stop"), error=upstream_failure())
        )

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in provider.stream(make_route(), make_request(), make_context()):
                events.append(event)

        assert [event.kind for event in events] == [StreamEventKind.CONTENT]

    async def test_a_failure_before_any_finish_reason_emits_no_terminal_event(
        self, wire_stream
    ) -> None:
        provider, _ = wire_stream(stream_of(content_chunk("Hello"), error=upstream_failure()))

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in provider.stream(make_route(), make_request(), make_context()):
                events.append(event)

        assert [event.kind for event in events] == [StreamEventKind.CONTENT]

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param(
                lambda: stream_of(content_chunk("Hello"), finish_chunk("stop")),
                id="completed",
            ),
            pytest.param(
                lambda: stream_of(
                    content_chunk("Hello"), finish_chunk("stop"), usage_chunk(25, 42, 67)
                ),
                id="usage-after-finish",
            ),
            pytest.param(
                lambda: stream_of(content_chunk("Hello")),
                id="ended-without-a-finish-reason",
            ),
            pytest.param(
                lambda: stream_of(
                    content_chunk("Hello"), finish_chunk("stop"), error=upstream_failure()
                ),
                id="failed-after-the-finish-reason",
            ),
            pytest.param(
                lambda: stream_of(content_chunk("Hello"), error=upstream_failure()),
                id="failed-mid-answer",
            ),
            pytest.param(lambda: upstream_failure(), id="failed-before-any-event"),
            pytest.param(
                lambda: stream_of(
                    content_chunk("Hello"),
                    finish_chunk("stop"),
                    usage_chunk(25, 42, 67),
                    error=upstream_failure(),
                ),
                id="failed-after-the-trailing-usage",
            ),
        ],
    )
    async def test_no_path_emits_more_than_one_terminal_event(self, wire_stream, outcome) -> None:
        """The whole point of the buffering: a terminal event is terminal in fact.

        Every path is drained — completion, trailing usage, an upstream that
        stops without a reason, and a failure at each of those points — and none
        of them may put more than one ``FINISH`` or ``ERROR`` into the stream, or
        put anything after one.
        """
        provider, _ = wire_stream(outcome())

        events: list[StreamEvent] = []
        try:
            async for event in provider.stream(make_route(), make_request(), make_context()):
                events.append(event)
        except Exception:
            pass

        terminal = [
            event
            for event in events
            if event.kind in (StreamEventKind.FINISH, StreamEventKind.ERROR)
        ]
        assert len(terminal) <= 1
        if terminal:
            assert events[-1] is terminal[0]

    async def test_a_failure_is_mapped_onto_the_taxonomy(self, wire_stream) -> None:
        provider, _ = wire_stream(
            litellm.APIConnectionError(
                message="the upstream connection failed",
                llm_provider="openai",
                model=UPSTREAM_MODEL,
            )
        )

        with pytest.raises(UnavailableError) as error:
            await collect(provider.stream(make_route(), make_request(), make_context()))

        assert error.value.reason is ErrorReason.PROVIDER_UNAVAILABLE

    async def test_a_gateway_error_from_the_translation_passes_through_unchanged(
        self, wire_stream
    ) -> None:
        refusal = CapabilityError("the upstream chunk carries a semantic the stream cannot carry")
        provider, _ = wire_stream(refusal)

        with pytest.raises(CapabilityError) as error:
            await collect(provider.stream(make_route(), make_request(), make_context()))

        assert error.value is refusal

    async def test_cancellation_propagates_unmapped(self, wire_stream) -> None:
        provider, _ = wire_stream(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await collect(provider.stream(make_route(), make_request(), make_context()))

    async def test_the_client_is_closed_when_the_stream_is_cancelled(
        self, wire_stream, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, stream = wire_stream(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await collect(provider.stream(make_route(), make_request(), make_context()))

        (call,) = stream.calls
        assert closed == [pinned_transport_of(call["client"])]

    async def test_the_provider_satisfies_the_streaming_protocol(self) -> None:
        provider = LiteLLMProvider()

        assert inspect.iscoroutinefunction(provider.stream) is False  # it is an async generator
        assert list(inspect.signature(provider.stream).parameters) == [
            "route",
            "request",
            "context",
        ]


# ---------------------------------------------------------------------------
# The library's own stream ordering
# ---------------------------------------------------------------------------


def _library_kind(chunk: ModelResponseStream) -> str:
    """What one chunk the library produced reports, ignoring everything it says."""
    choices = chunk.choices
    if not choices:
        return "usage"
    choice = choices[0]
    if getattr(choice, "finish_reason", None) is not None:
        return "finish"
    if getattr(getattr(choice, "delta", None), "content", None):
        return "content"
    return "usage"


async def _library_chunks(
    chunks: list[ModelResponseStream], *, stream_options: dict[str, object] | None = None
) -> list[ModelResponseStream]:
    """Drain ``CustomStreamWrapper``, the object ``acompletion(stream=True)`` hands back.

    The wrapper is driven with the library's own chunk records over a local async
    generator, so what runs is the installed library's end-of-stream logic rather
    than a stand-in for it — and no request is made.
    """

    async def generate():
        for chunk in chunks:
            yield chunk

    wrapper = CustomStreamWrapper(
        completion_stream=generate(),
        model=f"openai/{UPSTREAM_MODEL}",
        logging_obj=Logging(
            model=UPSTREAM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="acompletion",
            start_time=time.time(),
            litellm_call_id="contract-probe",
            function_id="contract-probe",
        ),
        custom_llm_provider="openai",
        stream_options=stream_options,
    )
    seen: list[ModelResponseStream] = []
    while True:
        try:
            seen.append(await wrapper.__anext__())
        except StopAsyncIteration:
            return seen


class TestTheLibrariesOwnChunkOrdering:
    """The chunk order the pinned library produces, which the buffering above rests on.

    This is a contract probe rather than a test of the binding: it drives
    ``litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper`` with a
    scripted chunk sequence and pins the order it emits. No network is involved.

    It exists because the buffering in
    :meth:`~model_gateway.providers.litellm_provider.LiteLLMProvider.stream` is a
    response to an observed property of the installed library, and that property
    is exactly what an upgrade would change. A docstring describing the
    assumption can be right and still go stale; this test cannot.
    """

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    async def test_a_trailing_usage_chunk_follows_the_finish_reason(self) -> None:
        """Usage reporting puts the usage report after the reason generation stopped."""
        chunks = await _library_chunks(
            [content_chunk("hi"), finish_chunk("stop"), usage_chunk(25, 42, 67)],
            stream_options={"include_usage": True},
        )
        assert [_library_kind(chunk) for chunk in chunks] == ["content", "finish", "usage"]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    async def test_the_trailing_chunk_carries_no_finish_reason(self) -> None:
        """It is a usage report, not a second terminal: nothing follows it."""
        chunks = await _library_chunks(
            [content_chunk("hi"), finish_chunk("stop"), usage_chunk(25, 42, 67)],
            stream_options={"include_usage": True},
        )
        assert [getattr(chunk.choices[0], "finish_reason", None) for chunk in chunks] == [
            None,
            "stop",
            None,
        ]

    async def test_without_usage_reporting_no_trailing_chunk_is_added(self) -> None:
        """The gateway asks for nothing beyond ``stream``, and gets none back."""
        chunks = await _library_chunks([content_chunk("hi"), finish_chunk("stop")])
        assert [_library_kind(chunk) for chunk in chunks] == ["content", "finish"]

    async def test_a_stream_the_upstream_never_finished_is_given_a_finish_reason(self) -> None:
        """The library synthesises ``stop`` — the limit the provider documents.

        The wrapper finishes a stream it never saw finish, so through it a
        truncated upstream is presented as a completed one and the two cannot be
        told apart from the chunks alone. Pinned here so the limit recorded in
        the provider's docstring is a tested fact rather than a hope.
        """
        chunks = await _library_chunks([content_chunk("hi")])
        assert [_library_kind(chunk) for chunk in chunks] == ["content", "finish"]
        assert getattr(chunks[-1].choices[0], "finish_reason", None) == "stop"


# ---------------------------------------------------------------------------
# Failure mapping
# ---------------------------------------------------------------------------


class TestFailures:
    """Every failure the call raises is mapped onto the gateway taxonomy."""

    @pytest.mark.parametrize(
        ("error", "category", "reason"),
        [
            (
                litellm.AuthenticationError(
                    message="bad key",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(401),
                ),
                UnauthenticatedError,
                "invalid_credentials",
            ),
            (
                litellm.PermissionDeniedError(
                    message="forbidden",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(403),
                ),
                PermissionDeniedError,
                "permission_denied",
            ),
            (
                litellm.NotFoundError(
                    message="no such model",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(404),
                ),
                NotFoundError,
                "resource_not_found",
            ),
            (
                litellm.RateLimitError(
                    message="slow down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(429),
                ),
                ResourceExhaustedError,
                "rate_limited",
            ),
            (
                litellm.Timeout(message="too slow", model=UPSTREAM_MODEL, llm_provider="openai"),
                DeadlineExceededError,
                "deadline_exceeded",
            ),
            (
                litellm.APIConnectionError(
                    message="connection reset",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIResponseValidationError(
                    message="bad shape",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIError(
                    status_code=503,
                    message="down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIError(
                    status_code=400,
                    message="bad",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.BadRequestError(
                    message="bad request",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(400),
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.ContextWindowExceededError(
                    message="too long",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(400),
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.UnsupportedParamsError(
                    message="unsupported param",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.ServiceUnavailableError(
                    message="down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(503),
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APITimeoutError(
                    request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions")
                ),
                DeadlineExceededError,
                "deadline_exceeded",
            ),
            (
                openai.APIConnectionError(
                    request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions")
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APIStatusError("server exploded", response=http_error(500), body=None),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APIStatusError("bad request", response=http_error(400), body=None),
                InternalError,
                "internal_error",
            ),
            (
                RuntimeError("something the binding did not expect"),
                InternalError,
                "internal_error",
            ),
        ],
    )
    async def test_every_failure_is_mapped_onto_the_taxonomy(
        self, wire, error: Exception, category: type, reason: str
    ) -> None:
        provider, _ = wire(error)

        with pytest.raises(category) as excinfo:
            await provider.complete(make_route(), make_request(), make_context())

        assert excinfo.value.reason.value == reason
        assert excinfo.value.__cause__ is error

    async def test_the_upstream_message_never_reaches_the_raised_error(self, wire) -> None:
        """The taxonomy sanitises its own payloads; no upstream text is forwarded."""
        secret = SECRET_KEY
        url = "https://api.openai.com/v1/chat/completions"
        error = litellm.AuthenticationError(
            message=f"invalid key {secret} for {url}",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(401),
        )
        provider, _ = wire(error)

        with pytest.raises(UnauthenticatedError) as excinfo:
            await provider.complete(make_route(), make_request(), make_context())

        payload = excinfo.value.to_payload(include_details=True)
        assert secret not in str(payload)
        assert url not in str(payload)

    async def test_a_gateway_error_from_the_translation_passes_through_unchanged(
        self, wire
    ) -> None:
        """A refusal part has no upstream representation: a refusal, not a mapped failure."""
        provider, _ = wire(response_from(completion_payload()))
        request = make_request(
            messages=(
                Message(
                    role=Role.ASSISTANT,
                    content=(RefusalPart(text="I cannot help with that"),),
                ),
            )
        )

        with pytest.raises(CapabilityError) as excinfo:
            await provider.complete(make_route(), request, make_context())

        assert excinfo.value.details["capability"] == "refusal"

    async def test_cancellation_propagates_unmapped(self, wire) -> None:
        """Cancellation is propagated, not reported as a gateway failure."""
        provider, _ = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.complete(make_route(), make_request(), make_context())

    async def test_the_client_is_closed_when_the_call_is_cancelled(self, wire, monkeypatch) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert closed == [pinned_transport_of(call["client"])]


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class TestTheProtocol:
    """The provider is usable wherever a :class:`TextProvider` is expected."""

    async def test_the_provider_satisfies_the_text_provider_protocol(self) -> None:
        provider = LiteLLMProvider()

        assert inspect.iscoroutinefunction(provider.complete)
        assert list(inspect.signature(provider.complete).parameters) == [
            "route",
            "request",
            "context",
        ]

    async def test_the_provider_works_through_the_protocol_type(self, wire) -> None:
        provider: TextProvider = wire(response_from(completion_payload()))[0]

        response = await provider.complete(make_route(), make_request(), make_context())

        assert response.model == "text-default"

    def test_the_provider_also_carries_the_image_operation(self) -> None:
        """The production service holds one provider, so every operation is on it.

        Without the image entry point the image operation has no provider at all:
        the core refuses every image request as unsupported however the routes are
        configured, so a deployment advertises an endpoint it cannot serve. The
        dialect checks the factory already runs at startup would not catch it,
        because they validate the routes rather than the object the core holds.
        """
        provider = LiteLLMProvider()

        assert isinstance(provider, ImageProvider)


# ---------------------------------------------------------------------------
# The seam is beneath the security boundary
# ---------------------------------------------------------------------------


class TestTheSeamIsBeneathTheBoundary:
    """What a caller may inject is the transport's inputs, never the client.

    The constructor accepts the pinned builder's inputs and nothing else, so no
    public input can replace the per-dialect client construction: it is
    impossible through the public API to obtain an unsafe client for a verified
    dialect, and the attempt does not need to raise to prevent one.
    """

    def test_the_constructor_accepts_no_client_or_client_factory(self) -> None:
        parameters = inspect.signature(LiteLLMProvider.__init__).parameters
        assert set(parameters) == {"self", "resolver", "verify", "inner_factory"}
        for name, parameter in parameters.items():
            if name == "self":
                continue
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
    async def test_custom_transport_inputs_still_yield_a_pinned_non_redirecting_client(
        self, dialect: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every injectable input is custom; the injected client is still the safe one.

        ``verify`` does not reach a supplied inner transport, as the transport
        documents; what the custom inputs must not change is the pinning and
        the refused redirect following, whichever of them is supplied.
        """
        completion = RecordingCompletion(response_from(completion_payload()))
        monkeypatch.setattr(litellm, "acompletion", completion)
        built: list[httpx.AsyncBaseTransport] = []

        def inner_factory() -> httpx.AsyncBaseTransport:
            transport = httpx.AsyncHTTPTransport()
            built.append(transport)
            return transport

        def resolver(host: str) -> tuple[str, ...]:
            return (PUBLIC_ADDRESS,)

        provider = LiteLLMProvider(
            resolver=resolver,
            verify=ssl.create_default_context(),
            inner_factory=inner_factory,
        )

        await provider.complete(make_route(format=dialect), make_request(), make_context())

        (call,) = completion.calls
        client = call["client"]
        if dialect == "openai":
            assert isinstance(client, openai.AsyncOpenAI)
            assert client.max_retries == 0
            http_client = client._client  # pyright: ignore[reportPrivateUsage]
        else:
            assert isinstance(client, AsyncHTTPHandler)
            assert client.follow_redirects is False
            http_client = client.client
            assert http_client._transport is client.transport  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        transport = http_client._transport  # pyright: ignore[reportPrivateUsage]
        assert isinstance(transport, PinnedAddressTransport)
        assert len(built) == 1
        assert transport._inner is built[0]  # pyright: ignore[reportPrivateUsage]
        assert transport._resolver is resolver  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# The .env guard
# ---------------------------------------------------------------------------


class TestTheEnvGuard:
    """Importing the binding must not load a working-directory ``.env``.

    The guard lives in :mod:`model_gateway.providers._litellm`, the seam every
    binding module imports the library through, and it *forces* the mode. These
    tests import a binding module in a subprocess — the provider on its own, and
    the translation module on its own, which crosses the seam with no other
    binding module imported first — with a ``.env`` in the working directory and
    no ``LITELLM_MODE`` in the environment, plus one run with an ambient
    ``LITELLM_MODE=DEV`` to prove the forced mode overrides even an explicit
    value: the mode is gateway-owned implementation state, not operator
    configuration.
    """

    PROBE = "GATEWAY_GUARD_PROBE_VARIABLE"

    def _run_import(
        self, code: str, tmp_path: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_importing_the_binding_leaves_the_working_directory_env_unloaded(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        # The probe must start unset, or the test would pass for the wrong reason.
        env = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
        code = (
            "import model_gateway.providers.litellm_provider\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr

    def test_importing_the_translation_module_directly_leaves_the_env_unloaded(
        self, tmp_path: Path
    ) -> None:
        """The translation module crosses the seam too: its direct import is guarded.

        Importing :mod:`model_gateway.providers.litellm_translation` on its own
        — without the provider module, and without this session's conftest guard —
        must still force the mode, because the seam is the one place the guard
        lives and every binding module imports the library through it rather than
        importing ``litellm`` directly.
        """
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        # The probe must start unset, or the test would pass for the wrong reason.
        env = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
        code = (
            "import model_gateway.providers.litellm_translation\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr

    def test_the_forced_mode_overrides_an_ambient_dev_value(self, tmp_path: Path) -> None:
        """An ambient ``LITELLM_MODE=DEV`` does not survive the seam's forced mode."""
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        env = {**os.environ, "LITELLM_MODE": "DEV"}
        code = (
            "import model_gateway.providers.litellm_provider\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr
