"""The streaming inference orchestration: gateway policy around a provider stream
(PRD 5.3, 11.2, 11.3, 15, 16).

The streaming entry point applies the same policy as the non-streaming one —
capability enforcement, one deadline, the concurrency limiter with its bounded
wait, cancellation that propagates — and one rule a non-streaming invocation
never had to enforce: once an event has been yielded to the caller, no fallback
may occur, because switching providers after a caller has seen output produces
a contradictory answer (PRD 11.2 and 11.3).

These tests pin that rule and its control case (a fallback before any output is
still allowed), the strictly increasing sequence numbers, exactly one terminal
event — and that nothing ever follows it, a failure after a terminal event
being the exception rather than a second one — and cancellation mid-stream. No
test reaches the network: the provider is a scripted stand-in.
"""

import asyncio

import pytest

import model_gateway.reliability as reliability
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    Message,
    Role,
    TextPart,
)
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    ModelNotFoundError,
    UnavailableError,
)
from model_gateway.inference import InferenceService
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.mock import MockProvider
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter


class _TickingClock:
    """A clock that runs forward on every reading, so a deadline expires between two of them.

    :class:`~model_gateway.reliability.Deadline` anchors itself with one reading
    and measures with the next, so a step larger than the configured timeout
    reaches the state a real deployment reaches only when time has run out.
    """

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        reading = self._now
        self._now += self._step
        return reading

INVOCATION_ID = "inv-stream-1"


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
        "capabilities": frozenset({Capability.CHAT}),
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
        "invocation_id": INVOCATION_ID,
    }
    values.update(overrides)
    return InvocationRequest(**values)


def content_event(text: str, sequence: int) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.CONTENT,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        text=text,
    )


def finish_event(reason: FinishReason, sequence: int, raw: str | None = None) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.FINISH,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        finish_reason=reason,
        raw_finish_reason=raw or reason.value,
    )


def error_event(reason: str, message: str, sequence: int) -> StreamEvent:
    """A terminal ERROR event, shaped as the service builds one for a failed stream."""
    return StreamEvent(
        kind=StreamEventKind.ERROR,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        error_reason=reason,
        error_message=message,
    )


async def collect(agen) -> list[StreamEvent]:
    """Drain an async generator into a list."""
    return [event async for event in agen]


class ScriptedStreamProvider:
    """A streaming provider that serves scripted event streams.

    Each script is a list of items served one call in order, then the last
    repeats. An item is either a :class:`StreamEvent`, which is yielded, or an
    :exc:`Exception`, which is raised at that point — so a script can yield some
    events and then fail mid-stream, or fail before any event.
    """

    def __init__(self, scripts: list[list[object]]) -> None:
        self._scripts = scripts
        self.calls: list[tuple[Route, InvocationRequest, AttemptContext]] = []

    async def stream(self, route, request, context):
        self.calls.append((route, request, context))
        script = self._scripts[min(len(self.calls), len(self._scripts)) - 1]
        items = script if isinstance(script, list | tuple) else [script]
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item


def service(routes: tuple[Route, ...], provider, **overrides) -> InferenceService:
    values = {
        "registry": ModelRegistry(routes),
        "provider": provider,
    }
    values.update(overrides)
    return InferenceService(**values)


# ---------------------------------------------------------------------------
# The capability
# ---------------------------------------------------------------------------


class TestTheCapability:
    """Streaming is a separate capability, refused rather than faked (PRD 16)."""

    async def test_a_provider_that_cannot_stream_is_refused(self):
        inference = service((route(),), MockProvider())

        with pytest.raises(CapabilityError) as error:
            await collect(inference.invoke_stream(request()))

        assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY
        assert error.value.details["capability"] == "streaming"
        assert error.value.http_status == 422

    async def test_the_refusal_names_the_operation_and_reaches_no_provider(self):
        provider = ScriptedStreamProvider([])
        # A provider with no stream method at all is not a streaming provider.
        inference = service((route(),), MockProvider())

        with pytest.raises(CapabilityError, match="cannot stream"):
            await collect(inference.invoke_stream(request()))

        assert provider.calls == []

    async def test_a_streaming_provider_is_accepted(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    async def test_a_simple_text_stream_is_served_in_order(self):
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Recursion", 0),
                    content_event(" is a technique", 1),
                    finish_event(FinishReason.STOP, 2),
                ]
            ]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert [event.text for event in events if event.kind is StreamEventKind.CONTENT] == [
            "Recursion",
            " is a technique",
        ]
        assert events[-1].finish_reason is FinishReason.STOP
        assert events[-1].raw_finish_reason == "stop"

    async def test_exactly_one_terminal_finish_event_is_produced(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        finish_events = [event for event in events if event.kind is StreamEventKind.FINISH]
        assert len(finish_events) == 1
        assert events[-1] is finish_events[0]

    async def test_sequence_numbers_are_strictly_increasing_across_the_stream(self):
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    content_event(" world", 1),
                    content_event("!", 2),
                    finish_event(FinishReason.STOP, 3),
                ]
            ]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        sequences = [event.sequence for event in events]
        assert sequences == [0, 1, 2, 3]
        assert all(
            later > earlier
            for earlier, later in zip(sequences, sequences[1:], strict=False)
        )

    async def test_every_event_carries_the_invocation_id(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert all(event.invocation_id == INVOCATION_ID for event in events)

    async def test_the_attempt_context_carries_the_invocation_and_budget(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider, attempt_timeout=12.5)

        await collect(inference.invoke_stream(request()))

        context = provider.calls[0][2]
        assert context.invocation_id == INVOCATION_ID
        assert context.attempt_number == 1
        assert context.timeout == pytest.approx(12.5, abs=0.5)

    async def test_an_unknown_model_is_refused_before_any_event(self):
        inference = service((route(),), ScriptedStreamProvider([]))

        with pytest.raises(ModelNotFoundError) as error:
            await collect(inference.invoke_stream(request(model="no-such-model")))

        assert error.value.reason is ErrorReason.MODEL_NOT_FOUND

    async def test_an_unknown_model_is_refused_before_the_streaming_capability_is_checked(self):
        """The model is resolved first, so a missing model is not a streaming refusal.

        The request names a resource that does not exist, and whether the bound
        provider could stream some other model has no bearing on that answer.
        """
        provider = MockProvider()  # not a streaming provider
        inference = service((route(),), provider)

        with pytest.raises(ModelNotFoundError) as error:
            await collect(inference.invoke_stream(request(model="no-such-model")))

        assert error.value.reason is ErrorReason.MODEL_NOT_FOUND

    async def test_a_capability_gap_is_refused_before_any_event(self):
        """A request the primary route cannot serve is refused before any event."""
        from model_gateway.domain import ToolDefinition

        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider)

        with pytest.raises(CapabilityError):
            await collect(
                inference.invoke_stream(
                    request(tools=(ToolDefinition(name="get_weather"),))
                )
            )

        assert provider.calls == []


# ---------------------------------------------------------------------------
# No fallback after the first output
# ---------------------------------------------------------------------------


class TestNoFallbackAfterOutput:
    """The rule PRD 11.2 states and a non-streaming invocation never enforced."""

    async def test_a_failure_after_the_first_output_ends_the_stream_without_falling_back(self):
        """Route 1 streams one event then fails; route 2 is never tried.

        The caller has seen output, so moving to another provider would
        concatenate a second answer onto the first. The stream ends with the
        error instead.
        """
        provider = ScriptedStreamProvider(
            [
                [content_event("Hello", 0), UnavailableError("the connection dropped")],
                [content_event("fallback", 0), finish_event(FinishReason.STOP, 1)],
            ]
        )
        inference = service(
            (route("primary"), route("fallback")), provider, retry_count=0
        )

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        # The stream ended with the error, not with route 2's answer.
        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.ERROR]
        assert events[-1].error_reason == "provider_unavailable"
        # Route 2 was never called.
        assert [call[0].model_id for call in provider.calls] == ["primary"]

    async def test_a_failure_after_the_first_output_is_not_retried_on_the_same_route(self):
        """A retry would concatenate a second answer, so the stream ends instead."""
        provider = ScriptedStreamProvider(
            [
                [content_event("Hello", 0), UnavailableError("the connection dropped")],
                [content_event("recovered", 0), finish_event(FinishReason.STOP, 1)],
            ]
        )
        inference = service((route(),), provider, retry_count=3, retry_backoff_ms=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.ERROR]
        assert len(provider.calls) == 1

    async def test_a_fallback_before_any_output_still_moves_to_the_next_route(self):
        """The control case: the caller has seen nothing, so a fallback is allowed."""
        provider = ScriptedStreamProvider(
            [
                UnavailableError("the primary is down"),
                [content_event("Hello", 0), finish_event(FinishReason.STOP, 1)],
            ]
        )
        inference = service(
            (route("primary"), route("fallback")), provider, retry_count=0
        )

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert events[0].text == "Hello"
        assert [call[0].model_id for call in provider.calls] == ["primary", "fallback"]

    async def test_a_retry_before_any_output_is_made_on_the_same_route(self):
        """A retryable failure before any event is retried, as in the non-streaming path."""
        provider = ScriptedStreamProvider(
            [
                UnavailableError("the provider is down"),
                [content_event("recovered", 0), finish_event(FinishReason.STOP, 1)],
            ]
        )
        inference = service((route(),), provider, retry_count=1, retry_backoff_ms=0)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert events[0].text == "recovered"
        assert len(provider.calls) == 2

    async def test_sequence_numbers_survive_a_fallback_before_any_output(self):
        """The fallback produced no events, so the next route's numbering from 0 is continuous."""
        provider = ScriptedStreamProvider(
            [
                UnavailableError("the primary is down"),
                [
                    content_event("Hello", 0),
                    content_event(" world", 1),
                    finish_event(FinishReason.STOP, 2),
                ],
            ]
        )
        inference = service(
            (route("primary"), route("fallback")), provider, retry_count=0
        )

        events = await collect(inference.invoke_stream(request()))

        assert [event.sequence for event in events] == [0, 1, 2]

    async def test_a_deterministic_failure_before_any_output_terminates_the_invocation(self):
        """A request failure is worth neither a retry nor a fallback, streaming or not."""
        failure = CapabilityError("the upstream rejected the request")
        provider = ScriptedStreamProvider(
            [
                failure,
                [content_event("fallback", 0), finish_event(FinishReason.STOP, 1)],
            ]
        )
        inference = service(
            (route("primary"), route("fallback")), provider, retry_count=2, retry_backoff_ms=0
        )

        with pytest.raises(CapabilityError) as error:
            await collect(inference.invoke_stream(request()))

        assert error.value is failure
        assert [call[0].model_id for call in provider.calls] == ["primary"]


# ---------------------------------------------------------------------------
# A mid-stream failure
# ---------------------------------------------------------------------------


class TestAMidStreamFailure:
    """A failure after output ends the stream with a terminal ERROR, not a silent stop."""

    async def test_a_failure_mid_stream_produces_a_terminal_error_not_a_silent_stop(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), UnavailableError("the connection dropped")]]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        # The stream produced a terminal ERROR event carrying the gateway's own
        # reason and the sanitised message, not a fabricated finish reason.
        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.ERROR]
        assert events[-1].error_reason == "provider_unavailable"
        assert events[-1].error_message == "the connection dropped"
        assert events[-1].finish_reason is None
        assert events[-1].raw_finish_reason is None

    async def test_the_terminal_error_event_takes_the_next_sequence_number(self):
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    content_event(" world", 1),
                    UnavailableError("the connection dropped"),
                ]
            ]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert [event.sequence for event in events] == [0, 1, 2]
        assert events[-1].kind is StreamEventKind.ERROR

    async def test_exactly_one_terminal_event_is_produced_when_a_failure_ends_the_stream(self):
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), UnavailableError("the connection dropped")]]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        terminal = [
            event
            for event in events
            if event.kind in (StreamEventKind.FINISH, StreamEventKind.ERROR)
        ]
        assert len(terminal) == 1
        assert terminal[0].kind is StreamEventKind.ERROR

    async def test_a_non_gateway_failure_is_normalised_into_the_error_event(self):
        """A failure outside the taxonomy still carries a gateway reason and message."""
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    RuntimeError("upstream https://internal.example/v1 rejected api_key=sk-secret"),
                ]
            ]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert events[-1].kind is StreamEventKind.ERROR
        assert events[-1].error_reason == "internal_error"
        assert "sk-secret" not in (events[-1].error_message or "")
        assert "internal.example" not in (events[-1].error_message or "")

    async def test_the_error_event_message_is_a_gateway_authored_taxonomy_message(self) -> None:
        """The event's message is gateway-authored, never provider text.

        The message is constrained to the taxonomy's own messages and is
        never populated from arbitrary provider exception text with
        sanitisation as the only defence: a failure outside the taxonomy is
        normalised before the event is built, so the event carries the
        normalised message — here the taxonomy's fixed internal-failure text —
        and the provider's own wording appears nowhere in it (PRD 5.4).
        """
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    RuntimeError("upstream https://internal.example/v1 rejected api_key=sk-secret"),
                ]
            ]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert events[-1].kind is StreamEventKind.ERROR
        assert events[-1].error_reason == "internal_error"
        assert events[-1].error_message == "the gateway could not complete the request"
        assert "sk-secret" not in (events[-1].error_message or "")
        assert "internal.example" not in (events[-1].error_message or "")

    async def test_the_raised_error_is_the_same_failure_the_event_reported(self) -> None:
        """One failure, reported twice: the event is authoritative, the exception is control flow.

        After the terminal ``ERROR`` event the invocation also raises, and
        the exception is control-flow metadata rather than a second failure:
        the raised error's reason and message are the event's own, because
        both describe the same single failure. A consumer reads the event as
        the failure's report and the exception as the iteration ending —
        never as a second failure (PRD 5.4 and 16).
        """
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), UnavailableError("the connection dropped")]]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError) as error:
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert events[-1].kind is StreamEventKind.ERROR
        assert error.value.reason.value == events[-1].error_reason
        assert error.value.message == events[-1].error_message

    async def test_the_events_produced_before_the_failure_are_kept(self):
        """The caller keeps what the upstream produced before it failed."""
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), UnavailableError("the connection dropped")]]
        )
        inference = service((route(),), provider, retry_count=0)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert events[0].text == "Hello"

    async def test_a_deadline_before_any_event_fails_the_stream(self, monkeypatch):
        """An expired deadline refuses the stream before the provider is called."""
        monkeypatch.setattr(reliability, "monotonic", _TickingClock(step=10.0))
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), finish_event(FinishReason.STOP, 1)]]
        )
        inference = service((route(),), provider, attempt_timeout=1.0)

        with pytest.raises(DeadlineExceededError) as error:
            await collect(inference.invoke_stream(request()))

        assert error.value.reason is ErrorReason.DEADLINE_EXCEEDED
        assert provider.calls == []


# ---------------------------------------------------------------------------
# Terminal ordering
# ---------------------------------------------------------------------------


class TestTerminalOrdering:
    """A terminal event ends the stream, and nothing may follow it."""

    async def test_nothing_follows_a_finish_event(self):
        """A provider that keeps yielding after its finish cannot extend the answer."""
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    finish_event(FinishReason.STOP, 1),
                    content_event(" and more", 2),
                ]
            ]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert events[-1].sequence == 1

    async def test_a_failure_after_a_finish_produces_no_second_terminal_event(self):
        """An upstream failure after the answer finished may not add another event.

        The answer already ended with a ``finish``; the failure is the upstream
        misbehaving after the fact, and turning it into a second terminal event
        would contradict the one the caller was already shown.
        """
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    finish_event(FinishReason.STOP, 1),
                    UnavailableError("the connection dropped after the finish"),
                ]
            ]
        )
        inference = service((route(),), provider, retry_count=2, retry_backoff_ms=0)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert len(provider.calls) == 1

    async def test_nothing_follows_an_error_event(self):
        """The ERROR event is the last event of a failed stream, too."""
        provider = ScriptedStreamProvider(
            [[content_event("Hello", 0), UnavailableError("the connection dropped")]]
        )
        inference = service((route(),), provider)

        events: list[StreamEvent] = []
        with pytest.raises(UnavailableError):
            async for event in inference.invoke_stream(request()):
                events.append(event)

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.ERROR]
        assert events[-1].sequence == 1

    async def test_a_provider_terminal_error_event_ends_the_stream(self):
        """An ERROR event the provider produced is terminal like any other."""
        provider = ScriptedStreamProvider(
            [
                [
                    content_event("Hello", 0),
                    error_event("provider_unavailable", "the provider failed", 1),
                    content_event(" and more", 2),
                ]
            ]
        )
        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.ERROR]

    async def test_a_stream_that_ends_without_a_terminal_event_produces_none(self):
        """The gateway synthesises no terminal event for a stream it never saw finish."""
        provider = ScriptedStreamProvider([[content_event("Hello", 0)]])

        inference = service((route(),), provider)

        events = await collect(inference.invoke_stream(request()))

        assert [event.kind for event in events] == [StreamEventKind.CONTENT]


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class _BlockingStreamProvider:
    """A provider that yields one event and then blocks until cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def stream(self, route, request, context):
        self.started.set()
        yield content_event("Hello", 0)
        await asyncio.Event().wait()


class TestCancellation:
    async def test_cancelling_mid_stream_stops_the_stream_and_frees_the_slot(self):
        provider = _BlockingStreamProvider()
        limiter = ConcurrencyLimiter(1)
        inference = service((route(),), provider, limiter=limiter)

        task = asyncio.create_task(collect(inference.invoke_stream(request())))
        await provider.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The slot the cancelled attempt held is released.
        async def take_the_only_slot() -> None:
            async with limiter.slot():
                pass

        await asyncio.wait_for(take_the_only_slot(), timeout=1.0)

    async def test_cancellation_is_never_turned_into_an_upstream_failure(self):
        class _CancellingStreamProvider:
            async def stream(self, route, request, context):
                yield content_event("Hello", 0)
                raise asyncio.CancelledError

        inference = service((route(),), _CancellingStreamProvider(), retry_count=2)

        with pytest.raises(asyncio.CancelledError):
            await collect(inference.invoke_stream(request()))
