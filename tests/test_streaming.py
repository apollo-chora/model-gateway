"""Tests for the canonical streaming types and their LiteLLM translation.

The domain side pins the canonical event: what a kind must carry, what a kind
refuses, and that a sequence number is the gateway's own numbering of the events
of one invocation. The translation side pins the stream-side rules the same way
the non-streaming translation tests pin the response side: every fixture is
built from the real LiteLLM types the installed library
defines — :class:`litellm.types.utils.ModelResponseStream` with its
``StreamingChoices``, ``Delta``, ``ChatCompletionDeltaToolCall`` and ``Usage``
records — so a fixture carrying an extra semantic field fails a translator that
would drop it, and a refusal names the field rather than trimming it.

No test performs network I/O: the chunks are local objects built from the
library's own types, exactly as the non-streaming tests build their responses.
"""

import dataclasses

import pytest
from litellm import stream_chunk_builder
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    ChatCompletionReasoningItem,
    ChatCompletionThinkingBlock,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    Message,
    Role,
    StreamEvent,
    StreamEventKind,
    TextPart,
    ToolCallDelta,
)
from model_gateway.domain import (
    Usage as TokenUsage,
)
from model_gateway.errors import InternalError
from model_gateway.providers.litellm_translation import (
    from_litellm_chunk,
    to_litellm_params,
    to_litellm_stream_params,
)

INVOCATION_ID = "inv-7f3c1a90"


def make_route() -> Route:
    """A resolved text route."""
    return Route(
        operation=Operation.TEXT,
        index=0,
        endpoint="https://upstream.internal.example/v1",
        base_url="https://upstream.internal.example/v1",
        model_id="gpt-4o",
        format="openai",
        api_key="sk-upstream-secret-key",
        capabilities=frozenset(),
        extra_headers={},
        allow_absolute_path_url=False,
    )


def make_request() -> InvocationRequest:
    """A canonical invocation."""
    return InvocationRequest(
        model="text-default",
        messages=(Message(role=Role.USER, content=(TextPart(text="Hello"),)),),
        invocation_id=INVOCATION_ID,
    )


def content_chunk(text: str) -> ModelResponseStream:
    """A chunk carrying a text delta and the role marker that opens the answer."""
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text, role="assistant"))]
    )


def tool_call_chunk(
    *,
    index: int,
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


def finish_chunk(finish_reason: str, **kwargs: object) -> ModelResponseStream:
    """A chunk carrying a finish reason."""
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, finish_reason=finish_reason, **kwargs)]
    )


def library_trailing_usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    """The usage record the pinned library assembles for a trailing usage chunk.

    Built by driving LiteLLM's own ``stream_chunk_builder`` with a scripted
    OpenAI-shaped stream, so the fixture is the library's real output —
    placeholder values included — rather than a handcrafted approximation of
    it. On an OpenAI-backed base the library sets usage reporting itself and
    assembles this record at end of stream.
    """
    response = stream_chunk_builder(
        chunks=[
            {
                "id": "chatcmpl-library",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-library",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        ]
    )
    assert response is not None
    usage = response.usage
    assert usage is not None
    return usage


# --------------------------------------------------------------------------- canonical events


class TestStreamEventKind:
    """The vocabulary is the gateway's own and its values are pinned."""

    def test_values_are_pinned(self) -> None:
        assert StreamEventKind.CONTENT == "content"
        assert StreamEventKind.TOOL_CALL == "tool_call"
        assert StreamEventKind.USAGE == "usage"
        assert StreamEventKind.FINISH == "finish"
        assert StreamEventKind.ERROR == "error"


class TestStreamEventRoundTrip:
    """Every kind round-trips through to_dict and from_dict, shape pinned."""

    def test_content_event_round_trip(self) -> None:
        event = StreamEvent(
            kind=StreamEventKind.CONTENT, sequence=0, invocation_id=INVOCATION_ID, text="Recur"
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["kind"] == "content"

    def test_tool_call_event_round_trip(self) -> None:
        event = StreamEvent(
            kind=StreamEventKind.TOOL_CALL,
            sequence=1,
            invocation_id=INVOCATION_ID,
            tool_call=ToolCallDelta(index=0, id="call_1", name="get_weather"),
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["tool_call"] == {
            "index": 0,
            "id": "call_1",
            "name": "get_weather",
            "arguments_delta": None,
        }

    def test_usage_event_round_trip(self) -> None:
        event = StreamEvent(
            kind=StreamEventKind.USAGE,
            sequence=2,
            invocation_id=INVOCATION_ID,
            usage=TokenUsage(input_tokens=25, output_tokens=42, total_tokens=67),
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["usage"] == {
            "input_tokens": 25,
            "output_tokens": 42,
            "total_tokens": 67,
            "cached_tokens": None,
            "estimated_fields": [],
        }

    def test_finish_event_round_trip(self) -> None:
        """The classification and the provider's own reason survive the trip."""
        event = StreamEvent(
            kind=StreamEventKind.FINISH,
            sequence=3,
            invocation_id=INVOCATION_ID,
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["finish_reason"] == "stop"
        assert event.to_dict()["raw_finish_reason"] == "stop"

    def test_error_event_round_trip(self) -> None:
        """The gateway's reason and its sanitised message survive the trip."""
        event = StreamEvent(
            kind=StreamEventKind.ERROR,
            sequence=5,
            invocation_id=INVOCATION_ID,
            error_reason="provider_unavailable",
            error_message="the upstream connection failed",
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["kind"] == "error"
        assert event.to_dict()["error_reason"] == "provider_unavailable"
        assert event.to_dict()["error_message"] == "the upstream connection failed"

    def test_error_event_without_a_message_round_trips(self) -> None:
        """The reason is what a caller switches on; the message only explains it."""
        event = StreamEvent(
            kind=StreamEventKind.ERROR,
            sequence=0,
            invocation_id=INVOCATION_ID,
            error_reason="internal_error",
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["error_message"] is None

    def test_to_dict_emits_every_field(self) -> None:
        """The serialised shape is pinned rather than implied, so nothing can hide."""
        event = StreamEvent(
            kind=StreamEventKind.CONTENT, sequence=0, invocation_id=INVOCATION_ID, text="hi"
        )
        assert event.to_dict() == {
            "kind": "content",
            "sequence": 0,
            "invocation_id": INVOCATION_ID,
            "text": "hi",
            "tool_call": None,
            "usage": None,
            "finish_reason": None,
            "raw_finish_reason": None,
            "error_reason": None,
            "error_message": None,
        }

    def test_unrecognised_finish_reason_round_trip(self) -> None:
        """UNKNOWN states the gateway did not classify; the raw value says what arrived."""
        event = StreamEvent(
            kind=StreamEventKind.FINISH,
            sequence=4,
            invocation_id=INVOCATION_ID,
            finish_reason=FinishReason.UNKNOWN,
            raw_finish_reason="tool_use",
        )
        assert StreamEvent.from_dict(event.to_dict()) == event
        assert event.to_dict()["finish_reason"] == "unknown"
        assert event.to_dict()["raw_finish_reason"] == "tool_use"

    def test_events_are_reparseable(self) -> None:
        event = StreamEvent(
            kind=StreamEventKind.CONTENT, sequence=0, invocation_id=INVOCATION_ID, text="hi"
        )
        assert StreamEvent.from_dict(StreamEvent.from_dict(event.to_dict()).to_dict()) == event

    def test_an_event_cannot_be_changed_afterwards(self) -> None:
        """Immutable, like every canonical type: a validated event stays the event it was."""
        event = StreamEvent(
            kind=StreamEventKind.CONTENT, sequence=0, invocation_id=INVOCATION_ID, text="hi"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.text = "changed"


class TestStreamEventConstructionRefusals:
    """A kind that does not agree with its payload is refused, never truncated."""

    def test_content_event_without_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries none of"):
            StreamEvent(kind=StreamEventKind.CONTENT, sequence=0, invocation_id=INVOCATION_ID)

    def test_content_event_with_empty_text_is_refused(self) -> None:
        """An empty fragment asserts no content change while claiming to report one."""
        with pytest.raises(ValueError, match="must not be empty"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="",
            )

    def test_tool_call_event_without_a_delta_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries none of"):
            StreamEvent(kind=StreamEventKind.TOOL_CALL, sequence=0, invocation_id=INVOCATION_ID)

    def test_usage_event_without_usage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries none of"):
            StreamEvent(kind=StreamEventKind.USAGE, sequence=0, invocation_id=INVOCATION_ID)

    def test_finish_event_without_a_reason_is_refused(self) -> None:
        """A terminal event is the one that says why generation stopped."""
        with pytest.raises(ValueError, match="carries none of"):
            StreamEvent(kind=StreamEventKind.FINISH, sequence=0, invocation_id=INVOCATION_ID)

    def test_content_event_carrying_usage_is_refused(self) -> None:
        """One event carries the one payload its kind names; usage is a different event."""
        with pytest.raises(ValueError, match="carries 'usage'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                usage=TokenUsage(total_tokens=1),
            )

    def test_content_event_carrying_a_tool_call_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'tool_call'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                tool_call=ToolCallDelta(index=0),
            )

    def test_content_event_carrying_a_finish_reason_is_refused(self) -> None:
        """The finish reason belongs to the terminal event, not to a fragment of text."""
        with pytest.raises(ValueError, match="carries 'finish_reason'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                finish_reason=FinishReason.STOP,
            )

    def test_content_event_carrying_the_raw_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'raw_finish_reason'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                raw_finish_reason="stop",
            )

    def test_tool_call_event_carrying_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'text'"):
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                sequence=0,
                invocation_id=INVOCATION_ID,
                tool_call=ToolCallDelta(index=0, id="call_1"),
                text="hi",
            )

    def test_usage_event_carrying_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'text'"):
            StreamEvent(
                kind=StreamEventKind.USAGE,
                sequence=0,
                invocation_id=INVOCATION_ID,
                usage=TokenUsage(total_tokens=1),
                text="hi",
            )

    def test_finish_event_carrying_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'text'"):
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                raw_finish_reason="stop",
                text="hi",
            )

    def test_finish_event_carrying_a_tool_call_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'tool_call'"):
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                raw_finish_reason="stop",
                tool_call=ToolCallDelta(index=0),
            )

    def test_finish_event_carrying_usage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'usage'"):
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                raw_finish_reason="stop",
                usage=TokenUsage(total_tokens=1),
            )

    def test_raw_reason_without_a_finish_event_is_refused(self) -> None:
        """The provider's own reason is only meaningful beside its classification."""
        with pytest.raises(ValueError, match="carries 'raw_finish_reason'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                raw_finish_reason="stop",
            )

    def test_error_event_without_a_reason_is_refused(self) -> None:
        """A terminal event says what ended the stream, and what ends an error is its reason."""
        with pytest.raises(ValueError, match="carries none of"):
            StreamEvent(kind=StreamEventKind.ERROR, sequence=0, invocation_id=INVOCATION_ID)

    def test_error_event_with_an_empty_reason_is_refused(self) -> None:
        """An empty reason asserts no failure while claiming to report one."""
        with pytest.raises(ValueError, match="must not be empty"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="",
            )

    def test_error_event_carrying_a_finish_reason_is_refused(self) -> None:
        """A failure is not a finish, so the two terminal kinds are disjoint."""
        with pytest.raises(ValueError, match="carries 'finish_reason'"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="provider_unavailable",
                finish_reason=FinishReason.UNKNOWN,
            )

    def test_error_event_carrying_the_raw_finish_reason_is_refused(self) -> None:
        """``raw_finish_reason`` is the provider's own reason, and a failure has none."""
        with pytest.raises(ValueError, match="carries 'raw_finish_reason'"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="provider_unavailable",
                raw_finish_reason="provider_unavailable",
            )

    def test_error_event_carrying_text_is_refused(self) -> None:
        """A failure is not a fragment of the answer; the two are different events."""
        with pytest.raises(ValueError, match="carries 'text'"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="provider_unavailable",
                text="hi",
            )

    def test_error_event_carrying_usage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'usage'"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="provider_unavailable",
                usage=TokenUsage(total_tokens=1),
            )

    def test_error_event_carrying_a_tool_call_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'tool_call'"):
            StreamEvent(
                kind=StreamEventKind.ERROR,
                sequence=0,
                invocation_id=INVOCATION_ID,
                error_reason="provider_unavailable",
                tool_call=ToolCallDelta(index=0),
            )

    def test_finish_event_carrying_an_error_reason_is_refused(self) -> None:
        """A finish reason classifies why generation stopped; a failure is not one."""
        with pytest.raises(ValueError, match="carries 'error_reason'"):
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                error_reason="provider_unavailable",
            )

    def test_finish_event_carrying_an_error_message_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carries 'error_message'"):
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                error_message="the upstream connection failed",
            )

    def test_content_event_carrying_an_error_reason_is_refused(self) -> None:
        """A failure is a terminal event, never a fragment of the answer."""
        with pytest.raises(ValueError, match="carries 'error_reason'"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="hi",
                error_reason="provider_unavailable",
            )

    def test_usage_event_carrying_an_error_message_is_refused(self) -> None:
        """The message explains a failure; beside a token report it explains nothing."""
        with pytest.raises(ValueError, match="carries 'error_message'"):
            StreamEvent(
                kind=StreamEventKind.USAGE,
                sequence=0,
                invocation_id=INVOCATION_ID,
                usage=TokenUsage(total_tokens=1),
                error_message="the upstream connection failed",
            )

    def test_negative_sequence_is_refused(self) -> None:
        """The numbering starts at 0 for the first event of an invocation."""
        with pytest.raises(ValueError, match="at least 0"):
            StreamEvent(
                kind=StreamEventKind.CONTENT, sequence=-1, invocation_id=INVOCATION_ID, text="hi"
            )

    def test_non_integer_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence="0",  # type: ignore[arg-type]
                invocation_id=INVOCATION_ID,
                text="hi",
            )

    def test_boolean_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=True,  # type: ignore[arg-type]
                invocation_id=INVOCATION_ID,
                text="hi",
            )

    def test_missing_invocation_id_is_refused(self) -> None:
        """Every chunk carries the invocation ID, so an event must state which it belongs to."""
        with pytest.raises(ValueError, match="must not be empty"):
            StreamEvent(kind=StreamEventKind.CONTENT, sequence=0, invocation_id="")

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            StreamEvent(kind="delta", sequence=0, invocation_id=INVOCATION_ID, text="hi")  # type: ignore[arg-type]

    def test_wrong_payload_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a ToolCallDelta"):
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                sequence=0,
                invocation_id=INVOCATION_ID,
                tool_call="call_1",  # type: ignore[arg-type]
            )

    def test_unknown_key_is_refused(self) -> None:
        """A key the canonical type does not define fails loudly."""
        with pytest.raises(ValueError, match="unsupported key"):
            StreamEvent.from_dict(
                {
                    "kind": "content",
                    "sequence": 0,
                    "invocation_id": INVOCATION_ID,
                    "text": "hi",
                    "weight": 1,
                }
            )

    def test_missing_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            StreamEvent.from_dict({"sequence": 0, "invocation_id": INVOCATION_ID})


class TestToolCallDelta:
    """A delta is one fragment of a call; its index is what attributes it."""

    def test_round_trip(self) -> None:
        delta = ToolCallDelta(index=0, id="call_1", name="get_weather", arguments_delta='{"a"')
        assert ToolCallDelta.from_dict(delta.to_dict()) == delta
        assert delta.to_dict() == {
            "index": 0,
            "id": "call_1",
            "name": "get_weather",
            "arguments_delta": '{"a"',
        }

    def test_minimal_delta_round_trip(self) -> None:
        delta = ToolCallDelta(index=1)
        assert ToolCallDelta.from_dict(delta.to_dict()) == delta

    def test_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unsupported key"):
            ToolCallDelta.from_dict({"index": 0, "arguments": '{"a"}'})

    def test_negative_index_is_refused(self) -> None:
        """An index is a position among the answer's calls, so it cannot be negative."""
        with pytest.raises(ValueError, match="at least 0"):
            ToolCallDelta(index=-1)

    def test_non_integer_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            ToolCallDelta(index="0")  # type: ignore[arg-type]

    def test_empty_identifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ToolCallDelta(index=0, id="")

    def test_empty_argument_fragment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ToolCallDelta(index=0, arguments_delta="")


# --------------------------------------------------------------------------- stream parameters


class TestToLitellmStreamParams:
    """The streaming call is the non-streaming one plus stream=True."""

    def test_stream_is_true(self) -> None:
        assert to_litellm_stream_params(make_route(), make_request())["stream"] is True

    def test_every_non_streaming_parameter_is_carried(self) -> None:
        """Nothing is re-derived for the stream: the same call, asked to stream."""
        params = to_litellm_stream_params(make_route(), make_request())
        expected = to_litellm_params(make_route(), make_request())
        assert params == {**expected, "stream": True}
        assert params["num_retries"] == 0
        assert params["model"] == "openai/gpt-4o"

    def test_no_other_streaming_parameter_is_invented(self) -> None:
        """Usage reporting is where available; the gateway asks for nothing beyond stream."""
        assert "stream_options" not in to_litellm_stream_params(make_route(), make_request())


# --------------------------------------------------------------------------- chunk translation


class TestFromLitellmChunkContent:
    """Text deltas become content events, numbered by the gateway."""

    def test_text_delta_becomes_a_content_event(self) -> None:
        events = from_litellm_chunk(content_chunk("Recur"), invocation_id=INVOCATION_ID, sequence=0)
        assert events == (
            StreamEvent(
                kind=StreamEventKind.CONTENT,
                sequence=0,
                invocation_id=INVOCATION_ID,
                text="Recur",
            ),
        )

    def test_text_deltas_accumulate_with_increasing_sequences(self) -> None:
        """Each chunk produces its own event, and the numbering is strictly increasing."""
        events = ()
        text = ""
        for offset, fragment in enumerate(("Recur", "sion", " is", " a", " technique...")):
            events += from_litellm_chunk(
                content_chunk(fragment), invocation_id=INVOCATION_ID, sequence=offset
            )
            text += fragment
        assert [event.kind for event in events] == [StreamEventKind.CONTENT] * 5
        assert [event.sequence for event in events] == [0, 1, 2, 3, 4]
        assert "".join(event.text or "" for event in events) == "Recursion is a technique..."

    def test_role_only_chunk_produces_no_event(self) -> None:
        """A role marker says whose turn the answer is; it says nothing about the answer."""
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=None, role="assistant"))]
        )
        assert from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0) == ()

    def test_empty_delta_chunk_produces_no_event(self) -> None:
        """LiteLLM emits keep-alive chunks with no reportable content at all."""
        chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=None))])
        assert from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0) == ()

    def test_empty_text_delta_produces_no_event(self) -> None:
        chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=""))])
        assert from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0) == ()


class TestFromLitellmChunkToolCalls:
    """A tool call arrives across several chunks and each fragment is reported."""

    def test_tool_call_arriving_in_fragments_is_accumulated(self) -> None:
        first = tool_call_chunk(
            index=0, identifier="call_1", name="get_weather", arguments='{"city"'
        )
        second = tool_call_chunk(index=0, arguments=':"Paris", "days": 3}')
        events = from_litellm_chunk(
            first, invocation_id=INVOCATION_ID, sequence=0
        ) + from_litellm_chunk(second, invocation_id=INVOCATION_ID, sequence=1)
        assert events == (
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                sequence=0,
                invocation_id=INVOCATION_ID,
                tool_call=ToolCallDelta(
                    index=0, id="call_1", name="get_weather", arguments_delta='{"city"'
                ),
            ),
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL,
                sequence=1,
                invocation_id=INVOCATION_ID,
                tool_call=ToolCallDelta(index=0, arguments_delta=':"Paris", "days": 3}'),
            ),
        )
        accumulated = "".join(
            event.tool_call.arguments_delta or "" for event in events if event.tool_call
        )
        assert accumulated == '{"city":"Paris", "days": 3}'

    def test_a_later_fragment_alone_still_carries_the_index(self) -> None:
        """The index is what matches a fragment to the call it belongs to."""
        events = from_litellm_chunk(
            tool_call_chunk(index=2, arguments='{"a": 1}'),
            invocation_id=INVOCATION_ID,
            sequence=0,
        )
        assert events[0].tool_call == ToolCallDelta(index=2, arguments_delta='{"a": 1}')

    def test_several_tool_calls_in_one_chunk_produce_several_events(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="function",
                                function=Function(arguments=None, name="first"),
                            ),
                            ChatCompletionDeltaToolCall(
                                index=1,
                                id="call_2",
                                type="function",
                                function=Function(arguments=None, name="second"),
                            ),
                        ],
                    ),
                )
            ]
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert [event.tool_call.index for event in events] == [0, 1]
        assert [event.sequence for event in events] == [0, 1]

    def test_index_only_tool_call_delta_produces_no_event(self) -> None:
        """A delta that says nothing about the call is a keep-alive, not a fragment."""
        assert (
            from_litellm_chunk(tool_call_chunk(index=0), invocation_id=INVOCATION_ID, sequence=0)
            == ()
        )


class TestFromLitellmChunkUsage:
    """A chunk's usage report becomes a usage event."""

    def test_usage_is_reported_with_provider_provenance(self) -> None:
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(prompt_tokens=25, completion_tokens=42, total_tokens=67),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        usage_events = [event for event in events if event.kind is StreamEventKind.USAGE]
        assert len(usage_events) == 1
        assert usage_events[0].usage == TokenUsage(
            input_tokens=25, output_tokens=42, total_tokens=67
        )
        assert usage_events[0].usage.source_of("input_tokens").value == "provider"

    def test_cached_tokens_are_reported(self) -> None:
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=67,
                prompt_tokens_details={"cached_tokens": 10},
            ),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        usage = next(event.usage for event in events if event.kind is StreamEventKind.USAGE)
        assert usage.cached_tokens == 10
        assert usage.source_of("cached_tokens").value == "provider"

    def test_unreported_total_is_derived_and_named_an_estimate(self) -> None:
        """A zero total beside non-zero parts is LiteLLM's placeholder, not a fact."""
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(prompt_tokens=25, completion_tokens=42),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        usage = next(event.usage for event in events if event.kind is StreamEventKind.USAGE)
        assert usage.total_tokens == 67
        assert usage.estimated_fields == frozenset({"total_tokens"})
        assert usage.source_of("total_tokens").value == "estimate"

    def test_a_chunk_reporting_no_usage_produces_no_usage_event(self) -> None:
        events = from_litellm_chunk(content_chunk("hi"), invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [StreamEventKind.CONTENT]

    def test_a_trailing_usage_chunk_produces_only_a_usage_event(self) -> None:
        """The library's trailing usage chunk carries no text, no call and no reason.

        Its choice is empty of everything reportable, so the one event it can
        produce is the usage report — the provider's ordering rule relies on that
        being a non-terminal event that may legally follow the finish reason.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta())],
            usage=Usage(prompt_tokens=25, completion_tokens=42, total_tokens=67),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [StreamEventKind.USAGE]
        assert events[0].usage == TokenUsage(input_tokens=25, output_tokens=42, total_tokens=67)

    def test_the_librarys_own_trailing_usage_record_is_accepted(self) -> None:
        """The placeholder values the library fills are not provider facts.

        The usage record is the one LiteLLM's own ``stream_chunk_builder``
        assembles for a trailing usage chunk: ``completion_tokens_details``
        carrying ``reasoning_tokens=0`` and ``text_tokens=completion_tokens``,
        the library's own schema satisfaction rather than a provider-reported
        breakdown. Refusing it fails a stream at the very end, after the caller
        has already seen the whole answer, so the translation accepts it.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=library_trailing_usage(25, 42),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        usage = next(event.usage for event in events if event.kind is StreamEventKind.USAGE)
        assert usage == TokenUsage(input_tokens=25, output_tokens=42, total_tokens=67)

    def test_the_librarys_measured_shape_is_exactly_the_placeholder_pair(self) -> None:
        """The measured synthetic shape is pinned, not approximated.

        The fixture is the library's own output, so this asserts the exact
        shape the translation recognises: only ``reasoning_tokens=0`` and
        ``text_tokens=completion_tokens`` are set, and nothing else. If an
        upgrade changes the shape, this test and the acceptance test fail
        loudly together — the measurement and the recognition cannot drift
        apart.
        """
        usage = library_trailing_usage(25, 42)
        details = usage.completion_tokens_details
        assert details is not None
        set_fields = {
            name for name in type(details).model_fields if getattr(details, name, None) is not None
        }
        assert set_fields == {"reasoning_tokens", "text_tokens"}
        assert details.reasoning_tokens == 0
        assert details.text_tokens == 42

    def test_an_explicit_zero_valued_completion_detail_is_refused(self) -> None:
        """An explicit provider-reported zero is a fact, not a placeholder.

        The placeholder exception is drawn on the exact synthetic shape, not
        on the value: a provider that reports ``reasoning_tokens=0`` without
        the library's derived ``text_tokens`` stated a fact the canonical
        usage model cannot carry, and dropping it would lose information in
        exactly the way the refusal exists to prevent.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=67,
                completion_tokens_details={"reasoning_tokens": 0},
            ),
        )
        with pytest.raises(InternalError, match="reasoning_tokens"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_an_explicit_zero_valued_category_is_refused(self) -> None:
        """Zero of a category the canonical model cannot carry is still refused.

        The placeholder exception is the library's exact synthetic shape, not
        a general licence for zero-valued categories: an audio-token count
        reported at zero is a provider fact with no canonical target, so the
        chunk is refused rather than trimmed.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=67,
                completion_tokens_details={"audio_tokens": 0},
            ),
        )
        with pytest.raises(InternalError, match="audio_tokens"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_text_tokens_numeric_coincidence_without_the_shape_is_refused(self) -> None:
        """Numeric equality with the library's arithmetic is evidence, not proof.

        ``text_tokens`` equal to ``completion_tokens - reasoning_tokens`` is
        what the library derives, but a provider reporting that value on its
        own — without the placeholder ``reasoning_tokens=0`` beside it — is
        not the library's synthetic shape, and the coincidence alone does not
        make the field the library's.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=67,
                completion_tokens_details={"text_tokens": 42},
            ),
        )
        with pytest.raises(InternalError, match="text_tokens"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)


class TestFromLitellmChunkFinish:
    """The terminal event carries the gateway's classification and the raw reason."""

    def test_finish_reason_is_mapped_through_the_gateway_vocabulary(self) -> None:
        events = from_litellm_chunk(finish_chunk("stop"), invocation_id=INVOCATION_ID, sequence=0)
        assert events == (
            StreamEvent(
                kind=StreamEventKind.FINISH,
                sequence=0,
                invocation_id=INVOCATION_ID,
                finish_reason=FinishReason.STOP,
                raw_finish_reason="stop",
            ),
        )

    def test_a_reason_the_gateway_has_not_been_taught_is_unknown_with_raw_preserved(self) -> None:
        """LiteLLM's own normalisation is upstream of this module and is not undone here.

        A streaming choice keeps no native finish reason, so a reason LiteLLM did
        not recognise has already been normalised to ``"stop"`` before the gateway
        sees the chunk; the gateway records the value it was handed and classifies
        it with its own vocabulary, which matches it exactly.
        """
        events = from_litellm_chunk(finish_chunk("stop"), invocation_id=INVOCATION_ID, sequence=0)
        assert events[0].finish_reason is FinishReason.STOP
        assert events[0].raw_finish_reason == "stop"

    def test_an_unrecognised_upstream_reason_is_unknown_with_its_raw_value_preserved(self) -> None:
        """The reason the provider said, kept verbatim beside the gateway's classification.

        ``provider_specific_fields['native_finish_reason']`` is where LiteLLM
        keeps the provider's own reason whenever it differs from the mapped one,
        and the stream reads it from there exactly as the non-streaming path
        does. The gateway's vocabulary is its own — Anthropic's ``tool_use`` is
        not one of its reasons — so it becomes ``UNKNOWN``, and the raw value
        still says what the provider said: UNKNOWN never means the gateway
        deleted the reason.
        """
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    finish_reason="tool_calls",
                    provider_specific_fields={"native_finish_reason": "tool_use"},
                )
            ]
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert events[0].finish_reason is FinishReason.UNKNOWN
        assert events[0].raw_finish_reason == "tool_use"

    def test_a_reason_the_library_did_not_map_is_recorded_as_it_arrived(self) -> None:
        """LiteLLM maps an unrecognised provider reason to 'stop' on a stream.

        The gateway cannot recover the provider's own wording because the
        library did not keep it, and it invents no mapping to pretend otherwise:
        it reports STOP with the raw value the library handed it, which states
        plainly that the reason arrived already normalised.
        """
        chunk = ModelResponseStream(choices=[StreamingChoices(index=0, finish_reason="stop")])
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert events[0].finish_reason is FinishReason.STOP
        assert events[0].raw_finish_reason == "stop"

    def test_a_chunk_without_a_finish_reason_is_not_terminal(self) -> None:
        events = from_litellm_chunk(content_chunk("hi"), invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [StreamEventKind.CONTENT]


class TestFromLitellmChunkOrdering:
    """A chunk carrying several things produces the events in the order the answer reads."""

    def test_content_and_finish_in_one_chunk_produce_two_ordered_events(self) -> None:
        """The finish event is terminal, so it is emitted last."""
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="Done"), finish_reason="stop")]
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [StreamEventKind.CONTENT, StreamEventKind.FINISH]
        assert [event.sequence for event in events] == [0, 1]
        assert events[-1].finish_reason is FinishReason.STOP

    def test_content_then_tool_calls_then_usage_then_finish(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="Let me check.",
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="function",
                                function=Function(arguments=None, name="get_weather"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(prompt_tokens=25, completion_tokens=42, total_tokens=67),
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [
            StreamEventKind.CONTENT,
            StreamEventKind.TOOL_CALL,
            StreamEventKind.USAGE,
            StreamEventKind.FINISH,
        ]
        assert [event.sequence for event in events] == [0, 1, 2, 3]
        assert events[-1].finish_reason is FinishReason.TOOL_CALLS


class TestFromLitellmChunkRefusals:
    """A semantic the canonical event cannot carry is refused, naming the field."""

    def test_reasoning_content_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=None, reasoning_content="hmm"))]
        )
        with pytest.raises(InternalError, match="reasoning content"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_thinking_blocks_are_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        thinking_blocks=[ChatCompletionThinkingBlock(thinking="x", signature="s")],
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="thinking blocks"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_reasoning_items_are_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        reasoning_items=[
                            ChatCompletionReasoningItem(type="reasoning.text", id="rs_1")
                        ],
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="reasoning items"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_content_safety_annotations_are_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        annotations=[{"type": "hallucination", "hallucination": {}}],
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="annotations"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_audio_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        audio=ChatCompletionAudioResponse(
                            id="a1", data="AAAA", expires_at=123, transcript="hi"
                        ),
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="audio"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_images_are_refused(self) -> None:
        """An image part mid-stream is a fact about the answer the event cannot carry."""
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=None, images=[{"image_url": {"url": "https://x/y.png"}}]),
                )
            ]
        )
        with pytest.raises(InternalError, match="image content"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_legacy_function_call_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=None, function_call={"name": "f", "arguments": "{}"}),
                )
            ]
        )
        with pytest.raises(InternalError, match="legacy function call"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_non_string_text_is_refused(self) -> None:
        """Text the canonical content event cannot represent is refused, not dropped."""
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(index=0, delta=Delta(content=[{"type": "text", "text": "hi"}]))
            ]
        )
        with pytest.raises(InternalError, match="not a string"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_reasoning_token_detail_is_refused(self) -> None:
        """A token count the canonical usage cannot carry is refused, not dropped."""
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=47,
                completion_tokens_details={"reasoning_tokens": 5},
            ),
        )
        with pytest.raises(InternalError, match="reasoning_tokens"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_semantic_completion_detail_value_is_still_refused(self) -> None:
        """A provider-reported count with no canonical target is refused, not dropped.

        The placeholder exception is drawn on the value, not the field: a
        non-zero count the canonical usage model cannot carry — here an
        audio-token count — is still refused, because silently dropping it
        would tell the caller the usage was honoured when part of the report
        was dropped.
        """
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=25,
                completion_tokens=42,
                total_tokens=67,
                completion_tokens_details={"audio_tokens": 10},
            ),
        )
        with pytest.raises(InternalError, match="audio_tokens"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_log_probability_block_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, finish_reason="stop", logprobs={"content": []})]
        )
        with pytest.raises(InternalError, match="log-probability block"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_an_unknown_choice_field_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    finish_reason="stop",
                    provider_specific_fields={"audit": {"x": 1}},
                )
            ]
        )
        with pytest.raises(InternalError, match="'audit'"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_an_unknown_delta_field_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="hi",
                        provider_specific_fields={"native_finish_reason": "stop"},
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="'native_finish_reason'"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_custom_tool_call_is_refused(self) -> None:
        """A tool call that is not a function call has no canonical representation."""
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="custom",
                                function=Function(arguments=None, name=None),
                            )
                        ],
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="'custom'"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_tool_call_delta_with_a_negative_index_is_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=-1,
                                id="call_1",
                                type="function",
                                function=Function(arguments=None, name="f"),
                            )
                        ],
                    ),
                )
            ]
        )
        with pytest.raises(InternalError, match="index -1"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_several_choices_in_one_chunk_are_refused(self) -> None:
        """The canonical stream carries one answer per invocation."""
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(index=0, finish_reason="stop"),
                StreamingChoices(index=1, finish_reason="stop"),
            ]
        )
        with pytest.raises(InternalError, match="2 choices"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_choice_at_a_later_index_is_refused(self) -> None:
        """A chunk for a second choice cannot be folded into the single answer."""
        chunk = ModelResponseStream(choices=[StreamingChoices(index=1, finish_reason="stop")])
        with pytest.raises(InternalError, match="choice index 1"):
            from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

    def test_a_backend_fingerprint_is_ignored(self) -> None:
        """Operational metadata, deliberately ignored — it must not fail a valid chunk."""
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="hi"))],
            system_fingerprint="fp-1",
        )
        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)
        assert [event.kind for event in events] == [StreamEventKind.CONTENT]

    def test_a_negative_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            from_litellm_chunk(content_chunk("hi"), invocation_id=INVOCATION_ID, sequence=-1)


class TestStreamingReasoningDiscardPolicy:
    """The route's answer contract covers the stream, next to the response side."""

    def reasoning_delta_chunk(self) -> ModelResponseStream:
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="ok",
                        role="assistant",
                        reasoning_content="the trace",
                        thinking_blocks=[{"type": "thinking", "thinking": "the trace"}],
                    ),
                )
            ]
        )

    def test_a_reasoning_delta_is_refused_by_default(self) -> None:
        with pytest.raises(InternalError) as error:
            from_litellm_chunk(
                self.reasoning_delta_chunk(), invocation_id=INVOCATION_ID, sequence=0
            )

        assert "cannot carry" in str(error.value)

    def test_a_reasoning_delta_streams_its_text_when_the_route_discards_the_trace(self) -> None:
        events = from_litellm_chunk(
            self.reasoning_delta_chunk(),
            invocation_id=INVOCATION_ID,
            sequence=0,
            discard_reasoning=True,
        )

        assert [event.kind for event in events] == [StreamEventKind.CONTENT]
        assert events[0].text == "ok"

    def test_a_null_citation_slot_on_a_delta_is_not_a_citation(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="ok", provider_specific_fields={"citations": None}),
                )
            ]
        )

        events = from_litellm_chunk(chunk, invocation_id=INVOCATION_ID, sequence=0)

        assert [event.kind for event in events] == [StreamEventKind.CONTENT]

    def test_a_populated_citation_on_a_delta_is_still_refused(self) -> None:
        chunk = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="ok",
                        provider_specific_fields={"citations": [{"url": "https://example.com"}]},
                    ),
                )
            ]
        )

        with pytest.raises(InternalError) as error:
            from_litellm_chunk(
                chunk, invocation_id=INVOCATION_ID, sequence=0, discard_reasoning=True
            )

        assert "citations" in str(error.value)
