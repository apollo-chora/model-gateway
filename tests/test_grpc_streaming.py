"""The InvokeStream RPC over a real in-process gRPC listener.

These tests drive a real gRPC server and a real client rather than a stub of
either, because the behaviour under test is the transport's own: which chunks a
caller reads, in what order, with what sequence numbers, and what ends the
stream. Almost every assertion is on the chunks that arrive at a client and on
the request the provider recorded, which is what a caller can actually observe;
the three at the end drive the handler directly to pin its cancellation seam,
which a disconnect over a socket can only race.

The provider is a scripted stand-in, so a stream can be shaped exactly: the
events it yields, the failure it raises after them, and a gate that holds it
open so a client disconnect can be observed stopping it. One test binds the
:mod:`model_gateway.providers.mock` provider instead, to prove the request that
reaches it is the canonical one the tenant rule requires.

The listener serves the InvokeStream RPC alone. The other three methods of the
service are other modules' subjects, and the generated servicer's own answer to
one — ``NotImplementedError`` — is the right answer to a call no test here makes.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import grpc
import grpc.aio
import pytest

from model_gateway.accounting import AccountingRecorder, MemoryAccountingSink
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import FinishReason, Role
from model_gateway.domain.generation import Usage
from model_gateway.domain.streaming import StreamEvent, StreamEventKind, ToolCallDelta
from model_gateway.errors import UnavailableError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.streaming import InvokeStream
from model_gateway.inference import InferenceService
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.mock import MockProvider, MockStreamSpec
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

API_KEY = "sk-gateway-test-key"
TENANT = "tenant-a"
# A tenant no configured key is bound to, so a body claiming it disagrees with
# the tenant the credential below carries.
OTHER_TENANT = "tenant-b"
# A second credential bound to that other tenant, so two callers are two
# principals rather than one.
OTHER_KEY = "sk-gateway-other-key"
INVOCATION_ID = "inv-stream-1"
TEXT_MODEL = "text-default"
QUESTION = "What is the weather in Paris?"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
UPSTREAM_KEY = "sk-upstream-secret-key"


# --------------------------------------------------------------------------- #
# The listener
# --------------------------------------------------------------------------- #


class _Servicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """The InvokeStream RPC alone, wired to the handler this file is about.

    The method is an async generator itself rather than a coroutine returning the
    handler's generator: gRPC's server-streaming behaviour iterates what the
    servicer method returns, so the handler's own generator has to be yielded
    through, not returned from a coroutine.
    """

    def __init__(self, authenticator: Authenticator, service: InferenceService) -> None:
        self._handler = InvokeStream(authenticator, service)

    async def InvokeStream(self, request, context):
        async for chunk in self._handler.invoke_stream(request, context):
            yield chunk


def _settings(*, auth_enabled: bool, api_keys: tuple[str, ...]) -> Settings:
    """Settings built from the test's own variables, never the process environment."""
    return Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true" if auth_enabled else "false",
            "GATEWAY_API_KEYS": ",".join(api_keys),
        }
    )


@pytest.fixture
async def serve():
    """A factory that starts an in-process listener for one provider.

    Yields a coroutine which builds the listener around ``provider``, starts it
    and returns a client stub. Every listener started is stopped and every
    channel closed when the test ends, so a test leaves none running.
    """
    listeners: list[tuple[grpc.aio.Server, grpc.aio.Channel]] = []

    async def _serve(
        provider,
        *,
        api_keys: tuple[str, ...] = (f"{API_KEY}:{TENANT}",),
        auth_enabled: bool = True,
        accounting=None,
    ):
        registry = ModelRegistry((route(),))
        service = InferenceService(registry, provider, accounting=accounting)
        settings = _settings(auth_enabled=auth_enabled, api_keys=api_keys)
        authenticator = Authenticator(
            ApiKeySet.from_settings(settings),
            auth_enabled=auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
        server = grpc.aio.server()
        port = server.add_insecure_port("127.0.0.1:0")
        gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
            _Servicer(authenticator, service), server
        )
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        listeners.append((server, channel))
        return gateway_pb2_grpc.ModelGatewayServiceStub(channel)

    yield _serve

    for server, channel in listeners:
        await channel.close()
        await server.stop(grace=None)


def route(model_id: str = "upstream-text", **overrides) -> Route:
    """A standalone text route; only its model ID is ever resolved."""
    values = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": UPSTREAM_ENDPOINT,
        "base_url": UPSTREAM_ENDPOINT,
        "model_id": model_id,
        "format": "openai",
        "api_key": UPSTREAM_KEY,
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def authorization(key: str = API_KEY) -> tuple[tuple[str, str], ...]:
    """The credential a call presents, in the gRPC form the transport reads."""
    return (("authorization", f"Bearer {key}"),)


def invoke_request(**overrides) -> pb.InvokeRequest:
    """A standalone InvokeRequest naming the routed text model."""
    values = {
        "invocation_id": INVOCATION_ID,
        "tenant_id": TENANT,
        "model": TEXT_MODEL,
        "messages": [_message(QUESTION)],
        # The contract refuses an unspecified modality, so the request states one.
        "response_modality": pb.RESPONSE_MODALITY_TEXT,
    }
    values.update(overrides)
    return pb.InvokeRequest(**values)


def _message(text: str) -> pb.Message:
    message = pb.Message(role=pb.ROLE_USER)
    message.content.add().text.text = text
    return message


async def collect(
    stub, request: pb.InvokeRequest, *, metadata: tuple[tuple[str, str], ...] | None = None
) -> list[pb.InvokeChunk]:
    """Read one streaming call to its end, which is what a caller does."""
    return [chunk async for chunk in stub.InvokeStream(request, metadata=metadata)]


async def first_chunk(call) -> pb.InvokeChunk:
    """The first chunk of a streaming call, leaving the call open.

    A streaming call is an async iterator rather than a synchronous one, so the
    first chunk is read by entering the iteration and leaving it there — which is
    what a caller does when it still has a stream to watch.
    """
    async for chunk in call:
        return chunk
    raise AssertionError("the stream ended before producing its first chunk")


# --------------------------------------------------------------------------- #
# A scripted provider
# --------------------------------------------------------------------------- #


class ScriptedStreamProvider:
    """A streaming provider serving a script of events and, optionally, a failure.

    Each item of the script is served in order: a :class:`StreamEvent` is
    yielded, an :exc:`Exception` is raised at that point, and an
    :class:`asyncio.Event` is waited on — so a script can produce some output and
    then fail mid-stream, produce nothing at all, or hold the stream open after
    its first event so a client disconnect has something to stop.

    Every call is recorded as a ``(route, request, context)`` triple, and every
    stream is counted when it ends — normally, on a failure, or because the
    caller went away and closed it — which is how a test tells that a disconnect
    stopped the upstream work.
    """

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.stream_calls: list[tuple[Route, object, AttemptContext]] = []
        self.streams_closed = 0

    async def stream(
        self, route: Route, request, context: AttemptContext
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append((route, request, context))
        try:
            for item in self._script:
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, asyncio.Event):
                    await item.wait()
                    continue
                yield item
        finally:
            self.streams_closed += 1


def content_event(text: str, sequence: int) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.CONTENT,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        text=text,
    )


def tool_call_event(delta: ToolCallDelta, sequence: int) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.TOOL_CALL,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        tool_call=delta,
    )


def usage_event(usage: Usage, sequence: int) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.USAGE,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        usage=usage,
    )


def finish_event(
    reason: FinishReason = FinishReason.STOP, sequence: int = 0, raw: str | None = None
) -> StreamEvent:
    return StreamEvent(
        kind=StreamEventKind.FINISH,
        sequence=sequence,
        invocation_id=INVOCATION_ID,
        finish_reason=reason,
        raw_finish_reason=raw or reason.value,
    )


async def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait for a fact the server has not delivered yet, or fail instead of hang.

    A cancellation reaches the server over the wire and is observed a moment
    after the client asks for it, so the assertions that follow a disconnect
    wait for the upstream to record what happened rather than racing it.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for the upstream stream to stop")
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- #
# The answer, as a caller reads it
# --------------------------------------------------------------------------- #


class TestTheAnswerStream:
    """What a caller reads from a completed stream."""

    async def test_a_streamed_answer_arrives_as_fragments_numbered_from_zero(self, serve):
        provider = ScriptedStreamProvider(
            [
                content_event("Recursion", 0),
                content_event(" is a technique", 1),
                content_event(".", 2),
                finish_event(sequence=3),
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        sequences = [chunk.sequence for chunk in chunks]
        assert sequences == [0, 1, 2, 3]
        assert all(
            later > earlier for earlier, later in zip(sequences, sequences[1:], strict=False)
        )
        assert [chunk.content_delta for chunk in chunks[:3]] == [
            "Recursion",
            " is a technique",
            ".",
        ]

    async def test_every_chunk_carries_the_invocation_it_belongs_to(self, serve):
        provider = ScriptedStreamProvider([content_event("Recursion", 0), finish_event(sequence=1)])
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert [chunk.invocation_id for chunk in chunks] == [INVOCATION_ID, INVOCATION_ID]

    async def test_the_last_chunk_of_a_completed_stream_is_the_finish(self, serve):
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), finish_event(FinishReason.STOP, 1)]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        last = chunks[-1]
        assert last.WhichOneof("payload") == "finish"
        assert last.finish.finish_reason == pb.FINISH_REASON_STOP
        assert last.finish.HasField("raw_finish_reason")
        assert last.finish.raw_finish_reason == "stop"
        assert not last.HasField("usage")

    async def test_a_finish_reporting_no_raw_reason_arrives_absent(self, serve):
        # The provider reported no reason of its own, and the optional field
        # says so by being absent: an empty string would assert the provider
        # supplied a blank one, which is a different fact.
        provider = ScriptedStreamProvider(
            [
                StreamEvent(
                    kind=StreamEventKind.FINISH,
                    sequence=0,
                    invocation_id=INVOCATION_ID,
                    finish_reason=FinishReason.STOP,
                    raw_finish_reason=None,
                )
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert len(chunks) == 1
        assert chunks[0].finish.finish_reason == pb.FINISH_REASON_STOP
        assert not chunks[0].finish.HasField("raw_finish_reason")

    async def test_a_finish_reporting_an_empty_raw_reason_keeps_it(self, serve):
        # The provider did report a reason and it was the empty string: the
        # field is present and empty, which is not the same statement as an
        # absent one.
        provider = ScriptedStreamProvider(
            [
                StreamEvent(
                    kind=StreamEventKind.FINISH,
                    sequence=0,
                    invocation_id=INVOCATION_ID,
                    finish_reason=FinishReason.STOP,
                    raw_finish_reason="",
                )
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert chunks[0].finish.HasField("raw_finish_reason")
        assert chunks[0].finish.raw_finish_reason == ""

    async def test_a_tool_call_arrives_as_a_fragment_naming_its_position(self, serve):
        provider = ScriptedStreamProvider(
            [
                content_event("Let me check.", 0),
                tool_call_event(ToolCallDelta(index=0, id="call-1", name="get_weather"), 1),
                tool_call_event(ToolCallDelta(index=0, arguments_delta='{"city"'), 2),
                finish_event(FinishReason.TOOL_CALLS, 3),
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert [chunk.sequence for chunk in chunks] == [0, 1, 2, 3]
        first, second = chunks[1].tool_call_delta, chunks[2].tool_call_delta
        assert (first.index, first.id, first.name) == (0, "call-1", "get_weather")
        assert first.arguments_delta == ""
        assert (second.index, second.id, second.name) == (0, "", "")
        assert second.arguments_delta == '{"city"'
        assert chunks[3].finish.finish_reason == pb.FINISH_REASON_TOOL_CALLS

    async def test_a_usage_report_arrives_with_only_the_counts_that_are_known(self, serve):
        provider = ScriptedStreamProvider(
            [
                usage_event(
                    Usage(
                        input_tokens=25,
                        output_tokens=42,
                        total_tokens=67,
                        estimated_fields=frozenset({"total_tokens"}),
                    ),
                    0,
                ),
                finish_event(sequence=1),
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        usage = chunks[0].usage
        assert usage.input_tokens == 25
        assert usage.output_tokens == 42
        assert usage.total_tokens == 67
        # The total is the gateway's own arithmetic, and the answer names it.
        assert usage.estimated_fields == [pb.USAGE_FIELD_TOTAL_TOKENS]
        # A count the provider never reported arrives absent, not as zero.
        assert not usage.HasField("cached_tokens")

    async def test_a_stream_carrying_no_output_still_ends_with_exactly_one_finish(self, serve):
        provider = ScriptedStreamProvider([finish_event(FinishReason.STOP, 0)])
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert len(chunks) == 1
        assert chunks[0].WhichOneof("payload") == "finish"
        assert chunks[0].finish.finish_reason == pb.FINISH_REASON_STOP

    async def test_the_request_reaches_the_provider_under_the_credentials_tenant(self, serve):
        """The round trip: the canonical request arrives, the answer comes back.

        The claim is left in the request to prove it is honoured when it agrees
        with the credential; the mismatch is its own refusal below.
        """
        provider = MockProvider(stream=MockStreamSpec(chunks=("Recursion", " is a technique")))
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert provider.stream_calls, "the provider was never called"
        _, request, _ = provider.stream_calls[0]
        assert request.invocation_id == INVOCATION_ID
        assert request.tenant_id == TENANT
        assert request.model == TEXT_MODEL
        message = request.messages[0]
        assert message.role is Role.USER
        assert [part.text for part in message.content] == [QUESTION]
        assert [chunk.content_delta for chunk in chunks] == ["Recursion", " is a technique", ""]
        assert chunks[-1].WhichOneof("payload") == "finish"


# --------------------------------------------------------------------------- #
# The end of the stream
# --------------------------------------------------------------------------- #


class TestTheEndOfTheStream:
    """What ends a stream, and what a caller is shown when it does."""

    async def test_a_provider_failure_mid_stream_ends_with_one_error_chunk(self, serve):
        """One failure, reported once: the error chunk, then a clean end.

        The caller has already seen output, so the stream does not silently
        stop; the terminal chunk carries the gateway's own reason and the
        sanitised message, and the call itself completes rather than failing.
        """
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), UnavailableError("the connection dropped")]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert [chunk.sequence for chunk in chunks] == [0, 1]
        assert chunks[0].content_delta == "Recursion"
        error = chunks[1].error
        assert error.reason == "provider_unavailable"
        assert error.message == "the connection dropped"
        # A failure is not a finish: it carries no reason generation stopped on.
        assert chunks[1].finish.finish_reason == pb.FINISH_REASON_UNSPECIFIED
        assert chunks[1].WhichOneof("payload") == "error"

    async def test_nothing_follows_the_terminal_error_chunk(self, serve):
        """The events the provider produced after its failure never reach a caller."""
        provider = ScriptedStreamProvider(
            [
                content_event("Recursion", 0),
                UnavailableError("the connection dropped"),
                content_event(" and more", 2),
                finish_event(sequence=3),
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert [chunk.sequence for chunk in chunks] == [0, 1]
        assert chunks[-1].WhichOneof("payload") == "error"

    async def test_the_error_chunk_takes_the_next_sequence_number(self, serve):
        provider = ScriptedStreamProvider(
            [
                content_event("Recursion", 0),
                content_event(" is a technique", 1),
                UnavailableError("the connection dropped"),
            ]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert [chunk.sequence for chunk in chunks] == [0, 1, 2]
        assert chunks[-1].error.reason == "provider_unavailable"

    async def test_an_unexpected_failure_is_not_published_as_a_chunk(self, serve):
        """A failure outside the taxonomy ends the call as an internal error.

        It has no gateway reason to report, so it is not turned into an event:
        the caller sees the chunks that were already delivered and then the
        failure itself, and the provider's own text reaches neither.
        """
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), RuntimeError("upstream exploded")]
        )
        stub = await serve(provider)

        call = stub.InvokeStream(invoke_request(), metadata=authorization())
        received = await first_chunk(call)
        assert received.content_delta == "Recursion"
        with pytest.raises(grpc.aio.AioRpcError) as error:
            async for _ in call:
                pass
        assert error.value.code() == grpc.StatusCode.INTERNAL

    async def test_a_failure_before_any_output_is_an_error_response_not_a_stream(self, serve):
        """A refusal the caller has not seen any output through is an error status.

        The provider fails before its first event, so the invocation has produced
        nothing a caller could be shown: the call ends with the error's own status
        and no chunk at all, and the reason a caller branches on is the taxonomy's
        in the aborted payload.
        """
        provider = MockProvider(
            stream=MockStreamSpec(error=UnavailableError("the upstream is down"))
        )
        stub = await serve(provider)

        with pytest.raises(grpc.aio.AioRpcError) as error:
            await collect(stub, invoke_request(), metadata=authorization())

        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        assert json.loads(error.value.details())["reason"] == "provider_unavailable"
        # The request was served and refused; it was not refused by routing.
        assert provider.stream_calls

    async def test_the_caller_keeps_the_output_the_stream_produced_before_it_failed(self, serve):
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), UnavailableError("the connection dropped")]
        )
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert chunks[0].content_delta == "Recursion"
        assert chunks[0].sequence == 0


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class TestARefusal:
    """A refusal is an error status and no stream at all."""

    async def test_a_claimed_tenant_that_disagrees_is_refused(self, serve):
        provider = ScriptedStreamProvider([content_event("Recursion", 0)])
        stub = await serve(provider)

        with pytest.raises(grpc.aio.AioRpcError) as error:
            await collect(stub, invoke_request(tenant_id=OTHER_TENANT), metadata=authorization())

        assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert provider.stream_calls == []

    async def test_a_claimed_tenant_that_agrees_is_honoured(self, serve):
        """The control case: a claim matching the credential's tenant is not a disagreement."""
        provider = MockProvider(stream=MockStreamSpec(chunks=("Recursion",)))
        stub = await serve(provider)

        chunks = await collect(stub, invoke_request(), metadata=authorization())

        assert provider.stream_calls[0][1].tenant_id == TENANT
        assert chunks[0].content_delta == "Recursion"

    async def test_a_call_with_no_credential_is_refused_unauthenticated(self, serve):
        provider = ScriptedStreamProvider([content_event("Recursion", 0)])
        stub = await serve(provider)

        with pytest.raises(grpc.aio.AioRpcError) as error:
            await collect(stub, invoke_request())

        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert provider.stream_calls == []

    async def test_a_call_with_a_credential_nobody_configured_is_refused(self, serve):
        provider = ScriptedStreamProvider([content_event("Recursion", 0)])
        stub = await serve(provider)

        with pytest.raises(grpc.aio.AioRpcError) as error:
            await collect(stub, invoke_request(), metadata=authorization("sk-gateway-wrong-key"))

        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert provider.stream_calls == []

    async def test_an_unknown_model_is_ignored_and_reaches_the_provider(self, serve):
        provider = ScriptedStreamProvider([content_event("Recursion", 0)])
        stub = await serve(provider)

        chunks = await collect(
            stub, invoke_request(model="no-such-model"), metadata=authorization()
        )

        assert chunks[0].content_delta == "Recursion"
        assert len(provider.stream_calls) == 1
        assert provider.stream_calls[0][1].model == "no-such-model"

    async def test_a_body_tenant_never_reaches_the_provider(self, serve):
        """A request field cannot create a tenant: an anonymous call has none.

        Authentication disabled, so the call proceeds anonymously and the claimed
        tenant is discarded rather than propagated.
        """
        provider = MockProvider(stream=MockStreamSpec(chunks=("Recursion",)))
        stub = await serve(provider, auth_enabled=False)

        chunks = await collect(stub, invoke_request(tenant_id=OTHER_TENANT))

        assert provider.stream_calls[0][1].tenant_id is None
        assert chunks[0].content_delta == "Recursion"


# --------------------------------------------------------------------------- #
# A caller that goes away
# --------------------------------------------------------------------------- #


class TestADisconnect:
    """A cancelled call stops the upstream work and ends without a terminal event."""

    async def test_a_caller_that_disconnects_stops_the_upstream_stream(self, serve):
        gate = asyncio.Event()
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), gate, finish_event(sequence=2)]
        )
        stub = await serve(provider)

        call = stub.InvokeStream(invoke_request(), metadata=authorization())
        first = await first_chunk(call)
        assert first.sequence == 0
        call.cancel()
        await wait_for(lambda: provider.streams_closed == 1)

    async def test_a_disconnected_call_ends_without_a_terminal_event(self, serve):
        """The stream simply ends, so a caller reads a truncated one.

        It stopped without a terminal event and without an error, which is not a
        completed stream and not a failed one: the finish the provider had
        scripted for after the gate never reaches a caller that went away.
        """
        gate = asyncio.Event()
        provider = ScriptedStreamProvider(
            [content_event("Recursion", 0), gate, finish_event(sequence=2)]
        )
        stub = await serve(provider)

        call = stub.InvokeStream(invoke_request(), metadata=authorization())
        received = [await first_chunk(call)]
        call.cancel()
        await wait_for(lambda: provider.streams_closed == 1)

        assert [chunk.sequence for chunk in received] == [0]
        assert all(chunk.WhichOneof("payload") not in ("finish", "error") for chunk in received)


class _CallerContext:
    """A ``ServicerContext`` double whose caller is there for a stated number of
    cancellation readings and gone for the rest.

    The handler reads cancellation once per event it is about to hand over, so
    ``stays_for=0`` refuses the first chunk and ``stays_for=1`` the second; a
    caller that never leaves is ``stays_for=None``. That pins the cancellation
    seam itself, which a disconnect over the socket can only race.
    """

    def __init__(self, *, stays_for: int | None) -> None:
        self._stays_for = stays_for
        self._readings = 0

    def invocation_metadata(self):
        return authorization()

    def cancelled(self) -> bool:
        self._readings += 1
        return self._stays_for is not None and self._readings > self._stays_for


def _handler(provider) -> InvokeStream:
    """The handler wired to a real inference service over one scripted provider."""
    settings = _settings(auth_enabled=True, api_keys=(f"{API_KEY}:{TENANT}",))
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=True,
        allow_anonymous_health=settings.allow_anonymous_health,
    )
    return InvokeStream(authenticator, InferenceService(ModelRegistry((route(),)), provider))


class TestTheCancellationSeam:
    """The handler's own answer to a caller that is no longer there.

    The disconnects above prove it over a socket — that the upstream attempt
    stops and that no terminal event reaches the caller. What they cannot pin is
    which of the two answers to a cancellation the handler takes, because the
    one a real disconnect takes is a race; these drive the seam directly and
    deterministically.
    """

    async def test_a_call_that_is_already_gone_is_answered_with_nothing(self):
        provider = ScriptedStreamProvider([content_event("Recursion", 0), finish_event(sequence=1)])

        chunks = [
            chunk
            async for chunk in _handler(provider).invoke_stream(
                invoke_request(), _CallerContext(stays_for=0)
            )
        ]

        assert chunks == []
        # Nothing was handed over, and the stream behind it was closed rather
        # than left running for a caller that is not there.
        assert provider.streams_closed == 1

    async def test_a_call_that_goes_away_stops_the_iteration_at_that_point(self):
        provider = ScriptedStreamProvider(
            [
                content_event("Recursion", 0),
                content_event(" and more", 1),
                finish_event(sequence=2),
            ]
        )

        chunks = [
            chunk
            async for chunk in _handler(provider).invoke_stream(
                invoke_request(), _CallerContext(stays_for=1)
            )
        ]

        # The chunk the caller was still there for arrived; the rest of the
        # answer never did, and no terminal event was synthesised for it.
        assert [chunk.sequence for chunk in chunks] == [0]
        assert chunks[0].content_delta == "Recursion"
        assert provider.streams_closed == 1

    async def test_a_call_that_is_still_live_streams_to_its_finish(self):
        """The control case: an ordinary caller is not a disconnected one."""
        provider = ScriptedStreamProvider([content_event("Recursion", 0), finish_event(sequence=1)])

        chunks = [
            chunk
            async for chunk in _handler(provider).invoke_stream(
                invoke_request(), _CallerContext(stays_for=None)
            )
        ]

        assert [chunk.sequence for chunk in chunks] == [0, 1]
        assert chunks[-1].WhichOneof("payload") == "finish"


# --------------------------------------------------------------------------- #
# Idempotency on the streaming path
# --------------------------------------------------------------------------- #

IDEMPOTENCY_KEY = "trace-7"


class TestIdempotency:
    """A keyed request is still served as a stream, never as a replay.

    Idempotency is a property of the non-streaming invocation: a replayed stream
    would present an answer that was never produced token by token in this call as
    one that was, which is the one thing the gateway never does. The key reaches
    the invocation — it is a request field like any other — and the provider is
    asked again on every call.
    """

    async def test_an_idempotency_key_reaches_the_invocation(self, serve):
        provider = MockProvider(stream=MockStreamSpec(chunks=("Recursion",)))
        stub = await serve(provider)

        chunks = await collect(
            stub, invoke_request(idempotency_key=IDEMPOTENCY_KEY), metadata=authorization()
        )

        _, request, _ = provider.stream_calls[0]
        assert request.idempotency_key == IDEMPOTENCY_KEY
        assert chunks[0].content_delta == "Recursion"

    async def test_a_repeat_under_the_same_key_reaches_the_provider_again(self, serve):
        # The second call is not answered from a record: the stream is produced
        # again, so the provider is asked again and the caller watches the answer
        # arrive as fragments rather than as a replay of one already given.
        provider = MockProvider(stream=MockStreamSpec(chunks=("Recursion",)))
        stub = await serve(provider)

        await collect(
            stub, invoke_request(idempotency_key=IDEMPOTENCY_KEY), metadata=authorization()
        )
        chunks = await collect(
            stub, invoke_request(idempotency_key=IDEMPOTENCY_KEY), metadata=authorization()
        )

        assert len(provider.stream_calls) == 2
        assert [chunk.content_delta for chunk in chunks] == ["Recursion", ""]


# --------------------------------------------------------------------------- #
# The authenticated principal reaches a record
# --------------------------------------------------------------------------- #


async def test_two_credentials_record_under_two_namespaces(serve):
    """InvokeStream forwards the principal it authenticated.

    The HTTP handlers have their own tests; this one covers the gRPC handler,
    which authenticates through its own path and tears its stream down through its
    own ``finally``, so a regression in forwarding the identity would not show up
    on the HTTP side.
    """
    sink = MemoryAccountingSink()
    provider = ScriptedStreamProvider([finish_event(sequence=0)])
    stub = await serve(
        provider,
        api_keys=(f"{API_KEY}:{TENANT}", f"{OTHER_KEY}:{OTHER_TENANT}"),
        accounting=AccountingRecorder(sink, None),
    )

    for key, tenant in ((API_KEY, TENANT), (OTHER_KEY, OTHER_TENANT)):
        assert await collect(stub, invoke_request(tenant_id=tenant), metadata=authorization(key))

    assert len(sink.records) == 2
    namespaces = [record.namespace for record in sink.records]
    assert all(namespaces)
    assert namespaces[0] != namespaces[1]
