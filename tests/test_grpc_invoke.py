"""The Invoke RPC, driven the way a caller drives it (PRD sections 5, 5.1, 5.2 and 5.4).

Every test here runs against a real ``grpc.aio`` server on an ephemeral loopback
port, reached through the generated :class:`ModelGatewayServiceStub`, so what is
asserted is what crosses the wire: the canonical invocation the provider was
handed, the protobuf answer the caller reads back, and the gRPC status a refusal
carries. Only the upstream is a stand-in — the provider is the one thing this
layer does not own — and it records every call, which is how "the provider was
never called" is proved rather than assumed.

The server is assembled here rather than through
:class:`~model_gateway.grpc.server.GrpcServer` because that class registers every
RPC of the service while this file tests one: a server carrying only ``Invoke``
keeps each failure attributable to the handler under test. It binds
``127.0.0.1:0``, so the port is the operating system's choice and no test can
collide with another or with a running deployment.

**An answer carrying a tool call, an image or a refusal is translated, not
refused.** The canonical answer is an ordered sequence of output items and the
contract carries the same shapes, so the inference service's text path — which
stamps the gateway's own facts onto the provider's answer and changes nothing
else — hands all three to the caller. Nothing on this path raises a capability
error for what an *answer* carries: the capability rules are request-side, and
they refuse what the caller stated before any provider is called (PRD section
9.5). The three tests named for those answers pin the behaviour that actually
happens, which is the opposite of refusing a provider's own successful answer.

The refusal tests assert the status a caller reads, which is what catches an
abort that never took effect: ``grpc.aio``'s ``ServicerContext.abort`` is a
coroutine, so an abort that is not awaited leaves the call to fail as
``UNKNOWN`` instead of the category's own status.
"""

from contextlib import asynccontextmanager

import grpc
import orjson
import pytest

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    IdempotencyStatus,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    RefusalPart,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import UnavailableError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.invoke import Invoke
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

API_KEY = "sk-gateway-test-key"
TENANT = "tenant-a"
# A tenant no configured key is bound to, so a claim of it disagrees with every
# credential this file's key set can produce.
UNBOUND_TENANT = "tenant-no-key-is-bound-to"
SECOND_KEY = "sk-gateway-second-credential"
SECOND_TENANT = "tenant-b"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
UPSTREAM_KEY = "sk-upstream-secret-key"
PUBLIC_MODEL = "text-default"
PRIMARY_MODEL = "upstream-text"
FALLBACK_MODEL = "fallback-text"

_HOST = "127.0.0.1"


def text_route(
    index: int,
    model_id: str,
    *,
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT, Capability.TOOLS}),
) -> Route:
    """One configured text route, with the capabilities a chat invocation needs."""
    return Route(
        operation=Operation.TEXT,
        index=index,
        endpoint=UPSTREAM_ENDPOINT,
        base_url=UPSTREAM_ENDPOINT,
        model_id=model_id,
        format="openai",
        api_key=UPSTREAM_KEY,
        capabilities=capabilities,
        extra_headers={},
        allow_absolute_path_url=False,
    )


def structured_output_registry() -> ModelRegistry:
    """A registry whose route can honour a structured-output requirement.

    A schema in the generation config is refused by the gateway core on a route
    that does not declare the capability, so a test about the schema itself needs
    a route that declares it.
    """
    return ModelRegistry(
        (
            text_route(
                0,
                PRIMARY_MODEL,
                capabilities=frozenset(
                    {Capability.CHAT, Capability.TOOLS, Capability.STRUCTURED_OUTPUT}
                ),
            ),
        )
    )


def make_registry(*, fallback: bool = False) -> ModelRegistry:
    """A registry serving ``text-default``, optionally with a second route behind it."""
    routes = [text_route(0, PRIMARY_MODEL)]
    if fallback:
        routes.append(text_route(1, FALLBACK_MODEL))
    return ModelRegistry(tuple(routes))


def user_message(text: str) -> pb.Message:
    return pb.Message(role=pb.ROLE_USER, content=(pb.ContentPart(text=pb.TextPart(text=text)),))


def request(**overrides) -> pb.InvokeRequest:
    """An ``InvokeRequest`` a caller could send: one user turn, the modality stated.

    ``response_modality`` is stated because the contract carries no default for
    it: an unstated modality is the ``*_UNSPECIFIED`` zero value, which the
    translation refuses rather than reading as text (see
    :func:`test_an_unstated_enum_is_refused`).
    """
    fields = {
        "model": PUBLIC_MODEL,
        "messages": (user_message("Explain recursion."),),
        "response_modality": pb.RESPONSE_MODALITY_TEXT,
    }
    fields.update(overrides)
    return pb.InvokeRequest(**fields)


def bearer(key: str | None = API_KEY) -> tuple[tuple[str, str], ...]:
    """The call metadata presenting ``key``, or nothing when ``key`` is ``None``."""
    return () if key is None else (("authorization", f"Bearer {key}"),)


class _InvokeOnlyServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """Dispatch ``Invoke`` to the handler under test, and nothing else."""

    def __init__(self, invoke: Invoke) -> None:
        self._invoke = invoke

    async def Invoke(self, request, context):
        return await self._invoke.invoke(request, context)


class ScriptedProvider:
    """A ``TextProvider`` double answering with the canonical response it was given.

    :class:`~model_gateway.providers.mock.MockProvider` builds its answer from
    text and tool calls, so an answer carrying an image part or a refusal is not
    a shape its spec can express; the tests about those answers state the
    canonical response directly. What the double records is the same triple the
    mock records, so "the provider was never called" reads the same either way.
    """

    def __init__(self, response: InvocationResponse) -> None:
        self._response = response
        self.calls: list[tuple[Route, InvocationRequest, object]] = []

    async def complete(self, route: Route, request, context) -> InvocationResponse:
        self.calls.append((route, request, context))
        return self._response


@asynccontextmanager
async def serve(
    provider,
    *,
    registry: ModelRegistry | None = None,
    auth_enabled: bool = True,
    two_credentials: bool = False,
):
    """A real in-process gRPC server serving one ``Invoke`` handler, and a stub for it.

    The server, the authenticator and the inference service are built from the
    same objects the listener builds them from, so a test exercises the real
    authentication policy and the real reliability policy rather than a stand-in
    for either. The port is reserved before the server starts, so the address the
    stub dials is the one this server bound and never a guess at a free port.
    """
    keys = f"{API_KEY}:{TENANT}"
    if two_credentials:
        keys = f"{keys},SECOND_KEY:{SECOND_TENANT}"
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true" if auth_enabled else "false",
            "GATEWAY_API_KEYS": keys,
        }
    )
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )
    service = InferenceService(registry if registry is not None else make_registry(), provider)
    server = grpc.aio.server()
    port = server.add_insecure_port(f"{_HOST}:0")
    gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
        _InvokeOnlyServicer(Invoke(authenticator, service)), server
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"{_HOST}:{port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.fixture
def provider() -> MockProvider:
    """The deterministic provider the fixture's server is wired to."""
    return MockProvider()


@pytest.fixture
async def stub(provider):
    """A running in-process server over ``provider``, stopped when the test ends."""
    async with serve(provider) as running:
        yield running


async def test_a_successful_invocation_round_trips(stub, provider):
    response = await stub.Invoke(request(invocation_id="inv-1"), metadata=bearer())

    assert response.invocation_id == "inv-1"
    assert response.model == PUBLIC_MODEL
    assert response.upstream_model == PRIMARY_MODEL
    assert [item.content.text.text for item in response.output] == ["Explain recursion."]
    assert response.finish_reason == pb.FINISH_REASON_STOP
    assert len(provider.calls) == 1


async def test_the_provider_is_handed_the_canonical_request(stub, provider):
    await stub.Invoke(
        request(
            invocation_id="inv-2",
            actor_id="actor-1",
            agent_id="agent-1",
            system_instruction="Answer briefly.",
        ),
        metadata=bearer(),
    )

    route, invocation, _ = provider.calls[0]
    assert route.model_id == PRIMARY_MODEL
    assert invocation.model == PUBLIC_MODEL
    assert invocation.invocation_id == "inv-2"
    assert invocation.actor_id == "actor-1"
    assert invocation.agent_id == "agent-1"
    assert invocation.system_instruction == "Answer briefly."
    assert [
        (
            message.role.value,
            "".join(part.text for part in message.content if isinstance(part, TextPart)),
        )
        for message in invocation.messages
    ] == [("user", "Explain recursion.")]


async def test_metadata_carries_json_values_beside_strings(stub, provider):
    # The canonical record is Mapping[str, JsonValue]: an integer and a boolean
    # beside a string are the same kind of fact, and the JSON text the contract
    # carries holds every one of them exactly, where the struct it replaced typed
    # every number as a double.
    invocation_request = request()
    invocation_request.metadata_json = b'{"trace": "abc", "attempt": 2, "cached": true}'

    await stub.Invoke(invocation_request, metadata=bearer())

    _, invocation, _ = provider.calls[0]
    assert invocation.metadata == {"trace": "abc", "attempt": 2, "cached": True}
    assert type(invocation.metadata["attempt"]) is int
    assert type(invocation.metadata["cached"]) is bool


async def test_the_invocation_carries_the_credentials_tenant(stub, provider):
    # The caller claims no tenant at all, so the tenant the provider sees can
    # only have come from the credential.
    await stub.Invoke(request(), metadata=bearer())

    _, invocation, _ = provider.calls[0]
    assert invocation.tenant_id == TENANT


async def test_a_claimed_tenant_matching_the_credential_is_accepted(stub, provider):
    response = await stub.Invoke(request(tenant_id=TENANT), metadata=bearer())

    _, invocation, _ = provider.calls[0]
    assert invocation.tenant_id == TENANT
    assert response.model == PUBLIC_MODEL


async def test_an_unauthenticated_call_is_refused_before_the_provider(stub, provider):
    with pytest.raises(grpc.aio.AioRpcError) as raised:
        await stub.Invoke(request(), metadata=bearer(None))

    assert raised.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert provider.calls == []


async def test_a_rejected_credential_is_refused(stub, provider):
    with pytest.raises(grpc.aio.AioRpcError) as raised:
        await stub.Invoke(request(), metadata=bearer("sk-not-a-configured-key"))

    assert raised.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert provider.calls == []


async def test_a_claimed_tenant_that_disagrees_is_refused(stub, provider):
    with pytest.raises(grpc.aio.AioRpcError) as raised:
        await stub.Invoke(request(tenant_id=UNBOUND_TENANT), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.PERMISSION_DENIED
    assert provider.calls == []


async def test_an_unknown_model_is_refused_and_the_provider_is_never_called(stub, provider):
    with pytest.raises(grpc.aio.AioRpcError) as raised:
        await stub.Invoke(request(model="no-such-model"), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.NOT_FOUND
    assert provider.calls == []


async def test_a_provider_failure_keeps_its_own_status():
    provider = MockProvider(MockSpec(error=UnavailableError("the upstream is unreachable")))

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(request(), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.UNAVAILABLE
    assert "provider_unavailable" in raised.value.details()
    assert len(provider.calls) == 1


async def test_an_unstated_enum_is_refused():
    # No response_modality: the wire carries the *_UNSPECIFIED zero value, which
    # is not a modality the contract defines.
    provider = MockProvider()

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(
                pb.InvokeRequest(model=PUBLIC_MODEL, messages=(user_message("hi"),)),
                metadata=bearer(),
            )

    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert provider.calls == []


async def test_a_content_part_carrying_no_kind_is_refused():
    provider = MockProvider()
    empty = pb.Message(role=pb.ROLE_USER, content=(pb.ContentPart(),))

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(request(messages=(empty,)), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert provider.calls == []


async def test_an_answer_carrying_a_tool_call_is_translated():
    provider = MockProvider(
        MockSpec(
            text="Let me look that up.",
            tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"query": "recursion"}),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert [item.WhichOneof("item") for item in response.output] == ["content", "tool_call"]
    assert response.output[0].content.text.text == "Let me look that up."
    call = response.output[1].tool_call
    assert call.id == "call_1"
    assert call.name == "lookup"
    assert orjson.loads(call.arguments_json) == {"query": "recursion"}
    assert response.finish_reason == pb.FINISH_REASON_TOOL_CALLS


async def test_a_tool_calls_integer_arguments_survive_as_integers():
    # The arguments cross as JSON text, not as a protobuf Value: a JSON integer
    # is a double in the well-known type, and one beyond 2^53 is not a double
    # at all, so the bytes are what keep an integer an integer.
    provider = MockProvider(
        MockSpec(
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="lookup",
                    arguments={"a": 1, "big": 9007199254740993},
                ),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    call = response.output[1].tool_call
    arguments = orjson.loads(call.arguments_json)
    assert arguments == {"a": 1, "big": 9007199254740993}
    assert type(arguments["a"]) is int
    assert type(arguments["big"]) is int


async def test_a_tool_call_reporting_no_arguments_carries_no_bytes_for_them():
    # None — the provider reported no arguments at all — is not an empty
    # object: it leaves the field unset, which reads back as no bytes.
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(ToolCall(id="call_1", name="lookup"),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.output[0].tool_call.arguments_json == b""


async def test_a_tool_calls_raw_string_arguments_stay_a_string():
    # A provider that did not parse its arguments reported text, and the text
    # stays text: the canonical record keeps a JSON string a string.
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(ToolCall(id="call_1", name="lookup", arguments="not json {"),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert orjson.loads(response.output[0].tool_call.arguments_json) == "not json {"


async def test_a_tool_call_in_the_conversation_reaches_the_provider_parsed():
    # The same fidelity on the request side: the bytes are decoded before the
    # invocation is built, so the provider is handed the JSON object the caller
    # wrote, with its integers intact.
    provider = MockProvider()
    call = pb.ToolCall(
        id="call_1", name="lookup", arguments_json=b'{"a": 1, "big": 9007199254740993}'
    )

    async with serve(provider) as stub:
        await stub.Invoke(
            request(messages=(pb.Message(role=pb.ROLE_ASSISTANT, tool_calls=(call,)),)),
            metadata=bearer(),
        )

    _, invocation, _ = provider.calls[0]
    arguments = invocation.messages[0].tool_calls[0].arguments
    assert arguments == {"a": 1, "big": 9007199254740993}
    assert type(arguments["a"]) is int


async def test_a_tool_call_carrying_unparsable_json_is_refused():
    # Bytes that are not JSON are a malformed request, not a tool call the
    # gateway may guess at: they are refused as an invalid argument before
    # any provider is called.
    provider = MockProvider()
    call = pb.ToolCall(id="call_1", name="lookup", arguments_json=b"not json")

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(
                request(messages=(pb.Message(role=pb.ROLE_ASSISTANT, tool_calls=(call,)),)),
                metadata=bearer(),
            )

    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert provider.calls == []


async def test_a_tool_choice_reaches_the_provider(stub, provider):
    await stub.Invoke(
        request(
            tools=(pb.ToolDefinition(name="lookup", description="Look a term up."),),
            tool_choice=pb.ToolChoice(mode=pb.TOOL_CHOICE_MODE_SPECIFIC, name="lookup"),
        ),
        metadata=bearer(),
    )

    _, invocation, _ = provider.calls[0]
    assert [declared.name for declared in invocation.tools] == ["lookup"]
    assert invocation.tool_choice is not None
    assert invocation.tool_choice.mode.value == "specific"
    assert invocation.tool_choice.name == "lookup"


async def test_a_boolean_json_schema_reaches_the_provider_as_a_boolean():
    # ``true`` is a complete JSON Schema — it admits every value — and the
    # canonical record carries it as the boolean it is, not as an object with a
    # claim inside it.
    provider = MockProvider()

    async with serve(provider, registry=structured_output_registry()) as stub:
        await stub.Invoke(
            request(
                generation_config=pb.GenerationConfig(
                    response_format=pb.ResponseFormat(
                        mode=pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA, schema_json=b"true"
                    )
                )
            ),
            metadata=bearer(),
        )

    _, invocation, _ = provider.calls[0]
    response_format = invocation.generation_config.response_format
    assert response_format.mode.value == "json_schema"
    assert response_format.schema is True


async def test_a_json_schema_object_reaches_the_provider_as_it_was_given():
    provider = MockProvider()
    schema = b'{"type": "object", "properties": {"name": {"type": "string"}}}'

    async with serve(provider, registry=structured_output_registry()) as stub:
        await stub.Invoke(
            request(
                generation_config=pb.GenerationConfig(
                    response_format=pb.ResponseFormat(
                        mode=pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA, schema_json=schema
                    )
                )
            ),
            metadata=bearer(),
        )

    _, invocation, _ = provider.calls[0]
    assert invocation.generation_config.response_format.schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(b'"str"', id="string"),
        pytest.param(b"null", id="null"),
        pytest.param(b"[1,2]", id="array"),
        pytest.param(b"42", id="number"),
    ],
)
async def test_a_schema_that_is_neither_object_nor_boolean_is_refused(schema):
    # A schema is an object or one of the two boolean schemas, and anything
    # else — a string naming a shape, a number, an array, null — is refused
    # rather than coerced, with the field that carried it named.
    provider = MockProvider()

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(
                request(
                    generation_config=pb.GenerationConfig(
                        response_format=pb.ResponseFormat(
                            mode=pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA, schema_json=schema
                        )
                    )
                ),
                metadata=bearer(),
            )

    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert "schema_json" in raised.value.details()
    assert provider.calls == []


async def test_an_answer_carrying_an_image_is_translated():
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(ImagePart(data="aW1hZ2UtYnl0ZXM=", media_type="image/png"),),
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    image = response.output[0].content.image
    assert image.WhichOneof("source") == "data"
    assert image.data == "aW1hZ2UtYnl0ZXM="
    assert image.media_type == "image/png"


async def test_an_answer_carrying_a_refusal_is_translated():
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(RefusalPart(text="I cannot help with that."),),
            finish_reason=FinishReason.CONTENT_FILTER,
            raw_finish_reason="content_filter",
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.output[0].content.refusal.text == "I cannot help with that."
    assert response.finish_reason == pb.FINISH_REASON_CONTENT_FILTER


async def test_an_unreported_usage_count_is_not_rendered_as_zero():
    provider = MockProvider(MockSpec(usage=Usage(input_tokens=25)))

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.usage.HasField("input_tokens")
    assert response.usage.input_tokens == 25
    assert not response.usage.HasField("output_tokens")
    assert not response.usage.HasField("total_tokens")
    assert not response.usage.HasField("cached_tokens")


async def test_the_usage_the_provider_reported_is_carried():
    provider = MockProvider(
        MockSpec(usage=Usage(input_tokens=25, output_tokens=42, total_tokens=67))
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.usage.HasField("input_tokens")
    assert response.usage.input_tokens == 25
    assert response.usage.output_tokens == 42
    assert response.usage.total_tokens == 67
    # Every count was the provider's own, so none is named as an estimate.
    assert list(response.usage.estimated_fields) == []


async def test_an_estimated_count_is_named_as_one_on_the_wire():
    # The provider reported the input and output counts and left the total to
    # the gateway, so the total is an estimate: the answer names it, which is
    # the only way a consumer can tell a reported count from a derived one.
    provider = MockProvider(
        MockSpec(
            usage=Usage(
                input_tokens=25,
                output_tokens=42,
                total_tokens=67,
                estimated_fields=frozenset({"total_tokens"}),
            )
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.usage.total_tokens == 67
    assert response.usage.estimated_fields == [pb.USAGE_FIELD_TOTAL_TOKENS]


async def test_attempt_count_is_one_when_the_primary_route_answered(stub):
    response = await stub.Invoke(request(), metadata=bearer())

    assert response.attempt_count == 1
    assert response.fallback_used is False


async def test_a_fallback_reports_the_attempts_the_gateway_made():
    provider = MockProvider(
        [
            MockSpec(error=UnavailableError("the primary route is unreachable")),
            MockSpec(text="Answered by the fallback."),
        ]
    )

    async with serve(provider, registry=make_registry(fallback=True)) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert [route.model_id for route, _, _ in provider.calls] == [
        PRIMARY_MODEL,
        FALLBACK_MODEL,
    ]
    assert response.attempt_count == 2
    assert response.fallback_used is True
    assert response.upstream_model == FALLBACK_MODEL
    assert response.output[0].content.text.text == "Answered by the fallback."


async def test_a_raw_finish_reason_is_carried_beside_its_classification():
    provider = MockProvider(MockSpec(finish_reason=FinishReason.UNKNOWN, raw_finish_reason="eos"))

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.finish_reason == pb.FINISH_REASON_UNKNOWN
    assert response.HasField("raw_finish_reason")
    assert response.raw_finish_reason == "eos"


async def test_an_absent_raw_finish_reason_arrives_absent():
    # The provider reported no reason of its own, and the optional field says
    # so by being absent: an empty string would assert the provider supplied a
    # blank one, which is a different fact.
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(TextPart(text="Explain recursion."),),
            finish_reason=FinishReason.UNKNOWN,
            raw_finish_reason=None,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.finish_reason == pb.FINISH_REASON_UNKNOWN
    assert not response.HasField("raw_finish_reason")


async def test_an_empty_raw_finish_reason_is_kept_apart_from_an_absent_one():
    # The provider did report a reason and it was the empty string: the field
    # is present and empty, which is not the same statement as an absent one.
    provider = MockProvider(MockSpec(finish_reason=FinishReason.STOP, raw_finish_reason=""))

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.HasField("raw_finish_reason")
    assert response.raw_finish_reason == ""


async def test_an_anonymous_call_can_claim_no_tenant():
    # With authentication disabled the caller is anonymous, and the tenant rule
    # discards its claim rather than propagating it: a request field can never
    # create a tenant.
    provider = MockProvider()

    async with serve(provider, auth_enabled=False) as stub:
        response = await stub.Invoke(request(tenant_id=UNBOUND_TENANT), metadata=bearer(None))

    _, invocation, _ = provider.calls[0]
    assert invocation.tenant_id is None
    assert response.model == PUBLIC_MODEL


# --------------------------------------------------------------------------- #
# Idempotency: one key, one answer, no duplicate accounting
# --------------------------------------------------------------------------- #

# A key the provider's answer is recorded under. The inference service keeps the
# record, so a repeat is answered from it and the provider is called once.
IDEMPOTENCY_KEY = "trace-7"


async def test_an_idempotency_key_reaches_the_invocation(stub, provider):
    response = await stub.Invoke(request(idempotency_key=IDEMPOTENCY_KEY), metadata=bearer())

    _, invocation, _ = provider.calls[0]
    assert invocation.idempotency_key == IDEMPOTENCY_KEY
    assert response.idempotency_status == pb.IDEMPOTENCY_STATUS_NEW
    assert len(provider.calls) == 1


async def test_a_repeat_under_the_same_key_is_answered_from_the_record(stub, provider):
    # The retry carries a freshly minted invocation id, which the gateway mints
    # and which is no part of the request's fingerprint, so it is the same
    # request: the answer comes from the record, identifier included.
    first = await stub.Invoke(
        request(invocation_id="inv-9", idempotency_key=IDEMPOTENCY_KEY), metadata=bearer()
    )
    second = await stub.Invoke(
        request(invocation_id="inv-10", idempotency_key=IDEMPOTENCY_KEY), metadata=bearer()
    )

    assert first.idempotency_status == pb.IDEMPOTENCY_STATUS_NEW
    assert second.idempotency_status == pb.IDEMPOTENCY_STATUS_REPLAYED
    assert second.invocation_id == first.invocation_id
    assert [item.content.text.text for item in second.output] == [
        item.content.text.text for item in first.output
    ]
    assert len(provider.calls) == 1


async def test_a_call_with_no_key_is_answered_not_applicable(stub, provider):
    response = await stub.Invoke(request(), metadata=bearer())

    _, invocation, _ = provider.calls[0]
    assert invocation.idempotency_key is None
    assert response.idempotency_status == pb.IDEMPOTENCY_STATUS_NOT_APPLICABLE


async def test_a_key_reused_for_another_payload_is_refused_as_a_conflict(stub, provider):
    """A key names one operation, so reuse is refused rather than executed.

    Executing it would run a request the caller never made and overwrite the record
    the key already holds, letting one key oscillate between unrelated payloads
    indefinitely. The provider is never called.
    """
    await stub.Invoke(request(idempotency_key=IDEMPOTENCY_KEY), metadata=bearer())

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await stub.Invoke(
            request(
                idempotency_key=IDEMPOTENCY_KEY,
                messages=(user_message("A different question entirely."),),
            ),
            metadata=bearer(),
        )

    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    payload = orjson.loads(error.value.details())
    assert payload["reason"] == "idempotency_conflict"
    assert payload["details"]["idempotency_key"] == IDEMPOTENCY_KEY
    assert len(provider.calls) == 1


async def test_an_empty_key_is_refused_rather_than_read_as_no_key():
    # The field is optional, so a stated empty key is a present field carrying
    # nothing — a different statement from an absent one — and the canonical
    # record refuses it rather than collapsing it into no key at all.
    provider = MockProvider()

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(request(idempotency_key=""), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert provider.calls == []


@pytest.mark.parametrize("status", list(IdempotencyStatus))
async def test_every_idempotency_status_is_rendered_on_the_wire(status):
    # The response side of the boundary: each canonical status is written as the
    # protobuf member naming it, on every answer — the enum's zero value is the
    # unspecified member rather than a status, so none is ever left unset.
    provider = ScriptedProvider(
        InvocationResponse(
            model=PUBLIC_MODEL,
            upstream_model=PRIMARY_MODEL,
            content=(TextPart(text="Explain recursion."),),
            finish_reason=FinishReason.STOP,
            idempotency_status=status,
        )
    )

    async with serve(provider) as stub:
        response = await stub.Invoke(request(), metadata=bearer())

    assert response.idempotency_status == getattr(pb, f"IDEMPOTENCY_STATUS_{status.name}")


async def test_a_failure_is_not_recorded_so_a_repeat_is_allowed_to_retry():
    # Only a completed answer is recorded: the first attempt fails, so there is
    # no record for the repeat to be answered from, and the retry reaches the
    # provider — the gateway claims no exactly-once execution.
    provider = MockProvider(
        [
            MockSpec(error=UnavailableError("the upstream is unreachable")),
            MockSpec(text="Answered on the retry."),
        ]
    )

    async with serve(provider) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await stub.Invoke(request(idempotency_key=IDEMPOTENCY_KEY), metadata=bearer())
        retried = await stub.Invoke(request(idempotency_key=IDEMPOTENCY_KEY), metadata=bearer())

    assert raised.value.code() is grpc.StatusCode.UNAVAILABLE
    assert retried.idempotency_status == pb.IDEMPOTENCY_STATUS_NEW
    assert len(provider.calls) == 2


async def test_two_credentials_sharing_a_key_are_two_independent_operations(provider):
    """The same key under two credentials is two operations over a real gRPC call.

    Credential A and credential B are separate trusted principals, so a key each
    uses names an operation of its own: neither replays the other's answer, and a
    repeat by A replays A's own. Both servers share one inference service and
    therefore one store, which is what makes the namespace the thing that separates
    them — two independent stores would not test the claim at all.
    """
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{API_KEY}:{TENANT},{SECOND_KEY}:{SECOND_TENANT}",
        }
    )
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=True,
        allow_anonymous_health=False,
    )
    shared_service = InferenceService(make_registry(), provider)

    server_a = grpc.aio.server()
    port_a = server_a.add_insecure_port(f"{_HOST}:0")
    server_b = grpc.aio.server()
    port_b = server_b.add_insecure_port(f"{_HOST}:0")
    for server in (server_a, server_b):
        gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
            _InvokeOnlyServicer(Invoke(authenticator, shared_service)), server
        )
    await server_a.start()
    await server_b.start()
    channel_a = grpc.aio.insecure_channel(f"{_HOST}:{port_a}")
    channel_b = grpc.aio.insecure_channel(f"{_HOST}:{port_b}")
    stub_a = gateway_pb2_grpc.ModelGatewayServiceStub(channel_a)
    stub_b = gateway_pb2_grpc.ModelGatewayServiceStub(channel_b)
    try:
        shared = request(idempotency_key="shared-key")

        first = await stub_a.Invoke(
            shared, metadata=(("authorization", f"Bearer {API_KEY}"),)
        )
        second = await stub_b.Invoke(
            shared, metadata=(("authorization", f"Bearer {SECOND_KEY}"),)
        )
        repeat = await stub_a.Invoke(
            shared, metadata=(("authorization", f"Bearer {API_KEY}"),)
        )
    finally:
        await channel_a.close()
        await channel_b.close()
        await server_a.stop(grace=None)
        await server_b.stop(grace=None)

    assert first.idempotency_status == pb.IDEMPOTENCY_STATUS_NEW
    assert second.idempotency_status == pb.IDEMPOTENCY_STATUS_NEW
    assert repeat.idempotency_status == pb.IDEMPOTENCY_STATUS_REPLAYED
    assert len(provider.calls) == 2


async def test_one_key_reused_for_another_payload_is_refused_over_grpc(stub, provider):
    """Reuse is refused before any provider is called, and the record is left intact."""
    await stub.Invoke(request(idempotency_key="key-1"), metadata=bearer())

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await stub.Invoke(
            request(
                idempotency_key="key-1",
                messages=(user_message("A different question entirely."),),
            ),
            metadata=bearer(),
        )

    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    assert len(provider.calls) == 1
