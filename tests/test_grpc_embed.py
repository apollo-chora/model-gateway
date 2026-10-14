"""Tests for the Embed RPC: the embeddings operation over gRPC (PRD sections 4.7, 5.1 and 9.5).

The handler is exercised through a real in-process gRPC server and the generated
stub, so what is tested is the RPC, its wire translation, its authentication and
its error mapping together — a caller's whole path rather than a call to a method.
The provider stands in at the binding boundary and every route names an IP
literal, so nothing reaches the network.

Each test pins one promise of the contract: that the canonical request reaches the
provider with the model, the input and the caller's own parameters, that the
vectors come back in the caller's input order, that the answer carries only the
usage counts the provider actually reported, and that every refusal answers with
the taxonomy's own status and reason — an absent credential, a credential that
matches nothing, an unknown model, an encoding the gateway cannot answer in, and a
vector of the wrong shape.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import Embedding, EmbeddingRequest, EmbeddingResponse, Usage
from model_gateway.errors import UnavailableError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.embed import Embed
from model_gateway.grpc.identity import Authenticator
from model_gateway.inference import InferenceService
from model_gateway.providers import MockProvider
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.mock import MockEmbedSpec
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

KEY = "sk-gateway-test-key"
WRONG_KEY = "sk-gateway-wrong-key"
TENANT = "tenant-a"
PUBLIC_MODEL = "embedding-default"
UPSTREAM_MODEL = "upstream-embedding"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
UPSTREAM_KEY = "sk-upstream-secret-key"
VECTOR = (0.12, -0.34, 0.56)


def route(**overrides: object) -> Route:
    """A standalone embeddings route, the shape a deployment configures."""
    values: dict[str, object] = {
        "operation": Operation.EMBEDDINGS,
        "index": 0,
        "endpoint": UPSTREAM_ENDPOINT,
        "base_url": UPSTREAM_ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": "openai",
        "api_key": UPSTREAM_KEY,
        "capabilities": frozenset({Capability.EMBEDDINGS}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def service(provider: object, **route_overrides: object) -> InferenceService:
    """An inference service over one embeddings route and the given provider."""
    return InferenceService(ModelRegistry((route(**route_overrides),)), provider)


def authenticator(*, auth_enabled: bool = True) -> Authenticator:
    """An authenticator over a key set with one key bound to ``TENANT``.

    With authentication disabled the configured key set is empty, which is the
    state a deployment that sets neither ``GATEWAY_AUTH_ENABLED`` nor
    ``GATEWAY_API_KEYS`` is in.
    """
    env: Mapping[str, str] = (
        {"GATEWAY_AUTH_ENABLED": "true", "GATEWAY_API_KEYS": f"{KEY}:{TENANT}"}
        if auth_enabled
        else {"GATEWAY_AUTH_ENABLED": "false", "GATEWAY_API_KEYS": ""}
    )
    settings = Settings.from_env(env)
    return Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )


class FakeEmbeddingProvider:
    """An ``EmbeddingProvider`` double: records the calls and answers from a script.

    The gRPC surface is what these tests are about, so the provider stands in at
    the binding boundary and no socket is opened. With no scripted answer it
    returns one vector per input text, at the position the caller's input holds; a
    test that needs an answer the gateway must refuse — a vector of the wrong
    length, a count the provider never reported — scripts it explicitly.

    Every call is recorded as a ``(route, request, context)`` triple, which is how
    a test proves the canonical request reached the provider.
    """

    def __init__(
        self,
        *,
        data: tuple[Embedding, ...] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.calls: list[tuple[Route, EmbeddingRequest, AttemptContext]] = []
        self._data = data
        self._usage = usage

    async def embed(
        self, route_: Route, request: EmbeddingRequest, context: AttemptContext
    ) -> EmbeddingResponse:
        self.calls.append((route_, request, context))
        return EmbeddingResponse(
            model=request.model,
            upstream_model=route_.model_id,
            data=self._data if self._data is not None else _vectors(request),
            usage=Usage() if self._usage is None else self._usage,
            invocation_id=context.invocation_id or "invocation-from-the-provider",
        )


def _vectors(request: EmbeddingRequest) -> tuple[Embedding, ...]:
    """One vector per input text, at the position that text holds."""
    return tuple(Embedding(index=index, values=VECTOR) for index in range(len(request.input)))


class EmbedOnly(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """The generated servicer with only the RPC under test implemented.

    The generated base class carries every method of the service and answers
    ``UNIMPLEMENTED`` for the ones it does not implement, so this test can serve
    the Embed RPC alone rather than depending on the sibling handlers.
    """

    def __init__(self, embed: Embed) -> None:
        self._embed = embed

    async def Embed(self, request, context):
        return await self._embed.embed(request, context)


@asynccontextmanager
async def served(
    inference: InferenceService, auth: Authenticator
) -> AsyncIterator[gateway_pb2_grpc.ModelGatewayServiceStub]:
    """Serve the Embed RPC on an ephemeral port and yield a stub connected to it."""
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
        EmbedOnly(Embed(auth, inference)), server
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=None)


async def call(
    stub: gateway_pb2_grpc.ModelGatewayServiceStub,
    request: pb.EmbedRequest,
    *,
    key: str | None = KEY,
):
    """Call Embed, presenting a credential or, with ``key=None``, none at all."""
    metadata = () if key is None else (("authorization", f"Bearer {key}"),)
    return await stub.Embed(request, metadata=metadata)


def payload(error: grpc.aio.AioRpcError) -> dict[str, object]:
    """The sanitised public payload a refusal carries, decoded from its details."""
    return json.loads(error.details())


async def test_embed_answers_with_the_vectors_the_provider_produced():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator()) as stub:
        response = await call(
            stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["first", "second", "third"])
        )

    assert response.object == "list"
    assert response.model == PUBLIC_MODEL
    assert response.upstream_model == UPSTREAM_MODEL
    assert response.invocation_id
    assert [item.index for item in response.data] == [0, 1, 2]
    assert [list(item.embedding) for item in response.data] == [list(VECTOR)] * 3


async def test_embed_hands_the_provider_the_canonical_request():
    # A route that accepts a caller-selectable dimensionality, because a stated
    # dimension is refused by the gateway core on a route that does not declare one.
    provider = FakeEmbeddingProvider()
    metadata = Struct()
    # The struct carries the JSON values the canonical record holds, an integer
    # beside a string, where a map of strings could carry only strings.
    metadata.update({"trace": "abc", "attempt": 2})
    async with served(service(provider, supports_dimensions_param=True), authenticator()) as stub:
        await call(
            stub,
            pb.EmbedRequest(
                model=PUBLIC_MODEL,
                input=["first", "second"],
                dimensions=3,
                user="caller-7",
                metadata=metadata,
            ),
        )

    assert len(provider.calls) == 1
    _, request, _ = provider.calls[0]
    assert request == EmbeddingRequest(
        model=PUBLIC_MODEL,
        input=("first", "second"),
        dimensions=3,
        user="caller-7",
        metadata={"trace": "abc", "attempt": 2},
    )


async def test_embed_returns_the_vectors_in_the_callers_input_order():
    provider = FakeEmbeddingProvider(
        data=(
            Embedding(index=2, values=(2.0,)),
            Embedding(index=0, values=(0.0,)),
            Embedding(index=1, values=(1.0,)),
        )
    )
    async with served(service(provider), authenticator()) as stub:
        response = await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a", "b", "c"]))

    assert [item.index for item in response.data] == [0, 1, 2]
    assert [item.embedding[0] for item in response.data] == [0.0, 1.0, 2.0]


async def test_embed_reports_only_the_usage_counts_the_provider_stated():
    provider = FakeEmbeddingProvider(usage=Usage(input_tokens=7))
    async with served(service(provider), authenticator()) as stub:
        response = await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]))

    assert response.usage.HasField("input_tokens")
    assert response.usage.input_tokens == 7
    assert not response.usage.HasField("total_tokens")
    assert not response.usage.HasField("output_tokens")
    assert not response.usage.HasField("cached_tokens")


async def test_embed_names_the_counts_the_gateway_derived():
    # The provider reported the input count and left the total to the gateway,
    # so the total is an estimate and the answer names it as one — the only
    # way a consumer can tell a reported count from a derived one.
    provider = FakeEmbeddingProvider(
        usage=Usage(input_tokens=7, total_tokens=7, estimated_fields=frozenset({"total_tokens"}))
    )
    async with served(service(provider), authenticator()) as stub:
        response = await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]))

    assert response.usage.total_tokens == 7
    assert response.usage.estimated_fields == [pb.USAGE_FIELD_TOTAL_TOKENS]


async def test_embed_refuses_a_base64_encoding_and_names_it():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(
                stub,
                pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"], encoding_format="base64"),
            )

    error = raised.value
    assert error.code() is grpc.StatusCode.FAILED_PRECONDITION
    body = payload(error)
    assert body["reason"] == "unsupported_capability"
    assert "'base64'" in body["message"]
    assert provider.calls == []


async def test_embed_refuses_a_wrong_dimension_vector_as_a_malformed_response():
    provider = FakeEmbeddingProvider(data=(Embedding(index=0, values=(0.1, 0.2, 0.3, 0.4)),))
    async with served(service(provider, dimensions=3), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]))

    error = raised.value
    assert error.code() is grpc.StatusCode.UNKNOWN
    assert payload(error)["reason"] == "malformed_response"
    assert len(provider.calls) == 1


async def test_embed_refuses_a_call_that_presents_no_credential():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]), key=None)

    error = raised.value
    assert error.code() is grpc.StatusCode.UNAUTHENTICATED
    assert payload(error)["reason"] == "authentication_required"
    assert provider.calls == []


async def test_embed_refuses_a_credential_that_matches_nothing():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]), key=WRONG_KEY)

    error = raised.value
    assert error.code() is grpc.StatusCode.UNAUTHENTICATED
    assert payload(error)["reason"] == "invalid_credentials"
    assert provider.calls == []


async def test_embed_refuses_an_unknown_model_without_calling_the_provider():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(stub, pb.EmbedRequest(model="no-such-model", input=["a"]))

    error = raised.value
    assert error.code() is grpc.StatusCode.NOT_FOUND
    assert payload(error)["reason"] == "model_not_found"
    assert provider.calls == []


async def test_embed_surfaces_a_provider_failure_as_its_own_status():
    provider = MockProvider(embed=MockEmbedSpec(error=UnavailableError("the upstream is down")))
    async with served(service(provider), authenticator()) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]))

    error = raised.value
    assert error.code() is grpc.StatusCode.UNAVAILABLE
    assert payload(error)["reason"] == "provider_unavailable"
    assert len(provider.embed_calls) == 1


async def test_embed_serves_an_anonymous_call_when_authentication_is_disabled():
    provider = FakeEmbeddingProvider()
    async with served(service(provider), authenticator(auth_enabled=False)) as stub:
        response = await call(stub, pb.EmbedRequest(model=PUBLIC_MODEL, input=["a"]), key=None)

    assert response.model == PUBLIC_MODEL
    assert len(provider.calls) == 1


async def test_the_embeddings_request_carries_no_tenant_claim():
    """PRD section 5.1 has nothing to reconcile on this RPC, and that is pinned here.

    Neither ``EmbedRequest`` nor the canonical embeddings request has a tenant
    field, so no call can claim a tenant and none can be refused with
    ``PERMISSION_DENIED`` for claiming one: the tenant an invocation acts under is
    the credential's, and this operation never states another. The absence is
    asserted rather than left implicit, because it is what makes the handler's
    reconciliation a statement of the rule instead of a check that could fail.
    """
    assert "tenant_id" not in {field.name for field in pb.EmbedRequest.DESCRIPTOR.fields}
    assert not hasattr(EmbeddingRequest(model=PUBLIC_MODEL, input=("a",)), "tenant_id")
