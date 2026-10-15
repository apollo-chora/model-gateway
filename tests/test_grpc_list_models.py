"""Tests for the ListModels RPC.

Every test drives the RPC over a real in-process gRPC server rather than calling
the handler directly, because the transport is part of what they pin: that the
servicer the generated contract expects dispatches to this handler, that a
credential travels in call metadata, and that a refusal arrives as a status the
caller branches on rather than as an exception. Only ``ListModels`` is served
here; the other three RPCs belong to their own modules and are tested there.

What is asserted is what a caller reads: the public models the deployment
configured, each with the capabilities its primary route declares and the limits
the deployment set — and, for a limit it did not set, no field at all rather than
a zero. The registry is built from routes directly, or from an environment
mapping whose endpoints are IP literals, so nothing here touches DNS.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
import pytest

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.list_models import ListModels
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

API_KEY = "sk-gateway-test-key"
WRONG_KEY = "sk-gateway-wrong-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
UPSTREAM_KEY = "sk-upstream-secret-key"
PUBLIC_ADDRESS = "93.184.216.34"

LIMIT_VARIABLES = (
    "TEXT_LLM_CONTEXT_WINDOW",
    "TEXT_LLM_MAX_OUTPUT_TOKENS",
    "EMBEDDINGS_DIMENSIONS",
)


@pytest.fixture(autouse=True)
def _hold_the_limit_variables_out_of_the_environment(monkeypatch):
    """Keep a registry built from routes alone free of the process environment.

    The advertised text limits are deployment configuration rather than route
    fields, so a registry constructed from routes reads them from ``os.environ``.
    Deleting them here is what makes a route-only registry in these tests the
    deployment the test describes rather than the one the developer's shell
    happens to export.
    """
    for variable in LIMIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def make_route(
    operation: Operation,
    model_id: str,
    *,
    capabilities: frozenset[Capability],
) -> Route:
    return Route(
        operation=operation,
        index=0,
        endpoint=UPSTREAM_ENDPOINT,
        base_url=UPSTREAM_ENDPOINT,
        model_id=model_id,
        format="openai",
        api_key=UPSTREAM_KEY,
        capabilities=capabilities,
        extra_headers={"x-tenant": UPSTREAM_KEY},
        allow_absolute_path_url=False,
    )


def configured_registry() -> ModelRegistry:
    """A registry over all three operations, each declaring its own capabilities."""
    return ModelRegistry(
        (
            make_route(
                Operation.TEXT,
                "upstream-text",
                capabilities=frozenset({Capability.CHAT, Capability.TOOLS}),
            ),
            make_route(
                Operation.IMAGE,
                "upstream-image",
                capabilities=frozenset({Capability.GENERATION}),
            ),
            make_route(
                Operation.EMBEDDINGS,
                "upstream-embedding",
                capabilities=frozenset({Capability.EMBEDDINGS}),
            ),
        )
    )


def environment(**overrides: str) -> dict[str, str]:
    """A complete deployment environment with IP-literal endpoints, so no DNS is needed."""
    env = {
        "TEXT_LLM_ENDPOINT": f"https://{PUBLIC_ADDRESS}/v1",
        "TEXT_LLM_MODEL_ID": "upstream-text",
        "TEXT_LLM_FORMAT": "openai",
        "TEXT_LLM_CAPABILITIES": "chat,tools",
        "IMAGE_GEN_ENDPOINT": f"https://{PUBLIC_ADDRESS}/images/v1",
        "IMAGE_GEN_MODEL_ID": "upstream-image",
        "IMAGE_GEN_FORMAT": "azure",
        "EMBEDDINGS_ENDPOINT": f"https://{PUBLIC_ADDRESS}/embeddings/v1",
        "EMBEDDINGS_MODEL_ID": "upstream-embedding",
        "EMBEDDINGS_FORMAT": "ollama",
    }
    env.update(overrides)
    return env


def authenticator(*, auth_enabled: bool, api_keys: tuple[str, ...] = ()) -> Authenticator:
    """The inbound authentication policy of a deployment with these settings."""
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true" if auth_enabled else "false",
            "GATEWAY_API_KEYS": ",".join(api_keys),
        }
    )
    return Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )


class _ListModelsServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """Serves ``ListModels`` and leaves the other RPCs unimplemented."""

    def __init__(self, list_models: ListModels) -> None:
        self._list_models = list_models

    async def ListModels(self, request, context):
        return await self._list_models.list_models(request, context)


@asynccontextmanager
async def open_listing(
    registry: ModelRegistry,
    *,
    auth_enabled: bool = False,
    api_keys: tuple[str, ...] = (),
) -> AsyncIterator[gateway_pb2_grpc.ModelGatewayServiceStub]:
    """Serve the listing on an ephemeral port and yield a connected stub."""
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
        _ListModelsServicer(
            ListModels(
                authenticator(auth_enabled=auth_enabled, api_keys=api_keys),
                registry,
            )
        ),
        server,
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=None)


def bearer(key: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {key}"),)


def refusal_reason(error: grpc.aio.AioRpcError) -> str:
    """The gateway's own stable reason, as it travels in the status details."""
    return json.loads(error.details())["reason"]


# --------------------------------------------------------------------------- #
# The listing
# --------------------------------------------------------------------------- #


async def test_list_models_lists_every_configured_public_model():
    async with open_listing(configured_registry()) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    assert [model.id for model in response.models] == [
        "text-default",
        "image-default",
        "embedding-default",
    ]


async def test_list_models_reports_each_models_own_capabilities():
    async with open_listing(configured_registry()) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    capabilities = {model.id: list(model.capabilities) for model in response.models}

    assert capabilities == {
        "text-default": ["chat", "tools"],
        "image-default": ["generation"],
        "embedding-default": ["embeddings"],
    }


async def test_list_models_omits_an_operation_with_no_configured_route():
    registry = ModelRegistry(
        (make_route(Operation.TEXT, "upstream-text", capabilities=frozenset({Capability.CHAT})),)
    )

    async with open_listing(registry) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    assert [model.id for model in response.models] == ["text-default"]


async def test_a_deployment_with_no_configured_routes_lists_nothing():
    async with open_listing(ModelRegistry(())) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    assert list(response.models) == []


async def test_each_model_carries_the_operation_its_public_id_names():
    async with open_listing(configured_registry()) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    operations = {model.id: model.operation for model in response.models}

    assert operations == {
        "text-default": pb.OPERATION_TEXT,
        "image-default": pb.OPERATION_IMAGE,
        "embedding-default": pb.OPERATION_EMBEDDINGS,
    }


async def test_a_route_declaring_no_capabilities_advertises_an_empty_set():
    registry = ModelRegistry(
        (make_route(Operation.TEXT, "upstream-text", capabilities=frozenset()),)
    )

    async with open_listing(registry) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    (model,) = response.models
    assert list(model.capabilities) == []


async def test_the_listing_carries_no_upstream_configuration():
    async with open_listing(configured_registry()) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    serialized = response.SerializeToString()

    assert UPSTREAM_KEY.encode() not in serialized
    assert UPSTREAM_ENDPOINT.encode() not in serialized
    assert b"upstream-text" not in serialized
    assert b"x-tenant" not in serialized


# --------------------------------------------------------------------------- #
# The advertised limits
# --------------------------------------------------------------------------- #


async def test_each_limit_the_deployment_configured_is_advertised():
    registry = ModelRegistry.from_env(
        environment(
            TEXT_LLM_CONTEXT_WINDOW="1048576",
            TEXT_LLM_MAX_OUTPUT_TOKENS="131072",
            EMBEDDINGS_DIMENSIONS="1024",
        )
    )

    async with open_listing(registry) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    models = {model.id: model for model in response.models}

    assert models["text-default"].context_window == 1048576
    assert models["text-default"].max_output_tokens == 131072
    assert models["embedding-default"].dimensions == 1024


async def test_a_limit_the_deployment_did_not_configure_is_absent_rather_than_zero():
    async with open_listing(ModelRegistry.from_env(environment())) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    for model in response.models:
        assert not model.HasField("context_window")
        assert not model.HasField("max_output_tokens")
        assert not model.HasField("dimensions")


async def test_the_text_limits_are_not_advertised_on_another_operations_model():
    registry = ModelRegistry.from_env(
        environment(TEXT_LLM_CONTEXT_WINDOW="1048576", TEXT_LLM_MAX_OUTPUT_TOKENS="131072")
    )

    async with open_listing(registry) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    models = {model.id: model for model in response.models}

    assert not models["image-default"].HasField("context_window")
    assert not models["image-default"].HasField("max_output_tokens")
    assert not models["embedding-default"].HasField("context_window")
    assert not models["embedding-default"].HasField("max_output_tokens")


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


async def test_a_call_without_a_credential_is_refused_when_authentication_is_enabled():
    async with open_listing(
        configured_registry(), auth_enabled=True, api_keys=(API_KEY,)
    ) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as refusal:
            await stub.ListModels(pb.ListModelsRequest())

    assert refusal.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert refusal_reason(refusal.value) == "authentication_required"


async def test_a_call_with_an_invalid_credential_is_refused():
    async with open_listing(
        configured_registry(), auth_enabled=True, api_keys=(API_KEY,)
    ) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as refusal:
            await stub.ListModels(pb.ListModelsRequest(), metadata=bearer(WRONG_KEY))

    assert refusal.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert refusal_reason(refusal.value) == "invalid_credentials"


async def test_a_call_with_a_valid_credential_lists_the_models():
    async with open_listing(
        configured_registry(), auth_enabled=True, api_keys=(API_KEY,)
    ) as stub:
        response = await stub.ListModels(pb.ListModelsRequest(), metadata=bearer(API_KEY))

    assert [model.id for model in response.models] == [
        "text-default",
        "image-default",
        "embedding-default",
    ]


async def test_a_deployment_with_authentication_disabled_lists_anonymously():
    async with open_listing(configured_registry(), auth_enabled=False) as stub:
        response = await stub.ListModels(pb.ListModelsRequest())

    assert [model.id for model in response.models] == [
        "text-default",
        "image-default",
        "embedding-default",
    ]
