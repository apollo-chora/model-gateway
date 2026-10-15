"""The gRPC listener wired into the application the server runs.

The four RPC modules are tested on their own. These tests prove the seams
between them: that the application the entrypoint builds starts the listener on
the configured port, that the standard health protocol answers on the same port,
and that a real gRPC client reaches the inference service over it — the path a
deployment actually takes, which no unit test of a single RPC covers.

The application is started and stopped directly, exactly as the factory tests
drive the HTTP application, because the listener's lifetime is the application's:
``on_start`` starts it and ``on_stop`` stops it.
"""

import socket

import grpc
import grpc.aio
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.grpc import gateway_pb2, gateway_pb2_grpc
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider
from model_gateway.registry import ModelRegistry

_SECRET_KEY = "sk-upstream-secret-key"
_UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"


def _registry() -> ModelRegistry:
    """The two operations the gateway serves, over a deterministic mock."""
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=_UPSTREAM_ENDPOINT,
                base_url=_UPSTREAM_ENDPOINT,
                model_id="upstream-text",
                format="openai",
                api_key=_SECRET_KEY,
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={},
                allow_absolute_path_url=False,
            ),
            Route(
                operation=Operation.EMBEDDINGS,
                index=0,
                endpoint=_UPSTREAM_ENDPOINT,
                base_url=_UPSTREAM_ENDPOINT,
                model_id="upstream-embedding",
                format="openai",
                api_key=_SECRET_KEY,
                capabilities=frozenset({Capability.EMBEDDINGS}),
                extra_headers={},
                allow_absolute_path_url=False,
                supports_dimensions_param=True,
            ),
        )
    )


def _free_port() -> int:
    """A port nothing else holds, for the test's listener to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request() -> gateway_pb2.InvokeRequest:
    """One valid Invoke request: a single user turn and a stated modality.

    ``RESPONSE_MODALITY_TEXT`` is stated explicitly because the mapping refuses
    the unspecified zero value rather than reading it as the default — so a
    request that says nothing is not a valid request on this contract.
    """
    request = gateway_pb2.InvokeRequest(
        invocation_id="inv-grpc-1",
        model="text-default",
        response_modality=gateway_pb2.RESPONSE_MODALITY_TEXT,
    )
    message = request.messages.add()
    message.role = gateway_pb2.ROLE_USER
    message.content.add().text.text = "hello"
    return request


@pytest.fixture
async def grpc_channel():
    """A channel to the listener the started application holds.

    ``serve_grpc`` is what makes the factory build the listener at all, so the
    flag is the whole of the wiring under test: without it the port is not bound
    and the channel cannot connect.
    """
    settings = Settings.from_env({"CHORA_GRPC_PORT": str(_free_port())})
    app = create_app(
        settings,
        registry=_registry(),
        service=InferenceService(_registry(), MockProvider()),
        serve_grpc=True,
    )
    await app.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{settings.grpc_port}")
    try:
        yield channel
    finally:
        await channel.close()
        await app.stop()


async def test_the_listener_binds_the_configured_port(grpc_channel):
    """The listener is on the port the deployment configured, and answers."""
    stub = gateway_pb2_grpc.ModelGatewayServiceStub(grpc_channel)
    response = await stub.ListModels(gateway_pb2.ListModelsRequest())
    assert [model.id for model in response.models] == ["text-default", "embedding-default"]


async def test_an_invoke_reaches_the_inference_service(grpc_channel):
    """A real gRPC client reaches the inference service through the listener."""
    stub = gateway_pb2_grpc.ModelGatewayServiceStub(grpc_channel)
    response = await stub.Invoke(_request())
    assert response.model == "text-default"
    assert response.upstream_model == "upstream-text"
    # The mock reports no token counts, and an unreported count stays absent.
    assert response.usage.HasField("input_tokens") is False


async def test_a_stream_reaches_the_caller_as_chunks(grpc_channel):
    stub = gateway_pb2_grpc.ModelGatewayServiceStub(grpc_channel)
    chunks = [chunk async for chunk in stub.InvokeStream(_request())]
    assert chunks, "a stream produced no chunks"
    assert chunks[-1].WhichOneof("payload") == "finish"
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


async def test_an_embed_reaches_the_caller(grpc_channel):
    stub = gateway_pb2_grpc.ModelGatewayServiceStub(grpc_channel)
    response = await stub.Embed(
        gateway_pb2.EmbedRequest(model="embedding-default", input=["one", "two"])
    )
    assert [item.index for item in response.data] == [0, 1]
    assert len(response.data[0].embedding) == 4


async def test_the_standard_health_protocol_answers_on_the_same_port(grpc_channel):
    """The health protocol shares the listener, so one port serves both."""
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    response = await stub.Check(health_pb2.HealthCheckRequest(service=""))
    assert response.status == health_pb2.HealthCheckResponse.SERVING


async def test_the_health_protocol_refuses_an_unknown_service_name(grpc_channel):
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await stub.Check(health_pb2.HealthCheckRequest(service="not-a-service"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND
