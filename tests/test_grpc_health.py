"""The standard health protocol the listener serves (PRD section 5).

The servicer is the ``grpcio-health-checking`` package's own, so these tests do
not re-test the package's state machine. They test the listener's side of the
protocol: that the standard servicer is the one registered, that the overall
service reports ``NOT_SERVING`` before the listener starts and ``SERVING``
after, that ``Check`` answers ``NOT_FOUND`` for an unknown service name, that
``Watch`` answers ``SERVICE_UNKNOWN`` for one and holds the call open — the
behaviour an external review found missing — and that a ``Watch`` on the
overall service is told the transition when the state changes.

The listener is started and stopped directly, the way the application's
``on_start`` and ``on_stop`` drive it, because the transitions under test are the
listener's: the state is set and flipped by the listener's lifetime, not by the
servicer's own.
"""

import asyncio
import socket

import grpc
import grpc.aio
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.grpc.server import GrpcServer
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

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
    """A port nothing else holds, for the listener to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _listener() -> GrpcServer:
    """An unstarted listener over the same objects the application holds."""
    settings = Settings.from_env({"CHORA_GRPC_PORT": str(_free_port())})
    registry = _registry()
    return GrpcServer(
        service=InferenceService(registry, MockProvider()),
        registry=registry,
        settings=settings,
        api_keys=ApiKeySet.from_settings(settings),
    )


@pytest.fixture
async def grpc_server():
    """A started listener on a channel, stopped at the end as the app drives it."""
    server = _listener()
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
    try:
        yield server, channel
    finally:
        await channel.close()
        await server.stop()


async def test_the_overall_service_is_not_serving_until_the_listener_starts():
    """Before the listener starts the overall service is NOT_SERVING.

    The listener binds its port in its constructor, so the window before
    ``start`` is real, but no RPC is answered in it — a ``Check`` cannot be
    answered until the server starts. What must be truthful in that window is
    the state, so it is read here directly rather than over the wire.
    """
    server = _listener()
    try:
        assert (
            server._health._server_status[""]
            == health_pb2.HealthCheckResponse.NOT_SERVING
        )
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
        try:
            stub = health_pb2_grpc.HealthStub(channel)
            response = await stub.Check(health_pb2.HealthCheckRequest(service=""))
            assert response.status == health_pb2.HealthCheckResponse.SERVING
        finally:
            await channel.close()
    finally:
        await server.stop()


async def test_check_on_the_overall_service_reports_serving(grpc_server):
    """The empty service name is the listener's overall health, and it is serving."""
    _, channel = grpc_server
    stub = health_pb2_grpc.HealthStub(channel)
    response = await stub.Check(health_pb2.HealthCheckRequest(service=""))
    assert response.status == health_pb2.HealthCheckResponse.SERVING


async def test_check_on_an_unknown_service_name_is_not_found(grpc_server):
    """Check answers NOT_FOUND for a service name the gateway does not carry."""
    _, channel = grpc_server
    stub = health_pb2_grpc.HealthStub(channel)
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await stub.Check(health_pb2.HealthCheckRequest(service="not-a-service"))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


async def test_watch_on_an_unknown_service_name_sends_service_unknown_and_stays_open(
    grpc_server,
):
    """Watch answers SERVICE_UNKNOWN for an unknown name and holds the call open.

    This is the behaviour the review flagged: the name is reported as
    SERVICE_UNKNOWN rather than refused, and the call does not terminate on its
    own. The stream staying open is asserted by the call not being done after a
    pause — a Watch that ended on its own would have completed it.
    """
    _, channel = grpc_server
    stub = health_pb2_grpc.HealthStub(channel)
    call = stub.Watch(health_pb2.HealthCheckRequest(service="not-a-service"))
    first = await asyncio.wait_for(call.read(), timeout=5)
    assert first.status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
    await asyncio.sleep(1)
    assert call.done() is False
    call.cancel()


async def test_watch_on_the_overall_service_is_told_the_transition(grpc_server):
    """A Watch on the overall service is notified when the state changes.

    The transition is driven through the registered servicer's ``set`` — the
    same call the listener's ``start`` and ``stop`` make — because what is
    under test is that the listener's watcher is notified, not the servicer's
    state machine.
    """
    server, channel = grpc_server
    stub = health_pb2_grpc.HealthStub(channel)
    call = stub.Watch(health_pb2.HealthCheckRequest(service=""))
    first = await asyncio.wait_for(call.read(), timeout=5)
    assert first.status == health_pb2.HealthCheckResponse.SERVING
    await server._health.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
    second = await asyncio.wait_for(call.read(), timeout=5)
    assert second.status == health_pb2.HealthCheckResponse.NOT_SERVING
    call.cancel()
