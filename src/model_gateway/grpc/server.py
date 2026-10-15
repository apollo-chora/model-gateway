"""Assemble, start and stop the gRPC listener (PRD sections 5 and 18).

The listener is built from the same objects the HTTP application holds — the
inference service, the model registry, the settings and the configured key set —
so a gRPC call and an HTTP request reach identical policy. Nothing here decides
what a caller may do; the servicer classes beside this module translate, and
the inference service decides.

The listener binds an insecure socket on ``CHORA_GRPC_PORT``, default ``9090``
(PRD section 5). Transport security is not offered: the gateway terminates no
TLS, and a deployment that needs it terminates it in front, the same way the HTTP
listener expects. Binding is verified rather than assumed —
``add_insecure_port`` returns ``0`` when the address cannot be bound, and
answering a probe on an unbound port would report a listener the deployment does
not have.

The standard health protocol is registered beside the gateway's own service on
the same port, so one listener serves both.
"""

from __future__ import annotations

import grpc
import structlog
from grpc_health.v1 import health_pb2, health_pb2_grpc

from model_gateway.config import ConfigurationError, Settings
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.embed import Embed
from model_gateway.grpc.health import HealthServicer
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.invoke import Invoke
from model_gateway.grpc.list_models import ListModels
from model_gateway.grpc.streaming import InvokeStream
from model_gateway.grpc.tracing import continue_trace
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

__all__ = ["GrpcServer", "ModelGatewayServiceServicer"]

logger = structlog.get_logger(__name__)

_BIND_ADDRESS = "0.0.0.0"


class GrpcServer:
    """The gRPC listener, ready to be started beside the HTTP server."""

    def __init__(
        self,
        *,
        service,
        registry: ModelRegistry,
        settings: Settings,
        api_keys: ApiKeySet,
    ) -> None:
        self._service = service
        self._registry = registry
        self._settings = settings
        self._authenticator = Authenticator(
            api_keys,
            auth_enabled=settings.auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
        # The same bound the HTTP transport applies, in this transport's own unit:
        # gRPC bounds a message rather than a body, and reports one over the bound
        # as RESOURCE_EXHAUSTED — the status the taxonomy maps for this condition —
        # so one configured value bounds both transports (PRD section 14). The
        # refusal a caller reads for an oversized message is the library's own
        # rather than the gateway's: it shares that status but not the gateway's
        # reason or payload, which is a property of where the bound is enforced
        # rather than a choice, and the HTTP side is where the gateway states it.
        self._grpc = grpc.aio.server(
            options=[("grpc.max_receive_message_length", settings.max_request_body_bytes)]
        )
        gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
            ModelGatewayServiceServicer(
                Invoke(self._authenticator, service),
                InvokeStream(self._authenticator, service),
                Embed(self._authenticator, service),
                ListModels(self._authenticator, registry),
            ),
            self._grpc,
        )
        # The package's constructor starts the overall service as SERVING, and
        # start() sets it again to notify any watcher that opened in between.
        # The state before start() is deliberately left alone: the listener
        # reserves its port here but answers no RPC until start(), so no probe
        # can read that state and there is nothing to correct.
        self._health = HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(self._health, self._grpc)
        self._port = self._grpc.add_insecure_port(f"{_BIND_ADDRESS}:{settings.grpc_port}")
        if not self._port:
            raise ConfigurationError(
                f"CHORA_GRPC_PORT is {settings.grpc_port} but the gRPC listener could not bind it; "
                "choose a free port"
            )

    @property
    def port(self) -> int:
        """The port the listener actually bound, which is the configured one."""
        return self._port

    async def start(self) -> None:
        """Start the listener and mark the overall health check serving."""
        await self._grpc.start()
        await self._health.set("", health_pb2.HealthCheckResponse.SERVING)
        logger.info(
            "model-gateway gRPC listener started",
            grpc_port=self._port,
            models=[info.id for info in self._registry.list_models()],
        )

    async def stop(self) -> None:
        """Stop the listener, refusing new calls and ending open ones.

        Graceful shutdown is entered before the server stops so a watcher is
        told the transition to NOT_SERVING rather than left with a vanished
        connection; entering it also ignores any later set, so a state change
        after this point cannot report a listener that has stopped.
        """
        await self._health.enter_graceful_shutdown()
        await self._grpc.stop(grace=None)
        logger.info("model-gateway gRPC listener stopped")


class ModelGatewayServiceServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """Dispatch each RPC to the class that implements it.

    The generated servicer requires one class to carry every method of the
    service, and this gateway keeps one class per RPC so each translation stands
    on its own and can be read and tested alone. This class is therefore a
    dispatcher and nothing else: it holds no state, makes no decision and
    contains no translation.

    Every collaborator takes the authenticator it needs: an inference RPC needs
    the inference service, the listing needs the model registry, and each
    authenticates the call it serves.
    """

    def __init__(
        self, invoke: Invoke, invoke_stream: InvokeStream, embed: Embed, list_models: ListModels
    ) -> None:
        self._invoke = invoke
        self._invoke_stream = invoke_stream
        self._embed = embed
        self._list_models = list_models

    async def Invoke(self, request, context):
        async with continue_trace(context):
            return await self._invoke.invoke(request, context)

    async def InvokeStream(self, request, context):
        """Stream the chunks of one invocation.

        This method is an async generator rather than an ``async def`` that
        returns one: gRPC's server side requires the behaviour of a
        server-streaming RPC to be an async generator itself, and an ``async
        def`` that returns one is answered with an empty stream rather than an
        error — the caller is told nothing at all.
        """
        async with continue_trace(context):
            async for chunk in self._invoke_stream.invoke_stream(request, context):
                yield chunk

    async def Embed(self, request, context):
        async with continue_trace(context):
            return await self._embed.embed(request, context)

    async def ListModels(self, request, context):
        async with continue_trace(context):
            return await self._list_models.list_models(request, context)
