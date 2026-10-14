"""The standard gRPC Health Checking Protocol (PRD section 5).

The protocol is the one load balancers, service meshes and orchestrators probe:
``Check`` answers one of three serving states for a service name, and ``Watch``
streams the state as it changes. The gateway registers the empty service name,
which is the server's overall health rather than any one RPC.

The servicer holds no state of its own: it reports the serving state the server
was started with. A gateway that has not started is not serving, one that has is,
and one that is shutting down is not — so ``Check`` before the listener starts
reports ``NOT_SERVING`` rather than answering for a server that would reject the
connection anyway.
"""

import asyncio

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

__all__ = ["HealthServicer"]

_OVERALL = ""
_POLL_SECONDS = 5.0


class HealthServicer(health_pb2_grpc.HealthServicer):
    """Answers the health protocol from the serving state the listener reports."""

    def __init__(self, *, serving: bool) -> None:
        self._serving = serving

    def _status(self) -> "health_pb2.HealthCheckResponse.ServingStatus":
        return (
            health_pb2.HealthCheckResponse.SERVING
            if self._serving
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )

    async def Check(self, request, context):
        """The serving state of one service name.

        An unknown service name is ``NOT_SERVING`` rather than ``UNKNOWN``: the
        name is not one this server carries, which for a health probe is the same
        statement as not serving, and it keeps the protocol's three-value
        vocabulary.
        """
        if request.service not in (_OVERALL,):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"the health protocol carries no service named {request.service!r}",
            )
        return health_pb2.HealthCheckResponse(status=self._status())

    async def Watch(self, request, context):
        """The serving state, and the change to it, as the call waits.

        The gateway does not move between serving states on its own, so this
        reports the current state once and then holds the call open until the
        consumer cancels it — which is what a probe wants, and what an
        implementation that never sends a first status would leave waiting.
        """
        if request.service not in (_OVERALL,):
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"the health protocol carries no service named {request.service!r}",
            )
        yield health_pb2.HealthCheckResponse(status=self._status())
        while not context.is_cancelled():
            await asyncio.sleep(_POLL_SECONDS)
