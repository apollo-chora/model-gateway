"""The standard gRPC Health Checking Protocol (PRD section 5).

The protocol is the one load balancers, service meshes and orchestrators probe:
``Check`` answers one of three serving states for a service name, and ``Watch``
streams the state as it changes. The gateway registers the empty service name,
which is the server's overall health rather than any one RPC.

The servicer is the ``grpcio-health-checking`` package's own implementation
rather than a clone written here. A clone that answers the protocol's shapes
while diverging from its rules — an unknown service name refused rather than
answered ``SERVICE_UNKNOWN``, a ``Watch`` that ends instead of holding the call
open — is a probe that reads a healthy gateway as sick or a sick one as well,
and the divergence surfaces only when a deployment's probes disagree with the
gateway. The package's implementation is the protocol's reference, so the
gateway serves that.

The re-exported implementation is the package's AsyncIO one, because the
listener is an ``grpc.aio.server``: the synchronous implementation serves
``Watch`` from a blocking iterator that runs on gRPC's migration thread pool,
and when such a call ends the pool's close callback never fires on this
server, so its iterator stays blocked forever and the process cannot exit.
The AsyncIO implementation serves the same protocol on the event loop.

The protocol is served without authentication. It is registered on the listener
directly, outside the gateway's authenticator, which only the gateway's own RPC
handlers consult: a prober presents no API key and has none to present, and the
gateway answers health on the same port as the service it probes, so one
listener serves both.

The servicer holds no state of its own beyond the serving state the listener
reports. A gateway that has not started is not serving, one that has is, and
one that is shutting down is not — so the listener sets the overall service to
``NOT_SERVING`` before it starts, ``SERVING`` in ``start`` (which notifies any
watchers), and enters graceful shutdown in ``stop`` before stopping the server,
so a watcher is told the transition rather than left with a vanished connection.
"""

from grpc_health.v1._async import HealthServicer

__all__ = ["HealthServicer"]
