"""The InvokeStream RPC: one request, a stream of chunks.

A streamed invocation is the same invocation as ``Invoke``, under the same policy:
the model is resolved first, the route's capability contract is enforced, one
deadline covers the whole stream and a concurrency slot is held for its duration.
None of that lives here. The inference service produces canonical events and this
module owns only the transport's half of the stream contract — how each event
reaches a caller, and what ends the stream.

Five rules shape the handler.

**The credential, not the body, is the tenant.** The call's identity comes from
:meth:`~model_gateway.grpc.identity.Authenticator.identity`, the claimed
``tenant_id`` is reconciled against it by
:func:`~model_gateway.security.authorization.reconcile_identity` — the same
transport-independent policy the HTTP layer calls — and the invocation is built
from the credential's tenant, never from the request field. An empty
``tenant_id`` reads as no claim, for the reason the Invoke handler gives: the
protobuf field is not ``optional``, so an absent claim and an empty
one are the same wire fact and neither is evidence of anything.

**The service's events are translated, never re-decided.** Every
:class:`~model_gateway.domain.streaming.StreamEvent` becomes one
:class:`~model_gateway.grpc.gateway_pb2.InvokeChunk` through
:func:`~model_gateway.grpc.mapping.chunk_from_event`, and what the caller reads
is exactly what the service produced: the gateway numbers its own events and
never takes a number from upstream, so a caller reads the same sequence, the
same invocation identifier and the same terminal event the service emitted.

**A refusal before the first event is an error response, not a stream.** An
unknown model, a capability gap, a provider that cannot stream, an expired
deadline — each ends the call with the gateway error's own gRPC status and with
no chunk at all. The caller has seen nothing, so an error status is the only
honest answer to a request the gateway could not serve, and a chunk published
beside it would promise a stream the invocation never produced.

**A failure after the first event ends the stream with one terminal chunk.** The
inference service reports such a failure as a terminal ``ERROR`` event and then
raises it, and the event is the authoritative report: the exception is
control-flow metadata that ends the iteration, never a second failure. This
handler therefore ends the stream at the raise and yields nothing further —
re-raising would report the one failure twice, and a second
error chunk would contradict the one the caller was already shown. A failure
outside the taxonomy is not a gateway failure at all: it is aborted as the
internal error it is rather than published as an event, and the provider's own
text reaches neither the caller nor the status.

**A caller that goes away stops the upstream work.** A cancelled call breaks the
iteration, emits no terminal event and closes the service's stream, so a
consumer that loses its connection reads a truncated stream — one that stopped
without a terminal event — and the in-flight upstream attempt is stopped rather
than spent on output nobody is left to receive. Cancellation is read from
``context.cancelled()``, which is the gRPC context's own answer to whether the
call is still live, and the handler polls it rather than discovering a
disconnect from a raised exception.

The abort is *awaited*, for the reason the Invoke handler gives: ``grpc.aio``'s
``ServicerContext.abort`` is a coroutine, so a call to it that is never awaited
aborts nothing and the caller reads ``UNKNOWN`` instead of the taxonomy's own
status. The status and the payload are read through the same two helpers
:func:`~model_gateway.grpc.errors.abort` is built from.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc
import structlog

from model_gateway.errors import GatewayError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc.errors import abort
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.mapping import chunk_from_event, invocation_request_from_proto
from model_gateway.inference import InferenceService
from model_gateway.security.authorization import reconcile_identity
from model_gateway.security.identity import caller_namespace, caller_tenant

__all__ = ["InvokeStream"]

logger = structlog.get_logger(__name__)


class InvokeStream:
    """Serve one InvokeStream RPC as the chunks its invocation streams."""

    def __init__(self, authenticator: Authenticator, service: InferenceService) -> None:
        self._authenticator = authenticator
        self._service = service

    async def invoke_stream(
        self, request: pb.InvokeRequest, context: grpc.ServicerContext
    ) -> AsyncIterator[pb.InvokeChunk]:
        """Stream the answer to one invocation, or refuse it.

        The order is the rule: the identity is established first, the claimed
        tenant is reconciled against it second, and only then is the canonical
        invocation built — from the credential's tenant. A request that fails
        authentication or claims a tenant its credential is not bound to never
        reaches the mapper, so no part of such a request can influence what the
        invocation says about who is calling.
        """
        try:
            identity = self._authenticator.identity(context)
            reconcile_identity(identity, request.tenant_id or None)
            invocation = invocation_request_from_proto(request, identity=identity)
        except Exception as error:
            await _abort(context, error, method="InvokeStream")

        stream = self._service.invoke_stream(
            invocation,
            namespace=caller_namespace(identity),
            tenant=caller_tenant(identity),
        )
        delivered = 0
        try:
            async for event in stream:
                if context.cancelled():
                    # The caller has gone away, so no terminal event is emitted —
                    # the stream simply ends and reads as truncated — and the
                    # upstream attempt is stopped rather than spent on output
                    # nobody is left to receive.
                    break
                yield chunk_from_event(event)
                delivered += 1
        except GatewayError as error:
            if delivered == 0:
                # The refusal reached the caller before the stream produced
                # anything, so it is an ordinary error response rather than a
                # stream.
                await _abort(context, error, method="InvokeStream")
            else:
                # The service has already emitted the terminal ERROR event and
                # the caller has seen it, so this raise ends the iteration. The
                # event is the failure's report; the exception is the control
                # flow that ends the stream, and re-raising it would report one
                # failure twice.
                logger.debug(
                    "invocation stream ended on a gateway failure",
                    reason=error.reason.value,
                )
                return
        except Exception as error:
            await _abort(context, error, method="InvokeStream")
        finally:
            # Close the service's stream on every exit — a completed stream, a
            # gateway failure, a refusal or a caller that went away — so the
            # upstream work behind it stops rather than outliving the call.
            await stream.aclose()


async def _abort(context: grpc.ServicerContext, error: Exception, *, method: str) -> None:
    """End the call with the taxonomy's own status and sanitised payload.

    ``Exception`` is caught and never ``BaseException``: a caller that
    disconnects cancels the call with :exc:`asyncio.CancelledError`, which is
    not a failure the gateway answers with a status, and swallowing it would
    leave an abandoned invocation running. A failure that is not a gateway error
    is recorded with its type name and nothing else before it is aborted — the
    same record the HTTP error handler keeps — because the aborted payload is
    sanitised and the type is the only thing about such a failure an operator
    may see.

    The abort is awaited because grpc.aio's ``ServicerContext.abort`` is a
    coroutine: a call to it that is not awaited never aborts anything, and the
    server answers ``UNKNOWN`` instead of the gateway's own status. The status
    and the payload are the taxonomy's, read through the same two helpers
    ``errors.abort`` is built from, so this publishes exactly what
    ``abort(context, error)`` publishes on a transport whose abort is
    synchronous.
    """
    if not isinstance(error, GatewayError):
        logger.error(
            "unhandled gRPC error",
            error_type=type(error).__name__,
            method=method,
        )
    await abort(context, error)
