"""The Embed RPC: one request, one list of vectors (PRD sections 4.7 and 5).

``Embed`` is the embeddings operation's entry point on the gRPC transport, and
this module is the whole of it: authenticate the call, translate the request into
the canonical embeddings request, hand that to
:class:`~model_gateway.inference.InferenceService` and translate the answer back.
No policy is decided here — model resolution, capability enforcement, the
deadline, retry, fallback, the input ordering and the dimensionality check all
run in the inference service, exactly as they do for an HTTP request to
``POST /v1/embeddings``, so a caller reaches the same behaviour over either
transport (PRD sections 4.7 and 5.2).

Three rules shape the handler.

**The credential is the tenant, and this request claims none.** The call's
identity comes from
:meth:`~model_gateway.grpc.identity.Authenticator.identity`, so a call that
presents no credential is refused when the deployment requires one. What the call
cannot do is claim a tenant: neither the protobuf ``EmbedRequest`` nor the
canonical :class:`~model_gateway.domain.embeddings.EmbeddingRequest` has a tenant
field, and the canonical request is built from the request alone because there is
nothing about identity for it to carry. The reconciliation of PRD section 5.1 is
therefore stated with the absent claim rather than left out, so the rule every
RPC handler applies is visible here too — and a tenant can no more come from an
embeddings body than from a text one.

**A failure is aborted, never raised.** Every failure leaves through the call's
own ``abort``, carrying the category's gRPC status and its already-sanitised
payload, so a caller branches on the same stable ``reason`` whichever transport it
used (PRD section 5.4). The status and the payload are the taxonomy's, read
through the same two helpers :func:`~model_gateway.grpc.errors.grpc_code` and
:func:`~model_gateway.grpc.errors.grpc_details` that
:func:`~model_gateway.grpc.errors.abort` is built from, and the abort is awaited
because ``grpc.aio``'s ``ServicerContext.abort`` is a coroutine: a call to it that
is not awaited aborts nothing, and the server would answer ``UNKNOWN`` instead of
the gateway's own status. The body catches ``Exception`` and never
``BaseException``: a caller that disconnects cancels the call with
:exc:`asyncio.CancelledError`, which is not a failure the gateway answers with a
status. A failure that is not a gateway error is recorded with its type name and
nothing else before it is aborted — the same record the HTTP error handler keeps —
because the aborted payload is sanitised and the type is the only thing about such
a failure an operator may see (PRD section 14).

**An answer is translated, not judged.** The canonical vectors are floats, one
per input text, placed at the position the caller's input holds, and the answer
reports the public model that was asked for, the upstream model that served it
and the invocation identifier the gateway minted (PRD section 4.7). A request
asking for a wire encoding the gateway cannot answer in is refused by the gateway
core with a capability error naming it, never here: the handler holds no mapping
logic and no policy, and it never answers such a caller in floats (PRD section
9.5).
"""

from __future__ import annotations

import grpc
import structlog

from model_gateway.errors import GatewayError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import mapping
from model_gateway.grpc.errors import abort
from model_gateway.grpc.identity import Authenticator
from model_gateway.inference import InferenceService
from model_gateway.security.authorization import reconcile_identity
from model_gateway.security.identity import caller_namespace, caller_tenant

__all__ = ["Embed"]

logger = structlog.get_logger(__name__)


class Embed:
    """Serve ``gateway.v1.ModelGatewayService/Embed``.

    Built by the listener with the authenticator and the inference service the
    deployment wired, so the handler owns no configuration: it authenticates
    through the one object that owns inbound authentication and serves the
    embeddings request through the one that owns policy.
    """

    def __init__(self, authenticator: Authenticator, service: InferenceService) -> None:
        self._authenticator = authenticator
        self._service = service

    async def embed(
        self, request: pb.EmbedRequest, context: grpc.ServicerContext
    ) -> pb.EmbedResponse:
        """Answer one embeddings call, or abort the call with the gateway's own status.

        The order is the rule: the identity is established first, the tenant the
        call states — none, for this operation — is reconciled against it second,
        and only then is the canonical request built. The mapper takes no
        identity, because the canonical embeddings request has no tenant to
        carry: the tenant an invocation acts under is the credential's, and an
        embeddings request never states one (PRD section 5.1).
        """
        try:
            identity = self._authenticator.identity(context)
            reconcile_identity(identity, None)
            embedding_request = mapping.embed_request_from_proto(request)
            response = await self._service.embed(
                embedding_request,
                namespace=caller_namespace(identity),
                tenant=caller_tenant(identity),
            )
            return mapping.embedding_response_to_proto(response)
        except Exception as error:
            if not isinstance(error, GatewayError):
                logger.error(
                    "unhandled gRPC error",
                    error_type=type(error).__name__,
                    method="Embed",
                )
            # The abort is awaited because grpc.aio's ServicerContext.abort is a
            # coroutine: a call to it that is not awaited never aborts anything,
            # and the server would answer UNKNOWN instead of the gateway's own
            # status. The status and the payload are the taxonomy's, read through
            # the same two helpers errors.abort is built from, so this publishes
            # exactly what abort(context, error) publishes on a transport whose
            # abort is synchronous.
            await abort(context, error)
