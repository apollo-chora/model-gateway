"""The Invoke RPC: one request, one complete answer.

``Invoke`` is the gRPC transport's unary entry point for a complete answer, and
this module is the whole of it: authenticate the call, reconcile the tenant it
claims, translate the request into the canonical invocation the request's
``response_modality`` names, hand that to
:class:`~model_gateway.inference.InferenceService` and translate the answer
back. ``TEXT`` is the text operation, ``IMAGE`` the image operation, and
``MULTIMODAL`` is refused: the gateway serves text and image answers separately,
so a multimodal answer has no canonical representation here and approximating one
would invent an answer nobody asked for. No policy is decided here — model
resolution, capability enforcement, the deadline, retry, fallback and
cancellation all run in the inference service, exactly as they do for an HTTP
request, so a caller reaches the same behaviour over either transport.

Four rules shape the handler.

**The credential, not the body, is the tenant.** The call's identity comes from
:meth:`~model_gateway.grpc.identity.Authenticator.identity`, the claimed
``tenant_id`` is reconciled against it by
:func:`~model_gateway.security.authorization.reconcile_identity` — the same
transport-independent policy the HTTP layer calls — and the invocation is built
from the credential's tenant, never from the request field. An
empty ``tenant_id`` reads as no claim: the protobuf field is not ``optional``,
so an absent claim and an empty one are the same wire fact and neither is
evidence of anything.

**The invocation identifier is the caller's.** The contract carries
``invocation_id`` on the request and echoes it on the response, so the handler
neither mints one nor renames the one it was given: a caller that correlates
its own calls by that identifier would be misled by a value the gateway
substituted. A caller that states none is answered with none.

**A failure is aborted, never raised.** Every failure leaves through the
taxonomy's own status and already-sanitised payload, so a caller branches on the
same stable ``reason`` whichever transport it used. The body
catches ``Exception`` and never ``BaseException``: a caller that disconnects
cancels the call with :exc:`asyncio.CancelledError`, which is not a failure the
gateway answers with a status, and swallowing it would leave an abandoned
invocation running. A failure that is not a gateway error is recorded with its
type name and nothing else before it is aborted — the same record the HTTP
error handler keeps — because the aborted payload is sanitised and the type is
the only thing about such a failure an operator may see.

The abort is *awaited*, and that is a property of this transport rather than a
preference. ``grpc.aio``'s ``ServicerContext.abort`` is a coroutine, so a call
to it that is never awaited aborts nothing: the handler returns ``None``, the
server fails to serialise a response that is not one, and the caller reads
``UNKNOWN`` instead of the category's own status. This handler therefore reads
the two properties the taxonomy already carries — through
:func:`~model_gateway.grpc.errors.grpc_code` and
:func:`~model_gateway.grpc.errors.grpc_details`, the same helpers
:func:`~model_gateway.grpc.errors.abort` is built from — and awaits the abort
itself, so the status a caller sees is the one the category declares.

**An answer is translated, not judged.** The canonical answer is an ordered
sequence of output items, and the contract carries the same shapes — a content
part, which may be text, an image or a refusal, and a tool call — so an answer
carrying any of them crosses the boundary faithfully. This is deliberately not
the request side's rule: a *request* a route cannot honour is refused before any
provider is called, because the gateway cannot silently drop a tool or an image
the caller stated, whereas an *answer* is the provider's own and the gateway
has nothing to enforce about it beyond the capability contract
the request already passed. What the provider returned is what the caller
reads.
"""

from __future__ import annotations

import grpc
import structlog

from model_gateway.errors import GatewayError, InvalidArgumentError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import mapping
from model_gateway.grpc.errors import abort
from model_gateway.grpc.identity import Authenticator
from model_gateway.inference import InferenceService
from model_gateway.security.authorization import reconcile_identity
from model_gateway.security.identity import caller_namespace, caller_tenant

__all__ = ["Invoke"]

logger = structlog.get_logger(__name__)


class Invoke:
    """Serve ``gateway.v1.ModelGatewayService/Invoke``.

    Built by the listener with the authenticator and the inference service the
    deployment wired, so the handler owns no configuration: it authenticates
    through the one object that owns inbound authentication and serves the
    invocation through the one that owns policy.
    """

    def __init__(self, authenticator: Authenticator, service: InferenceService) -> None:
        self._authenticator = authenticator
        self._service = service

    async def invoke(
        self, request: pb.InvokeRequest, context: grpc.ServicerContext
    ) -> pb.InvokeResponse:
        """Answer one invocation, or abort the call with the gateway's own status.

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
            if request.response_modality == pb.RESPONSE_MODALITY_IMAGE:
                # The contract carries the modality on the one Invoke RPC, so
                # this entry point serves the operation the request declares
                # rather than assuming text: an image call answered by the text
                # operation is an answer to a request nobody made.
                image_request = mapping.image_request_from_proto(request, identity=identity)
                image_response = await self._service.generate(
                    image_request,
                    namespace=caller_namespace(identity),
                    tenant=caller_tenant(identity),
                )
                return mapping.image_response_to_proto(image_response)
            if request.response_modality == pb.RESPONSE_MODALITY_MULTIMODAL:
                raise InvalidArgumentError(
                    "the gateway serves text and image answers separately; a multimodal answer "
                    "has no canonical representation here, so the modality is refused rather "
                    "than approximated"
                )
            invocation = mapping.invocation_request_from_proto(request, identity=identity)
            response = await self._service.invoke(
                invocation,
                namespace=caller_namespace(identity),
                tenant=caller_tenant(identity),
            )
            return mapping.invoke_response_to_proto(response)
        except Exception as error:
            if not isinstance(error, GatewayError):
                logger.error(
                    "unhandled gRPC error",
                    error_type=type(error).__name__,
                    method="Invoke",
                )
            # The abort is awaited because grpc.aio's ServicerContext.abort is a
            # coroutine: a call to it that is not awaited never aborts anything,
            # and the server would answer UNKNOWN instead of the gateway's own
            # status. The status and the payload are the taxonomy's, read through
            # the same two helpers errors.abort is built from, so this publishes
            # exactly what abort(context, error) publishes on a transport whose
            # abort is synchronous.
            await abort(context, error)
