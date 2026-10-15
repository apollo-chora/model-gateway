"""The ``ListModels`` RPC: the gateway's model listing over gRPC.

The listing is the registry's, not the inference service's. The registry is the
only thing that maps a public model ID onto the routes configured for its
operation, and a :class:`~model_gateway.registry.ModelInfo` is the whole of what
that mapping publishes: an ID, the operation it names, the capabilities its
primary route declares and the limits the deployment configured. This RPC
therefore reads the registry directly — there is no invocation to build, no route
to resolve and no attempt to make — and renders the result through
:func:`~model_gateway.grpc.mapping.list_models_response_to_proto`, so the gRPC
listing and ``GET /v1/models`` are two renderings of one value rather than two
answers that could drift apart.

**What the listing carries, and what it cannot.** A ``ModelInfo`` holds no route,
so no upstream endpoint, credential, upstream model ID or extra header can reach
a caller through this RPC: that is a property of the type rather than a matter of
remembering to strip fields, exactly as it is on the HTTP path. The one thing
logged here is the set of public model IDs, which is public by definition.

**The listing is authenticated when the deployment authenticates.** Every
endpoint but the health endpoints requires a valid credential, and
``ListModels`` is not one of them, so the call is authenticated before any
configured state is read. The request carries no identity claim of its own —
``ListModelsRequest`` is empty — so there is no claimed tenant to reconcile here;
the tenant rule is what the RPCs carrying an invocation apply, and this one has
nothing a caller could claim a tenant with.

**A failure is a gateway error, not an exception.** A refused credential is
aborted with the taxonomy's own status and sanitised payload rather than answered
as a gRPC ``UNKNOWN``, and a failure raised by the registry or
the mapping is classified the same way, so a caller reads a gateway reason
whichever transport it used.
"""

from __future__ import annotations

import grpc
import structlog

from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc.errors import abort
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.mapping import list_models_response_to_proto
from model_gateway.registry import ModelRegistry

__all__ = ["ListModels"]

logger = structlog.get_logger(__name__)


class ListModels:
    """The model listing, read from the registry.

    It holds the authenticator and the registry and nothing else. The listing is
    a read of configured state rather than an inference, so there is no service
    to call and nothing to hold between calls.
    """

    def __init__(self, authenticator: Authenticator, registry: ModelRegistry) -> None:
        self._authenticator = authenticator
        self._registry = registry

    async def list_models(
        self, request: pb.ListModelsRequest, context: grpc.ServicerContext
    ) -> pb.ListModelsResponse:
        """Answer one ``ListModels`` call.

        The call is authenticated first, so a call that presents no credential
        is refused before any configured state is read when the deployment
        requires one. The registry is then read and its entries rendered in the
        registry's own canonical operation order, which is the order the HTTP
        listing uses too: one entry per configured operation, and no entry for an
        operation the deployment configured no route for.
        """
        try:
            self._authenticator.identity(context)
            infos = self._registry.list_models()
            response = list_models_response_to_proto(infos)
        except Exception as error:
            # CancelledError is a BaseException, so a caller that disconnects is
            # never caught here and never turned into a status. The status itself
            # is aborted with the taxonomy's own code and sanitised payload;
            # grpc.aio's ServicerContext.abort is a coroutine, so it is awaited
            # rather than called through the synchronous errors.abort helper.
            await abort(context, error)
            raise
        logger.info(
            "model-gateway gRPC model listing served",
            models=[info.id for info in infos],
        )
        return response
