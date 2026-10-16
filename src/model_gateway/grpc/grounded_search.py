"""The GroundedSearch RPC: one directive, one grounded answer with citations.

``GroundedSearch`` is the grounded web-search operation's entry point on the
gRPC transport, and this module is the whole of it: authenticate the call,
reconcile the tenant it claims, validate the request, translate it into the
canonical grounded search, hand that to the Exa adapter and translate the
answer back. No policy is decided here — the search, the citations and the
synthesis all run in the adapter, exactly as they would for any other caller.

Three rules shape the handler.

**The credential, not the body, is the tenant.** The call's identity comes from
:meth:`~model_gateway.grpc.identity.Authenticator.identity`, and the claimed
``tenant_id`` is reconciled against it by
:func:`~model_gateway.security.authorization.reconcile_identity` — the same
transport-independent policy the other RPCs apply. The grounded search needs a
tenant to attribute the egress to, so a request that states none is refused as
an invalid argument after the reconciliation, and one that states a tenant its
credential is not bound to is refused as a permission denial.

**A failure is aborted, never raised.** A failure the adapter raises carries a
stable reason token, and this handler maps the token onto the gRPC status the
governance chain declares for it — the same mapping the Go adapter's
``mapGroundedError`` performs — so a caller branches on the same reason
whichever transport it used. A failure that is not a grounded failure is
classified by the taxonomy's own normaliser, and the abort is awaited because
``grpc.aio``'s ``ServicerContext.abort`` is a coroutine.

**An empty citation list is answered, not refused.** A search that finds no
renderable citation is an in-band anomaly, not an error: the adapter returns it
with empty citations and the unspecified finish reason, and the handler passes
it through so the caller hedges rather than presenting web content as knowledge.
"""

from __future__ import annotations

import grpc
import structlog

from model_gateway.domain import GroundedSearchRequest
from model_gateway.errors import (
    FailedPreconditionError,
    GatewayError,
    InternalError,
    InvalidArgumentError,
    ResourceExhaustedError,
    UnavailableError,
)
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import mapping
from model_gateway.grpc.errors import abort
from model_gateway.grpc.identity import Authenticator
from model_gateway.providers.exa import ExaConfig, ExaGroundedSearch, GroundedError
from model_gateway.security.authorization import reconcile_identity

__all__ = ["GroundedSearch"]

logger = structlog.get_logger(__name__)

_REQUIRED_FIELDS: tuple[str, ...] = ("tenant_id", "gcid", "agent_id", "directive", "action_code")
"""The fields a grounded search states or is refused.

The tenant drives the egress attribution, the learner the search is attributed
to, the agent the calling skill, the directive the question itself and the
action code the metered egress the call is priced under. An un-priced egress is
not permitted, so a request missing any of them is an invalid argument naming
the field rather than a search that runs unaccounted.
"""


class GroundedSearch:
    """Serve ``gateway.v1.ModelGatewayService/GroundedSearch``.

    Built by the listener with the authenticator and the Exa configuration the
    deployment wired, so the handler owns no configuration: it authenticates
    through the one object that owns inbound authentication and searches through
    the one adapter that owns the operation.
    """

    def __init__(self, authenticator: Authenticator, config: ExaConfig) -> None:
        self._authenticator = authenticator
        self._adapter = ExaGroundedSearch(config)

    async def grounded_search(
        self, request: pb.GroundedSearchRequest, context: grpc.ServicerContext
    ) -> pb.GroundedSearchResponse:
        """Answer one grounded search, or abort the call with the gateway's own status.

        The order is the rule: the identity is established first, the claimed
        tenant is reconciled against it second, the request's required fields are
        validated third, and only then is the canonical search built and handed to
        the adapter. A request that fails authentication, states a tenant its
credential is not bound to, or is missing a required field never reaches the
adapter, so no part of such a request can influence the egress.
        """
        try:
            identity = self._authenticator.identity(context)
            reconcile_identity(identity, request.tenant_id or None)
            grounded_request = mapping.grounded_search_request_from_proto(request)
            _validate(grounded_request)
            result = await self._adapter.search(grounded_request)
            return mapping.grounded_search_response_to_proto(result)
        except GroundedError as error:
            # A grounded failure carries its own reason token, which the
            # governance chain already classified; the mapping reads it and
            # aborts with the status that reason declares, the same way the Go
            # adapter's mapGroundedError does.
            await abort(context, _as_gateway_error(error))
        except Exception as error:
            if not isinstance(error, GatewayError):
                logger.error(
                    "unhandled gRPC error",
                    error_type=type(error).__name__,
                    method="GroundedSearch",
                )
            await abort(context, error)


def _validate(request: GroundedSearchRequest) -> None:
    """Refuse a grounded search that states too little, naming the missing field."""
    for field in _REQUIRED_FIELDS:
        if not getattr(request, field):
            raise InvalidArgumentError(f"{field} is required")


_GROUNDED_CATEGORIES: dict[str, type[GatewayError]] = {
    "kill_switch_engaged": FailedPreconditionError,
    "external_egress_disabled": FailedPreconditionError,
    "budget_exhausted": FailedPreconditionError,
    "daily_ceiling_exceeded": ResourceExhaustedError,
    "invalid_argument": InvalidArgumentError,
    "vendor_unavailable": UnavailableError,
}
"""The gateway error category each grounded reason token maps to.

The egress preconditions (kill switch, egress disabled, budget exhausted) are
failed preconditions the caller must resolve; the daily ceiling is resource
exhaustion the caller retries later; a vendor failure is unavailability; and an
unrecognised reason — a gateway misconfiguration above all — is an internal
error rather than something the caller could act on.
"""


def _as_gateway_error(error: GroundedError) -> GatewayError:
    """The gateway error a grounded failure's reason token selects.

    The reason tokens are the governance chain's own, and each selects the
    category whose transport status the chain declares for it. A reason outside
    the table is an internal error: it names a failure the gateway cannot
    classify, which is a misconfiguration rather than a caller's remedy.
    """
    category = _GROUNDED_CATEGORIES.get(error.reason, InternalError)
    return category(error.detail or error.reason)
