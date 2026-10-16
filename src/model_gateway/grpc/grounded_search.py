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

**A wired governance chain gates the search before the egress.** When the
deployment built the handler with a
:class:`~model_gateway.governance.manager.GovernanceManager`, the search runs
the same chain an invoke does — the egress gate, the budget decision and the
mana quote for a fail-closed action — before the Exa call, and any refusal
aborts with the status the reason token declares. A deployment that wired no
governance manager gates nothing.
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
from model_gateway.governance.budget import BudgetDecision, decide_strictest
from model_gateway.governance.mana import is_fail_closed
from model_gateway.governance.manager import GovernanceManager
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
    the one adapter that owns the operation. The optional governance manager is
    the chain the search runs through before the egress, when the deployment
    wired one.
    """

    def __init__(
        self,
        authenticator: Authenticator,
        config: ExaConfig,
        governance: GovernanceManager | None = None,
    ) -> None:
        self._authenticator = authenticator
        self._adapter = ExaGroundedSearch(config)
        self._governance = governance

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
            if self._governance is not None:
                await _authorize_grounded(self._governance, grounded_request)
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


async def _authorize_grounded(
    governance: GovernanceManager, request: GroundedSearchRequest
) -> None:
    """Run the governance chain for one grounded search, or raise on a refusal.

    Grounded search is the gateway's one controlled egress surface, so it is
    gated by the same chain an invoke runs, in the same order: the egress gate
    first — the kill switch, the entitlement and the daily ceiling, read
    uncached — then the tenant budget decision, then the mana quote for a
    fail-closed action. Any refusal raises a :class:`GroundedError` carrying
    the refusing mechanism's own reason token, which the handler maps onto the
    gRPC status the governance chain declares for it, so a grounded denial
    reaches the caller as the same status the Exa adapter's own denials do.
    A deployment that wired no governance manager gates nothing here.
    """
    egress = await governance.egress.authorize(request.tenant_id)
    if not egress.allowed:
        raise GroundedError(egress.reason, egress.detail)
    async with governance.budget.active_budgets(request.tenant_id) as windows:
        decision = decide_strictest(windows)
    if decision is BudgetDecision.BLOCK:
        raise GroundedError(
            "budget_exhausted",
            "the tenant's active LLM budget window is exhausted under its block policy",
        )
    if governance.mana is not None and request.action_code and is_fail_closed(request.action_code):
        quote = await governance.mana.quote(request.gcid, request.action_code)
        if not quote.affordable and not quote.unknown_action:
            raise GroundedError(
                "insufficient_mana",
                f"required={quote.required} available={quote.available} "
                f"action={request.action_code}",
            )


_GROUNDED_CATEGORIES: dict[str, type[GatewayError]] = {
    "kill_switch_engaged": FailedPreconditionError,
    "external_egress_disabled": FailedPreconditionError,
    "budget_exhausted": FailedPreconditionError,
    "daily_ceiling_exceeded": ResourceExhaustedError,
    "egress_unreadable": UnavailableError,
    "insufficient_mana": ResourceExhaustedError,
    "invalid_argument": InvalidArgumentError,
    "vendor_unavailable": UnavailableError,
}
"""The gateway error category each grounded reason token maps to.

The egress preconditions (kill switch, egress disabled, budget exhausted) are
failed preconditions the caller must resolve; the daily ceiling is resource
exhaustion the caller retries later; a mana shortfall is resource exhaustion
the caller tops up; an egress store the gateway could not read is a gateway
fault rather than a verdict, so it is unavailability like a vendor failure; and
an unrecognised reason — a gateway misconfiguration above all — is an internal
error rather than something the caller could act on.
"""


def _as_gateway_error(error: GroundedError) -> GatewayError:
    """The gateway error a grounded failure's reason token selects.

    The reason tokens are the governance chain's own, and each selects the
    category whose transport status the chain declares for it. The message
    leads with the reason token, the same way the Go adapter's
    ``mapGroundedError`` does, so a caller can prefix-match the denial reason
    off the status without a structured field on the wire.

    When the failure carries partial spend — the Exa search billed but the
    synthesis failed — the usage is included in the error details so the
    caller can account the cost the provider already charged.
    """
    category = _GROUNDED_CATEGORIES.get(error.reason, InternalError)
    message = f"{error.reason}: {error.detail}" if error.detail else error.reason
    details = None
    if error.usage is not None:
        details = {
            "partial_grounding_units": str(error.usage.grounding_units),
            "partial_cost_micros": str(error.usage.cost_micros),
        }
    return category(message, details=details)
