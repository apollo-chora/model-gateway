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

**The invocation identifier is resolved once, then carried everywhere.** A
caller may omit ``invocation_id`` — the contract does not require it — and the
handler mints one with :func:`uuid.uuid4` rather than leaving it blank, because
the settlement the search owes is keyed by that identifier: the outbox insert
is the idempotency guard, and a blank key would collapse every event onto one
row. The resolved value is the one the canonical request, the adapter result,
the budget debit, the outbox row, the mana idempotency key and the response all
carry, so a redelivery of one search settles nothing.

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

**A wired governance chain gates the search before the egress and settles it
after.** When the deployment built the handler with a
:class:`~model_gateway.governance.manager.GovernanceManager`, the search runs
the same chain an invoke does — the egress gate, the budget decision and the
mana quote for a fail-closed action — before the Exa call, and any refusal
aborts with the status the reason token declares. A search that is served then
settles the same way an invoke does: the egress counter is incremented, the
budget debit and the token-usage outbox record commit in one transaction, and
the mana debit follows inside it. A search that fails after billable work —
the Exa search succeeded and the synthesis failed — settles the partial spend
only: the budget debit and the outbox record, with no mana debit and no egress
counter, because the call never completed. A deployment that wired no
governance manager gates and settles nothing.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import grpc
import structlog

from model_gateway.domain import GroundedSearchRequest, GroundedSearchResult
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
from model_gateway.governance.outbox import build_token_usage_payload, token_usage_envelope
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
        tenant is reconciled against it second, the invocation identifier is
        resolved third — minted when the caller omitted it — the request's
        required fields are validated fourth, and only then is the canonical
        search built and handed to the adapter. A request that fails
        authentication, states a tenant its credential is not bound to, or is
        missing a required field never reaches the adapter, so no part of such a
        request can influence the egress.
        """
        try:
            identity = self._authenticator.identity(context)
            reconcile_identity(identity, request.tenant_id or None)
            # The invocation id is resolved once, here, and that one value is
            # what the request, the result and every settlement write carry: a
            # caller that omits it gets a minted one, and a caller that states
            # one keeps it, so the outbox's idempotency key is never blank.
            invocation_id = request.invocation_id or str(uuid.uuid4())
            grounded_request = replace(
                mapping.grounded_search_request_from_proto(request), invocation_id=invocation_id
            )
            _validate(grounded_request)
            if self._governance is not None:
                await _authorize_grounded(self._governance, grounded_request)
            result = await self._adapter.search(grounded_request)
            if self._governance is not None:
                await _settle_grounded(self._governance, grounded_request, result)
            return mapping.grounded_search_response_to_proto(
                replace(result, invocation_id=invocation_id)
            )
        except GroundedError as error:
            # A grounded failure carries its own reason token, which the
            # governance chain already classified; the mapping reads it and
            # aborts with the status that reason declares, the same way the Go
            # adapter's mapGroundedError does. A failure after billable work
            # settles the partial spend first — the provider charged for the
            # search, and the tenant's ledger must account it — then the
            # vendor's own error reaches the caller unchanged.
            if self._governance is not None and error.usage is not None:
                await _settle_grounded_partial(self._governance, grounded_request, error)
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
        quote = await governance.mana.quote(request.gcid, request.tenant_id, request.action_code)
        if not quote.affordable and not quote.unknown_action:
            raise GroundedError(
                "insufficient_mana",
                f"required={quote.required} available={quote.available} "
                f"action={request.action_code}",
            )


async def _settle_grounded(
    governance: GovernanceManager,
    request: GroundedSearchRequest,
    result: GroundedSearchResult,
) -> None:
    """Settle one served grounded search against the governance ledger, best-effort.

    A search that produced an answer is a call the gateway owes for, so it
    settles the way an invoke does: the egress counter is incremented first,
    then the budget debit and the token-usage outbox record commit in one
    transaction — with the mana debit following inside it. The usage id is the
    invocation id, so a redelivered search settles nothing: the outbox insert
    reports the row already exists and the learner is not charged twice.

    Every step is best-effort: a failure is logged, never raised, because the
    answer is already in the caller's hands and a ledger the operator
    misconfigured is not a reason to withhold it — the same rule the invoke
    settlement follows. The counter and the invoke settlement are settled
    independently: a counter that cannot be recorded does not stop the budget
    and the usage from being, and the reverse holds too.
    """
    try:
        await governance.egress.record_egress(request.tenant_id)
    except Exception as error:
        logger.warning(
            "grounded egress counter not recorded",
            error_type=type(error).__name__,
            invocation_id=request.invocation_id,
        )
    usage = result.usage
    occurred_at = datetime.now(UTC)
    usage_id = request.invocation_id
    try:
        await governance.settle_invoke(
            tenant_id=request.tenant_id,
            gcid=request.gcid,
            action_code=request.action_code,
            dispatch_key=None,
            invocation_id=request.invocation_id,
            usage_id=usage_id,
            cost_micros=usage.cost_micros,
            payload=build_token_usage_payload(
                usage_id=usage_id,
                tenant_id=request.tenant_id,
                gcid=request.gcid,
                # The synthesis leg ran the adapter's configured model, and
                # the result names the version that produced the tokens: the
                # event records the model that actually served the call.
                model_id=result.model_version,
                invocation_id=request.invocation_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_micros=usage.cost_micros,
                action_code=request.action_code,
                agent_role=request.agent_id,
                occurred_at=occurred_at,
            ),
            envelope=token_usage_envelope(
                usage_id=usage_id,
                tenant_id=request.tenant_id,
                gcid=request.gcid,
                occurred_at=occurred_at,
                traceparent=request.traceparent,
                tracestate=request.tracestate,
            ),
        )
    except Exception as error:
        logger.error(
            "grounded settlement failed",
            error_type=type(error).__name__,
            invocation_id=request.invocation_id,
        )


async def _settle_grounded_partial(
    governance: GovernanceManager,
    request: GroundedSearchRequest,
    error: GroundedError,
) -> None:
    """Settle the partial spend of a failed grounded search, best-effort.

    When the Exa search succeeded and billed but the synthesis failed, the
    provider has charged for work the gateway owes for, and the usage the
    error carries is the adapter's own record of it. The settlement is the
    atomic pair a served search writes — the budget debit and the token-usage
    outbox row in one transaction, under the invocation id so a redelivery of
    the same failed search settles nothing — and nothing else.

    What a failed call does not earn is withheld: there is no mana debit,
    because the call never completed and the learner's wallet is not charged
    for an answer that was never produced, and the egress counter is not
    incremented, because the search returned no answer to count against the
    tenant's daily ceiling. The vendor's own error is untouched: it is still
    the caller's answer, mapped onto the status its reason token declares.

    Every step is best-effort: a settlement failure is logged, never raised,
    for the same reason the served settlement's is — a ledger the operator
    misconfigured is not a reason to withhold the error.
    """
    usage = error.usage
    if usage is None:
        return
    occurred_at = datetime.now(UTC)
    usage_id = request.invocation_id
    try:
        await governance.budget.debit_and_enqueue(
            tenant_id=request.tenant_id,
            usage_id=usage_id,
            gcid=request.gcid,
            cost_micros=usage.cost_micros,
            payload=build_token_usage_payload(
                usage_id=usage_id,
                tenant_id=request.tenant_id,
                gcid=request.gcid,
                # The synthesis never completed, so no model produced tokens:
                # the event records the search's own spend under no model.
                model_id="",
                invocation_id=request.invocation_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_micros=usage.cost_micros,
                action_code=request.action_code,
                agent_role=request.agent_id,
                occurred_at=occurred_at,
            ),
            envelope=token_usage_envelope(
                usage_id=usage_id,
                tenant_id=request.tenant_id,
                gcid=request.gcid,
                occurred_at=occurred_at,
                traceparent=request.traceparent,
                tracestate=request.tracestate,
            ),
        )
    except Exception as settle_error:
        logger.error(
            "grounded partial settlement failed",
            error_type=type(settle_error).__name__,
            invocation_id=request.invocation_id,
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
