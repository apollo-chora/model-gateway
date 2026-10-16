"""The governance chain: one decision for one invoke, composed.

The six mechanisms — companion containment, the dispatch debit claim, the
tenant budget, the mana meter, the external-egress gate and the token-usage
outbox — are each their own module with its own rules. This module is the
composition the invocation flow runs: :meth:`GovernanceManager.authorize_invoke`
asks each mechanism in the order the platform's own governance pipeline
declares, and :meth:`GovernanceManager.settle_invoke` settles the ones a
served call drew down.

**The order is the platform's, restated.** Containment first, because a
contained turn must be refused before any debit is even possible. The debit
claim next, because a redelivered dispatch bills nothing and must not reach
the budget or the mana meter as if it were a first call. The budget third,
because an exhausted window refuses the call before any compute is spent. The
mana quote last among the gates, and only for a fail-closed action: a
shortfall on a premium-learner action refuses the call, while every other
action is served and charged best-effort afterwards.

**A denied call never reaches the provider.** The refusal carries the
mechanism's own reason token and detail, so the transport maps it to the
status the governance chain declares for that reason and a caller can branch
on the same token whichever transport it used.

**Settlement is atomic and best-effort.** The budget debit and the outbox
insert commit in one transaction, so a crash between them cannot record a
usage the budget was not debited for, or debit a budget for a usage that was
not recorded. The mana debit follows the commit and is best-effort: a charge
that cannot be recorded is logged, never raised, because the call has already
been served and an unrecorded charge is a bookkeeping outcome rather than a
reason to un-serve it.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from model_gateway.errors import (
    FailedPreconditionError,
    GatewayError,
    ResourceExhaustedError,
    UnavailableError,
)
from model_gateway.governance.budget import BudgetDecision, BudgetManager, decide_strictest
from model_gateway.governance.db import Database
from model_gateway.governance.egress import EgressGate
from model_gateway.governance.idempotency import IdempotencyClaims
from model_gateway.governance.mana import ManaDebitResult, ManaMeter, is_fail_closed
from model_gateway.governance.outbox import Outbox
from model_gateway.governance.suspension import SuspensionChecker

__all__ = [
    "COMPANION_SURFACES",
    "FAIL_CLOSED_ACTIONS",
    "FAIL_CLOSED_PREFIXES",
    "GovernanceDecision",
    "GovernanceManager",
    "governance_refusal",
    "is_companion_surface",
]

logger = structlog.get_logger(__name__)

COMPANION_SURFACES: frozenset[str] = frozenset({"companion_chat", "companion_diagnosis"})
"""The surfaces the companion containment tables govern (ADR-252 D1, ADR-254 D7).

A turn on one of these surfaces is a Learning Companion turn, and the platform
and tenant containment tables are read for it uncached on every call. A turn
on any other surface — or one that states no surface at all — is not a
companion turn and is never read against the containment tables.
"""

FAIL_CLOSED_ACTIONS: frozenset[str] = frozenset(
    {
        "knowledge_graph_traverse",
        "boss_challenge_atom_gen",
        "daily_dose_coach",
        "coach_session_full",
    }
)
"""The action codes that hard-block on a definitive mana shortfall.

A fail-closed action pre-flight a dry-run affordability check before any
compute is spent, because a zero-balance learner must not receive a premium
action for free. A shortfall is refused with the two balances as the caller's
upsell context.
"""

FAIL_CLOSED_PREFIXES: tuple[str, ...] = ("companion_chat", "familiar_chat")
"""The action-code prefixes that are fail-closed, matching :data:`FAIL_CLOSED_ACTIONS`.

The companion chat turns price under per-tier codes that carry the base name
as a prefix (``companion_chat_turn_standard`` and its siblings, and the
legacy ``familiar_chat_turn_*`` during the rename overlap), so the
classification matches by prefix for these two and exactly for the rest.
"""


def is_companion_surface(surface: str) -> bool:
    """Whether a stamped surface is a governed companion turn surface."""
    return surface in COMPANION_SURFACES


@dataclass(frozen=True)
class GovernanceDecision:
    """What the governance chain decided for one invoke.

    ``allowed`` is the whole answer for a caller that only needs the gate.
    ``reason`` and ``detail`` are the operator-facing half: ``reason`` is the
    refusing mechanism's own token — ``companion_suspended``,
    ``suspension_unreadable``, ``budget_exhausted``, ``insufficient_mana`` —
    and ``detail`` is that mechanism's own explanation. ``deduped`` marks an
    invocation whose dispatch key was already claimed: it is allowed through
    (the turn is served) but settles nothing, because the first delivery
    already paid for it.
    """

    allowed: bool
    reason: str
    detail: str
    deduped: bool = False


_REFUSAL_CATEGORIES: dict[str, type[GatewayError]] = {
    "companion_suspended": FailedPreconditionError,
    "budget_exhausted": FailedPreconditionError,
    "insufficient_mana": ResourceExhaustedError,
    # A store the gateway could not read is a gateway fault, not a verdict
    # the caller can resolve: it is unavailability, not a failed precondition.
    "suspension_unreadable": UnavailableError,
}
"""The gateway error category each denied governance reason token selects.

The containment and budget denials are failed preconditions the caller must
resolve; the mana shortfall is resource exhaustion the caller tops up; and a
containment store that could not be read is a gateway fault, mapped to
unavailability for the same reason the egress gate's own unreadable is.
"""


def governance_refusal(decision: GovernanceDecision) -> GatewayError:
    """The gateway error a denied governance decision selects.

    The reason tokens are the governance chain's own, and each selects the
    category whose transport status the chain declares for it. The mechanism's
    token rides in the error's details under ``reason``, so a caller branches
    on the same token whichever transport it used. A reason outside the table
    is a failed precondition: it names a refusal the gateway cannot further
    classify, which is safer than inventing a category for it.
    """
    category = _REFUSAL_CATEGORIES.get(decision.reason, FailedPreconditionError)
    return category(
        f"the invocation was refused by governance: {decision.reason}",
        details={"reason": decision.reason, "detail": decision.detail},
    )


class GovernanceManager:
    """The governance chain for one invoke: authorize before, settle after.

    The manager holds the six mechanisms and the database they share, so the
    invocation flow runs the chain without knowing how many mechanisms there
    are or in what order they run. The mana meter is optional: a deployment
    that has not wired the identity service's mana port runs the chain
    un-metered, exactly as it runs without a budget window.
    """

    def __init__(
        self,
        db: Database,
        budget: BudgetManager,
        outbox: Outbox,
        idempotency: IdempotencyClaims,
        suspension: SuspensionChecker,
        egress: EgressGate,
        mana: ManaMeter | None = None,
    ) -> None:
        self._db = db
        self._budget = budget
        self._outbox = outbox
        self._idempotency = idempotency
        self._suspension = suspension
        self._egress = egress
        self._mana = mana

    @property
    def budget(self) -> BudgetManager:
        """The tenant budget mechanism, for the grounded-egress gate's decision."""
        return self._budget

    @property
    def egress(self) -> EgressGate:
        """The external-egress gate, for the grounded-egress gate's kill switch."""
        return self._egress

    @property
    def mana(self) -> ManaMeter | None:
        """The mana meter, for the grounded-egress gate's fail-closed quote."""
        return self._mana

    async def authorize_invoke(
        self,
        tenant_id: str,
        gcid: str | None,
        action_code: str | None,
        dispatch_key: str | None,
        surface: str | None,
        invocation_id: str,
    ) -> GovernanceDecision:
        """Run the full governance chain for an Invoke request.

        Returns a :class:`GovernanceDecision` with ``allowed=True`` or
        ``allowed=False``; if ``allowed=False``, the decision carries the
        reason and detail, and the caller must refuse the invocation with
        :func:`governance_refusal` before any provider is called. A
        ``deduped=True`` decision is allowed through but settles nothing.

        The tenant-scoped mechanisms — containment and budget — run only
        when the request states a tenant: a tenant-less call is not a tenant
        turn, and the tables they read are keyed by tenant. The learner-scoped
        ones — the debit claim and the mana quote — run whenever the request
        states the learner, because they are keyed by GCID and no tenant.
        """
        # 1. Suspension check (companion surfaces only). A contained turn is
        # refused before any debit is even possible, so this is the outermost
        # gate; a non-companion surface passes without a read.
        if tenant_id and surface and is_companion_surface(surface):
            decision = await self._suspension.check(tenant_id, action_code or "")
            if not decision.allowed:
                return decision

        # 2. Idempotency claim. A redelivered dispatch finds its key already
        # claimed and is allowed through unbilled: the turn is served, but
        # the budget and the mana meter must not be charged a second time.
        if gcid and dispatch_key:
            claimed = await self._idempotency.claim(
                gcid, dispatch_key, action_code or "unspecified", invocation_id
            )
            if not claimed:
                return GovernanceDecision(
                    allowed=True,
                    reason="dispatch_deduped",
                    detail="the dispatch key was already claimed; this turn settles nothing",
                    deduped=True,
                )

        # 3. Budget decision. Every active window is read under one lock and
        # the strictest decision across them is the one honoured, so two
        # concurrent invocations of one tenant cannot both read the same
        # ``spent`` and both decide there is room.
        if tenant_id:
            async with self._budget.active_budgets(tenant_id) as windows:
                decision = decide_strictest(windows)
            if decision is BudgetDecision.BLOCK:
                return GovernanceDecision(
                    allowed=False,
                    reason="budget_exhausted",
                    detail="the tenant's active LLM budget window is exhausted "
                    "under its block policy",
                )

        # 4. Mana quote (fail-closed actions only). A definitive shortfall
        # refuses the call before any compute is spent; an unpriced action is
        # un-metered, and a meter that cannot be read fails open, so neither
        # denies here.
        if self._mana is not None and gcid and action_code and is_fail_closed(action_code):
            quote = await self._mana.quote(gcid, action_code)
            if not quote.affordable and not quote.unknown_action:
                return GovernanceDecision(
                    allowed=False,
                    reason="insufficient_mana",
                    detail=f"required={quote.required} available={quote.available} "
                    f"action={action_code}",
                )

        return GovernanceDecision(allowed=True, reason="", detail="")

    async def settle_invoke(
        self,
        tenant_id: str,
        gcid: str | None,
        action_code: str | None,
        dispatch_key: str | None,
        invocation_id: str,
        usage_id: str,
        cost_micros: int,
        payload: bytes,
        envelope: dict[str, str],
    ) -> None:
        """Settle a successful invocation: debit budget + insert outbox atomically.

        The two writes commit in one transaction, so a crash between them
        cannot record a usage the budget was not debited for, or debit a budget
        for a usage that was not recorded. A usage whose outbox row already
        exists — a redelivery of the same usage — settles nothing: the insert
        reports it did not land, and the learner is not charged twice for one
        recorded call.

        The mana debit follows the commit and is best-effort: a charge that
        cannot be recorded is logged, never raised, because the call has
        already been served. ``dispatch_key`` is carried onto the event
        envelope so the redelivery proof can read which dispatch the usage
        belonged to.
        """
        envelope = {**envelope, **({"dispatch_key": dispatch_key} if dispatch_key else {})}
        settled = await self._budget.debit_and_enqueue(
            tenant_id=tenant_id,
            usage_id=usage_id,
            gcid=gcid or "",
            cost_micros=cost_micros,
            payload=payload,
            envelope=envelope,
        )
        if not settled:
            logger.info(
                "usage already recorded; settling nothing",
                usage_id=usage_id,
                invocation_id=invocation_id,
            )
            return
        if self._mana is None or not gcid or not action_code:
            return
        debit = await self._mana.debit(gcid, action_code, invocation_id)
        if debit.result is not ManaDebitResult.OK:
            logger.warning(
                "mana debit not recorded after a served call",
                action_code=action_code,
                gcid=gcid,
                invocation_id=invocation_id,
                result=debit.result.value,
                error=debit.error,
            )
