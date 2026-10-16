"""Per-tenant LLM budget admission and settlement.

Every invocation a tenant makes draws down a windowed USD cap the operator
configures in ``per_tenant_llm_budget``. This module is the gateway's half of
that contract: it reads the active windows, decides what they allow, and
settles the spend against them.

**The read locks every active window row, across the decision.**
:meth:`BudgetManager.active_budgets` selects every window that currently
encloses ``now()`` with ``FOR UPDATE`` and holds the lock for the caller's
whole block, so two concurrent invocations of one tenant cannot both read the
same ``spent`` and both decide there is room: the second waits for the first's
settlement and then reads the incremented value. A tenant may carry more than
one active window — overlapping periods, or a window that opened before the
current one closed — so :func:`decide_strictest` applies the strictest
decision across all of them: a block anywhere refuses the call, a downgrade
anywhere re-routes it, and an alert anywhere reports it. The lock is held only
for the caller's transaction, which is what bounds contention even when one
tenant fires concurrent calls.

**The single-window read is the most recent, for callers that want it.**
:meth:`BudgetManager.read_active_budget` returns the most recent active
window over the same locked read; it holds the lock only for the read itself,
so a caller that decides outside it does so without the lock the decision
needs — :meth:`active_budgets` is the form that holds it.

**The decision is a pure function of the windows.** :func:`decide` takes one
window, or its absence, and nothing else, so the policy — allow, downgrade,
block, alert — can be checked against every shape a window can take without a
database; :func:`decide_strictest` lifts that to many. A tenant with no
active window is allowed through: the budget is an operator opt-in, and
refusing a tenant that was never configured would fail closed on a deployment
choice the gateway does not own. ``budget_required`` exists for the
deployment that does want the closed behaviour.

**Settlement tolerates a missing window.** :meth:`BudgetManager.debit` updates
the active window and accepts a zero-row result: the window may have closed
between the decision and the settlement, and charging a tenant against a window
that no longer exists would attach a bookkeeping error to no row. Money is
integer micros (``BIGINT``) end to end; nothing here converts to or from a
float.

**Settlement is atomic with the usage record.**
:meth:`BudgetManager.debit_and_enqueue` writes the token-usage outbox row and
debits the budget in one transaction, so a crash between them cannot record a
usage the budget was not debited for, or debit a budget for a usage that was
not recorded. The outbox insert is the idempotency guard: a usage whose row
already exists settles nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from model_gateway.domain.serialization import as_int
from model_gateway.governance.db import Database, rows_affected, validate_uuid
from model_gateway.governance.outbox import INSERT_TOKEN_USAGE, validate_usage_event

__all__ = [
    "BudgetDecision",
    "BudgetManager",
    "BudgetPolicy",
    "BudgetWindow",
    "decide",
    "decide_strictest",
]

_SELECT_ACTIVE_BUDGETS = """
    SELECT budget_id, budget_usd_micros, spent_usd_micros, policy, downgrade_to_logical_model_id
    FROM per_tenant_llm_budget
    WHERE tenant_id = $1::uuid
      AND budget_period_start <= now()
      AND budget_period_end > now()
    ORDER BY budget_period_start DESC
    FOR UPDATE
"""
"""Every active window for the tenant, locked ``FOR UPDATE`` for the block's duration."""

_DEBIT_ACTIVE_BUDGET = """
    UPDATE per_tenant_llm_budget
    SET spent_usd_micros = spent_usd_micros + $2, updated_at = now()
    WHERE tenant_id = $1::uuid
      AND budget_period_start <= now()
      AND budget_period_end > now()
"""


class BudgetPolicy(StrEnum):
    """The enforcement policy an exhausted window carries (migration 0008).

    ``DOWNGRADE`` requires ``downgrade_to_logical_model_id`` to name the model
    the call re-routes to; the database enforces that invariant, and the
    absence of a fallback for the other two policies is what their definition
    is.
    """

    BLOCK = "block"
    DOWNGRADE = "downgrade"
    ALERT = "alert"


class BudgetDecision(StrEnum):
    """What the gateway may do with one invocation against one window.

    ``ALLOW`` serves the resolved model; ``DOWNGRADE`` serves the window's
    cheaper fallback and records the downgrade; ``BLOCK`` refuses the call
    before any vendor dispatch; ``ALERT`` lets the call through and reports
    the breach for the dashboard.
    """

    ALLOW = "allow"
    DOWNGRADE = "downgrade"
    BLOCK = "block"
    ALERT = "alert"


_STRICTNESS: dict[BudgetDecision, int] = {
    BudgetDecision.ALLOW: 0,
    BudgetDecision.ALERT: 1,
    BudgetDecision.DOWNGRADE: 2,
    BudgetDecision.BLOCK: 3,
}
"""How strictly each decision constrains the caller, for the strictest-of-many fold."""


@dataclass(frozen=True)
class BudgetWindow:
    """One active budget row, as the gateway reads it.

    ``downgrade_to_logical_model_id`` is ``None`` unless ``policy`` is
    ``DOWNGRADE``: the database's check constraint makes a downgrade policy
    without a fallback unrepresentable, and the type states that rather than
    trusting it. Both amounts are integer micros of USD, validated to the
    bounds the database's own ``CHECK`` constraints declare — a positive cap
    and a non-negative spend.
    """

    budget_id: uuid.UUID
    budget_usd_micros: int
    spent_usd_micros: int
    policy: BudgetPolicy
    downgrade_to_logical_model_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.budget_id, uuid.UUID):
            raise ValueError(f"a budget window 'budget_id' must be a UUID, got {self.budget_id!r}")
        object.__setattr__(
            self,
            "budget_usd_micros",
            as_int("budget window 'budget_usd_micros'", self.budget_usd_micros, minimum=1),
        )
        object.__setattr__(
            self,
            "spent_usd_micros",
            as_int("budget window 'spent_usd_micros'", self.spent_usd_micros, minimum=0),
        )


def decide(budget: BudgetWindow | None, *, budget_required: bool = False) -> BudgetDecision:
    """The decision for one invocation against one window, or its absence.

    A tenant with no active window is allowed unless the deployment has made
    budgets required, in which case the absence is a block: the operator has
    said every tenant is capped, and a tenant with no cap row is not one the
    gateway may serve. An exhausted window — ``spent >= budget``, the boundary
    included — decides by its policy: ``DOWNGRADE`` re-routes to the cheaper
    model, ``ALERT`` lets the call through and reports it, and ``BLOCK`` — the
    policy of an exhausted window under any other reading, including one the
    gateway does not recognise — refuses the call. Refusing on an unrecognised
    policy is the fail-safe direction: an unknown policy must not become a
    silent allowance.
    """
    if budget is None:
        return BudgetDecision.BLOCK if budget_required else BudgetDecision.ALLOW
    if budget.spent_usd_micros < budget.budget_usd_micros:
        return BudgetDecision.ALLOW
    if budget.policy is BudgetPolicy.DOWNGRADE:
        return BudgetDecision.DOWNGRADE
    if budget.policy is BudgetPolicy.ALERT:
        return BudgetDecision.ALERT
    return BudgetDecision.BLOCK


def decide_strictest(
    windows: Sequence[BudgetWindow], *, budget_required: bool = False
) -> BudgetDecision:
    """The strictest decision across every active window, or their absence.

    A tenant may carry more than one active window, and the strictest of them
    is the one the caller must honour: a block anywhere refuses the call, a
    downgrade anywhere re-routes it, and an alert anywhere reports it, so a
    tenant with a exhausted block window alongside an allowed one is refused
    rather than served against the window that had room. With no window at
    all the decision is :func:`decide`'s own — allowed, or blocked when the
    deployment has made budgets required.
    """
    if not windows:
        return decide(None, budget_required=budget_required)
    return max((decide(window) for window in windows), key=lambda decision: _STRICTNESS[decision])


def _window_from_row(row: Mapping[str, object]) -> BudgetWindow:
    """One active budget window as the database row states it."""
    return BudgetWindow(
        budget_id=row["budget_id"],
        budget_usd_micros=row["budget_usd_micros"],
        spent_usd_micros=row["spent_usd_micros"],
        policy=BudgetPolicy(row["policy"]),
        downgrade_to_logical_model_id=row["downgrade_to_logical_model_id"],
    )


class BudgetManager:
    """Budget admission and settlement against ``per_tenant_llm_budget``.

    Every operation runs in its own transaction with the tenant's RLS GUC set,
    so one tenant's budget rows are the only ones visible and the window a
    decision reads is the window a settlement debits.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def read_active_budget(self, tenant_id: str) -> BudgetWindow | None:
        """The tenant's most recent window enclosing now, or ``None``.

        The row is locked ``FOR UPDATE`` for the read's duration. The lock is
        released before the caller decides, so a decision made from this
        window is made without it — :meth:`active_budgets` is the form that
        holds the lock across the decision, and what the governance chain
        uses. ``None`` means the tenant has no active window — which
        :func:`decide` reads as an allowed call, not an error.
        """
        async with self.active_budgets(tenant_id) as windows:
            return windows[0] if windows else None

    @asynccontextmanager
    async def active_budgets(self, tenant_id: str) -> AsyncIterator[tuple[BudgetWindow, ...]]:
        """Every active window enclosing now, locked ``FOR UPDATE`` for the block.

        The lock is held for the whole of the caller's block, so the decision
        the block makes is made against rows no concurrent settlement can
        interleave with: a second invocation of the same tenant waits for the
        first's settlement and then reads the incremented ``spent``, rather
        than both reading the same value and both deciding there is room. A
        tenant with no active window yields an empty tuple, which
        :func:`decide_strictest` reads as an allowed call. A ``tenant_id``
        that is not a UUID's shape is refused before a connection is taken,
        the same caller bug the other governance mechanisms refuse.
        """
        validate_uuid(tenant_id, "tenant_id")
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            rows = await connection.fetch(_SELECT_ACTIVE_BUDGETS, tenant_id)
            yield tuple(_window_from_row(row) for row in rows)

    async def debit(self, tenant_id: str, usd_micros: int) -> None:
        """Add ``usd_micros`` to the tenant's active window.

        A no-op when there is no active window — the window may have closed
        since the decision — and for a non-positive amount, which carries
        nothing to settle. Tolerating the zero-row update is what lets
        settlement run unconditionally after a call: the caller does not have
        to hold the read's lock for the debit to be correct, and an update that
        matches nothing is a fact about the window, not an error.
        """
        validate_uuid(tenant_id, "tenant_id")
        amount = as_int("budget debit 'usd_micros'", usd_micros, minimum=0)
        if amount == 0:
            return
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            await connection.execute(_DEBIT_ACTIVE_BUDGET, tenant_id, amount)

    async def debit_and_enqueue(
        self,
        tenant_id: str,
        usage_id: str,
        gcid: str,
        cost_micros: int,
        payload: bytes,
        envelope: Mapping[str, str],
    ) -> bool:
        """Write the token-usage outbox row and debit the budget in one transaction.

        The two writes commit together, so a crash between them cannot record
        a usage the budget was not debited for, or debit a budget for a usage
        that was not recorded — the outbox's own idempotency guard is what
        makes the pair atomic rather than merely adjacent. The insert runs
        first with ``ON CONFLICT DO NOTHING``: when it lands (one row), the
        debit follows in the same transaction; when it does not (the usage
        was already recorded), the debit is skipped and ``False`` is returned,
        so a redelivery of one usage cannot charge the tenant twice. A
        non-positive amount carries nothing to settle and skips only the
        debit — the usage is still recorded. Both statements run under the
        tenant's RLS GUCs, because the outbox and the budget tables key their
        policies on them.
        """
        envelope_json = validate_usage_event(usage_id, tenant_id, gcid, payload, envelope)
        amount = as_int("budget debit 'usd_micros'", cost_micros, minimum=0)
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            status = await connection.execute(
                INSERT_TOKEN_USAGE, usage_id, tenant_id, gcid, bytes(payload), envelope_json
            )
            if rows_affected(status) != 1:
                return False
            if amount:
                await connection.execute(_DEBIT_ACTIVE_BUDGET, tenant_id, amount)
        return True
