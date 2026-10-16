"""Per-tenant LLM budget admission and settlement.

Every invocation a tenant makes draws down a windowed USD cap the operator
configures in ``per_tenant_llm_budget``. This module is the gateway's half of
that contract: it reads the active window, decides what the window allows, and
settles the spend against it.

**The read locks the window row.** :meth:`BudgetManager.read_active_budget`
selects the most recent window that currently encloses ``now()`` with
``FOR UPDATE``, so two concurrent invocations of one tenant cannot both read
the same ``spent`` and both decide there is room: the second waits for the
first's settlement and then reads the incremented value. The lock is held only
for the caller's transaction, which is what bounds contention even when one
tenant fires concurrent calls.

**The decision is a pure function of the window.** :func:`decide` takes the
window, or its absence, and nothing else, so the policy — allow, downgrade,
block, alert — can be checked against every shape a window can take without a
database. A tenant with no active window is allowed through: the budget is an
operator opt-in, and refusing a tenant that was never configured would fail
closed on a deployment choice the gateway does not own. ``budget_required``
exists for the deployment that does want the closed behaviour.

**Settlement tolerates a missing window.** :meth:`BudgetManager.debit` updates
the active window and accepts a zero-row result: the window may have closed
between the decision and the settlement, and charging a tenant against a window
that no longer exists would attach a bookkeeping error to no row. Money is
integer micros (``BIGINT``) end to end; nothing here converts to or from a
float.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum

from model_gateway.domain.serialization import as_int
from model_gateway.governance.db import Database, validate_uuid

__all__ = [
    "BudgetDecision",
    "BudgetManager",
    "BudgetPolicy",
    "BudgetWindow",
    "decide",
]

_SELECT_ACTIVE_BUDGET = """
    SELECT budget_id, budget_usd_micros, spent_usd_micros, policy, downgrade_to_logical_model_id
    FROM per_tenant_llm_budget
    WHERE tenant_id = $1::uuid
      AND budget_period_start <= now()
      AND budget_period_end > now()
    ORDER BY budget_period_start DESC
    LIMIT 1
    FOR UPDATE
"""

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

        The row is locked ``FOR UPDATE`` for the transaction's duration, so a
        concurrent settlement cannot interleave between this read and the debit
        it gates. ``None`` means the tenant has no active window — which
        :func:`decide` reads as an allowed call, not an error.
        """
        validate_uuid(tenant_id, "tenant_id")
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            row = await connection.fetchrow(_SELECT_ACTIVE_BUDGET, tenant_id)
        if row is None:
            return None
        return BudgetWindow(
            budget_id=row["budget_id"],
            budget_usd_micros=row["budget_usd_micros"],
            spent_usd_micros=row["spent_usd_micros"],
            policy=BudgetPolicy(row["policy"]),
            downgrade_to_logical_model_id=row["downgrade_to_logical_model_id"],
        )

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
