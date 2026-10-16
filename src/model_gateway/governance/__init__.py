"""Governance mechanisms: the platform controls the gateway enforces.

The gateway's governance mechanisms are its contract with the platform around
it: a tenant's spend is capped, a completed call's usage is recorded exactly
once, a redelivered dispatch is debited once, a contained companion is refused
before it is billed, grounded egress is gated by a kill switch, a per-tenant
entitlement and a daily ceiling, and every metered action draws down one
learner's mana wallet. Each mechanism states its own rules; what they share —
one Postgres database, one connection discipline, one tenant-isolation scheme —
lives in :mod:`model_gateway.governance.db`, and the mana meter's only shared
surface is the identity service it calls.

Every database-backed mechanism is opt-in at the settings layer: a deployment
configures ``CHORA_OBSERVABILITY_DSN`` to turn governance on, and the modules
here are imported and wired by the invocation flow that uses them.
"""

from model_gateway.governance.budget import (
    BudgetDecision,
    BudgetManager,
    BudgetPolicy,
    BudgetWindow,
    decide,
)
from model_gateway.governance.db import Database
from model_gateway.governance.egress import EgressDecision, EgressGate
from model_gateway.governance.idempotency import IdempotencyClaims
from model_gateway.governance.mana import (
    ManaClient,
    ManaDebit,
    ManaDebitResult,
    ManaMeter,
    ManaQuote,
    ManaResult,
    UnknownAction,
    is_fail_closed,
)
from model_gateway.governance.mana_client import GrpcManaClient
from model_gateway.governance.manager import (
    COMPANION_SURFACES,
    FAIL_CLOSED_ACTIONS,
    FAIL_CLOSED_PREFIXES,
    GovernanceDecision,
    GovernanceManager,
    governance_refusal,
    is_companion_surface,
)
from model_gateway.governance.outbox import Outbox
from model_gateway.governance.suspension import SuspensionChecker, SuspensionDecision

__all__ = [
    "BudgetDecision",
    "BudgetManager",
    "BudgetPolicy",
    "BudgetWindow",
    "COMPANION_SURFACES",
    "Database",
    "EgressDecision",
    "EgressGate",
    "FAIL_CLOSED_ACTIONS",
    "FAIL_CLOSED_PREFIXES",
    "GovernanceDecision",
    "GovernanceManager",
    "GrpcManaClient",
    "IdempotencyClaims",
    "ManaClient",
    "ManaDebit",
    "ManaDebitResult",
    "ManaMeter",
    "ManaQuote",
    "ManaResult",
    "Outbox",
    "SuspensionChecker",
    "SuspensionDecision",
    "UnknownAction",
    "decide",
    "decide_strictest",
    "governance_refusal",
    "is_companion_surface",
    "is_fail_closed",
]
