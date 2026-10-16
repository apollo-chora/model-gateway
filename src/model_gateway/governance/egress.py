"""External egress governance: kill switch, entitlement, daily ceiling.

Grounded web search is the gateway's one controlled egress surface, and this
module is the fail-closed gate in front of it. Three ``chora_observability``
tables answer one question — may this tenant make a grounded call right now —
and each can refuse (ADR-231 D6).

**The kill switch is read first, with no tenant context.** It is a single
platform-wide row with no row-level security, so it is read in a bare
transaction; an engaged row refuses every tenant's egress. The entitlement and
the daily counter are tenant-scoped and RLS-enforcing, so both are read in the
tenant's own transaction under ``SET LOCAL chora.tenant_id``, in that order,
because the ceiling is only meaningful once the entitlement has been granted.

**A tenant with no entitlement row is refused.** External egress defaults OFF:
the absence of a row is the franchise default, not an oversight, and a tenant
that was never granted egress must not receive it because a row is missing. The
same refusal covers a row that is present but ``egress_enabled = FALSE``.

**The ceiling is inclusive.** ``call_count >= daily_call_ceiling`` refuses, so
a ceiling of zero refuses every call and a tenant already at its ceiling refuses
the next one. The counter is per ``(tenant, day)``, so a tenant that has not
called today is at zero and is read as absent rather than as an error.

**Every read error fails closed.** A missing relation before the migration
lands, a dropped connection, a malformed row — the gate cannot know, so it
refuses with ``egress_unreadable`` rather than granting egress it could not
verify.

**The counter is incremented post-success.** :meth:`EgressGate.record_egress`
upserts the tenant's row for today, so a redelivered success cannot be counted
twice by a second insert and two concurrent calls cannot lose an increment to
one another.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_gateway.governance.db import Database, validate_uuid

__all__ = [
    "EgressDecision",
    "EgressGate",
]

_KILL_SWITCH = "SELECT engaged FROM platform_egress_killswitch LIMIT 1"

_ENTITLEMENT = """
    SELECT egress_enabled, daily_call_ceiling
    FROM external_egress_policy
    WHERE tenant_id = $1::uuid
"""

_DAILY_USAGE = """
    SELECT call_count
    FROM grounded_search_daily_usage
    WHERE tenant_id = $1::uuid AND usage_date = current_date
"""

_RECORD_EGRESS = """
    INSERT INTO grounded_search_daily_usage (tenant_id, usage_date, call_count)
    VALUES ($1::uuid, current_date, 1)
    ON CONFLICT (tenant_id, usage_date)
    DO UPDATE SET call_count = call_count + 1, updated_at = now()
"""

_KILL_SWITCH_ENGAGED = "kill_switch_engaged"
_EGRESS_DISABLED = "external_egress_disabled"
_DAILY_CEILING_EXCEEDED = "daily_ceiling_exceeded"
_UNREADABLE = "egress_unreadable"


@dataclass(frozen=True)
class EgressDecision:
    """What one grounded call may do against the egress tables.

    ``allowed`` is the whole answer for a caller that only needs the gate; the
    reason token is the operator-facing half, one of the governance chain's own
    — ``kill_switch_engaged``, ``external_egress_disabled``,
    ``daily_ceiling_exceeded`` — and ``egress_unreadable`` names the store the
    gateway could not read, with the error's message as its detail.
    """

    allowed: bool
    reason: str = ""
    detail: str = ""


class EgressGate:
    """The fail-closed external-egress gate and its daily counter.

    Every check takes its own transactions rather than caching, so a kill switch
    an operator engages mid-flight takes effect on the next call. The kill
    switch's read is bare and the entitlement and ceiling reads carry the
    tenant's GUCs, which is what makes the platform table's lack of isolation
    and the tenant tables' RLS invisible to the caller.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def authorize(self, tenant_id: str) -> EgressDecision:
        """Whether this tenant may make one grounded call right now.

        The kill switch is read first and refuses on its own, so an engaged
        platform switch costs one read and never reaches the tenant tables. The
        entitlement is read next and refuses when it is absent or disabled, and
        only then is the daily counter read against the ceiling the entitlement
        named. Any read error — from either transaction — denies with
        ``egress_unreadable`` and the error's message, because egress the
        gateway could not verify is egress it must not grant. A ``tenant_id``
        that is not a UUID's shape is refused before a connection is taken, the
        same caller bug the other governance mechanisms refuse.
        """
        validate_uuid(tenant_id, "tenant_id")
        try:
            async with self._db.transaction() as connection:
                switch = await connection.fetchrow(_KILL_SWITCH)
            if switch is not None and switch["engaged"]:
                return EgressDecision(allowed=False, reason=_KILL_SWITCH_ENGAGED)
            async with self._db.transaction(tenant_id=tenant_id) as connection:
                entitlement = await connection.fetchrow(_ENTITLEMENT, tenant_id)
                if entitlement is None or not entitlement["egress_enabled"]:
                    return EgressDecision(allowed=False, reason=_EGRESS_DISABLED)
                usage = await connection.fetchrow(_DAILY_USAGE, tenant_id)
                ceiling = entitlement["daily_call_ceiling"]
                count = usage["call_count"] if usage is not None else 0
                if count >= ceiling:
                    return EgressDecision(allowed=False, reason=_DAILY_CEILING_EXCEEDED)
        except Exception as error:
            return EgressDecision(allowed=False, reason=_UNREADABLE, detail=str(error))
        return EgressDecision(allowed=True)

    async def record_egress(self, tenant_id: str) -> None:
        """Count one completed grounded call against today's ceiling.

        The upsert is the whole of the write: a first call for the day inserts
        the row at one, and every later call increments it, so a redelivered
        success and two concurrent calls cannot collapse onto one count. The
        write runs under the tenant's GUCs, because the counter table is
        RLS-enforcing like the entitlement it belongs to. A ``tenant_id`` that
        is not a UUID's shape is refused before a connection is taken.
        """
        validate_uuid(tenant_id, "tenant_id")
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            await connection.execute(_RECORD_EGRESS, tenant_id)
