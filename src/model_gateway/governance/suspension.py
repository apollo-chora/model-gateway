"""Companion containment: the platform and tenant suspension read.

An operator can contain the Learning Companion without a deploy — platform-wide,
per tenant, or per skill — and the gateway is the one un-bypassable chokepoint
that enforces it. This module is the gateway's read of the two
``chora_observability`` tables that hold that containment (ADR-252 D1, read
uncached per turn by ADR-254 D7).

**The platform table is read first, with no tenant context.** It carries no
tenant column and no row-level security, so it is read in a bare transaction.
The tenant table is keyed on ``chora.tenant_id`` and is read under ``SET LOCAL``
in the tenant's own transaction. A row matches when its ``skill_key`` is ``NULL``
(every companion turn in that scope) or equals the call's action code, and the
platform row wins when both match, so a platform-wide containment is never
shadowed by a tenant's narrower one.

**Absent rows allow; an unreadable store denies.** The two are told apart by
error, never by an empty result: a missing row means nobody is contained, while
a read that failed means the gateway cannot know, and a control it cannot read
must not become a control it silently skips. Every read error — a missing
relation before the migration lands, a dropped connection, a malformed row —
fails closed with ``suspension_unreadable``.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_gateway.governance.db import Database, validate_uuid

__all__ = [
    "SuspensionChecker",
    "SuspensionDecision",
]

_PLATFORM_SUSPENSION = """
    SELECT COALESCE(skill_key, ''), reason
    FROM platform_companion_suspension
    WHERE engaged = TRUE
      AND (skill_key IS NULL OR skill_key = $1)
    ORDER BY (skill_key IS NULL) DESC, engaged_at DESC
    LIMIT 1
"""

_TENANT_SUSPENSION = """
    SELECT skill_key, reason
    FROM companion_suspension_policy
    WHERE tenant_id = $1::uuid
      AND engaged = TRUE
      AND (skill_key IS NULL OR skill_key = $2)
    ORDER BY (skill_key IS NULL) DESC, engaged_at DESC
    LIMIT 1
"""

_SUSPENDED = "companion_suspended"
_UNREADABLE = "suspension_unreadable"


@dataclass(frozen=True)
class SuspensionDecision:
    """What one companion turn may do against the containment tables.

    ``allowed`` is the whole answer for a caller that only needs the gate. The
    reason and detail are the operator-facing half: ``companion_suspended``
    names the containment that refused the turn, and its detail names the scope
    it came from, the skill it matched and the operator's own reason, while
    ``suspension_unreadable`` names the store the gateway could not read.
    """

    allowed: bool
    reason: str = ""
    detail: str = ""


def _contained(scope: str, skill_key: object, reason: object) -> SuspensionDecision:
    """The denial for one engaged row, naming its scope, skill and reason.

    An all-skills row carries no skill key, so it is reported as the empty
    skill rather than as the ``None`` the tenant column can hold — the two
    tables store the same "every companion turn" fact in two shapes, and the
    detail should not depend on which one answered.
    """
    skill = skill_key if isinstance(skill_key, str) else ""
    return SuspensionDecision(
        allowed=False,
        reason=_SUSPENDED,
        detail=f"scope={scope} skill={skill} reason={reason}",
    )


class SuspensionChecker:
    """The companion containment read: one uncached round per companion turn.

    Every check takes its own transactions rather than caching, so a containment
    an operator engages mid-flight is enforced on the next turn and a release is
    equally immediate. The platform read is bare and the tenant read carries the
    tenant's GUCs, which is what makes the two tables' different isolation
    schemes invisible to the caller.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def check(self, tenant_id: str, skill_key: str) -> SuspensionDecision:
        """Whether one companion turn is contained, and why when it is.

        The platform table is read first and the tenant table only when it
        yielded nothing, so a platform-wide containment costs one read and a
        tenant's own costs two. A matched engaged row denies with
        ``companion_suspended``; no row on either table allows. Any read error —
        from either transaction — denies with ``suspension_unreadable`` and the
        error's message, because a containment the gateway could not read is a
        containment it must assume is in force. A ``tenant_id`` that is not a
        UUID's shape is refused before a connection is taken, the same caller
        bug the other governance mechanisms refuse.
        """
        validate_uuid(tenant_id, "tenant_id")
        try:
            async with self._db.transaction() as connection:
                platform = await connection.fetchrow(_PLATFORM_SUSPENSION, skill_key)
            if platform is not None:
                return _contained("platform", platform["skill_key"], platform["reason"])
            async with self._db.transaction(tenant_id=tenant_id) as connection:
                tenant = await connection.fetchrow(_TENANT_SUSPENSION, tenant_id, skill_key)
            if tenant is not None:
                return _contained("tenant", tenant["skill_key"], tenant["reason"])
        except Exception as error:
            return SuspensionDecision(allowed=False, reason=_UNREADABLE, detail=str(error))
        return SuspensionDecision(allowed=True)
