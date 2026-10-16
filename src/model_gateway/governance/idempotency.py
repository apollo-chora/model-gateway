"""Invocation idempotency: one dispatch key, one debit.

Pub/Sub is at-least-once, so a dispatched agent turn can be redelivered and
re-invoke the gateway with the same dispatch idempotency key. The learner must
be billed once. ``invoke_debit_claims`` is the gateway's ledger of
dispatch-keyed debits: the first claim on a key inserts the row and wins —
the caller debits — and every later claim on the same key inserts nothing,
debits nothing, and is counted on the winning row (``dedupe_count``,
``last_deduped_invocation_id``, ``last_deduped_at``) so the redelivery proof
can read "this key billed once, deduped N times" straight off the ledger.

**The claim is the debit's gate, not a courtesy.** A claim that lost the race
returns ``False`` and the caller must not debit: two concurrent redeliveries
of one turn would otherwise both pass the check and both charge. The
``INSERT ... ON CONFLICT DO NOTHING`` is what makes the race safe — the
database serialises the two claims, and exactly one of them inserts.

**An unkeyed turn is not deduplicated.** An empty dispatch key means the turn
carries no deduplication contract, so the claim passes through: there is
nothing to dedupe against, and refusing to debit an unkeyed turn would bill
nobody. The key's other half — the learner's GCID — is not optional: a claim
without it has no identity to be keyed by. An empty action code defaults to
``'unspecified'`` so the claim triple still dedupes under an explicit label.
"""

from __future__ import annotations

from model_gateway.governance.db import Database, rows_affected, validate_uuid

__all__ = [
    "IdempotencyClaims",
]

_DEFAULT_ACTION_CODE = "unspecified"
"""The action code a claim carries when the caller stamped none."""

_CLAIM_DEBIT = """
    INSERT INTO invoke_debit_claims (gcid, dispatch_idempotency_key, action_code, invocation_id)
    VALUES ($1::uuid, $2, $3, $4)
    ON CONFLICT (gcid, dispatch_idempotency_key, action_code) DO NOTHING
"""

_RECORD_DEDUPE = """
    UPDATE invoke_debit_claims
    SET dedupe_count = dedupe_count + 1,
        last_deduped_invocation_id = $4,
        last_deduped_at = now()
    WHERE gcid = $1::uuid
      AND dispatch_idempotency_key = $2
      AND action_code = $3
"""


class IdempotencyClaims:
    """Debit claims on ``invoke_debit_claims``, one per dispatch key.

    The table carries no RLS of its own — it is keyed by the learner's GCID and
    written by the gateway on its own transaction — so the claim's transaction
    runs bare, with no tenant GUC set.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def claim(
        self,
        gcid: str,
        dispatch_key: str,
        action_code: str,
        invocation_id: str,
    ) -> bool:
        """Claim the debit for one dispatch key; ``True`` when this caller won.

        The first claim on a ``(gcid, dispatch_key, action_code)`` triple
        inserts its row and wins; a later claim on the same triple inserts
        nothing, records itself as a dedupe on the winning row, and loses. An
        empty ``dispatch_key`` passes the claim through without touching the
        database — an unkeyed turn has nothing to dedupe. A ``gcid`` that is
        empty or not a UUID, and an empty ``invocation_id``, are refused as
        caller bugs before a connection is taken: the claim's identity is
        unusable without them, and the ledger row the winner writes must name
        the invocation that won the key.
        """
        if not dispatch_key:
            return True
        validate_uuid(gcid, "gcid")
        if not invocation_id:
            raise ValueError("idempotency claim 'invocation_id' must not be empty")
        if not isinstance(action_code, str):
            raise ValueError(
                f"idempotency claim 'action_code' must be a string, got {action_code!r}"
            )
        action = action_code or _DEFAULT_ACTION_CODE
        async with self._db.transaction() as connection:
            status = await connection.execute(
                _CLAIM_DEBIT, gcid, dispatch_key, action, invocation_id
            )
            if rows_affected(status) == 1:
                return True
            await connection.execute(_RECORD_DEDUPE, gcid, dispatch_key, action, invocation_id)
            return False
