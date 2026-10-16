"""Token-usage events, written to the transactional outbox.

A completed invocation's usage is recorded in ``outbox_events`` and drained to
the canonical topic ``chora.observability.token_usage.recorded.v1`` by the
observability service's relay. The outbox exists so the record and the call it
describes cannot diverge: the row is written in the caller's own transaction,
and a relay that never runs leaves a row that can be drained later, rather than
an event published after a write that then rolled back.

**The insert is the idempotency guard.** ``idempotency_key`` is unique over
every row in the table, and this module always sets it — to the usage id,
which is unique per invocation — because the column's default would collapse
every event onto one empty-string row and the second insert would fail on the
unique index. ``ON CONFLICT DO NOTHING`` makes a redelivery of the same usage a
clean no-op: the insert reports whether it actually landed, and the caller
settles the budget only when it did.

**The payload is bytes and the envelope a flat string map.** ``payload`` is
the opaque event body the consumer decodes, written to ``BYTEA``; ``envelope``
is the attribute map the relay publishes alongside it, written to ``JSONB``
with every value a string — the only shape the relay's reader parses. Neither
is parsed here: the gateway's contract with the consumer is the bytes it
wrote, not a shape this module re-validates.
"""

from __future__ import annotations

from collections.abc import Mapping

import orjson

from model_gateway.governance.db import Database, rows_affected, validate_uuid

__all__ = [
    "INSERT_TOKEN_USAGE",
    "Outbox",
    "validate_usage_event",
]

TOPIC = "chora.observability.token_usage.recorded.v1"
"""The canonical topic every token-usage event is drained to."""

AGGREGATE_TYPE = "token_usage"
"""The aggregate type stamped on every token-usage event."""

INSERT_TOKEN_USAGE = """
    INSERT INTO outbox_events (
        id, tenant_id, gcid, agid, aggregate_type, aggregate_id,
        event_type, topic, payload, envelope, occurred_at, status,
        retry_count, idempotency_key
    )
    VALUES (
        $1, $2, $3, '', 'token_usage', $1,
        'chora.observability.token_usage.recorded.v1',
        'chora.observability.token_usage.recorded.v1',
        $4, $5::jsonb, now(), 'pending', 0, $1
    )
    ON CONFLICT (idempotency_key) DO NOTHING
"""


def validate_usage_event(
    usage_id: str,
    tenant_id: str,
    gcid: str,
    payload: bytes,
    envelope: Mapping[str, str],
) -> str:
    """Validate one token-usage event and return its serialised envelope.

    The validation :meth:`Outbox.insert_token_usage` refuses — a blank usage
    id, a tenant that is not a UUID, a payload that is not bytes, an envelope
    that is not a flat string map — is shared with the atomic settlement in
    :meth:`~model_gateway.governance.budget.BudgetManager.debit_and_enqueue`,
    which writes the same row in the same transaction as the budget debit. It
    lives here so both writers state the event's rules once.
    """
    if not isinstance(usage_id, str) or not usage_id:
        raise ValueError("outbox 'usage_id' must be a non-empty string")
    validate_uuid(tenant_id, "tenant_id")
    if not isinstance(gcid, str):
        raise ValueError("outbox 'gcid' must be a string")
    if not isinstance(payload, bytes | bytearray):
        raise ValueError("outbox 'payload' must be bytes")
    if not isinstance(envelope, Mapping):
        raise ValueError("outbox 'envelope' must be a mapping of strings to strings")
    for key, value in envelope.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("outbox 'envelope' must map strings to strings")
    return orjson.dumps(dict(envelope)).decode("utf-8")


class Outbox:
    """The token-usage outbox: one insert per completed invocation.

    The insert runs in a transaction with the tenant's RLS GUCs set — the
    outbox table keys its policy on ``app.current_tenant`` — so a redelivery of
    one usage cannot be attributed to another tenant's row.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def insert_token_usage(
        self,
        usage_id: str,
        tenant_id: str,
        gcid: str,
        payload: bytes,
        envelope: Mapping[str, str],
    ) -> bool:
        """Record one token-usage event; ``True`` when the row actually landed.

        ``usage_id`` names the event and is its idempotency key, so a
        redelivery of the same invocation inserts nothing and reports
        ``False`` — the signal the caller must not settle the budget a second
        time for a call it has already paid for. ``gcid`` is the subject the
        usage is attributed to and may be empty for system events. The
        envelope is serialised to JSON here; the payload is stored as the bytes
        it is given.
        """
        envelope_json = validate_usage_event(usage_id, tenant_id, gcid, payload, envelope)
        async with self._db.transaction(tenant_id=tenant_id) as connection:
            status = await connection.execute(
                INSERT_TOKEN_USAGE, usage_id, tenant_id, gcid, bytes(payload), envelope_json
            )
        return rows_affected(status) == 1
