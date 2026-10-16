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

**The payload is a protobuf and the envelope a flat string map.** ``payload``
is the canonical ``chora.observability.v1.TokenUsageRecorded`` message the
consumer ``proto.Unmarshal``s, written to ``BYTEA``; ``envelope`` is the
attribute map the relay publishes alongside it, written to ``JSONB`` with
every value a string — the only shape the relay's reader parses. Neither is
parsed here: the gateway's contract with the consumer is the bytes it
wrote, not a shape this module re-validates. The payload is built by
:func:`build_token_usage_payload` from the checked-in contract
``proto/observability/v1/token_usage.proto``, whose field numbers and types
mirror the Go gateway's own — the consumer decodes by field number, so a
renumbered field here is a silently misread ledger row there.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import orjson

from model_gateway.governance.db import Database, rows_affected, validate_uuid
from model_gateway.governance.observability.v1 import token_usage_pb2

__all__ = [
    "INSERT_TOKEN_USAGE",
    "SCHEMA_VERSION",
    "SOURCE_SERVICE",
    "Outbox",
    "build_token_usage_payload",
    "token_usage_envelope",
    "validate_usage_event",
]

TOPIC = "chora.observability.token_usage.recorded.v1"
"""The canonical topic every token-usage event is drained to."""

AGGREGATE_TYPE = "token_usage"
"""The aggregate type stamped on every token-usage event."""

SOURCE_SERVICE = "chora-model-gateway"
"""The provenance stamped on every token-usage event envelope."""

SCHEMA_VERSION = 1
"""The schema major version of the token-usage event payload."""

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


def token_usage_envelope(
    *,
    usage_id: str,
    tenant_id: str,
    gcid: str,
    occurred_at: datetime | None = None,
    traceparent: str = "",
    tracestate: str = "",
) -> dict[str, str]:
    """The flat canonical envelope map for one token-usage event.

    The keys are the contract's own: the relay's reader reconstructs the
    Pub/Sub message attributes from this map and validates them, so a key the
    contract does not name is a key the consumer never sees. Every value is a
    string, the only shape that reader parses. ``traceparent`` and
    ``tracestate`` are empty when the invocation carried no W3C trace context;
    the consumer synthesises deterministic stubs for an absent traceparent
    rather than dropping the correlation.
    """
    return {
        "event_id": usage_id,
        "idempotency_key": usage_id,
        "tenant_id": tenant_id,
        "gcid": gcid,
        "traceparent": traceparent,
        "tracestate": tracestate,
        "occurred_at": (occurred_at or datetime.now(UTC)).isoformat(),
        "source_service": SOURCE_SERVICE,
        "schema_version": str(SCHEMA_VERSION),
    }


def build_token_usage_payload(
    *,
    usage_id: str,
    tenant_id: str,
    gcid: str,
    model_id: str,
    invocation_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cost_micros: int = 0,
    action_code: str = "",
    agent_role: str = "",
    occurred_at: datetime | None = None,
) -> bytes:
    """Build the canonical ``chora.observability.v1.TokenUsageRecorded`` payload.

    The bytes are the Pub/Sub message body the observability service's
    consumer ``proto.Unmarshal``s into its own copy of the message, so the
    field numbers and types are the contract's, not this module's: the same
    shape the Go gateway marshals for the same event. The envelope carries
    the dedupe key (the usage id, which is unique per invocation), the tenant
    and learner the usage is attributed to, the W3C trace context the egress
    span anchors to, and the producer's provenance.

    A token count the provider never reported is zero, not absent: the
    contract's counts are plain ``int64``, which cannot distinguish the two,
    and the consumer rejects a negative count — so an unreported count is
    serialised as the zero the contract reads as "unknown". ``occurred_at``
    defaults to the gateway's own clock when the caller supplies none.
    """
    event = token_usage_pb2.TokenUsageRecorded()
    event.envelope.event_id = usage_id
    event.envelope.idempotency_key = usage_id
    event.envelope.tenant_id = tenant_id
    event.envelope.gcid = gcid
    event.envelope.occurred_at.FromDatetime(occurred_at or datetime.now(UTC))
    event.envelope.source_service = SOURCE_SERVICE
    event.envelope.schema_version = SCHEMA_VERSION
    event.usage_id = usage_id
    event.tenant_id = tenant_id
    event.gcid = gcid
    event.model_id = model_id
    event.input_tokens = input_tokens
    event.output_tokens = output_tokens
    event.cached_tokens = cached_tokens
    event.cost_micros = cost_micros
    event.invocation_id = invocation_id
    event.agent_role = agent_role
    event.action_code = action_code
    return event.SerializeToString()


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
