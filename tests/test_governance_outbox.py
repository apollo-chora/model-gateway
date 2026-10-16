"""Tests for the token-usage outbox insert.

Tested against a stub connection: what is under test is that the insert names
the usage id as its own idempotency key (so a redelivery cannot collide with a
different event or double-settle the same one), that the payload is stored as
bytes and the envelope as JSON, that the row lands in the tenant's RLS
transaction, and that the return value tells the caller whether the row
actually landed.
"""

from __future__ import annotations

import orjson
import pytest

from model_gateway.governance.db import Database
from model_gateway.governance.outbox import AGGREGATE_TYPE, TOPIC, Outbox

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_GCID = "987e6543-e89b-42d3-a456-426614174999"
_USAGE_ID = "0192a4b6-7f3e-7c1a-9d2e-5f6a7b8c9d0e"


class StubTransaction:
    def __init__(self, connection: StubConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> StubConnection:
        return self._connection

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class StubConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.status = "INSERT 0 1"

    def transaction(self) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return self.status


class StubPool:
    def __init__(self, connection: StubConnection) -> None:
        self.connection = connection

    def acquire(self) -> _PooledConnection:
        return _PooledConnection(self)


class _PooledConnection:
    def __init__(self, pool: StubPool) -> None:
        self._pool = pool

    async def __aenter__(self) -> StubConnection:
        return self._pool.connection

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _main_call(connection: StubConnection, needle: str) -> tuple[str, tuple[object, ...]]:
    """The first recorded call whose query contains ``needle``.

    The transaction's SET LOCAL calls precede the insert under test, so the
    call is matched by its SQL rather than by its position.
    """
    for query, params in connection.calls:
        if needle in query:
            return query, params
    raise AssertionError(f"no call matched {needle!r}; calls were {connection.calls}")


@pytest.fixture
async def connection() -> StubConnection:
    return StubConnection()


@pytest.fixture
async def outbox(connection: StubConnection) -> Outbox:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return Outbox(db)


_ENVELOPE = {
    "event_id": _USAGE_ID,
    "idempotency_key": _USAGE_ID,
    "tenant_id": _TENANT_ID,
    "gcid": _GCID,
    "traceparent": "00-abcdef0123456789abcdef0123456789-0123456789abcdef01-01",
    "occurred_at": "2026-10-16T00:00:00.000000Z",
    "source_service": "chora-model-gateway",
    "schema_version": "1",
}


class TestInsertTokenUsage:
    async def test_true_when_the_row_lands(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        assert await outbox.insert_token_usage(
            _USAGE_ID, _TENANT_ID, _GCID, b"\x00\x01payload", _ENVELOPE
        )

    async def test_false_when_a_redelivery_conflicts(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        connection.status = "INSERT 0 0"
        assert not await outbox.insert_token_usage(
            _USAGE_ID, _TENANT_ID, _GCID, b"\x00\x01payload", _ENVELOPE
        )

    async def test_insert_names_the_usage_id_as_its_idempotency_key(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, b"{}", _ENVELOPE)

        query, params = _main_call(connection, "INSERT INTO outbox_events")
        assert "ON CONFLICT (idempotency_key) DO NOTHING" in query
        assert params[0] == _USAGE_ID
        assert params[1] == _TENANT_ID
        assert params[2] == _GCID
        assert params[3] == b"{}"

    async def test_envelope_is_serialised_to_json(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, b"{}", _ENVELOPE)

        _, params = _main_call(connection, "INSERT INTO outbox_events")
        assert orjson.loads(params[4]) == _ENVELOPE  # type: ignore[arg-type]

    async def test_canonical_event_shape(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, b"{}", _ENVELOPE)

        query, _ = _main_call(connection, "INSERT INTO outbox_events")
        assert f"'{TOPIC}'" in query
        assert f"'{AGGREGATE_TYPE}'" in query

    async def test_runs_inside_the_tenant_guc_transaction(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, b"{}", _ENVELOPE)

        gucs = [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]

    async def test_gcid_may_be_empty_for_system_events(
        self, outbox: Outbox, connection: StubConnection
    ) -> None:
        await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, "", b"{}", _ENVELOPE)
        _, params = _main_call(connection, "INSERT INTO outbox_events")
        assert params[2] == ""


class TestValidation:
    async def test_refuses_an_empty_usage_id(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="usage_id"):
            await outbox.insert_token_usage("", _TENANT_ID, _GCID, b"{}", _ENVELOPE)

    async def test_refuses_an_empty_tenant_id(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await outbox.insert_token_usage(_USAGE_ID, "", _GCID, b"{}", _ENVELOPE)

    async def test_refuses_a_non_uuid_tenant_id(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await outbox.insert_token_usage(_USAGE_ID, "not-a-uuid", _GCID, b"{}", _ENVELOPE)

    async def test_refuses_a_non_string_gcid(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="gcid"):
            await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, None, b"{}", _ENVELOPE)  # type: ignore[arg-type]

    async def test_refuses_a_non_bytes_payload(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="payload"):
            await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, "{}", _ENVELOPE)  # type: ignore[arg-type]

    async def test_refuses_a_non_mapping_envelope(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="envelope"):
            await outbox.insert_token_usage(_USAGE_ID, _TENANT_ID, _GCID, b"{}", "json")  # type: ignore[arg-type]

    async def test_refuses_a_non_string_envelope_value(self, outbox: Outbox) -> None:
        with pytest.raises(ValueError, match="envelope"):
            await outbox.insert_token_usage(
                _USAGE_ID, _TENANT_ID, _GCID, b"{}", {"schema_version": 1}  # type: ignore[dict-item]
            )
