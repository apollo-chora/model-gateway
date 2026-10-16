"""Tests for the external-egress gate and its daily counter.

Tested against a stub connection: what is under test is the evaluation order —
the kill switch first with no tenant context, then the entitlement and the daily
counter under the tenant GUC — the inclusive ceiling, the fail-closed behaviour
on a read error, and the upsert that counts a completed call. The SQL is pinned
here so a change to the queries the gateway runs against the three egress tables
cannot pass silently.
"""

from __future__ import annotations

import pytest

from model_gateway.governance.db import Database
from model_gateway.governance.egress import EgressDecision, EgressGate

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_CEILING = 50


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
        self.kill_switch_row: dict[str, object] | None = None
        self.entitlement_row: dict[str, object] | None = None
        self.usage_row: dict[str, object] | None = None
        self.kill_switch_error: Exception | None = None
        self.entitlement_error: Exception | None = None
        self.usage_error: Exception | None = None
        self.status = "INSERT 0 1"

    def transaction(self) -> StubTransaction:
        return StubTransaction(self)

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((query, args))
        if "platform_egress_killswitch" in query:
            if self.kill_switch_error is not None:
                raise self.kill_switch_error
            return self.kill_switch_row
        if "external_egress_policy" in query:
            if self.entitlement_error is not None:
                raise self.entitlement_error
            return self.entitlement_row
        if "grounded_search_daily_usage" in query:
            if self.usage_error is not None:
                raise self.usage_error
            return self.usage_row
        raise AssertionError(f"unexpected query: {query}")

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

    The transaction's SET LOCAL calls precede the tenant reads under test, so
    the call is matched by its SQL rather than by its position.
    """
    for query, params in connection.calls:
        if needle in query:
            return query, params
    raise AssertionError(f"no call matched {needle!r}; calls were {connection.calls}")


@pytest.fixture
async def connection() -> StubConnection:
    return StubConnection()


@pytest.fixture
async def gate(connection: StubConnection) -> EgressGate:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return EgressGate(db)


@pytest.fixture
async def entitled(connection: StubConnection) -> StubConnection:
    """A tenant granted egress with headroom, and the connection serving it."""
    connection.kill_switch_row = {"engaged": False}
    connection.entitlement_row = {"egress_enabled": True, "daily_call_ceiling": _CEILING}
    connection.usage_row = {"call_count": _CEILING - 1}
    return connection


class TestAuthorize:
    async def test_allows_a_tenant_with_headroom(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        assert await gate.authorize(_TENANT_ID) == EgressDecision(allowed=True)

    async def test_an_engaged_kill_switch_denies(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        connection.kill_switch_row = {"engaged": True}
        connection.entitlement_row = {"egress_enabled": True, "daily_call_ceiling": _CEILING}

        decision = await gate.authorize(_TENANT_ID)

        assert decision == EgressDecision(allowed=False, reason="kill_switch_engaged")
        assert not [call for call in connection.calls if "external_egress_policy" in call[0]]

    async def test_a_missing_entitlement_row_denies(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        connection.kill_switch_row = {"engaged": False}

        decision = await gate.authorize(_TENANT_ID)

        assert decision == EgressDecision(allowed=False, reason="external_egress_disabled")

    async def test_a_disabled_entitlement_denies(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {"egress_enabled": False, "daily_call_ceiling": _CEILING}

        decision = await gate.authorize(_TENANT_ID)

        assert decision.reason == "external_egress_disabled"

    async def test_a_missing_kill_switch_row_is_not_engaged(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        entitled.kill_switch_row = None

        assert await gate.authorize(_TENANT_ID) == EgressDecision(allowed=True)

    async def test_the_ceiling_is_inclusive(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        entitled.usage_row = {"call_count": _CEILING}

        decision = await gate.authorize(_TENANT_ID)

        assert decision == EgressDecision(allowed=False, reason="daily_ceiling_exceeded")

    async def test_a_zero_ceiling_denies_every_call(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        entitled.entitlement_row = {"egress_enabled": True, "daily_call_ceiling": 0}
        entitled.usage_row = None

        assert (await gate.authorize(_TENANT_ID)).reason == "daily_ceiling_exceeded"

    async def test_a_tenant_that_has_not_called_today_is_at_zero(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        entitled.usage_row = None

        assert await gate.authorize(_TENANT_ID) == EgressDecision(allowed=True)

    async def test_the_kill_switch_read_runs_before_any_tenant_guc(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        await gate.authorize(_TENANT_ID)

        assert "platform_egress_killswitch" in entitled.calls[0][0]
        assert not [call for call in entitled.calls[:1] if call[0].startswith("SET LOCAL")]

    async def test_the_entitlement_and_ceiling_read_under_the_tenant_guc(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        await gate.authorize(_TENANT_ID)

        gucs = [call for call in entitled.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]
        _, entitlement_params = _main_call(entitled, "external_egress_policy")
        _, usage_params = _main_call(entitled, "grounded_search_daily_usage")
        assert entitlement_params == (_TENANT_ID,)
        assert usage_params == (_TENANT_ID,)

    async def test_the_ceiling_query_uses_today(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        await gate.authorize(_TENANT_ID)

        query, _ = _main_call(entitled, "grounded_search_daily_usage")
        assert "usage_date = current_date" in query

    async def test_a_kill_switch_read_error_fails_closed(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        connection.kill_switch_error = RuntimeError("relation does not exist")

        decision = await gate.authorize(_TENANT_ID)

        assert decision == EgressDecision(
            allowed=False, reason="egress_unreadable", detail="relation does not exist"
        )

    async def test_an_entitlement_read_error_fails_closed(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_error = RuntimeError("connection lost")

        decision = await gate.authorize(_TENANT_ID)

        assert decision.allowed is False
        assert decision.reason == "egress_unreadable"
        assert decision.detail == "connection lost"

    async def test_a_ceiling_read_error_fails_closed(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        entitled.usage_error = RuntimeError("connection lost")

        decision = await gate.authorize(_TENANT_ID)

        assert decision.reason == "egress_unreadable"

    async def test_an_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await gate.authorize("not-a-uuid")
        assert connection.calls == []


class TestRecordEgress:
    async def test_upserts_todays_counter(self, gate: EgressGate, entitled: StubConnection) -> None:
        await gate.record_egress(_TENANT_ID)

        query, params = _main_call(entitled, "grounded_search_daily_usage")
        assert "INSERT INTO grounded_search_daily_usage" in query
        assert "ON CONFLICT (tenant_id, usage_date)" in query
        assert "call_count = grounded_search_daily_usage.call_count + 1" in query
        assert "updated_at = now()" in query
        assert params == (_TENANT_ID,)

    async def test_runs_under_the_tenant_guc(
        self, gate: EgressGate, entitled: StubConnection
    ) -> None:
        await gate.record_egress(_TENANT_ID)

        gucs = [call for call in entitled.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]

    async def test_an_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, gate: EgressGate, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await gate.record_egress("not-a-uuid")
        assert connection.calls == []
