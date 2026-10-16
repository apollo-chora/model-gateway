"""Tests for budget admission and settlement.

The manager is tested against a stub connection: what is under test is that the
read locks the active window ``FOR UPDATE`` inside a tenant-GUC transaction,
that the decision is a pure function of the window, and that settlement
tolerates a window that is no longer there. The SQL is pinned here so a change
to the query the gateway runs against ``per_tenant_llm_budget`` cannot pass
silently.
"""

from __future__ import annotations

import uuid

import pytest

from model_gateway.governance.budget import (
    BudgetDecision,
    BudgetManager,
    BudgetPolicy,
    BudgetWindow,
    decide,
)
from model_gateway.governance.db import Database

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_BUDGET_ID = uuid.uuid4()


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
        self.row: dict[str, object] | None = None
        self.status = "UPDATE 0"

    def transaction(self) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return self.status

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((query, args))
        return self.row


class StubPool:
    def __init__(self, connection: StubConnection) -> None:
        self.connection = connection
        self.acquired = 0

    def acquire(self) -> _PooledConnection:
        self.acquired += 1
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

    The transaction's SET LOCAL calls precede the statement under test, so the
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
async def manager(connection: StubConnection) -> BudgetManager:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return BudgetManager(db)


def _window(
    *,
    budget_micros: int = 1_000_000,
    spent_micros: int = 0,
    policy: BudgetPolicy = BudgetPolicy.BLOCK,
    downgrade_to: str | None = None,
) -> BudgetWindow:
    return BudgetWindow(
        budget_id=_BUDGET_ID,
        budget_usd_micros=budget_micros,
        spent_usd_micros=spent_micros,
        policy=policy,
        downgrade_to_logical_model_id=downgrade_to,
    )


class TestReadActiveBudget:
    async def test_reads_the_active_window_with_a_lock(
        self,
        manager: BudgetManager,
        connection: StubConnection,
    ) -> None:
        connection.row = {
            "budget_id": _BUDGET_ID,
            "budget_usd_micros": 1_000_000,
            "spent_usd_micros": 250_000,
            "policy": "block",
            "downgrade_to_logical_model_id": None,
        }

        window = await manager.read_active_budget(_TENANT_ID)

        assert window == _window(spent_micros=250_000)
        query, params = _main_call(connection, "per_tenant_llm_budget")
        assert "FOR UPDATE" in query
        assert params == (_TENANT_ID,)

    async def test_runs_inside_the_tenant_guc_transaction(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        await manager.read_active_budget(_TENANT_ID)

        gucs = [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]

    async def test_no_active_window_is_none(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        assert await manager.read_active_budget(_TENANT_ID) is None

    async def test_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await manager.read_active_budget("not-a-uuid")
        assert connection.calls == []


class TestDecide:
    def test_allows_a_window_under_its_cap(self) -> None:
        decision = decide(_window(budget_micros=1_000_000, spent_micros=999_999))
        assert decision is BudgetDecision.ALLOW

    def test_exact_cap_is_exhausted(self) -> None:
        decision = decide(_window(budget_micros=1_000_000, spent_micros=1_000_000))
        assert decision is BudgetDecision.BLOCK

    def test_blocks_an_exhausted_block_window(self) -> None:
        assert (
            decide(_window(spent_micros=2_000_000, policy=BudgetPolicy.BLOCK))
            is BudgetDecision.BLOCK
        )

    def test_downgrades_an_exhausted_downgrade_window(self) -> None:
        assert (
            decide(
                _window(
                    spent_micros=2_000_000,
                    policy=BudgetPolicy.DOWNGRADE,
                    downgrade_to="gemini-2.5-flash-lite",
                )
            )
            is BudgetDecision.DOWNGRADE
        )

    def test_alerts_an_exhausted_alert_window(self) -> None:
        assert (
            decide(_window(spent_micros=2_000_000, policy=BudgetPolicy.ALERT))
            is BudgetDecision.ALERT
        )

    def test_no_window_fails_open(self) -> None:
        assert decide(None) is BudgetDecision.ALLOW

    def test_no_window_blocks_when_budget_is_required(self) -> None:
        assert decide(None, budget_required=True) is BudgetDecision.BLOCK


class TestBudgetWindow:
    def test_refuses_a_non_uuid_budget_id(self) -> None:
        with pytest.raises(ValueError, match="budget_id"):
            BudgetWindow(
                budget_id="not-a-uuid",  # type: ignore[arg-type]
                budget_usd_micros=1,
                spent_usd_micros=0,
                policy=BudgetPolicy.BLOCK,
                downgrade_to_logical_model_id=None,
            )

    def test_refuses_a_non_positive_cap(self) -> None:
        with pytest.raises(ValueError, match="budget_usd_micros"):
            _window(budget_micros=0)

    def test_refuses_a_negative_spend(self) -> None:
        with pytest.raises(ValueError, match="spent_usd_micros"):
            _window(spent_micros=-1)


class TestDebit:
    async def test_settles_against_the_active_window(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        connection.status = "UPDATE 1"

        await manager.debit(_TENANT_ID, 50_000)

        query, params = _main_call(connection, "UPDATE per_tenant_llm_budget")
        assert "spent_usd_micros = spent_usd_micros + $2" in query
        assert params == (_TENANT_ID, 50_000)

    async def test_runs_inside_the_tenant_guc_transaction(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        await manager.debit(_TENANT_ID, 50_000)

        gucs = [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]

    async def test_a_window_that_closed_is_tolerated(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        connection.status = "UPDATE 0"
        await manager.debit(_TENANT_ID, 50_000)

    async def test_zero_amount_is_a_no_op(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        await manager.debit(_TENANT_ID, 0)
        assert connection.calls == []

    async def test_negative_amount_is_refused(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="usd_micros"):
            await manager.debit(_TENANT_ID, -1)
        assert connection.calls == []

    async def test_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, manager: BudgetManager, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await manager.debit("not-a-uuid", 50_000)
        assert connection.calls == []
