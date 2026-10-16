"""Tests for the governance Database: pool lifecycle, transaction discipline, helpers.

The Database class is tested against a stub pool rather than a live Postgres:
what is under test is the discipline — GUCs set inside the transaction, commit
on success, rollback on error, UUID validated before a connection is taken —
and a stub records exactly the calls whose order and content that discipline
is. The SQL itself is pinned by the budget, outbox and idempotency tests.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from model_gateway.governance.db import Database, rows_affected, validate_uuid

_DSN = "postgresql://chora_observability_app_rw@localhost/chora_observability"
_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"


class StubTransaction:
    """Records the outcome of the transaction it wraps, as asyncpg's would."""

    def __init__(self, connection: StubConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> StubConnection:
        self._connection.begun += 1
        return self._connection

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self._connection.committed += 1
        else:
            self._connection.rolled_back += 1
        return False


class StubConnection:
    """Records every call; returns the row, rows and status the test configures."""

    def __init__(self) -> None:
        self.begun = 0
        self.committed = 0
        self.rolled_back = 0
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.status = "UPDATE 0"
        self.row: dict[str, object] | None = None
        self.rows: list[dict[str, object]] = []

    def transaction(self) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return self.status

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((query, args))
        return self.row

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.calls.append((query, args))
        return self.rows


class _PooledConnection:
    """The handle pool.acquire() yields; releases back to the pool on exit."""

    def __init__(self, pool: StubPool) -> None:
        self._pool = pool

    async def __aenter__(self) -> StubConnection:
        return self._pool.connection

    async def __aexit__(self, *exc: object) -> bool:
        self._pool.released += 1
        return False


class StubPool:
    """Hands out one stub connection and counts acquisitions.

    Mirrors the asyncpg Pool's own query methods, which run on a pooled
    connection outside any transaction — the shape the Database helpers use.
    """

    def __init__(self, connection: StubConnection) -> None:
        self.connection = connection
        self.acquired = 0
        self.released = 0
        self.closed = False

    def acquire(self) -> _PooledConnection:
        self.acquired += 1
        return _PooledConnection(self)

    async def close(self) -> None:
        self.closed = True

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        return await self.connection.fetchrow(query, *args)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        return await self.connection.fetch(query, *args)

    async def execute(self, query: str, *args: object) -> str:
        return await self.connection.execute(query, *args)


class TestValidateUuid:
    def test_accepts_a_uuid(self) -> None:
        validate_uuid(_TENANT_ID, "tenant_id")

    def test_accepts_uppercase_hex(self) -> None:
        validate_uuid(_TENANT_ID.upper(), "tenant_id")

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "not-a-uuid",
            "123e4567e89b42d3a456426614174000",
            "123e4567-e89b-42d3-a456-42661417400",
        ],
    )
    def test_refuses_a_non_uuid(self, value: str) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            validate_uuid(value, "tenant_id")

    def test_refuses_a_non_string(self) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            validate_uuid(None, "tenant_id")  # type: ignore[arg-type]


class TestRowsAffected:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("INSERT 0 1", 1),
            ("INSERT 0 0", 0),
            ("UPDATE 3", 3),
            ("DELETE 0", 0),
            ("SET", 0),
        ],
    )
    def test_parses_the_count_from_the_tag_tail(self, status: str, expected: int) -> None:
        assert rows_affected(status) == expected


@pytest.fixture
async def connection() -> StubConnection:
    return StubConnection()


@pytest.fixture
async def pool(connection: StubConnection, monkeypatch: pytest.MonkeyPatch) -> StubPool:
    stub = StubPool(connection)

    async def fake_create_pool(dsn: str) -> StubPool:
        assert dsn == _DSN
        return stub

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    return stub


@pytest.fixture
async def database(pool: StubPool) -> Database:
    db = Database(_DSN)
    await db.connect()
    return db


class TestConnectAndClose:
    async def test_connect_builds_the_pool_from_the_dsn(self, database: Database) -> None:
        assert database.dsn == _DSN

    async def test_close_releases_the_pool(self, database: Database) -> None:
        await database.close()

    async def test_connect_refuses_an_empty_dsn(self) -> None:
        with pytest.raises(ValueError, match="dsn"):
            Database("")

    async def test_queries_refuse_an_unopened_pool(self) -> None:
        db = Database(_DSN)
        with pytest.raises(RuntimeError, match="connect"):
            await db.fetch_one("SELECT 1")
        with pytest.raises(RuntimeError, match="connect"):
            await db.execute("SELECT 1")


class TestTransaction:
    async def test_sets_both_gucs_in_order_and_commits(
        self, database: Database, connection: StubConnection
    ) -> None:
        async with database.transaction(tenant_id=_TENANT_ID):
            pass

        gucs = [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]
        assert (connection.begun, connection.committed, connection.rolled_back) == (1, 1, 0)

    async def test_no_gucs_without_a_tenant(
        self, database: Database, connection: StubConnection
    ) -> None:
        async with database.transaction():
            await connection.execute("INSERT INTO invoke_debit_claims DEFAULT VALUES")

        assert not [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert connection.committed == 1

    async def test_rolls_back_on_error(
        self, database: Database, connection: StubConnection
    ) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with database.transaction(tenant_id=_TENANT_ID):
                raise RuntimeError("boom")

        assert (connection.begun, connection.committed, connection.rolled_back) == (1, 0, 1)

    async def test_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, database: Database, pool: StubPool, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            async with database.transaction(tenant_id="not-a-uuid"):
                pass

        assert pool.acquired == 0
        assert connection.calls == []


class TestQueryHelpers:
    async def test_fetch_one_returns_the_first_row(
        self,
        database: Database,
        connection: StubConnection,
    ) -> None:
        budget_id = uuid.uuid4()
        connection.row = {"budget_id": budget_id}
        row = await database.fetch_one("SELECT budget_id FROM per_tenant_llm_budget", _TENANT_ID)
        assert row == {"budget_id": budget_id}

    async def test_fetch_one_returns_none_without_a_row(self, database: Database) -> None:
        assert await database.fetch_one("SELECT 1 WHERE false") is None

    async def test_fetch_all_returns_every_row(
        self, database: Database, connection: StubConnection
    ) -> None:
        connection.rows = [{"n": 1}, {"n": 2}]
        rows = await database.fetch_all("SELECT 1 AS n")
        assert rows == [{"n": 1}, {"n": 2}]

    async def test_execute_returns_the_command_tag(self, database: Database) -> None:
        assert await database.execute("UPDATE nothing SET x = 1") == "UPDATE 0"

    async def test_ping_reaches_the_database(
        self, database: Database, connection: StubConnection
    ) -> None:
        connection.row = {"?column?": 1}
        await database.ping()
        assert connection.calls == [("SELECT 1", ())]
