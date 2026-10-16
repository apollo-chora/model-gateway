"""Tests for invocation idempotency debit claims.

Tested against a stub connection: what is under test is the claim's race
semantics — the first insert wins and reports ``True``, a conflicting insert
records itself as a dedupe on the winning row and reports ``False`` — and the
validation that keeps a claim's identity usable. The table carries no RLS, so
the claim's transaction must set no tenant GUC; a test pins that.
"""

from __future__ import annotations

import pytest

from model_gateway.governance.db import Database
from model_gateway.governance.idempotency import IdempotencyClaims

_GCID = "123e4567-e89b-42d3-a456-426614174000"
_INVOCATION_ID = "0192a4b6-7f3e-7c1a-9d2e-5f6a7b8c9d0e"
_DISPATCH_KEY = "dispatch:run:42:turn:7"


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


@pytest.fixture
async def connection() -> StubConnection:
    return StubConnection()


@pytest.fixture
async def claims(connection: StubConnection) -> IdempotencyClaims:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return IdempotencyClaims(db)


class TestClaim:
    async def test_first_claim_wins(
        self, claims: IdempotencyClaims, connection: StubConnection
    ) -> None:
        assert await claims.claim(_GCID, _DISPATCH_KEY, "invoke", _INVOCATION_ID)

        query, params = connection.calls[0]
        assert "INSERT INTO invoke_debit_claims" in query
        assert "ON CONFLICT (gcid, dispatch_idempotency_key, action_code) DO NOTHING" in query
        assert params == (_GCID, _DISPATCH_KEY, "invoke", _INVOCATION_ID)

    async def test_second_claim_is_deduped(
        self, claims: IdempotencyClaims, connection: StubConnection
    ) -> None:
        connection.status = "INSERT 0 0"

        assert not await claims.claim(_GCID, _DISPATCH_KEY, "invoke", _INVOCATION_ID)

        dedupe_query, dedupe_params = connection.calls[1]
        assert "UPDATE invoke_debit_claims" in dedupe_query
        assert "dedupe_count = dedupe_count + 1" in dedupe_query
        assert "last_deduped_invocation_id = $4" in dedupe_query
        assert dedupe_params == (_GCID, _DISPATCH_KEY, "invoke", _INVOCATION_ID)

    async def test_unkeyed_turn_passes_through(
        self, claims: IdempotencyClaims, connection: StubConnection
    ) -> None:
        assert await claims.claim(_GCID, "", "invoke", _INVOCATION_ID)
        assert connection.calls == []

    async def test_empty_action_code_defaults_to_unspecified(
        self, claims: IdempotencyClaims, connection: StubConnection
    ) -> None:
        await claims.claim(_GCID, _DISPATCH_KEY, "", _INVOCATION_ID)
        _, params = connection.calls[0]
        assert params[2] == "unspecified"

    async def test_claim_sets_no_tenant_guc(
        self, claims: IdempotencyClaims, connection: StubConnection
    ) -> None:
        await claims.claim(_GCID, _DISPATCH_KEY, "invoke", _INVOCATION_ID)
        assert not [call for call in connection.calls if call[0].startswith("SET LOCAL")]


class TestValidation:
    async def test_refuses_an_empty_gcid(self, claims: IdempotencyClaims) -> None:
        with pytest.raises(ValueError, match="gcid"):
            await claims.claim("", _DISPATCH_KEY, "invoke", _INVOCATION_ID)

    async def test_refuses_a_non_uuid_gcid(self, claims: IdempotencyClaims) -> None:
        with pytest.raises(ValueError, match="gcid"):
            await claims.claim("not-a-uuid", _DISPATCH_KEY, "invoke", _INVOCATION_ID)

    async def test_refuses_an_empty_invocation_id(self, claims: IdempotencyClaims) -> None:
        with pytest.raises(ValueError, match="invocation_id"):
            await claims.claim(_GCID, _DISPATCH_KEY, "invoke", "")

    async def test_refuses_a_non_string_action_code(self, claims: IdempotencyClaims) -> None:
        with pytest.raises(ValueError, match="action_code"):
            await claims.claim(_GCID, _DISPATCH_KEY, 7, _INVOCATION_ID)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gcid", ["", "not-a-uuid"])
    async def test_validation_precedes_any_database_work(
        self, claims: IdempotencyClaims, connection: StubConnection, gcid: str
    ) -> None:
        with pytest.raises(ValueError, match="gcid"):
            await claims.claim(gcid, _DISPATCH_KEY, "invoke", _INVOCATION_ID)
        assert connection.calls == []
