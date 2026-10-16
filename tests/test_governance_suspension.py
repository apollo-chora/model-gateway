"""Tests for the companion containment read.

Tested against a stub connection: what is under test is the evaluation order —
the platform table first, with no tenant context, then the tenant table under
the tenant GUC — the fail-closed behaviour on a read error, and the detail that
names the scope, the matched skill and the operator's reason. The SQL is pinned
here so a change to the queries the gateway runs against the two containment
tables cannot pass silently.
"""

from __future__ import annotations

import pytest

from model_gateway.governance.db import Database
from model_gateway.governance.suspension import SuspensionChecker, SuspensionDecision

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_SKILL_KEY = "companion_chat_turn_standard"


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
        self.platform_row: dict[str, object] | None = None
        self.tenant_row: dict[str, object] | None = None
        self.platform_error: Exception | None = None
        self.tenant_error: Exception | None = None

    def transaction(self) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        return "SET"

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((query, args))
        if "platform_companion_suspension" in query:
            if self.platform_error is not None:
                raise self.platform_error
            return self.platform_row
        if "companion_suspension_policy" in query:
            if self.tenant_error is not None:
                raise self.tenant_error
            return self.tenant_row
        raise AssertionError(f"unexpected query: {query}")


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

    The transaction's SET LOCAL calls precede the tenant read under test, so the
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
async def checker(connection: StubConnection) -> SuspensionChecker:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return SuspensionChecker(db)


class TestCheck:
    async def test_allows_when_no_row_matches(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        assert await checker.check(_TENANT_ID, _SKILL_KEY) == SuspensionDecision(allowed=True)

    async def test_platform_row_denies(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        connection.platform_row = {"skill_key": _SKILL_KEY, "reason": "policy breach"}

        decision = await checker.check(_TENANT_ID, _SKILL_KEY)

        assert decision == SuspensionDecision(
            allowed=False,
            reason="companion_suspended",
            detail=f"scope=platform skill={_SKILL_KEY} reason=policy breach",
        )

    async def test_tenant_row_denies(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        connection.tenant_row = {"skill_key": None, "reason": "tenant hold"}

        decision = await checker.check(_TENANT_ID, _SKILL_KEY)

        assert decision.reason == "companion_suspended"
        assert decision.detail == "scope=tenant skill= reason=tenant hold"

    async def test_a_platform_row_skips_the_tenant_read(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        connection.platform_row = {"skill_key": "", "reason": "platform hold"}
        connection.tenant_row = {"skill_key": _SKILL_KEY, "reason": "tenant hold"}

        decision = await checker.check(_TENANT_ID, _SKILL_KEY)

        assert "scope=platform" in decision.detail
        assert not [call for call in connection.calls if "companion_suspension_policy" in call[0]]

    async def test_platform_read_runs_before_any_tenant_guc(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        await checker.check(_TENANT_ID, _SKILL_KEY)

        assert "platform_companion_suspension" in connection.calls[0][0]
        assert not [call for call in connection.calls[:1] if call[0].startswith("SET LOCAL")]

    async def test_tenant_read_runs_under_the_tenant_guc(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        await checker.check(_TENANT_ID, _SKILL_KEY)

        gucs = [call for call in connection.calls if call[0].startswith("SET LOCAL")]
        assert gucs == [
            (f"SET LOCAL chora.tenant_id = '{_TENANT_ID}'", ()),
            (f"SET LOCAL app.current_tenant = '{_TENANT_ID}'", ()),
        ]

    async def test_platform_query_matches_all_skills_or_the_skill(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        await checker.check(_TENANT_ID, _SKILL_KEY)

        query, params = _main_call(connection, "platform_companion_suspension")
        assert "COALESCE(skill_key, '')" in query
        assert "skill_key IS NULL OR skill_key = $1" in query
        assert "ORDER BY (skill_key IS NULL) DESC, engaged_at DESC" in query
        assert params == (_SKILL_KEY,)

    async def test_tenant_query_scopes_the_tenant_and_the_skill(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        await checker.check(_TENANT_ID, _SKILL_KEY)

        query, params = _main_call(connection, "companion_suspension_policy")
        assert "tenant_id = $1::uuid" in query
        assert "skill_key IS NULL OR skill_key = $2" in query
        assert params == (_TENANT_ID, _SKILL_KEY)

    async def test_a_platform_read_error_fails_closed(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        connection.platform_error = RuntimeError("relation does not exist")

        decision = await checker.check(_TENANT_ID, _SKILL_KEY)

        assert decision == SuspensionDecision(
            allowed=False, reason="suspension_unreadable", detail="relation does not exist"
        )

    async def test_a_tenant_read_error_fails_closed(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        connection.tenant_error = RuntimeError("connection lost")

        decision = await checker.check(_TENANT_ID, _SKILL_KEY)

        assert decision.allowed is False
        assert decision.reason == "suspension_unreadable"
        assert decision.detail == "connection lost"

    async def test_an_invalid_tenant_is_refused_before_a_connection_is_taken(
        self, checker: SuspensionChecker, connection: StubConnection
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            await checker.check("not-a-uuid", _SKILL_KEY)
        assert connection.calls == []
