"""Postgres access for the governance mechanisms.

The gateway's governance mechanisms — budget admission, the token-usage
outbox, invocation idempotency — share one Postgres database,
``chora_observability``, and one connection discipline. This module owns that
discipline so each mechanism states its own rules and not the plumbing.

**Every tenant-scoped statement runs inside a transaction that has set the
tenant GUCs first.** The tables carry row-level security keyed on
``chora.tenant_id`` (``per_tenant_llm_budget``) and ``app.current_tenant``
(``outbox_events``); a connection that has not set them sees no rows at all,
and one that sets them outside a transaction leaks the value onto the next
borrower of the same pooled connection. ``SET LOCAL`` inside the transaction is
the only form that is both effective and contained, so the transaction context
manager applies it and nothing else does.

**A tenant id is validated as a UUID before it is inlined.** ``SET LOCAL``
cannot be parameterised — the driver prepares the statement and a placeholder
is not accepted there — so the id is written into the SQL text. That is safe
only because the value has first been checked to be exactly a UUID's shape; a
malformed id is a caller bug refused before a connection is taken from the
pool, not a database error surfaced after one.

**The pool is opened and closed with the process.** Governance is opt-in: a
deployment that configures no DSN never opens a pool, and one that does owns
the pool's lifecycle — ``connect`` at startup, ``close`` at shutdown — because
an unclosed pool keeps the event loop from exiting cleanly.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

__all__ = [
    "Database",
    "rows_affected",
    "validate_uuid",
]

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
"""The exact shape of a UUID, any version — the only form ``SET LOCAL`` may inline."""


def validate_uuid(value: str, name: str) -> None:
    """Refuse ``value`` unless it is exactly a UUID's shape.

    The check is a shape check, not a registry lookup: every UUID version the
    database may hold passes, and anything else — empty, a placeholder, a
    tenant name — is refused where it was named rather than after a connection
    was taken to reject it. The same check guards the ``::uuid`` casts the
    governance queries rely on, so a malformed id fails as a caller bug
    instead of as a database error.
    """
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise ValueError(f"{name} must be a UUID, got {value!r}")


def rows_affected(status: str) -> int:
    """The row count in an asyncpg command tag (``"INSERT 0 1"`` → 1).

    asyncpg reports the outcome of a statement as its command tag rather than
    as a row count, and the tag's shape depends on the statement: ``INSERT``
    carries an oid and a count (``"INSERT 0 1"``), ``UPDATE`` and ``DELETE``
    carry only a count (``"UPDATE 3"``), and a statement that affects no rows
    reports zero (``"INSERT 0 0"``). The count is always the tag's last token,
    which is what an ``ON CONFLICT DO NOTHING`` insert's caller needs to tell a
    landed row from a skipped one.
    """
    try:
        return int(status.rsplit(" ", 1)[-1])
    except ValueError:
        return 0


class Database:
    """A pooled Postgres connection with the gateway's transaction discipline.

    The pool is created by :meth:`connect` and closed by :meth:`close`; the
    query helpers and :meth:`transaction` refuse to run against an unopened
    pool rather than opening one implicitly, so a deployment that configures
    no DSN never pays for a connection it will not use.
    """

    def __init__(self, dsn: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("a database 'dsn' must be a non-empty connection string")
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def dsn(self) -> str:
        """The connection string this pool is built from."""
        return self._dsn

    async def connect(self) -> None:
        """Create the connection pool. A second call is a no-op."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._dsn)

    async def close(self) -> None:
        """Close the pool, releasing every connection it holds."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> None:
        """Check the database is reachable, backing the readiness probe."""
        await self.fetch_one("SELECT 1")

    @asynccontextmanager
    async def transaction(self, tenant_id: str | None = None) -> AsyncIterator[asyncpg.Connection]:
        """Run the wrapped block in one transaction, with the tenant GUCs set if given.

        The transaction begins before the GUCs are applied, so ``SET LOCAL``
        takes effect for the transaction's whole duration and is rolled back
        with it — the value can never outlive the transaction or leak onto the
        next borrower of the connection. Both GUCs are set, in this order,
        because the governance tables key their RLS on different ones:
        ``chora.tenant_id`` for the budget tables, ``app.current_tenant`` for
        the outbox. With no ``tenant_id`` the transaction runs bare, which is
        the correct form for the tables that carry no RLS at all.

        Commits on success and rolls back on any error from the block. A
        ``tenant_id`` that is not a UUID's shape is refused before a
        connection is acquired, because the id is inlined into the ``SET LOCAL``
        statement and must be safe to inline.
        """
        pool = self._pool_or_raise()
        if tenant_id is not None:
            validate_uuid(tenant_id, "tenant_id")
        async with pool.acquire() as connection, connection.transaction():
            if tenant_id is not None:
                await connection.execute(f"SET LOCAL chora.tenant_id = '{tenant_id}'")
                await connection.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")
            yield connection

    async def fetch_one(self, query: str, *args: object) -> asyncpg.Record | None:
        """Run ``query`` on a pooled connection and return its first row or ``None``."""
        return await self._pool_or_raise().fetchrow(query, *args)

    async def fetch_all(self, query: str, *args: object) -> list[asyncpg.Record]:
        """Run ``query`` on a pooled connection and return every row it matched."""
        return await self._pool_or_raise().fetch(query, *args)

    async def execute(self, query: str, *args: object) -> str:
        """Run ``query`` on a pooled connection and return its command tag."""
        return await self._pool_or_raise().execute(query, *args)

    def _pool_or_raise(self) -> asyncpg.Pool:
        """The open pool, or a refusal naming what is missing."""
        if self._pool is None:
            raise RuntimeError("the database pool is not open; call connect() first")
        return self._pool
