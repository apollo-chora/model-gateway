"""Postgres integration fixtures for the governance mechanisms.

The governance mechanisms' unit tests run against stub connections, which
establish the gateway's discipline but not the database behaviour behind it:
row-level security, the ``FOR UPDATE`` read lock, ``ON CONFLICT DO NOTHING``
dedupe and the claim race are properties of Postgres, not of the gateway's SQL.
These fixtures spin up a real Postgres container, apply the chora-observability
migrations the gateway's tables come from, and hand the tests a
:class:`~model_gateway.governance.db.Database` connected as the application
role — so the RLS policies, the locks and the conflict handling under test are
the database's own.

The container is started once per test session and the migrations applied once;
each test gets its own connection pool over the same schema, so tests share the
database but not a pool. Tests that need isolation use distinct tenant ids.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

from model_gateway.governance.db import Database

POSTGRES_IMAGE = "pgvector/pgvector:0.8.7-pg18"
"""The container image, matching the chora-stack Postgres service."""

_APP_ROLE = "chora_observability_app_rw"
_APP_PASSWORD = "app_rw_test"

_MIGRATIONS_DIR = Path(
    os.environ.get(
        "CHORA_OBSERVABILITY_MIGRATIONS",
        str(Path(__file__).resolve().parents[2] / "chora-observability" / "migrations"),
    )
)
"""The directory holding the chora-observability migrations."""

_MIGRATION_FILES = (
    "0001_initial.sql",
    "0003_outbox.sql",
    "0006_outbox_d6_canonical.up.sql",
    "0008_per_tenant_llm_budget.up.sql",
    "0014_external_egress_policy.up.sql",
    "0018_companion_suspension.up.sql",
    "0019_invoke_debit_claims.up.sql",
)
"""The migrations that create the gateway's tables, in apply order.

0001 is a prerequisite rather than a gateway table: 0008, 0014 and 0018 create
triggers that execute ``observability_set_updated_at()``, which 0001 defines. The
rest are the gateway's own — the outbox (0003, 0006), the per-tenant budget
(0008), the egress tables (0014), the companion containment tables (0018) and
the debit-claim ledger (0019).
"""


def _normalize_scheme(dsn: str) -> str:
    """Reduce a driver-qualified scheme to the plain ``postgresql`` asyncpg accepts.

    ``PostgresContainer.get_connection_url`` emits a SQLAlchemy-style
    ``postgresql+psycopg2`` URL; asyncpg accepts only ``postgresql`` (or
    ``postgres``), so the driver suffix is dropped before connecting.
    """
    parts = urlsplit(dsn)
    if parts.scheme.startswith("postgresql+"):
        parts = parts._replace(scheme="postgresql")
    return urlunsplit(parts)


def _app_dsn(admin_dsn: str) -> str:
    """The connection string for the application role, from the admin one."""
    parts = urlsplit(admin_dsn)
    host = parts.netloc.rsplit("@", 1)[-1]
    netloc = f"{_APP_ROLE}:{_APP_PASSWORD}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


async def _prepare_database(admin_dsn: str) -> None:
    """Create the application role and apply the migrations, as the admin user.

    The role is created before the migrations so the migrations' own grant
    blocks — each guarded by ``IF EXISTS (SELECT 1 FROM pg_roles ...)`` — apply
    to it. A final ``GRANT`` covers the tables the migrations grant to a
    different role name (the outbox, and the budget table's ``app_rw``), so the
    application role can read and write every gateway table whichever name the
    migration checked for.
    """
    connection = await asyncpg.connect(admin_dsn)
    try:
        await connection.execute(
            f"CREATE ROLE {_APP_ROLE} WITH LOGIN PASSWORD '{_APP_PASSWORD}' NOBYPASSRLS"
        )
        for name in _MIGRATION_FILES:
            script = (_MIGRATIONS_DIR / name).read_text(encoding="utf-8")
            await connection.execute(script)
        await connection.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {_APP_ROLE}"
        )
        await connection.execute(
            f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO {_APP_ROLE}"
        )
        await connection.execute(
            f"GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO {_APP_ROLE}"
        )
    finally:
        await connection.close()


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """The Postgres container, started once per test session and stopped at its end."""
    container = PostgresContainer(POSTGRES_IMAGE)
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def postgres_admin_dsn(postgres_container: PostgresContainer) -> Iterator[str]:
    """The container's admin connection string, with the governance schema applied."""
    admin_dsn = _normalize_scheme(postgres_container.get_connection_url())
    asyncio.run(_prepare_database(admin_dsn))
    yield admin_dsn


@pytest.fixture(scope="session")
def postgres_dsn(postgres_admin_dsn: str) -> Iterator[str]:
    """The container's application-role connection string.

    The migrations are applied once per session; the yielded string names the
    application role, so a database built from it runs under the RLS policies
    the migrations enable.
    """
    yield _app_dsn(postgres_admin_dsn)


_GOVERNANCE_TABLES = (
    "outbox_events",
    "outbox_dead_letters",
    "per_tenant_llm_budget",
    "external_egress_policy",
    "grounded_search_daily_usage",
    "companion_suspension_policy",
    "platform_companion_suspension",
    "invoke_debit_claims",
    "platform_egress_killswitch",
)
"""The gateway's governance tables, reset before each test.

``outbox_dead_letters`` is included because it references ``outbox_events`` and
must be truncated alongside it.
"""


async def _reset_governance_tables(admin_dsn: str) -> None:
    """Truncate the governance tables and re-seed the kill switch, as the admin.

    The container is session-scoped, so the schema — and any rows a previous
    test wrote — persist across tests. Each test starts from the empty schema
    the migrations create, so one test's tenant rows cannot leak into another's.
    The truncate runs as the admin (a superuser), which bypasses the RLS
    policies the application role is subject to.
    """
    connection = await asyncpg.connect(admin_dsn)
    try:
        await connection.execute(f"TRUNCATE {', '.join(_GOVERNANCE_TABLES)}")
        await connection.execute(
            "INSERT INTO platform_egress_killswitch (singleton, engaged) VALUES (TRUE, FALSE)"
        )
    finally:
        await connection.close()


@pytest.fixture
async def postgres_db(
    postgres_dsn: str, postgres_admin_dsn: str
) -> AsyncIterator[Database]:
    """A governance Database connected to the test container's application role.

    The governance tables are reset before each test, so a test sees only the
    rows it writes itself.
    """
    await _reset_governance_tables(postgres_admin_dsn)
    db = Database(postgres_dsn)
    await db.connect()
    try:
        yield db
    finally:
        await db.close()
