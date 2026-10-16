"""Real-Postgres integration tests for the governance mechanisms.

The governance unit tests run against stub connections, which pin the gateway's
discipline but not the database behaviour the parity review flagged: row-level
security, the ``FOR UPDATE`` read lock, ``ON CONFLICT DO NOTHING`` dedupe and the
claim race are properties of Postgres, not of the gateway's SQL. These tests run
the real mechanisms against a real Postgres container — the chora-observability
migrations applied, the application role connected under RLS — so the settlement
and idempotency claims are checked against the database's own concurrency and
isolation rather than a stub's script.

The container and schema come from :mod:`conftest_governance`; each test gets its
own connection pool over the shared schema and uses distinct tenant ids for
isolation.
"""

from __future__ import annotations

import asyncio

import grpc
import pytest

from model_gateway.config import Settings
from model_gateway.domain import GroundedUsage
from model_gateway.governance import (
    BudgetDecision,
    BudgetManager,
    Database,
    EgressGate,
    GovernanceManager,
    IdempotencyClaims,
    ManaMeter,
    ManaResult,
    Outbox,
    SuspensionChecker,
    decide,
)
from model_gateway.governance.db import rows_affected
from model_gateway.governance.outbox import build_token_usage_payload, token_usage_envelope
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc.grounded_search import GroundedSearch
from model_gateway.grpc.identity import Authenticator
from model_gateway.providers.exa import ExaConfig, GroundedError
from model_gateway.security.api_keys import ApiKeySet

pytestmark = pytest.mark.postgres

_TENANT_A = "123e4567-e89b-42d3-a456-426614174000"
_TENANT_B = "123e4567-e89b-42d3-a456-426614174001"
_GCID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

_API_KEY = "sk-gateway-postgres-test"
_API_TENANT = _TENANT_A


class _StubManaClient:
    """A mana client that succeeds every quote and debit, and records its calls."""

    def __init__(self) -> None:
        self.quote_calls: list[tuple[str, str, str]] = []
        self.debit_calls: list[tuple[str, str, str, str]] = []

    async def quote(self, gcid: str, tenant_id: str, action_code: str) -> ManaResult:
        self.quote_calls.append((gcid, tenant_id, action_code))
        return ManaResult(success=True, balance_after_units=100)

    async def debit(
        self, gcid: str, tenant_id: str, action_code: str, idem_key: str
    ) -> ManaResult:
        self.debit_calls.append((gcid, tenant_id, action_code, idem_key))
        return ManaResult(success=True, balance_after_units=90)


def _manager(db: Database, *, mana: ManaMeter | None = None) -> GovernanceManager:
    """The full governance chain over one database, with an optional mana meter."""
    return GovernanceManager(
        db,
        BudgetManager(db),
        Outbox(db),
        IdempotencyClaims(db),
        SuspensionChecker(db),
        EgressGate(db),
        mana=mana,
    )


async def _insert_budget(
    db: Database,
    tenant_id: str,
    *,
    budget_micros: int,
    spent_micros: int,
    policy: str = "block",
) -> None:
    """Insert an active budget window for a tenant, under the tenant's RLS GUCs."""
    async with db.transaction(tenant_id=tenant_id) as connection:
        await connection.execute(
            """
            INSERT INTO per_tenant_llm_budget (
                tenant_id, budget_period_start, budget_period_end,
                budget_usd_micros, spent_usd_micros, policy
            )
            VALUES ($1::uuid, now() - interval '1 day', now() + interval '1 day',
                    $2, $3, $4::llm_budget_policy)
            """,
            tenant_id,
            budget_micros,
            spent_micros,
            policy,
        )


async def _insert_egress_policy(
    db: Database,
    tenant_id: str,
    *,
    enabled: bool = True,
    ceiling: int = 50,
) -> None:
    """Insert an egress entitlement for a tenant, under the tenant's RLS GUCs."""
    async with db.transaction(tenant_id=tenant_id) as connection:
        await connection.execute(
            """
            INSERT INTO external_egress_policy (tenant_id, egress_enabled, daily_call_ceiling)
            VALUES ($1::uuid, $2, $3)
            """,
            tenant_id,
            enabled,
            ceiling,
        )


# ---------------------------------------------------------------------------
# 1. RLS isolation
# ---------------------------------------------------------------------------


class TestRlsIsolation:
    """Row-level security: one tenant's rows are invisible to another's.

    The budget, egress and containment tables key their RLS on
    ``chora.tenant_id`` and the outbox on ``app.current_tenant``; a connection
    that has set another tenant's GUC sees none of them, so a read returns
    nothing and an update matches nothing. These tests connect as the
    application role (no ``BYPASSRLS``), so the policies are the database's own.
    """

    async def test_tenant_a_reads_and_updates_its_own_rows(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)

        window = await BudgetManager(db).read_active_budget(_TENANT_A)

        assert window is not None
        assert window.budget_usd_micros == 1_000_000
        # Tenant A can update its own row: the update matches one row.
        async with db.transaction(tenant_id=_TENANT_A) as connection:
            status = await connection.execute(
                "UPDATE per_tenant_llm_budget SET spent_usd_micros = 5 WHERE tenant_id = $1::uuid",
                _TENANT_A,
            )
        assert rows_affected(status) == 1

    async def test_tenant_b_cannot_read_tenant_a_rows(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)
        await _insert_egress_policy(db, _TENANT_A)
        async with db.transaction(tenant_id=_TENANT_A) as connection:
            await connection.execute(
                """
                INSERT INTO companion_suspension_policy
                    (id, tenant_id, skill_key, engaged, reason, engaged_by)
                VALUES (gen_random_uuid(), $1::uuid, NULL, TRUE, 'containment', $2::uuid)
                """,
                _TENANT_A,
                _GCID,
            )

        # Tenant B's budget read finds nothing: the RLS policy filters A's row.
        assert await BudgetManager(db).read_active_budget(_TENANT_B) is None
        # Tenant B's entitlement read finds nothing.
        async with db.transaction(tenant_id=_TENANT_B) as connection:
            row = await connection.fetchrow(
                "SELECT egress_enabled FROM external_egress_policy WHERE tenant_id = $1::uuid",
                _TENANT_A,
            )
        assert row is None
        # Tenant B's containment read finds nothing, so B is not contained.
        assert (await SuspensionChecker(db).check(_TENANT_B, "")).allowed

    async def test_tenant_b_cannot_update_tenant_a_rows(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)

        async with db.transaction(tenant_id=_TENANT_B) as connection:
            status = await connection.execute(
                "UPDATE per_tenant_llm_budget SET spent_usd_micros = 999999 "
                "WHERE tenant_id = $1::uuid",
                _TENANT_A,
            )

        # The update matched nothing: B cannot spend A's budget.
        assert rows_affected(status) == 0
        window = await BudgetManager(db).read_active_budget(_TENANT_A)
        assert window is not None
        assert window.spent_usd_micros == 0

    async def test_outbox_rows_are_isolated_by_tenant(self, postgres_db: Database) -> None:
        db = postgres_db
        outbox = Outbox(db)
        payload = build_token_usage_payload(
            usage_id="usage-rls",
            tenant_id=_TENANT_A,
            gcid=_GCID,
            model_id="model",
            invocation_id="inv-rls",
        )
        envelope = token_usage_envelope(usage_id="usage-rls", tenant_id=_TENANT_A, gcid=_GCID)
        assert await outbox.insert_token_usage("usage-rls", _TENANT_A, _GCID, payload, envelope)

        # Tenant B's outbox read finds nothing: the policy keys on app.current_tenant.
        async with db.transaction(tenant_id=_TENANT_B) as connection:
            row = await connection.fetchrow(
                "SELECT id FROM outbox_events WHERE id = $1", "usage-rls"
            )
        assert row is None


# ---------------------------------------------------------------------------
# 2. Budget admission and settlement
# ---------------------------------------------------------------------------


class TestBudgetAdmission:
    """The budget window's decision and settlement, against real Postgres."""

    async def test_admission_decision_and_debit(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)
        budget = BudgetManager(db)

        window = await budget.read_active_budget(_TENANT_A)
        assert window is not None
        assert window.spent_usd_micros == 0
        assert decide(window) is BudgetDecision.ALLOW

        await budget.debit(_TENANT_A, 50_000)

        window = await budget.read_active_budget(_TENANT_A)
        assert window is not None
        assert window.spent_usd_micros == 50_000

    async def test_an_exhausted_window_blocks(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=1_000_000)

        window = await BudgetManager(db).read_active_budget(_TENANT_A)

        assert window is not None
        assert decide(window) is BudgetDecision.BLOCK


# ---------------------------------------------------------------------------
# 3. Outbox insert and idempotency
# ---------------------------------------------------------------------------


class TestOutboxIdempotency:
    """The outbox insert is the idempotency guard: one usage, one row."""

    async def test_insert_then_redelivery_does_not_duplicate(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        outbox = Outbox(db)
        payload = build_token_usage_payload(
            usage_id="usage-1",
            tenant_id=_TENANT_A,
            gcid=_GCID,
            model_id="model",
            invocation_id="inv-1",
        )
        envelope = token_usage_envelope(usage_id="usage-1", tenant_id=_TENANT_A, gcid=_GCID)

        assert await outbox.insert_token_usage("usage-1", _TENANT_A, _GCID, payload, envelope)

        async with db.transaction(tenant_id=_TENANT_A) as connection:
            row = await connection.fetchrow(
                "SELECT idempotency_key, aggregate_type FROM outbox_events WHERE id = $1",
                "usage-1",
            )
        assert row is not None
        assert row["idempotency_key"] == "usage-1"
        assert row["aggregate_type"] == "token_usage"

        # A redelivery of the same usage inserts nothing and reports False.
        assert not await outbox.insert_token_usage(
            "usage-1", _TENANT_A, _GCID, payload, envelope
        )

        async with db.transaction(tenant_id=_TENANT_A) as connection:
            count = await connection.fetchval(
                "SELECT count(*) FROM outbox_events WHERE id = $1", "usage-1"
            )
        assert count == 1


# ---------------------------------------------------------------------------
# 4. Idempotency claims
# ---------------------------------------------------------------------------


class TestIdempotencyClaims:
    """One dispatch key, one debit: the first claim wins, later ones are counted."""

    async def test_claim_wins_then_dedupe_count_increments(self, postgres_db: Database) -> None:
        db = postgres_db
        claims = IdempotencyClaims(db)

        assert await claims.claim(_GCID, "dispatch-key-1", "companion_chat", "inv-1")

        row = await db.fetch_one(
            """
            SELECT dedupe_count, last_deduped_invocation_id
            FROM invoke_debit_claims
            WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3
            """,
            _GCID,
            "dispatch-key-1",
            "companion_chat",
        )
        assert row is not None
        assert row["dedupe_count"] == 0

        # A redelivery of the same dispatch key loses and is counted on the winner.
        assert not await claims.claim(_GCID, "dispatch-key-1", "companion_chat", "inv-2")

        row = await db.fetch_one(
            """
            SELECT dedupe_count, last_deduped_invocation_id
            FROM invoke_debit_claims
            WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3
            """,
            _GCID,
            "dispatch-key-1",
            "companion_chat",
        )
        assert row is not None
        assert row["dedupe_count"] == 1
        assert row["last_deduped_invocation_id"] == "inv-2"


# ---------------------------------------------------------------------------
# 5. Egress authorization
# ---------------------------------------------------------------------------


class TestEgressAuthorization:
    """The fail-closed egress gate: kill switch, entitlement, daily ceiling."""

    async def test_authorize_then_record_increments_the_daily_counter(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT_A, enabled=True, ceiling=50)
        gate = EgressGate(db)

        assert (await gate.authorize(_TENANT_A)).allowed

        await gate.record_egress(_TENANT_A)

        async with db.transaction(tenant_id=_TENANT_A) as connection:
            count = await connection.fetchval(
                """
                SELECT call_count FROM grounded_search_daily_usage
                WHERE tenant_id = $1::uuid AND usage_date = current_date
                """,
                _TENANT_A,
            )
        assert count == 1
        # Still under the ceiling.
        assert (await gate.authorize(_TENANT_A)).allowed

    async def test_a_daily_ceiling_refuses_the_next_call(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT_A, enabled=True, ceiling=2)
        gate = EgressGate(db)

        await gate.record_egress(_TENANT_A)
        await gate.record_egress(_TENANT_A)

        decision = await gate.authorize(_TENANT_A)

        assert not decision.allowed
        assert decision.reason == "daily_ceiling_exceeded"

    async def test_a_disabled_entitlement_refuses(self, postgres_db: Database) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT_A, enabled=False)

        decision = await EgressGate(db).authorize(_TENANT_A)

        assert not decision.allowed
        assert decision.reason == "external_egress_disabled"

    async def test_an_engaged_kill_switch_refuses_every_tenant(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT_A, enabled=True, ceiling=50)
        async with db.transaction() as connection:
            await connection.execute(
                "UPDATE platform_egress_killswitch SET engaged = TRUE"
            )

        decision = await EgressGate(db).authorize(_TENANT_A)

        assert not decision.allowed
        assert decision.reason == "kill_switch_engaged"


# ---------------------------------------------------------------------------
# 6. Reconciliation invariant
# ---------------------------------------------------------------------------


class TestReconciliationInvariant:
    """N invocations produce N outbox rows and N mana debits, with no unmetered path."""

    async def test_n_invocations_settle_n_outbox_rows_and_n_mana_debits(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        cost_micros = 1_000
        n = 5

        for index in range(n):
            decision = await manager.authorize_invoke(
                _TENANT_A, _GCID, "companion_chat", f"dispatch-{index}", "qgen", f"inv-{index}"
            )
            assert decision.allowed
            await manager.settle_invoke(
                tenant_id=_TENANT_A,
                gcid=_GCID,
                action_code="companion_chat",
                dispatch_key=f"dispatch-{index}",
                invocation_id=f"inv-{index}",
                usage_id=f"usage-{index}",
                cost_micros=cost_micros,
                payload=b"{}",
                envelope={"invocation_id": f"inv-{index}"},
            )

        # N outbox rows, one per invocation, each keyed by its own usage id.
        row = await db.fetch_one(
            "SELECT count(*) AS n, count(DISTINCT idempotency_key) AS keys "
            "FROM outbox_events WHERE tenant_id = $1",
            _TENANT_A,
        )
        assert row is not None
        assert row["n"] == n
        assert row["keys"] == n
        # N mana debits, each under its own invocation id as the idempotency key.
        assert len(mana_client.debit_calls) == n
        assert len({call[3] for call in mana_client.debit_calls}) == n
        # N budget debits: the spent amount is exactly N times the per-call cost.
        window = await BudgetManager(db).read_active_budget(_TENANT_A)
        assert window is not None
        assert window.spent_usd_micros == n * cost_micros


# ---------------------------------------------------------------------------
# 7. Concurrent budget settlement
# ---------------------------------------------------------------------------


class TestConcurrentSettlement:
    """Concurrent invocations are both authorized and both settle: a bounded overshoot.

    The budget read locks ``FOR UPDATE`` for the read only; settlement runs after
    the provider call, outside the lock. Two concurrent invocations of one tenant
    may therefore both be authorized against the same snapshot and both settle
    afterwards — the budget absorbs the overshoot (the Go gateway's own
    behaviour) rather than preventing it. The concurrent settlements must still
    lose no update: the row lock serializes the debits.
    """

    async def test_two_concurrent_invocations_both_authorized_and_settled(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT_A, budget_micros=100_000_000, spent_micros=90_000_000)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))

        async def authorize(index: int) -> None:
            decision = await manager.authorize_invoke(
                _TENANT_A, _GCID, "companion_chat", f"dispatch-{index}", "qgen", f"inv-{index}"
            )
            assert decision.allowed

        async def settle(index: int) -> None:
            await manager.settle_invoke(
                tenant_id=_TENANT_A,
                gcid=_GCID,
                action_code="companion_chat",
                dispatch_key=f"dispatch-{index}",
                invocation_id=f"inv-{index}",
                usage_id=f"usage-{index}",
                cost_micros=20_000_000,
                payload=b"{}",
                envelope={"invocation_id": f"inv-{index}"},
            )

        # Both authorizations run concurrently against the same $90-of-$100 snapshot:
        # the FOR UPDATE read lock serializes the reads, so both see $90 and both pass.
        await asyncio.gather(authorize(1), authorize(2))
        # Both settlements run concurrently; the row lock serializes the debits.
        await asyncio.gather(settle(1), settle(2))

        window = await BudgetManager(db).read_active_budget(_TENANT_A)

        assert window is not None
        # The bounded overshoot: $90 + $20 + $20 = $130, both served, no lost update.
        assert window.spent_usd_micros == 130_000_000
        assert len(mana_client.debit_calls) == 2


# ---------------------------------------------------------------------------
# 8. Partial failure settlement
# ---------------------------------------------------------------------------


class _AbortedCall(Exception):
    """The abort a real ServicerContext raises to terminate the handler."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class _FakeServicerContext:
    """A minimal grpc.aio ServicerContext: invocation metadata and abort."""

    def __init__(self, metadata: tuple[tuple[str, str], ...]) -> None:
        self._metadata = metadata
        self.aborted: tuple[grpc.StatusCode, str] | None = None

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.aborted = (code, details)
        raise _AbortedCall(code, details)


class _FailingAdapter:
    """A grounded adapter that raises the error it was given."""

    def __init__(self, error: GroundedError) -> None:
        self._error = error

    async def search(self, request: object) -> None:
        raise self._error


def _grounded_handler(
    db: Database,
    adapter: _FailingAdapter,
    *,
    mana: ManaMeter | None = None,
) -> GroundedSearch:
    """A GroundedSearch handler wired to a scripted adapter and a governance chain."""
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_API_KEY}:{_API_TENANT}",
        }
    )
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )
    handler = GroundedSearch(
        authenticator,
        ExaConfig(
            api_key="sk-exa-test",
            synthesis_model="synthesis-model",
            synthesis_endpoint="https://synthesis.example/v1",
            synthesis_api_key="sk-synthesis-test",
        ),
        governance=_manager(db, mana=mana),
    )
    handler._adapter = adapter
    return handler


class TestPartialFailureSettlement:
    """A failure after billable work settles the partial spend, and nothing else.

    The Exa search succeeded and billed but the synthesis failed, so the
    provider charged for work the gateway owes for. The settlement is the atomic
    pair a served search writes — the budget debit and the outbox row in one
    transaction — and nothing more: no mana debit (the call never completed)
    and no egress count (no answer was produced).
    """

    async def test_a_grounded_failure_with_cost_settles_budget_and_outbox_only(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT_A, enabled=True, ceiling=50)
        await _insert_budget(db, _TENANT_A, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        adapter = _FailingAdapter(
            GroundedError(
                "vendor_unavailable",
                "the synthesis call failed",
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
            )
        )
        handler = _grounded_handler(db, adapter, mana=ManaMeter(mana_client))
        request = pb.GroundedSearchRequest(
            invocation_id="inv-partial-1",
            tenant_id=_TENANT_A,
            gcid=_GCID,
            agent_id="familiar_seeker",
            crew_kind="familiar",
            directive="What is the capital of France?",
            max_results=5,
            action_code="external_egress.search",
        )
        context = _FakeServicerContext((("authorization", f"Bearer {_API_KEY}"),))

        with pytest.raises(_AbortedCall) as error:
            await handler.grounded_search(request, context)

        # The vendor's own error is the caller's answer, unchanged.
        assert error.value.code == grpc.StatusCode.UNAVAILABLE
        # The partial spend settled: the budget was debited by the search's cost.
        window = await BudgetManager(db).read_active_budget(_TENANT_A)
        assert window is not None
        assert window.spent_usd_micros == 10_000
        # The outbox row was inserted under the invocation id.
        row = await db.fetch_one(
            "SELECT id FROM outbox_events WHERE id = $1", "inv-partial-1"
        )
        assert row is not None
        # The failed call earned neither a mana debit nor an egress count.
        assert mana_client.debit_calls == []
        async with db.transaction(tenant_id=_TENANT_A) as connection:
            count = await connection.fetchval(
                "SELECT count(*) FROM grounded_search_daily_usage WHERE tenant_id = $1::uuid",
                _TENANT_A,
            )
        assert count == 0
