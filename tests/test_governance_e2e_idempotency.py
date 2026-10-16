"""End-to-end governance: the real transport, the real ledger, one charge per dispatch.

The governance mechanisms are tested in isolation against stub connections, and
:mod:`tests.test_governance_postgres_integration` runs them against a real
Postgres. Neither crosses the boundary this module crosses: a caller sending a
request through the real gRPC listener, whose answer is settled against a real
Postgres ledger. That is the composition a production deployment runs, and the
question the parity review left open — can a successful response be produced
without the intended durable usage record, or can one dispatch be charged twice
after a retry — is a question about that composition rather than about any one
mechanism.

Every test drives the gateway through :class:`~model_gateway.grpc.server.
GrpcServer`, the same listener the application starts, wired to a
:class:`~model_gateway.governance.manager.GovernanceManager` over the database
the ``postgres_db`` fixture hands it. The mana meter is a stub client, because
the identity service is not part of this process; what is asserted about the
charge reads the database's own rows and the stub's own call record, never the
gateway's intent.

The tests state the two halves of the invariant. A served invocation writes
exactly one outbox row and debits the budget once; a redelivery of the same
dispatch key is served but settles nothing more; a refused invocation settles
nothing at all; and a failure after billable work settles exactly the partial
spend it incurred and no more.

The mana debit is a separate invariant, because it follows the committed
settlement rather than sharing it. A debit that fails after the budget and
outbox transaction commits is logged, never raised: the call has already been
served, and an unrecorded mana charge is a bookkeeping outcome rather than a
reason to un-serve it. A redelivery of the same dispatch key is still served,
but settles nothing — no second budget debit, no second outbox row and no second
mana debit — so the retry can neither double-charge the learner nor blindly
re-attempt the charge that never landed.
"""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager
from dataclasses import replace

import asyncpg
import grpc
import orjson
import pytest
from structlog.testing import capture_logs

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import GroundedFinishReason, GroundedSearchResult, GroundedUsage, Usage
from model_gateway.governance import (
    BudgetManager,
    Database,
    EgressGate,
    GovernanceManager,
    IdempotencyClaims,
    ManaMeter,
    ManaResult,
    Outbox,
    SuspensionChecker,
)
from model_gateway.governance.observability.v1 import token_usage_pb2
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.server import GrpcServer
from model_gateway.inference import InferenceService
from model_gateway.providers.exa import GroundedError
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

pytestmark = [pytest.mark.postgres, pytest.mark.e2e]

_HOST = "127.0.0.1"

_TENANT = "123e4567-e89b-42d3-a456-426614174000"
"""The tenant the API key is bound to; every settlement is keyed on it."""

_GCID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
"""The learner the mana meter and the dispatch-claim ledger are keyed by."""

_API_KEY = "sk-gateway-e2e-idempotency"
"""The credential the stub dials with; bound to :data:`_TENANT` alone."""

_PUBLIC_MODEL = "text-default"
_UPSTREAM_MODEL = "upstream-text"
_UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"

_ACTION_CODE = "companion_chat"
"""The priced action the invoke is metered under; fail-closed by prefix."""

_ESTIMATED_COST = 0.001
_COST_MICROS = 1_000
"""The price the costed mock reports, in dollars and in the micros it settles to."""


# ---------------------------------------------------------------------------
# The stub mana client
# ---------------------------------------------------------------------------


class _StubManaClient:
    """A mana client that succeeds every quote and debit, and records its calls.

    The identity service is a separate process, so the meter is driven through
    this seam: a quote always reports the learner can afford the action and a
    debit always succeeds, and every call is recorded so a test asserts exactly
    what was charged — once, under which invocation id — rather than trusting
    the gateway's intent.
    """

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


class _FailingManaClient(_StubManaClient):
    """A stub mana client whose quotes succeed but whose debits raise.

    The crash boundary under test sits between the committed settlement and the
    external charge, so the seam must fail exactly there: the quote passes (the
    learner is admitted) and the debit raises (the charge never lands). The
    parent's call record still stands, so a test asserts the debit was attempted
    — and failed — rather than trusting the gateway's intent.
    """

    async def debit(
        self, gcid: str, tenant_id: str, action_code: str, idem_key: str
    ) -> ManaResult:
        self.debit_calls.append((gcid, tenant_id, action_code, idem_key))
        raise RuntimeError("the mana service is unreachable")


class _CostedMockProvider(MockProvider):
    """A mock provider whose answer carries a price, so settlement has a cost.

    The gateway's budget is debited in integer micros of a USD amount the
    canonical answer reports, and the plain mock reports no price; this one
    attaches one to exercise the debit through the invocation path.
    """

    async def complete(self, route, request, context):
        response = await super().complete(route, request, context)
        return replace(response, estimated_cost=_ESTIMATED_COST, cost_currency="USD")


class _ScriptedAdapter:
    """An Exa adapter double that raises the grounded failure it was given."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls: list[object] = []

    async def search(self, grounded_request) -> object:
        self.calls.append(grounded_request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """A port nothing else holds, for the listener to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _text_registry() -> ModelRegistry:
    """A registry whose one text route can honour a chat invocation."""
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=_UPSTREAM_ENDPOINT,
                base_url=_UPSTREAM_ENDPOINT,
                model_id=_UPSTREAM_MODEL,
                format="openai",
                api_key="sk-upstream-secret",
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


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


@asynccontextmanager
async def _serve(service, registry: ModelRegistry, governance: GovernanceManager):
    """A real listener on the production wiring, and a stub dialled at its port.

    The listener is the application's own :class:`~model_gateway.grpc.server.
    GrpcServer`, started on an ephemeral loopback port and reached through the
    generated stub, so what crosses the wire is the production transport path:
    the authenticator the settings build, the inference service the test wired
    to governance, and the dispatcher that serves each RPC.
    """
    settings = Settings.from_env(
        {
            "CHORA_GRPC_PORT": str(_free_port()),
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_API_KEY}:{_TENANT}",
        }
    )
    server = GrpcServer(
        service=service,
        registry=registry,
        settings=settings,
        api_keys=ApiKeySet.from_settings(settings),
        governance=governance,
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"{_HOST}:{server.port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel)
    finally:
        await channel.close()
        await server.stop()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _user_message(text: str) -> pb.Message:
    return pb.Message(role=pb.ROLE_USER, content=(pb.ContentPart(text=pb.TextPart(text=text)),))


def _invoke_request(**overrides) -> pb.InvokeRequest:
    """An Invoke request the governance chain can settle: stamped and metered.

    ``surface`` is stated because an unstamped turn is refused before any
    provider is called, and ``gcid``/``action_code`` because the mana debit and
    the dispatch claim are keyed by them.
    """
    fields = {
        "model": _PUBLIC_MODEL,
        "messages": (_user_message("Explain recursion."),),
        "response_modality": pb.RESPONSE_MODALITY_TEXT,
        "tenant_id": _TENANT,
        "gcid": _GCID,
        "action_code": _ACTION_CODE,
        "surface": "qgen",
    }
    fields.update(overrides)
    return pb.InvokeRequest(**fields)


def _grounded_request(**overrides) -> pb.GroundedSearchRequest:
    fields = {
        "invocation_id": "inv-grounded-e2e",
        "tenant_id": _TENANT,
        "gcid": _GCID,
        "agent_id": "familiar_seeker",
        "crew_kind": "familiar",
        "directive": "What is the capital of France?",
        "max_results": 5,
        "action_code": "external_egress.search",
    }
    fields.update(overrides)
    return pb.GroundedSearchRequest(**fields)


def _bearer() -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {_API_KEY}"),)


# ---------------------------------------------------------------------------
# Database seeding and reading
# ---------------------------------------------------------------------------


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


async def _outbox_rows(db: Database, tenant_id: str) -> list[asyncpg.Record]:
    """The tenant's outbox rows, read through the tenant GUC the RLS policy keys on."""
    async with db.transaction(tenant_id=tenant_id) as connection:
        # ``outbox_events.tenant_id`` is TEXT, so the id is compared as text.
        return await connection.fetch(
            "SELECT id, idempotency_key, payload, envelope FROM outbox_events "
            "WHERE tenant_id = $1 ORDER BY occurred_at",
            tenant_id,
        )


async def _spent_micros(db: Database, tenant_id: str) -> int:
    """The tenant's active-window spend, or zero when it has no active window."""
    window = await BudgetManager(db).read_active_budget(tenant_id)
    return window.spent_usd_micros if window is not None else 0


def _envelope(row: asyncpg.Record) -> dict[str, str]:
    """The stored envelope map, whether the driver handed back text or a mapping."""
    stored = row["envelope"]
    if isinstance(stored, str | bytes | bytearray):
        return orjson.loads(stored)
    return dict(stored)


async def _claim_row(
    db: Database, gcid: str, dispatch_key: str, action_code: str
) -> asyncpg.Record | None:
    """The dispatch-claim row for one key, or ``None`` when the key was never claimed."""
    return await db.fetch_one(
        """
        SELECT dedupe_count, last_deduped_invocation_id
        FROM invoke_debit_claims
        WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3
        """,
        gcid,
        dispatch_key,
        action_code,
    )


# ---------------------------------------------------------------------------
# 1. A successful invocation produces its durable usage record
# ---------------------------------------------------------------------------


class TestSuccessfulInvocationProducesDurableUsageRecord:
    """A served invoke over the wire leaves exactly one outbox row and one debit."""

    async def test_a_served_invoke_records_one_usage_and_debits_the_budget(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider(MockSpec(usage=Usage(input_tokens=10, output_tokens=20)))
        service = InferenceService(_text_registry(), provider, governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            response = await stub.Invoke(
                _invoke_request(invocation_id="inv-e2e-1"), metadata=_bearer()
            )

        # The caller read a successful answer carrying its own invocation id.
        assert response.invocation_id == "inv-e2e-1"
        assert response.model == _PUBLIC_MODEL
        assert len(provider.calls) == 1

        # Exactly one durable usage record landed, and its idempotency key is its
        # own usage id — the row a redelivery would collide with.
        rows = await _outbox_rows(db, _TENANT)
        assert len(rows) == 1
        row = rows[0]
        assert row["idempotency_key"] == row["id"]
        event = token_usage_pb2.TokenUsageRecorded.FromString(row["payload"])
        assert event.usage_id == row["id"]
        assert event.invocation_id == "inv-e2e-1"
        assert event.tenant_id == _TENANT
        assert event.input_tokens == 10
        assert event.output_tokens == 20
        assert event.cost_micros == _COST_MICROS

        # The budget was debited by the answer's own price, and the learner was
        # charged once under the invocation id the answer carries.
        assert await _spent_micros(db, _TENANT) == _COST_MICROS
        assert mana_client.debit_calls == [(_GCID, _TENANT, _ACTION_CODE, "inv-e2e-1")]


# ---------------------------------------------------------------------------
# 2. A retry under the same dispatch key does not double-charge
# ---------------------------------------------------------------------------


class TestRetryWithSameDispatchKeyDoesNotDoubleCharge:
    """A redelivered dispatch is served, but settles nothing a second time."""

    async def test_a_redelivered_dispatch_is_served_but_charged_once(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider()
        service = InferenceService(_text_registry(), provider, governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            first = await stub.Invoke(
                _invoke_request(
                    invocation_id="inv-first", dispatch_idempotency_key="dispatch-e2e-1"
                ),
                metadata=_bearer(),
            )
            rows_after_first = await _outbox_rows(db, _TENANT)
            spent_after_first = await _spent_micros(db, _TENANT)
            # The retry is a fresh invocation carrying the same dispatch key, the
            # way an at-least-once redelivery repeats one turn.
            second = await stub.Invoke(
                _invoke_request(
                    invocation_id="inv-second", dispatch_idempotency_key="dispatch-e2e-1"
                ),
                metadata=_bearer(),
            )

        # Both turns were served — the redelivery is answered, not refused.
        assert first.invocation_id == "inv-first"
        assert second.invocation_id == "inv-second"
        assert len(provider.calls) == 2

        # But the ledger holds one row, one debit and one charge: the retry added
        # nothing to any of them.
        assert len(rows_after_first) == 1
        assert len(await _outbox_rows(db, _TENANT)) == 1
        assert spent_after_first == _COST_MICROS
        assert await _spent_micros(db, _TENANT) == _COST_MICROS
        assert mana_client.debit_calls == [(_GCID, _TENANT, _ACTION_CODE, "inv-first")]

        # The dedupe is recorded on the winning claim row, which is the ledger's
        # own proof that the key billed once and the redelivery was counted.
        claim = await _claim_row(db, _GCID, "dispatch-e2e-1", _ACTION_CODE)
        assert claim is not None
        assert claim["dedupe_count"] == 1
        assert claim["last_deduped_invocation_id"] == "inv-second"

        # The outbox row the first call wrote carries the dispatch key it was
        # billed under, so the durable record names the dispatch it belongs to.
        row = (await _outbox_rows(db, _TENANT))[0]
        assert _envelope(row)["dispatch_key"] == "dispatch-e2e-1"


# ---------------------------------------------------------------------------
# 3. A refused invocation produces no usage record
# ---------------------------------------------------------------------------


class TestFailedInvocationProducesNoUsageRecord:
    """A refusal before the provider settles nothing at all."""

    async def test_an_exhausted_budget_is_refused_without_a_usage_record(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=1_000_000)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider()
        service = InferenceService(_text_registry(), provider, governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(_invoke_request(invocation_id="inv-denied"), metadata=_bearer())

        # The caller read the governance refusal, and the provider was never reached.
        assert error.value.code() is grpc.StatusCode.FAILED_PRECONDITION
        assert "budget_exhausted" in error.value.details()
        assert provider.calls == []

        # Nothing settled: no outbox row, no further budget spend, no mana debit.
        assert await _outbox_rows(db, _TENANT) == []
        assert await _spent_micros(db, _TENANT) == 1_000_000
        assert mana_client.debit_calls == []


# ---------------------------------------------------------------------------
# 4. A failure after billable work produces the partial settlement only
# ---------------------------------------------------------------------------


class TestPartialFailureProducesPartialSettlement:
    """A synthesis failure after a billed search settles the partial spend, and no more."""

    async def test_a_synthesis_failure_settles_the_partial_spend_only(
        self, postgres_db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT, enabled=True, ceiling=50)
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        adapter = _ScriptedAdapter(
            GroundedError(
                "vendor_unavailable",
                "the synthesis call failed",
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
            )
        )
        # The adapter is stood in where the listener would build it from its Exa
        # config, so what is scripted is the vendor's answer, not the transport.
        monkeypatch.setattr(
            "model_gateway.grpc.grounded_search.ExaGroundedSearch",
            lambda _config: adapter,
        )
        service = InferenceService(_text_registry(), _CostedMockProvider(), governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        # The vendor's own error is the caller's answer, with the partial spend
        # named so the caller can account the cost already incurred.
        assert error.value.code() is grpc.StatusCode.UNAVAILABLE
        assert "vendor_unavailable" in error.value.details()
        assert "partial_cost_micros" in error.value.details()
        assert len(adapter.calls) == 1

        # The partial spend settled: one outbox row under the invocation id and
        # the budget debited by exactly the search's cost.
        rows = await _outbox_rows(db, _TENANT)
        assert len(rows) == 1
        event = token_usage_pb2.TokenUsageRecorded.FromString(rows[0]["payload"])
        assert event.invocation_id == "inv-grounded-e2e"
        assert event.cost_micros == 10_000
        assert await _spent_micros(db, _TENANT) == 10_000

        # But the failed call earned no mana debit: the learner is not charged for
        # an answer that was never produced.
        assert mana_client.debit_calls == []


# ---------------------------------------------------------------------------
# 5. A failed mana debit after settlement is logged, never un-served
# ---------------------------------------------------------------------------


class TestManaDebitFailureAfterSettlement:
    """A debit that fails after the committed settlement leaves the call served."""

    async def test_a_failed_mana_debit_is_logged_and_the_call_still_served(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _FailingManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider(MockSpec(usage=Usage(input_tokens=10, output_tokens=20)))
        service = InferenceService(_text_registry(), provider, governance=manager)

        with capture_logs() as logs:
            async with _serve(service, _text_registry(), manager) as stub:
                response = await stub.Invoke(
                    _invoke_request(invocation_id="inv-e2e-mana-fail"), metadata=_bearer()
                )

        # The caller read a successful answer: the failed debit is a bookkeeping
        # outcome, never a reason to un-serve a call that was already served.
        assert response.invocation_id == "inv-e2e-mana-fail"
        assert response.model == _PUBLIC_MODEL
        assert len(provider.calls) == 1

        # The budget was debited and the outbox row landed, in the transaction
        # that committed before the debit was attempted.
        assert await _spent_micros(db, _TENANT) == _COST_MICROS
        rows = await _outbox_rows(db, _TENANT)
        assert len(rows) == 1
        row = rows[0]
        assert row["idempotency_key"] == row["id"]
        event = token_usage_pb2.TokenUsageRecorded.FromString(row["payload"])
        assert event.invocation_id == "inv-e2e-mana-fail"
        assert event.cost_micros == _COST_MICROS

        # The mana debit was attempted exactly once, under the invocation id, and
        # it failed.
        assert mana_client.debit_calls == [(_GCID, _TENANT, _ACTION_CODE, "inv-e2e-mana-fail")]

        # The failure is logged loudly rather than raised to the caller.
        recorded = [
            entry
            for entry in logs
            if entry["event"] == "mana debit not recorded after a served call"
        ]
        assert len(recorded) == 1
        assert recorded[0]["result"] == "error"
        assert recorded[0]["invocation_id"] == "inv-e2e-mana-fail"


# ---------------------------------------------------------------------------
# 6. A redelivery does not re-attempt the mana debit
# ---------------------------------------------------------------------------


class TestRedeliveredDispatchDoesNotReAttemptManaDebit:
    """A redelivered dispatch settles nothing, so its mana debit is not repeated."""

    async def test_a_redelivered_dispatch_does_not_re_attempt_the_mana_debit(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider()
        service = InferenceService(_text_registry(), provider, governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            first = await stub.Invoke(
                _invoke_request(
                    invocation_id="inv-mana-first", dispatch_idempotency_key="dispatch-mana-1"
                ),
                metadata=_bearer(),
            )
            second = await stub.Invoke(
                _invoke_request(
                    invocation_id="inv-mana-second", dispatch_idempotency_key="dispatch-mana-1"
                ),
                metadata=_bearer(),
            )

        # Both turns were served, but the mana debit ran once: the redelivery
        # settled nothing and so did not charge the learner a second time.
        assert first.invocation_id == "inv-mana-first"
        assert second.invocation_id == "inv-mana-second"
        assert len(provider.calls) == 2
        assert mana_client.debit_calls == [(_GCID, _TENANT, _ACTION_CODE, "inv-mana-first")]
        assert await _spent_micros(db, _TENANT) == _COST_MICROS
        assert len(await _outbox_rows(db, _TENANT)) == 1


# ---------------------------------------------------------------------------
# 7. A crash between settlement and the mana debit, then a retry
# ---------------------------------------------------------------------------


class TestCrashBetweenSettlementAndManaDebitBillsOnce:
    """A crash after the committed settlement bills the learner exactly once."""

    async def test_a_crash_before_the_mana_debit_does_not_double_charge_on_retry(
        self, postgres_db: Database
    ) -> None:
        db = postgres_db
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _FailingManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        provider = _CostedMockProvider()
        service = InferenceService(_text_registry(), provider, governance=manager)

        with capture_logs() as logs:
            async with _serve(service, _text_registry(), manager) as stub:
                first = await stub.Invoke(
                    _invoke_request(
                        invocation_id="inv-crash-first", dispatch_idempotency_key="dispatch-crash-1"
                    ),
                    metadata=_bearer(),
                )
                rows_after_first = await _outbox_rows(db, _TENANT)
                spent_after_first = await _spent_micros(db, _TENANT)
                second = await stub.Invoke(
                    _invoke_request(
                        invocation_id="inv-crash-second",
                        dispatch_idempotency_key="dispatch-crash-1",
                    ),
                    metadata=_bearer(),
                )

        # The first call was served and its budget + outbox transaction committed
        # before the mana debit was attempted — the crash boundary.
        assert first.invocation_id == "inv-crash-first"
        assert second.invocation_id == "inv-crash-second"
        assert len(provider.calls) == 2
        assert spent_after_first == _COST_MICROS
        assert len(rows_after_first) == 1

        # The debit failed, loudly logged, and the retry was still served.
        assert mana_client.debit_calls == [(_GCID, _TENANT, _ACTION_CODE, "inv-crash-first")]
        recorded = [
            entry
            for entry in logs
            if entry["event"] == "mana debit not recorded after a served call"
        ]
        assert len(recorded) == 1

        # The retry settled nothing: one budget debit, one outbox row and no
        # second mana debit, so the learner is never charged twice for one dispatch.
        assert await _spent_micros(db, _TENANT) == _COST_MICROS
        assert len(await _outbox_rows(db, _TENANT)) == 1


# ---------------------------------------------------------------------------
# 8. A redelivered grounded search does not double-charge
# ---------------------------------------------------------------------------


class TestGroundedSearchRetryDoesNotDoubleCharge:
    """A redelivered grounded search is served, but settles nothing a second time.

    The dispatch-claim mechanism protects the grounded search the same way it
    protects an invoke: the first search with an invocation id claims the
    dispatch, runs the Exa call and settles; a redelivery with the same
    invocation id finds the dispatch already claimed, skips the Exa call and
    returns a deduped response — no second budget debit, no second outbox row,
    no second mana debit.
    """

    async def test_a_redelivered_grounded_search_is_served_but_charged_once(
        self, postgres_db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT, enabled=True, ceiling=50)
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        adapter = _ScriptedAdapter(
            GroundedSearchResult(
                invocation_id="inv-grounded-e2e",
                grounded_answer="Paris is the capital.",
                web_search_queries=("What is the capital of France?",),
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
                vendor="exa",
                model_version="synthesis-model",
                finish_reason=GroundedFinishReason.COMPLETE,
                completed_at="2026-10-16T00:00:00+00:00",
            )
        )
        monkeypatch.setattr(
            "model_gateway.grpc.grounded_search.ExaGroundedSearch",
            lambda _config: adapter,
        )
        service = InferenceService(_text_registry(), _CostedMockProvider(), governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-e2e"), metadata=_bearer()
            )
            rows_after_first = await _outbox_rows(db, _TENANT)
            spent_after_first = await _spent_micros(db, _TENANT)
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-e2e"), metadata=_bearer()
            )

        # Both searches were served — the redelivery is answered, not refused.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == ""
        # The Exa adapter was called only once — the redelivery skipped it.
        assert len(adapter.calls) == 1
        # But the learner was billed once: one outbox row, one budget debit.
        assert len(rows_after_first) == 1
        assert len(await _outbox_rows(db, _TENANT)) == 1
        assert spent_after_first == 10_000
        assert await _spent_micros(db, _TENANT) == 10_000
        assert mana_client.debit_calls == [
            (_GCID, _TENANT, "external_egress.search", "inv-grounded-e2e")
        ]

        # The dedupe is recorded on the winning claim row, which is the ledger's
        # own proof that the key billed once and the redelivery was counted.
        claim = await _claim_row(db, _GCID, "inv-grounded-e2e", "external_egress.search")
        assert claim is not None
        assert claim["dedupe_count"] == 1
        assert claim["last_deduped_invocation_id"] == "inv-grounded-e2e"

    async def test_a_grounded_search_with_a_different_invocation_id_is_charged(
        self, postgres_db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = postgres_db
        await _insert_egress_policy(db, _TENANT, enabled=True, ceiling=50)
        await _insert_budget(db, _TENANT, budget_micros=1_000_000, spent_micros=0)
        mana_client = _StubManaClient()
        manager = _manager(db, mana=ManaMeter(mana_client))
        adapter = _ScriptedAdapter(
            GroundedSearchResult(
                invocation_id="inv-grounded-e2e",
                grounded_answer="Paris is the capital.",
                web_search_queries=("What is the capital of France?",),
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
                vendor="exa",
                model_version="synthesis-model",
                finish_reason=GroundedFinishReason.COMPLETE,
                completed_at="2026-10-16T00:00:00+00:00",
            )
        )
        monkeypatch.setattr(
            "model_gateway.grpc.grounded_search.ExaGroundedSearch",
            lambda _config: adapter,
        )
        service = InferenceService(_text_registry(), _CostedMockProvider(), governance=manager)

        async with _serve(service, _text_registry(), manager) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-2"), metadata=_bearer()
            )

        # Both searches were served and both were charged — different invocation
        # ids are different dispatches.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == "Paris is the capital."
        assert len(adapter.calls) == 2
        assert len(await _outbox_rows(db, _TENANT)) == 2
        assert await _spent_micros(db, _TENANT) == 20_000
        assert len(mana_client.debit_calls) == 2
