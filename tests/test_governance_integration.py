"""The governance chain, composed: the manager, the invoke path, and the settle.

The six mechanisms are tested in isolation elsewhere. What is under test here is
the composition the invocation flow runs: :class:`GovernanceManager` deciding
one invoke against a stub database, the :class:`~model_gateway.inference.
InferenceService` refusing a denied call before any provider is called and
settling a served one, and the settlement producing exactly one outbox row and
one mana debit per served invocation.

The database is a stub connection playing the rows the chain reads; the mana
meter is a stub client; the provider is the gateway's own
:class:`~model_gateway.providers.mock.MockProvider`, so "the provider was never
called" is asserted from the provider's own call record rather than assumed.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import grpc
import pytest

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    GroundedFinishReason,
    GroundedSearchResult,
    GroundedUsage,
    InvocationRequest,
    Message,
    Role,
    TextPart,
    Usage,
)
from model_gateway.errors import (
    FailedPreconditionError,
    ResourceExhaustedError,
    UnavailableError,
)
from model_gateway.governance import (
    BudgetManager,
    BudgetPolicy,
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
from model_gateway.grpc.grounded_search import GroundedSearch
from model_gateway.grpc.identity import Authenticator
from model_gateway.inference import InferenceService
from model_gateway.providers.exa import ExaConfig, GroundedError
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.security.api_keys import ApiKeySet

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_GCID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_BUDGET_ID = uuid.uuid4()


class StubTransaction:
    def __init__(self, connection: StubConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> StubConnection:
        return self._connection

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _PooledConnection:
    def __init__(self, pool: StubPool) -> None:
        self._pool = pool

    async def __aenter__(self) -> StubConnection:
        return self._pool.connection

    async def __aexit__(self, *exc: object) -> bool:
        return False


class StubPool:
    def __init__(self, connection: StubConnection) -> None:
        self.connection = connection

    def acquire(self) -> _PooledConnection:
        return _PooledConnection(self)


class StubManaClient:
    """A mana client replaying a scripted quote and debit, and recording its calls."""

    def __init__(
        self,
        *,
        quote_result: ManaResult | None = None,
        debit_result: ManaResult | None = None,
    ) -> None:
        self.quote_result = quote_result or ManaResult(success=True, balance_after_units=100)
        self.debit_result = debit_result or ManaResult(success=True, balance_after_units=90)
        self.quote_calls: list[tuple[str, str, str]] = []
        self.debit_calls: list[tuple[str, str, str, str]] = []

    async def quote(self, gcid: str, tenant_id: str, action_code: str) -> ManaResult:
        self.quote_calls.append((gcid, tenant_id, action_code))
        return self.quote_result

    async def debit(
        self, gcid: str, tenant_id: str, action_code: str, idem_key: str
    ) -> ManaResult:
        self.debit_calls.append((gcid, tenant_id, action_code, idem_key))
        return self.debit_result


class StubConnection:
    """A pooled-connection stand-in playing the rows the governance chain reads.

    Every statement is recorded as a ``(query, args)`` triple, so a test asserts
    the chain read the tables it was meant to and wrote the rows it was meant to.
    The scripted state is per-table: budget windows, the two containment rows,
    the outbox insert's landing, the debit's row count and the claim's win.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.budget_rows: list[dict[str, object]] = []
        self.platform_suspension_row: dict[str, object] | None = None
        self.tenant_suspension_row: dict[str, object] | None = None
        self.kill_switch_row: dict[str, object] | None = None
        self.entitlement_row: dict[str, object] | None = None
        self.usage_row: dict[str, object] | None = None
        self.outbox_status = "INSERT 0 1"
        self.update_status = "UPDATE 1"
        self.claim_status = "INSERT 0 1"
        self.claim_statuses: list[str] = []
        self.fail_needles: dict[str, Exception] = {}

    def transaction(self, tenant_id: str | None = None) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        for needle, error in self.fail_needles.items():
            if needle in query:
                raise error
        if "invoke_debit_claims" in query and "INSERT" in query:
            if self.claim_statuses:
                return self.claim_statuses.pop(0)
            return self.claim_status
        if "outbox_events" in query:
            return self.outbox_status
        if "per_tenant_llm_budget" in query and "UPDATE" in query:
            return self.update_status
        return "UPDATE 0"

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((query, args))
        for needle, error in self.fail_needles.items():
            if needle in query:
                raise error
        if "platform_companion_suspension" in query:
            return self.platform_suspension_row
        if "companion_suspension_policy" in query:
            return self.tenant_suspension_row
        if "platform_egress_killswitch" in query:
            return self.kill_switch_row
        if "external_egress_policy" in query:
            return self.entitlement_row
        if "grounded_search_daily_usage" in query:
            return self.usage_row
        return None

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.calls.append((query, args))
        if "per_tenant_llm_budget" in query:
            return list(self.budget_rows)
        return []

    # -- assertions on the recorded calls ---------------------------------- #

    def wrote_outbox_rows(self) -> int:
        return len([call for call in self.calls if "INSERT INTO outbox_events" in call[0]])

    def recorded_egress(self) -> int:
        return len(
            [call for call in self.calls if "INSERT INTO grounded_search_daily_usage" in call[0]]
        )

    def debited_budget(self) -> int:
        return len(
            [
                call
                for call in self.calls
                if "UPDATE per_tenant_llm_budget" in call[0] and "spent_usd_micros" in call[0]
            ]
        )

    def claimed_debit(self) -> int:
        return len([call for call in self.calls if "INSERT INTO invoke_debit_claims" in call[0]])

    def egress_reads(self) -> int:
        return len([call for call in self.calls if "platform_egress_killswitch" in call[0]])

    def budget_window_reads(self) -> int:
        return len(
            [
                call
                for call in self.calls
                if "per_tenant_llm_budget" in call[0] and "SELECT" in call[0]
            ]
        )

    def read_containment(self) -> int:
        return len([call for call in self.calls if "companion_suspension" in call[0]])

    def budget_reads(self) -> int:
        return len([call for call in self.calls if "per_tenant_llm_budget" in call[0]])


def _route(**overrides) -> Route:
    values = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.example/v1",
        "base_url": "https://upstream.example/v1",
        "model_id": "upstream-model",
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def _request(**overrides) -> InvocationRequest:
    values = {
        "invocation_id": "inv-1",
        "tenant_id": _TENANT_ID,
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
        "gcid": _GCID,
    }
    values.update(overrides)
    return InvocationRequest(**values)


def _budget_window(
    *,
    budget_micros: int = 1_000_000,
    spent_micros: int = 0,
    policy: BudgetPolicy = BudgetPolicy.BLOCK,
) -> dict[str, object]:
    return {
        "budget_id": _BUDGET_ID,
        "budget_usd_micros": budget_micros,
        "spent_usd_micros": spent_micros,
        "policy": policy.value,
        "downgrade_to_logical_model_id": None,
    }


def _manager(
    connection: StubConnection,
    *,
    mana: ManaMeter | None = None,
) -> GovernanceManager:
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    return GovernanceManager(
        db,
        BudgetManager(db),
        Outbox(db),
        IdempotencyClaims(db),
        SuspensionChecker(db),
        EgressGate(db),
        mana=mana,
    )


def _service(
    provider: MockProvider,
    governance: GovernanceManager | None = None,
) -> InferenceService:
    from model_gateway.registry import ModelRegistry

    registry = ModelRegistry((_route(),))
    return InferenceService(registry, provider, governance=governance)


class _CostedMockProvider(MockProvider):
    """A mock provider whose answer carries a price, so settlement has a cost.

    The gateway's budget is debited in integer micros of a USD amount the
    canonical answer reports; the plain mock reports no price, so this one
    attaches one to exercise the debit through the invoke path.
    """

    async def complete(self, route, request, context):
        from dataclasses import replace

        response = await super().complete(route, request, context)
        return replace(response, estimated_cost=0.001, cost_currency="USD")


# ---------------------------------------------------------------------------
# authorize_invoke: the chain's decisions
# ---------------------------------------------------------------------------


class TestAuthorizeInvoke:
    async def test_a_surface_less_request_is_refused_without_reading_a_table(self) -> None:
        connection = StubConnection()
        manager = _manager(connection)

        decision = await manager.authorize_invoke("", None, None, None, None, "inv-1")

        # An unstamped surface is refused loudly, before any table is read:
        # containment cannot be evaluated for an undeclared turn, and the
        # refusal must not be a silent pass-through.
        assert not decision.allowed
        assert decision.reason == "surface_unstamped"
        assert connection.calls == []

    async def test_a_tenant_with_no_budget_window_is_allowed(self) -> None:
        connection = StubConnection()
        manager = _manager(connection)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, "companion_chat", None, "qgen", "inv-1"
        )

        assert decision.allowed
        # The tenant budget was read and found no window; the fail-open absence.
        # No learner-scoped action ran (the meter is not wired), and no
        # containment read ran (a non-companion surface is not contained).
        assert connection.budget_reads() == 1
        assert connection.read_containment() == 0

    async def test_an_exhausted_budget_window_blocks(self) -> None:
        connection = StubConnection()
        connection.budget_rows = [
            _budget_window(spent_micros=2_000_000, policy=BudgetPolicy.BLOCK)
        ]
        manager = _manager(connection)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, None, None, "qgen", "inv-1"
        )

        assert not decision.allowed
        assert decision.reason == "budget_exhausted"

    async def test_a_contained_companion_surface_is_suspended(self) -> None:
        connection = StubConnection()
        connection.platform_suspension_row = {
            "skill_key": None,
            "reason": "operator containment",
        }
        manager = _manager(connection)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, None, None, "companion_chat", "inv-1"
        )

        assert not decision.allowed
        assert decision.reason == "companion_suspended"
        assert "operator containment" in decision.detail

    async def test_a_non_companion_surface_is_not_read_against_containment(self) -> None:
        connection = StubConnection()
        connection.platform_suspension_row = {
            "skill_key": None,
            "reason": "operator containment",
        }
        manager = _manager(connection)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, None, None, "qgen", "inv-1"
        )

        assert decision.allowed
        assert connection.read_containment() == 0

    async def test_a_redelivered_dispatch_key_is_allowed_through_unbilled(self) -> None:
        connection = StubConnection()
        connection.claim_status = "INSERT 0 0"  # the claim lost the race
        manager = _manager(connection)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, "companion_chat", "dispatch-key-1", "qgen", "inv-2"
        )

        assert decision.allowed
        assert decision.deduped
        assert connection.claimed_debit() == 1

    async def test_a_fail_closed_action_with_a_shortfall_is_refused(self) -> None:
        connection = StubConnection()
        mana = ManaMeter(
            StubManaClient(
                quote_result=ManaResult(
                    success=False, required_units=50, current_balance_units=20
                )
            )
        )
        manager = _manager(connection, mana=mana)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, "companion_chat_turn_standard", None, "qgen", "inv-1"
        )

        assert not decision.allowed
        assert decision.reason == "insufficient_mana"
        assert "required=50" in decision.detail

    async def test_a_fail_open_action_is_never_quoted(self) -> None:
        connection = StubConnection()
        mana_client = StubManaClient()
        mana = ManaMeter(mana_client)
        manager = _manager(connection, mana=mana)

        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, "content_moderation", None, "qgen", "inv-1"
        )

        assert decision.allowed
        assert mana_client.quote_calls == []

    async def test_an_unpriced_fail_closed_action_is_un_metered_not_refused(self) -> None:
        connection = StubConnection()
        mana = ManaMeter(
            StubManaClient(
                quote_result=ManaResult(success=True, balance_after_units=100)
            )
        )
        manager = _manager(connection, mana=mana)

        # An action the catalogue does not price quotes affordable; only a
        # definitive shortfall on a priced action refuses.
        decision = await manager.authorize_invoke(
            _TENANT_ID, _GCID, "coach_session_full", None, "qgen", "inv-1"
        )

        assert decision.allowed


# ---------------------------------------------------------------------------
# settle_invoke: the atomic settlement
# ---------------------------------------------------------------------------


class TestSettleInvoke:
    async def test_a_served_call_writes_one_outbox_row_and_debits_the_budget(self) -> None:
        connection = StubConnection()
        mana_client = StubManaClient()
        manager = _manager(connection, mana=ManaMeter(mana_client))

        await manager.settle_invoke(
            tenant_id=_TENANT_ID,
            gcid=_GCID,
            action_code="companion_chat",
            dispatch_key=None,
            invocation_id="inv-1",
            usage_id="usage-1",
            cost_micros=50_000,
            payload=b'{"tokens": 10}',
            envelope={"invocation_id": "inv-1"},
        )

        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        assert mana_client.debit_calls == [(_GCID, _TENANT_ID, "companion_chat", "inv-1")]

    async def test_a_usage_that_already_landed_settles_nothing(self) -> None:
        connection = StubConnection()
        connection.outbox_status = "INSERT 0 0"  # ON CONFLICT DO NOTHING skipped
        mana_client = StubManaClient()
        manager = _manager(connection, mana=ManaMeter(mana_client))

        await manager.settle_invoke(
            tenant_id=_TENANT_ID,
            gcid=_GCID,
            action_code="companion_chat",
            dispatch_key=None,
            invocation_id="inv-1",
            usage_id="usage-1",
            cost_micros=50_000,
            payload=b"{}",
            envelope={},
        )

        # The row already existed: no budget debit, no mana debit — the caller
        # must not be charged twice for one recorded usage.
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 0
        assert mana_client.debit_calls == []

    async def test_a_zero_cost_call_records_the_usage_without_a_debit(self) -> None:
        connection = StubConnection()
        mana_client = StubManaClient()
        manager = _manager(connection, mana=ManaMeter(mana_client))

        await manager.settle_invoke(
            tenant_id=_TENANT_ID,
            gcid=_GCID,
            action_code=None,
            dispatch_key=None,
            invocation_id="inv-1",
            usage_id="usage-1",
            cost_micros=0,
            payload=b"{}",
            envelope={},
        )

        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 0
        # No action code: un-metered, so no mana debit either.
        assert mana_client.debit_calls == []


# ---------------------------------------------------------------------------
# The invoke path: a denied call never reaches the provider
# ---------------------------------------------------------------------------


class TestInvokeGovernance:
    async def test_a_suspended_call_is_refused_before_the_provider(self) -> None:
        connection = StubConnection()
        connection.platform_suspension_row = {
            "skill_key": None,
            "reason": "operator containment",
        }
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        with pytest.raises(FailedPreconditionError) as error:
            await service.invoke(
                _request(surface="companion_chat"), namespace="ns", tenant=_TENANT_ID
            )

        assert error.value.details["reason"] == "companion_suspended"
        assert provider.calls == []

    async def test_an_unstamped_surface_is_refused_before_the_provider(self) -> None:
        connection = StubConnection()
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        with pytest.raises(FailedPreconditionError) as error:
            await service.invoke(_request(), namespace="ns", tenant=_TENANT_ID)

        # The refusal is loud: an undeclared turn is refused as surface_unstamped
        # by the governance chain, never silently passed as "not a companion
        # turn", and the provider is never reached.
        assert error.value.details["reason"] == "surface_unstamped"
        assert provider.calls == []

    async def test_an_overdrawn_call_is_refused_before_the_provider(self) -> None:
        connection = StubConnection()
        connection.budget_rows = [
            _budget_window(spent_micros=2_000_000, policy=BudgetPolicy.BLOCK)
        ]
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        with pytest.raises(FailedPreconditionError) as error:
            await service.invoke(
                _request(surface="qgen"), namespace="ns", tenant=_TENANT_ID
            )

        assert error.value.details["reason"] == "budget_exhausted"
        assert provider.calls == []

    async def test_an_insufficient_mana_call_is_refused_before_the_provider(self) -> None:
        connection = StubConnection()
        mana = ManaMeter(
            StubManaClient(
                quote_result=ManaResult(
                    success=False, required_units=500, current_balance_units=10
                )
            )
        )
        provider = MockProvider()
        service = _service(provider, _manager(connection, mana=mana))

        with pytest.raises(ResourceExhaustedError) as error:
            await service.invoke(
                _request(action_code="coach_session_full", surface="qgen"),
                namespace="ns",
                tenant=_TENANT_ID,
            )

        assert error.value.details["reason"] == "insufficient_mana"
        assert provider.calls == []

    async def test_an_unreadable_containment_store_is_refused_as_unavailable(self) -> None:
        connection = StubConnection()
        connection.fail_needles["platform_companion_suspension"] = RuntimeError(
            "relation does not exist"
        )
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        with pytest.raises(UnavailableError) as error:
            await service.invoke(
                _request(surface="companion_diagnosis"), namespace="ns", tenant=_TENANT_ID
            )

        assert error.value.details["reason"] == "suspension_unreadable"
        assert provider.calls == []

    async def test_a_served_call_reaches_the_provider_and_settles(self) -> None:
        connection = StubConnection()
        mana_client = StubManaClient()
        provider = _CostedMockProvider(
            MockSpec(usage=Usage(input_tokens=10, output_tokens=20))
        )
        service = _service(provider, _manager(connection, mana=ManaMeter(mana_client)))

        response = await service.invoke(
            _request(action_code="companion_chat", surface="qgen"),
            namespace="ns",
            tenant=_TENANT_ID,
        )

        assert len(provider.calls) == 1
        assert response.invocation_id == "inv-1"
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        assert len(mana_client.debit_calls) == 1

    async def test_a_redelivered_dispatch_is_served_but_settles_nothing(self) -> None:
        connection = StubConnection()
        # The first claim wins the key; the second loses it and is deduped.
        connection.claim_statuses = ["INSERT 0 1", "INSERT 0 0"]
        mana_client = StubManaClient()
        provider = _CostedMockProvider()
        service = _service(provider, _manager(connection, mana=ManaMeter(mana_client)))

        first = await service.invoke(
            _request(
                invocation_id="inv-1",
                action_code="companion_chat",
                dispatch_idempotency_key="dispatch-key-1",
                surface="qgen",
            ),
            namespace="ns",
            tenant=_TENANT_ID,
        )
        second = await service.invoke(
            _request(
                invocation_id="inv-2",
                action_code="companion_chat",
                dispatch_idempotency_key="dispatch-key-1",
                surface="qgen",
            ),
            namespace="ns",
            tenant=_TENANT_ID,
        )

        # Both turns were served — the redelivery is answered, not refused.
        assert first.invocation_id == "inv-1"
        assert second.invocation_id == "inv-2"
        assert len(provider.calls) == 2
        # But the learner was billed once: one claim, one outbox row, one debit.
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        assert len(mana_client.debit_calls) == 1

    async def test_a_settlement_failure_never_fails_the_invocation(self) -> None:
        connection = StubConnection()
        connection.fail_needles["INSERT INTO outbox_events"] = RuntimeError("connection lost")
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        response = await service.invoke(
            _request(surface="qgen"), namespace="ns", tenant=_TENANT_ID
        )

        # The provider was called and its answer returned; the settlement
        # failure was logged, not raised.
        assert len(provider.calls) == 1
        assert response.invocation_id == "inv-1"

    async def test_n_served_invocations_produce_n_outbox_rows_and_n_mana_debits(self) -> None:
        connection = StubConnection()
        mana_client = StubManaClient()
        provider = _CostedMockProvider(
            MockSpec(usage=Usage(input_tokens=5, output_tokens=5))
        )
        service = _service(provider, _manager(connection, mana=ManaMeter(mana_client)))

        for index in range(5):
            await service.invoke(
                _request(
                    invocation_id=f"inv-{index}",
                    action_code="companion_chat",
                    surface="qgen",
                ),
                namespace="ns",
                tenant=_TENANT_ID,
            )

        assert len(provider.calls) == 5
        assert connection.wrote_outbox_rows() == 5
        assert connection.debited_budget() == 5
        assert len(mana_client.debit_calls) == 5

    async def test_a_tenantless_request_is_served_ungoverned(self) -> None:
        connection = StubConnection()
        provider = MockProvider()
        service = _service(provider, _manager(connection))

        response = await service.invoke(
            _request(tenant_id=None, surface="qgen"), namespace="ns", tenant=None
        )

        # No tenant: no budget to read, no outbox to write, no mana to debit —
        # but the call itself is served exactly as it was before governance.
        assert len(provider.calls) == 1
        assert response.invocation_id == "inv-1"
        assert connection.calls == []


# ---------------------------------------------------------------------------
# The grounded-search egress gate, over the gRPC wire
# ---------------------------------------------------------------------------

_GRPC_API_KEY = "sk-gateway-test-key"
_GRPC_TENANT = _TENANT_ID


class _ScriptedAdapter:
    """An Exa adapter double answering with the result it was given."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls: list[object] = []

    async def search(self, grounded_request) -> GroundedSearchResult:
        self.calls.append(grounded_request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _grounded_result() -> GroundedSearchResult:
    return GroundedSearchResult(
        invocation_id="inv-grounded-1",
        grounded_answer="Paris is the capital.",
        web_search_queries=("What is the capital of France?",),
        usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
        vendor="exa",
        model_version="synthesis-model",
        finish_reason=GroundedFinishReason.COMPLETE,
        completed_at="2026-10-16T00:00:00+00:00",
    )


def _grounded_request(**overrides) -> pb.GroundedSearchRequest:
    fields = {
        "invocation_id": "inv-grounded-1",
        "tenant_id": _GRPC_TENANT,
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
    return (("authorization", f"Bearer {_GRPC_API_KEY}"),)


def _grounded_handler(
    connection: StubConnection,
    adapter: _ScriptedAdapter,
    *,
    mana: ManaMeter | None = None,
) -> GroundedSearch:
    """A GroundedSearch handler wired to a scripted adapter and a governance manager."""
    db = Database("postgresql://localhost/chora_observability")
    db._pool = StubPool(connection)  # type: ignore[assignment]
    manager = GovernanceManager(
        db,
        BudgetManager(db),
        Outbox(db),
        IdempotencyClaims(db),
        SuspensionChecker(db),
        EgressGate(db),
        mana=mana,
    )
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_GRPC_API_KEY}:{_GRPC_TENANT}",
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
        governance=manager,
    )
    # The adapter is stood in after construction, the way the listener would
    # have built it from its config, so what is scripted is the adapter's answer.
    handler._adapter = adapter
    return handler


class _GroundedOnlyServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    def __init__(self, handler) -> None:
        self._handler = handler

    async def GroundedSearch(self, req, context):
        return await self._handler.grounded_search(req, context)


@asynccontextmanager
async def _serve_grounded(handler):
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
        _GroundedOnlyServicer(handler), server
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=None)


class TestGroundedSearchGovernance:
    async def test_an_engaged_kill_switch_aborts_before_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": True}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "kill_switch_engaged" in error.value.details()
        assert adapter.calls == []

    async def test_a_disabled_entitlement_aborts_before_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": False,
            "daily_call_ceiling": 50,
        }
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "external_egress_disabled" in error.value.details()
        assert adapter.calls == []

    async def test_an_exhausted_budget_aborts_before_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [
            _budget_window(spent_micros=2_000_000, policy=BudgetPolicy.BLOCK)
        ]
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "budget_exhausted" in error.value.details()
        assert adapter.calls == []

    async def test_a_fail_closed_mana_shortfall_aborts_before_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        mana = ManaMeter(
            StubManaClient(
                quote_result=ManaResult(
                    success=False, required_units=50, current_balance_units=20
                )
            )
        )
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=mana)

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(
                    _grounded_request(action_code="companion_chat"), metadata=_bearer()
                )

        assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "insufficient_mana" in error.value.details()
        assert adapter.calls == []

    async def test_a_daily_ceiling_aborts_as_resource_exhausted(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 50}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "daily_ceiling_exceeded" in error.value.details()
        assert adapter.calls == []

    async def test_an_entitled_search_reaches_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 1}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            response = await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert response.grounded_answer == "Paris is the capital."
        assert len(adapter.calls) == 1


class _CountingConnection(StubConnection):
    """A stub connection whose daily usage counter increments as egress is recorded.

    The base stub plays one static row; the ceiling test needs the counter the
    gate reads to move as the settlement records it, so a tenant that keeps
    calling eventually reaches the ceiling the entitlement named.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entitlement_row = {"egress_enabled": True, "daily_call_ceiling": 2}
        self._count = 0

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        if "grounded_search_daily_usage" in query:
            return {"call_count": self._count}
        return await super().fetchrow(query, *args)

    async def execute(self, query: str, *args: object) -> str:
        if "grounded_search_daily_usage" in query and "INSERT" in query:
            self._count += 1
        return await super().execute(query, *args)


class TestGroundedSearchSettlement:
    """A served grounded search settles the way an invoke does.

    The authorization half is tested above. What is under test here is the
    other half the handler owes: the egress counter incremented, the budget
    debited and the outbox row inserted in one transaction, and the mana
    debit following under the invocation id as its idempotency key — every
    step best-effort, so a settlement failure never fails the search.
    """

    async def test_a_served_search_increments_the_daily_counter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            response = await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert response.grounded_answer == "Paris is the capital."
        assert connection.recorded_egress() == 1

    async def test_a_served_search_writes_one_outbox_row_and_debits_the_budget(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1

    async def test_a_served_search_debits_mana_under_the_invocation_id(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        # The debit is keyed by the invocation id, so a redelivered search
        # collapses onto the ledger row the first settlement wrote, and it
        # carries the tenant the egress is priced under.
        assert mana_client.debit_calls == [
            (_GCID, _GRPC_TENANT, "external_egress.search", "inv-grounded-1")
        ]

    async def test_a_served_search_records_the_model_that_produced_the_tokens(
        self,
    ) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        # The outbox payload is the canonical TokenUsageRecorded protobuf the
        # consumer unmarshals: the model the synthesis leg ran is recorded on
        # it, not left blank.
        insert = next(
            call for call in connection.calls if "INSERT INTO outbox_events" in call[0]
        )
        event = token_usage_pb2.TokenUsageRecorded.FromString(insert[1][3])
        assert event.model_id == "synthesis-model"
        assert event.cost_micros == 10_000
        assert event.tenant_id == _GRPC_TENANT

    async def test_a_failed_search_settles_the_partial_spend_without_mana_or_egress(
        self,
    ) -> None:
        """A synthesis failure after a billed search settles what it owed for.

        The provider charged for the Exa search even though the synthesis
        failed, so the budget is debited and the outbox row recorded under the
        invocation id. What the failed call did not earn is withheld: no mana
        debit (the call never completed) and no egress counter (no answer was
        produced). The vendor's own error still reaches the caller.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(
            GroundedError(
                "vendor_unavailable",
                "the synthesis call failed",
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
            )
        )
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        # The vendor error is the caller's answer, unchanged.
        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        assert "vendor_unavailable" in error.value.details()
        assert "partial_cost_micros" in error.value.details()
        # The partial spend settled: the outbox row landed and the budget was
        # debited — the atomic pair a served search writes.
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        # But the failed call earned neither a mana debit nor an egress count.
        assert mana_client.debit_calls == []
        assert connection.recorded_egress() == 0

    async def test_a_failed_search_without_partial_spend_settles_nothing(self) -> None:
        """A grounded failure carrying no usage settles nothing at all.

        A refusal at the egress gate or the budget decision happens before any
        billable work, so there is no spend to account and the vendor's error is
        the whole of the answer.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(GroundedError("vendor_unavailable", "the search failed"))
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        assert "partial_cost_micros" not in error.value.details()
        assert connection.wrote_outbox_rows() == 0
        assert connection.debited_budget() == 0
        assert mana_client.debit_calls == []
        assert connection.recorded_egress() == 0

    async def test_a_settlement_failure_never_fails_the_search(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.fail_needles["INSERT INTO outbox_events"] = RuntimeError("connection lost")
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            response = await stub.GroundedSearch(_grounded_request(), metadata=_bearer())

        # The answer is returned; the settlement failure was logged, not raised.
        assert response.grounded_answer == "Paris is the capital."
        assert len(adapter.calls) == 1

    async def test_repeated_grounded_calls_eventually_hit_the_configured_ceiling(self) -> None:
        connection = _CountingConnection()
        connection.kill_switch_row = {"engaged": False}
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-2"), metadata=_bearer()
            )
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(
                    _grounded_request(invocation_id="inv-grounded-3"), metadata=_bearer()
                )

        # The first two searches were served and settled; the third was
        # refused at the gate once the counter reached the ceiling of two.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == "Paris is the capital."
        assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "daily_ceiling_exceeded" in error.value.details()
        assert len(adapter.calls) == 2
        assert connection.recorded_egress() == 2

    async def test_an_omitted_invocation_id_is_minted_and_settled_under(self) -> None:
        """A caller that omits the invocation id still settles under one.

        The contract does not require ``invocation_id``, so a caller may omit
        it; the handler mints one, and that single value is what the response,
        the budget debit, the outbox row and the mana idempotency key all carry
        — rather than a blank key the outbox would refuse.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            response = await stub.GroundedSearch(
                _grounded_request(invocation_id=""), metadata=_bearer()
            )

        # The gateway minted one: the response carries a non-empty id, and the
        # adapter was handed that same one on the canonical request.
        assert response.invocation_id
        assert adapter.calls[0].invocation_id == response.invocation_id
        # The minted id is the key the settlement is written under: one budget
        # debit, one outbox row and one mana debit, all sharing it.
        assert connection.debited_budget() == 1
        assert connection.wrote_outbox_rows() == 1
        assert mana_client.debit_calls == [
            (_GCID, _GRPC_TENANT, "external_egress.search", response.invocation_id)
        ]
        insert = next(
            call for call in connection.calls if "INSERT INTO outbox_events" in call[0]
        )
        assert insert[1][0] == response.invocation_id
        event = token_usage_pb2.TokenUsageRecorded.FromString(insert[1][3])
        assert event.usage_id == response.invocation_id
        assert event.invocation_id == response.invocation_id

    async def test_an_omitted_invocation_id_is_minted_for_a_partial_settlement(self) -> None:
        """A synthesis failure after a billed search settles under a minted id.

        The partial settlement is keyed by the invocation id too, so a caller
        that omitted it must still produce one: the outbox row and the budget
        debit land under the minted id rather than being refused for a blank
        key, and the failed call still earns neither a mana debit nor an egress
        count.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(
            GroundedError(
                "vendor_unavailable",
                "the synthesis call failed",
                usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
            )
        )
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GroundedSearch(
                    _grounded_request(invocation_id=""), metadata=_bearer()
                )

        assert error.value.code() == grpc.StatusCode.UNAVAILABLE
        # The partial spend settled under a minted, non-empty key: one outbox
        # row and one budget debit, with the row's own usage id agreeing.
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        insert = next(
            call for call in connection.calls if "INSERT INTO outbox_events" in call[0]
        )
        assert insert[1][0]
        event = token_usage_pb2.TokenUsageRecorded.FromString(insert[1][3])
        assert event.usage_id == insert[1][0]
        assert event.invocation_id == insert[1][0]
        # The failed call earned neither a mana debit nor an egress count.
        assert mana_client.debit_calls == []
        assert connection.recorded_egress() == 0


class TestGroundedSearchDispatchClaim:
    """A redelivered grounded search is served, but settles nothing a second time.

    The dispatch-claim mechanism protects the grounded search the same way it
    protects an invoke: the first search with an invocation id claims the
    dispatch, runs the Exa call and settles; a redelivery with the same
    invocation id finds the dispatch already claimed, skips the Exa call and
    returns a deduped response — no second budget debit, no second outbox row,
    no second mana debit.
    """

    async def test_a_redelivered_search_is_served_but_charged_once(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        # The first claim wins the key; the second loses it and is deduped.
        connection.claim_statuses = ["INSERT 0 1", "INSERT 0 0"]
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )

        # Both searches were served — the redelivery is answered, not refused.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == ""
        # The Exa adapter was called only once — the redelivery skipped it.
        assert len(adapter.calls) == 1
        # But the learner was billed once: one claim, one outbox row, one debit.
        assert connection.wrote_outbox_rows() == 1
        assert connection.debited_budget() == 1
        assert len(mana_client.debit_calls) == 1
        assert mana_client.debit_calls == [
            (_GCID, _GRPC_TENANT, "external_egress.search", "inv-grounded-1")
        ]

    async def test_a_redelivered_search_with_a_different_invocation_id_is_charged(
        self,
    ) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
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
        assert connection.wrote_outbox_rows() == 2
        assert connection.debited_budget() == 2
        assert len(mana_client.debit_calls) == 2

    async def test_a_served_search_claims_the_dispatch_before_the_adapter(self) -> None:
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        mana_client = StubManaClient()
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter, mana=ManaMeter(mana_client))

        async with _serve_grounded(handler) as stub:
            await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )

        # The dispatch was claimed exactly once, before the Exa call.
        assert connection.claimed_debit() == 1

    async def test_the_claim_is_recorded_before_the_exa_call_begins(self) -> None:
        """The adapter sees the claim in the ledger at the moment it is invoked.

        The ordering is the whole of the dedup contract: the claim must be
        durable before the Exa call starts, so a crash after the call cannot
        leave an unbilled search and a redelivery cannot be charged twice. The
        adapter asserts it from its own call record rather than the test
        asserting it after the fact.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]

        class _ClaimCheckingAdapter:
            """An adapter that asserts the dispatch was claimed before it ran."""

            def __init__(self, outcome: object, connection: StubConnection) -> None:
                self._outcome = outcome
                self._connection = connection
                self.calls: list[object] = []

            async def search(self, grounded_request) -> GroundedSearchResult:
                assert self._connection.claimed_debit() == 1
                self.calls.append(grounded_request)
                return self._outcome

        adapter = _ClaimCheckingAdapter(_grounded_result(), connection)
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            response = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )

        assert response.grounded_answer == "Paris is the capital."
        assert len(adapter.calls) == 1

    async def test_a_redelivered_search_skips_the_egress_gate(self) -> None:
        """A redelivery is deduped before the egress gate, not refused by it.

        The operator engages the kill switch between the two deliveries, so the
        gate the first call passed would now refuse the second. The redelivery
        is still answered with the deduped response — the claim short-circuits
        the gates — and the kill switch is not read a second time.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        # The first claim wins the key; the second loses it and is deduped.
        connection.claim_statuses = ["INSERT 0 1", "INSERT 0 0"]
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )
            reads_after_first = connection.egress_reads()
            connection.kill_switch_row = {"engaged": True}
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )

        # The redelivery is answered with the deduped response, not refused by
        # the gate that would now deny it, and the Exa call was skipped.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == ""
        assert len(adapter.calls) == 1
        # The gate was read once, by the first call only: the redelivery's
        # dedup happened before any egress read.
        assert reads_after_first == 1
        assert connection.egress_reads() == 1

    async def test_a_redelivered_search_skips_the_budget_decision(self) -> None:
        """A redelivery is deduped before the budget decision, not refused by it.

        The tenant's window is exhausted between the two deliveries, so the
        budget decision the first call passed would now refuse the second. The
        redelivery is still answered with the deduped response, and the budget
        is not read a second time.
        """
        connection = StubConnection()
        connection.kill_switch_row = {"engaged": False}
        connection.entitlement_row = {
            "egress_enabled": True,
            "daily_call_ceiling": 50,
        }
        connection.usage_row = {"call_count": 0}
        connection.budget_rows = [_budget_window(spent_micros=0, policy=BudgetPolicy.BLOCK)]
        # The first claim wins the key; the second loses it and is deduped.
        connection.claim_statuses = ["INSERT 0 1", "INSERT 0 0"]
        adapter = _ScriptedAdapter(_grounded_result())
        handler = _grounded_handler(connection, adapter)

        async with _serve_grounded(handler) as stub:
            first = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )
            reads_after_first = connection.budget_window_reads()
            connection.budget_rows = [
                _budget_window(spent_micros=2_000_000, policy=BudgetPolicy.BLOCK)
            ]
            second = await stub.GroundedSearch(
                _grounded_request(invocation_id="inv-grounded-1"), metadata=_bearer()
            )

        # The redelivery is answered with the deduped response, not refused with
        # the budget_exhausted the exhausted window would declare.
        assert first.grounded_answer == "Paris is the capital."
        assert second.grounded_answer == ""
        assert len(adapter.calls) == 1
        # The budget was read once, by the first call only: the redelivery's
        # dedup happened before any budget read.
        assert reads_after_first == 1
        assert connection.budget_window_reads() == 1
