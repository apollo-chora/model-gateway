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
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.grounded_search import GroundedSearch
from model_gateway.grpc.identity import Authenticator
from model_gateway.inference import InferenceService
from model_gateway.providers.exa import ExaConfig
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
        self.quote_calls: list[tuple[str, str]] = []
        self.debit_calls: list[tuple[str, str, str]] = []

    async def quote(self, gcid: str, action_code: str) -> ManaResult:
        self.quote_calls.append((gcid, action_code))
        return self.quote_result

    async def debit(self, gcid: str, action_code: str, idem_key: str) -> ManaResult:
        self.debit_calls.append((gcid, action_code, idem_key))
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
        assert mana_client.debit_calls == [(_GCID, "companion_chat", "inv-1")]

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
        "gcid": "gcid-1",
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
        # collapses onto the ledger row the first settlement wrote.
        assert mana_client.debit_calls == [("gcid-1", "external_egress.search", "inv-grounded-1")]

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
