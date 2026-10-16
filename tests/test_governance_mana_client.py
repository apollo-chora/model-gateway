"""The real mana client, and its wiring into the application the process runs.

The meter's policy is tested against a stub client elsewhere. What is under
test here is the transport half of that seam: :class:`GrpcManaClient` speaking
chora-identity's ``ManaService`` over a real in-process gRPC server — the
request it puts on the wire (a dry-run quote with no units, a real debit
under the caller's idempotency key), the outcome it reads back (an unpriced
action as the service's ``INVALID_ARGUMENT``, every other failure propagated
for the meter to fail open on) — and ``create_app`` wiring the client into the
governance chain when the deployment names the identity service's address,
closing the channel with the process.

The grounded-search settlement is driven end to end too: a served search over
the wired application debits mana through the real channel, records the
egress, debits the budget and inserts the outbox row.

The identity service is a stub registered on the generated servicer, so what
is asserted is what crosses the wire — the request fields the client set and
the response fields it read — without a chora-identity deployment.
"""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager

import grpc
import pytest

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import GroundedFinishReason, GroundedSearchResult, GroundedUsage
from model_gateway.governance import (
    Database,
    GovernanceManager,
    GrpcManaClient,
    ManaClient,
    ManaMeter,
    ManaResult,
    UnknownAction,
)
from model_gateway.governance.identity.v1 import mana_service_pb2, mana_service_pb2_grpc
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_GCID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_DSN = "postgresql://localhost/chora_observability"
_UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
_API_KEY = "sk-gateway-test-key"

_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# The stub identity service
# ---------------------------------------------------------------------------


class StubManaService(mana_service_pb2_grpc.ManaServiceServicer):
    """A ``ManaService`` double replaying a scripted outcome and recording its calls."""

    def __init__(self) -> None:
        self.calls: list[mana_service_pb2.DeductManaRequest] = []
        self.response = mana_service_pb2.DeductManaResponse(
            success=True, balance_after_units=90
        )
        self.code: grpc.StatusCode | None = None
        self.details = ""

    async def DeductMana(self, request, context):
        self.calls.append(request)
        if self.code is not None:
            await context.abort(self.code, self.details)
        return self.response


@asynccontextmanager
async def _serve_mana(service: StubManaService):
    """A real in-process gRPC server serving one ``ManaService``, and its address."""
    server = grpc.aio.server()
    port = server.add_insecure_port(f"{_HOST}:0")
    mana_service_pb2_grpc.add_ManaServiceServicer_to_server(service, server)
    await server.start()
    try:
        yield f"{_HOST}:{port}"
    finally:
        await server.stop(grace=None)


def _client(address: str) -> GrpcManaClient:
    return GrpcManaClient(grpc.aio.insecure_channel(address))


# ---------------------------------------------------------------------------
# GrpcManaClient against the wire
# ---------------------------------------------------------------------------


class TestGrpcManaClient:
    async def test_a_quote_runs_a_dry_run_with_no_units_under_a_fresh_key(self) -> None:
        service = StubManaService()

        async with _serve_mana(service) as address:
            client = _client(address)

            result = await client.quote(_GCID, "companion_chat")

        assert result == ManaResult(success=True, balance_after_units=90)
        assert len(service.calls) == 1
        call = service.calls[0]
        assert call.gcid == _GCID
        assert call.action_code == "companion_chat"
        assert call.dry_run is True
        assert call.units == 0
        # The service requires a non-empty key even for a dry run; a fresh one
        # records no dedup state because a dry run writes no ledger row.
        assert call.idempotency_key != ""

    async def test_a_debit_runs_a_real_deduction_under_the_callers_key(self) -> None:
        service = StubManaService()

        async with _serve_mana(service) as address:
            client = _client(address)

            result = await client.debit(_GCID, "companion_chat", "idem-key-1")

        assert result == ManaResult(success=True, balance_after_units=90)
        assert len(service.calls) == 1
        call = service.calls[0]
        assert call.dry_run is False
        assert call.idempotency_key == "idem-key-1"

    async def test_a_shortfall_is_read_with_both_balances(self) -> None:
        service = StubManaService()
        service.response = mana_service_pb2.DeductManaResponse(
            success=False, required_units=50, current_balance_units=20
        )

        async with _serve_mana(service) as address:
            client = _client(address)

            result = await client.debit(_GCID, "companion_chat", "idem-key-1")

        assert result == ManaResult(
            success=False, required_units=50, current_balance_units=20
        )

    async def test_an_unpriced_action_is_the_services_invalid_argument(self) -> None:
        service = StubManaService()
        service.code = grpc.StatusCode.INVALID_ARGUMENT
        service.details = "unknown action code"

        async with _serve_mana(service) as address:
            client = _client(address)

            with pytest.raises(UnknownAction):
                await client.quote(_GCID, "not_in_the_catalogue")

    async def test_another_failure_propagates_for_the_meter_to_fail_open(self) -> None:
        service = StubManaService()
        service.code = grpc.StatusCode.UNAVAILABLE
        service.details = "identity unreachable"

        async with _serve_mana(service) as address:
            client = _client(address)

            with pytest.raises(grpc.aio.AioRpcError) as error:
                await client.debit(_GCID, "companion_chat", "idem-key-1")

        assert error.value.code() is grpc.StatusCode.UNAVAILABLE

    async def test_the_client_satisfies_the_protocol(self) -> None:
        channel = grpc.aio.insecure_channel(f"{_HOST}:1")
        try:
            assert isinstance(GrpcManaClient(channel), ManaClient)
        finally:
            await channel.close()


# ---------------------------------------------------------------------------
# The wiring through create_app
# ---------------------------------------------------------------------------


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


class StubConnection:
    """A pooled-connection stand-in playing the rows the governance chain reads.

    Every write is counted as it lands, so a test asserts the settlement
    happened from the counters rather than from a call log.
    """

    def __init__(self) -> None:
        self.entitlement_row = {"egress_enabled": True, "daily_call_ceiling": 50}
        self.usage_row = {"call_count": 0}
        self.kill_switch_row = {"engaged": False}
        self.egress_recorded = 0
        self.outbox_inserts = 0
        self.budget_updates = 0

    def transaction(self, tenant_id: str | None = None) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        if "grounded_search_daily_usage" in query:
            self.egress_recorded += 1
            return "INSERT 0 1"
        if "outbox_events" in query:
            self.outbox_inserts += 1
            return "INSERT 0 1"
        if "per_tenant_llm_budget" in query and "UPDATE" in query:
            self.budget_updates += 1
            return "UPDATE 1"
        return "UPDATE 0"

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        if "platform_egress_killswitch" in query:
            return self.kill_switch_row
        if "external_egress_policy" in query:
            return self.entitlement_row
        if "grounded_search_daily_usage" in query:
            return self.usage_row
        return None

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        return []


class FakeDatabase(Database):
    """A ``Database`` whose pool is a stub, so the chain runs without Postgres."""

    def __init__(self, dsn: str, connection: StubConnection) -> None:
        super().__init__(dsn)
        self.connect_calls = 0
        self.close_calls = 0
        self._pool = StubPool(connection)  # type: ignore[assignment]

    async def connect(self) -> None:
        self.connect_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def _wire_database(monkeypatch: pytest.MonkeyPatch, connection: StubConnection) -> list:
    """Replace the factory's Database with the stub-pool fake; returns the fakes."""
    created: list[FakeDatabase] = []

    def fake_database(dsn: str) -> FakeDatabase:
        assert dsn == _DSN
        db = FakeDatabase(dsn, connection)
        created.append(db)
        return db

    monkeypatch.setattr("model_gateway.app.Database", fake_database)
    return created


def _registry() -> ModelRegistry:
    """One text route over a deterministic upstream, the factory's own."""
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=_UPSTREAM_ENDPOINT,
                base_url=_UPSTREAM_ENDPOINT,
                model_id="upstream-text",
                format="openai",
                api_key="sk-upstream-secret",
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


def _free_port() -> int:
    """A port nothing else holds, for the gRPC listener to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


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


def _grounded_request(**overrides) -> pb.GroundedSearchRequest:
    fields = {
        "invocation_id": "inv-grounded-1",
        "tenant_id": _TENANT_ID,
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
    return (("authorization", f"Bearer {_API_KEY}"),)


class TestManaWiring:
    """create_app wires the mana meter when the deployment names the identity."""

    async def test_the_mana_meter_is_wired_when_the_identity_address_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = StubConnection()
        created: dict[str, object] = {}
        real_service = InferenceService

        def recording_service(registry, provider, **kwargs):
            created.update(kwargs)
            return real_service(registry, provider, **kwargs)

        monkeypatch.setattr("model_gateway.app.InferenceService", recording_service)
        _wire_database(monkeypatch, connection)

        app = create_app(
            Settings.from_env(
                {
                    "CHORA_OBSERVABILITY_DSN": _DSN,
                    "CHORA_IDENTITY_GRPC_ADDR": f"{_HOST}:9999",
                }
            ),
            registry=_registry(),
        )
        await app.start()
        try:
            governance = created["governance"]
            assert isinstance(governance, GovernanceManager)
            assert isinstance(governance.mana, ManaMeter)
            assert isinstance(governance.mana._client, GrpcManaClient)
        finally:
            await app.stop()

    async def test_a_dsn_without_an_identity_address_runs_un_metered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = StubConnection()
        created: dict[str, object] = {}
        real_service = InferenceService

        def recording_service(registry, provider, **kwargs):
            created.update(kwargs)
            return real_service(registry, provider, **kwargs)

        monkeypatch.setattr("model_gateway.app.InferenceService", recording_service)
        _wire_database(monkeypatch, connection)

        create_app(Settings.from_env({"CHORA_OBSERVABILITY_DSN": _DSN}), registry=_registry())

        governance = created["governance"]
        assert isinstance(governance, GovernanceManager)
        # No identity address: no mana client, so the chain runs un-metered —
        # exactly as it runs without a budget window.
        assert governance.mana is None

    async def test_the_mana_channel_closes_with_the_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = StubConnection()
        created_clients: list[GrpcManaClient] = []
        real_client = GrpcManaClient

        def recording_client(channel: grpc.aio.Channel) -> GrpcManaClient:
            client = real_client(channel)
            created_clients.append(client)
            return client

        monkeypatch.setattr("model_gateway.app.GrpcManaClient", recording_client)
        _wire_database(monkeypatch, connection)

        app = create_app(
            Settings.from_env(
                {
                    "CHORA_OBSERVABILITY_DSN": _DSN,
                    "CHORA_IDENTITY_GRPC_ADDR": f"{_HOST}:9999",
                }
            ),
            registry=_registry(),
        )
        await app.start()
        assert len(created_clients) == 1
        channel = created_clients[0]._channel
        await app.stop()
        assert channel.get_state(try_to_connect=False) is grpc.ChannelConnectivity.SHUTDOWN


class TestGroundedSettlementEndToEnd:
    """A served grounded search settles over the wired application, for real."""

    async def test_a_served_search_debits_mana_and_records_the_egress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connection = StubConnection()
        mana_service = StubManaService()
        adapter = _ScriptedAdapter(_grounded_result())

        _wire_database(monkeypatch, connection)
        monkeypatch.setattr(
            "model_gateway.grpc.grounded_search.ExaGroundedSearch",
            lambda _config: adapter,
        )

        async with _serve_mana(mana_service) as mana_address:
            settings = Settings.from_env(
                {
                    "CHORA_OBSERVABILITY_DSN": _DSN,
                    "CHORA_IDENTITY_GRPC_ADDR": mana_address,
                    "CHORA_GRPC_PORT": str(_free_port()),
                    "GATEWAY_AUTH_ENABLED": "true",
                    "GATEWAY_API_KEYS": f"{_API_KEY}:{_TENANT_ID}",
                }
            )
            app = create_app(settings, registry=_registry(), serve_grpc=True)
            await app.start()
            try:
                channel = grpc.aio.insecure_channel(f"{_HOST}:{settings.grpc_port}")
                try:
                    stub = gateway_pb2_grpc.ModelGatewayServiceStub(channel)
                    response = await stub.GroundedSearch(
                        _grounded_request(), metadata=_bearer()
                    )
                finally:
                    await channel.close()
            finally:
                await app.stop()

        assert response.grounded_answer == "Paris is the capital."
        # The mana debit crossed the real channel to the identity service: a
        # real deduction, not a dry run, under the invocation id as the key.
        assert len(mana_service.calls) == 1
        debit = mana_service.calls[0]
        assert debit.dry_run is False
        assert debit.gcid == "gcid-1"
        assert debit.action_code == "external_egress.search"
        assert debit.idempotency_key == "inv-grounded-1"
        # The egress was counted, the outbox row landed and the budget was
        # debited — the settlement the handler owed after the served search.
        assert connection.egress_recorded == 1
        assert connection.outbox_inserts == 1
        assert connection.budget_updates == 1
