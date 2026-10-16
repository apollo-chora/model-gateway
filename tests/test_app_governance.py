"""The governance chain wired into the application the process runs.

The mechanisms and the factory are tested separately. What is under test here
is the seam between them: that ``create_app`` builds the chain when the
deployment configures a DSN, hands it to the inference service and to the
gRPC listener's grounded search, opens the database at startup and closes it
at shutdown, and that a request the chain denies never reaches the provider.

The database is a stub pool playing the rows the chain reads — every read
finds no window, no containment row and no egress policy, so a governed call
is authorised on the strength of the fail-open absence — which exercises the
wiring without Postgres and without standing up a provider account.
"""

from __future__ import annotations

import socket
import uuid

import pytest
from blacksheep import JSONContent
from blacksheep.testing import TestClient

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.governance import Database, GovernanceManager
from model_gateway.grpc.grounded_search import GroundedSearch
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider
from model_gateway.registry import ModelRegistry

_TENANT_ID = "123e4567-e89b-42d3-a456-426614174000"
_DSN = "postgresql://localhost/chora_observability"
_UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
_API_KEY = "sk-gateway-test-key"


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
    """A pooled-connection stand-in playing the rows the governance chain reads."""

    def __init__(self) -> None:
        self.budget_rows: list[dict[str, object]] = []

    def transaction(self, tenant_id: str | None = None) -> StubTransaction:
        return StubTransaction(self)

    async def execute(self, query: str, *args: object) -> str:
        if "outbox_events" in query:
            return "INSERT 0 1"
        if "per_tenant_llm_budget" in query and "UPDATE" in query:
            return "UPDATE 1"
        return "UPDATE 0"

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        return None

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        if "per_tenant_llm_budget" in query:
            return list(self.budget_rows)
        return []


class FakeDatabase(Database):
    """A ``Database`` whose pool is a stub, so the chain runs without Postgres.

    ``connect`` and ``close`` are recorded rather than run, so a test asserts
    the process opened the pool at startup and closed it at shutdown.
    """

    def __init__(self, dsn: str, connection: StubConnection) -> None:
        super().__init__(dsn)
        self.connect_calls = 0
        self.close_calls = 0
        self._pool = StubPool(connection)  # type: ignore[assignment]

    async def connect(self) -> None:
        self.connect_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


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
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def connection() -> StubConnection:
    return StubConnection()


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


class TestGovernanceWiring:
    async def test_governance_is_wired_into_the_inference_service(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        captured: dict[str, object] = {}
        real_service = InferenceService

        def recording_service(registry, provider, **kwargs):
            captured.update(kwargs)
            return real_service(registry, provider, **kwargs)

        monkeypatch.setattr("model_gateway.app.InferenceService", recording_service)
        _wire_database(monkeypatch, connection)

        create_app(
            Settings.from_env({"CHORA_OBSERVABILITY_DSN": _DSN}),
            registry=_registry(),
        )

        # The service the factory built was handed the governance chain: the
        # budget, the outbox and the containment read are its to run.
        governance = captured["governance"]
        assert isinstance(governance, GovernanceManager)

    async def test_governance_is_wired_into_the_grounded_search(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        captured: dict[str, object] = {}
        real_grounded = GroundedSearch

        def recording_grounded(authenticator, config, *, governance=None):
            captured["governance"] = governance
            return real_grounded(authenticator, config, governance=governance)

        monkeypatch.setattr(
            "model_gateway.grpc.server.GroundedSearch", recording_grounded
        )
        _wire_database(monkeypatch, connection)

        settings = Settings.from_env(
            {"CHORA_OBSERVABILITY_DSN": _DSN, "CHORA_GRPC_PORT": str(_free_port())}
        )
        create_app(settings, registry=_registry(), serve_grpc=True)

        assert isinstance(captured["governance"], GovernanceManager)

    async def test_a_deployment_with_no_dsn_wires_no_governance(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        created = _wire_database(monkeypatch, connection)

        app = create_app(Settings.from_env({}), registry=_registry())
        await app.start()
        try:
            # No DSN: no database is built and no chain exists to wire, so the
            # invocation flow runs exactly as it did before governance.
            assert created == []
        finally:
            await app.stop()

    async def test_the_database_opens_at_startup_and_closes_at_shutdown(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        created = _wire_database(monkeypatch, connection)

        app = create_app(
            Settings.from_env({"CHORA_OBSERVABILITY_DSN": _DSN}),
            registry=_registry(),
        )
        await app.start()
        assert created[0].connect_calls == 1
        await app.stop()
        assert created[0].close_calls == 1


class TestDenialThroughTheWiredApplication:
    async def test_a_denied_request_never_reaches_the_provider(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        constructed: list[object] = []

        class RecordingProvider:
            """A provider double that records whether it was ever called."""

            def __init__(self) -> None:
                self.calls = 0
                constructed.append(self)

            async def complete(self, route, request, context):
                self.calls += 1
                raise AssertionError("a denied request reached the provider")

        # The HTTP surface stamps openai_compat on every request it serves, so
        # the chain's stamped-surface check passes; the denial this test is
        # about comes from the tenant's exhausted budget window, and it must
        # arrive before the provider is called.
        connection.budget_rows = [
            {
                "budget_id": uuid.uuid4(),
                "budget_usd_micros": 1_000_000,
                "spent_usd_micros": 2_000_000,
                "policy": "block",
                "downgrade_to_logical_model_id": None,
            }
        ]
        monkeypatch.setattr("model_gateway.app.LiteLLMProvider", RecordingProvider)
        _wire_database(monkeypatch, connection)

        app = create_app(
            Settings.from_env(
                {
                    "CHORA_OBSERVABILITY_DSN": _DSN,
                    "GATEWAY_AUTH_ENABLED": "true",
                    "GATEWAY_API_KEYS": f"{_API_KEY}:{_TENANT_ID}",
                }
            ),
            registry=_registry(),
        )
        await app.start()
        try:
            response = await TestClient(app).post(
                "/v1/chat/completions",
                content=JSONContent(
                    {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
                ),
                headers={"Authorization": f"Bearer {_API_KEY}"},
            )
        finally:
            await app.stop()

        # The budget denial is a loud failed precondition, and the provider
        # was never constructed beyond the factory's own binding and never
        # called.
        assert response.status == 412
        assert "budget_exhausted" in await response.text()
        assert constructed[0].calls == 0  # type: ignore[attr-defined]

    async def test_a_served_request_passes_the_surface_check_and_reaches_the_provider(
        self, monkeypatch: pytest.MonkeyPatch, connection: StubConnection
    ) -> None:
        """The HTTP surface's stamped surface satisfies the chain's first check.

        With the governance chain wired and nothing blocking the tenant, the
        ``openai_compat`` surface the HTTP surface stamps carries the
        invocation through the stamped-surface check — the check that refused
        every HTTP inference request as ``surface_unstamped`` before the
        surface was stamped — and the provider is called.
        """
        constructed: list[MockProvider] = []

        def recording_provider(*args: object, **kwargs: object) -> MockProvider:
            provider = MockProvider()
            constructed.append(provider)
            return provider

        monkeypatch.setattr("model_gateway.app.LiteLLMProvider", recording_provider)
        _wire_database(monkeypatch, connection)

        app = create_app(
            Settings.from_env({"CHORA_OBSERVABILITY_DSN": _DSN}),
            registry=_registry(),
        )
        await app.start()
        try:
            response = await TestClient(app).post(
                "/v1/chat/completions",
                content=JSONContent(
                    {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
                ),
            )
        finally:
            await app.stop()

        assert response.status == 200
        assert len(constructed) == 1
        assert len(constructed[0].calls) == 1
        _, invocation, _context = constructed[0].calls[0]
        assert invocation.surface == "openai_compat"
