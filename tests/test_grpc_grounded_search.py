"""The GroundedSearch RPC, driven the way a caller drives it.

Every test here runs against a real ``grpc.aio`` server on an ephemeral loopback
port, reached through the generated :class:`ModelGatewayServiceStub`, so what is
asserted is what crosses the wire: the canonical grounded search the adapter was
handed, the protobuf answer the caller reads back, and the gRPC status a refusal
carries. Only the adapter is a stand-in — the one thing this layer does not own
— and it records every call, which is how "the adapter was never called" is
proved rather than assumed.

The server is assembled here rather than through
:class:`~model_gateway.grpc.server.GrpcServer` because that class registers every
RPC of the service while this file tests one: a server carrying only
``GroundedSearch`` keeps each failure attributable to the handler under test. It
binds ``127.0.0.1:0``, so the port is the operating system's choice and no test
can collide with another or with a running deployment.

The adapter is stood in at :mod:`model_gateway.grpc.grounded_search`'s own
``ExaGroundedSearch``, because the handler builds the adapter from its config
the way the listener does — so what is scripted is the adapter's answer, not the
handler's wiring.
"""

from contextlib import asynccontextmanager

import grpc
import pytest

from model_gateway.config import Settings
from model_gateway.domain import GroundedFinishReason, GroundedSearchResult, GroundedUsage
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc
from model_gateway.grpc.grounded_search import GroundedSearch
from model_gateway.grpc.identity import Authenticator
from model_gateway.providers.exa import ExaConfig, GroundedError
from model_gateway.security.api_keys import ApiKeySet

API_KEY = "sk-gateway-test-key"
TENANT = "tenant-a"
UNBOUND_TENANT = "tenant-no-key-is-bound-to"

_HOST = "127.0.0.1"

_REQUIRED_FIELDS = ("tenant_id", "gcid", "agent_id", "directive", "action_code")


def config() -> ExaConfig:
    """An adapter config; the adapter itself is stood in, so nothing is contacted."""
    return ExaConfig(
        api_key="sk-exa-test",
        synthesis_model="synthesis-model",
        synthesis_endpoint="https://synthesis.example/v1",
        synthesis_api_key="sk-synthesis-test",
    )


def request(**overrides) -> pb.GroundedSearchRequest:
    """A ``GroundedSearchRequest`` a caller could send."""
    fields = {
        "invocation_id": "inv-grounded-1",
        "tenant_id": TENANT,
        "gcid": "gcid-1",
        "agent_id": "familiar_seeker",
        "crew_kind": "familiar",
        "directive": "What is the capital of France?",
        "max_results": 5,
        "action_code": "external_egress.search",
        "traceparent": "00-abc-def-01",
        "tracestate": "congo=t61rcWkgMzE",
    }
    fields.update(overrides)
    return pb.GroundedSearchRequest(**fields)


def bearer(key: str | None = API_KEY) -> tuple[tuple[str, str], ...]:
    """The call metadata presenting ``key``, or nothing when ``key`` is ``None``."""
    return () if key is None else (("authorization", f"Bearer {key}"),)


class ScriptedAdapter:
    """An adapter double answering with the result or the failure it was given."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls: list[object] = []

    async def search(self, grounded_request) -> GroundedSearchResult:
        self.calls.append(grounded_request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def adapter_result() -> GroundedSearchResult:
    """A canonical grounded search result the adapter could have produced."""
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


class _GroundedSearchOnlyServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    """Dispatch ``GroundedSearch`` to the handler under test, and nothing else."""

    def __init__(self, handler: GroundedSearch) -> None:
        self._handler = handler

    async def GroundedSearch(self, req, context):
        return await self._handler.grounded_search(req, context)


@asynccontextmanager
async def serve(adapter: ScriptedAdapter, *, auth_enabled: bool = True):
    """A real in-process gRPC server serving one ``GroundedSearch`` handler, and a stub."""
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true" if auth_enabled else "false",
            "GATEWAY_API_KEYS": f"{API_KEY}:{TENANT}",
        }
    )
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )
    handler = GroundedSearch(authenticator, config())
    server = grpc.aio.server()
    port = server.add_insecure_port(f"{_HOST}:0")
    gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
        _GroundedSearchOnlyServicer(handler), server
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"{_HOST}:{port}")
    try:
        yield gateway_pb2_grpc.ModelGatewayServiceStub(channel), adapter
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.fixture
def wire_adapter(monkeypatch: pytest.MonkeyPatch):
    """Patch the handler's adapter class with a scripted stand-in."""

    def _wire(adapter: ScriptedAdapter):
        monkeypatch.setattr(
            "model_gateway.grpc.grounded_search.ExaGroundedSearch", lambda _config: adapter
        )

    return _wire


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_a_grounded_search_reaches_the_adapter_and_answers(wire_adapter):
    adapter = ScriptedAdapter(adapter_result())
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        response = await stub.GroundedSearch(request(), metadata=bearer())

    assert response.grounded_answer == "Paris is the capital."
    assert response.vendor == "exa"
    assert response.model_version == "synthesis-model"
    assert response.finish_reason == pb.FINISH_REASON_STOP
    assert response.usage.grounding_units == 1
    assert response.usage.cost_micros == 10_000
    # The adapter was handed the caller's directive, nothing rewritten.
    assert len(adapter.calls) == 1
    assert adapter.calls[0].directive == "What is the capital of France?"


async def test_a_search_with_no_citation_is_answered_not_refused(wire_adapter):
    adapter = ScriptedAdapter(
        GroundedSearchResult(
            invocation_id="inv-grounded-1",
            citations=(),
            grounded_answer="",
            usage=GroundedUsage(grounding_units=1, cost_micros=10_000),
            vendor="exa",
            finish_reason=GroundedFinishReason.UNSPECIFIED,
            completed_at="2026-10-16T00:00:00+00:00",
        )
    )
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        response = await stub.GroundedSearch(request(), metadata=bearer())

    assert len(response.citations) == 0
    assert response.grounded_answer == ""
    assert response.finish_reason == pb.FINISH_REASON_UNSPECIFIED
    # The search billed, so the usage carries the surcharge even with no answer.
    assert response.usage.cost_micros == 10_000


# ---------------------------------------------------------------------------
# Validation: a request that states too little is refused, naming the field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", _REQUIRED_FIELDS)
async def test_a_missing_required_field_is_an_invalid_argument(wire_adapter, field):
    adapter = ScriptedAdapter(adapter_result())
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(**{field: ""}), metadata=bearer())

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert field in error.value.details()
    # A request that fails validation never reaches the adapter.
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# The governance denials map onto the statuses the chain declares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, status",
    [
        pytest.param("kill_switch_engaged", grpc.StatusCode.FAILED_PRECONDITION, id="kill"),
        pytest.param("external_egress_disabled", grpc.StatusCode.FAILED_PRECONDITION, id="egress"),
        pytest.param("budget_exhausted", grpc.StatusCode.FAILED_PRECONDITION, id="budget"),
        pytest.param("daily_ceiling_exceeded", grpc.StatusCode.RESOURCE_EXHAUSTED, id="ceiling"),
        pytest.param("vendor_unavailable", grpc.StatusCode.UNAVAILABLE, id="vendor"),
    ],
)
async def test_a_governance_denial_maps_onto_its_status(wire_adapter, reason, status):
    adapter = ScriptedAdapter(GroundedError(reason, f"denied: {reason}"))
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(), metadata=bearer())

    assert error.value.code() == status
    assert reason in error.value.details()


async def test_a_synthesis_failure_includes_partial_spend_in_the_error_details(wire_adapter):
    """A grounded failure carrying partial usage publishes it in the gRPC error details.

    When the Exa search succeeds but the synthesis fails, the provider billed
    for the search. The error response carries the partial cost so the caller
    can account it — the same reason the Go adapter returns the usage with the
    error.
    """
    partial_usage = GroundedUsage(grounding_units=1, cost_micros=10_000)
    adapter = ScriptedAdapter(
        GroundedError("vendor_unavailable", "the synthesis call failed", usage=partial_usage)
    )
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(), metadata=bearer())

    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    details = error.value.details()
    assert "partial_cost_micros" in details
    assert "10000" in details


async def test_an_unconfigured_gateway_is_an_internal_error(wire_adapter):
    adapter = ScriptedAdapter(GroundedError("not_configured", "the Exa API key is not configured"))
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(), metadata=bearer())

    assert error.value.code() == grpc.StatusCode.INTERNAL


async def test_an_unrecognised_reason_is_an_internal_error(wire_adapter):
    adapter = ScriptedAdapter(GroundedError("something_unexpected", "a bug"))
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(), metadata=bearer())

    assert error.value.code() == grpc.StatusCode.INTERNAL


# ---------------------------------------------------------------------------
# Identity: the credential is the tenant, and authentication is required
# ---------------------------------------------------------------------------


async def test_a_tenant_claim_disagreeing_with_the_credential_is_refused(wire_adapter):
    adapter = ScriptedAdapter(adapter_result())
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(tenant_id=UNBOUND_TENANT), metadata=bearer())

    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
    # A refused claim never reaches the adapter.
    assert adapter.calls == []


async def test_a_call_without_a_credential_is_unauthenticated(wire_adapter):
    adapter = ScriptedAdapter(adapter_result())
    wire_adapter(adapter)

    async with serve(adapter) as (stub, adapter):
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.GroundedSearch(request(), metadata=bearer(None))

    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert adapter.calls == []


async def test_anonymous_calls_proceed_when_authentication_is_off(wire_adapter):
    adapter = ScriptedAdapter(adapter_result())
    wire_adapter(adapter)

    async with serve(adapter, auth_enabled=False) as (stub, adapter):
        response = await stub.GroundedSearch(request(tenant_id=TENANT), metadata=bearer(None))

    assert response.grounded_answer == "Paris is the capital."
    assert len(adapter.calls) == 1
