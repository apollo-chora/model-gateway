"""Usage and cost accounting: the record, the pricing, the sinks (PRD section 13).

These tests prove the rules the module is built on: a token count is either
reported or estimated and the difference travels with it, cost comes from
configured pricing rather than from a vendor name, and a cost is reported only
when every category the deployment priced is known — an incomplete sum is not
reported as a cost. They also prove the service records a completed answer under
the deployment's accounting setting, that a recording failure is logged rather
than swallowed or propagated, and that the projection of a record covers every
field the record declares.
"""

import asyncio
import dataclasses
from datetime import UTC

import orjson
import pytest
from blacksheep import Application, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient
from structlog.testing import capture_logs

from model_gateway.accounting import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    AccountingRecorder,
    InvocationOutcome,
    MemoryAccountingSink,
    NullAccountingSink,
    PricingTable,
    UsageRecord,
    pricing_from_env,
)
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.config.settings import ConfigurationError
from model_gateway.domain import FinishReason, InvocationResponse, TextPart, Usage
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.http.auth import create_auth_middleware
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.embeddings import register_embeddings_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet


def route(model_id: str = "upstream-text", **overrides) -> Route:
    """One text route, with the capabilities a chat invocation needs.

    ``model_id`` is positional so a test can name the route it is building:
    ``route("upstream-model")`` and ``route("fallback-model")`` are then
    self-describing, while ``route()`` is the default upstream.
    """
    values: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.internal.example/v1",
        "base_url": "https://upstream.internal.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)  # type: ignore[arg-type]


def response(**overrides) -> InvocationResponse:
    """One completed answer, with a usage report the provider gave."""
    values: dict[str, object] = {
        "model": "text-default",
        "upstream_model": "upstream-text",
        "invocation_id": "inv-accounting-1",
        "content": (TextPart("hello"),),
        "finish_reason": FinishReason.STOP,
        "usage": Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=2,
        ),
        "attempt_count": 2,
        "fallback_used": True,
    }
    values.update(overrides)
    return InvocationResponse(**values)  # type: ignore[arg-type]


def outcome(**overrides) -> InvocationOutcome:
    """The answer facts a record takes, as a completed answer states them.

    ``response()`` is what a provider returns; this is what the recorder is
    handed, and the two are separate because a streamed answer has no response
    object to read those facts from.
    """
    values: dict[str, object] = {
        "invocation_id": "inv-accounting-1",
        "upstream_model": "upstream-text",
        "usage": Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=2,
        ),
    }
    values.update(overrides)
    return InvocationOutcome(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def test_pricing_comes_from_configuration_not_from_a_vendor_name():
    """A model's prices are whatever the deployment configured, per million tokens."""
    pricing = pricing_from_env(
        {
            "COST_PRICING_JSON": orjson.dumps(
                {
                    "text-default": {
                        "currency": "USD",
                        "tokens": {"input": 3.0, "output": 15.0, "cached": 0.3},
                    }
                }
            ).decode()
        }
    )

    cost, currency = pricing.estimate(
        Usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=1_000_000),
        "text-default",
    )

    assert cost == pytest.approx(18.3)
    assert currency == "USD"


def test_a_model_with_no_configured_price_records_usage_and_no_cost():
    """An unpriced model is recorded with no cost rather than an invented one."""
    cost, currency = PricingTable().estimate(
        Usage(input_tokens=10, output_tokens=5), "text-default"
    )

    assert (cost, currency) == (None, None)


def test_a_priced_category_the_provider_never_reported_makes_the_cost_unknown():
    """A partial sum is not a cost.

    The entry prices output tokens, so the deployment has said what output costs.
    A provider that reported no output tokens has left a charge the gateway cannot
    see, and reporting the input half as *the* cost would understate spend while
    looking like a complete figure.
    """
    cost, currency = _table("USD", {"input": 3.0, "output": 15.0}).estimate(
        Usage(input_tokens=1_000_000, output_tokens=None), "text-default"
    )

    assert (cost, currency) == (None, None)


def test_a_category_the_entry_does_not_price_is_not_a_missing_charge():
    """An operator who priced only input tokens has said what input tokens cost."""
    cost, currency = _table("USD", {"input": 3.0}).estimate(
        Usage(input_tokens=1_000_000, output_tokens=None), "text-default"
    )

    assert cost == pytest.approx(3.0)
    assert currency == "USD"


def test_no_known_billable_count_is_not_a_cost_of_zero():
    """Zero would read as evidence of a free invocation, which nothing established."""
    cost, currency = _table("USD", {"input": 3.0}).estimate(Usage(), "text-default")

    assert (cost, currency) == (None, None)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("not json", id="not JSON at all"),
        pytest.param("[]", id="a JSON array"),
        pytest.param('{"text-default": "expensive"}', id="a scalar entry"),
        pytest.param('{"": {"currency": "USD", "tokens": {"input": 3.0}}}', id="an empty model id"),
        pytest.param(
            '{"text-default": {"tokens": {"input": 3.0}}}', id="an entry with no currency"
        ),
        pytest.param('{"text-default": {"currency": "USD"}}', id="an entry with no tokens"),
        pytest.param(
            '{"text-default": {"currency": "  ", "tokens": {"input": 3.0}}}',
            id="an empty currency",
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {"input": 3.0}, "note": "hi"}}',
            id="an unknown entry key",
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {}}}', id="an entry that prices nothing"
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {"prompt": 3.0}}}',
            id="an unknown token category",
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {"input": "3.0"}}}',
            id="a string price",
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {"input": true}}}',
            id="a boolean price",
        ),
        pytest.param(
            '{"text-default": {"currency": "USD", "tokens": {"input": -3.0}}}',
            id="a negative price",
        ),
    ],
)
def test_a_malformed_pricing_value_is_refused_at_startup(raw):
    """A deployment that misconfigures its prices learns now, not after a month of bad bills.

    Each case is a value that would otherwise reach a cost calculation and produce
    a number the gateway could not stand behind.
    """
    with pytest.raises(ConfigurationError):
        pricing_from_env({"COST_PRICING_JSON": raw})


def test_a_non_finite_price_is_refused():
    """JSON cannot carry one, but the value object is reachable without JSON."""
    for bad in (float("inf"), float("nan"), float("-inf")):
        with pytest.raises(ValueError):
            _price("USD", {"input": bad})


def test_an_absent_pricing_value_is_an_empty_table():
    """A deployment that configures no prices is not a misconfiguration."""
    assert pricing_from_env({}).prices == {}
    assert pricing_from_env({"COST_PRICING_JSON": ""}).prices == {}


def test_a_table_built_by_hand_is_validated_too():
    """The parser is a caller like any other: every table is one the gateway can price against.

    A table assembled outside the environment path must not be able to carry a key
    that is not a model ID or a value that is not a price, because `estimate` would
    then reach a shape it cannot price and raise inside an invocation's accounting
    rather than where the mistake was made.
    """
    with pytest.raises(ValueError):
        PricingTable(prices={"text-default": {"input": 3.0}})
    with pytest.raises(ValueError):
        PricingTable(prices={"": _price("USD", {"input": 3.0})})
    with pytest.raises(ValueError):
        PricingTable(prices=[("text-default", _price("USD", {"input": 3.0}))])


def _table(currency: str, tokens) -> PricingTable:
    """A one-model pricing table, for the estimate tests."""
    return PricingTable(prices={"text-default": _price(currency, tokens)})


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


_RECORD_KEYS = frozenset(
    {
        "invocation_id",
        "timestamp",
        "namespace",
        "tenant",
        "public_model",
        "route_model_id",
        "operation",
        "capabilities",
        "upstream_model",
        "status",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "total_tokens",
        "estimated_tokens",
        "latency_ms",
        "attempts",
        "fallback_used",
        "error_reason",
        "estimated_cost",
        "cost_currency",
    }
)
"""Every field a record carries, and therefore every key its projection must have."""


def test_the_projection_is_exhaustive_over_the_record():
    """A field added to the record cannot be silently absent from what operators export.

    The check is made against the dataclass rather than against a hand-written list,
    so it is the *record* that defines the schema: a new field with no projection
    fails here, and a projection key with no field fails here too.
    """
    assert {field.name for field in dataclasses.fields(UsageRecord)} == _RECORD_KEYS
    assert set(_record().to_dict()) == _RECORD_KEYS


def test_a_record_carries_every_fact_section_13_asks_for():
    record = _record()
    projected = record.to_dict()

    assert record.status == "completed"
    assert record.timestamp.tzinfo is UTC
    assert record.namespace == "cred-digest"
    assert record.tenant == "tenant-a"
    assert record.public_model == "text-default"
    assert record.route_model_id == "upstream-text"
    assert record.operation is Operation.TEXT
    assert record.capabilities == frozenset({"chat"})
    assert record.upstream_model == "upstream-text"
    assert (
        record.input_tokens,
        record.output_tokens,
        record.cached_tokens,
        record.total_tokens,
    ) == (10, 5, 2, 15)
    assert record.attempts == 2
    assert record.fallback_used is True
    assert record.error_reason is None
    assert record.latency_ms > 0

    assert projected["invocation_id"] == "inv-accounting-1"
    assert projected["route_model_id"] == "upstream-text"
    assert projected["operation"] == "text"
    assert projected["capabilities"] == ["chat"]
    assert projected["total_tokens"] == 15
    assert projected["timestamp"] == record.timestamp.isoformat()


def test_a_record_separates_reported_counts_from_derived_ones():
    """Provenance travels with the record, so a derived count is never a reported one."""
    recorder = AccountingRecorder(
        MemoryAccountingSink(), _table("USD", {"input": 3.0, "cached": 0.3})
    )

    usage = Usage(input_tokens=10, output_tokens=None, cached_tokens=2)
    record = recorder.record(
        _request(),
        outcome(usage=usage),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=2,
        fallback_used=True,
    )

    assert record.input_tokens == 10
    assert record.output_tokens is None
    assert record.cached_tokens == 2
    # Nothing was derived here: the provider reported every count it gave.
    assert record.estimated_tokens == frozenset()
    # And the cost covers exactly the categories the entry prices, all of them known.
    assert record.estimated_cost == pytest.approx(3.06e-05)


def test_a_derived_count_is_marked_estimated_in_the_record():
    """A count the gateway computed is never recorded as one the provider reported."""
    recorder = AccountingRecorder(MemoryAccountingSink(), None)

    usage = Usage(
        input_tokens=10,
        total_tokens=10,
        estimated_fields=frozenset({"total_tokens"}),
    )
    record = recorder.record(
        _request(),
        outcome(usage=usage),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=1,
        fallback_used=False,
    )

    assert record.total_tokens == 10
    assert record.estimated_tokens == frozenset({"total_tokens"})
    assert record.to_dict()["estimated_tokens"] == ["total_tokens"]


def test_a_record_refuses_a_route_of_another_operation():
    """The route is an audit fact, so a mismatched pairing is refused rather than recorded.

    A caller that handed the recorder the wrong route would produce a record that
    looks valid and is not, and nothing downstream compares the two. The pairing is
    known only where the record is built, so it is checked there rather than
    trusted.
    """
    recorder = AccountingRecorder(MemoryAccountingSink(), None)

    with pytest.raises(ValueError, match="embeddings"):
        recorder.record(
            _request(),
            outcome(),
            route(operation=Operation.EMBEDDINGS, capabilities=frozenset({Capability.EMBEDDINGS})),
            namespace="cred-digest",
            latency_ms=1.0,
            attempts=1,
            fallback_used=False,
        )


def _price(currency: str, tokens):
    from model_gateway.accounting import ModelPrice

    return ModelPrice(currency=currency, tokens=tokens)


# --------------------------------------------------------------------------- #
# The sinks
# --------------------------------------------------------------------------- #


def test_a_null_sink_discards_records():
    sink = NullAccountingSink()

    sink.record(_record())

    assert len(sink) == 0


def test_a_memory_sink_holds_records_and_drops_the_oldest():
    """Eviction drops the oldest once full, so the newest records survive."""
    sink = MemoryAccountingSink(capacity=2)
    first = _record()
    second = _record()
    third = _record()

    sink.record(first)
    sink.record(second)
    sink.record(third)

    assert sink.records == (second, third)


def test_a_memory_sink_of_capacity_one_holds_only_the_newest_record():
    """The bound is a bound at every capacity, including the smallest one."""
    sink = MemoryAccountingSink(capacity=1)

    sink.record(_record())
    newest = _record()
    sink.record(newest)

    assert sink.records == (newest,)
    assert len(sink) == 1


def test_a_memory_sink_never_exceeds_its_capacity():
    """Many insertions past the bound leave exactly the newest records, in order."""
    sink = MemoryAccountingSink(capacity=3)
    records = [_record() for _ in range(20)]

    for record in records:
        sink.record(record)

    assert sink.records == tuple(records[-3:])
    assert len(sink) == 3


def test_a_memory_sink_refuses_a_capacity_that_is_not_a_positive_integer():
    """A bound that is not a bound is a configuration mistake, not a sink."""
    for bad in (0, -1, True, 1.5, "3", None):
        with pytest.raises(ValueError):
            MemoryAccountingSink(capacity=bad)


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


async def test_the_service_records_a_completed_answer_when_accounting_is_enabled():
    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
        accounting_enabled=True,
    )

    await service.invoke(_request(), namespace="cred-digest")

    assert len(sink.records) == 1


async def test_the_service_does_not_record_when_accounting_is_disabled():
    """A deployment that turns accounting off is not silently accounting."""
    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
        accounting_enabled=False,
    )

    await service.invoke(_request(), namespace="cred-digest")

    assert len(sink) == 0


async def test_a_recording_failure_is_logged_and_does_not_fail_the_invocation():
    """A broken store must not withhold an answer the caller is waiting on.

    Surviving is only half the contract: the failure must also reach the operator,
    because a silently swallowed one would turn a broken store into a month of
    unaccounted usage. The log is therefore asserted, not assumed.
    """
    sink = _ExplodingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    with capture_logs() as logs:
        answered = await service.invoke(_request(), namespace="cred-digest")

    assert answered.text == "hello"
    assert sink.attempts == 1
    reported = _storage_failures(logs)
    assert len(reported) == 1
    assert reported[0]["error_type"] == "RuntimeError"
    assert reported[0]["invocation_id"] == answered.invocation_id


async def test_a_recorder_defect_does_not_fail_the_invocation():
    """The service's guard covers a recorder that fails before it reaches the sink.

    Wrapping only the final write would leave a defect in record construction or
    cost estimation able to replace the caller's answer with an error. The service
    wraps the whole call, and this proves that boundary rather than the sink's.
    """

    class _BrokenPricing:
        def estimate(self, usage, model):
            raise RuntimeError("the pricing table is broken")

    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, _BrokenPricing()),
    )

    with capture_logs() as logs:
        answered = await service.invoke(_request(), namespace="cred-digest")

    assert answered.text == "hello"
    assert len(sink) == 0
    assert len(_storage_failures(logs)) == 1


def _storage_failures(logs):
    """The log entries reporting that a usage record could not be stored."""
    return [entry for entry in logs if entry["event"] == "usage record could not be stored"]


class _ExplodingSink:
    """A store that fails the way a defective one would, to prove the failure surfaces."""

    def __init__(self):
        self.attempts = 0

    def record(self, record):
        self.attempts += 1
        raise RuntimeError("the accounting store is broken")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _record():
    return AccountingRecorder(MemoryAccountingSink(), PricingTable()).record(
        _request(),
        outcome(),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=2,
        fallback_used=True,
    )


def _request(**overrides):
    from model_gateway.domain import InvocationRequest, Message, Role, TextPart

    values: dict[str, object] = {
        "model": "text-default",
        "messages": (Message(role=Role.USER, content=(TextPart("hello"),)),),
        "tenant_id": "tenant-a",
        "idempotency_key": None,
    }
    values.update(overrides)
    return InvocationRequest(**values)  # type: ignore[arg-type]


def _one_route_registry():
    from model_gateway.registry import ModelRegistry

    return ModelRegistry((route(),))


class _EchoProvider:
    """The deterministic provider the other suites use: one scripted answer."""

    def __init__(self):
        self.calls = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        return response()


class _FlakyProvider:
    """Fails the first call with a given gateway error, then answers.

    A fallback is only walkable for a failure another route may recover from, so
    a provider that cannot fail at all cannot produce one; and the walk decides
    from the gateway's own taxonomy, not from an exception type. It answers all
    three operations, so one scripted failure drives a text, an embeddings or an
    image fallback through the same helper.
    """

    def __init__(self, error):
        self._error = error
        self.calls = []
        self._embedding = _EmbeddingProvider()
        self._image = _ImageProvider()

    def _maybe_fail(self, route, request, context):
        self.calls.append((route, request, context))
        if len(self.calls) == 1:
            raise self._error

    async def complete(self, route, request, context):
        self._maybe_fail(route, request, context)
        return response()

    async def embed(self, route, request, context):
        self._maybe_fail(route, request, context)
        return await self._embedding.embed(route, request, context)

    async def generate(self, route, request, context):
        self._maybe_fail(route, request, context)
        return await self._image.generate(route, request, context)


# --------------------------------------------------------------------------- #
# What the service attributes
# --------------------------------------------------------------------------- #


async def test_the_record_carries_the_trusted_principal_and_the_route_facts():
    """The record attributes the answer to the credential, not to the request."""
    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.invoke(_request(), namespace="cred-digest")

    record = sink.records[0]
    assert record.namespace == "cred-digest"
    assert record.tenant == "tenant-a"
    assert record.public_model == "text-default"
    assert record.upstream_model == "upstream-text"
    assert record.operation is Operation.TEXT
    assert record.capabilities == frozenset({"chat"})
    assert record.latency_ms > 0
    assert record.attempts == 1
    assert record.fallback_used is False


async def test_an_unconfigured_price_table_records_usage_with_no_cost():
    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, None),
    )

    await service.invoke(_request(), namespace="cred-digest")

    assert sink.records[0].estimated_cost is None


# --------------------------------------------------------------------------- #
# The factory wiring
# --------------------------------------------------------------------------- #


def test_the_factory_builds_a_recorder_when_usage_tracking_is_on():
    from model_gateway.app import _accounting
    from model_gateway.config.settings import Settings

    recorder = _accounting(Settings.from_env({"USAGE_TRACKING_ENABLED": "true"}))

    assert isinstance(recorder, AccountingRecorder)


def test_the_factory_builds_no_recorder_when_usage_tracking_is_off():
    from model_gateway.app import _accounting
    from model_gateway.config.settings import Settings

    assert _accounting(Settings.from_env({"USAGE_TRACKING_ENABLED": "false"})) is None


def test_cost_tracking_without_usage_records_usage_and_no_cost():
    """Usage is recorded but priced against nothing, which is the honest state."""
    from model_gateway.app import _accounting
    from model_gateway.config.settings import Settings

    recorder = _accounting(
        Settings.from_env(
            {
                "USAGE_TRACKING_ENABLED": "true",
                "COST_TRACKING_ENABLED": "true",
                "USAGE_STORAGE": "none",
            }
        )
    )

    assert recorder._pricing is not None
    assert recorder._pricing.prices == {}


def test_a_durable_storage_mode_without_persistence_is_refused():
    """Asking for durability the deployment cannot provide fails at startup."""
    from model_gateway.app import _accounting
    from model_gateway.config.settings import Settings

    with pytest.raises(ConfigurationError):
        _accounting(
            Settings.from_env(
                {
                    "USAGE_TRACKING_ENABLED": "true",
                    "USAGE_STORAGE": "sqlite",
                }
            )
        )


# --------------------------------------------------------------------------- #
# Route attribution on fallback
# --------------------------------------------------------------------------- #


async def test_a_fallback_answer_is_recorded_against_the_route_that_served_it():
    """The record names the route that answered, not the one that was asked for.

    Before the fix, accounting was handed the plan's primary route, so a fallback
    answer was recorded against a route that never produced it: upstream_model
    would be credited with an answer it failed to give, and the route that
    actually served the request would appear to have served nothing.
    """
    from model_gateway.errors import UnavailableError

    sink = MemoryAccountingSink()
    primary = route("upstream-model")
    fallback = route("fallback-model")
    provider = _FlakyProvider(UnavailableError("the primary is down"))
    service = InferenceService(
        ModelRegistry((primary, fallback)),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    result = await service.invoke(_request(), namespace="cred-digest")

    assert result.fallback_used is True
    record = sink.records[0]
    assert record.route_model_id == "fallback-model"
    assert record.upstream_model == "fallback-model"
    assert len(provider.calls) == 2


async def test_a_primary_answer_is_still_recorded_against_the_primary():
    """The common case: no fallback, and the record names the serving route."""
    sink = MemoryAccountingSink()
    provider = _EchoProvider()
    service = InferenceService(
        ModelRegistry((route(),)),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.invoke(_request(), namespace="cred-digest")

    assert sink.records[0].route_model_id == "upstream-text"


async def test_one_invocation_produces_exactly_one_record():
    """The record belongs to the invocation, never to the caller's own retries.

    A caller that drives `invoke` inside its own retry loop must not end up with two
    records for a single invocation: the record is written inside the service, once
    per invocation that completed, and the identifier it carries is the one that
    invocation produced. This guards that ownership from either side — a retry that
    re-entered the service, or a record written per attempt instead of per invocation.
    """

    class _NumberedProvider:
        """Answers with a distinct invocation identifier per call."""

        def __init__(self):
            self.count = 0

        async def complete(self, route, request, context):
            self.count += 1
            return response(invocation_id=f"inv-{self.count}")

    sink = MemoryAccountingSink()
    service = InferenceService(
        _one_route_registry(),
        _NumberedProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    first = await service.invoke(_request(), namespace="cred-digest")
    second = await service.invoke(_request(), namespace="cred-digest")

    assert len(sink.records) == 2
    assert [record.invocation_id for record in sink.records] == [
        first.invocation_id,
        second.invocation_id,
    ]


# --------------------------------------------------------------------------- #
# Every operation records
# --------------------------------------------------------------------------- #


async def test_an_embeddings_fallback_is_recorded_against_the_route_that_served_it():
    """The walk's own facts reach an embeddings record, not only a text one.

    A failed primary followed by a serving fallback is the case that exercises
    both facts at once: the attempt count is two, the fallback flag is set, and
    the route named is the one that answered. An embeddings answer carries neither
    fact (PRD section 5.2), so this is the walk's report and not the answer's.
    """
    from model_gateway.errors import UnavailableError

    sink = MemoryAccountingSink()
    provider = _FlakyProvider(UnavailableError("the primary is down"))
    service = InferenceService(
        ModelRegistry((_embedding_route("embed-primary"), _embedding_route("embed-fallback"))),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.embed(_embedding_request(), namespace="cred-digest")

    record = sink.records[0]
    assert record.attempts == 2
    assert record.fallback_used is True
    assert record.route_model_id == "embed-fallback"
    assert record.upstream_model == "embed-fallback"
    assert record.operation is Operation.EMBEDDINGS


async def test_an_image_fallback_is_recorded_against_the_route_that_served_it():
    from model_gateway.errors import UnavailableError

    sink = MemoryAccountingSink()
    provider = _FlakyProvider(UnavailableError("the primary is down"))
    service = InferenceService(
        ModelRegistry((_image_route("image-primary"), _image_route("image-fallback"))),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.generate(_image_request(), namespace="cred-digest")

    record = sink.records[0]
    assert record.attempts == 2
    assert record.fallback_used is True
    assert record.route_model_id == "image-fallback"
    assert record.operation is Operation.IMAGE


def test_a_record_carries_the_status_and_reason_a_failed_answer_states():
    """A streamed answer that failed after output is recorded as failed, with its reason.

    The three statuses are the streaming walk's: a non-streaming invocation that
    produced an answer always completed, because a failure there produces no
    answer to record.
    """
    recorder = AccountingRecorder(MemoryAccountingSink(), None)

    failed = recorder.record(
        _request(),
        outcome(status=STATUS_FAILED, error_reason="upstream_error"),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=1,
        fallback_used=False,
    )
    stopped = recorder.record(
        _request(),
        outcome(status=STATUS_INCOMPLETE),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=1,
        fallback_used=False,
    )
    completed = recorder.record(
        _request(),
        outcome(),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=1,
        fallback_used=False,
    )

    assert failed.status == STATUS_FAILED
    assert failed.error_reason == "upstream_error"
    assert stopped.status == STATUS_INCOMPLETE
    # A stream that stopped without a terminal event declared no failure, so it
    # carries no reason for one.
    assert stopped.error_reason is None
    assert completed.status == STATUS_COMPLETED
    assert completed.error_reason is None


def _embedding_route(model_id: str = "upstream-embed") -> Route:
    return route(
        model_id,
        operation=Operation.EMBEDDINGS,
        capabilities=frozenset({Capability.EMBEDDINGS}),
    )


def _image_route(model_id: str = "upstream-image") -> Route:
    return route(
        model_id,
        operation=Operation.IMAGE,
        capabilities=frozenset({Capability.GENERATION}),
    )


def _embedding_request():
    from model_gateway.domain.embeddings import EmbeddingRequest

    return EmbeddingRequest(model="embedding-default", input=("hello",))


def _image_request():
    from model_gateway.domain.images import ImageRequest

    return ImageRequest(model="image-default", prompt="a red fox")


class _EmbeddingProvider:
    """One vector per input text, which is what the gateway checks an answer against."""

    def __init__(self):
        self.calls = []

    async def embed(self, route, request, context):
        from model_gateway.domain.embeddings import Embedding, EmbeddingResponse

        self.calls.append((route, request, context))
        return EmbeddingResponse(
            model=request.model,
            upstream_model=route.model_id,
            data=tuple(
                Embedding(index=index, values=(0.1, 0.2)) for index, _ in enumerate(request.input)
            ),
            usage=Usage(input_tokens=4),
            invocation_id="provider-ignored",
        )


class _ImageProvider:
    """One image, which is what a request stating no count asks for."""

    def __init__(self):
        self.calls = []

    async def generate(self, route, request, context):
        from model_gateway.domain.images import GeneratedImage, ImageResponse

        self.calls.append((route, request, context))
        return ImageResponse(
            model=request.model,
            upstream_model=route.model_id,
            created=0,
            data=(GeneratedImage(b64_json="Zm9v"),),
            usage=Usage(input_tokens=5),
            invocation_id="provider-ignored",
        )


async def test_an_embeddings_invocation_is_recorded():
    """§13 asks for a record of every invocation that produced an answer."""
    sink = MemoryAccountingSink()
    service = InferenceService(
        ModelRegistry((_embedding_route(),)),
        _EmbeddingProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.embed(_embedding_request(), namespace="cred-digest")

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.operation is Operation.EMBEDDINGS
    assert record.capabilities == frozenset({"embeddings"})
    assert record.public_model == "embedding-default"
    assert record.route_model_id == "upstream-embed"
    assert record.upstream_model == "upstream-embed"
    assert record.namespace == "cred-digest"
    assert record.input_tokens == 4
    assert record.attempts == 1
    assert record.fallback_used is False


async def test_an_image_invocation_is_recorded():
    sink = MemoryAccountingSink()
    service = InferenceService(
        ModelRegistry((_image_route(),)),
        _ImageProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.generate(_image_request(), namespace="cred-digest")

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.operation is Operation.IMAGE
    assert record.capabilities == frozenset({"generation"})
    assert record.public_model == "image-default"
    assert record.upstream_model == "upstream-image"
    assert record.namespace == "cred-digest"
    assert record.attempts == 1


async def test_the_tenant_is_absent_for_an_operation_whose_request_carries_none():
    """The canonical embeddings and image requests have no tenant field (§5.1).

    The tenant an invocation acts under is the credential's, so the record states
    a tenant only where the request stated one rather than borrowing the
    namespace's value for a field the request never carried.
    """
    sink = MemoryAccountingSink()
    service = InferenceService(
        ModelRegistry((_embedding_route(),)),
        _EmbeddingProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    await service.embed(_embedding_request(), namespace="cred-digest")

    assert sink.records[0].tenant is None


async def test_an_embeddings_refusal_leaves_no_record():
    """A refusal is not an answer, so there is nothing to account for."""
    from model_gateway.errors import CapabilityError

    class _NotAnEmbeddingProvider:
        """A text provider, which cannot embed and so is refused the request."""

        async def complete(self, route, request, context):
            raise AssertionError("the gateway must not call a provider it refused")

    sink = MemoryAccountingSink()
    service = InferenceService(
        ModelRegistry((_embedding_route(),)),
        _NotAnEmbeddingProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    with pytest.raises(CapabilityError):
        await service.embed(_embedding_request(), namespace="cred-digest")

    assert len(sink) == 0


# --------------------------------------------------------------------------- #
# A streamed answer records too
# --------------------------------------------------------------------------- #


def _event(kind, sequence, **overrides) -> StreamEvent:
    values: dict[str, object] = {
        "kind": kind,
        "sequence": sequence,
        "invocation_id": "inv-stream-1",
    }
    values.update(overrides)
    return StreamEvent(**values)  # type: ignore[arg-type]


class _ScriptedStreamProvider:
    """A streaming provider that serves one scripted event stream per call.

    An item is a :class:`StreamEvent`, which is yielded, or an exception, which is
    raised at that point — so a script can yield some events and then fail
    mid-stream, or produce nothing at all. Each event is stamped with the
    invocation identifier the attempt was handed, which is what a real provider
    does: the gateway's identifier is the one the attempt context carries, and a
    scripted event does not get to name its own invocation.
    """

    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, route, request, context):
        self.calls.append((route, request, context))
        script = self._scripts[min(len(self.calls), len(self._scripts)) - 1]
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield dataclasses.replace(item, invocation_id=context.invocation_id)


def _streaming_service(provider, sink, **overrides) -> InferenceService:
    return InferenceService(
        _one_route_registry(),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
        **overrides,
    )


async def _drain(stream) -> list[StreamEvent]:
    return [event async for event in stream]


async def test_a_completed_stream_is_recorded():
    """§13 asks for a record of every invocation that produced an answer, streams included."""
    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [
                _event(StreamEventKind.CONTENT, 0, text="hello"),
                _event(
                    StreamEventKind.USAGE,
                    1,
                    usage=Usage(input_tokens=7, output_tokens=3, total_tokens=10),
                ),
                _event(StreamEventKind.FINISH, 2, finish_reason=FinishReason.STOP),
            ]
        ),
        sink,
    )

    await _drain(
        service.invoke_stream(_request(invocation_id="inv-stream-1"), namespace="cred-digest")
    )

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.status == STATUS_COMPLETED
    assert record.invocation_id == "inv-stream-1"
    assert record.namespace == "cred-digest"
    assert record.public_model == "text-default"
    assert record.route_model_id == "upstream-text"
    assert record.upstream_model == "upstream-text"
    assert record.operation is Operation.TEXT
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (7, 3, 10)
    assert record.attempts == 1
    assert record.fallback_used is False
    assert record.error_reason is None


async def test_a_stream_that_produced_nothing_leaves_no_record():
    """A refusal before the first event is not an answer, so nothing is accounted for."""
    sink = MemoryAccountingSink()
    service = _streaming_service(_ScriptedStreamProvider([]), sink)

    await _drain(
        service.invoke_stream(_request(invocation_id="inv-stream-1"), namespace="cred-digest")
    )

    assert len(sink) == 0


async def test_a_stream_that_failed_after_output_is_recorded_as_failed():
    """The failure's own gateway reason travels with the record."""
    from model_gateway.errors import UnavailableError

    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [
                _event(StreamEventKind.CONTENT, 0, text="hel"),
                UnavailableError("the upstream went away"),
            ]
        ),
        sink,
    )

    with pytest.raises(UnavailableError):
        await _drain(
            service.invoke_stream(
                _request(invocation_id="inv-stream-1"), namespace="cred-digest"
            )
        )

    record = sink.records[0]
    assert record.status == STATUS_FAILED
    assert record.error_reason == "provider_unavailable"
    assert record.route_model_id == "upstream-text"


async def test_a_stream_that_stopped_without_a_terminal_event_is_incomplete():
    """A truncated stream is not a failed one, and the record does not report it as one."""
    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [
                _event(StreamEventKind.CONTENT, 0, text="hel"),
                _event(StreamEventKind.CONTENT, 1, text="lo"),
            ]
        ),
        sink,
    )

    await _drain(
        service.invoke_stream(_request(invocation_id="inv-stream-1"), namespace="cred-digest")
    )

    record = sink.records[0]
    assert record.status == STATUS_INCOMPLETE
    assert record.error_reason is None
    assert record.route_model_id == "upstream-text"


async def test_a_stream_the_caller_abandons_is_still_recorded():
    """The provider was called, so the gateway owes for the output it produced."""
    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [
                _event(StreamEventKind.CONTENT, 0, text="hel"),
                _event(StreamEventKind.FINISH, 1, finish_reason=FinishReason.STOP),
            ]
        ),
        sink,
    )

    stream = service.invoke_stream(_request(invocation_id="inv-stream-1"), namespace="cred-digest")
    first = await stream.__anext__()
    assert first.text == "hel"
    await stream.aclose()

    assert len(sink.records) == 1
    assert sink.records[0].status == STATUS_INCOMPLETE


async def test_a_streamed_answer_is_not_recorded_when_accounting_is_disabled():
    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [_event(StreamEventKind.FINISH, 0, finish_reason=FinishReason.STOP)]
        ),
        sink,
        accounting_enabled=False,
    )

    await _drain(
        service.invoke_stream(_request(invocation_id="inv-stream-1"), namespace="cred-digest")
    )

    assert len(sink) == 0


# --------------------------------------------------------------------------- #
# The authenticated principal reaches a record through the HTTP surface
# --------------------------------------------------------------------------- #

_KEY_A = "sk-gateway-key-a"
_KEY_B = "sk-gateway-key-b"


def _embeddings_app(sink) -> Application:
    """An application serving only the embeddings endpoint, with authentication on.

    Built by hand rather than through the factory so the test can hand the service
    the sink it will read. The point is the transport's own contribution — which
    principal a request is recorded under — and the factory builds its sink where
    a test cannot reach it.
    """
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_KEY_A}:tenant-a,{_KEY_B}:tenant-b",
        }
    )
    service = InferenceService(
        ModelRegistry((_embedding_route(),)),
        _EmbeddingProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )
    app = Application(router=Router())
    register_error_handlers(app)
    register_embeddings_routes(app.router, service)
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


async def test_two_credentials_record_under_two_namespaces_through_the_http_surface():
    """The handler's contribution: which principal an authenticated request records under.

    Two callers presenting two different keys are two principals, and the record
    each invocation leaves must name its own. A direct service test cannot show
    this, because it supplies the namespace itself; the check is therefore made
    through the real middleware and the real handler, which is the wiring that
    previously dropped the identity entirely.
    """
    sink = MemoryAccountingSink()
    app = _embeddings_app(sink)

    await app.start()
    try:
        client = TestClient(app)
        for key in (_KEY_A, _KEY_B):
            response = await client.post(
                "/v1/embeddings",
                content=JSONContent({"model": "embedding-default", "input": "hello"}),
                headers={"Authorization": f"Bearer {key}"},
            )
            assert response.status == 200
    finally:
        await app.stop()

    assert len(sink.records) == 2
    namespaces = [record.namespace for record in sink.records]
    assert all(namespaces)
    assert namespaces[0] != namespaces[1]
    # The canonical embeddings request states no tenant, so neither record does —
    # the principal is carried by the namespace and not invented as a tenant.
    assert [record.tenant for record in sink.records] == [None, None]


# --------------------------------------------------------------------------- #
# The accounting attempt survives the teardown
# --------------------------------------------------------------------------- #


class _BlockingCloseProvider:
    """A streaming provider that parks while it is being closed.

    It is how a test gets a cancellation to land inside the teardown: the provider
    yields one event and then blocks in its own ``finally``, so closing the
    service's stream — and every generator between it and this one — is suspended
    when the consumer is cancelled.
    """

    def __init__(self, closing, release):
        self._closing = closing
        self._release = release

    async def stream(self, route, request, context):
        try:
            yield dataclasses.replace(
                _event(StreamEventKind.CONTENT, 0, text="hel"),
                invocation_id=context.invocation_id,
            )
        finally:
            self._closing.set()
            await self._release.wait()


async def test_a_stream_cancelled_while_it_is_being_closed_is_still_recorded():
    """A caller cancelled mid-teardown must not take the record with it.

    The stream is closed before the record is written and that close is awaited,
    so a cancellation delivered while it is in flight is exactly the case that
    would otherwise skip the accounting attempt: the gateway would have called the
    provider and recorded nothing. The record is written anyway, the status says
    the stream stopped without a terminal event, and the cancellation still
    propagates — nothing here swallows it.
    """
    sink = MemoryAccountingSink()
    closing = asyncio.Event()
    release = asyncio.Event()
    service = _streaming_service(_BlockingCloseProvider(closing, release), sink)

    async def consume():
        # The close is awaited inside this task on purpose: a consumer that only
        # breaks out of the loop leaves the generator to be finalised by the event
        # loop in another task, and the cancellation would then have nothing to
        # interrupt.
        stream = service.invoke_stream(
            _request(invocation_id="inv-stream-1"), namespace="cred-digest"
        )
        await stream.__anext__()
        await stream.aclose()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(closing.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)
    release.set()

    assert len(sink.records) == 1
    assert sink.records[0].status == STATUS_INCOMPLETE
    assert sink.records[0].namespace == "cred-digest"


async def test_a_streamed_invocation_without_an_identifier_is_given_one():
    """The gateway mints the identifier, so the events and the record name one invocation."""
    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [_event(StreamEventKind.FINISH, 0, finish_reason=FinishReason.STOP)]
        ),
        sink,
    )

    events = await _drain(service.invoke_stream(_request(), namespace="cred-digest"))

    record = sink.records[0]
    assert record.invocation_id
    assert record.invocation_id == events[0].invocation_id


async def test_a_failed_stream_records_the_reason_its_terminal_event_carries():
    """One normalised failure, so the event and the record cannot disagree about why."""
    from model_gateway.errors import UnavailableError

    sink = MemoryAccountingSink()
    service = _streaming_service(
        _ScriptedStreamProvider(
            [
                _event(StreamEventKind.CONTENT, 0, text="hel"),
                UnavailableError("the upstream went away"),
            ]
        ),
        sink,
    )

    events = []
    with pytest.raises(UnavailableError):
        async for event in service.invoke_stream(
            _request(invocation_id="inv-stream-1"), namespace="cred-digest"
        ):
            events.append(event)

    terminal = events[-1]
    record = sink.records[0]
    assert terminal.kind is StreamEventKind.ERROR
    assert record.status == STATUS_FAILED
    assert record.error_reason == terminal.error_reason


def test_an_outcome_refuses_a_status_and_a_reason_that_cannot_both_be_true():
    """The vocabulary is only meaningful if the two fields agree."""
    with pytest.raises(ValueError, match="must be one of"):
        outcome(status="finished")
    with pytest.raises(ValueError, match="must carry the gateway reason"):
        outcome(status=STATUS_FAILED)
    with pytest.raises(ValueError, match="only a failed outcome"):
        outcome(status=STATUS_COMPLETED, error_reason="provider_unavailable")
    with pytest.raises(ValueError, match="only a failed outcome"):
        outcome(status=STATUS_INCOMPLETE, error_reason="provider_unavailable")


def _chat_app(sink, provider) -> Application:
    """An application serving only the chat endpoint, with authentication on.

    The streaming counterpart of :func:`_embeddings_app`: the embeddings test
    proves the identity reaches a record through the non-streaming surface, and
    this proves it through the streaming one, which is a different handler and
    therefore a different wiring to regress.
    """
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_KEY_A}:tenant-a,{_KEY_B}:tenant-b",
        }
    )
    service = InferenceService(
        ModelRegistry((route(),)),
        provider,
        accounting=AccountingRecorder(sink, PricingTable()),
    )
    app = Application(router=Router())
    register_error_handlers(app)
    register_chat_routes(app.router, service)
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


async def test_two_credentials_record_under_two_namespaces_through_the_streaming_surface():
    """The streaming handler forwards the identity too.

    A test of the non-streaming surface would keep passing if the streaming branch
    stopped forwarding the principal, because the two are different handlers.
    """
    sink = MemoryAccountingSink()
    app = _chat_app(
        sink,
        _ScriptedStreamProvider(
            [_event(StreamEventKind.FINISH, 0, finish_reason=FinishReason.STOP)]
        ),
    )

    await app.start()
    try:
        client = TestClient(app)
        for key in (_KEY_A, _KEY_B):
            response = await client.post(
                "/v1/chat/completions",
                content=JSONContent(
                    {
                        "model": "text-default",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    }
                ),
                headers={"Authorization": f"Bearer {key}"},
            )
            assert response.status == 200
            assert await response.text()

        assert len(sink.records) == 2
        namespaces = [record.namespace for record in sink.records]
        assert all(namespaces)
        assert namespaces[0] != namespaces[1]
        assert [record.status for record in sink.records] == [
            STATUS_COMPLETED,
            STATUS_COMPLETED,
        ]
    finally:
        await app.stop()
