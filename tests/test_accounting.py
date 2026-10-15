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

import dataclasses
from datetime import UTC

import orjson
import pytest
from structlog.testing import capture_logs

from model_gateway.accounting import (
    AccountingRecorder,
    MemoryAccountingSink,
    NullAccountingSink,
    PricingTable,
    UsageRecord,
    pricing_from_env,
)
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.config.settings import ConfigurationError
from model_gateway.domain import FinishReason, InvocationResponse, TextPart, Usage
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry


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
        response(usage=usage),
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
        response(usage=usage),
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
            response(),
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
        response(),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=2,
        fallback_used=True,
    )


def _request():
    from model_gateway.domain import InvocationRequest, Message, Role, TextPart

    return InvocationRequest(
        model="text-default",
        messages=(Message(role=Role.USER, content=(TextPart("hello"),)),),
        tenant_id="tenant-a",
        idempotency_key=None,
    )


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
    """Fails the first call with a given gateway error, then answers like the echo.

    A fallback is only walkable for a failure another route may recover from, so
    a provider that cannot fail at all cannot produce one; and the walk decides
    from the gateway's own taxonomy, not from an exception type.
    """

    def __init__(self, error):
        self._error = error
        self.calls = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        if len(self.calls) == 1:
            raise self._error
        return response()


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


def test_the_attempt_facts_come_from_the_caller_not_from_the_answer():
    """The record's attempt count and fallback flag are the walk's, not the answer's.

    Only the text answer carries those two facts, so the recorder is handed them
    rather than reading them off the response: an embeddings or image answer has
    no field for either (PRD section 5.2), and a recorder that read the response
    would have to invent a value for those operations.
    """
    recorder = AccountingRecorder(MemoryAccountingSink(), PricingTable())

    record = recorder.record(
        _request(),
        response(attempt_count=7, fallback_used=True),
        route(),
        namespace="cred-digest",
        latency_ms=1.0,
        attempts=3,
        fallback_used=False,
    )

    assert record.attempts == 3
    assert record.fallback_used is False


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
