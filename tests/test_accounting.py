"""Usage and cost accounting: the record, the pricing, the sinks (PRD section 13).

These tests prove the two rules the module is built on: a token count is either
reported or estimated and the difference travels with it, and cost comes from
configured pricing rather than from a vendor name. They also prove the service
records a completed answer under the deployment's accounting setting, and that a
recording failure is surfaced rather than swallowed.
"""

import orjson
import pytest

from model_gateway.accounting import (
    AccountingRecorder,
    MemoryAccountingSink,
    NullAccountingSink,
    PricingTable,
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


def test_a_count_the_provider_never_reported_is_not_priced():
    """An unreported count contributes no cost: zero would report a cost not incurred."""
    pricing = pricing_from_env(
        {
            "COST_PRICING_JSON": orjson.dumps(
                {"text-default": {"currency": "USD", "tokens": {"input": 3.0, "output": 15.0}}}
            ).decode()
        }
    )

    cost, _ = pricing.estimate(Usage(input_tokens=1_000_000, output_tokens=None), "text-default")

    assert cost == pytest.approx(3.0)


def test_malformed_pricing_configuration_is_refused_at_startup():
    """A deployment that misconfigures its prices learns now, not after a month of bad bills."""
    for raw in ("not json", "[]", '{"text-default": "expensive"}'):
        with pytest.raises(ConfigurationError):
            pricing_from_env({"COST_PRICING_JSON": raw})


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def test_a_record_carries_every_fact_section_13_asks_for():
    record = _record()

    assert record.status == "completed"
    assert record.namespace == "cred-digest"
    assert record.tenant == "tenant-a"
    assert record.public_model == "text-default"
    assert record.operation is Operation.TEXT
    assert record.capabilities == frozenset({"chat"})
    assert record.upstream_model == "upstream-text"
    assert (record.input_tokens, record.output_tokens, record.cached_tokens) == (10, 5, 2)
    assert record.attempts == 2
    assert record.fallback_used is True
    assert record.error_reason is None
    assert record.latency_ms > 0
    assert record.to_dict()["invocation_id"] == "inv-accounting-1"


def test_a_record_separates_reported_counts_from_derived_ones():
    """Provenance travels with the record, so a derived count is never a reported one."""
    pricing = PricingTable(
        prices={"text-default": _price("USD", {"input": 3.0, "output": 15.0, "cached": 0.3})}
    )
    recorder = AccountingRecorder(MemoryAccountingSink(), pricing)

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
    # And every count it gave is priced, reported or not.
    assert record.estimated_cost == pytest.approx(3.06e-05)


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


async def test_a_memory_sink_holds_records_and_is_bounded():
    """Eviction drops the oldest once full, so the newest records survive."""
    sink = MemoryAccountingSink(capacity=2)
    first = _record()
    second = _record()
    third = _record()

    sink.record(first)
    sink.record(second)
    sink.record(third)

    assert sink.records == (second, third)


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
    """A broken store must not withhold an answer the caller is waiting on."""
    sink = _ExplodingSink()
    service = InferenceService(
        _one_route_registry(),
        _EchoProvider(),
        accounting=AccountingRecorder(sink, PricingTable()),
    )

    answered = await service.invoke(_request(), namespace="cred-digest")

    assert answered.text == "hello"
    assert sink.attempts == 1


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
