"""Idempotency in the inference service's non-streaming invocation (PRD section 12).

These tests drive the real :class:`~model_gateway.inference.InferenceService` over
a :class:`~model_gateway.providers.mock.MockProvider`, so what is asserted is the
wiring rather than the store: a repeat under one key is answered from the record
with ``REPLAYED`` and the provider is called once across both, a retry carrying a
freshly minted invocation identifier is still the same request, a key reused for
a different payload reaches the provider again and is answered with the second
answer, a request without a key never reaches the store, a failure is never
recorded so a retry is allowed to retry, and an unknown model is still a
model-not-found error even when the key has a cached record.
"""

import pytest

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    IdempotencyStatus,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    TextPart,
)
from model_gateway.errors import ModelNotFoundError, UnavailableError
from model_gateway.idempotency import (
    IdempotencyRecord,
    MemoryIdempotencyStore,
    RequestFingerprint,
)
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.registry import ModelRegistry

UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
EXTRA_HEADER_VALUE = "tenant-header-value"


def registry() -> ModelRegistry:
    """The one text route the gateway's own application tests build, primary first."""
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=UPSTREAM_ENDPOINT,
                base_url=UPSTREAM_ENDPOINT,
                model_id="upstream-text",
                format="openai",
                api_key=SECRET_KEY,
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={"x-tenant": EXTRA_HEADER_VALUE},
                allow_absolute_path_url=False,
            ),
        )
    )


def request(**overrides) -> InvocationRequest:
    """One canonical text request carrying an idempotency key."""
    values: dict[str, object] = {
        "model": "text-default",
        "invocation_id": "inv-1",
        "idempotency_key": "key-1",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
    }
    values.update(overrides)
    return InvocationRequest(**values)  # type: ignore[arg-type]


def service(store: MemoryIdempotencyStore, provider: MockProvider) -> InferenceService:
    """A service over the one-route registry, with the store injected."""
    return InferenceService(registry(), provider, idempotency_store=store)


def london() -> InvocationRequest:
    """The same key and model, a different question."""
    return request(
        messages=(
            Message(role=Role.USER, content=(TextPart(text="What is the weather in London?"),)),
        )
    )


async def test_a_repeat_under_one_key_is_answered_from_the_record() -> None:
    """One provider call, and the repeat served from the record with ``REPLAYED``."""
    store = MemoryIdempotencyStore()
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = service(store, provider)

    first = await gateway.invoke(request())
    second = await gateway.invoke(request())

    assert first.idempotency_status is IdempotencyStatus.NEW
    assert second.idempotency_status is IdempotencyStatus.REPLAYED
    assert second.text == first.text
    assert len(provider.calls) == 1
    assert len(store) == 1


async def test_a_repeat_with_a_freshly_minted_invocation_id_is_the_same_request() -> None:
    """Rule 2: the gateway mints the identifier, so it cannot distinguish requests."""
    store = MemoryIdempotencyStore()
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = service(store, provider)

    first = await gateway.invoke(request(invocation_id="inv-1"))
    second = await gateway.invoke(request(invocation_id="inv-2"))

    assert first.idempotency_status is IdempotencyStatus.NEW
    assert second.idempotency_status is IdempotencyStatus.REPLAYED
    assert second.invocation_id == first.invocation_id == "inv-1"
    assert len(provider.calls) == 1


async def test_one_key_reused_for_another_payload_is_served_as_a_new_request() -> None:
    """Rule 1: the fingerprint covers the payload, so a reused key is two requests."""
    store = MemoryIdempotencyStore()
    provider = MockProvider((MockSpec(text="sunny in Paris"), MockSpec(text="rainy in London")))
    gateway = service(store, provider)

    first = await gateway.invoke(request())
    second = await gateway.invoke(london())

    assert first.idempotency_status is IdempotencyStatus.NEW
    assert second.idempotency_status is IdempotencyStatus.NEW
    assert second.text == "rainy in London"
    assert second.text != first.text
    assert len(provider.calls) == 2
    assert len(store) == 1


async def test_a_request_without_a_key_never_reaches_the_store() -> None:
    """No key, no deduplication: the answer reports ``NOT_APPLICABLE`` and nothing is recorded."""
    store = MemoryIdempotencyStore()
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = service(store, provider)

    response = await gateway.invoke(request(idempotency_key=None))

    assert response.idempotency_status is IdempotencyStatus.NOT_APPLICABLE
    assert len(provider.calls) == 1
    assert len(store) == 0


async def test_a_failed_invocation_is_not_recorded() -> None:
    """Rule 3: a failure is never cached, so the retry is allowed to retry."""
    store = MemoryIdempotencyStore()
    provider = MockProvider(
        (
            MockSpec(error=UnavailableError("the upstream is down")),
            MockSpec(text="sunny in Paris"),
        )
    )
    gateway = service(store, provider)

    with pytest.raises(UnavailableError):
        await gateway.invoke(request())
    second = await gateway.invoke(request())

    assert second.idempotency_status is IdempotencyStatus.NEW
    assert second.text == "sunny in Paris"
    assert len(provider.calls) == 2
    assert len(store) == 1


async def test_an_unknown_model_is_still_model_not_found_with_a_cached_record() -> None:
    """The model is resolved before the record is consulted, so the ordering holds."""
    store = MemoryIdempotencyStore()
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = service(store, provider)

    await gateway.invoke(request())
    assert len(store) == 1

    with pytest.raises(ModelNotFoundError):
        await gateway.invoke(request(model="not-a-public-model"))

    assert len(provider.calls) == 1


async def test_a_service_without_an_injected_store_deduplicates_from_its_own() -> None:
    """The default is an in-memory store, so a service built plainly still deduplicates."""
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = InferenceService(registry(), provider)

    first = await gateway.invoke(request())
    second = await gateway.invoke(request())

    assert first.idempotency_status is IdempotencyStatus.NEW
    assert second.idempotency_status is IdempotencyStatus.REPLAYED
    assert len(provider.calls) == 1


class BrokenStore:
    """A store that fails the way a defective one would, to prove the failure surfaces."""

    def get(self, key: str) -> IdempotencyRecord | None:
        raise RuntimeError("the idempotency store is broken")

    def put(self, key: str, fingerprint: RequestFingerprint, response: InvocationResponse) -> None:
        raise RuntimeError("the idempotency store is broken")


async def test_a_store_failure_surfaces_rather_than_being_swallowed() -> None:
    """A store defect is a gateway defect: it must not be swallowed into a re-run."""
    provider = MockProvider(MockSpec(text="sunny in Paris"))
    gateway = InferenceService(registry(), provider, idempotency_store=BrokenStore())

    with pytest.raises(RuntimeError, match="the idempotency store is broken"):
        await gateway.invoke(request())

    assert provider.calls == []
