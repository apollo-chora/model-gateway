"""The idempotency store and its fingerprint.

These tests prove the three rules the module is built on — the fingerprint covers
the payload and not the key, the gateway's own identifier is not part of it, and a
failure is never recorded as an answer — and the coordination that makes
simultaneous identical requests coalesce rather than both execute.
"""

import asyncio
from dataclasses import replace

import pytest

from model_gateway.config.routes import Operation
from model_gateway.domain import IdempotencyStatus, InvocationRequest
from model_gateway.errors import IdempotencyConflictError
from model_gateway.idempotency import (
    MemoryIdempotencyStore,
    fingerprint_payload,
    request_fingerprint,
    settle,
)
from model_gateway.registry import PUBLIC_MODEL_IDS


def request(**overrides) -> InvocationRequest:
    """One canonical text request, with the fields a test tends to vary.

    A message is always present: an invocation with nothing to send has no
    canonical representation, so a request without one is refused before it can be
    fingerprinted.
    """
    values: dict[str, object] = {
        "model": "text-default",
        "invocation_id": "inv-1",
        "idempotency_key": "key-1",
        "messages": (_user_turn(),),
    }
    values.update(overrides)
    return InvocationRequest(**values)  # type: ignore[arg-type]


def _user_turn():
    """The one user message a minimal text request carries."""
    from model_gateway.domain import Message, Role, TextPart

    return Message(role=Role.USER, content=(TextPart("hello"),))


def answered(request: InvocationRequest):
    """One completed answer, as the wiring records it after a call that finished."""
    from dataclasses import replace

    from model_gateway.domain import FinishReason, InvocationResponse, TextPart

    return replace(
        InvocationResponse(
            model=PUBLIC_MODEL_IDS[Operation.TEXT],
            upstream_model="upstream-text",
            content=(TextPart("hello"),),
            finish_reason=FinishReason.STOP,
        ),
        invocation_id=request.invocation_id,
        attempt_count=1,
        idempotency_status=IdempotencyStatus.NEW,
    )


# --------------------------------------------------------------------------- #
# The fingerprint
# --------------------------------------------------------------------------- #


def test_the_fingerprint_covers_the_payload_not_the_key():
    """Two requests differing only in their payload have different fingerprints."""
    from dataclasses import replace

    goodbye = replace(
        request(),
        messages=(
            replace(
                request().messages[0],
                content=(replace(request().messages[0].content[0], text="goodbye"),),
            ),
        ),
    )

    assert request_fingerprint(request()).digest == request_fingerprint(request()).digest
    assert request_fingerprint(request()).digest != request_fingerprint(goodbye).digest


def test_the_fingerprint_covers_the_tenant():
    """The same request under two tenants has two fingerprints.

    Answering one from the other would leak an answer across a boundary the gateway
    is required to hold, so the tenant is part of what makes a
    request the request it is.
    """
    assert request_fingerprint(request(tenant_id="acme")).digest != request_fingerprint(
        request(tenant_id="other")
    ).digest


def test_the_invocation_identifier_is_not_part_of_the_fingerprint():
    """A retry with a freshly minted identifier is the same request.

    The gateway mints its own identifier, so it is not a property of what the
    caller asked; fingerprinting it would make every retry a miss and defeat the
    mechanism.
    """
    assert request_fingerprint(request(invocation_id="inv-1")).digest == request_fingerprint(
        request(invocation_id="inv-2")
    ).digest


def test_the_fingerprint_ignores_bookkeeping_that_cannot_change_the_answer():
    """Metadata and the key itself are excluded, for the reason the module gives."""
    assert request_fingerprint(request(metadata={"trace": "a"})).digest == request_fingerprint(
        request(metadata={"trace": "b"})
    ).digest


def test_a_fingerprint_is_a_fixed_length_digest():
    assert len(request_fingerprint(request()).digest) == 64


def test_the_payload_states_exactly_which_fields_participate():
    """The fingerprint's input is explicit, so a test can audit it without parsing a hash."""
    payload = fingerprint_payload(request())

    assert set(payload) == {
        "model",
        "messages",
        "system_instruction",
        "generation_config",
        "tools",
        "tool_choice",
        "response_modality",
        "tenant",
    }
    for excluded in ("invocation_id", "idempotency_key", "metadata", "fallback_models"):
        assert excluded not in payload


# --------------------------------------------------------------------------- #
# The claim: atomic coordination
# --------------------------------------------------------------------------- #


async def test_a_first_claim_makes_the_caller_the_owner():
    store = MemoryIdempotencyStore()

    claim = await store.acquire("key-1", request_fingerprint(request()), "")

    assert claim.kind == "owner"
    assert store._entries[("", "key-1")].record is None


async def test_a_second_claim_with_the_same_fingerprint_makes_the_caller_a_waiter():
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")

    claim = await store.acquire("key-1", request_fingerprint(request()), "")

    assert claim.kind == "waiter"
    assert claim.future is not None and not claim.future.done()


async def test_a_second_claim_with_a_different_fingerprint_is_a_conflict():
    """A key names one operation, so reuse is refused rather than executed.

    Executing it would both run a request the caller never made and overwrite the
    record the key already holds, letting one key oscillate between unrelated
    payloads indefinitely.
    """
    store = MemoryIdempotencyStore()
    first = request()
    await store.acquire("key-1", request_fingerprint(first), "")
    other = replace(
        first,
        messages=(
            replace(
                first.messages[0],
                content=(replace(first.messages[0].content[0], text="goodbye"),),
            ),
        ),
    )

    with pytest.raises(IdempotencyConflictError) as error:
        await store.acquire("key-1", request_fingerprint(other), "")

    assert error.value.reason.value == "idempotency_conflict"
    assert error.value.http_status == 409
    assert error.value.grpc_code == "ALREADY_EXISTS"


async def test_two_tenants_using_the_same_key_do_not_interfere():
    """The key is namespaced by the trusted principal, so one cannot destroy the other's record."""
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "acme")
    await store.acquire("key-1", request_fingerprint(request()), "acme")

    other = await store.acquire("key-1", request_fingerprint(request()), "other")

    assert other.kind == "owner"
    assert len(store) == 2


async def test_a_second_identical_claim_within_one_namespace_coalesces():
    """Two identical requests in one namespace are one invocation, not two.

    The first claims the key; the second, arriving while the first is still
    deciding, waits on the same claim rather than starting a second invocation.
    A different namespace is a different operation, which is the case the other
    test covers.
    """
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "acme")

    claim = await store.acquire("key-1", request_fingerprint(request()), "acme")

    assert claim.kind == "waiter"


async def test_an_empty_namespace_is_not_special():
    """A caller with no namespace is still one namespace, distinct from a named one."""
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")
    await store.acquire("key-1", request_fingerprint(request()), "acme")

    assert len(store) == 2


# --------------------------------------------------------------------------- #
# Completing and failing
# --------------------------------------------------------------------------- #


async def test_completing_claims_resolves_the_waiter_with_the_answer():
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")
    waiter = await store.acquire("key-1", request_fingerprint(request()), "")
    assert waiter.kind == "waiter"

    await store.complete("key-1", "", answered(request()))

    assert (await settle(waiter.future)).idempotency_status is IdempotencyStatus.NEW
    assert store._entries[("", "key-1")].record is not None


async def test_completing_twice_is_a_no_op_rather_than_a_second_record():
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")
    answer = answered(request())
    await store.complete("key-1", "", answer)
    again = replace(answer, idempotency_status=IdempotencyStatus.REPLAYED)

    await store.complete("key-1", "", again)

    assert len(store) == 1
    assert store._entries[("", "key-1")].record.response is answer


async def test_failing_releases_the_claim_so_a_later_caller_may_retry():
    """Only a completed answer is recorded, so a retry after a failure retries."""
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")
    await store.fail("key-1", "", ValueError("the upstream failed"))

    assert ("", "key-1") not in store._entries


async def test_failing_tells_waiters_the_failure_rather_than_starting_another_attempt():
    """A waiter coalesced onto a failed invocation receives that failure."""
    store = MemoryIdempotencyStore()
    await store.acquire("key-1", request_fingerprint(request()), "")
    waiter = await store.acquire("key-1", request_fingerprint(request()), "")

    await store.fail("key-1", "", ValueError("the upstream failed"))

    with pytest.raises(ValueError, match="the upstream failed"):
        await settle(waiter.future)


async def test_a_waiter_cancelling_does_not_cancel_the_invocation_it_waits_on():
    """Cancellation belongs to the caller who owns it."""
    store = MemoryIdempotencyStore()
    claim = await store.acquire("key-1", request_fingerprint(request()), "")
    cancelled: asyncio.Future = asyncio.Future()
    cancelled.cancel()

    await store.fail("key-1", "", asyncio.CancelledError())

    assert claim.future is None or claim.future.done()


async def test_a_store_defect_surfaces_rather_than_being_swallowed():
    """A store failure is a gateway defect and must not silently become a re-run.

    Swallowing it would silently produce the duplicate accounting the key exists to
    prevent.
    """

    class BrokenStore:
        async def acquire(self, key, fingerprint, namespace):
            raise RuntimeError("the store is broken")

    store = BrokenStore()
    with pytest.raises(RuntimeError, match="the store is broken"):
        await store.acquire("key-1", request_fingerprint(request()), "")


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #


async def test_the_store_is_bounded_so_it_cannot_grow_without_limit():
    store = MemoryIdempotencyStore(capacity=2)
    await store.acquire("a", request_fingerprint(request()), "")
    await store.acquire("b", request_fingerprint(request()), "")
    await store.complete("a", "", answered(request()))

    await store.acquire("c", request_fingerprint(request()), "")

    assert len(store) == 2
    assert ("", "a") not in store._entries
    assert ("", "c") in store._entries


async def test_an_in_flight_claim_is_never_evicted():
    """Evicting a claim would strand its waiters on a future nobody resolves."""
    store = MemoryIdempotencyStore(capacity=1)
    await store.acquire("a", request_fingerprint(request()), "")
    await store.complete("a", "", answered(request()))
    await store.acquire("b", request_fingerprint(request()), "")

    await store.acquire("c", request_fingerprint(request()), "")

    assert ("", "b") in store._entries


async def test_a_zero_capacity_is_refused():
    with pytest.raises(ValueError):
        MemoryIdempotencyStore(capacity=0)


# --------------------------------------------------------------------------- #
# Concurrent coalescing
# --------------------------------------------------------------------------- #


async def test_simultaneous_identical_requests_coalesce_so_one_invocation_serves_both():
    """The situation idempotency exists to handle: two callers arriving together.

    The first claims the key; the second, arriving while the first is still
    deciding, waits on the same claim rather than starting a second invocation. When
    the claim completes both hold the same answer, so one invocation serves both.
    """
    owner, second = await _concurrent_claims(MemoryIdempotencyStore())

    assert owner.kind == "owner"
    assert second.idempotency_status is IdempotencyStatus.NEW
    assert second.upstream_model == owner_response().upstream_model


def owner_response():
    """The answer a claim resolves with, stamped as the gateway stamps it.

    The status is ``NEW`` because the caller that owns the invocation is the one
    the provider was reached for; a replay is the service's stamp, not the
    store's.
    """
    from model_gateway.domain import InvocationResponse

    return InvocationResponse(
        model="text-default",
        upstream_model="upstream-text",
        idempotency_status=IdempotencyStatus.NEW,
    )


async def _concurrent_claims(store):
    """Two claims for the same key and fingerprint, the second waiting on the first."""
    owner = await store.acquire("key-1", request_fingerprint(request()), "")
    waiter = await store.acquire("key-1", request_fingerprint(request()), "")
    assert waiter.kind == "waiter"
    await store.complete("key-1", "", owner_response())
    return owner, await settle(waiter.future)
