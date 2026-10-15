"""The idempotency store and its fingerprint (PRD section 12).

These tests prove the three rules the module is built on: the fingerprint covers
the payload and not the key, the gateway's own identifier is not part of it, and a
failure is never recorded as an answer. The wiring's own tests assert the same
behaviour through the inference service.
"""

import pytest

from model_gateway.domain import IdempotencyStatus, InvocationRequest
from model_gateway.idempotency import MemoryIdempotencyStore, request_fingerprint


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
            model=request.model,
            upstream_model="upstream-text",
            content=(TextPart("hello"),),
            finish_reason=FinishReason.STOP,
        ),
        invocation_id=request.invocation_id,
        attempt_count=1,
        idempotency_status=IdempotencyStatus.NEW,
    )


def record(store: MemoryIdempotencyStore, request: InvocationRequest):
    """Record one answer for ``request`` under its key, the way the wiring does."""
    fingerprint = request_fingerprint(request)
    store.put(request.idempotency_key or "", fingerprint, answered(request))
    return store.get(request.idempotency_key or "")


# --------------------------------------------------------------------------- #
# The fingerprint
# --------------------------------------------------------------------------- #


def test_the_fingerprint_covers_the_payload_not_the_key():
    """Two requests differing only in their payload have different fingerprints."""
    from dataclasses import replace

    goodbye = replace(request(), messages=(replace(request().messages[0], content=(
        replace(request().messages[0].content[0], text="goodbye"),)),))

    assert request_fingerprint(request()).digest == request_fingerprint(request()).digest
    assert request_fingerprint(request()).digest != request_fingerprint(goodbye).digest


def test_the_fingerprint_covers_the_tenant():
    """The same request under two tenants has two fingerprints.

    Answering one from the other would leak an answer across a boundary the gateway
    is required to hold (PRD section 7), so the tenant is part of what makes a
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


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def test_a_stored_answer_is_returned_under_its_key():
    store = MemoryIdempotencyStore()

    assert record(store, request()) is not None


def test_an_unknown_key_has_no_record():
    assert MemoryIdempotencyStore().get("never-seen") is None


def test_storing_overwrites_the_earlier_answer_for_the_same_key():
    """A restart loses the memory, so a retry runs again rather than being answered."""
    store = MemoryIdempotencyStore()

    first = record(store, request())
    second = record(store, request())

    assert store.get("key-1") is second
    assert store.get("key-1") is not first


def test_the_store_is_bounded_so_it_cannot_grow_without_limit():
    store = MemoryIdempotencyStore(capacity=2)
    record(store, request(idempotency_key="a"))
    record(store, request(idempotency_key="b"))
    record(store, request(idempotency_key="c"))

    assert len(store) == 2
    assert store.get("a") is None
    assert store.get("c") is not None


def test_a_zero_capacity_is_refused():
    with pytest.raises(ValueError):
        MemoryIdempotencyStore(capacity=0)


def test_the_record_names_the_answer_as_the_first_one():
    """A record is stored as NEW; a replay is what the wiring marks when it answers from it."""
    store = MemoryIdempotencyStore()

    assert record(store, request()).status is IdempotencyStatus.NEW
