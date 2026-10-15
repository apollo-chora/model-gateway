"""Idempotency: one key, one answer, no duplicate accounting (PRD section 12).

An invocation may carry an idempotency key. A repeat of the same request is
answered from the gateway's own record of the first answer rather than by asking
the provider again, which is what prevents duplicate accounting: the provider is
called once, and every equivalent repeat is served from the record.

Four rules shape the module, and each is a reason the obvious implementation is
wrong.

**The key is namespaced by the trusted principal, not used raw.** The store is keyed
by a namespace derived from the credential — its identifier and its tenant — beside
the caller's key. Two tenants using the same key are two independent operations,
and letting one overwrite the other's record is a cross-tenant interference
channel: the second silently destroys the first's replay. A tenant is a property of
the credential, never of the request (PRD section 5.1), so the namespace comes from
the transport's identity rather than from anything the caller states.

**A key names one operation, so reuse is a conflict, not a re-run.** Once a key has
a record, a repeat carrying a different fingerprint is refused with an idempotency
conflict rather than executed as a new request. The alternative — execute and
overwrite — lets one key oscillate between unrelated payloads indefinitely, and
hides a caller's mistake of reusing a key it believed was unique.

**Concurrent identical requests are coalesced, not just sequential repeats.** The
first caller claims the key and every later caller with the same fingerprint waits
on that claim's outcome instead of starting a second invocation. Without this, two
simultaneous retries — the situation idempotency exists to handle — both call the
provider. Cancellation belongs to the caller who owns it: a waiter going away does
not cancel the invocation it is waiting on.

**Only a completed answer is recorded.** A failure is not cached, so a retry after a
transient failure is allowed to retry. A waiter coalesced onto a failed invocation
receives that invocation's failure rather than starting another provider attempt
inside the same request. The gateway therefore claims no exactly-once upstream
execution (PRD section 11.3).

The fingerprint is taken over a canonical JSON encoding of the request's semantic
content, never over ``repr()``: a dataclass's repr is an implementation
representation that changes when the class does, which would invalidate every
stored fingerprint and couple the contract to Python's formatting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol

import orjson

from model_gateway.domain import (
    IdempotencyStatus,
    InvocationRequest,
    InvocationResponse,
)
from model_gateway.domain.serialization import as_int, as_required_text
from model_gateway.errors import IdempotencyConflictError

__all__ = [
    "IdempotencyRecord",
    "IdempotencyStore",
    "MemoryIdempotencyStore",
    "RequestFingerprint",
    "fingerprint_payload",
    "request_fingerprint",
    "settle",
]

_DEFAULT_CAPACITY = 10_000

_ClaimKind = Literal["completed", "owner", "waiter"]


@dataclass(frozen=True)
class RequestFingerprint:
    """What makes two requests the same request, as a digest.

    The digest is a SHA-256 over the canonical JSON of the request's semantic
    content. Which fields participate and which are deliberately excluded is stated
    at :func:`request_fingerprint`.
    """

    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", as_required_text("fingerprint 'digest'", self.digest))


def fingerprint_payload(request: InvocationRequest) -> dict[str, Any]:
    """The request's semantic content as the JSON the fingerprint is taken over.

    Exposed so a test can assert exactly which fields participate without parsing a
    hash, and so a durable store built later takes its fingerprint from the same
    bytes rather than re-deriving them.
    """
    return {
        "model": request.model,
        "messages": [message.to_dict() for message in request.messages],
        "system_instruction": request.system_instruction,
        "generation_config": request.generation_config.to_dict(),
        "tools": [tool.to_dict() for tool in request.tools],
        "tool_choice": request.tool_choice.to_dict() if request.tool_choice is not None else None,
        "response_modality": request.response_modality.value,
        "fallback_models": list(request.fallback_models),
        "tenant": request.tenant_id,
    }


def request_fingerprint(request: InvocationRequest) -> RequestFingerprint:
    """The fingerprint of one invocation request.

    Every field the answer depends on is fingerprinted: the model, the
    conversation, the instruction, the generation config, the tools and the tool
    choice, the modality, the permitted fallbacks and the tenant the request acts
    under. The tenant is included because the same key under two tenants is two
    requests, and answering one from the other would leak an answer across a
    boundary the gateway is required to hold (PRD section 7).

    Three fields are deliberately excluded, each for a stated reason:

    * ``invocation_id`` — the gateway mints its own identifier and discards a
      provider's, so it is not a property of what the caller asked. Fingerprinting
      it would make every retry a miss and defeat the mechanism.
    * ``idempotency_key`` — the key names the deduplication, not the request, so it
      cannot be part of what makes the request itself.
    * ``metadata`` — bookkeeping that never reaches the model or a provider, so it
      cannot change the answer. This holds while ``FORWARDABLE_METADATA_KEYS``
      stays empty; a metadata key that becomes semantically consumed changes this
      fingerprint with that contract, not before it.
    """
    digest = sha256(
        orjson.dumps(fingerprint_payload(request), option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    return RequestFingerprint(digest=digest)


@dataclass(frozen=True)
class IdempotencyRecord:
    """One recorded answer, keyed by the key the caller chose and the request's shape.

    ``status`` records whether the stored answer was the first one (``NEW``) or a
    replay of it (``REPLAYED``), which the caller can read to learn whether the
    provider was reached.
    """

    key: str
    fingerprint: RequestFingerprint
    response: InvocationResponse
    status: IdempotencyStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", as_required_text("idempotency record 'key'", self.key))


@dataclass
class _Entry:
    """One namespaced key's state: a claim while running, a record once done.

    ``future`` is set while the invocation is in flight and is resolved with its
    outcome; ``record`` is set once that outcome was a completed answer.
    """

    fingerprint: str
    future: asyncio.Future[Any] | None
    record: IdempotencyRecord | None = None


@dataclass(frozen=True)
class _Claim:
    """What an atomic claim found: a finished answer, a claim to own, or a wait.

    Exactly one of ``record`` and ``future`` is set, according to ``kind``.
    """

    kind: _ClaimKind
    record: IdempotencyRecord | None = None
    future: asyncio.Future[Any] | None = None


class IdempotencyStore(Protocol):
    """The gateway's record of the answers it has given, by namespace and key.

    ``acquire`` is the whole of the coordination and it is atomic: it performs no
    ``await`` between reading the store and writing the claim, so within one event
    loop no other caller can observe the key between those two steps. That is what
    makes two simultaneous identical requests coalesce instead of both executing.
    """

    async def acquire(
        self, key: str, fingerprint: RequestFingerprint, namespace: str
    ) -> _Claim:
        """Claim ``key`` in ``namespace``, or report what is already there.

        Returns ``completed`` with the stored answer when the key has finished and
        the fingerprint matches; ``waiter`` with the in-flight future when another
        caller owns a claim for the same fingerprint; and ``owner`` with a fresh
        future when this caller has claimed it.

        Raises :class:`~model_gateway.errors.IdempotencyConflictError` when the key
        exists with a *different* fingerprint: the caller reused a key it believed
        was unique, which is refused rather than executed or overwritten.
        """
        ...

    async def complete(self, key: str, namespace: str, response: InvocationResponse) -> None:
        """Record the answer to the invocation this caller owns.

        Replaces the claim with the answer, resolves every waiter's future with it,
        and sets the record's status to ``NEW``. A no-op when there is no claim to
        complete — a key completed twice, or one whose claim was already released.
        """
        ...

    async def fail(self, key: str, namespace: str, error: BaseException) -> None:
        """Release the claim, giving every waiter the failure it was coalesced onto.

        The claim is removed so a later caller may retry; waiters that were already
        waiting receive ``error`` rather than starting a second provider attempt of
        their own.
        """
        ...


class MemoryIdempotencyStore:
    """An in-memory store, sufficient for a single-instance deployment.

    Bounded so a deployment that never evicts cannot grow without limit: once the
    store is full, the oldest *completed* record is dropped. An in-flight claim is
    never evicted, because evicting it would strand its waiters on a future nobody
    resolves.

    The bound weakens the replay guarantee in a stated way: a repeat is deduplicated
    only while its record remains available in this instance. A deployment that needs
    the guarantee across restarts or instances configures durable storage, which is a
    separate concern requiring configured persistence (PRD section 12).
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = as_int("idempotency store 'capacity'", capacity, minimum=1)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._order: list[tuple[str, str]] = []

    async def acquire(
        self, key: str, fingerprint: RequestFingerprint, namespace: str
    ) -> _Claim:
        """Atomically claim the key, or report what is already there.

        No ``await`` appears between the read and the write, which is the whole of
        the atomicity: a second caller arriving while the first is still deciding
        sees the claim already made.
        """
        namespaced = (namespace, key)
        entry = self._entries.get(namespaced)
        if entry is not None and entry.fingerprint != fingerprint.digest:
            raise IdempotencyConflictError(
                f"the idempotency key {key!r} was already used for a different request; a key"
                "names one operation, not two",
                details={
                    "idempotency_key": key,
                    "namespace": namespace,
                },
            )
        if entry is not None and entry.record is not None:
            return _Claim("completed", record=entry.record)
        if entry is not None:
            return _Claim("waiter", future=entry.future)

        future: asyncio.Future[Any] = asyncio.Future()
        self._entries[namespaced] = _Entry(fingerprint=fingerprint.digest, future=future)
        self._order.append(namespaced)
        self._evict()
        return _Claim("owner", future=future)

    async def complete(
        self, key: str, namespace: str, response: InvocationResponse
    ) -> None:
        """Record the answer and resolve every waiter's future with it."""
        entry = self._entries.get((namespace, key))
        if entry is None or entry.future is None:
            return
        future = entry.future
        entry.record = IdempotencyRecord(
            key=key,
            fingerprint=RequestFingerprint(digest=entry.fingerprint),
            response=response,
            status=IdempotencyStatus.NEW,
        )
        entry.future = None
        _resolve_future(future, response)

    async def fail(self, key: str, namespace: str, error: BaseException) -> None:
        """Release the claim so a later caller may retry, and tell the waiters why."""
        entry = self._entries.pop((namespace, key), None)
        if entry is None:
            return
        if entry.future is not None:
            _resolve_future(entry.future, error)

    def _evict(self) -> None:
        """Drop the oldest completed record once the store is over its capacity."""
        while len(self._entries) > self._capacity:
            victim = next(
                (name for name in self._order if self._entries[name].record is not None),
                None,
            )
            if victim is None:
                return
            self._order.remove(victim)
            del self._entries[victim]

    def __len__(self) -> int:
        """How many claims and records the store holds, for tests and operators."""
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MemoryIdempotencyStore(entries={len(self._entries)}, capacity={self._capacity})"


def _resolve_future(future: asyncio.Future[Any], outcome: Any) -> None:
    """Resolve a claim's future with its outcome, tolerating one already resolved.

    A waiter may have been cancelled while waiting, which cancels the future it is
    holding; the owner then has nothing to resolve, and that is a normal outcome
    rather than an error. The future is resolved by the owner alone, never by a
    waiter, so a waiter going away can never cancel the invocation.
    """
    if future.done():
        return
    if isinstance(outcome, Exception):
        future.set_exception(outcome)
        return
    future.set_result(outcome)


async def settle(future: asyncio.Future[Any]) -> Any:
    """The outcome of a claim a caller is waiting on: the answer, or the failure.

    A cancellation of the caller's own request propagates rather than being reported
    as the invocation's failure: the caller went away, and that is not a fact about
    the invocation.
    """
    return await future
