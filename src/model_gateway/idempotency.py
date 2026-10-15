"""Idempotency: one key, one answer, no duplicate accounting (PRD section 12).

An invocation may carry an idempotency key. A repeat of the same request is
answered from the gateway's own record of the first answer rather than by asking
the provider again, which is what prevents duplicate accounting: the provider is
called once, and every equivalent repeat is served from the record.

Three rules shape the module, and each is a reason the obvious implementation is
wrong.

**The fingerprint covers the payload, not the key.** A key names a caller's intent
to deduplicate; it says nothing about what was asked. The same key sent with two
different bodies is two requests, not one, so the record stores a fingerprint of
the request and a repeat whose fingerprint differs is refused rather than
answered from the wrong answer. Otherwise a caller that reused a key by mistake
would silently receive an answer to a request it never made.

**The invocation identifier is not part of the fingerprint.** The gateway mints
its own identifier and discards a provider's, so a retry carrying the same key and
a freshly minted identifier is the same request. Fingerprinting the identifier
would make every retry a miss and defeat the mechanism.

**Only a completed answer is recorded.** A failure is not cached, because a retry
after a transient failure should be allowed to retry, and recording a failure as
if it were an answer would tell the caller the invocation succeeded. The
consequence is that the gateway claims no exactly-once upstream execution: a
provider without idempotency support, or a timeout that leaves the outcome
unknown, may produce more than one upstream call, and this module does not
pretend otherwise (PRD section 11.3).

The store is in memory, which is sufficient for a single-instance deployment as
PRD section 12 states. Durable persistence is a separate concern: it needs
configured storage, and an in-process map is not shared between instances.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from model_gateway.domain import IdempotencyStatus, InvocationRequest, InvocationResponse
from model_gateway.domain.serialization import as_int, as_required_text

__all__ = [
    "IdempotencyRecord",
    "IdempotencyStore",
    "MemoryIdempotencyStore",
    "RequestFingerprint",
    "request_fingerprint",
]

_DEFAULT_CAPACITY = 10_000


@dataclass(frozen=True)
class RequestFingerprint:
    """What makes two requests the same request, as a digest.

    The digest is a SHA-256 over the canonical request's semantic content. The
    fields it covers are listed at :func:`request_fingerprint`, which is where the
    deliberate exclusions are argued.
    """

    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", as_required_text("fingerprint 'digest'", self.digest))


def request_fingerprint(request: InvocationRequest) -> RequestFingerprint:
    """The fingerprint of one invocation request.

    Every field the answer depends on is fingerprinted: the model, the
    conversation, the instruction, the generation config, the tools and the tool
    choice, the modality, the permitted fallbacks and the tenant the request acts
    under. The tenant is included because the same key under two tenants is two
    requests, and answering one from the other would leak an answer across a
    boundary the gateway is required to hold (PRD section 7).

    The invocation identifier is excluded, for the reason the module docstring
    gives: the gateway mints it, so it is not a property of what the caller asked.
    ``metadata`` and ``idempotency_key`` are excluded too — bookkeeping never
    changes the answer, and the key names the deduplication rather than the request.
    """
    payload = "|".join(
        [
            request.model,
            repr(request.messages),
            repr(request.system_instruction),
            repr(request.generation_config),
            repr(request.tools),
            repr(request.tool_choice),
            repr(request.response_modality),
            repr(request.fallback_models),
            repr(request.tenant_id),
        ]
    )
    return RequestFingerprint(digest=sha256(payload.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class IdempotencyRecord:
    """One recorded answer, keyed by the key the caller chose and the request's shape.

    ``fingerprint`` is what a repeat is checked against, so a key reused for a
    different payload is refused rather than answered. ``status`` records whether
    the stored answer was the first one (``NEW``) or a replay of it (``REPLAYED``),
    which the caller can read to learn whether the provider was reached.
    """

    key: str
    fingerprint: RequestFingerprint
    response: InvocationResponse
    status: IdempotencyStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", as_required_text("idempotency record 'key'", self.key))


class IdempotencyStore(Protocol):
    """The gateway's record of the answers it has given, by key and request shape.

    The protocol is deliberately small: a repeat needs to find its answer, and a
    completed invocation needs to store one. Persistence is the implementation's
    business — PRD section 12 requires in-memory for a single instance, and a
    durable store needs configured storage this module does not presume.
    """

    def get(self, key: str) -> IdempotencyRecord | None:
        """The record for ``key``, or ``None`` when no answer was recorded under it.

        A record whose fingerprint does not match the request being made is not
        returned: the key was reused for a different payload, which is a caller
        error the gateway answers rather than papering over (PRD section 12).
        """
        ...

    def put(self, key: str, fingerprint: RequestFingerprint, response: InvocationResponse) -> None:
        """Record the answer to one request under ``key``.

        Storing overwrites any earlier record for the same key and fingerprint, so
        a deployment that restarts loses its memory and a retry then runs again
        rather than being answered from a record that no longer exists.
        """
        ...


class MemoryIdempotencyStore:
    """An in-memory store, sufficient for a single-instance deployment.

    Bounded so a deployment that never evicts cannot grow without limit: the
    oldest record is dropped once the store is full. Eviction is by insertion
    order rather than by last use, because an LRU costs a timestamp write on every
    read and the bound's purpose is to stop unbounded growth, not to keep the
    hottest keys — a deployment that needs that should configure durable storage.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = as_int("idempotency store 'capacity'", capacity, minimum=1)
        self._records: dict[str, IdempotencyRecord] = {}

    def get(self, key: str) -> IdempotencyRecord | None:
        """The record for ``key``, or ``None`` when none was recorded."""
        return self._records.get(key)

    def put(self, key: str, fingerprint: RequestFingerprint, response: InvocationResponse) -> None:
        """Record the answer, evicting the oldest record if the store is full."""
        self._records[key] = IdempotencyRecord(
            key=key,
            fingerprint=fingerprint,
            response=response,
            status=IdempotencyStatus.NEW,
        )
        while len(self._records) > self._capacity:
            self._records.pop(next(iter(self._records)))

    def __len__(self) -> int:
        """How many answers the store holds, for tests and operators."""
        return len(self._records)

    def __repr__(self) -> str:
        return f"MemoryIdempotencyStore(records={len(self._records)}, capacity={self._capacity})"
