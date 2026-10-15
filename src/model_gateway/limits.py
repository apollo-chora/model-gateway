"""Per-client request-rate limits and per-tenant concurrency limits.

Two admission decisions, both taken before an invocation reaches any provider and
both keyed on the trusted principal rather than on anything a request carries.
Each limiter takes an opaque principal string, so what the key *is* stays a
decision of the layer that knows the caller — the transport, which holds the
authenticated identity — rather than one this module guesses at.

**A request rate is a property of the caller.** One credential's burst must not
consume the capacity the others are entitled to, so the bucket is per principal.
The principal the gateway already keys policy on is the credential identifier —
the namespace idempotency uses — and never a
caller-supplied field: a limit keyed on a request value is evaded by changing that
value, and a caller-supplied value is not a principal at all.

**A concurrency bound is a property of the tenant.** One tenant's burst must not
occupy the gateway's whole upstream pool and starve the others, which a single
global bound cannot prevent because it does not know who is asking. This bound is
therefore separate from
:class:`~model_gateway.reliability.ConcurrencyLimiter`, which bounds upstream
attempts across *all* callers: that limiter protects the gateway and the
providers, and this one shares the pool fairly between tenants. Both are enforced
and neither replaces the other, because they answer different questions — how much
upstream work may be in flight, and whose work it is.

**A refusal is taken before the walk, not inside it.** The rate limit is checked
at the invocation's entry, before a route is selected and before any provider is
called. That is not merely an optimisation. The walk treats a ``RATE_LIMITED``
failure as route-local and therefore fallback-eligible, because
an *upstream's* rate limit belongs to the route that answered it — another route
is not necessarily limited. The gateway's own limit is a property of the caller
and of no route, so raising it inside the walk would make the gateway walk a
request from one provider to the next in an attempt to escape a limit that follows
the caller, spending every configured provider's capacity on the way.

**Both limiters take an opaque principal, and neither bounds how many it may see.**
That separation is deliberate: what the principal *is* belongs to the layer that
holds the authenticated identity, and this module has no access to the credential
set and must not invent one. The consequence is that the cardinality of these maps
is a property of the *call sites* rather than of the limiters — a deployment whose
transports supply the trusted principal sees a set bounded by the credentials it
configured, while an in-process caller that passed a fresh string on every call
would grow both maps without limit. The service's ``namespace`` and ``tenant``
arguments carry that contract and state it; this module does not enforce it.

Eviction is not the answer to that, and it is worth saying why: dropping a bucket
resets the rate history it holds, and dropping a tenant's limiter splits that
tenant's concurrency bound between two limiters, so a deployment with unbounded
principals would get limits that are both unbounded in memory and wrong in effect.
A deployment that needs a bound on principals needs a bound on its callers, which
is what authentication is for.

**Both limits are configured into existence rather than always on.** A deployment
that configures no request rate and no per-tenant bound is one that has not said
what its callers are entitled to, and inventing a bound would refuse requests no
configuration said to refuse — the same reasoning that makes an absent optional
capability disabled rather than fatal.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite

from model_gateway.errors import ErrorReason, ResourceExhaustedError
from model_gateway.reliability import ConcurrencyLimiter

__all__ = ["RateLimiter", "TenantConcurrencyLimiter"]


@dataclass
class _Bucket:
    """One principal's token bucket: what is left, and when it was last refilled."""

    tokens: float
    updated: float


class RateLimiter:
    """A per-principal token bucket: a burst, then a sustained rate.

    The bucket holds ``requests`` tokens and refills at ``requests`` per
    ``window_seconds``, so a caller may burst up to the configured number and is
    held to the configured rate over time. A bucket rather than a fixed window,
    because a fixed window lets a caller spend a full window's requests at the end
    of one window and another full window's at the start of the next — twice the
    configured rate across the boundary, which is the one moment a rate limit
    exists to smooth.

    The clock is injectable so a test can advance time rather than sleep, which is
    what makes a refill testable at all: a limiter observable only by waiting a
    real window would be tested by waiting, and a test that sleeps is slow and
    flaky where one that advances a clock is exact.
    """

    def __init__(
        self,
        *,
        requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
            raise ValueError(f"requests must be a positive integer, got {requests!r}")
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, int | float):
            raise ValueError(f"window_seconds must be a number, got {window_seconds!r}")
        window = float(window_seconds)
        # Stated as the good case rather than the negation of a bad one, because
        # every comparison with NaN is false: ``not window > 0`` is False for NaN,
        # so a NaN window would pass a check written that way and make every
        # bucket look empty forever.
        if not isfinite(window) or window <= 0:
            raise ValueError(
                f"window_seconds must be a finite positive number, got {window_seconds!r}"
            )
        refill = requests / window
        # A window small enough to overflow the division is not a rate: it would
        # refill a bucket to full after any positive elapsed interval, which is
        # the one value that turns a limit into no limit at all.
        if not isfinite(refill) or refill <= 0:
            raise ValueError(
                f"{requests} requests per {window_seconds!r} seconds is not a finite "
                f"positive refill rate, got {refill!r}"
            )
        self._requests = requests
        self._window_seconds = window
        self._refill_per_second = refill
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    @property
    def requests(self) -> int:
        """The burst a principal may spend before the rate holds it to the window."""
        return self._requests

    @property
    def window_seconds(self) -> float:
        """The window the rate is quoted over."""
        return self._window_seconds

    def check(self, principal: str) -> None:
        """Spend one token for ``principal``, or refuse the request.

        A principal that has never been seen starts with a full bucket, so a fresh
        deployment serves its first burst rather than refusing it.

        Raises :class:`~model_gateway.errors.ResourceExhaustedError` with reason
        ``RATE_LIMITED`` when the bucket is empty. The refusal names neither the
        principal nor any credential material: it is a fact about the caller's own
        rate, and the caller already knows which credential it presented.
        """
        now = self._clock()
        bucket = self._buckets.get(principal)
        if bucket is None:
            bucket = _Bucket(tokens=float(self._requests), updated=now)
            self._buckets[principal] = bucket
        else:
            elapsed = now - bucket.updated
            if elapsed > 0:
                bucket.tokens = min(
                    float(self._requests), bucket.tokens + elapsed * self._refill_per_second
                )
                bucket.updated = now
        if bucket.tokens < 1.0:
            raise ResourceExhaustedError(
                "the caller has reached its request rate limit; the limit refills "
                "continuously, so a request made after a short wait will be served",
                reason=ErrorReason.RATE_LIMITED,
                details={
                    "requests": str(self._requests),
                    "window_seconds": str(self._window_seconds),
                },
            )
        bucket.tokens -= 1.0


class TenantConcurrencyLimiter:
    """Bounds the simultaneous invocations one principal may have in flight.

    One :class:`~model_gateway.reliability.ConcurrencyLimiter` per principal, so a
    tenant's burst is bounded by its own limit rather than by how many others
    happen to be calling. The per-principal limiters are built on first use and
    kept, and how many there can be is the caller's business rather than this
    class's: it accepts any principal string and neither knows nor can check which
    ones a deployment is entitled to see (see the module docstring).

    What the key should be belongs to the caller, and the caller is expected to key
    on the tenant the credential is bound to. A credential bound to no tenant is
    its own principal, so keying such a caller on its credential
    identifier rather than on a shared empty tenant is what keeps one tenant-less
    caller from consuming the bound the others are entitled to.
    """

    def __init__(self, *, maximum: int, max_waiters: int | None = None) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError(f"maximum must be a positive integer, got {maximum!r}")
        self._maximum = maximum
        self._max_waiters = max_waiters
        self._limiters: dict[str, ConcurrencyLimiter] = {}

    @property
    def maximum(self) -> int:
        """The bound on simultaneous invocations for any one principal."""
        return self._maximum

    @property
    def principals(self) -> tuple[str, ...]:
        """The principals a limiter has been built for, for tests and operators."""
        return tuple(self._limiters)

    @asynccontextmanager
    async def slot(self, principal: str, timeout: float | None = None) -> AsyncIterator[None]:
        """Hold one of ``principal``'s slots until the wrapped body exits.

        The wait is bounded by ``timeout`` — the invocation's remaining deadline —
        and a caller that arrives when the principal's queue is full is refused
        immediately with reason ``CONCURRENCY_LIMIT_EXCEEDED``. Both behaviours are
        the shared limiter's, because the two bounds differ only in what they are
        keyed on: a per-tenant bound that queued without limit would hold one
        tenant's payloads resident exactly as an unbounded global queue would.
        """
        async with self._limiter_for(principal).slot(timeout):
            yield

    def _limiter_for(self, principal: str) -> ConcurrencyLimiter:
        limiter = self._limiters.get(principal)
        if limiter is None:
            limiter = ConcurrencyLimiter(
                self._maximum, max_waiters=self._max_waiters, subject="tenant"
            )
            self._limiters[principal] = limiter
        return limiter
