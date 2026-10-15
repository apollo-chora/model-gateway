"""Per-client rate limits and per-tenant concurrency limits (PRD section 14).

These tests prove the two decisions are taken per principal rather than per
deployment: one caller's burst cannot consume another's entitlement, a bucket
refills at the configured rate rather than at whatever a real clock happens to
give, and a per-principal concurrency bound refuses a full queue immediately
instead of holding the caller's payload resident.

Time is advanced by hand rather than slept through. A rate limiter that could only
be observed by waiting a real window would be tested by waiting, and a test that
sleeps is slow and flaky where one that advances a clock is exact.
"""

import asyncio

import pytest

from model_gateway.errors import DeadlineExceededError, ErrorReason, ResourceExhaustedError
from model_gateway.limits import RateLimiter, TenantConcurrencyLimiter


class _Clock:
    """A monotonic clock a test advances by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _settle(predicate, *, turns: int = 500) -> None:
    """Let the event loop run until ``predicate`` holds, or fail rather than hang."""
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


# --------------------------------------------------------------------------- #
# The request rate
# --------------------------------------------------------------------------- #


def test_a_fresh_principal_may_spend_its_whole_burst():
    """A deployment serves its first burst rather than refusing it."""
    limiter = RateLimiter(requests=3, window_seconds=60, clock=_Clock())

    for _ in range(3):
        limiter.check("cred-a")

    with pytest.raises(ResourceExhaustedError) as refused:
        limiter.check("cred-a")

    assert refused.value.reason is ErrorReason.RATE_LIMITED
    assert refused.value.http_status == 429


def test_the_bucket_refills_at_the_configured_rate():
    """A sixty-a-minute bucket grants one request per second, not one per window."""
    clock = _Clock()
    limiter = RateLimiter(requests=60, window_seconds=60, clock=clock)
    for _ in range(60):
        limiter.check("cred-a")

    clock.advance(1.0)
    limiter.check("cred-a")

    with pytest.raises(ResourceExhaustedError):
        limiter.check("cred-a")


def test_a_refill_never_exceeds_the_burst():
    """A bucket left idle is full, not overflowing: a caller cannot bank credit."""
    clock = _Clock()
    limiter = RateLimiter(requests=2, window_seconds=60, clock=clock)

    limiter.check("cred-a")
    clock.advance(3_600.0)
    limiter.check("cred-a")
    limiter.check("cred-a")

    with pytest.raises(ResourceExhaustedError):
        limiter.check("cred-a")


def test_one_principals_spending_does_not_touch_another():
    """The bucket is per caller, which is the whole point of a per-client limit."""
    limiter = RateLimiter(requests=1, window_seconds=60, clock=_Clock())

    limiter.check("cred-a")
    with pytest.raises(ResourceExhaustedError):
        limiter.check("cred-a")

    limiter.check("cred-b")


def test_a_refusal_names_neither_the_principal_nor_its_credential():
    """It is a fact about the caller's own rate, not a statement about who it is."""
    limiter = RateLimiter(requests=1, window_seconds=60, clock=_Clock())
    limiter.check("credential-digest-that-must-not-leak")

    with pytest.raises(ResourceExhaustedError) as refused:
        limiter.check("credential-digest-that-must-not-leak")

    assert "credential-digest-that-must-not-leak" not in refused.value.message
    assert refused.value.details == {"requests": "1", "window_seconds": "60.0"}


def test_a_configured_limit_reports_what_it_was_configured_with():
    limiter = RateLimiter(requests=30, window_seconds=60, clock=_Clock())

    assert limiter.requests == 30
    assert limiter.window_seconds == 60.0


@pytest.mark.parametrize(
    ("requests", "window"),
    [
        (0, 60.0),
        (-1, 60.0),
        (True, 60.0),
        (1.5, 60.0),
        ("3", 60.0),
        (1, 0.0),
        (1, -1.0),
        (1, "60"),
    ],
)
def test_a_limit_that_is_not_a_limit_is_refused(requests, window):
    """A bound that bounds nothing is a configuration mistake, not a limit."""
    with pytest.raises(ValueError):
        RateLimiter(requests=requests, window_seconds=window)


# --------------------------------------------------------------------------- #
# The per-principal concurrency bound
# --------------------------------------------------------------------------- #


async def test_one_principals_simultaneous_invocations_are_bounded():
    """The callers past the bound queue rather than running alongside the holders."""
    limiter = TenantConcurrencyLimiter(maximum=2, max_waiters=8)
    release = asyncio.Event()
    inside = 0
    peak = 0

    async def hold():
        nonlocal inside, peak
        async with limiter.slot("tenant-a"):
            inside += 1
            peak = max(peak, inside)
            await release.wait()
            inside -= 1

    tasks = [asyncio.create_task(hold()) for _ in range(5)]
    await _settle(lambda: inside == 2)
    assert peak == 2

    release.set()
    await asyncio.gather(*tasks)

    assert peak == 2


async def test_one_principals_bound_does_not_hold_another_back():
    """The bound is per tenant, so one tenant's saturation is not another's."""
    limiter = TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    release = asyncio.Event()
    held: set[str] = set()

    async def hold(principal):
        async with limiter.slot(principal):
            held.add(principal)
            await release.wait()
            held.discard(principal)

    first = asyncio.create_task(hold("tenant-a"))
    await _settle(lambda: held == {"tenant-a"})
    second = asyncio.create_task(hold("tenant-b"))

    await _settle(lambda: held == {"tenant-a", "tenant-b"})
    assert not first.done()
    assert not second.done()

    release.set()
    await asyncio.gather(first, second)


async def test_a_full_queue_is_refused_immediately():
    """Shedding load now is the answer the caller would have reached after a long wait."""
    limiter = TenantConcurrencyLimiter(maximum=1, max_waiters=0)
    release = asyncio.Event()
    held: set[str] = set()

    async def hold():
        async with limiter.slot("tenant-a"):
            held.add("a")
            await release.wait()

    task = asyncio.create_task(hold())
    await _settle(lambda: held == {"a"})

    with pytest.raises(ResourceExhaustedError) as refused:
        async with limiter.slot("tenant-a"):
            pass

    assert refused.value.reason is ErrorReason.CONCURRENCY_LIMIT_EXCEEDED

    release.set()
    await task


async def test_a_wait_for_a_slot_is_bounded_by_the_deadline_it_is_given():
    """An invocation that cannot be admitted in time fails rather than staying queued."""
    limiter = TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    release = asyncio.Event()
    held: set[str] = set()

    async def hold():
        async with limiter.slot("tenant-a"):
            held.add("a")
            await release.wait()

    task = asyncio.create_task(hold())
    await _settle(lambda: held == {"a"})

    with pytest.raises(DeadlineExceededError):
        async with limiter.slot("tenant-a", timeout=0.01):
            pass

    release.set()
    await task


async def test_a_slot_is_released_when_its_holder_is_cancelled():
    """A cancelled invocation frees its slot rather than leaving the principal stuck."""
    limiter = TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with limiter.slot("tenant-a"):
            entered.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with limiter.slot("tenant-a"):
        pass


async def test_a_limiter_is_built_only_for_a_principal_that_asks_for_one():
    """The map is bounded by the principals that actually call, not by the bound."""
    limiter = TenantConcurrencyLimiter(maximum=2)

    assert limiter.principals == ()
    assert limiter.maximum == 2

    async with limiter.slot("tenant-a"):
        assert limiter.principals == ("tenant-a",)


def test_a_concurrency_bound_that_is_not_a_bound_is_refused():
    for bad in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError):
            TenantConcurrencyLimiter(maximum=bad)
