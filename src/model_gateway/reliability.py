"""Reliability primitives: the policy the gateway's retry, fallback and
concurrency loop is built from (PRD sections 11.3 and 15).

This module is pure policy. It imports the gateway's own error taxonomy, the
standard library's clock and its random source, and nothing else: no HTTP, no
inference library, no I/O. The one nondeterminism it introduces — the jitter on
the retry schedule — is injectable, so the production default is the only place
randomness enters and a caller that needs a reproducible schedule supplies its
own source. The orchestration that uses these primitives is a separate slice
and codes against the names this module exports.

* :class:`Deadline` is the single total invocation budget. Provider timeouts,
  connect timeouts, retries and fallbacks all draw from the one remaining
  budget it reports, and no attempt restarts it.
* :func:`is_retryable` classifies a failure from the gateway's own categories
  *and reasons*, never from a library exception type, and answers whether
  another attempt **on the same route** is worth making.
* :func:`is_fallback_eligible` asks the other half of that question — whether
  another **route** is worth trying — because the two answers differ for the
  same failure.
* :func:`backoff_seconds` computes the delay before the next attempt: the
  exponential schedule the configured base defines, jittered so a shared
  outage does not synchronise every caller's retries, and capped so no single
  wait can dominate an invocation budget.
* :class:`ConcurrencyLimiter` bounds both the number of simultaneous upstream
  attempts and the number of callers queued for one, so an overload is shed
  instead of buffered in process memory.

PRD section 15 gives the gateway one deadline per invocation and requires that
a retry must not exceed the caller's remaining deadline. These primitives are
how that is enforced: time is subtracted, never reset, and a wait that would
consume the last of the budget is refused rather than slept.
"""

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from model_gateway.errors import (
    DeadlineExceededError,
    ErrorReason,
    ResourceExhaustedError,
    UnavailableError,
)

__all__ = [
    "BACKOFF_CEILING_SECONDS",
    "BACKOFF_JITTER_FRACTION",
    "WAITER_QUEUE_MULTIPLE",
    "ConcurrencyLimiter",
    "Deadline",
    "backoff_seconds",
    "is_fallback_eligible",
    "is_retryable",
]

# The ceiling on one backoff delay, in seconds. Without a ceiling a long retry
# schedule would outlast any invocation budget, and with the deployment's
# default 110 second provider timeout a 30 second delay still leaves room for
# the attempt itself. The schedule is exponential in the attempt count, so the
# ceiling is what keeps "retries must be bounded" (PRD 11.2) true for every
# retry count a deployment configures. It caps the jittered delay, so it is a
# bound on the wait itself and not merely on the schedule underneath it.
BACKOFF_CEILING_SECONDS = 30.0

# How much randomness one retry wait carries, as a fraction of the exponential
# delay it is drawn from. The jittered delay lies in ``[delay, delay * (1 +
# BACKOFF_JITTER_FRACTION))``, so 1.0 lets a wait run up to twice the schedule's
# value. The default is large enough to break up the waves a shared provider
# outage produces — callers that failed together wake spread across a whole
# doubling of the delay — while the delay itself stays the floor, so jitter can
# never make a retry fire earlier than the configured backoff intends.
BACKOFF_JITTER_FRACTION = 1.0

# How many waiters the concurrency limiter queues per slot, for its default
# bound. A limiter with ``maximum`` slots therefore holds at most
# ``maximum * (1 + WAITER_QUEUE_MULTIPLE)`` callers resident — the cohort in
# flight plus the queue — which absorbs a burst a few times the pool's width
# and sheds beyond it. See :class:`ConcurrencyLimiter` for why an unbounded
# queue is not an option.
WAITER_QUEUE_MULTIPLE = 4


def _usable_seconds(value: object) -> bool:
    """Whether ``value`` is a number of seconds a budget could be built from.

    ``None`` is the gateway's own "no budget" marker and is handled by the
    callers; this check is for numbers. It mirrors the rule
    :class:`~model_gateway.providers.base.AttemptContext` applies to the
    per-attempt timeout it feeds: a number must be finite and positive. A
    budget of zero seconds is not a budget, it is a refusal in disguise.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return isfinite(value) and value > 0


@dataclass(frozen=True)
class Deadline:
    """One invocation budget covering every attempt, retry and fallback (PRD 15).

    The gateway enforces a single total deadline per invocation, and this
    object is that deadline as a value: provider timeouts, connect timeouts,
    retries and fallbacks all draw from the one remaining budget reported
    here, and no attempt restarts it. An invocation constructs one
    :class:`Deadline` and threads that same instance through every attempt,
    retry and fallback, so a retry consumes the same budget rather than
    restarting it: the clock is anchored when the invocation begins, and each
    :meth:`Deadline.remaining` call subtracts the time passed since that
    anchor — including the time earlier attempts, retries and fallbacks of
    the same invocation already spent.

    ``seconds`` is the total budget the invocation starts with and
    ``started_at`` is the monotonic instant it was anchored at. ``None`` for
    both means the gateway set no deadline: an unbounded deadline, which is
    not the same as a budget of zero. The two are always given together — a
    bounded deadline needs an anchor — and :meth:`Deadline.after` is the
    constructor callers use.

    Remaining time is never negative: a deadline that has fully elapsed
    reports zero. :meth:`Deadline.budget_for` therefore hands an expired
    deadline's caller an attempt budget of zero — an attempt may be refused,
    but it is never given a negative number of seconds.
    """

    seconds: float | None
    started_at: float | None

    def __post_init__(self) -> None:
        if (self.seconds is None) != (self.started_at is None):
            raise ValueError(
                "seconds and started_at must both be set or both be None: a bounded "
                f"deadline needs an anchor, got seconds={self.seconds!r} and "
                f"started_at={self.started_at!r}"
            )
        if self.seconds is not None and not _usable_seconds(self.seconds):
            raise ValueError(
                f"seconds must be a positive number of seconds or None, got {self.seconds!r}"
            )

    @classmethod
    def after(cls, seconds: float | None) -> "Deadline":
        """Anchor a new deadline ``seconds`` from now, or an unbounded one.

        ``None`` means the gateway set no deadline for the invocation and
        produces a deadline that never expires and has no budget to hand an
        attempt. A number must be finite and positive; anything else raises
        :exc:`ValueError` here rather than surfacing later as a negative
        budget.
        """
        return cls(seconds=seconds, started_at=monotonic() if seconds is not None else None)

    @property
    def expired(self) -> bool:
        """Whether the budget is fully elapsed; never true when unbounded."""
        left = self.remaining()
        return left is not None and left <= 0.0

    def remaining(self, *, now: Callable[[], float] | None = None) -> float | None:
        """Seconds left in the budget, or ``None`` when unbounded.

        The result is clamped at zero: a deadline that has overrun reports no
        time left rather than a negative amount, so every caller that
        subtracts from it stays honest. ``now`` is injectable for tests and
        callers computing against their own clock; it defaults to the
        monotonic clock the anchor was taken on.
        """
        if self.seconds is None or self.started_at is None:
            return None
        clock = now if now is not None else monotonic
        left = self.seconds - (clock() - self.started_at)
        return max(left, 0.0)

    def budget_for(
        self,
        configured: float | None,
        *,
        now: Callable[[], float] | None = None,
    ) -> float | None:
        """The budget one attempt may use: the smaller of two bounds.

        An attempt is bounded twice over: by ``configured``, the provider
        timeout the deployment configured, and by the invocation deadline
        itself. The attempt may use the smaller of the two, and that is what
        is returned:

        * neither bound set — ``None``: the gateway set no deadline and no
          provider timeout, so the attempt is unbounded. ``None`` is the
          gateway's own "no budget" marker; it is not a budget of zero.
        * only the configured timeout — that timeout.
        * only the deadline — what remains of it, which is zero once the
          deadline has expired.
        * both — the smaller of the two.

        An expired deadline therefore yields a budget of zero, never a
        negative one: an attempt may be refused, but it is never handed time
        it cannot use. Because the deadline is one clock for the whole
        invocation, the budget this returns for a retry is what is left
        after the attempts before it — the retry draws from the same budget,
        it does not restart it.

        ``configured`` must be a finite positive number of seconds or
        ``None``, matching the validation
        :class:`~model_gateway.providers.base.AttemptContext` applies to the
        timeout it is handed on.
        """
        if configured is not None and not _usable_seconds(configured):
            raise ValueError(
                f"configured must be a positive number of seconds or None, got {configured!r}"
            )
        left = self.remaining(now=now)
        if configured is None:
            return left
        if left is None:
            return configured
        return min(configured, left)


def is_retryable(error: BaseException, *, remaining: float | None = None) -> bool:
    """Whether another attempt **on the same route** is worth making.

    The answer comes from the gateway's own taxonomy — the category the failure
    was classified into, and the reason that category permits — and from nothing
    else. A library exception type is never inspected: a failure that has not
    been classified into the taxonomy has no gateway meaning yet, so it is not
    retryable. Callers pass such a failure through
    :func:`~model_gateway.errors.normalise_exception` first, which lands it in
    :class:`~model_gateway.errors.UnavailableError` or
    :class:`~model_gateway.errors.InternalError`, both decided below.

    This is the same-route question only. Whether another *route* is worth
    trying is :func:`is_fallback_eligible`'s, and the two answers differ: an
    upstream rate limit is worth another route but not another attempt at the
    route that just refused the request.

    Retryable, because another attempt at the same route can still succeed:

    * :class:`~model_gateway.errors.UnavailableError` — no provider could serve
      the request (``PROVIDER_UNAVAILABLE``). A provider being down is
      transient: the retry may reach it after it recovers, and PRD section 11.2
      names connection failures a retry trigger.
    * :class:`~model_gateway.errors.DeadlineExceededError` — a deadline expired,
      retryable only while the invocation deadline still has budget left.
      ``remaining`` is the seconds left in the invocation's :class:`Deadline`,
      passed as ``deadline.remaining()``: a provider or attempt timeout does not
      by itself exhaust the invocation budget, so while time remains another
      attempt may fit inside it. Once the budget is gone (``remaining`` is zero)
      there is nothing left to fit an attempt into. ``remaining=None`` means the
      gateway set no invocation deadline at all, so a provider timeout consumes
      no gateway budget and stays retryable while the retry count allows — PRD
      section 11.2 lists timeouts a retry trigger, and with no deadline the
      retry count is the only bound.

    Not retryable on the same route, even though the category is transient:

    * :class:`~model_gateway.errors.ResourceExhaustedError` — here the *reason*
      decides, and neither reason is a same-route retry trigger.
      ``CONCURRENCY_LIMIT_EXCEEDED`` is the gateway's own
      :class:`ConcurrencyLimiter` refusing work, and calling a model route again
      does nothing to relieve the gateway's own overload.
      ``RATE_LIMITED`` is an upstream refusing the request rate, and retrying it
      on the same route without knowing how long to wait is a blind 429 retry
      that adds load to the provider that just refused it; the taxonomy carries
      no ``Retry-After`` field to wait the right amount, so until one exists the
      rate limit is not retried into. It is not discarded either: it stays
      :func:`is_fallback_eligible`, because the limit belongs to the route that
      answered and another route may not be limited.

    Not retryable, because repeating the call cannot change the answer:
    invalid arguments, authentication and permission failures, a missing
    resource or model, a capability the route does not advertise, a failed
    precondition, and internal errors. The first group describes the request
    or the caller and would repeat identically; a capability failure is a
    property of the route that was chosen; and an internal error is a gateway
    defect, which PRD section 15 classes as a deterministic error category —
    deterministic failures are reported, not retried into.
    """
    if isinstance(error, DeadlineExceededError):
        return remaining is None or remaining > 0
    if isinstance(error, ResourceExhaustedError):
        return False
    return isinstance(error, UnavailableError)


def is_fallback_eligible(error: BaseException, *, remaining: float | None = None) -> bool:
    """Whether another **route** is worth trying after ``error``.

    The companion of :func:`is_retryable`, which answers the same-route
    question. The two are deliberately separate, because a failure can be worth
    one and not the other: an upstream rate limit is worth another route but not
    another attempt at the route that just refused the request.

    Eligible, because another route is a different provider call that may
    succeed:

    * :class:`~model_gateway.errors.UnavailableError` — the route could not
      serve the request at all, so a different route may.
    * :class:`~model_gateway.errors.ResourceExhaustedError` with reason
      ``RATE_LIMITED`` — a rate limit is a property of the route that answered
      it, and another route is not necessarily limited. The same category with
      reason ``CONCURRENCY_LIMIT_EXCEEDED`` is *not* eligible: that rejection
      came from the gateway's own :class:`ConcurrencyLimiter`, which every route
      shares, so another route would be refused in exactly the same way and
      walking to it only spends invocation budget.
    * :class:`~model_gateway.errors.DeadlineExceededError` — while the
      invocation deadline still has budget left (``remaining`` as in
      :func:`is_retryable`), another route may fit inside it.

    Everything else is not eligible: the failure describes the request or the
    caller and would repeat on any route, or it is a gateway defect.

    The invocation's fallback walk is its intended caller: this is the test that
    separates a route worth walking to from one that is not.
    """
    if isinstance(error, DeadlineExceededError):
        return remaining is None or remaining > 0
    if isinstance(error, UnavailableError):
        return True
    if isinstance(error, ResourceExhaustedError):
        return error.reason is ErrorReason.RATE_LIMITED
    return False


def _jitter_draw(source: Callable[[], float]) -> float:
    """One draw from the injected random source, checked against its contract.

    The source stands for a probability, so a draw outside ``[0.0, 1.0]`` is a
    programming error rather than a schedule: it would produce a wait outside
    the range :func:`backoff_seconds` documents. It is refused here, while the
    source that broke the contract is still in hand.
    """
    draw = source()
    if (
        isinstance(draw, bool)
        or not isinstance(draw, int | float)
        or not isfinite(draw)
        or not 0.0 <= draw <= 1.0
    ):
        raise ValueError(f"the random source must return a number in [0.0, 1.0], got {draw!r}")
    return float(draw)


def backoff_seconds(
    attempt: int,
    base_ms: float,
    *,
    remaining: float | None = None,
    random_source: Callable[[], float] | None = None,
) -> float:
    """The delay in seconds before the next attempt: exponential, jittered, capped.

    ``attempt`` is the 1-based number of the attempt that just failed, and the
    schedule grows exponentially from ``base_ms`` — the configured
    ``PROVIDER_RETRY_BACKOFF_MS``: one base after the first attempt, two after
    the second, four after the third, ``base_ms * 2 ** (attempt - 1)``
    milliseconds in general.

    The growth is bounded by :data:`BACKOFF_CEILING_SECONDS`, a fixed ceiling of
    30 seconds, so no single wait can dominate an invocation budget.

    Jitter. The delay is the exponential value extended by a uniformly random
    fraction of itself, so it lies in ``[delay, delay * (1 +
    BACKOFF_JITTER_FRACTION))``. Without jitter a provider outage synchronises
    every caller's retries into waves that arrive together and hit the provider
    again the moment it recovers; the jitter spreads them across a doubling of
    the delay instead. It only ever extends a wait, never shortens it: the
    exponential value is the floor the configured base intends, and firing
    earlier than that would add pressure to a provider that is already
    struggling.

    ``random_source`` is the injection point: a callable returning a float in
    ``[0.0, 1.0]``, defaulting to :func:`random.random`. Production passes
    nothing and gets the module's own source; a test passes a fixed one, so
    ``lambda: 0.0`` reproduces the un-jittered schedule exactly and ``lambda:
    1.0`` produces the largest wait the fraction allows. A source that returns
    anything else raises :exc:`ValueError`.

    A delay that would consume the whole remaining budget is refused, not
    shortened. Pass ``remaining`` — the seconds left in the invocation's
    :class:`Deadline`, as ``deadline.remaining()`` — and a positive delay that
    is not strictly less than it raises
    :class:`~model_gateway.errors.DeadlineExceededError`. PRD section 15
    requires that a retry must not exceed the caller's remaining deadline, and
    clamping the delay to what was left would sleep until the deadline expired
    and then fail anyway, having knowingly waited for a retry that could never
    run; refusing now says the same thing immediately and leaves the budget
    unconsumed. A delay of zero is still returned when the budget is spent —
    there is nothing to wait for, so nothing is consumed, and the caller's own
    budget check decides what happens next. ``remaining=None`` means the gateway
    set no deadline, and nothing is refused.

    An ``attempt`` below 1 or a negative ``base_ms`` raises :exc:`ValueError`: a
    schedule with no first step is not a schedule.
    """
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be a positive integer, got {attempt!r}")
    if isinstance(base_ms, bool) or not isinstance(base_ms, int | float):
        raise ValueError(f"base_ms must be a non-negative number, got {base_ms!r}")
    if not isfinite(base_ms) or base_ms < 0:
        raise ValueError(f"base_ms must be a non-negative number, got {base_ms!r}")
    source = random.random if random_source is None else random_source
    exponential = base_ms * 2 ** (attempt - 1) / 1000
    jittered = exponential * (1.0 + BACKOFF_JITTER_FRACTION * _jitter_draw(source))
    delay = min(jittered, BACKOFF_CEILING_SECONDS)
    if remaining is not None and delay > 0.0 and delay >= remaining:
        raise DeadlineExceededError(
            f"the retry backoff of {delay:.3f}s would consume the invocation's "
            f"remaining budget of {remaining:.3f}s, so no retry could run after it"
        )
    return delay


def _require_wait_timeout(timeout: float | None) -> None:
    """Refuse a slot-wait bound that is not a usable number of seconds.

    ``None`` is the gateway's "no bound" marker. A number must be finite and
    non-negative: zero means the caller takes a slot only if one is free right
    now, and anything below zero is a bound no wait could satisfy. A boolean is
    an ``int`` in Python but is not a duration, so it is refused with the rest.
    """
    if timeout is None:
        return
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not isfinite(timeout)
        or timeout < 0.0
    ):
        raise ValueError(
            f"timeout must be a non-negative number of seconds or None, got {timeout!r}"
        )


class ConcurrencyLimiter:
    """Bounds the simultaneous upstream attempts and the queue waiting for one.

    The deployment's ``PROVIDER_MAX_CONCURRENT_REQUESTS`` becomes this limiter's
    ``maximum``, and :meth:`ConcurrencyLimiter.slot` is the seam the
    orchestration wraps an attempt in: ``async with limiter.slot():`` holds a
    slot for exactly the duration of the attempt, and the number of attempts in
    flight never exceeds the maximum.

    Two bounds, because one is not enough. ``maximum`` bounds the attempts in
    flight, which is what would otherwise grow without limit inside the gateway.
    ``max_waiters`` bounds the callers *queued* for a slot, because an unbounded
    queue turns process memory into the overload buffer: ten thousand
    invocations against a hundred slots would leave nine thousand nine hundred
    resident tasks, each holding payload memory, framework state and a socket,
    waiting for a turn most of them will never get. A caller that arrives when
    ``max_waiters`` callers are already queued is refused **immediately** with
    :class:`~model_gateway.errors.ResourceExhaustedError` carrying reason
    ``CONCURRENCY_LIMIT_EXCEEDED`` instead of being queued. That refusal is
    controlled overload shedding rather than a failure: the gateway is telling
    the caller to back off now rather than accepting work it cannot serve in
    time, and it is the answer the caller would have reached after a long wait,
    minus the memory and the latency. ``max_waiters`` defaults to
    :data:`WAITER_QUEUE_MULTIPLE` times ``maximum``, so a burst a few times the
    pool's width is absorbed and a sustained overload is shed.

    Waiting is bounded by the invocation deadline. :meth:`slot` takes an
    optional ``timeout`` that the caller passes its remaining deadline to
    (``deadline.remaining()``), and a wait that outlives it raises
    :class:`~model_gateway.errors.DeadlineExceededError` rather than blocking
    until a slot happens to free. Under a saturated provider an unbounded wait
    leaves arbitrarily many already-expired invocations resident, each still
    holding its payload and its framework state and each still ahead of work
    that could actually be served. ``timeout=None`` waits without a bound, which
    is what an invocation with no deadline asked for.

    Queueing behaviour. Queued waiters are admitted as slots become available.
    No admission order is promised and none is relied on: the order is whatever
    the semaphore underneath does, and starvation is not intentionally
    introduced — every waiter is admitted as long as slots keep freeing. A
    caller that needs a stronger guarantee than that needs an explicit queue,
    which this limiter deliberately is not.

    A slot is released on every exit from the ``async with`` body — success,
    exception, or cancellation — so a limiter never deadlocks at its limit: a
    caller that cannot get a slot waits until one frees, and every holder frees
    exactly one. A caller cancelled while waiting never acquires a slot, and a
    caller whose wait runs out never acquires one either. ``maximum`` must be a
    positive integer; a limit of zero or less would admit nothing and is refused
    at construction. ``max_waiters`` must be a non-negative integer: zero
    refuses every caller that would have had to queue, which is a coherent, if
    extreme, policy rather than an error.
    """

    def __init__(self, maximum: int, *, max_waiters: int | None = None) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise ValueError(f"maximum must be a positive integer, got {maximum!r}")
        if maximum < 1:
            raise ValueError(f"maximum must be a positive integer, got {maximum!r}")
        if max_waiters is None:
            max_waiters = maximum * WAITER_QUEUE_MULTIPLE
        elif isinstance(max_waiters, bool) or not isinstance(max_waiters, int) or max_waiters < 0:
            raise ValueError(f"max_waiters must be a non-negative integer, got {max_waiters!r}")
        self._maximum = maximum
        self._max_waiters = max_waiters
        self._slots = asyncio.Semaphore(maximum)
        self._waiting = 0

    @property
    def maximum(self) -> int:
        """The configured bound on simultaneous upstream attempts."""
        return self._maximum

    @property
    def max_waiters(self) -> int:
        """The bound on callers queued for a slot before the limiter sheds load."""
        return self._max_waiters

    @asynccontextmanager
    async def slot(self, timeout: float | None = None) -> AsyncIterator[None]:
        """Hold one upstream-attempt slot until the wrapped body exits.

        Waits for a slot when the maximum is already in use, then yields with
        the slot held and releases it on every exit: the body returning,
        raising, or being cancelled. A body cancelled mid-attempt releases its
        slot, and a caller cancelled while waiting never acquires one.

        ``timeout`` bounds that wait: pass the invocation's remaining deadline
        (``deadline.remaining()``) and a wait that outlives it raises
        :class:`~model_gateway.errors.DeadlineExceededError` instead of blocking
        until a slot frees. ``None`` waits without a bound. A caller that
        arrives when ``max_waiters`` callers are already queued is refused
        immediately with :class:`~model_gateway.errors.ResourceExhaustedError`
        and never reaches the wait at all.
        """
        _require_wait_timeout(timeout)
        await self._acquire(timeout)
        try:
            yield
        finally:
            self._slots.release()

    async def _acquire(self, timeout: float | None) -> None:
        """Take a slot, queue for one within both bounds, or refuse the caller.

        Three outcomes. A free slot is taken at once and no queue is entered. A
        caller that has to wait is admitted to the queue while it has room,
        waits — bounded by ``timeout`` when there is one — and takes the slot
        that wait grants it. A caller that finds the queue full, or whose wait
        runs out, is refused without ever holding a slot.

        ``locked()`` is the queue test, not only the count test: Python's
        semaphore reports itself locked while any waiter is queued, so a caller
        arriving behind waiting ones queues rather than taking the slot they
        were about to be given.
        """
        if not self._slots.locked():
            await self._slots.acquire()
            return
        if self._waiting >= self._max_waiters:
            raise ResourceExhaustedError(
                f"the gateway's upstream concurrency limit is saturated: "
                f"{self._max_waiters} callers are already queued behind "
                f"{self._maximum} slots",
                reason=ErrorReason.CONCURRENCY_LIMIT_EXCEEDED,
                details={"maximum": str(self._maximum), "max_waiters": str(self._max_waiters)},
            )
        self._waiting += 1
        try:
            if timeout is None:
                await self._slots.acquire()
            else:
                await asyncio.wait_for(self._slots.acquire(), timeout)
        except TimeoutError as expired:
            raise DeadlineExceededError(
                f"the invocation deadline expired while waiting for an upstream concurrency "
                f"slot, with all {self._maximum} slots in use"
            ) from expired
        finally:
            self._waiting -= 1
