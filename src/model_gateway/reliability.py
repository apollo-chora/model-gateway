"""Reliability primitives: the policy the gateway's retry, fallback and
concurrency loop is built from (PRD sections 11.3 and 15).

This module is pure policy. It imports the gateway's own error taxonomy and
nothing else: no HTTP, no inference library, no I/O. The orchestration that
uses these primitives is a separate slice and codes against the names this
module exports.

* :class:`Deadline` is the single total invocation budget. Provider timeouts,
  connect timeouts, retries and fallbacks all draw from the one remaining
  budget it reports, and no attempt restarts it.
* :func:`is_retryable` classifies a failure from the gateway's own categories
  and reasons, never from a library exception type.
* :func:`backoff_seconds` computes the delay before the next attempt: the
  exponential schedule the configured base defines, capped so a retry can
  never outlive the deadline it draws from.
* :class:`ConcurrencyLimiter` bounds the number of simultaneous upstream
  attempts, queueing the rest fairly until a slot frees.

PRD section 15 gives the gateway one deadline per invocation and requires that
a retry must not exceed the caller's remaining deadline. These primitives are
how that is enforced: time is subtracted, never reset, and every delay is
clamped against what is left.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from model_gateway.errors import (
    DeadlineExceededError,
    ResourceExhaustedError,
    UnavailableError,
)

__all__ = [
    "BACKOFF_CEILING_SECONDS",
    "ConcurrencyLimiter",
    "Deadline",
    "backoff_seconds",
    "is_retryable",
]

# The ceiling on one backoff delay, in seconds. Without a ceiling a long retry
# schedule would outlast any invocation budget, and with the deployment's
# default 110 second provider timeout a 30 second delay still leaves room for
# the attempt itself. The schedule is exponential in the attempt count, so the
# ceiling is what keeps "retries must be bounded" (PRD 11.2) true for every
# retry count a deployment configures.
BACKOFF_CEILING_SECONDS = 30.0


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
    """Whether the failure ``error`` is worth another attempt or a fallback.

    The answer comes from the gateway's own taxonomy — the category the
    failure was classified into, and the reason that category permits — and
    from nothing else. A library exception type is never inspected: a failure
    that has not been classified into the taxonomy has no gateway meaning yet,
    so it is not retryable. Callers pass such a failure through
    :func:`~model_gateway.errors.normalise_exception` first, which lands it in
    :class:`~model_gateway.errors.UnavailableError` or
    :class:`~model_gateway.errors.InternalError`, both decided below.

    Retryable, because a later attempt or another route can still succeed:

    * :class:`~model_gateway.errors.UnavailableError` — no provider could
      serve the request (``PROVIDER_UNAVAILABLE``). A provider being down is
      transient: the retry may reach it after it recovers, and PRD section
      11.2 names connection failures a retry trigger.
    * :class:`~model_gateway.errors.ResourceExhaustedError` — an upstream
      rate, concurrency or quota limit (``RATE_LIMITED`` and
      ``CONCURRENCY_LIMIT_EXCEEDED``). PRD section 11.2 lists HTTP 429 a
      retry trigger, and the remedy for both reasons is the same shape: wait
      and retry, which is what the backoff schedule exists for.
    * :class:`~model_gateway.errors.DeadlineExceededError` — a deadline
      expired, retryable only while the invocation deadline still has budget
      left. ``remaining`` is the seconds left in the invocation's
      :class:`Deadline`, passed as ``deadline.remaining()``: a provider or
      attempt timeout does not by itself exhaust the invocation budget, so
      while time remains another attempt may fit inside it. Once the budget is
      gone (``remaining`` is zero) there is nothing left to fit an attempt
      into. ``remaining=None`` means the gateway set no invocation deadline at
      all, so a provider timeout consumes no gateway budget and stays
      retryable while the retry count allows — PRD section 11.2 lists timeouts
      a retry trigger, and with no deadline the retry count is the only bound.

    Not retryable, because repeating the call cannot change the answer:
    invalid arguments, authentication and permission failures, a missing
    resource or model, a capability the route does not advertise, a failed
    precondition, and internal errors. The first group describes the request
    or the caller and would repeat identically; a capability failure is a
    property of the route that was chosen, and PRD section 11.2 filters
    fallback candidates by the same test, so no eligible route can serve it;
    and an internal error is a gateway defect, which PRD section 15 classes as
    a deterministic error category — deterministic failures are reported, not
    retried into.
    """
    if isinstance(error, DeadlineExceededError):
        return remaining is None or remaining > 0
    return isinstance(error, (UnavailableError, ResourceExhaustedError))


def backoff_seconds(
    attempt: int,
    base_ms: float,
    *,
    remaining: float | None = None,
) -> float:
    """The delay in seconds before the next attempt, exponential and capped.

    ``attempt`` is the 1-based number of the attempt that just failed, and the
    schedule grows exponentially from ``base_ms`` — the configured
    ``PROVIDER_RETRY_BACKOFF_MS``: one base after the first attempt, two after
    the second, four after the third, ``base_ms * 2 ** (attempt - 1)``
    milliseconds in general.

    The growth is bounded by :data:`BACKOFF_CEILING_SECONDS`, a fixed ceiling
    of 30 seconds, so no single wait can dominate an invocation budget.

    The delay never exceeds the invocation's remaining budget: pass
    ``remaining`` — the seconds left in the invocation's :class:`Deadline`, as
    ``deadline.remaining()`` — and the delay is clamped to it. PRD section 15
    requires that a retry must not exceed the caller's remaining deadline, and
    the wait is part of what the retry consumes, so the clamp is applied here
    rather than left to the caller. ``remaining=None`` means the gateway set no
    deadline, and clamps nothing. An exhausted budget clamps the delay to
    zero: retrying is still permitted, it simply does not wait first.

    There is no jitter, deliberately: the delay is a pure function of its
    arguments, so a retry schedule is reproducible in tests and in records.
    A deployment that wants jitter can wrap this function with its own.

    An ``attempt`` below 1 or a negative ``base_ms`` raises :exc:`ValueError`:
    a schedule with no first step is not a schedule.
    """
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be a positive integer, got {attempt!r}")
    if isinstance(base_ms, bool) or not isinstance(base_ms, int | float):
        raise ValueError(f"base_ms must be a non-negative number, got {base_ms!r}")
    if not isfinite(base_ms) or base_ms < 0:
        raise ValueError(f"base_ms must be a non-negative number, got {base_ms!r}")
    delay = min(base_ms * 2 ** (attempt - 1) / 1000, BACKOFF_CEILING_SECONDS)
    if remaining is not None:
        delay = min(delay, max(remaining, 0.0))
    return delay


class ConcurrencyLimiter:
    """Bounds the number of simultaneous upstream attempts (PRD section 15).

    The deployment's ``PROVIDER_MAX_CONCURRENT_REQUESTS`` becomes this
    limiter's maximum, and :meth:`ConcurrencyLimiter.slot` is the seam the
    orchestration wraps an attempt in: ``async with limiter.slot():`` holds a
    slot for exactly the duration of the attempt, and the number of attempts
    in flight never exceeds the maximum.

    Queueing behaviour. Waiting callers queue in arrival order and are
    admitted in that order — first come, first served — so a burst of new
    callers can queue behind the maximum without starving any one caller
    indefinitely: a slot that frees always goes to the longest-waiting caller,
    and a newcomer queues behind everyone already waiting rather than
    barging ahead of them. The queue of waiting callers is unbounded; what is
    bounded is the number of callers actually inside a slot, which is the
    thing that would otherwise grow without limit. A waiting caller holds no
    upstream connection and no provider resource, so the bound that matters —
    simultaneous upstream attempts — is the one enforced here; how long a
    caller may wait is bounded by its own invocation deadline, and a caller
    cancelled while waiting leaves the queue without consuming a slot.

    A slot is released on every exit from the ``async with`` body — success,
    exception, or cancellation — so a limiter never deadlocks at its limit: a
    caller that cannot get a slot waits until one frees, and every holder
    frees exactly one. ``maximum`` must be a positive integer; a limit of
    zero or less would admit nothing and is refused at construction.
    """

    def __init__(self, maximum: int) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise ValueError(f"maximum must be a positive integer, got {maximum!r}")
        if maximum < 1:
            raise ValueError(f"maximum must be a positive integer, got {maximum!r}")
        self._maximum = maximum
        self._slots = asyncio.Semaphore(maximum)

    @property
    def maximum(self) -> int:
        """The configured bound on simultaneous upstream attempts."""
        return self._maximum

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one upstream-attempt slot until the wrapped body exits.

        Waits for a slot when the maximum is already in use, then yields with
        the slot held and releases it on every exit: the body returning,
        raising, or being cancelled. A body cancelled mid-attempt releases
        its slot, and a caller cancelled while waiting never acquires one.
        """
        await self._slots.acquire()
        try:
            yield
        finally:
            self._slots.release()
