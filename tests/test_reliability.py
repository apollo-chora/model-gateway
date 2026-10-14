"""Tests for the reliability primitives (PRD sections 11.3 and 15).

The primitives are pure policy, so these are unit tests throughout. The
:class:`~model_gateway.reliability.Deadline` tests pin the arithmetic to an
injectable clock, so nothing here waits on real time. The classification tests
walk every category of the gateway's taxonomy, the backoff tests pin the
schedule's growth, its ceiling and its budget clamp, and the limiter tests
hold it under contention — including the two cancellation paths a retry loop
depends on: a caller cancelled while queued for a slot, and a holder
cancelled inside its slot.
"""

import asyncio
from time import monotonic

import pytest

from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    FailedPreconditionError,
    InternalError,
    InvalidArgumentError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.reliability import (
    BACKOFF_CEILING_SECONDS,
    ConcurrencyLimiter,
    Deadline,
    backoff_seconds,
    is_retryable,
)

MESSAGE = "a failure the gateway authored"


class TestDeadline:
    def test_a_deadline_without_seconds_is_unbounded(self):
        deadline = Deadline.after(None)
        assert not deadline.expired
        assert deadline.remaining() is None

    def test_a_budget_is_counted_down_from_the_anchor(self):
        deadline = Deadline.after(100.0)
        started = deadline.started_at
        assert deadline.remaining(now=lambda: started + 30.0) == pytest.approx(70.0)
        assert not deadline.expired

    def test_an_elapsed_budget_never_goes_negative(self):
        deadline = Deadline(seconds=1.0, started_at=monotonic() - 2.5)
        assert deadline.remaining() == 0.0
        assert deadline.expired

    def test_an_unbounded_deadline_never_expires(self):
        assert not Deadline.after(None).expired

    @pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), "soon", True])
    def test_an_unusable_budget_is_refused(self, seconds):
        with pytest.raises(ValueError, match="positive number of seconds"):
            Deadline.after(seconds)

    def test_seconds_and_the_anchor_must_be_given_together(self):
        with pytest.raises(ValueError, match="both be set or both be None"):
            Deadline(seconds=5.0, started_at=None)
        with pytest.raises(ValueError, match="both be set or both be None"):
            Deadline(seconds=None, started_at=12345.0)


class TestAttemptBudget:
    def test_no_bounds_leaves_the_attempt_unbounded(self):
        assert Deadline.after(None).budget_for(None) is None

    def test_only_a_configured_timeout_bounds_the_attempt(self):
        assert Deadline.after(None).budget_for(110.0) == 110.0

    def test_only_a_deadline_bounds_the_attempt(self):
        deadline = Deadline.after(100.0)
        started = deadline.started_at
        assert deadline.budget_for(None, now=lambda: started + 30.0) == pytest.approx(70.0)

    def test_the_deadline_wins_when_it_is_the_smaller_bound(self):
        deadline = Deadline.after(20.0)
        started = deadline.started_at
        assert deadline.budget_for(110.0, now=lambda: started + 10.0) == pytest.approx(10.0)

    def test_the_configured_timeout_wins_when_it_is_the_smaller_bound(self):
        deadline = Deadline.after(100.0)
        started = deadline.started_at
        assert deadline.budget_for(10.0, now=lambda: started + 30.0) == pytest.approx(10.0)

    def test_an_expired_deadline_yields_zero_not_a_negative_budget(self):
        deadline = Deadline(seconds=5.0, started_at=monotonic() - 10.0)
        assert deadline.expired
        assert deadline.remaining() == 0.0
        assert deadline.budget_for(110.0) == 0.0
        assert deadline.budget_for(None) == 0.0

    def test_a_retry_draws_from_the_budget_the_invocation_already_consumed(self):
        deadline = Deadline.after(100.0)
        started = deadline.started_at
        first = deadline.budget_for(110.0, now=lambda: started + 30.0)
        second = deadline.budget_for(110.0, now=lambda: started + 60.0)
        assert first == pytest.approx(70.0)
        assert second == pytest.approx(40.0)
        assert second < first

    @pytest.mark.parametrize("configured", [0, -5, float("nan"), float("inf"), "fast", True])
    def test_an_unusable_configured_timeout_is_refused(self, configured):
        with pytest.raises(ValueError, match="positive number of seconds"):
            Deadline.after(None).budget_for(configured)


class TestRetryableFailures:
    def test_an_unavailable_provider_is_retryable(self):
        assert is_retryable(UnavailableError(MESSAGE))

    @pytest.mark.parametrize(
        "reason", [ErrorReason.RATE_LIMITED, ErrorReason.CONCURRENCY_LIMIT_EXCEEDED]
    )
    def test_a_resource_exhausted_condition_is_retryable(self, reason):
        assert is_retryable(ResourceExhaustedError(MESSAGE, reason=reason))

    @pytest.mark.parametrize("remaining", [None, 0.5, 110.0])
    def test_a_deadline_with_budget_left_is_retryable(self, remaining):
        assert is_retryable(DeadlineExceededError(MESSAGE), remaining=remaining)

    def test_a_deadline_with_no_budget_left_is_not_retryable(self):
        assert not is_retryable(DeadlineExceededError(MESSAGE), remaining=0.0)


class TestNonRetryableFailures:
    @pytest.mark.parametrize(
        "error",
        [
            InvalidArgumentError(MESSAGE),
            UnauthenticatedError(MESSAGE),
            UnauthenticatedError(MESSAGE, reason=ErrorReason.INVALID_CREDENTIALS),
            PermissionDeniedError(MESSAGE),
            NotFoundError(MESSAGE),
            ModelNotFoundError(MESSAGE),
            FailedPreconditionError(MESSAGE),
            CapabilityError(MESSAGE),
            InternalError(MESSAGE),
        ],
    )
    def test_a_deterministic_failure_is_not_retryable(self, error):
        assert not is_retryable(error)

    def test_an_unclassified_failure_is_not_retryable(self):
        assert not is_retryable(RuntimeError("litellm.APIConnectionError: boom"))
        assert not is_retryable(ValueError("some provider's own ValueError"))


class TestBackoff:
    def test_the_first_retry_waits_one_base(self):
        assert backoff_seconds(1, 500) == pytest.approx(0.5)

    def test_each_failed_attempt_doubles_the_wait(self):
        assert backoff_seconds(2, 500) == pytest.approx(1.0)
        assert backoff_seconds(3, 500) == pytest.approx(2.0)
        assert backoff_seconds(4, 500) == pytest.approx(4.0)

    def test_the_growth_is_bounded_by_the_documented_ceiling(self):
        assert backoff_seconds(7, 500) == pytest.approx(BACKOFF_CEILING_SECONDS)
        assert backoff_seconds(20, 500) == pytest.approx(BACKOFF_CEILING_SECONDS)

    def test_a_zero_base_waits_nothing(self):
        assert backoff_seconds(3, 0) == 0.0

    def test_the_wait_never_exceeds_the_remaining_budget(self):
        assert backoff_seconds(4, 500, remaining=1.0) == pytest.approx(1.0)
        assert backoff_seconds(4, 500, remaining=2.0) == pytest.approx(2.0)

    def test_an_exhausted_budget_clamps_the_wait_to_zero(self):
        assert backoff_seconds(4, 500, remaining=0.0) == 0.0

    def test_no_budget_clamps_nothing(self):
        assert backoff_seconds(4, 500, remaining=None) == pytest.approx(4.0)

    def test_the_delay_is_deterministic(self):
        assert backoff_seconds(3, 250) == backoff_seconds(3, 250)

    @pytest.mark.parametrize("attempt", [0, -2])
    def test_a_schedule_with_no_first_step_is_refused(self, attempt):
        with pytest.raises(ValueError, match="positive integer"):
            backoff_seconds(attempt, 500)

    def test_a_negative_base_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            backoff_seconds(1, -1)


class TestConcurrencyLimiter:
    def test_the_configured_maximum_is_exposed(self):
        assert ConcurrencyLimiter(100).maximum == 100

    @pytest.mark.parametrize("maximum", [0, -1, 2.5, "many", True])
    def test_a_limit_that_admits_nothing_is_refused(self, maximum):
        with pytest.raises(ValueError, match="positive integer"):
            ConcurrencyLimiter(maximum)

    async def test_simultaneous_attempts_never_exceed_the_maximum(self):
        limiter = ConcurrencyLimiter(3)
        in_flight = 0
        peak = 0

        async def attempt():
            nonlocal in_flight, peak
            async with limiter.slot():
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0)
                in_flight -= 1

        await asyncio.gather(*(attempt() for _ in range(24)))
        assert peak <= limiter.maximum
        assert peak == limiter.maximum

    async def test_waiters_are_admitted_in_arrival_order(self):
        limiter = ConcurrencyLimiter(1)
        release = asyncio.Event()
        holder_ready = asyncio.Event()
        admitted = []

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        async def wait_in_line(index):
            async with limiter.slot():
                admitted.append(index)

        holder = asyncio.create_task(hold())
        await holder_ready.wait()
        waiters = [asyncio.create_task(wait_in_line(index)) for index in range(5)]
        for _ in waiters:
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(holder, *waiters)
        assert admitted == [0, 1, 2, 3, 4]

    async def test_a_cancelled_waiter_consumes_no_slot(self):
        limiter = ConcurrencyLimiter(1)
        release = asyncio.Event()
        holder_ready = asyncio.Event()
        acquired = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        async def take_slot():
            async with limiter.slot():
                acquired.set()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()

        cancelled = asyncio.create_task(take_slot())
        await asyncio.sleep(0)
        assert not acquired.is_set()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not acquired.is_set()

        release.set()
        await holder
        next_in_line = asyncio.create_task(take_slot())
        await asyncio.wait_for(acquired.wait(), timeout=1.0)
        assert acquired.is_set()
        await next_in_line

    async def test_a_holder_cancelled_inside_its_slot_releases_it(self):
        limiter = ConcurrencyLimiter(1)
        holder_ready = asyncio.Event()
        acquired = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await asyncio.sleep(30.0)

        async def take_slot():
            async with limiter.slot():
                acquired.set()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

        next_in_line = asyncio.create_task(take_slot())
        await asyncio.wait_for(acquired.wait(), timeout=1.0)
        await next_in_line

    async def test_slots_are_returned_after_every_use(self):
        limiter = ConcurrencyLimiter(2)
        for _ in range(20):
            async with limiter.slot():
                pass
