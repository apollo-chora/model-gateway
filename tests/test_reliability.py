"""Tests for the reliability primitives.

The primitives are pure policy, so these are unit tests throughout. The
:class:`~model_gateway.reliability.Deadline` tests pin the arithmetic to an
injectable clock and the backoff tests pin the jitter to an injectable random
source, so nothing here waits on real time or on chance. The classification
tests walk every category of the gateway's taxonomy — by reason as well as by
category — and ask both the same-route and the route-level question; the
backoff tests pin the schedule's growth, its ceiling, its jitter and its
refusal to spend the last of the budget; and the limiter tests hold it under
contention, covering the deadline-bounded wait, the waiter-queue bound and the
two cancellation paths a retry loop depends on: a caller cancelled while queued
for a slot, and a holder cancelled inside its slot.
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
    MalformedResponseError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.reliability import (
    BACKOFF_CEILING_SECONDS,
    BACKOFF_JITTER_FRACTION,
    WAITER_QUEUE_MULTIPLE,
    ConcurrencyLimiter,
    Deadline,
    backoff_seconds,
    is_fallback_eligible,
    is_retryable,
)

MESSAGE = "a failure the gateway authored"


def no_jitter() -> float:
    """A random source that reproduces the un-jittered schedule exactly."""
    return 0.0


def full_jitter() -> float:
    """A random source that produces the largest wait the fraction allows."""
    return 1.0


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
    """The same-route question: what another attempt at the same route could fix."""

    def test_an_unavailable_provider_is_retryable(self):
        assert is_retryable(UnavailableError(MESSAGE))

    @pytest.mark.parametrize("remaining", [None, 0.5, 110.0])
    def test_a_deadline_with_budget_left_is_retryable(self, remaining):
        assert is_retryable(DeadlineExceededError(MESSAGE), remaining=remaining)

    def test_a_deadline_with_no_budget_left_is_not_retryable(self):
        assert not is_retryable(DeadlineExceededError(MESSAGE), remaining=0.0)


class TestResourceExhaustedClassification:
    """One category, two reasons, and the reasons are not the same failure."""

    @pytest.mark.parametrize(
        "reason", [ErrorReason.RATE_LIMITED, ErrorReason.CONCURRENCY_LIMIT_EXCEEDED]
    )
    def test_no_reason_of_the_category_is_retried_on_the_same_route(self, reason):
        assert not is_retryable(ResourceExhaustedError(MESSAGE, reason=reason))

    def test_a_local_concurrency_rejection_is_not_worth_another_route_either(self):
        """The limiter is gateway-wide, so every route would refuse the same way."""
        error = ResourceExhaustedError(MESSAGE, reason=ErrorReason.CONCURRENCY_LIMIT_EXCEEDED)
        assert not is_fallback_eligible(error)

    def test_an_upstream_rate_limit_is_worth_another_route_but_not_a_retry(self):
        """The limit belongs to the route that answered; another may not be limited."""
        error = ResourceExhaustedError(MESSAGE, reason=ErrorReason.RATE_LIMITED)
        assert is_fallback_eligible(error)
        assert not is_retryable(error)


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
    def test_a_deterministic_failure_is_neither_retried_nor_fallen_back_from(self, error):
        assert not is_retryable(error)
        assert not is_fallback_eligible(error)

    def test_an_unclassified_failure_is_neither_retried_nor_fallen_back_from(self):
        assert not is_retryable(RuntimeError("litellm.APIConnectionError: boom"))
        assert not is_retryable(ValueError("some provider's own ValueError"))
        assert not is_fallback_eligible(RuntimeError("litellm.APIConnectionError: boom"))


class TestFallbackEligibility:
    """The route-level question: what another route could fix."""

    def test_an_unavailable_provider_is_worth_another_route(self):
        assert is_fallback_eligible(UnavailableError(MESSAGE))

    @pytest.mark.parametrize("remaining", [None, 0.5, 110.0])
    def test_a_deadline_with_budget_left_is_worth_another_route(self, remaining):
        assert is_fallback_eligible(DeadlineExceededError(MESSAGE), remaining=remaining)

    def test_a_deadline_with_no_budget_left_is_not_worth_another_route(self):
        assert not is_fallback_eligible(DeadlineExceededError(MESSAGE), remaining=0.0)

    @pytest.mark.parametrize("remaining", [None, 0.5, 110.0])
    def test_a_malformed_upstream_response_is_worth_another_route(self, remaining):
        """The fault lies in that upstream's answer, so another route may fix it."""
        assert is_fallback_eligible(MalformedResponseError(MESSAGE), remaining=remaining)


class TestMalformedResponseClassification:
    """The two answers differ for a malformed upstream response, by design.

    The request was valid and the route was reachable; what came back was
    structurally wrong. Retrying the route that just answered wrongly is
    unlikely to produce a different answer, but the same request to a
    different provider may well succeed — the two decisions the taxonomy
    keeps separate.
    """

    def test_a_malformed_upstream_response_is_not_retried_on_the_same_route(self):
        assert not is_retryable(MalformedResponseError(MESSAGE))

    def test_a_malformed_upstream_response_is_fallback_eligible(self):
        assert is_fallback_eligible(MalformedResponseError(MESSAGE))

    def test_a_gateway_defect_is_still_neither_retried_nor_walked(self):
        """Widening internal would make a programming bug fallback-eligible."""
        assert not is_retryable(InternalError(MESSAGE))
        assert not is_fallback_eligible(InternalError(MESSAGE))


class TestBackoff:
    def test_the_first_retry_waits_one_base(self):
        assert backoff_seconds(1, 500, random_source=no_jitter) == pytest.approx(0.5)

    def test_each_failed_attempt_doubles_the_wait(self):
        assert backoff_seconds(2, 500, random_source=no_jitter) == pytest.approx(1.0)
        assert backoff_seconds(3, 500, random_source=no_jitter) == pytest.approx(2.0)
        assert backoff_seconds(4, 500, random_source=no_jitter) == pytest.approx(4.0)

    def test_the_growth_is_bounded_by_the_documented_ceiling(self):
        assert backoff_seconds(7, 500, random_source=no_jitter) == pytest.approx(
            BACKOFF_CEILING_SECONDS
        )
        assert backoff_seconds(20, 500, random_source=full_jitter) == pytest.approx(
            BACKOFF_CEILING_SECONDS
        )

    def test_a_zero_base_waits_nothing(self):
        assert backoff_seconds(3, 0) == 0.0

    def test_the_wait_is_jittered_by_default(self):
        """Production passes no source, so waits vary instead of arriving in waves."""
        assert len({backoff_seconds(3, 250) for _ in range(200)}) > 1

    def test_the_default_jitter_stays_inside_the_documented_range(self):
        delay = backoff_seconds(3, 250)
        assert 1.0 <= delay <= 1.0 * (1.0 + BACKOFF_JITTER_FRACTION)

    def test_jitter_extends_the_wait_and_never_shortens_it(self):
        floor = backoff_seconds(3, 250, random_source=no_jitter)
        ceiling = backoff_seconds(3, 250, random_source=full_jitter)
        assert floor == pytest.approx(1.0)
        assert ceiling == pytest.approx(1.0 * (1.0 + BACKOFF_JITTER_FRACTION))

    @pytest.mark.parametrize("draw", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_the_jittered_wait_stays_inside_the_documented_range(self, draw):
        delay = backoff_seconds(4, 500, random_source=lambda: draw)
        assert 4.0 <= delay <= 4.0 * (1.0 + BACKOFF_JITTER_FRACTION)

    def test_the_injected_source_makes_the_schedule_reproducible(self):
        assert backoff_seconds(3, 250, random_source=full_jitter) == backoff_seconds(
            3, 250, random_source=full_jitter
        )

    def test_no_budget_refuses_nothing(self):
        assert backoff_seconds(4, 500, remaining=None, random_source=no_jitter) == pytest.approx(
            4.0
        )

    def test_a_wait_that_fits_the_remaining_budget_is_returned(self):
        assert backoff_seconds(1, 500, remaining=5.0, random_source=no_jitter) == pytest.approx(0.5)

    @pytest.mark.parametrize("remaining", [0.0, 0.5, 1.0])
    def test_a_wait_that_would_consume_the_budget_fails_the_deadline(self, remaining):
        with pytest.raises(DeadlineExceededError):
            backoff_seconds(4, 500, remaining=remaining, random_source=no_jitter)

    def test_the_refusal_is_the_gateways_deadline_error(self):
        with pytest.raises(DeadlineExceededError) as error:
            backoff_seconds(4, 500, remaining=1.0, random_source=no_jitter)
        assert error.value.reason is ErrorReason.DEADLINE_EXCEEDED
        assert error.value.http_status == 504

    def test_a_zero_wait_is_returned_even_when_the_budget_is_spent(self):
        """There is nothing to wait for, so nothing is consumed; the caller decides next."""
        assert backoff_seconds(4, 0, remaining=0.0, random_source=no_jitter) == 0.0

    @pytest.mark.parametrize("draw", [-0.1, 1.5, float("nan"), float("inf"), "half", True])
    def test_a_source_outside_the_unit_interval_is_refused(self, draw):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            backoff_seconds(1, 500, random_source=lambda: draw)

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

    def test_the_waiter_bound_defaults_to_a_multiple_of_the_limit(self):
        assert ConcurrencyLimiter(7).max_waiters == 7 * WAITER_QUEUE_MULTIPLE

    def test_the_waiter_bound_is_configurable(self):
        assert ConcurrencyLimiter(7, max_waiters=2).max_waiters == 2

    @pytest.mark.parametrize("maximum", [0, -1, 2.5, "many", True])
    def test_a_limit_that_admits_nothing_is_refused(self, maximum):
        with pytest.raises(ValueError, match="positive integer"):
            ConcurrencyLimiter(maximum)

    @pytest.mark.parametrize("max_waiters", [-1, 1.5, "many", True])
    def test_a_waiter_bound_that_is_not_a_count_is_refused(self, max_waiters):
        with pytest.raises(ValueError, match="non-negative integer"):
            ConcurrencyLimiter(2, max_waiters=max_waiters)

    @pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), "soon", True])
    async def test_an_unusable_wait_bound_is_refused(self, timeout):
        limiter = ConcurrencyLimiter(1)
        with pytest.raises(ValueError, match="non-negative number of seconds"):
            async with limiter.slot(timeout=timeout):
                pass

    async def test_simultaneous_attempts_never_exceed_the_maximum(self):
        limiter = ConcurrencyLimiter(3, max_waiters=64)
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

    async def test_every_queued_waiter_is_admitted_as_slots_free(self):
        """Admission order is not promised; being admitted eventually is."""
        limiter = ConcurrencyLimiter(1, max_waiters=5)
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
        assert sorted(admitted) == [0, 1, 2, 3, 4]

    async def test_a_full_waiter_queue_is_refused_immediately(self):
        limiter = ConcurrencyLimiter(1, max_waiters=1)
        release = asyncio.Event()
        holder_ready = asyncio.Event()
        queued = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        async def queue_up():
            async with limiter.slot():
                queued.set()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()
        waiter = asyncio.create_task(queue_up())
        for _ in range(3):
            await asyncio.sleep(0)

        with pytest.raises(ResourceExhaustedError) as error:
            async with limiter.slot():
                pytest.fail("a caller refused for a full queue must not hold a slot")

        assert error.value.reason is ErrorReason.CONCURRENCY_LIMIT_EXCEEDED
        assert error.value.http_status == 429

        release.set()
        await asyncio.gather(holder, waiter)
        assert queued.is_set()

    async def test_a_zero_waiter_bound_refuses_every_caller_that_would_queue(self):
        limiter = ConcurrencyLimiter(1, max_waiters=0)
        holder_ready = asyncio.Event()
        release = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()

        with pytest.raises(ResourceExhaustedError):
            async with limiter.slot():
                pytest.fail("no waiter may be queued when the bound is zero")

        release.set()
        await holder

    async def test_a_free_slot_is_taken_even_when_no_waiter_may_queue(self):
        limiter = ConcurrencyLimiter(1, max_waiters=0)
        async with limiter.slot(timeout=0.0):
            pass

    async def test_a_timed_out_wait_raises_the_deadline_error(self):
        limiter = ConcurrencyLimiter(1)
        holder_ready = asyncio.Event()
        release = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()

        with pytest.raises(DeadlineExceededError) as error:
            async with limiter.slot(timeout=0.01):
                pytest.fail("a caller whose wait ran out must not hold a slot")

        assert error.value.reason is ErrorReason.DEADLINE_EXCEEDED
        assert error.value.http_status == 504

        release.set()
        await holder

    async def test_a_zero_timeout_refuses_to_wait_at_all(self):
        limiter = ConcurrencyLimiter(1)
        holder_ready = asyncio.Event()
        release = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()

        with pytest.raises(DeadlineExceededError):
            async with limiter.slot(timeout=0.0):
                pytest.fail("a zero timeout must not wait for a slot")

        release.set()
        await holder

    async def test_a_timed_out_waiter_leaves_the_slot_for_the_next_caller(self):
        limiter = ConcurrencyLimiter(1)
        holder_ready = asyncio.Event()
        release = asyncio.Event()
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

        with pytest.raises(DeadlineExceededError):
            async with limiter.slot(timeout=0.01):
                pytest.fail("a caller whose wait ran out must not hold a slot")

        release.set()
        await holder
        await asyncio.wait_for(take_slot(), timeout=1.0)
        assert acquired.is_set()

    async def test_a_wait_that_fits_its_timeout_still_takes_the_slot(self):
        limiter = ConcurrencyLimiter(1)
        holder_ready = asyncio.Event()
        release = asyncio.Event()
        acquired = asyncio.Event()

        async def hold():
            async with limiter.slot():
                holder_ready.set()
                await release.wait()

        async def take_slot():
            async with limiter.slot(timeout=5.0):
                acquired.set()

        holder = asyncio.create_task(hold())
        await holder_ready.wait()
        waiter = asyncio.create_task(take_slot())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(holder, waiter)
        assert acquired.is_set()

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
