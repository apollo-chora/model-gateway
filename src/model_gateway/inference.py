"""The inference service: gateway-owned policy around one provider call.

The service owns policy; the provider owns translation. That split is PRD
section 9's boundary, and this module is the policy side of it:

* **Routing.** The public model ID in the request is resolved to the routes
  configured for its operation by the registry. An unknown ID is a
  :class:`~model_gateway.errors.ModelNotFoundError` raised here, before any
  provider is called, so a caller-supplied model name cannot reach an
  unconfigured upstream endpoint (PRD section 10).
* **Capability enforcement.** The request is validated against the primary
  route's capability contract before anything is selected. The registry
  advertises the primary route's capabilities as the public model's own (PRD
  section 4.3), so a request the primary cannot honour is refused with a
  :class:`~model_gateway.errors.CapabilityError` even when a later route declares
  what the request needs (PRD section 9.5): serving it would answer a request the
  public contract never advertised for that model. Silently dropping a tool, an
  image or a structured-output requirement is forbidden, and so is obtaining one
  through a fallback. Only then are the remaining routes filtered, each by the
  same test, into the fallbacks that could serve the request unchanged, so a
  fallback may preserve the advertised capabilities but can never expand them.
* **Attempt execution.** The provider is handed one resolved route, the request
  and an :class:`~model_gateway.providers.base.AttemptContext` carrying the
  attempt's own facts: the invocation it belongs to, which attempt it is and the
  budget that attempt may use (PRD section 9.6). The budget is the gateway's, so
  a provider never reads a deadline out of configuration it does not own, and the
  core never wraps the call in a timeout the provider cannot see.
* **Normalisation.** The provider's answer comes back as upstream data and is
  re-normalised at this boundary: the public model that was asked for, the
  upstream model that actually served it and the invocation's own reliability
  facts are the gateway's to state, because the gateway chose the route and made
  the attempts (PRD sections 5.2 and 9).
* **Reliability.** One invocation has one deadline, one retry loop and one
  attempt count (PRD sections 11.3 and 15), and the loop is built from the
  primitives of :mod:`model_gateway.reliability`. :func:`_select_routes` decides
  the primary route and the fallbacks that could serve the request;
  :meth:`InferenceService._invoke_on` anchors a single
  :class:`~model_gateway.reliability.Deadline` when the invocation begins and
  walks that route order, asking two separate questions about every failure.
  *May another attempt on this route help?*
  (:func:`~model_gateway.reliability.is_retryable`) decides whether the route is
  retried, within the configured retry count, waiting
  :func:`~model_gateway.reliability.backoff_seconds` between attempts. *May
  another route help?* (:func:`~model_gateway.reliability.is_fallback_eligible`)
  decides whether the invocation moves on to the next fallback. The two answers
  differ for the same failure, and the difference is the point: an upstream rate
  limit is worth another route but not another attempt at the route that refused,
  while a request the upstream rejected as invalid is worth neither — repeating
  it there cannot change the answer, and no other provider can change it either,
  because the answer is a property of the request. A failure that is neither
  retryable nor fallback-eligible therefore terminates the invocation where it
  happened, rather than spending another provider's capacity on a request that
  cannot succeed.
  Every attempt derives its budget from the one deadline, so a retry or a
  fallback consumes what is left rather than starting a fresh budget, and no
  attempt is started once that budget is gone; the backoff schedule counts a
  route's own retries rather than the invocation's attempts, so a fallback's
  first retry waits one base instead of inheriting the primary route's growth.
  When no route could serve the invocation, the failure reported is the last
  attempt's, which is the most informative one; a deadline that ran out before
  another attempt could begin is reported as a deadline error instead.

The provider performs exactly one call and owns protocol translation and
execution and nothing else (:mod:`model_gateway.providers.base`).

**``attempt_count`` counts gateway attempts, never upstream requests.** The
gateway counts the calls it makes. The inference library has internal retry
paths no supported knob disables (PRD section 11.3), so one gateway attempt can
carry more than one wire request and this count may understate what the upstream
saw. It must never be documented, reported or recorded as a wire-request count.

**Cancellation propagates.** :exc:`asyncio.CancelledError` is a
:exc:`BaseException` and is never caught here, so a cancelled invocation stops
at the attempt in flight, releases the concurrency slot that attempt held, and
makes no further attempt (PRD section 15).
"""

import asyncio
from dataclasses import dataclass, replace

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Route
from model_gateway.domain import (
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    ResponseFormatMode,
)
from model_gateway.errors import CapabilityError, DeadlineExceededError
from model_gateway.providers.base import AttemptContext, TextProvider, _is_usable_budget
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import (
    ConcurrencyLimiter,
    Deadline,
    backoff_seconds,
    is_fallback_eligible,
    is_retryable,
)

__all__ = ["InferenceService", "InvocationState"]

_DEFAULT_MAX_CONCURRENT_REQUESTS = 100
"""The concurrency bound a service is built with when no limiter is injected.

It is ``PROVIDER_MAX_CONCURRENT_REQUESTS``'s own default (PRD section 15), so a
service constructed without a deployment's configuration still bounds its
simultaneous upstream attempts rather than leaving them unbounded. The
application factory passes the configured value, or a limiter of its own.
"""


@dataclass(frozen=True)
class InvocationState:
    """What the gateway has done for one invocation so far (PRD sections 5.2 and 13).

    The response reports facts about the gateway's own execution — how many
    provider attempts were made and whether a fallback path was taken — and those
    are facts about the invocation rather than about any one answer a provider
    gave. They live here so that ``_normalise`` states them instead of inventing
    them: a hardcoded ``attempt_count`` of one is a claim about the reliability
    loop, which is exactly what this object can answer truthfully.

    ``attempt_count`` starts at zero, because a state exists before the first
    call, and counts the attempts that were made — including ones that raised,
    because a failed attempt is work the gateway did (PRD section 13). It counts
    *gateway* attempts: one call this service made, on one route, under one
    attempt number. It is not a wire-request count. The inference library has
    internal retry paths that no supported knob switches off (PRD section 11.3),
    so a single attempt here can carry more than one upstream request, and this
    count may understate what the upstream saw. Reporting it as the number of
    upstream requests would be a claim the gateway cannot support.

    ``fallback_used`` records that the invocation took a *fallback path*: a route
    other than the primary one served it because the normal primary path could
    not (PRD section 5.2). It is deliberately narrower than "a route other than
    the first configured one was used", which was also true whenever capability
    routing passed the primary over. The primary route is the one the public
    model advertises, and a request it cannot honour is refused rather than
    routed elsewhere, so capability routing can never set this flag. With that
    definition, ``attempt_count=1`` and ``fallback_used=False`` says the primary
    route served the invocation on its first attempt, rather than merely that no
    later route was configured.

    :meth:`with_attempt` is the one place the state advances, so the retry and
    fallback loop has a single seam to move it and no field to add in a second
    place.
    """

    attempt_count: int = 0
    fallback_used: bool = False

    def with_attempt(self, *, on_fallback: bool = False) -> "InvocationState":
        """The state after recording that one more provider attempt is made.

        The attempt is recorded before the call, so an attempt that raises counts
        too: the count describes what the gateway did, and a failure is not less
        of an attempt than a success (PRD section 13).

        ``on_fallback`` states that this attempt is made on a route other than
        the primary one, after the primary path could not serve the invocation.
        That is the one thing that makes ``fallback_used`` true (PRD section
        5.2), and it is sticky, because one such route is enough for the
        invocation to have fallen back. It defaults to false because an attempt
        on the primary route is never a fallback, however many routes are
        configured behind it.
        """
        return replace(
            self,
            attempt_count=self.attempt_count + 1,
            fallback_used=self.fallback_used or on_fallback,
        )


class InferenceService:
    """Policy for a text invocation: resolve, select, call, normalise."""

    def __init__(
        self,
        registry: ModelRegistry,
        provider: TextProvider,
        *,
        attempt_timeout: float | None = None,
        retry_count: int = 0,
        retry_backoff_ms: int = 0,
        max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT_REQUESTS,
        limiter: ConcurrencyLimiter | None = None,
    ) -> None:
        """Wire the service to the registry, the provider and the reliability policy.

        ``attempt_timeout`` is the one number the deployment configures today —
        ``PROVIDER_TIMEOUT_SECONDS`` — and it plays two roles at once: it is the
        invocation's total budget *and* the ceiling on each attempt inside it
        (PRD sections 9.6 and 15). Those are logically distinct bounds and this
        parameter does not yet separate them. The deadline is anchored from it
        when an invocation begins, and every attempt is handed the smaller of it
        and what is left of the deadline, so retries and fallbacks draw from one
        budget instead of each getting a fresh one. Serving a longer invocation
        budget with a shorter per-attempt timeout, or the reverse, needs a second
        configured value and is a future refinement; today there is exactly one.
        ``None`` means the gateway sets neither bound — no deadline and no
        per-attempt ceiling — which is not the same as a budget of zero. A value
        no attempt could use is refused here rather than on the first request.

        ``retry_count`` is how many times one route may be retried after its
        first attempt fails retryably — the configured ``PROVIDER_RETRY_COUNT`` —
        so a route is attempted at most ``retry_count + 1`` times.
        ``retry_backoff_ms`` is the base delay between those attempts, the
        configured ``PROVIDER_RETRY_BACKOFF_MS``, from which
        :func:`~model_gateway.reliability.backoff_seconds` derives its
        exponential schedule. Both default to no retry at all: a service built
        without a reliability policy makes one attempt per route, and the
        application factory passes the deployment's configured values.

        ``max_concurrent_requests`` is the configured maximum
        (``PROVIDER_MAX_CONCURRENT_REQUESTS``) that the concurrency limiter is
        built from, and ``limiter`` replaces that limiter outright, so a test can
        hold a small limit or a stub. One slot is held for the duration of one
        attempt and released before any backoff wait: holding it across a retry's
        delay would let a few slow callers occupy every slot while doing nothing,
        and starve the callers queued behind them. Waiting for a slot is bounded
        by what is left of the invocation's deadline, so a caller that cannot be
        served in time stops queueing rather than waiting for a slot it could no
        longer use.
        """
        if attempt_timeout is not None and not _is_usable_budget(attempt_timeout):
            raise ValueError(
                f"attempt_timeout must be a positive number of seconds or None, got "
                f"{attempt_timeout!r}"
            )
        _require_count("retry_count", retry_count)
        _require_count("retry_backoff_ms", retry_backoff_ms)
        self._registry = registry
        self._provider = provider
        self._attempt_timeout = attempt_timeout
        self._retry_count = retry_count
        self._retry_backoff_ms = retry_backoff_ms
        self._limiter = (
            limiter if limiter is not None else ConcurrencyLimiter(max_concurrent_requests)
        )

    async def invoke(self, request: InvocationRequest) -> InvocationResponse:
        """Invoke the public model named by ``request.model``.

        Resolves the ID to its configured routes, validates the request against
        the primary route's capability contract — the contract the registry
        advertises for the model — then serves the invocation under one deadline
        and one retry loop (PRD sections 11.2, 11.3 and 15). Raises
        :class:`~model_gateway.errors.ModelNotFoundError` for an unknown ID and
        :class:`~model_gateway.errors.CapabilityError` when the primary route
        cannot honour the request; otherwise raises the failure of the last
        attempt that was made, a
        :class:`~model_gateway.errors.DeadlineExceededError` when the invocation
        budget ran out before another attempt could begin, or — when the failure
        says the request itself cannot succeed — that failure, raised where it
        happened rather than carried to another provider. See
        :meth:`_invoke_on` for which failures are which.

        The returned ``attempt_count`` counts the gateway's own attempts and is
        not a count of upstream requests: the inference library can issue more
        than one request inside a single attempt (PRD section 11.3).
        """
        routes = self._registry.routes_for(request.model)
        return await self._invoke_on(request, routes)

    async def _invoke_on(
        self, request: InvocationRequest, routes: tuple[Route, ...]
    ) -> InvocationResponse:
        """Serve one invocation from the configured route order, primary first.

        Selection decides which routes may serve the request; execution walks
        them under the invocation's single deadline. Each failure is asked two
        separate questions, because the answers differ.

        *May another attempt on this route help?*
        :func:`~model_gateway.reliability.is_retryable` answers it, and a yes
        retries the same route while the configured retry count allows.

        *May another route help?* :func:`~model_gateway.reliability.is_fallback_eligible`
        answers that one, and a yes moves the invocation on to the next route in
        ``plan.fallbacks``. A fallback that answers is what makes
        ``fallback_used`` true (PRD section 5.2).

        A failure that answers no to both terminates the invocation where it
        happened, and the refusal raised is that failure. This is the case the
        two questions exist to separate: a request the upstream rejected as
        invalid, a credential it refused or a capability it does not have is a
        property of the request or of the caller, so it would repeat identically
        on every route — walking the rest of the fallbacks would spend their
        capacity on a request that cannot succeed, and would make the caller's
        answer depend on which providers happen to be configured. A route-local
        failure that is simply not worth repeating on the same route — an
        upstream rate limit, say — is not retried here but is still walked to
        the next route.

        One :class:`~model_gateway.reliability.Deadline` is anchored here, when
        the invocation begins, and every attempt derives its budget from it, so
        retries and fallbacks consume one budget rather than each getting a fresh
        one (PRD section 15). An attempt is never started once that budget is
        spent: the invocation fails with a
        :class:`~model_gateway.errors.DeadlineExceededError` and no provider is
        called.

        When no route could serve the invocation, the failure raised is the last
        attempt's rather than the first: the most recent failure is the one that
        says why the invocation could not be served (PRD section 11.2).

        The wait between attempts happens outside the concurrency slot, so a
        caller in backoff holds nothing another caller could be using, and it is
        waited only when a retry will actually follow it: the last attempt a
        route is allowed never sleeps for a retry that will not be made. A wait
        that cannot fit inside what is left of the deadline is not made either —
        see :func:`_back_off` — and the route is then treated as used up, so the
        invocation either falls back or reports the failure it already has.
        """
        plan = _select_routes(routes, request)
        deadline = Deadline.after(self._attempt_timeout)
        state = InvocationState()
        failure: BaseException | None = None

        for position, route in enumerate((plan.primary, *plan.fallbacks)):
            for retry in range(self._retry_count + 1):
                if _budget_is_spent(deadline.budget_for(self._attempt_timeout)):
                    raise _deadline_exceeded(request, state) from failure
                state = state.with_attempt(on_fallback=position > 0)
                try:
                    response = await self._attempt(route, request, state, deadline)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failure = error
                    remaining = deadline.remaining()
                    if (
                        retry < self._retry_count
                        and is_retryable(error, remaining=remaining)
                        and await _back_off(retry + 1, self._retry_backoff_ms, deadline)
                    ):
                        continue
                    if not is_fallback_eligible(error, remaining=remaining):
                        raise
                    break
                return _normalise(route, request, response, state)

        if failure is None:
            raise _deadline_exceeded(request, state)
        raise failure

    async def _attempt(
        self,
        route: Route,
        request: InvocationRequest,
        state: InvocationState,
        deadline: Deadline,
    ) -> InvocationResponse:
        """One provider call against one route, inside one concurrency slot.

        The slot is held for this call and released when it returns or raises, so
        the number of attempts in flight never exceeds the configured maximum,
        and a cancelled attempt releases its slot like any other exit.

        Waiting for that slot is bounded by what is left of the invocation's
        deadline: the remaining time is handed to the limiter, which raises
        :class:`~model_gateway.errors.DeadlineExceededError` rather than letting
        an invocation that cannot be served before its deadline queue behind
        others indefinitely. Under a saturated provider that wait is what would
        otherwise keep an already-doomed invocation resident, still holding its
        payload and still ahead of work that could be served.

        The budget handed to the provider is derived here, *after* the slot is
        held, from the invocation's one deadline: waiting for a slot consumes
        invocation budget too, so the budget is computed at the moment the
        attempt starts rather than before the wait (PRD section 15). An attempt
        whose budget is already spent is not made — the provider is not called
        and a :class:`~model_gateway.errors.DeadlineExceededError` is raised
        instead.
        """
        async with self._limiter.slot(deadline.remaining()):
            budget = deadline.budget_for(self._attempt_timeout)
            if _budget_is_spent(budget):
                raise _deadline_exceeded(request, state)
            context = AttemptContext(
                invocation_id=request.invocation_id,
                attempt_number=state.attempt_count,
                timeout=budget,
            )
            return await self._provider.complete(route, request, context)


def _require_count(name: str, value: int) -> None:
    """Refuse a retry count or backoff that is not a non-negative integer.

    A negative count would mean fewer than one attempt per route, and a boolean
    is an ``int`` in Python but is not a count of anything. Both are refused at
    construction rather than turning into a scheduling surprise on the first
    request.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _budget_is_spent(budget: float | None) -> bool:
    """Whether an attempt budget is exhausted, so no attempt may be started with it.

    ``None`` is the gateway's own "no budget" marker — the deployment configured
    no timeout, so no deadline was anchored and no attempt is handed a ceiling
    of its own — and is never spent. Zero is: a deadline that has run out hands
    back zero, and an attempt given zero seconds is one the gateway has already
    refused (PRD section 15).
    """
    return budget is not None and budget <= 0.0


async def _back_off(attempt: int, base_ms: float, deadline: Deadline) -> bool:
    """Wait the delay that follows a failed attempt, when there is one to wait.

    ``attempt`` is the 1-based number of the attempt that just failed **on this
    route**, and the delay is
    :func:`~model_gateway.reliability.backoff_seconds`'s exponential schedule,
    bounded by what is left of ``deadline`` so a wait can never outlive the
    invocation's budget. The schedule is per route, not per invocation: a
    fallback's first retry backs off one base, exactly as the primary route's
    first retry did, rather than inheriting exponential growth from the attempts
    an earlier route already spent. A delay of zero waits nothing — the retry is
    still made, it simply does not sleep first. The sleep happens outside any
    concurrency slot, so a caller in backoff occupies no upstream capacity.

    Returns whether a retry still follows the wait. It does not when the delay
    would consume the whole remaining budget: ``backoff_seconds`` refuses that
    schedule with a :class:`~model_gateway.errors.DeadlineExceededError` rather
    than clamping it, because sleeping until the deadline expired would spend the
    rest of the invocation's budget on a retry that could never run. That
    refusal ends this route's retries, not the invocation: it is reported to the
    caller of this function as ``False`` so that
    :meth:`InferenceService._invoke_on` treats the route as used up — falling
    back if another route is eligible and the failure says so, and otherwise
    reporting the attempt's own failure, which is still the more informative
    answer while the invocation deadline itself has not expired.
    """
    try:
        delay = backoff_seconds(attempt, base_ms, remaining=deadline.remaining())
    except DeadlineExceededError:
        return False
    if delay > 0:
        await asyncio.sleep(delay)
    return True


def _deadline_exceeded(request: InvocationRequest, state: InvocationState) -> DeadlineExceededError:
    """The refusal for an invocation whose budget is gone (PRD section 15).

    The message and the details are built from gateway vocabulary and name only
    the public model and the number of gateway attempts made, so no upstream
    endpoint, model or credential can travel through them.
    """
    return DeadlineExceededError(
        f"the invocation deadline for model {request.model!r} expired before a provider "
        f"could serve it",
        details={"model": request.model, "attempts": str(state.attempt_count)},
    )


@dataclass(frozen=True)
class _RoutePlan:
    """Which routes may serve one invocation, decided before any call is made.

    ``primary`` is the route the public model advertises — the first configured
    route — and the one every invocation is attempted on. ``fallbacks`` are the
    later routes that could serve the same request unchanged, in configured
    order: the candidates a genuine failure fallback may walk (PRD section 11.2).

    Empty ``fallbacks`` is not a refusal. It says only that the invocation has
    nowhere to go if the primary path fails, which is the ordinary case for a
    model with one configured route.
    """

    primary: Route
    fallbacks: tuple[Route, ...]


def _select_routes(routes: tuple[Route, ...], request: InvocationRequest) -> _RoutePlan:
    """The primary route that must serve ``request``, and the fallbacks that may.

    Capability enforcement happens here, in this order, because the registry
    advertises the *primary* route's capabilities as the public model's own
    (:class:`~model_gateway.registry.ModelInfo`). The primary route is therefore
    validated first: a request it cannot honour is refused with a
    :class:`~model_gateway.errors.CapabilityError` even when a later route
    declares what the request needs, because serving it would answer a request
    the public model never advertised (PRD sections 4.3 and 9.5).

    Only then are the remaining routes filtered, each by the same test, into the
    fallbacks. A route that does not declare everything the request needs is
    skipped rather than attempted, so a capability gap in a fallback cannot make
    an invocation fatal that the primary route could have served (PRD section
    11.2); and because the request was already validated against the primary,
    filtering afterwards is what keeps a fallback from expanding the advertised
    capabilities. A fallback may preserve them, never add to them.

    This is a decision about eligibility alone, not the walk itself: it retries
    nothing and abandons no route for having failed. The walk that consumes
    ``fallbacks`` is :meth:`InferenceService._invoke_on`'s, which attempts them
    in this order when the routes before them cannot serve the invocation.

    Raises :class:`~model_gateway.errors.CapabilityError` when the primary route
    cannot serve the request.
    """
    if not routes:
        raise _no_eligible_route(request, None)
    primary = routes[0]
    missing = _first_missing_capability(primary, request)
    if missing is not None:
        raise _no_eligible_route(request, missing)
    return _RoutePlan(
        primary=primary,
        fallbacks=tuple(
            route for route in routes[1:] if _first_missing_capability(route, request) is None
        ),
    )


def _first_missing_capability(route: Route, request: InvocationRequest) -> Capability | None:
    """The first capability ``route`` lacks for ``request``, or ``None`` if it has them all.

    The order is the order :func:`_required_capabilities` reports, so the
    capability a refusal names is deterministic.
    """
    for capability in _required_capabilities(request):
        if capability not in route.capabilities:
            return capability
    return None


def _no_eligible_route(
    request: InvocationRequest, missing: Capability | None
) -> CapabilityError:
    """The refusal for a request the advertised model cannot serve (PRD section 9.5).

    ``missing`` is the first capability the primary route lacks for this request,
    which is the capability the public model does not advertise. It is ``None``
    only when the operation has no routes at all, which the registry refuses
    before selection runs.

    A later route declaring the capability does not change the answer: the
    public contract is the primary route's, and a fallback may preserve it but
    never expand it.
    """
    if missing is None:
        return CapabilityError(
            f"no route is configured for {request.model!r}, so this invocation cannot be served"
        )
    return CapabilityError(
        f"model {request.model!r} does not advertise {missing.value!r}: its primary route does "
        f"not declare that capability, so this invocation cannot be served unchanged",
        details={"capability": missing.value},
    )


def _required_capabilities(request: InvocationRequest) -> tuple[Capability, ...]:
    """The capabilities a text invocation asks of the route that serves it.

    A chat invocation needs the chat capability; declaring tools needs the tools
    capability, because a provider that cannot call tools cannot be made to; an
    image in the conversation needs the vision capability, because dropping it
    would change what the caller said (PRD sections 9.4 and 9.5); and a
    generation config carrying a non-text response format needs the
    structured-output capability, because a route that cannot honour the schema
    would otherwise answer with free text while the caller believed its
    requirement had been met (PRD section 9.5). A text-mode response format asks
    for nothing beyond chat: it is the provider's own default.

    A system instruction asks for nothing beyond chat. It is baseline chat
    semantics that every chat binding must map or refuse (PRD sections 4.4, 9.1
    and 9.5), so a route declaring chat already claims to handle it, and a
    capability of its own would move that obligation somewhere it could be left
    unset (:class:`~model_gateway.capabilities.Capability`).

    The first missing capability is reported, in this order, so the refusal is
    deterministic.
    """
    required: list[Capability] = [Capability.CHAT]
    if request.tools:
        required.append(Capability.TOOLS)
    if _carries_images(request):
        required.append(Capability.VISION)
    if _requires_structured_output(request):
        required.append(Capability.STRUCTURED_OUTPUT)
    return tuple(required)


def _requires_structured_output(request: InvocationRequest) -> bool:
    """Whether the generation config asks for an output shape a provider must honour."""
    response_format = request.generation_config.response_format
    return response_format is not None and response_format.mode is not ResponseFormatMode.TEXT


def _carries_images(request: InvocationRequest) -> bool:
    """Whether any message in the conversation carries an image part."""
    return any(
        isinstance(part, ImagePart)
        for message in request.messages
        for part in message.content
    )


def _normalise(
    route: Route,
    request: InvocationRequest,
    response: InvocationResponse,
    state: InvocationState,
) -> InvocationResponse:
    """Stamp the gateway's own facts onto a provider's answer.

    A provider's response is upstream data, never forwarded unchanged (PRD
    section 9). The gateway chose the route, so it states which public model was
    asked for and which upstream model answered; and the gateway made the
    attempts, so ``attempt_count`` and ``fallback_used`` are read from ``state``
    rather than assumed. A provider cannot report how many times the gateway
    called, and neither may this function invent the number.

    ``invocation_id`` is always the gateway's own, never the provider's. The
    gateway mints the identifier before the call, and PRD section 5.2 describes
    the response field as the gateway invocation identifier, so a value a
    provider supplies is discarded: a buggy or hostile provider must not be able
    to rename the invocation in the caller's own record of it. A provider's own
    request or response identifier is a different fact and must not be published
    here; it belongs in a field of its own — an upstream request ID reported
    beside ``upstream_model`` — and until such a field exists it is not reported
    at all, rather than passed off as the invocation's identity.
    """
    return replace(
        response,
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=request.invocation_id,
        attempt_count=state.attempt_count,
        fallback_used=state.fallback_used,
    )
