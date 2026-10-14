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
  primitives of :mod:`model_gateway.reliability`. :func:`_select_routes_for` decides
  the primary route and the fallbacks that could serve the request;
  :meth:`InferenceService._walk` anchors a single
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

The provider performs one gateway attempt and owns protocol translation and
execution and nothing else (:mod:`model_gateway.providers.base`); one attempt
may carry more than one wire request, because a provider's library can retry
internally (PRD section 11.3).

**Cancellation propagates.** :exc:`asyncio.CancelledError` is a
:exc:`BaseException` and is never caught here, so a cancelled invocation stops
at the attempt in flight, releases the concurrency slot that attempt held, and
makes no further attempt (PRD section 15).

**Embeddings are served under the same policy.** :meth:`InferenceService.embed`
is the embeddings operation's entry point, and it is the same walk: the public
model is resolved to the routes configured for embeddings, the primary route's
capability contract is enforced, one deadline covers the invocation, and every
attempt is made inside a concurrency slot whose wait is bounded by what is left
of that deadline. The differences are the capability the route must declare
(``embeddings``), the answer the operation produces, and the route properties
that gate the request's ``dimensions``: a stated dimension is honoured only on
a route that accepts the parameter, where it is forwarded and becomes the
expected output length, and is refused with a capability error naming it
anywhere else — a fallback that cannot honour the same request is not a
candidate, so a fallback may preserve the advertised contract but never expand
it. Two checks belong to this layer because they read facts the provider does
not own: the vectors are **ordered against the caller's input**, and their
length is **checked against the dimensionality the served route declares**,
so an upstream answer that cannot be matched to the request is refused rather
than passed on as a wrong-shaped answer — a malformed-response error, refused
on the route that answered wrongly and walked to an eligible fallback (PRD
sections 4.7, 6.3, 9.5 and 11.2).

**``attempt_count`` counts gateway attempts, never upstream requests.** The
gateway counts the calls it makes. The inference library has internal retry
paths no supported knob disables (PRD section 11.3), so one gateway attempt can
carry more than one wire request and this count may understate what the upstream
saw. It must never be documented, reported or recorded as a wire-request count.
The embeddings path makes the same claim and no stronger one: it does not offer
exactly-once upstream execution either.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TypeVar
from uuid import uuid4

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Route
from model_gateway.domain import (
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    ResponseFormatMode,
)
from model_gateway.domain.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    GatewayError,
    MalformedResponseError,
    normalise_exception,
)
from model_gateway.providers.base import (
    AttemptContext,
    EmbeddingProvider,
    StreamingTextProvider,
    TextProvider,
    _is_usable_budget,
)
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import (
    ConcurrencyLimiter,
    Deadline,
    backoff_seconds,
    is_fallback_eligible,
    is_retryable,
)

__all__ = ["InferenceService", "InvocationState"]

_STREAMING_CAPABILITY = "streaming"

_EMBEDDINGS_CAPABILITY = Capability.EMBEDDINGS.value
"""The capability an embeddings invocation asks of the route that serves it.

The refusal names it so a caller learns which operation is missing rather than
receiving a generic failure (PRD section 9.5).
"""

_FLOAT_ENCODING = "float"
"""The one wire encoding the gateway's embeddings answer is given in (PRD 4.7).

It is the upstream's own default as well, so a caller who asked for it — or
asked for nothing — is served by asking the provider for nothing extra.
"""

_ENCODING_CAPABILITY = "encoded_output"
"""The capability named when a caller asks for an encoding the gateway cannot answer in.

The gateway's embeddings response is the float form, so any other encoding a
canonical request can still carry is a behaviour this gateway does not have;
naming it keeps the refusal in the same vocabulary the chat endpoint's feature
refusals use (PRD section 9.5).
"""

_Result = TypeVar("_Result")
"""The answer one operation produces: a completion, a stream, or vectors."""

_Serve = Callable[[Route, "InvocationState", Deadline], Awaitable[_Result]]
"""One attempt on one route, finished: the call and the gateway's own stamping."""

_TERMINAL_EVENT_KINDS = frozenset({StreamEventKind.FINISH, StreamEventKind.ERROR})
"""The event kinds that end a stream, after which nothing may follow.

A stream reaches at most one of them: ``FINISH`` for an answer that completed,
``ERROR`` for one that failed after it had already produced output. They are the
gateway's own vocabulary for "this stream is over", so the walk in
:meth:`InferenceService._invoke_stream_on` reads them rather than trusting the
provider to stop at the right point: a provider that keeps yielding after a
terminal event, or that raises after one, must not be able to put anything —
least of all a second terminal event — after the answer the caller was already
told was over.
"""

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


class InferenceService:
    """Policy for one inference invocation: resolve, select, call, normalise.

    The service is built once per process and serves every operation the
    gateway exposes: :meth:`invoke` for a text answer, :meth:`invoke_stream`
    for a streamed one, and :meth:`embed` for vectors. The operations differ in
    what the provider is asked for and what comes back, not in how the gateway
    chooses a route, budgets the invocation, bounds concurrency or decides
    whether a failure is worth another attempt — that policy is written once,
    in :meth:`_walk`, and every operation runs through it.
    """

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

        ``provider`` is the binding the deployment chose, and it is typed as a
        :class:`~model_gateway.providers.base.TextProvider` because every
        operation the gateway serves needs one. Whether it *also* implements
        :class:`~model_gateway.providers.base.EmbeddingProvider` is asked at the
        moment an embeddings request arrives, not here: a provider that cannot
        embed is a perfectly usable text provider, and refusing to construct the
        service would disable chat over an operation the deployment never asks
        for. The refusal belongs to the request that needs the missing
        capability, and it names it (PRD sections 6 and 9.5).
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

    async def invoke_stream(self, request: InvocationRequest) -> AsyncIterator[StreamEvent]:
        """Invoke the public model named by ``request.model``, streaming the answer.

        The streaming counterpart of :meth:`invoke`, under the same policy: the
        ID is resolved to its configured routes, the request is validated against
        the primary route's capability contract, and the invocation runs under
        one deadline and one retry loop (PRD sections 11.2, 11.3 and 15). The
        difference is what a failure may do once the answer has begun.

        **The public model is resolved before streaming eligibility is asked.**
        A request naming a model the gateway does not expose is a
        :class:`~model_gateway.errors.ModelNotFoundError`, and it is refused as
        that whatever the bound provider can do: the request addresses a resource
        that does not exist, so whether some other model could have been streamed
        has no bearing on the answer. Asking the provider's capability first would
        report a streaming refusal for a model that is not even configured, which
        is a less accurate answer to a question the request already answered.

        **No fallback after the first output.** Before the first event reaches
        the caller, a retryable or fallback-eligible failure may still move to
        another route, exactly as :meth:`invoke` does — the caller has seen
        nothing, so nothing it was told can be contradicted. After the first
        event, a failure ends the stream instead: switching providers, or even
        retrying the same route, would concatenate a second answer onto the
        first, and a caller that has seen output cannot be given a contradictory
        one (PRD sections 11.2, 11.3 and 16). The stream then ends with exactly
        one terminal :class:`~model_gateway.domain.streaming.StreamEvent` of kind
        ``ERROR`` that reports the failure's own gateway reason and sanitised
        message, rather than stopping silently or mislabelling the failure as a
        finish.

        Raises :class:`~model_gateway.errors.ModelNotFoundError` for an unknown
        ID, :class:`~model_gateway.errors.CapabilityError` when the primary route
        cannot honour the request or the provider cannot stream, and otherwise
        the failure of the last attempt — surfaced the same way :meth:`invoke`
        surfaces it, except that a failure after the first output is reported
        through the terminal ``ERROR`` event and then raised. The event is
        authoritative and the exception is control-flow metadata: both describe
        the same single failure, so the exception must not be read as a second
        one.
        """
        routes = self._registry.routes_for(request.model)
        if not isinstance(self._provider, StreamingTextProvider):
            raise CapabilityError(
                "this provider cannot stream a chat completion; the gateway does not answer a "
                "streaming request with a non-streamed body, so the request is refused",
                details={"capability": _STREAMING_CAPABILITY},
            )
        stream = self._invoke_stream_on(request, routes)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed the texts named by ``request`` through the public model's routes.

        The embeddings counterpart of :meth:`invoke`, under the same policy: the
        public model ID is resolved to the routes configured for the embeddings
        operation, the request is validated against the primary route's
        capability contract — which for this operation means the ``embeddings``
        capability, and only that — and the invocation runs under one deadline
        and one retry loop (PRD sections 4.7, 6.3, 11.2, 11.3 and 15).

        **The model is resolved before the provider's capability is asked.** A
        request naming a model the gateway does not expose is a
        :class:`~model_gateway.errors.ModelNotFoundError`, and it is refused as
        that whatever the bound provider can do: the request addresses a
        resource that does not exist, so whether some other operation could have
        been served has no bearing on the answer (PRD section 10).

        **A provider that cannot embed is refused, never improvised.** The
        gateway does not answer an embeddings request with a vector it invented
        or with an empty one, because either would tell the caller its request
        was served when it was not. The refusal is a
        :class:`~model_gateway.errors.CapabilityError` naming ``embeddings``
        (PRD section 9.5).

        **A caller who asked for a wire encoding the gateway cannot produce is
        refused too.** The canonical answer is the float form (PRD section 4.7),
        so ``encoding_format`` of ``float`` — or none at all, which is the
        upstream's own default — is honoured, and ``base64`` is refused with a
        capability error naming the encoding rather than answered in floats
        (PRD section 9.5).

        **A caller dimension is honoured only where the route accepts one.** A
        route declares whether it accepts a caller-selectable dimensionality
        (``EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM``, or a
        ``supports_dimensions_param`` entry). Where it does, the caller's
        ``dimensions`` is forwarded to the upstream and becomes the expected
        output length. Where it does not, the route's dimensionality is fixed at
        the one it declares, and a request that states a dimension is refused
        with a capability error naming the parameter before any provider is
        called — the parameter is neither forwarded against the route's
        configuration nor silently dropped (PRD sections 6.3 and 9.5).

        **The answer is checked against the request before it is returned.** Each
        vector is placed at the position the upstream reported, so ``data`` is in
        the caller's input order whatever order the upstream used, and the
        position set must be exactly the caller's indices: an upstream that
        returned fewer, more or duplicated vectors has not answered the request,
        and the invocation is refused rather than answered with a vector matched
        to the wrong text (PRD section 4.7). Every vector's length must equal the
        dimensionality the *served route* declares — read per route, not from one
        operation-wide setting — or the one the caller asked for, when the route
        accepts the parameter and the caller asked for one, so a wrong-shaped
        vector is refused rather than returned (PRD sections 4.7 and 6.3). A
        refusal of this kind is a
        :class:`~model_gateway.errors.MalformedResponseError`: the fault lies in
        that upstream's answer, so the invocation falls back to another eligible
        route rather than retrying the route that answered wrongly.

        The response states the public model that was asked for, the upstream
        model that actually served it, and the invocation identifier the gateway
        minted for this call. It carries no attempt count or fallback flag: the
        canonical embeddings response has no field for them, and the gateway does
        not publish a number the caller's contract does not define.

        Raises :class:`~model_gateway.errors.ModelNotFoundError` for an unknown
        ID, :class:`~model_gateway.errors.CapabilityError` when the primary route
        cannot serve embeddings, does not accept a stated caller dimension, or
        the provider cannot embed, and otherwise the failure of the last
        attempt — surfaced exactly as :meth:`invoke` surfaces it, including the
        retry and fallback decisions, which are made from the gateway's own
        taxonomy and never from a library exception type.

        ``attempt_count`` is not reported here, but the same limit applies to
        what this method does: it counts the calls the *gateway* makes, and the
        inference library's internal retry paths mean one of those calls can
        carry more than one upstream request (PRD section 11.3). The embeddings
        path therefore makes no exactly-once claim either.
        """
        routes = self._registry.routes_for(request.model)
        if not isinstance(self._provider, EmbeddingProvider):
            raise CapabilityError(
                "this provider cannot produce embeddings; the gateway does not answer an "
                "embeddings request with a vector it fabricated, so the request is refused",
                details={"capability": _EMBEDDINGS_CAPABILITY},
            )
        _reject_unhonourable_encoding(request)
        plan = _select_routes_for(
            routes,
            request.model,
            (Capability.EMBEDDINGS,),
            caller_dimension=request.dimensions is not None,
        )
        _reject_unsupported_dimension(request, plan.primary)
        invocation_id = uuid4().hex

        async def serve(
            route: Route, state: InvocationState, deadline: Deadline
        ) -> EmbeddingResponse:
            response = await self._embed_attempt(route, request, state, deadline, invocation_id)
            return _normalise_embeddings(route, request, response, invocation_id=invocation_id)

        return await self._walk(request.model, plan, serve)

    async def _invoke_on(
        self, request: InvocationRequest, routes: tuple[Route, ...]
    ) -> InvocationResponse:
        """Serve one text invocation from the configured route order, primary first.

        The text operation's entry into the shared walk. Selection is the text
        operation's own — :func:`_select_routes` measures the request against the
        primary route's advertised capabilities and refuses what the public model
        never advertised (PRD sections 4.3 and 9.5) — and everything after it is
        :meth:`_walk`'s: one deadline, the same-route retry test, the separate
        fallback test, the bounded backoff and the last attempt's failure
        reported.

        What this method contributes to the walk is the answer shape. The
        provider's completion is stamped with the gateway's own facts by
        :func:`_normalise`, which is where ``attempt_count`` and
        ``fallback_used`` come from rather than being assumed (PRD section 5.2).
        """
        return await self._walk(
            request.model,
            _select_routes(routes, request),
            self._text_serve(request),
        )

    async def _walk(
        self,
        model: str,
        plan: _RoutePlan,
        serve: _Serve[_Result],
    ) -> _Result:
        """Walk one operation's route plan under one deadline and one retry loop.

        This is the policy the non-streaming operations share — the text
        invocation's (PRD section 11.2) and the embeddings one's (PRD section
        4.7) — written once so the two cannot drift apart. The streaming walk is
        deliberately not this one: it carries the extra rule that no fallback and
        no retry may follow the first delivered event (PRD sections 11.2 and 16),
        which is a rule about output the caller has already seen rather than a
        different reliability policy.

        ``model`` is the public model ID the request named, carried so a refusal
        can name the resource it was about without the request itself; ``plan``
        is which routes may serve it; ``serve`` performs one attempt on one route
        and returns the finished result, the place each operation's own answer
        shape is handled.

        Each failure is asked two separate questions, because the answers differ.

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
        deadline = Deadline.after(self._attempt_timeout)
        state = InvocationState()
        failure: BaseException | None = None

        for position, route in enumerate((plan.primary, *plan.fallbacks)):
            for retry in range(self._retry_count + 1):
                if _budget_is_spent(deadline.budget_for(self._attempt_timeout)):
                    raise _deadline_exceeded(model, state) from failure
                state = state.with_attempt(on_fallback=position > 0)
                try:
                    return await serve(route, state, deadline)
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

        if failure is None:
            raise _deadline_exceeded(model, state)
        raise failure

    def _text_serve(
        self, request: InvocationRequest
    ) -> _Serve[InvocationResponse]:
        """Bind one text invocation's attempt to its answer shape.

        The walk performs the call and nothing else; this closure is where the
        text operation's own answer is produced — the single provider call plus
        the gateway's own stamping of the result (PRD section 5.2).
        """

        async def serve(
            route: Route, state: InvocationState, deadline: Deadline
        ) -> InvocationResponse:
            response = await self._attempt(route, request, state, deadline)
            return _normalise(route, request, response, state)

        return serve

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
        async with self._attempt_context(
            request.model, request.invocation_id, state, deadline
        ) as context:
            return await self._provider.complete(route, request, context)

    async def _embed_attempt(
        self,
        route: Route,
        request: EmbeddingRequest,
        state: InvocationState,
        deadline: Deadline,
        invocation_id: str,
    ) -> EmbeddingResponse:
        """One embeddings call against one route, inside one concurrency slot.

        The embeddings counterpart of :meth:`_attempt`, and the same single
        attempt under the same controls: the slot is held for the duration of the
        call and released when it ends — success, failure or cancellation — and
        the budget handed to the provider is derived from the invocation's one
        deadline after the slot is held, so waiting for a slot consumes
        invocation budget too (PRD section 15).

        The provider is handed the route, the request and an
        :class:`~model_gateway.providers.base.AttemptContext` carrying the
        attempt's own facts, exactly as :meth:`_attempt` hands them to
        :meth:`~model_gateway.providers.base.TextProvider.complete`. The
        invocation identifier is the one the gateway minted for this invocation,
        which the canonical embeddings request does not carry and the canonical
        response must state (PRD section 5.2).
        """
        async with self._attempt_context(
            request.model, invocation_id, state, deadline
        ) as context:
            return await self._provider.embed(route, request, context)

    @asynccontextmanager
    async def _attempt_context(
        self,
        model: str,
        invocation_id: str | None,
        state: InvocationState,
        deadline: Deadline,
    ) -> AsyncIterator[AttemptContext]:
        """Hold one concurrency slot and hand back this attempt's own facts.

        The slot is held for the duration of the attempt and released when it
        returns or raises, so the number of attempts in flight never exceeds the
        configured maximum, and a cancelled attempt releases its slot like any
        other exit.

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

        ``model`` is the public model ID the invocation named, carried so the
        refusal can name the resource it was about; ``invocation_id`` is the
        gateway's own identifier for the invocation, which a text request carries
        and an embeddings request does not (PRD section 5.2).
        """
        async with self._limiter.slot(deadline.remaining()):
            budget = deadline.budget_for(self._attempt_timeout)
            if _budget_is_spent(budget):
                raise _deadline_exceeded(model, state)
            yield AttemptContext(
                invocation_id=invocation_id,
                attempt_number=state.attempt_count,
                timeout=budget,
            )

    async def _invoke_stream_on(
        self, request: InvocationRequest, routes: tuple[Route, ...]
    ) -> AsyncIterator[StreamEvent]:
        """Serve one streaming invocation from the configured route order.

        The streaming counterpart of :meth:`_invoke_on`, and the same walk:
        the primary route first, then the fallbacks that could serve the
        request, under the invocation's single deadline, with the same two
        questions asked of every failure. The difference is the rule PRD
        sections 11.2 and 11.3 state and a non-streaming invocation never had
        to enforce: **once an event has been yielded to the caller, no fallback
        and no retry may occur.**

        Before the first event, the walk is exactly :meth:`_invoke_on`'s: a
        retryable failure is retried on the same route within the configured
        count, a fallback-eligible failure moves to the next route, and a
        failure that answers no to both terminates the invocation where it
        happened. The caller has seen nothing, so moving to another provider
        contradicts nothing.

        After the first event, a failure ends the stream. The events already
        yielded stay — they are what the upstream produced — and the stream is
        closed with exactly one terminal ``ERROR`` event that carries the
        failure's own gateway reason and sanitised message, rather than stopping
        silently, restarting on another provider, or mislabelling the failure as
        a finish (PRD section 16). The error is then raised, so the caller knows
        the invocation failed; the terminal event is what the stream itself
        shows.

        **A terminal event is the last thing a stream ever contains.** The walk
        stops the moment one is yielded: the generator returns immediately after
        handing a ``FINISH`` or an ``ERROR`` to the caller, so no event — not
        even a second terminal one — can follow it. That holds even when the
        provider misbehaves: a provider that yields more events after its finish,
        or raises after it, cannot put anything after the terminal event the
        caller was already shown. A failure that raises after a terminal event
        has been produced is re-raised unchanged rather than turned into a
        second terminal event.

        Sequence numbers are strictly increasing across the whole stream. A
        provider numbers its own events from 0, and a route is only retried or
        walked away from before it has produced any event, so the numbers a
        caller sees never repeat or go backwards. The terminal event after a
        failure takes the next number after the last one yielded.

        A provider stream that ends without producing a terminal event — the
        upstream went away, or simply never reported a finish reason — ends the
        generator without one too. The gateway synthesises no terminal event
        for an answer it never saw finish, so a caller whose stream stops reads
        a stream that is over without a terminal event, not a failed one (PRD
        section 16).
        """
        plan = _select_routes(routes, request)
        deadline = Deadline.after(self._attempt_timeout)
        state = InvocationState()
        failure: BaseException | None = None
        output_started = False
        terminal_emitted = False
        last_sequence = -1

        for position, route in enumerate((plan.primary, *plan.fallbacks)):
            for retry in range(self._retry_count + 1):
                if _budget_is_spent(deadline.budget_for(self._attempt_timeout)):
                    raise _deadline_exceeded(request, state) from failure
                state = state.with_attempt(on_fallback=position > 0)
                attempt_events = self._stream_attempt(route, request, state, deadline)
                try:
                    async for event in attempt_events:
                        last_sequence = event.sequence
                        output_started = True
                        yield event
                        if event.kind in _TERMINAL_EVENT_KINDS:
                            terminal_emitted = True
                            return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failure = error
                    if terminal_emitted:
                        # A terminal event already ended the stream, so this
                        # failure adds nothing to it — re-raising it as the
                        # second terminal event would contradict the one the
                        # caller was already shown.
                        raise
                    if output_started:
                        yield _error_event(request, error, sequence=last_sequence + 1)
                        terminal_emitted = True
                        raise
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
                finally:
                    await attempt_events.aclose()
                return

        if failure is None:
            raise _deadline_exceeded(request, state)
        raise failure

    async def _stream_attempt(
        self,
        route: Route,
        request: InvocationRequest,
        state: InvocationState,
        deadline: Deadline,
    ) -> AsyncIterator[StreamEvent]:
        """One provider stream against one route, inside one concurrency slot.

        The streaming counterpart of :meth:`_attempt`: the slot is held for the
        duration of the stream and released when it ends — success, failure or
        cancellation — and the budget handed to the provider is derived from
        the invocation's one deadline after the slot is held, so waiting for a
        slot consumes invocation budget too (PRD section 15).

        The provider is handed the route, the request and an
        :class:`~model_gateway.providers.base.AttemptContext` carrying the
        attempt's own facts, exactly as :meth:`_attempt` hands them to
        :meth:`~model_gateway.providers.base.TextProvider.complete`. The
        provider performs one gateway attempt and translates each upstream
        chunk; it does not retry and does not choose between routes.
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
            upstream = self._provider.stream(route, request, context)
            try:
                async for event in upstream:
                    yield event
            finally:
                await upstream.aclose()


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
    :meth:`InferenceService._walk` treats the route as used up — falling
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


def _deadline_exceeded(model: str, state: InvocationState) -> DeadlineExceededError:
    """The refusal for an invocation whose budget is gone (PRD section 15).

    The message and the details are built from gateway vocabulary and name only
    the public model and the number of gateway attempts made, so no upstream
    endpoint, model or credential can travel through them.
    """
    return DeadlineExceededError(
        f"the invocation deadline for model {model!r} expired before a provider "
        f"could serve it",
        details={"model": model, "attempts": str(state.attempt_count)},
    )


def _error_event(
    request: InvocationRequest, error: BaseException, *, sequence: int
) -> StreamEvent:
    """The terminal event that ends a stream which failed after output began.

    A failure after the first event cannot be retried or walked to another
    route — the caller has already seen output, and a second answer would
    contradict it (PRD sections 11.2 and 11.3). The stream therefore ends with
    exactly one terminal ``ERROR`` event, so the caller is not left with a
    stream that simply stopped, and is not told the answer finished when it did
    not.

    The event carries the gateway's own stable reason and its sanitised message,
    never a finish reason. A finish reason classifies *why generation stopped*,
    and there is none: the provider failed rather than finishing, so borrowing
    ``FinishReason.UNKNOWN`` and parking the gateway's reason in
    ``raw_finish_reason`` would describe a finish that never happened and put
    gateway vocabulary in a field reserved for the provider's own (PRD sections
    5.2, 5.4 and 9.1). ``error_reason`` is the value of the failure's
    :class:`~model_gateway.errors.ErrorReason`, which is what a caller switches
    on; ``error_message`` is the same sanitised message the error carries, so
    the reason and the text agree with the error that is raised alongside the
    event.

    The message is constrained to the taxonomy's own gateway-authored messages.
    It is never populated from arbitrary provider exception text with
    sanitisation as the only defence: a failure that is not already a
    :class:`~model_gateway.errors.GatewayError` is normalised first — here,
    before the event is built, exactly as the HTTP boundary normalises it — so
    the reason is one of the taxonomy's own and the message is gateway-authored
    rather than the provider's (PRD section 5.4). The event is built from the
    normalised error, never from the raw one.
    """
    gateway_error = error if isinstance(error, GatewayError) else normalise_exception(error)
    return StreamEvent(
        kind=StreamEventKind.ERROR,
        sequence=sequence,
        invocation_id=request.invocation_id or "",
        error_reason=gateway_error.reason.value,
        error_message=gateway_error.message,
    )


def _select_routes(routes: tuple[Route, ...], request: InvocationRequest) -> _RoutePlan:
    """The primary route that must serve a text ``request``, and the fallbacks that may.

    The text operation's capabilities are what the request is measured against;
    every other operation selects through :func:`_select_routes_for` with its own
    capability, so the enforcement below is stated once.

    Raises :class:`~model_gateway.errors.CapabilityError` when the primary route
    cannot serve the request.
    """
    return _select_routes_for(routes, request.model, _required_capabilities(request))


def _select_routes_for(
    routes: tuple[Route, ...],
    model: str,
    required: tuple[Capability, ...],
    *,
    caller_dimension: bool = False,
) -> _RoutePlan:
    """The primary route that must serve a request, and the fallbacks that may.

    ``required`` is the capability set the operation asks of the route that
    serves it: the text operation's own list for a chat invocation
    (:func:`_required_capabilities`), and the single ``embeddings`` capability
    for an embeddings one. ``model`` is the public model ID the request named,
    carried so a refusal names the resource it was about. ``caller_dimension``
    is whether an embeddings request states a caller dimension, which only a
    route that accepts the ``dimensions`` parameter can honour unchanged.

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

    The primary is measured by the capability test alone, deliberately: a
    primary that lacks the ``dimensions`` parameter is refused by the caller's
    own dimension check — :func:`_reject_unsupported_dimension`, which names the
    parameter — rather than by a generic capability refusal here. The
    ``caller_dimension`` filter applies to fallbacks only, where skipping a
    route that cannot honour the request unchanged is the whole of the rule.

    This is a decision about eligibility alone, not the walk itself: it retries
    nothing and abandons no route for having failed. The walk that consumes
    ``fallbacks`` is :meth:`InferenceService._walk`'s, which attempts them in
    this order when the routes before them cannot serve the invocation.
    """
    if not routes:
        raise _no_eligible_route(model, None)
    primary = routes[0]
    missing = _first_missing_capability(primary, required)
    if missing is not None:
        raise _no_eligible_route(model, missing)
    return _RoutePlan(
        primary=primary,
        fallbacks=tuple(
            route for route in routes[1:] if _route_can_serve(route, required, caller_dimension)
        ),
    )


def _route_can_serve(
    route: Route, required: tuple[Capability, ...], caller_dimension: bool
) -> bool:
    """Whether ``route`` could serve this request unchanged.

    The capability test first, exactly as the primary is validated: a route
    that does not declare an operation's capability cannot serve it (PRD
    section 9.5). The dimension test applies only when the caller stated a
    dimension: a route that does not accept the ``dimensions`` parameter
    cannot honour a request that asks for one, so it is not a candidate. The
    primary route establishes whether caller-selectable dimensions are part of
    the model's contract, and a fallback that cannot honour the same request is
    not one — a fallback may preserve the advertised contract, never expand it
    (PRD sections 9.5 and 11.2).
    """
    if _first_missing_capability(route, required) is not None:
        return False
    return not caller_dimension or route.supports_dimensions_param


def _first_missing_capability(
    route: Route, required: tuple[Capability, ...]
) -> Capability | None:
    """The first capability ``route`` lacks, or ``None`` if it has them all.

    ``required`` is reported in a deterministic order by whoever built it, so the
    capability a refusal names is deterministic.
    """
    for capability in required:
        if capability not in route.capabilities:
            return capability
    return None


def _no_eligible_route(model: str, missing: Capability | None) -> CapabilityError:
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
            f"no route is configured for {model!r}, so this invocation cannot be served"
        )
    return CapabilityError(
        f"model {model!r} does not advertise {missing.value!r}: its primary route does "
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


def _reject_unhonourable_encoding(request: EmbeddingRequest) -> None:
    """Refuse a wire encoding the gateway's embeddings answer cannot be given in.

    The canonical embeddings answer is the float form (PRD section 4.7), and so
    is the OpenAI-compatible response the endpoint renders. ``float`` — and no
    stated encoding at all, which is the upstream's own default — is therefore
    honoured without asking the provider for anything else, and anything the
    canonical request can still carry but the gateway cannot answer in, today
    ``base64``, is refused here with a capability error naming the encoding.

    The refusal is deliberate rather than a gap. Answering a caller who asked
    for base64 with floats would tell it its request was honoured while the
    gateway silently ignored the encoding it asked for, which PRD section 9.5
    forbids; and the gateway cannot return base64 instead, because the canonical
    vector is a tuple of numbers and no field of the response carries an encoded
    form. Refusing is the only answer that keeps the caller's view of the
    request and the gateway's view of it identical (PRD section 4).
    """
    encoding = request.encoding_format
    if encoding is None or encoding == _FLOAT_ENCODING:
        return
    raise CapabilityError(
        f"the gateway cannot answer an embeddings request in the {encoding!r} encoding: its "
        f"embeddings response is the float form, so the request is refused rather than "
        f"answered in an encoding the caller did not ask for",
        details={"capability": _ENCODING_CAPABILITY, "parameter": "encoding_format"},
    )


def _reject_unsupported_dimension(request: EmbeddingRequest, primary: Route) -> None:
    """Refuse a caller dimension the primary route's contract does not carry.

    A route declares whether it accepts a caller-selectable dimensionality
    (``EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM``, or a ``supports_dimensions_param``
    entry). Where it does not, the route's dimensionality is fixed at the one
    it declares, so a stated dimension is a parameter the route cannot honour.
    It is refused here — before any provider is called, before the walk begins
    — rather than forwarded against the route's configuration or silently
    dropped, and the refusal names the parameter so a caller learns which one
    the route does not accept (PRD sections 6.3 and 9.5).

    The primary route establishes whether caller-selectable dimensions are part
    of the model's advertised contract, and no later route can expand it
    (PRD section 9.5): a fallback that cannot honour the same request is not a
    candidate, which :func:`_route_can_serve` enforces the other way round.
    """
    if request.dimensions is None or primary.supports_dimensions_param:
        return
    raise CapabilityError(
        f"model {request.model!r} does not accept a 'dimensions' parameter: its primary route "
        f"does not declare supports_dimensions_param, so the request is refused rather than "
        f"forwarded against the route's configuration or silently dropped",
        details={"capability": "dimensions", "parameter": "dimensions"},
    )


def _normalise_embeddings(
    route: Route,
    request: EmbeddingRequest,
    response: EmbeddingResponse,
    *,
    invocation_id: str,
) -> EmbeddingResponse:
    """Order the vectors against the caller's input and check their shape.

    A provider's embeddings answer is upstream data, never forwarded unchanged
    (PRD section 9), and the two facts a provider cannot state are the gateway's
    own: which public model was asked for and which upstream model answered, and
    which invocation this answer belongs to.

    The vectors are placed at the positions the upstream reported, so ``data`` is
    in the caller's input order whatever order the upstream used, and each
    vector's own ``index`` is the position it was placed at. PRD section 4.7
    requires the gateway to preserve input ordering, and preserving it means more
    than passing an array through: the position of a vector has to be the
    position of the text it embeds, or the caller matches a vector to the wrong
    document and cannot tell that it did. An answer whose positions are not
    exactly the caller's — too few vectors, too many, a duplicate position, a
    position outside the input — cannot be ordered at all, so the invocation is
    refused rather than answered with a partial or misaligned result.

    Every vector's length is then checked against the dimensionality the
    *route being served* declares — ``route.dimensions``, the value
    ``EMBEDDINGS_DIMENSIONS`` gave the block's primary route, or a
    ``dimensions`` entry gave a fallback — or, when the caller asked for one
    and the route accepts the parameter, the dimensionality the caller asked
    for, because that is what the route was asked to honour. A route that does
    not accept the parameter is checked against its own declared dimensionality
    because that is the only length it can produce, and a request that states a
    dimension to such a route is refused before it reaches this check
    (:func:`_reject_unsupported_dimension`). ``None`` means nothing is declared
    and nothing is checked: an unconfigured dimensionality is not a
    dimensionality of zero (PRD sections 4.3 and 6.3). A vector of the wrong
    length is refused rather than returned, because a wrong-shaped vector is a
    wrong answer that a consumer of fixed-width vectors would accept and then
    misuse (PRD section 4.7).

    Both refusals are :class:`~model_gateway.errors.MalformedResponseError`.
    They are properties of what *this upstream* returned, not of the caller's
    request: the request was valid and the route was reachable, and the answer
    it produced was structurally wrong. That is precisely the shape of failure
    another eligible route can fix, so the invocation walks to the next route
    while the route that just answered wrongly is not worth retrying immediately
    (PRD section 11.2). A gateway defect, by contrast, is
    :class:`~model_gateway.errors.InternalError` and terminates the invocation
    where it happened.
    """
    expected_dimensions = (
        request.dimensions
        if request.dimensions is not None and route.supports_dimensions_param
        else route.dimensions
    )
    return replace(
        response,
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=invocation_id,
        data=_ordered_vectors(request.input, response.data, expected=expected_dimensions),
    )


def _ordered_vectors(
    inputs: tuple[str, ...],
    data: tuple[Embedding, ...],
    *,
    expected: int | None,
) -> tuple[Embedding, ...]:
    """``data`` in the caller's input order, checked for completeness and shape."""
    by_index: dict[int, Embedding] = {}
    for embedding in data:
        if embedding.index in by_index:
            raise MalformedResponseError(
                f"the upstream returned more than one embedding for input position "
                f"{embedding.index}; the caller's inputs cannot be matched to a single vector "
                f"each, so the answer is refused rather than misaligned"
            )
        by_index[embedding.index] = embedding

    wanted = set(range(len(inputs)))
    if set(by_index) != wanted:
        raise MalformedResponseError(
            f"the upstream returned {len(by_index)} embeddings for {len(inputs)} inputs; the "
            f"answer must carry exactly one vector per input, in the caller's order, so it is "
            f"refused rather than matched to the wrong text"
        )

    ordered: list[Embedding] = []
    for index in range(len(inputs)):
        vector = by_index[index]
        if expected is not None and len(vector.values) != expected:
            raise MalformedResponseError(
                f"the upstream returned an embedding of {len(vector.values)} dimensions where "
                f"{expected} were expected; a vector of the wrong length is refused rather than "
                f"returned as the caller's answer"
            )
        ordered.append(vector)
    return tuple(ordered)
