"""The inference service: gateway-owned policy around one provider call.

The service owns policy; the provider owns translation. That split is the
boundary, and this module is the policy side of it:

* **Routing.** The public model ID in the request is resolved to the routes
  configured for its operation by the registry. An unknown ID is a
  :class:`~model_gateway.errors.ModelNotFoundError` raised here, before any
  provider is called, so a caller-supplied model name cannot reach an
    unconfigured upstream endpoint.
* **Capability enforcement.** The request is validated against the primary
  route's capability contract before anything is selected. The registry
    advertises the primary route's capabilities as the public model's own, so
  a request the primary cannot honour is refused with a
  :class:`~model_gateway.errors.CapabilityError` even when a later route declares
    what the request needs: serving it would answer a request the
  public contract never advertised for that model. Silently dropping a tool, an
  image or a structured-output requirement is forbidden, and so is obtaining one
  through a fallback. Only then are the remaining routes filtered, each by the
  same test, into the fallbacks that could serve the request unchanged, so a
  fallback may preserve the advertised capabilities but can never expand them.
* **Attempt execution.** The provider is handed one resolved route, the request
  and an :class:`~model_gateway.providers.base.AttemptContext` carrying the
  attempt's own facts: the invocation it belongs to, which attempt it is and the
    budget that attempt may use. The budget is the gateway's, so
  a provider never reads a deadline out of configuration it does not own, and the
  core never wraps the call in a timeout the provider cannot see.
* **Normalisation.** The provider's answer comes back as upstream data and is
  re-normalised at this boundary: the public model that was asked for, the
  upstream model that actually served it and the invocation's own reliability
  facts are the gateway's to state, because the gateway chose the route and made
    the attempts.
* **Reliability.** One invocation has one deadline, one retry loop and one
    attempt count, and the loop is built from the
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

**An idempotency key is answered from the gateway's own record.** A
non-streaming invocation may carry one, and a repeat of the same request is
answered from the record of the first answer rather than by asking the provider
again — the provider is called once and every equivalent repeat is served from
the record, which is what prevents duplicate accounting. The
record is keyed by the caller's key and matched against a fingerprint of the
request, so a key reused for a different body is a second request and is served
as one rather than answered from the wrong answer; the gateway-minted invocation
identifier is not part of the fingerprint, so a retry carrying a freshly minted
one is still the same request. Only a completed answer is recorded, so a retry
after a transient failure is allowed to retry and the gateway claims no
exactly-once upstream execution. The record is consulted after the model is
resolved and before any provider is called, so an unknown model is still a
model-not-found error even when the key has a cached record. Streaming is
deliberately excluded: a replayed stream would present an answer that was never
produced token by token in this call as a stream, which the gateway does not do.

The provider performs one gateway attempt and owns protocol translation and
execution and nothing else (:mod:`model_gateway.providers.base`); one attempt
may carry more than one wire request, because a provider's library can retry
internally.

**Cancellation propagates.** :exc:`asyncio.CancelledError` is a
:exc:`BaseException` and is never caught here, so a cancelled invocation stops
at the attempt in flight, releases the concurrency slot that attempt held, and
makes no further attempt.

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
on the route that answered wrongly and walked to an eligible fallback.

**``attempt_count`` counts gateway attempts, never upstream requests.** The
gateway counts the calls it makes. The inference library has internal retry
paths no supported knob disables, so one gateway attempt can
carry more than one wire request and this count may understate what the upstream
saw. It must never be documented, reported or recorded as a wire-request count.
The embeddings path makes the same claim and no stronger one: it does not offer
exactly-once upstream execution either.

**Images are served under the same policy.** :meth:`InferenceService.generate`
is the image operation's entry point, and it is the same walk: the public model
is resolved to the routes configured for the image operation, the provider's
capability is asked before anything is attempted, the primary route's
capability contract is enforced — which for this operation means the
``generation`` capability, and only that — and one deadline covers the
invocation, with every attempt made inside a concurrency slot whose wait is
bounded by what is left of that deadline. The answer is checked against the
request before it is returned:
a caller that stated ``n`` is answered with exactly that many images, and an
upstream that returned another number has not answered the request, so the
invocation is refused rather than answered with the wrong number of images —
a :class:`~model_gateway.errors.MalformedResponseError`, refused on the route
that answered wrongly and walked to an eligible fallback. The image path makes
the same no-exactly-once claim as the others: the gateway counts its own
attempts, and the inference library's internal retry paths mean one of those
calls can carry more than one upstream request.

**Image editing is absent from the surface.** The gateway's canonical image
request carries no field for an image input, so an edit cannot be forwarded
to any route however its capabilities read, and an edit answered without it
would report an image edited that never left the gateway.
The service therefore has no edit entry point at all: an operation the
gateway categorically cannot perform is left out rather than published and
guaranteed to fail, and a configured route that declares the ``editing``
capability is refused at startup by the application factory.
"""

import asyncio
import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from typing import TypeVar
from uuid import uuid4

import structlog
from opentelemetry.trace import Tracer

from model_gateway.accounting import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    AccountingRecorder,
    InvocationOutcome,
)
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Route
from model_gateway.domain import (
    IdempotencyStatus,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    ResponseFormatMode,
    Usage,
)
from model_gateway.domain.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from model_gateway.domain.images import ImageRequest, ImageResponse
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    GatewayError,
    MalformedResponseError,
    ResourceExhaustedError,
    normalise_exception,
)
from model_gateway.idempotency import (
    IdempotencyStore,
    MemoryIdempotencyStore,
    request_fingerprint,
    settle,
)
from model_gateway.limits import RateLimiter, TenantConcurrencyLimiter
from model_gateway.metrics import Metrics
from model_gateway.providers.base import (
    AttemptContext,
    EmbeddingProvider,
    ImageProvider,
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
receiving a generic failure.
"""

_FLOAT_ENCODING = "float"
"""The one wire encoding the gateway's embeddings answer is given in.

It is the upstream's own default as well, so a caller who asked for it — or
asked for nothing — is served by asking the provider for nothing extra.
"""

_ENCODING_CAPABILITY = "encoded_output"
"""The capability named when a caller asks for an encoding the gateway cannot answer in.

The gateway's embeddings response is the float form, so any other encoding a
canonical request can still carry is a behaviour this gateway does not have;
naming it keeps the refusal in the same vocabulary the chat endpoint's feature
refusals use.
"""

_IMAGE_CAPABILITY = Capability.GENERATION.value
"""The capability an image invocation asks of the route that serves it.

Image generation is the image operation's own capability, the default every
image route declares; the refusal names it so a caller learns which operation
is missing rather than receiving a generic failure.
"""

_Result = TypeVar("_Result")
"""The answer one operation produces: a completion, a stream, or vectors."""

_Serve = Callable[[Route, "InvocationState", Deadline], Awaitable[tuple[_Result, Route]]]
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

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_CONCURRENT_REQUESTS = 100
"""The concurrency bound a service is built with when no limiter is injected.

It is ``PROVIDER_MAX_CONCURRENT_REQUESTS``'s own default, so a
service constructed without a deployment's configuration still bounds its
simultaneous upstream attempts rather than leaving them unbounded. The
application factory passes the configured value, or a limiter of its own.
"""


@dataclass(frozen=True)
class InvocationState:
    """What the gateway has done for one invocation so far.

    The response reports facts about the gateway's own execution — how many
    provider attempts were made and whether a fallback path was taken — and those
    are facts about the invocation rather than about any one answer a provider
    gave. They live here so that ``_normalise`` states them instead of inventing
    them: a hardcoded ``attempt_count`` of one is a claim about the reliability
    loop, which is exactly what this object can answer truthfully.

    ``attempt_count`` starts at zero, because a state exists before the first
    call, and counts the attempts that were made — including ones that raised,
    because a failed attempt is work the gateway did. It counts
    *gateway* attempts: one call this service made, on one route, under one
    attempt number. It is not a wire-request count. The inference library has
    internal retry paths that no supported knob switches off,
    so a single attempt here can carry more than one upstream request, and this
    count may understate what the upstream saw. Reporting it as the number of
    upstream requests would be a claim the gateway cannot support.

    ``fallback_used`` records that the invocation took a *fallback path*: a route
    other than the primary one served it because the normal primary path could
    not. It is deliberately narrower than "a route other than
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
        of an attempt than a success.

        ``on_fallback`` states that this attempt is made on a route other than
        the primary one, after the primary path could not serve the invocation.
        That is the one thing that makes ``fallback_used`` true,
        and it is sticky, because one such route is enough for the
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
    order: the candidates a genuine failure fallback may walk.

    Empty ``fallbacks`` is not a refusal. It says only that the invocation has
    nowhere to go if the primary path fails, which is the ordinary case for a
    model with one configured route.
    """

    primary: Route
    fallbacks: tuple[Route, ...]


@dataclass
class _StreamFacts:
    """What a streaming invocation has produced so far, filled in by its walk.

    A generator cannot return the facts an invocation's accounting needs, so the
    streaming walk writes them here as it runs and the service reads them once the
    stream is over. This is the streaming counterpart of the answer a
    non-streaming walk returns, and it exists so that a streamed invocation
    reaches the same record path as a completed one rather than a parallel one.

    ``route`` is unset until the first event is handed to the caller, and it is
    how the service tells a stream that produced an answer from one that produced
    nothing: a refusal before the first event leaves it ``None``, and there is
    then no answer to account for. ``state`` is the walk state at that same
    moment, which is where the attempt count and the fallback flag come from.

    **Any event counts, not only one carrying answer content.** A usage report is
    the provider stating it did work, so a stream that reached the provider is one
    the gateway owes for whatever it carried; a stream that emitted nothing at all
    is the only one with no answer to account for. That is deliberately the same
    criterion the walk already applies to decide that no further fallback or retry
    may occur, so the rule that ends the walk and the
    rule that decides whether there is something to record cannot drift apart.

    ``usage`` is the last token report the provider sent, or ``None`` when it sent
    none — the canonical stream carries usage on events of its own rather than on
    the answer, so a provider that reports nothing leaves no counts rather than
    invented ones. ``status`` and ``error_reason`` are set when the stream reaches
    a terminal event, and left unset otherwise; the service reads an unset status
    as ``incomplete``, which is what a stream that produced output and then
    stopped without a terminal event is.
    """

    route: Route | None = None
    state: InvocationState = field(default_factory=InvocationState)
    usage: Usage | None = None
    status: str | None = None
    error_reason: str | None = None


def _counted_invocation(method):
    """Count one invocation's arrival, its duration and how it ended.

    A decorator rather than a wrapper written into each of the four operations,
    because the four differ in what they serve and the counting is identical for
    all of them. The scope is the whole operation, so a refusal from a gateway
    limit is counted like any other outcome — it is an invocation the gateway
    declined to serve, which is what the error count is for.

    Defined before the class because a decorator is looked up when the class body
    is executed, not when the method is called.
    """

    @functools.wraps(method)
    async def counted(self, request, **kwargs):
        async with self._counted():
            return await method(self, request, **kwargs)

    return counted


def _counted_stream(method):
    """The streaming counterpart of :func:`_counted_invocation`.

    The scope is the stream's whole life rather than the call that produced the
    generator. An async generator does nothing until it is iterated, so a
    coroutine-shaped decorator would count a request that had not begun and finish
    counting before it had.
    """

    @functools.wraps(method)
    async def counted(self, request, **kwargs):
        async with self._counted():
            inner = method(self, request, **kwargs)
            try:
                async for event in inner:
                    yield event
            finally:
                # The inner generator is closed explicitly rather than left to the
                # event loop's finalization: it is what records the stream and
                # releases its tenant slot, and a consumer that stops early must
                # have that happen now rather than whenever the loop gets to it.
                await inner.aclose()

    return counted


def _traced_invocation(name: str):
    """A decorator factory: one span around an operation, named for it.

    A decorator rather than a wrapper written into each operation, for the reason
    :func:`_counted_invocation` gives. The name is a parameter because a span name
    is what a trace backend groups by, and four operations sharing one would make a
    trace unreadable.
    """

    def decorate(method):
        @functools.wraps(method)
        async def traced(self, request, **kwargs):
            with self._span(name, request):
                return await method(self, request, **kwargs)

        return traced

    return decorate


def _traced_stream(name: str):
    """The streaming counterpart of :func:`_traced_invocation`.

    The span covers the stream's whole life, as the counting does: an invocation is
    in flight while its answer is arriving, and a span that ended at the first
    event would report a fraction of the work it stands for.
    """

    def decorate(method):
        @functools.wraps(method)
        async def traced(self, request, **kwargs):
            with self._span(name, request):
                inner = method(self, request, **kwargs)
                try:
                    async for event in inner:
                        yield event
                finally:
                    await inner.aclose()

        return traced

    return decorate


class InferenceService:
    """Policy for one inference invocation: resolve, select, call, normalise.

    The service is built once per process and serves every operation the
    gateway exposes: :meth:`invoke` for a text answer, :meth:`invoke_stream`
    for a streamed one, :meth:`embed` for vectors, and :meth:`generate` for
    images. The operations differ in what the provider is
    asked for and what comes back, not in how the gateway chooses a route,
    budgets the invocation, bounds concurrency or decides whether a failure
    is worth another attempt — that policy is written once, in :meth:`_walk`,
    and every operation runs through it.
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
        idempotency_store: IdempotencyStore | None = None,
        accounting: AccountingRecorder | None = None,
        accounting_enabled: bool = True,
        rate_limiter: RateLimiter | None = None,
        tenant_limiter: TenantConcurrencyLimiter | None = None,
        metrics: Metrics | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Wire the service to the registry, the provider and the reliability policy.

        ``attempt_timeout`` is the one number the deployment configures today —
        ``PROVIDER_TIMEOUT_SECONDS`` — and it plays two roles at once: it is the
        invocation's total budget *and* the ceiling on each attempt inside it
        Those are logically distinct bounds and this
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
        capability, and it names it. Image generation
        is asked the same way, through
        :class:`~model_gateway.providers.base.ImageProvider`, when an image
        request arrives: a provider that cannot generate is a usable text and
        embeddings provider, and an image request it cannot serve is refused
        rather than handed to it.

        ``idempotency_store`` is where the answers to keyed invocations are
        recorded, so a repeat of the same request is answered from the record
        rather than by asking the provider again. It defaults to
        an in-memory store, which is sufficient for a single-instance deployment;
        a deployment that needs its records to survive a restart or to be shared
        between instances configures a store of its own and passes it here. A
        store that fails is a gateway defect and surfaces as one: it is never
        swallowed into a re-run, because silently asking the provider again is
        exactly the duplicate accounting the key exists to prevent.

        ``rate_limiter`` and ``tenant_limiter`` are the two admission decisions of
        the gateway, and both default to ``None``, which is a deployment that
        configured neither. They are taken at each operation's entry, before the
        model is resolved and before any provider is called, and that placement is
        load-bearing rather than tidy: the walk treats a ``RATE_LIMITED`` failure
        as route-local and therefore fallback-eligible, because
        an *upstream's* rate limit belongs to the route that answered it. The
        gateway's own limit belongs to the caller and to no route, so raising it
        inside the walk would make the gateway walk a request from one provider to
        the next trying to escape a limit that follows the caller, spending every
        configured provider's capacity on the way.

        ``rate_limiter`` bounds a caller's request rate and ``tenant_limiter``
        bounds the invocations one tenant may have in flight at once. The two are
        keyed differently on purpose: the rate is per *caller*, because one
        credential's burst must not consume what the others are entitled to, while
        the concurrency bound is per *tenant*, because it exists to share the
        gateway's upstream pool fairly between tenants and a single global bound
        cannot do that without knowing who is asking.

        ``metrics`` is the registry the invocation's own facts are counted into, or
        ``None`` for a deployment that publishes none. The counts are taken at this
        boundary rather than in the transports because both transports reach the
        same operations here: an HTTP request and a gRPC call for one invocation
        would otherwise be counted twice, or counted differently by each.

        ``tracer`` is where the invocation's span goes, or ``None`` for a deployment
        that does not trace. It is a tracer rather than a provider because the
        provider is the process's and the tracer names this component within it.
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
        self._idempotency_store = (
            idempotency_store if idempotency_store is not None else MemoryIdempotencyStore()
        )
        self._accounting = accounting
        self._accounting_enabled = accounting_enabled
        self._rate_limiter = rate_limiter
        self._tenant_limiter = tenant_limiter
        self._metrics = metrics
        self._tracer = tracer

    @contextmanager
    def _span(self, name: str, request: object) -> Iterator[object]:
        """One invocation's span, or nothing when the deployment does not trace.

        The public model is recorded as an attribute because it is the one fact a
        caller of a trace needs to group spans by, and it is the gateway's own
        vocabulary rather than anything the caller supplied as content — the same
        distinction the log redaction draws.
        """
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            model = getattr(request, "model", None)
            if model is not None:
                span.set_attribute("gateway.model", str(model))
            yield span

    @asynccontextmanager
    async def _counted(self) -> AsyncIterator[None]:
        """Count one invocation's arrival, its duration and how it ended.

        Taken around the whole operation so the duration is the invocation's, not
        one attempt's, and so a refusal from a gateway limit is counted as an error
        like any other — it is an invocation the gateway declined to serve, which is
        exactly what the error count is for. A deployment that publishes no metrics
        pays nothing: the wrapper is one branch and no clock reading.

        The duration is observed on every exit, including a cancellation, because
        an invocation that was cut short still occupied the gateway for as long as
        it ran and the active count would drift if it were not released.
        """
        metrics = self._metrics
        if metrics is None:
            yield
            return
        started = time.monotonic()
        metrics.requests_total.inc()
        metrics.active_requests.inc()
        try:
            yield
        except GatewayError:
            metrics.errors_total.inc()
            raise
        finally:
            metrics.active_requests.dec()
            metrics.request_duration_seconds.observe(time.monotonic() - started)

    def _count_limit_rejection(self) -> None:
        """Count a request the gateway's own limit refused.

        Called where the refusal is raised rather than where the error is caught,
        because this is the only place that can tell the gateway's own refusal from
        an upstream's: both arrive as a resource-exhausted error, and only one of
        them is the gateway declining rather than the provider reporting.
        """
        if self._metrics is not None:
            self._metrics.rate_limit_rejections_total.inc()

    def _count_retry(self) -> None:
        """Count one attempt made after a retryable failure on the same route."""
        if self._metrics is not None:
            self._metrics.retries_total.inc()

    def _count_answer(self, outcome: InvocationOutcome, state: InvocationState) -> None:
        """Count what one completed answer contributed: its tokens and its fallback.

        Called before the accounting branch rather than inside it, because a
        deployment that records no usage still wants the counts: the two settings
        answer different questions, and turning the record off does not turn the
        metrics off.

        A count the provider never reported is not added. The metrics are sums of
        what the gateway knows, and adding a zero for an unknown count would make
        the total claim that nothing was spent where the truth is that nothing was
        reported — the same distinction the record preserves.
        """
        metrics = self._metrics
        if metrics is None:
            return
        if state.fallback_used:
            metrics.fallbacks_total.inc()
        usage = outcome.usage
        if usage.input_tokens is not None:
            metrics.tokens_input_total.inc(usage.input_tokens)
        if usage.output_tokens is not None:
            metrics.tokens_output_total.inc(usage.output_tokens)

    def _admit(self, namespace: str, tenant: str | None) -> None:
        """Take the admission decisions that can be answered without waiting.

        The request rate is checked here, at the operation's entry, and a caller
        over its rate is refused before the model is resolved and before a tenant
        slot is asked for — the cheapest decision first, and the one a caller that
        cannot be served at all should learn about rather than queue behind.

        The rate limit is keyed on ``namespace``, which is the credential
        identifier the rest of the gateway already keys policy on, and never on
        anything a request carries.
        """
        if self._rate_limiter is None:
            return
        try:
            self._rate_limiter.check(namespace)
        except ResourceExhaustedError:
            self._count_limit_rejection()
            raise

    def _tenant_slot(self, namespace: str, tenant: str | None, deadline: Deadline):
        """The per-tenant concurrency slot an invocation holds for its whole duration.

        Keyed on the tenant the credential is bound to, or on the credential
        identifier when it is bound to none. A tenant-less credential is its own
        principal, so falling back to the identifier keeps one
        of them from consuming the bound the others are entitled to — which a
        shared empty key would do.

        The wait for a slot is bounded by what is left of the invocation's
        deadline, and that deadline is anchored before admission rather than after
        it. One budget therefore covers the whole invocation — the queue wait and
        everything the walk then does with it — instead of the wait being bounded
        by one full timeout and the walk then starting a fresh budget of the same
        length, which would let an invocation take twice the configured time from
        the caller's point of view.

        A deployment that configured no bound gets a slot that holds nothing,
        rather than a ``None`` each call site has to remember to test.
        """
        if self._tenant_limiter is None:
            return _unbounded_slot()
        return self._counted_slot(tenant or namespace, deadline.remaining())

    @asynccontextmanager
    async def _counted_slot(self, principal: str, timeout: float | None) -> AsyncIterator[None]:
        """Take a tenant slot, counting a refusal as the gateway's own.

        The counting is here rather than at the entry point because this is the
        only place that can tell the gateway's concurrency refusal from an
        upstream's rate limit: both arrive as a resource-exhausted error. Whether
        the slot was ever entered is what separates them — a refusal from the
        limiter is raised before the body runs, so the flag is still false.
        """
        entered = False
        try:
            async with self._tenant_limiter.slot(principal, timeout):  # type: ignore[union-attr]
                entered = True
                yield
        except ResourceExhaustedError:
            if not entered:
                self._count_limit_rejection()
            raise

    @_counted_invocation
    @_traced_invocation("invoke")
    async def invoke(
        self,
        request: InvocationRequest,
        *,
        namespace: str | None = None,
        tenant: str | None = None,
    ) -> InvocationResponse:
        """Invoke the public model named by ``request.model``.

        Resolves the ID to its configured routes, validates the request against
        the primary route's capability contract — the contract the registry
        advertises for the model — then serves the invocation under one deadline
        and one retry loop. Raises
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
        than one request inside a single attempt.

        **An idempotency key is answered from the gateway's own record.** A
        request carrying one is deduplicated here rather than upstream: a repeat
        whose fingerprint matches a recorded answer is answered from that record
        with :attr:`~model_gateway.domain.IdempotencyStatus.REPLAYED`, and the
        provider is not called again — one provider call, and every equivalent
        repeat served from the record, which is what prevents duplicate
        accounting. The key is namespaced by the trusted
        principal, so two credentials using the same key are two operations and
        cannot overwrite each other's records.

        The coordination is atomic, because simultaneous identical requests are
        the situation idempotency exists to handle: the first caller claims the
        key, and every later caller with the same fingerprint waits on that
        claim's outcome rather than starting a second invocation. A waiter
        going away does not cancel the invocation it is waiting on. A repeat
        whose fingerprint differs is refused with an
        :class:`~model_gateway.errors.IdempotencyConflictError`, because a key
        names one operation and executing the second would both run a request
        the caller never made and overwrite the record the key already holds.

        Only a completed answer is recorded: a failure propagates, releases the
        claim so a later caller may retry, and gives every waiter that failure
        rather than starting another provider attempt inside the same request. The
        gateway therefore claims no exactly-once upstream execution. The claim
        is consulted after the model is resolved and before any provider is
        called, so an unknown model is a
        :class:`~model_gateway.errors.ModelNotFoundError` even when the key has
        a cached record. A request without a key is unaffected by all of this:
        it is never recorded and reports ``NOT_APPLICABLE``.
        """
        routes = self._registry.routes_for(request.model)
        # One deadline is anchored here, before admission, so the wait for a
        # tenant slot draws from the same budget the walk then spends rather
        # than from a full timeout of its own followed by a fresh one.
        deadline = Deadline.after(self._attempt_timeout)
        started = time.monotonic()
        self._admit(namespace or "", tenant)
        async with self._tenant_slot(namespace or "", tenant, deadline):
            return await self._invoke_admitted(
                request, routes, namespace or "", started, deadline
            )

    async def _invoke_admitted(
        self,
        request: InvocationRequest,
        routes: tuple[Route, ...],
        namespace: str,
        started: float,
        deadline: Deadline,
    ) -> InvocationResponse:
        """Serve an invocation that has been admitted, under the idempotency policy.

        Separate from :meth:`invoke` so that the admission decisions wrap exactly
        one body: an admitted invocation holds its tenant slot for everything that
        follows — including the wait on another caller's idempotency claim, which is
        work this invocation is resident for — and the serving code does not have to
        be re-indented to say so.

        ``deadline`` is the one anchored before admission, so the time the queue
        wait consumed is time the walk no longer has.
        """
        key = request.idempotency_key
        if key is None:
            response, route, state = await self._invoke_on(request, routes, deadline)
            # The answer and its record carry one identifier, minted here when the
            # request did not already carry one: the record must name the
            # invocation the caller was told about, not a second one.
            answer = replace(response, invocation_id=request.invocation_id or uuid4().hex)
            self._record(request, _outcome_of(answer), route, state, namespace, started)
            return answer

        fingerprint = request_fingerprint(request)
        claim = await self._idempotency_store.acquire(key, fingerprint, namespace)
        if claim.kind == "completed":
            return replace(claim.record.response, idempotency_status=IdempotencyStatus.REPLAYED)
        if claim.kind == "waiter":
            # A waiter did not initiate provider execution, and the status names
            # this request's handling rather than the invocation's. The owner's
            # answer is what it is handed, but the stamp is its own: telling a
            # coalesced caller "new" would report that its request reached the
            # provider, which is the accounting fact the field exists to carry.
            response = await settle(claim.future)
            # A coalesced waiter did not reach the provider and did not add to
            # the execution the owner performed, so it is not accounted: the
            # the owner's record is the one record for that execution.
            return replace(response, idempotency_status=IdempotencyStatus.REPLAYED)

        try:
            response, route, state = await self._invoke_on(request, routes, deadline)
        except BaseException as error:
            await self._idempotency_store.fail(key, namespace, error)
            raise
        recorded = replace(
            response,
            idempotency_status=IdempotencyStatus.NEW,
            invocation_id=request.invocation_id or uuid4().hex,
        )
        await self._idempotency_store.complete(key, namespace, recorded)
        self._record(request, _outcome_of(recorded), route, state, namespace, started)
        return recorded

    def _record(
        self,
        request: InvocationRequest | EmbeddingRequest | ImageRequest,
        outcome: InvocationOutcome,
        route: Route,
        state: InvocationState,
        namespace: str,
        started: float,
    ) -> None:
        """Record one invocation's answer and cost, when accounting is enabled.

        Every operation records, because the accounting contract asks for a record
        of every invocation that produced an answer and not only of a text one, and the
        walk knows for all of them what the record needs. ``outcome`` is the
        answer's own facts, and stating them rather than reading them off a
        response is what lets a streamed answer be recorded through this same
        path: a stream has no response object, so it states the same facts
        directly. ``state`` carries the two facts only the walk knows — how many
        attempts the gateway made and whether a fallback served the invocation —
        and it is passed rather than read off the answer because the canonical
        embeddings and image answers have no field for either
        while the record has one for both.

        Only an answer is recorded, and only when the caller asked for it: a
        deployment that disables accounting is not silently accounting. A failure
        that produced no output propagates to the caller and leaves no record,
        because there is no answer to account for; a streamed answer that produced
        output and then failed is an answer, and is recorded as one.

        A recording failure must not fail the invocation. The answer has already
        been produced and the caller is waiting on it; a store the operator
        misconfigured is not a reason to withhold it, and swallowing the error
        would turn accounting into a correctness problem. The failure is logged
        for the operator instead.
        """
        self._count_answer(outcome, state)
        if self._accounting is None or not self._accounting_enabled:
            return
        try:
            self._accounting.record(
                request,
                outcome,
                route,
                namespace=namespace,
                latency_ms=(time.monotonic() - started) * 1000,
                attempts=state.attempt_count,
                fallback_used=state.fallback_used,
            )
        except Exception as error:
            logger.error(
                "usage record could not be stored",
                error_type=type(error).__name__,
                invocation_id=outcome.invocation_id,
            )

    def _record_stream(
        self,
        request: InvocationRequest,
        facts: "_StreamFacts",
        namespace: str,
        started: float,
    ) -> None:
        """Record a streamed invocation that produced output, when accounting is on.

        A stream that produced nothing leaves no record, because there is no
        answer to account for: a refusal before the first event reaches the caller
        as an ordinary error response and no provider produced anything.

        A stream that produced output and then ended without a terminal event is
        recorded as ``incomplete`` rather than ``failed``. The stream contract
        draws that distinction for the caller — a truncated stream is not a failed one — and
        a record that collapsed the two would report a failure the stream never
        declared.

        ``usage`` is the last report the provider sent, or an empty report when it
        sent none. The canonical stream carries usage on events of its own, so a
        provider that reports nothing leaves a record with no counts rather than
        with invented ones.

        ``upstream_model`` is the serving route's model ID. A stream reports no
        upstream model of its own — the canonical events carry content, usage and a
        finish reason and no model name — so the record names the route the
        gateway sent the request to, which is the same value the non-streaming
        path is stamped with. It is not an independently reported provider model
        and is not presented as one.

        This is one *attempt* to record, not a guarantee that a record was stored:
        a sink that fails is logged rather than propagated, so a broken store
        loses the record without failing the invocation, exactly as it does on the
        non-streaming paths.
        """
        if facts.route is None:
            return
        status = facts.status or STATUS_INCOMPLETE
        self._record(
            request,
            InvocationOutcome(
                invocation_id=request.invocation_id or "",
                upstream_model=facts.route.model_id,
                usage=facts.usage or Usage(),
                status=status,
                error_reason=facts.error_reason if status == STATUS_FAILED else None,
            ),
            facts.route,
            facts.state,
            namespace,
            started,
        )

    @_counted_stream
    @_traced_stream("invoke_stream")
    async def invoke_stream(
        self,
        request: InvocationRequest,
        *,
        namespace: str | None = None,
        tenant: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Invoke the public model named by ``request.model``, streaming the answer.

        The streaming counterpart of :meth:`invoke`, under the same policy: the
        ID is resolved to its configured routes, the request is validated against
        the primary route's capability contract, and the invocation runs under
        one deadline and one retry loop. The
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
        one. The stream then ends with exactly
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

        **Idempotency does not apply here, deliberately.** A replayed stream would
        be a synthetic stream of an answer that was never produced token by token
        in this call, which contradicts the gateway's standing rule that it never
        presents a non-streamed answer as streaming. A request that carries an
        idempotency key is therefore served as a stream like any other, and its
        ``idempotency_status`` stays ``NOT_APPLICABLE``.

        **A streamed answer is accounted for too.** A stream that produced output
        makes one record attempt, made when the stream is over — whether it
        finished, failed after output, or stopped without a terminal event — and a
        stream that produced nothing makes none, because there is no answer to
        account for. It is an attempt rather than a guarantee
        that a record was stored, because a sink that fails is logged rather than
        propagated, exactly as on the non-streaming paths. The attempt is made even
        when the caller goes away mid-stream: the provider was called and the
        gateway owes for the output it produced, and the record says
        ``incomplete`` rather than ``failed`` so it reports what happened rather
        than what did not. ``namespace`` is the trusted principal the invocation
        acted under, which the streaming request does not carry and the
        authenticated caller does.

        **The accounting attempt survives the teardown.** The stream is closed
        before the record is written, and that close is awaited, so a caller
        cancelled while it is in flight would otherwise take the record with it —
        the gateway would have called the provider and recorded nothing. The close
        therefore sits in an inner ``finally`` and the recording in an outer one,
        so the record is still written when the close is interrupted. Recording is
        synchronous, so it cannot itself be interrupted, and the cancellation then
        continues to propagate: nothing here swallows it.

        ``invocation_id`` is minted here when the caller supplied none, exactly as
        :meth:`invoke` mints it for an answer. The canonical stream event types its
        identifier as required text, so a request without one could not be served
        at all, and the record would name no invocation.

        **The tenant slot is held for the stream's whole life**, and the record is
        written before it is released: an invocation is in flight for as long as its
        answer is arriving, and it is still in flight at the moment its usage is
        accounted for. A stream refused a slot is refused before any event is
        produced, so the refusal reaches the caller as an ordinary error response
        rather than as a stream that failed.
        """
        routes = self._registry.routes_for(request.model)
        # Admission comes first, before the streaming capability is asked: the
        # caller's own limit is a fact about the caller and is the first thing to
        # say, and a caller over its rate should learn that rather than a fact
        # about the provider it happens to have asked.
        deadline = Deadline.after(self._attempt_timeout)
        self._admit(namespace or "", tenant)
        if not isinstance(self._provider, StreamingTextProvider):
            raise CapabilityError(
                "this provider cannot stream a chat completion; the gateway does not answer a "
                "streaming request with a non-streamed body, so the request is refused",
                details={"capability": _STREAMING_CAPABILITY},
            )
        if request.invocation_id is None:
            request = replace(request, invocation_id=uuid4().hex)
        started = time.monotonic()
        async with self._tenant_slot(namespace or "", tenant, deadline):
            facts = _StreamFacts()
            stream = self._invoke_stream_on(request, routes, facts, deadline=deadline)
            try:
                try:
                    async for event in stream:
                        yield event
                finally:
                    await stream.aclose()
            finally:
                self._record_stream(request, facts, namespace or "", started)

    @_counted_invocation
    @_traced_invocation("embed")
    async def embed(
        self,
        request: EmbeddingRequest,
        *,
        namespace: str | None = None,
        tenant: str | None = None,
    ) -> EmbeddingResponse:
        """Embed the texts named by ``request`` through the public model's routes.

        The embeddings counterpart of :meth:`invoke`, under the same policy: the
        public model ID is resolved to the routes configured for the embeddings
        operation, the request is validated against the primary route's
        capability contract — which for this operation means the ``embeddings``
        capability, and only that — and the invocation runs under one deadline
        and one retry loop.

        **The model is resolved before the provider's capability is asked.** A
        request naming a model the gateway does not expose is a
        :class:`~model_gateway.errors.ModelNotFoundError`, and it is refused as
        that whatever the bound provider can do: the request addresses a
        resource that does not exist, so whether some other operation could have
        been served has no bearing on the answer.

        **A provider that cannot embed is refused, never improvised.** The
        gateway does not answer an embeddings request with a vector it invented
        or with an empty one, because either would tell the caller its request
        was served when it was not. The refusal is a
        :class:`~model_gateway.errors.CapabilityError` naming ``embeddings``.

        **A caller who asked for a wire encoding the gateway cannot produce is
        refused too.** The canonical answer is the float form,
        so ``encoding_format`` of ``float`` — or none at all, which is the
        upstream's own default — is honoured, and ``base64`` is refused with a
        capability error naming the encoding rather than answered in floats.

        **A caller dimension is honoured only where the route accepts one.** A
        route declares whether it accepts a caller-selectable dimensionality
        (``EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM``, or a
        ``supports_dimensions_param`` entry). Where it does, the caller's
        ``dimensions`` is forwarded to the upstream and becomes the expected
        output length. Where it does not, the route's dimensionality is fixed at
        the one it declares, and a request that states a dimension is refused
        with a capability error naming the parameter before any provider is
        called — the parameter is neither forwarded against the route's
        configuration nor silently dropped.

        **The answer is checked against the request before it is returned.** Each
        vector is placed at the position the upstream reported, so ``data`` is in
        the caller's input order whatever order the upstream used, and the
        position set must be exactly the caller's indices: an upstream that
        returned fewer, more or duplicated vectors has not answered the request,
        and the invocation is refused rather than answered with a vector matched
        to the wrong text. Every vector's length must equal the
        dimensionality the *served route* declares — read per route, not from one
        operation-wide setting — or the one the caller asked for, when the route
        accepts the parameter and the caller asked for one, so a wrong-shaped
        vector is refused rather than returned. A
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
        carry more than one upstream request. The embeddings
        path therefore makes no exactly-once claim either.

        **Every completed invocation leaves a usage record.** A served answer is
        handed to the accounting sink with the same facts the text operation
        records — the serving route, the public model, the upstream model, the
        token counts and their provenance, the latency and the walk's attempt and
        fallback facts — under the same ``USAGE_TRACKING_ENABLED`` setting. The
        canonical embeddings response carries no attempt count, so that fact
        reaches the record from the walk rather than from the answer. ``namespace``
        is the trusted principal the invocation acted under, which the embeddings
        request does not carry and the authenticated caller
        does.
        """
        routes = self._registry.routes_for(request.model)
        # Admission comes first, before the provider's capability and the request's
        # own constraints are asked: the caller's limit is a fact about the caller
        # and is the first thing to say, and a caller over its rate should learn
        # that rather than a fact about the provider or the request it sent.
        deadline = Deadline.after(self._attempt_timeout)
        self._admit(namespace or "", tenant)
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
        started = time.monotonic()

        async with self._tenant_slot(namespace or "", tenant, deadline):
            invocation_id = uuid4().hex

            async def serve(
                route: Route, state: InvocationState, deadline: Deadline
            ) -> tuple[EmbeddingResponse, Route]:
                response = await self._embed_attempt(
                    route, request, state, deadline, invocation_id
                )
                return (
                    _normalise_embeddings(route, request, response, invocation_id=invocation_id),
                    route,
                )

            result, route, state = await self._walk(
                request.model, plan, serve, deadline=deadline
            )
            self._record(request, _outcome_of(result), route, state, namespace or "", started)
            return result

    @_counted_invocation
    @_traced_invocation("generate")
    async def generate(
        self,
        request: ImageRequest,
        *,
        namespace: str | None = None,
        tenant: str | None = None,
    ) -> ImageResponse:
        """Generate the images the prompt names through the public model's routes.

        The image operation's counterpart of :meth:`invoke`, under the same
        policy: the public model ID is resolved to the routes configured for
        the image operation, the request is validated against the primary
        route's capability contract — which for this operation means the
        ``generation`` capability, and only that — and the invocation runs
        under one deadline and one retry loop.

        **The model is resolved before the provider's capability is asked.** A
        request naming a model the gateway does not expose is a
        :class:`~model_gateway.errors.ModelNotFoundError`, and it is refused
        as that whatever the bound provider can do: the request addresses a
        resource that does not exist, so whether some other operation could
        have been served has no bearing on the answer.

        **A provider that cannot generate images is refused, never
        improvised.** The gateway does not answer an image request with an
        image it invented, and an empty result is not an answer either, so
        the refusal is a :class:`~model_gateway.errors.CapabilityError` naming
        ``generation``.

        **The answer is checked against the request before it is returned.** A
        caller that stated ``n`` is answered with exactly that many images; an
        upstream that returned another number has not answered the request,
        and the invocation is refused rather than answered with the wrong
        number of images. A caller who stated no count is
        checked against none: the upstream's own default applies, and the
        gateway cannot know what number that was. A refusal of this kind is a
        :class:`~model_gateway.errors.MalformedResponseError`: the fault lies
        in that upstream's answer, so the invocation walks to another eligible
        route rather than retrying the route that answered wrongly.

        The response states the public model that was asked for, the upstream
        model that actually served it, and the invocation identifier the
        gateway minted for this call. It carries no attempt count or fallback
        flag: the canonical image response has no field for them, and the
        gateway does not publish a number the caller's contract does not
        define.

        Raises :class:`~model_gateway.errors.ModelNotFoundError` for an
        unknown ID, :class:`~model_gateway.errors.CapabilityError` when the
        primary route cannot serve image generation or the provider cannot
        generate, and otherwise the failure of the last attempt — surfaced
        exactly as :meth:`invoke` surfaces it, including the retry and fallback
        decisions, which are made from the gateway's own taxonomy and never
        from a library exception type.

        ``attempt_count`` is not reported here, but the same limit applies:
        it counts the calls the *gateway* makes, and the inference library's
        internal retry paths mean one of those calls can carry more than one
        upstream request. The image path makes no
        exactly-once claim.

        **Every completed invocation leaves a usage record.** A served answer is
        handed to the accounting sink with the same facts the text operation
        records — the serving route, the public model, the upstream model, the
        token counts and their provenance, the latency and the walk's attempt and
        fallback facts — under the same ``USAGE_TRACKING_ENABLED`` setting. The
        canonical image response carries no attempt count, so that fact reaches
        the record from the walk rather than from the answer.
        ``namespace`` is the trusted principal the invocation acted under, which
        the image request does not carry and the authenticated caller does.
        """
        routes = self._registry.routes_for(request.model)
        # Admission comes first, before the provider's capability and the route's
        # are asked, for the reason :meth:`embed` gives.
        deadline = Deadline.after(self._attempt_timeout)
        self._admit(namespace or "", tenant)
        _require_image_provider(self._provider)
        plan = _select_routes_for(routes, request.model, (Capability.GENERATION,))
        started = time.monotonic()

        async with self._tenant_slot(namespace or "", tenant, deadline):
            invocation_id = uuid4().hex

            async def serve(
                route: Route, state: InvocationState, deadline: Deadline
            ) -> tuple[ImageResponse, Route]:
                response = await self._image_attempt(
                    route, request, state, deadline, invocation_id
                )
                return (
                    _normalise_images(route, request, response, invocation_id=invocation_id),
                    route,
                )

            result, route, state = await self._walk(
                request.model, plan, serve, deadline=deadline
            )
            self._record(request, _outcome_of(result), route, state, namespace or "", started)
            return result

    async def _invoke_on(
        self, request: InvocationRequest, routes: tuple[Route, ...], deadline: Deadline
    ) -> tuple[InvocationResponse, Route, InvocationState]:
        """Serve one text invocation from the configured route order, primary first.

        The text operation's entry into the shared walk. Selection is the text
        operation's own — :func:`_select_routes` measures the request against the
        primary route's advertised capabilities and refuses what the public model
        never advertised — and everything after it is
        :meth:`_walk`'s: one deadline, the same-route retry test, the separate
        fallback test, the bounded backoff and the last attempt's failure
        reported.

        What this method contributes to the walk is the answer shape. The
        provider's completion is stamped with the gateway's own facts by
        :func:`_normalise`, which is where ``attempt_count`` and
        ``fallback_used`` come from rather than being assumed.
        """
        result, answered_by, state = await self._walk(
            request.model,
            _select_routes(routes, request),
            self._text_serve(request),
            deadline=deadline,
        )
        return result, answered_by, state

    async def _walk(
        self,
        model: str,
        plan: _RoutePlan,
        serve: _Serve[_Result],
        *,
        deadline: Deadline | None = None,
    ) -> tuple[_Result, Route, InvocationState]:
        """Walk one operation's route plan under one deadline and one retry loop.

        This is the policy the non-streaming operations share — the text
        invocation's and the embeddings one's — written once so the two cannot
        drift apart. The streaming walk is
        deliberately not this one: it carries the extra rule that no fallback and
        no retry may follow the first delivered event,
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
        ``fallback_used`` true.

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
        one. An attempt is never started once that budget is
        spent: the invocation fails with a
        :class:`~model_gateway.errors.DeadlineExceededError` and no provider is
        called.

        When no route could serve the invocation, the failure raised is the last
        attempt's rather than the first: the most recent failure is the one that
        says why the invocation could not be served.

        The wait between attempts happens outside the concurrency slot, so a
        caller in backoff holds nothing another caller could be using, and it is
        waited only when a retry will actually follow it: the last attempt a
        route is allowed never sleeps for a retry that will not be made. A wait
        that cannot fit inside what is left of the deadline is not made either —
        see :func:`_back_off` — and the route is then treated as used up, so the
        invocation either falls back or reports the failure it already has.

        ``deadline`` is the invocation's one deadline, anchored by the caller when
        it admitted the invocation so that a queue wait draws from the same budget
        as the walk. ``None`` anchors one here, which is what a caller that took no
        admission decision wants and what the tests that drive the walk directly
        rely on.
        """
        if deadline is None:
            deadline = Deadline.after(self._attempt_timeout)
        state = InvocationState()
        failure: BaseException | None = None

        for position, route in enumerate((plan.primary, *plan.fallbacks)):
            for retry in range(self._retry_count + 1):
                if _budget_is_spent(deadline.budget_for(self._attempt_timeout)):
                    raise _deadline_exceeded(model, state) from failure
                state = state.with_attempt(on_fallback=position > 0)
                try:
                    result, answered_by = await serve(route, state, deadline)
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
                        self._count_retry()
                        continue
                    if not is_fallback_eligible(error, remaining=remaining):
                        raise
                    break
                return result, answered_by, state

        if failure is None:
            raise _deadline_exceeded(model, state)
        raise failure

    def _text_serve(
        self, request: InvocationRequest
    ) -> _Serve[InvocationResponse]:
        """Bind one text invocation's attempt to its answer shape.

        The walk performs the call and nothing else; this closure is where the
        text operation's own answer is produced — the single provider call plus
        the gateway's own stamping of the result.
        """

        async def serve(
            route: Route, state: InvocationState, deadline: Deadline
        ) -> tuple[InvocationResponse, Route]:
            response = await self._attempt(route, request, state, deadline)
            return _normalise(route, request, response, state), route

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
        attempt starts rather than before the wait. An attempt
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
        invocation budget too.

        The provider is handed the route, the request and an
        :class:`~model_gateway.providers.base.AttemptContext` carrying the
        attempt's own facts, exactly as :meth:`_attempt` hands them to
        :meth:`~model_gateway.providers.base.TextProvider.complete`. The
        invocation identifier is the one the gateway minted for this invocation,
        which the canonical embeddings request does not carry and the canonical
        response must state.
        """
        async with self._attempt_context(
            request.model, invocation_id, state, deadline
        ) as context:
            return await self._provider.embed(route, request, context)

    async def _image_attempt(
        self,
        route: Route,
        request: ImageRequest,
        state: InvocationState,
        deadline: Deadline,
        invocation_id: str,
    ) -> ImageResponse:
        """One image call against one route, inside one concurrency slot.

        The image counterpart of :meth:`_embed_attempt`, and the same single
        attempt under the same controls: the slot is held for the duration of
        the call and released when it ends — success, failure or cancellation —
        and the budget handed to the provider is derived from the invocation's
        one deadline after the slot is held, so waiting for a slot consumes
        invocation budget too.

        The provider is handed the route, the request and an
        :class:`~model_gateway.providers.base.AttemptContext` carrying the
        attempt's own facts, exactly as :meth:`_embed_attempt` hands them to
        :meth:`~model_gateway.providers.base.EmbeddingProvider.embed`. The
        invocation identifier is the one the gateway minted for this
        invocation, which the canonical image request does not carry and the
        canonical response must state.
        """
        async with self._attempt_context(
            request.model, invocation_id, state, deadline
        ) as context:
            return await self._provider.generate(route, request, context)

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
        attempt starts rather than before the wait. An attempt
        whose budget is already spent is not made — the provider is not called
        and a :class:`~model_gateway.errors.DeadlineExceededError` is raised
        instead.

        ``model`` is the public model ID the invocation named, carried so the
        refusal can name the resource it was about; ``invocation_id`` is the
        gateway's own identifier for the invocation, which a text request carries
        and an embeddings request does not.
        """
        async with self._limiter.slot(deadline.remaining()):
            budget = deadline.budget_for(self._attempt_timeout)
            if _budget_is_spent(budget):
                raise _deadline_exceeded(model, state)
            context = AttemptContext(
                invocation_id=invocation_id,
                attempt_number=state.attempt_count,
                timeout=budget,
            )
            # Counted here rather than in the provider, because this is the one
            # seam every non-streaming attempt passes through and a provider
            # cannot be trusted to report how many times it was called.
            if self._metrics is None:
                yield context
                return
            self._metrics.upstream_requests_total.inc()
            started = time.monotonic()
            try:
                yield context
            finally:
                self._metrics.upstream_duration_seconds.observe(time.monotonic() - started)

    async def _invoke_stream_on(
        self,
        request: InvocationRequest,
        routes: tuple[Route, ...],
        facts: "_StreamFacts",
        *,
        deadline: Deadline | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Serve one streaming invocation from the configured route order.

        The streaming counterpart of :meth:`_invoke_on`, and the same walk:
        the primary route first, then the fallbacks that could serve the
        request, under the invocation's single deadline, with the same two
        questions asked of every failure. The difference is the rule a
        non-streaming invocation never had to enforce: **once an event has been
        yielded to the caller, no fallback
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
        a finish. The error is then raised, so the caller knows
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
        a stream that is over without a terminal event, not a failed one.

        **``facts`` is what the walk leaves behind for the record.** A generator
        cannot return the facts an invocation's accounting needs, so the route
        that produced output, the walk state, the last usage the provider
        reported and the terminal state are accumulated as the walk runs and read
        once the stream is over. They are set *before* each event is handed over,
        not after: a consumer that stops on a terminal event closes this
        generator at the ``yield``, and the stream still reached its terminal
        state. A walk that produced no output leaves ``facts.route`` unset, which
        is how the caller of this method knows there is no answer to account for.
        """
        plan = _select_routes(routes, request)
        if deadline is None:
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
                        if facts.route is None:
                            # The route that produced the first event is the one
                            # that served the invocation: nothing may fall back or
                            # retry once output has begun, so it cannot change.
                            facts.route = route
                            facts.state = state
                        if event.usage is not None:
                            facts.usage = event.usage
                        if event.kind in _TERMINAL_EVENT_KINDS:
                            facts.status = (
                                STATUS_COMPLETED
                                if event.kind is StreamEventKind.FINISH
                                else STATUS_FAILED
                            )
                            facts.error_reason = event.error_reason
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
                        facts.status = STATUS_FAILED
                        facts.error_reason = _gateway_reason(error)
                        yield _error_event(request, error, sequence=last_sequence + 1)
                        terminal_emitted = True
                        raise
                    remaining = deadline.remaining()
                    if (
                        retry < self._retry_count
                        and is_retryable(error, remaining=remaining)
                        and await _back_off(retry + 1, self._retry_backoff_ms, deadline)
                    ):
                        self._count_retry()
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
        slot consumes invocation budget too.

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
            # The streaming counterpart of the counting in _attempt_context: one
            # upstream request per gateway attempt, timed over the whole stream
            # rather than over its first event.
            if self._metrics is not None:
                self._metrics.upstream_requests_total.inc()
            started = time.monotonic()
            try:
                async for event in upstream:
                    yield event
            finally:
                await upstream.aclose()
                if self._metrics is not None:
                    self._metrics.upstream_duration_seconds.observe(
                        time.monotonic() - started
                    )


@asynccontextmanager
async def _unbounded_slot() -> AsyncIterator[None]:
    """A slot a deployment that configured no per-tenant bound holds and nothing more.

    It exists so the entry points can take a slot unconditionally: a bound that was
    not configured is then not a special case each of them has to remember to test,
    and the ``async with`` that holds it is the same statement either way.
    """
    yield


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
    refused.
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
    """The refusal for an invocation whose budget is gone.

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
    contradict it. The stream therefore ends with
    exactly one terminal ``ERROR`` event, so the caller is not left with a
    stream that simply stopped, and is not told the answer finished when it did
    not.

    The event carries the gateway's own stable reason and its sanitised message,
    never a finish reason. A finish reason classifies *why generation stopped*,
    and there is none: the provider failed rather than finishing, so borrowing
    ``FinishReason.UNKNOWN`` and parking the gateway's reason in
    ``raw_finish_reason`` would describe a finish that never happened and put
    gateway vocabulary in a field reserved for the provider's own.
    ``error_reason`` is the value of the failure's
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
    rather than the provider's. The event is built from the
    normalised error, never from the raw one.
    """
    gateway_error = _normalised_error(error)
    return StreamEvent(
        kind=StreamEventKind.ERROR,
        sequence=sequence,
        invocation_id=request.invocation_id or "",
        error_reason=gateway_error.reason.value,
        error_message=gateway_error.message,
    )


def _normalised_error(error: BaseException) -> GatewayError:
    """A failure as the gateway's own taxonomy states it.

    A failure that is already a :class:`~model_gateway.errors.GatewayError` is
    itself; anything else is normalised, so the reason is one of the taxonomy's
    own and the message is gateway-authored rather than the provider's. The
    terminal event a failed stream carries and the record that
    stream leaves are both built from this one answer, so they cannot disagree
    about why the invocation failed.
    """
    return error if isinstance(error, GatewayError) else normalise_exception(error)


def _gateway_reason(error: BaseException) -> str:
    """The stable reason a failure is reported under, in the taxonomy's vocabulary."""
    return _normalised_error(error).reason.value


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
    the public model never advertised.

    Only then are the remaining routes filtered, each by the same test, into the
    fallbacks. A route that does not declare everything the request needs is
    skipped rather than attempted, so a capability gap in a fallback cannot make
    an invocation fatal that the primary route could have served; and because
    the request was already validated against the primary,
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
    that does not declare an operation's capability cannot serve it. The
    dimension test applies only when the caller stated a
    dimension: a route that does not accept the ``dimensions`` parameter
    cannot honour a request that asks for one, so it is not a candidate. The
    primary route establishes whether caller-selectable dimensions are part of
    the model's contract, and a fallback that cannot honour the same request is
    not one — a fallback may preserve the advertised contract, never expand it.
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
    """The refusal for a request the advertised model cannot serve.

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
    would change what the caller said; and a
    generation config carrying a non-text response format needs the
    structured-output capability, because a route that cannot honour the schema
    would otherwise answer with free text while the caller believed its
    requirement had been met. A text-mode response format asks
    for nothing beyond chat: it is the provider's own default.

    A system instruction asks for nothing beyond chat. It is baseline chat
    semantics that every chat binding must map or refuse, so a route declaring
    chat already claims to handle it, and a
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


def _outcome_of(
    response: InvocationResponse | EmbeddingResponse | ImageResponse,
) -> InvocationOutcome:
    """The answer facts a record takes from a canonical answer, whichever one it is.

    All three canonical answers carry the same three facts the record needs from
    an answer — the invocation identifier, the upstream model and the usage
    report — and a non-streaming path only ever reaches this with an answer that
    completed, so the status is the completed one. A streamed answer has no such
    object and states these facts directly (see ``_StreamFacts``), which is why
    the record's input is a value object rather than a response type.
    """
    return InvocationOutcome(
        invocation_id=response.invocation_id or "",
        upstream_model=response.upstream_model,
        usage=response.usage,
    )


def _normalise(
    route: Route,
    request: InvocationRequest,
    response: InvocationResponse,
    state: InvocationState,
) -> InvocationResponse:
    """Stamp the gateway's own facts onto a provider's answer.

    A provider's response is upstream data, never forwarded unchanged. The
    gateway chose the route, so it states which public model was
    asked for and which upstream model answered; and the gateway made the
    attempts, so ``attempt_count`` and ``fallback_used`` are read from ``state``
    rather than assumed. A provider cannot report how many times the gateway
    called, and neither may this function invent the number.

    ``invocation_id`` is always the gateway's own, never the provider's. The
    gateway mints the identifier before the call, and the response field is the
    gateway invocation identifier, so a value a
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

    The canonical embeddings answer is the float form, and so
    is the OpenAI-compatible response the endpoint renders. ``float`` — and no
    stated encoding at all, which is the upstream's own default — is therefore
    honoured without asking the provider for anything else, and anything the
    canonical request can still carry but the gateway cannot answer in, today
    ``base64``, is refused here with a capability error naming the encoding.

    The refusal is deliberate rather than a gap. Answering a caller who asked
    for base64 with floats would tell it its request was honoured while the
    gateway would be silently ignoring the encoding it asked for, which is
    forbidden; and the gateway cannot return base64 instead, because the canonical
    vector is a tuple of numbers and no field of the response carries an encoded
    form. Refusing is the only answer that keeps the caller's view of the
    request and the gateway's view of it identical.
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
    the route does not accept.

    The primary route establishes whether caller-selectable dimensions are part
    of the model's advertised contract, and no later route can expand it
    A fallback that cannot honour the same request is not a
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


def _require_image_provider(provider: TextProvider) -> None:
    """Refuse an image invocation the bound provider cannot serve.

    Image generation is a capability of its own, not a mode of
    :class:`~model_gateway.providers.base.TextProvider`: it is a different
    upstream operation with a different answer — images rather than words — so
    a provider that cannot generate is refused rather than handed the request
    and left to improvise. There is no honest degradation: a text provider
    asked for an image could only invent one, and either would tell the caller
    its request was served when it was not. The refusal names ``generation``
    so a caller learns which operation is missing.
    """
    if not isinstance(provider, ImageProvider):
        raise CapabilityError(
            "this provider cannot produce images for the 'generation' capability; the gateway "
            "does not answer an image request with an image it fabricated, so the request is "
            "refused",
            details={"capability": _IMAGE_CAPABILITY},
        )


def _normalise_images(
    route: Route,
    request: ImageRequest,
    response: ImageResponse,
    *,
    invocation_id: str,
) -> ImageResponse:
    """Stamp the gateway's own facts onto a provider's image answer.

    A provider's image answer is upstream data, never forwarded unchanged
    , and the two facts a provider cannot state are the
    gateway's own: which public model was asked for and which upstream model
    answered, and which invocation this answer belongs to.

    The answer is checked against the request before anything is stamped: a
    caller that stated ``n`` asked for exactly that many images, and an
    upstream that returned another number has not answered the request, so
    the invocation is refused rather than answered with the wrong number of
    images. A caller who stated no count is checked against
    none — the upstream's own default applies, and the gateway cannot know
    what number that was, so an unstated count is never a refusal.

    The refusal is a :class:`~model_gateway.errors.MalformedResponseError`.
    It is a property of what *this upstream* returned, not of the caller's
    request: the request was valid and the route was reachable, and the answer
    it produced was wrong. That is precisely the shape of failure another
    eligible route can fix, so the invocation walks to the next route while
    the route that just answered wrongly is not worth retrying immediately.
    """
    _check_image_count(request, response)
    return replace(
        response,
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=invocation_id,
    )


def _check_image_count(request: ImageRequest, response: ImageResponse) -> None:
    """Refuse an image answer that does not carry the number of images asked for.

    ``ImageRequest.n`` is how many images the caller wants, and it is
    forwarded to the upstream as stated; an answer carrying another number is
    an answer to a different request, and returning it would tell the caller
    it received ``n`` images when it did not. An unstated
    count means the upstream's own default applies, and no expected number
    exists to check against.
    """
    if request.n is None:
        return
    returned = len(response.data)
    if returned != request.n:
        raise MalformedResponseError(
            f"the upstream returned {returned} images where {request.n} were requested; an "
            f"answer that does not carry the number of images asked for is refused rather "
            f"than returned as the caller's answer"
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
    , and the two facts a provider cannot state are the gateway's
    own: which public model was asked for and which upstream model answered, and
    which invocation this answer belongs to.

    The vectors are placed at the positions the upstream reported, so ``data`` is
    in the caller's input order whatever order the upstream used, and each
    vector's own ``index`` is the position it was placed at. The gateway
    preserves input ordering, and preserving it means more
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
    dimensionality of zero. A vector of the wrong
    length is refused rather than returned, because a wrong-shaped vector is a
    wrong answer that a consumer of fixed-width vectors would accept and then
    misuse.

    Both refusals are :class:`~model_gateway.errors.MalformedResponseError`.
    They are properties of what *this upstream* returned, not of the caller's
    request: the request was valid and the route was reachable, and the answer
    it produced was structurally wrong. That is precisely the shape of failure
    another eligible route can fix, so the invocation walks to the next route
    while the route that just answered wrongly is not worth retrying
    immediately. A gateway defect, by contrast, is
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
