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
from blacksheep import Application, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    TextPart,
    Usage,
)
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import DeadlineExceededError, ErrorReason, ResourceExhaustedError
from model_gateway.http.auth import create_auth_middleware
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.inference import InferenceService
from model_gateway.limits import RateLimiter, TenantConcurrencyLimiter
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet


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


async def _yield_times(turns: int = 50) -> None:
    """Let the event loop run ``turns`` times, so a queued task gets every chance to run."""
    for _ in range(turns):
        await asyncio.sleep(0)


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
        # NaN passes a check written as ``not window > 0``, because every
        # comparison with NaN is false — so the check has to be the good case.
        (1, float("nan")),
        (1, float("inf")),
        # A window small enough to overflow the refill division is not a rate: it
        # refills a bucket to full after any positive interval.
        (1, 1e-320),
    ],
)
def test_a_limit_that_is_not_a_limit_is_refused(requests, window):
    """A bound that bounds nothing is a configuration mistake, not a limit."""
    with pytest.raises(ValueError):
        RateLimiter(requests=requests, window_seconds=window)


def test_a_clock_that_goes_backwards_does_not_grant_credit():
    """A monotonic clock is what the limiter relies on, and a step back must not refill.

    The behaviour is conservative rather than clever: a negative interval refills
    nothing and leaves the bucket's last-update time alone, so a clock adjustment
    cannot hand a caller tokens it did not earn. Refills resume once the clock
    catches up to where it was.
    """
    clock = _Clock()
    limiter = RateLimiter(requests=1, window_seconds=60, clock=clock)
    limiter.check("cred-a")

    clock.advance(-30.0)
    with pytest.raises(ResourceExhaustedError):
        limiter.check("cred-a")

    clock.advance(90.0)
    limiter.check("cred-a")


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


async def test_a_limiter_is_built_lazily_for_the_principal_that_asks_for_one():
    """A principal gets its limiter on first use rather than at construction.

    It does **not** show that the map is bounded, and it is named for laziness
    rather than for a bound for exactly that reason: this class accepts any
    principal string and grows with distinct keys, and what bounds the set is the
    call sites supplying trusted principals rather than anything enforced here.
    """
    limiter = TenantConcurrencyLimiter(maximum=2)

    assert limiter.principals == ()
    assert limiter.maximum == 2

    async with limiter.slot("tenant-a"):
        assert limiter.principals == ("tenant-a",)


def test_a_concurrency_bound_that_is_not_a_bound_is_refused():
    for bad in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError):
            TenantConcurrencyLimiter(maximum=bad)


# --------------------------------------------------------------------------- #
# The admission decisions, as the service takes them
# --------------------------------------------------------------------------- #


def _route(model_id: str = "upstream-text", **overrides) -> Route:
    values: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.internal.example/v1",
        "base_url": "https://upstream.internal.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)  # type: ignore[arg-type]


def _request(**overrides) -> InvocationRequest:
    values: dict[str, object] = {
        "model": "text-default",
        "messages": (Message(role=Role.USER, content=(TextPart("hello"),)),),
        "tenant_id": "tenant-a",
        "idempotency_key": None,
    }
    values.update(overrides)
    return InvocationRequest(**values)  # type: ignore[arg-type]


def _response() -> InvocationResponse:
    return InvocationResponse(
        model="text-default",
        upstream_model="upstream-text",
        invocation_id="inv-limits-1",
        content=(TextPart("hello"),),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class _EchoProvider:
    """The deterministic provider the other suites use: one scripted answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        return _response()


class _BlockingProvider:
    """A provider that holds the first attempt open, so a bound has something to bound."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[Route, object, object]] = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        self.entered.set()
        await self.release.wait()
        return _response()


class _StreamingProvider:
    """A provider whose only answer is a stream that finishes at once."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def stream(self, route, request, context):
        self.calls.append((route, request, context))
        yield StreamEvent(
            kind=StreamEventKind.FINISH,
            sequence=0,
            invocation_id=context.invocation_id or "",
            finish_reason=FinishReason.STOP,
        )


def _service(provider, *, routes=None, **overrides) -> InferenceService:
    return InferenceService(ModelRegistry(routes or (_route(),)), provider, **overrides)


def _fresh_limiter(requests: int = 1) -> RateLimiter:
    """A limiter whose clock a test can advance, shared by the service-level tests."""
    return RateLimiter(requests=requests, window_seconds=60, clock=_Clock())


async def test_a_rate_limited_caller_is_refused_before_any_provider_is_called():
    """The decision is taken at the entry, so a refused request costs nothing upstream."""
    provider = _EchoProvider()
    service = _service(provider, rate_limiter=_fresh_limiter())

    await service.invoke(_request(), namespace="cred-a")

    with pytest.raises(ResourceExhaustedError) as refused:
        await service.invoke(_request(), namespace="cred-a")

    assert refused.value.reason is ErrorReason.RATE_LIMITED
    assert refused.value.http_status == 429
    assert len(provider.calls) == 1


async def test_a_rate_limit_is_never_walked_to_another_route():
    """The gateway's own limit belongs to the caller, and the walk must not see it.

    ``is_fallback_eligible`` treats ``RATE_LIMITED`` as route-local on purpose: an
    *upstream's* rate limit belongs to the route that answered it, and another route
    is not necessarily limited. The gateway's own limit follows the caller, so
    raising it inside the walk would make the gateway try every configured provider
    to escape a limit that would refuse it at each of them.
    """
    provider = _EchoProvider()
    service = _service(
        provider,
        routes=(_route("primary"), _route("fallback")),
        rate_limiter=_fresh_limiter(),
    )

    await service.invoke(_request(), namespace="cred-a")
    with pytest.raises(ResourceExhaustedError):
        await service.invoke(_request(), namespace="cred-a")

    assert [call[0].model_id for call in provider.calls] == ["primary"]


async def test_one_callers_rate_limit_does_not_refuse_another():
    provider = _EchoProvider()
    service = _service(provider, rate_limiter=_fresh_limiter())

    await service.invoke(_request(), namespace="cred-a")
    with pytest.raises(ResourceExhaustedError):
        await service.invoke(_request(), namespace="cred-a")

    await service.invoke(_request(), namespace="cred-b")

    assert len(provider.calls) == 2


async def test_a_tenants_simultaneous_invocations_are_bounded():
    """The second invocation of a one-slot tenant waits rather than reaching upstream."""
    provider = _BlockingProvider()
    service = _service(
        provider, tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    )

    first = asyncio.create_task(
        service.invoke(_request(), namespace="cred-a", tenant="tenant-a")
    )
    await asyncio.wait_for(provider.entered.wait(), timeout=5.0)
    second = asyncio.create_task(
        service.invoke(_request(), namespace="cred-b", tenant="tenant-a")
    )
    await _yield_times()

    assert not second.done()
    assert len(provider.calls) == 1

    provider.release.set()
    await asyncio.gather(first, second)


async def test_one_tenants_bound_does_not_hold_another_back():
    """Two credentials of two tenants are two principals for this bound."""
    provider = _BlockingProvider()
    service = _service(
        provider, tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    )

    first = asyncio.create_task(
        service.invoke(_request(), namespace="cred-a", tenant="tenant-a")
    )
    await asyncio.wait_for(provider.entered.wait(), timeout=5.0)
    second = asyncio.create_task(
        service.invoke(_request(), namespace="cred-b", tenant="tenant-b")
    )
    await _settle(lambda: len(provider.calls) == 2)

    provider.release.set()
    await asyncio.gather(first, second)


async def test_a_tenant_slot_is_released_when_the_invocation_ends():
    """A leaked slot would refuse the tenant's next invocation, which this would catch."""
    provider = _EchoProvider()
    service = _service(
        provider, tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=0)
    )

    for _ in range(3):
        await service.invoke(_request(), namespace="cred-a", tenant="tenant-a")

    assert len(provider.calls) == 3


async def test_a_streamed_invocation_is_admitted_too():
    """The streaming entry point takes the same decisions as the non-streaming ones."""
    provider = _StreamingProvider()
    service = _service(
        provider,
        rate_limiter=_fresh_limiter(),
        tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=0),
    )

    stream = service.invoke_stream(_request(), namespace="cred-a", tenant="tenant-a")
    assert [event async for event in stream]

    with pytest.raises(ResourceExhaustedError) as refused:
        async for _ in service.invoke_stream(
            _request(), namespace="cred-a", tenant="tenant-a"
        ):
            pass

    assert refused.value.reason is ErrorReason.RATE_LIMITED
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------- #
# The tenant reaches the bound through the transport
# --------------------------------------------------------------------------- #

_KEY_A = "sk-gateway-key-a"
_KEY_B = "sk-gateway-key-b"
_SHARED_TENANT = "shared-tenant"


def _chat_app(provider) -> Application:
    """An application serving the chat endpoint, with two keys bound to one tenant.

    Built by hand so the service can be handed a per-tenant bound and a provider
    that holds its attempt open, which is what makes the bound observable.
    """
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_KEY_A}:{_SHARED_TENANT},{_KEY_B}:{_SHARED_TENANT}",
        }
    )
    service = InferenceService(
        ModelRegistry((_route(),)),
        provider,
        tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=4),
    )
    app = Application(router=Router())
    register_error_handlers(app)
    register_chat_routes(app.router, service)
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


async def test_two_credentials_of_one_tenant_share_its_concurrency_bound():
    """The bound is per tenant, so two credentials bound to one tenant share it.

    This is the test that proves the transport hands the service the tenant rather
    than only the credential: keying the bound on the credential — which is what
    happens when the tenant is not passed — would let both of these through, and
    the assertion that the second is still waiting is what catches that.
    """
    provider = _BlockingProvider()
    app = _chat_app(provider)
    body = {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}

    await app.start()
    try:
        client = TestClient(app)
        first = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                content=JSONContent(body),
                headers={"Authorization": f"Bearer {_KEY_A}"},
            )
        )
        await asyncio.wait_for(provider.entered.wait(), timeout=5.0)
        second = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                content=JSONContent(body),
                headers={"Authorization": f"Bearer {_KEY_B}"},
            )
        )
        await _yield_times()

        assert not second.done()
        assert len(provider.calls) == 1

        provider.release.set()
        responses = await asyncio.gather(first, second)
        assert [response.status for response in responses] == [200, 200]
    finally:
        await app.stop()


# --------------------------------------------------------------------------- #
# A stream holds its tenant's slot for as long as it is arriving
# --------------------------------------------------------------------------- #


class _BlockingStreamProvider:
    """A streaming provider that holds its stream open between two events.

    A paused stream is the case a per-tenant bound is most likely to leak on: the
    invocation is resident while it waits for the next chunk, so it must still
    occupy the slot it was admitted to.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[Route, object, object]] = []

    async def stream(self, route, request, context):
        self.calls.append((route, request, context))
        invocation_id = context.invocation_id or ""
        yield StreamEvent(
            kind=StreamEventKind.CONTENT,
            sequence=0,
            invocation_id=invocation_id,
            text="hel",
        )
        self.entered.set()
        await self.release.wait()
        yield StreamEvent(
            kind=StreamEventKind.FINISH,
            sequence=1,
            invocation_id=invocation_id,
            finish_reason=FinishReason.STOP,
        )


async def test_a_tenant_slot_is_held_for_the_whole_life_of_a_stream():
    """A paused stream still occupies its tenant's slot, and frees it when it ends.

    The rate limiter is deliberately absent here: the point is the concurrency
    bound on the streaming path, and a rate-limit refusal would answer the second
    request before the bound could.
    """
    provider = _BlockingStreamProvider()
    service = _service(
        provider, tenant_limiter=TenantConcurrencyLimiter(maximum=1, max_waiters=4)
    )

    async def consume():
        return [event async for event in service.invoke_stream(_request(), tenant="tenant-a")]

    first = asyncio.create_task(consume())
    await asyncio.wait_for(provider.entered.wait(), timeout=5.0)

    second = asyncio.create_task(consume())
    await _yield_times()

    assert not second.done()
    assert len(provider.calls) == 1

    provider.release.set()
    await asyncio.gather(first, second)

    assert len(provider.calls) == 2


# --------------------------------------------------------------------------- #
# Every operation takes the decisions, and the refusal is a 429
# --------------------------------------------------------------------------- #


class _EmbeddingProvider:
    """A provider that answers an embeddings request with one vector."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def embed(self, route, request, context):
        from model_gateway.domain.embeddings import Embedding, EmbeddingResponse

        self.calls.append((route, request, context))
        return EmbeddingResponse(
            model=request.model,
            upstream_model=route.model_id,
            data=(Embedding(index=0, values=(0.1, 0.2)),),
            usage=Usage(input_tokens=1),
            invocation_id="provider-ignored",
        )


class _ImageProvider:
    """A provider that answers an image request with one image."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def generate(self, route, request, context):
        from model_gateway.domain.images import GeneratedImage, ImageResponse

        self.calls.append((route, request, context))
        return ImageResponse(
            model=request.model,
            upstream_model=route.model_id,
            created=0,
            data=(GeneratedImage(b64_json="Zm9v"),),
            usage=Usage(input_tokens=1),
            invocation_id="provider-ignored",
        )


async def test_an_embeddings_invocation_is_rate_limited():
    """The embeddings entry point takes the same decision as the text one."""
    from model_gateway.domain.embeddings import EmbeddingRequest

    provider = _EmbeddingProvider()
    service = _service(
        provider,
        routes=(_route("upstream-embed", operation=Operation.EMBEDDINGS,
                       capabilities=frozenset({Capability.EMBEDDINGS})),),
        rate_limiter=_fresh_limiter(),
    )
    request = EmbeddingRequest(model="embedding-default", input=("hello",))

    await service.embed(request, namespace="cred-a")
    with pytest.raises(ResourceExhaustedError) as refused:
        await service.embed(request, namespace="cred-a")

    assert refused.value.reason is ErrorReason.RATE_LIMITED
    assert len(provider.calls) == 1


async def test_an_image_invocation_is_rate_limited():
    """The image entry point takes the same decision as the text one."""
    from model_gateway.domain.images import ImageRequest

    provider = _ImageProvider()
    service = _service(
        provider,
        routes=(_route("upstream-image", operation=Operation.IMAGE,
                       capabilities=frozenset({Capability.GENERATION})),),
        rate_limiter=_fresh_limiter(),
    )
    request = ImageRequest(model="image-default", prompt="a red fox")

    await service.generate(request, namespace="cred-a")
    with pytest.raises(ResourceExhaustedError) as refused:
        await service.generate(request, namespace="cred-a")

    assert refused.value.reason is ErrorReason.RATE_LIMITED
    assert len(provider.calls) == 1


def _limited_chat_app(provider, **overrides) -> Application:
    """A chat application with a request rate of one and no concurrency bound."""
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_KEY_A}:{_SHARED_TENANT}",
            **overrides,
        }
    )
    service = InferenceService(
        ModelRegistry((_route(),)),
        provider,
        rate_limiter=RateLimiter(requests=1, window_seconds=60),
    )
    app = Application(router=Router())
    register_error_handlers(app)
    register_chat_routes(app.router, service)
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


async def test_a_rate_limit_refusal_is_a_429_through_the_http_surface():
    """The transport's own contract for the refusal, end to end rather than by taxonomy."""
    app = _limited_chat_app(_EchoProvider())
    body = {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}

    await app.start()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {_KEY_A}"}
        first = await client.post(
            "/v1/chat/completions", content=JSONContent(body), headers=headers
        )
        second = await client.post(
            "/v1/chat/completions", content=JSONContent(body), headers=headers
        )

        assert first.status == 200
        assert second.status == 429
        payload = await second.json()
        assert payload["reason"] == "rate_limited"
        # The public payload is reason and message only: details are on the
        # exception but are not published by default.
        assert "details" not in payload
    finally:
        await app.stop()
