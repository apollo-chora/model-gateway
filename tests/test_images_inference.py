"""The inference service's image path: the same policy, a different operation.

The image entry point is the non-streaming text entry point applied to a
different operation, so these tests are about the two things that differ and
the things that must not: the capability the route has to declare and the
provider has to implement, the image count that only this operation checks,
and the one deadline, retry loop, fallback walk, concurrency slot and
cancellation rule the operations share.

Nothing here reaches the network: every provider is a stand-in over in-memory
values and every route is a standalone value.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

import model_gateway.reliability as reliability
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain.generation import Usage
from model_gateway.domain.images import GeneratedImage, ImageRequest, ImageResponse
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    InternalError,
    InvalidArgumentError,
    MalformedResponseError,
    ModelNotFoundError,
    ResourceExhaustedError,
    UnavailableError,
)
from model_gateway.inference import InferenceService
from model_gateway.providers.base import AttemptContext, ImageProvider
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter

PAYLOAD = "aW1hZ2UtYnl0ZXM="
PUBLIC_MODEL = "image-default"


def route(model_id: str = "image-model", **overrides: object) -> Route:
    """A standalone image route."""
    values: dict[str, object] = {
        "operation": Operation.IMAGE,
        "index": 0,
        "endpoint": "https://upstream.example/v1",
        "base_url": "https://upstream.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.GENERATION}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def request(**overrides: object) -> ImageRequest:
    """A standalone canonical image request."""
    values: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "prompt": "A futuristic city at sunset",
        "size": "1024x1024",
        "response_format": "b64_json",
        "n": 2,
    }
    values.update(overrides)
    return ImageRequest(**values)


def images(count: int, *, url: str | None = None) -> tuple[GeneratedImage, ...]:
    """A tuple of generated images, each carrying the same base64 payload."""
    if url is not None:
        return tuple(GeneratedImage(url=url) for _ in range(count))
    return tuple(GeneratedImage(b64_json=PAYLOAD) for _ in range(count))


def response_for(
    route_: Route,
    request_: ImageRequest,
    context: AttemptContext,
    *,
    data: tuple[GeneratedImage, ...] | None = None,
) -> ImageResponse:
    """The canonical answer a provider would build for this attempt."""
    return ImageResponse(
        model=request_.model,
        upstream_model=route_.model_id,
        created=1700000000,
        data=data if data is not None else images(request_.n or 1),
        usage=Usage(),
        invocation_id=context.invocation_id or "inv-from-the-provider",
    )


def too_few(route_: Route, request_: ImageRequest, context: AttemptContext) -> ImageResponse:
    """An answer carrying fewer images than the caller asked for."""
    return response_for(route_, request_, context, data=images(1))


def too_many(route_: Route, request_: ImageRequest, context: AttemptContext) -> ImageResponse:
    """An answer carrying more images than the caller asked for."""
    return response_for(route_, request_, context, data=images(3))


class FakeImageProvider:
    """A provider that answers each call from a script, recording what it was handed.

    An outcome is a :class:`~model_gateway.domain.images.ImageResponse`, a
    callable that builds one from the attempt, or a :exc:`BaseException` to
    raise. The last outcome repeats once the script is exhausted, so a test can
    script a failure followed by a recovery — or the reverse — and still know
    exactly what the next call will get.
    """

    def __init__(self, *outcomes: object) -> None:
        assert outcomes
        self._outcomes = list(outcomes)
        self.calls: list[tuple[Route, ImageRequest, AttemptContext]] = []

    async def generate(
        self, route_: Route, request_: ImageRequest, context: AttemptContext
    ) -> ImageResponse:
        self.calls.append((route_, request_, context))
        outcome = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(route_, request_, context)
        return outcome


class FakeTextProvider:
    """A provider that answers text and nothing else: no ``generate`` method at all."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, route_: Route, request_: object, context: object) -> object:
        self.calls += 1
        raise AssertionError("a text-only provider must never be handed an image request")


def build(
    routes: tuple[Route, ...], provider: object, **overrides: object
) -> InferenceService:
    """An inference service over a registry built from ``routes``."""
    return InferenceService(ModelRegistry(routes), provider, **overrides)


# ---------------------------------------------------------------------------
# Resolving the model and the capability
# ---------------------------------------------------------------------------


class TestTheModelAndTheCapability:
    """What is refused before a provider is called, and in which order."""

    async def test_an_unknown_model_is_a_model_not_found_error(self) -> None:
        provider = FakeImageProvider(response_for)

        with pytest.raises(ModelNotFoundError):
            await build((route(),), provider).generate(request(model="no-such-model"))

        assert provider.calls == []

    async def test_the_model_is_resolved_before_the_providers_capability_is_asked(
        self,
    ) -> None:
        provider = FakeTextProvider()

        with pytest.raises(ModelNotFoundError):
            await build((route(),), provider).generate(request(model="no-such-model"))

        assert provider.calls == 0

    async def test_a_provider_that_cannot_generate_is_refused(self) -> None:
        with pytest.raises(CapabilityError) as error:
            await build((route(),), FakeTextProvider()).generate(request())

        assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY
        assert error.value.details["capability"] == Capability.GENERATION.value

    async def test_the_refusal_names_the_operation_that_is_missing(self) -> None:
        with pytest.raises(CapabilityError) as error:
            await build((route(),), FakeTextProvider()).generate(request())

        assert Capability.GENERATION.value in error.value.message

    async def test_a_primary_route_without_the_capability_is_refused(self) -> None:
        image_route = route(capabilities=frozenset({Capability.EDITING}))

        with pytest.raises(CapabilityError) as error:
            await build((image_route,), FakeImageProvider(response_for)).generate(request())

        assert error.value.details["capability"] == Capability.GENERATION.value

    async def test_a_later_route_cannot_expand_the_advertised_capability(self) -> None:
        """The public contract is the primary route's, so a later route cannot add to it.

        The primary declares nothing; a fallback that declares ``generation``
        does not make the model advertise it, and the request is refused
        naming the missing capability rather than served by the route that
        happens to have it.
        """
        routes = (
            route(capabilities=frozenset()),
            route(model_id="fallback", index=1),
        )

        with pytest.raises(CapabilityError) as error:
            await build(routes, FakeImageProvider(response_for)).generate(request())

        assert error.value.details["capability"] == Capability.GENERATION.value

    async def test_a_provider_satisfies_the_image_protocol_structurally(self) -> None:
        provider = FakeImageProvider(response_for)

        assert isinstance(provider, ImageProvider)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    """The invocation the service makes and the facts it states."""

    async def test_the_response_states_the_gateways_own_facts(self) -> None:
        provider = FakeImageProvider(response_for)

        response = await build((route(),), provider).generate(request())

        assert response.model == PUBLIC_MODEL
        assert response.upstream_model == "image-model"
        assert len(response.data) == 2

    async def test_the_invocation_identifier_is_the_gateways_and_reaches_the_attempt(
        self,
    ) -> None:
        provider = FakeImageProvider(response_for)

        response = await build((route(),), provider).generate(request())

        assert response.invocation_id
        assert len(response.invocation_id) == 32
        assert provider.calls[0][2].invocation_id == response.invocation_id

    async def test_the_providers_own_invocation_identifier_is_discarded(self) -> None:
        """The gateway mints the identifier; a provider cannot rename the invocation."""

        def answer(
            route_: Route, request_: ImageRequest, context: AttemptContext
        ) -> ImageResponse:
            return ImageResponse(
                model=request_.model,
                upstream_model=route_.model_id,
                created=1700000000,
                data=images(1),
                usage=Usage(),
                invocation_id="inv-from-the-upstream",
            )

        response = await build((route(),), FakeImageProvider(answer)).generate(
            request(n=1)
        )

        assert response.invocation_id != "inv-from-the-upstream"

    async def test_the_request_reaches_the_provider_unchanged(self) -> None:
        provider = FakeImageProvider(response_for)
        requested = request(prompt="A lighthouse at dusk", size="512x512", n=3)

        await build((route(),), provider).generate(requested)

        (_, received, _) = provider.calls[0]
        assert received.prompt == "A lighthouse at dusk"
        assert received.size == "512x512"
        assert received.n == 3

    async def test_a_single_image_is_one_image(self) -> None:
        provider = FakeImageProvider(response_for)

        response = await build((route(),), provider).generate(request(n=1))

        assert len(response.data) == 1


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------


class TestTheAnswer:
    """An answer that does not carry the number of images asked for is refused."""

    async def test_a_matching_count_is_served(self) -> None:
        provider = FakeImageProvider(response_for)

        response = await build((route(),), provider).generate(request(n=3))

        assert len(response.data) == 3

    async def test_an_answer_with_too_few_images_is_refused(self) -> None:
        provider = FakeImageProvider(too_few)

        with pytest.raises(MalformedResponseError) as error:
            await build((route(),), provider).generate(request(n=2))

        assert error.value.reason is ErrorReason.MALFORMED_RESPONSE

    async def test_an_answer_with_too_many_images_is_refused(self) -> None:
        provider = FakeImageProvider(too_many)

        with pytest.raises(MalformedResponseError):
            await build((route(),), provider).generate(request(n=1))

    async def test_an_unstated_count_is_never_a_refusal(self) -> None:
        """The upstream's own default applies, and the gateway cannot know its number."""
        provider = FakeImageProvider(response_for)

        response = await build((route(),), provider).generate(request(n=None))

        assert len(response.data) == 1


# ---------------------------------------------------------------------------
# The malformed answer and the fallback
# ---------------------------------------------------------------------------


class TestTheMalformedAnswerWalks:
    """A wrong-shaped answer is refused on that route and walked to another."""

    async def test_a_malformed_answer_walks_to_the_fallback(self) -> None:
        provider = FakeImageProvider(too_few, response_for)
        routes = (route(), route(model_id="fallback", index=1))

        response = await build(routes, provider, retry_count=2).generate(request(n=2))

        assert response.upstream_model == "fallback"
        assert len(response.data) == 2
        assert [route_.model_id for route_, _, _ in provider.calls] == [
            "image-model",
            "fallback",
        ]

    async def test_a_malformed_answer_is_not_retried_on_the_route_that_answered_wrongly(
        self,
    ) -> None:
        provider = FakeImageProvider(too_few)

        with pytest.raises(MalformedResponseError):
            await build((route(),), provider, retry_count=3).generate(request(n=2))

        assert len(provider.calls) == 1

    async def test_a_fallback_that_answers_wrongly_too_reports_its_failure(self) -> None:
        provider = FakeImageProvider(too_few, too_few)
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(MalformedResponseError):
            await build(routes, provider).generate(request(n=2))

        assert len(provider.calls) == 2

    async def test_a_malformed_answer_is_not_walked_past_a_route_without_the_capability(
        self,
    ) -> None:
        provider = FakeImageProvider(too_few, response_for)
        routes = (
            route(),
            route(model_id="text-only", index=1, capabilities=frozenset()),
            route(model_id="fallback", index=2),
        )

        response = await build(routes, provider).generate(request(n=2))

        assert response.upstream_model == "fallback"


# ---------------------------------------------------------------------------
# The reliability loop
# ---------------------------------------------------------------------------


class TestTheReliabilityLoop:
    """One deadline, one retry loop and one fallback walk, exactly as for text."""

    async def test_the_attempt_context_carries_the_budget_and_the_attempt_number(
        self,
    ) -> None:
        provider = FakeImageProvider(response_for)

        await build((route(),), provider, attempt_timeout=30.0).generate(request())

        (_, _, context) = provider.calls[0]
        assert context.attempt_number == 1
        assert context.timeout == pytest.approx(30.0, abs=0.1)

    async def test_a_retryable_failure_is_retried_on_the_same_route(self) -> None:
        provider = FakeImageProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )

        response = await build((route(),), provider, retry_count=1).generate(request())

        assert response.upstream_model == "image-model"
        assert [context.attempt_number for _, _, context in provider.calls] == [1, 2]

    async def test_a_rate_limit_moves_to_the_other_route_without_retrying(self) -> None:
        provider = FakeImageProvider(
            ResourceExhaustedError("the upstream rate limit was reached"), response_for
        )
        routes = (route(), route(model_id="fallback", index=1))

        response = await build(routes, provider, retry_count=2).generate(request())

        assert response.upstream_model == "fallback"
        assert len(provider.calls) == 2

    async def test_a_fallback_route_serves_when_the_primary_cannot(self) -> None:
        provider = FakeImageProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )
        routes = (route(), route(model_id="fallback", index=1))

        response = await build(routes, provider).generate(request())

        assert response.upstream_model == "fallback"
        assert [route_.model_id for route_, _, _ in provider.calls] == [
            "image-model",
            "fallback",
        ]

    async def test_a_fallback_without_the_capability_is_skipped(self) -> None:
        provider = FakeImageProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )
        routes = (
            route(),
            route(model_id="chat-only", index=1, capabilities=frozenset({Capability.CHAT})),
            route(model_id="capable", index=2),
        )

        response = await build(routes, provider).generate(request())

        assert response.upstream_model == "capable"

    async def test_a_deterministic_failure_ends_the_invocation_where_it_happened(
        self,
    ) -> None:
        """No other provider can change the answer to a request that cannot succeed."""
        provider = FakeImageProvider(InvalidArgumentError("the request is invalid"))
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(InvalidArgumentError):
            await build(routes, provider, retry_count=2).generate(request())

        assert len(provider.calls) == 1

    async def test_a_gateway_defect_terminates_where_it_happened(self) -> None:
        """An internal error is the gateway's own defect: no route can fix it."""
        provider = FakeImageProvider(
            InternalError("the gateway could not read its own answer")
        )
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(InternalError):
            await build(routes, provider, retry_count=2).generate(request())

        assert len(provider.calls) == 1

    async def test_the_last_attempts_failure_is_the_one_reported(self) -> None:
        provider = FakeImageProvider(
            UnavailableError("the first route is unavailable"),
            UnavailableError("the second route is unavailable"),
        )
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(UnavailableError) as error:
            await build(routes, provider).generate(request())

        assert "second route" in error.value.message

    async def test_a_cancelled_image_invocation_propagates(self) -> None:
        provider = FakeImageProvider(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await build((route(),), provider).generate(request())

    async def test_a_cancelled_attempt_makes_no_further_attempt(self) -> None:
        provider = FakeImageProvider(asyncio.CancelledError())
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(asyncio.CancelledError):
            await build(routes, provider, retry_count=2).generate(request())

        assert len(provider.calls) == 1

    async def test_an_expired_deadline_refuses_the_invocation_without_calling_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reliability, "monotonic", _TickingClock(step=10.0))
        provider = FakeImageProvider(response_for)
        inference = build((route(),), provider, attempt_timeout=1.0)

        with pytest.raises(DeadlineExceededError) as error:
            await inference.generate(request())

        assert provider.calls == []
        assert error.value.reason is ErrorReason.DEADLINE_EXCEEDED
        assert "upstream.example" not in str(error.value)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class _RecordingLimiter:
    """A stub limiter that records the bound each caller asked to wait within."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    @asynccontextmanager
    async def slot(self, timeout: float | None = None):
        self.timeouts.append(timeout)
        yield


class _TickingClock:
    """A clock that runs forward on every reading, so a deadline expires between two of them.

    :class:`~model_gateway.reliability.Deadline` anchors itself with one reading
    and measures with the next, so a step larger than the configured timeout
    reaches the state a real deployment reaches only when time has run out.
    """

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        reading = self._now
        self._now += self._step
        return reading


class TestConcurrency:
    """The wait for a slot is bounded by the invocation deadline."""

    async def test_the_limiter_is_handed_what_is_left_of_the_deadline(self) -> None:
        limiter = _RecordingLimiter()
        provider = FakeImageProvider(response_for)

        await build((route(),), provider, attempt_timeout=7.0, limiter=limiter).generate(
            request()
        )

        assert limiter.timeouts[0] == pytest.approx(7.0, abs=0.1)

    async def test_a_limiter_refuses_an_invocation_it_cannot_admit_in_time(self) -> None:
        """A wait that outlives the deadline raises rather than queueing forever."""

        class _FullLimiter:
            @asynccontextmanager
            async def slot(self, timeout: float | None = None):
                raise DeadlineExceededError(
                    "the wait for a concurrency slot expired",
                    details={"attempts": "0"},
                )
                yield

        inference = build(
            (route(),), FakeImageProvider(response_for), limiter=_FullLimiter()
        )

        with pytest.raises(DeadlineExceededError):
            await inference.generate(request())

    async def test_the_service_bounds_simultaneous_attempts_by_the_configured_maximum(
        self,
    ) -> None:
        inference = build(
            (route(),),
            FakeImageProvider(response_for),
            max_concurrent_requests=3,
        )

        assert inference._limiter.maximum == 3

    async def test_the_default_limiter_is_built_from_the_configured_default(self) -> None:
        inference = build((route(),), FakeImageProvider(response_for))

        assert isinstance(inference._limiter, ConcurrencyLimiter)
        assert inference._limiter.maximum > 0
