"""The inference service's embeddings path: the same policy, a different operation.

The embeddings entry point is the non-streaming text entry point applied to a
different operation, so these tests are about the two things that differ and the
things that must not: the capability the route has to declare and the provider
has to implement, the ordering and dimensionality checks that only this operation
needs, and the one deadline, retry loop, fallback walk, concurrency slot and
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
from model_gateway.domain.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from model_gateway.domain.generation import Usage
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
from model_gateway.providers.base import AttemptContext, EmbeddingProvider
from model_gateway.registry import PUBLIC_MODEL_IDS, ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter

VECTOR = (0.12, -0.34, 0.56)
PUBLIC_MODEL = "embedding-default"


def route(model_id: str = "embedding-model", **overrides: object) -> Route:
    """A standalone embeddings route."""
    values: dict[str, object] = {
        "operation": Operation.EMBEDDINGS,
        "index": 0,
        "endpoint": "https://upstream.example/v1",
        "base_url": "https://upstream.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.EMBEDDINGS}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def request(**overrides: object) -> EmbeddingRequest:
    """A standalone canonical embeddings request."""
    values: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "input": ("First document", "Second document"),
    }
    values.update(overrides)
    return EmbeddingRequest(**values)


def vectors(*values: tuple[float, ...], indices: list[int] | None = None) -> tuple[Embedding, ...]:
    """A tuple of embeddings, positioned by ``indices`` or in the order given."""
    positions = indices if indices is not None else list(range(len(values)))
    return tuple(
        Embedding(index=index, values=value)
        for index, value in zip(positions, values, strict=True)
    )


def response_for(
    route_: Route,
    request_: EmbeddingRequest,
    context: AttemptContext,
    *,
    data: tuple[Embedding, ...] | None = None,
) -> EmbeddingResponse:
    """The canonical answer a provider would build for this attempt."""
    return EmbeddingResponse(
        model=PUBLIC_MODEL_IDS[route_.operation],
        upstream_model=route_.model_id,
        data=(
            data
            if data is not None
            else vectors(*[VECTOR for _ in request_.input])
        ),
        usage=Usage(input_tokens=10, total_tokens=10),
        invocation_id=context.invocation_id or "inv-from-the-provider",
    )


class FakeEmbeddingProvider:
    """A provider that answers each call from a script, recording what it was handed.

    An outcome is a :class:`~model_gateway.domain.embeddings.EmbeddingResponse`, a
    callable that builds one from the attempt, or a :exc:`BaseException` to raise.
    The last outcome repeats once the script is exhausted, so a test can script a
    recovery after a failure.
    """

    def __init__(self, *outcomes: object) -> None:
        assert outcomes
        self._outcomes = list(outcomes)
        self.calls: list[tuple[Route, EmbeddingRequest, AttemptContext]] = []

    async def embed(
        self, route_: Route, request_: EmbeddingRequest, context: AttemptContext
    ) -> EmbeddingResponse:
        self.calls.append((route_, request_, context))
        outcome = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(route_, request_, context)
        return outcome


class FakeTextProvider:
    """A provider that answers text and nothing else: no ``embed`` method at all."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, route_: Route, request_: object, context: object) -> object:
        self.calls += 1
        raise AssertionError("a text-only provider must never be handed an embeddings request")


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

    async def test_an_unknown_model_is_ignored_and_the_operation_serves_it(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        result = await build((route(),), provider).embed(request(model="no-such-model"))

        assert result.model == PUBLIC_MODEL
        assert len(provider.calls) == 1
        assert provider.calls[0][1].model == "no-such-model"

    async def test_the_operations_routes_are_resolved_before_the_providers_capability_is_asked(
        self,
    ) -> None:
        """An unconfigured operation is refused as that, whatever the provider can do."""
        provider = FakeEmbeddingProvider(lambda route_, request_, context: None)

        with pytest.raises(ModelNotFoundError):
            await build((), provider).embed(request(model="no-such-model"))

        assert provider.calls == []

    async def test_a_provider_that_cannot_embed_is_refused(self) -> None:
        """The gateway does not answer an embeddings request with a fabricated vector."""
        provider = FakeTextProvider()

        with pytest.raises(CapabilityError) as error:
            await build((route(),), provider).embed(request())

        assert error.value.details["capability"] == "embeddings"
        assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY

    async def test_the_refusal_names_the_operation_that_is_missing(self) -> None:
        with pytest.raises(CapabilityError) as error:
            await build((route(),), FakeTextProvider()).embed(request())

        assert "embeddings" in error.value.message

    async def test_a_primary_route_without_the_capability_is_refused(self) -> None:
        provider = FakeEmbeddingProvider(lambda route_, request_, context: None)
        chat_only = route(capabilities=frozenset({Capability.CHAT}))

        with pytest.raises(CapabilityError) as error:
            await build((chat_only,), provider).embed(request())

        assert error.value.details["capability"] == "embeddings"
        assert provider.calls == []

    async def test_a_later_route_cannot_expand_the_advertised_capability(self) -> None:
        """The public contract is the primary route's, so a fallback cannot rescue it."""
        provider = FakeEmbeddingProvider(lambda route_, request_, context: None)
        chat_only = route(capabilities=frozenset({Capability.CHAT}))

        with pytest.raises(CapabilityError):
            await build((chat_only, route(model_id="fallback")), provider).embed(request())

        assert provider.calls == []

    async def test_a_provider_satisfies_the_embedding_protocol_structurally(self) -> None:
        assert isinstance(FakeEmbeddingProvider(lambda route_, request_, context: None),
                          EmbeddingProvider)


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------


class TestTheAnswer:
    """What the service stamps on the provider's answer, and what it checks."""

    async def test_the_response_states_the_gateway_own_facts(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(),), provider).embed(request())

        assert response.model == PUBLIC_MODEL
        assert response.upstream_model == "embedding-model"
        assert response.invocation_id
        assert response.usage == Usage(input_tokens=10, total_tokens=10)

    async def test_the_invocation_identifier_is_the_gateways_and_reaches_the_attempt(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(),), provider).embed(request())

        (_, _, context) = provider.calls[0]
        assert context.invocation_id == response.invocation_id

    async def test_the_providers_own_invocation_identifier_is_discarded(self) -> None:
        """A provider must not be able to rename the invocation in the caller's record."""
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(),), provider).embed(request())

        assert response.invocation_id != "inv-from-the-provider"

    async def test_the_vectors_are_returned_in_the_callers_input_order(self) -> None:
        """The upstream's order is not the contract; the caller's order is."""
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_,
                request_,
                context,
                data=vectors((0.7, 0.8, 0.9), VECTOR, indices=[1, 0]),
            )
        )

        response = await build((route(),), provider).embed(request())

        assert [embedding.index for embedding in response.data] == [0, 1]
        assert [embedding.values for embedding in response.data] == [VECTOR, (0.7, 0.8, 0.9)]

    async def test_a_single_input_is_one_vector(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(),), provider).embed(request(input=("only one",)))

        assert len(response.data) == 1
        assert response.data[0].index == 0

    async def test_a_missing_vector_is_refused(self) -> None:
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors(VECTOR)
            )
        )

        with pytest.raises(MalformedResponseError) as error:
            await build((route(),), provider).embed(request())

        assert error.value.reason is ErrorReason.MALFORMED_RESPONSE

    async def test_a_duplicate_position_is_refused(self) -> None:
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors(VECTOR, VECTOR, indices=[0, 0])
            )
        )

        with pytest.raises(MalformedResponseError):
            await build((route(),), provider).embed(request())

    async def test_a_position_outside_the_input_is_refused(self) -> None:
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors(VECTOR, VECTOR, indices=[0, 7])
            )
        )

        with pytest.raises(MalformedResponseError):
            await build((route(),), provider).embed(request())


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------


class TestDimensionality:
    """The shape of the vectors is checked against what the route declares."""

    async def test_a_declared_dimensionality_is_enforced(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(dimensions=3),), provider).embed(request())

        assert len(response.data[0].values) == 3

    async def test_a_vector_of_the_wrong_length_is_refused(self) -> None:
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4))
            )
        )

        with pytest.raises(MalformedResponseError) as error:
            await build((route(dimensions=3),), provider).embed(request())

        assert error.value.reason is ErrorReason.MALFORMED_RESPONSE

    async def test_a_wrong_shaped_answer_walks_to_the_fallback(self) -> None:
        """The fault lies in the primary's answer, so another route can fix it.

        The malformed answer is not a property of the request: repeating it on
        the same route cannot help, but the fallback serves the same request
        correctly, so the invocation walks rather than terminating.
        """
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4))
            ),
            response_for,
        )

        response = await build(
            (route(dimensions=3), route(model_id="fallback", index=1, dimensions=3)),
            provider,
            retry_count=2,
        ).embed(request())

        assert response.upstream_model == "fallback"
        assert [route_.model_id for route_, _, _ in provider.calls] == [
            "embedding-model",
            "fallback",
        ]

    async def test_an_unconfigured_dimensionality_is_not_checked(self) -> None:
        """An unconfigured dimensionality is not a dimensionality of zero."""
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4, 0.5))
            )
        )

        response = await build((route(),), provider).embed(request(input=("only one",)))

        assert len(response.data[0].values) == 5

    async def test_the_callers_stated_dimension_is_what_is_checked(self) -> None:
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4))
            )
        )

        response = await build(
            (route(dimensions=3, supports_dimensions_param=True),), provider
        ).embed(request(input=("only one",), dimensions=4))

        assert len(response.data[0].values) == 4

    async def test_the_callers_stated_dimension_is_enforced_too(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        with pytest.raises(MalformedResponseError):
            await build(
                (route(dimensions=3, supports_dimensions_param=True),), provider
            ).embed(request(dimensions=4))


# ---------------------------------------------------------------------------
# The dimension parameter
# ---------------------------------------------------------------------------


class TestTheDimensionParameter:
    """A caller's stated dimension is honoured only where the route accepts it."""

    async def test_a_stated_dimension_on_a_route_that_does_not_accept_it_is_refused(
        self,
    ) -> None:
        """The route's dimensionality is fixed; a stated dimension is refused, not dropped."""
        provider = FakeEmbeddingProvider(response_for)

        with pytest.raises(CapabilityError) as error:
            await build((route(dimensions=3),), provider).embed(request(dimensions=4))

        assert error.value.details["parameter"] == "dimensions"
        assert "dimensions" in error.value.message
        assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY

    async def test_the_refusal_happens_before_any_upstream_call(self) -> None:
        """No provider is called, and the model and the parameter are both named."""
        provider = FakeEmbeddingProvider(response_for)

        with pytest.raises(CapabilityError) as error:
            await build((route(),), provider).embed(request(dimensions=4))

        assert provider.calls == []
        assert PUBLIC_MODEL in error.value.message
        assert "dimensions" in error.value.message

    async def test_a_route_that_accepts_the_parameter_is_served(self) -> None:
        """The stated dimension is forwarded there and becomes the expected length."""
        provider = FakeEmbeddingProvider(
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4))
            )
        )

        response = await build(
            (route(supports_dimensions_param=True, dimensions=4),), provider
        ).embed(request(input=("only one",), dimensions=4))

        assert response.model == PUBLIC_MODEL
        assert len(response.data[0].values) == 4

    async def test_no_stated_dimension_is_never_refused(self) -> None:
        """The refusal is about the parameter, not the route's fixed dimensionality."""
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(dimensions=3),), provider).embed(request())

        assert len(response.data[0].values) == 3

    async def test_a_later_route_cannot_expand_the_dimension_contract(self) -> None:
        """The primary establishes the contract; a fallback cannot add the parameter."""
        provider = FakeEmbeddingProvider(response_for)

        with pytest.raises(CapabilityError) as error:
            await build(
                (route(), route(model_id="fallback", index=1, supports_dimensions_param=True)),
                provider,
            ).embed(request(dimensions=4))

        assert error.value.details["parameter"] == "dimensions"
        assert provider.calls == []


# ---------------------------------------------------------------------------
# Fallbacks and the dimension contract
# ---------------------------------------------------------------------------


class TestFallbacksAndTheDimensionContract:
    """A fallback may preserve the advertised contract, never expand it."""

    async def test_a_fallback_that_cannot_honour_the_dimension_is_not_a_candidate(
        self,
    ) -> None:
        """The caller stated a dimension; only an accepting route may serve it."""
        provider = FakeEmbeddingProvider(
            UnavailableError("the primary is unavailable"), response_for
        )
        routes = (
            route(supports_dimensions_param=True, dimensions=4),
            route(model_id="fallback", index=1, dimensions=4),
        )

        with pytest.raises(UnavailableError):
            await build(routes, provider).embed(request(input=("only one",), dimensions=4))

        assert [route_.model_id for route_, _, _ in provider.calls] == ["embedding-model"]

    async def test_a_fallback_that_can_honour_the_dimension_serves_it(self) -> None:
        provider = FakeEmbeddingProvider(
            UnavailableError("the primary is unavailable"),
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4))
            ),
        )
        routes = (
            route(supports_dimensions_param=True, dimensions=4),
            route(model_id="fallback", index=1, supports_dimensions_param=True, dimensions=4),
        )

        response = await build(routes, provider).embed(
            request(input=("only one",), dimensions=4)
        )

        assert response.upstream_model == "fallback"
        assert len(response.data[0].values) == 4

    async def test_a_fallback_is_checked_against_its_own_declared_dimensionality(
        self,
    ) -> None:
        """The per-route rule: each route is held to the dimensionality it declares."""
        provider = FakeEmbeddingProvider(
            UnavailableError("the primary is unavailable"), response_for
        )
        routes = (
            route(dimensions=3),
            route(model_id="fallback", index=1, dimensions=4),
        )

        with pytest.raises(MalformedResponseError):
            await build(routes, provider).embed(request(input=("only one",)))

        assert [route_.model_id for route_, _, _ in provider.calls] == [
            "embedding-model",
            "fallback",
        ]

    async def test_a_fallback_with_no_declared_dimensionality_checks_none(self) -> None:
        """The fallback declares none, so none is checked — not the primary's value."""
        provider = FakeEmbeddingProvider(
            UnavailableError("the primary is unavailable"),
            lambda route_, request_, context: response_for(
                route_, request_, context, data=vectors((0.1, 0.2, 0.3, 0.4, 0.5))
            ),
        )
        routes = (route(dimensions=3), route(model_id="fallback", index=1))

        response = await build(routes, provider).embed(request(input=("only one",)))

        assert response.upstream_model == "fallback"
        assert len(response.data[0].values) == 5


# ---------------------------------------------------------------------------
# The encoding
# ---------------------------------------------------------------------------


class TestTheEncoding:
    """The gateway's answer is the float form, and nothing else is quietly substituted."""

    async def test_the_float_encoding_is_accepted(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        response = await build((route(),), provider).embed(request(encoding_format="float"))

        assert len(response.data) == 2

    async def test_no_stated_encoding_is_accepted(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        await build((route(),), provider).embed(request())

        assert len(provider.calls) == 1

    async def test_an_encoding_the_gateway_cannot_answer_in_is_refused(self) -> None:
        """Answering a base64 request with floats would be a silent downgrade."""
        provider = FakeEmbeddingProvider(response_for)

        with pytest.raises(CapabilityError) as error:
            await build((route(),), provider).embed(request(encoding_format="base64"))

        assert error.value.details["parameter"] == "encoding_format"
        assert provider.calls == []


# ---------------------------------------------------------------------------
# The reliability loop
# ---------------------------------------------------------------------------


class TestTheReliabilityLoop:
    """One deadline, one retry loop and one fallback walk, exactly as for text."""

    async def test_the_attempt_context_carries_the_budget_and_the_attempt_number(self) -> None:
        provider = FakeEmbeddingProvider(response_for)

        await build((route(),), provider, attempt_timeout=30.0).embed(request())

        (_, _, context) = provider.calls[0]
        assert context.attempt_number == 1
        assert context.timeout == pytest.approx(30.0, abs=0.1)

    async def test_a_retryable_failure_is_retried_on_the_same_route(self) -> None:
        provider = FakeEmbeddingProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )

        response = await build((route(),), provider, retry_count=1).embed(request())

        assert response.upstream_model == "embedding-model"
        assert [context.attempt_number for _, _, context in provider.calls] == [1, 2]

    async def test_a_rate_limit_moves_to_the_other_route_without_retrying(self) -> None:
        provider = FakeEmbeddingProvider(
            ResourceExhaustedError("the upstream rate limit was reached"), response_for
        )
        routes = (route(), route(model_id="fallback", index=1))

        response = await build(routes, provider, retry_count=2).embed(request())

        assert response.upstream_model == "fallback"
        assert len(provider.calls) == 2

    async def test_a_fallback_route_serves_when_the_primary_cannot(self) -> None:
        provider = FakeEmbeddingProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )
        routes = (route(), route(model_id="fallback", index=1))

        response = await build(routes, provider).embed(request())

        assert response.upstream_model == "fallback"
        assert [route_.model_id for route_, _, _ in provider.calls] == [
            "embedding-model",
            "fallback",
        ]

    async def test_a_fallback_without_the_capability_is_skipped(self) -> None:
        provider = FakeEmbeddingProvider(
            UnavailableError("the upstream is unavailable"), response_for
        )
        routes = (
            route(),
            route(model_id="chat-only", index=1, capabilities=frozenset({Capability.CHAT})),
            route(model_id="capable", index=2),
        )

        response = await build(routes, provider).embed(request())

        assert response.upstream_model == "capable"

    async def test_a_deterministic_failure_ends_the_invocation_where_it_happened(self) -> None:
        """No other provider can change the answer to a request that cannot succeed."""
        provider = FakeEmbeddingProvider(InvalidArgumentError("the request is invalid"))
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(InvalidArgumentError):
            await build(routes, provider, retry_count=2).embed(request())

        assert len(provider.calls) == 1

    async def test_a_gateway_defect_terminates_where_it_happened(self) -> None:
        """An internal error is the gateway's own defect: no route can fix it.

        The contrast with a malformed upstream answer: that fault lies in one
        route's answer, so the invocation walks to an eligible fallback, while
        a defect in the gateway's own code is neither retried on the route nor
        walked to the next.
        """
        provider = FakeEmbeddingProvider(
            InternalError("the gateway could not read its own answer")
        )
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(InternalError):
            await build(routes, provider, retry_count=2).embed(request())

        assert len(provider.calls) == 1

    async def test_the_last_attempts_failure_is_the_one_reported(self) -> None:
        provider = FakeEmbeddingProvider(
            UnavailableError("the first route is unavailable"),
            UnavailableError("the second route is unavailable"),
        )
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(UnavailableError) as error:
            await build(routes, provider).embed(request())

        assert "second route" in error.value.message

    async def test_a_cancelled_embeddings_invocation_propagates(self) -> None:
        provider = FakeEmbeddingProvider(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await build((route(),), provider).embed(request())

    async def test_a_cancelled_attempt_makes_no_further_attempt(self) -> None:
        provider = FakeEmbeddingProvider(asyncio.CancelledError())
        routes = (route(), route(model_id="fallback", index=1))

        with pytest.raises(asyncio.CancelledError):
            await build(routes, provider, retry_count=2).embed(request())

        assert len(provider.calls) == 1

    async def test_an_expired_deadline_refuses_the_invocation_without_calling_a_provider(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(reliability, "monotonic", _TickingClock(step=10.0))
        provider = FakeEmbeddingProvider(response_for)
        inference = build((route(),), provider, attempt_timeout=1.0)

        with pytest.raises(DeadlineExceededError) as error:
            await inference.embed(request())

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


class _ControllableClock:
    """A clock that runs forward only when a test advances it."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


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
    """The embeddings attempt holds a slot, and its wait is bounded like any other."""

    async def test_the_limiter_is_given_the_remaining_deadline_to_bound_the_wait(self) -> None:
        limiter = _RecordingLimiter()
        provider = FakeEmbeddingProvider(response_for)

        await build((route(),), provider, attempt_timeout=10.0, limiter=limiter).embed(request())

        (timeout,) = limiter.timeouts
        assert timeout is not None
        assert 0.0 < timeout <= 10.0

    async def test_an_invocation_with_no_deadline_asks_the_limiter_for_no_bound(self) -> None:
        limiter = _RecordingLimiter()
        provider = FakeEmbeddingProvider(response_for)

        await build((route(),), provider, limiter=limiter).embed(request())

        assert limiter.timeouts == [None]

    async def test_an_invocation_that_cannot_get_a_slot_in_time_fails_promptly(self) -> None:
        limiter = ConcurrencyLimiter(1)
        provider = FakeEmbeddingProvider(response_for)
        inference = build((route(),), provider, attempt_timeout=0.2, limiter=limiter)

        async with limiter.slot():
            with pytest.raises(DeadlineExceededError):
                await asyncio.wait_for(inference.embed(request()), timeout=5.0)

        assert provider.calls == []

    async def test_the_slot_is_released_when_the_attempt_ends(self) -> None:
        limiter = ConcurrencyLimiter(1)
        provider = FakeEmbeddingProvider(response_for)
        inference = build((route(),), provider, limiter=limiter)

        await inference.embed(request())
        await inference.embed(request())

        assert len(provider.calls) == 2
