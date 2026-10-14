"""The inference service: gateway-owned policy around one provider call (PRD 4.3, 9.5, 10, 11.2).

The service owns policy and the provider owns translation, so these tests are
about the policy: a public model ID resolves to configured routes before any
provider is called, an unknown ID is a model-not-found error that reaches no
provider, the capabilities the request needs are validated against the primary
route — the one whose capabilities the public model advertises — so a fallback
may preserve the advertised capabilities but never expand them, and the
provider's answer comes back stamped with the gateway's own facts.

They are also the reliability loop's tests (PRD 11.3 and 15): one deadline per
invocation that every attempt draws its budget from, retries on the same route
bounded by the configured count and gated on the gateway's own retryability
classification, a fallback walked only when a failure says another route is
worth trying — a deterministic request failure ends the invocation instead of
being sent to another provider — the last attempt's failure reported when no
route could serve, a backoff that grows with the route's own retries, a
concurrency slot whose wait is bounded by what is left of the deadline and whose
acquisition is charged to the attempt that follows it, and cancellation that
propagates instead of turning into an upstream failure — including the inference
library's own internal retry, which the gateway cannot switch off.
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

import model_gateway.inference as inference_module
import model_gateway.reliability as reliability
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    GenerationConfig,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    Message,
    ResponseFormat,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    ToolDefinition,
    Usage,
)
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    FailedPreconditionError,
    InternalError,
    InvalidArgumentError,
    ModelNotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.inference import InferenceService, InvocationState, _select_routes
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.litellm_provider import LiteLLMProvider
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter


def route(model_id: str = "upstream-model", **overrides) -> Route:
    """A standalone text route."""
    values = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.example/v1",
        "base_url": "https://upstream.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)


def request(**overrides) -> InvocationRequest:
    """A standalone canonical request."""
    values = {
        "model": "text-default",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
    }
    values.update(overrides)
    return InvocationRequest(**values)


def service(routes: tuple[Route, ...], spec: MockSpec | None = None) -> InferenceService:
    registry = ModelRegistry(routes)
    return InferenceService(registry, MockProvider(MockSpec() if spec is None else spec))


def full_route(**overrides) -> Route:
    values = {
        "capabilities": frozenset(
            {
                Capability.CHAT,
                Capability.TOOLS,
                Capability.VISION,
                Capability.STRUCTURED_OUTPUT,
            }
        ),
    }
    values.update(overrides)
    return route(**values)


def structured_config() -> GenerationConfig:
    """A generation config requiring a JSON shape the provider has to honour."""
    return GenerationConfig(
        response_format=ResponseFormat(
            mode=ResponseFormatMode.JSON_SCHEMA, schema={"type": "object"}
        )
    )


class _FixedProvider:
    """A provider that returns one fixed response, to see what the service stamps on."""

    def __init__(self, response: InvocationResponse) -> None:
        self._response = response
        self.calls: list[tuple[Route, InvocationRequest]] = []

    async def complete(
        self, route: Route, request: InvocationRequest, context
    ) -> InvocationResponse:
        self.calls.append((route, request))
        return self._response


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_invoke_returns_the_providers_response():
    result = await service((route(),), MockSpec(text="Recursion is a technique...")).invoke(
        request()
    )

    assert isinstance(result, InvocationResponse)
    assert result.text == "Recursion is a technique..."


async def test_invoke_calls_the_provider_with_the_primary_route():
    provider = MockProvider()
    registry = ModelRegistry((route("primary"), route("fallback")))
    inference = InferenceService(registry, provider)

    await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary"]


async def test_invoke_calls_the_provider_exactly_once():
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(), route("fallback"))), provider)

    await inference.invoke(request())

    assert len(provider.calls) == 1


async def test_invoke_passes_the_request_through_unchanged():
    provider = MockProvider()
    invocation = request(invocation_id="inv-1", model="text-default")
    inference = InferenceService(ModelRegistry((route(),)), provider)

    await inference.invoke(invocation)

    assert provider.calls[0][1] is invocation


async def test_invoke_reports_one_attempt_and_no_fallback():
    result = await service((route(),)).invoke(request())

    assert result.attempt_count == 1
    assert result.fallback_used is False


# ---------------------------------------------------------------------------
# Resolving the public model ID
# ---------------------------------------------------------------------------


async def test_invoke_resolves_an_unknown_model_id_to_model_not_found():
    inference = service((route(),))

    with pytest.raises(ModelNotFoundError) as error:
        await inference.invoke(request(model="gpt-sunburst"))

    assert error.value.reason is ErrorReason.MODEL_NOT_FOUND
    assert error.value.http_status == 404


async def test_invoke_reaches_no_provider_for_an_unknown_model_id():
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider)

    with pytest.raises(ModelNotFoundError):
        await inference.invoke(request(model="gpt-sunburst"))

    assert provider.calls == []


async def test_invoke_refuses_a_public_id_whose_operation_is_unconfigured():
    inference = service((route(),))

    with pytest.raises(ModelNotFoundError):
        await inference.invoke(request(model="image-default"))


async def test_invoke_never_passes_a_callers_model_name_to_the_provider():
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider)

    with pytest.raises(ModelNotFoundError):
        await inference.invoke(request(model="https://unconfigured.example/v1/internal-model"))

    assert provider.calls == []


# ---------------------------------------------------------------------------
# Capability enforcement
# ---------------------------------------------------------------------------


async def test_invoke_refuses_when_the_primary_route_cannot_chat():
    inference = service((route(capabilities=frozenset({Capability.TOOLS})),))

    with pytest.raises(CapabilityError):
        await inference.invoke(request())


async def test_invoke_refuses_tools_the_route_does_not_declare():
    inference = service((route(capabilities=frozenset({Capability.CHAT})),))

    with pytest.raises(CapabilityError):
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))


async def test_invoke_refuses_images_the_route_does_not_declare():
    conversation = request(
        messages=(
            Message(
                role=Role.USER,
                content=(
                    TextPart(text="what is in this image?"),
                    ImagePart(data="AAAA", media_type="image/png"),
                ),
            ),
        )
    )

    with pytest.raises(CapabilityError):
        await service((route(capabilities=frozenset({Capability.CHAT})),)).invoke(conversation)


async def test_invoke_accepts_tools_the_route_declares():
    result = await service(
        (full_route(),),
        MockSpec(tool_calls=(ToolCall(id="call_01", name="get_weather"),)),
    ).invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert result.tool_calls == (ToolCall(id="call_01", name="get_weather"),)


async def test_invoke_accepts_images_the_route_declares():
    conversation = request(
        messages=(
            Message(
                role=Role.USER,
                content=(
                    TextPart(text="what is in this image?"),
                    ImagePart(data="AAAA", media_type="image/png"),
                ),
            ),
        )
    )

    result = await service((full_route(),)).invoke(conversation)

    assert result.text == "what is in this image?"


async def test_the_capability_error_names_the_missing_capability():
    inference = service((route(capabilities=frozenset({Capability.CHAT})),))

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY
    assert error.value.details["capability"] == "tools"
    assert error.value.http_status == 422


async def test_invoke_reaches_no_provider_when_a_capability_is_missing():
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry((route(capabilities=frozenset({Capability.CHAT})),)), provider
    )

    with pytest.raises(CapabilityError):
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert provider.calls == []


# ---------------------------------------------------------------------------
# A provider failure
# ---------------------------------------------------------------------------


async def test_invoke_provides_a_provider_error_unchanged():
    failure = UnavailableError("the upstream provider is unavailable")
    inference = service((route(),), MockSpec(error=failure))

    with pytest.raises(UnavailableError) as error:
        await inference.invoke(request())

    assert error.value is failure


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


async def test_invoke_stamps_the_routing_fields_the_gateway_owns():
    provider = _FixedProvider(
        InvocationResponse(
            model="someone-elses-model",
            upstream_model="someone-elses-upstream",
            attempt_count=4,
            fallback_used=True,
        )
    )
    inference = InferenceService(ModelRegistry((route(),)), provider)

    result = await inference.invoke(request())

    assert result.model == "text-default"
    assert result.upstream_model == "upstream-model"
    assert result.attempt_count == 1
    assert result.fallback_used is False


async def test_invoke_fills_an_invocation_id_the_provider_did_not_echo():
    provider = _FixedProvider(
        InvocationResponse(model="text-default", upstream_model="upstream-model")
    )
    inference = InferenceService(ModelRegistry((route(),)), provider)

    result = await inference.invoke(request(invocation_id="inv-1"))

    assert result.invocation_id == "inv-1"


async def test_invoke_always_stamps_the_gateways_own_invocation_id():
    """A provider does not get to name the invocation in the caller's record."""
    provider = _FixedProvider(
        InvocationResponse(
            model="text-default",
            upstream_model="upstream-model",
            invocation_id="a-provider-chosen-id",
        )
    )
    inference = InferenceService(ModelRegistry((route(),)), provider)

    result = await inference.invoke(request(invocation_id="inv-1"))

    assert result.invocation_id == "inv-1"


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def test_invoke_refuses_a_schema_the_route_does_not_declare():
    inference = service((route(capabilities=frozenset({Capability.CHAT})),))

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(generation_config=structured_config()))

    assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY
    assert error.value.details["capability"] == "structured-output"


async def test_invoke_accepts_a_schema_the_route_declares():
    result = await service((full_route(),)).invoke(request(generation_config=structured_config()))

    assert result.text == "What is the weather in Paris?"


async def test_a_json_object_response_format_needs_the_capability_too():
    config = GenerationConfig(response_format=ResponseFormat(mode=ResponseFormatMode.JSON_OBJECT))
    inference = service((route(capabilities=frozenset({Capability.CHAT})),))

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(generation_config=config))

    assert error.value.details["capability"] == "structured-output"


async def test_a_text_response_format_asks_for_nothing_beyond_chat():
    config = GenerationConfig(response_format=ResponseFormat(mode=ResponseFormatMode.TEXT))

    result = await service((route(),)).invoke(request(generation_config=config))

    assert result.text == "What is the weather in Paris?"


async def test_an_unset_response_format_asks_for_nothing_beyond_chat():
    result = await service((route(),)).invoke(request(generation_config=GenerationConfig()))

    assert result.text == "What is the weather in Paris?"


async def test_a_structured_output_refusal_reaches_no_provider():
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry((route(capabilities=frozenset({Capability.CHAT})),)), provider
    )

    with pytest.raises(CapabilityError):
        await inference.invoke(request(generation_config=structured_config()))

    assert provider.calls == []


# ---------------------------------------------------------------------------
# A fallback may preserve the advertised capabilities, never expand them
# ---------------------------------------------------------------------------


async def test_a_fallback_cannot_add_tools_the_primary_does_not_advertise():
    """The public model advertises the primary route's capabilities, and only those."""
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT})),
                route("fallback", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
            )
        ),
        provider,
    )

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert error.value.reason is ErrorReason.UNSUPPORTED_CAPABILITY
    assert error.value.details["capability"] == "tools"
    assert provider.calls == []


async def test_the_refusal_follows_the_capabilities_the_registry_advertises():
    """What `GET /v1/models` publishes for the model is what a request is checked against."""
    routes = (
        route("primary", capabilities=frozenset({Capability.CHAT})),
        route("fallback", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
    )
    registry = ModelRegistry(routes)
    inference = InferenceService(registry, MockProvider())

    advertised = registry.get("text-default").capabilities

    assert advertised == frozenset({Capability.CHAT})
    with pytest.raises(CapabilityError):
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))


async def test_a_fallback_cannot_add_the_output_shape_the_primary_does_not_advertise():
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT})),
                route(
                    "fallback",
                    capabilities=frozenset({Capability.CHAT, Capability.STRUCTURED_OUTPUT}),
                ),
            )
        ),
        provider,
    )

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(generation_config=structured_config()))

    assert error.value.details["capability"] == "structured-output"
    assert provider.calls == []


async def test_a_fallback_cannot_add_vision_the_primary_does_not_advertise():
    conversation = request(
        messages=(
            Message(
                role=Role.USER,
                content=(
                    TextPart(text="what is in this image?"),
                    ImagePart(data="AAAA", media_type="image/png"),
                ),
            ),
        )
    )
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT})),
                route("fallback", capabilities=frozenset({Capability.CHAT, Capability.VISION})),
            )
        ),
        provider,
    )

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(conversation)

    assert error.value.details["capability"] == "vision"
    assert provider.calls == []


async def test_a_request_the_advertised_model_cannot_serve_reaches_no_route():
    """Two routes that cannot serve the request refuse it without calling either."""
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT})),
                route("fallback", capabilities=frozenset({Capability.CHAT})),
            )
        ),
        provider,
    )

    with pytest.raises(CapabilityError):
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert provider.calls == []


async def test_the_refusal_names_the_capability_the_advertised_model_lacks():
    inference = service(
        (
            route("primary", capabilities=frozenset({Capability.CHAT})),
            route("fallback", capabilities=frozenset({Capability.CHAT, Capability.VISION})),
        )
    )

    with pytest.raises(CapabilityError) as error:
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert error.value.details["capability"] == "tools"


# ---------------------------------------------------------------------------
# Which routes may serve an invocation
# ---------------------------------------------------------------------------


async def test_the_primary_route_is_still_used_first_when_it_is_eligible():
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
                route("fallback", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
            )
        ),
        provider,
    )

    result = await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert [call[0].model_id for call in provider.calls] == ["primary"]
    assert result.fallback_used is False


async def test_a_fallback_that_cannot_honour_the_request_is_not_a_candidate():
    """Primary, a chat-only fallback and a capable fallback: only the capable one may serve."""
    primary = route("primary", capabilities=frozenset({Capability.CHAT, Capability.TOOLS}))
    chat_only = route("chat-only", capabilities=frozenset({Capability.CHAT}))
    capable = route("capable", capabilities=frozenset({Capability.CHAT, Capability.TOOLS}))

    plan = _select_routes(
        (primary, chat_only, capable), request(tools=(ToolDefinition(name="get_weather"),))
    )

    assert plan.primary is primary
    assert plan.fallbacks == (capable,)


async def test_a_fallback_that_preserves_the_advertised_capabilities_is_a_candidate():
    primary = full_route(model_id="primary")
    fallback = full_route(model_id="fallback")

    plan = _select_routes(
        (primary, fallback), request(tools=(ToolDefinition(name="get_weather"),))
    )

    assert plan.primary is primary
    assert plan.fallbacks == (fallback,)


async def test_a_model_with_one_route_has_no_fallback_candidates():
    primary = route("primary")

    plan = _select_routes((primary,), request())

    assert plan.primary is primary
    assert plan.fallbacks == ()


async def test_the_primary_route_is_attempted_before_any_fallback():
    """The primary route is tried first; only the eligible fallback follows it."""
    provider = MockProvider(MockSpec(error=UnavailableError("the primary route is down")))
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
                route("chat-only", capabilities=frozenset({Capability.CHAT})),
                route("capable", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
            )
        ),
        provider,
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert [call[0].model_id for call in provider.calls] == ["primary", "capable"]


# ---------------------------------------------------------------------------
# What `fallback_used` means
# ---------------------------------------------------------------------------


async def test_the_primary_route_serving_the_invocation_is_not_a_fallback():
    """Configured fallbacks behind the primary do not make the primary a fallback."""
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry((full_route(model_id="primary"), full_route(model_id="fallback"))),
        provider,
    )

    result = await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert result.upstream_model == "primary"
    assert result.attempt_count == 1
    assert result.fallback_used is False


async def test_an_attempt_on_a_fallback_route_is_what_sets_the_flag():
    """A non-primary route serving the invocation after the primary path could not."""
    state = InvocationState().with_attempt()

    assert state.fallback_used is False
    assert state.with_attempt(on_fallback=True).fallback_used is True


async def test_the_flag_stays_set_once_a_fallback_route_has_been_used():
    """Sticky, because one route other than the primary is enough to have fallen back."""
    state = InvocationState().with_attempt(on_fallback=True).with_attempt()

    assert (state.attempt_count, state.fallback_used) == (2, True)


async def test_a_response_never_claims_a_fallback_capability_routing_did_not_take():
    """A request the primary route serves reports no fallback, however many routes exist."""
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry(
            (
                route("primary", capabilities=frozenset({Capability.CHAT})),
                route("fallback", capabilities=frozenset({Capability.CHAT})),
            )
        ),
        provider,
    )

    result = await inference.invoke(request())

    assert result.attempt_count == 1
    assert result.fallback_used is False


# ---------------------------------------------------------------------------
# The attempt the provider is handed
# ---------------------------------------------------------------------------


async def test_the_attempt_context_carries_the_invocation_and_the_attempt_number():
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider)

    await inference.invoke(request(invocation_id="inv-1"))

    context = provider.calls[0][2]
    assert context.invocation_id == "inv-1"
    assert context.attempt_number == 1


async def test_the_attempt_context_carries_the_budget_the_gateway_set():
    """The budget comes from the invocation deadline, so it is what is left of the timeout."""
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider, attempt_timeout=12.5)

    await inference.invoke(request())

    assert provider.calls[0][2].timeout == pytest.approx(12.5, abs=0.5)


async def test_an_unset_budget_is_no_budget_rather_than_zero():
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider)

    await inference.invoke(request())

    assert provider.calls[0][2].timeout is None


async def test_the_attempt_context_carries_no_routing_policy():
    """The route travels beside the context, never inside it (PRD section 9.2)."""
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider)

    await inference.invoke(request())

    assert set(vars(provider.calls[0][2])) == {"invocation_id", "attempt_number", "timeout"}


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan")])
def test_a_budget_no_attempt_could_use_is_refused_at_construction(budget):
    with pytest.raises(ValueError, match="attempt_timeout must be a positive number"):
        InferenceService(ModelRegistry((route(),)), MockProvider(), attempt_timeout=budget)


# ---------------------------------------------------------------------------
# One deadline for the invocation
# ---------------------------------------------------------------------------


class _ControllableClock:
    """A monotonic clock a test moves forward by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

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


class _BudgetSpendingProvider:
    """A provider that spends part of the invocation's budget and then fails."""

    def __init__(self, clock: _ControllableClock, *, spend: float, error: BaseException) -> None:
        self._clock = clock
        self._spend = spend
        self._error = error
        self.calls: list[tuple[Route, InvocationRequest, AttemptContext]] = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        self._clock.advance(self._spend)
        raise self._error


async def test_an_expired_deadline_refuses_the_invocation_without_calling_a_provider(monkeypatch):
    monkeypatch.setattr(reliability, "monotonic", _TickingClock(step=10.0))
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider, attempt_timeout=1.0)

    with pytest.raises(DeadlineExceededError) as error:
        await inference.invoke(request())

    assert provider.calls == []
    assert error.value.reason is ErrorReason.DEADLINE_EXCEEDED
    assert error.value.http_status == 504
    assert "upstream.example" not in str(error.value)


async def test_no_retry_is_started_once_the_deadline_is_spent(monkeypatch):
    """The remaining budget cannot cover another attempt, so none is made."""
    clock = _ControllableClock()
    monkeypatch.setattr(reliability, "monotonic", clock)
    provider = _BudgetSpendingProvider(clock, spend=5.0, error=UnavailableError("too slow"))
    inference = InferenceService(
        ModelRegistry((route(),)), provider, attempt_timeout=1.0, retry_count=2
    )

    with pytest.raises(DeadlineExceededError) as error:
        await inference.invoke(request())

    assert len(provider.calls) == 1
    assert error.value.details["attempts"] == "1"


async def test_every_attempt_draws_its_budget_from_the_one_deadline():
    """A retry is handed what is left of the invocation's budget, not a fresh timeout."""
    provider = MockProvider(
        (MockSpec(error=UnavailableError("the provider is down")), MockSpec(text="recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route(),)), provider, attempt_timeout=30.0, retry_count=1
    )

    result = await inference.invoke(request())

    first, second = provider.calls[0][2].timeout, provider.calls[1][2].timeout
    assert 0 < second <= first <= 30.0
    assert result.attempt_count == 2


# ---------------------------------------------------------------------------
# Retries on the same route
# ---------------------------------------------------------------------------


async def test_a_retryable_failure_is_retried_on_the_same_route():
    provider = MockProvider(
        (MockSpec(error=UnavailableError("the provider is down")), MockSpec(text="recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=1, retry_backoff_ms=0
    )

    result = await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["upstream-model", "upstream-model"]
    assert result.text == "recovered"
    assert result.attempt_count == 2
    assert result.fallback_used is False


async def test_retries_are_bounded_by_the_configured_count():
    provider = MockProvider(MockSpec(error=UnavailableError("still down")))
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=2, retry_backoff_ms=0
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert len(provider.calls) == 3


async def test_no_wait_happens_after_the_last_attempt_a_route_may_make(monkeypatch):
    """A backoff is only waited when a retry will actually follow it."""
    waits: list[int] = []

    async def recording_back_off(attempt: int, base_ms: float, deadline) -> bool:
        waits.append(attempt)
        return True

    monkeypatch.setattr(inference_module, "_back_off", recording_back_off)
    provider = MockProvider(MockSpec(error=UnavailableError("still down")))
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=2, retry_backoff_ms=5000
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert waits == [1, 2]
    assert len(provider.calls) == 3


async def test_a_failure_a_repeat_cannot_change_is_not_retried():
    failure = InvalidArgumentError("the upstream rejected the request")
    provider = MockProvider(MockSpec(error=failure))
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=3, retry_backoff_ms=0
    )

    with pytest.raises(InvalidArgumentError) as error:
        await inference.invoke(request())

    assert error.value is failure
    assert len(provider.calls) == 1


async def test_the_attempt_number_advances_with_each_retry():
    provider = MockProvider(
        (MockSpec(error=UnavailableError("down")), MockSpec(text="recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=1, retry_backoff_ms=0
    )

    await inference.invoke(request())

    assert [call[2].attempt_number for call in provider.calls] == [1, 2]


async def test_a_retry_waits_the_configured_backoff_before_the_next_attempt():
    provider = MockProvider(
        (MockSpec(error=UnavailableError("down")), MockSpec(text="recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route(),)), provider, retry_count=1, retry_backoff_ms=200
    )

    started = asyncio.get_running_loop().time()
    await inference.invoke(request())
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed >= 0.15


async def test_a_fallbacks_first_retry_backs_off_one_base_not_the_invocations_count(monkeypatch):
    """The backoff grows with the route's own retries, not with the invocation's attempts.

    Two routes, one retry each: the invocation's fourth attempt is the fallback's
    first retry, and it waits the base delay — one step of the schedule. A
    schedule driven by the global attempt count would hand it the third step, four
    times the base, growing exponentially with failures the fallback never made.
    """
    waits: list[int] = []

    async def recording_back_off(attempt: int, base_ms: float, deadline) -> bool:
        waits.append(attempt)
        return True

    monkeypatch.setattr(inference_module, "_back_off", recording_back_off)
    provider = MockProvider(MockSpec(error=UnavailableError("still down")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        retry_count=1,
        retry_backoff_ms=500,
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == [
        "primary",
        "primary",
        "fallback",
        "fallback",
    ]
    assert waits == [1, 1]


async def test_a_backoff_that_cannot_fit_the_deadline_ends_the_route_not_the_invocation():
    """A wait that would outlive the deadline is refused, so the route is used up.

    The invocation still has budget; what it does not have is enough of it to
    sleep through a backoff and then make the retry. The retry is therefore not
    made, and the invocation goes on to the route that can still be tried inside
    the time left, rather than sleeping until the deadline expired and failing
    there anyway.
    """
    provider = MockProvider(MockSpec(error=UnavailableError("the provider is down")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        attempt_timeout=1.0,
        retry_count=2,
        retry_backoff_ms=5000,
    )

    started = asyncio.get_running_loop().time()
    with pytest.raises(UnavailableError):
        await inference.invoke(request())
    elapsed = asyncio.get_running_loop().time() - started

    assert [call[0].model_id for call in provider.calls] == ["primary", "fallback"]
    assert elapsed < 2.0


async def test_no_retry_is_made_when_none_is_configured():
    """A service built without a retry policy makes one attempt per route."""
    provider = MockProvider(
        (MockSpec(error=UnavailableError("fail once")), MockSpec(text="recovered"))
    )
    inference = InferenceService(ModelRegistry((route(),)), provider)

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


async def test_a_fallback_serves_the_invocation_when_the_primary_path_fails():
    provider = MockProvider(
        (
            MockSpec(error=UnavailableError("the primary route is down")),
            MockSpec(text="the fallback answered"),
        )
    )
    inference = InferenceService(ModelRegistry((route("primary"), route("fallback"))), provider)

    result = await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary", "fallback"]
    assert result.upstream_model == "fallback"
    assert result.fallback_used is True
    assert result.attempt_count == 2


async def test_a_route_is_retried_before_a_fallback_is_tried():
    provider = MockProvider(
        (MockSpec(error=UnavailableError("primary down")), MockSpec(text="primary recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        retry_count=1,
        retry_backoff_ms=0,
    )

    result = await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary", "primary"]
    assert result.upstream_model == "primary"
    assert result.fallback_used is False


async def test_a_deterministic_failure_terminates_the_invocation_at_that_route():
    """A request the upstream rejected as invalid is not sent to another provider.

    Repeating the call cannot change the answer, and neither can changing the
    provider: the answer is a property of the request. Walking the rest of the
    fallbacks would spend their capacity on a request that cannot succeed, and
    would make the caller's answer depend on which providers happen to be
    configured. So the invocation ends where it happened, with that failure.
    """
    failure = InvalidArgumentError("the primary rejected the request")
    provider = MockProvider(
        (
            MockSpec(error=failure),
            MockSpec(text="the fallback answered"),
        )
    )
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        retry_count=2,
        retry_backoff_ms=0,
    )

    with pytest.raises(InvalidArgumentError) as error:
        await inference.invoke(request())

    assert error.value is failure
    assert [call[0].model_id for call in provider.calls] == ["primary"]


@pytest.mark.parametrize(
    "failure",
    [
        InvalidArgumentError("the upstream rejected the request"),
        UnauthenticatedError("the upstream refused the route's credentials"),
        PermissionDeniedError("the upstream denied the request"),
        CapabilityError("the upstream cannot honour the request"),
        FailedPreconditionError("the upstream is not in a state to serve this"),
        InternalError("the gateway could not complete the request"),
    ],
)
async def test_every_deterministic_category_terminates_rather_than_falling_back(failure):
    """The whole deterministic group, not just invalid arguments."""
    provider = MockProvider((MockSpec(error=failure), MockSpec(text="the fallback answered")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        retry_count=2,
        retry_backoff_ms=0,
    )

    with pytest.raises(type(failure)):
        await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary"]


async def test_a_route_local_failure_falls_back_without_being_retried_on_that_route():
    """An upstream rate limit is worth another route but not another attempt here.

    This is the case the two questions exist to separate: the failure is not
    retryable on the route that answered, and it is still fallback-eligible, so
    the invocation moves on rather than repeating the call or giving up.
    """
    refusal = ResourceExhaustedError(
        "the upstream is rate limiting this route", reason=ErrorReason.RATE_LIMITED
    )
    provider = MockProvider((MockSpec(error=refusal), MockSpec(text="the fallback answered")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))),
        provider,
        retry_count=3,
        retry_backoff_ms=0,
    )

    result = await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary", "fallback"]
    assert result.upstream_model == "fallback"
    assert result.fallback_used is True
    assert result.attempt_count == 2


async def test_the_gateways_own_concurrency_refusal_terminates_the_invocation():
    """Every route shares the gateway's limiter, so another route is refused identically."""
    refusal = ResourceExhaustedError(
        "the gateway's upstream concurrency limit is saturated",
        reason=ErrorReason.CONCURRENCY_LIMIT_EXCEEDED,
    )
    provider = MockProvider((MockSpec(error=refusal), MockSpec(text="the fallback answered")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))), provider, retry_count=3
    )

    with pytest.raises(ResourceExhaustedError) as error:
        await inference.invoke(request())

    assert error.value.reason is ErrorReason.CONCURRENCY_LIMIT_EXCEEDED
    assert [call[0].model_id for call in provider.calls] == ["primary"]


async def test_the_last_attempts_failure_is_the_one_reported():
    """The most recent failure is the one that says why no route could serve."""
    fallback_failure = InvalidArgumentError("the fallback rejected the request")
    provider = MockProvider(
        (
            MockSpec(error=UnavailableError("the primary route is down")),
            MockSpec(error=fallback_failure),
        )
    )
    inference = InferenceService(ModelRegistry((route("primary"), route("fallback"))), provider)

    with pytest.raises(InvalidArgumentError) as error:
        await inference.invoke(request())

    assert error.value is fallback_failure
    assert [call[0].model_id for call in provider.calls] == ["primary", "fallback"]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class _ConcurrencyProbe:
    """A provider that records how many of its calls were in flight at once."""

    def __init__(self) -> None:
        self.calls = 0
        self.peak = 0
        self._in_flight = 0

    async def complete(self, route, request, context):
        self.calls += 1
        self._in_flight += 1
        self.peak = max(self.peak, self._in_flight)
        try:
            await asyncio.sleep(0)
            return InvocationResponse(model=request.model, upstream_model=route.model_id)
        finally:
            self._in_flight -= 1


class _RecordingLimiter:
    """A stub limiter that records when a slot is taken, given back, and with what bound."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.timeouts: list[float | None] = []

    @asynccontextmanager
    async def slot(self, timeout: float | None = None):
        self.events.append("enter")
        self.timeouts.append(timeout)
        try:
            yield
        finally:
            self.events.append("exit")


async def test_simultaneous_attempts_are_bounded_by_the_injected_limiter():
    provider = _ConcurrencyProbe()
    inference = InferenceService(
        ModelRegistry((route(),)), provider, limiter=ConcurrencyLimiter(2)
    )

    await asyncio.gather(*(inference.invoke(request()) for _ in range(6)))

    assert provider.calls == 6
    assert provider.peak == 2


async def test_the_configured_maximum_builds_the_limiter():
    provider = _ConcurrencyProbe()
    inference = InferenceService(ModelRegistry((route(),)), provider, max_concurrent_requests=1)

    await asyncio.gather(*(inference.invoke(request()) for _ in range(4)))

    assert provider.calls == 4
    assert provider.peak == 1


async def test_the_concurrency_slot_is_released_before_the_backoff_wait(monkeypatch):
    """A caller waiting to retry holds no slot another caller could be using."""
    limiter = _RecordingLimiter()

    async def recording_back_off(attempt: int, base_ms: float, deadline) -> bool:
        limiter.events.append("wait")
        return True

    monkeypatch.setattr(inference_module, "_back_off", recording_back_off)
    provider = MockProvider(
        (MockSpec(error=UnavailableError("down")), MockSpec(text="recovered"))
    )
    inference = InferenceService(
        ModelRegistry((route(),)),
        provider,
        retry_count=1,
        retry_backoff_ms=1000,
        limiter=limiter,
    )

    await inference.invoke(request())

    assert limiter.events == ["enter", "exit", "wait", "enter", "exit"]


async def test_the_limiter_is_given_the_remaining_deadline_to_bound_the_wait():
    """The slot wait is bounded by the invocation's deadline, not by the slot's holder."""
    limiter = _RecordingLimiter()
    inference = InferenceService(
        ModelRegistry((route(),)), MockProvider(), attempt_timeout=10.0, limiter=limiter
    )

    await inference.invoke(request())

    (timeout,) = limiter.timeouts
    assert timeout is not None
    assert 0.0 < timeout <= 10.0


async def test_an_invocation_with_no_deadline_asks_the_limiter_for_no_bound():
    limiter = _RecordingLimiter()
    inference = InferenceService(ModelRegistry((route(),)), MockProvider(), limiter=limiter)

    await inference.invoke(request())

    assert limiter.timeouts == [None]


async def test_an_invocation_that_cannot_get_a_slot_in_time_fails_promptly():
    """A caller queued behind a held slot gives up when its own deadline passes.

    Without the bound this invocation would wait for the holder, however long
    that took, and only then discover that its own deadline had been spent. The
    wait here is wrapped in a timeout of its own so a regression fails the test
    instead of hanging it.
    """
    limiter = ConcurrencyLimiter(1)
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry((route(),)), provider, attempt_timeout=0.2, limiter=limiter
    )

    async with limiter.slot():
        with pytest.raises(DeadlineExceededError):
            await asyncio.wait_for(inference.invoke(request()), timeout=5.0)

    assert provider.calls == []


class _ClockAdvancingLimiter:
    """A stub limiter that spends invocation budget while a caller waits for a slot."""

    def __init__(self, clock: _ControllableClock, spend: float) -> None:
        self._clock = clock
        self._spend = spend

    @asynccontextmanager
    async def slot(self, timeout: float | None = None):
        self._clock.advance(self._spend)
        yield


async def test_the_attempt_budget_is_recomputed_after_the_slot_is_acquired(monkeypatch):
    """Waiting for a slot consumes invocation budget, so the attempt gets what is left."""
    clock = _ControllableClock()
    monkeypatch.setattr(reliability, "monotonic", clock)
    provider = MockProvider()
    inference = InferenceService(
        ModelRegistry((route(),)),
        provider,
        attempt_timeout=10.0,
        limiter=_ClockAdvancingLimiter(clock, spend=4.0),
    )

    await inference.invoke(request())

    assert provider.calls[0][2].timeout == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class _BlockingProvider:
    """A provider whose call blocks until the invocation is cancelled."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, InvocationRequest, AttemptContext]] = []
        self.started = asyncio.Event()

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        self.started.set()
        await asyncio.Event().wait()


class _CancellingProvider:
    """A provider whose call raises :exc:`asyncio.CancelledError`."""

    async def complete(self, route, request, context):
        raise asyncio.CancelledError


async def test_cancelling_an_invocation_stops_further_attempts_and_frees_the_slot():
    provider = _BlockingProvider()
    limiter = ConcurrencyLimiter(1)
    inference = InferenceService(
        ModelRegistry((route(),)),
        provider,
        retry_count=3,
        retry_backoff_ms=0,
        limiter=limiter,
    )

    task = asyncio.create_task(inference.invoke(request()))
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(provider.calls) == 1

    async def take_the_only_slot() -> None:
        async with limiter.slot():
            pass

    await asyncio.wait_for(take_the_only_slot(), timeout=1.0)


async def test_cancellation_is_never_turned_into_an_upstream_failure():
    inference = InferenceService(
        ModelRegistry((route(),)), _CancellingProvider(), retry_count=2, retry_backoff_ms=0
    )

    with pytest.raises(asyncio.CancelledError):
        await inference.invoke(request())


# The inference library has internal retry paths no supported knob disables (PRD
# section 11.3), so "the gateway made no further attempt" is not the same claim
# as "no further wire request was made". These two tests are about the second
# claim, and they are driven through the real binding and the real library over a
# fake wire: the transport below is injected as the binding's inner transport, so
# every request is recorded and no socket is ever opened.
#
# What they prove, and what they do not. They prove that on the pinned library,
# for the ``anthropic`` dialect — whose HTTP path retries a connection error
# inside the library, which is the internal retry these tests are about — a
# cancellation delivered while a wire request is in flight is not swallowed by
# that retry, and no further wire request follows it. They do not exercise the
# library's other internal retry path, the message-reformat loop that resends on
# a malformed-request response, because that one is triggered by a response body
# rather than by a connection error; and they do not exercise a real socket. Both
# omissions are covered by the same mechanism rather than by a different one: the
# retry loops catch the library's own exception hierarchy, and cancellation
# arrives as :exc:`asyncio.CancelledError`, a :exc:`BaseException` those loops do
# not name.

_UPSTREAM_ADDRESS = "93.184.216.34"


def _stub_resolver(host: str) -> tuple[str, ...]:
    """Resolve every hostname to one public address, so no test touches DNS."""
    return (_UPSTREAM_ADDRESS,)


class _FailingWireTransport(httpx.AsyncBaseTransport):
    """A wire that refuses every connection, recording each request it was asked to send."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise httpx.ConnectError("this wire refuses every connection", request=request)


class _BlockingWireTransport(httpx.AsyncBaseTransport):
    """A wire that records every request and holds each one open until it is released."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        raise httpx.ConnectError("this wire refuses every connection", request=request)


def _gateway_over(transport: httpx.AsyncBaseTransport, **overrides) -> InferenceService:
    """A gateway invocation that reaches the real inference library and a fake wire.

    The route's dialect is ``anthropic`` because that is the HTTP path the pinned
    library retries a connection error on by itself.
    """
    binding = LiteLLMProvider(resolver=_stub_resolver, inner_factory=lambda: transport)
    return InferenceService(
        ModelRegistry((route(format="anthropic"),)), binding, **overrides
    )


async def test_the_librarys_internal_retry_does_reach_the_wire_when_nothing_cancels():
    """The control for the cancellation test below.

    Without this, "no second wire request after cancellation" would prove
    nothing: it could equally be a failure the library never retries at all.
    This pins that the connection error used below is one the library retries on
    its own, so exactly one wire request is the gateway's and the second is the
    library's.
    """
    wire = _FailingWireTransport()

    with pytest.raises(UnavailableError):
        await _gateway_over(wire).invoke(request())

    assert len(wire.requests) == 2


async def test_cancelling_the_invocation_stops_the_librarys_internal_retry():
    """A cancelled invocation makes no further wire request, retry paths included.

    The gateway's own retry loop is already covered above; this is the second
    loop, the one inside the inference library that the gateway cannot switch
    off. The invocation is cancelled with a wire request in flight, which is the
    state the library's connection-error retry would resume from.
    """
    wire = _BlockingWireTransport()
    gateway = _gateway_over(wire, retry_count=3, retry_backoff_ms=0)

    invocation = asyncio.create_task(gateway.invoke(request()))
    await asyncio.wait_for(wire.started.wait(), timeout=5.0)
    assert len(wire.requests) == 1

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    wire.release.set()
    for _ in range(50):
        await asyncio.sleep(0)

    assert len(wire.requests) == 1


# ---------------------------------------------------------------------------
# The response the service returns
# ---------------------------------------------------------------------------


async def test_invoke_returns_the_usage_the_provider_reported():
    usage = Usage(input_tokens=25, output_tokens=42, total_tokens=67)
    result = await service((route(),), MockSpec(usage=usage)).invoke(request())

    assert result.usage == usage


async def test_invoke_returns_the_finish_reason_the_provider_reported():
    result = await service(
        (route(),), MockSpec(finish_reason=FinishReason.LENGTH, raw_finish_reason="max_tokens")
    ).invoke(request())

    assert result.finish_reason is FinishReason.LENGTH
    assert result.raw_finish_reason == "max_tokens"
