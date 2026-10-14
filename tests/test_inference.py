"""The inference service: gateway-owned policy around one provider call (PRD 9, 9.5, 10, 11.3).

The service owns policy and the provider owns translation, so these tests are
about the policy: a public model ID resolves to configured routes before any
provider is called, an unknown ID is a model-not-found error that reaches no
provider, the capabilities the request needs decide which configured route may
serve it, and the provider's answer comes back stamped with the gateway's own
facts.

They also pin the boundary of this slice deliberately: there are no retries, no
fallbacks and no deadline policy yet, and a provider that fails on the route
selection chose stays failed. The reliability slice owns that loop; these tests
are what it must not change.
"""

import pytest

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
from model_gateway.errors import CapabilityError, ErrorReason, ModelNotFoundError, UnavailableError
from model_gateway.inference import InferenceService
from model_gateway.providers.mock import MockProvider, MockSpec
from model_gateway.registry import ModelRegistry


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
# Route selection
# ---------------------------------------------------------------------------


async def test_an_ineligible_primary_is_skipped_for_an_eligible_fallback():
    """A route that cannot honour the request is passed over, not failed on."""
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

    result = await inference.invoke(request(tools=(ToolDefinition(name="get_weather"),)))

    assert [call[0].model_id for call in provider.calls] == ["fallback"]
    assert result.upstream_model == "fallback"
    assert result.attempt_count == 1
    assert result.fallback_used is True


async def test_selection_skips_a_route_that_cannot_honour_the_output_shape():
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

    result = await inference.invoke(request(generation_config=structured_config()))

    assert [call[0].model_id for call in provider.calls] == ["fallback"]
    assert result.fallback_used is True


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


async def test_an_ineligible_route_is_skipped_rather_than_fatal():
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


async def test_the_refusal_names_the_first_configured_routes_missing_capability():
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
    provider = MockProvider()
    inference = InferenceService(ModelRegistry((route(),)), provider, attempt_timeout=12.5)

    await inference.invoke(request())

    assert provider.calls[0][2].timeout == 12.5


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
# The reliability boundary of this slice
# ---------------------------------------------------------------------------


async def test_invoke_does_not_retry_a_failing_provider():
    """The retry loop is the later slice's; a failure here stays a failure."""
    inference = service(
        (route(),),
        MockSpec(error=UnavailableError("the first attempt failed")),
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request())


async def test_invoke_does_not_fall_back_to_a_second_route():
    """The fallback walk is the later slice's; the primary route is the route."""
    provider = MockProvider(MockSpec(error=UnavailableError("the primary route is down")))
    inference = InferenceService(
        ModelRegistry((route("primary"), route("fallback"))), provider
    )

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert [call[0].model_id for call in provider.calls] == ["primary"]


async def test_a_provider_that_fails_once_then_succeeds_is_not_retried():
    """Even a retry that would succeed is not made: the retry loop is not this slice's."""
    provider = MockProvider(
        (MockSpec(error=UnavailableError("fail once")), MockSpec(text="recovered"))
    )
    inference = InferenceService(ModelRegistry((route(),)), provider)

    with pytest.raises(UnavailableError):
        await inference.invoke(request())

    assert len(provider.calls) == 1


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
