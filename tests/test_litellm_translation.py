"""Tests for the transport-agnostic LiteLLM translation (PRD sections 9.1, 9.3, 9.5, 9.6, 13).

Every fixture here is built from the real LiteLLM types the installed library
defines — by constructing :class:`litellm.types.utils.ModelResponse` and its
records directly, or by parsing a realistic upstream payload through LiteLLM's
own :func:`convert_to_model_response_object` — so the tests exercise the shape a
real provider response has once LiteLLM has decoded it, not a handcrafted
dictionary that merely resembles one.

No test performs network I/O: the fixtures are local objects, and the error
mapping is exercised by constructing LiteLLM's exception classes in memory.

The preservation rule of PRD section 9.1 is tested on both sides. On the
request side, a canonical element with no upstream equivalent (a refusal part)
is rejected rather than dropped. On the response side, the four-way
classification is pinned: a semantic field the canonical model cannot represent
— a log-probability block, a reasoning trace, a provider-specific annotation —
is rejected rather than silently trimmed, an unknown provider-specific field
is refused until it is classified, and a backend fingerprint is classified
non-semantic operational metadata and deliberately ignored. The fixtures
deliberately carry those extra fields so a translator that drops a semantic,
or fails one it should ignore, fails the test.
"""

import httpx
import litellm
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import (
    ChoiceLogprobs,
    Choices,
    ModelResponse,
)
from litellm.types.utils import (
    Message as LiteLLMMessage,
)

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    GenerationConfig,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    Message,
    RefusalPart,
    ResponseFormat,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    Usage,
)
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    InternalError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.providers.litellm_translation import (
    _NON_SEMANTIC_RESPONSE_FIELDS,
    from_litellm_response,
    map_litellm_error,
    to_litellm_params,
)

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "gpt-4o"

ALL_CAPABILITIES = frozenset(
    {
        Capability.CHAT,
        Capability.TOOLS,
        Capability.VISION,
        Capability.STRUCTURED_OUTPUT,
    }
)


def make_route(**overrides: object) -> Route:
    """A resolved text route; override any field to model a different configuration."""
    fields: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": "openai",
        "api_key": SECRET_KEY,
        "capabilities": ALL_CAPABILITIES,
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request(**overrides: object) -> InvocationRequest:
    """A canonical invocation; override any field to model a different request."""
    fields: dict[str, object] = {
        "model": "text-default",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
        "invocation_id": "inv-7f3c1a90",
    }
    fields.update(overrides)
    return InvocationRequest(**fields)


def response_from(payload: dict[str, object]) -> ModelResponse:
    """Build a real ``ModelResponse`` by parsing a payload through LiteLLM's own converter.

    This is the shape a real provider response has once LiteLLM has decoded it,
    including the fields LiteLLM itself adds or normalises.
    """
    response = ModelResponse()
    convert_to_model_response_object(response_object=payload, model_response_object=response)
    return response


def completion_payload(**overrides: object) -> dict[str, object]:
    """A realistic OpenAI-shaped upstream completion payload."""
    payload: dict[str, object] = {
        "id": "chatcmpl-9x2f",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recursion is a technique...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 42,
            "total_tokens": 67,
        },
    }
    payload.update(overrides)
    return payload


def http_error(status: int) -> httpx.Response:
    """An HTTP error response with a request attached, as LiteLLM exceptions carry."""
    return httpx.Response(
        status_code=status,
        request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions"),
    )


def user_message(text: str = "Hello") -> Message:
    return Message(role=Role.USER, content=(TextPart(text=text),))


# --------------------------------------------------------------------------- request side


class TestToLitellmParams:
    """The canonical request reaches the upstream call in full (PRD sections 9.1 and 9.6)."""

    def test_model_string_is_provider_prefixed(self) -> None:
        params = to_litellm_params(make_route(), make_request())
        assert params["model"] == f"openai/{UPSTREAM_MODEL}"

    def test_base_url_api_key_and_extra_headers_reach_upstream(self) -> None:
        route = make_route(extra_headers={"X-Org": "acme", "X-Project": "42"})
        params = to_litellm_params(route, make_request())
        assert params["api_base"] == ENDPOINT
        assert params["api_key"] == SECRET_KEY
        assert params["extra_headers"] == {"X-Org": "acme", "X-Project": "42"}

    def test_retries_are_disabled(self) -> None:
        """The gateway core owns the retry loop, so LiteLLM must not retry (PRD 11.3)."""
        params = to_litellm_params(make_route(), make_request())
        assert params["num_retries"] == 0

    def test_system_instruction_forwarded_as_separate_system_message(self) -> None:
        """The instruction is forwarded unmerged, ahead of the conversation (PRD 9.5)."""
        request = make_request(
            system_instruction="Answer concisely.",
            messages=(
                Message(role=Role.SYSTEM, content=(TextPart(text="You are helpful."),)),
                user_message(),
            ),
        )
        params = to_litellm_params(make_route(), request)
        messages = params["messages"]
        assert messages[0] == {"role": "system", "content": "Answer concisely."}
        assert messages[1] == {"role": "system", "content": "You are helpful."}
        assert messages[2] == {"role": "user", "content": "Hello"}

    def test_conversation_roles_are_forwarded(self) -> None:
        request = make_request(
            messages=(
                Message(role=Role.SYSTEM, content=(TextPart(text="sys"),)),
                Message(role=Role.DEVELOPER, content=(TextPart(text="dev"),)),
                Message(role=Role.USER, content=(TextPart(text="user"),)),
                Message(role=Role.ASSISTANT, content=(TextPart(text="assistant"),)),
                Message(role=Role.TOOL, content=(TextPart(text="tool"),), tool_call_id="call_1"),
            )
        )
        params = to_litellm_params(make_route(), request)
        roles = [message["role"] for message in params["messages"]]
        assert roles == ["system", "developer", "user", "assistant", "tool"]

    def test_tool_message_carries_its_tool_call_id(self) -> None:
        """A result that cannot be matched to its call is not a result (PRD 9.3)."""
        request = make_request(
            messages=(
                Message(
                    role=Role.TOOL,
                    content=(TextPart(text='{"temp_c": 18}'),),
                    tool_call_id="call_01",
                ),
            )
        )
        params = to_litellm_params(make_route(), request)
        assert params["messages"][0]["tool_call_id"] == "call_01"

    def test_assistant_tool_calls_are_serialised(self) -> None:
        request = make_request(
            messages=(
                Message(
                    role=Role.ASSISTANT,
                    content=(TextPart(text="Let me look that up."),),
                    tool_calls=(
                        ToolCall(
                            id="call_01",
                            name="get_weather",
                            arguments={"city": "Paris", "days": 1},
                        ),
                    ),
                ),
            )
        )
        params = to_litellm_params(make_route(), request)
        tool_calls = params["messages"][0]["tool_calls"]
        assert tool_calls == [
            {
                "id": "call_01",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris", "days": 1}',
                },
            }
        ]

    def test_image_parts_become_ordered_content_blocks(self) -> None:
        """Text and images interleaved stay interleaved (PRD 9.4)."""
        request = make_request(
            messages=(
                Message(
                    role=Role.USER,
                    content=(
                        TextPart(text="What is in this image?"),
                        ImagePart(data="iVBORw0KGgo=", media_type="image/png"),
                        TextPart(text="And this one?"),
                        ImagePart(url="https://images.example/paris.jpg"),
                    ),
                ),
            )
        )
        params = to_litellm_params(make_route(), request)
        assert params["messages"][0]["content"] == [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "And this one?"},
            {"type": "image_url", "image_url": {"url": "https://images.example/paris.jpg"}},
        ]

    def test_single_text_part_is_a_plain_string(self) -> None:
        params = to_litellm_params(make_route(), make_request())
        assert params["messages"][0]["content"] == "What is the weather in Paris?"

    def test_temperature_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(temperature=0.7))
        assert to_litellm_params(make_route(), request)["temperature"] == 0.7

    def test_top_p_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(top_p=0.9))
        assert to_litellm_params(make_route(), request)["top_p"] == 0.9

    def test_max_output_tokens_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(max_output_tokens=512))
        assert to_litellm_params(make_route(), request)["max_tokens"] == 512

    def test_stop_sequences_reach_upstream(self) -> None:
        request = make_request(
            generation_config=GenerationConfig(stop_sequences=("\n\nHuman:", "END"))
        )
        assert to_litellm_params(make_route(), request)["stop"] == ["\n\nHuman:", "END"]

    def test_seed_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(seed=7))
        assert to_litellm_params(make_route(), request)["seed"] == 7

    def test_presence_penalty_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(presence_penalty=0.1))
        assert to_litellm_params(make_route(), request)["presence_penalty"] == 0.1

    def test_frequency_penalty_reaches_upstream(self) -> None:
        request = make_request(generation_config=GenerationConfig(frequency_penalty=0.2))
        assert to_litellm_params(make_route(), request)["frequency_penalty"] == 0.2

    def test_json_object_response_format_reaches_upstream(self) -> None:
        request = make_request(
            generation_config=GenerationConfig(
                response_format=ResponseFormat(mode=ResponseFormatMode.JSON_OBJECT)
            )
        )
        assert to_litellm_params(make_route(), request)["response_format"] == {
            "type": "json_object"
        }

    def test_json_schema_response_format_reaches_upstream(self) -> None:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        request = make_request(
            generation_config=GenerationConfig(
                response_format=ResponseFormat(
                    mode=ResponseFormatMode.JSON_SCHEMA,
                    schema=schema,
                    name="weather_report",
                    strict=True,
                )
            )
        )
        assert to_litellm_params(make_route(), request)["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "weather_report",
                "schema": schema,
                "strict": True,
            },
        }

    def test_text_response_format_is_omitted(self) -> None:
        """A text format asks for nothing beyond chat; it is the provider's default (PRD 6.1)."""
        request = make_request(
            generation_config=GenerationConfig(
                response_format=ResponseFormat(mode=ResponseFormatMode.TEXT)
            )
        )
        assert "response_format" not in to_litellm_params(make_route(), request)

    def test_unset_generation_parameters_are_omitted(self) -> None:
        """Unset and zero are different requests; no vendor default is invented (PRD 5.1)."""
        params = to_litellm_params(make_route(), make_request())
        for name in (
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
        ):
            assert name not in params

    def test_tools_reach_upstream(self) -> None:
        request = make_request(
            tools=(
                ToolDefinition(
                    name="get_weather",
                    description="Look up the current weather for a city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        )
        params = to_litellm_params(make_route(), request)
        assert params["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (ToolChoice(mode=ToolChoiceMode.AUTO), "auto"),
            (ToolChoice(mode=ToolChoiceMode.NONE), "none"),
            (ToolChoice(mode=ToolChoiceMode.REQUIRED), "required"),
            (
                ToolChoice(mode=ToolChoiceMode.SPECIFIC, name="get_weather"),
                {"type": "function", "function": {"name": "get_weather"}},
            ),
        ],
    )
    def test_tool_choice_reaches_upstream(
        self, choice: ToolChoice, expected: object
    ) -> None:
        request = make_request(
            tools=(ToolDefinition(name="get_weather", parameters={"type": "object"}),),
            tool_choice=choice,
        )
        assert to_litellm_params(make_route(), request)["tool_choice"] == expected

    def test_refusal_part_in_request_is_rejected(self) -> None:
        """A refusal the caller authored has no upstream representation (PRD 9.1, 5.2)."""
        request = make_request(
            messages=(
                Message(role=Role.ASSISTANT, content=(RefusalPart(text="I can't answer that."),)),
            )
        )
        with pytest.raises(CapabilityError) as error:
            to_litellm_params(make_route(), request)
        assert "refusal" in error.value.message
        assert error.value.details["capability"] == "refusal"

    def test_metadata_is_never_forwarded(self) -> None:
        """Metadata is bookkeeping, not a provider parameter bag (PRD 5.1)."""
        request = make_request(metadata={"request_source": "console", "trace": {"sampled": True}})
        params = to_litellm_params(make_route(), request)
        assert "metadata" not in params
        assert "trace" not in params

    def test_routing_and_accounting_fields_are_not_forwarded(self) -> None:
        """The gateway chose the route and mints the identity; the upstream is told neither."""
        request = make_request(
            fallback_models=("text-fallback",),
            idempotency_key="idem-42",
            tenant_id="acme",
            actor_id="svc-bot",
        )
        params = to_litellm_params(make_route(), request)
        assert "fallback_models" not in params
        assert "idempotency_key" not in params
        assert "tenant_id" not in params
        assert "actor_id" not in params


# --------------------------------------------------------------------------- response side


class TestFromLitellmResponse:
    """The upstream response is normalised into the canonical model (PRD 9.1, 5.2, 13)."""

    def test_gateway_identity_is_always_stamped(self) -> None:
        """The gateway's own identity, never an upstream value (PRD 5.2, 9)."""
        response = response_from(
            completion_payload(id="chatcmpl-upstream", model="gpt-4o-2024-08-06")
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.model == "text-default"
        assert result.upstream_model == UPSTREAM_MODEL
        assert result.invocation_id == "inv-7f3c1a90"

    def test_upstream_model_is_the_route_model_id_not_the_prefixed_string(self) -> None:
        result = from_litellm_response(
            make_route(), make_request(), response_from(completion_payload())
        )
        assert result.upstream_model == UPSTREAM_MODEL
        assert "/" not in result.upstream_model

    def test_text_content_is_mapped(self) -> None:
        result = from_litellm_response(
            make_route(), make_request(), response_from(completion_payload())
        )
        assert result.content == (TextPart(text="Recursion is a technique..."),)
        assert result.text == "Recursion is a technique..."

    def test_tool_calls_carry_identifier_and_structured_arguments(self) -> None:
        response = response_from(
            completion_payload(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Let me check the weather.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris", "days": 3}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.tool_calls == (
            ToolCall(
                id="call_1",
                name="get_weather",
                arguments={"city": "Paris", "days": 3},
            ),
        )
        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert result.raw_finish_reason == "tool_calls"

    def test_malformed_arguments_are_kept_raw(self) -> None:
        """The gateway records what arrived rather than repairing it (PRD 9.3)."""
        response = response_from(
            completion_payload(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "not json{",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.tool_calls[0].arguments == "not json{"

    def test_ordered_output_preserves_text_then_tool_calls(self) -> None:
        """The order the upstream shape preserves (PRD 9.1)."""
        response = response_from(
            completion_payload(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Let me check.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.content == (
            TextPart(text="Let me check."),
            ToolCall(id="call_1", name="get_weather", arguments={}),
        )

    def test_response_images_are_mapped(self) -> None:
        response = response_from(
            completion_payload(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Here you go.",
                            "images": [
                                {
                                    "type": "image_url",
                                    "index": 0,
                                    "image_url": {"url": "https://images.example/out.png"},
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ]
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.images == (ImagePart(url="https://images.example/out.png"),)

    def test_finish_reason_is_mapped_through_the_gateway_vocabulary(self) -> None:
        result = from_litellm_response(
            make_route(), make_request(), response_from(completion_payload())
        )
        assert result.finish_reason is FinishReason.STOP
        assert result.raw_finish_reason == "stop"

    def test_unrecognised_finish_reason_is_unknown_with_raw_preserved(self) -> None:
        """UNKNOWN never means the gateway deleted what upstream said (PRD 5.2).

        LiteLLM normalises both of these onto its OpenAI-shaped vocabulary, but
        the gateway's vocabulary is its own and matches exactly: a reason the
        gateway has not been taught becomes UNKNOWN, and the raw value still
        says what arrived.
        """
        for raw in ("end_turn", "max_tokens"):
            response = response_from(
                completion_payload(
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": raw,
                        }
                    ]
                )
            )
            result = from_litellm_response(make_route(), make_request(), response)
            assert result.finish_reason is FinishReason.UNKNOWN
            assert result.raw_finish_reason == raw

    def test_usage_is_provider_reported(self) -> None:
        response = response_from(
            completion_payload(
                usage={
                    "prompt_tokens": 25,
                    "completion_tokens": 42,
                    "total_tokens": 67,
                }
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.usage == Usage(
            input_tokens=25, output_tokens=42, total_tokens=67, estimated_fields=frozenset()
        )
        assert result.usage.source_of("input_tokens").value == "provider"
        assert result.usage.source_of("total_tokens").value == "provider"

    def test_usage_total_is_estimated_when_the_provider_reported_no_total(self) -> None:
        """A zero total beside non-zero parts is LiteLLM's placeholder, not a fact (PRD 13)."""
        response = response_from(
            completion_payload(
                usage={
                    "prompt_tokens": 25,
                    "completion_tokens": 42,
                }
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.usage.input_tokens == 25
        assert result.usage.output_tokens == 42
        assert result.usage.total_tokens == 67
        assert result.usage.estimated_fields == frozenset({"total_tokens"})
        assert result.usage.source_of("total_tokens").value == "estimate"

    def test_usage_cached_tokens_are_provider_reported(self) -> None:
        response = response_from(
            completion_payload(
                usage={
                    "prompt_tokens": 25,
                    "completion_tokens": 42,
                    "total_tokens": 67,
                    "prompt_tokens_details": {"cached_tokens": 10},
                }
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.usage.cached_tokens == 10
        assert result.usage.source_of("cached_tokens").value == "provider"

    def test_absent_usage_is_unknown_not_zero(self) -> None:
        """None means not known, which is not the same as zero (PRD 13)."""
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Hi", role="assistant"),
                )
            ]
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result.usage == Usage()

    def test_multiple_choices_are_rejected(self) -> None:
        """The canonical response represents one answer; the rest cannot be dropped."""
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="a", role="assistant"),
                ),
                Choices(
                    finish_reason="stop",
                    index=1,
                    message=LiteLLMMessage(content="b", role="assistant"),
                ),
            ]
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "choices" in error.value.message

    def test_logprobability_block_is_rejected(self) -> None:
        """A log-probability block has no canonical target; the response is refused (PRD 9.1)."""
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Hi", role="assistant"),
                    logprobs=ChoiceLogprobs(content=[]),
                )
            ]
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "log-probability" in error.value.message

    def test_provider_specific_field_is_rejected(self) -> None:
        """A provider annotation the canonical model cannot carry is refused, not trimmed."""
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Hi", role="assistant"),
                    provider_specific_fields={"annotation": "extra"},
                )
            ]
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "annotation" in error.value.message

    def test_extra_message_field_from_upstream_is_rejected(self) -> None:
        """A payload field LiteLLM cannot map lands in provider_specific_fields and is refused."""
        response = response_from(
            completion_payload(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hi",
                            "some_provider_field": "value",
                        },
                        "finish_reason": "stop",
                    }
                ]
            )
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "some_provider_field" in error.value.message

    def test_reasoning_content_is_rejected(self) -> None:
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        content="Hi", role="assistant", reasoning_content="thinking..."
                    ),
                )
            ]
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "reasoning content" in error.value.message

    def test_system_fingerprint_is_ignored(self) -> None:
        """Backend metadata, not answer semantics: the response is not failed (PRD 9.1).

        ``system_fingerprint`` identifies the upstream's infrastructure, not
        anything about the answer — two backends serving the same model return
        the same tokens — so it is classified non-semantic operational
        metadata and deliberately ignored, from the explicit set
        :data:`~model_gateway.providers.litellm_translation._NON_SEMANTIC_RESPONSE_FIELDS`.
        It must not fail an otherwise valid response, and it must not be
        treated as a semantic either: the canonical response has no field for
        it and nothing downstream depends on it.
        """
        assert "system_fingerprint" in _NON_SEMANTIC_RESPONSE_FIELDS
        response = response_from(completion_payload(system_fingerprint="fp_abc123"))

        result = from_litellm_response(make_route(), make_request(), response)

        assert result.text == "Recursion is a technique..."

    def test_reasoning_token_detail_is_rejected(self) -> None:
        """A completion-token breakdown count has no canonical target (PRD 9.1)."""
        response = response_from(
            completion_payload(
                usage={
                    "prompt_tokens": 25,
                    "completion_tokens": 42,
                    "total_tokens": 67,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                }
            )
        )
        with pytest.raises(InternalError) as error:
            from_litellm_response(make_route(), make_request(), response)
        assert "reasoning_tokens" in error.value.message

    def test_full_round_trip_from_a_realistic_payload(self) -> None:
        """A complete realistic payload normalises into the canonical response.

        The payload carries a backend fingerprint beside the mapped fields: it
        is classified non-semantic operational metadata and must not fail the
        round trip.
        """
        response = response_from(
            completion_payload(
                id="chatcmpl-roundtrip",
                system_fingerprint="fp_roundtrip",
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Recursion is a technique...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": 25,
                    "completion_tokens": 42,
                    "total_tokens": 67,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            )
        )
        result = from_litellm_response(make_route(), make_request(), response)
        assert result == InvocationResponse(
            model="text-default",
            upstream_model=UPSTREAM_MODEL,
            invocation_id="inv-7f3c1a90",
            content=(TextPart(text="Recursion is a technique..."),),
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=Usage(
                input_tokens=25,
                output_tokens=42,
                total_tokens=67,
                cached_tokens=10,
            ),
        )


# --------------------------------------------------------------------------- error mapping


class TestMapLitellmError:
    """The LiteLLM exception hierarchy maps onto the gateway taxonomy (PRD 5.4, 9)."""

    def test_authentication_error_is_unauthenticated(self) -> None:
        error = litellm.AuthenticationError(
            message="bad key",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(401),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnauthenticatedError)
        assert mapped.reason.value == "invalid_credentials"
        assert mapped.http_status == 401

    def test_permission_error_is_permission_denied(self) -> None:
        error = litellm.PermissionDeniedError(
            message="forbidden",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(403),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, PermissionDeniedError)
        assert mapped.http_status == 403

    def test_not_found_error_is_not_found(self) -> None:
        error = litellm.NotFoundError(
            message="no model",
            model="gpt-9",
            llm_provider="openai",
            response=http_error(404),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, NotFoundError)
        assert mapped.http_status == 404

    def test_rate_limit_error_is_resource_exhausted(self) -> None:
        error = litellm.RateLimitError(
            message="slow down",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(429),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, ResourceExhaustedError)
        assert mapped.reason.value == "rate_limited"
        assert mapped.http_status == 429

    def test_timeout_error_is_deadline_exceeded(self) -> None:
        error = litellm.Timeout(message="too slow", model=UPSTREAM_MODEL, llm_provider="openai")
        mapped = map_litellm_error(error)
        assert isinstance(mapped, DeadlineExceededError)
        assert mapped.http_status == 504

    def test_connection_error_is_unavailable(self) -> None:
        error = litellm.APIConnectionError(
            message="connection reset",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)
        assert mapped.http_status == 503

    def test_service_unavailable_error_is_unavailable(self) -> None:
        error = litellm.ServiceUnavailableError(
            message="down",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(503),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)

    def test_bad_gateway_error_is_unavailable(self) -> None:
        error = litellm.BadGatewayError(
            message="bad gateway",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(502),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)

    def test_internal_server_error_is_unavailable(self) -> None:
        error = litellm.InternalServerError(
            message="boom",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(500),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)

    def test_api_error_with_5xx_status_is_unavailable(self) -> None:
        error = litellm.APIError(
            status_code=500,
            message="weird",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)

    def test_api_error_with_4xx_status_is_internal(self) -> None:
        """The upstream refused a request the gateway had already validated."""
        error = litellm.APIError(
            status_code=400,
            message="bad",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, InternalError)

    def test_malformed_response_is_unavailable(self) -> None:
        error = litellm.APIResponseValidationError(
            message="bad shape",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, UnavailableError)

    def test_bad_request_error_is_internal(self) -> None:
        error = litellm.BadRequestError(
            message="bad request",
            model=UPSTREAM_MODEL,
            llm_provider="openai",
            response=http_error(400),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, InternalError)

    def test_context_window_exceeded_is_internal(self) -> None:
        error = litellm.ContextWindowExceededError(
            message="too long",
            model=UPSTREAM_MODEL,
            llm_provider="openai",
            response=http_error(400),
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, InternalError)

    def test_unsupported_params_error_is_internal(self) -> None:
        error = litellm.UnsupportedParamsError(
            message="unsupported param",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
        )
        mapped = map_litellm_error(error)
        assert isinstance(mapped, InternalError)

    def test_gateway_error_passes_through_unchanged(self) -> None:
        """An already-classified error is not re-classified (PRD 5.4)."""
        error = CapabilityError("already classified", details={"capability": "chat"})
        assert map_litellm_error(error) is error

    def test_unclassified_exception_is_internal(self) -> None:
        mapped = map_litellm_error(ValueError("something unexpected"))
        assert isinstance(mapped, InternalError)

    def test_upstream_message_never_reaches_the_public_payload(self) -> None:
        """The taxonomy sanitises its own payloads; no upstream text is forwarded (PRD 5.4)."""
        secret = "sk-upstream-secret-key"
        url = "https://api.openai.com/v1/chat/completions"
        error = litellm.AuthenticationError(
            message=f"invalid key {secret} for {url}",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(401),
        )
        mapped = map_litellm_error(error)
        payload = mapped.to_payload(include_details=True)
        assert secret not in str(payload)
        assert url not in str(payload)
        assert "openai" not in str(payload)
        assert mapped.message == "the upstream rejected the route's credentials"
