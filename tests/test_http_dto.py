"""Wire-translation tests for the OpenAI-compatible DTOs (PRD sections 4.3, 4.4, 9.1, 9.5)."""

import pytest

from model_gateway.capabilities import Capability
from model_gateway.domain import (
    FinishReason,
    ImagePart,
    InvocationResponse,
    Role,
    TextPart,
    ToolCall,
    ToolChoiceMode,
    Usage,
)
from model_gateway.errors import CapabilityError, InvalidArgumentError
from model_gateway.http.dto import ChatCompletion, ChatRequest, ModelCard, model_list_payload


def body(**overrides):
    payload = {
        "model": "text-default",
        "messages": [{"role": "user", "content": "Explain recursion."}],
    }
    payload.update(overrides)
    return payload


def response(**overrides):
    fields = {
        "model": "text-default",
        "upstream_model": "upstream-model",
        "invocation_id": "abc123",
        "content": (TextPart("Recursion is a technique..."),),
        "finish_reason": FinishReason.STOP,
        "raw_finish_reason": "stop",
        "usage": Usage(input_tokens=25, output_tokens=42, total_tokens=67),
    }
    fields.update(overrides)
    return InvocationResponse(**fields)


class TestModelCard:
    def test_omits_metadata_the_route_never_stated(self):
        card = ModelCard(id="text-default", capabilities=frozenset({Capability.CHAT}))

        assert card.to_dict() == {
            "id": "text-default",
            "object": "model",
            "created": 0,
            "owned_by": "gateway",
            "capabilities": ["chat"],
        }

    def test_reports_the_metadata_the_route_states(self):
        card = ModelCard(
            id="embedding-default",
            capabilities=frozenset({Capability.EMBEDDINGS}),
            context_window=8192,
            max_output_tokens=2048,
            dimensions=1024,
        )

        assert card.to_dict() == {
            "id": "embedding-default",
            "object": "model",
            "created": 0,
            "owned_by": "gateway",
            "capabilities": ["embeddings"],
            "context_window": 8192,
            "max_output_tokens": 2048,
            "dimensions": 1024,
        }

    def test_capabilities_are_sorted_so_the_body_is_stable(self):
        card = ModelCard(
            id="text-default",
            capabilities=frozenset({Capability.VISION, Capability.CHAT, Capability.TOOLS}),
        )

        assert card.to_dict()["capabilities"] == ["chat", "tools", "vision"]

    def test_list_envelope_is_an_openai_list(self):
        cards = [ModelCard(id="text-default", capabilities=frozenset({Capability.CHAT}))]

        assert model_list_payload(cards) == {"object": "list", "data": [cards[0].to_dict()]}


class TestChatRequestMessages:
    @pytest.mark.parametrize(
        ("role", "extra"),
        [
            ("system", {}),
            ("developer", {}),
            ("user", {}),
            ("assistant", {}),
            ("tool", {"tool_call_id": "call_1"}),
        ],
    )
    def test_parses_every_canonical_role(self, role, extra):
        payload = body(messages=[{"role": role, "content": "hello", **extra}])

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[0].role is Role(role)
        assert request.invocation.messages[0].content == (TextPart("hello"),)

    def test_keeps_a_multi_turn_conversation_in_order(self):
        payload = body(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain recursion."},
                {"role": "assistant", "content": "Recursion is a technique..."},
                {"role": "user", "content": "In one line."},
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert [message.role for message in request.invocation.messages] == [
            Role.SYSTEM,
            Role.USER,
            Role.ASSISTANT,
            Role.USER,
        ]

    def test_parses_content_parts(self):
        payload = body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    ],
                }
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[0].content == (
            TextPart("What is this?"),
            ImagePart(url="https://example.com/a.png"),
        )

    def test_parses_a_base64_data_uri_into_inline_image_content(self):
        payload = body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        }
                    ],
                }
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[0].content == (
            ImagePart(data="QUJD", media_type="image/png"),
        )

    def test_refuses_an_image_url_scheme_the_gateway_will_not_carry(self):
        payload = body(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}],
                }
            ]
        )

        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(payload)

        assert error.value.reason.value == "invalid_request"

    def test_refuses_an_unknown_role(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(messages=[{"role": "robot", "content": "hi"}]))

        assert "must be one of" in error.value.message

    def test_refuses_an_unknown_message_key(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(
                body(messages=[{"role": "user", "content": "hi", "audio": {}}])
            )

        assert "unsupported key 'audio'" in error.value.message

    def test_refuses_a_tool_message_without_the_call_it_answers(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(messages=[{"role": "tool", "content": "42"}]))

        assert "tool_call_id" in error.value.message

    def test_parses_an_assistant_tool_call_with_json_arguments(self):
        payload = body(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                        }
                    ],
                },
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[1].tool_calls == (
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
        )

    def test_keeps_tool_arguments_verbatim_when_they_do_not_parse(self):
        payload = body(
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{not json"},
                        }
                    ],
                }
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[0].tool_calls[0].arguments == "{not json"

    def test_refuses_a_message_that_is_not_an_object(self):
        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(body(messages=["hello"]))

    def test_refuses_a_body_without_messages(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload({"model": "text-default"})

        assert "messages" in error.value.message

    def test_refuses_an_empty_conversation(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(messages=[]))

        assert "at least one message" in error.value.message


class TestChatRequestParameters:
    def test_maps_the_sampling_parameters(self):
        payload = body(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["END"],
            seed=7,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        )

        config = ChatRequest.from_payload(payload).invocation.generation_config

        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_output_tokens == 2048
        assert config.stop_sequences == ("END",)
        assert config.seed == 7
        assert config.presence_penalty == 0.5
        assert config.frequency_penalty == 0.25

    def test_leaves_omitted_parameters_unset(self):
        config = ChatRequest.from_payload(body()).invocation.generation_config

        assert config.temperature is None
        assert config.max_output_tokens is None
        assert config.response_format is None

    def test_accepts_a_single_stop_sequence(self):
        config = ChatRequest.from_payload(body(stop="END")).invocation.generation_config

        assert config.stop_sequences == ("END",)

    def test_refuses_a_temperature_the_canonical_type_cannot_carry(self):
        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(body(temperature=-1))

    def test_maps_the_legacy_user_field_onto_the_actor_identity(self):
        request = ChatRequest.from_payload(body(user="caller-7"))

        assert request.invocation.actor_id == "caller-7"

    def test_maps_metadata_onto_the_invocation_metadata(self):
        request = ChatRequest.from_payload(body(metadata={"trace": "abc"}))

        assert request.invocation.metadata == {"trace": "abc"}

    def test_refuses_metadata_that_names_a_canonical_field(self):
        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(body(metadata={"temperature": 1}))

    def test_parses_a_json_object_response_format(self):
        request = ChatRequest.from_payload(body(response_format={"type": "json_object"}))

        response_format = request.invocation.generation_config.response_format

        assert response_format.mode.value == "json_object"

    def test_parses_a_json_schema_response_format(self):
        payload = body(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            }
        )

        request = ChatRequest.from_payload(payload)
        response_format = request.invocation.generation_config.response_format

        assert response_format.mode.value == "json_schema"
        assert response_format.name == "answer"
        assert response_format.strict is True

    def test_refuses_a_json_schema_response_format_without_a_schema(self):
        payload = body(response_format={"type": "json_schema", "json_schema": {"name": "answer"}})

        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(payload)

    def test_parses_tool_declarations(self):
        payload = body(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up the weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.tools[0].name == "get_weather"
        assert request.invocation.tools[0].description == "Look up the weather"
        assert request.invocation.tools[0].parameters == {"type": "object"}

    def test_refuses_a_tool_strict_flag_the_canonical_type_cannot_carry(self):
        payload = body(
            tools=[{"type": "function", "function": {"name": "t", "strict": True}}],
        )

        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(payload)

        assert "strict" in error.value.message

    def test_parses_a_named_tool_choice(self):
        payload = body(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        choice = ChatRequest.from_payload(payload).invocation.tool_choice

        assert choice.mode is ToolChoiceMode.SPECIFIC
        assert choice.name == "get_weather"

    def test_refuses_a_named_tool_choice_the_invocation_does_not_declare(self):
        payload = body(tool_choice={"type": "function", "function": {"name": "missing"}})

        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(payload)

    def test_refuses_a_parameter_the_openai_format_does_not_define(self):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(verbosity="high"))

        assert "verbosity" in error.value.message

    def test_refuses_a_recognised_feature_the_gateway_cannot_honour(self):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(n=3))

        assert error.value.reason.value == "unsupported_capability"
        assert error.value.details["capability"] == "multiple_choices"
        assert error.value.http_status == 422

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n": 1},
            {"store": False},
            {"logprobs": False},
            {"parallel_tool_calls": True},
            {"modalities": ["text"]},
            {"stream_options": None},
        ],
    )
    def test_accepts_a_parameter_asking_for_the_openai_default(self, overrides):
        assert ChatRequest.from_payload(body(**overrides)).invocation.model == "text-default"

    def test_carries_stream_rather_than_acting_on_it(self):
        assert ChatRequest.from_payload(body(stream=True)).stream is True

    def test_refuses_a_stream_flag_that_is_not_a_boolean(self):
        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload(body(stream="yes"))

    def test_refuses_a_body_that_is_not_an_object(self):
        with pytest.raises(InvalidArgumentError):
            ChatRequest.from_payload([1, 2, 3])

    def test_builds_the_canonical_invocation(self):
        request = ChatRequest.from_payload(body(temperature=0.7, user="caller-7"))

        invocation = request.to_invocation(invocation_id="inv-1")

        assert invocation.model == "text-default"
        assert invocation.invocation_id == "inv-1"
        assert invocation.actor_id == "caller-7"
        assert invocation.response_modality.value == "text"


class TestChatCompletion:
    def test_matches_the_documented_response_shape(self):
        completion = ChatCompletion.from_invocation(
            response(), completion_id="chatcmpl-1", created=1700000000
        )

        assert completion.to_dict() == {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "text-default",
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
            "usage": {"prompt_tokens": 25, "completion_tokens": 42, "total_tokens": 67},
        }

    def test_reports_a_tool_call_answer_with_null_content(self):
        completion = ChatCompletion.from_invocation(
            response(
                content=(ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),),
                finish_reason=FinishReason.TOOL_CALLS,
                raw_finish_reason="tool_calls",
            ),
            completion_id="chatcmpl-1",
            created=0,
        )

        message = completion.to_dict()["choices"][0]["message"]

        assert message["content"] is None
        assert message["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'},
            }
        ]
        assert completion.to_dict()["choices"][0]["finish_reason"] == "tool_calls"

    def test_reports_tool_arguments_it_could_not_parse_verbatim(self):
        completion = ChatCompletion.from_invocation(
            response(content=(ToolCall(id="c", name="t", arguments="{not json"),)),
            completion_id="chatcmpl-1",
            created=0,
        )

        call = completion.to_dict()["choices"][0]["message"]["tool_calls"][0]

        assert call["function"]["arguments"] == "{not json"

    def test_reports_a_refusal_beside_the_content(self):
        from model_gateway.domain import RefusalPart

        completion = ChatCompletion.from_invocation(
            response(content=(RefusalPart("I cannot help with that."),)),
            completion_id="chatcmpl-1",
            created=0,
        )

        message = completion.to_dict()["choices"][0]["message"]

        assert message["refusal"] == "I cannot help with that."

    def test_reports_an_unclassified_finish_reason_as_the_provider_gave_it(self):
        completion = ChatCompletion.from_invocation(
            response(finish_reason=FinishReason.UNKNOWN, raw_finish_reason="eos"),
            completion_id="chatcmpl-1",
            created=0,
        )

        assert completion.to_dict()["choices"][0]["finish_reason"] == "eos"

    def test_reports_no_finish_reason_when_the_provider_gave_none(self):
        completion = ChatCompletion.from_invocation(
            response(finish_reason=FinishReason.UNKNOWN, raw_finish_reason=None),
            completion_id="chatcmpl-1",
            created=0,
        )

        assert completion.to_dict()["choices"][0]["finish_reason"] is None

    def test_refuses_image_output_the_response_shape_cannot_carry(self):
        with pytest.raises(CapabilityError) as error:
            ChatCompletion.from_invocation(
                response(content=(TextPart("here"), ImagePart(data="QUJD"))),
                completion_id="chatcmpl-1",
                created=0,
            )

        assert error.value.details["capability"] == "image_output"

    def test_derives_the_total_the_provider_omitted(self):
        completion = ChatCompletion.from_invocation(
            response(usage=Usage(input_tokens=10, output_tokens=5)),
            completion_id="chatcmpl-1",
            created=0,
        )

        assert completion.to_dict()["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
