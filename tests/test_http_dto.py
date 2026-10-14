"""Wire-translation tests for the OpenAI-compatible DTOs (PRD sections 4.3, 4.4, 9.1, 9.5)."""

import pytest

from model_gateway.capabilities import Capability
from model_gateway.domain import (
    FinishReason,
    ImagePart,
    InvocationResponse,
    RefusalPart,
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
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ]
        )

        request = ChatRequest.from_payload(payload)

        assert request.invocation.messages[0].content == (
            TextPart("What is this?"),
            ImagePart(data="QUJD", media_type="image/png"),
        )

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/a.png",
            "http://example.com/a.png",
            "http://127.0.0.1/a.png",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/a.png",
            "http://[::1]/a.png",
        ],
    )
    def test_refuses_a_remote_image_url_the_gateway_cannot_screen(self, url):
        payload = body(
            messages=[
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
            ]
        )

        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(payload)

        assert error.value.reason.value == "unsupported_capability"
        assert error.value.details["capability"] == "remote_image_fetch"
        assert error.value.http_status == 422

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
            {"modalities": ["text"]},
            {"stream_options": {}},
        ],
    )
    def test_accepts_a_parameter_asking_for_the_openai_default(self, overrides):
        assert ChatRequest.from_payload(body(**overrides)).invocation.model == "text-default"

    @pytest.mark.parametrize("value", [True, False])
    def test_refuses_an_explicit_parallel_tool_calls_flag(self, value):
        """Neither direction is a promise this slice can keep.

        ``true`` asks for parallel tool calls and ``false`` asks for a guarantee
        of sequential ones; nothing here implements or preserves either, so
        accepting the flag would advertise a capability the gateway lacks.
        """
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(parallel_tool_calls=value))

        assert error.value.reason.value == "unsupported_capability"
        assert error.value.details["capability"] == "parallel_tool_calls"
        assert error.value.details["parameter"] == "parallel_tool_calls"

    def test_accepts_an_absent_parallel_tool_calls_flag(self):
        assert ChatRequest.from_payload(body()).invocation.model == "text-default"

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


class TestChatRequestParameterWireTypes:
    """Each recognised parameter's wire type is checked before its value is read.

    A value of the wrong JSON type is malformed input, not a request for a
    feature, so it is refused with a 400 naming the parameter and the type
    supplied — including the values a value-only reading would misjudge:
    ``true`` for an integer parameter, because ``True == 1``, and the falsy
    ``0``, ``""``, ``[]`` and ``{}`` for a boolean or object parameter. A
    correctly typed value that is the OpenAI default asks for nothing and is
    accepted; any other correctly typed value keeps the capability refusal
    (PRD section 9.5).
    """

    @pytest.mark.parametrize("value", [True, 1.5, "2", None, [], {}])
    def test_refuses_an_n_that_is_not_an_integer(self, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(n=value))

        assert error.value.reason.value == "invalid_request"
        assert "parameter 'n'" in error.value.message
        assert "an integer" in error.value.message

    @pytest.mark.parametrize("value", [0, 2])
    def test_refuses_an_n_other_than_the_default_the_gateway_matches(self, value):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(n=value))

        assert error.value.details["capability"] == "multiple_choices"
        assert error.value.details["parameter"] == "n"

    @pytest.mark.parametrize("name", ["logprobs", "store"])
    @pytest.mark.parametrize("value", [0, 1, "", [], {}, None])
    def test_refuses_a_flag_that_is_not_a_boolean(self, name, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert f"parameter {name!r}" in error.value.message
        assert "a boolean" in error.value.message

    @pytest.mark.parametrize("name", ["logprobs", "store"])
    def test_accepts_the_false_default(self, name):
        assert ChatRequest.from_payload(body(**{name: False})).invocation.model == "text-default"

    @pytest.mark.parametrize(
        ("name", "capability"),
        [("logprobs", "log_probabilities"), ("store", "response_storage")],
    )
    def test_refuses_a_flag_that_asks_for_the_feature(self, name, capability):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(**{name: True}))

        assert error.value.details["capability"] == capability
        assert error.value.details["parameter"] == name

    @pytest.mark.parametrize("value", [True, 5.5, "5", [], {}])
    def test_refuses_a_top_logprobs_that_is_not_an_integer_or_null(self, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(top_logprobs=value))

        assert "parameter 'top_logprobs'" in error.value.message

    def test_accepts_a_null_top_logprobs(self):
        assert ChatRequest.from_payload(body(top_logprobs=None)).invocation.model == "text-default"

    def test_refuses_a_top_logprobs_that_asks_for_log_probabilities(self):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(top_logprobs=5))

        assert error.value.details["capability"] == "log_probabilities"
        assert error.value.details["parameter"] == "top_logprobs"

    @pytest.mark.parametrize("name", ["logit_bias", "stream_options"])
    @pytest.mark.parametrize("value", [None, 0, "", []])
    def test_refuses_a_parameter_that_is_not_an_object(self, name, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert f"parameter {name!r}" in error.value.message
        assert "an object" in error.value.message

    @pytest.mark.parametrize("name", ["logit_bias", "stream_options"])
    def test_accepts_the_empty_object_default(self, name):
        assert ChatRequest.from_payload(body(**{name: {}})).invocation.model == "text-default"

    @pytest.mark.parametrize(
        ("name", "value", "capability"),
        [
            ("logit_bias", {"123": 5}, "logit_bias"),
            ("stream_options", {"include_usage": True}, "streaming"),
        ],
    )
    def test_refuses_a_populated_object(self, name, value, capability):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert error.value.details["capability"] == capability
        assert error.value.details["parameter"] == name

    @pytest.mark.parametrize("value", [0, "yes", [], {}])
    def test_refuses_a_parallel_tool_calls_that_is_not_a_boolean_or_null(self, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(parallel_tool_calls=value))

        assert "parameter 'parallel_tool_calls'" in error.value.message

    def test_accepts_a_null_parallel_tool_calls(self):
        assert (
            ChatRequest.from_payload(body(parallel_tool_calls=None)).invocation.model
            == "text-default"
        )

    @pytest.mark.parametrize("value", ["text", {}, None, 0])
    def test_refuses_a_modalities_that_is_not_an_array(self, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(modalities=value))

        assert "parameter 'modalities'" in error.value.message

    def test_accepts_the_text_only_default(self):
        assert (
            ChatRequest.from_payload(body(modalities=["text"])).invocation.model
            == "text-default"
        )

    @pytest.mark.parametrize("value", [[], ["text", "audio"]])
    def test_refuses_a_modalities_that_asks_for_non_text_output(self, value):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(modalities=value))

        assert error.value.details["capability"] == "non_text_output"
        assert error.value.details["parameter"] == "modalities"

    @pytest.mark.parametrize("name", ["audio", "prediction", "web_search_options"])
    @pytest.mark.parametrize("value", [[], "", 5])
    def test_refuses_an_object_parameter_of_the_wrong_type(self, name, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert f"parameter {name!r}" in error.value.message

    @pytest.mark.parametrize("name", ["audio", "prediction", "web_search_options"])
    def test_accepts_a_null_object_parameter(self, name):
        assert ChatRequest.from_payload(body(**{name: None})).invocation.model == "text-default"

    @pytest.mark.parametrize(
        ("name", "capability"),
        [
            ("audio", "audio_output"),
            ("prediction", "predicted_output"),
            ("web_search_options", "grounded_search"),
        ],
    )
    def test_refuses_a_populated_object_parameter(self, name, capability):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(**{name: {}}))

        assert error.value.details["capability"] == capability
        assert error.value.details["parameter"] == name

    @pytest.mark.parametrize("name", ["reasoning_effort", "service_tier", "function_call"])
    @pytest.mark.parametrize("value", [5, [], {}])
    def test_refuses_a_string_parameter_of_the_wrong_type(self, name, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert f"parameter {name!r}" in error.value.message

    @pytest.mark.parametrize("name", ["reasoning_effort", "service_tier", "function_call"])
    def test_accepts_a_null_string_parameter(self, name):
        assert ChatRequest.from_payload(body(**{name: None})).invocation.model == "text-default"

    @pytest.mark.parametrize(
        ("name", "value", "capability"),
        [
            ("reasoning_effort", "low", "reasoning_effort"),
            ("service_tier", "auto", "service_tier"),
            ("function_call", "none", "legacy_functions"),
        ],
    )
    def test_refuses_a_string_parameter_that_asks_for_the_feature(self, name, value, capability):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(**{name: value}))

        assert error.value.details["capability"] == capability
        assert error.value.details["parameter"] == name

    @pytest.mark.parametrize("value", ["fn", 5, None])
    def test_refuses_a_functions_that_is_not_an_array_or_object(self, value):
        with pytest.raises(InvalidArgumentError) as error:
            ChatRequest.from_payload(body(functions=value))

        assert "parameter 'functions'" in error.value.message

    @pytest.mark.parametrize("value", [[], {}])
    def test_accepts_an_empty_functions(self, value):
        assert ChatRequest.from_payload(body(functions=value)).invocation.model == "text-default"

    @pytest.mark.parametrize(
        "value",
        [
            [{"type": "function", "function": {"name": "get_weather"}}],
            {"get_weather": {"description": "Look up the weather"}},
        ],
    )
    def test_refuses_a_populated_functions(self, value):
        with pytest.raises(CapabilityError) as error:
            ChatRequest.from_payload(body(functions=value))

        assert error.value.details["capability"] == "legacy_functions"
        assert error.value.details["parameter"] == "functions"


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

    def test_reports_text_followed_by_tool_calls(self):
        completion = ChatCompletion.from_invocation(
            response(
                content=(
                    TextPart("Checking the weather now."),
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                raw_finish_reason="tool_calls",
            ),
            completion_id="chatcmpl-1",
            created=0,
        )

        message = completion.to_dict()["choices"][0]["message"]

        assert message["content"] == "Checking the weather now."
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.parametrize(
        "content",
        [
            (
                ToolCall(id="call_1", name="get_weather"),
                TextPart("The weather is sunny."),
            ),
            (
                TextPart("Checking."),
                ToolCall(id="call_1", name="get_weather"),
                TextPart("It is sunny."),
            ),
            (
                ToolCall(id="call_1", name="get_weather"),
                RefusalPart("I cannot share that."),
            ),
        ],
    )
    def test_refuses_an_output_order_the_response_shape_cannot_carry(self, content):
        """Concatenating the text and hoisting the calls would invent an answer.

        The OpenAI message records no order between ``content`` and
        ``tool_calls``, so an interleaved canonical response has no faithful
        representation in it (PRD section 9.1).
        """
        with pytest.raises(CapabilityError) as error:
            ChatCompletion.from_invocation(
                response(
                    content=content,
                    finish_reason=FinishReason.TOOL_CALLS,
                    raw_finish_reason="tool_calls",
                ),
                completion_id="chatcmpl-1",
                created=0,
            )

        assert error.value.reason.value == "unsupported_capability"
        assert error.value.details["capability"] == "interleaved_output"
        assert error.value.http_status == 422

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
