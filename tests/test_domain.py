"""Canonical domain types and their golden fixtures (PRD sections 4, 5, 9.1, 9.2, 9.3, 13)."""

import dataclasses
import json
from pathlib import Path

import pytest

from model_gateway.domain import (
    FinishReason,
    FrozenJsonObject,
    GenerationConfig,
    IdempotencyStatus,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    Message,
    ResponseModality,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResult,
    Usage,
    UsageSource,
    content_part_from_dict,
    freeze_json,
)

FIXTURES = Path(__file__).parent / "fixtures" / "canonical"

FIXTURE_TYPES = {
    "invocation_request.json": InvocationRequest,
    "invocation_response.json": InvocationResponse,
    "invocation_response_image.json": InvocationResponse,
}

FIXTURE_NAMES = tuple(sorted(FIXTURE_TYPES))

USER_MESSAGE = Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),))

REQUEST = InvocationRequest(
    model="text-default",
    messages=(USER_MESSAGE,),
    metadata={"trace": {"sampled": True}},
)
RESPONSE = InvocationResponse(model="text-default", upstream_model="upstream-model")


def request(**overrides) -> InvocationRequest:
    values = {"model": "text-default", "messages": (USER_MESSAGE,)}
    values.update(overrides)
    return InvocationRequest(**values)


def response(**overrides) -> InvocationResponse:
    values = {"model": "text-default", "upstream_model": "upstream-model"}
    values.update(overrides)
    return InvocationResponse(**values)


def fixture(name: str) -> tuple[object, str]:
    """A fixture's parsed document and its exact text."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return json.loads(text), text


# --------------------------------------------------------------------------- fixtures


def test_every_fixture_has_a_canonical_type() -> None:
    """A fixture added without a type would otherwise go unchecked."""
    assert tuple(sorted(path.name for path in FIXTURES.glob("*.json"))) == FIXTURE_NAMES


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_round_trip_reproduces_the_same_structure(name: str) -> None:
    document, text = fixture(name)
    parsed = FIXTURE_TYPES[name].from_dict(document)

    assert parsed.to_dict() == document
    assert json.dumps(parsed.to_dict(), indent=2) + "\n" == text


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_round_trip_is_stable_under_reparsing(name: str) -> None:
    document, _ = fixture(name)
    once = FIXTURE_TYPES[name].from_dict(document)
    twice = FIXTURE_TYPES[name].from_dict(once.to_dict())

    assert twice == once


def test_request_fixture_pins_a_multi_turn_conversation() -> None:
    document, _ = fixture("invocation_request.json")
    parsed = InvocationRequest.from_dict(document)

    assert [message.role for message in parsed.messages] == [
        Role.SYSTEM,
        Role.DEVELOPER,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.USER,
    ]


def test_request_fixture_pins_a_tool_call_with_structured_arguments() -> None:
    document, _ = fixture("invocation_request.json")
    parsed = InvocationRequest.from_dict(document)

    assistant = parsed.messages[3]
    assert assistant.tool_calls == (
        ToolCall(
            id="call_01",
            name="get_weather",
            arguments={"city": "Paris", "units": "metric", "days": 1},
        ),
    )
    assert assistant.tool_calls[0].arguments == {"city": "Paris", "units": "metric", "days": 1}
    assert parsed.messages[4].tool_call_id == "call_01"
    assert parsed.tool_choice == ToolChoice.specific("get_weather")


def test_request_fixture_pins_a_multimodal_message() -> None:
    document, _ = fixture("invocation_request.json")
    parsed = InvocationRequest.from_dict(document)

    parts = parsed.messages[5].content
    assert isinstance(parts[0], TextPart)
    assert parts[1].data is not None and parts[1].url is None
    assert parts[2].data is None and parts[2].url == "https://images.example/paris.jpg"
    assert parts[2].media_type == "image/jpeg"


def test_response_fixture_pins_a_tool_call_with_structured_arguments() -> None:
    document, _ = fixture("invocation_response.json")
    parsed = InvocationResponse.from_dict(document)

    assert parsed.finish_reason is FinishReason.TOOL_CALLS
    assert parsed.tool_calls[0].arguments == {"city": "Paris", "units": "metric", "days": 1}
    assert parsed.usage.source is UsageSource.PROVIDER
    assert parsed.usage.cached_tokens == 256
    assert parsed.attempt_count == 2
    assert parsed.fallback_used is True
    assert parsed.idempotency_status is IdempotencyStatus.REPLAYED
    assert parsed.estimated_cost == 0.00042
    assert parsed.cost_currency == "USD"


def test_image_response_fixture_pins_image_output_and_estimated_usage() -> None:
    document, _ = fixture("invocation_response_image.json")
    parsed = InvocationResponse.from_dict(document)

    assert parsed.text == ""
    assert parsed.images[0].media_type == "image/png"
    assert parsed.usage.source is UsageSource.ESTIMATE
    assert parsed.usage.is_estimated is True
    assert parsed.usage.output_tokens is None
    assert parsed.estimated_cost is None


# --------------------------------------------------------------------------- enums


def test_role_vocabulary() -> None:
    assert [role.value for role in Role] == ["system", "developer", "user", "assistant", "tool"]


def test_finish_reason_vocabulary() -> None:
    assert [reason.value for reason in FinishReason] == [
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "unknown",
    ]


def test_usage_source_vocabulary() -> None:
    assert [source.value for source in UsageSource] == ["provider", "estimate"]


def test_response_modality_vocabulary() -> None:
    assert [modality.value for modality in ResponseModality] == ["text", "image", "multimodal"]


def test_idempotency_status_vocabulary() -> None:
    assert [status.value for status in IdempotencyStatus] == [
        "not_applicable",
        "new",
        "replayed",
        "indeterminate",
    ]


def test_tool_choice_mode_vocabulary() -> None:
    assert [mode.value for mode in ToolChoiceMode] == ["auto", "none", "required", "specific"]


@pytest.mark.parametrize(
    "reason", [FinishReason.STOP, FinishReason.LENGTH, FinishReason.TOOL_CALLS]
)
def test_finish_reason_coerce_accepts_its_own_values(reason: FinishReason) -> None:
    assert FinishReason.coerce(reason.value) is reason


def test_finish_reason_coerce_maps_an_unrecognised_reason_to_unknown() -> None:
    """An unknown provider reason is classified, not dropped (PRD section 5.2)."""
    assert FinishReason.coerce("end_turn") is FinishReason.UNKNOWN
    assert FinishReason.coerce("") is FinishReason.UNKNOWN


def test_finish_reason_coerce_maps_a_missing_reason_to_unknown() -> None:
    assert FinishReason.coerce(None) is FinishReason.UNKNOWN


def test_finish_reason_coerce_does_not_guess_at_case() -> None:
    """Inventing a mapping the gateway was never given is provider translation."""
    assert FinishReason.coerce("STOP") is FinishReason.UNKNOWN
    assert FinishReason.coerce("Stop") is FinishReason.UNKNOWN


def test_response_maps_an_unrecognised_finish_reason_to_unknown() -> None:
    assert response(finish_reason="max_tokens").finish_reason is FinishReason.UNKNOWN


def test_response_defaults_an_absent_finish_reason_to_unknown() -> None:
    assert response().finish_reason is FinishReason.UNKNOWN


def test_response_refuses_a_finish_reason_that_is_not_a_string() -> None:
    with pytest.raises(ValueError, match="finish_reason must be a string"):
        response(finish_reason=5)


def test_message_accepts_a_role_by_its_string_value() -> None:
    assert Message(role="user").role is Role.USER


def test_message_refuses_a_role_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="message 'role' must be one of"):
        Message(role="wizard")


def test_message_refuses_a_role_that_is_not_a_string() -> None:
    with pytest.raises(ValueError, match="message 'role' must be a string"):
        Message(role=3)


def test_message_refuses_an_unknown_role_when_parsing() -> None:
    """A role the gateway cannot name cannot be carried into a message (PRD 9.3)."""
    with pytest.raises(ValueError, match="message 'role' must be one of"):
        Message.from_dict({"role": "wizard", "content": []})


# --------------------------------------------------------------------------- immutability


INSTANCES = (
    ("text part", TextPart(text="hi"), "text"),
    ("image part", ImagePart(data="AAAA"), "data"),
    ("message", USER_MESSAGE, "role"),
    ("tool definition", ToolDefinition(name="get_weather"), "name"),
    ("tool call", ToolCall(id="call_01", name="get_weather"), "id"),
    ("tool result", ToolResult(tool_call_id="call_01"), "content"),
    ("tool choice", ToolChoice.specific("get_weather"), "mode"),
    ("generation config", GenerationConfig(temperature=0.2), "temperature"),
    ("usage", Usage(input_tokens=1), "input_tokens"),
    ("request", REQUEST, "model"),
    ("response", RESPONSE, "model"),
)


@pytest.mark.parametrize(
    ("instance", "attribute"),
    [pytest.param(instance, attribute, id=label) for label, instance, attribute in INSTANCES],
)
def test_values_are_frozen(instance: object, attribute: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, attribute, "changed")


@pytest.mark.parametrize(
    ("instance", "attribute"),
    [pytest.param(instance, attribute, id=label) for label, instance, attribute in INSTANCES],
)
def test_values_are_hashable(instance: object, attribute: str) -> None:
    assert hash(instance) == hash(instance)
    assert len({instance}) == 1


def test_equal_json_objects_hash_alike_whatever_the_key_order() -> None:
    first = freeze_json("first", {"a": 1, "b": {"c": [1, 2]}})
    second = freeze_json("second", {"b": {"c": [1, 2]}, "a": 1})

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda value: value.__setitem__("new", 1), id="setitem"),
        pytest.param(lambda value: value.update({"new": 1}), id="update"),
        pytest.param(lambda value: value.pop("a"), id="pop"),
        pytest.param(lambda value: value.popitem(), id="popitem"),
        pytest.param(lambda value: value.clear(), id="clear"),
        pytest.param(lambda value: value.setdefault("new", 1), id="setdefault"),
        pytest.param(lambda value: value.__ior__({"new": 1}), id="ior"),
        pytest.param(lambda value: value.__delitem__("a"), id="delitem"),
    ],
)
def test_json_objects_refuse_every_mutator(mutate) -> None:
    """A shallow freeze would let a validated request acquire a field afterwards."""
    with pytest.raises(TypeError, match="immutable"):
        mutate(freeze_json("value", {"a": 1}))


def test_nested_json_is_frozen_all_the_way_down() -> None:
    definition = ToolDefinition(
        name="get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )

    assert isinstance(definition.parameters, FrozenJsonObject)
    assert isinstance(definition.parameters["properties"], FrozenJsonObject)
    assert isinstance(definition.parameters["properties"]["city"], FrozenJsonObject)
    with pytest.raises(TypeError, match="immutable"):
        definition.parameters["properties"]["city"]["type"] = "number"


def test_freezing_a_json_object_does_not_retain_the_source_mapping() -> None:
    source = {"city": "Paris"}
    call = ToolCall(id="call_01", name="get_weather", arguments=source)

    source["city"] = "Berlin"
    source["units"] = "metric"

    assert call.arguments == {"city": "Paris"}


def test_json_arrays_become_tuples() -> None:
    definition = ToolDefinition(name="t", parameters={"required": ["city"]})

    assert definition.parameters["required"] == ("city",)
    assert isinstance(definition.parameters["required"], tuple)


def test_collections_are_normalised_to_tuples() -> None:
    message = Message(role=Role.USER, content=[TextPart(text="hi")])
    config = GenerationConfig(stop_sequences=["END"])

    assert message.content == (TextPart(text="hi"),)
    assert config.stop_sequences == ("END",)
    assert isinstance(config.stop_sequences, tuple)


def test_json_objects_survive_copying_and_pickling_as_frozen_types() -> None:
    import copy
    import pickle

    original = freeze_json("value", {"a": {"b": 1}})

    assert isinstance(copy.deepcopy(original), FrozenJsonObject)
    assert isinstance(pickle.loads(pickle.dumps(original)), FrozenJsonObject)


# --------------------------------------------------------------------------- content parts


def test_text_part_round_trip() -> None:
    assert TextPart.from_dict({"type": "text", "text": "hi"}) == TextPart(text="hi")
    assert TextPart(text="hi").to_dict() == {"type": "text", "text": "hi"}


def test_text_part_allows_empty_text() -> None:
    assert TextPart(text="").text == ""


def test_text_part_refuses_a_non_string() -> None:
    with pytest.raises(ValueError, match="text part 'text' must be a string"):
        TextPart(text=1)


def test_text_part_refuses_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="unsupported key 'weight'"):
        TextPart.from_dict({"type": "text", "text": "hi", "weight": 1})


def test_image_part_carries_base64_data() -> None:
    part = ImagePart(data="AAAA", media_type="image/png")

    assert part.data == "AAAA"
    assert part.url is None
    assert part.media_type == "image/png"


def test_image_part_carries_an_authorised_url() -> None:
    part = ImagePart(url="https://images.example/a.png")

    assert part.url == "https://images.example/a.png"
    assert part.data is None
    assert part.media_type is None


def test_image_part_refuses_both_a_payload_and_a_url() -> None:
    with pytest.raises(ValueError, match="exactly one of 'data'"):
        ImagePart(data="AAAA", url="https://images.example/a.png")


def test_image_part_refuses_neither_a_payload_nor_a_url() -> None:
    with pytest.raises(ValueError, match="exactly one of 'data'"):
        ImagePart(media_type="image/png")


def test_image_part_round_trip() -> None:
    part = ImagePart(data="AAAA")

    assert ImagePart.from_dict(part.to_dict()) == part
    assert part.to_dict() == {"type": "image", "data": "AAAA", "url": None, "media_type": None}


def test_content_part_dispatches_on_its_type() -> None:
    assert content_part_from_dict({"type": "text", "text": "hi"}) == TextPart(text="hi")
    assert content_part_from_dict({"type": "image", "data": "AAAA"}) == ImagePart(data="AAAA")


def test_content_part_refuses_an_unknown_type() -> None:
    """A part the gateway cannot represent must not be skipped (PRD section 9.3)."""
    with pytest.raises(ValueError, match="does not define"):
        content_part_from_dict({"type": "audio", "data": "AAAA"})


def test_content_part_refuses_a_missing_type() -> None:
    with pytest.raises(ValueError, match="discriminator"):
        content_part_from_dict({"text": "hi"})


def test_content_part_refuses_a_non_object() -> None:
    with pytest.raises(ValueError, match="content part must be a JSON object"):
        content_part_from_dict(["text"])


def test_message_parses_its_content_parts() -> None:
    message = Message.from_dict(
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}, {"type": "image", "data": "AAAA"}],
            "name": None,
            "tool_calls": [],
            "tool_call_id": None,
        }
    )

    assert message.content == (TextPart(text="hi"), ImagePart(data="AAAA"))


def test_message_round_trip() -> None:
    message = Message(role=Role.USER, content=(TextPart(text="hi"),), name="alice")

    assert Message.from_dict(message.to_dict()) == message
    assert message.to_dict() == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
        "name": "alice",
        "tool_calls": [],
        "tool_call_id": None,
    }


def test_message_defaults() -> None:
    message = Message(role=Role.USER)

    assert message.content == ()
    assert message.name is None
    assert message.tool_calls == ()
    assert message.tool_call_id is None


def test_message_refuses_tool_calls_from_a_non_assistant() -> None:
    call = ToolCall(id="call_01", name="get_weather")

    with pytest.raises(ValueError, match="cannot carry tool calls"):
        Message(role=Role.USER, tool_calls=(call,))


def test_assistant_message_may_carry_tool_calls() -> None:
    call = ToolCall(id="call_01", name="get_weather")

    assert Message(role=Role.ASSISTANT, tool_calls=(call,)).tool_calls == (call,)


def test_message_refuses_a_tool_call_id_from_a_non_tool() -> None:
    with pytest.raises(ValueError, match="cannot carry a tool call id"):
        Message(role=Role.ASSISTANT, tool_call_id="call_01")


def test_tool_message_must_carry_the_call_it_answers() -> None:
    with pytest.raises(ValueError, match="must carry the 'tool_call_id'"):
        Message(role=Role.TOOL, content=(TextPart(text="18C"),))


def test_tool_message_carries_its_call_id() -> None:
    message = Message(role=Role.TOOL, tool_call_id="call_01", content=(TextPart(text="18C"),))

    assert message.tool_call_id == "call_01"


def test_message_refuses_a_content_part_that_is_not_a_part() -> None:
    with pytest.raises(ValueError, match="must be a TextPart or an ImagePart"):
        Message(role=Role.USER, content=("hi",))


# --------------------------------------------------------------------------- tools


def test_tool_definition_defaults() -> None:
    definition = ToolDefinition(name="get_weather")

    assert definition.description == ""
    assert definition.parameters == {}
    assert isinstance(definition.parameters, FrozenJsonObject)


def test_tool_definition_round_trip() -> None:
    definition = ToolDefinition(
        name="get_weather",
        description="Look up the weather.",
        parameters={"type": "object", "required": ["city"]},
    )

    assert ToolDefinition.from_dict(definition.to_dict()) == definition
    assert definition.to_dict() == {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {"type": "object", "required": ["city"]},
    }


def test_tool_definition_requires_a_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ToolDefinition(name="")


def test_tool_definition_keeps_the_schema_it_was_given() -> None:
    """A schema the gateway had rewritten would describe a different tool."""
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}

    assert ToolDefinition(name="t", parameters=parameters).parameters == parameters


def test_tool_call_defaults() -> None:
    call = ToolCall(id="call_01", name="get_weather")

    assert call.arguments is None


def test_tool_call_keeps_structured_arguments() -> None:
    call = ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris"})

    assert isinstance(call.arguments, FrozenJsonObject)
    assert call.to_dict()["arguments"] == {"city": "Paris"}


def test_tool_call_keeps_unparsed_arguments_as_a_raw_string() -> None:
    """Repairing malformed arguments would present a call the model never made."""
    call = ToolCall(id="call_01", name="get_weather", arguments='{"city": ')

    assert call.arguments == '{"city": '
    assert ToolCall.from_dict(call.to_dict()) == call


def test_tool_call_keeps_no_arguments_apart_from_empty_ones() -> None:
    assert ToolCall(id="c", name="t").arguments is None
    assert ToolCall(id="c", name="t", arguments={}).arguments == {}


def test_tool_call_round_trip() -> None:
    for arguments in ({"city": "Paris"}, "not json", None):
        call = ToolCall(id="call_01", name="get_weather", arguments=arguments)

        assert ToolCall.from_dict(call.to_dict()) == call


def test_tool_call_requires_an_id_and_a_name() -> None:
    with pytest.raises(ValueError, match="tool call 'id' must not be empty"):
        ToolCall(id="", name="get_weather")
    with pytest.raises(ValueError, match="tool call 'name' must not be empty"):
        ToolCall(id="call_01", name="")


def test_tool_call_refuses_arguments_that_are_not_json() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        ToolCall(id="call_01", name="get_weather", arguments=b"bytes")


def test_tool_result_defaults() -> None:
    result = ToolResult(tool_call_id="call_01")

    assert result.content == ""
    assert result.is_error is False


def test_tool_result_round_trip() -> None:
    result = ToolResult(tool_call_id="call_01", content="18C", is_error=True)

    assert ToolResult.from_dict(result.to_dict()) == result
    assert result.to_dict() == {
        "tool_call_id": "call_01",
        "content": "18C",
        "is_error": True,
    }


def test_tool_result_requires_the_call_it_answers() -> None:
    with pytest.raises(ValueError, match="tool_call_id' must not be empty"):
        ToolResult(tool_call_id="")


def test_tool_choice_factories() -> None:
    assert ToolChoice.auto() == ToolChoice(mode=ToolChoiceMode.AUTO)
    assert ToolChoice.none() == ToolChoice(mode=ToolChoiceMode.NONE)
    assert ToolChoice.required() == ToolChoice(mode=ToolChoiceMode.REQUIRED)
    assert ToolChoice.specific("get_weather") == ToolChoice(
        mode=ToolChoiceMode.SPECIFIC, name="get_weather"
    )


def test_tool_choice_defaults_to_auto() -> None:
    assert ToolChoice().mode is ToolChoiceMode.AUTO
    assert ToolChoice().name is None


def test_tool_choice_refuses_specific_without_a_name() -> None:
    with pytest.raises(ValueError, match="must name the tool"):
        ToolChoice(mode=ToolChoiceMode.SPECIFIC)


def test_tool_choice_refuses_a_name_for_a_non_specific_mode() -> None:
    with pytest.raises(ValueError, match="must not name a tool"):
        ToolChoice(mode=ToolChoiceMode.REQUIRED, name="get_weather")


def test_tool_choice_round_trip() -> None:
    for choice in (ToolChoice.auto(), ToolChoice.none(), ToolChoice.required()):
        assert ToolChoice.from_dict(choice.to_dict()) == choice

    specific = ToolChoice.specific("get_weather")
    assert ToolChoice.from_dict(specific.to_dict()) == specific
    assert specific.to_dict() == {"mode": "specific", "name": "get_weather"}


def test_tool_choice_refuses_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="tool choice 'mode' must be one of"):
        ToolChoice(mode="sometimes")


# --------------------------------------------------------------------------- generation config


def test_generation_config_defaults_leave_every_parameter_unset() -> None:
    config = GenerationConfig()

    assert config.temperature is None
    assert config.top_p is None
    assert config.max_output_tokens is None
    assert config.stop_sequences == ()
    assert config.seed is None
    assert config.presence_penalty is None
    assert config.frequency_penalty is None


def test_generation_config_keeps_unset_apart_from_zero() -> None:
    """Substituting a vendor default would encode one vendor in the core."""
    assert GenerationConfig().temperature is None
    assert GenerationConfig(temperature=0.0).temperature == 0.0


def test_generation_config_round_trip() -> None:
    config = GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=512,
        stop_sequences=("END", "\n\nHuman:"),
        seed=7,
        presence_penalty=0.0,
        frequency_penalty=0.5,
    )

    assert GenerationConfig.from_dict(config.to_dict()) == config
    assert config.to_dict() == {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_output_tokens": 512,
        "stop_sequences": ["END", "\n\nHuman:"],
        "seed": 7,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.5,
    }


def test_generation_config_round_trips_an_unset_parameter_set() -> None:
    config = GenerationConfig()

    assert GenerationConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("temperature", -0.1, id="negative temperature"),
        pytest.param("top_p", -0.1, id="negative top_p"),
        pytest.param("top_p", 1.5, id="top_p above one"),
        pytest.param("max_output_tokens", 0, id="zero output budget"),
        pytest.param("max_output_tokens", -1, id="negative output budget"),
        pytest.param("temperature", float("nan"), id="nan temperature"),
        pytest.param("temperature", float("inf"), id="infinite temperature"),
    ],
)
def test_generation_config_enforces_provider_independent_bounds(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("temperature", "warm", id="temperature"),
        pytest.param("max_output_tokens", 1.5, id="max_output_tokens"),
        pytest.param("seed", "seven", id="seed"),
        pytest.param("stop_sequences", "END", id="bare stop string"),
        pytest.param("stop_sequences", ["", "END"], id="empty stop string"),
    ],
)
def test_generation_config_refuses_a_value_of_the_wrong_shape(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**{field: value})


def test_generation_config_refuses_a_boolean_count() -> None:
    with pytest.raises(ValueError, match="max_output_tokens must be an integer"):
        GenerationConfig(max_output_tokens=True)


def test_generation_config_refuses_an_unknown_key() -> None:
    with pytest.raises(ValueError, match="unsupported key 'stream'"):
        GenerationConfig.from_dict({"temperature": 0.2, "stream": True})


def test_generation_config_parses_an_absent_parameter_set() -> None:
    assert GenerationConfig.from_dict({}) == GenerationConfig()


# --------------------------------------------------------------------------- usage


def test_usage_defaults_are_unknown_not_zero() -> None:
    """A count the provider never reported is not the same as a count of zero."""
    usage = Usage()

    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.cached_tokens is None
    assert usage.source is UsageSource.PROVIDER
    assert usage.is_estimated is False


def test_usage_distinguishes_reported_counts_from_estimates() -> None:
    reported = Usage(input_tokens=412, output_tokens=38, total_tokens=450)
    estimated = Usage(input_tokens=412, output_tokens=38, total_tokens=450, source="estimate")

    assert reported.is_estimated is False
    assert estimated.is_estimated is True
    assert reported != estimated


def test_usage_keeps_an_unreported_cached_count_apart_from_zero() -> None:
    assert Usage(cached_tokens=None).cached_tokens is None
    assert Usage(cached_tokens=0).cached_tokens == 0


def test_usage_round_trip() -> None:
    usage = Usage(input_tokens=412, output_tokens=38, total_tokens=450, cached_tokens=256)

    assert Usage.from_dict(usage.to_dict()) == usage
    assert usage.to_dict() == {
        "input_tokens": 412,
        "output_tokens": 38,
        "total_tokens": 450,
        "cached_tokens": 256,
        "source": "provider",
    }


def test_usage_round_trips_an_unknown_record() -> None:
    usage = Usage()

    assert Usage.from_dict(usage.to_dict()) == usage


def test_usage_defaults_an_absent_source_to_provider() -> None:
    assert Usage.from_dict({"input_tokens": 1}).source is UsageSource.PROVIDER


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "total_tokens", "cached_tokens"]
)
def test_usage_refuses_a_negative_count(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be at least 0"):
        Usage(**{field: -1})


def test_usage_refuses_a_boolean_count() -> None:
    with pytest.raises(ValueError, match="input_tokens must be an integer"):
        Usage(input_tokens=True)


def test_usage_refuses_an_unknown_source() -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        Usage(source="guessed")


# --------------------------------------------------------------------------- invocation request


def test_request_requires_a_model() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        request(model="")


def test_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        request(messages=())


def test_request_defaults() -> None:
    invocation = request()

    assert invocation.invocation_id is None
    assert invocation.tenant_id is None
    assert invocation.actor_id is None
    assert invocation.agent_id is None
    assert invocation.system_instruction is None
    assert invocation.generation_config == GenerationConfig()
    assert invocation.tools == ()
    assert invocation.tool_choice is None
    assert invocation.response_modality is ResponseModality.TEXT
    assert invocation.fallback_models == ()
    assert invocation.idempotency_key is None
    assert invocation.metadata == {}


def test_request_metadata_is_frozen_json() -> None:
    invocation = request(metadata={"trace": {"sampled": True}})

    assert isinstance(invocation.metadata, FrozenJsonObject)
    assert isinstance(invocation.metadata["trace"], FrozenJsonObject)
    with pytest.raises(TypeError, match="immutable"):
        invocation.metadata["trace"]["sampled"] = False


def test_request_refuses_metadata_that_is_not_json() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        request(metadata={"when": object()})


def test_request_round_trip() -> None:
    invocation = request(
        invocation_id="inv-1",
        tenant_id="acme",
        actor_id="svc",
        agent_id="agent",
        system_instruction="Answer concisely.",
        generation_config=GenerationConfig(temperature=0.2, stop_sequences=("END",)),
        tools=(ToolDefinition(name="get_weather"),),
        tool_choice=ToolChoice.specific("get_weather"),
        response_modality=ResponseModality.MULTIMODAL,
        fallback_models=("text-default",),
        idempotency_key="idem-1",
        metadata={"trace": {"sampled": True}},
    )

    assert InvocationRequest.from_dict(invocation.to_dict()) == invocation


def test_request_round_trips_its_defaults() -> None:
    invocation = request()

    assert InvocationRequest.from_dict(invocation.to_dict()) == invocation


def test_request_refuses_a_required_tool_choice_without_tools() -> None:
    with pytest.raises(ValueError, match="requires at least one declared tool"):
        request(tool_choice=ToolChoice.required())


def test_request_refuses_a_specific_choice_naming_an_undeclared_tool() -> None:
    with pytest.raises(ValueError, match="which the invocation does not declare"):
        request(
            tools=(ToolDefinition(name="get_weather"),),
            tool_choice=ToolChoice.specific("send_email"),
        )


def test_request_accepts_a_specific_choice_naming_a_declared_tool() -> None:
    invocation = request(
        tools=(ToolDefinition(name="get_weather"),),
        tool_choice=ToolChoice.specific("get_weather"),
    )

    assert invocation.tool_choice == ToolChoice.specific("get_weather")


def test_request_accepts_a_none_choice_without_tools() -> None:
    assert request(tool_choice=ToolChoice.none()).tool_choice == ToolChoice.none()


def test_request_refuses_a_message_that_is_not_a_message() -> None:
    with pytest.raises(ValueError, match=r"messages\[0\] must be a Message"):
        request(messages=({"role": "user"},))


def test_request_refuses_a_tool_that_is_not_a_declaration() -> None:
    with pytest.raises(ValueError, match=r"tools\[0\] must be a ToolDefinition"):
        request(tools=({"name": "get_weather"},))


def test_request_refuses_fallback_models_given_as_a_bare_string() -> None:
    with pytest.raises(ValueError, match="fallback_models must be a JSON array of strings"):
        request(fallback_models="text-default")


def test_request_refuses_a_generation_config_that_is_not_one() -> None:
    with pytest.raises(ValueError, match="generation_config must be a GenerationConfig"):
        request(generation_config={"temperature": 0.2})


def test_request_parses_its_defaults_from_a_minimal_document() -> None:
    parsed = InvocationRequest.from_dict(
        {"model": "text-default", "messages": [{"role": "user", "content": []}]}
    )

    assert parsed == request(messages=(Message(role=Role.USER),))


def test_request_refuses_a_document_with_no_messages() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        InvocationRequest.from_dict({"model": "text-default", "messages": []})


# --------------------------------------------------------------------------- invocation response


def test_response_defaults() -> None:
    result = response()

    assert result.invocation_id is None
    assert result.text == ""
    assert result.tool_calls == ()
    assert result.images == ()
    assert result.finish_reason is FinishReason.UNKNOWN
    assert result.usage == Usage()
    assert result.estimated_cost is None
    assert result.cost_currency is None
    assert result.attempt_count == 1
    assert result.fallback_used is False
    assert result.idempotency_status is IdempotencyStatus.NOT_APPLICABLE


def test_response_requires_the_model_and_the_upstream_model() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        response(model="")
    with pytest.raises(ValueError, match="upstream_model must not be empty"):
        response(upstream_model="")


def test_response_refuses_a_zero_attempt_count() -> None:
    with pytest.raises(ValueError, match="attempt_count must be at least 1"):
        response(attempt_count=0)


def test_response_reports_the_upstream_model_separately_from_the_public_one() -> None:
    """A caller that cannot tell which upstream answered cannot reconcile its own books."""
    result = response(model="text-default", upstream_model="fallback-chat-model")

    assert (result.model, result.upstream_model) == ("text-default", "fallback-chat-model")


def test_response_round_trip() -> None:
    result = response(
        invocation_id="inv-1",
        text="Checking the weather now.",
        tool_calls=(ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris"}),),
        images=(ImagePart(data="AAAA", media_type="image/png"),),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=412, output_tokens=38, total_tokens=450, cached_tokens=256),
        estimated_cost=0.00042,
        cost_currency="USD",
        attempt_count=2,
        fallback_used=True,
        idempotency_status=IdempotencyStatus.REPLAYED,
    )

    assert InvocationResponse.from_dict(result.to_dict()) == result


def test_response_round_trips_its_defaults() -> None:
    result = response()

    assert InvocationResponse.from_dict(result.to_dict()) == result


def test_response_requires_a_cost_and_a_currency_together() -> None:
    with pytest.raises(ValueError, match="must be given together"):
        response(estimated_cost=0.5)
    with pytest.raises(ValueError, match="must be given together"):
        response(cost_currency="USD")


@pytest.mark.parametrize("currency", ["usd", "US", "USDD", "123"])
def test_response_refuses_a_currency_that_is_not_an_iso_code(currency: str) -> None:
    with pytest.raises(ValueError, match="ISO 4217"):
        response(estimated_cost=0.5, cost_currency=currency)


def test_response_refuses_a_negative_cost() -> None:
    with pytest.raises(ValueError, match="estimated_cost must be at least 0"):
        response(estimated_cost=-0.5, cost_currency="USD")


def test_response_refuses_an_unknown_idempotency_status() -> None:
    with pytest.raises(ValueError, match="idempotency_status must be one of"):
        response(idempotency_status="maybe")


def test_response_accepts_an_idempotency_status_by_its_string_value() -> None:
    assert response(idempotency_status="new").idempotency_status is IdempotencyStatus.NEW


def test_response_refuses_an_image_that_is_not_an_image_part() -> None:
    with pytest.raises(ValueError, match=r"images\[0\] must be an ImagePart"):
        response(images=({"data": "AAAA"},))


def test_response_refuses_a_tool_call_that_is_not_a_tool_call() -> None:
    with pytest.raises(ValueError, match=r"tool_calls\[0\] must be a ToolCall"):
        response(tool_calls=({"id": "call_01"},))


def test_response_refuses_a_usage_that_is_not_a_usage() -> None:
    with pytest.raises(ValueError, match="usage must be a Usage"):
        response(usage={"input_tokens": 1})


def test_response_parses_its_defaults_from_a_minimal_document() -> None:
    parsed = InvocationResponse.from_dict(
        {"model": "text-default", "upstream_model": "upstream-model"}
    )

    assert parsed == response()


# --------------------------------------------------------------------------- explicit failure


@pytest.mark.parametrize(
    ("kind", "document"),
    [
        pytest.param(
            InvocationRequest,
            {"model": "text-default", "messages": [], "stream": True},
            id="request",
        ),
        pytest.param(
            InvocationResponse,
            {"model": "text-default", "upstream_model": "upstream", "logprobs": []},
            id="response",
        ),
        pytest.param(
            Message,
            {"role": "user", "content": [], "refusal": None},
            id="message",
        ),
        pytest.param(
            ToolDefinition,
            {"name": "t", "strict": True},
            id="tool definition",
        ),
        pytest.param(
            ToolCall,
            {"id": "call_01", "name": "t", "index": 0},
            id="tool call",
        ),
        pytest.param(
            ToolResult,
            {"tool_call_id": "call_01", "content": "18C", "is_error": False, "extra": 1},
            id="tool result",
        ),
        pytest.param(
            ToolChoice,
            {"mode": "auto", "name": None, "disable_parallel_tool_use": True},
            id="tool choice",
        ),
        pytest.param(
            GenerationConfig,
            {"temperature": 0.2, "logit_bias": {}},
            id="generation config",
        ),
        pytest.param(
            Usage,
            {"input_tokens": 1, "reasoning_tokens": 0},
            id="usage",
        ),
        pytest.param(
            ImagePart,
            {"data": "AAAA", "detail": "high"},
            id="image part",
        ),
        pytest.param(
            TextPart,
            {"text": "hi", "cache_control": {"type": "ephemeral"}},
            id="text part",
        ),
    ],
)
def test_an_unrepresentable_field_fails_explicitly(kind: type, document: dict) -> None:
    """PRD section 9.3: a parameter must never disappear silently."""
    with pytest.raises(ValueError, match="unsupported key"):
        kind.from_dict(document)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"bytes", id="bytes"),
        pytest.param({1, 2}, id="set"),
        pytest.param(object(), id="arbitrary object"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_a_value_json_cannot_carry_fails_explicitly(value: object) -> None:
    with pytest.raises(ValueError):
        freeze_json("metadata", value)


def test_a_non_string_json_key_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="non-string key"):
        freeze_json("metadata", {1: "one"})


@pytest.mark.parametrize(
    ("kind", "document"),
    [
        pytest.param(InvocationRequest, "not an object", id="request"),
        pytest.param(InvocationResponse, [], id="response"),
        pytest.param(Message, None, id="message"),
        pytest.param(Usage, 3, id="usage"),
    ],
)
def test_a_document_that_is_not_an_object_fails_explicitly(kind: type, document: object) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        kind.from_dict(document)
