"""The protobuf↔canonical translation, asserted field by field (PRD section 5).

Every test here drives the mappers in :mod:`model_gateway.grpc.mapping` directly
with generated protobuf messages, so what is asserted is what crosses the wire:
the canonical record a protobuf request builds, and the protobuf message a
canonical answer renders to. No server is involved — the transport's handlers are
covered by the sibling RPC test files, and this file owns the translation alone.

Four rules shape the file.

**A Struct carries the whole of JSON, at the cost of exact integers.** Metadata
crosses as a ``google.protobuf.Struct``, so a nested object, a boolean and a list
all reach the canonical record; the well-known type's numbers are doubles, which
is why a small integer compares equal to its float form rather than arriving as
a different value.

**A JSON Schema may be a boolean.** ``ResponseFormat.schema`` is a
``google.protobuf.Value`` so that ``true`` and ``false`` — both valid schema
documents — cross as themselves, and a schema of any other kind is refused by the
mapper rather than coerced.

**Tool-call arguments travel as JSON text.** ``bytes arguments_json`` keeps a
JSON integer exact, which the well-known value could not; the mapper decodes with
:func:`orjson.loads`, so a JSON string stays a string, an object becomes a
mapping, ``null`` and an absent field both become ``None``, and any other JSON
value is refused.

**A request the canonical constructors refuse is an invalid argument, not an
internal error.** The canonical types validate heavily — the model, the
conversation, the tool choice against the declared tools, the generation bounds,
the metadata keys — and a ``ValueError`` from one is wrapped as
:class:`InvalidArgumentError` naming the rule, exactly as the embeddings request
mapper already wraps.
"""

import orjson
import pytest
from google.protobuf.struct_pb2 import Struct, Value

from model_gateway.domain import (
    FinishReason,
    InvocationResponse,
    ResponseFormat,
    ResponseFormatMode,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.domain.json_value import FrozenJsonObject
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import InvalidArgumentError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import mapping
from model_gateway.security.identity import Identity

PUBLIC_MODEL = "text-default"
TENANT = "tenant-a"


def _list_value(values):
    """A ``google.protobuf.Value`` carrying a JSON array."""
    value = Value()
    value.list_value.extend(values)
    return value


def identity() -> Identity:
    """The authenticated identity the invocation mapper expects."""
    return Identity(authenticated=True, tenant=TENANT, credential_id="cred-1", key_label="key-1")


def user_message(text: str = "Explain recursion.") -> pb.Message:
    return pb.Message(role=pb.ROLE_USER, content=(pb.ContentPart(text=pb.TextPart(text=text)),))


def invoke_request(**overrides) -> pb.InvokeRequest:
    """An ``InvokeRequest`` a caller could send: one user turn, the modality stated."""
    fields = {
        "model": PUBLIC_MODEL,
        "messages": (user_message(),),
        "response_modality": pb.RESPONSE_MODALITY_TEXT,
    }
    fields.update(overrides)
    return pb.InvokeRequest(**fields)


def completion(**overrides) -> InvocationResponse:
    """An ``InvocationResponse`` a provider could have produced."""
    fields = {
        "model": PUBLIC_MODEL,
        "upstream_model": "upstream-text",
        "content": (TextPart(text="Answered."),),
        "finish_reason": FinishReason.STOP,
    }
    fields.update(overrides)
    return InvocationResponse(**fields)


# --------------------------------------------------------------------------- #
# Metadata: a Struct both ways
# --------------------------------------------------------------------------- #


def test_a_struct_and_a_json_object_convert_both_ways():
    metadata = FrozenJsonObject({"trace": {"sampled": True}, "ids": [1, 2], "flag": True})
    struct = mapping._mapping_to_struct(metadata)
    assert isinstance(struct, Struct)
    assert mapping._struct_to_json(struct) == dict(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"trace": {"sampled": True}}, id="nested object"),
        pytest.param({"flag": True}, id="boolean"),
        pytest.param({"ids": [1, 2, 3]}, id="list"),
    ],
)
def test_request_metadata_of_every_json_kind_reaches_the_invocation(metadata):
    # A Struct types numbers as doubles and the canonical record freezes a JSON
    # array into a tuple, so the list comes back as (1.0, 2.0, 3.0); the nested
    # object and the boolean cross unchanged. Equality, not type identity, is
    # what a caller comparing the round trip can rely on.
    request = invoke_request()
    request.metadata.update(metadata)

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    expected = (
        {"ids": (1.0, 2.0, 3.0)} if "ids" in metadata else metadata
    )
    assert invocation.metadata == expected


def test_absent_request_metadata_becomes_an_empty_object():
    invocation = mapping.invocation_request_from_proto(invoke_request(), identity=identity())

    assert invocation.metadata == {}


def test_absent_embed_metadata_stays_absent():
    # The canonical embeddings request distinguishes no metadata from empty
    # metadata, and a Struct field has presence, so the distinction survives.
    request = pb.EmbedRequest(model=PUBLIC_MODEL, input=("first",))

    embedding_request = mapping.embed_request_from_proto(request)

    assert embedding_request.metadata is None


def test_present_embed_metadata_reaches_the_canonical_request():
    request = pb.EmbedRequest(model=PUBLIC_MODEL, input=("first",))
    request.metadata.update({"trace": "abc"})

    embedding_request = mapping.embed_request_from_proto(request)

    assert embedding_request.metadata == {"trace": "abc"}


# --------------------------------------------------------------------------- #
# A JSON Schema may be an object or a boolean
# --------------------------------------------------------------------------- #


def test_a_boolean_json_schema_round_trips():
    canonical = ResponseFormat(mode=ResponseFormatMode.JSON_SCHEMA, schema=True)

    proto = mapping._response_format_to_proto(canonical)

    assert proto.schema.WhichOneof("kind") == "bool_value"
    assert proto.schema.bool_value is True
    assert mapping.response_format_from_proto("response_format", proto) == canonical


def test_an_object_json_schema_round_trips():
    schema = FrozenJsonObject({"type": "object", "properties": {"city": {"type": "string"}}})
    canonical = ResponseFormat(mode=ResponseFormatMode.JSON_SCHEMA, schema=schema)

    proto = mapping._response_format_to_proto(canonical)

    assert proto.schema.WhichOneof("kind") == "struct_value"
    assert mapping.response_format_from_proto("response_format", proto) == canonical


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(Value(number_value=42.0), id="number"),
        pytest.param(Value(string_value="true"), id="string"),
        pytest.param(Value(), id="null"),
        pytest.param(_list_value([1.0]), id="array"),
    ],
)
def test_a_schema_of_any_other_kind_is_refused(schema):
    proto = pb.ResponseFormat(mode=pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA)
    proto.schema.CopyFrom(schema)

    with pytest.raises(InvalidArgumentError, match="'schema'"):
        mapping.response_format_from_proto("response_format 'response_format'", proto)


def test_a_boolean_schema_in_a_generation_config_reaches_the_canonical_request():
    config = pb.GenerationConfig()
    config.response_format.mode = pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA
    config.response_format.schema.CopyFrom(Value(bool_value=True))

    invocation = mapping.invocation_request_from_proto(
        invoke_request(generation_config=config), identity=identity()
    )

    assert invocation.generation_config.response_format == ResponseFormat(
        mode=ResponseFormatMode.JSON_SCHEMA, schema=True
    )


def test_a_json_schema_format_without_a_schema_is_refused():
    # The canonical constructor refuses a 'json_schema' mode carrying no schema,
    # and the request mapper wraps that refusal as an invalid argument rather
    # than letting it escape as the internal error a ValueError would become.
    config = pb.GenerationConfig()
    config.response_format.mode = pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA

    with pytest.raises(InvalidArgumentError, match="schema"):
        mapping.invocation_request_from_proto(
            invoke_request(generation_config=config), identity=identity()
        )


# --------------------------------------------------------------------------- #
# Usage: the derived counts are named, not implied
# --------------------------------------------------------------------------- #


def test_derived_counts_are_written_as_the_closed_enum():
    usage = Usage(input_tokens=25, total_tokens=25, estimated_fields={"total_tokens"})

    proto = mapping._usage_to_proto(usage)

    assert list(proto.estimated_fields) == [pb.USAGE_FIELD_TOTAL_TOKENS]


def test_every_derived_count_is_written():
    usage = Usage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        cached_tokens=4,
        estimated_fields={"input_tokens", "cached_tokens"},
    )

    proto = mapping._usage_to_proto(usage)

    # Written in sorted name order, which is alphabetical: 'cached_tokens'
    # sorts before 'input_tokens'.
    assert list(proto.estimated_fields) == [
        pb.USAGE_FIELD_CACHED_TOKENS,
        pb.USAGE_FIELD_INPUT_TOKENS,
    ]


def test_no_derived_counts_writes_an_empty_list():
    proto = mapping._usage_to_proto(Usage(input_tokens=25))

    assert list(proto.estimated_fields) == []


def test_derived_counts_carry_through_the_response():
    response = completion(usage=Usage(total_tokens=25, estimated_fields={"total_tokens"}))

    proto = mapping.invoke_response_to_proto(response)

    assert list(proto.usage.estimated_fields) == [pb.USAGE_FIELD_TOTAL_TOKENS]


# --------------------------------------------------------------------------- #
# raw_finish_reason: absent and empty stay distinct
# --------------------------------------------------------------------------- #


def test_an_absent_raw_finish_reason_stays_absent():
    proto = mapping.invoke_response_to_proto(completion())

    assert not proto.HasField("raw_finish_reason")


def test_an_empty_raw_finish_reason_stays_empty():
    proto = mapping.invoke_response_to_proto(completion(raw_finish_reason=""))

    assert proto.HasField("raw_finish_reason")
    assert proto.raw_finish_reason == ""


def test_a_stated_raw_finish_reason_is_written():
    proto = mapping.invoke_response_to_proto(completion(raw_finish_reason="eos"))

    assert proto.HasField("raw_finish_reason")
    assert proto.raw_finish_reason == "eos"


def test_an_absent_stream_raw_finish_reason_stays_absent():
    # The canonical StreamEvent refuses an empty raw_finish_reason at
    # construction, so the empty case cannot arise on the streaming path; the
    # optional field is still written only when a reason is present, which is
    # what keeps the two wire states distinct should one ever arrive.
    event = StreamEvent(
        kind=StreamEventKind.FINISH,
        sequence=2,
        invocation_id="inv-1",
        finish_reason=FinishReason.STOP,
    )

    chunk = mapping.chunk_from_event(event)

    assert not chunk.finish.HasField("raw_finish_reason")


# --------------------------------------------------------------------------- #
# Tool-call arguments: JSON text, exactly as canonical holds them
# --------------------------------------------------------------------------- #


def test_tool_call_arguments_as_an_object_round_trip():
    call = ToolCall(id="call_1", name="lookup", arguments={"query": "recursion"})

    proto = mapping._tool_call_to_proto(call)

    assert proto.arguments_json == b'{"query":"recursion"}'
    assert mapping.tool_call_from_proto("tool_calls", proto) == call


def test_tool_call_arguments_keep_a_json_integer_exact():
    # The reason arguments travel as bytes: the well-known value types every
    # number as a double, which cannot carry an integer beyond 2^53.
    call = ToolCall(id="call_1", name="lookup", arguments={"big": 2**53 + 1})

    proto = mapping._tool_call_to_proto(call)

    assert orjson.loads(proto.arguments_json) == {"big": 2**53 + 1}
    assert mapping.tool_call_from_proto("tool_calls", proto) == call


def test_tool_call_arguments_as_a_raw_string_round_trip():
    # A string that is not JSON is a fact of its own: it crosses quoted, so it
    # decodes back to the same string instead of being refused as malformed.
    call = ToolCall(id="call_1", name="lookup", arguments='{"city": ')

    proto = mapping._tool_call_to_proto(call)

    assert orjson.loads(proto.arguments_json) == '{"city": '
    assert mapping.tool_call_from_proto("tool_calls", proto) == call


def test_tool_call_arguments_absent_stays_absent():
    proto = pb.ToolCall(id="call_1", name="lookup")

    call = mapping.tool_call_from_proto("tool_calls", proto)

    assert call.arguments is None
    assert mapping._tool_call_to_proto(call).arguments_json == b""


def test_a_tool_call_argument_that_is_neither_object_nor_string_is_refused():
    proto = pb.ToolCall(id="call_1", name="lookup", arguments_json=b"42")

    with pytest.raises(InvalidArgumentError, match="arguments_json"):
        mapping.tool_call_from_proto("tool_calls", proto)


def test_malformed_argument_json_is_refused():
    proto = pb.ToolCall(id="call_1", name="lookup", arguments_json=b"{not json")

    with pytest.raises(InvalidArgumentError, match="arguments_json"):
        mapping.tool_call_from_proto("tool_calls", proto)


def test_request_tool_call_arguments_reach_the_invocation():
    request = invoke_request(
        messages=(
            pb.Message(
                role=pb.ROLE_ASSISTANT,
                tool_calls=(
                    pb.ToolCall(id="call_1", name="lookup", arguments_json=b'{"query": "x"}'),
                ),
            ),
        )
    )

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    assert invocation.messages[0].tool_calls[0].arguments == {"query": "x"}


def test_response_tool_call_arguments_cross_as_json_text():
    response = completion(
        finish_reason=FinishReason.TOOL_CALLS,
        content=(ToolCall(id="call_1", name="lookup", arguments={"query": "recursion"}),),
    )

    proto = mapping.invoke_response_to_proto(response)

    assert proto.output[0].tool_call.arguments_json == b'{"query":"recursion"}'
    assert "arguments" not in pb.ToolCall.DESCRIPTOR.fields_by_name


# --------------------------------------------------------------------------- #
# The canonical field's own name
# --------------------------------------------------------------------------- #


def test_the_system_instruction_reaches_the_invocation():
    invocation = mapping.invocation_request_from_proto(
        invoke_request(system_instruction="Answer briefly."), identity=identity()
    )

    assert invocation.system_instruction == "Answer briefly."


def test_absent_system_instruction_stays_absent():
    invocation = mapping.invocation_request_from_proto(invoke_request(), identity=identity())

    assert invocation.system_instruction is None


# --------------------------------------------------------------------------- #
# A request the canonical constructors refuse is an invalid argument
# --------------------------------------------------------------------------- #


def test_an_empty_model_is_an_invalid_argument():
    with pytest.raises(InvalidArgumentError, match="model"):
        mapping.invocation_request_from_proto(invoke_request(model=""), identity=identity())


def test_an_invocation_with_no_messages_is_an_invalid_argument():
    with pytest.raises(InvalidArgumentError, match="messages"):
        mapping.invocation_request_from_proto(invoke_request(messages=()), identity=identity())


def test_a_tool_choice_naming_an_undeclared_tool_is_an_invalid_argument():
    request = invoke_request(
        tools=(pb.ToolDefinition(name="lookup", description="Look a term up."),),
        tool_choice=pb.ToolChoice(mode=pb.TOOL_CHOICE_MODE_SPECIFIC, name="not-declared"),
    )

    with pytest.raises(InvalidArgumentError, match="tool"):
        mapping.invocation_request_from_proto(request, identity=identity())


def test_a_reserved_metadata_key_is_an_invalid_argument():
    request = invoke_request()
    request.metadata.update({"temperature": "0.2"})

    with pytest.raises(InvalidArgumentError, match="metadata"):
        mapping.invocation_request_from_proto(request, identity=identity())
