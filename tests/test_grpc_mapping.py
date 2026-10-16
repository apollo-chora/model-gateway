"""The protobuf↔canonical translation, asserted field by field.

Every test here drives the mappers in :mod:`model_gateway.grpc.mapping` directly
with generated protobuf messages, so what is asserted is what crosses the wire:
the canonical record a protobuf request builds, and the protobuf message a
canonical answer renders to. No server is involved — the transport's handlers are
covered by the sibling RPC test files, and this file owns the translation alone.

Four rules shape the file.

**Arbitrary JSON crosses as its canonical JSON text.** Metadata, a tool's
parameters and a JSON Schema travel as ``bytes`` carrying JSON text, so a nested
object, a boolean and a list all reach the canonical record with their types and
values intact. The well-known protobuf types were tried for these fields and
rejected: ``Struct`` and ``Value`` represent every number as a double, which
changes both the type and the value of a JSON integer.

**A JSON Schema may be a boolean.** ``ResponseFormat.schema_json`` carries the
schema's JSON text, so ``true`` and ``false`` — both valid schema documents —
cross as themselves, and a schema of any other kind is refused by the mapper
rather than coerced.

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

from model_gateway.domain import (
    FinishReason,
    GroundedCitation,
    GroundedFinishReason,
    GroundedSearchResult,
    GroundedUsage,
    InvocationResponse,
    ResponseFormat,
    ResponseFormatMode,
    TextPart,
    ToolCall,
    Usage,
    thaw_json,
)
from model_gateway.domain.json_value import FrozenJsonObject
from model_gateway.domain.streaming import StreamEvent, StreamEventKind
from model_gateway.errors import InvalidArgumentError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import mapping
from model_gateway.security.identity import Identity

PUBLIC_MODEL = "text-default"
TENANT = "tenant-a"


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
# Metadata: JSON text both ways
# --------------------------------------------------------------------------- #


def test_canonical_json_text_and_a_json_object_convert_both_ways():
    # The boundary carries JSON text, not a well-known type: the canonical
    # mapping goes out as its text and comes back as the same mapping, with an
    # integer an integer and a boolean a boolean — and a JSON array as the tuple
    # the canonical record freezes it into.
    metadata = FrozenJsonObject({"trace": {"sampled": True}, "ids": [1, 2], "flag": True})
    request = invoke_request()
    request.metadata_json = orjson.dumps(thaw_json(metadata))

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    assert invocation.metadata == {"trace": {"sampled": True}, "ids": (1, 2), "flag": True}
    assert type(invocation.metadata["ids"][0]) is int
    assert type(invocation.metadata["flag"]) is bool


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"trace": {"sampled": True}}, id="nested object"),
        pytest.param({"flag": True}, id="boolean"),
        pytest.param({"ids": [1, 2, 3]}, id="list"),
    ],
)
def test_request_metadata_of_every_json_kind_reaches_the_invocation(metadata):
    # The bytes carry the canonical text, so every JSON kind crosses as itself:
    # the list comes back as a tuple of integers, where the struct the contract
    # used to carry typed every number as a double and returned (1.0, 2.0, 3.0).
    request = invoke_request()
    request.metadata_json = orjson.dumps(metadata)

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    expected = {"ids": (1, 2, 3)} if "ids" in metadata else metadata
    assert invocation.metadata == expected


def test_absent_request_metadata_becomes_an_empty_object():
    invocation = mapping.invocation_request_from_proto(invoke_request(), identity=identity())

    assert invocation.metadata == {}


def test_absent_embed_metadata_stays_absent():
    # The canonical embeddings request distinguishes no metadata from empty
    # metadata, and the mapper reads the field's absence — the empty bytes an
    # unset field carries — as the first of the two.
    request = pb.EmbedRequest(model=PUBLIC_MODEL, input=("first",))

    embedding_request = mapping.embed_request_from_proto(request)

    assert embedding_request.metadata is None


def test_an_empty_embed_model_is_mapped_to_none():
    request = pb.EmbedRequest(model="", input=("first",))

    embedding_request = mapping.embed_request_from_proto(request)

    assert embedding_request.model is None


def test_present_embed_metadata_reaches_the_canonical_request():
    request = pb.EmbedRequest(model=PUBLIC_MODEL, input=("first",))
    request.metadata_json = b'{"trace": "abc"}'

    embedding_request = mapping.embed_request_from_proto(request)

    assert embedding_request.metadata == {"trace": "abc"}


# --------------------------------------------------------------------------- #
# A JSON Schema may be an object or a boolean
# --------------------------------------------------------------------------- #


def test_a_boolean_json_schema_round_trips():
    canonical = ResponseFormat(mode=ResponseFormatMode.JSON_SCHEMA, schema=True)

    proto = mapping._response_format_to_proto(canonical)

    assert proto.schema_json == b"true"
    assert mapping.response_format_from_proto("response_format", proto) == canonical


def test_an_object_json_schema_round_trips():
    schema = FrozenJsonObject({"type": "object", "properties": {"city": {"type": "string"}}})
    canonical = ResponseFormat(mode=ResponseFormatMode.JSON_SCHEMA, schema=schema)

    proto = mapping._response_format_to_proto(canonical)

    assert orjson.loads(proto.schema_json) == dict(schema)
    assert mapping.response_format_from_proto("response_format", proto) == canonical


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(b"42", id="number"),
        pytest.param(b'"str"', id="string"),
        pytest.param(b"null", id="null"),
        pytest.param(b"[1,2]", id="array"),
    ],
)
def test_a_schema_of_any_other_kind_is_refused(schema):
    proto = pb.ResponseFormat(mode=pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA)
    proto.schema_json = schema

    with pytest.raises(InvalidArgumentError, match="'schema_json'"):
        mapping.response_format_from_proto("response_format 'response_format'", proto)


def test_a_boolean_schema_in_a_generation_config_reaches_the_canonical_request():
    config = pb.GenerationConfig()
    config.response_format.mode = pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA
    config.response_format.schema_json = b"true"

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
# A tool's parameters: JSON text, with a schema's integers intact
# --------------------------------------------------------------------------- #


def test_a_tool_definitions_parameters_reach_the_invocation_as_they_were_given():
    request = invoke_request(
        tools=(
            pb.ToolDefinition(
                name="lookup",
                description="Look a term up.",
                parameters_json=b'{"type": "object", "properties": {"n": {"type": "integer"}}}',
            ),
        )
    )

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    assert invocation.tools[0].parameters == {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }


def test_a_tool_definitions_parameters_carry_a_large_integer_constraint_exactly():
    # The reason parameters travel as bytes: the struct they replaced typed
    # every number as a double, and 2^53 + 1 is not a double at all, so the
    # constraint would cross changed.
    request = invoke_request(
        tools=(
            pb.ToolDefinition(
                name="lookup",
                description="Look a term up.",
                parameters_json=orjson.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer", "maximum": 9007199254740993}
                        },
                    }
                ),
            ),
        )
    )

    invocation = mapping.invocation_request_from_proto(request, identity=identity())

    maximum = invocation.tools[0].parameters["properties"]["n"]["maximum"]
    assert maximum == 9007199254740993
    assert type(maximum) is int


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


def test_an_empty_model_is_mapped_to_none():
    invocation = mapping.invocation_request_from_proto(
        invoke_request(model=""), identity=identity()
    )

    assert invocation.model is None


def test_an_unset_model_is_mapped_to_none():
    invocation = mapping.invocation_request_from_proto(
        pb.InvokeRequest(
            messages=(user_message(),), response_modality=pb.RESPONSE_MODALITY_TEXT
        ),
        identity=identity(),
    )

    assert invocation.model is None


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
    request.metadata_json = b'{"temperature": "0.2"}'

    with pytest.raises(InvalidArgumentError, match="metadata"):
        mapping.invocation_request_from_proto(request, identity=identity())


# --------------------------------------------------------------------------- #
# Grounded search: the request and response cross both ways
# --------------------------------------------------------------------------- #


def grounded_request(**overrides) -> pb.GroundedSearchRequest:
    """A ``GroundedSearchRequest`` a caller could send."""
    fields = {
        "invocation_id": "inv-grounded-1",
        "tenant_id": TENANT,
        "gcid": "gcid-1",
        "agent_id": "familiar_seeker",
        "crew_kind": "familiar",
        "directive": "What is the capital of France?",
        "max_results": 5,
        "action_code": "external_egress.search",
        "traceparent": "00-abc-def-01",
        "tracestate": "congo=t61rcWkgMzE",
    }
    fields.update(overrides)
    return pb.GroundedSearchRequest(**fields)


def grounded_result(**overrides) -> GroundedSearchResult:
    """A ``GroundedSearchResult`` the adapter could have produced."""
    fields = {
        "invocation_id": "inv-grounded-1",
        "citations": (
            GroundedCitation(
                uri="https://a.example/one",
                title="Source One",
                domain="a.example",
                snippet="text from one",
                confidence=0.9,
            ),
        ),
        "grounded_answer": "Paris is the capital.",
        "search_entry_point_html": (
            '<ul><li><a href="https://a.example/one">Source One</a></li></ul>'
        ),
        "web_search_queries": ("What is the capital of France?",),
        "usage": GroundedUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            grounding_units=1,
            cost_micros=10000,
        ),
        "vendor": "exa",
        "model_version": "synthesis-model",
        "finish_reason": GroundedFinishReason.COMPLETE,
        "completed_at": "2026-10-16T00:00:00+00:00",
    }
    fields.update(overrides)
    return GroundedSearchResult(**fields)


def test_a_grounded_search_request_crosses_both_ways():
    canonical = mapping.grounded_search_request_from_proto(grounded_request())

    assert canonical.invocation_id == "inv-grounded-1"
    assert canonical.tenant_id == TENANT
    assert canonical.gcid == "gcid-1"
    assert canonical.agent_id == "familiar_seeker"
    assert canonical.crew_kind == "familiar"
    assert canonical.directive == "What is the capital of France?"
    assert canonical.max_results == 5
    assert canonical.action_code == "external_egress.search"
    assert canonical.traceparent == "00-abc-def-01"
    assert canonical.tracestate == "congo=t61rcWkgMzE"


def test_a_grounded_search_response_crosses_both_ways():
    proto = mapping.grounded_search_response_to_proto(grounded_result())

    assert proto.invocation_id == "inv-grounded-1"
    assert [citation.uri for citation in proto.citations] == ["https://a.example/one"]
    assert proto.citations[0].title == "Source One"
    assert proto.citations[0].domain == "a.example"
    assert proto.citations[0].snippet == "text from one"
    assert proto.citations[0].confidence == pytest.approx(0.9)
    assert proto.grounded_answer == "Paris is the capital."
    assert proto.search_entry_point_html == (
        '<ul><li><a href="https://a.example/one">Source One</a></li></ul>'
    )
    assert list(proto.web_search_queries) == ["What is the capital of France?"]
    assert proto.usage.input_tokens == 10
    assert proto.usage.output_tokens == 20
    assert proto.usage.total_tokens == 30
    assert proto.usage.grounding_units == 1
    assert proto.usage.cost_micros == 10000
    assert proto.vendor == "exa"
    assert proto.model_version == "synthesis-model"
    assert proto.finish_reason == pb.FINISH_REASON_STOP
    assert proto.finish_detail == ""
    assert proto.completed_at == "2026-10-16T00:00:00+00:00"


def test_an_empty_citation_list_crosses_as_empty():
    # The in-band anomaly: the search found nothing renderable, so the
    # response carries no citations and the caller hedges. The mapping writes
    # the empty list rather than dropping it or refusing it.
    proto = mapping.grounded_search_response_to_proto(
        grounded_result(
            citations=(), grounded_answer="", finish_reason=GroundedFinishReason.UNSPECIFIED
        )
    )

    assert len(proto.citations) == 0
    assert proto.grounded_answer == ""
    assert proto.finish_reason == pb.FINISH_REASON_UNSPECIFIED


@pytest.mark.parametrize(
    "reason, expected",
    [
        pytest.param(
            GroundedFinishReason.UNSPECIFIED, pb.FINISH_REASON_UNSPECIFIED, id="unspecified"
        ),
        pytest.param(GroundedFinishReason.COMPLETE, pb.FINISH_REASON_STOP, id="complete"),
    ],
)
def test_a_grounded_finish_reason_maps_onto_the_proto_enum(reason, expected):
    proto = mapping.grounded_search_response_to_proto(grounded_result(finish_reason=reason))

    assert proto.finish_reason == expected


def test_an_unknown_grounded_finish_reason_is_refused():
    result = grounded_result()
    # A finish reason outside the canonical vocabulary is not a value the
    # gateway could publish, so the mapping refuses it rather than guessing.
    object.__setattr__(result, "finish_reason", "not-a-reason")

    with pytest.raises(InvalidArgumentError, match="finish_reason"):
        mapping.grounded_search_response_to_proto(result)
