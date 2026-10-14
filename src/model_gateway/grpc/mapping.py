"""Translation between the protobuf contract and the canonical domain (PRD 5).

This module is the whole of the gRPC transport's *translation*. Every RPC handler
builds a canonical request with the request-side mappers and renders a canonical
answer with the response-side ones, so the handlers hold no mapping logic and no
policy.

Three rules shape the module:

**An unknown enum value is refused, not defaulted.** A protobuf enum carries an
explicit ``*_UNSPECIFIED`` zero value, and a caller that sets it — or sends a
number this contract does not define — is asking for something the gateway cannot
honour. Mapping it onto the gateway's own default would answer a request the
caller never made, so it raises :class:`InvalidArgumentError` naming the field
and the value that arrived.

**A count the provider never reported stays absent, never zero.** The canonical
usage record types its counts as optional for exactly that reason, and the
protobuf message carries them as ``optional`` fields, so an absent field and a
stated zero stay distinct across the boundary.

**What arrived is recorded, not repaired.** A tool call's arguments are kept as a
parsed JSON object, a raw string, or nothing — three distinct facts in the
canonical record — and the mappers preserve the distinction rather than coercing
one into another.

``google.protobuf.Struct`` numbers are doubles, so a small integer inside a
metadata object arrives as a float and one beyond 2^53 is not represented
exactly. That is a property of the well-known type rather than a choice this
module makes. Tool-call arguments do not cross through it: they travel as JSON
text, which carries every integer exactly.
"""

from collections.abc import Mapping
from enum import Enum

import orjson
from google.protobuf.struct_pb2 import Struct, Value

from model_gateway.domain import (
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    GenerationConfig,
    IdempotencyStatus,
    ImagePart,
    InvocationRequest,
    Message,
    OutputItem,
    RefusalPart,
    ResponseFormat,
    ResponseFormatMode,
    ResponseModality,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    Usage,
)
from model_gateway.domain.json_value import JsonValue, thaw_json
from model_gateway.domain.streaming import StreamEvent, StreamEventKind, ToolCallDelta
from model_gateway.errors import InvalidArgumentError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.registry import ModelInfo

__all__ = [
    "chunk_from_event",
    "embed_request_from_proto",
    "embedding_response_to_proto",
    "generation_config_from_proto",
    "image_part_from_proto",
    "invocation_request_from_proto",
    "invoke_response_to_proto",
    "list_models_response_to_proto",
    "message_from_proto",
    "model_info_to_proto",
    "tool_call_from_proto",
    "tool_definition_from_proto",
]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
#
# The mappings are written out rather than derived from the names, because the
# protobuf enum is prefixed (``ROLE_SYSTEM``) and the canonical member is not
# (``System``), and a name-mangling rule that strips one prefix is a second
# rule to keep in sync. An explicit table makes the vocabulary visible and
# makes an unlisted value impossible to map by accident.

_ROLES: dict[int, Role] = {
    pb.ROLE_SYSTEM: Role.SYSTEM,
    pb.ROLE_DEVELOPER: Role.DEVELOPER,
    pb.ROLE_USER: Role.USER,
    pb.ROLE_ASSISTANT: Role.ASSISTANT,
    pb.ROLE_TOOL: Role.TOOL,
}
_RESPONSE_FORMAT_MODES: dict[int, ResponseFormatMode] = {
    pb.RESPONSE_FORMAT_MODE_TEXT: ResponseFormatMode.TEXT,
    pb.RESPONSE_FORMAT_MODE_JSON_OBJECT: ResponseFormatMode.JSON_OBJECT,
    pb.RESPONSE_FORMAT_MODE_JSON_SCHEMA: ResponseFormatMode.JSON_SCHEMA,
}
_RESPONSE_MODALITIES: dict[int, ResponseModality] = {
    pb.RESPONSE_MODALITY_TEXT: ResponseModality.TEXT,
    pb.RESPONSE_MODALITY_IMAGE: ResponseModality.IMAGE,
    pb.RESPONSE_MODALITY_MULTIMODAL: ResponseModality.MULTIMODAL,
}
_TOOL_CHOICE_MODES: dict[int, ToolChoiceMode] = {
    pb.TOOL_CHOICE_MODE_AUTO: ToolChoiceMode.AUTO,
    pb.TOOL_CHOICE_MODE_NONE: ToolChoiceMode.NONE,
    pb.TOOL_CHOICE_MODE_REQUIRED: ToolChoiceMode.REQUIRED,
    pb.TOOL_CHOICE_MODE_SPECIFIC: ToolChoiceMode.SPECIFIC,
}

# The closed vocabulary of counts the gateway may have derived. The canonical
# record names each count directly (``input_tokens``), the protobuf enum is
# prefixed (``USAGE_FIELD_INPUT_TOKENS``), and the table between them is written
# out for the same reason the other enums are: an explicit mapping makes the
# vocabulary visible and makes an unlisted name impossible to map by accident.
_USAGE_FIELDS: dict[str, int] = {
    "input_tokens": pb.USAGE_FIELD_INPUT_TOKENS,
    "output_tokens": pb.USAGE_FIELD_OUTPUT_TOKENS,
    "total_tokens": pb.USAGE_FIELD_TOTAL_TOKENS,
    "cached_tokens": pb.USAGE_FIELD_CACHED_TOKENS,
}


def _canonical_member(table: Mapping[int, Enum], where: str, value: int) -> Enum:
    """The canonical member a protobuf enum number selects, or a refusal.

    A caller that sends the ``*_UNSPECIFIED`` zero value, or a number this
    contract does not define, is asking for something the gateway cannot honour.
    Mapping it onto the gateway's own default would answer a request the caller
    never made, so the refusal names the field and the value that arrived.
    """
    try:
        return table[value]
    except KeyError as error:
        defined = ", ".join(str(number) for number in table)
        raise InvalidArgumentError(f"{where} must be one of {defined}, got {value!r}") from error


# --------------------------------------------------------------------------- #
# Well-known types
# --------------------------------------------------------------------------- #


def _value_to_json(value: Value) -> JsonValue:
    """A ``google.protobuf.Value`` as the JSON value it carries.

    ``null_value`` is ``None``, which is the same statement the canonical record
    makes wherever a JSON null can arrive: nothing was said, rather than an
    empty object or an empty list.
    """
    kind = value.WhichOneof("kind")
    if kind is None or kind == "null_value":
        return None
    if kind == "bool_value":
        return value.bool_value
    if kind == "number_value":
        return value.number_value
    if kind == "string_value":
        return value.string_value
    if kind == "struct_value":
        return _struct_to_json(value.struct_value)
    return [_value_to_json(item) for item in value.list_value.values]


def _struct_to_json(struct: Struct) -> dict[str, JsonValue]:
    """A ``google.protobuf.Struct`` as a JSON object."""
    return {name: _value_to_json(item) for name, item in struct.fields.items()}


def _json_to_value(value: JsonValue) -> Value:
    """A JSON value as a ``google.protobuf.Value``."""
    result = Value()
    if value is None:
        result.null_value = 0
    elif isinstance(value, bool):
        result.bool_value = value
    elif isinstance(value, (int, float)):
        result.number_value = value
    elif isinstance(value, str):
        result.string_value = value
    elif isinstance(value, Mapping):
        result.struct_value.update(value)
    else:
        result.list_value.extend(value)
    return result


def _mapping_to_struct(metadata: Mapping[str, JsonValue]) -> Struct:
    """A metadata mapping as a ``google.protobuf.Struct``.

    The canonical record holds a frozen JSON object; the protobuf one holds a
    struct, and the two are the same statement.
    """
    result = Struct()
    result.update(dict(metadata))
    return result


# --------------------------------------------------------------------------- #
# Request side: protobuf -> canonical
# --------------------------------------------------------------------------- #


def image_part_from_proto(where: str, image: pb.ImagePart) -> ImagePart:
    """A canonical image part, carrying exactly one of data and url.

    A part with neither, or with both, is not a representation the gateway can
    send: those are different requests and the gateway cannot guess which was
    meant.
    """
    kind = image.WhichOneof("source")
    if kind == "data":
        return ImagePart(data=image.data or None, media_type=image.media_type or None)
    if kind == "url":
        return ImagePart(url=image.url or None, media_type=image.media_type or None)
    raise InvalidArgumentError(
        f"{where} must carry exactly one of 'data' (a base64 payload) or 'url' (an "
        f"authorised remote URL), got {kind or 'none'!r}"
    )


def content_part_from_proto(where: str, part: pb.ContentPart):
    """One canonical content part, dispatched on the protobuf oneof."""
    kind = part.WhichOneof("part")
    if kind == "text":
        return TextPart(text=part.text.text)
    if kind == "refusal":
        return RefusalPart(text=part.refusal.text)
    if kind == "image":
        return image_part_from_proto(where, part.image)
    raise InvalidArgumentError(
        f"{where} must carry exactly one content kind, got {kind or 'none'!r}"
    )


def tool_call_from_proto(where: str, call: pb.ToolCall) -> ToolCall:
    """A canonical tool call, recording what arrived rather than repairing it.

    ``arguments_json`` is the canonical JSON text, and :func:`orjson.loads`
    reproduces the canonical rule exactly: a JSON string parses to a Python
    ``str`` and stays one, an object becomes a mapping, and ``null`` parses to
    ``None``. An absent field and a ``null`` payload are the same statement —
    the caller said nothing about arguments — so both become ``None``. Any other
    JSON value — a bare number, a boolean, an array — is not a state the
    canonical record defines and is refused with :class:`InvalidArgumentError`
    naming the field, rather than escaping as a ``ValueError`` the handler would
    report as an internal error.
    """
    if not call.arguments_json:
        return ToolCall(id=call.id, name=call.name, arguments=None)
    try:
        return ToolCall(id=call.id, name=call.name, arguments=orjson.loads(call.arguments_json))
    except ValueError as error:
        raise InvalidArgumentError(f"{where} 'arguments_json' is not valid: {error}") from error


def tool_definition_from_proto(where: str, definition: pb.ToolDefinition) -> ToolDefinition:
    """A canonical tool declaration.

    ``parameters`` is stored as the caller supplied it and never rewritten: a
    schema the gateway filled defaults into would describe a different tool.
    """
    parameters: Mapping[str, JsonValue] = {}
    if definition.HasField("parameters"):
        parameters = _struct_to_json(definition.parameters)
    return ToolDefinition(
        name=definition.name,
        description=definition.description,
        parameters=parameters,
    )


def _schema_from_value(where: str, schema: Value) -> Mapping[str, JsonValue] | bool:
    """A JSON Schema carried by a ``google.protobuf.Value``, or a refusal.

    A JSON Schema is an object or one of the two booleans the specification
    permits. ``struct_value`` and ``bool_value`` are those two kinds; a string,
    a number, an array or null is none of them, and is refused rather than
    coerced, exactly as the canonical type refuses it.
    """
    kind = schema.WhichOneof("kind")
    if kind == "struct_value":
        return _struct_to_json(schema.struct_value)
    if kind == "bool_value":
        return schema.bool_value
    raise InvalidArgumentError(
        f"{where} must be a JSON object or a boolean, got {kind or 'null_value'!r}"
    )


def response_format_from_proto(where: str, response_format: pb.ResponseFormat) -> ResponseFormat:
    """A canonical structured-output requirement.

    ``schema`` is read through :func:`_schema_from_value`: a JSON Schema may be
    an object or a boolean, and ``google.protobuf.Value`` is the one well-known
    type that carries both, so a boolean schema crosses the boundary as itself
    instead of being coerced into an object.
    """
    mode = _canonical_member(_RESPONSE_FORMAT_MODES, f"{where} 'mode'", response_format.mode)
    schema: Mapping[str, JsonValue] | bool | None = None
    if response_format.HasField("schema"):
        schema = _schema_from_value(f"{where} 'schema'", response_format.schema)
    return ResponseFormat(
        mode=mode,
        schema=schema,
        name=response_format.name or None,
        strict=response_format.strict,
    )


def generation_config_from_proto(where: str, config: pb.GenerationConfig) -> GenerationConfig:
    """A canonical generation config.

    An unset field means "not specified", which the provider's own default
    answers; a stated zero is a different request from an unstated one, so the
    optional fields are read through ``HasField`` rather than defaulted.
    """
    return GenerationConfig(
        temperature=config.temperature if config.HasField("temperature") else None,
        top_p=config.top_p if config.HasField("top_p") else None,
        max_output_tokens=config.max_output_tokens
        if config.HasField("max_output_tokens")
        else None,
        stop_sequences=tuple(config.stop_sequences),
        seed=config.seed if config.HasField("seed") else None,
        presence_penalty=config.presence_penalty if config.HasField("presence_penalty") else None,
        frequency_penalty=config.frequency_penalty
        if config.HasField("frequency_penalty")
        else None,
        response_format=(
            response_format_from_proto(f"{where} 'response_format'", config.response_format)
            if config.HasField("response_format")
            else None
        ),
    )


def message_from_proto(where: str, message: pb.Message) -> Message:
    """One canonical turn.

    The role's own rules — only an assistant proposes a call, only a tool message
    answers one — are enforced by :class:`Message` at construction, so an
    inconsistent conversation fails here rather than upstream, where the
    provider's complaint would be less specific.
    """
    role = _canonical_member(_ROLES, f"{where} 'role'", message.role)
    return Message(
        role=role,
        content=tuple(
            content_part_from_proto(f"{where} content[{index}]", part)
            for index, part in enumerate(message.content)
        ),
        name=message.name or None,
        tool_calls=tuple(
            tool_call_from_proto(f"{where} 'tool_calls'", call) for call in message.tool_calls
        ),
        tool_call_id=message.tool_call_id or None,
    )


def invocation_request_from_proto(request: pb.InvokeRequest, *, identity) -> InvocationRequest:
    """The canonical invocation a gRPC request asks for.

    ``identity`` is the trusted identity the transport authenticated, not a value
    read from the request: the tenant the invocation acts under is the
    credential's, and a claimed ``tenant_id`` has already been reconciled against
    it by the caller (PRD section 5.1).

    Every canonical constructor this mapping reaches — each message, the tool
    declarations, the generation config and its response format, the tool choice
    and the invocation itself — validates what it is given, and a failure there
    is a request the gateway cannot honour rather than a gateway bug, so a
    ``ValueError`` from one becomes :class:`InvalidArgumentError` naming the
    rule. The HTTP request DTOs apply the same rule the same way.
    """
    try:
        return InvocationRequest(
            model=request.model,
            messages=tuple(
                message_from_proto(f"messages[{index}]", message)
                for index, message in enumerate(request.messages)
            ),
            invocation_id=request.invocation_id or None,
            tenant_id=identity.tenant,
            actor_id=request.actor_id or None,
            agent_id=request.agent_id or None,
            system_instruction=request.system_instruction or None,
            generation_config=generation_config_from_proto(
                "generation_config", request.generation_config
            ),
            tools=tuple(
                tool_definition_from_proto(f"tools[{index}]", definition)
                for index, definition in enumerate(request.tools)
            ),
            tool_choice=(
                ToolChoice(
                    mode=_canonical_member(
                        _TOOL_CHOICE_MODES, "tool_choice 'mode'", request.tool_choice.mode
                    ),
                    name=request.tool_choice.name or None,
                )
                if request.HasField("tool_choice")
                else None
            ),
            response_modality=_canonical_member(
                _RESPONSE_MODALITIES, "response_modality", request.response_modality
            ),
            fallback_models=tuple(request.fallback_models),
            idempotency_key=request.idempotency_key or None,
            metadata=_struct_to_json(request.metadata),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"the invocation request is not valid: {error}") from error


def embed_request_from_proto(request: pb.EmbedRequest) -> EmbeddingRequest:
    """The canonical embeddings request a gRPC request asks for.

    The canonical type's own validation — at least one input, no blank entry, a
    positive ``dimensions``, an encoding the gateway answers — is a gateway rule
    and not a protobuf one, so a failure is an invalid argument naming the rule
    rather than an internal error. The HTTP embeddings endpoint applies the same
    rule the same way.
    """
    try:
        return EmbeddingRequest(
            model=request.model,
            input=tuple(request.input),
            dimensions=request.dimensions if request.HasField("dimensions") else None,
            encoding_format=request.encoding_format or None,
            user=request.user or None,
            metadata=(
                _struct_to_json(request.metadata) if request.HasField("metadata") else None
            ),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"the embeddings request is not valid: {error}") from error


# --------------------------------------------------------------------------- #
# Response side: canonical -> protobuf
# --------------------------------------------------------------------------- #


def _usage_to_proto(usage: Usage) -> pb.Usage:
    """The protobuf usage object, carrying only the counts that are known.

    A count the gateway derived is named in ``estimated_fields`` through the
    closed vocabulary of :data:`_USAGE_FIELDS`, so a consumer can tell a reported
    fact from the gateway's own arithmetic; the names are written in sorted
    order so the same record always serialises the same way.
    """
    result = pb.Usage()
    if usage.input_tokens is not None:
        result.input_tokens = usage.input_tokens
    if usage.output_tokens is not None:
        result.output_tokens = usage.output_tokens
    if usage.total_tokens is not None:
        result.total_tokens = usage.total_tokens
    if usage.cached_tokens is not None:
        result.cached_tokens = usage.cached_tokens
    result.estimated_fields.extend(_USAGE_FIELDS[name] for name in sorted(usage.estimated_fields))
    return result


def _finish_reason_to_proto(reason: FinishReason) -> int:
    return getattr(pb, f"FINISH_REASON_{reason.name}")


def _idempotency_status_to_proto(status: IdempotencyStatus) -> int:
    return getattr(pb, f"IDEMPOTENCY_STATUS_{status.name}")


def _content_part_to_proto(part) -> pb.ContentPart:
    """One protobuf content part."""
    result = pb.ContentPart()
    if isinstance(part, TextPart):
        result.text.text = part.text
    elif isinstance(part, RefusalPart):
        result.refusal.text = part.text
    else:
        image = result.image
        if part.data is not None:
            image.data = part.data
        else:
            image.url = part.url or ""
        if part.media_type is not None:
            image.media_type = part.media_type
    return result


def _response_format_to_proto(response_format) -> pb.ResponseFormat:
    """The protobuf structured-output requirement.

    A schema that is an object or a boolean is written as the
    ``google.protobuf.Value`` of the same kind, so a boolean JSON Schema crosses
    the boundary as itself; no schema leaves the field unset, which the canonical
    record reads as the caller requiring no particular shape.
    """
    result = pb.ResponseFormat(
        mode=getattr(pb, f"RESPONSE_FORMAT_MODE_{response_format.mode.name}"),
        name=response_format.name or "",
        strict=response_format.strict,
    )
    if response_format.schema is not None:
        result.schema.CopyFrom(_json_to_value(response_format.schema))
    return result


def _tool_call_to_proto(call: ToolCall) -> pb.ToolCall:
    """One protobuf tool call.

    A parsed JSON object, a raw string, and nothing are three distinct facts.
    :func:`orjson.dumps` serialises an object or a string as the JSON text of
    the same kind — a JSON string is quoted, so it decodes back to a string —
    and nothing leaves ``arguments_json`` empty, which the canonical record reads
    as the provider having reported no arguments at all rather than an empty
    object. ``bytes`` rather than the well-known value is what keeps a JSON
    integer exact: ``google.protobuf.Value`` types every number as a double.
    """
    result = pb.ToolCall(id=call.id, name=call.name)
    if call.arguments is not None:
        result.arguments_json = orjson.dumps(thaw_json(call.arguments))
    return result


def _output_item_to_proto(item: OutputItem) -> pb.OutputItem:
    """One protobuf output item: a content part or a tool call."""
    result = pb.OutputItem()
    if isinstance(item, ToolCall):
        result.tool_call.CopyFrom(_tool_call_to_proto(item))
    else:
        result.content.CopyFrom(_content_part_to_proto(item))
    return result


def invoke_response_to_proto(response) -> pb.InvokeResponse:
    """The protobuf answer to one Invoke.

    ``raw_finish_reason`` is written only when the canonical record carries one:
    the field is optional, and the canonical record distinguishes "the provider
    reported no reason" (``None``) from "the provider supplied an empty one"
    (``""``), so writing ``""`` unconditionally would collapse the two.
    """
    result = pb.InvokeResponse(
        invocation_id=response.invocation_id or "",
        model=response.model,
        upstream_model=response.upstream_model,
        finish_reason=_finish_reason_to_proto(response.finish_reason),
        attempt_count=response.attempt_count,
        fallback_used=response.fallback_used,
        idempotency_status=_idempotency_status_to_proto(response.idempotency_status),
    )
    if response.raw_finish_reason is not None:
        result.raw_finish_reason = response.raw_finish_reason
    result.output.extend(_output_item_to_proto(item) for item in response.content)
    result.usage.CopyFrom(_usage_to_proto(response.usage))
    if response.estimated_cost is not None:
        result.estimated_cost = response.estimated_cost
        result.cost_currency = response.cost_currency or ""
    return result


def _tool_call_delta_to_proto(delta: ToolCallDelta) -> pb.ToolCallDelta:
    """One protobuf tool-call fragment.

    Each of id, name and arguments_delta is set only when this fragment reported
    it: a consumer accumulates fragments by ``index``, and a field the fragment
    said nothing about must arrive unset rather than empty.
    """
    result = pb.ToolCallDelta(index=delta.index)
    if delta.id is not None:
        result.id = delta.id
    if delta.name is not None:
        result.name = delta.name
    if delta.arguments_delta is not None:
        result.arguments_delta = delta.arguments_delta
    return result


def chunk_from_event(event: StreamEvent) -> pb.InvokeChunk:
    """One protobuf chunk carrying one canonical stream event.

    The terminal event is exactly one of finish and error, and the stream
    contract of PRD section 16 carries over unchanged: a chunk carrying a finish
    payload ends a completed stream, one carrying an error payload ends a stream
    that failed after output began, and a stream that stopped without reporting a
    finish reason produces neither.
    """
    result = pb.InvokeChunk(invocation_id=event.invocation_id, sequence=event.sequence)
    if event.kind == StreamEventKind.CONTENT:
        result.content_delta = event.text or ""
    elif event.kind == StreamEventKind.TOOL_CALL:
        result.tool_call_delta.CopyFrom(_tool_call_delta_to_proto(event.tool_call))
    elif event.kind == StreamEventKind.USAGE:
        result.usage.CopyFrom(_usage_to_proto(event.usage))
    elif event.kind == StreamEventKind.FINISH:
        result.finish.finish_reason = _finish_reason_to_proto(event.finish_reason)
        # Optional, so an absent reason stays absent: the canonical record
        # distinguishes "the provider reported no reason" from "the provider
        # supplied an empty one", and the two must not collapse on the wire.
        if event.raw_finish_reason is not None:
            result.finish.raw_finish_reason = event.raw_finish_reason
    else:
        result.error.reason = event.error_reason or "unknown"
        result.error.message = event.error_message or ""
    return result


def embedding_response_to_proto(response: EmbeddingResponse) -> pb.EmbedResponse:
    """The protobuf answer to one Embed.

    ``object`` is ``list``, as the HTTP embeddings shape states it, and each
    vector is placed at the index the upstream reported, so the answer's data is
    in the caller's input order whatever order the upstream used.
    """
    result = pb.EmbedResponse(
        object="list",
        model=response.model,
        upstream_model=response.upstream_model,
        invocation_id=response.invocation_id or "",
    )
    result.usage.CopyFrom(_usage_to_proto(response.usage))
    result.data.extend(
        pb.Embedding(index=embedding.index, embedding=embedding.values)
        for embedding in response.data
    )
    return result


def model_info_to_proto(info: ModelInfo) -> pb.ModelInfo:
    """One protobuf model, with the capabilities its primary route declares."""
    result = pb.ModelInfo(
        id=info.id,
        operation=getattr(pb, f"OPERATION_{info.operation.name}"),
    )
    result.capabilities.extend(sorted(capability.value for capability in info.capabilities))
    if info.context_window is not None:
        result.context_window = info.context_window
    if info.max_output_tokens is not None:
        result.max_output_tokens = info.max_output_tokens
    if info.dimensions is not None:
        result.dimensions = info.dimensions
    return result


def list_models_response_to_proto(infos: tuple[ModelInfo, ...]) -> pb.ListModelsResponse:
    """The protobuf model listing."""
    result = pb.ListModelsResponse()
    result.models.extend(model_info_to_proto(info) for info in infos)
    return result
