"""Transport-agnostic translation between the canonical domain and LiteLLM (PRD section 9).

This module is the half of the LiteLLM binding that knows nothing about how a call
is performed. It translates a canonical :class:`~model_gateway.domain.InvocationRequest`
into the keyword arguments of ``litellm.acompletion`` and a LiteLLM
:class:`~litellm.types.utils.ModelResponse` back into a canonical
:class:`~model_gateway.domain.InvocationResponse`, and maps LiteLLM's exception
hierarchy onto the gateway error taxonomy. The outbound transport — the connection,
the address pinning, the redirect policy — is a later slice's; this module only
shapes data.

Three rules govern every function here. They are PRD section 9.1 applied to this
boundary, and each is enforced rather than assumed:

* **Request side.** Every semantically relevant element of the canonical request
  is represented in the upstream call, or the invocation is rejected as
  unsupported. An element with no LiteLLM equivalent raises a
  :class:`~model_gateway.errors.CapabilityError` naming it; a parameter is never
  dropped because no upstream target exists for it. Routing, retry, accounting,
  identity and idempotency fields stay out of the upstream call: the gateway
  chose the route, owns the retry loop and mints the invocation identity, so
  none of ``model`` (the public ID), ``invocation_id``, ``fallback_models``,
  ``idempotency_key`` or ``metadata`` is forwarded (PRD sections 5.1, 9.2 and
  12).
* **Response side.** Every field of the upstream response is classified
  before it is acted on, by four rules (PRD section 9.1):

  - a canonical semantic is mapped into the canonical model;
  - a known semantic the canonical model cannot represent — log probabilities,
    reasoning content, audio, a legacy function call, multiple choices, an
    unrepresentable usage detail — is refused with an
    :class:`~model_gateway.errors.InternalError` naming it, because the caller
    must not be told the answer was honoured when part of it was dropped;
  - a known non-semantic operational metadata field is deliberately ignored,
    from the explicit set :data:`_NON_SEMANTIC_RESPONSE_FIELDS` — a provider
    diagnostic that does not change the answer must not fail an otherwise
    valid response once it is understood;
  - an unknown provider-specific field is still refused until it is
    classified, so a new provider annotation cannot pass unexamined.

  The gateway's own identity is stamped on the response — ``model`` is the
  public ID the caller asked for, ``upstream_model`` is the route's model ID
  and ``invocation_id`` is the gateway's own — never a value the upstream
  supplied (PRD sections 5.2 and 9).
* **Errors.** LiteLLM's exception hierarchy is mapped onto the gateway taxonomy
  by class, never by parsing an upstream message. Every message the taxonomy
  publishes is gateway-authored and fixed, so an upstream message, URL or
  credential cannot reach a caller through this module; the original exception
  is attached as ``__cause__`` for the logging layer, which must not format
  exception chains by default (PRD sections 5.4 and 17).

The LiteLLM shapes this module reads and writes are the installed library's own:
``litellm.types.utils.ModelResponse`` with its ``Choices``, ``Message``,
``ChatCompletionMessageToolCall`` and ``Usage`` records, and the
``litellm.exceptions`` hierarchy, which extends the OpenAI exception classes.
Tests build fixtures from those real types — by constructing them, or by parsing
a realistic upstream payload through LiteLLM's own
``convert_to_model_response_object`` — so a fixture carrying an extra semantic
field fails a translator that would drop it.
"""

import json
from collections.abc import Mapping
from typing import Any

from model_gateway.config.routes import Route
from model_gateway.domain import (
    ContentPart,
    FinishReason,
    GenerationConfig,
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    Message,
    OutputItem,
    ResponseFormat,
    ResponseFormatMode,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    freeze_json,
    thaw_json,
)
from model_gateway.domain import (
    Usage as TokenUsage,
)
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    GatewayError,
    InternalError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
    normalise_exception,
)
from model_gateway.providers._litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    LiteLLMMessage,
    LiteLLMUsage,
    ModelResponse,
    litellm,
    openai,
)

__all__ = ["from_litellm_response", "map_litellm_error", "to_litellm_params"]


def to_litellm_params(route: Route, request: InvocationRequest) -> dict[str, object]:
    """Translate a canonical invocation into ``litellm.acompletion`` keyword arguments.

    The route supplies everything the upstream call needs to be addressed: the
    provider-prefixed model string ``<format>/<model_id>`` (PRD section 9.6),
    the resolved base URL, the route's credential and its non-authentication
    extra headers. Retries are disabled with ``num_retries=0`` because the
    gateway core owns the retry loop and the invocation deadline, so there is
    exactly one retry layer and one deadline (PRD section 11.3).

    The request supplies the conversation, the tools and the generation
    parameters. An unset generation parameter is omitted rather than sent as a
    default, so the provider's own default applies; the gateway deliberately
    substitutes no vendor default (PRD section 5.1). A ``text`` response format
    is likewise omitted, because it is the provider's own default and asks for
    nothing beyond ``chat`` (PRD section 6.1); a non-text response format is the
    structured-output requirement of PRD section 9.5 and is sent in full.

    The invocation-level ``system_instruction`` is forwarded as a ``system``
    message of its own, ahead of the conversation, and is never merged into a
    ``system`` message inside ``messages``: the two are separate elements and
    folding one into the other would change what the provider sees (PRD
    sections 5.1 and 9.5).

    Raises :class:`~model_gateway.errors.CapabilityError` naming the element
    when a semantically relevant part of the request has no LiteLLM upstream
    representation — today that is a refusal part in a caller-authored message,
    which a chat completion request cannot carry. It is never mapped to
    ordinary text, because a refusal and text saying much the same thing are
    different events (PRD section 5.2).
    """
    params: dict[str, object] = {
        "model": route.litellm_model,
        "api_base": route.base_url,
        "api_key": route.api_key,
        "extra_headers": dict(route.extra_headers),
        "num_retries": 0,
        "messages": _messages(request),
    }
    _add_generation_params(params, request.generation_config)
    if request.tools:
        params["tools"] = [_tool(definition) for definition in request.tools]
    if request.tool_choice is not None:
        params["tool_choice"] = _tool_choice(request.tool_choice)
    return params


def from_litellm_response(
    route: Route,
    request: InvocationRequest,
    response: ModelResponse,
) -> InvocationResponse:
    """Normalise a LiteLLM completion into the canonical response.

    ``response`` is upstream data, never forwarded unchanged (PRD section 9).
    Exactly one choice is accepted: the canonical response represents a single
    answer, so a response carrying several choices is rejected rather than
    silently reduced to its first. The answer is read in the order the upstream
    shape preserves — the message text, then any images, then its tool calls.
    LiteLLM's ``Message`` keeps content and tool calls in separate fields and so
    does not record where a tool call fell within the text; the binding therefore
    uses that fixed order and loses no ordering the upstream kept.

    The finish reason is mapped through the gateway's own vocabulary with the
    provider's raw reason preserved verbatim beside it. LiteLLM maps recognised
    reasons onto its OpenAI-shaped vocabulary and keeps the original in
    ``provider_specific_fields`` whenever the two differ, so the native reason is
    read from there when it is present and from the mapped reason otherwise. A
    reason the gateway has not been taught becomes ``FinishReason.UNKNOWN``
    with the raw value still recorded, so ``UNKNOWN`` never means the gateway
    deleted what upstream said (PRD section 5.2).

    Usage is mapped with per-count provenance (PRD section 13): provider-reported
    counts are recorded as reported, and a total LiteLLM left at zero beside
    non-zero parts — its placeholder for an unreported total — is derived from
    the parts and named in ``estimated_fields``. Cached tokens come from the
    provider's own prompt-token details.

    The gateway's own identity is stamped on the result: ``model`` is the public
    ID the caller asked for, ``upstream_model`` is the route's configured model
    ID (not the provider-prefixed string the binding built) and
    ``invocation_id`` is the gateway's own. No upstream value is trusted for any
    of them (PRD sections 5.2 and 9.6).

    Raises :class:`~model_gateway.errors.InternalError` naming the field when
    the response carries a semantic the canonical model cannot represent — a
    log-probability block, a reasoning trace, a provider-specific annotation,
    a server-tool-use record or an extra usage-detail count (PRD section 9.1).
    A backend fingerprint is no longer one of them: it is classified
    non-semantic operational metadata and deliberately ignored.
    """
    choices = response.choices
    if len(choices) != 1:
        raise InternalError(
            f"the upstream response carries {len(choices)} choices; the canonical response "
            f"represents exactly one answer, so the remaining choices cannot be dropped"
        )
    choice = choices[0]
    message = choice.message
    _reject_unmappable(response, choice, message)
    raw_finish_reason = _raw_finish_reason(choice)
    return InvocationResponse(
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=request.invocation_id,
        content=_output(message),
        finish_reason=FinishReason.coerce(raw_finish_reason),
        raw_finish_reason=raw_finish_reason,
        usage=_usage(getattr(response, "usage", None)),
    )


def map_litellm_error(exc: BaseException) -> GatewayError:
    """Map a LiteLLM failure onto the gateway error taxonomy (PRD sections 5.4 and 9).

    The mapping is by exception class, never by parsing an upstream message, and
    every message the returned error carries is gateway-authored and fixed. An
    upstream message, URL or credential therefore cannot reach a caller through
    this function; the original exception is attached as ``__cause__`` so the
    logging layer can still reach it behind its explicit opt-in (PRD section 17).

    A :class:`~model_gateway.errors.GatewayError` is returned unchanged: it has
    already been classified and sanitised, and re-classifying it would demote a
    precise reason to an internal one. Anything the LiteLLM hierarchy does not
    name is converted by :func:`~model_gateway.errors.normalise_exception`, which
    discards the original text from the public payload entirely.
    """
    if isinstance(exc, GatewayError):
        return exc
    error = _classify_provider_error(exc)
    if error is None:
        return normalise_exception(exc)
    error.__cause__ = exc
    error.__suppress_context__ = True
    return error


# --------------------------------------------------------------------------- request side


def _messages(request: InvocationRequest) -> list[dict[str, object]]:
    """The conversation as LiteLLM message dicts, the instruction forwarded unmerged.

    The invocation-level instruction becomes a ``system`` message of its own at
    the head of the conversation. A ``system`` message inside ``messages`` is
    the conversation-level equivalent and is forwarded as the caller supplied it;
    the two are never folded together (PRD sections 5.1 and 9.5).
    """
    messages: list[dict[str, object]] = []
    if request.system_instruction is not None:
        messages.append({"role": Role.SYSTEM.value, "content": request.system_instruction})
    messages.extend(_message(message) for message in request.messages)
    return messages


def _message(message: Message) -> dict[str, object]:
    """One canonical message as a LiteLLM message dict."""
    result: dict[str, object] = {
        "role": message.role.value,
        "content": _content(message.content),
    }
    if message.name is not None:
        result["name"] = message.name
    if message.tool_calls:
        result["tool_calls"] = [_outgoing_tool_call(call) for call in message.tool_calls]
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    return result


def _content(parts: tuple[ContentPart, ...]) -> str | list[dict[str, object]]:
    """The message's content parts as text or as ordered content blocks.

    A single text part is sent as the plain string the OpenAI-shaped API
    expects; anything else becomes an ordered block list that preserves the
    part sequence — text and images interleaved stay interleaved (PRD 9.4).
    """
    if not parts:
        return ""
    blocks: list[dict[str, object]] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image_url", "image_url": {"url": _image_url(part)}})
        else:
            raise CapabilityError(
                "a refusal part in a request message has no LiteLLM upstream representation: "
                "a refusal is the model's own decision to decline, so a chat completion request "
                "the caller authors cannot carry one, and forwarding it as ordinary text would "
                "collapse two different events into one",
                details={"capability": "refusal"},
            )
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return str(blocks[0]["text"])
    return blocks


def _image_url(part: ImagePart) -> str:
    """The image as a URL LiteLLM accepts: a data URL for a payload, the remote URL otherwise."""
    if part.data is not None:
        media_type = (
            part.media_type if part.media_type is not None else "application/octet-stream"
        )
        return f"data:{media_type};base64,{part.data}"
    if part.url is None:
        raise CapabilityError(
            "an image part carries neither base64 data nor a remote URL, so it has no LiteLLM "
            "upstream representation",
            details={"capability": "image"},
        )
    return part.url


def _outgoing_tool_call(call: ToolCall) -> dict[str, object]:
    """A tool call from the conversation history in LiteLLM's OpenAI-shaped form.

    Structured arguments are serialised; a raw string is forwarded exactly as
    the caller supplied it, because repairing malformed arguments would present
    a call the model never made; ``None`` — no arguments at all — becomes the
    empty string LiteLLM's ``Function`` substitutes for absent arguments.
    """
    arguments = call.arguments
    if isinstance(arguments, Mapping):
        rendered: str = json.dumps(thaw_json(arguments))
    elif arguments is None:
        rendered = ""
    else:
        rendered = arguments
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": rendered},
    }


def _tool(definition: ToolDefinition) -> dict[str, object]:
    """A tool declaration in LiteLLM's OpenAI-shaped function form (PRD 9.3)."""
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": thaw_json(definition.parameters),
        },
    }


def _tool_choice(choice: ToolChoice) -> str | dict[str, str]:
    """The tool-choice policy in LiteLLM's OpenAI-shaped vocabulary (PRD 9.3)."""
    if choice.mode is ToolChoiceMode.SPECIFIC:
        name = choice.name
        if name is None:
            raise ValueError(
                "a 'specific' tool choice must name the tool the model is required to call"
            )
        return {"type": "function", "function": {"name": name}}
    return choice.mode.value


def _add_generation_params(
    params: dict[str, object], config: GenerationConfig
) -> None:
    """Add the generation parameters that were set, leaving the rest to the provider.

    "Unset" and "zero" are different requests, so an unset parameter is omitted
    rather than sent as a default the gateway invented (PRD section 5.1).
    """
    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.max_output_tokens is not None:
        params["max_tokens"] = config.max_output_tokens
    if config.stop_sequences:
        params["stop"] = list(config.stop_sequences)
    if config.seed is not None:
        params["seed"] = config.seed
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    response_format = config.response_format
    if response_format is not None and response_format.mode is not ResponseFormatMode.TEXT:
        params["response_format"] = _response_format(response_format)


def _response_format(response_format: ResponseFormat) -> dict[str, object]:
    """A structured-output requirement in LiteLLM's OpenAI-shaped form (PRD 9.5)."""
    if response_format.mode is ResponseFormatMode.JSON_OBJECT:
        return {"type": "json_object"}
    schema = response_format.schema
    rendered = thaw_json(schema) if isinstance(schema, Mapping) else schema
    json_schema: dict[str, object] = {"schema": rendered, "strict": response_format.strict}
    if response_format.name is not None:
        json_schema["name"] = response_format.name
    return {"type": "json_schema", "json_schema": json_schema}


# --------------------------------------------------------------------------- response side


def _raw_finish_reason(choice: Choices) -> str:
    """The provider's own finish reason, verbatim (PRD section 5.2).

    LiteLLM maps recognised reasons onto its OpenAI-shaped vocabulary and keeps
    the original in ``provider_specific_fields`` whenever the two differ, so the
    native reason is that value when it is present and the mapped reason
    otherwise.
    """
    fields = getattr(choice, "provider_specific_fields", None) or {}
    native = fields.get("native_finish_reason")
    if native is not None:
        return str(native)
    return str(choice.finish_reason)


def _output(message: LiteLLMMessage) -> tuple[OutputItem, ...]:
    """The message's answer as an ordered sequence of canonical output items.

    Text, then images, then tool calls: the order the upstream shape preserves.
    LiteLLM's ``Message`` records content, images and tool calls in separate
    fields, so the relative order of a tool call and the text around it is not
    something the upstream kept, and this fixed order loses nothing it did.
    """
    items: list[OutputItem] = []
    if message.content is not None:
        items.append(TextPart(text=message.content))
    items.extend(_output_image(entry) for entry in getattr(message, "images", None) or ())
    items.extend(_incoming_tool_call(call) for call in message.tool_calls or ())
    return tuple(items)

def _output_image(entry: Mapping[str, object]) -> ImagePart:
    """An upstream image entry as a canonical image part (PRD 9.4)."""
    image_url = entry.get("image_url") if isinstance(entry, Mapping) else None
    url = image_url.get("url") if isinstance(image_url, Mapping) else None
    if not isinstance(url, str) or not url:
        raise InternalError(
            "an upstream image entry carries no URL, and an image without one has no canonical "
            "representation"
        )
    return ImagePart(url=url)


def _incoming_tool_call(call: Any) -> ToolCall:
    """A tool call the model proposed, with its identifier and structured arguments.

    The arguments are a parsed JSON object when they parse to one and the raw
    string when they do not — the gateway records what arrived rather than
    repairing it (PRD section 9.3).
    """
    if not isinstance(call, ChatCompletionMessageToolCall):
        raise InternalError(
            "an upstream tool call is not a function call, and the canonical tool call "
            "representation has no way to carry it"
        )
    name = call.function.name
    if name is None:
        raise InternalError(
            "an upstream tool call carries no function name, and a call the gateway cannot "
            "name has no canonical representation"
        )
    return ToolCall(id=str(call.id), name=name, arguments=_arguments(call.function.arguments))


def _arguments(raw: str | None) -> Mapping[str, Any] | str | None:
    """The tool arguments: a parsed object when they parse, the raw string when they do not."""
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict):
        return freeze_json("tool call arguments", parsed)
    return raw


def _usage(usage: LiteLLMUsage | None) -> TokenUsage:
    """The upstream usage as a canonical usage record (PRD section 13).

    A ``None`` usage means the provider reported nothing, which is not the same
    as zero. Counts on a present LiteLLM ``Usage`` are provider-reported,
    except the total: LiteLLM fills an unreported count with zero, so a total
    of zero beside non-zero parts is that placeholder rather than a reported
    fact, and the gateway derives the total from the parts and names it as the
    estimate it is.
    """
    if usage is None:
        return TokenUsage()
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    estimated: set[str] = set()
    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens
        estimated.add("total_tokens")
    _reject_unrepresentable_usage(usage)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=_cached_tokens(usage),
        estimated_fields=frozenset(estimated),
    )


def _cached_tokens(usage: LiteLLMUsage) -> int | None:
    """The provider-reported cached-token count, or ``None`` when it reported none."""
    details = usage.prompt_tokens_details
    if details is None:
        return None
    return details.cached_tokens


def _reject_unrepresentable_usage(usage: LiteLLMUsage) -> None:
    """Refuse usage fields the canonical model cannot carry (PRD section 9.1).

    Only ``cached_tokens`` has a canonical target among the prompt-token
    details, and none among the completion-token details; a server-tool-use
    record has none either. Each is a provider-reported fact, so the response
    is rejected rather than trimmed. LiteLLM's own ``cost`` is deliberately not
    checked: cost is the gateway's, computed from configured pricing data
    (PRD section 13), not a provider semantic the binding forwards.

    A refusal that recurs — extra token categories in the usage details, such
    as a reasoning-token count — is a signal that the canonical usage model
    (PRD section 13) may genuinely need extending with a field for that
    category. Until it does, the refusal stands: the semantic is not
    allowlisted into :data:`_NON_SEMANTIC_RESPONSE_FIELDS`, because a token
    count is a reported fact about the answer, not backend metadata, and
    ignoring it would hide a difference the caller was never told about.
    """
    prompt_details = usage.prompt_tokens_details
    if prompt_details is not None:
        _reject_extra_detail_fields(
            prompt_details, "prompt token details", frozenset({"cached_tokens"})
        )
    completion_details = usage.completion_tokens_details
    if completion_details is not None:
        _reject_extra_detail_fields(completion_details, "completion token details", frozenset())
    if getattr(usage, "server_tool_use", None) is not None:
        raise InternalError(
            "the upstream usage carries a server-tool-use record, which the canonical usage "
            "representation cannot carry"
        )


def _reject_extra_detail_fields(details: Any, where: str, allowed: frozenset[str]) -> None:
    """Refuse a usage-detail field with no canonical target, naming the first one found."""
    for name in type(details).model_fields:
        if name in allowed:
            continue
        if getattr(details, name, None) is not None:
            raise InternalError(
                f"the upstream usage carries {where[:-1]} field {name!r}, which the canonical "
                f"usage representation cannot carry"
            )


# Response fields classified as provider operational metadata rather than answer
# semantics — the third branch of the four-way rule, the deliberate-ignore
# branch. A field belongs here only by the same act of classification that
# would otherwise refuse it, and the reason it is non-semantic is recorded
# beside it; listing a field is a decision that it changes nothing about the
# answer, never a place to park a semantic the canonical model cannot hold.
# The upstream's own identity fields — ``id``, ``created``, ``model`` — need no
# entry: the gateway stamps its own identity on the response (PRD sections 5.2
# and 9.6), so an upstream identity is deliberately neither mapped nor refused.
_NON_SEMANTIC_RESPONSE_FIELDS: frozenset[str] = frozenset(
    {
        # The fingerprint of the upstream backend that produced the answer: it
        # identifies provider infrastructure, not anything about the answer.
        # Two backends serving the same model return the same tokens, so the
        # fingerprint is backend metadata that does not change what the caller
        # was told — it must not fail an otherwise valid response.
        "system_fingerprint",
    }
)


def _reject_unmappable(response: ModelResponse, choice: Choices, message: LiteLLMMessage) -> None:
    """Refuse a response carrying a semantic the canonical model cannot represent.

    Each check names the field it refuses, so a drop cannot pass undetected:
    a response the canonical model cannot fully represent is rejected as
    unsupported rather than silently trimmed (PRD section 9.1). The one
    response-level field the binding inspects, ``system_fingerprint``, is
    deliberately not here: it is classified non-semantic operational metadata
    and ignored, from :data:`_NON_SEMANTIC_RESPONSE_FIELDS`.
    """
    if getattr(choice, "logprobs", None) is not None:
        raise InternalError(
            "the upstream response carries a log-probability block, which the canonical "
            "representation cannot carry"
        )
    _reject_extra_provider_fields(
        getattr(choice, "provider_specific_fields", None),
        "choice",
        frozenset({"native_finish_reason"}),
    )
    _reject_extra_provider_fields(
        getattr(message, "provider_specific_fields", None), "message", frozenset()
    )
    if getattr(message, "reasoning_content", None) is not None:
        raise InternalError(
            "the upstream message carries reasoning content, which the canonical representation "
            "cannot carry"
        )
    if getattr(message, "thinking_blocks", None) is not None:
        raise InternalError(
            "the upstream message carries thinking blocks, which the canonical representation "
            "cannot carry"
        )
    if getattr(message, "reasoning_items", None) is not None:
        raise InternalError(
            "the upstream message carries reasoning items, which the canonical representation "
            "cannot carry"
        )
    if getattr(message, "annotations", None) is not None:
        raise InternalError(
            "the upstream message carries content-safety annotations, which the canonical "
            "representation cannot carry"
        )
    if getattr(message, "audio", None) is not None:
        raise InternalError(
            "the upstream message carries audio content, which the canonical representation "
            "cannot carry"
        )
    if message.function_call is not None:
        raise InternalError(
            "the upstream message carries a legacy function call, which the canonical tool call "
            "representation cannot carry"
        )


def _reject_extra_provider_fields(
    fields: Mapping[str, Any] | None, where: str, allowed: frozenset[str]
) -> None:
    """Refuse provider-specific fields with no canonical target, naming the first one found.

    On a choice, ``native_finish_reason`` is allowed because it is the mapped
    raw finish reason; on a message nothing is. Any other key is a
    provider-specific semantic the canonical model cannot carry.
    """
    if not fields:
        return
    extra = sorted(set(fields) - allowed)
    if extra:
        raise InternalError(
            f"the upstream {where} carries provider-specific field {extra[0]!r}, which the "
            f"canonical representation cannot carry"
        )


# --------------------------------------------------------------------------- error mapping


def _classify_provider_error(exc: BaseException) -> GatewayError | None:
    """Map one LiteLLM exception onto the taxonomy, or ``None`` if it is not one LiteLLM raises.

    LiteLLM's exception classes extend the OpenAI ones, so the OpenAI base
    checks below also cover the LiteLLM subclasses that share a mapping. The
    specific classes are named first so a refusal names the exact failure; a
    raw OpenAI exception LiteLLM raised directly lands on the same mapping.
    """
    if isinstance(exc, litellm.AuthenticationError):
        return UnauthenticatedError(
            "the upstream rejected the route's credentials",
            reason=ErrorReason.INVALID_CREDENTIALS,
        )
    if isinstance(exc, litellm.PermissionDeniedError):
        return PermissionDeniedError("the upstream denied the request")
    if isinstance(exc, litellm.NotFoundError):
        return NotFoundError("the upstream reported the model was not found")
    if isinstance(exc, litellm.RateLimitError):
        return ResourceExhaustedError("the upstream rate limit was reached")
    if isinstance(exc, litellm.Timeout):
        return DeadlineExceededError("the upstream timed out")
    if isinstance(exc, litellm.APIConnectionError):
        return UnavailableError("the upstream connection failed")
    if isinstance(exc, litellm.APIResponseValidationError):
        return UnavailableError("the upstream returned a malformed response")
    if isinstance(exc, litellm.APIError):
        return _map_api_error(exc)
    if isinstance(exc, openai.APITimeoutError):
        return DeadlineExceededError("the upstream timed out")
    if isinstance(exc, openai.APIConnectionError):
        return UnavailableError("the upstream connection failed")
    if isinstance(exc, openai.APIStatusError):
        # LiteLLM's remaining status errors, raised as the OpenAI bases or as
        # LiteLLM's own 4xx subclasses (BadRequestError,
        # ContextWindowExceededError, UnsupportedParamsError,
        # UnprocessableEntityError): a 5xx is an unavailable upstream, and any
        # other status is the upstream refusing a well-formed request.
        status = exc.status_code
        if isinstance(status, int) and status >= 500:
            return UnavailableError("the upstream is unavailable")
        return InternalError("the upstream rejected the request")
    return None


def _map_api_error(exc: litellm.APIError) -> GatewayError:
    """Map LiteLLM's generic API error by its status code.

    A 5xx means the upstream could not serve the request; anything else is the
    upstream refusing a request the gateway had already validated, which is an
    unclassified upstream failure rather than a caller error.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return UnavailableError("the upstream is unavailable")
    return InternalError("the upstream rejected the request")
