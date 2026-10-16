"""Canonical invocation request and response.

These two types are the gateway core's own contract: what it accepts from a
caller and what it reports back. Both are written in the gateway's vocabulary —
public model IDs, canonical content, canonical finish reasons — and neither
carries a provider's wire format. Mapping either side onto LiteLLM is the
binding's work.

A response reports more than the model's answer. It says which upstream model
actually served it, how many provider attempts that took, whether a fallback was
used, what happened to any idempotency key, and what the call cost. A caller that
cannot see those cannot reconcile its own accounting, nor tell a first attempt
from a rescued one.

The answer itself is an ordered sequence of output items rather than a text field
beside an image field, and the finish reason is reported beside the provider's own
untranslated reason. Both exist for the same reason: a canonical state that cannot
distinguish two different upstream events has already lost information by the time
a caller sees it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Self

from model_gateway.domain.content import Message, as_messages
from model_gateway.domain.generation import (
    FinishReason,
    GenerationConfig,
    Usage,
    as_generation_config,
    as_usage,
)
from model_gateway.domain.json_value import (
    FrozenJsonObject,
    JsonDocument,
    JsonValue,
    thaw_json,
)
from model_gateway.domain.output import (
    OutputItem,
    as_output_items,
    output_item_from_dict,
    output_item_to_dict,
)
from model_gateway.domain.parts import ImagePart, RefusalPart, TextPart
from model_gateway.domain.serialization import (
    as_enumeration,
    as_int,
    as_json_object,
    as_object,
    as_optional_boolean,
    as_optional_float,
    as_optional_text,
    as_required_text,
    as_sequence,
    as_text,
    as_text_tuple,
    describe,
    reject_unknown_keys,
)
from model_gateway.domain.tools import (
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    as_tool_choice,
    as_tool_definitions,
)

__all__ = [
    "IdempotencyStatus",
    "InvocationRequest",
    "InvocationResponse",
    "ResponseModality",
]

_REQUEST_KEYS = frozenset(
    {
        "invocation_id",
        "tenant_id",
        "actor_id",
        "agent_id",
        "model",
        "messages",
        "system_instruction",
        "generation_config",
        "tools",
        "tool_choice",
        "response_modality",
        "fallback_models",
        "idempotency_key",
        "metadata",
        "gcid",
        "action_code",
        "dispatch_idempotency_key",
        "surface",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "invocation_id",
        "model",
        "upstream_model",
        "content",
        "finish_reason",
        "raw_finish_reason",
        "usage",
        "estimated_cost",
        "cost_currency",
        "attempt_count",
        "fallback_used",
        "idempotency_status",
    }
)

_RESERVED_METADATA_KEYS = _REQUEST_KEYS | frozenset(
    parameter.name for parameter in fields(GenerationConfig)
)
"""Field names metadata may not use: the canonical request and its generation parameters.

Derived from the types themselves rather than restated, so a parameter added to
``GenerationConfig`` is covered by the metadata rule without a second edit.
"""

FORWARDABLE_METADATA_KEYS: frozenset[str] = frozenset()
"""Metadata keys a separately specified feature has claimed, and may forward.

It is empty: no feature consumes metadata today, so no metadata reaches a model
or a provider, and a key belongs here only once the feature that consumes it has
its own specification. ``metadata`` is request metadata,
and treating it as a forwarding channel by default is what would turn it into a
way to smuggle an unrepresentable semantic upstream.
"""

_CURRENCY_LENGTH = 3

_OPTIONAL_TEXT = (
    "invocation_id",
    "tenant_id",
    "actor_id",
    "agent_id",
    "system_instruction",
    "idempotency_key",
    "gcid",
    "action_code",
    "dispatch_idempotency_key",
    "surface",
)


class ResponseModality(StrEnum):
    """The kind of output the caller asked for."""

    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class IdempotencyStatus(StrEnum):
    """What the gateway did about a repeated invocation.

    * ``NOT_APPLICABLE`` — the invocation carried no idempotency key, so
      deduplication was never in play.
    * ``NEW`` — the key was accepted and this invocation is the first with it.
    * ``REPLAYED`` — the key matched an earlier invocation with the same request
      fingerprint, and that invocation's recorded result was returned instead of
      calling the provider again.
    * ``INDETERMINATE`` — the gateway cannot say whether the provider executed
      the request. That is the honest answer after a timeout against a provider
      with no idempotency support, and claiming exactly-once execution in that
      case is forbidden.

    A key reused with a different payload is not a replay: the fingerprint differs,
    so the second request is served as ``NEW`` and the provider is called again,
    overwriting the record under that key. Answering it from the stored record
    would give a caller the answer to a request it never made, which is the
    accident the fingerprint exists to prevent.
    """

    NOT_APPLICABLE = "not_applicable"
    NEW = "new"
    REPLAYED = "replayed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class InvocationRequest:
    """One canonical inference invocation.

    ``model`` is the public model ID the caller named, or ``None`` when it named
    none. It is never an upstream model name, and it is not what selects the
    routes: the operation is resolved from the transport the call arrived on —
    the HTTP endpoint or the gRPC method — and the routes configured for that
    operation serve it. A stated model is carried for the caller's own
    correlation and is ignored for routing, so a caller-supplied model name
    cannot reach an unconfigured upstream endpoint.

    ``system_instruction`` is the invocation-level instruction.
    A ``Role.SYSTEM`` message inside ``messages`` is the conversation-level
    equivalent, and the two are not interchangeable: the gateway forwards both
    and merges neither, because silently folding one into the other would change
    what the provider sees.

    ``fallback_models`` names the public model IDs the caller permits as
    fallbacks. An empty tuple means the caller imposes no restriction and the
    configured route order applies.

    ``metadata`` is non-sensitive request metadata and nothing more. It is
    arbitrary JSON held as an immutable mapping, so a validated request cannot
    acquire a field afterwards, and the gateway must not place credentials in it.

    ``gcid``, ``action_code``, ``dispatch_idempotency_key`` and ``surface`` are
    the governance fields the platform's invocation flow stamps. ``gcid`` is the
    caller's learner identity, the key the mana meter and the debit-claim ledger
    are keyed by; ``action_code`` is the priced action the call is metered under;
    ``dispatch_idempotency_key`` is the dispatch key a redelivered turn repeats,
    so one dispatch is debited once; ``surface`` is the caller's surface, which
    the companion containment read keys on. All four are optional and all four
    are carried for governance only: none is forwarded to the model or the
    provider, and a request that states none of them is simply ungoverned —
    the budget fails open, the containment read is skipped and the call is
    served un-metered.

    Two rules bound what it may carry. Both are enforced here rather than left as
    prose, because a semantic parked in metadata is a semantic the gateway
    neither represents nor forwards, and it would disappear between the caller
    and the provider exactly as a dropped parameter does:

    * It is never forwarded to the model or the provider unless a separately
      specified feature consumes a key. No feature does today, so the binding
      must not send it upstream; :attr:`forwardable_metadata` states which keys
      such a feature has claimed, and is empty.
    * It is non-inference-semantic. A key naming a canonical invocation field or
      a generation parameter is refused, because such a key can only be an
      attempt to carry a model parameter outside the contract the gateway
      validates — where it would either be dropped silently or reach the provider
      unvalidated.
    """

    messages: tuple[Message, ...]
    invocation_id: str | None = None
    tenant_id: str | None = None
    actor_id: str | None = None
    agent_id: str | None = None
    system_instruction: str | None = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice | None = None
    response_modality: ResponseModality = ResponseModality.TEXT
    fallback_models: tuple[str, ...] = ()
    idempotency_key: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=FrozenJsonObject)
    model: str | None = None
    gcid: str | None = None
    action_code: str | None = None
    dispatch_idempotency_key: str | None = None
    surface: str | None = None

    def __post_init__(self) -> None:
        model = as_optional_text("model", self.model)
        messages = as_messages("messages", self.messages)
        if not messages:
            raise ValueError(
                "messages must carry at least one message; an invocation with nothing to send "
                "has no canonical representation"
            )
        tools = as_tool_definitions("tools", self.tools)
        tool_choice = as_tool_choice("tool_choice", self.tool_choice)
        _check_tool_choice(tool_choice, tools)

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "messages", messages)
        for name in _OPTIONAL_TEXT:
            object.__setattr__(self, name, as_optional_text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "generation_config",
            as_generation_config("generation_config", self.generation_config),
        )
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "tool_choice", tool_choice)
        object.__setattr__(
            self,
            "response_modality",
            as_enumeration("response_modality", self.response_modality, ResponseModality),
        )
        object.__setattr__(
            self, "fallback_models", as_text_tuple("fallback_models", self.fallback_models)
        )
        metadata = as_json_object("metadata", self.metadata)
        _check_metadata_keys(metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def forwardable_metadata(self) -> FrozenJsonObject:
        """The metadata a separately specified feature has claimed; empty while none does.

        This is the whole of the metadata the gateway may send upstream. A
        consumer that needs to know what may be forwarded reads it here rather
        than reaching for ``metadata`` directly, so a key cannot start reaching a
        provider merely because some call site forgot the rule.
        """
        return FrozenJsonObject(
            {key: value for key, value in self.metadata.items() if key in FORWARDABLE_METADATA_KEYS}
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "invocation_id": self.invocation_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "system_instruction": self.system_instruction,
            "generation_config": self.generation_config.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
            "tool_choice": None if self.tool_choice is None else self.tool_choice.to_dict(),
            "response_modality": self.response_modality.value,
            "fallback_models": list(self.fallback_models),
            "idempotency_key": self.idempotency_key,
            "metadata": thaw_json(self.metadata),
            "gcid": self.gcid,
            "action_code": self.action_code,
            "dispatch_idempotency_key": self.dispatch_idempotency_key,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "invocation request"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _REQUEST_KEYS)

        return cls(
            model=as_optional_text(f"{where} 'model'", payload.get("model")),
            messages=tuple(
                Message.from_dict(item)
                for item in as_sequence(f"{where} 'messages'", payload.get("messages"))
            ),
            invocation_id=as_optional_text(
                f"{where} 'invocation_id'", payload.get("invocation_id")
            ),
            tenant_id=as_optional_text(f"{where} 'tenant_id'", payload.get("tenant_id")),
            actor_id=as_optional_text(f"{where} 'actor_id'", payload.get("actor_id")),
            agent_id=as_optional_text(f"{where} 'agent_id'", payload.get("agent_id")),
            system_instruction=as_optional_text(
                f"{where} 'system_instruction'", payload.get("system_instruction")
            ),
            generation_config=_nested(
                payload.get("generation_config"),
                GenerationConfig.from_dict,
                GenerationConfig(),
            ),
            tools=tuple(
                ToolDefinition.from_dict(item)
                for item in as_sequence(f"{where} 'tools'", payload.get("tools"))
            ),
            tool_choice=_optional_nested(payload.get("tool_choice"), ToolChoice.from_dict),
            response_modality=as_enumeration(
                f"{where} 'response_modality'",
                payload.get("response_modality", ResponseModality.TEXT),
                ResponseModality,
            ),
            fallback_models=as_text_tuple(
                f"{where} 'fallback_models'", payload.get("fallback_models")
            ),
            idempotency_key=as_optional_text(
                f"{where} 'idempotency_key'", payload.get("idempotency_key")
            ),
            metadata=as_json_object(f"{where} 'metadata'", payload.get("metadata")),
            gcid=as_optional_text(f"{where} 'gcid'", payload.get("gcid")),
            action_code=as_optional_text(f"{where} 'action_code'", payload.get("action_code")),
            dispatch_idempotency_key=as_optional_text(
                f"{where} 'dispatch_idempotency_key'", payload.get("dispatch_idempotency_key")
            ),
            surface=as_optional_text(f"{where} 'surface'", payload.get("surface")),
        )


@dataclass(frozen=True)
class InvocationResponse:
    """The canonical result of one invocation.

    ``model`` is the public model ID of the operation that served the invocation —
    the registry's fixed ID for the text operation, stated by the gateway rather
    than by the caller, because the operation is resolved from the transport and
    a model ID the caller may state is carried for correlation and ignored for
    routing — and ``upstream_model`` is the route's own model ID, the one that
    actually served the request. The two are reported separately because a
    fallback may have served it, and a caller that cannot tell which upstream
    answered cannot reconcile its own accounting. ``upstream_model`` is the
    route's configured model ID, not the provider-prefixed ``<format>/<model_id>``
    string the binding builds, because the provider format is a binding detail.

    ``content`` is the model's answer as an ordered sequence of output items, and
    it is the only place the answer is stored. Text, images, refusals and tool
    calls can interleave — a model may emit text, then an image, then more text,
    or explain itself, call a tool and explain the result — and separate ``text``,
    ``images`` and ``tool_calls`` fields cannot express that: concatenating the
    text parts loses where the images fell. ``text``, ``images``, ``tool_calls``
    and ``refusals`` are therefore derived views over ``content``, convenient for
    the common case and never authoritative.

    ``finish_reason`` is the gateway's classification, and ``raw_finish_reason``
    is the provider's own reason recorded verbatim. The pair exists because
    ``UNKNOWN`` must never mean "the gateway deleted what upstream told us": when
    a reason could not be classified, the raw value still says what arrived, and
    ``None`` means the provider supplied no reason at all. The classification is
    checked against its source — a reason the vocabulary recognises cannot be
    reported as anything else — so a consumer can trust that the two agree.

    ``attempt_count`` counts provider attempts, including retries of the same
    route, and is at least one because a response exists only once an attempt
    produced one. ``fallback_used`` records that the invocation took a fallback
    path: a route other than the primary one served it because the normal
    primary path could not. It is deliberately narrower than "the answering
    route was anything other than the first configured one", which was also
    true whenever capability routing passed the primary over. Capability
    routing can never set this flag: the registry advertises the primary route's
    capabilities as the public model's own, so a request the
    primary cannot honour is refused rather than routed elsewhere.

    ``estimated_cost`` and ``cost_currency`` come from configured pricing data
    and are ``None`` when cost tracking is off or no price is
    configured for the route; ``None`` never means zero. An amount is always
    reported with the currency it is denominated in, so a figure is never
    ambiguous.
    """

    model: str
    upstream_model: str
    invocation_id: str | None = None
    content: tuple[OutputItem, ...] = ()
    finish_reason: FinishReason = FinishReason.UNKNOWN
    raw_finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    estimated_cost: float | None = None
    cost_currency: str | None = None
    attempt_count: int = 1
    fallback_used: bool = False
    idempotency_status: IdempotencyStatus = IdempotencyStatus.NOT_APPLICABLE

    def __post_init__(self) -> None:
        estimated_cost = as_optional_float("estimated_cost", self.estimated_cost, minimum=0.0)
        cost_currency = _as_currency("cost_currency", self.cost_currency)
        if (estimated_cost is None) != (cost_currency is None):
            raise ValueError(
                "estimated_cost and cost_currency must be given together; an amount without its "
                "currency, or a currency without an amount, is not a cost the gateway can report"
            )

        finish_reason = _as_finish_reason(self.finish_reason)
        raw_finish_reason = _as_raw_finish_reason(self.raw_finish_reason)
        _check_finish_reason(finish_reason, raw_finish_reason)

        object.__setattr__(self, "model", as_required_text("model", self.model))
        object.__setattr__(
            self, "upstream_model", as_required_text("upstream_model", self.upstream_model)
        )
        object.__setattr__(
            self, "invocation_id", as_optional_text("invocation_id", self.invocation_id)
        )
        object.__setattr__(self, "content", as_output_items("content", self.content))
        object.__setattr__(self, "finish_reason", finish_reason)
        object.__setattr__(self, "raw_finish_reason", raw_finish_reason)
        object.__setattr__(self, "usage", as_usage("usage", self.usage))
        object.__setattr__(self, "estimated_cost", estimated_cost)
        object.__setattr__(self, "cost_currency", cost_currency)
        object.__setattr__(
            self, "attempt_count", as_int("attempt_count", self.attempt_count, minimum=1)
        )
        object.__setattr__(
            self,
            "fallback_used",
            as_optional_boolean("fallback_used", self.fallback_used, False),
        )
        object.__setattr__(
            self,
            "idempotency_status",
            as_enumeration("idempotency_status", self.idempotency_status, IdempotencyStatus),
        )

    @property
    def text(self) -> str:
        """The text parts concatenated in order; derived from ``content``, never stored."""
        return "".join(item.text for item in self.content if isinstance(item, TextPart))

    @property
    def images(self) -> tuple[ImagePart, ...]:
        """The image parts in order; derived from ``content``, never stored."""
        return tuple(item for item in self.content if isinstance(item, ImagePart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The tool calls in order; derived from ``content``, never stored."""
        return tuple(item for item in self.content if isinstance(item, ToolCall))

    @property
    def refusals(self) -> tuple[RefusalPart, ...]:
        """The refusal parts in order; derived from ``content``, never stored."""
        return tuple(item for item in self.content if isinstance(item, RefusalPart))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "invocation_id": self.invocation_id,
            "model": self.model,
            "upstream_model": self.upstream_model,
            "content": [output_item_to_dict(item) for item in self.content],
            "finish_reason": self.finish_reason.value,
            "raw_finish_reason": self.raw_finish_reason,
            "usage": self.usage.to_dict(),
            "estimated_cost": self.estimated_cost,
            "cost_currency": self.cost_currency,
            "attempt_count": self.attempt_count,
            "fallback_used": self.fallback_used,
            "idempotency_status": self.idempotency_status.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "invocation response"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESPONSE_KEYS)

        return cls(
            model=as_required_text(f"{where} 'model'", payload.get("model")),
            upstream_model=as_required_text(
                f"{where} 'upstream_model'", payload.get("upstream_model")
            ),
            invocation_id=as_optional_text(
                f"{where} 'invocation_id'", payload.get("invocation_id")
            ),
            content=tuple(
                output_item_from_dict(item)
                for item in as_sequence(f"{where} 'content'", payload.get("content"))
            ),
            finish_reason=_as_finish_reason(payload.get("finish_reason")),
            raw_finish_reason=_as_raw_finish_reason(payload.get("raw_finish_reason")),
            usage=_nested(payload.get("usage"), Usage.from_dict, Usage()),
            estimated_cost=as_optional_float(
                f"{where} 'estimated_cost'", payload.get("estimated_cost"), minimum=0.0
            ),
            cost_currency=_as_currency(f"{where} 'cost_currency'", payload.get("cost_currency")),
            attempt_count=as_int(
                f"{where} 'attempt_count'", payload.get("attempt_count", 1), minimum=1
            ),
            fallback_used=as_optional_boolean(
                f"{where} 'fallback_used'", payload.get("fallback_used"), False
            ),
            idempotency_status=as_enumeration(
                f"{where} 'idempotency_status'",
                payload.get("idempotency_status", IdempotencyStatus.NOT_APPLICABLE),
                IdempotencyStatus,
            ),
        )


def _check_tool_choice(tool_choice: ToolChoice | None, tools: tuple[ToolDefinition, ...]) -> None:
    """Refuse a tool choice the declared tools cannot satisfy."""
    if tool_choice is None or tool_choice.mode in (ToolChoiceMode.AUTO, ToolChoiceMode.NONE):
        return
    if not tools:
        raise ValueError(
            f"tool_choice {tool_choice.mode.value!r} requires at least one declared tool; the "
            f"model cannot be made to call a tool the invocation does not declare"
        )
    declared = frozenset(tool.name for tool in tools)
    if tool_choice.name is not None and tool_choice.name not in declared:
        names = ", ".join(sorted(declared))
        raise ValueError(
            f"tool_choice names {tool_choice.name!r}, which the invocation does not declare; "
            f"declared tools are {names}"
        )


def _as_finish_reason(value: object) -> FinishReason:
    """A finish reason, mapping anything unrecognised onto ``UNKNOWN``.

    A provider reason the gateway does not know is not an error: it is mapped to
    ``UNKNOWN`` so the field still says something true, rather than being dropped
    or turned into a failure. A value that is not a string at
    all is malformed input, and is refused.
    """
    if isinstance(value, FinishReason):
        return value
    if value is None or isinstance(value, str):
        return FinishReason.coerce(value)
    raise ValueError(f"finish_reason must be a string, got {describe(value)}")


def _as_raw_finish_reason(value: object) -> str | None:
    """The provider's own finish reason, recorded verbatim, or ``None`` if it gave none.

    The value is kept exactly as the provider supplied it, including an empty
    string. The difference between "the provider said nothing" and "the provider
    said something the gateway could not classify" is the whole reason for
    keeping it, and a verbatim record is not something to tidy up.
    """
    if value is None:
        return None
    return as_text("raw_finish_reason", value)


def _check_finish_reason(finish_reason: FinishReason, raw: str | None) -> None:
    """Refuse a classification the recorded source reason contradicts.

    A raw reason the vocabulary recognises has exactly one correct
    classification, so reporting another one leaves a consumer unable to tell
    which of the two to believe. A raw reason the vocabulary does not recognise
    is left to the binding's own judgement — including ``UNKNOWN``, which is the
    honest answer when the gateway has not been taught the reason — and ``None``
    means the provider supplied none.
    """
    if raw is None:
        return
    recognised = FinishReason.coerce(raw)
    if recognised is not FinishReason.UNKNOWN and recognised is not finish_reason:
        raise ValueError(
            f"finish_reason {finish_reason.value!r} contradicts raw_finish_reason {raw!r}, which "
            f"the gateway classifies as {recognised.value!r}; a response must not report a "
            f"classification its own recorded source does not support"
        )


def _check_metadata_keys(metadata: Mapping[str, JsonValue]) -> None:
    """Refuse metadata that names a canonical field.

    Metadata is non-inference-semantic. A key that spells a canonical invocation
    or generation parameter is an attempt to carry a model parameter outside the
    contract the gateway validates, and such a parameter would either be dropped
    silently or reach a provider unchecked.
    """
    reserved = sorted(set(metadata) & _RESERVED_METADATA_KEYS)
    if reserved:
        raise ValueError(
            f"metadata key {reserved[0]!r} is a canonical invocation field name; metadata is "
            f"non-inference-semantic and is never forwarded to a model or provider, so a model "
            f"parameter parked there is a semantic the gateway neither represents nor sends"
        )


def _as_currency(where: str, value: object) -> str | None:
    """An ISO 4217 currency code, or ``None``."""
    currency = as_optional_text(where, value)
    if currency is None:
        return None
    if len(currency) != _CURRENCY_LENGTH or not currency.isascii() or not currency.isalpha():
        raise ValueError(
            f"{where} must be a three-letter ISO 4217 currency code such as 'USD', got {currency!r}"
        )
    if currency != currency.upper():
        raise ValueError(f"{where} must be an uppercase ISO 4217 code, got {currency!r}")
    return currency


def _nested[T](value: object, parse: Callable[[object], T], default: T) -> T:
    """Parse a nested canonical object, or fall back to its default."""
    if value is None:
        return default
    return parse(value)


def _optional_nested[T](value: object, parse: Callable[[object], T]) -> T | None:
    """Parse an optional nested canonical object."""
    if value is None:
        return None
    return parse(value)
