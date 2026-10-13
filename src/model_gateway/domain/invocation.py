"""Canonical invocation request and response (PRD sections 4.4, 5.1, 5.2, 12 and 13).

These two types are the gateway core's own contract: what it accepts from a
caller and what it reports back. Both are written in the gateway's vocabulary —
public model IDs, canonical content, canonical finish reasons — and neither
carries a provider's wire format. Mapping either side onto LiteLLM is the
binding's work (PRD section 9).

A response reports more than the model's answer. It says which upstream model
actually served it, how many provider attempts that took, whether a fallback was
used, what happened to any idempotency key, and what the call cost. A caller that
cannot see those cannot reconcile its own accounting, nor tell a first attempt
from a rescued one (PRD sections 5.2, 11.2, 12 and 13).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from model_gateway.domain.content import ImagePart, Message, as_image_parts, as_messages
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
    as_tool_calls,
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
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "invocation_id",
        "model",
        "upstream_model",
        "text",
        "tool_calls",
        "images",
        "finish_reason",
        "usage",
        "estimated_cost",
        "cost_currency",
        "attempt_count",
        "fallback_used",
        "idempotency_status",
    }
)

_CURRENCY_LENGTH = 3

_OPTIONAL_TEXT = (
    "invocation_id",
    "tenant_id",
    "actor_id",
    "agent_id",
    "system_instruction",
    "idempotency_key",
)


class ResponseModality(StrEnum):
    """The kind of output the caller asked for (PRD section 5.1)."""

    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class IdempotencyStatus(StrEnum):
    """What the gateway did about a repeated invocation (PRD sections 5.2 and 12).

    * ``NOT_APPLICABLE`` — the invocation carried no idempotency key, so
      deduplication was never in play.
    * ``NEW`` — the key was accepted and this invocation is the first with it.
    * ``REPLAYED`` — the key matched an earlier invocation with the same request
      fingerprint, and that invocation's recorded result was returned instead of
      calling the provider again.
    * ``INDETERMINATE`` — the gateway cannot say whether the provider executed
      the request. That is the honest answer after a timeout against a provider
      with no idempotency support, and PRD section 12 forbids claiming
      exactly-once execution in that case.

    A key reused with a different payload is a conflict rather than a status: it
    is refused before the provider is called.
    """

    NOT_APPLICABLE = "not_applicable"
    NEW = "new"
    REPLAYED = "replayed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class InvocationRequest:
    """One canonical inference invocation (PRD sections 4.4, 5.1 and 12).

    ``model`` is the public model ID, never an upstream model name: callers name
    one of the gateway's fixed logical models and the route resolver decides
    which upstream serves it (PRD section 10). Caller-supplied model names must
    not be able to reach an unconfigured endpoint, so the canonical request
    carries the public ID and nothing else.

    ``system_instruction`` is the invocation-level instruction of PRD section
    5.1. A ``Role.SYSTEM`` message inside ``messages`` is the conversation-level
    equivalent, and the two are not interchangeable: the gateway forwards both
    and merges neither, because silently folding one into the other would change
    what the provider sees (PRD section 9.3).

    ``fallback_models`` names the public model IDs the caller permits as
    fallbacks. An empty tuple means the caller imposes no restriction and the
    configured route order applies (PRD section 11.2).

    ``metadata`` is non-sensitive request metadata. It is arbitrary JSON held as
    an immutable mapping, so a validated request cannot acquire a field
    afterwards, and the gateway must not place credentials in it.
    """

    model: str
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

    def __post_init__(self) -> None:
        model = as_required_text("model", self.model)
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
        object.__setattr__(self, "metadata", as_json_object("metadata", self.metadata))

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
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "invocation request"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _REQUEST_KEYS)

        return cls(
            model=as_required_text(f"{where} 'model'", payload.get("model")),
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
        )


@dataclass(frozen=True)
class InvocationResponse:
    """The canonical result of one invocation (PRD sections 4.4, 5.2 and 13).

    ``model`` is the public model ID the caller asked for, and ``upstream_model``
    is the route's own model ID — the one that actually served the request. The
    two are reported separately because a fallback may have served it, and a
    caller that cannot tell which upstream answered cannot reconcile its own
    accounting (PRD sections 5.2 and 11.2). ``upstream_model`` is the route's
    configured model ID, not the provider-prefixed ``<format>/<model_id>`` string
    the binding builds, because the provider format is a binding detail
    (PRD section 9.4).

    ``attempt_count`` counts provider attempts, including retries of the same
    route, and is at least one because a response exists only once an attempt
    produced one. ``fallback_used`` says whether the answering route was anything
    other than the first configured one.

    ``estimated_cost`` and ``cost_currency`` come from configured pricing data
    (PRD section 13) and are ``None`` when cost tracking is off or no price is
    configured for the route; ``None`` never means zero. An amount is always
    reported with the currency it is denominated in, so a figure is never
    ambiguous.
    """

    model: str
    upstream_model: str
    invocation_id: str | None = None
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    images: tuple[ImagePart, ...] = ()
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: Usage = field(default_factory=Usage)
    estimated_cost: float | None = None
    cost_currency: str | None = None
    attempt_count: int = 1
    fallback_used: bool = False
    idempotency_status: IdempotencyStatus = IdempotencyStatus.NOT_APPLICABLE

    def __post_init__(self) -> None:
        estimated_cost = as_optional_float(
            "estimated_cost", self.estimated_cost, minimum=0.0
        )
        cost_currency = _as_currency("cost_currency", self.cost_currency)
        if (estimated_cost is None) != (cost_currency is None):
            raise ValueError(
                "estimated_cost and cost_currency must be given together; an amount without its "
                "currency, or a currency without an amount, is not a cost the gateway can report"
            )

        object.__setattr__(self, "model", as_required_text("model", self.model))
        object.__setattr__(
            self, "upstream_model", as_required_text("upstream_model", self.upstream_model)
        )
        object.__setattr__(
            self, "invocation_id", as_optional_text("invocation_id", self.invocation_id)
        )
        object.__setattr__(self, "text", as_text("text", self.text))
        object.__setattr__(self, "tool_calls", as_tool_calls("tool_calls", self.tool_calls))
        object.__setattr__(self, "images", as_image_parts("images", self.images))
        object.__setattr__(self, "finish_reason", _as_finish_reason(self.finish_reason))
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
            as_enumeration(
                "idempotency_status", self.idempotency_status, IdempotencyStatus
            ),
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "invocation_id": self.invocation_id,
            "model": self.model,
            "upstream_model": self.upstream_model,
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "images": [image.to_dict() for image in self.images],
            "finish_reason": self.finish_reason.value,
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
            text=as_text(f"{where} 'text'", payload.get("text", "")),
            tool_calls=tuple(
                ToolCall.from_dict(item)
                for item in as_sequence(f"{where} 'tool_calls'", payload.get("tool_calls"))
            ),
            images=tuple(
                ImagePart.from_dict(item)
                for item in as_sequence(f"{where} 'images'", payload.get("images"))
            ),
            finish_reason=_as_finish_reason(payload.get("finish_reason")),
            usage=_nested(payload.get("usage"), Usage.from_dict, Usage()),
            estimated_cost=as_optional_float(
                f"{where} 'estimated_cost'", payload.get("estimated_cost"), minimum=0.0
            ),
            cost_currency=_as_currency(
                f"{where} 'cost_currency'", payload.get("cost_currency")
            ),
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
    """Refuse a tool choice the declared tools cannot satisfy (PRD section 9.3)."""
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
    or turned into a failure (PRD section 5.2). A value that is not a string at
    all is malformed input, and is refused.
    """
    if isinstance(value, FinishReason):
        return value
    if value is None or isinstance(value, str):
        return FinishReason.coerce(value)
    raise ValueError(f"finish_reason must be a string, got {describe(value)}")


def _as_currency(where: str, value: object) -> str | None:
    """An ISO 4217 currency code, or ``None``."""
    currency = as_optional_text(where, value)
    if currency is None:
        return None
    if len(currency) != _CURRENCY_LENGTH or not currency.isascii() or not currency.isalpha():
        raise ValueError(
            f"{where} must be a three-letter ISO 4217 currency code such as 'USD', got "
            f"{currency!r}"
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
