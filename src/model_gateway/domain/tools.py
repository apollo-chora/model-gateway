"""Canonical tool calling: declarations, calls, results and choice (PRD section 9.3).

The gateway declares tools and relays calls; it never executes one. A tool call
is a request from the model, and the caller — or a trusted executor the caller
registered — decides whether and how to run it (PRD section 9.3). Nothing in this
module has an opinion about how a provider spells a tool, which is LiteLLM's
work (PRD section 9).

A call's arguments are kept in the form the provider reported them: a parsed JSON
object when they parsed, the raw string when they did not, and ``None`` when the
provider reported no arguments at all. Those are three different facts and the
canonical type keeps them apart rather than collapsing them into one.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from model_gateway.domain.json_value import (
    FrozenJsonObject,
    JsonDocument,
    JsonValue,
    thaw_json,
)
from model_gateway.domain.parts import (
    ContentPart,
    TextPart,
    as_content_parts,
    content_part_from_dict,
)
from model_gateway.domain.serialization import (
    as_enumeration,
    as_json_object,
    as_object,
    as_optional_boolean,
    as_optional_text,
    as_required_text,
    as_sequence,
    as_text,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "ToolCall",
    "ToolChoice",
    "ToolChoiceMode",
    "ToolDefinition",
    "ToolResult",
    "as_tool_calls",
    "as_tool_choice",
    "as_tool_definitions",
]

_DEFINITION_KEYS = frozenset({"name", "description", "parameters"})
_CALL_KEYS = frozenset({"id", "name", "arguments"})
_RESULT_KEYS = frozenset({"tool_call_id", "content", "is_error"})
_CHOICE_KEYS = frozenset({"mode", "name"})


class ToolChoiceMode(StrEnum):
    """How the model may use the tools an invocation declares (PRD section 9.3)."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class ToolDefinition:
    """A tool the caller declares for an invocation (PRD section 9.3).

    ``parameters`` is the JSON Schema for the tool's arguments, stored as the
    caller supplied it and never rewritten: a schema the gateway had filled in
    defaults for would describe a different tool than the one declared. An empty
    mapping means the tool takes no arguments.

    Declaring a tool authorises nothing. It tells the model what exists; the
    gateway still must not execute what the model asks for (PRD section 9.3).
    """

    name: str
    description: str = ""
    parameters: Mapping[str, JsonValue] = field(default_factory=FrozenJsonObject)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", as_required_text("tool definition 'name'", self.name))
        object.__setattr__(
            self, "description", as_text("tool definition 'description'", self.description)
        )
        object.__setattr__(
            self, "parameters", as_json_object("tool definition 'parameters'", self.parameters)
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": thaw_json(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "tool definition"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _DEFINITION_KEYS)
        return cls(
            name=as_required_text(f"{where} 'name'", payload.get("name")),
            description=as_text(f"{where} 'description'", payload.get("description", "")),
            parameters=as_json_object(f"{where} 'parameters'", payload.get("parameters")),
        )


@dataclass(frozen=True)
class ToolCall:
    """A call the model proposed (PRD section 9.3).

    ``arguments`` is the parsed JSON object when the provider supplied
    well-formed arguments, and the raw string when it did not: the gateway
    records what arrived rather than repairing it, because substituting an empty
    object for malformed arguments would present a call the model never made.
    ``None`` means the provider reported no arguments at all, which is not the
    same as ``{}`` — the arguments of a tool that takes none.

    The gateway does not execute this call (PRD section 9.3); it returns the call
    so that the caller, or a trusted executor, can.
    """

    id: str
    name: str
    arguments: Mapping[str, JsonValue] | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", as_required_text("tool call 'id'", self.id))
        object.__setattr__(self, "name", as_required_text("tool call 'name'", self.name))
        arguments = self.arguments
        if arguments is not None and not isinstance(arguments, str):
            object.__setattr__(
                self, "arguments", as_json_object("tool call 'arguments'", arguments)
            )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": thaw_json(self.arguments),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "tool call"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _CALL_KEYS)
        return cls(
            id=as_required_text(f"{where} 'id'", payload.get("id")),
            name=as_required_text(f"{where} 'name'", payload.get("name")),
            arguments=_as_arguments(f"{where} 'arguments'", payload.get("arguments")),
        )


@dataclass(frozen=True)
class ToolResult:
    """The outcome a tool call produced (PRD section 9.3).

    ``content`` is an ordered sequence of content parts — the same
    representation a message carries — so a tool that returns an image, or text
    and an image together, needs no different shape from a tool that returns
    text. The gateway does not execute tools and so never produces this record
    itself; it exists so that a trusted executor's result has a canonical form,
    and widening it now costs nothing while leaving it as ``str`` would
    knowingly require a migration as soon as a tool returns something else
    (PRD section 9.4).

    ``is_error`` records a failed execution as a result rather than as an
    exception: a tool that failed is information the model needs, not a reason to
    abandon the invocation.

    A tool result is a standalone record of one execution. Inside a conversation
    the same fact appears as a ``Role.TOOL`` message carrying the matching
    ``tool_call_id``.
    """

    tool_call_id: str
    content: tuple[ContentPart, ...] = ()
    is_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            as_required_text("tool result 'tool_call_id'", self.tool_call_id),
        )
        object.__setattr__(
            self, "content", as_content_parts("tool result 'content'", self.content)
        )
        object.__setattr__(
            self,
            "is_error",
            as_optional_boolean("tool result 'is_error'", self.is_error, False),
        )

    @property
    def text(self) -> str:
        """The result's text parts, concatenated in order; a derived view, never stored."""
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": [part.to_dict() for part in self.content],
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "tool result"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESULT_KEYS)
        return cls(
            tool_call_id=as_required_text(f"{where} 'tool_call_id'", payload.get("tool_call_id")),
            content=tuple(
                content_part_from_dict(item)
                for item in as_sequence(f"{where} 'content'", payload.get("content"))
            ),
            is_error=as_optional_boolean(f"{where} 'is_error'", payload.get("is_error"), False),
        )


@dataclass(frozen=True)
class ToolChoice:
    """The caller's tool-choice policy (PRD section 9.3).

    ``NONE`` means tools may be declared but must not be called. It is not the
    same as sending no choice at all, which is ``None`` on the invocation and
    leaves the decision to the provider's own default; the two are kept apart
    because a provider that defaults to calling tools would behave differently
    under each.

    ``SPECIFIC`` names the one tool the model must call, and that name must be
    one of the tools the invocation declares — a choice naming an undeclared tool
    is not a request any provider could honour (PRD section 9.5).
    """

    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    name: str | None = None

    def __post_init__(self) -> None:
        mode = as_enumeration("tool choice 'mode'", self.mode, ToolChoiceMode)
        name = as_optional_text("tool choice 'name'", self.name)
        if mode is ToolChoiceMode.SPECIFIC and name is None:
            raise ValueError(
                "a 'specific' tool choice must name the tool the model is required to call"
            )
        if mode is not ToolChoiceMode.SPECIFIC and name is not None:
            raise ValueError(
                f"a {mode.value!r} tool choice must not name a tool; only 'specific' names one"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "name", name)

    @classmethod
    def auto(cls) -> Self:
        """Let the model decide whether to call a tool."""
        return cls(mode=ToolChoiceMode.AUTO)

    @classmethod
    def none(cls) -> Self:
        """Declare tools, but require that none is called."""
        return cls(mode=ToolChoiceMode.NONE)

    @classmethod
    def required(cls) -> Self:
        """Require the model to call at least one of the declared tools."""
        return cls(mode=ToolChoiceMode.REQUIRED)

    @classmethod
    def specific(cls, name: str) -> Self:
        """Require the model to call the named tool."""
        return cls(mode=ToolChoiceMode.SPECIFIC, name=name)

    def to_dict(self) -> dict[str, JsonDocument]:
        return {"mode": self.mode.value, "name": self.name}

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "tool choice"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _CHOICE_KEYS)
        return cls(
            mode=as_enumeration(f"{where} 'mode'", payload.get("mode"), ToolChoiceMode),
            name=as_optional_text(f"{where} 'name'", payload.get("name")),
        )


def as_tool_definitions(where: str, value: object) -> tuple[ToolDefinition, ...]:
    """Coerce a sequence of tool declarations, refusing anything else."""
    definitions = as_sequence(where, value)
    for index, definition in enumerate(definitions):
        if not isinstance(definition, ToolDefinition):
            raise ValueError(
                f"{where}[{index}] must be a ToolDefinition, got {describe(definition)}"
            )
    return definitions


def as_tool_calls(where: str, value: object) -> tuple[ToolCall, ...]:
    """Coerce a sequence of tool calls, refusing anything else."""
    calls = as_sequence(where, value)
    for index, call in enumerate(calls):
        if not isinstance(call, ToolCall):
            raise ValueError(f"{where}[{index}] must be a ToolCall, got {describe(call)}")
    return calls


def as_tool_choice(where: str, value: object) -> ToolChoice | None:
    """A tool choice, or ``None`` when the caller expressed none."""
    if value is None:
        return None
    if not isinstance(value, ToolChoice):
        raise ValueError(f"{where} must be a ToolChoice, got {describe(value)}")
    return value


def _as_arguments(where: str, value: object) -> Mapping[str, JsonValue] | str | None:
    if value is None or isinstance(value, str):
        return value
    return as_json_object(where, value)
