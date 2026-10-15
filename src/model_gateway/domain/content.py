"""Canonical conversation turns: roles and messages.

A message is a role plus an ordered sequence of content parts, so one message can
carry text and images together. The parts themselves live in
:mod:`model_gateway.domain.parts`, because a message is not their only home: a
tool result and a model's output are ordered part sequences too.

A message holds a tuple of parts, so a conversation that has been validated
cannot change while it is being sent.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.parts import (
    ContentPart,
    as_content_parts,
    content_part_from_dict,
    reject_refusal_parts,
)
from model_gateway.domain.serialization import (
    as_enumeration,
    as_object,
    as_optional_text,
    as_sequence,
    describe,
    reject_unknown_keys,
)
from model_gateway.domain.tools import ToolCall, as_tool_calls

__all__ = ["Message", "Role", "as_messages"]

_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_calls", "tool_call_id"})


class Role(StrEnum):
    """The author of a message.

    The vocabulary is the gateway's own. A role it does not recognise cannot be
    carried into a message at all, because there is no canonical way to present
    it to a provider and inventing one would change the conversation the caller
    asked for.
    """

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """One turn of a conversation.

    ``tool_calls`` belongs to an assistant message, the only role that proposes a
    call, and ``tool_call_id`` belongs to a tool message, the only role that
    answers one — and a tool message must carry it, because a result that cannot
    be matched to its call is not a result. Both rules are enforced here so that
    an inconsistent conversation fails at construction rather than upstream,
    where the provider's complaint would be less specific.

    ``content`` is an ordered sequence of content parts. It may be empty: a
    provider may report an assistant turn whose only content is a tool call, and
    a tool result is not always text. Only an assistant message may carry a
    refusal part: a refusal is the model's own decision to decline, so a message
    the caller authors cannot contain one.
    """

    role: Role
    content: tuple[ContentPart, ...] = ()
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        role = as_enumeration("message 'role'", self.role, Role)
        content = as_content_parts("message 'content'", self.content)
        name = as_optional_text("message 'name'", self.name)
        tool_calls = as_tool_calls("message 'tool_calls'", self.tool_calls)
        tool_call_id = as_optional_text("message 'tool_call_id'", self.tool_call_id)

        if role is not Role.ASSISTANT:
            reject_refusal_parts(f"a {role.value!r} message", content)
        if tool_calls and role is not Role.ASSISTANT:
            raise ValueError(
                f"a {role.value!r} message cannot carry tool calls; only an assistant message "
                f"proposes a call"
            )
        if tool_call_id is not None and role is not Role.TOOL:
            raise ValueError(
                f"a {role.value!r} message cannot carry a tool call id; only a tool message "
                f"answers a call"
            )
        if role is Role.TOOL and tool_call_id is None:
            raise ValueError(
                "a tool message must carry the 'tool_call_id' it answers; a result that cannot "
                "be matched to its call cannot be represented"
            )

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "tool_call_id", tool_call_id)

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "role": self.role.value,
            "content": [part.to_dict() for part in self.content],
            "name": self.name,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "message"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _MESSAGE_KEYS)
        return cls(
            role=as_enumeration(f"{where} 'role'", payload.get("role"), Role),
            content=tuple(
                content_part_from_dict(item)
                for item in as_sequence(f"{where} 'content'", payload.get("content"))
            ),
            name=as_optional_text(f"{where} 'name'", payload.get("name")),
            tool_calls=tuple(
                ToolCall.from_dict(item)
                for item in as_sequence(f"{where} 'tool_calls'", payload.get("tool_calls"))
            ),
            tool_call_id=as_optional_text(f"{where} 'tool_call_id'", payload.get("tool_call_id")),
        )


def as_messages(where: str, value: object) -> tuple[Message, ...]:
    """Coerce a sequence of messages, refusing anything else."""
    messages = as_sequence(where, value)
    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise ValueError(f"{where}[{index}] must be a Message, got {describe(message)}")
    return messages
