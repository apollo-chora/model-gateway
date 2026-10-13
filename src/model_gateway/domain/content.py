"""Canonical conversation content: roles, content parts and messages (PRD 4.4, 5.1, 9.2).

A message is a role plus a sequence of content parts, so one message can carry
text and images together (PRD section 9.2). An image arrives either as base64
data or as an authorised remote URL, and never as anything else: the gateway does
not fetch an image on its own authority, so a URL that reaches this type is one
the URL policy has already admitted.

Parts and messages hold tuples rather than lists, so a conversation that has been
validated cannot change while it is being sent.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.serialization import (
    as_enumeration,
    as_object,
    as_optional_text,
    as_sequence,
    as_text,
    describe,
    reject_unknown_keys,
)
from model_gateway.domain.tools import ToolCall, as_tool_calls

__all__ = [
    "ContentPart",
    "ImagePart",
    "Message",
    "Role",
    "TextPart",
    "as_image_parts",
    "as_messages",
    "content_part_from_dict",
]

_TYPE = "type"
_TEXT = "text"
_IMAGE = "image"

_TEXT_PART_KEYS = frozenset({_TYPE, _TEXT})
_IMAGE_PART_KEYS = frozenset({_TYPE, "data", "url", "media_type"})
_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_calls", "tool_call_id"})


class Role(StrEnum):
    """The author of a message (PRD section 4.4).

    The vocabulary is the gateway's own. A role it does not recognise cannot be
    carried into a message at all, because there is no canonical way to present
    it to a provider and inventing one would change the conversation the caller
    asked for (PRD section 9.3).
    """

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class TextPart:
    """A run of text inside a message (PRD section 9.2).

    Empty text is allowed: a provider may legitimately report a text part with
    nothing in it, and refusing it would leave the gateway unable to represent a
    response it actually received.
    """

    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", as_text("text part 'text'", self.text))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {_TYPE: _TEXT, _TEXT: self.text}

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "text part"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _TEXT_PART_KEYS)
        return cls(text=as_text(f"{where} 'text'", payload.get(_TEXT)))


@dataclass(frozen=True)
class ImagePart:
    """An image inside a message (PRD section 9.2).

    Exactly one of ``data`` and ``url`` is set. ``data`` is the base64 payload
    exactly as supplied — the gateway passes it through and never decodes it.
    ``url`` is a remote URL the URL policy has already admitted; the gateway does
    not fetch it here, and the fetch, when it happens, is subject to the same
    address pinning, redirect and size controls as any upstream call (PRD
    sections 9.2 and 14.1). ``media_type`` is the image's media type, such as
    ``image/png``, and ``None`` means the provider must infer it.

    A part carrying both a payload and a URL, or neither, is refused: those are
    different requests and the gateway cannot guess which was meant.
    """

    data: str | None = None
    url: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        data = as_optional_text("image part 'data'", self.data)
        url = as_optional_text("image part 'url'", self.url)
        if (data is None) == (url is None):
            raise ValueError(
                "an image part must carry exactly one of 'data' (a base64 payload) or 'url' "
                "(an authorised remote URL); a part with neither, or with both, is not a "
                "representation the gateway can send"
            )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "url", url)
        object.__setattr__(
            self, "media_type", as_optional_text("image part 'media_type'", self.media_type)
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            _TYPE: _IMAGE,
            "data": self.data,
            "url": self.url,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "image part"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _IMAGE_PART_KEYS)
        return cls(
            data=as_optional_text(f"{where} 'data'", payload.get("data")),
            url=as_optional_text(f"{where} 'url'", payload.get("url")),
            media_type=as_optional_text(f"{where} 'media_type'", payload.get("media_type")),
        )


type ContentPart = TextPart | ImagePart


def content_part_from_dict(data: object) -> ContentPart:
    """Parse one content part, dispatching on its ``type`` discriminator.

    The discriminator is what makes a part unambiguous on the wire. A part whose
    type the gateway does not know cannot be represented and is refused rather
    than skipped, because a dropped part silently changes the message the model
    sees (PRD section 9.3).
    """
    where = "content part"
    payload = as_object(where, data)
    kind = payload.get(_TYPE)
    if kind == _TEXT:
        return TextPart.from_dict(payload)
    if kind == _IMAGE:
        return ImagePart.from_dict(payload)
    if not isinstance(kind, str):
        raise ValueError(
            f"{where} must carry a string {_TYPE!r} discriminator, got {describe(kind)}"
        )
    raise ValueError(
        f"{where} has type {kind!r}, which the canonical content representation does not "
        f"define; supported types are {_TEXT!r} and {_IMAGE!r} (PRD section 9.3)"
    )


@dataclass(frozen=True)
class Message:
    """One turn of a conversation (PRD sections 4.4 and 5.1).

    ``tool_calls`` belongs to an assistant message, the only role that proposes a
    call, and ``tool_call_id`` belongs to a tool message, the only role that
    answers one — and a tool message must carry it, because a result that cannot
    be matched to its call is not a result. Both rules are enforced here so that
    an inconsistent conversation fails at construction rather than upstream,
    where the provider's complaint would be less specific (PRD section 9.1).
    """

    role: Role
    content: tuple[ContentPart, ...] = ()
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        role = as_enumeration("message 'role'", self.role, Role)
        content = _as_content_parts("message 'content'", self.content)
        name = as_optional_text("message 'name'", self.name)
        tool_calls = as_tool_calls("message 'tool_calls'", self.tool_calls)
        tool_call_id = as_optional_text("message 'tool_call_id'", self.tool_call_id)

        if tool_calls and role is not Role.ASSISTANT:
            raise ValueError(
                f"a {role.value!r} message cannot carry tool calls; only an assistant message "
                f"proposes a call (PRD section 9.1)"
            )
        if tool_call_id is not None and role is not Role.TOOL:
            raise ValueError(
                f"a {role.value!r} message cannot carry a tool call id; only a tool message "
                f"answers a call (PRD section 9.1)"
            )
        if role is Role.TOOL and tool_call_id is None:
            raise ValueError(
                "a tool message must carry the 'tool_call_id' it answers; a result that cannot "
                "be matched to its call cannot be represented (PRD section 9.1)"
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


def as_image_parts(where: str, value: object) -> tuple[ImagePart, ...]:
    """Coerce a sequence of image parts, refusing anything else."""
    parts = as_sequence(where, value)
    for index, part in enumerate(parts):
        if not isinstance(part, ImagePart):
            raise ValueError(f"{where}[{index}] must be an ImagePart, got {describe(part)}")
    return parts


def _as_content_parts(where: str, value: object) -> tuple[ContentPart, ...]:
    parts = as_sequence(where, value)
    for index, part in enumerate(parts):
        if not isinstance(part, TextPart | ImagePart):
            raise ValueError(
                f"{where}[{index}] must be a TextPart or an ImagePart, got {describe(part)}"
            )
    return parts
