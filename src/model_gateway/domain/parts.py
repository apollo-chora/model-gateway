"""Canonical content parts (PRD sections 5.1, 9.1 and 9.4).

A content part is one element of an ordered sequence. Three canonical types are
such a sequence: a message is a role plus its parts, a tool result is the parts
the tool produced, and a model's output is the parts it emitted, in the order it
emitted them. Defining the parts once, here, is what lets those three share one
representation instead of three that drift apart — and it is why this module
depends on nothing but the JSON and parsing vocabulary.

An image arrives either as base64 data or as an authorised remote URL, and never
as anything else: the gateway does not fetch an image on its own authority, so a
URL that reaches this type is one the URL policy has already admitted.

Every part is a frozen dataclass holding immutable values only, so a sequence of
parts cannot change while it is being sent.
"""

from dataclasses import dataclass
from typing import Self

from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.serialization import (
    as_object,
    as_optional_text,
    as_sequence,
    as_text,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "CONTENT_PART_TYPES",
    "ContentPart",
    "ImagePart",
    "RefusalPart",
    "TextPart",
    "as_content_parts",
    "content_part_from_dict",
]

_TYPE = "type"
_TEXT = "text"
_IMAGE = "image"
_REFUSAL = "refusal"

_TEXT_PART_KEYS = frozenset({_TYPE, _TEXT})
_IMAGE_PART_KEYS = frozenset({_TYPE, "data", "url", "media_type"})

CONTENT_PART_TYPES: frozenset[str] = frozenset({_TEXT, _IMAGE, _REFUSAL})
"""The ``type`` discriminators the canonical content representation defines."""


@dataclass(frozen=True)
class TextPart:
    """A run of text inside a message (PRD section 9.4).

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
    """An image inside a message (PRD section 9.4).

    Exactly one of ``data`` and ``url`` is set. ``data`` is the base64 payload
    exactly as supplied — the gateway passes it through and never decodes it.
    ``url`` is a remote URL the URL policy has already admitted; the gateway does
    not fetch it here, and the fetch, when it happens, is subject to the same
    address pinning, redirect and size controls as any upstream call (PRD
    sections 9.4 and 14.1). ``media_type`` is the image's media type, such as
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


@dataclass(frozen=True)
class RefusalPart:
    """A provider's refusal to answer, kept apart from ordinary text (PRD 5.2, 9.1).

    A provider that declines on safety grounds says so, and a model that answers
    with text saying much the same thing is a different event: the first is the
    provider's own decision about the request, the second is the model's answer.
    Collapsing a refusal into text loses that distinction; dropping it loses the
    response. A refusal is therefore a content part of its own, carrying the
    provider's refusal text verbatim.

    A refusal is an output-side fact. No caller asks a model to refuse, so this
    part never expresses an instruction. It is nonetheless a ``ContentPart``, and
    so may appear in a message: an assistant refusal from an earlier turn is
    conversation history, and re-sending that history is how a caller continues a
    conversation whose last answer was a refusal. A refusal part inside a
    ``Role.USER`` message is not a request the gateway invents meaning for — it is
    forwarded as the text it is, exactly as the caller wrote it.

    Empty refusal text is allowed for the same reason empty text is: a provider
    may report a refusal with nothing in it, and refusing that would leave the
    gateway unable to represent a response it received.
    """

    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", as_text("refusal part 'text'", self.text))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {_TYPE: _REFUSAL, _TEXT: self.text}

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "refusal part"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _TEXT_PART_KEYS)
        return cls(text=as_text(f"{where} 'text'", payload.get(_TEXT)))


type ContentPart = TextPart | ImagePart | RefusalPart


def content_part_from_dict(data: object) -> ContentPart:
    """Parse one content part, dispatching on its ``type`` discriminator.

    The discriminator is what makes a part unambiguous on the wire. A part whose
    type the gateway does not know cannot be represented and is refused rather
    than skipped, because a dropped part silently changes the message the model
    sees (PRD section 9.1).
    """
    where = "content part"
    payload = as_object(where, data)
    kind = payload.get(_TYPE)
    if kind == _TEXT:
        return TextPart.from_dict(payload)
    if kind == _IMAGE:
        return ImagePart.from_dict(payload)
    if kind == _REFUSAL:
        return RefusalPart.from_dict(payload)
    if not isinstance(kind, str):
        raise ValueError(
            f"{where} must carry a string {_TYPE!r} discriminator, got {describe(kind)}"
        )
    supported = ", ".join(repr(item) for item in sorted(CONTENT_PART_TYPES))
    raise ValueError(
        f"{where} has type {kind!r}, which the canonical content representation does not "
        f"define; supported types are {supported} (PRD section 9.1)"
    )


def as_content_parts(where: str, value: object) -> tuple[ContentPart, ...]:
    """Coerce a sequence of content parts, refusing anything else."""
    parts = as_sequence(where, value)
    for index, part in enumerate(parts):
        if not isinstance(part, TextPart | ImagePart | RefusalPart):
            raise ValueError(
                f"{where}[{index}] must be a TextPart, an ImagePart or a RefusalPart, got "
                f"{describe(part)}"
            )
    return parts
