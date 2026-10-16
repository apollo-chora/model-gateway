"""Canonical image generation request and response.

Image generation is the second of the gateway's three operations, and its
canonical types follow the same rules as the other two: the request names a
public model ID and carries the caller's parameters, the response reports the
public ID beside the upstream model that served it, and both round-trip through
``to_dict`` and ``from_dict`` with every field always emitted. Nothing here
names a vendor or a wire format — translating these types
into a provider's image call is the LiteLLM binding's work.

**A generated image carries exactly one payload.** ``GeneratedImage.b64_json``
is the image itself and ``GeneratedImage.url`` is where the caller may fetch it,
and the two are alternatives rather than companions: the OpenAI-compatible shape
this type mirrors puts one or the other in a result entry, and a caller reads
whichever is present. An entry carrying both would leave that caller choosing
between two answers the gateway never said were the same image, and an entry
carrying neither would hand it a result with no image in it at all. Both states
are refused, so the caller's rule — read the one that is set — is always well
defined.

**A generated image's URL is untrusted provider output.** ``GeneratedImage.url``
is carried through from the upstream's answer unchanged, so it survives this
type's parsing because it is a non-empty string, not because the gateway screened
it: nothing here resolves it, applies the URL policy to it, pins a connection to
it or follows it. A caller fetches it on its own authority and under its own
policy; the gateway must never treat it as a gateway-fetchable trusted URL. It is
the mirror image of the remote image *input* the gateway refuses:
an inbound ``image_url`` is rejected because the gateway cannot screen and
fetch it safely, and an outbound ``url`` is not thereby safe to fetch either —
neither has passed the URL policy.

**Only the documented parameter values are accepted.** ``size`` is the
documented ``WxH`` form and ``response_format`` one of the two wire encodings
the OpenAI-compatible image API defines, ``url`` and ``b64_json``.
A value outside either set is refused rather than dropped or defaulted,
for the same reason the embeddings request refuses an undocumented encoding: a
caller told its request was honoured while the gateway quietly substituted
another size or another representation cannot tell the two apart.
``n`` is positive when stated, because a request for zero images is not a
request any provider could honour, and ``prompt`` must carry something, because
an image generation request with nothing to generate from has no canonical
representation.

**What a route can do is the route's business, not this type's.** Whether a
model supports generation alone or generation and editing is the ``generation``
and ``editing`` capabilities its primary route declares, and
that check belongs to the inference service and the model registry rather than
to the canonical request. ``metadata`` is the caller's bookkeeping — routing
hints, correlation and provenance attributes — held as an immutable JSON object
and never forwarded to the model or the provider.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from model_gateway.domain.generation import Usage, as_usage
from model_gateway.domain.json_value import FrozenJsonObject, JsonDocument, thaw_json
from model_gateway.domain.serialization import (
    as_int,
    as_json_object,
    as_object,
    as_optional_int,
    as_optional_text,
    as_required_text,
    as_sequence,
    describe,
    reject_unknown_keys,
)

__all__ = [
    "IMAGE_RESPONSE_FORMATS",
    "GeneratedImage",
    "ImageRequest",
    "ImageResponse",
    "as_images",
]

_IMAGE_KEYS = frozenset({"b64_json", "url", "revised_prompt"})
_REQUEST_KEYS = frozenset({"model", "prompt", "size", "response_format", "n", "metadata"})
_RESPONSE_KEYS = frozenset(
    {"model", "upstream_model", "created", "data", "usage", "invocation_id"}
)

IMAGE_RESPONSE_FORMATS: frozenset[str] = frozenset({"url", "b64_json"})
"""The wire representations an image request may ask for.

The gateway's image endpoint is OpenAI-compatible, and that API documents exactly
two: ``url``, a location the caller fetches the image from — untrusted provider
output the gateway neither screens nor fetches, as :class:`GeneratedImage`
documents — and ``b64_json``, the image itself base64-encoded. A request naming
anything else is refused rather than silently answered in the gateway's own
default, because a caller told its request was honoured while the gateway ignored
the representation it asked for cannot tell the two apart.
"""

_SIZE_PATTERN = re.compile(r"(\d+)x(\d+)")


@dataclass(frozen=True)
class GeneratedImage:
    """One generated image, carried as its payload or as its location.

    ``b64_json`` is the image itself, base64-encoded, and ``url`` is where the
    caller may fetch it. Exactly one of the two is set: they are alternatives in
    the shape this type mirrors, so a caller reads whichever is present, and an
    entry carrying both or neither leaves that rule undefined. The refusal of
    both states is what keeps the caller's reading of a result entry — and of
    the serialised form, which emits both keys always — unambiguous.

    ``url`` is untrusted provider output, not a URL the gateway vouches for. It
    arrives in the upstream's answer and is carried through unchanged, so it
    survives this type's parsing because it is a non-empty string rather than
    because anything checked it: no resolver, no URL-policy decision, no pinned
    connection and no redirect evaluation stands behind it. A caller may fetch it
    on its own authority and under its own policy; the gateway must not treat it
    as a gateway-fetchable trusted URL, and must not hand it to an operation that
    would fetch it on the gateway's behalf. This is the mirror image of the
    remote image *input* the gateway refuses for exactly that reason:
    an inbound ``image_url`` is rejected because the gateway cannot screen
    and fetch it safely, and an outbound ``url`` is not thereby safe to fetch
    either.

    ``revised_prompt`` is the prompt the provider reports having actually used,
    when it rewrites the caller's prompt to describe the image it produced. It
    is reported beside the original rather than instead of it, so a caller can
    see what it asked for and what the provider did with it.
    """

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __post_init__(self) -> None:
        b64_json = as_optional_text("generated image 'b64_json'", self.b64_json)
        url = as_optional_text("generated image 'url'", self.url)
        if b64_json is None and url is None:
            raise ValueError(
                "a generated image must carry either a 'b64_json' payload or a 'url'; this one "
                "carries neither, so it is a result entry with no image in it and a caller "
                "cannot tell which of the two to read"
            )
        if b64_json is not None and url is not None:
            raise ValueError(
                "a generated image must carry either a 'b64_json' payload or a 'url', not both; "
                "the two are alternatives in the shape this type mirrors, so a caller reading "
                "one of them would be choosing between two answers the gateway never said were "
                "the same image"
            )

        object.__setattr__(self, "b64_json", b64_json)
        object.__setattr__(self, "url", url)
        object.__setattr__(
            self,
            "revised_prompt",
            as_optional_text("generated image 'revised_prompt'", self.revised_prompt),
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "b64_json": self.b64_json,
            "url": self.url,
            "revised_prompt": self.revised_prompt,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "generated image"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _IMAGE_KEYS)

        return cls(
            b64_json=as_optional_text(f"{where} 'b64_json'", payload.get("b64_json")),
            url=as_optional_text(f"{where} 'url'", payload.get("url")),
            revised_prompt=as_optional_text(
                f"{where} 'revised_prompt'", payload.get("revised_prompt")
            ),
        )


@dataclass(frozen=True)
class ImageRequest:
    """One canonical image generation invocation.

    ``model`` is the public model ID the caller named, or ``None`` when it named
    none. It is never an upstream model name, and it is not what selects the
    routes: the operation is resolved from the transport the call arrived on,
    and the routes configured for the image operation serve it. A stated model
    is carried for the caller's own correlation and is ignored for routing, so
    a caller-supplied model name cannot reach an unconfigured endpoint.

    ``prompt`` is what the image is generated from and must carry something: an
    image request with nothing to generate from has no canonical representation.
    The prompt is kept verbatim — the gateway neither trims nor rewrites the
    text a provider will be asked to illustrate.

    ``size`` is the documented ``WxH`` form (for example ``1024x1024``) and
    ``response_format`` one of :data:`IMAGE_RESPONSE_FORMATS`; each is refused
    when set to anything else rather than dropped or defaulted. ``n`` is how
    many images the caller wants and is positive when set, because a request for
    zero images is not one any provider could honour; ``None`` means the caller
    stated no count and the provider's own default applies, which is not the
    same request as ``n=1``.

    ``metadata`` is non-sensitive request metadata held as an immutable JSON
    object, never forwarded to the model or the provider.
    """

    prompt: str
    size: str | None = None
    response_format: str | None = None
    n: int | None = None
    metadata: Mapping[str, object] | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", as_optional_text("model", self.model))
        object.__setattr__(self, "prompt", as_required_text("prompt", self.prompt))
        object.__setattr__(self, "size", _as_size("size", self.size))
        object.__setattr__(
            self,
            "response_format",
            _as_response_format("response_format", self.response_format),
        )
        object.__setattr__(self, "n", as_optional_int("n", self.n, minimum=1))
        object.__setattr__(self, "metadata", _as_metadata("metadata", self.metadata))

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "size": self.size,
            "response_format": self.response_format,
            "n": self.n,
            "metadata": None if self.metadata is None else thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "image request"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _REQUEST_KEYS)

        return cls(
            model=as_optional_text(f"{where} 'model'", payload.get("model")),
            prompt=as_required_text(f"{where} 'prompt'", payload.get("prompt")),
            size=_as_size(f"{where} 'size'", payload.get("size")),
            response_format=_as_response_format(
                f"{where} 'response_format'", payload.get("response_format")
            ),
            n=as_optional_int(f"{where} 'n'", payload.get("n"), minimum=1),
            metadata=_as_metadata(f"{where} 'metadata'", payload.get("metadata")),
        )


@dataclass(frozen=True)
class ImageResponse:
    """The canonical result of one image generation invocation.

    ``model`` is the public model ID of the image operation, stated by the
    gateway rather than by the caller, and ``upstream_model`` the route's own
    model ID, reported separately because a fallback may have served it.
    ``created`` is when the gateway assembled the
    answer and is non-negative, because a negative timestamp is not a time any
    answer was produced at.

    ``data`` holds the generated images and must carry at least one: a response
    reporting that an image was generated and returning none of them is not an
    answer to the request that was made. ``usage`` is the accounting of the call,
    and ``invocation_id`` the identifier of the invocation
    this response belongs to, so a caller can reconcile the response with the
    request it sent.
    """

    model: str
    upstream_model: str
    created: int
    data: tuple[GeneratedImage, ...]
    usage: Usage
    invocation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", as_required_text("model", self.model))
        object.__setattr__(
            self, "upstream_model", as_required_text("upstream_model", self.upstream_model)
        )
        object.__setattr__(self, "created", as_int("created", self.created, minimum=0))
        object.__setattr__(self, "data", _as_data("data", self.data))
        object.__setattr__(self, "usage", as_usage("usage", self.usage))
        object.__setattr__(
            self, "invocation_id", as_required_text("invocation_id", self.invocation_id)
        )

    def to_dict(self) -> dict[str, JsonDocument]:
        return {
            "model": self.model,
            "upstream_model": self.upstream_model,
            "created": self.created,
            "data": [image.to_dict() for image in self.data],
            "usage": self.usage.to_dict(),
            "invocation_id": self.invocation_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "image response"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESPONSE_KEYS)

        return cls(
            model=as_required_text(f"{where} 'model'", payload.get("model")),
            upstream_model=as_required_text(
                f"{where} 'upstream_model'", payload.get("upstream_model")
            ),
            created=as_int(f"{where} 'created'", payload.get("created"), minimum=0),
            data=tuple(
                GeneratedImage.from_dict(item)
                for item in as_sequence(f"{where} 'data'", payload.get("data"))
            ),
            usage=Usage.from_dict(payload.get("usage")),
            invocation_id=as_required_text(
                f"{where} 'invocation_id'", payload.get("invocation_id")
            ),
        )


def as_images(where: str, value: object) -> tuple[GeneratedImage, ...]:
    """Coerce a sequence of generated images, refusing anything else."""
    items = as_sequence(where, value)
    for index, item in enumerate(items):
        if not isinstance(item, GeneratedImage):
            raise ValueError(
                f"{where}[{index}] must be a GeneratedImage, got {describe(item)}"
            )
    return items


def _as_data(where: str, value: object) -> tuple[GeneratedImage, ...]:
    """A non-empty tuple of generated images."""
    items = as_images(where, value)
    if not items:
        raise ValueError(
            f"{where} must carry at least one generated image; a response reporting that an "
            f"image was generated and returning none of them is not an answer to the request "
            f"that was made"
        )
    return items


def _as_size(where: str, value: object) -> str | None:
    """The documented ``WxH`` size, or ``None`` when the caller stated none."""
    size = as_optional_text(where, value)
    if size is None:
        return None
    match = _SIZE_PATTERN.fullmatch(size)
    if match is None or int(match.group(1)) < 1 or int(match.group(2)) < 1:
        raise ValueError(
            f"{where} must be the documented 'WxH' form, for example '1024x1024'; got {size!r}. "
            f"A size outside that form is refused rather than dropped, because a caller told "
            f"its request was honoured while the gateway substituted another size cannot tell "
            f"the two apart"
        )
    return size


def _as_response_format(where: str, value: object) -> str | None:
    """One of the documented wire representations, or ``None`` when none was stated."""
    response_format = as_optional_text(where, value)
    if response_format is None:
        return None
    if response_format not in IMAGE_RESPONSE_FORMATS:
        supported = ", ".join(repr(item) for item in sorted(IMAGE_RESPONSE_FORMATS))
        raise ValueError(
            f"{where} must be one of {supported}; got {response_format!r}. A representation "
            f"outside the documented set is refused rather than answered in the gateway's own "
            f"default, because a caller told its request was honoured while the gateway ignored "
            f"the representation it asked for cannot tell the two apart"
        )
    return response_format


def _as_metadata(where: str, value: object) -> FrozenJsonObject | None:
    """Arbitrary JSON as an immutable object; ``None`` when the caller supplied none.

    ``None`` is kept as ``None`` rather than folded into an empty object, so a
    request that carried no metadata round-trips to one that still carries none.
    """
    if value is None:
        return None
    return as_json_object(where, value)
