"""The image endpoints: ``POST /v1/images/generations`` and
``POST /v1/images/edits`` (PRD section 4.6).

The handlers are thin on purpose, exactly as the other endpoints' are. They
decode the body, translate it into the canonical image request, call the
inference service and render the canonical response back into the OpenAI
shape. They resolve no route, retry nothing, count nothing and check nothing:
routing, fallback, retry, the deadline, the image count and the editing
capability are the gateway core's (PRD sections 4.6, 9.2 and 9.5), and a
second copy of that logic on the HTTP path is how the two transports drift
apart.

**The accepted request is the documented subset, and nothing else is
ignored.** The generations body is the canonical request's own fields —
``model``, ``prompt``, ``size``, ``response_format``, ``n`` and ``metadata``
— and the edits body adds the image input the operation is defined by. A key
the gateway does not understand is refused with an invalid-request error
naming it rather than dropped, which is the compatibility boundary of PRD
section 4: a client SDK that serialises a new default parameter receives an
error it can act on instead of an assurance that a request it did not make
was honoured. ``size``, ``response_format`` and ``n`` are the canonical
request's own values, so an undocumented one is refused by the canonical
type rather than silently substituted (PRD section 9.1).

**The image input is a base64 payload or a data URL, validated where it
arrives.** ``POST /v1/images/edits`` accepts the caller's image as a base64
payload, as a ``data:<mediatype>;base64,<payload>`` URL, or as an array of
those — the representation PRD section 4.6 asks for in place of multipart form
data, and the one this endpoint documents. A payload that is not decodable
base64, or an edit that carries no image at all, is refused naming ``image``:
a gateway that accepted a corrupt image and failed upstream, or worse edited
something else, has not served the caller.

**Editing is enforced from the route's declared capabilities, and refused
for a reason the gateway owns.** The edits handler carries the request to the
inference service, which asks the route to declare ``generation`` *and*
``editing`` — the second being the operator's own statement
(``IMAGE_GEN_CAPABILITIES``; PRD section 6.2). A route that declares
``generation`` alone is refused with a capability error naming ``editing``
(PRD section 4.6). A route that declares both is refused too, because the
canonical image request carries no field for an image input and the gateway
cannot forward the caller's image to any route; the refusal is the same
capability error. The validated representation is what lets a caller tell a
refused edit from a silently dropped image.

**The response is the documented shape.** ``created`` is when the gateway
assembled the answer and each entry of ``data`` carries the payload it was
given — ``b64_json`` or ``url``, never both — beside ``revised_prompt`` when
the upstream reported one. The gateway's own accounting — the invocation
identifier, the upstream model, the attempt count — is deliberately not in
this payload: the OpenAI image shape has no field for it (PRD sections 5.2
and 13).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from blacksheep import HTTPException, Request, Response
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.domain.images import GeneratedImage, ImageRequest, ImageResponse
from model_gateway.domain.json_value import FrozenJsonObject
from model_gateway.domain.serialization import (
    as_json_object,
    as_object,
    as_optional_int,
    as_optional_text,
    as_required_text,
    reject_unknown_keys,
)
from model_gateway.errors import InvalidArgumentError

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService

__all__ = [
    "ImageGenerationResponse",
    "create_image_edits",
    "create_image_generations",
    "image_edit_request_from_payload",
    "image_generation_request_from_payload",
    "register_image_routes",
]


ImagesHandler = Callable[[Request], Awaitable[Response]]

_GENERATION_REQUEST_KEYS = frozenset(
    {"model", "prompt", "size", "response_format", "n", "metadata"}
)
"""The request fields ``POST /v1/images/generations`` accepts.

The set is closed on purpose, exactly as the other endpoints': PRD section 4
makes the gateway's OpenAI compatibility a supported subset rather than
drop-in compatibility, so a parameter the gateway does not understand is
refused rather than ignored.
"""

_EDIT_REQUEST_KEYS = frozenset(
    {"model", "prompt", "image", "size", "response_format", "n", "metadata"}
)
"""The request fields ``POST /v1/images/edits`` accepts: the generation set plus ``image``.

``image`` is the edit's own input, in the base64 representation this endpoint
documents (PRD section 4.6); everything else is the generations request's
field, because an edit is an image invocation with an image to start from.
"""

_DATA_URL = "data:"
_BASE64_MARKER = ";base64,"


@dataclass(frozen=True)
class ImageGenerationResponse:
    """An OpenAI image generations response (PRD section 4.6).

    ``created`` is when the gateway assembled the answer and ``data`` the
    images the upstream produced, each carrying the payload it was given and
    the revised prompt the upstream reported beside it. Both are read from the
    canonical response; nothing here is computed, and nothing the documented
    shape does not define is added to it.
    """

    created: int
    data: tuple[GeneratedImage, ...]

    @classmethod
    def from_image_response(cls, response: ImageResponse) -> Self:
        """Translate a canonical image response into the OpenAI shape."""
        return cls(created=response.created, data=response.data)

    def to_dict(self) -> dict[str, object]:
        return {
            "created": self.created,
            "data": [_image_payload(image) for image in self.data],
        }


def image_generation_request_from_payload(payload: object) -> ImageRequest:
    """Parse a decoded JSON body into the canonical image request.

    The parse is the gateway's, not the wire format's: by the time a caller
    has a canonical request, every field has a canonical home and every rule
    the canonical type enforces has already been applied — the documented
    ``WxH`` size, the two documented response encodings, a positive ``n``
    — so an undocumented value is refused here rather than discovered
    halfway through an invocation.

    A ``ValueError`` from the canonical types or from the serialisation
    helpers is a validation failure of this request and becomes an
    invalid-request error carrying the domain's own gateway-authored message,
    which names the field that failed. An unknown key is refused the same
    way: the boundary does not ignore a parameter it does not understand
    (PRD section 4).
    """
    where = "the request body"
    try:
        body = as_object(where, payload)
        reject_unknown_keys(where, body, _GENERATION_REQUEST_KEYS)
        return ImageRequest(
            model=as_required_text(f"{where} 'model'", body.get("model")),
            prompt=as_required_text(f"{where} 'prompt'", body.get("prompt")),
            size=as_optional_text(f"{where} 'size'", body.get("size")),
            response_format=as_optional_text(
                f"{where} 'response_format'", body.get("response_format")
            ),
            n=as_optional_int(f"{where} 'n'", body.get("n"), minimum=1),
            metadata=_as_metadata(f"{where} 'metadata'", body.get("metadata")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid image request: {error}") from error


def image_edit_request_from_payload(payload: object) -> ImageRequest:
    """Parse a decoded JSON body into the canonical image request for an edit.

    The same parse as :func:`image_generation_request_from_payload`, plus the
    edit's own field: ``image``, the caller's image in the documented base64
    representation. The image has no home in the canonical image request and
    the inference service refuses the edit for that reason; what this boundary
    owes the caller is that the representation is real — an absent or corrupt
    image is refused here, naming the field, rather than discovered upstream
    or answered with an edit of something else (PRD sections 4.6 and 9.5).
    """
    where = "the request body"
    try:
        body = as_object(where, payload)
        reject_unknown_keys(where, body, _EDIT_REQUEST_KEYS)
        _reject_unparseable_image(f"{where} 'image'", body.get("image"))
        return ImageRequest(
            model=as_required_text(f"{where} 'model'", body.get("model")),
            prompt=as_required_text(f"{where} 'prompt'", body.get("prompt")),
            size=as_optional_text(f"{where} 'size'", body.get("size")),
            response_format=as_optional_text(
                f"{where} 'response_format'", body.get("response_format")
            ),
            n=as_optional_int(f"{where} 'n'", body.get("n"), minimum=1),
            metadata=_as_metadata(f"{where} 'metadata'", body.get("metadata")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid image edit request: {error}") from error


def create_image_generations(service: InferenceService) -> ImagesHandler:
    """Build the handler for ``POST /v1/images/generations`` (PRD section 4.6)."""

    async def image_generations(request: Request) -> Response:
        image_request = image_generation_request_from_payload(await _read_json_body(request))
        response = await service.generate(image_request)
        return json(ImageGenerationResponse.from_image_response(response).to_dict())

    return image_generations


def create_image_edits(service: InferenceService) -> ImagesHandler:
    """Build the handler for ``POST /v1/images/edits`` (PRD section 4.6)."""

    async def image_edits(request: Request) -> Response:
        image_request = image_edit_request_from_payload(await _read_json_body(request))
        response = await service.edit(image_request)
        return json(ImageGenerationResponse.from_image_response(response).to_dict())

    return image_edits


def register_image_routes(router: Router, service: InferenceService) -> None:
    """Add the image endpoints to ``router``."""
    router.add_post("/v1/images/generations", create_image_generations(service))
    router.add_post("/v1/images/edits", create_image_edits(service))


async def _read_json_body(request: Request) -> object:
    """Decode the request body, refusing anything that is not a JSON document.

    BlackSheep raises its own ``BadRequestFormat`` when the declared content
    type is JSON and the body is not, and returns ``None`` when the content
    type never declared JSON at all. Both are malformed input for this
    endpoint, and both become an invalid-request error carrying the
    taxonomy's reason rather than the framework's plain-text body, so every
    failure on this path answers in one shape (PRD section 5.4).
    """
    try:
        payload = await request.json()
    except HTTPException as error:
        raise InvalidArgumentError(
            "the request body must be valid JSON and the Content-Type header must declare it"
        ) from error
    if payload is None:
        raise InvalidArgumentError(
            "the request body must be a JSON object sent with Content-Type: application/json"
        )
    return payload


def _reject_unparseable_image(where: str, value: object) -> None:
    """Refuse an edit whose image input is absent or not the documented form.

    The image is accepted as a base64 payload, as a ``data:`` URL carrying
    one, or as an array of those; anything else — no image at all, a
    non-string, a payload that does not decode — is refused rather than
    carried to a provider that would edit something else or fail on a
    corrupt body (PRD section 4.6).
    """
    if value is None:
        raise ValueError(
            f"{where} must be a base64 image payload, a base64 data URL, or an array of those; "
            f"an edit with no image is not an edit"
        )
    _as_image_input(where, value)


def _as_image_input(where: str, value: object) -> tuple[str, ...]:
    """The caller's image input as validated strings, in the order given.

    A single image is promoted to a one-element tuple here rather than
    refused, because on this wire form the two are unambiguous: the endpoint
    knows the field's documented shape, and a caller that sent one image
    asked for exactly one image to be edited.
    """
    if isinstance(value, list):
        return tuple(
            _as_image_item(f"{where}[{index}]", item) for index, item in enumerate(value)
        )
    return (_as_image_item(where, value),)


def _as_image_item(where: str, value: object) -> str:
    """One image as a base64 payload or a ``data:`` URL, or a refusal."""
    text = as_required_text(where, value)
    payload = text.split(_BASE64_MARKER, 1)[1] if text.startswith(_DATA_URL) else text
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(
            f"{where} must be a base64 image payload or a base64 data URL, got {text!r}"
        ) from None
    return text


def _as_metadata(where: str, value: object) -> FrozenJsonObject | None:
    """Non-sensitive request metadata as an immutable object, or ``None`` when absent.

    ``None`` is kept as ``None`` rather than folded into an empty object, so a
    request that carried no metadata does not acquire an empty one. The value
    is never forwarded upstream: PRD section 5.1 makes request metadata caller
    bookkeeping, and the binding sends none of it.
    """
    if value is None:
        return None
    return as_json_object(where, value)


def _image_payload(image: GeneratedImage) -> dict[str, object]:
    """One image in the OpenAI shape: the payload it carries, and the revised prompt.

    The canonical type guarantees exactly one of ``b64_json`` and ``url`` is
    set, so the entry carries the one the upstream gave and never both.
    ``revised_prompt`` is the prompt the provider reports having actually
    used; it is reported beside the caller's prompt rather than instead of
    it, and omitted when the upstream did not revise anything, because the
    documented shape leaves it optional.
    """
    entry: dict[str, object] = {}
    if image.b64_json is not None:
        entry["b64_json"] = image.b64_json
    if image.url is not None:
        entry["url"] = image.url
    if image.revised_prompt is not None:
        entry["revised_prompt"] = image.revised_prompt
    return entry
