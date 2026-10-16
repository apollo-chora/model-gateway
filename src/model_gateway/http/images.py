"""The image endpoints: ``POST /v1/images/generations``.

The handlers are thin on purpose, exactly as the other endpoints' are. They
decode the body, translate it into the canonical image request, call the
inference service and render the canonical response back into the OpenAI
shape. They resolve no route, retry nothing, count nothing and check nothing:
routing, fallback, retry, the deadline and the image count are the gateway
core's, and a second copy of that logic on the HTTP path is how the two
transports drift apart.

**The accepted request is the documented subset, and nothing else is
ignored.** The generations body is the canonical request's own fields —
``model``, ``prompt``, ``size``, ``response_format``, ``n`` and ``metadata``
— and a key the gateway does not understand is refused with an
invalid-request error naming it rather than dropped — the gateway's
compatibility boundary: a client SDK that serialises a new
default parameter receives an error it can act on instead of an assurance
that a request it did not make was honoured. ``size``, ``response_format``
and ``n`` are the canonical request's own values, so an undocumented one is
refused by the canonical type rather than silently substituted.

**Image editing is absent, not advertised-and-refused.** The gateway's
canonical image request carries no field for an image input, so an edit
cannot be forwarded to any route however its capabilities read, and an edit
answered without it would report an image edited that never left the gateway.
The gateway therefore registers no edits endpoint at all:
an operation it categorically cannot perform is left out of the surface
rather than published and guaranteed to fail, and a configured route that
declares the ``editing`` capability is refused at startup by the application
factory with a message saying editing is not implemented.

**The response is the documented shape.** ``created`` is when the gateway
assembled the answer and each entry of ``data`` carries the payload it was
given — ``b64_json`` or ``url``, never both — beside ``revised_prompt`` when
the upstream reported one. The gateway's own accounting — the invocation
identifier, the upstream model, the attempt count — is deliberately not in
this payload: the OpenAI image shape has no field for it.
"""

from __future__ import annotations

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
from model_gateway.http.auth import get_identity
from model_gateway.security.identity import caller_namespace, caller_tenant

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService

__all__ = [
    "ImageGenerationResponse",
    "create_image_generations",
    "image_generation_request_from_payload",
    "register_image_routes",
]


ImagesHandler = Callable[[Request], Awaitable[Response]]

_GENERATION_REQUEST_KEYS = frozenset(
    {"model", "prompt", "size", "response_format", "n", "metadata"}
)
"""The request fields ``POST /v1/images/generations`` accepts.

The set is closed on purpose, exactly as the other endpoints': the gateway's
OpenAI compatibility is a supported subset rather than drop-in compatibility,
so a parameter the gateway does not understand is refused rather than
ignored.
"""


@dataclass(frozen=True)
class ImageGenerationResponse:
    """An OpenAI image generations response.

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
    way: the boundary does not ignore a parameter it does not understand.
    """
    where = "the request body"
    try:
        body = as_object(where, payload)
        reject_unknown_keys(where, body, _GENERATION_REQUEST_KEYS)
        return ImageRequest(
            model=as_optional_text(f"{where} 'model'", body.get("model")),
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


def create_image_generations(service: InferenceService) -> ImagesHandler:
    """Build the handler for ``POST /v1/images/generations``."""

    async def image_generations(request: Request) -> Response:
        image_request = image_generation_request_from_payload(await _read_json_body(request))
        identity = get_identity(request)
        response = await service.generate(
            image_request,
            namespace=caller_namespace(identity),
            tenant=caller_tenant(identity),
        )
        return json(ImageGenerationResponse.from_image_response(response).to_dict())

    return image_generations


def register_image_routes(router: Router, service: InferenceService) -> None:
    """Add the image endpoints to ``router``."""
    router.add_post("/v1/images/generations", create_image_generations(service))


async def _read_json_body(request: Request) -> object:
    """Decode the request body, refusing anything that is not a JSON document.

    BlackSheep raises its own ``BadRequestFormat`` when the declared content
    type is JSON and the body is not, and returns ``None`` when the content
    type never declared JSON at all. Both are malformed input for this
    endpoint, and both become an invalid-request error carrying the
    taxonomy's reason rather than the framework's plain-text body, so every
    failure on this path answers in one shape.
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


def _as_metadata(where: str, value: object) -> FrozenJsonObject | None:
    """Non-sensitive request metadata as an immutable object, or ``None`` when absent.

    ``None`` is kept as ``None`` rather than folded into an empty object, so a
    request that carried no metadata does not acquire an empty one. The value
    is never forwarded upstream: request metadata is caller bookkeeping, and the
    binding sends none of it.
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
