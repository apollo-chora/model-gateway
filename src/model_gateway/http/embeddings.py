"""The embeddings endpoint: ``POST /v1/embeddings`` (PRD section 4.7).

The handler is thin on purpose, exactly as the chat endpoint's is. It decodes the
body, translates it into the canonical embeddings request, calls the inference
service and renders the canonical response back into the OpenAI shape. It
resolves no route, retries nothing, counts nothing and orders nothing: routing,
fallback, retry, the deadline, the input ordering and the dimensionality check
are the gateway core's (PRD sections 4.7, 9.2 and 11.3), and a second copy of
that logic on the HTTP path is how the two transports drift apart.

**The accepted request is the documented subset, and nothing else is ignored.**
The body is the canonical request's own fields — ``model``, ``input``,
``dimensions``, ``encoding_format``, ``user`` and ``metadata`` — so a parameter
the gateway does not understand is refused with an invalid-request error naming
it rather than dropped, which is the compatibility boundary of PRD section 4.
``input`` is the one field whose wire form is wider than the canonical one: the
OpenAI format accepts a single string or an array of strings, and both are
normalised to the tuple of texts the canonical request holds, in the caller's
order. An array is what the canonical request always carries, because ordering is
part of the contract and a bare string would be an ambiguous one-element array.

**The response is the documented shape.** ``object`` is ``list``, ``model`` is the
public model ID the caller asked for, each entry of ``data`` carries
``object: "embedding"``, its ``index`` and its ``embedding``, and ``usage``
carries ``prompt_tokens`` and ``total_tokens``. The gateway's own accounting —
the invocation identifier, the upstream model, the attempt count — is
deliberately not in this payload: the OpenAI embeddings shape has no field for
it, and adding one would be a contract the gateway invented. That accounting
belongs to the gRPC contract and to usage records (PRD sections 5.2 and 13).

**The vectors are the float form.** The endpoint renders the canonical vectors,
which are numbers, so a request asking for another wire encoding is refused by
the gateway core with a capability error naming it rather than answered in floats
(PRD section 9.5). The endpoint itself decides nothing about that: it translates,
and the policy lives where every other transport can reach it.

The invocation identifier is minted by the inference service for this operation,
because the canonical embeddings request carries no identifier of its own. It
reaches the response's own canonical form, and the caller-facing payload omits it
only because the documented shape has no field for it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from blacksheep import HTTPException, Request, Response
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.domain.embeddings import Embedding, EmbeddingRequest, EmbeddingResponse
from model_gateway.domain.generation import Usage
from model_gateway.domain.json_value import FrozenJsonObject
from model_gateway.domain.serialization import (
    as_json_object,
    as_object,
    as_optional_int,
    as_optional_text,
    as_required_text,
    as_text,
    describe,
    reject_unknown_keys,
)
from model_gateway.errors import InvalidArgumentError
from model_gateway.http.auth import get_identity
from model_gateway.security.identity import caller_namespace

if TYPE_CHECKING:
    from model_gateway.inference import InferenceService

__all__ = [
    "EmbeddingsResponse",
    "create_embeddings",
    "embedding_request_from_payload",
    "register_embeddings_routes",
]

EmbeddingsHandler = Callable[[Request], Awaitable[Response]]

_OBJECT_LIST = "list"
_OBJECT_EMBEDDING = "embedding"

_REQUEST_KEYS = frozenset(
    {"model", "input", "dimensions", "encoding_format", "user", "metadata"}
)
"""The request fields this endpoint accepts, which are the canonical request's own.

The set is closed on purpose: PRD section 4 makes the gateway's OpenAI
compatibility a supported subset rather than drop-in compatibility, so a
parameter the gateway does not understand is refused rather than ignored. A
client SDK that serialises a new default receives an error it can act on instead
of an assurance that a request it did not make was honoured.
"""


@dataclass(frozen=True)
class EmbeddingsResponse:
    """An OpenAI embeddings response (PRD section 4.7).

    ``data`` is the vectors in the caller's input order, each carrying the
    position it was placed at, and ``usage`` is the token accounting of the call.
    Both are read from the canonical response; nothing here is computed, and
    nothing the documented shape does not define is added to it.
    """

    model: str
    data: tuple[Embedding, ...]
    usage: Usage

    @classmethod
    def from_embedding_response(cls, response: EmbeddingResponse) -> Self:
        """Translate a canonical embeddings response into the OpenAI shape."""
        return cls(model=response.model, data=response.data, usage=response.usage)

    def to_dict(self) -> dict[str, object]:
        return {
            "object": _OBJECT_LIST,
            "model": self.model,
            "data": [_embedding_payload(embedding) for embedding in self.data],
            "usage": _usage_payload(self.usage),
        }


def embedding_request_from_payload(payload: object) -> EmbeddingRequest:
    """Parse a decoded JSON body into the canonical embeddings request.

    The parse is the gateway's, not the wire format's: by the time a caller has a
    canonical request, every field has a canonical home and every rule the
    canonical type enforces has already been applied. Two of those rules matter
    before any provider is called — an input with nothing to embed, and a blank
    entry that would shift every later index — so they are enforced here rather
    than discovered halfway through an invocation.

    A ``ValueError`` from the canonical types or from the serialisation helpers
    is a validation failure of this request and becomes an invalid-request error
    carrying the domain's own gateway-authored message, which names the field
    that failed. An unknown key is refused the same way: the boundary does not
    ignore a parameter it does not understand (PRD section 4).
    """
    where = "the request body"
    try:
        body = as_object(where, payload)
        reject_unknown_keys(where, body, _REQUEST_KEYS)
        return EmbeddingRequest(
            model=as_required_text(f"{where} 'model'", body.get("model")),
            input=_as_input(f"{where} 'input'", body.get("input")),
            dimensions=as_optional_int(
                f"{where} 'dimensions'", body.get("dimensions"), minimum=1
            ),
            encoding_format=as_optional_text(
                f"{where} 'encoding_format'", body.get("encoding_format")
            ),
            user=as_optional_text(f"{where} 'user'", body.get("user")),
            metadata=_as_metadata(f"{where} 'metadata'", body.get("metadata")),
        )
    except ValueError as error:
        raise InvalidArgumentError(f"invalid embeddings request: {error}") from error


def create_embeddings(service: InferenceService) -> EmbeddingsHandler:
    """Build the handler for ``POST /v1/embeddings`` (PRD section 4.7)."""

    async def embeddings(request: Request) -> Response:
        embedding_request = embedding_request_from_payload(await _read_json_body(request))
        identity = get_identity(request)
        response = await service.embed(
            embedding_request, namespace=caller_namespace(identity)
        )
        return json(EmbeddingsResponse.from_embedding_response(response).to_dict())

    return embeddings


def register_embeddings_routes(router: Router, service: InferenceService) -> None:
    """Add the embeddings endpoint to ``router``."""
    router.add_post("/v1/embeddings", create_embeddings(service))


async def _read_json_body(request: Request) -> object:
    """Decode the request body, refusing anything that is not a JSON document.

    BlackSheep raises its own ``BadRequestFormat`` when the declared content type
    is JSON and the body is not, and returns ``None`` when the content type never
    declared JSON at all. Both are malformed input for this endpoint, and both
    become an invalid-request error carrying the taxonomy's reason rather than
    the framework's plain-text body, so every failure on this path answers in one
    shape (PRD section 5.4).
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


def _as_input(where: str, value: object) -> tuple[str, ...]:
    """The texts to embed, in the order the caller gave them.

    The OpenAI format accepts one string or an array of them, and both are
    normalised to the tuple the canonical request holds. A bare string is
    promoted to a one-element array here rather than refused as the canonical
    type would refuse it, because on this wire form the two are unambiguous: the
    endpoint knows the field's documented shape, and a caller that sent a string
    asked for exactly one text to be embedded.
    """
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(as_text(f"{where}[{index}]", item) for index, item in enumerate(value))
    raise InvalidArgumentError(
        f"{where} must be a string or an array of strings, got {describe(value)}"
    )


def _as_metadata(where: str, value: object) -> FrozenJsonObject | None:
    """Non-sensitive request metadata as an immutable object, or ``None`` when absent.

    ``None`` is kept as ``None`` rather than folded into an empty object, so a
    request that carried no metadata does not acquire an empty one. The value is
    never forwarded upstream: PRD section 5.1 makes request metadata caller
    bookkeeping, and the binding sends none of it.
    """
    if value is None:
        return None
    return as_json_object(where, value)


def _embedding_payload(embedding: Embedding) -> dict[str, object]:
    """One vector in the OpenAI shape: an object, its position and its numbers."""
    return {
        "object": _OBJECT_EMBEDDING,
        "index": embedding.index,
        "embedding": list(embedding.values),
    }


def _usage_payload(usage: Usage) -> dict[str, int]:
    """The OpenAI embeddings usage object, which counts only prompt tokens.

    The embeddings shape reports ``prompt_tokens`` and ``total_tokens`` and has no
    completion count, because there is no completion. Its counts are typed as
    integers, so a count the provider never reported is presented as zero here;
    that presentation is the boundary's, not the record's, and the canonical
    :class:`~model_gateway.domain.Usage` keeps an unreported count apart from a
    zero one (PRD section 13). A total the provider omitted is taken as the
    prompt count, which is arithmetic the caller can repeat.
    """
    prompt = 0 if usage.input_tokens is None else usage.input_tokens
    total = usage.total_tokens
    if total is None:
        total = prompt
    return {"prompt_tokens": prompt, "total_tokens": total}
