"""The OpenAI-compatible HTTP surface (PRD sections 4.3, 4.4, 5.4 and 9.5).

**This surface is a supported OpenAI-compatible subset, not a drop-in
replacement for the OpenAI API.** The routes, the request fields and the response
shapes are the OpenAI ones, and a caller written against the documented subset
works unchanged. Everything outside that subset is refused rather than ignored:
a request parameter the OpenAI format does not define is answered with a 400
naming the key, and a parameter the format does define that this gateway cannot
honour — ``n`` above one, ``logprobs``, ``parallel_tool_calls``, the legacy
``functions`` family, remote image URLs, streaming — is answered with a
capability error naming the feature. That strictness is deliberate (PRD section
9.5 forbids silently discarding a parameter) and it has a consequence worth
stating before anyone markets this endpoint: a newer client SDK that serialises a
new parameter, or a new default value for an existing one, receives a 400 from a
gateway that has not been taught it yet. Compatibility here means the subset is
stable and its refusals are explicit, not that any OpenAI client will work
unmodified.

Six modules make up this package, and the split follows the boundaries the PRD
draws rather than the file sizes:

:mod:`~model_gateway.http.dto`
    The wire types, and the whole of the translation between the OpenAI format
    and the canonical domain. Everything that decides what a request *means*
    lives here, so no handler has to.
:mod:`~model_gateway.http.models`
    ``GET /v1/models`` and ``GET /v1/models/{model_id}``.
:mod:`~model_gateway.http.chat`
    ``POST /v1/chat/completions``.
:mod:`~model_gateway.http.embeddings`
    ``POST /v1/embeddings``.
:mod:`~model_gateway.http.errors``
    The one place a :class:`~model_gateway.errors.GatewayError` becomes an HTTP
    response, and the one place an unclassified failure is sanitised.
:mod:`~model_gateway.http.images`
    ``POST /v1/images/generations`` and ``POST /v1/images/edits``.

Handlers are built by factory functions rather than reaching for module-level
state, so a route's dependencies are its arguments: ``list_models(registry)``
returns the handler, and the application factory decides which registry it gets.
That is also what lets the tests exercise the surface without an environment,
a network or a provider.

Neither the registry nor the inference service is imported at runtime by these
modules — only for typing. They are the gateway core, this package is the
northbound interface, and the application factory is where the two meet
(PRD section 23).
"""

from model_gateway.http.chat import create_chat_completion, register_chat_routes
from model_gateway.http.dto import (
    ChatCompletion,
    ChatMessage,
    ChatRequest,
    ModelCard,
    model_list_payload,
)
from model_gateway.http.embeddings import (
    EmbeddingsResponse,
    create_embeddings,
    embedding_request_from_payload,
    register_embeddings_routes,
)
from model_gateway.http.errors import (
    handle_gateway_error,
    handle_unexpected_error,
    register_error_handlers,
)
from model_gateway.http.images import (
    ImageGenerationResponse,
    create_image_edits,
    create_image_generations,
    image_edit_request_from_payload,
    image_generation_request_from_payload,
    register_image_routes,
)
from model_gateway.http.models import get_model, list_models, register_model_routes

__all__ = [
    "ChatCompletion",
    "ChatMessage",
    "ChatRequest",
    "EmbeddingsResponse",
    "ImageGenerationResponse",
    "ModelCard",
    "create_chat_completion",
    "create_embeddings",
    "create_image_edits",
    "create_image_generations",
    "embedding_request_from_payload",
    "get_model",
    "handle_gateway_error",
    "handle_unexpected_error",
    "image_edit_request_from_payload",
    "image_generation_request_from_payload",
    "list_models",
    "model_list_payload",
    "register_chat_routes",
    "register_embeddings_routes",
    "register_error_handlers",
    "register_image_routes",
    "register_model_routes",
]
