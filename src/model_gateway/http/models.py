"""The model endpoints: ``GET /v1/models`` and ``GET /v1/models/{model_id}`` (PRD 4.3).

Both endpoints read the registry and publish capability metadata. They publish
nothing else, and that is a design constraint rather than a happy accident: the
registry holds routes, and a route holds an upstream endpoint, a credential and
any extra headers configured for it. The response is built from a
:class:`~model_gateway.http.dto.ModelCard`, whose fields are a public model ID,
a capability set and three optional limits, so there is no field an endpoint or a
key could travel in (PRD section 4.3, acceptance criterion 12).

An unknown model ID is not a 404 invented here. The registry raises
:class:`~model_gateway.errors.ModelNotFoundError`, which carries the taxonomy's
``model_not_found`` reason, and the error handler renders it. Resolving the ID
through the registry — rather than checking it against the list this module
produced — is what keeps the two endpoints in agreement and what keeps a
caller-supplied model name from reaching an unconfigured upstream (PRD 10).

The registry is imported for typing only. These handlers use three methods of its
interface and never construct one, so the module that builds it is the
application factory's dependency rather than this one's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from blacksheep import Request, Response
from blacksheep.server.responses import json
from blacksheep.server.routing import Router

from model_gateway.http.dto import ModelCard, model_list_payload

if TYPE_CHECKING:
    from model_gateway.registry import ModelRegistry

__all__ = ["get_model", "list_models", "register_model_routes"]

ListHandler = Callable[[Request], Awaitable[Response]]
DetailHandler = Callable[..., Awaitable[Response]]


def list_models(registry: ModelRegistry) -> ListHandler:
    """Build the handler for ``GET /v1/models`` (PRD section 4.3)."""

    async def handler(request: Request) -> Response:
        cards = [ModelCard.from_model_info(info) for info in registry.list_models()]
        return json(model_list_payload(cards))

    return handler


def get_model(registry: ModelRegistry) -> DetailHandler:
    """Build the handler for ``GET /v1/models/{model_id}`` (PRD section 4.3)."""

    async def handler(request: Request, model_id: str) -> Response:
        return json(ModelCard.from_model_info(registry.get(model_id)).to_dict())

    return handler


def register_model_routes(router: Router, registry: ModelRegistry) -> None:
    """Add both model endpoints to ``router``."""
    router.add_get("/v1/models", list_models(registry))
    router.add_get("/v1/models/{model_id}", get_model(registry))
