"""Application factory.

One call builds everything a serving process needs: validated settings, logging,
the JSON encoder, the model registry read from the environment, the inbound
authentication middleware, the health endpoints, the OpenAI-compatible HTTP
surface and the error mapping that turns a gateway failure into a status and a
sanitised body.

The registry is built here, before the application is handed to the server, so
that configuration the gateway cannot run with fails at startup rather than on
the first request (PRD sections 6 and 21). Nothing is resolved lazily: a
deployment whose routes are malformed does not start.

When no service is injected, the factory builds the production one — the
LiteLLM binding of PRD section 9 over the registry's routes, with the attempt
budget from ``Settings`` — so a process started with configuration alone
actually calls upstream. An injected service is still used verbatim: tests and
callers bind the provider they want, naming the deterministic
:class:`~model_gateway.providers.MockProvider` when they need the stand-in
(PRD section 20). There is no configuration flag that enables the mock: an
operator cannot turn fake inference on by accident, and cannot turn it on on
purpose either.

The gateway core never names a provider, so the binding is swappable here without
touching routing, policy or the public contract (PRD sections 9 and 23).
"""

import orjson
import structlog
from blacksheep import Application
from blacksheep.settings.json import json_settings

from model_gateway.app_logging import configure_logging
from model_gateway.config import Settings
from model_gateway.health import healthz, readyz
from model_gateway.http import (
    register_chat_routes,
    register_error_handlers,
    register_model_routes,
)
from model_gateway.http.auth import create_auth_middleware
from model_gateway.inference import InferenceService
from model_gateway.providers.litellm_provider import LiteLLMProvider
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

logger = structlog.get_logger(__name__)


def _orjson_dumps(obj) -> str:
    return orjson.dumps(obj).decode("utf-8")


def configure_json() -> None:
    json_settings.use(dumps=_orjson_dumps)


def create_app(
    settings: Settings | None = None,
    *,
    registry: ModelRegistry | None = None,
    service: InferenceService | None = None,
) -> Application:
    """Build the application, ready to be started by the server.

    ``registry`` and ``service`` are keyword-only. An injected ``service`` is
    used verbatim — tests and callers pass
    ``InferenceService(registry, provider)`` naming the provider this process
    should call upstream with. When no service is injected the factory builds
    the production one: :class:`~model_gateway.providers.litellm_provider.LiteLLMProvider`
    over the registry's routes, with ``PROVIDER_TIMEOUT_SECONDS`` as the budget
    every attempt context carries (PRD section 9.6). That budget is the
    provider's own ceiling today; a later slice makes it the minimum of the
    configured timeout and the invocation deadline remaining when the attempt
    begins (PRD section 15), so a retry can never outlive the caller's budget.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    registry = registry or ModelRegistry.from_env()

    if service is None:
        service = InferenceService(
            registry,
            LiteLLMProvider(),
            attempt_timeout=settings.provider_timeout_seconds,
        )

    api_keys = ApiKeySet.from_settings(settings)
    app = Application()
    app.middlewares.append(
        create_auth_middleware(
            api_keys,
            auth_enabled=settings.auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
    )
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    register_model_routes(app.router, registry)
    register_chat_routes(app.router, service)
    register_error_handlers(app)

    @app.on_start
    async def _on_start(app: Application) -> None:
        logger.info(
            "model-gateway starting",
            environment=settings.environment,
            http_port=settings.http_port,
            models=[info.id for info in registry.list_models()],
        )

    @app.on_stop
    async def _on_stop(app: Application) -> None:
        logger.info("model-gateway stopped")

    return app
