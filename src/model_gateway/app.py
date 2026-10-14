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
from model_gateway.config import ConfigurationError, Settings
from model_gateway.config.routes import Operation
from model_gateway.errors import FailedPreconditionError
from model_gateway.health import healthz, readyz
from model_gateway.http import (
    register_chat_routes,
    register_embeddings_routes,
    register_error_handlers,
    register_model_routes,
)
from model_gateway.http.auth import create_auth_middleware
from model_gateway.inference import InferenceService
from model_gateway.providers.litellm_provider import (
    LiteLLMProvider,
    ensure_supported_dialect,
)
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter
from model_gateway.security.api_keys import ApiKeySet

logger = structlog.get_logger(__name__)

_BINDING_OPERATIONS: frozenset[Operation] = frozenset({Operation.TEXT, Operation.EMBEDDINGS})
"""The operations the production binding runs, and so the ones its start validates.

The binding has one entry point per operation: the text completion and the
embedding call. An operation it does not run has no route this process would
serve, so a dialect it could not keep behind the transport controls is not a
reason to refuse to start — that route is inert here.
"""


def _ensure_binding_can_run(registry: ModelRegistry) -> None:
    """Refuse to start when a configured route names a dialect the binding cannot run.

    The binding refuses such a route at the attempt too, and this is not a
    duplicate of that check: a deployment whose configuration the transport
    controls cannot survive should not reach the point of serving requests at
    all (PRD sections 6, 14.1 and 21). Every operation the binding runs is
    checked — the text and embeddings operations today — so no route this
    process would actually serve is unusable at startup and discoverable only
    when a caller arrives. An operation with no binding is not checked, because
    this process would never run its routes. The check runs only when this
    factory builds the binding, since an injected service brings its own
    provider.
    """
    for model in registry.list_models():
        if model.operation not in _BINDING_OPERATIONS:
            continue
        for route in registry.routes_for(model.id):
            try:
                ensure_supported_dialect(route.format)
            except FailedPreconditionError as error:
                raise ConfigurationError(
                    f"{model.id}: {error}. The gateway does not start with a route it cannot "
                    f"run behind its transport controls"
                ) from error


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
    over the registry's routes, with the deployment's configured reliability
    policy — ``PROVIDER_TIMEOUT_SECONDS`` as both the invocation deadline and
    the per-attempt ceiling (one configured value plays both roles today; they
    are logically distinct bounds and separating them is a future refinement),
    ``PROVIDER_RETRY_COUNT`` and ``PROVIDER_RETRY_BACKOFF_MS`` as the retry
    schedule, and a limiter built from ``PROVIDER_MAX_CONCURRENT_REQUESTS``
    (PRD section 9.6). The service anchors one deadline per invocation and
    hands each attempt the smaller of the configured timeout and what is left
    of that deadline, so a retry can never outlive the caller's budget (PRD
    section 15).

    Building that binding also validates the routes it would run: a route of any
    operation the binding serves — text or embeddings — whose dialect it cannot
    keep behind the transport controls fails the start (PRD section 14.1). An
    injected service is not validated, because it brings its own provider and
    this factory cannot know what that provider can run.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    registry = registry or ModelRegistry.from_env()

    if service is None:
        _ensure_binding_can_run(registry)
        service = InferenceService(
            registry,
            LiteLLMProvider(),
            attempt_timeout=settings.provider_timeout_seconds,
            retry_count=settings.provider_retry_count,
            retry_backoff_ms=settings.provider_retry_backoff_ms,
            limiter=ConcurrencyLimiter(settings.provider_max_concurrent_requests),
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
    register_embeddings_routes(app.router, service)
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
