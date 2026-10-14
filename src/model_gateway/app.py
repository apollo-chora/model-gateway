"""Application factory.

One call builds everything a serving process needs: validated settings, logging,
the JSON encoder, the model registry read from the environment, the health
endpoints, the OpenAI-compatible HTTP surface and the error mapping that turns a
gateway failure into a status and a sanitised body.

The registry is built here, before the application is handed to the server, so
that configuration the gateway cannot run with fails at startup rather than on
the first request (PRD sections 6 and 21). Nothing is resolved lazily: a
deployment whose routes are malformed does not start.

The inference service is deliberately *not* built here. This slice has no
production inference provider — the LiteLLM binding of PRD section 9 is not
written yet — so a factory that picked one would be choosing it on the operator's
behalf, and the only provider it could pick is the deterministic mock. A process
started that way advertises ``text-default``, accepts a production chat request
and answers it with an echo: a fake indistinguishable, from the outside, from
working upstream inference. So the factory refuses to start rather than degrading
to one, and :data:`NO_PROVIDER_MESSAGE` says what to do about it.

The registry and the service are both injectable, and injecting the service is
the only way to obtain a serving application. Tests and development code pass
``InferenceService(registry, provider)`` explicitly — naming
:class:`~model_gateway.providers.MockProvider` when they want the deterministic
stand-in — and the binding will be wired at this same seam once it exists. There
is no configuration flag that enables the mock: an operator cannot turn fake
inference on by accident, and cannot turn it on on purpose either.

The gateway core never names a provider, so the binding is swappable here without
touching routing, policy or the public contract (PRD sections 9 and 23).
"""

import orjson
import structlog
from blacksheep import Application
from blacksheep.settings.json import json_settings

from model_gateway.app_logging import configure_logging
from model_gateway.config import ConfigurationError, Settings
from model_gateway.health import healthz, readyz
from model_gateway.http import (
    register_chat_routes,
    register_error_handlers,
    register_model_routes,
)
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

logger = structlog.get_logger(__name__)

NO_PROVIDER_MESSAGE = (
    "no inference provider is wired: this build has no production inference provider, so "
    "create_app() will not select one implicitly and will not fall back to the deterministic "
    "mock provider; pass service=InferenceService(registry, provider) explicitly, naming the "
    "provider this process should call upstream with"
)
"""Why startup fails, and what the caller does about it.

It is a module constant so the message is one string rather than a copy per call
site, and so a test can assert against it rather than against prose that drifts.
"""


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

    ``service`` is required and keyword-only: there is no production provider to
    build one from, so an omitted service is a startup failure carrying
    :data:`NO_PROVIDER_MESSAGE` rather than an application served by the mock
    (see the module docstring). ``ConfigurationError`` is the same failure the
    rest of startup raises, so a deployment that cannot run stops before the
    server binds a port.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    if service is None:
        raise ConfigurationError(NO_PROVIDER_MESSAGE)

    registry = registry or ModelRegistry.from_env()

    app = Application()
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
