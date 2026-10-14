"""Application factory.

One call builds everything a serving process needs: validated settings, logging,
the JSON encoder, the model registry read from the environment, the inference
service, the health endpoints, the OpenAI-compatible HTTP surface and the error
mapping that turns a gateway failure into a status and a sanitised body.

The registry and the inference service are built here, before the application is
handed to the server, so that configuration the gateway cannot run with fails at
startup rather than on the first request (PRD sections 6 and 21). Nothing is
resolved lazily: a deployment whose routes are malformed does not start.

Both are also injectable. The factory builds them by default and accepts
alternatives, which is what lets a test exercise the real routes and the real
error handling against a registry and a provider it controls, without an
environment, a network or an upstream account. The injection points are the
seam, not a second code path: an injected registry is used exactly as a
constructed one is.

Which provider this process calls upstream with is :func:`_text_provider`'s
decision and nothing else's. The gateway core never names a provider, so the
binding is swappable here without touching routing, policy or the public
contract (PRD sections 9 and 23).
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
from model_gateway.inference import InferenceService
from model_gateway.providers import MockProvider, TextProvider
from model_gateway.registry import ModelRegistry

logger = structlog.get_logger(__name__)


def _orjson_dumps(obj) -> str:
    return orjson.dumps(obj).decode("utf-8")


def configure_json() -> None:
    json_settings.use(dumps=_orjson_dumps)


def _text_provider() -> TextProvider:
    """The text provider upstream calls are made through.

    The deterministic mock is the provider this slice wires: it performs the
    gateway's routing, policy, capability enforcement and error mapping end to
    end with no upstream account, which is what makes the HTTP surface testable
    and the service startable (PRD section 20). The LiteLLM binding replaces it
    at this seam — the return type is the provider protocol, and no caller above
    this function names a provider (PRD section 9).
    """
    return MockProvider()


def create_app(
    settings: Settings | None = None,
    *,
    registry: ModelRegistry | None = None,
    service: InferenceService | None = None,
) -> Application:
    """Build the application, ready to be started by the server."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    registry = registry or ModelRegistry.from_env()
    service = service or InferenceService(registry, _text_provider())

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
