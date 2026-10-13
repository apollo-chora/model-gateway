"""Application factory."""

import orjson
import structlog
from blacksheep import Application
from blacksheep.settings.json import json_settings

from model_gateway.app_logging import configure_logging
from model_gateway.config import Settings
from model_gateway.health import healthz, readyz

logger = structlog.get_logger(__name__)


def _orjson_dumps(obj) -> str:
    return orjson.dumps(obj).decode("utf-8")


def configure_json() -> None:
    json_settings.use(dumps=_orjson_dumps)


def create_app(settings: Settings | None = None) -> Application:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    app = Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)

    @app.on_start
    async def _on_start(app: Application) -> None:
        logger.info(
            "model-gateway starting",
            environment=settings.environment,
            http_port=settings.http_port,
        )

    @app.on_stop
    async def _on_stop(app: Application) -> None:
        logger.info("model-gateway stopped")

    return app
