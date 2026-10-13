"""Entry point: ``python -m model_gateway`` or ``uv run model-gateway``."""

from blacksheep import Application
from granian import Granian

from model_gateway.app import create_app
from model_gateway.config import Settings


def create_asgi_app() -> Application:
    """Factory invoked by granian to build the ASGI application."""
    return create_app()


def main() -> None:
    settings = Settings.from_env()
    Granian(
        "model_gateway.__main__:create_asgi_app",
        interface="asgi",
        factory=True,
        address="0.0.0.0",
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    ).serve()


if __name__ == "__main__":
    main()
