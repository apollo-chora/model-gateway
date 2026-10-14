"""Entry point: ``python -m model_gateway`` or ``uv run model-gateway``."""

from blacksheep import Application
from granian import Granian

from model_gateway.app import create_app
from model_gateway.config import Settings


def create_asgi_app() -> Application:
    """Factory invoked by granian to build the ASGI application.

    The gRPC listener starts beside the HTTP one (PRD section 18): both are
    started and stopped by the application, so a process serves the two
    transports under one lifetime, configured by the same environment.
    """
    return create_app(serve_grpc=True)


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
