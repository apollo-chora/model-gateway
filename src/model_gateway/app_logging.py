"""Structured logging via structlog, rendered as JSON through orjson."""

import logging
import sys

import orjson
import structlog


def configure_logging(level: int | str = logging.INFO) -> None:
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(serializer=_orjson_dumps),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _orjson_dumps(obj, **kwargs) -> str:
    return orjson.dumps(obj, **kwargs).decode("utf-8")
