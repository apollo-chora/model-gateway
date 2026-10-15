"""Error mapping from the gateway taxonomy onto HTTP responses.

The taxonomy in :mod:`model_gateway.errors` already knows which HTTP status each
category carries, so this module decides nothing about status codes. It has two
jobs: render a :class:`~model_gateway.errors.GatewayError` as the sanitised JSON
payload a caller switches on, and make sure that a failure which is *not* a
gateway error still leaves as one.

The second job is the important one. A provider library raising its own
exception, a bug in the gateway, a decoding failure — none of those may reach a
caller as a traceback, a provider message or an HTML error page, because a
traceback carries the code path and a provider message carries whatever the
provider chose to include: an upstream URL, a request id, sometimes the prompt
itself. :func:`~model_gateway.errors.normalise_exception` turns
any such failure into an internal gateway error with a fixed, gateway-authored
message, and this module returns that instead.

Registration covers two exception classes. ``GatewayError`` covers everything the
gateway raised deliberately. ``Exception`` is the backstop, and it does not
interfere with the framework's own ``HTTPException`` handling: BlackSheep routes
those through a separate lookup that stops at ``HTTPException`` before reaching
the handlers registered here, so a 404 for an unknown path or a 400 for an
unreadable body keeps the framework's behaviour.

Logging follows the same rule. The record names the reason and the exception
*type*; it does not format the exception chain, because
``normalise_exception`` keeps the original exception as ``__cause__`` and that
chain may hold a prompt, a response or a credential. A log line that would help
debug a defect is worth having; one that spills a caller's prompt into the log
store is not.
"""

import structlog
from blacksheep import Application, Request, Response
from blacksheep.server.responses import json

from model_gateway.errors import GatewayError, normalise_exception

__all__ = [
    "handle_gateway_error",
    "handle_unexpected_error",
    "register_error_handlers",
]

logger = structlog.get_logger(__name__)


async def handle_gateway_error(
    app: Application, request: Request, error: Exception
) -> Response:
    """Render a classified gateway error with its own status and reason.

    The payload is the taxonomy's, not this module's: ``reason`` is the stable
    machine-readable identifier a caller branches on, and ``message`` is already
    sanitised at construction. Nothing here reaches for ``str(error)``, so the
    public body is exactly what the taxonomy decided to publish.
    """
    gateway_error = normalise_exception(error)
    return json(gateway_error.to_payload(), status=gateway_error.http_status)


async def handle_unexpected_error(
    app: Application, request: Request, error: Exception
) -> Response:
    """Turn any unclassified failure into a sanitised 500.

    The original exception is discarded from the response and kept only as
    ``__cause__``, so the provider's message, type name and traceback cannot
    leak. That is also why the log record below carries the exception's type and
    nothing else: the message is the one thing most likely to contain a prompt or
    a credential.
    """
    gateway_error = normalise_exception(error)
    logger.error(
        "unhandled gateway error",
        reason=gateway_error.reason.value,
        error_type=type(error).__name__,
        method=request.method,
        path=request.url.path.decode(),
    )
    return json(gateway_error.to_payload(), status=gateway_error.http_status)


def register_error_handlers(app: Application) -> None:
    """Install the taxonomy mapping and the sanitising backstop on ``app``.

    ``GatewayError`` is registered first so that a deliberate failure is matched
    by its own category; the ``Exception`` backstop then catches everything else.
    """
    app.exception_handler(GatewayError)(handle_gateway_error)
    app.exception_handler(Exception)(handle_unexpected_error)
