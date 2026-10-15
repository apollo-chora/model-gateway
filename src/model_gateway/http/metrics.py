"""The metrics endpoint: ``GET /metrics`` (PRD section 17).

The exposition surface for :mod:`model_gateway.metrics`. It is a module of its own
because the registry is policy — which numbers the gateway keeps and what each one
means — while this is transport: a content type, a status and a body that is
deliberately not JSON like every other endpoint the gateway serves.

The content type is the format's own, and it carries the version: a scraper
negotiates on it, and serving the text as ``text/plain`` without the version would
tell a client that understands OpenMetrics that this is plain text it should not
try to parse as anything else.

**The endpoint is authenticated like every other path.** PRD section 14.2 exempts
exactly two paths, ``/healthz`` and ``/readyz``, and a metrics scrape is not one
of them: the exemption exists so a probe can reach a gateway that is unwell, and
an operator's scraper can carry a credential the way any other client does.
Publishing the gateway's counts to an unauthenticated caller would also publish
how much traffic it is carrying, which is not a health signal. What the body
carries is counts and durations and no request content, which is what makes it
safe to publish at all — the same rule that keeps prompts and credentials out of
the logs (PRD section 17).
"""

from collections.abc import Awaitable, Callable

from blacksheep import Request, Response
from blacksheep.contents import Content
from blacksheep.server.routing import Router

from model_gateway.metrics import Metrics, render_prometheus

__all__ = ["MetricsHandler", "create_metrics", "register_metrics_route"]

MetricsHandler = Callable[[Request], Awaitable[Response]]

_CONTENT_TYPE = b"text/plain; version=0.0.4; charset=utf-8"
"""The Prometheus text exposition format's media type, version included."""


def create_metrics(metrics: Metrics) -> MetricsHandler:
    """Build the handler for ``GET /metrics``."""

    async def expose(request: Request) -> Response:
        # The body declares its own media type, because the format's carries a
        # `version` parameter that the response's typed content type does not model
        # and that a scraper negotiates on.
        body = render_prometheus(metrics).encode("utf-8")
        return Response(200, None, Content(_CONTENT_TYPE, body))

    return expose


def register_metrics_route(router: Router, metrics: Metrics) -> None:
    """Add the metrics endpoint to ``router``."""
    router.add_get("/metrics", create_metrics(metrics))
