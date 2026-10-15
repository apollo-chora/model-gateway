"""The middleware that continues an inbound trace (PRD section 17).

Section 17 asks for W3C Trace Context support, and for a gateway that means
something specific: an inbound ``traceparent`` is the parent of the span the
gateway produces, so a caller's trace reaches the upstream through the gateway
rather than stopping at it. This middleware is where that happens — it extracts the
context from the request headers and attaches it for the duration of the request,
so every span started inside the handler is a child of the caller's.

**The extraction is unconditional and the recording is not.** Nothing here asks
whether tracing is enabled: extracting a context is a header parse that writes
nothing, and the spans that would use it are only produced when a tracer was
configured. Branching here would mean two code paths for one behaviour, and the one
a deployment ran would be the one least tested.

**The context is detached on every exit**, including a refusal raised by a later
middleware or a handler. A context left attached would leak into whatever the
event loop ran next, which on a shared worker means one request's trace becoming
another's parent.
"""

from collections.abc import Awaitable, Callable

from blacksheep import Request, Response
from opentelemetry.context import attach, detach

from model_gateway.observability import extract_trace_context

__all__ = ["create_tracing_middleware"]

NextHandler = Callable[[Request], Awaitable[Response]]


def create_tracing_middleware() -> Callable[[Request, NextHandler], Awaitable[Response]]:
    """Build the middleware that continues the caller's trace."""

    async def tracing_middleware(request: Request, handler: NextHandler) -> Response:
        carrier = {
            key.decode("ascii", "replace"): value.decode("ascii", "replace")
            for key, value in request.headers
        }
        token = attach(extract_trace_context(carrier))
        try:
            return await handler(request)
        finally:
            detach(token)

    return tracing_middleware
