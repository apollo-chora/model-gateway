"""Continuing a caller's trace over gRPC (PRD section 17).

gRPC carries propagation in the call's metadata, which is the same key-and-value
shape the HTTP middleware reads out of headers — so the two transports continue a
trace the same way and a caller that traces over one is traced over the other. That
is the whole of this module: read the carrier, attach the context, detach it on
every exit.

**The context is detached on every exit, including an abort.** A context left
attached would leak into whatever the event loop ran next, and on a shared worker
that means one call's trace becoming another's parent — a trace that is wrong in a
way no error reports.

**The extraction is unconditional and the recording is not**, exactly as on the
HTTP side: attaching a context writes nothing, and the spans that would use it
exist only when a tracer was configured (see :mod:`model_gateway.observability`).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
from opentelemetry.context import attach, detach

from model_gateway.observability import extract_trace_context

__all__ = ["continue_trace"]


@asynccontextmanager
async def continue_trace(context: grpc.ServicerContext) -> AsyncIterator[None]:
    """Attach the caller's trace context for the duration of one RPC."""
    carrier = {key: value for key, value in context.invocation_metadata()}
    token = attach(extract_trace_context(carrier))
    try:
        yield
    finally:
        detach(token)
