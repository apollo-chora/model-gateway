"""The inbound body-size limit.

A request-body size limit is a deployment concern, and this is where the HTTP
transport applies one. The bound is the deployment's, from
``GATEWAY_MAX_REQUEST_BODY_BYTES``,
and it exists because the gateway reads a body into memory before it can validate
anything in it: without a bound, the size of a caller's request is the amount of
memory the gateway will spend on it, and that is true before authentication has
decided anything.

**A declared length is refused before the body is read.** ``Content-Length`` states
the size up front, so a request that announces more than the bound is answered
without the gateway buffering a byte of it — the refusal is cheapest exactly where
it matters most. A length the gateway cannot parse is not treated as a small one:
it falls through to the measured path below.

**A request that declares no length is measured.** A chunked body carries no
``Content-Length``, so the only way to bound it is to read it and measure what
arrived. Reading here costs nothing the handler would not spend — the framework
caches the body, so the handler's own read is the same bytes rather than a second
one — and the alternative, letting a chunked body through unchecked, would be a
limit a caller evades by declining to state a size.

**The refusal is a resource exhaustion, not a malformed request.** The caller sent
a well-formed request the deployment will not carry, which is what 413 is for, and
gRPC reports the same condition as ``RESOURCE_EXHAUSTED``. The reason is
``payload_too_large`` so the two can be told apart from the rate and concurrency
refusals they share a gRPC status with.

**The limit is applied after authentication.** Both checks are cheap and neither
reads the body, so the order is a statement about what an unidentified caller is
told: that it is unidentified, rather than something about the gateway's
configuration. With authentication disabled the size check is the only one that
runs, which is the deployment that needs it most.

**What the measured path bounds, and what it does not.** It bounds what the gateway
*accepts*: a body over the bound is never handed to a handler, so no request
validation, routing or provider work happens on it. It does not bound what the
server spent receiving that body, because a middleware is given a body the server
has already read. Framing guarantees — that a message ends where its declared
length says, that a chunked body is well formed, and that an edge proxy and this
server agree on both — belong to the server and to whatever sits in front of it,
and a middleware cannot repair a disagreement about where a request ends after the
fact. The declared-length path is the cheap refusal either way, because it never
asks for the body at all.
"""

from collections.abc import Awaitable, Callable

from blacksheep import Request, Response

from model_gateway.errors import PayloadTooLargeError

__all__ = ["create_body_limit_middleware"]

NextHandler = Callable[[Request], Awaitable[Response]]


def create_body_limit_middleware(
    maximum_bytes: int,
) -> Callable[[Request, NextHandler], Awaitable[Response]]:
    """Build the middleware that refuses a body larger than ``maximum_bytes``."""

    def refusal(observed: str) -> PayloadTooLargeError:
        return PayloadTooLargeError(
            f"the request body is larger than this gateway accepts; the limit is "
            f"{maximum_bytes} bytes",
            details={"limit_bytes": str(maximum_bytes), "observed": observed},
        )

    async def body_limit_middleware(request: Request, handler: NextHandler) -> Response:
        declared = request.headers.get_first(b"content-length")
        if declared is not None:
            try:
                announced = int(declared)
            except ValueError:
                announced = None
            if announced is not None and announced > maximum_bytes:
                raise refusal(f"declared {announced} bytes")
            if announced is not None:
                return await handler(request)

        body = await request.read()
        if body is not None and len(body) > maximum_bytes:
            raise refusal(f"received {len(body)} bytes")
        return await handler(request)

    return body_limit_middleware
