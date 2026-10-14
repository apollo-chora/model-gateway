"""Translate a gateway failure into a gRPC status (PRD section 5.4).

The error taxonomy already carries a gRPC status name on every category and
already sanitises every message and detail, so the transport has exactly one
job: read those two properties and abort with them. No exception type is
matched and no message text is inspected here, which is what makes the mapping
safe — a category can never publish a reason its own status does not own, and
a caller that switches on ``reason`` behaves the same over either transport.

The gRPC status name travels as a :class:`~model_gateway.errors.GrpcStatus` string
constant holding the standard name from the gRPC status code list, so it is
mapped onto the enum's member by name rather than through a second table that
could drift from the taxonomy.

What a caller receives
-----------------------
``context.abort`` takes a status code and a details string. The details string
is the sanitised public payload as JSON, so a gRPC caller reads the same
``reason`` and ``message`` an HTTP caller reads in a response body — and, because
the payload is JSON rather than prose, a machine client can parse it without
relying on the status code alone. ``details`` is included when the category
carries it: it is gateway-authored and sanitised at construction, and the gRPC
details string is the natural place for it, whereas the HTTP body keeps its
documented reason-and-message shape.
"""

import json

import grpc

from model_gateway.errors import GatewayError, normalise_exception

__all__ = ["abort", "grpc_code", "grpc_details"]


def _as_gateway_error(error: BaseException) -> GatewayError:
    """Classify an arbitrary failure without re-classifying a gateway error.

    A :class:`~model_gateway.errors.GatewayError` is returned unchanged: it has
    already been classified and sanitised, and running it through
    :func:`~model_gateway.errors.normalise_exception` again would demote a
    precise reason to ``INTERNAL_ERROR``.
    """
    return error if isinstance(error, GatewayError) else normalise_exception(error)


def grpc_code(error: BaseException) -> grpc.StatusCode:
    """The gRPC status code a failure maps to."""
    return getattr(grpc.StatusCode, _as_gateway_error(error).grpc_code.value)


def grpc_details(error: BaseException) -> str:
    """The sanitised public payload of a failure, as a JSON string."""
    payload = _as_gateway_error(error).to_payload(include_details=True)
    return json.dumps(payload, separators=(",", ":"))


def abort(context: grpc.ServicerContext, error: BaseException) -> None:
    """Abort the call with the gateway error's own status and payload.

    Raises gRPC's own abort error to terminate the handler; it returns nothing.
    """
    gateway_error = _as_gateway_error(error)
    context.abort(
        getattr(grpc.StatusCode, gateway_error.grpc_code.value),
        json.dumps(gateway_error.to_payload(include_details=True), separators=(",", ":")),
    )
