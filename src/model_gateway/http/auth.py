"""Inbound HTTP authentication: credential extraction, scope and trusted identity.

This module is the whole of the gateway's inbound authentication on the HTTP
surface (PRD sections 7 and 14). For every request it answers three questions,
and nothing else:

**Who is calling?** A credential is extracted from the request — ``Authorization:
Bearer <key>`` or ``x-api-key: <key>`` — and validated against the configured
key set. The result is a :class:`~model_gateway.security.identity.Identity`:
the one trusted-identity type the rest of the gateway switches on. The identity
is derived from the credential alone, so a tenant is a property of the key and
never of the request (PRD section 5.1).

**May they call this?** When authentication is enabled, every ``/v1/*``
endpoint requires a valid credential — the model listing included, because it
exposes configured model IDs. The health endpoints are exempt only when
``GATEWAY_ALLOW_ANONYMOUS_HEALTH`` is true; when it is false they require a
credential too. When authentication is disabled, every request proceeds with
the anonymous identity.

**Whose tenant is it?** The authenticated identity is the sole source of
tenant identity. A request body's ``tenant_id`` is never authorization
evidence: when it disagrees with the authenticated tenant the request is
refused with a :class:`~model_gateway.errors.PermissionDeniedError` rather
than silently ignored, and when the caller is unauthenticated it is discarded
rather than propagated.

The presented key is never logged, published or echoed. Log records carry the
non-secret ``key_id`` instead (PRD section 17).
"""

from collections.abc import Awaitable, Callable

import structlog
from blacksheep import Request, Response

from model_gateway.errors import PermissionDeniedError, UnauthenticatedError
from model_gateway.security.api_keys import ApiKeySet
from model_gateway.security.identity import Identity

__all__ = [
    "authenticate_request",
    "create_auth_middleware",
    "enforce_tenant",
    "extract_credential",
    "get_identity",
]

_AUTHORIZATION_HEADER = b"authorization"
_API_KEY_HEADER = b"x-api-key"
_BEARER_SCHEME = "bearer"
_IDENTITY_ATTRIBUTE = "gateway_identity"
_HEALTH_PATHS: frozenset[bytes] = frozenset({b"/healthz", b"/readyz"})

logger = structlog.get_logger(__name__)

NextHandler = Callable[..., Awaitable[Response]]
"""The next link in the chain: the route handler, or the next middleware."""


def extract_credential(request: Request) -> str | None:
    """The API key the request presents, or ``None`` when it presents none.

    ``Authorization: Bearer <key>`` is the preferred form; ``x-api-key: <key>``
    is accepted from clients that cannot set Authorization. A malformed
    Authorization header — a scheme that is not ``Bearer``, or a bearer token
    that is empty — raises :class:`~model_gateway.errors.UnauthenticatedError`
    rather than a 400: the caller's credential is wrong, not the shape of its
    request. When both headers are present and disagree, the request is
    rejected rather than having one of them picked silently.
    """
    authorization = _decode(request.get_first_header(_AUTHORIZATION_HEADER))
    api_key = _header_text(request.get_first_header(_API_KEY_HEADER))

    if authorization is None:
        return api_key

    token = _bearer_token(authorization)
    if api_key is not None and api_key != token:
        raise UnauthenticatedError(
            "the Authorization and x-api-key headers present different credentials"
        )
    return token


def authenticate_request(api_keys: ApiKeySet, request: Request, *, exempt: bool) -> Identity:
    """Validate the request's credential and return the trusted identity.

    ``exempt`` marks a path the deployment allows without a credential — a
    health endpoint under ``GATEWAY_ALLOW_ANONYMOUS_HEALTH``. An exempt request
    that presents no credential proceeds anonymously; one that presents a
    credential still has it validated, so a rejected key is never waved through
    on an exempt path. A non-exempt request without a credential is refused
    with ``authentication_required``.
    """
    presented = extract_credential(request)
    if presented is None:
        if exempt:
            return api_keys.anonymous()
        raise UnauthenticatedError(
            "no credential was presented; send the API key as "
            "'Authorization: Bearer <key>' or 'x-api-key: <key>'"
        )
    identity = api_keys.authenticate(presented)
    logger.info("request authenticated", key_id=identity.key_id)
    return identity


def enforce_tenant(identity: Identity | None, body_tenant_id: str | None) -> None:
    """Refuse a body ``tenant_id`` that disagrees with the authenticated tenant.

    The authenticated identity is the sole source of tenant identity (PRD
    section 5.1): a body ``tenant_id`` can never supply or override a tenant.
    When the caller is authenticated and the body names a different tenant —
    including one the credential is not bound to at all — the request is a
    :class:`~model_gateway.errors.PermissionDeniedError`. An absent body
    ``tenant_id`` is fine, and one that matches the authenticated tenant is
    fine.

    When the caller is unauthenticated there is nothing trusted to enforce
    against, so the body value is not authoritative: it is discarded by
    :meth:`~model_gateway.http.dto.ChatRequest.to_invocation` rather than
    propagated as the invocation's tenant.
    """
    if identity is None or not identity.authenticated:
        return
    if body_tenant_id is not None and body_tenant_id != identity.tenant:
        raise PermissionDeniedError(
            "the request body's tenant_id does not match the tenant bound to the "
            "presented credential; a request-body tenant_id is not authorization "
            "evidence and cannot supply or override the credential's tenant",
            details={"key_id": identity.key_id},
        )


def get_identity(request: Request) -> Identity | None:
    """The trusted identity the authentication middleware assigned to ``request``.

    ``None`` when the middleware did not run — a hand-built application, or a
    route registered outside the application factory. Callers treat ``None`` as
    unauthenticated: no tenant is enforced, and none is propagated.
    """
    return getattr(request, _IDENTITY_ATTRIBUTE, None)


def create_auth_middleware(
    api_keys: ApiKeySet, *, auth_enabled: bool, allow_anonymous_health: bool
) -> Callable[[Request, NextHandler], Awaitable[Response]]:
    """Build the middleware that authenticates every request.

    The middleware assigns the trusted identity to the request as the
    ``gateway_identity`` attribute before the handler runs, so handlers read
    the identity the credential produced rather than anything a request body
    claims (PRD section 5.1). It decides nothing else: routing, body parsing
    and response shaping stay with the handlers.
    """

    async def auth_middleware(request: Request, handler: NextHandler) -> Response:
        if not auth_enabled:
            identity = api_keys.anonymous()
        else:
            exempt = request.url.path in _HEALTH_PATHS and allow_anonymous_health
            try:
                identity = authenticate_request(api_keys, request, exempt=exempt)
            except UnauthenticatedError as error:
                logger.warning(
                    "authentication failed",
                    reason=error.reason.value,
                    method=request.method,
                    path=request.url.path.decode("ascii", "replace"),
                )
                raise
        setattr(request, _IDENTITY_ATTRIBUTE, identity)
        return await handler(request)

    return auth_middleware


def _decode(value: bytes | None) -> str | None:
    """A header value as text, or ``None`` when the header is absent.

    Unlike :func:`_header_text` this does not collapse a blank value to absent:
    a present-but-empty ``Authorization`` header is a malformed credential, not
    a missing one.
    """
    if value is None:
        return None
    return value.decode("latin-1")


def _header_text(value: bytes | None) -> str | None:
    """A header value as text, or ``None`` when the header is absent or blank."""
    if value is None:
        return None
    text = value.decode("latin-1").strip()
    return text or None


def _bearer_token(authorization: str) -> str:
    """The token of an ``Authorization`` header, refusing any other form."""
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != _BEARER_SCHEME:
        raise UnauthenticatedError("the Authorization header must be 'Bearer <key>'")
    return parts[1]
