"""Inbound authentication and tenant isolation on the gRPC transport.

The transport-independent rules live elsewhere, and this module is the whole of
the gateway's inbound authentication on the gRPC surface (PRD sections 5.1 and
7): it extracts the credential, validates it and yields the trusted identity.
Nothing here decides what a caller may do — only who the gateway believes the
caller is.

**Who is calling?** A call presents ``authorization: Bearer <key>`` or
``x-api-key`` metadata, and the presentation is validated against the configured
key set — the same :class:`~model_gateway.security.ApiKeySet` the HTTP
middleware uses, so the two transports cannot disagree about a caller. The
grammar is the HTTP one's, applied to gRPC metadata: a credential presented
twice is refused rather than reduced to the first value, a malformed bearer
credential is refused rather than read as a 400, and two credential metadata
presenting different keys are refused rather than resolved by picking one.

**What is never trusted.** A ``tenant_id`` in a request body is a claim, not
evidence. The transport-independent rule that a claimed tenant must never become
or override the authenticated tenant lives in
:mod:`model_gateway.security.authorization`, so the HTTP and gRPC transports call
the same code rather than each reimplementing it (PRD section 5.1). Every RPC
handler reconciles the claim through it before it builds an invocation, and the
invocation's tenant is always the credential's, never the caller's.

The presented key is never logged, published or echoed; log records carry the
non-secret ``key_label`` instead (PRD section 17).
"""

from collections.abc import Iterable

import grpc
import structlog

from model_gateway.errors import UnauthenticatedError
from model_gateway.security.api_keys import ApiKeySet
from model_gateway.security.identity import Identity

__all__ = ["Authenticator", "authenticate", "extract_credential"]

_AUTHORIZATION = "authorization"
_API_KEY = "x-api-key"
_BEARER_SCHEME = "bearer"

logger = structlog.get_logger(__name__)


def extract_credential(metadata: Iterable[tuple[str, str]]) -> str | None:
    """The API key a call presents, or ``None`` when it presents none.

    ``authorization: Bearer <key>`` is the preferred form; ``x-api-key: <key>`` is
    accepted from clients that cannot set authorization.

    Each credential key must be presented exactly once: zero values means the key
    is absent, and more than one raises rather than being reduced to whichever
    value comes first. A repeated credential is an ambiguous presentation — the
    request-smuggling shape in which a forward proxy and a reverse proxy each act
    on a different value — and "first wins" would let the ignored value smuggle
    past anything that reads only the first. A malformed authorization value — a
    scheme that is not ``bearer``, or a bearer token that is empty — raises the
    same error rather than a 400: the caller's credential is wrong, not the shape
    of its request. When both keys are present and disagree, the call is rejected
    rather than having one of them picked silently.
    """
    authorization = _credential_value(metadata, _AUTHORIZATION, blank_is_absent=False)
    api_key = _credential_value(metadata, _API_KEY, blank_is_absent=True)

    if authorization is None:
        return api_key

    token = _bearer_token(authorization)
    if api_key is not None and api_key != token:
        raise UnauthenticatedError(
            "the authorization and x-api-key metadata present different credentials"
        )
    return token


def _credential_value(
    metadata: Iterable[tuple[str, str]], key: str, *, blank_is_absent: bool
) -> str | None:
    """One metadata key's value, or ``None`` when the call did not present it.

    Case-insensitive, as metadata keys are. A blank value counts as absent for
    ``x-api-key`` only: an empty API key is not a credential, whereas an empty
    authorization value is a malformed credential.
    """
    values = [value for name, value in metadata if name.lower() == key]
    if not values:
        return None
    if len(values) > 1:
        raise UnauthenticatedError(f"the {key} metadata was presented more than once")
    value = values[0]
    return None if blank_is_absent and not value else value


def _bearer_token(authorization: str) -> str:
    """The token from a bearer credential, or a refusal naming the rule."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != _BEARER_SCHEME or not token:
        raise UnauthenticatedError(
            "the authorization metadata must carry a Bearer token, as 'authorization: Bearer <key>'"
        )
    return token


def authenticate(api_keys: ApiKeySet, context: grpc.ServicerContext) -> Identity:
    """Validate the call's credential and return the trusted identity.

    A call that presents no credential is refused with ``authentication_required``
    when inbound authentication is enabled, and proceeds with the anonymous
    identity when it is not. A call that presents one has it validated; a key that
    matches nothing is refused with ``invalid_credentials``. The distinction is
    what tells a caller whether to obtain a credential or to replace the one it
    sent.
    """
    return api_keys.authenticate(extract_credential(_metadata_of(context)))


def _metadata_of(context: grpc.ServicerContext) -> tuple[tuple[str, str], ...]:
    """The call's invocation metadata as name-value pairs, once.

    Read once and reused: a handler that reads it per key would see a different
    sequence for each read on some implementations, and the ambiguity rule this
    module applies needs a single consistent view of the call.
    """
    return tuple(context.invocation_metadata() or ())


class Authenticator:
    """The inbound authentication policy a gRPC servicer applies.

    Held by the server and passed to every handler, so a handler authenticates by
    calling :meth:`identity` rather than reaching for the settings or the key set
    itself. That keeps the transport's policy in one object and makes a test able
    to drive an authenticated or an anonymous call without constructing a real
    credential store.
    """

    def __init__(
        self,
        api_keys: ApiKeySet,
        *,
        auth_enabled: bool,
        allow_anonymous_health: bool,
    ) -> None:
        self._api_keys = api_keys
        self._auth_enabled = auth_enabled
        self._allow_anonymous_health = allow_anonymous_health

    @property
    def auth_enabled(self) -> bool:
        """Whether inbound authentication is required for this deployment."""
        return self._auth_enabled

    def identity(self, context: grpc.ServicerContext, *, exempt: bool = False) -> Identity:
        """The trusted identity of the call, or a refusal.

        ``exempt`` marks a call the deployment allows without a credential — the
        health endpoints under ``GATEWAY_ALLOW_ANONYMOUS_HEALTH``. An exempt call
        that presents no credential proceeds anonymously; one that presents a
        credential still has it validated, so a rejected key is never waved through
        on an exempt path.
        """
        presented = extract_credential(_metadata_of(context))
        if presented is None:
            if exempt and self._allow_anonymous_health:
                return self._api_keys.anonymous()
            if not self._auth_enabled:
                return self._api_keys.anonymous()
            raise UnauthenticatedError("no API key was presented")
        return self._api_keys.authenticate(presented)
