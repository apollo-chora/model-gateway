"""Transport-independent authorization for a trusted identity.

The rule that a caller's *claimed* tenant must never become or override its
*authenticated* tenant is a property of the credential model, not of any one
wire format. It lives here so that the HTTP and gRPC transports call the same
code instead of each reimplementing it: a transport extracts the credential and
builds the :class:`~model_gateway.security.Identity`, and this module reconciles
whatever tenant the caller claimed against it.

:func:`reconcile_identity` covers the whole of the rule:

- a claimed tenant that disagrees with the authenticated tenant — including one
  the credential is not bound to at all — is refused with
  :class:`~model_gateway.errors.PermissionDeniedError`;
- an absent claim is fine, and a claim matching the authenticated tenant is
  fine;
- an authenticated key with no tenant cannot be given one by a claim: the claim
  disagrees with ``None`` and is refused;
- an unauthenticated caller's claim is discarded rather than propagated, so a
  request field can never create a tenant.

The function validates and raises; it does not rewrite the invocation. The
trusted tenant a caller acts under is always ``Identity.tenant``, read from the
credential, and a transport builds its invocation from that identity rather than
from the body.

Only the credential's ``tenant`` and ``credential_id`` are principals for
authorization; caller-supplied ``actor_id`` and ``agent_id`` are attribution
labels and are not consulted here.
"""

from model_gateway.errors import PermissionDeniedError
from model_gateway.security.identity import Identity

__all__ = ["reconcile_identity"]


def reconcile_identity(identity: Identity | None, claimed_tenant: str | None) -> None:
    """Refuse a claimed tenant that disagrees with the authenticated tenant.

    The authenticated identity is the sole source of tenant identity: a
    claimed tenant can never supply or override a tenant. When
    the caller is authenticated and claims a different tenant — including one
    the credential is not bound to at all — the request is a
    :class:`~model_gateway.errors.PermissionDeniedError`. An absent claim is
    fine, and one that matches the authenticated tenant is fine.

    When the caller is unauthenticated there is nothing trusted to reconcile
    against, so the claim is not authoritative and is discarded: this function
    returns without raising, and no tenant is propagated. A ``None`` identity
    means the authentication middleware never ran, and is treated the same way.
    """
    if identity is None or not identity.authenticated:
        return
    if claimed_tenant is not None and claimed_tenant != identity.tenant:
        details = {"key_label": identity.key_label} if identity.key_label is not None else None
        raise PermissionDeniedError(
            "the claimed tenant does not match the tenant bound to the presented "
            "credential; a claimed tenant is not authorization evidence and cannot "
            "supply or override the credential's tenant",
            details=details,
        )
