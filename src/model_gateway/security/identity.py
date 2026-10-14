"""Trusted identity of an inbound caller (PRD section 7).

Inbound authentication is credential-based: a caller presents an API key, and
everything the gateway believes about that caller is derived from the key
alone. :class:`Identity` is the result of that derivation — the one
trusted-identity type the rest of the gateway switches on.

Two rules define the type:

- A tenant is a property of the credential, never of the request. ``tenant``
  is ``None`` unless the presented key is explicitly bound to one, and a
  request-body ``tenant_id`` can never supply or override it (PRD section 5.1).
- ``key_id`` is a label, not a credential. It identifies which configured key
  was presented so that logs and accounting can correlate requests, without
  the key material itself leaving the process boundary.
"""

from dataclasses import dataclass

__all__ = ["Identity"]


@dataclass(frozen=True)
class Identity:
    """Who the gateway believes the caller is, derived from the credential only.

    ``authenticated``
        ``True`` only after a presented key matched a configured one. The
        anonymous identity — assigned when authentication is disabled — is the
        only ``authenticated=False`` identity, and it carries no tenant.

    ``tenant``
        The tenant the caller's credential is bound to, or ``None`` when the
        configured key binds to no tenant. An authenticated caller with
        ``tenant is None`` is authenticated but tenant-less; an
        unauthenticated caller has NO tenant by construction, and no
        request-body ``tenant_id`` can supply one (PRD section 5.1).

    ``key_id``
        A short, non-secret label for the presented key, safe to log. It is
        derived from the key (see :class:`ApiKeySet`) and never is the key.
        ``None`` for the anonymous identity.
    """

    authenticated: bool
    tenant: str | None
    key_id: str | None
