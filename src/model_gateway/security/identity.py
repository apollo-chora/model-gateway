"""Trusted identity of an inbound caller (PRD sections 5.1 and 7).

Inbound authentication is credential-based: a caller presents an API key, and
everything the gateway believes about that caller is derived from the key
alone. :class:`Identity` is the result of that derivation — the one
trusted-identity type the rest of the gateway switches on.

Four rules define the type:

- A tenant is a property of the credential, never of the request. ``tenant``
  is ``None`` unless the presented key is explicitly bound to one, and a
  request-body ``tenant_id`` can never supply or override it (PRD section 5.1).
- The credential, not the caller, is the principal. ``credential_id`` is the
  collision-resistant internal identifier downstream policy keys on — rate
  limiting, concurrency limits, idempotency and accounting all use it, and
  nothing a caller sends may stand in for it.
- ``key_label`` is a display string, not a principal. It is a deliberately
  truncated digest, for correlating log lines only; see
  :mod:`model_gateway.security.api_keys` for why it must not be relied on as a
  secret or used as a key.
- Caller-supplied ``actor_id`` and ``agent_id`` are attribution labels, never
  principals. They record who or what the caller says it is, and they must
  never key a quota, a rate limit, a concurrency limit or an authorization
  decision. Only ``tenant`` and ``credential_id`` may (PRD section 5.1).
"""

from dataclasses import dataclass

__all__ = ["Identity"]


@dataclass(frozen=True)
class Identity:
    """Who the gateway believes the caller is, derived from the credential only.

    ``authenticated``
        ``True`` only after a presented key matched a configured one. The
        anonymous identity — assigned when authentication is disabled — is the
        only ``authenticated=False`` identity, and it carries no tenant, no
        credential identifier and no key label. ``__post_init__`` enforces
        both directions of this: an authenticated identity must carry a
        credential identifier and a key label, and an anonymous one must carry
        none of the three, so no transport can construct a contradictory
        form.

    ``tenant``
        The tenant the caller's credential is bound to, or ``None`` when the
        configured key binds to no tenant. An authenticated caller with
        ``tenant is None`` is authenticated but tenant-less; an
        unauthenticated caller has NO tenant by construction, and no
        request-body ``tenant_id`` can supply one (PRD section 5.1).

    ``credential_id``
        The collision-resistant identifier of the presented credential — the
        full-length digest of the configured key — for the gateway's own use.
        Downstream policy (rate limiting, concurrency, idempotency, accounting)
        uses it as the principal that names the credential, so it must never be
        a truncated display value. It is internal gateway identity: never
        exposed in public API responses or ordinary logs, while trusted
        internal persistence and policy components may store and use it —
        durable accounting may write it to a database, distributed rate
        limiting may carry it through a shared store, and tracing or audit
        records may include it. It is an unsalted deterministic digest, not a
        credential: it is never accepted as authentication material, but
        possessing it permits offline candidate testing of the same kind the
        short label allows, with definitive verification rather than prefix
        matching. It is therefore sensitive internal data. A keyed derivation
        would avoid that exposure, at the cost of key-management and rotation
        obligations that are not warranted now. ``None`` for the anonymous
        identity.

    ``key_label``
        A short, non-secret label for the presented key, for logs and display
        only. It is a truncated digest (see :class:`ApiKeySet`) and is
        deliberately NOT collision-resistant, so it must never be used as a
        principal: use ``credential_id`` for that. ``None`` for the anonymous
        identity.
    """

    authenticated: bool
    tenant: str | None
    credential_id: str | None
    key_label: str | None

    def __post_init__(self) -> None:
        """Reject the two identity states that contradict what the fields mean.

        An authenticated identity is derived from a matched credential, so it
        must carry that credential's identifier and label; an anonymous
        identity is derived from nothing, so it must carry no tenant, no
        credential identifier and no key label. Both forms are refused here so
        that no transport — HTTP or gRPC — can construct one.
        """
        if self.authenticated:
            if self.credential_id is None or self.key_label is None:
                raise ValueError(
                    "an authenticated identity must carry the credential identifier and key "
                    "label of the matched credential; an authenticated caller with no "
                    "credential is not a state the gateway can represent"
                )
        elif (
            self.tenant is not None
            or self.credential_id is not None
            or self.key_label is not None
        ):
            raise ValueError(
                "an anonymous identity carries no tenant, no credential identifier and no "
                "key label; an unauthenticated caller with a credential is not a state "
                "the gateway can represent"
            )
