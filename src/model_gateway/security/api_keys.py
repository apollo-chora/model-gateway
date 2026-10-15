"""Inbound credential store: API-key parsing, verification and identity.

This module owns the gateway's inbound API keys: the format
of ``GATEWAY_API_KEYS``, how a presented key is checked, and the
:class:`~model_gateway.security.Identity` the check produces. It is the only
place a configured key is compared against a presentation.

Key format
----------
``GATEWAY_API_KEYS`` is a comma-separated list of entries; the settings layer
splits the environment value on commas and trims each element, so every
element of ``Settings.api_keys`` is one whole entry, either ``<key>`` or
``<key>:<tenant>``. Surrounding whitespace is trimmed on the entry and on
both halves, and blank entries are dropped. The rest is rejected at startup,
because ambiguous configuration must be refused rather than resolved by
guessing:

- an entry with an empty key part (``:tenant``);
- an entry with an empty tenant part (``key:``) — it reads as a tenant
  binding the operator started and did not finish, and silently reading it as
  tenant-less would hide the mistake;
- an entry with more than one ``:`` — the format is ``<key>`` or
  ``<key>:<tenant>`` and nothing else;
- a key appearing twice, with the same or a different tenant — which binding
  was intended is unknowable, and the duplicate would otherwise decide which
  tenant a presentation yields;
- ``GATEWAY_AUTH_ENABLED=true`` with no keys configured — every request
  would be rejected with a valid key none the less.

Configuration-time keys are plaintext secrets
---------------------------------------------
The keys live in the operator's environment as plaintext: that is the
configuration model, not a defect this module can remove.
At startup each configured key is reduced to its SHA-256 digest and the
plaintext is discarded, so the store holds a digest rather than the key for the
process lifetime. The gateway never logs, echoes or serialises a key — error
messages name the rule that was violated and never the key that violated it,
and the store's repr shows only derived labels. Secret redaction and the rule
that credentials are never logged apply to everything in this module.

Verification
------------
A presentation is hashed to a fixed-length SHA-256 digest and compared against
every configured key's precomputed digest with :func:`hmac.compare_digest`, and
the scan runs to the last key whether or not an earlier one matched. Because the
scan never returns early, the number of comparisons does not depend on which key
matched: the match position is not revealed. Every comparison is between two
256-bit digests, so the work a comparison does does not depend on the length
of the presented key or on ordinary content equality between the digests.

What the scan does NOT hide is the number of configured credentials. Total work
is proportional to that number, so the time an attempt takes still correlates
with how many keys the deployment configures, and an observer who can time the
gateway can estimate that count. Hiding it is deliberately not attempted: the
count is not a secret worth the complexity, because it says nothing about the
credentials themselves and an attacker cannot act on it.

A wrong key is rejected with :class:`~model_gateway.errors.UnauthenticatedError`
carrying ``INVALID_CREDENTIALS``; no credential at all carries
``AUTHENTICATION_REQUIRED``. Neither message echoes the presented key.

``authenticate`` verifies regardless of :attr:`ApiKeySet.enabled`; whether a
request may proceed without authentication is the caller's decision, made by
consulting ``enabled`` and assigning :meth:`ApiKeySet.anonymous`.

``credential_id`` and ``key_label``
-----------------------------------
``Identity`` carries two derived values for a matched key, and they serve
different consumers:

- ``credential_id`` is the full SHA-256 digest of the key: a
  collision-resistant identifier, and the principal downstream policy — rate
  limiting, concurrency limits, idempotency and accounting — keys on.
- ``key_label`` is the first 12 hexadecimal characters of the same digest, a
  short display string for logs. It is deliberately truncated and must never be
  used as a principal: 48 bits can collide at scale, so two different keys could
  share a label. It exists to let an operator correlate log lines, nothing more.

Neither is a credential. A hash prefix is not encryption, and a configured key
may be low entropy: the example keys (``primary-key``,
``internal-tooling``) are guessable, so an attacker holding a logged label can
test candidate keys offline until one hashes to it. The label is therefore a
correlation aid and not a secret to rely on. The operational consequence is that
credentials must be generated with high entropy, and the label must be treated
as public.
"""

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

from model_gateway.config import ConfigurationError, Settings
from model_gateway.errors import ErrorReason, UnauthenticatedError
from model_gateway.security.identity import Identity

__all__ = ["ApiKeySet"]

_KEY_LABEL_LENGTH = 12

_SETTINGS_VARIABLE = "GATEWAY_API_KEYS"

_AUTH_ENABLED_VARIABLE = "GATEWAY_AUTH_ENABLED"


def _digest_for(key: str) -> str:
    """The full-length SHA-256 digest of ``key``, as hexadecimal text.

    Fixed at 64 characters whatever the key's length, so every comparison this
    module performs is between two equal-length strings.
    """
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class _KeyBinding:
    """One configured key and the tenant it is bound to.

    ``digest`` is the full SHA-256 digest of the key, computed once at startup;
    the plaintext key is not retained. It is compared against presentations and
    nothing else, and the repr shows only the truncated ``key_label``.
    """

    digest: str
    tenant: str | None

    @property
    def credential_id(self) -> str:
        """The collision-resistant identifier downstream policy keys on."""
        return self.digest

    @property
    def key_label(self) -> str:
        """The short display label for logs; never a principal."""
        return self.digest[:_KEY_LABEL_LENGTH]

    def __repr__(self) -> str:
        return f"_KeyBinding(tenant={self.tenant!r}, key_label={self.key_label!r})"


@dataclass(frozen=True)
class ApiKeySet:
    """The gateway's configured inbound API keys, and what a presentation yields.

    Built through :meth:`from_settings` so that every construction path has
    passed startup validation: the set either holds unambiguous, unique
    key-to-tenant bindings or was never constructed.
    """

    _bindings: tuple[_KeyBinding, ...]
    _enabled: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "ApiKeySet":
        """Build the store from ``settings``, rejecting ambiguous configuration.

        The bindings come from ``settings.api_keys`` (the settings layer has
        already split the comma-separated environment value), and
        ``enabled`` reflects ``settings.auth_enabled``. A startup rejection
        raises :class:`~model_gateway.config.ConfigurationError`; the
        messages name the violated rule and never the key material.
        """
        if settings.auth_enabled and not settings.api_keys:
            raise ConfigurationError(
                f"{_AUTH_ENABLED_VARIABLE} is true but {_SETTINGS_VARIABLE} is empty; "
                "set one or more comma-separated keys, or set "
                f"{_AUTH_ENABLED_VARIABLE}=false"
            )
        return cls(
            _bindings=_parse_bindings(settings.api_keys),
            _enabled=settings.auth_enabled,
        )

    @property
    def enabled(self) -> bool:
        """Whether inbound authentication is required for this deployment."""
        return self._enabled

    def authenticate(self, presented: str | None) -> Identity:
        """Verify a presented key against every configured key.

        Returns the :class:`Identity` of the binding that matched, with the
        trusted tenant derived from the credential only. Raises
        :class:`~model_gateway.errors.UnauthenticatedError` when no
        credential was presented (``AUTHENTICATION_REQUIRED``) or when the
        presented key matches nothing (``INVALID_CREDENTIALS``). The error
        messages never echo the presented key.

        The presentation is hashed once to a fixed-length digest, and the scan
        compares that digest against every configured key's precomputed digest
        with :func:`hmac.compare_digest`, never returning early on a match. The
        match position therefore does not determine the number of comparisons;
        the number of configured keys is not hidden, and hiding it is
        deliberately not attempted.
        """
        if not presented:
            raise UnauthenticatedError("no API key was presented")

        digest = _digest_for(presented)
        match: _KeyBinding | None = None
        for binding in self._bindings:
            if hmac.compare_digest(digest, binding.digest):
                match = binding
        if match is None:
            raise UnauthenticatedError(
                "the presented API key is not valid",
                reason=ErrorReason.INVALID_CREDENTIALS,
            )
        return Identity(
            authenticated=True,
            tenant=match.tenant,
            credential_id=match.credential_id,
            key_label=match.key_label,
        )

    def anonymous(self) -> Identity:
        """The identity assigned when authentication is disabled.

        Unauthenticated means exactly this and nothing more: the caller has
        NO tenant (``tenant is None`` — a request-body ``tenant_id`` can never
        supply one), no credential identifier and no key label.
        """
        return Identity(authenticated=False, tenant=None, credential_id=None, key_label=None)

    def __repr__(self) -> str:
        return f"ApiKeySet(enabled={self._enabled!r}, bindings={self._bindings!r})"


def _parse_bindings(entries: Iterable[str]) -> tuple[_KeyBinding, ...]:
    """Parse configured entries into bindings, rejecting ambiguous ones.

    Every entry is ``<key>`` or ``<key>:<tenant>`` after trimming; blank
    entries are dropped. Each key is reduced to its digest here, and only the
    digest is kept, so the plaintext never outlives construction. Anything
    unreadable — an empty key half, an empty tenant half, a second separator, a
    key that appears twice — raises
    :class:`~model_gateway.config.ConfigurationError` naming the rule and
    never the key material.
    """
    bindings: list[_KeyBinding] = []
    seen: set[str] = set()
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue

        key_part, separator, tenant_part = entry.partition(":")
        if not key_part.strip():
            raise ConfigurationError(
                f"{_SETTINGS_VARIABLE} contains an entry with an empty key part"
            )
        if not separator:
            key, tenant = entry, None
        else:
            if ":" in tenant_part:
                raise ConfigurationError(
                    f"{_SETTINGS_VARIABLE} contains an entry with more than one "
                    f"separator; the format is <key> or <key>:<tenant>"
                )
            tenant = tenant_part.strip()
            if not tenant:
                raise ConfigurationError(
                    f"{_SETTINGS_VARIABLE} contains an entry with an empty tenant part; "
                    "a key is either plain (<key>) or tenant-bound (<key>:<tenant>)"
                )
            key = key_part.strip()

        digest = _digest_for(key)
        if digest in seen:
            raise ConfigurationError(
                f"{_SETTINGS_VARIABLE} contains a key more than once; each key must "
                "appear exactly once"
            )
        seen.add(digest)
        bindings.append(_KeyBinding(digest=digest, tenant=tenant))
    return tuple(bindings)
