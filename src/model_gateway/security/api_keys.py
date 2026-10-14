"""Inbound credential store: API-key parsing, verification and identity.

This module owns the gateway's inbound API keys (PRD section 7): the format
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
guessing (PRD sections 6 and 11.1):

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
The keys live in the operator's environment as plaintext: that is the PRD's
configuration model (sections 6 and 18), not a defect this module can remove.
They are kept in memory for the process lifetime so that presentations can be
verified, and the gateway never logs, echoes or serialises them — error
messages name the rule that was violated and never the key that violated it,
and the store's repr shows only derived labels. Secret redaction (PRD section
14) and the rule that credentials are never logged (PRD section 17) apply to
everything in this module.

Verification
------------
A presentation is compared against every configured key with
:func:`hmac.compare_digest`, and the scan runs to the last key whether or not
an earlier one matched. Returning on the first match would make response time
reveal which key matched — and, with it, how many keys are configured — so
the number of comparisons is constant for every outcome. A wrong key is
rejected with :class:`~model_gateway.errors.UnauthenticatedError` carrying
``INVALID_CREDENTIALS``; no credential at all carries
``AUTHENTICATION_REQUIRED``. Neither message echoes the presented key.

``authenticate`` verifies regardless of :attr:`ApiKeySet.enabled`; whether a
request may proceed without authentication is the caller's decision, made by
consulting ``enabled`` and assigning :meth:`ApiKeySet.anonymous`.

``key_id``
---------
The label carried on ``Identity.key_id`` is the first 12 hexadecimal
characters of the SHA-256 digest of the UTF-8 key. The digest is one-way, so
the key cannot be recovered from the label; its only purpose is letting logs
and accounting tell presentations of one configured key apart from another.
It is never a credential and must never be used as one.
"""

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

from model_gateway.config import ConfigurationError, Settings
from model_gateway.errors import ErrorReason, UnauthenticatedError
from model_gateway.security.identity import Identity

__all__ = ["ApiKeySet"]

_KEY_ID_LENGTH = 12

_SETTINGS_VARIABLE = "GATEWAY_API_KEYS"

_AUTH_ENABLED_VARIABLE = "GATEWAY_AUTH_ENABLED"


def key_id_for(key: str) -> str:
    """Return the short, non-secret log label for ``key``.

    The first 12 hexadecimal characters of the SHA-256 digest of the UTF-8
    key: one-way, safe to log, and usable only to correlate presentations of
    the same configured key — never to recover the key or to authenticate
    with it.
    """
    return hashlib.sha256(key.encode()).hexdigest()[:_KEY_ID_LENGTH]


@dataclass(frozen=True)
class _KeyBinding:
    """One configured key and the tenant it is bound to.

    ``key`` is plaintext secret material. It is compared against
    presentations and nothing else: it is never logged, echoed or included
    in the repr, which shows only the derived ``key_id``.
    """

    key: str
    tenant: str | None

    @property
    def key_id(self) -> str:
        return key_id_for(self.key)

    def __repr__(self) -> str:
        return f"_KeyBinding(tenant={self.tenant!r}, key_id={self.key_id!r})"


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

        The scan compares against all configured keys with
        :func:`hmac.compare_digest` and never returns early on a match, so
        the comparison count is constant for every outcome.
        """
        if not presented:
            raise UnauthenticatedError("no API key was presented")

        match: _KeyBinding | None = None
        for binding in self._bindings:
            if hmac.compare_digest(presented.encode(), binding.key.encode()):
                match = binding
        if match is None:
            raise UnauthenticatedError(
                "the presented API key is not valid",
                reason=ErrorReason.INVALID_CREDENTIALS,
            )
        return Identity(authenticated=True, tenant=match.tenant, key_id=match.key_id)

    def anonymous(self) -> Identity:
        """The identity assigned when authentication is disabled.

        Unauthenticated means exactly this and nothing more: the caller has
        NO tenant (``tenant is None`` — a request-body ``tenant_id`` can never
        supply one, PRD section 5.1) and no key label.
        """
        return Identity(authenticated=False, tenant=None, key_id=None)

    def __repr__(self) -> str:
        return f"ApiKeySet(enabled={self._enabled!r}, bindings={self._bindings!r})"


def _parse_bindings(entries: Iterable[str]) -> tuple[_KeyBinding, ...]:
    """Parse configured entries into bindings, rejecting ambiguous ones.

    Every entry is ``<key>`` or ``<key>:<tenant>`` after trimming; blank
    entries are dropped. Anything unreadable — an empty key half, an empty
    tenant half, a second separator, a key that appears twice — raises
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

        if key in seen:
            raise ConfigurationError(
                f"{_SETTINGS_VARIABLE} contains a key more than once; each key must "
                "appear exactly once"
            )
        seen.add(key)
        bindings.append(_KeyBinding(key=key, tenant=tenant))
    return tuple(bindings)
