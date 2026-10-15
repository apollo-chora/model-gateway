"""Upstream URL policy: deterministic endpoint joining, scheme and SSRF checks.

The endpoint handed to LiteLLM is composed from a configured base URL and an
optional configured path. Composition is literal: the base
URL's path is preserved and the configured path's segments are appended to it,
with no component repaired, reordered or dropped. A full URL in the path
setting is refused unless the route opts in explicitly.

Every URL the gateway is about to contact is validated first: the scheme must
be HTTPS unless insecure transport is explicitly enabled for development, and
every address a host resolves to must be publicly routable. The redirect
decision below — a same-origin hop permitted after revalidation, a
cross-origin hop refused unless the caller opts in, credentials never carried
across an origin change — is specified as policy but has no caller in a
request path: the binding has no redirect-evaluation step, so a redirect is
refused rather than assessed. Credentials must arrive through route
configuration; a URL that embeds userinfo is refused.

Transport contract
------------------
These checks are configuration-time screening only, and a passing URL is not a
guarantee of SSRF prevention. Screening resolves a hostname that the transport
will resolve again, and the two answers can differ, so the component that
performs the outbound request is required to:

- connect to the address that was validated (DNS pinning) instead of
  re-resolving the hostname and connecting wherever it now points, and
- disable automatic redirect following, so that a redirect surfaces instead of
  being followed silently. No hop is evaluated today: the binding has no
  redirect-evaluation step, so every redirect is refused and surfaces to the
  caller as an upstream error, and :func:`evaluate_redirect` has no caller in a
  request path.

Without both obligations met, this module is advisory only.
"""

import ipaddress
import re
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

Resolver = Callable[[str], Sequence[str]]

_SECURE_SCHEME = "https"
_INSECURE_SCHEME = "http"

_DEFAULT_PORTS = {_INSECURE_SCHEME: 80, _SECURE_SCHEME: 443}

RESERVED_AUTH_HEADERS: frozenset[str] = frozenset({"authorization", "x-api-key", "api-key"})

ALLOWED_HOSTS_VARIABLE = "GATEWAY_ALLOWED_UPSTREAM_HOSTS"
"""The variable naming the upstream hosts a deployment permits."""

_HOSTNAME_LABEL = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\Z")

_HOSTNAME_MAXIMUM = 253


def _is_hostname(host: str) -> bool:
    """Whether ``host`` is a DNS hostname or an IPv4 literal.

    Dot-separated labels of letters, digits and hyphens, each not beginning or ending
    with a hyphen, the whole thing at most 253 characters. That is the grammar the
    allowlist states, and an entry outside it is refused rather than kept as text
    that can never match a host the gateway would resolve — which is the failure an
    operator would never see, because a list that quietly kept a typo looks like a
    list that works.
    """
    if not host or len(host) > _HOSTNAME_MAXIMUM:
        return False
    return all(_HOSTNAME_LABEL.match(label) for label in host.split("."))


def parse_allowed_hosts(raw: str | None) -> frozenset[str]:
    """The hostnames an allowlist setting names, or an empty set for none.

    Empty is not "allow nothing": it is "no allowlist is configured", which is the
    behaviour every deployment had before this setting existed. A deployment that
    wants the restriction states one, and only the hosts it names may then be
    configured as an endpoint.

    An entry is a **hostname** and not an authority, so a scheme, a port, a path,
    userinfo, an IPv6 literal and whitespace are each **refused** rather than
    ignored, and so is anything outside the hostname grammar :func:`_is_hostname`
    states. An entry the gateway silently ignored would be an operator believing a
    host was permitted or refused when it was not. Matching is case-insensitive,
    because hostnames are.
    """
    if raw is None:
        return frozenset()
    hosts = {entry.strip().lower() for entry in raw.split(",")}
    hosts.discard("")
    for host in sorted(hosts):
        if any(character in host for character in "/@:[] \t"):
            raise UrlPolicyError(
                f"an entry in {ALLOWED_HOSTS_VARIABLE} is not a hostname; a scheme, a port, a "
                f"path, userinfo and an IPv6 literal are not accepted here"
            )
        if not _is_hostname(host):
            raise UrlPolicyError(
                f"an entry in {ALLOWED_HOSTS_VARIABLE} is not a hostname; labels are letters, "
                f"digits and hyphens, a label does not begin or end with a hyphen, and the whole "
                f"name is at most {_HOSTNAME_MAXIMUM} characters"
            )
    return frozenset(hosts)


class UrlPolicyError(ValueError):
    """An upstream URL violates the gateway's URL policy.

    Messages name the violated rule. They never carry the URL itself, because a
    URL may embed credentials or tokens.
    """


@dataclass(frozen=True)
class RedirectDecision:
    """Whether a redirect may be followed, and whether credentials may go with it."""

    allowed: bool
    forward_credentials: bool
    reason: str


def is_reserved_auth_header(name: str) -> bool:
    """Whether ``name`` is an authentication header a route's own configuration owns.

    Configuration validation uses this to reject reserved names supplied through
    ``*_EXTRA_HEADERS_JSON``.
    """
    return name.strip().lower() in RESERVED_AUTH_HEADERS


def system_resolver(host: str) -> tuple[str, ...]:
    """Resolve ``host`` to every address the operating system reports for it."""
    infos = socket.getaddrinfo(host, None)
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


def is_blocked_address(address: str) -> bool:
    """Whether ``address`` is an address class the gateway must never contact."""
    return _blocked_reason(_parse_address(address, "address")) is not None


def check_host(host: str, *, resolver: Resolver = system_resolver) -> None:
    """Reject a host that resolves to any blocked address.

    Loopback, private, unique-local, link-local (which includes the cloud
    metadata address 169.254.169.254), multicast, reserved, unspecified and
    otherwise non-routable addresses are all refused. Every address the host
    resolves to is checked, not only the literal host string.

    This is configuration-time screening only, and it does not by itself prevent
    SSRF. A hostname screened here is resolved again by the transport, and the
    second answer can differ, so the transport must connect to the address that
    was validated instead (see the transport contract in the module docstring).
    """
    text = host.strip().strip("[]")
    if not text:
        raise UrlPolicyError("URL must include a host")

    literal = _parse_address_or_none(text)
    if literal is not None:
        addresses = (str(literal),)
    else:
        addresses = tuple(resolver(text))
        if not addresses:
            raise UrlPolicyError(f"host {text!r} did not resolve to any address")

    blocked: list[str] = []
    for address in addresses:
        reason = _blocked_reason(_parse_address(address, f"address resolved for host {text!r}"))
        if reason is not None:
            blocked.append(f"{address} ({reason})")
    if blocked:
        raise UrlPolicyError(f"host {text!r} resolves to a blocked address: {', '.join(blocked)}")


def check_allowed_host(host: str, allowed_hosts: frozenset[str]) -> None:
    """Reject a host an allowlist does not name.

    **The allowlist narrows and never widens.** A host it names still has to pass
    every other check in this module, including the address screen, so configuring
    one cannot make a blocked address reachable — it is a second, operator-owned
    statement about which upstreams this deployment talks to, not an exemption from
    the first. That is the property that makes it worth having: an operator who wants
    the gateway to reach exactly two providers can say so, and a route added later
    that reaches a third fails the start rather than being discovered in a bill or an
    incident.

    An empty allowlist is no allowlist: every host the rest of this module permits
    stays permitted.
    """
    if not allowed_hosts:
        return
    if host.strip().strip("[]").lower() not in allowed_hosts:
        raise UrlPolicyError(
            f"the endpoint host {host!r} is not one this deployment permits; "
            f"{ALLOWED_HOSTS_VARIABLE} names {len(allowed_hosts)} host(s) and this is not one "
            f"of them"
        )


def join_endpoint(
    base_url: str,
    path: str | None = None,
    *,
    allow_absolute_path_url: bool = False,
) -> str:
    """Compose an endpoint from a base URL and an optional configured path.

    The base URL may carry a path of its own, and the configured path's segments
    are appended to it literally: ``/v1`` joined with ``/custom`` gives
    ``/v1/custom``, and ``/v1`` joined with ``/v1/custom`` gives
    ``/v1/v1/custom``. Nothing is repaired or guessed at, because a repeated
    component is meaningful to some APIs (``/api/models`` joined with
    ``/models/generate``) and is operator error elsewhere.

    Nothing is discarded either: a base path always survives joining, and a base
    URL or path carrying a query string, a fragment or ``.``/``..`` segments is
    rejected instead of being trimmed.

    Empty segments and trailing slashes are normalised away, so the result never
    ends in ``/`` and never contains ``//``.

    An absolute URL in ``path`` is refused unless ``allow_absolute_path_url``
    is set, in which case it replaces the base URL entirely.
    """
    base = _parse_url(base_url, what="endpoint")
    text = (path or "").strip()
    if not text:
        return base.rebuild()

    if _is_absolute(text):
        if not allow_absolute_path_url:
            raise UrlPolicyError(
                "path setting is an absolute URL; the route's absolute-path-URL opt-in is required"
            )
        return _parse_url(text, what="path setting").rebuild()

    relative = urlsplit(text)
    if relative.query or relative.fragment:
        raise UrlPolicyError("path setting must not carry a query string or fragment")
    segments = _path_segments(relative.path, what="path setting")
    return base.rebuild(path_segments=base.path_segments + segments)


def validate_upstream_url(
    url: str,
    *,
    allow_insecure_transport: bool = False,
    resolver: Resolver = system_resolver,
) -> None:
    """Apply scheme policy and SSRF checks to an absolute upstream URL.

    Configuration-time screening only, not a guarantee of SSRF prevention: the
    checks run once, against a DNS answer the transport will obtain again. The
    transport that performs the outbound request is required to pin the
    validated address and to disable automatic redirect following, so that a
    redirect surfaces instead of being followed silently. No hop is evaluated:
    the binding refuses redirects rather than assessing them (see the module
    docstring).
    """
    parsed = _parse_url(url, what="URL")
    _check_scheme(parsed.scheme, allow_insecure_transport=allow_insecure_transport)
    check_host(parsed.host, resolver=resolver)


def resolve_endpoint(
    base_url: str,
    path: str | None = None,
    *,
    allow_absolute_path_url: bool = False,
    allow_insecure_transport: bool = False,
    allowed_hosts: frozenset[str] = frozenset(),
    resolver: Resolver = system_resolver,
) -> str:
    """Join a route's endpoint and validate it before it is handed to LiteLLM.

    The result is the ``api_base`` value the binding passes to LiteLLM. Its one
    caller is the
    route parser, which is where configuration is validated: the binding receives a
    resolved ``Route`` rather than resolving one, so this is configuration-time
    screening and not a per-request step. ``allowed_hosts`` is the deployment's
    optional allowlist, applied to the joined endpoint so an absolute-path override
    is checked against it too.
    """
    endpoint = join_endpoint(base_url, path, allow_absolute_path_url=allow_absolute_path_url)
    validate_upstream_url(
        endpoint,
        allow_insecure_transport=allow_insecure_transport,
        resolver=resolver,
    )
    check_allowed_host(urlsplit(endpoint).hostname or "", allowed_hosts)
    return endpoint


def evaluate_redirect(
    source_url: str,
    target_url: str,
    *,
    allow_cross_origin: bool = False,
    allow_insecure_transport: bool = False,
    resolver: Resolver = system_resolver,
) -> RedirectDecision:
    """Decide whether a redirect from ``source_url`` to ``target_url`` may proceed.

    The destination is revalidated as a fresh upstream URL, so a redirect can
    never reach a blocked address or downgrade to plain HTTP. A redirect that
    stays within one origin proceeds and keeps the route's credentials.

    A redirect that changes scheme, host or port is refused unless
    ``allow_cross_origin`` is set. The request being redirected is a POST that
    may carry custom authentication headers, provider or account headers, and a
    body holding private prompts and tool payloads, so deciding which of those
    may cross an origin is riskier than refusing the hop. When the opt-in is set
    the hop proceeds, but credentials are never forwarded across the origin
    change.

    No request path calls this helper. It states the policy for controlled
    redirect handling, which is not yet implemented: the binding has no
    redirect-evaluation step, so a transport that disables automatic redirect
    following returns the 3xx to the caller unchanged and the hop is refused
    rather than assessed.

    ``source_url`` is the URL the gateway already validated, so a malformed
    source raises :class:`UrlPolicyError`; a malformed target is a refusal.
    """
    source = _parse_url(source_url, what="source URL")
    try:
        target = _parse_url(target_url, what="redirect target")
        _check_scheme(target.scheme, allow_insecure_transport=allow_insecure_transport)
        check_host(target.host, resolver=resolver)
    except UrlPolicyError as error:
        return RedirectDecision(allowed=False, forward_credentials=False, reason=str(error))

    if target.origin == source.origin:
        return RedirectDecision(allowed=True, forward_credentials=True, reason="same origin")
    if not allow_cross_origin:
        return RedirectDecision(
            allowed=False,
            forward_credentials=False,
            reason="redirect changes origin; cross-origin redirects require an explicit opt-in",
        )
    return RedirectDecision(
        allowed=True,
        forward_credentials=False,
        reason="origin changed; credentials are not forwarded",
    )


@dataclass(frozen=True)
class _Url:
    scheme: str
    netloc: str
    host: str
    port: int
    path_segments: tuple[str, ...]

    @property
    def origin(self) -> tuple[str, str, int]:
        return (self.scheme, self.host, self.port)

    def rebuild(self, path_segments: tuple[str, ...] | None = None) -> str:
        segments = self.path_segments if path_segments is None else path_segments
        path = "/" + "/".join(segments) if segments else ""
        return f"{self.scheme}://{self.netloc}{path}"


def _parse_url(url: str, *, what: str) -> _Url:
    text = url.strip()
    if not text:
        raise UrlPolicyError(f"{what} is empty")

    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        raise UrlPolicyError(f"{what} must be an absolute URL with a scheme and host")
    if parts.query or parts.fragment:
        raise UrlPolicyError(f"{what} must not carry a query string or fragment")
    if parts.username is not None or parts.password is not None:
        raise UrlPolicyError(
            f"{what} must not embed credentials; supply them through route configuration"
        )

    host = parts.hostname
    if not host:
        raise UrlPolicyError(f"{what} must include a host")
    try:
        port = parts.port
    except ValueError as error:
        raise UrlPolicyError(f"{what} has an invalid port") from error

    return _Url(
        scheme=parts.scheme,
        netloc=parts.netloc,
        host=host,
        port=port or _DEFAULT_PORTS.get(parts.scheme, 0),
        path_segments=_path_segments(parts.path, what=what),
    )


def _path_segments(path: str, *, what: str) -> tuple[str, ...]:
    segments: list[str] = []
    for segment in path.split("/"):
        if not segment:
            continue
        if segment in {".", ".."}:
            raise UrlPolicyError(f"{what} must not contain '.' or '..' path segments")
        segments.append(segment)
    return tuple(segments)


def _is_absolute(text: str) -> bool:
    return text.startswith("//") or bool(urlsplit(text).scheme)


def _check_scheme(scheme: str, *, allow_insecure_transport: bool) -> None:
    if scheme == _SECURE_SCHEME:
        return
    if scheme == _INSECURE_SCHEME:
        if allow_insecure_transport:
            return
        raise UrlPolicyError(
            "plain http is only allowed when insecure transport is explicitly enabled"
        )
    raise UrlPolicyError(f"unsupported URL scheme {scheme!r}; https is required")


def _parse_address_or_none(
    address: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(address)
    except ValueError:
        return None


def _parse_address(
    address: str,
    what: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    parsed = _parse_address_or_none(address.strip())
    if parsed is None:
        raise UrlPolicyError(f"{what} is not an IP address")
    return parsed


def _blocked_reason(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link-local"
    if address.is_multicast:
        return "multicast"
    if address.is_unspecified:
        return "unspecified"
    if address.is_reserved:
        return "reserved"
    if address.is_private:
        return "private or unique-local"
    if not address.is_global:
        return "not globally routable"
    return None
