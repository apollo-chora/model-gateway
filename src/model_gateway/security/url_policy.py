"""Upstream URL policy: deterministic endpoint joining, scheme and SSRF checks.

The endpoint handed to LiteLLM is composed from a configured base URL and an
optional configured path (PRD section 8). Composition is additive: the base
URL's own path is preserved, and segments the configured path re-states are
collapsed instead of duplicated. A full URL in the path setting is refused
unless the route opts in explicitly.

Every URL the gateway is about to contact is validated first: the scheme must
be HTTPS unless insecure transport is explicitly enabled for development, and
every address a host resolves to must be publicly routable (PRD sections 9.2
and 14). Redirects are revalidated, and a route's credentials never cross an
origin change (PRD section 7).
"""

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

Resolver = Callable[[str], Sequence[str]]

_SECURE_SCHEME = "https"
_INSECURE_SCHEME = "http"

_DEFAULT_PORTS = {_INSECURE_SCHEME: 80, _SECURE_SCHEME: 443}

RESERVED_AUTH_HEADERS: frozenset[str] = frozenset({"authorization", "x-api-key", "api-key"})


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
    ``*_EXTRA_HEADERS_JSON`` (PRD section 7).
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
    """Reject a host that resolves to any blocked address (PRD sections 9.2 and 14).

    Loopback, private, unique-local, link-local (which includes the cloud
    metadata address 169.254.169.254), multicast, reserved, unspecified and
    otherwise non-routable addresses are all refused. Every address the host
    resolves to is checked, not only the literal host string.
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


def join_endpoint(
    base_url: str,
    path: str | None = None,
    *,
    allow_absolute_path_url: bool = False,
) -> str:
    """Compose an endpoint from a base URL and an optional configured path (PRD section 8).

    The base URL may carry a path of its own, and the configured path is joined
    to it. Segments the configured path repeats from the end of the base path
    are collapsed, so ``/v1`` is never duplicated and an operation path is never
    appended twice. Path components are never discarded silently: a base path
    survives joining, and a base URL or path carrying a query string, a fragment
    or ``.``/``..`` segments is rejected instead of being trimmed.

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
    return base.rebuild(path_segments=_merge_paths(base.path_segments, segments))


def validate_upstream_url(
    url: str,
    *,
    allow_insecure_transport: bool = False,
    resolver: Resolver = system_resolver,
) -> None:
    """Apply scheme policy and SSRF checks to an absolute upstream URL (PRD section 14)."""
    parsed = _parse_url(url, what="URL")
    _check_scheme(parsed.scheme, allow_insecure_transport=allow_insecure_transport)
    check_host(parsed.host, resolver=resolver)


def resolve_endpoint(
    base_url: str,
    path: str | None = None,
    *,
    allow_absolute_path_url: bool = False,
    allow_insecure_transport: bool = False,
    resolver: Resolver = system_resolver,
) -> str:
    """Join a route's endpoint and validate it before it is handed to LiteLLM.

    This is the entry point configuration validation and the LiteLLM binding
    use: the result is the ``api_base`` value of PRD section 9.4.
    """
    endpoint = join_endpoint(base_url, path, allow_absolute_path_url=allow_absolute_path_url)
    validate_upstream_url(
        endpoint,
        allow_insecure_transport=allow_insecure_transport,
        resolver=resolver,
    )
    return endpoint


def evaluate_redirect(
    source_url: str,
    target_url: str,
    *,
    allow_insecure_transport: bool = False,
    resolver: Resolver = system_resolver,
) -> RedirectDecision:
    """Decide whether a redirect from ``source_url`` to ``target_url`` may proceed.

    The destination is revalidated as a fresh upstream URL, so a redirect can
    never reach a blocked address or downgrade to plain HTTP. Credentials are
    forwarded only within one origin: a redirect that changes scheme, host or
    port follows without them (PRD sections 7 and 14).

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


def _merge_paths(
    base_segments: tuple[str, ...],
    path_segments: tuple[str, ...],
) -> tuple[str, ...]:
    for size in range(min(len(base_segments), len(path_segments)), 0, -1):
        if base_segments[-size:] == path_segments[:size]:
            return base_segments + path_segments[size:]
    return base_segments + path_segments


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
