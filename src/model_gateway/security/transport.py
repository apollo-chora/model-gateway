"""The outbound transport: the connection-time half of the URL policy (PRD section 14.1).

Screening a URL when configuration is loaded does not prevent SSRF on its own.
:mod:`~model_gateway.security.url_policy` validates the endpoint the operator
configured, and the binding resolves that hostname again when it performs the
request. Between the two, DNS may answer differently — the time-of-check to
time-of-use gap PRD section 14.1 describes. The controls that close it belong to
whatever opens the socket, which is this module:

* **Address pinning.** :class:`PinnedAddressTransport` resolves the request's
  hostname through the URL policy's resolver, validates every address it gets
  with the URL policy's own blocked-address rules, and connects to a validated
  address instead of re-resolving. An address the policy refuses raises
  :class:`~model_gateway.security.url_policy.UrlPolicyError`, and nothing is
  sent: there is no fall-through to a direct connection.
* **No automatic redirect following.** :func:`build_async_client` returns a
  client built with ``follow_redirects=False``, so a 3xx surfaces to the caller
  as the response it is and is never followed. The gateway has no redirect-
  evaluation step, so the hop is refused rather than assessed and surfaces to
  the caller as an upstream error (PRD section 14.1). This transport neither
  follows nor evaluates a redirect: connecting is its job.

Pinning would be worthless if it changed who the upstream thinks it is talking
to, so the original hostname survives it. The socket goes to the validated
address while the ``Host`` header and the TLS server name — and therefore
certificate validation — stay the hostname the caller asked for.

Why a client must be passed to the binding per call
---------------------------------------------------
LiteLLM builds its own HTTP client when the caller does not supply one, and it
hardcodes ``follow_redirects=True`` while doing so. A client supplied as
``client=`` to ``litellm.acompletion`` is therefore the only way either control
reaches the wire; omitted, redirects are followed automatically and the
connection is made to whatever the hostname resolves to at connect time.
Omitting it loses both controls silently — no error, no warning, and no visible
difference until a rebinding or a redirect is attempted. The same call site must
keep ``num_retries=0``, because a client supplied without it leaves the retry
count at LiteLLM's own default and would stack a second retry loop on the
gateway's (PRD section 11.3).

Which client that is has two answers, because the pinned library does not accept
the same object on every provider path, and the difference is a security
boundary rather than a preference. Both shapes are built from the transport
here; :mod:`model_gateway.providers.litellm_provider` is where one is chosen per
dialect, and only for dialects whose path it has verified:

* The OpenAI SDK path takes an ``openai.AsyncOpenAI``. A bare httpx client is
  rejected by it (the SDK reads ``api_key`` and ``base_url`` off the client it
  is handed), so :func:`build_async_client` supplies the httpx client that the
  SDK object is built around.
* The generic httpx path — the shared ``base_llm_http_handler`` and the
  Anthropic handler both use it — takes an ``AsyncHTTPHandler``. Anything else
  is *silently discarded*: the handler substitutes its own default client, which
  re-resolves the hostname and follows redirects, and raises nothing. That path
  also retries a connection error by rebuilding a client through
  ``AsyncHTTPHandler.create_client``, and the rebuild is constructed from the
  handler's own ``transport`` attribute. :func:`build_pinned_transport` exists
  so the binding can hand the handler the pinned transport itself: every client
  that handler creates, first or rebuilt, then carries the pinning and the
  disabled redirect following, and the retry cannot leave the controls.

A refusal raised here does not arrive at the gateway as its own type. The
provider SDK beneath LiteLLM catches anything its own hierarchy does not name
and re-raises it as a connection error, so the gateway sees a connection failure
whose ``__cause__`` chain carries the ``UrlPolicyError``. The refusal happens
before a socket is opened, which is the property that matters; a caller that must
tell the two apart reads the cause chain.
"""

import ipaddress
import ssl
from math import isfinite

import httpx

from model_gateway.security.url_policy import (
    Resolver,
    UrlPolicyError,
    check_host,
    system_resolver,
)

__all__ = [
    "PinnedAddressTransport",
    "attempt_timeout",
    "build_async_client",
    "build_pinned_transport",
]

_SECURE_SCHEME = "https"
_INSECURE_SCHEME = "http"


class PinnedAddressTransport(httpx.AsyncBaseTransport):
    """An httpx transport that connects only to an address the URL policy accepted.

    Every request is rewritten before it is sent: the URL's host is replaced by
    the validated address the resolver returned, so the socket terminates where
    policy was checked rather than wherever the hostname points at connect time
    (PRD section 14.1). The hostname itself is not lost — it stays in the ``Host``
    header and is handed to the connection as the TLS server name, so the
    authority the upstream sees and the certificate it is validated against are
    unchanged.

    The first validated address is used. There is deliberately no fallback to
    another address the resolver returned: trying a second address after a failed
    connect is exactly the re-resolution this control removes. Every candidate
    was validated anyway, so choosing between them is not a policy decision; an
    operator that needs a different destination changes what the name resolves
    to.

    ``resolver`` is the URL policy's resolver, injected so that tests never touch
    the network and a deployment can resolve through its own stub.
    ``allow_insecure_transport`` mirrors the URL policy's development opt-in and
    is the only way a plain ``http`` request leaves this transport.

    ``inner`` is the transport that performs the socket I/O — by default
    :class:`httpx.AsyncHTTPTransport`, which owns pooling, TLS configuration and
    limits. It is a seam: a test supplies one that answers without a socket, and
    a deployment that needs an option this constructor does not expose supplies
    its own. Pinning is unaffected by which one is used, because it happens
    before the inner transport is called.
    """

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        allow_insecure_transport: bool = False,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._resolver = resolver
        self._allow_insecure_transport = allow_insecure_transport
        self._inner: httpx.AsyncBaseTransport = (
            httpx.AsyncHTTPTransport() if inner is None else inner
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send ``request`` to a validated address, or refuse it before connecting.

        Raises :class:`~model_gateway.security.url_policy.UrlPolicyError` when the
        request's scheme is not acceptable, when the host does not resolve, or
        when any address the host resolves to is one the policy blocks. Nothing
        is sent in that case: the inner transport is not called, so no socket is
        opened.

        A redirect response is returned to the caller unchanged, and the
        destination is not even resolved. This transport follows no redirect and
        evaluates none; the gateway has no redirect-evaluation step, so the hop
        is refused rather than assessed and surfaces to the caller as an
        upstream error (PRD section 14.1).
        """
        return await self._inner.handle_async_request(self._pin(request))

    async def aclose(self) -> None:
        """Close the inner transport, and with it every pooled connection."""
        await self._inner.aclose()

    def _pin(self, request: httpx.Request) -> httpx.Request:
        """``request`` rewritten to a validated address, with its hostname preserved.

        The URL's host is replaced and nothing else about it: scheme, port, path
        and query stay the caller's, because the request is the binding's own
        composition of the route's endpoint and the operation path (PRD section
        8).

        The ``Host`` header is the authority the request already carried, and is
        derived from the original URL only when it carried none — a client-built
        request always carries one. Preserving it is what keeps the authority the
        upstream sees independent of the address the socket reached.

        ``sni_hostname`` is the extension httpcore passes to TLS as the server
        name, so it is the original hostname too and certificate validation is
        performed against the certificate that hostname calls for rather than
        against the address. Without it the TLS layer would be handed the pinned
        address and every certificate would fail to match.
        """
        url = request.url
        _check_scheme(url.scheme, allow_insecure_transport=self._allow_insecure_transport)
        addresses = _validated_addresses(url.host, self._resolver)

        headers = httpx.Headers(request.headers)
        if "host" not in headers:
            headers["host"] = url.netloc.decode("ascii")

        return httpx.Request(
            method=request.method,
            url=url.copy_with(host=addresses[0]),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": url.host},
        )


def attempt_timeout(
    budget: float | None,
    *,
    connect_timeout: float | None = None,
) -> httpx.Timeout:
    """Derive an outbound timeout from one attempt's budget (PRD section 15).

    ``budget`` is the seconds this attempt may use — ``AttemptContext.timeout``,
    drawn from the invocation's single deadline. It becomes every phase of the
    call, so no connect, read, write or pool wait outlives the deadline the
    gateway set. ``None`` means the gateway set no budget for this attempt, which
    is not a budget of zero: the client then carries no deadline of its own.

    ``connect_timeout`` is the operator's connect-phase budget
    (``PROVIDER_CONNECT_TIMEOUT_SECONDS``), capped by the attempt budget so that a
    slow connect cannot consume more time than the attempt has left. When it is
    ``None`` the attempt budget covers the connect phase too, which cannot exceed
    the deadline because it is the deadline.

    The result is the client's default. A per-call ``timeout`` handed to httpx
    replaces it for that call, so a binding that passes one must pass the same
    attempt budget it passes here.
    """
    _positive_seconds(budget, what="budget")
    _positive_seconds(connect_timeout, what="connect timeout")
    if connect_timeout is None:
        return httpx.Timeout(timeout=budget)
    connect = connect_timeout if budget is None else min(connect_timeout, budget)
    return httpx.Timeout(timeout=budget, connect=connect)


def build_pinned_transport(
    *,
    resolver: Resolver = system_resolver,
    allow_insecure_transport: bool = False,
    verify: ssl.SSLContext | bool = True,
    inner: httpx.AsyncBaseTransport | None = None,
) -> PinnedAddressTransport:
    """Build the transport that pins every connection to a validated address.

    This is the control itself, without a client wrapped around it. A binding
    that must hand LiteLLM something other than an httpx client — the generic
    httpx path takes an ``AsyncHTTPHandler``, and that handler rebuilds its
    client from its own ``transport`` attribute — supplies this to it, so every
    client the library creates from then on, first or rebuilt, is pinned and
    does not follow redirects. See the module docstring.

    ``verify`` is httpx's TLS verification setting: ``True`` for the default
    trust store, ``False`` to disable verification, or an
    :class:`ssl.SSLContext` for a private CA
    (``ssl.create_default_context(cafile=...)``). It reaches the inner transport,
    so it does not apply when one is supplied — a supplied transport owns its own
    TLS configuration.
    """
    return PinnedAddressTransport(
        resolver=resolver,
        allow_insecure_transport=allow_insecure_transport,
        inner=inner if inner is not None else httpx.AsyncHTTPTransport(verify=verify),
    )


def build_async_client(
    *,
    budget: float | None,
    connect_timeout: float | None = None,
    resolver: Resolver = system_resolver,
    allow_insecure_transport: bool = False,
    verify: ssl.SSLContext | bool = True,
    inner: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the httpx client one attempt is made through.

    The returned client pins every connection to a validated address, never
    follows a redirect, and carries the attempt's timeout (PRD sections 9, 14.1
    and 15). See the module docstring for why both controls are required, why
    LiteLLM's own client cannot supply them, and which provider paths accept this
    client rather than :func:`build_pinned_transport`.

    One client per attempt: it is cheap, it owns a connection pool, and its
    timeout is this attempt's. Close it when the attempt ends — ``async with
    build_async_client(...) as client:`` — or the pooled connections outlive the
    call (PRD section 15).

    ``verify`` is httpx's TLS verification setting: ``True`` for the default
    trust store, ``False`` to disable verification, or an
    :class:`ssl.SSLContext` for a private CA
    (``ssl.create_default_context(cafile=...)``). It reaches the inner transport,
    so it does not apply when one is supplied — a supplied transport owns its own
    TLS configuration.

    No proxy is configured and none is taken from the environment. A proxy that
    re-resolves the hostname independently recreates the very gap this client
    closes, so it cannot be the compensating control PRD section 14.1 describes;
    supplying the transport explicitly is what keeps the environment from
    installing one between the gateway and the upstream.
    """
    return httpx.AsyncClient(
        transport=build_pinned_transport(
            resolver=resolver,
            allow_insecure_transport=allow_insecure_transport,
            verify=verify,
            inner=inner,
        ),
        timeout=attempt_timeout(budget, connect_timeout=connect_timeout),
        follow_redirects=False,
        trust_env=False,
    )


def _resolve(host: str, resolver: Resolver) -> tuple[str, ...]:
    """Every address ``host`` resolves to, without resolving a literal address.

    A literal address is its own answer, exactly as the URL policy treats it, so
    an endpoint configured as an IP is never looked up and cannot be made to mean
    something else by a resolver.
    """
    text = host.strip().strip("[]")
    if not text:
        raise UrlPolicyError("URL must include a host")
    try:
        return (str(ipaddress.ip_address(text)),)
    except ValueError:
        return tuple(resolver(text))


def _validated_addresses(host: str, resolver: Resolver) -> tuple[str, ...]:
    """The addresses ``host`` resolved to, once the URL policy has accepted them.

    Resolution happens exactly once and the answers are handed to the policy's own
    :func:`~model_gateway.security.url_policy.check_host`, so the verdict and the
    address this transport connects to cannot disagree: the addresses validated
    are the addresses used. One blocked answer is enough to refuse the request,
    even beside a public one, and the error names the address class rather than
    the URL.
    """
    addresses = _resolve(host, resolver)
    check_host(host, resolver=lambda _host: addresses)
    return addresses


def _check_scheme(scheme: str, *, allow_insecure_transport: bool) -> None:
    """Refuse a scheme this transport will not speak (PRD section 14).

    The URL policy applies the same rule when an endpoint is configured. It is
    applied again here because this is the last point before a socket, and a
    request URL that never passed configuration screening reaches the transport
    too — a redirect destination, or a URL composed by the binding.
    """
    if scheme == _SECURE_SCHEME:
        return
    if scheme == _INSECURE_SCHEME:
        if allow_insecure_transport:
            return
        raise UrlPolicyError(
            "plain http is only allowed when insecure transport is explicitly enabled"
        )
    raise UrlPolicyError(f"unsupported URL scheme {scheme!r}; https is required")


def _positive_seconds(value: float | None, *, what: str) -> None:
    """Refuse a value that is not a usable number of seconds."""
    if value is None:
        return
    usable = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and isfinite(value)
        and value > 0
    )
    if not usable:
        raise ValueError(f"{what} must be a positive number of seconds or None, got {value!r}")
