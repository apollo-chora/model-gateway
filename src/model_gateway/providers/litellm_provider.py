"""The LiteLLM text provider: one gateway attempt through the pinned binding (PRD section 9).

This is the production :class:`~model_gateway.providers.base.TextProvider`. It
adds execution to the transport-agnostic translation of
:mod:`model_gateway.providers.litellm_translation`: build the call's parameters,
perform one gateway attempt through ``litellm.acompletion`` with the safe
outbound client injected, and normalise the answer into the canonical
response. What it owns is PRD section 9's boundary and nothing more — routing,
retry policy, accounting, identity and the public contract stay in the gateway
core (:mod:`model_gateway.providers.base`).

Controls, and why each is there
-------------------------------
``num_retries=0``
    Set by :func:`~model_gateway.providers.litellm_translation.to_litellm_params`
    and never overridden here. It is mandatory, not defensive: it suppresses both
    the LiteLLM-level retry loop and the OpenAI-SDK retry loop, which is what
    leaves the gateway core as the only retry owner (PRD section 11.3). The
    injected client is *not* a substitute: passing
    ``client=AsyncOpenAI(max_retries=0)`` without ``num_retries=0`` silently
    resets the SDK's ``max_retries`` to 2, so retries return unless the
    parameter travels beside the client.

The injected client
    A per-call ``client=`` kwarg, built from the transport's own pieces
    (:mod:`model_gateway.security.transport`). The connection is the binding's
    to make, so it is the binding's to make safe (PRD section 14.1): the client
    pins the connection to the address the gateway validated and disables
    automatic redirect following, which LiteLLM otherwise forces on
    (``follow_redirects=True`` is hardcoded in the client it builds for itself).
    A per-call ``transport=`` kwarg cannot do this — LiteLLM treats it as a body
    parameter — so client injection is the only mechanism that reaches the wire.
    Which object has to be injected is not one question but one per HTTP path,
    and a dialect whose path is unverified is refused rather than run: see "The
    HTTP path each dialect takes" below. One client per attempt, closed when the
    attempt ends: the binding owns no client lifecycle beyond the call it is
    making.

The attempt timeout
    ``context.timeout`` is passed through as the call's ``timeout`` — the
    budget the gateway set for this attempt (PRD section 9.6). ``None`` means
    the gateway set no budget, and the binding passes the absence through
    rather than inventing one of its own.

Error mapping
    Every exception the call raises is mapped by
    :func:`~model_gateway.providers.litellm_translation.map_litellm_error` and
    re-raised, so a caller never sees a provider-shaped error and no upstream
    message, URL or credential reaches the public payload (PRD sections 5.4
    and 9). A refusal raised by the transport itself — a host that resolves to
    a blocked address at connect time — arrives as a connection failure whose
    cause chain carries the
    :class:`~model_gateway.security.url_policy.UrlPolicyError`, and is mapped as
    the connection failure it was reported as. Cancellation is the deliberate
    boundary of that rule:
    :exc:`asyncio.CancelledError` and the other ``BaseException``s propagate
    unmapped, because PRD section 15 requires the gateway to propagate
    cancellation rather than report it as a gateway failure.

The HTTP path each dialect takes
-------------------------------
Which client LiteLLM honours is decided by the path its ``acompletion`` call
takes for that provider dialect, and the paths disagree. Every entry below was
determined against the pinned ``litellm-core`` 1.106.0rc1, and each is a claim
about which client the call actually uses, so each is proved by a test in
:mod:`tests.test_litellm_client_injection` rather than asserted here.

``openai``
    The OpenAI SDK path (``litellm/llms/openai/openai.py``). ``client=`` must be
    an ``openai.AsyncOpenAI``: the handler reads ``api_key`` and ``base_url`` off
    the object it is given, so a bare httpx client fails the call before any
    request is made. The binding constructs that SDK client around the httpx
    client of
    :func:`~model_gateway.security.transport.build_async_client`, and the SDK
    never replaces the ``http_client`` it was built with, so the pinning and the
    disabled redirect following hold for every request the SDK makes.

``anthropic``, ``openrouter``
    The generic httpx path: the shared handler in
    ``litellm/llms/custom_httpx/llm_http_handler.py`` for OpenRouter, and
    ``litellm/llms/anthropic/chat/handler.py`` for the Messages API. ``client=``
    must be an ``AsyncHTTPHandler``, and anything else is *silently discarded* —
    the handler substitutes its own default client, which re-resolves the
    hostname at connect time and follows redirects, and raises nothing. This
    path also retries a connection error by rebuilding a client from the
    handler's own ``transport`` attribute, which is why the binding hands it the
    pinned transport itself
    (:func:`~model_gateway.security.transport.build_pinned_transport`) rather
    than a client wrapped around one: every client that handler creates, first
    or rebuilt, carries the pinning and the disabled redirect following.

Everything else
    Refused, at the attempt and at startup. A dialect outside
    :data:`SUPPORTED_DIALECTS` is refused with
    :class:`~model_gateway.errors.FailedPreconditionError` rather than run on a
    path whose client handling has not been verified, because a path that
    discards the injected client loses both controls with no error to notice.
    Adding a dialect means adding a verified entry to :data:`SUPPORTED_DIALECTS`
    together with the test that proves its path.

What is still not closed
------------------------
The generic path's connection-error retry is a second upstream request that no
supported knob suppresses — ``num_retries=0`` does not reach it — so
``attempt_count`` may still understate what reached the upstream, and a request
the binding retried internally may have been partially processed, or billed,
while the gateway records a single attempt (PRD section 11.3). It is no longer a
transport escape: both attempts go through the pinned, non-redirecting
transport, and the retry cannot reach an address or an origin the first attempt
was refused. An independently enforced egress restriction remains the
compensating control PRD section 14.1 describes for any path outside the table
above.

The ``.env`` guard
-----------------
Importing ``litellm`` calls ``dotenv.load_dotenv()`` at import time while
``LITELLM_MODE`` is ``DEV``, its default: any ``.env`` in the working directory
is loaded into ``os.environ``, and
:meth:`model_gateway.config.settings.Settings.from_env` would then read
configuration the operator never set for this process. The mode is gateway-
owned implementation state, not operator configuration, and it is *forced* to
``PRODUCTION`` by :mod:`model_gateway.providers._litellm` — the seam this
module imports the library through, and the earliest point any importer of the
binding can reach — so no ambient value, not even an explicit operator
``LITELLM_MODE=DEV``, can re-enable the load and the process environment stays
the only configuration source (PRD section 6). The mode's only other effect is
on LiteLLM's proxy CLI, which this gateway never runs. The test session's own
copy of the guard, needed because the binding tests import ``litellm``
directly, is :file:`tests/conftest.py`.
"""

import ssl
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Final

import httpx

from model_gateway.config.routes import Route
from model_gateway.domain import InvocationRequest, InvocationResponse
from model_gateway.errors import FailedPreconditionError, GatewayError
from model_gateway.providers._litellm import AsyncHTTPHandler, litellm, openai
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.litellm_translation import (
    from_litellm_response,
    map_litellm_error,
    to_litellm_params,
)
from model_gateway.security.transport import (
    attempt_timeout,
    build_async_client,
    build_pinned_transport,
)
from model_gateway.security.url_policy import Resolver, system_resolver

__all__ = ["SUPPORTED_DIALECTS", "LiteLLMProvider", "ensure_supported_dialect"]

_OPENAI_SDK_DIALECTS: Final[frozenset[str]] = frozenset({"openai"})
_HTTP_HANDLER_DIALECTS: Final[frozenset[str]] = frozenset({"anthropic", "openrouter"})

SUPPORTED_DIALECTS: Final[frozenset[str]] = _OPENAI_SDK_DIALECTS | _HTTP_HANDLER_DIALECTS
"""The provider dialects this binding has verified end to end.

Each member's ``acompletion`` path was determined against the pinned
``litellm-core`` and has a test proving that the client the binding injects is
the client the call uses. Membership is not a claim about what the upstream
supports: it is a claim about which HTTP path the binding can keep behind the
transport controls of PRD section 14.1.
"""


def ensure_supported_dialect(dialect: str) -> str:
    """Return ``dialect`` when its HTTP path is verified, and refuse it otherwise.

    A dialect whose path has not been verified may be one whose handler discards
    the injected client and substitutes its own, which re-resolves the hostname
    at connect time and follows redirects with nothing raised to notice. That is
    a silent loss of the controls PRD section 14.1 requires, so it is refused
    rather than attempted: the refusal is the deployment's signal that the route
    is configured with a dialect the binding cannot keep safe, and it names the
    dialects it can. The name is configuration, not a secret, so it is safe in a
    caller-visible message.
    """
    if dialect in SUPPORTED_DIALECTS:
        return dialect
    verified = ", ".join(repr(name) for name in sorted(SUPPORTED_DIALECTS))
    raise FailedPreconditionError(
        f"the inference binding cannot run the {dialect!r} provider dialect: its HTTP path has "
        f"not been verified to honour the injected client, and a path that discards it "
        f"re-resolves the hostname and follows redirects with nothing raised. Verified "
        f"dialects are {verified}"
    )


class _AttemptClient:
    """One attempt's injected client, and the lifecycle that closes it.

    LiteLLM must receive the client itself, and for the generic httpx path that
    client is an ``AsyncHTTPHandler`` whose own ``__aenter__`` yields the inner
    httpx client instead of the handler. Keeping the injected object and the
    context manager in one object is what lets ``async with`` close exactly the
    client LiteLLM was handed.
    """

    __slots__ = ("_client", "_close")

    def __init__(self, client: Any, close: Callable[[], Awaitable[None]]) -> None:
        self._client = client
        self._close = close

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *exc_info: object) -> None:
        await self._close()


def _openai_sdk_client(
    route: Route,
    budget: float | None,
    resolver: Resolver,
    verify: ssl.SSLContext | bool,
    inner_factory: Callable[[], httpx.AsyncBaseTransport],
) -> _AttemptClient:
    """The client the OpenAI SDK path is given: an SDK client over the pinned httpx client.

    The SDK object is what ``client=`` must carry on this path, and the httpx
    client it is built around is the one that owns the connection, so the
    transport controls live there. ``max_retries=0`` is set here as well as
    through ``num_retries=0`` because the SDK retries with the same httpx client
    and the gateway owns the only retry loop (PRD section 11.3).

    ``resolver``, ``verify`` and ``inner_factory`` are this attempt's transport
    inputs (see :class:`LiteLLMProvider`): they choose which policy-checked
    address is connected to, which trust store validates the certificate and
    which transport performs the socket I/O. Whether the connection is pinned
    and redirects are refused is not theirs to decide — the httpx client is
    built by :func:`~model_gateway.security.transport.build_async_client`
    around the pinned transport whatever they are.
    """
    client = openai.AsyncOpenAI(
        api_key=route.api_key,
        base_url=route.base_url,
        http_client=build_async_client(
            budget=budget,
            resolver=resolver,
            verify=verify,
            inner=inner_factory(),
        ),
        timeout=attempt_timeout(budget),
        max_retries=0,
    )
    return _AttemptClient(client, client.close)


def _http_handler_client(
    budget: float | None,
    resolver: Resolver,
    verify: ssl.SSLContext | bool,
    inner_factory: Callable[[], httpx.AsyncBaseTransport],
) -> _AttemptClient:
    """The client the generic httpx path is given, carrying the pinned transport itself.

    ``AsyncHTTPHandler`` builds its client from its own ``transport`` attribute,
    and rebuilds it the same way when it retries a connection error, so handing
    it the pinned transport is what keeps that retry inside the controls.
    ``follow_redirects`` is ``False`` because the handler's default is the
    library's own ``True``.

    The pinned transport is
    :func:`~model_gateway.security.transport.build_pinned_transport` — never a
    caller-supplied one — built from this attempt's ``resolver``, ``verify``
    and ``inner_factory``, so the client the handler creates, first or
    rebuilt, cannot leave the controls.
    """
    handler = AsyncHTTPHandler(
        timeout=attempt_timeout(budget),
        transport=build_pinned_transport(
            resolver=resolver,
            verify=verify,
            inner=inner_factory(),
        ),
        follow_redirects=False,
    )
    return _AttemptClient(handler, handler.close)


class LiteLLMProvider:
    """The production text provider: translation plus one execution (PRD section 9).

    One gateway attempt in, one canonical answer out. The provider performs one
    gateway attempt for the route it is handed — a single invocation of the
    pinned binding — and applies the attempt's own budget to it; it does not
    retry, does not choose between routes and knows nothing about fallbacks —
    those are the gateway core's (PRD sections 9.2 and 11.3). Every control the
    call carries — ``num_retries=0``, the injected client, the attempt timeout,
    the error mapping — is documented in the module docstring, where the
    reasons for each live.

    What a caller may inject, and what it may not
    --------------------------------------------
    The per-dialect client construction is the security boundary, and it is not
    injectable: the binding itself builds the OpenAI SDK client for the
    ``openai`` dialect and the ``AsyncHTTPHandler`` for the generic dialects,
    always around
    :func:`~model_gateway.security.transport.build_pinned_transport`. What a
    caller may supply is that builder's own inputs, because none of them
    decides whether a connection is pinned or a redirect is followed:

    ``resolver``
        The URL policy's resolver: it chooses which addresses a hostname is
        looked up to, and every answer is still validated by the policy's own
        blocked-address rules before a socket is opened, so a custom resolver
        can change which policy-checked address is used, never whether
        checking happens. A deployment may resolve through its own stub; a
        test never touches DNS.

    ``verify``
        httpx's TLS verification setting: the default trust store, or an
        :class:`ssl.SSLContext` for a private CA. It chooses who is trusted,
        not where the connection goes. It does not apply when ``inner_factory``
        supplies a transport — a supplied transport owns its own TLS
        configuration, as the transport documents.

    ``inner_factory``
        Called once per attempt to build the transport that performs socket
        I/O, by default :class:`httpx.AsyncHTTPTransport`. Pinning rewrites
        the request before the inner transport is called, so whichever
        transport is supplied, it receives only requests already addressed to
        a validated address, with the original hostname preserved for TLS and
        ``Host``.

    What is not injectable — the builder, the SDK client and the handler — is
    what makes the failure mode impossible rather than merely unexercised:
    through the public API no caller can obtain a client for a verified
    dialect that re-resolves the hostname at connect time or follows a
    redirect, because every path from the constructor to the wire runs through
    :func:`~model_gateway.security.transport.build_pinned_transport` and the
    per-dialect wrapping. A dialect whose path is unverified is refused before
    any client is built at all (:func:`ensure_supported_dialect`).
    """

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        verify: ssl.SSLContext | bool = True,
        inner_factory: Callable[[], httpx.AsyncBaseTransport] = httpx.AsyncHTTPTransport,
    ) -> None:
        """Build the provider over the pinned transport's own inputs.

        The inputs are the ones documented in the class docstring: ``resolver``
        chooses which policy-checked address is connected to, ``verify`` which
        trust store validates the certificate, and ``inner_factory`` — called
        once per attempt — which transport performs the socket I/O. The client
        they are inputs *to* is always the binding's own per-dialect
        construction, so supplying them cannot substitute for it.
        """
        self._resolver = resolver
        self._verify = verify
        self._inner_factory = inner_factory

    async def complete(
        self,
        route: Route,
        request: InvocationRequest,
        context: AttemptContext,
    ) -> InvocationResponse:
        """Perform the one gateway attempt for this route (PRD section 9.6).

        The call's parameters are the translation of ``route`` and ``request``
        (:func:`~model_gateway.providers.litellm_translation.to_litellm_params`)
        plus this attempt's own controls: the timeout ``context`` carries and
        the client built for this attempt's budget, closed when the attempt
        ends. Translation and dialect refusals are already gateway errors and
        propagate unchanged; the mapping covers the call itself, whose
        exceptions are provider-shaped. The answer is normalised through
        :func:`~model_gateway.providers.litellm_translation.from_litellm_response`
        and returned as the canonical response.
        """
        params = to_litellm_params(route, request)
        params["timeout"] = context.timeout
        try:
            async with self._attempt_client(route, context.timeout) as client:
                response = await litellm.acompletion(**params, client=client)
        except GatewayError:
            raise
        except Exception as exc:
            raise map_litellm_error(exc) from exc
        return from_litellm_response(route, request, response)

    def _attempt_client(
        self, route: Route, budget: float | None
    ) -> AbstractAsyncContextManager[Any]:
        """The client this attempt injects, chosen for the route's HTTP path.

        The dialect is checked first, and the check is unconditional: the
        refusal is about the HTTP path ``litellm.acompletion`` takes for that
        dialect, which none of the injectable transport inputs changes, so
        supplying them does not make an unverified dialect safe. The check
        wins even over *which* client a verified dialect is given — that
        choice is the binding's own, built from this attempt's transport
        inputs.
        """
        ensure_supported_dialect(route.format)
        if route.format in _HTTP_HANDLER_DIALECTS:
            return _http_handler_client(
                budget, self._resolver, self._verify, self._inner_factory
            )
        return _openai_sdk_client(
            route, budget, self._resolver, self._verify, self._inner_factory
        )
