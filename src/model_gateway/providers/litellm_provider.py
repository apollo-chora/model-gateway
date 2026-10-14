"""The LiteLLM text provider: one upstream call through the pinned binding (PRD section 9).

This is the production :class:`~model_gateway.providers.base.TextProvider`. It
adds execution to the transport-agnostic translation of
:mod:`model_gateway.providers.litellm_translation`: build the call's parameters,
perform exactly one upstream request through ``litellm.acompletion`` with the
safe outbound client injected, and normalise the answer into the canonical
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

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Final

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


def _openai_sdk_client(route: Route, budget: float | None) -> _AttemptClient:
    """The client the OpenAI SDK path is given: an SDK client over the pinned httpx client.

    The SDK object is what ``client=`` must carry on this path, and the httpx
    client it is built around is the one that owns the connection, so the
    transport controls live there. ``max_retries=0`` is set here as well as
    through ``num_retries=0`` because the SDK retries with the same httpx client
    and the gateway owns the only retry loop (PRD section 11.3).
    """
    client = openai.AsyncOpenAI(
        api_key=route.api_key,
        base_url=route.base_url,
        http_client=build_async_client(budget=budget),
        timeout=attempt_timeout(budget),
        max_retries=0,
    )
    return _AttemptClient(client, client.close)


def _http_handler_client(budget: float | None) -> _AttemptClient:
    """The client the generic httpx path is given, carrying the pinned transport itself.

    ``AsyncHTTPHandler`` builds its client from its own ``transport`` attribute,
    and rebuilds it the same way when it retries a connection error, so handing
    it the pinned transport is what keeps that retry inside the controls.
    ``follow_redirects`` is ``False`` because the handler's default is the
    library's own ``True``.
    """
    handler = AsyncHTTPHandler(
        timeout=attempt_timeout(budget),
        transport=build_pinned_transport(),
        follow_redirects=False,
    )
    return _AttemptClient(handler, handler.close)


class LiteLLMProvider:
    """The production text provider: translation plus one execution (PRD section 9).

    One attempt in, one canonical answer out. The provider performs exactly one
    upstream request for the route it is handed and applies the attempt's own
    budget to it; it does not retry, does not choose between routes and knows
    nothing about fallbacks — those are the gateway core's (PRD sections 9.2
    and 11.3). Every control the call carries — ``num_retries=0``, the injected
    client, the attempt timeout, the error mapping — is documented in the module
    docstring, where the reasons for each live.

    The client factory is injectable so a caller can stand in for the transport's
    client. Left unset — the production default — the binding builds the client
    the route's dialect requires, from the pinned transport, and refuses a
    dialect whose HTTP path it has not verified
    (:func:`ensure_supported_dialect`).
    """

    def __init__(self, *, client_factory: Callable[..., Any] | None = None) -> None:
        """Build the provider, optionally over an explicit client factory.

        ``client_factory`` is called once per attempt with the attempt's budget
        as ``budget=`` and returns the client that attempt's request is injected
        with. Supplying one replaces the per-dialect construction for a verified
        dialect: the object the factory returns is what LiteLLM receives, which
        is what lets a test stand in for the transport's client and never touch
        a network. It does not make an unverified dialect acceptable, because the
        refusal is about the path the library takes rather than about the client
        it is given.
        """
        self._client_factory = client_factory

    async def complete(
        self,
        route: Route,
        request: InvocationRequest,
        context: AttemptContext,
    ) -> InvocationResponse:
        """Perform the one upstream call for this attempt (PRD section 9.6).

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
        dialect, which no injected client changes, so an explicitly supplied
        factory does not make an unverified dialect safe. It wins only over
        *which* client a verified dialect is given.
        """
        ensure_supported_dialect(route.format)
        if self._client_factory is not None:
            return self._client_factory(budget=budget)
        if route.format in _HTTP_HANDLER_DIALECTS:
            return _http_handler_client(budget)
        return _openai_sdk_client(route, budget)
