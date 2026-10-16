"""The LiteLLM image provider: one gateway attempt through the pinned binding.

This is the production :class:`~model_gateway.providers.base.ImageProvider`. It
performs one gateway attempt through ``litellm.aimage_generation`` with the
safe outbound client injected, and normalises the answer into the canonical
image response. What it owns is the provider boundary and nothing more —
routing, retry policy, accounting, identity and the public contract stay in
the gateway core (:mod:`model_gateway.providers.base`).

The image entry point is a different library entry point from the text and
embedding ones, so none of their transport verification carries over. Every
claim below was determined against the pinned ``litellm-core`` 1.106.0rc1 by
driving the library's own entry point against a local recording transport and
a local HTTP server, and each claim about which client the call uses is proved
by a test in :mod:`tests.test_litellm_image` rather than asserted here.

The HTTP path each dialect takes
-------------------------------
``litellm.image_dispatch`` per provider dialect, and the image dispatch is not
the text one's. Of the candidate dialects the binding considered, three are
runnable here:

``openai`` — the OpenAI SDK path
    ``litellm/llms/openai/openai.py``. ``client=`` must be an
    ``openai.AsyncOpenAI``: the handler reads ``api_key`` and ``base_url`` off
    the object it is given, so anything else fails the call outright with an
    ``APIConnectionError`` — a loud failure, not a silent loss, but a failure
    all the same. The binding constructs that SDK client around the httpx
    client of
    :func:`~model_gateway.security.transport.build_async_client`, the same
    construction the text path uses, with the one image-entry difference
    documented under "The retry count" below.

``gemini``, ``vertex_ai`` — the generic path, served *synchronously*
    Both go through ``BaseLLMHTTPHandler.image_generation_handler``, which the
    dispatch calls without ``_is_async`` — so the synchronous branch runs
    whatever entry point was used. Measured: that branch honours a synchronous
    ``HTTPHandler`` and silently discards an ``AsyncHTTPHandler`` — the call
    then succeeds on the library's own default client, which re-resolves the
    hostname at connect time and follows redirects, with nothing raised. The
    binding hands that path a synchronous ``HTTPHandler`` built around
    :class:`~model_gateway.security.transport.SyncPinnedAddressTransport`, so
    the synchronous client this path requires is pinned and non-redirecting
    exactly as the asynchronous one is.

``azure`` — refused
    ``AzureChatCompletions.aimage_generation`` passes ``client=None`` to
    ``make_async_azure_httpx_request`` regardless of what it was handed
    (measured: the injected client is silently discarded and the call proceeds
    on the library's own client). A path that discards the injected client
    loses both transport controls with no error to notice, so it is refused
    rather than run, exactly as the text path treats an unverified dialect.

``anthropic``, ``openrouter`` and everything else — refused. The image
dispatch has no branch for the Anthropic Messages API, and a model that matches
no branch at all does not fail: the dispatch returns an *empty*
``ImageResponse`` without contacting anything (measured) — an empty 200 is
worse than a raised error, so a dialect outside :data:`IMAGE_DIALECTS` is
refused at the attempt and at startup.

The injected client's two controls
---------------------------------
Address pinning and disabled redirect following reach the wire through the
injected client on both supported paths, and the measurements that make it
evidence rather than assertion:

* With the injected client (``follow_redirects=False``), a 302 from the
  upstream is *not* followed: it surfaces as an ``APIError`` on the SDK path
  and an ``APIConnectionError`` on the generic path (measured, one request
  each).
* Without it, both paths *follow* redirects (measured: the SDK's default
  client followed the hop; the generic path's default handler followed it
  until ``TooManyRedirects``).

The retry count
---------------
``num_retries=0`` travels on every call — but
the image entry point does not read it. It pops ``max_retries`` off the
request data (default 2) and assigns it onto an injected SDK client through
``_set_dynamic_params_on_client``, so on the ``openai`` path a plain injected
SDK client is forcibly retried twice: measured, a 500-ing upstream saw
**three** wire requests inside one gateway attempt, with or without
``num_retries`` on the call. :class:`_ZeroRetryAsyncOpenAI` pins the count as
a read-only property the library's assignment cannot move; measured again,
the pinned client produces **exactly one** wire request per gateway attempt
on a 500-ing upstream and on a connection error. The generic path makes
exactly one wire request too: the synchronous ``HTTPHandler.post`` has no
connection-error retry of its own (measured, one attempt), so no gateway
attempt on a supported route carries more than one upstream request.

The unsupported-parameter constraint
-------------------------------------
The constraint is real and was measured: for a model whose
provider configuration does not list ``response_format`` (or ``size``) among
its supported parameters, the library raises ``UnsupportedParamsError``
*before any request is made* — including on the OpenAI GPT configuration,
which every non-dall-e model on the ``openai`` dialect selects. The binding
never lets that surface as an opaque failure: it asks the same provider
configuration object the library will use — for the bare model ID, because
that is the string the library's dispatch hands to it — and forwards only what
that set contains.

A stated parameter the configuration does not accept is refused with a
:class:`~model_gateway.errors.CapabilityError` naming it, because the route
promised the parameter and the upstream cannot honour it.
One deliberate exception, measured rather than assumed: three specific
configurations that do not forward ``response_format`` produce base64 image
data by construction — the Google and Vertex Imagen transformations emit
``inlineData`` and ``bytesBase64Encoded``, and the OpenAI GPT configuration's
upstream defaults to ``b64_json`` — so a caller who asked for ``b64_json`` on
one of those routes is honoured by the upstream's own single representation and
the parameter is not forwarded. The exception is keyed on those configurations'
classes (:data:`_BASE64_ONLY_IMAGE_CONFIGS`), not on the value the caller
stated and not on membership of :data:`IMAGE_DIALECTS`, so a dialect the
binding has not measured cannot inherit it by joining the set. A caller who
asked for ``url`` on any route that does not forward the parameter cannot be
honoured at all, and is refused. (The dall-e configurations list
``response_format`` and forward it explicitly; their upstream returns what the
caller asked for.)

The translation of the upstream answer
--------------------------------------
The four-way field classification, applied to every field the binding
inspects:

* each data entry's ``b64_json``/``url``/``revised_prompt`` are mapped into
  :class:`~model_gateway.domain.images.GeneratedImage`; the ``url`` is carried
  through as untrusted provider output, neither fetched nor screened here, and
  must not be treated as a gateway-fetchable URL just because it survived
  canonical parsing (:class:`~model_gateway.domain.images.GeneratedImage`
  documents the contract). An entry carrying a
  provider-specific field (a ``thought_signature`` from a Gemini-shaped
  answer, say — measured arriving as ``provider_specific_fields``) or
  carrying neither payload nor location is refused with a
  :class:`~model_gateway.errors.MalformedResponseError` naming it, so another
  eligible route may serve the request;
* the usage is mapped with the embeddings path's discipline: a non-zero count
  is the provider-reported fact it can only be, a zero or absent count is
  recorded as unknown — the library fabricates an all-zero usage when the
  upstream reported none, and that placeholder is indistinguishable from a
  reported zero (measured) — and a total of zero beside non-zero parts is
  derived from the parts and named in ``estimated_fields``. A usage field the
  canonical model cannot carry (a ``web_search_requests`` annotation, say) is
  refused with the field named;
* the upstream's own identity fields (``created``, ``model``) are neither
  mapped nor refused: the gateway stamps its own assembly time, and the
  upstream model is reported as ``upstream_model`` beside the public ID;
* ``hidden_params`` (response headers, cost, call id) is operational metadata
  and is deliberately ignored, as is ``system_fingerprint``, the backend
  fingerprint named as the field that must not fail an otherwise valid
  response.

One binding limitation, recorded rather than accepted: the pinned library
drops unknown *top-level* response fields before the binding sees them — a
``system_fingerprint`` in the upstream answer never arrives on the
``ImageResponse`` object (measured). Fields on a data entry and on the usage
do arrive, so those two levels are classified as above; a top-level
annotation is irrecoverable at this boundary, exactly as the embeddings
path's zero placeholder is.

The ``.env`` guard
-----------------
The module imports the library through :mod:`model_gateway.providers._litellm`,
the seam that forces ``LITELLM_MODE`` to ``PRODUCTION`` before the first
``import litellm``, so no ambient value can re-enable the library's import-
time ``dotenv.load_dotenv()``. The test session's own copy of
the guard lives in :file:`tests/conftest.py`.
"""

import inspect
import ssl
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Final

import httpx
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.gemini.image_generation.transformation import GoogleImageGenConfig
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.llms.vertex_ai.image_generation.vertex_imagen_transformation import (
    VertexAIImagenImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from model_gateway.config.routes import Route
from model_gateway.domain.generation import Usage
from model_gateway.domain.images import GeneratedImage, ImageRequest, ImageResponse
from model_gateway.errors import (
    CapabilityError,
    FailedPreconditionError,
    GatewayError,
    MalformedResponseError,
)
from model_gateway.providers._litellm import litellm, openai
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.litellm_translation import map_litellm_error
from model_gateway.registry import PUBLIC_MODEL_IDS
from model_gateway.security.transport import (
    attempt_timeout,
    build_async_client,
    build_sync_client,
)
from model_gateway.security.url_policy import Resolver, system_resolver

__all__ = [
    "IMAGE_DIALECTS",
    "LiteLLMImageProvider",
    "ensure_supported_image_dialect",
]

_OPENAI_SDK_DIALECTS: Final[frozenset[str]] = frozenset({"openai"})
_GENERIC_DIALECTS: Final[frozenset[str]] = frozenset({"gemini", "vertex_ai"})

_BASE64_ONLY_IMAGE_CONFIGS: Final[frozenset[type]] = frozenset(
    {
        GPTImageGenerationConfig,
        GoogleImageGenConfig,
        VertexAIImagenImageGenerationConfig,
    }
)
"""The provider image configurations measured to produce base64 image data by construction.

These are the configurations that do not list ``response_format`` among their
supported parameters and whose upstream has no other representation to return:
the Google and Vertex Imagen transformations emit ``inlineData`` and
``bytesBase64Encoded``, and the OpenAI GPT configuration's upstream defaults to
``b64_json``. Only these make the ``b64_json`` honour-by-default case in
:func:`_forward_response_format` true. Membership is by exact class, not by
dialect and not by subclass, so a dialect that merely joins
:data:`IMAGE_DIALECTS` — or a configuration that subclasses one of these —
cannot inherit the case without its own measurement.
"""

IMAGE_DIALECTS: Final[frozenset[str]] = _OPENAI_SDK_DIALECTS | _GENERIC_DIALECTS
"""The provider dialects this binding has verified end to end for the image entry point.

The candidate dialects were each driven against the pinned
``litellm-core``; membership is a claim about which HTTP path the binding can
keep behind the transport controls it enforces, and the verification
that produced it is recorded in the module docstring. The set is
entry-point-specific: a dialect verified for ``acompletion`` or
``aembedding`` is not thereby verified here — the image dispatch differs, and
on the generic dialects it serves a synchronous path.
"""


def ensure_supported_image_dialect(dialect: str) -> str:
    """Return ``dialect`` when its image HTTP path is verified, and refuse it otherwise.

    A dialect whose image path the binding cannot keep behind the transport
    controls is refused rather than attempted, and the reason is always one of
    three: the library discards the injected client (``azure`` passes ``None`` to
    its own request builder), the image dispatch has no usable branch for the
    dialect (``anthropic`` is not in the dispatch at all and answers an empty
    ``ImageResponse`` without contacting anything; ``openrouter`` is dispatched
    but answers with an empty ``data`` list), or the path has simply not been
    verified. Where a refused path does reach the upstream, the connection is
    made to whatever the hostname resolves to at connect time and redirects are
    followed, with nothing raised to notice; where it does not, an empty 200 is
    worse than a raised error. Being synchronous is *not* one of those reasons:
    the generic image dialects are synchronous, and the binding pins them through
    a synchronous pinned transport, so the refusal turns on whether the injected
    client is honoured and the path is verified, never on whether the path is
    asynchronous. The refusal names the dialects the image binding can run; the
    name is configuration, not a secret, so it is safe in a caller-visible
    message.
    """
    if dialect in IMAGE_DIALECTS:
        return dialect
    verified = ", ".join(repr(name) for name in sorted(IMAGE_DIALECTS))
    raise FailedPreconditionError(
        f"the image binding cannot run the {dialect!r} provider dialect: its HTTP path has "
        f"not been verified to honour the injected client for the image entry point. The "
        f"paths refused here are the ones that discard the injected client, the ones the "
        f"image dispatch has no usable branch for, and the ones not yet verified; where such "
        f"a call proceeds it runs on the library's own client, which re-resolves the hostname "
        f"and follows redirects with nothing raised. Being synchronous is not one of those "
        f"reasons — the generic image dialects are synchronous and the binding pins them "
        f"through a synchronous pinned transport. Verified image dialects are {verified}"
    )


class _ZeroRetryAsyncOpenAI(openai.AsyncOpenAI):
    """An OpenAI SDK client whose retry count the library cannot restore.

    The image entry point is the one call this binding makes where the library
    itself restores a retry count on the client it is handed: it pops
    ``max_retries`` off the request data — default 2, because ``num_retries``
    is not read anywhere on the image path — and assigns it onto an injected
    client through the SDK handler's ``_set_dynamic_params_on_client``.
    Measured against the pinned library: a 500-ing upstream then saw three wire
    requests inside one gateway attempt. The property pins the count at the
    ``max_retries=0`` the binding built the client with, so the forced
    assignment is a no-op and one gateway attempt carries exactly one wire
    request.
    """

    @property
    def max_retries(self) -> int:
        return 0

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        return


class _AttemptClient:
    """One attempt's injected client, and the lifecycle that closes it.

    The SDK client's ``close`` is asynchronous, so the wrapper awaits it when
    the close returns a coroutine; a synchronous close would be called plain.
    """

    __slots__ = ("_client", "_close")

    def __init__(self, client: Any, close: Callable[[], Any]) -> None:
        self._client = client
        self._close = close

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *exc_info: object) -> None:
        result = self._close()
        if inspect.isawaitable(result):
            await result


def _openai_sdk_image_client(
    route: Route,
    budget: float | None,
    resolver: Resolver,
    verify: ssl.SSLContext | bool,
    inner_factory: Callable[[], httpx.AsyncBaseTransport],
) -> _AttemptClient:
    """The client the SDK image path is given: a retry-pinned SDK client over the pinned client.

    The construction is the text path's OpenAI client construction — the SDK
    object is what ``client=`` must carry on this path, built around the httpx
    client of :func:`~model_gateway.security.transport.build_async_client` —
    with the one image-entry-point difference: the SDK client is
    :class:`_ZeroRetryAsyncOpenAI`, because the image entry point restores
    ``max_retries=2`` on whatever client it is handed and the gateway owns the
    only retry loop.
    """
    client = _ZeroRetryAsyncOpenAI(
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


def _generic_image_client(
    budget: float | None,
    resolver: Resolver,
    verify: ssl.SSLContext | bool,
    inner_factory: Callable[[], httpx.BaseTransport],
) -> _AttemptClient:
    """The client the generic image path is given: a synchronous handler over the pinned transport.

    ``BaseLLMHTTPHandler.image_generation_handler`` serves the generic dialects
    synchronously, so ``client=`` must carry a synchronous ``HTTPHandler``
    there; an ``AsyncHTTPHandler`` is silently discarded and the call runs on
    the library's own default client, which re-resolves the hostname and
    follows redirects. The handler is built from the synchronous client of
    :func:`~model_gateway.security.transport.build_sync_client`, with
    ``follow_redirects=False`` because the handler's own default is the
    library's ``True`` and ``trust_env=False`` so no environment proxy is
    installed between the gateway and the upstream.
    """
    handler = HTTPHandler(
        timeout=attempt_timeout(budget),
        client=build_sync_client(
            budget=budget,
            resolver=resolver,
            verify=verify,
            inner=inner_factory(),
        ),
        follow_redirects=False,
    )
    return _AttemptClient(handler, handler.client.close)


class LiteLLMImageProvider:
    """The production image provider: translation plus one execution.

    One gateway attempt in, one canonical answer out. The provider performs one
    gateway attempt for the route it is handed — a single invocation of the
    pinned binding — and applies the attempt's own budget to it; it does not
    retry, does not choose between routes and knows nothing about fallbacks —
    those are the gateway core's. Every control the
    call carries — ``num_retries=0``, the injected client, the attempt timeout,
    the error mapping — is documented in the module docstring, where the
    reasons and the measurements behind each live.

    What a caller may inject, and what it may not
    --------------------------------------------
    The per-dialect client construction is the security boundary, and it is not
    injectable: the binding itself builds the retry-pinned SDK client for the
    ``openai`` dialect and the synchronous ``HTTPHandler`` for the generic
    dialects, always around a pinned, non-redirecting transport. What a caller
    may supply is that pinning's own inputs, because none of them decides
    whether a connection is pinned or a redirect is followed:

    ``resolver``
        The URL policy's resolver: it chooses which addresses a hostname is
        looked up to, and every answer is still validated by the policy's own
        blocked-address rules before a socket is opened, so a custom resolver
        can change which policy-checked address is used, never whether
        checking happens.

    ``verify``
        httpx's TLS verification setting: the default trust store, or an
        :class:`ssl.SSLContext` for a private CA. It chooses who is trusted,
        not where the connection goes. It does not apply when an
        ``inner_factory`` supplies a transport — a supplied transport owns its
        own TLS configuration, as the transport documents.

    ``inner_factory`` and ``sync_inner_factory``
        Called once per attempt to build the transport that performs socket
        I/O: an asynchronous one for the OpenAI SDK path, a synchronous one for
        the generic path, which is the one HTTP path this binding reaches that
        is synchronous. Pinning rewrites the request before the inner transport
        is called, so whichever transport is supplied, it receives only requests
        already addressed to a validated address, with the original hostname
        preserved for TLS and ``Host``.

    What is not injectable — the builder, the SDK client and the handler — is
    what makes the failure mode impossible rather than merely unexercised:
    through the public API no caller can obtain a client for a verified
    dialect that re-resolves the hostname at connect time or follows a
    redirect, because every path from the constructor to the wire runs through
    the pinned transport and the per-dialect wrapping. A dialect whose path is
    unverified is refused before any client is built at all
    (:func:`ensure_supported_image_dialect`).
    """

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        verify: ssl.SSLContext | bool = True,
        inner_factory: Callable[[], httpx.AsyncBaseTransport] = httpx.AsyncHTTPTransport,
        sync_inner_factory: Callable[[], httpx.BaseTransport] = httpx.HTTPTransport,
    ) -> None:
        """Build the provider over the pinned transport's own inputs.

        The inputs are the ones documented in the class docstring: ``resolver``
        chooses which policy-checked address is connected to, ``verify`` which
        trust store validates the certificate, and the two inner factories —
        called once per attempt — which transport performs the socket I/O on
        each path. The client they are inputs *to* is always the binding's own
        per-dialect construction, so supplying them cannot substitute for it.
        """
        self._resolver = resolver
        self._verify = verify
        self._inner_factory = inner_factory
        self._sync_inner_factory = sync_inner_factory

    async def generate(
        self,
        route: Route,
        request: ImageRequest,
        context: AttemptContext,
    ) -> ImageResponse:
        """Perform the one image attempt for this route.

        The dialect is checked first, unconditionally: the refusal is about
        the HTTP path ``litellm.aimage_generation`` takes for that dialect,
        which none of the injectable transport inputs changes. The call's
        parameters are the translation of ``route`` and ``request`` — the
        provider-prefixed model string, the resolved base URL, the route's
        credential and extra headers, the prompt, only the image parameters
        the dialect's provider configuration accepts, ``num_retries=0`` and the
        timeout ``context`` carries — plus this attempt's own client, built
        for this attempt's budget and closed when the attempt ends.

        The answer is normalised through :func:`_from_litellm_image` and
        returned as the canonical response. Errors are mapped exactly as the
        text path maps them: a :class:`~model_gateway.errors.GatewayError` — a
        dialect refusal, a parameter the upstream cannot honour, or an answer
        the canonical representation cannot carry — propagates unchanged, and
        anything the call raises is mapped by
        :func:`~model_gateway.providers.litellm_translation.map_litellm_error`
        and re-raised, so a caller never sees a provider-shaped error.
        Cancellation propagates unmapped, because it is a :exc:`BaseException`
        the mapping does not catch.
        """
        ensure_supported_image_dialect(route.format)
        config = _route_image_config(route)
        params = _image_params(route, request, context, config)
        try:
            async with self._attempt_client(route, context.timeout) as client:
                response = await litellm.aimage_generation(**params, client=client)
        except GatewayError:
            raise
        except litellm.UnsupportedParamsError as error:
            raise CapabilityError(
                "the upstream refused an image parameter the gateway forwarded for this "
                "route, so the request as asked cannot be served",
                details={"capability": "image_parameters"},
            ) from error
        except Exception as exc:
            raise map_litellm_error(exc) from exc
        return _from_litellm_image(route, request, response, invocation_id=context.invocation_id)

    def _attempt_client(
        self, route: Route, budget: float | None
    ) -> AbstractAsyncContextManager[Any]:
        """The client this attempt injects, chosen for the route's HTTP path.

        The dialect is checked first, and the check is unconditional: the
        refusal is about the HTTP path the image entry point takes for that
        dialect, which none of the injectable transport inputs changes. The
        check wins even over *which* client a verified dialect is given — that
        choice is the binding's own, built from this attempt's transport
        inputs.
        """
        ensure_supported_image_dialect(route.format)
        if route.format in _GENERIC_DIALECTS:
            return _generic_image_client(
                budget, self._resolver, self._verify, self._sync_inner_factory
            )
        return _openai_sdk_image_client(
            route, budget, self._resolver, self._verify, self._inner_factory
        )


@dataclass(frozen=True)
class _ImageConfig:
    """The library's own provider image configuration for a route, and what it accepts.

    The library raises ``UnsupportedParamsError`` for a request parameter its
    provider configuration does not list, before any request is made (measured),
    so the binding asks the same configuration object the library will use. The
    lookup is by the bare model ID because that is the string the library's
    dispatch hands to it after stripping the provider prefix — asking with the
    prefixed ``<format>/<model_id>`` string would select the OpenAI GPT
    configuration for *every* model, including dall-e (measured).

    ``configuration`` is ``None`` when the library recognises no configuration
    for the route. ``supported`` is what that configuration lists for the bare
    model ID. ``base64_only`` is the measured property
    :func:`_forward_response_format` turns on: whether the configuration is one
    of :data:`_BASE64_ONLY_IMAGE_CONFIGS`, which is true only of configurations
    whose upstream was driven and found to have no representation but base64.
    """

    configuration: BaseImageGenerationConfig | None
    supported: frozenset[str]

    @property
    def base64_only(self) -> bool:
        """Whether this configuration's upstream was measured to produce base64 only.

        Membership is by exact class: a subclass of a measured configuration is
        a different configuration and inherits nothing from this one.
        """
        return (
            self.configuration is not None
            and type(self.configuration) in _BASE64_ONLY_IMAGE_CONFIGS
        )


def _route_image_config(route: Route) -> _ImageConfig:
    """The library's provider image configuration for this route, and what it accepts."""
    configuration = ProviderConfigManager.get_provider_image_generation_config(
        model=route.model_id, provider=LlmProviders(route.format)
    )
    if configuration is None:
        return _ImageConfig(configuration=None, supported=frozenset())
    return _ImageConfig(
        configuration=configuration,
        supported=frozenset(configuration.get_supported_openai_params(route.model_id)),
    )


def _image_params(
    route: Route,
    request: ImageRequest,
    context: AttemptContext,
    config: _ImageConfig,
) -> dict[str, object]:
    """Translate an image request into ``litellm.aimage_generation`` keyword arguments.

    The route supplies everything the upstream call needs to be addressed,
    exactly as it does for a completion: the provider-prefixed model string
    ``<format>/<model_id>``, the resolved base URL, the
    route's credential and its non-authentication extra headers. Retries are
    disabled with ``num_retries=0`` because the gateway core owns the retry
    loop and the invocation deadline, so there is exactly one retry layer and
    one deadline — the parameter travels on every call even
    though the image entry point is the one call this binding makes that does
    not read it.

    The request supplies the prompt, forwarded verbatim, and the parameters the
    caller stated. A stated parameter the dialect's provider configuration
    does not accept is refused with a capability error naming it rather than
    forwarded to the library's own opaque refusal;
    ``response_format`` is the one parameter with a measured honour-by-default
    case, handled by :func:`_forward_response_format`.
    """
    params: dict[str, object] = {
        "model": route.litellm_model,
        "api_base": route.base_url,
        "api_key": route.api_key,
        "extra_headers": dict(route.extra_headers),
        "prompt": request.prompt,
        "num_retries": 0,
        "timeout": context.timeout,
    }
    _forward_param(params, config.supported, "n", request.n)
    _forward_param(params, config.supported, "size", request.size)
    _forward_response_format(params, config, request.response_format)
    return params


def _forward_param(
    params: dict[str, object], supported: frozenset[str], name: str, value: object
) -> None:
    """Forward one stated parameter the configuration accepts, or refuse the invocation."""
    if value is None:
        return
    if name in supported:
        params[name] = value
        return
    raise CapabilityError(
        f"the route's dialect and model do not accept the {name!r} image parameter, so the "
        f"request as asked cannot be served",
        details={"capability": name},
    )


def _forward_response_format(
    params: dict[str, object], config: _ImageConfig, response_format: str | None
) -> None:
    """Forward ``response_format`` where the configuration accepts it, classify where not.

    Each branch is a measured fact about the configuration the library itself
    will use, never a rule about the value the caller stated. A configuration
    that lists ``response_format`` — the dall-e configurations on the ``openai``
    dialect — forwards it and its upstream honours it directly. A configuration
    in :data:`_BASE64_ONLY_IMAGE_CONFIGS` does not list it and has no
    representation but base64: the Google Imagen transformations emit
    ``inlineData`` and ``bytesBase64Encoded``, and the OpenAI GPT
    configuration's upstream defaults to ``b64_json``. Only for those is a
    stated ``b64_json`` honoured by the upstream's own single representation,
    and only there is the parameter dropped. Anything else is refused with a
    capability error naming ``response_format``, because forwarding it would
    reach the library's opaque ``UnsupportedParamsError`` and dropping it would
    claim a representation was honoured on the strength of no measurement at
    all.

    The base64 branch is keyed on the configuration's exact class rather than on
    the stated value, so a dialect that merely joins :data:`IMAGE_DIALECTS` —
    and a configuration that subclasses a measured one — cannot inherit it. A
    library upgrade that hands a verified dialect a different configuration
    therefore turns the drop into a refusal, which is loud, rather than into a
    silent claim that an unmeasured upstream returned what was asked for.
    """
    if response_format is None:
        return
    if "response_format" in config.supported:
        params["response_format"] = response_format
        return
    if response_format == "b64_json" and config.base64_only:
        return
    if response_format == "b64_json":
        raise CapabilityError(
            "the route's dialect and model cannot be honoured for a 'b64_json' request: the "
            "configuration does not accept the response_format parameter, and this route's "
            "upstream has not been measured to produce base64 image data by default",
            details={"capability": "response_format"},
        )
    raise CapabilityError(
        "the route's dialect and model cannot return images as urls: the configuration does "
        "not accept the response_format parameter, and the images it produces are base64-only",
        details={"capability": "response_format"},
    )


def _from_litellm_image(
    route: Route,
    request: ImageRequest,
    response: object,
    *,
    invocation_id: str | None,
) -> ImageResponse:
    """Normalise a LiteLLM image answer into the canonical response.

    ``response`` is upstream data, never forwarded unchanged.
    The upstream's identity fields are neither mapped nor refused: the gateway
    stamps its own assembly time, and the route's own model ID is reported as
    ``upstream_model`` beside the public one. ``hidden_params`` — response
    headers, cost, call id — is operational metadata and deliberately ignored.

    Raises :class:`~model_gateway.errors.MalformedResponseError` naming the
    field when the upstream answer carries a semantic the canonical
    representation cannot carry: an image entry with no payload and no
    location, an entry carrying a provider-specific field, a usage field with
    no canonical target, or no image at all. The refusal is the malformed-
    response error rather than an internal one: the upstream answered, and its
    answer is what cannot serve the request, so another eligible route may
    answer correctly.
    """
    if not invocation_id:
        raise FailedPreconditionError(
            "the attempt carries no invocation identifier, and the canonical image response "
            "must state the invocation it belongs to; the gateway mints that identifier "
            "before the call"
        )
    if response is None:
        raise MalformedResponseError("the upstream returned no image response")
    _reject_unrepresentable_image_response(response)
    try:
        return ImageResponse(
            model=PUBLIC_MODEL_IDS[route.operation],
            upstream_model=route.model_id,
            created=int(time.time()),
            data=tuple(_generated_image_from(item) for item in _image_entries(response)),
            usage=from_litellm_image_usage(getattr(response, "usage", None)),
            invocation_id=invocation_id,
        )
    except ValueError as error:
        raise MalformedResponseError(
            "the upstream returned an image answer the canonical representation cannot carry"
        ) from error


def _image_entries(response: object) -> tuple[object, ...]:
    """The image entries the upstream answer carries, as an iterable of entries.

    The library types ``data`` as a list and fills it with an empty one when
    the upstream reported nothing, so the usual case is a list. A value that is
    not iterable at all — a ``null`` the library passed through, say — becomes
    an empty tuple here rather than a ``TypeError`` escaping the binding: the
    gateway core then refuses the answer for carrying no image, which is a
    gateway error a caller can act on rather than an unclassified failure.
    """
    data = getattr(response, "data", None)
    if data is None or isinstance(data, str | bytes):
        return ()
    try:
        return tuple(data)
    except TypeError:
        return ()


def _generated_image_from(item: object) -> GeneratedImage:
    """One upstream image entry as a canonical image.

    The pinned library hands the entries back as
    :class:`~litellm.types.utils.ImageObject` objects or as the raw
    dictionaries its own converter produced, so both shapes are read the same
    way. An entry the canonical representation cannot carry is refused rather
    than coerced.

    An entry's ``url`` is copied through verbatim as the untrusted provider
    output it is: it is not resolved, not screened against the URL policy and
    not fetched, here or anywhere else in the binding, and its survival of this
    mapping is no evidence that it is safe to fetch
    (:class:`~model_gateway.domain.images.GeneratedImage`).
    """
    b64_json = _image_field(item, "b64_json")
    url = _image_field(item, "url")
    revised_prompt = _image_field(item, "revised_prompt")
    provider_specific = _image_field(item, "provider_specific_fields")
    if provider_specific:
        names = ", ".join(sorted(str(name) for name in _mapping_keys(provider_specific)))
        raise MalformedResponseError(
            f"the upstream image entry carries a provider-specific field the canonical "
            f"representation cannot carry: {names}"
        )
    try:
        return GeneratedImage(b64_json=b64_json, url=url, revised_prompt=revised_prompt)
    except ValueError as error:
        raise MalformedResponseError(
            "the upstream returned an image entry the canonical representation cannot carry"
        ) from error


def _image_field(item: object, name: str) -> object:
    """One field of an upstream image entry, whichever shape it arrived in."""
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _mapping_keys(value: object) -> tuple[object, ...]:
    """The keys of a mapping, or the value itself when it is not one."""
    if isinstance(value, Mapping):
        return tuple(value)
    return (value,)


_IMAGE_RESPONSE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "created",
        "data",
        "usage",
        # declared fields of the library's ImageResponse that echo request
        # parameters rather than describing the answer
        "background",
        "output_format",
        "quality",
        "size",
    }
)
"""Fields the binding inspects on the upstream response and classifies."""

_IGNORED_RESPONSE_FIELDS: Final[frozenset[str]] = frozenset({"hidden_params", "system_fingerprint"})
"""Known non-semantic operational metadata, deliberately ignored.

``hidden_params`` carries response headers, cost and the library's own call
id; ``system_fingerprint`` identifies the upstream infrastructure that
produced the answer rather than anything about it. Neither changes what the
caller was told, so neither fails an otherwise valid response.
"""


def _reject_unrepresentable_image_response(response: object) -> None:
    """Refuse a response carrying a field the canonical model cannot represent.

    Every field the binding can observe is classified: the known semantics are
    mapped (:func:`_from_litellm_image`), the operational metadata in
    :data:`_IGNORED_RESPONSE_FIELDS` is deliberately ignored, and anything else
    is refused with the field named, so a new provider annotation cannot pass
    unexamined. The pinned library already drops unknown *top-level* fields
    before the binding sees them (measured — a ``system_fingerprint`` in the
    upstream answer never arrives), so this check is the act of classification
    on the fields that do arrive, not a claim that every drop is observable.
    """
    for name in _response_field_names(response):
        if name in _IGNORED_RESPONSE_FIELDS:
            continue
        if name not in _IMAGE_RESPONSE_FIELDS:
            raise MalformedResponseError(
                "the upstream image response carries a field the canonical representation "
                f"cannot carry: {name}"
            )


def _response_field_names(response: object) -> tuple[str, ...]:
    """The field names the binding can observe on the upstream response."""
    names = set(vars(response))
    names.add("hidden_params")
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, Mapping):
        names.update(extra)
    return tuple(sorted(names))


_IMAGE_USAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        # the ImageUsage naming of the same counts, which the library assigns
        # beside the LiteLLMUsage-declared fields on one object (measured)
        "input_tokens",
        "input_tokens_details",
        "output_tokens",
        "output_tokens_details",
    }
)
"""The usage fields the binding reads or classifies, whichever shape they arrive in."""


def from_litellm_image_usage(usage: object) -> Usage:
    """The upstream image usage as a canonical usage record.

    The mapping carries the embeddings path's zero discipline, for the same
    measured reason: ``litellm.image_generation`` fabricates an all-zero usage
    when the upstream reported none, and that placeholder is indistinguishable
    from a reported zero in every representation the binding can read, so a
    zero count cannot be presented as a provider-reported fact. A non-zero
    count cannot be the placeholder, so it is recorded as the fact it can only
    be; a zero or absent count is recorded as unknown. A total of zero beside
    non-zero parts is derived from the parts and named in ``estimated_fields``.

    A usage field outside :data:`_IMAGE_USAGE_FIELDS` — a ``web_search_requests``
    annotation, say — is a provider-reported fact the canonical usage model
    cannot carry, so the response is refused with the field named rather than
    trimmed.
    """
    if usage is None:
        return Usage()
    _reject_unrepresentable_image_usage(usage)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    estimated: set[str] = set()
    if total_tokens == 0 and ((input_tokens or 0) > 0 or (output_tokens or 0) > 0):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        estimated.add("total_tokens")
    return Usage(
        input_tokens=_established(input_tokens),
        output_tokens=_established(output_tokens),
        total_tokens=(total_tokens if "total_tokens" in estimated else _established(total_tokens)),
        cached_tokens=_established(_image_cached_tokens(usage)),
        estimated_fields=frozenset(estimated),
    )


def _reject_unrepresentable_image_usage(usage: object) -> None:
    """Refuse a usage field the canonical model cannot carry, naming it."""
    for name in _usage_field_names(usage):
        if name not in _IMAGE_USAGE_FIELDS:
            raise MalformedResponseError(
                "the upstream usage carries a field the canonical usage representation "
                f"cannot carry: {name}"
            )


def _usage_field_names(usage: object) -> tuple[str, ...]:
    """The usage field names the binding can observe, in either of the two shapes it arrives in."""
    names = set(vars(usage))
    extra = getattr(usage, "model_extra", None)
    if isinstance(extra, Mapping):
        names.update(extra)
    return tuple(sorted(names))


def _image_cached_tokens(usage: object) -> int | None:
    """The provider-reported cached-token count, or ``None`` when it reported none."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    return getattr(details, "cached_tokens", None)


def _established(count: int | None) -> int | None:
    """A count the binding can hold to the upstream; a zero on this path cannot be one.

    The library's placeholder for an unreported usage is all-zero, and a
    reported zero is the same object, so a zero count here is never
    establishable as a provider fact and is recorded as unknown. A non-zero
    count cannot be the placeholder, so it is recorded as reported.
    """
    if count is None or count == 0:
        return None
    return count
