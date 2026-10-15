"""Tests for the LiteLLM image provider: one gateway attempt through the pinned binding.

Two kinds of test run here, and neither reaches the network:

* The client-injection tests monkeypatch ``litellm.aimage_generation`` with a
  recording stand-in and inspect the object the attempt hands it, exactly as
  the text path's injection tests do — the claim under test is *which client*
  the library's image path honours, and the pinned library proves that claim by
  the object it is given.
* The real-call tests drive the actual ``litellm.aimage_generation`` with the
  injected client built around a recording transport, so no socket is opened at
  all: the library dispatches, transforms and converts for real, and only the
  transport beneath the injected client is a stand-in. These pin the
  end-to-end behaviour the module docstring records as measured — the request
  each dialect actually sends, the one-wire-request-per-attempt retry count,
  and the redirect refusal.

The translation fixtures are the real LiteLLM types, built through the
library's own converter or constructed directly, carrying the additional
semantic fields a happy-path answer would not need — a provider-specific
field on an image entry, a usage annotation — so a translator that drops them
fails the test.
"""

import asyncio
import json

import httpx
import litellm
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai.image_generation.dall_e_3_transformation import (
    DallE3ImageGenerationConfig,
)
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.utils import Usage as LiteLLMUsage

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain.images import GeneratedImage, ImageRequest
from model_gateway.errors import (
    CapabilityError,
    FailedPreconditionError,
    InternalError,
    MalformedResponseError,
    UnavailableError,
)
from model_gateway.providers.base import AttemptContext, ImageProvider
from model_gateway.providers.litellm_image import (
    IMAGE_DIALECTS,
    LiteLLMImageProvider,
    _forward_response_format,
    _ImageConfig,
    _route_image_config,
    _ZeroRetryAsyncOpenAI,
    ensure_supported_image_dialect,
    from_litellm_image_usage,
)
from model_gateway.security.transport import PinnedAddressTransport, SyncPinnedAddressTransport

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "dall-e-3"
PUBLIC_ADDRESS = "93.184.216.34"

OPENAI_DIALECT = "openai"
GPT_MODEL_ID = "gpt-image-1"
GENERIC_DIALECTS = ("gemini", "vertex_ai")
UNVERIFIED_DIALECT = "azure"


def make_route(**overrides: object) -> Route:
    """A resolved image route; override ``format`` to model a different dialect."""
    fields: dict[str, object] = {
        "operation": Operation.IMAGE,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": OPENAI_DIALECT,
        "api_key": SECRET_KEY,
        "capabilities": frozenset({Capability.GENERATION}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request(**overrides: object) -> ImageRequest:
    """A canonical image request; override any field to model a different request."""
    fields: dict[str, object] = {
        "model": "image-default",
        "prompt": "a futuristic city at sunset",
    }
    fields.update(overrides)
    return ImageRequest(**fields)


def make_context(**overrides: object) -> AttemptContext:
    """A standalone attempt context; the budget is the provider's only deadline input."""
    fields: dict[str, object] = {
        "invocation_id": "inv-7f3c1a90",
        "attempt_number": 1,
        "timeout": 30.0,
    }
    fields.update(overrides)
    return AttemptContext(**fields)


def image_payload(
    *,
    b64_json: str | None = "aGVsbG8=",
    url: str | None = None,
    revised_prompt: str | None = "a revised prompt",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    """An OpenAI-shaped upstream image payload, with the fields a real provider adds."""
    data: dict[str, object] = {}
    if b64_json is not None:
        data["b64_json"] = b64_json
    if url is not None:
        data["url"] = url
    if revised_prompt is not None:
        data["revised_prompt"] = revised_prompt
    payload: dict[str, object] = {
        "id": "img-9x2f",
        "created": 1730000000,
        "model": "dall-e-3-2024-08-06",
        "data": [data],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def image_response(payload: dict[str, object]) -> ImageResponse:
    """A real ``ImageResponse``, built through LiteLLM's own converter from a payload.

    The converter's image path builds a fresh ``ImageResponse`` from the payload
    rather than mutating the one passed in, so the return value is the object
    with the payload's entries on it.
    """
    response = ImageResponse()
    return convert_to_model_response_object(
        response_object=payload,
        model_response_object=response,
        response_type="image_generation",
    )


class RecordingImageGeneration:
    """A stand-in for ``litellm.aimage_generation`` that records the kwargs it was called with."""

    def __init__(self, response: ImageResponse | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response if response is not None else image_response(image_payload())

    async def __call__(self, **kwargs: object) -> ImageResponse:
        self.calls.append(kwargs)
        return self._response


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Wire a provider to the real per-dialect construction and a recording entry point."""

    def wire_provider(
        response: ImageResponse | None = None,
    ) -> tuple[LiteLLMImageProvider, RecordingImageGeneration]:
        entry = RecordingImageGeneration(response)
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        return LiteLLMImageProvider(), entry

    return wire_provider


def injected_client(entry: RecordingImageGeneration, index: int = 0) -> object:
    """The object the attempt handed LiteLLM as ``client=``."""
    return entry.calls[index]["client"]


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


class TestTheRefusal:
    """A dialect whose image HTTP path is unverified is refused, not run."""

    def test_the_verified_dialects_are_the_three_the_binding_proves(self) -> None:
        assert frozenset({OPENAI_DIALECT, *GENERIC_DIALECTS}) == IMAGE_DIALECTS

    @pytest.mark.parametrize("dialect", sorted(IMAGE_DIALECTS))
    def test_a_verified_dialect_is_returned_unchanged(self, dialect: str) -> None:
        assert ensure_supported_image_dialect(dialect) == dialect

    @pytest.mark.parametrize("dialect", [UNVERIFIED_DIALECT, "anthropic", "openrouter"])
    def test_the_refusal_names_the_dialect_and_the_verified_set(self, dialect: str) -> None:
        with pytest.raises(FailedPreconditionError) as excinfo:
            ensure_supported_image_dialect(dialect)

        message = excinfo.value.message
        assert dialect in message
        for verified in sorted(IMAGE_DIALECTS):
            assert verified in message

    def test_the_refusal_does_not_offer_synchronicity_as_a_reason(self) -> None:
        """A synchronous pinned transport exists and works, so sync is not why a path is refused.

        The generic image dialects are served synchronously and are verified, so
        a message that named synchronicity as a reason for refusal would send a
        maintainer after the wrong thing. The three real reasons are the ones
        stated instead.
        """
        with pytest.raises(FailedPreconditionError) as excinfo:
            ensure_supported_image_dialect(UNVERIFIED_DIALECT)

        message = excinfo.value.message
        assert "discard the injected client" in message
        assert "no usable branch" in message
        assert "not yet verified" in message
        assert "Being synchronous is not one of those reasons" in message
        assert "synchronous pinned transport" in message

    @pytest.mark.parametrize("dialect", [UNVERIFIED_DIALECT, "anthropic", "openrouter"])
    async def test_an_unverified_dialect_is_refused_before_any_call(self, wire, dialect) -> None:
        provider, entry = wire()

        with pytest.raises(FailedPreconditionError):
            await provider.generate(
                make_route(format=dialect), make_request(), make_context()
            )

        assert entry.calls == []

    async def test_custom_transport_inputs_do_not_make_an_unverified_dialect_acceptable(
        self,
    ) -> None:
        """The refusal is about the path the library takes, not about the inputs it is given.

        Every injectable input is custom, and the refusal still wins before any
        client is built.
        """

        def resolver(host: str) -> tuple[str, ...]:
            raise AssertionError(f"the resolver must not be called for {host}")

        provider = LiteLLMImageProvider(resolver=resolver, verify=False)

        with pytest.raises(FailedPreconditionError):
            await provider.generate(
                make_route(format=UNVERIFIED_DIALECT), make_request(), make_context()
            )


# ---------------------------------------------------------------------------
# Which client each dialect's HTTP path is given
# ---------------------------------------------------------------------------


class TestTheOpenAISdkPath:
    """``openai`` goes through the OpenAI SDK, which reads its own fields off ``client=``."""

    async def test_the_sdk_is_given_a_retry_pinned_sdk_client_over_the_pinned_httpx_client(
        self, wire
    ) -> None:
        provider, entry = wire()

        await provider.generate(make_route(), make_request(), make_context())

        client = injected_client(entry)
        assert isinstance(client, _ZeroRetryAsyncOpenAI)
        # The SDK reads these off the object it is handed, which is why a bare
        # httpx client or an HTTP handler fails this path outright.
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert isinstance(http_client, httpx.AsyncClient)
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert isinstance(_pinned_transport_of(client), PinnedAddressTransport)

    def test_the_pinned_client_keeps_its_zero_retry_count_when_the_library_restores_two(
        self,
    ) -> None:
        """The image entry point assigns ``max_retries`` onto an injected client.

        Measured: the library's handler pops ``max_retries`` off the request data
        (default 2 — ``num_retries`` is not read on the image path) and assigns
        it onto the client it is handed, so a plain SDK client is retried twice
        on a 500-ing upstream. The pinned subclass's property makes that
        assignment a no-op.
        """
        client = _ZeroRetryAsyncOpenAI(
            api_key=SECRET_KEY,
            base_url=ENDPOINT,
            http_client=httpx.AsyncClient(),
            max_retries=0,
        )

        client.max_retries = 2  # what _set_dynamic_params_on_client does on this path

        assert client.max_retries == 0


def _sync_pinned_transport_of(handler: object) -> SyncPinnedAddressTransport:
    """The pinned transport the generic handler's client connects through."""
    transport = handler.client._transport  # pyright: ignore[reportPrivateUsage]
    assert isinstance(transport, SyncPinnedAddressTransport)
    return transport


def _pinned_transport_of(client: object) -> PinnedAddressTransport:
    """The pinned transport the OpenAI SDK client connects through."""
    http_client = client._client  # pyright: ignore[reportPrivateUsage]
    transport = http_client._transport  # pyright: ignore[reportPrivateUsage]
    assert isinstance(transport, PinnedAddressTransport)
    return transport


class TestTheGenericImagePath:
    """``gemini`` and ``vertex_ai`` share a synchronous handler that honours a sync handler."""

    @pytest.mark.parametrize("dialect", GENERIC_DIALECTS)
    async def test_the_handler_is_given_the_synchronous_pinned_transport(
        self, wire, dialect
    ) -> None:
        """The image generic path serves synchronously, so the injected client is a sync one.

        Measured against the pinned library: an ``AsyncHTTPHandler`` on this path
        is silently discarded and the call runs on the library's own default
        client, which re-resolves the hostname and follows redirects; a
        synchronous ``HTTPHandler`` is honoured.
        """
        provider, entry = wire()

        await provider.generate(make_route(format=dialect), make_request(), make_context())

        handler = injected_client(entry)
        assert isinstance(handler, HTTPHandler)
        assert handler.follow_redirects is False
        assert isinstance(_sync_pinned_transport_of(handler), SyncPinnedAddressTransport)
        http_client = handler.client
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert not http_client._mounts  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize("dialect", GENERIC_DIALECTS)
    async def test_the_sync_client_is_built_from_the_injected_inputs(
        self, monkeypatch: pytest.MonkeyPatch, dialect
    ) -> None:
        """The pinned transport is built from the caller's own inner transport."""
        built: list[httpx.BaseTransport] = []

        def sync_inner_factory() -> httpx.BaseTransport:
            transport = httpx.HTTPTransport()
            built.append(transport)
            return transport

        def resolver(host: str) -> tuple[str, ...]:
            raise AssertionError(f"the resolver must not be called for {host}")

        entry = RecordingImageGeneration()
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider(
            resolver=resolver, sync_inner_factory=sync_inner_factory
        )

        await provider.generate(make_route(format=dialect), make_request(), make_context())

        handler = injected_client(entry)
        assert len(built) == 1
        assert (
            _sync_pinned_transport_of(handler)._inner is built[0]  # pyright: ignore[reportPrivateUsage]
        )


# ---------------------------------------------------------------------------
# One transport per attempt
# ---------------------------------------------------------------------------


class TestTheAttemptsOwnTransport:
    """The transport is this attempt's, and it is closed when the attempt ends."""

    @pytest.mark.parametrize("dialect", [OPENAI_DIALECT, *GENERIC_DIALECTS])
    async def test_each_attempt_gets_its_own_transport_and_closes_it(
        self, wire, monkeypatch, dialect
    ) -> None:
        closed: list[PinnedAddressTransport | SyncPinnedAddressTransport] = []
        original_aclose = PinnedAddressTransport.aclose
        original_close = SyncPinnedAddressTransport.close

        async def recording_aclose(self: PinnedAddressTransport) -> None:
            closed.append(self)
            await original_aclose(self)

        def recording_close(self: SyncPinnedAddressTransport) -> None:
            closed.append(self)
            original_close(self)

        monkeypatch.setattr(PinnedAddressTransport, "aclose", recording_aclose)
        monkeypatch.setattr(SyncPinnedAddressTransport, "close", recording_close)
        provider, entry = wire()

        await provider.generate(make_route(format=dialect), make_request(), make_context())
        await provider.generate(make_route(format=dialect), make_request(), make_context())

        first = _transport_of(injected_client(entry, 0), dialect)
        second = _transport_of(injected_client(entry, 1), dialect)
        assert first is not second
        assert closed == [first, second]


def _transport_of(client: object, dialect: str) -> PinnedAddressTransport:
    """The pinned transport each dialect's injected client connects through."""
    if dialect in GENERIC_DIALECTS:
        return _sync_pinned_transport_of(client)
    return _pinned_transport_of(client)


# ---------------------------------------------------------------------------
# The real call, driven network-free through the recording transport
# ---------------------------------------------------------------------------


class RecordingAsyncTransport(httpx.AsyncBaseTransport):
    """An inner transport that answers without a socket and records what it was sent."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self._response is not None:
            return self._response
        return httpx.Response(
            200,
            json={
                "created": 1730000000,
                "data": [{"b64_json": "aGVsbG8=", "revised_prompt": "a revised prompt"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "input_tokens_details": {"image_tokens": 5, "text_tokens": 5},
                },
            },
            request=request,
        )


class RecordingSyncTransport(httpx.BaseTransport):
    """The synchronous twin of :class:`RecordingAsyncTransport`."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self._response is not None:
            return self._response
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": "aGVsbG8=",
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                },
            },
            request=request,
        )


def recording_resolver(host: str) -> list[str]:
    """Stand-in for DNS: every hostname resolves to one public address."""
    return [PUBLIC_ADDRESS]


class TestTheRealOpenAISdkCall:
    """The real ``litellm.aimage_generation`` on the SDK path, network-free."""

    async def test_the_request_the_sdk_path_sends(self) -> None:
        """The recorded wire request is the operation the route performs."""
        inner = RecordingAsyncTransport()
        provider = LiteLLMImageProvider(resolver=recording_resolver, inner_factory=lambda: inner)

        response = await provider.generate(
            make_route(extra_headers={"X-Custom": "value"}),
            make_request(size="1024x1024", response_format="b64_json", n=2),
            make_context(),
        )

        assert len(inner.requests) == 1
        request = inner.requests[0]
        # The connection is pinned to the validated address; the hostname is
        # preserved for the Host header and TLS.
        assert request.url.host == PUBLIC_ADDRESS
        assert request.url.path == "/v1/images/generations"
        assert request.headers["host"] == "upstream.internal.example"
        assert request.headers["Authorization"] == f"Bearer {SECRET_KEY}"
        assert request.headers["X-Custom"] == "value"
        body = json.loads(request.content)
        assert body["prompt"] == "a futuristic city at sunset"
        assert body["model"] == UPSTREAM_MODEL
        assert body["n"] == 2
        assert body["size"] == "1024x1024"
        assert body["response_format"] == "b64_json"

        assert isinstance(response.data[0], GeneratedImage)
        assert response.data[0].b64_json == "aGVsbG8="
        assert response.data[0].revised_prompt == "a revised prompt"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.model == "image-default"
        assert response.upstream_model == UPSTREAM_MODEL
        assert response.invocation_id == "inv-7f3c1a90"

    async def test_a_500_upstream_is_answered_by_exactly_one_wire_request(self) -> None:
        """The pinned SDK client makes one wire request per gateway attempt.

        The assertion is the wire count through the real entry point, not the
        client's ``max_retries`` attribute: the attribute is only how the pin is
        implemented, and a library upgrade that set it another way would leave an
        attribute test passing while the retries came back. Measured against the
        pinned library: the image entry point assigns ``max_retries`` (default 2)
        onto the client it is handed, and a plain SDK client is retried twice —
        three wire requests inside one gateway attempt (measured again with the
        pin removed from this same setup: three requests arrive here). The
        binding's pinned subclass keeps the count at zero, so the gateway's
        attempt count and the upstream's request count agree.
        """
        inner = RecordingAsyncTransport(
            httpx.Response(
                500, json={"error": {"message": "upstream exploded", "type": "server_error"}}
            )
        )
        provider = LiteLLMImageProvider(resolver=recording_resolver, inner_factory=lambda: inner)

        with pytest.raises(UnavailableError):
            await provider.generate(make_route(), make_request(), make_context())

        assert len(inner.requests) == 1

    async def test_a_redirect_is_not_followed(self) -> None:
        """A 302 surfaces to the caller rather than being followed."""
        inner = RecordingAsyncTransport(
            httpx.Response(302, headers={"Location": "/elsewhere"}, json={"error": "moved"})
        )
        provider = LiteLLMImageProvider(resolver=recording_resolver, inner_factory=lambda: inner)

        with pytest.raises(InternalError):
            await provider.generate(make_route(), make_request(), make_context())

        assert len(inner.requests) == 1


class TestTheRealGenericCall:
    """The real ``litellm.aimage_generation`` on the generic path, network-free."""

    async def test_the_request_the_generic_path_sends(self) -> None:
        """The Google Imagen transformation reaches the wire as the upstream's own protocol."""
        inner = RecordingSyncTransport()
        provider = LiteLLMImageProvider(
            resolver=recording_resolver, sync_inner_factory=lambda: inner
        )

        response = await provider.generate(
            make_route(format="gemini", model_id="gemini-2.5-flash-image-preview"),
            make_request(size="1024x1024"),
            make_context(),
        )

        assert len(inner.requests) == 1
        request = inner.requests[0]
        # The connection is pinned to the validated address; the hostname is
        # preserved for the Host header and TLS.
        assert request.url.host == PUBLIC_ADDRESS
        assert request.url.path == "/v1/models/gemini-2.5-flash-image-preview:generateContent"
        assert request.headers["host"] == "upstream.internal.example"
        assert request.headers["x-goog-api-key"] == SECRET_KEY
        body = json.loads(request.content)
        assert body["contents"] == [{"parts": [{"text": "a futuristic city at sunset"}]}]
        assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"

        assert isinstance(response.data[0], GeneratedImage)
        assert response.data[0].b64_json == "aGVsbG8="

    async def test_the_vertex_route_sends_the_predict_protocol(self) -> None:
        request_sent = httpx.Request("POST", ENDPOINT)
        inner = RecordingSyncTransport(
            httpx.Response(
                200,
                json={"predictions": [{"bytesBase64Encoded": "aGVsbG8="}]},
                request=request_sent,
            )
        )
        provider = LiteLLMImageProvider(
            resolver=recording_resolver, sync_inner_factory=lambda: inner
        )

        response = await provider.generate(
            make_route(format="vertex_ai", model_id="imagegeneration@006"),
            make_request(),
            make_context(),
        )

        request = inner.requests[0]
        assert request.url.path == "/v1"
        assert request.headers["host"] == "upstream.internal.example"
        body = json.loads(request.content)
        assert body["instances"] == [{"prompt": "a futuristic city at sunset"}]
        assert response.data[0].b64_json == "aGVsbG8="

    async def test_the_generic_path_makes_exactly_one_wire_request_on_a_500(self) -> None:
        """The synchronous handler has no connection-error retry of its own (measured)."""
        inner = RecordingSyncTransport(
            httpx.Response(
                500, json={"error": {"message": "upstream exploded", "type": "server_error"}}
            )
        )
        provider = LiteLLMImageProvider(
            resolver=recording_resolver, sync_inner_factory=lambda: inner
        )

        with pytest.raises(UnavailableError):
            await provider.generate(
                make_route(format="gemini", model_id="gemini-2.5-flash-image-preview"),
                make_request(),
                make_context(),
            )

        assert len(inner.requests) == 1


# ---------------------------------------------------------------------------
# The unsupported-parameter constraint
# ---------------------------------------------------------------------------


class TestTheResponseFormatConstraint:
    """A stated image parameter the configuration does not accept is refused, not forwarded."""

    async def test_a_dall_e_route_forwards_response_format(self, wire) -> None:
        provider, entry = wire()

        await provider.generate(
            make_route(), make_request(response_format="b64_json"), make_context()
        )

        params = entry.calls[0]
        assert params["response_format"] == "b64_json"

    @pytest.mark.parametrize("model_id", [GPT_MODEL_ID, "not-a-real-model"])
    async def test_a_gpt_route_honours_b64_json_by_the_upstreams_default(
        self, wire, model_id
    ) -> None:
        """The OpenAI GPT configuration's upstream produces base64 by default (measured).

        That configuration does not list ``response_format``, so forwarding it
        would raise the library's opaque refusal before any request; the
        caller's b64_json ask is honoured by the upstream's own single
        representation, and the parameter is not forwarded.
        """
        provider, entry = wire()

        await provider.generate(
            make_route(model_id=model_id),
            make_request(response_format="b64_json"),
            make_context(),
        )

        params = entry.calls[0]
        assert "response_format" not in params

    @pytest.mark.parametrize("dialect", GENERIC_DIALECTS)
    async def test_a_generic_route_honours_b64_json_by_the_upstreams_default(
        self, wire, dialect
    ) -> None:
        """The Google Imagen transformations produce base64 by construction (measured).

        The configuration does not accept ``response_format`` on these dialects,
        so forwarding it would raise the library's opaque refusal before any
        request; the caller's b64_json ask is honoured by the upstream's own
        single representation, and the parameter is not forwarded.
        """
        provider, entry = wire()

        await provider.generate(
            make_route(format=dialect), make_request(response_format="b64_json"), make_context()
        )

        params = entry.calls[0]
        assert "response_format" not in params

    @pytest.mark.parametrize(
        "dialect, model_id",
        [
            ("openai", GPT_MODEL_ID),
            ("openai", "not-a-real-model"),
            ("gemini", "gemini-2.5-flash-image-preview"),
            ("vertex_ai", "imagegeneration@006"),
        ],
    )
    async def test_a_route_that_cannot_return_urls_is_refused_naming_the_parameter(
        self, wire, dialect, model_id
    ) -> None:
        provider, entry = wire()

        with pytest.raises(CapabilityError) as excinfo:
            await provider.generate(
                make_route(format=dialect, model_id=model_id),
                make_request(response_format="url"),
                make_context(),
            )

        assert excinfo.value.details["capability"] == "response_format"
        assert entry.calls == []

    async def test_the_librarys_own_refusal_never_reaches_a_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the library refuses a parameter anyway, the failure is a gateway error.

        The binding's pre-check forwards only supported parameters, so the
        library's ``UnsupportedParamsError`` should never fire; if a library
        change makes it fire, the binding classifies it as the capability
        refusal it is, not as an unavailable upstream.
        """
        entry = RecordingImageGeneration()

        async def refusing(**kwargs: object) -> ImageResponse:
            entry.calls.append(kwargs)
            raise litellm.UnsupportedParamsError(
                status_code=400, message="Setting `response_format` is not supported"
            )

        monkeypatch.setattr(litellm, "aimage_generation", refusing)
        provider = LiteLLMImageProvider()

        with pytest.raises(CapabilityError):
            await provider.generate(
                make_route(), make_request(response_format="b64_json"), make_context()
            )


class TestTheBase64OnlyConfigurations:
    """The honour-by-default case is keyed on the measured configuration, not on the value.

    The case exists because three configurations were driven and found to have
    no representation but base64. It must not be expressible as the general rule
    "drop the parameter when the requested value is probably the default",
    because a dialect that merely joins :data:`IMAGE_DIALECTS` would then inherit
    it without any measurement of its own.
    """

    @pytest.mark.parametrize(
        "dialect, model_id",
        [
            ("openai", GPT_MODEL_ID),
            ("openai", "not-a-real-model"),
            ("gemini", "gemini-2.5-flash-image-preview"),
            ("gemini", "imagen-3.0-generate-002"),
            ("vertex_ai", "imagegeneration@006"),
            ("vertex_ai", "imagen-3.0-generate-002"),
        ],
    )
    def test_every_verified_route_that_drops_the_parameter_is_a_measured_configuration(
        self, dialect: str, model_id: str
    ) -> None:
        """Every verified route that does not forward ``response_format`` is one that was measured.

        The b64_json case is true of three configurations and of nothing else.
        A library upgrade that hands a verified dialect a different
        configuration object fails here — and in the binding it turns the drop
        into a refusal, which is loud, rather than into a silent claim that an
        unmeasured upstream returned what was asked for.
        """
        config = _route_image_config(make_route(format=dialect, model_id=model_id))

        assert "response_format" not in config.supported
        assert config.base64_only

    def test_a_measured_configuration_drops_the_parameter(self) -> None:
        config = _ImageConfig(
            configuration=GPTImageGenerationConfig(), supported=frozenset()
        )
        params: dict[str, object] = {}

        _forward_response_format(params, config, "b64_json")

        assert params == {}

    def test_an_unmeasured_configuration_cannot_inherit_the_case(self) -> None:
        """The branch is keyed on the configuration's class, not on the stated value.

        ``DallE3ImageGenerationConfig`` is not a measured base64-only
        configuration — its own upstream honours ``response_format`` — so a
        route whose configuration is that one, with the parameter stripped from
        what it accepts, models a dialect that has joined the set without being
        measured. The stated ``b64_json`` is refused rather than dropped.
        """
        config = _ImageConfig(
            configuration=DallE3ImageGenerationConfig(), supported=frozenset()
        )
        params: dict[str, object] = {}

        with pytest.raises(CapabilityError) as excinfo:
            _forward_response_format(params, config, "b64_json")

        assert params == {}
        assert excinfo.value.details["capability"] == "response_format"


# ---------------------------------------------------------------------------
# The translation of the upstream answer
# ---------------------------------------------------------------------------


class TestTheResponseTranslation:
    """Every semantically relevant field is mapped or the answer is refused."""

    async def test_the_happy_path_maps_every_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = RecordingImageGeneration(
            image_response(
                image_payload(
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30,
                        "input_tokens_details": {"image_tokens": 5, "text_tokens": 5},
                    }
                )
            )
        )
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        response = await provider.generate(make_route(), make_request(), make_context())

        assert response.model == "image-default"
        assert response.upstream_model == UPSTREAM_MODEL
        assert response.invocation_id == "inv-7f3c1a90"
        # The upstream's own identity is neither mapped nor refused: the
        # gateway stamps its own assembly time.
        assert response.created != 1730000000
        (image,) = response.data
        assert image.b64_json == "aGVsbG8="
        assert image.url is None
        assert image.revised_prompt == "a revised prompt"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.usage.estimated_fields == frozenset()

    async def test_a_url_entry_maps_onto_the_canonical_location(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = RecordingImageGeneration(
            image_response(image_payload(b64_json=None, url="https://cdn.example/img.png"))
        )
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        response = await provider.generate(make_route(), make_request(), make_context())

        (image,) = response.data
        assert image.url == "https://cdn.example/img.png"
        assert image.b64_json is None

    async def test_a_provider_specific_field_on_an_entry_is_refused_naming_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Gemini answer's thought_signature arrives on the entry (measured).

        The canonical image has no target for it, and silently dropping it would
        hide a content-provenance annotation the caller was never told about, so
        the answer is refused and another eligible route may serve it.
        """
        entry = RecordingImageGeneration(
            ImageResponse(
                created=1730000000,
                data=[
                    ImageObject(
                        b64_json="aGVsbG8=",
                        provider_specific_fields={"thought_signature": "sig-abc"},
                    )
                ],
            )
        )
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        with pytest.raises(MalformedResponseError) as excinfo:
            await provider.generate(make_route(), make_request(), make_context())

        assert "thought_signature" in excinfo.value.message

    async def test_an_entry_with_no_image_in_it_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = RecordingImageGeneration(
            ImageResponse(
                created=1730000000,
                data=[ImageObject(b64_json=None, url=None)],
            )
        )
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        with pytest.raises(MalformedResponseError):
            await provider.generate(make_route(), make_request(), make_context())

    async def test_an_answer_with_no_images_at_all_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response reporting a generation and returning none is not an answer."""
        entry = RecordingImageGeneration(ImageResponse(created=1730000000, data=[]))
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        with pytest.raises(MalformedResponseError):
            await provider.generate(make_route(), make_request(), make_context())

    async def test_a_usage_annotation_that_cannot_be_carried_is_refused_naming_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage.web_search_requests = 3
        entry = RecordingImageGeneration(
            ImageResponse(
                created=1730000000,
                data=[ImageObject(b64_json="aGVsbG8=")],
            )
        )
        # The library's own transform assigns the usage attribute rather than
        # passing it to the constructor, whose field type would reject the
        # LiteLLMUsage shape — so the annotation is set the way it arrives.
        entry._response.usage = usage
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        with pytest.raises(MalformedResponseError) as excinfo:
            await provider.generate(make_route(), make_request(), make_context())

        assert "web_search_requests" in excinfo.value.message

    async def test_an_unknown_top_level_field_is_refused_until_it_is_classified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Classification is an act: an unexamined provider annotation cannot pass.

        The pinned library drops unknown top-level fields before the binding
        sees them (measured), so this pins the classification of one that does
        arrive: the backend fingerprint is operational metadata and is ignored;
        anything else is refused with the field named.
        """
        entry = RecordingImageGeneration(
            ImageResponse(
                created=1730000000,
                data=[ImageObject(b64_json="aGVsbG8=")],
            )
        )
        monkeypatch.setattr(litellm, "aimage_generation", entry)
        provider = LiteLLMImageProvider()

        entry._response.system_fingerprint = "fp_abc"
        ignored = await provider.generate(make_route(), make_request(), make_context())
        assert ignored.data[0].b64_json == "aGVsbG8="

        entry._response.some_new_annotation = "examine me"
        with pytest.raises(MalformedResponseError) as excinfo:
            await provider.generate(make_route(), make_request(), make_context())
        assert "some_new_annotation" in excinfo.value.message


class TestTheUsageTranslation:
    """The usage discipline is the embeddings path's, for the same measured reason."""

    def test_a_non_zero_count_is_the_provider_reported_fact(self) -> None:
        response = image_response(
            image_payload(
                usage={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "input_tokens_details": {"image_tokens": 5, "text_tokens": 5},
                }
            )
        )

        canonical = from_litellm_image_usage(response.usage)
        assert canonical.input_tokens == 10
        assert canonical.output_tokens == 20
        assert canonical.total_tokens == 30
        assert canonical.estimated_fields == frozenset()

    def test_the_fabricated_all_zero_placeholder_is_recorded_as_unknown(self) -> None:
        """The library fabricates an all-zero usage when the upstream reported none.

        Measured, and the same irrecoverable binding limitation as the
        embeddings path's: the placeholder is indistinguishable from a reported
        zero, so a zero count is never presented as a provider fact.
        """
        response = image_response(image_payload(usage=None))

        canonical = from_litellm_image_usage(response.usage)

        assert canonical.input_tokens is None
        assert canonical.output_tokens is None
        assert canonical.total_tokens is None
        assert canonical.cached_tokens is None

    def test_a_zero_total_beside_non_zero_parts_is_derived_and_named_an_estimate(
        self,
    ) -> None:
        usage = LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=0)

        canonical = from_litellm_image_usage(usage)

        assert canonical.input_tokens == 10
        assert canonical.output_tokens == 20
        assert canonical.total_tokens == 30
        assert canonical.estimated_fields == frozenset({"total_tokens"})

    def test_an_absent_usage_is_unknown_counts(self) -> None:
        canonical = from_litellm_image_usage(None)

        assert canonical.input_tokens is None
        assert canonical.output_tokens is None
        assert canonical.total_tokens is None


# ---------------------------------------------------------------------------
# The request translation
# ---------------------------------------------------------------------------


class TestTheRequestTranslation:
    """The call carries the route's addressing, the prompt, and the attempt's controls."""

    async def test_the_call_carries_everything_the_upstream_needs(self, wire) -> None:
        provider, entry = wire()

        await provider.generate(
            make_route(extra_headers={"X-Custom": "value"}),
            make_request(size="1024x1024", n=3),
            make_context(timeout=12.5),
        )

        params = entry.calls[0]
        assert params["model"] == f"{OPENAI_DIALECT}/{UPSTREAM_MODEL}"
        assert params["api_base"] == ENDPOINT
        assert params["api_key"] == SECRET_KEY
        assert params["extra_headers"] == {"X-Custom": "value"}
        assert params["prompt"] == "a futuristic city at sunset"
        assert params["num_retries"] == 0
        assert params["timeout"] == 12.5
        assert params["n"] == 3
        assert params["size"] == "1024x1024"

    async def test_no_timeout_budget_is_passed_through_as_none(self, wire) -> None:
        provider, entry = wire()

        await provider.generate(make_route(), make_request(), make_context(timeout=None))

        assert entry.calls[0]["timeout"] is None

    async def test_unstated_parameters_are_not_sent(self, wire) -> None:
        provider, entry = wire()

        await provider.generate(make_route(), make_request(), make_context())

        params = entry.calls[0]
        assert "n" not in params
        assert "size" not in params
        assert "response_format" not in params

    async def test_caller_bookkeeping_is_never_forwarded(self, wire) -> None:
        """``metadata`` is the caller's, not the upstream's."""
        provider, entry = wire()

        await provider.generate(
            make_route(), make_request(metadata={"tenant": "acme"}), make_context()
        )

        params = entry.calls[0]
        assert "metadata" not in params
        assert "user" not in params


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestTheErrorMapping:
    """Every failure the call raises is classified into the gateway taxonomy."""

    async def test_an_unavailable_upstream_is_mapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            litellm,
            "aimage_generation",
            _raising(
                litellm.InternalServerError(
                    message="boom", llm_provider="openai", model="dall-e-3"
                )
            ),
        )
        provider = LiteLLMImageProvider()

        with pytest.raises(UnavailableError):
            await provider.generate(make_route(), make_request(), make_context())

    async def test_a_connection_failure_is_mapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            litellm,
            "aimage_generation",
            _raising(
                litellm.APIConnectionError(
                    message="no route", llm_provider="openai", model="dall-e-3"
                )
            ),
        )
        provider = LiteLLMImageProvider()

        with pytest.raises(UnavailableError):
            await provider.generate(make_route(), make_request(), make_context())

    async def test_a_redirect_surfacing_on_the_sdk_path_is_an_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            litellm,
            "aimage_generation",
            _raising(
                litellm.APIError(
                    status_code=302, message="moved", llm_provider="openai", model="dall-e-3"
                )
            ),
        )
        provider = LiteLLMImageProvider()

        with pytest.raises(InternalError):
            await provider.generate(make_route(), make_request(), make_context())

    async def test_cancellation_propagates_unmapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def cancelling(**kwargs: object) -> ImageResponse:
            raise asyncio.CancelledError()

        monkeypatch.setattr(litellm, "aimage_generation", cancelling)
        provider = LiteLLMImageProvider()

        with pytest.raises(asyncio.CancelledError):
            await provider.generate(make_route(), make_request(), make_context())


def _raising(error: Exception):
    async def raise_it(**kwargs: object) -> ImageResponse:
        raise error

    return raise_it


# ---------------------------------------------------------------------------
# The provider satisfies the protocol
# ---------------------------------------------------------------------------


class TestTheProtocol:
    def test_the_provider_satisfies_the_image_provider_protocol(self) -> None:
        assert isinstance(LiteLLMImageProvider(), ImageProvider)
