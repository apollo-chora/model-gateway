"""The client the binding injects, per provider dialect (PRD sections 9 and 14.1).

The security boundary here is *which object* ``litellm.acompletion`` is handed,
because the pinned library honours a different one on each of its HTTP paths and
silently discards the rest. So these tests inspect that object rather than the
network: the completion is replaced with a recording stand-in and the transport
never opens a socket. No test reaches the network.

Two claims are proved rather than asserted, both against the pinned library's own
code so that a change in it fails here instead of at the security boundary:

* the object each supported dialect receives is one its path honours, and
* the generic httpx path's connection-error retry — which rebuilds its client
  through ``AsyncHTTPHandler.create_client`` — rebuilds it on the pinned
  transport, so the retry cannot leave the controls.

A dialect outside :data:`~model_gateway.providers.litellm_provider.SUPPORTED_DIALECTS`
is refused, at the attempt and at startup.
"""

import httpx
import litellm
import openai
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import ModelResponse

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import ConfigurationError, Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import InvocationRequest, Message, Role, TextPart
from model_gateway.errors import FailedPreconditionError
from model_gateway.inference import InferenceService
from model_gateway.providers.base import AttemptContext
from model_gateway.providers.litellm_provider import (
    SUPPORTED_DIALECTS,
    LiteLLMProvider,
    ensure_supported_dialect,
)
from model_gateway.registry import ModelRegistry
from model_gateway.security.transport import PinnedAddressTransport

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "gpt-4o"

OPENAI_SDK_DIALECT = "openai"
HTTP_HANDLER_DIALECTS = ("anthropic", "openrouter")
UNVERIFIED_DIALECT = "vertex_ai"


def make_route(**overrides: object) -> Route:
    """A resolved text route; override ``format`` to model a different dialect."""
    fields: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": OPENAI_SDK_DIALECT,
        "api_key": SECRET_KEY,
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request() -> InvocationRequest:
    """A canonical invocation, minimal and well-formed."""
    return InvocationRequest(
        model="text-default",
        messages=(Message(role=Role.USER, content=(TextPart(text="hi"),)),),
        invocation_id="inv-7f3c1a90",
    )


def make_context(**overrides: object) -> AttemptContext:
    """A standalone attempt context; the budget is the provider's only deadline input."""
    fields: dict[str, object] = {
        "invocation_id": "inv-7f3c1a90",
        "attempt_number": 1,
        "timeout": 30.0,
    }
    fields.update(overrides)
    return AttemptContext(**fields)


def completion_response() -> ModelResponse:
    """A real ``ModelResponse``, built through LiteLLM's own converter from a payload."""
    response = ModelResponse()
    convert_to_model_response_object(
        response_object={
            "id": "chatcmpl-9x2f",
            "object": "chat.completion",
            "created": 1730000000,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        model_response_object=response,
    )
    return response


class RecordingCompletion:
    """A stand-in for ``litellm.acompletion`` that records the kwargs it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        return completion_response()


class StubClient:
    """A stand-in for a client a caller supplied: buildable and closable, never connected."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "StubClient":
        self.entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True


class StubClientFactory:
    """A factory that builds a fresh stub client per call and records what it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.clients: list[StubClient] = []

    def __call__(self, **kwargs: object) -> StubClient:
        self.calls.append(kwargs)
        client = StubClient()
        self.clients.append(client)
        return client


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Wire a provider to the real per-dialect construction and a recording completion."""

    def wire_provider() -> tuple[LiteLLMProvider, RecordingCompletion]:
        completion = RecordingCompletion()
        monkeypatch.setattr(litellm, "acompletion", completion)
        return LiteLLMProvider(), completion

    return wire_provider


def injected_client(completion: RecordingCompletion, index: int = 0) -> object:
    """The object the attempt handed LiteLLM as ``client=``."""
    return completion.calls[index]["client"]


def pinned_transport_of(client: object) -> PinnedAddressTransport:
    """The pinned transport ``client`` connects through, whichever shape it is.

    Both shapes keep the httpx client that owns the connection in one attribute;
    the OpenAI SDK's is private, which is why this is a helper rather than an
    assertion in each test.
    """
    http_client = (
        client._client  # pyright: ignore[reportPrivateUsage]  # the SDK keeps its httpx client here
        if isinstance(client, openai.AsyncOpenAI)
        else client.client  # pyright: ignore[reportAttributeAccessIssue]  # AsyncHTTPHandler property
    )
    transport = http_client._transport  # pyright: ignore[reportPrivateUsage]  # httpx keeps its transport here
    assert isinstance(transport, PinnedAddressTransport)
    return transport


# ---------------------------------------------------------------------------
# Which client each dialect's HTTP path is given
# ---------------------------------------------------------------------------


class TestTheOpenAISdkPath:
    """``openai`` goes through the OpenAI SDK, which reads its own fields off ``client=``."""

    async def test_the_sdk_is_given_an_sdk_client_over_the_pinned_httpx_client(self, wire) -> None:
        provider, completion = wire()

        await provider.complete(
            make_route(format=OPENAI_SDK_DIALECT), make_request(), make_context()
        )

        client = injected_client(completion)
        assert isinstance(client, openai.AsyncOpenAI)
        # The SDK reads these off the object it is handed, which is why a bare
        # httpx client fails this path outright rather than losing the controls.
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        assert client.max_retries == 0
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert isinstance(http_client, httpx.AsyncClient)
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)


class TestTheGenericHttpxPath:
    """``anthropic`` and ``openrouter`` share a handler that discards anything else."""

    @pytest.mark.parametrize("dialect", HTTP_HANDLER_DIALECTS)
    async def test_the_handler_is_given_the_pinned_transport_itself(self, wire, dialect) -> None:
        provider, completion = wire()

        await provider.complete(make_route(format=dialect), make_request(), make_context())

        handler = injected_client(completion)
        assert isinstance(handler, AsyncHTTPHandler)
        assert isinstance(handler.transport, PinnedAddressTransport)
        assert handler.follow_redirects is False
        assert isinstance(handler.client._transport, PinnedAddressTransport)  # pyright: ignore[reportPrivateUsage]
        assert handler.client.follow_redirects is False

    @pytest.mark.parametrize("dialect", HTTP_HANDLER_DIALECTS)
    async def test_the_handlers_connection_error_retry_rebuilds_on_the_pinned_transport(
        self, wire, dialect
    ) -> None:
        """The retry that motivated this: a rebuilt client must still be ours.

        ``AsyncHTTPHandler.post`` catches a connection error and calls
        ``create_client`` for a second attempt, which is the code path that
        escapes the controls when the handler was built without a transport of
        its own. This pins that the rebuild carries the pinned transport, with
        redirect following off and no environment proxy.
        """
        provider, completion = wire()
        await provider.complete(make_route(format=dialect), make_request(), make_context())
        handler = injected_client(completion)
        assert isinstance(handler, AsyncHTTPHandler)

        rebuilt = handler.create_client(timeout=httpx.Timeout(5.0), event_hooks=None)

        assert rebuilt._transport is handler.transport  # pyright: ignore[reportPrivateUsage]
        assert isinstance(rebuilt._transport, PinnedAddressTransport)  # pyright: ignore[reportPrivateUsage]
        assert rebuilt.follow_redirects is False
        assert rebuilt.trust_env is False
        await rebuilt.aclose()

    @pytest.mark.parametrize("dialect", HTTP_HANDLER_DIALECTS)
    async def test_the_httpx_client_of_the_generic_path_never_reaches_the_environment(
        self, wire, dialect
    ) -> None:
        """No proxy is installed between the gateway and the upstream (PRD section 14.1)."""
        provider, completion = wire()

        await provider.complete(make_route(format=dialect), make_request(), make_context())

        handler = injected_client(completion)
        assert handler.client.trust_env is False  # pyright: ignore[reportAttributeAccessIssue]
        assert not handler.client._mounts  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# One transport per attempt
# ---------------------------------------------------------------------------


class TestTheAttemptsOwnTransport:
    """The transport is this attempt's, and it is closed when the attempt ends."""

    @pytest.mark.parametrize(
        "dialect", [OPENAI_SDK_DIALECT, *HTTP_HANDLER_DIALECTS]
    )
    async def test_each_attempt_gets_its_own_transport_and_closes_it(
        self, wire, monkeypatch, dialect
    ) -> None:
        closed: list[PinnedAddressTransport] = []
        original = PinnedAddressTransport.aclose

        async def recording(self: PinnedAddressTransport) -> None:
            closed.append(self)
            await original(self)

        monkeypatch.setattr(PinnedAddressTransport, "aclose", recording)
        provider, completion = wire()

        await provider.complete(make_route(format=dialect), make_request(), make_context())
        await provider.complete(make_route(format=dialect), make_request(), make_context())

        first = pinned_transport_of(injected_client(completion, 0))
        second = pinned_transport_of(injected_client(completion, 1))
        assert first is not second
        assert closed == [first, second]


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


class TestTheRefusal:
    """A dialect whose HTTP path is unverified is refused, not run (PRD section 14.1)."""

    def test_the_verified_dialects_are_the_three_the_binding_proves(self) -> None:
        assert frozenset(
            {OPENAI_SDK_DIALECT, *HTTP_HANDLER_DIALECTS}
        ) == SUPPORTED_DIALECTS

    @pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
    def test_a_verified_dialect_is_returned_unchanged(self, dialect: str) -> None:
        assert ensure_supported_dialect(dialect) == dialect

    def test_the_refusal_names_the_dialect_and_the_verified_set(self) -> None:
        with pytest.raises(FailedPreconditionError) as excinfo:
            ensure_supported_dialect(UNVERIFIED_DIALECT)

        message = excinfo.value.message
        assert UNVERIFIED_DIALECT in message
        for dialect in sorted(SUPPORTED_DIALECTS):
            assert dialect in message

    async def test_an_unverified_dialect_is_refused_before_any_call(self, wire) -> None:
        provider, completion = wire()

        with pytest.raises(FailedPreconditionError):
            await provider.complete(
                make_route(format=UNVERIFIED_DIALECT), make_request(), make_context()
            )

        assert completion.calls == []

    async def test_an_explicit_factory_does_not_make_an_unverified_dialect_acceptable(self) -> None:
        """The refusal is about the path the library takes, not about the client it is given."""
        factory = StubClientFactory()
        provider = LiteLLMProvider(client_factory=factory)

        with pytest.raises(FailedPreconditionError):
            await provider.complete(
                make_route(format=UNVERIFIED_DIALECT), make_request(), make_context()
            )

        assert factory.calls == []

    async def test_an_explicit_factory_still_supplies_the_client_for_a_verified_dialect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The seam is unchanged: what the factory returns is what LiteLLM receives."""
        factory = StubClientFactory()
        completion = RecordingCompletion()
        monkeypatch.setattr(litellm, "acompletion", completion)
        provider = LiteLLMProvider(client_factory=factory)

        await provider.complete(
            make_route(format=HTTP_HANDLER_DIALECTS[0]), make_request(), make_context()
        )

        assert factory.calls == [{"budget": 30.0}]
        assert injected_client(completion) is factory.clients[0]


class _StubProvider:
    """A provider an injected service can be built over; the factory never calls it."""

    async def complete(self, route: Route, request: InvocationRequest, context: AttemptContext):
        raise AssertionError("the stub provider is never called")


class TestTheStartupRefusal:
    """A deployment that cannot run its text routes does not start (PRD sections 6 and 21)."""

    def test_a_text_route_with_an_unverified_dialect_fails_the_start(self) -> None:
        registry = ModelRegistry((make_route(format=UNVERIFIED_DIALECT),))

        with pytest.raises(ConfigurationError) as excinfo:
            create_app(Settings.from_env({}), registry=registry)

        assert UNVERIFIED_DIALECT in str(excinfo.value)

    @pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
    def test_a_text_route_with_a_verified_dialect_starts(self, dialect: str) -> None:
        registry = ModelRegistry((make_route(format=dialect),))

        assert create_app(Settings.from_env({}), registry=registry) is not None

    def test_an_injected_service_is_not_validated(self) -> None:
        """An injected service brings its own provider, so the factory cannot judge its routes."""
        registry = ModelRegistry((make_route(format=UNVERIFIED_DIALECT),))
        service = InferenceService(registry, _StubProvider())

        assert create_app(Settings.from_env({}), registry=registry, service=service) is not None
