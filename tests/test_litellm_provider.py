"""Tests for the LiteLLM provider: one gateway attempt through the pinned binding
(PRD 9, 11.3, 14.1).

No test reaches the network: ``litellm.acompletion`` is monkeypatched with a
recording stand-in, so the real per-dialect client is built and closed around a
transport that is never asked for a request, and the response fixtures are the
real LiteLLM types — built through LiteLLM's own converter from a realistic
upstream payload — so what runs is exactly what the installed binding does with
a real provider's answer.

The failure paths construct LiteLLM's and the OpenAI SDK's exception classes
in memory, which is where the mapping of PRD section 5.4 lives. The tests that
pin the ``.env`` guard run a binding module's import in a subprocess with a
``.env`` in its working directory: that side effect only exists at import time,
and only a fresh process can show whether the seam suppressed it.
"""

import asyncio
import inspect
import os
import ssl
import subprocess
import sys
from pathlib import Path

import httpx
import litellm
import openai
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import ModelResponse

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    RefusalPart,
    Role,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    InternalError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.providers.base import AttemptContext, TextProvider
from model_gateway.providers.litellm_provider import SUPPORTED_DIALECTS, LiteLLMProvider
from model_gateway.security.transport import PinnedAddressTransport

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "gpt-4o"
PUBLIC_ADDRESS = "93.184.216.34"

ALL_CAPABILITIES = frozenset(
    {
        Capability.CHAT,
        Capability.TOOLS,
        Capability.VISION,
        Capability.STRUCTURED_OUTPUT,
    }
)


def make_route(**overrides: object) -> Route:
    """A resolved text route; override any field to model a different configuration."""
    fields: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": "openai",
        "api_key": SECRET_KEY,
        "capabilities": ALL_CAPABILITIES,
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request(**overrides: object) -> InvocationRequest:
    """A canonical invocation; override any field to model a different request."""
    fields: dict[str, object] = {
        "model": "text-default",
        "messages": (
            Message(role=Role.USER, content=(TextPart(text="What is the weather in Paris?"),)),
        ),
        "invocation_id": "inv-7f3c1a90",
    }
    fields.update(overrides)
    return InvocationRequest(**fields)


def make_context(**overrides: object) -> AttemptContext:
    """A standalone attempt context; the budget is the provider's only deadline input."""
    fields: dict[str, object] = {
        "invocation_id": "inv-7f3c1a90",
        "attempt_number": 1,
        "timeout": 30.0,
    }
    fields.update(overrides)
    return AttemptContext(**fields)


def http_error(status: int) -> httpx.Response:
    """An HTTP error response with a request attached, as LiteLLM exceptions carry."""
    return httpx.Response(
        status_code=status,
        request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions"),
    )


def completion_payload(**overrides: object) -> dict[str, object]:
    """A realistic OpenAI-shaped upstream completion payload."""
    payload: dict[str, object] = {
        "id": "chatcmpl-9x2f",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recursion is a technique...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 42,
            "total_tokens": 67,
        },
    }
    payload.update(overrides)
    return payload


def response_from(payload: dict[str, object]) -> ModelResponse:
    """Build a real ``ModelResponse`` by parsing a payload through LiteLLM's own converter.

    This is the shape a real provider response has once LiteLLM has decoded it,
    including the fields LiteLLM itself adds or normalises.
    """
    response = ModelResponse()
    convert_to_model_response_object(response_object=payload, model_response_object=response)
    return response


def pinned_transport_of(client: object) -> PinnedAddressTransport:
    """The pinned transport ``client`` connects through, whichever shape it is.

    Both shapes keep the httpx client that owns the connection in one
    attribute; the OpenAI SDK's is private, which is why this is a helper
    rather than an assertion in each test.
    """
    http_client = (
        client._client  # pyright: ignore[reportPrivateUsage]  # the SDK keeps its httpx client here
        if isinstance(client, openai.AsyncOpenAI)
        else client.client  # pyright: ignore[reportAttributeAccessIssue]  # AsyncHTTPHandler property
    )
    transport = http_client._transport  # pyright: ignore[reportPrivateUsage]  # httpx keeps its transport here
    assert isinstance(transport, PinnedAddressTransport)
    return transport


def recording_aclose(monkeypatch: pytest.MonkeyPatch) -> list[PinnedAddressTransport]:
    """Record every pinned transport closed while the test runs."""
    closed: list[PinnedAddressTransport] = []
    original = PinnedAddressTransport.aclose

    async def record(self: PinnedAddressTransport) -> None:
        closed.append(self)
        await original(self)

    monkeypatch.setattr(PinnedAddressTransport, "aclose", record)
    return closed


class RecordingCompletion:
    """A stand-in for ``litellm.acompletion`` that records each call and answers from a script."""

    def __init__(self, outcome: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._outcome = outcome

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Wire a provider to a scripted completion; the real client is built, never connected."""

    def wire_outcome(
        outcome: object,
    ) -> tuple[LiteLLMProvider, RecordingCompletion]:
        completion = RecordingCompletion(outcome)
        monkeypatch.setattr(litellm, "acompletion", completion)
        return LiteLLMProvider(), completion

    return wire_outcome


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestTheCall:
    """What one attempt hands to the pinned binding, and what comes back."""

    async def test_the_parameters_are_the_translation_plus_the_attempts_own_controls(
        self, wire
    ) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert call["model"] == "openai/gpt-4o"
        assert call["api_base"] == ENDPOINT
        assert call["api_key"] == SECRET_KEY
        assert call["extra_headers"] == {}
        assert call["messages"] == [
            {"role": "user", "content": "What is the weather in Paris?"}
        ]
        assert call["num_retries"] == 0
        assert call["timeout"] == 30.0

    async def test_the_injected_client_is_the_sdk_client_over_the_pinned_transport(
        self, wire
    ) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        client = call["client"]
        assert isinstance(client, openai.AsyncOpenAI)
        # The SDK reads these off the object it is handed, which is why a bare
        # httpx client fails this path outright rather than losing the controls.
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        assert client.max_retries == 0
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)

    async def test_the_client_is_closed_when_the_attempt_ends(self, wire, monkeypatch) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert closed == [pinned_transport_of(call["client"])]

    async def test_the_attempt_budget_is_the_calls_timeout(self, wire) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=12.5))

        assert completion.calls[0]["timeout"] == 12.5

    async def test_no_budget_is_passed_through_as_none(self, wire) -> None:
        """The gateway set no budget; the binding invents none of its own."""
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=None))

        assert completion.calls[0]["timeout"] is None

    async def test_each_attempt_gets_a_fresh_client_with_its_own_budget(
        self, wire, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context(timeout=7.0))
        await provider.complete(make_route(), make_request(), make_context(timeout=7.0))

        first, second = (call["client"] for call in completion.calls)
        assert first is not second
        assert closed == [pinned_transport_of(first), pinned_transport_of(second)]
        assert first.timeout == httpx.Timeout(7.0)
        assert second.timeout == httpx.Timeout(7.0)

    async def test_the_call_is_made_exactly_once_per_attempt(self, wire) -> None:
        provider, completion = wire(response_from(completion_payload()))

        await provider.complete(make_route(), make_request(), make_context())

        assert len(completion.calls) == 1

    async def test_the_instruction_is_forwarded_as_its_own_system_message(self, wire) -> None:
        """The invocation-level instruction and a conversation system message stay unmerged."""
        provider, completion = wire(response_from(completion_payload()))
        request = make_request(
            system_instruction="answer concisely",
            messages=(
                Message(role=Role.SYSTEM, content=(TextPart(text="be terse"),)),
                Message(role=Role.USER, content=(TextPart(text="hi"),)),
            ),
        )

        await provider.complete(make_route(), request, make_context())

        assert completion.calls[0]["messages"] == [
            {"role": "system", "content": "answer concisely"},
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    async def test_the_answer_is_normalised_into_the_canonical_response(self, wire) -> None:
        provider, _ = wire(response_from(completion_payload()))

        response = await provider.complete(make_route(), make_request(), make_context())

        assert isinstance(response, InvocationResponse)
        assert response.model == "text-default"
        assert response.upstream_model == UPSTREAM_MODEL
        assert response.invocation_id == "inv-7f3c1a90"
        assert response.text == "Recursion is a technique..."
        assert response.finish_reason is FinishReason.STOP
        assert response.raw_finish_reason == "stop"
        assert response.usage == Usage(input_tokens=25, output_tokens=42, total_tokens=67)

    async def test_a_tool_call_survives_the_round_trip(self, wire) -> None:
        payload = completion_payload(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_01",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )
        provider, _ = wire(response_from(payload))

        response = await provider.complete(make_route(), make_request(), make_context())

        assert response.tool_calls == (
            ToolCall(id="call_01", name="get_weather", arguments={"city": "Paris"}),
        )
        assert response.finish_reason is FinishReason.TOOL_CALLS


# ---------------------------------------------------------------------------
# Failure mapping
# ---------------------------------------------------------------------------


class TestFailures:
    """Every failure the call raises is mapped onto the gateway taxonomy (PRD 5.4, 9)."""

    @pytest.mark.parametrize(
        ("error", "category", "reason"),
        [
            (
                litellm.AuthenticationError(
                    message="bad key",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(401),
                ),
                UnauthenticatedError,
                "invalid_credentials",
            ),
            (
                litellm.PermissionDeniedError(
                    message="forbidden",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(403),
                ),
                PermissionDeniedError,
                "permission_denied",
            ),
            (
                litellm.NotFoundError(
                    message="no such model",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(404),
                ),
                NotFoundError,
                "resource_not_found",
            ),
            (
                litellm.RateLimitError(
                    message="slow down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(429),
                ),
                ResourceExhaustedError,
                "rate_limited",
            ),
            (
                litellm.Timeout(message="too slow", model=UPSTREAM_MODEL, llm_provider="openai"),
                DeadlineExceededError,
                "deadline_exceeded",
            ),
            (
                litellm.APIConnectionError(
                    message="connection reset",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIResponseValidationError(
                    message="bad shape",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIError(
                    status_code=503,
                    message="down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                litellm.APIError(
                    status_code=400,
                    message="bad",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.BadRequestError(
                    message="bad request",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(400),
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.ContextWindowExceededError(
                    message="too long",
                    model=UPSTREAM_MODEL,
                    llm_provider="openai",
                    response=http_error(400),
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.UnsupportedParamsError(
                    message="unsupported param",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                ),
                InternalError,
                "internal_error",
            ),
            (
                litellm.ServiceUnavailableError(
                    message="down",
                    llm_provider="openai",
                    model=UPSTREAM_MODEL,
                    response=http_error(503),
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APITimeoutError(
                    request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions")
                ),
                DeadlineExceededError,
                "deadline_exceeded",
            ),
            (
                openai.APIConnectionError(
                    request=httpx.Request(method="POST", url=f"{ENDPOINT}/chat/completions")
                ),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APIStatusError("server exploded", response=http_error(500), body=None),
                UnavailableError,
                "provider_unavailable",
            ),
            (
                openai.APIStatusError("bad request", response=http_error(400), body=None),
                InternalError,
                "internal_error",
            ),
            (
                RuntimeError("something the binding did not expect"),
                InternalError,
                "internal_error",
            ),
        ],
    )
    async def test_every_failure_is_mapped_onto_the_taxonomy(
        self, wire, error: Exception, category: type, reason: str
    ) -> None:
        provider, _ = wire(error)

        with pytest.raises(category) as excinfo:
            await provider.complete(make_route(), make_request(), make_context())

        assert excinfo.value.reason.value == reason
        assert excinfo.value.__cause__ is error

    async def test_the_upstream_message_never_reaches_the_raised_error(self, wire) -> None:
        """The taxonomy sanitises its own payloads; no upstream text is forwarded (PRD 5.4)."""
        secret = SECRET_KEY
        url = "https://api.openai.com/v1/chat/completions"
        error = litellm.AuthenticationError(
            message=f"invalid key {secret} for {url}",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(401),
        )
        provider, _ = wire(error)

        with pytest.raises(UnauthenticatedError) as excinfo:
            await provider.complete(make_route(), make_request(), make_context())

        payload = excinfo.value.to_payload(include_details=True)
        assert secret not in str(payload)
        assert url not in str(payload)

    async def test_a_gateway_error_from_the_translation_passes_through_unchanged(
        self, wire
    ) -> None:
        """A refusal part has no upstream representation: a refusal, not a mapped failure."""
        provider, _ = wire(response_from(completion_payload()))
        request = make_request(
            messages=(
                Message(
                    role=Role.ASSISTANT,
                    content=(RefusalPart(text="I cannot help with that"),),
                ),
            )
        )

        with pytest.raises(CapabilityError) as excinfo:
            await provider.complete(make_route(), request, make_context())

        assert excinfo.value.details["capability"] == "refusal"

    async def test_cancellation_propagates_unmapped(self, wire) -> None:
        """PRD section 15: cancellation is propagated, not reported as a gateway failure."""
        provider, _ = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.complete(make_route(), make_request(), make_context())

    async def test_the_client_is_closed_when_the_call_is_cancelled(
        self, wire, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, completion = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.complete(make_route(), make_request(), make_context())

        (call,) = completion.calls
        assert closed == [pinned_transport_of(call["client"])]


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class TestTheProtocol:
    """The provider is usable wherever a :class:`TextProvider` is expected."""

    async def test_the_provider_satisfies_the_text_provider_protocol(self) -> None:
        provider = LiteLLMProvider()

        assert inspect.iscoroutinefunction(provider.complete)
        assert list(inspect.signature(provider.complete).parameters) == [
            "route",
            "request",
            "context",
        ]

    async def test_the_provider_works_through_the_protocol_type(self, wire) -> None:
        provider: TextProvider = wire(response_from(completion_payload()))[0]

        response = await provider.complete(make_route(), make_request(), make_context())

        assert response.model == "text-default"


# ---------------------------------------------------------------------------
# The seam is beneath the security boundary
# ---------------------------------------------------------------------------


class TestTheSeamIsBeneathTheBoundary:
    """What a caller may inject is the transport's inputs, never the client (PRD 14.1).

    The constructor accepts the pinned builder's inputs and nothing else, so no
    public input can replace the per-dialect client construction: it is
    impossible through the public API to obtain an unsafe client for a verified
    dialect, and the attempt does not need to raise to prevent one.
    """

    def test_the_constructor_accepts_no_client_or_client_factory(self) -> None:
        parameters = inspect.signature(LiteLLMProvider.__init__).parameters
        assert set(parameters) == {"self", "resolver", "verify", "inner_factory"}
        for name, parameter in parameters.items():
            if name == "self":
                continue
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
    async def test_custom_transport_inputs_still_yield_a_pinned_non_redirecting_client(
        self, dialect: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every injectable input is custom; the injected client is still the safe one.

        ``verify`` does not reach a supplied inner transport, as the transport
        documents; what the custom inputs must not change is the pinning and
        the refused redirect following, whichever of them is supplied.
        """
        completion = RecordingCompletion(response_from(completion_payload()))
        monkeypatch.setattr(litellm, "acompletion", completion)
        built: list[httpx.AsyncBaseTransport] = []

        def inner_factory() -> httpx.AsyncBaseTransport:
            transport = httpx.AsyncHTTPTransport()
            built.append(transport)
            return transport

        def resolver(host: str) -> tuple[str, ...]:
            return (PUBLIC_ADDRESS,)

        provider = LiteLLMProvider(
            resolver=resolver,
            verify=ssl.create_default_context(),
            inner_factory=inner_factory,
        )

        await provider.complete(make_route(format=dialect), make_request(), make_context())

        (call,) = completion.calls
        client = call["client"]
        if dialect == "openai":
            assert isinstance(client, openai.AsyncOpenAI)
            assert client.max_retries == 0
            http_client = client._client  # pyright: ignore[reportPrivateUsage]
        else:
            assert isinstance(client, AsyncHTTPHandler)
            assert client.follow_redirects is False
            http_client = client.client
            assert http_client._transport is client.transport  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        transport = http_client._transport  # pyright: ignore[reportPrivateUsage]
        assert isinstance(transport, PinnedAddressTransport)
        assert len(built) == 1
        assert transport._inner is built[0]  # pyright: ignore[reportPrivateUsage]
        assert transport._resolver is resolver  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# The .env guard
# ---------------------------------------------------------------------------


class TestTheEnvGuard:
    """Importing the binding must not load a working-directory ``.env`` (PRD section 6).

    The guard lives in :mod:`model_gateway.providers._litellm`, the seam every
    binding module imports the library through, and it *forces* the mode. These
    tests import a binding module in a subprocess — the provider on its own, and
    the translation module on its own, which crosses the seam with no other
    binding module imported first — with a ``.env`` in the working directory and
    no ``LITELLM_MODE`` in the environment, plus one run with an ambient
    ``LITELLM_MODE=DEV`` to prove the forced mode overrides even an explicit
    value: the mode is gateway-owned implementation state, not operator
    configuration.
    """

    PROBE = "GATEWAY_GUARD_PROBE_VARIABLE"

    def _run_import(
        self, code: str, tmp_path: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_importing_the_binding_leaves_the_working_directory_env_unloaded(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        # The probe must start unset, or the test would pass for the wrong reason.
        env = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
        code = (
            "import model_gateway.providers.litellm_provider\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr

    def test_importing_the_translation_module_directly_leaves_the_env_unloaded(
        self, tmp_path: Path
    ) -> None:
        """The translation module crosses the seam too: its direct import is guarded.

        Importing :mod:`model_gateway.providers.litellm_translation` on its own
        — without the provider module, and without this session's conftest guard —
        must still force the mode, because the seam is the one place the guard
        lives and every binding module imports the library through it rather than
        importing ``litellm`` directly.
        """
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        # The probe must start unset, or the test would pass for the wrong reason.
        env = {key: value for key, value in os.environ.items() if key != "LITELLM_MODE"}
        code = (
            "import model_gateway.providers.litellm_translation\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr

    def test_the_forced_mode_overrides_an_ambient_dev_value(self, tmp_path: Path) -> None:
        """An ambient ``LITELLM_MODE=DEV`` does not survive the seam's forced mode."""
        (tmp_path / ".env").write_text(f"{self.PROBE}=from-env-file\n", encoding="utf-8")
        env = {**os.environ, "LITELLM_MODE": "DEV"}
        code = (
            "import model_gateway.providers.litellm_provider\n"
            "import os\n"
            f"raise SystemExit(0 if os.environ.get({self.PROBE!r}) is None else 1)\n"
        )

        result = self._run_import(code, tmp_path, env)

        assert result.returncode == 0, result.stderr
