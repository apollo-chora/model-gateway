"""The embeddings path of the LiteLLM provider: one attempt through the binding
(PRD 4.7, 6.3, 9.6, 11.3, 14.1).

No test reaches the network: ``litellm.aembedding`` is monkeypatched with a
recording stand-in, so the real per-dialect client is built and closed around a
transport that is never asked for a request, and the response fixtures are the
real LiteLLM types — built through LiteLLM's own converter from a realistic
upstream payload — so what runs is exactly what the installed binding does with
a real provider's answer.

What these tests pin is that the embeddings operation is not a second, less
careful way to reach an upstream: the call carries the translation of the route
and the request, ``num_retries=0``, the attempt's own budget and the *same*
per-dialect client the text path injects, and every failure is mapped onto the
gateway's taxonomy before it can reach a caller.
"""

import asyncio
import inspect
import ssl

import httpx
import litellm
import openai
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import Embedding as LiteLLMEmbedding
from litellm.types.utils import EmbeddingResponse as LiteLLMEmbeddingResponse

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from model_gateway.domain.generation import Usage, UsageSource
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    FailedPreconditionError,
    InternalError,
    MalformedResponseError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
)
from model_gateway.providers.base import AttemptContext, EmbeddingProvider
from model_gateway.providers.litellm_provider import LiteLLMProvider
from model_gateway.security.transport import PinnedAddressTransport

ENDPOINT = "https://upstream.internal.example/v1"
SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_MODEL = "text-embedding-3-small"
PUBLIC_ADDRESS = "93.184.216.34"


def make_route(**overrides: object) -> Route:
    """A resolved embeddings route; override any field to model a configuration."""
    fields: dict[str, object] = {
        "operation": Operation.EMBEDDINGS,
        "index": 0,
        "endpoint": ENDPOINT,
        "base_url": ENDPOINT,
        "model_id": UPSTREAM_MODEL,
        "format": "openai",
        "api_key": SECRET_KEY,
        "capabilities": frozenset({Capability.EMBEDDINGS}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    fields.update(overrides)
    return Route(**fields)


def make_request(**overrides: object) -> EmbeddingRequest:
    """A canonical embeddings request; override any field to model a caller's ask."""
    fields: dict[str, object] = {
        "model": "embedding-default",
        "input": ("First document", "Second document"),
    }
    fields.update(overrides)
    return EmbeddingRequest(**fields)


def make_context(**overrides: object) -> AttemptContext:
    """A standalone attempt context; the budget is the provider's only deadline input."""
    fields: dict[str, object] = {
        "invocation_id": "inv-7f3c1a90",
        "attempt_number": 1,
        "timeout": 30.0,
    }
    fields.update(overrides)
    return AttemptContext(**fields)


def embedding_payload(**overrides: object) -> dict[str, object]:
    """A realistic OpenAI-shaped upstream embeddings payload."""
    payload: dict[str, object] = {
        "object": "list",
        "model": UPSTREAM_MODEL,
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.12, -0.34, 0.56]},
            {"object": "embedding", "index": 1, "embedding": [0.7, 0.8, 0.9]},
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    payload.update(overrides)
    return payload


def response_from(payload: dict[str, object]) -> LiteLLMEmbeddingResponse:
    """Build a real ``EmbeddingResponse`` through LiteLLM's own converter.

    This is the shape a real provider's answer has once the library has decoded
    it, down to the fact that the converter leaves ``data`` as the raw records
    the provider sent rather than wrapping each one in an ``Embedding``.
    """
    return convert_to_model_response_object(response_object=payload, response_type="embedding")


def http_error(status: int) -> httpx.Response:
    """An HTTP error response with a request attached, as LiteLLM exceptions carry."""
    return httpx.Response(
        status_code=status,
        request=httpx.Request(method="POST", url=f"{ENDPOINT}/embeddings"),
    )


def pinned_transport_of(client: object) -> PinnedAddressTransport:
    """The pinned transport ``client`` connects through, whichever shape it is."""
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


class RecordingEmbedding:
    """A stand-in for ``litellm.aembedding`` that records each call and answers from a script."""

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
    """Wire a provider to a scripted embedding call; the real client is built, never connected."""

    def wire_outcome(outcome: object) -> tuple[LiteLLMProvider, RecordingEmbedding]:
        embedding = RecordingEmbedding(outcome)
        monkeypatch.setattr(litellm, "aembedding", embedding)
        return LiteLLMProvider(), embedding

    return wire_outcome


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestTheCall:
    """What one embeddings attempt hands to the pinned binding, and what comes back."""

    async def test_the_parameters_are_the_route_the_texts_and_the_attempts_controls(
        self, wire
    ) -> None:
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context())

        (call,) = embedding.calls
        assert call["model"] == "openai/text-embedding-3-small"
        assert call["api_base"] == ENDPOINT
        assert call["api_key"] == SECRET_KEY
        assert call["extra_headers"] == {}
        assert call["input"] == ["First document", "Second document"]
        assert call["num_retries"] == 0
        assert call["timeout"] == 30.0

    async def test_the_injected_client_is_the_sdk_client_over_the_pinned_transport(
        self, wire
    ) -> None:
        """The same per-dialect construction the text path uses, not a second one."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context())

        (call,) = embedding.calls
        client = call["client"]
        assert isinstance(client, openai.AsyncOpenAI)
        assert client.api_key == SECRET_KEY
        assert str(client.base_url).rstrip("/") == ENDPOINT
        assert client.max_retries == 0
        http_client = client._client  # pyright: ignore[reportPrivateUsage]
        assert http_client.follow_redirects is False
        assert http_client.trust_env is False
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)

    @pytest.mark.parametrize("dialect", ["anthropic", "openrouter"])
    async def test_the_generic_dialects_are_given_the_pinned_http_handler(
        self, wire, dialect: str
    ) -> None:
        """The other verified dialects reach the embeddings call through their own client."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(format=dialect), make_request(), make_context())

        (call,) = embedding.calls
        client = call["client"]
        assert isinstance(client, AsyncHTTPHandler)
        assert client.follow_redirects is False
        assert client.client._transport is client.transport  # pyright: ignore[reportPrivateUsage]
        assert isinstance(pinned_transport_of(client), PinnedAddressTransport)

    async def test_the_client_is_closed_when_the_attempt_ends(self, wire, monkeypatch) -> None:
        closed = recording_aclose(monkeypatch)
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context())

        (call,) = embedding.calls
        assert closed == [pinned_transport_of(call["client"])]

    async def test_each_attempt_gets_a_fresh_client_with_its_own_budget(
        self, wire, monkeypatch
    ) -> None:
        closed = recording_aclose(monkeypatch)
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context(timeout=7.0))
        await provider.embed(make_route(), make_request(), make_context(timeout=7.0))

        first, second = (call["client"] for call in embedding.calls)
        assert first is not second
        assert closed == [pinned_transport_of(first), pinned_transport_of(second)]
        assert first.timeout == httpx.Timeout(7.0)
        assert second.timeout == httpx.Timeout(7.0)

    async def test_the_attempt_budget_is_the_calls_timeout(self, wire) -> None:
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context(timeout=12.5))

        assert embedding.calls[0]["timeout"] == 12.5

    async def test_no_budget_is_passed_through_as_none(self, wire) -> None:
        """The gateway set no budget; the binding invents none of its own."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context(timeout=None))

        assert embedding.calls[0]["timeout"] is None

    async def test_the_call_is_made_exactly_once_per_attempt(self, wire) -> None:
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context())

        assert len(embedding.calls) == 1

    async def test_a_stated_dimension_is_forwarded_when_the_route_enables_it(self, wire) -> None:
        """The route's own flag decides whether the caller's ask reaches the upstream."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(
            make_route(supports_dimensions_param=True), make_request(dimensions=256),
            make_context(),
        )

        assert embedding.calls[0]["dimensions"] == 256

    async def test_a_stated_dimension_is_not_forwarded_when_the_route_does_not_enable_it(
        self, wire
    ) -> None:
        """The operator configured the route as not accepting the parameter."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(dimensions=256), make_context())

        assert "dimensions" not in embedding.calls[0]

    async def test_an_unstated_dimension_is_omitted_rather_than_defaulted(self, wire) -> None:
        """The provider's own default applies; the gateway substitutes no vendor default."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(), make_context())

        assert "dimensions" not in embedding.calls[0]

    async def test_the_wire_encoding_is_never_sent(self, wire) -> None:
        """The answer is the float form, which is also the upstream's own default."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(make_route(), make_request(encoding_format="float"), make_context())

        assert "encoding_format" not in embedding.calls[0]

    async def test_the_callers_user_and_metadata_are_not_forwarded(self, wire) -> None:
        """PRD section 5.1: caller bookkeeping never reaches the model or the provider."""
        provider, embedding = wire(response_from(embedding_payload()))

        await provider.embed(
            make_route(),
            make_request(user="caller-7", metadata={"trace": {"sampled": True}}),
            make_context(),
        )

        assert "user" not in embedding.calls[0]
        assert "metadata" not in embedding.calls[0]

    async def test_the_answer_is_normalised_into_the_canonical_response(self, wire) -> None:
        provider, _ = wire(response_from(embedding_payload()))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert isinstance(response, EmbeddingResponse)
        assert response.model == "embedding-default"
        assert response.upstream_model == UPSTREAM_MODEL
        assert response.invocation_id == "inv-7f3c1a90"
        assert [embedding.values for embedding in response.data] == [
            (0.12, -0.34, 0.56),
            (0.7, 0.8, 0.9),
        ]
        assert [embedding.index for embedding in response.data] == [0, 1]
        assert response.usage == Usage(input_tokens=10, output_tokens=None, total_tokens=10)

    async def test_the_upstreams_own_order_and_positions_are_reported_verbatim(self, wire) -> None:
        """Ordering the vectors against the caller's input is the service's work, not this one's."""
        payload = embedding_payload(
            data=[
                {"object": "embedding", "index": 1, "embedding": [0.7, 0.8, 0.9]},
                {"object": "embedding", "index": 0, "embedding": [0.12, -0.34, 0.56]},
            ]
        )
        provider, _ = wire(response_from(payload))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert [embedding.index for embedding in response.data] == [1, 0]

    async def test_a_vector_the_library_wrapped_is_read_the_same_way(self, wire) -> None:
        """The library hands the records back as objects on one path and dicts on another."""
        payload = embedding_payload(
            data=[LiteLLMEmbedding(object="embedding", index=0, embedding=[0.1, 0.2, 0.3])]
        )
        provider, _ = wire(response_from(payload))

        response = await provider.embed(make_route(), make_request(input=("one",)), make_context())

        assert [embedding.values for embedding in response.data] == [(0.1, 0.2, 0.3)]

    async def test_a_base64_encoded_vector_is_refused(self, wire) -> None:
        """A semantic the canonical representation cannot carry is refused, never coerced."""
        payload = embedding_payload(
            data=[{"object": "embedding", "index": 0, "embedding": "AAAAAAAAAAA="}]
        )
        provider, _ = wire(response_from(payload))

        with pytest.raises(MalformedResponseError) as error:
            await provider.embed(make_route(), make_request(input=("one",)), make_context())

        assert error.value.reason is ErrorReason.MALFORMED_RESPONSE

    async def test_a_vector_of_non_numbers_is_refused(self, wire) -> None:
        payload = embedding_payload(
            data=[{"object": "embedding", "index": 0, "embedding": ["a", "b"]}]
        )
        provider, _ = wire(response_from(payload))

        with pytest.raises(MalformedResponseError):
            await provider.embed(make_route(), make_request(input=("one",)), make_context())

    async def test_a_position_that_is_not_an_integer_is_refused(self, wire) -> None:
        payload = embedding_payload(
            data=[{"object": "embedding", "index": "0", "embedding": [0.1, 0.2]}]
        )
        provider, _ = wire(response_from(payload))

        with pytest.raises(MalformedResponseError):
            await provider.embed(make_route(), make_request(input=("one",)), make_context())

    async def test_an_absent_usage_is_reported_as_unknown(self, wire) -> None:
        """The library fabricates an all-zero usage, and a zero cannot be told from a reported zero.

        ``EmbeddingResponse`` always holds a ``Usage`` object — the class supplies
        its own when the upstream reported nothing — and the converter fills the
        counts onto that fabricated object, so an absent usage and a reported
        zero are the same object at this boundary and the gateway cannot tell
        them apart. Within that limit the binding reports what it can hold to
        the upstream: every count of the placeholder is recorded as unknown,
        because presenting the placeholder as a reported fact is the one error
        this boundary must not make (PRD section 13).
        """
        payload = embedding_payload()
        payload.pop("usage")
        provider, _ = wire(response_from(payload))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert response.usage.input_tokens is None
        assert response.usage.total_tokens is None

    async def test_a_reported_non_zero_count_is_recorded_as_reported(self, wire) -> None:
        """A non-zero count cannot be the library's placeholder, so it is a fact it can only be.

        The placeholder is all-zero by construction, so a count above zero can
        only have come from the upstream and is recorded with provider
        provenance. The zero counts beside it — ambiguous between a reported
        zero, a converter default and the placeholder — stay unknown.
        """
        provider, _ = wire(response_from(embedding_payload()))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert response.usage.input_tokens == 10
        assert response.usage.total_tokens == 10
        assert response.usage.output_tokens is None
        assert response.usage.source_of("input_tokens") is UsageSource.PROVIDER
        assert response.usage.source_of("total_tokens") is UsageSource.PROVIDER

    async def test_a_total_of_zero_beside_reported_parts_is_derived_and_named_an_estimate(
        self, wire
    ) -> None:
        """The embeddings path treats an unestablished total as the completion path does.

        A total of zero beside non-zero parts is the library's placeholder
        rather than a reported fact, so the gateway derives it from the parts
        and names it in ``estimated_fields`` — the same rule :func:`_usage`
        applies on the completion path, so the two cannot drift apart.
        """
        payload = embedding_payload(usage={"prompt_tokens": 10})
        provider, _ = wire(response_from(payload))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens is None
        assert response.usage.total_tokens == 10
        assert response.usage.estimated_fields == frozenset({"total_tokens"})
        assert response.usage.source_of("total_tokens") is UsageSource.ESTIMATE

    async def test_an_answer_carrying_no_records_is_reported_as_no_vectors(self, wire) -> None:
        """A ``null`` the library passed through becomes no vectors, not a ``TypeError``.

        The gateway core refuses an answer with no vector per input, which is a
        gateway error a caller can act on; a ``TypeError`` escaping the binding
        would reach the same caller as an unclassified internal failure instead
        (PRD section 5.4).
        """
        payload = embedding_payload(data=None)
        provider, _ = wire(response_from(payload))

        response = await provider.embed(make_route(), make_request(), make_context())

        assert response.data == ()


# ---------------------------------------------------------------------------
# Failure mapping
# ---------------------------------------------------------------------------


class TestFailures:
    """Every failure the embeddings call raises is mapped onto the taxonomy (PRD 5.4, 9)."""

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
            await provider.embed(make_route(), make_request(), make_context())

        assert excinfo.value.reason.value == reason
        assert excinfo.value.__cause__ is error

    async def test_the_upstream_message_never_reaches_the_raised_error(self, wire) -> None:
        url = "https://api.openai.com/v1/embeddings"
        error = litellm.AuthenticationError(
            message=f"invalid key {SECRET_KEY} for {url}",
            llm_provider="openai",
            model=UPSTREAM_MODEL,
            response=http_error(401),
        )
        provider, _ = wire(error)

        with pytest.raises(UnauthenticatedError) as excinfo:
            await provider.embed(make_route(), make_request(), make_context())

        payload = excinfo.value.to_payload(include_details=True)
        assert SECRET_KEY not in str(payload)
        assert url not in str(payload)

    async def test_a_gateway_error_from_the_translation_passes_through_unchanged(
        self, wire
    ) -> None:
        refusal = CapabilityError("the upstream answer cannot be carried canonically")
        provider, _ = wire(refusal)

        with pytest.raises(CapabilityError) as excinfo:
            await provider.embed(make_route(), make_request(), make_context())

        assert excinfo.value is refusal

    async def test_an_unverified_dialect_is_refused_before_any_client_is_built(self, wire) -> None:
        """A dialect outside the verified set cannot reach the embeddings call either."""
        provider, embedding = wire(response_from(embedding_payload()))

        with pytest.raises(FailedPreconditionError) as error:
            await provider.embed(make_route(format="vertex_ai"), make_request(), make_context())

        assert error.value.reason is ErrorReason.FAILED_PRECONDITION
        assert embedding.calls == []

    async def test_an_attempt_without_an_invocation_identifier_is_refused(self, wire) -> None:
        """The canonical embeddings response must state the invocation it belongs to."""
        provider, embedding = wire(response_from(embedding_payload()))

        with pytest.raises(FailedPreconditionError):
            await provider.embed(make_route(), make_request(), make_context(invocation_id=None))

        assert len(embedding.calls) == 1

    async def test_cancellation_propagates_unmapped(self, wire) -> None:
        provider, _ = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.embed(make_route(), make_request(), make_context())

    async def test_the_client_is_closed_when_the_call_is_cancelled(self, wire, monkeypatch) -> None:
        closed = recording_aclose(monkeypatch)
        provider, embedding = wire(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await provider.embed(make_route(), make_request(), make_context())

        (call,) = embedding.calls
        assert closed == [pinned_transport_of(call["client"])]


# ---------------------------------------------------------------------------
# Usage provenance
# ---------------------------------------------------------------------------


class TestUsageProvenance:
    """Whether an absent usage can be told from a reported zero at this boundary.

    This is a contract probe rather than a test of the binding: it drives the
    pinned library's own ``litellm.aembedding`` entry point with scripted wires
    and pins what the binding receives. It exists because the mapping in
    :func:`~model_gateway.providers.litellm_translation.from_litellm_embedding_usage`
    rests on a measured property of the installed library — that the
    distinction is destroyed — and a docstring describing that property can
    be right and still go stale; this test cannot.
    """

    @pytest.mark.parametrize("dialect", ["openai", "openrouter"])
    async def test_an_absent_usage_and_a_reported_zero_are_the_same_object(
        self, dialect: str
    ) -> None:
        """Every representation the binding can read agrees between the two.

        Both payloads — one with no ``usage`` key, one with an explicit
        all-zero usage — are driven through the library's own entry point,
        and the two answers are compared in the usage's pydantic field set,
        its ``__dict__``, its ``model_dump()`` and its extras, and in the
        response's own field set and the keys of its ``hidden_params``. They
        agree, so the binding cannot tell an unreported usage from a reported
        zero and must not present the placeholder's zeros as provider-reported
        fact.
        """
        absent = await _library_embedding(dialect, _payload_without_usage())
        reported_zero = await _library_embedding(dialect, _payload_with_zero_usage())

        assert absent.usage.model_fields_set == reported_zero.usage.model_fields_set
        assert absent.usage.__dict__ == reported_zero.usage.__dict__
        assert absent.usage.model_dump() == reported_zero.usage.model_dump()
        assert absent.usage.model_extra == reported_zero.usage.model_extra
        assert absent.model_fields_set == reported_zero.model_fields_set
        assert absent.model_extra == reported_zero.model_extra
        assert set(absent.hidden_params) == set(reported_zero.hidden_params)

    @pytest.mark.parametrize("dialect", ["openai", "openrouter"])
    async def test_the_placeholder_the_library_fabricates_is_all_zero(self, dialect: str) -> None:
        """The fabrication itself is pinned: an answer carrying no usage yields zeros.

        Not the visible field alone — the whole object is the placeholder, and
        it is the same object a reported zero produces, which is why the
        binding records a zero count as unknown rather than as reported.
        """
        response = await _library_embedding(dialect, _payload_without_usage())

        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0

    async def test_the_anthropic_dialect_is_refused_by_the_library_itself(self) -> None:
        """The library's embedding dispatch has no anthropic branch, so there is no usage to read.

        The binding's SUPPORTED_DIALECTS includes anthropic for the text path;
        on the embeddings path the pinned library raises before any client is
        used. Pinned here because the dialect's behaviour is a property of the
        library, not of the gateway, and an upgrade that added the branch
        would change what a dialect's route can be.
        """
        with pytest.raises(litellm.BadRequestError):
            await _library_embedding("anthropic", _payload_without_usage())


class ScriptedWire(httpx.AsyncBaseTransport):
    """A wire that answers every request with one scripted upstream payload."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=self.payload,
            headers={"content-type": "application/json"},
            request=request,
        )


def _payload_without_usage() -> dict[str, object]:
    """A realistic upstream embeddings payload carrying no usage at all."""
    return {
        "object": "list",
        "model": UPSTREAM_MODEL,
        "data": [{"object": "embedding", "index": 0, "embedding": [0.12, -0.34, 0.56]}],
    }


def _payload_with_zero_usage() -> dict[str, object]:
    """The same payload with an explicit all-zero usage object."""
    payload = _payload_without_usage()
    payload["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return payload


async def _library_embedding(
    dialect: str, payload: dict[str, object]
) -> LiteLLMEmbeddingResponse:
    """Drive the pinned library's own ``aembedding`` entry point with a scripted wire.

    The client is the per-dialect construction the binding injects — an
    ``openai.AsyncOpenAI`` on the SDK path, an ``AsyncHTTPHandler`` on the
    generic one — around a wire that is never a socket, so what runs is
    exactly the library's response construction for a real provider's answer.
    """
    wire = ScriptedWire(payload)
    if dialect == "openai":
        client: openai.AsyncOpenAI | AsyncHTTPHandler = openai.AsyncOpenAI(
            api_key=SECRET_KEY,
            base_url=ENDPOINT,
            http_client=httpx.AsyncClient(transport=wire),
            max_retries=0,
        )
    else:
        client = AsyncHTTPHandler(timeout=30.0, transport=wire, follow_redirects=False)
    try:
        return await litellm.aembedding(
            model=f"{dialect}/{UPSTREAM_MODEL}",
            input=["hello"],
            api_base=ENDPOINT,
            api_key=SECRET_KEY,
            num_retries=0,
            client=client,
        )
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class TestTheProtocol:
    """The provider is usable wherever an :class:`EmbeddingProvider` is expected."""

    async def test_the_provider_satisfies_the_embedding_protocol(self) -> None:
        provider = LiteLLMProvider()

        assert isinstance(provider, EmbeddingProvider)
        assert inspect.iscoroutinefunction(provider.embed)
        assert list(inspect.signature(provider.embed).parameters) == [
            "route",
            "request",
            "context",
        ]

    async def test_the_embedding_capability_does_not_replace_the_text_one(self) -> None:
        """One provider serves both operations, and neither protocol is faked for it.

        ``TextProvider`` is deliberately not runtime-checkable, so this pins the
        structural fact instead: the same object answers a completion and an
        embedding call, and the embeddings protocol was added beside the text one
        rather than replacing it.
        """
        provider = LiteLLMProvider()

        assert inspect.iscoroutinefunction(provider.complete)
        assert isinstance(provider, EmbeddingProvider)

    async def test_the_provider_works_through_the_protocol_type(self, wire) -> None:
        provider: EmbeddingProvider = wire(response_from(embedding_payload()))[0]

        response = await provider.embed(make_route(), make_request(), make_context())

        assert response.model == "embedding-default"

    async def test_custom_transport_inputs_still_yield_a_pinned_non_redirecting_client(
        self, wire
    ) -> None:
        """What a caller may inject is the transport's inputs, never the client itself."""
        provider, embedding = wire(response_from(embedding_payload()))
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

        await provider.embed(make_route(), make_request(), make_context())

        (call,) = embedding.calls
        client = call["client"]
        assert isinstance(client, openai.AsyncOpenAI)
        assert client.max_retries == 0
        transport = pinned_transport_of(client)
        assert len(built) == 1
        assert transport._inner is built[0]  # pyright: ignore[reportPrivateUsage]
        assert transport._resolver is resolver  # pyright: ignore[reportPrivateUsage]
