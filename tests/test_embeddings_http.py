"""Tests for the embeddings endpoint: ``POST /v1/embeddings``.

The handler is exercised through a real BlackSheep application, so what is tested
is the route, the wire translation and the error mapping together. Nothing
reaches the network: the provider stands in at the binding boundary and every
route names an IP literal.

The endpoint's contract has three parts, and each is pinned here: the accepted
request is the documented subset and nothing outside it is ignored, the response
is the documented OpenAI shape, and every refusal — an unknown model, a
parameter the gateway does not understand, a capability the route does not have —
answers with the taxonomy's own status and reason rather than a framework error.
"""

import pytest
from blacksheep import Application, Content, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.domain.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from model_gateway.domain.generation import Usage
from model_gateway.errors import UnavailableError
from model_gateway.http.embeddings import register_embeddings_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
PUBLIC_MODEL = "embedding-default"
VECTOR = (0.12, -0.34, 0.56)


def make_registry() -> ModelRegistry:
    return ModelRegistry(
        (
            Route(
                operation=Operation.EMBEDDINGS,
                index=0,
                endpoint=UPSTREAM_ENDPOINT,
                base_url=UPSTREAM_ENDPOINT,
                model_id="upstream-embedding",
                format="openai",
                api_key=SECRET_KEY,
                capabilities=frozenset({Capability.EMBEDDINGS}),
                extra_headers={},
                allow_absolute_path_url=False,
                supports_dimensions_param=True,
            ),
        )
    )


def embedding_response(**overrides: object) -> EmbeddingResponse:
    """A canonical embeddings response, the shape a provider returns."""
    fields: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "upstream_model": "upstream-embedding",
        "data": (
            Embedding(index=0, values=VECTOR),
            Embedding(index=1, values=(0.7, 0.8, 0.9)),
        ),
        "usage": Usage(input_tokens=10, total_tokens=10),
        "invocation_id": "inv-7f3c1a90",
    }
    fields.update(overrides)
    return EmbeddingResponse(**fields)


class FakeEmbeddingProvider:
    """An ``EmbeddingProvider`` double: records the calls and returns a canned response.

    The HTTP surface is what these tests are about, so the provider stands in for
    the upstream at the binding boundary and no socket is opened. With no canned
    result it answers one vector per input text, so a test that varies the input
    still gets a well-shaped answer.
    """

    def __init__(
        self,
        result: EmbeddingResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[Route, EmbeddingRequest]] = []
        self.result = result
        self.error = error

    async def embed(
        self, route: Route, request: EmbeddingRequest, context: object
    ) -> EmbeddingResponse:
        self.calls.append((route, request))
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return embedding_response(
            data=tuple(
                Embedding(index=index, values=VECTOR)
                for index in range(len(request.input))
            )
        )


class FakeTextProvider:
    """A provider that answers text and nothing else: no ``embed`` method at all."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, route: Route, request: object, context: object) -> object:
        self.calls += 1
        raise AssertionError("a text-only provider must never be handed an embeddings request")


def build_app(provider: object) -> Application:
    registry = make_registry()
    app = Application(router=Router())
    register_error_handlers(app)
    register_embeddings_routes(app.router, InferenceService(registry, provider))
    return app


@pytest.fixture(autouse=True)
def no_declared_dimensions(monkeypatch: pytest.MonkeyPatch):
    """These tests declare no embedding dimensionality, so none is checked.

    The registry reads the advertised limits from the process environment, and a
    developer who happens to have exported ``EMBEDDINGS_DIMENSIONS`` would
    otherwise change what these tests assert about the endpoint.
    """
    monkeypatch.delenv("EMBEDDINGS_DIMENSIONS", raising=False)


@pytest.fixture
async def provider():
    return FakeEmbeddingProvider()


@pytest.fixture
async def client(provider):
    app = build_app(provider)
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "input": ["First document", "Second document"],
    }
    body.update(overrides)
    return body


async def post(client, body: object):
    return await client.post("/v1/embeddings", content=JSONContent(body))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    """The documented request and the documented response."""

    async def test_a_list_of_texts_is_embedded(self, client, provider) -> None:
        response = await post(client, payload())

        assert response.status == 200
        assert len(provider.calls) == 1

    async def test_the_response_is_the_documented_shape(self, client) -> None:
        response = await post(client, payload())
        body = await response.json()

        assert set(body) == {"object", "model", "data", "usage"}
        assert body["object"] == "list"
        assert body["model"] == PUBLIC_MODEL
        assert body["usage"] == {"prompt_tokens": 10, "total_tokens": 10}
        assert [set(entry) for entry in body["data"]] == [
            {"object", "index", "embedding"},
            {"object", "index", "embedding"},
        ]
        assert [entry["object"] for entry in body["data"]] == ["embedding", "embedding"]
        assert [entry["index"] for entry in body["data"]] == [0, 1]
        assert body["data"][0]["embedding"] == list(VECTOR)

    async def test_a_single_string_is_one_text_to_embed(self, client) -> None:
        response = await post(client, payload(input="First document"))
        body = await response.json()

        assert response.status == 200
        assert [entry["index"] for entry in body["data"]] == [0]

    async def test_the_texts_reach_the_provider_in_the_callers_order(
        self, client, provider
    ) -> None:
        await post(client, payload(input=["Second document", "First document"]))

        (_, request) = provider.calls[0]
        assert request.input == ("Second document", "First document")
        assert request.model == PUBLIC_MODEL

    async def test_the_vectors_come_back_in_the_callers_order(self, client) -> None:
        """The endpoint renders what the service ordered, not what the upstream sent."""
        provider = FakeEmbeddingProvider(
            embedding_response(
                data=(
                    Embedding(index=1, values=(0.7, 0.8, 0.9)),
                    Embedding(index=0, values=VECTOR),
                )
            )
        )
        app = build_app(provider)
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        body = await response.json()
        assert [entry["index"] for entry in body["data"]] == [0, 1]
        assert body["data"][0]["embedding"] == list(VECTOR)

    async def test_the_stated_dimension_and_encoding_reach_the_request(
        self, client, provider
    ) -> None:
        await post(client, payload(dimensions=256, encoding_format="float"))

        (_, request) = provider.calls[0]
        assert request.dimensions == 256
        assert request.encoding_format == "float"

    async def test_the_callers_user_and_metadata_are_accepted_and_not_forwarded(
        self, client, provider
    ) -> None:
        response = await post(client, payload(user="caller-7", metadata={"trace": "t-1"}))

        assert response.status == 200
        (_, request) = provider.calls[0]
        assert request.user == "caller-7"
        assert request.metadata == {"trace": "t-1"}

    async def test_the_response_never_carries_upstream_configuration(self, client) -> None:
        response = await post(client, payload())
        body = await response.text()

        assert SECRET_KEY not in body
        assert UPSTREAM_ENDPOINT not in body
        assert "upstream-embedding" not in body

    async def test_the_invocation_identifier_is_not_published(self, client) -> None:
        """The documented shape has no field for it, so the endpoint invents none."""
        response = await post(client, payload())
        body = await response.json()

        assert "id" not in body
        assert "invocation_id" not in body

    async def test_an_unreported_usage_is_presented_as_zero(self, client) -> None:
        provider = FakeEmbeddingProvider(embedding_response(usage=Usage()))
        app = build_app(provider)
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        assert (await response.json())["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------------------
# The request boundary
# ---------------------------------------------------------------------------


class TestTheRequestBoundary:
    """A parameter the gateway does not understand is refused, never ignored."""

    async def test_an_unknown_parameter_is_refused(self, client) -> None:
        response = await post(client, payload(truncate="END"))

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"

    async def test_the_refusal_names_the_parameter(self, client) -> None:
        response = await post(client, payload(truncate="END"))

        assert "truncate" in (await response.json())["message"]

    async def test_a_missing_model_is_served(self, client) -> None:
        response = await post(client, {"input": ["First document"]})

        assert response.status == 200
        assert (await response.json())["model"] == PUBLIC_MODEL

    async def test_a_missing_input_is_refused(self, client) -> None:
        response = await post(client, {"model": PUBLIC_MODEL})

        assert response.status == 400

    async def test_an_empty_input_array_is_refused(self, client) -> None:
        response = await post(client, payload(input=[]))

        assert response.status == 400

    async def test_a_blank_input_entry_is_refused(self, client) -> None:
        response = await post(client, payload(input=["First document", "   "]))

        assert response.status == 400

    async def test_an_input_that_is_neither_a_string_nor_an_array_is_refused(self, client) -> None:
        response = await post(client, payload(input=42))

        assert response.status == 400

    async def test_an_undocumented_encoding_is_refused(self, client) -> None:
        response = await post(client, payload(encoding_format="hex"))

        assert response.status == 400

    async def test_a_non_positive_dimension_is_refused(self, client) -> None:
        response = await post(client, payload(dimensions=0))

        assert response.status == 400

    async def test_a_body_that_is_not_a_json_object_is_refused(self, client) -> None:
        response = await post(client, [1, 2, 3])

        assert response.status == 400

    async def test_a_body_that_is_not_json_is_refused(self, client) -> None:
        response = await client.post(
            "/v1/embeddings", content=Content(b"application/json", b"{not json")
        )

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"

    async def test_a_refused_request_reaches_no_provider(self, client, provider) -> None:
        await post(client, payload(truncate="END"))

        assert provider.calls == []


# ---------------------------------------------------------------------------
# Refusals that are about the route or the provider
# ---------------------------------------------------------------------------


class TestCapabilityRefusals:
    """A capability the route does not have is refused with the taxonomy's reason."""

    async def test_an_unknown_model_is_ignored_and_the_request_is_served(
        self, client, provider
    ) -> None:
        response = await post(client, payload(model="no-such-model"))

        assert response.status == 200
        assert (await response.json())["model"] == PUBLIC_MODEL
        (_, request) = provider.calls[0]
        assert request.model == "no-such-model"

    async def test_a_provider_that_cannot_embed_is_refused(self) -> None:
        app = build_app(FakeTextProvider())
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"

    async def test_the_capability_refusal_names_the_operation(self) -> None:
        app = build_app(FakeTextProvider())
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        assert "embeddings" in (await response.json())["message"]

    async def test_an_encoding_the_gateway_cannot_answer_in_is_refused(
        self, client, provider
    ) -> None:
        response = await post(client, payload(encoding_format="base64"))

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"
        assert provider.calls == []

    async def test_a_provider_failure_is_mapped_onto_the_taxonomy(self) -> None:
        failure = UnavailableError("the upstream is unavailable")
        app = build_app(FakeEmbeddingProvider(error=failure))
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        assert response.status == 503
        assert (await response.json())["reason"] == "provider_unavailable"

    async def test_an_unclassified_failure_is_sanitised(self) -> None:
        app = build_app(FakeEmbeddingProvider(error=RuntimeError(f"boom {SECRET_KEY}")))
        await app.start()
        try:
            response = await post(TestClient(app), payload())
        finally:
            await app.stop()

        body = await response.text()
        assert response.status == 500
        assert SECRET_KEY not in body
        assert "boom" not in body
