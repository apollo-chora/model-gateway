"""Tests for the image endpoints: ``POST /v1/images/generations`` and
``POST /v1/images/edits`` (PRD sections 4.6, 6.2, 9.5, 10 and 14.1).

The handlers are exercised through a real BlackSheep application, so what is
tested is the route, the wire translation and the error mapping together.
Nothing reaches the network: the provider stands in at the binding boundary
and every route is a standalone value.

The endpoints' contract has three parts, and each is pinned here: the accepted
request is the documented subset and nothing outside it is ignored, the
response is the documented OpenAI shape, and every refusal — an unknown model,
a parameter the gateway does not understand, a capability the route does not
have — answers with the taxonomy's own status and reason rather than a
framework error. The application factory's own registration of the
generations route is pinned too, since a handler that answers only on a
hand-built application is how an endpoint comes to 404 on a real deployment.
"""

import pytest
from blacksheep import Application, Content, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import ConfigurationError, Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain.generation import Usage
from model_gateway.domain.images import GeneratedImage, ImageRequest, ImageResponse
from model_gateway.errors import UnavailableError
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.images import register_image_routes
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
PUBLIC_MODEL = "image-default"
PAYLOAD = "aW1hZ2UtYnl0ZXM="


def make_registry(*, editing: bool = False) -> ModelRegistry:
    """A registry with one image route, declaring ``generation`` and maybe ``editing``."""
    capabilities = (
        frozenset({Capability.GENERATION, Capability.EDITING})
        if editing
        else frozenset({Capability.GENERATION})
    )
    return ModelRegistry(
        (
            Route(
                operation=Operation.IMAGE,
                index=0,
                endpoint=UPSTREAM_ENDPOINT,
                base_url=UPSTREAM_ENDPOINT,
                model_id="image-model",
                format="openai",
                api_key=SECRET_KEY,
                capabilities=capabilities,
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


def image_response(**overrides: object) -> ImageResponse:
    """A canonical image response, the shape a provider returns."""
    fields: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "upstream_model": "image-model",
        "created": 1700000000,
        "data": (GeneratedImage(b64_json=PAYLOAD), GeneratedImage(b64_json=PAYLOAD)),
        "usage": Usage(),
        "invocation_id": "inv-7f3c1a90",
    }
    fields.update(overrides)
    return ImageResponse(**fields)


class FakeImageProvider:
    """An ``ImageProvider`` double: records the calls and returns a canned response.

    The HTTP surface is what these tests are about, so the provider stands in
    for the upstream at the binding boundary and no socket is opened.
    """

    def __init__(
        self,
        result: ImageResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[Route, ImageRequest]] = []
        self.result = result
        self.error = error

    async def generate(
        self, route: Route, request: ImageRequest, context: object
    ) -> ImageResponse:
        self.calls.append((route, request))
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return image_response()


class FakeTextProvider:
    """A provider that answers text and nothing else: no ``generate`` method at all."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, route: Route, request: object, context: object) -> object:
        self.calls += 1
        raise AssertionError("a text-only provider must never be handed an image request")


def build_app(provider: object) -> Application:
    registry = make_registry()
    app = Application(router=Router())
    register_error_handlers(app)
    register_image_routes(app.router, InferenceService(registry, provider))
    return app


@pytest.fixture
async def provider():
    return FakeImageProvider()


@pytest.fixture
async def client(provider):
    app = build_app(provider)
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"model": PUBLIC_MODEL, "prompt": "A futuristic city at sunset"}
    body.update(overrides)
    return body


def edit_payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": PUBLIC_MODEL,
        "prompt": "Make it rainy",
        "image": PAYLOAD,
    }
    body.update(overrides)
    return body


async def post_generations(client, body: object):
    return await client.post("/v1/images/generations", content=JSONContent(body))


async def post_edits(client, body: object):
    return await client.post("/v1/images/edits", content=JSONContent(body))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    """The documented request and the documented response (PRD section 4.6)."""

    async def test_a_generation_request_is_served(self, client, provider) -> None:
        response = await post_generations(client, payload())

        assert response.status == 200
        assert len(provider.calls) == 1

    async def test_the_response_is_the_documented_shape(self, client) -> None:
        response = await post_generations(client, payload())
        body = await response.json()

        assert set(body) == {"created", "data"}
        assert body["created"] == 1700000000
        assert [set(entry) for entry in body["data"]] == [{"b64_json"}, {"b64_json"}]
        assert body["data"][0]["b64_json"] == PAYLOAD

    async def test_a_url_answer_is_returned_as_a_url(self, client) -> None:
        provider = FakeImageProvider(
            image_response(data=(GeneratedImage(url="https://cdn.example/i.png"),))
        )
        app = build_app(provider)
        await app.start()
        try:
            response = await post_generations(
                TestClient(app), payload(model=PUBLIC_MODEL)
            )
        finally:
            await app.stop()

        body = await response.json()
        assert [set(entry) for entry in body["data"]] == [{"url"}]
        assert body["data"][0]["url"] == "https://cdn.example/i.png"

    async def test_an_entry_never_carries_both_payloads(self, client) -> None:
        """The canonical type allows exactly one, so the wire carries exactly one."""
        response = await post_generations(client, payload())
        body = await response.json()

        for entry in body["data"]:
            assert ("b64_json" in entry) != ("url" in entry)

    async def test_a_revised_prompt_is_returned_beside_the_payload(self, client) -> None:
        provider = FakeImageProvider(
            image_response(data=(GeneratedImage(b64_json=PAYLOAD, revised_prompt="A city"),))
        )
        app = build_app(provider)
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        body = await response.json()
        assert body["data"][0]["revised_prompt"] == "A city"

    async def test_the_stated_parameters_reach_the_request(self, client, provider) -> None:
        await post_generations(
            client, payload(size="1024x1024", response_format="b64_json", n=2)
        )

        (_, request) = provider.calls[0]
        assert request.size == "1024x1024"
        assert request.response_format == "b64_json"
        assert request.n == 2

    async def test_the_prompt_reaches_the_request_verbatim(self, client, provider) -> None:
        await post_generations(client, payload(prompt="  A lighthouse at dusk  "))

        (_, request) = provider.calls[0]
        assert request.prompt == "  A lighthouse at dusk  "

    async def test_the_response_never_carries_upstream_configuration(self, client) -> None:
        response = await post_generations(client, payload())
        body = await response.text()

        assert SECRET_KEY not in body
        assert UPSTREAM_ENDPOINT not in body
        assert "image-model" not in body

    async def test_the_invocation_identifier_is_not_published(self, client) -> None:
        """The documented shape has no field for it, so the endpoint invents none."""
        response = await post_generations(client, payload())
        body = await response.json()

        assert "id" not in body
        assert "invocation_id" not in body
        assert "model" not in body

    async def test_the_usage_is_not_published(self, client) -> None:
        """The OpenAI image shape has no usage field, so none is invented."""
        response = await post_generations(client, payload())
        body = await response.json()

        assert "usage" not in body


# ---------------------------------------------------------------------------
# The request boundary
# ---------------------------------------------------------------------------


class TestTheRequestBoundary:
    """A parameter the gateway does not understand is refused, never ignored (PRD 4)."""

    async def test_an_unknown_parameter_is_refused(self, client) -> None:
        response = await post_generations(client, payload(truncate="END"))

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"

    async def test_the_refusal_names_the_parameter(self, client) -> None:
        response = await post_generations(client, payload(truncate="END"))

        assert "truncate" in (await response.json())["message"]

    async def test_a_missing_model_is_refused(self, client) -> None:
        response = await post_generations(client, {"prompt": "A futuristic city"})

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"

    async def test_a_missing_prompt_is_refused(self, client) -> None:
        response = await post_generations(client, {"model": PUBLIC_MODEL})

        assert response.status == 400

    async def test_an_empty_prompt_is_refused(self, client) -> None:
        response = await post_generations(client, payload(prompt=""))

        assert response.status == 400

    async def test_a_size_outside_the_documented_form_is_refused(self, client) -> None:
        for size in ("1024X1024", "1024*1024", "1024x1024x1024", "square", "1024"):
            response = await post_generations(client, payload(size=size))

            assert response.status == 400

    async def test_an_undocumented_response_format_is_refused(self, client) -> None:
        response = await post_generations(client, payload(response_format="jpeg"))

        assert response.status == 400

    async def test_a_non_positive_count_is_refused(self, client) -> None:
        for count in (0, -1):
            response = await post_generations(client, payload(n=count))

            assert response.status == 400

    async def test_a_count_that_is_not_an_integer_is_refused(self, client) -> None:
        response = await post_generations(client, payload(n="two"))

        assert response.status == 400

    async def test_metadata_that_is_not_a_json_object_is_refused(self, client) -> None:
        response = await post_generations(client, payload(metadata="tenant-1"))

        assert response.status == 400

    async def test_metadata_holds_no_upstream_configuration(self, client, provider) -> None:
        await post_generations(client, payload(metadata={"trace": "t-1"}))

        (_, request) = provider.calls[0]
        assert request.metadata == {"trace": "t-1"}

    async def test_a_body_that_is_not_a_json_object_is_refused(self, client) -> None:
        response = await post_generations(client, [1, 2, 3])

        assert response.status == 400

    async def test_a_body_that_is_not_json_is_refused(self, client) -> None:
        response = await client.post(
            "/v1/images/generations", content=Content(b"application/json", b"{not json")
        )

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"

    async def test_a_refused_request_reaches_no_provider(self, client, provider) -> None:
        await post_generations(client, payload(truncate="END"))

        assert provider.calls == []


# ---------------------------------------------------------------------------
# The edits endpoint's image input
# ---------------------------------------------------------------------------


class TestTheEditImageInput:
    """The documented representation is validated where it arrives (PRD 4.6)."""

    async def test_a_base64_payload_is_accepted(self, client, provider) -> None:
        response = await post_edits(client, edit_payload())

        assert response.status == 422

    async def test_a_data_url_is_accepted(self, client) -> None:
        response = await post_edits(
            client, edit_payload(image=f"data:image/png;base64,{PAYLOAD}")
        )

        assert response.status == 422

    async def test_an_array_of_images_is_accepted(self, client) -> None:
        response = await post_edits(client, edit_payload(image=[PAYLOAD, PAYLOAD]))

        assert response.status == 422

    async def test_a_missing_image_is_refused_naming_the_field(self, client) -> None:
        response = await post_edits(client, {"model": PUBLIC_MODEL, "prompt": "rainy"})

        assert response.status == 400
        assert (await response.json())["reason"] == "invalid_request"
        assert "image" in (await response.json())["message"]

    async def test_a_payload_that_is_not_base64_is_refused(self, client) -> None:
        response = await post_edits(client, edit_payload(image="not base64!"))

        assert response.status == 400
        assert "image" in (await response.json())["message"]

    async def test_an_empty_payload_is_refused(self, client) -> None:
        response = await post_edits(client, edit_payload(image=""))

        assert response.status == 400

    async def test_an_image_that_is_not_a_string_is_refused(self, client) -> None:
        response = await post_edits(client, edit_payload(image=42))

        assert response.status == 400

    async def test_an_unknown_parameter_on_edits_is_refused(self, client) -> None:
        response = await post_edits(client, edit_payload(mask=PAYLOAD))

        assert response.status == 400
        assert "mask" in (await response.json())["message"]

    async def test_an_edit_answers_no_provider(self, client, provider) -> None:
        await post_edits(client, edit_payload())

        assert provider.calls == []


# ---------------------------------------------------------------------------
# Refusals that are about the route or the provider
# ---------------------------------------------------------------------------


class TestCapabilityRefusals:
    """A capability the route does not have is refused with the taxonomy's reason."""

    async def test_an_unknown_model_is_a_model_not_found_error(self, client, provider) -> None:
        response = await post_generations(client, payload(model="no-such-model"))

        assert response.status == 404
        assert (await response.json())["reason"] == "model_not_found"
        assert provider.calls == []

    async def test_a_provider_that_cannot_generate_is_refused(self) -> None:
        app = build_app(FakeTextProvider())
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"

    async def test_the_capability_refusal_names_the_operation(self) -> None:
        app = build_app(FakeTextProvider())
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        assert Capability.GENERATION.value in (await response.json())["message"]

    async def test_a_route_without_the_generation_capability_is_refused(self) -> None:
        registry = ModelRegistry(
            (
                Route(
                    operation=Operation.IMAGE,
                    index=0,
                    endpoint=UPSTREAM_ENDPOINT,
                    base_url=UPSTREAM_ENDPOINT,
                    model_id="image-model",
                    format="openai",
                    api_key=SECRET_KEY,
                    capabilities=frozenset({Capability.EDITING}),
                    extra_headers={},
                    allow_absolute_path_url=False,
                ),
            )
        )
        app = Application(router=Router())
        register_error_handlers(app)
        register_image_routes(app.router, InferenceService(registry, FakeImageProvider()))
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        assert response.status == 422
        assert Capability.GENERATION.value in (await response.json())["message"]

    async def test_an_edit_on_a_generation_only_route_is_refused(self) -> None:
        """The configured provider supports generation but not editing (PRD 4.6)."""
        app = build_app(FakeImageProvider())
        await app.start()
        try:
            response = await post_edits(TestClient(app), edit_payload())
        finally:
            await app.stop()

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"
        assert Capability.EDITING.value in (await response.json())["message"]

    async def test_the_editing_refusal_names_the_capability(self) -> None:
        app = build_app(FakeImageProvider())
        await app.start()
        try:
            response = await post_edits(TestClient(app), edit_payload())
        finally:
            await app.stop()

        assert Capability.EDITING.value in (await response.json())["message"]

    async def test_an_edit_on_a_route_that_declares_editing_is_still_refused(self) -> None:
        """The route's declared capabilities are enforced; none can be handed an edit.

        A route that declares ``generation`` and ``editing`` passes the
        capability check and is then refused for the gateway's own reason: the
        canonical image request carries no field for an image input, so the
        caller's image cannot be forwarded and an edit answered without it
        would report an image edited that never left the gateway (PRD
        section 9.5). The refusal is the same capability error, so a caller
        cannot confuse an unrepresentable edit with a served one.
        """
        registry = make_registry(editing=True)
        provider = FakeImageProvider()
        app = Application(router=Router())
        register_error_handlers(app)
        register_image_routes(app.router, InferenceService(registry, provider))
        await app.start()
        try:
            response = await post_edits(TestClient(app), edit_payload())
        finally:
            await app.stop()

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"
        assert Capability.EDITING.value in (await response.json())["message"]
        assert provider.calls == []

    async def test_an_edit_resolves_the_model_before_its_capabilities(self) -> None:
        app = build_app(FakeImageProvider())
        await app.start()
        try:
            response = await post_edits(TestClient(app), edit_payload(model="no-such-model"))
        finally:
            await app.stop()

        assert response.status == 404
        assert (await response.json())["reason"] == "model_not_found"

    async def test_a_provider_failure_is_mapped_onto_the_taxonomy(self) -> None:
        failure = UnavailableError("the upstream is unavailable")
        app = build_app(FakeImageProvider(error=failure))
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        assert response.status == 503
        assert (await response.json())["reason"] == "provider_unavailable"

    async def test_an_unclassified_failure_is_sanitised(self) -> None:
        app = build_app(FakeImageProvider(error=RuntimeError(f"boom {SECRET_KEY}")))
        await app.start()
        try:
            response = await post_generations(TestClient(app), payload())
        finally:
            await app.stop()

        body = await response.text()
        assert response.status == 500
        assert SECRET_KEY not in body
        assert "boom" not in body


# ---------------------------------------------------------------------------
# The malformed answer, through the endpoint
# ---------------------------------------------------------------------------


class TestTheMalformedAnswer:
    """A wrong-shaped answer is refused on that route and walked to another."""

    async def test_an_answer_with_the_wrong_image_count_walks_to_a_fallback(self) -> None:
        """The provider returns one image where two were asked; the fallback serves."""
        wrong_count = image_response(data=(GeneratedImage(b64_json=PAYLOAD),))
        outcomes = [wrong_count, image_response()]

        first = Route(
            operation=Operation.IMAGE,
            index=0,
            endpoint=UPSTREAM_ENDPOINT,
            base_url=UPSTREAM_ENDPOINT,
            model_id="image-model",
            format="openai",
            api_key=SECRET_KEY,
            capabilities=frozenset({Capability.GENERATION}),
            extra_headers={},
            allow_absolute_path_url=False,
        )
        fallback = Route(
            operation=Operation.IMAGE,
            index=1,
            endpoint="https://fallback.internal.example/v1",
            base_url="https://fallback.internal.example/v1",
            model_id="fallback-image",
            format="openai",
            api_key=SECRET_KEY,
            capabilities=frozenset({Capability.GENERATION}),
            extra_headers={},
            allow_absolute_path_url=False,
        )

        class ScriptedProvider:
            """A provider whose answers come from a script, like the service tests'."""

            def __init__(self) -> None:
                self.calls: list[Route] = []

            async def generate(
                self, route: Route, request: ImageRequest, context: object
            ) -> ImageResponse:
                self.calls.append(route)
                return outcomes.pop(0)

        provider = ScriptedProvider()
        app = Application(router=Router())
        register_error_handlers(app)
        register_image_routes(
            app.router, InferenceService(ModelRegistry((first, fallback)), provider)
        )
        await app.start()
        try:
            response = await post_generations(
                TestClient(app), payload(model=PUBLIC_MODEL, n=2)
            )
        finally:
            await app.stop()

        assert response.status == 200
        body = await response.json()
        assert len(body["data"]) == 2
        assert [route_.model_id for route_ in provider.calls] == [
            "image-model",
            "fallback-image",
        ]


# ---------------------------------------------------------------------------
# The application factory
# ---------------------------------------------------------------------------


class TestTheFactory:
    """The routes the server actually runs are registered (PRD sections 4.6 and 19)."""

    async def test_the_factory_registers_the_generations_endpoint(self) -> None:
        """``create_app`` answers ``POST /v1/images/generations``, not just a hand-built app.

        The image handler's own tests build their own application, so none of
        them can tell whether the factory registers the route at all — which is
        exactly how an endpoint that answers on a hand-built application comes
        to return 404 on one started from configuration alone.
        """
        registry = make_registry()
        provider = FakeImageProvider()
        app = create_app(
            Settings.from_env({}),
            registry=registry,
            service=InferenceService(registry, provider),
        )
        await app.start()
        try:
            response = await TestClient(app).post(
                "/v1/images/generations",
                content=JSONContent({"model": PUBLIC_MODEL, "prompt": "A lighthouse"}),
            )
        finally:
            await app.stop()

        assert response.status == 200
        body = await response.json()
        assert set(body) == {"created", "data"}
        assert len(provider.calls) == 1

    async def test_the_factory_registers_the_edits_endpoint(self) -> None:
        registry = make_registry()
        provider = FakeImageProvider()
        app = create_app(
            Settings.from_env({}),
            registry=registry,
            service=InferenceService(registry, provider),
        )
        await app.start()
        try:
            response = await TestClient(app).post(
                "/v1/images/edits",
                content=JSONContent(
                    {"model": PUBLIC_MODEL, "prompt": "rainy", "image": PAYLOAD}
                ),
            )
        finally:
            await app.stop()

        assert response.status == 422
        assert (await response.json())["reason"] == "unsupported_capability"
        assert provider.calls == []


# ---------------------------------------------------------------------------
# The startup check
# ---------------------------------------------------------------------------


class TestTheStartupCheck:
    """An image route whose dialect is unverified refuses the start (PRD 6.2, 14.1)."""

    def test_an_image_route_with_an_unverified_dialect_refuses_the_start(self) -> None:
        """The binding cannot run the dialect, so the deployment must not start.

        The text and embeddings operations are checked the same way; the image
        operation was excluded only because it had no binding, and a binding
        now exists. ``azure`` is refused because its path discards the injected
        client outright (PRD section 14.1). The refusal happens at startup
        rather than on the first caller, because a deployment whose
        configuration the transport controls cannot survive should not reach the
        point of serving requests (PRD section 14.1).
        """
        registry = ModelRegistry(
            (
                Route(
                    operation=Operation.IMAGE,
                    index=0,
                    endpoint=UPSTREAM_ENDPOINT,
                    base_url=UPSTREAM_ENDPOINT,
                    model_id="image-model",
                    format="azure",
                    api_key=SECRET_KEY,
                    capabilities=frozenset({Capability.GENERATION}),
                    extra_headers={},
                    allow_absolute_path_url=False,
                ),
            )
        )

        with pytest.raises(ConfigurationError) as error:
            create_app(Settings.from_env({}), registry=registry)

        assert "azure" in str(error.value)

    def test_an_image_route_with_a_verified_dialect_starts(self) -> None:
        registry = make_registry()

        app = create_app(Settings.from_env({}), registry=registry)

        assert app is not None

    def test_the_image_route_is_checked_beside_the_other_operations(self) -> None:
        """A verified text route does not make an unverified image route acceptable.

        The check covers every operation the binding runs, so a deployment that
        configures one good operation and one route the transport controls
        cannot survive still refuses to start — the image route is the reason,
        not the text one beside it.
        """
        text_route = Route(
            operation=Operation.TEXT,
            index=0,
            endpoint=UPSTREAM_ENDPOINT,
            base_url=UPSTREAM_ENDPOINT,
            model_id="upstream-text",
            format="openai",
            api_key=SECRET_KEY,
            capabilities=frozenset({Capability.CHAT}),
            extra_headers={},
            allow_absolute_path_url=False,
        )
        image_route = Route(
            operation=Operation.IMAGE,
            index=0,
            endpoint=UPSTREAM_ENDPOINT,
            base_url=UPSTREAM_ENDPOINT,
            model_id="image-model",
            format="azure",
            api_key=SECRET_KEY,
            capabilities=frozenset({Capability.GENERATION}),
            extra_headers={},
            allow_absolute_path_url=False,
        )
        registry = ModelRegistry((text_route, image_route))

        with pytest.raises(ConfigurationError) as error:
            create_app(Settings.from_env({}), registry=registry)

        assert "image-default" in str(error.value)
