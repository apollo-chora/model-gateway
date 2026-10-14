"""Tests for the model endpoints (PRD sections 4.3, 10 and acceptance criterion 12)."""

import pytest
from blacksheep import Application
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.models import register_model_routes
from model_gateway.registry import ModelRegistry

SECRET_KEY = "sk-upstream-secret-key"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"


def make_route(
    operation: Operation,
    model_id: str,
    *,
    index: int = 0,
    capabilities: frozenset[Capability],
) -> Route:
    return Route(
        operation=operation,
        index=index,
        endpoint=UPSTREAM_ENDPOINT,
        base_url=UPSTREAM_ENDPOINT,
        model_id=model_id,
        format="openai",
        api_key=SECRET_KEY,
        capabilities=capabilities,
        extra_headers={"x-tenant": SECRET_KEY},
        allow_absolute_path_url=False,
    )


def configured_registry() -> ModelRegistry:
    """A registry for a text route and an embedding route, and no image route."""
    return ModelRegistry(
        (
            make_route(Operation.TEXT, "upstream-text", capabilities=frozenset({Capability.CHAT})),
            make_route(
                Operation.EMBEDDINGS,
                "upstream-embedding",
                capabilities=frozenset({Capability.EMBEDDINGS}),
            ),
        )
    )


def build_app(registry: ModelRegistry) -> Application:
    app = Application(router=Router())
    register_error_handlers(app)
    register_model_routes(app.router, registry)
    return app


@pytest.fixture
async def client():
    app = build_app(configured_registry())
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


async def test_list_returns_an_openai_list_envelope(client):
    body = await (await client.get("/v1/models")).json()

    assert body["object"] == "list"
    assert isinstance(body["data"], list)


async def test_list_returns_only_configured_operations(client):
    body = await (await client.get("/v1/models")).json()

    assert [entry["id"] for entry in body["data"]] == ["text-default", "embedding-default"]


async def test_list_entries_carry_the_model_object_shape(client):
    entry = (await (await client.get("/v1/models")).json())["data"][0]

    assert entry["object"] == "model"
    assert entry["created"] == 0
    assert entry["owned_by"] == "gateway"
    assert entry["capabilities"] == ["chat"]


async def test_list_reports_each_operations_own_capabilities(client):
    listing = await (await client.get("/v1/models")).json()
    entries = {entry["id"]: entry for entry in listing["data"]}

    assert entries["embedding-default"]["capabilities"] == ["embeddings"]


async def test_get_returns_one_model_object(client):
    response = await client.get("/v1/models/text-default")

    assert response.status == 200
    body = await response.json()
    assert body["id"] == "text-default"
    assert body["object"] == "model"
    assert body["created"] == 0
    assert body["owned_by"] == "gateway"
    assert body["capabilities"] == ["chat"]


async def test_get_refuses_an_unknown_model_with_the_model_not_found_reason(client):
    response = await client.get("/v1/models/no-such-model")

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"


async def test_get_refuses_an_upstream_model_name(client):
    response = await client.get("/v1/models/upstream-text")

    assert response.status == 404
    assert (await response.json())["reason"] == "model_not_found"


async def test_a_registry_with_no_routes_lists_nothing():
    app = build_app(ModelRegistry(()))
    await app.start()
    try:
        response = await TestClient(app).get("/v1/models")
    finally:
        await app.stop()

    assert await response.json() == {"object": "list", "data": []}


@pytest.mark.parametrize(
    "path",
    ["/v1/models", "/v1/models/text-default", "/v1/models/no-such-model"],
)
async def test_no_response_body_carries_upstream_configuration(client, path):
    body = await (await client.get(path)).text()

    assert SECRET_KEY not in body
    assert UPSTREAM_ENDPOINT not in body
    assert "x-tenant" not in body
    assert "upstream-text" not in body
