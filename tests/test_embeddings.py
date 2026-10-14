"""Canonical embeddings types and the embeddings model ID (PRD sections 4.7, 6.3, 10, 13).

Two contracts meet here. The first is the canonical representation: an
embeddings request and response round-trip through ``to_dict`` and ``from_dict``
with every field emitted, and every state the representation cannot hold is
refused rather than dropped (PRD sections 4.7 and 9.1). The second is the public
model ID ``embedding-default``, which resolves onto the configured embeddings
routes exactly as ``text-default`` and ``image-default`` resolve onto theirs,
and appears in the model listing with the dimensions the deployment configured
(PRD sections 4.3 and 10).

Nothing here reaches the network: every endpoint is an IP literal and every
request is built by hand.
"""

import dataclasses
import json

import pytest
from blacksheep import Application
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

import model_gateway.domain as domain
from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation
from model_gateway.domain.embeddings import (
    ENCODING_FORMATS,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from model_gateway.domain.generation import Usage
from model_gateway.errors import ErrorReason, ModelNotFoundError
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.models import register_model_routes
from model_gateway.registry import ModelRegistry

PUBLIC_ADDRESS = "93.184.216.34"

VECTOR = Embedding(index=0, values=(0.12, -0.34, 0.56))

REQUEST = EmbeddingRequest(
    model="embedding-default",
    input=("First document", "Second document"),
    metadata={"trace": {"sampled": True}},
)

RESPONSE = EmbeddingResponse(
    model="embedding-default",
    upstream_model="embedding-model",
    data=(VECTOR, Embedding(index=1, values=(0.7, 0.8, 0.9))),
    usage=Usage(input_tokens=10, total_tokens=10),
    invocation_id="inv-7f3c1a90",
)


def request(**overrides) -> EmbeddingRequest:
    values = {"model": "embedding-default", "input": ("First document",)}
    values.update(overrides)
    return EmbeddingRequest(**values)


def response(**overrides) -> EmbeddingResponse:
    values = {
        "model": "embedding-default",
        "upstream_model": "embedding-model",
        "data": (VECTOR,),
        "usage": Usage(input_tokens=10, total_tokens=10),
        "invocation_id": "inv-7f3c1a90",
    }
    values.update(overrides)
    return EmbeddingResponse(**values)


# --------------------------------------------------------------------------- the vector


def test_the_domain_package_re_exports_every_canonical_embeddings_type():
    """``model_gateway.domain`` re-exports the canonical types by convention.

    ``model_gateway.domain.embeddings`` is the definition, and the package is
    what every other module imports from, so a type that exists only in the
    submodule is a type half the gateway cannot name.
    """
    assert domain.ENCODING_FORMATS is ENCODING_FORMATS
    assert domain.Embedding is Embedding
    assert domain.EmbeddingRequest is EmbeddingRequest
    assert domain.EmbeddingResponse is EmbeddingResponse


def test_an_embedding_round_trips():
    assert Embedding.from_dict(VECTOR.to_dict()) == VECTOR


def test_an_embedding_emits_every_field():
    assert VECTOR.to_dict() == {"index": 0, "values": [0.12, -0.34, 0.56]}


def test_an_integer_component_is_recorded_as_the_number_it_is():
    embedding = Embedding(index=0, values=(1, 2))

    assert embedding.values == (1.0, 2.0)
    assert embedding.to_dict() == {"index": 0, "values": [1.0, 2.0]}


def test_an_embedding_is_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        VECTOR.index = 1
    assert len({VECTOR, Embedding(index=0, values=(0.12, -0.34, 0.56))}) == 1


def test_an_embedding_refuses_a_negative_index():
    with pytest.raises(ValueError, match="'index' must be at least 0"):
        Embedding(index=-1, values=(0.1,))


def test_an_embedding_refuses_a_non_integer_index():
    with pytest.raises(ValueError, match="'index' must be an integer"):
        Embedding(index="0", values=(0.1,))

    with pytest.raises(ValueError, match="'index' must be an integer"):
        Embedding(index=True, values=(0.1,))


def test_an_embedding_refuses_an_empty_vector():
    with pytest.raises(ValueError, match="at least one number"):
        Embedding(index=0, values=())


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_an_embedding_refuses_a_non_finite_component(value):
    with pytest.raises(ValueError, match="must be a finite number"):
        Embedding(index=0, values=(0.1, value))


@pytest.mark.parametrize("value", [None, "0.1", True, [0.1], {"value": 0.1}])
def test_an_embedding_refuses_a_component_that_is_not_a_number(value):
    with pytest.raises(ValueError, match=r"'values'\[1\] must be a number"):
        Embedding(index=0, values=(0.1, value))


def test_an_embedding_refuses_a_key_it_does_not_define():
    with pytest.raises(ValueError, match="unsupported key 'vector'"):
        Embedding.from_dict({"index": 0, "values": [0.1], "vector": [0.1]})


def test_an_embedding_refuses_a_missing_field():
    with pytest.raises(ValueError, match="'values'"):
        Embedding.from_dict({"index": 0})


def test_an_embedding_refuses_a_payload_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        Embedding.from_dict([0, 0.1])


# --------------------------------------------------------------------------- the request


def test_a_request_round_trips_with_every_optional_field_unset():
    minimal = request()

    assert minimal.to_dict() == {
        "model": "embedding-default",
        "input": ["First document"],
        "dimensions": None,
        "encoding_format": None,
        "user": None,
        "metadata": None,
    }
    assert EmbeddingRequest.from_dict(minimal.to_dict()) == minimal


def test_a_request_round_trips_with_every_optional_field_set():
    document = REQUEST.to_dict()

    assert document == {
        "model": "embedding-default",
        "input": ["First document", "Second document"],
        "dimensions": None,
        "encoding_format": None,
        "user": None,
        "metadata": {"trace": {"sampled": True}},
    }
    assert EmbeddingRequest.from_dict(document) == REQUEST


def test_a_request_round_trip_is_stable_under_reparsing():
    once = EmbeddingRequest.from_dict(REQUEST.to_dict())
    twice = EmbeddingRequest.from_dict(once.to_dict())

    assert twice == once


def test_a_request_records_a_stated_dimension_and_encoding():
    parsed = request(dimensions=1024, encoding_format="float", user="caller-7")

    assert parsed.dimensions == 1024
    assert parsed.encoding_format == "float"
    assert parsed.user == "caller-7"
    assert EmbeddingRequest.from_dict(parsed.to_dict()) == parsed


def test_a_request_keeps_absent_metadata_distinct_from_empty_metadata():
    absent = request(metadata=None)
    empty = request(metadata={})

    assert absent.metadata is None
    assert empty.metadata == {}
    assert EmbeddingRequest.from_dict(absent.to_dict()).metadata is None
    assert EmbeddingRequest.from_dict(empty.to_dict()).metadata == {}


def test_a_request_holds_its_metadata_immutably():
    metadata = request(metadata={"trace": {"sampled": True}}).metadata

    with pytest.raises(TypeError):
        metadata["trace"] = "no"


def test_a_request_is_frozen_and_hashable():
    assert len({request(), request()}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        REQUEST.model = "text-default"


def test_a_request_refuses_an_empty_model_id():
    with pytest.raises(ValueError, match="model must not be empty"):
        request(model="")


def test_a_request_refuses_an_empty_input():
    with pytest.raises(ValueError, match="at least one string"):
        request(input=())


def test_a_request_refuses_a_bare_string_as_its_input():
    with pytest.raises(ValueError, match="a single string is not accepted"):
        request(input="First document")


def test_a_request_refuses_a_blank_input_entry():
    with pytest.raises(ValueError, match=r"input\[0\] must not be empty"):
        request(input=("",))

    with pytest.raises(ValueError, match=r"input\[1\] is blank"):
        request(input=("First document", "   "))


def test_a_request_keeps_its_input_verbatim():
    assert request(input=("  padded text  ",)).input == ("  padded text  ",)


def test_a_request_refuses_a_non_positive_dimension():
    with pytest.raises(ValueError, match="dimensions must be at least 1"):
        request(dimensions=0)

    with pytest.raises(ValueError, match="dimensions must be at least 1"):
        request(dimensions=-1)


def test_a_request_refuses_a_non_integer_dimension():
    with pytest.raises(ValueError, match="dimensions must be an integer"):
        request(dimensions="1024")


def test_the_documented_encoding_formats_are_the_two_wire_encodings():
    assert frozenset({"float", "base64"}) == ENCODING_FORMATS


@pytest.mark.parametrize("encoding", sorted(ENCODING_FORMATS))
def test_a_request_accepts_a_documented_encoding_format(encoding):
    assert request(encoding_format=encoding).encoding_format == encoding


def test_a_request_refuses_an_undocumented_encoding_format():
    with pytest.raises(ValueError, match="must be one of 'base64', 'float'"):
        request(encoding_format="hex")


def test_a_request_refuses_a_whitespace_encoding_format():
    with pytest.raises(ValueError, match="must be one of 'base64', 'float'"):
        request(encoding_format="  ")


def test_a_request_refuses_a_blank_user():
    with pytest.raises(ValueError, match="user must not be empty"):
        request(user="")


def test_a_request_refuses_metadata_that_is_not_json():
    with pytest.raises(ValueError, match="cannot carry"):
        request(metadata={"bad": {1, 2}})


def test_a_request_refuses_metadata_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        request(metadata=["trace"])


def test_a_request_refuses_a_key_it_does_not_define():
    with pytest.raises(ValueError, match="unsupported key 'encoding_formats'"):
        EmbeddingRequest.from_dict(
            {"model": "embedding-default", "input": ["a"], "encoding_formats": "float"}
        )


# --------------------------------------------------------------------------- the response


def test_a_response_round_trips():
    assert EmbeddingResponse.from_dict(RESPONSE.to_dict()) == RESPONSE


def test_a_response_round_trip_is_stable_under_reparsing():
    once = EmbeddingResponse.from_dict(RESPONSE.to_dict())
    twice = EmbeddingResponse.from_dict(once.to_dict())

    assert twice == once


def test_a_response_emits_every_field():
    assert RESPONSE.to_dict() == {
        "model": "embedding-default",
        "upstream_model": "embedding-model",
        "data": [
            {"index": 0, "values": [0.12, -0.34, 0.56]},
            {"index": 1, "values": [0.7, 0.8, 0.9]},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": None,
            "total_tokens": 10,
            "cached_tokens": None,
            "estimated_fields": [],
        },
        "invocation_id": "inv-7f3c1a90",
    }


def test_a_response_preserves_the_order_of_the_vectors():
    parsed = EmbeddingResponse.from_dict(RESPONSE.to_dict())

    assert [embedding.index for embedding in parsed.data] == [0, 1]


def test_a_response_is_frozen_and_hashable():
    assert len({response(), response()}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        RESPONSE.model = "text-default"


def test_a_response_refuses_a_vector_that_is_not_an_embedding():
    with pytest.raises(ValueError, match=r"data\[0\] must be an Embedding"):
        response(data=((0, (0.1,)),))


def test_a_response_refuses_usage_that_is_not_a_usage_record():
    with pytest.raises(ValueError, match="usage must be a Usage"):
        response(usage={"input_tokens": 10})


def test_a_response_refuses_an_empty_invocation_id():
    with pytest.raises(ValueError, match="invocation_id must not be empty"):
        response(invocation_id="")


def test_a_response_refuses_an_empty_upstream_model():
    with pytest.raises(ValueError, match="upstream_model must not be empty"):
        response(upstream_model="")


def test_a_response_refuses_a_key_it_does_not_define():
    payload = {**RESPONSE.to_dict(), "object": "list"}

    with pytest.raises(ValueError, match="unsupported key 'object'"):
        EmbeddingResponse.from_dict(payload)


def test_a_response_refuses_a_missing_usage():
    payload = {key: value for key, value in RESPONSE.to_dict().items() if key != "usage"}

    with pytest.raises(ValueError, match="usage must be a JSON object, got null"):
        EmbeddingResponse.from_dict(payload)


# --------------------------------------------------------------------------- the registry


def text_block(**overrides: str) -> dict[str, str]:
    env = {
        "TEXT_LLM_ENDPOINT": f"https://{PUBLIC_ADDRESS}/v1",
        "TEXT_LLM_MODEL_ID": "primary-model",
        "TEXT_LLM_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def embeddings_block(**overrides: str) -> dict[str, str]:
    env = {
        "EMBEDDINGS_ENDPOINT": f"https://{PUBLIC_ADDRESS}/embeddings/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def embeddings_fallback(model_id: str) -> dict[str, str]:
    return {
        "EMBEDDINGS_ROUTES_JSON": json.dumps(
            [
                {
                    "endpoint": f"https://{PUBLIC_ADDRESS}/fallback/v1",
                    "model_id": model_id,
                    "format": "openai",
                }
            ]
        )
    }


def registry(env: dict[str, str]) -> ModelRegistry:
    return ModelRegistry.from_env(env)


def build_app(models: ModelRegistry) -> Application:
    app = Application(router=Router())
    register_error_handlers(app)
    register_model_routes(app.router, models)
    return app


def test_embedding_default_resolves_to_the_configured_embeddings_routes():
    models = registry({**text_block(), **embeddings_block()})

    routes = models.routes_for("embedding-default")

    assert [route.operation for route in routes] == [Operation.EMBEDDINGS]
    assert [route.model_id for route in routes] == ["embedding-model"]


def test_embedding_default_resolves_to_its_fallbacks_in_configured_order():
    models = registry(
        {
            **text_block(),
            **embeddings_block(),
            **embeddings_fallback("embedding-fallback"),
        }
    )

    routes = models.routes_for("embedding-default")

    assert [(route.index, route.model_id) for route in routes] == [
        (0, "embedding-model"),
        (1, "embedding-fallback"),
    ]


def test_embedding_default_is_not_served_by_another_operations_routes():
    models = registry({**text_block(), **embeddings_block()})

    assert [route.operation for route in models.routes_for("text-default")] == [Operation.TEXT]
    assert [route.operation for route in models.routes_for("embedding-default")] == [
        Operation.EMBEDDINGS
    ]


def test_embedding_default_advertises_the_embeddings_capability():
    models = registry(embeddings_block())

    assert models.get("embedding-default").capabilities == frozenset({Capability.EMBEDDINGS})


def test_an_unknown_public_model_id_raises_the_model_not_found_error():
    models = registry(embeddings_block())

    with pytest.raises(ModelNotFoundError) as error:
        models.get("embedding-default-2")

    assert error.value.reason is ErrorReason.MODEL_NOT_FOUND
    assert error.value.http_status == 404


def test_a_public_model_id_whose_operation_is_unconfigured_is_not_found():
    models = registry(text_block())

    with pytest.raises(ModelNotFoundError):
        models.get("embedding-default")

    with pytest.raises(ModelNotFoundError):
        models.routes_for("embedding-default")


def test_list_models_omits_embeddings_when_it_is_unconfigured():
    models = registry({**text_block(), **embeddings_block()})
    without = registry(text_block())

    assert [info.id for info in models.list_models()] == [
        "text-default",
        "embedding-default",
    ]
    assert [info.id for info in without.list_models()] == ["text-default"]


def test_list_models_reports_the_configured_embedding_dimensions():
    models = registry(
        {
            **text_block(),
            **embeddings_block(),
            "EMBEDDINGS_DIMENSIONS": "1024",
        }
    )

    entries = {info.id: info for info in models.list_models()}

    assert entries["embedding-default"].dimensions == 1024
    assert entries["embedding-default"].context_window is None
    assert entries["text-default"].dimensions is None


def test_an_unconfigured_embedding_dimension_is_none_rather_than_zero():
    models = registry(embeddings_block())

    assert models.get("embedding-default").dimensions is None


async def test_the_model_listing_carries_the_embedding_dimensions():
    models = registry(
        {
            **text_block(),
            **embeddings_block(),
            "EMBEDDINGS_DIMENSIONS": "1024",
        }
    )
    app = build_app(models)
    await app.start()
    try:
        listing = await (await TestClient(app).get("/v1/models")).json()
    finally:
        await app.stop()

    entries = {entry["id"]: entry for entry in listing["data"]}

    assert entries["embedding-default"]["dimensions"] == 1024
    assert entries["embedding-default"]["capabilities"] == ["embeddings"]
    assert "dimensions" not in entries["text-default"]


async def test_the_model_listing_omits_embeddings_when_it_is_unconfigured():
    app = build_app(registry(text_block()))
    await app.start()
    try:
        listing = await (await TestClient(app).get("/v1/models")).json()
    finally:
        await app.stop()

    assert [entry["id"] for entry in listing["data"]] == ["text-default"]
