"""The public model registry: fixed public model IDs onto configured routes.

The registry is the only thing that maps a public ID onto a route, so these tests
pin both directions of that contract: a configured operation resolves to the ID
the caller used and to the routes the deployment configured for it, and an ID
the gateway does not expose resolves to nothing at all — no route, no endpoint,
no error message that carries one.

The advertised limits are deployment configuration rather than route fields, so
they are read from the environment the registry was built from. Tests that build
a registry from routes alone hold the limit variables out of the process
environment; tests that build one from an environment put them in it. Every
endpoint used here is an IP literal or is resolved by a stand-in, so nothing
touches DNS.
"""

import dataclasses
import json

import pytest

from model_gateway.capabilities import Capability
from model_gateway.config import ConfigurationError
from model_gateway.config.routes import Operation, parse_routes
from model_gateway.errors import ErrorReason, ModelNotFoundError
from model_gateway.registry import PUBLIC_MODEL_IDS, ModelInfo, ModelRegistry

PUBLIC_ADDRESS = "93.184.216.34"

LIMIT_VARIABLES = (
    "TEXT_LLM_CONTEXT_WINDOW",
    "TEXT_LLM_MAX_OUTPUT_TOKENS",
    "EMBEDDINGS_DIMENSIONS",
)

SECRET_KEY = "sk-upstream-secret-key"


@pytest.fixture(autouse=True)
def _hold_the_limit_variables_out_of_the_environment(monkeypatch):
    """Keep a registry built from routes alone free of the process environment.

    A deployment's limits are not route fields, so a registry constructed from
    routes reads them from ``os.environ``. Deleting them here is what makes the
    "an unconfigured limit is none" assertions below mean something.
    """
    for variable in LIMIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def public_resolver(host: str) -> list[str]:
    """Stand-in for DNS: every hostname resolves to one public address."""
    return [PUBLIC_ADDRESS]


def block(**overrides: str) -> dict[str, str]:
    """A valid TEXT_LLM singleton block with ``overrides`` applied."""
    env = {
        "TEXT_LLM_ENDPOINT": "https://primary.example/v1",
        "TEXT_LLM_MODEL_ID": "primary-model",
        "TEXT_LLM_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def image_block(**overrides: str) -> dict[str, str]:
    env = {
        "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_FORMAT": "azure",
    }
    env.update(overrides)
    return env


def embeddings_block(**overrides: str) -> dict[str, str]:
    env = {
        "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_FORMAT": "ollama",
    }
    env.update(overrides)
    return env


def environment(**overrides: str) -> dict[str, str]:
    """A complete environment with IP-literal endpoints, so no DNS is needed."""
    env = {
        "TEXT_LLM_ENDPOINT": f"https://{PUBLIC_ADDRESS}/v1",
        "TEXT_LLM_MODEL_ID": "primary-model",
        "TEXT_LLM_FORMAT": "openai",
        "IMAGE_GEN_ENDPOINT": f"https://{PUBLIC_ADDRESS}/images/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_FORMAT": "azure",
        "EMBEDDINGS_ENDPOINT": f"https://{PUBLIC_ADDRESS}/embeddings/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_FORMAT": "ollama",
    }
    env.update(overrides)
    return env


def routes_json(*entries: dict[str, object]) -> dict[str, str]:
    return {"TEXT_LLM_ROUTES_JSON": json.dumps(list(entries))}


def entry(**fields) -> dict[str, object]:
    """A self-contained ROUTES_JSON entry with ``fields`` applied."""
    route = {
        "endpoint": "https://fallback.example/v1",
        "model_id": "fallback-model",
        "format": "openai",
    }
    route.update(fields)
    return route


def registry(env: dict[str, str]) -> ModelRegistry:
    """A registry over the routes in ``env``, resolved without DNS."""
    return ModelRegistry(parse_routes(env, resolver=public_resolver))


# ---------------------------------------------------------------------------
# The fixed public model IDs
# ---------------------------------------------------------------------------


def test_the_public_model_ids_are_the_fixed_set():
    assert PUBLIC_MODEL_IDS == {
        Operation.TEXT: "text-default",
        Operation.IMAGE: "image-default",
        Operation.EMBEDDINGS: "embedding-default",
    }


def test_the_public_model_id_mapping_is_read_only():
    with pytest.raises(TypeError):
        PUBLIC_MODEL_IDS[Operation.TEXT] = "renamed"


# ---------------------------------------------------------------------------
# Listing the configured operations
# ---------------------------------------------------------------------------


def test_list_models_lists_every_configured_operation_in_canonical_order():
    models = registry({**block(), **image_block(), **embeddings_block()})

    assert [(info.id, info.operation) for info in models.list_models()] == [
        ("text-default", Operation.TEXT),
        ("image-default", Operation.IMAGE),
        ("embedding-default", Operation.EMBEDDINGS),
    ]


def test_list_models_omits_an_unconfigured_operation():
    models = registry(block())

    assert [(info.id, info.operation) for info in models.list_models()] == [
        ("text-default", Operation.TEXT)
    ]


def test_list_models_omits_every_unconfigured_operation():
    assert registry({}).list_models() == ()


def test_list_models_is_stable_across_calls():
    models = registry({**block(), **embeddings_block()})

    assert models.list_models() == models.list_models()


def test_list_models_takes_the_capabilities_from_the_primary_route():
    models = registry(
        {**block(TEXT_LLM_CAPABILITIES="chat,tools,vision"), **image_block(), **embeddings_block()}
    )

    capabilities = {info.id: info.capabilities for info in models.list_models()}

    assert capabilities["text-default"] == frozenset(
        {Capability.CHAT, Capability.TOOLS, Capability.VISION}
    )
    assert capabilities["image-default"] == frozenset({Capability.GENERATION})
    assert capabilities["embedding-default"] == frozenset({Capability.EMBEDDINGS})


def test_capabilities_of_a_fallback_route_are_not_advertised():
    models = registry(
        {
            **block(TEXT_LLM_CAPABILITIES="chat"),
            **routes_json(entry(capabilities=["chat", "tools", "vision"])),
        }
    )

    (info,) = models.list_models()

    assert info.capabilities == frozenset({Capability.CHAT})


def test_one_operations_routes_do_not_leak_into_another_operations_entry():
    models = registry(
        {
            **block(),
            "EMBEDDINGS_ENDPOINT": "https://primary.example/v1",
            "EMBEDDINGS_MODEL_ID": "primary-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    text, embeddings = models.list_models()

    assert (text.id, embeddings.id) == ("text-default", "embedding-default")
    assert text.operation is not embeddings.operation
    assert text.capabilities == frozenset({Capability.CHAT})
    assert embeddings.capabilities == frozenset({Capability.EMBEDDINGS})


def test_get_returns_the_listed_entry_for_each_public_id():
    models = registry({**block(), **image_block(), **embeddings_block()})

    listed = {info.id: info for info in models.list_models()}

    assert models.get("text-default") == listed["text-default"]
    assert models.get("image-default") == listed["image-default"]
    assert models.get("embedding-default") == listed["embedding-default"]


# ---------------------------------------------------------------------------
# The advertised limits
# ---------------------------------------------------------------------------


def test_the_text_limits_come_from_the_text_configuration():
    models = ModelRegistry.from_env(
        environment(
            TEXT_LLM_CONTEXT_WINDOW="1048576",
            TEXT_LLM_MAX_OUTPUT_TOKENS="131072",
        )
    )

    (text, image, embeddings) = models.list_models()

    assert (text.context_window, text.max_output_tokens) == (1048576, 131072)
    assert (image.context_window, image.max_output_tokens) == (None, None)
    assert (embeddings.context_window, embeddings.max_output_tokens) == (None, None)


def test_the_embeddings_dimensions_come_from_the_embeddings_configuration():
    models = ModelRegistry.from_env(environment(EMBEDDINGS_DIMENSIONS="1024"))

    (text, image, embeddings) = models.list_models()

    assert embeddings.dimensions == 1024
    assert text.dimensions is None
    assert image.dimensions is None


def test_the_advertised_dimensions_are_the_primary_routes_own():
    """A fallback route's declared dimensionality is not the model's contract."""
    env = environment(EMBEDDINGS_DIMENSIONS="1024")
    env["EMBEDDINGS_ROUTES_JSON"] = json.dumps(
        [{"endpoint": f"https://{PUBLIC_ADDRESS}/fallback/v1", "model_id": "fallback",
          "format": "ollama", "dimensions": 512}]
    )
    models = ModelRegistry.from_env(env)

    assert models.get("embedding-default").dimensions == 1024

    undeclared = environment()
    undeclared["EMBEDDINGS_ROUTES_JSON"] = json.dumps(
        [{"endpoint": f"https://{PUBLIC_ADDRESS}/fallback/v1", "model_id": "fallback",
          "format": "ollama", "dimensions": 512}]
    )
    models = ModelRegistry.from_env(undeclared)

    assert models.get("embedding-default").dimensions is None


def test_an_unconfigured_limit_is_none():
    models = ModelRegistry.from_env(environment())

    for info in models.list_models():
        assert (info.context_window, info.max_output_tokens, info.dimensions) == (None, None, None)


def test_a_blank_limit_is_absent():
    models = ModelRegistry.from_env(
        environment(
            TEXT_LLM_CONTEXT_WINDOW="   ",
            TEXT_LLM_MAX_OUTPUT_TOKENS="",
            EMBEDDINGS_DIMENSIONS=" ",
        )
    )

    for info in models.list_models():
        assert (info.context_window, info.max_output_tokens, info.dimensions) == (None, None, None)


def test_a_limit_is_read_as_the_integer_it_was_configured_as():
    models = ModelRegistry.from_env(
        environment(
            TEXT_LLM_CONTEXT_WINDOW=" 8000 ",
            TEXT_LLM_MAX_OUTPUT_TOKENS="+4096",
            EMBEDDINGS_DIMENSIONS=" 512 ",
        )
    )

    limits = {info.id: info for info in models.list_models()}

    assert limits["text-default"].context_window == 8000
    assert limits["text-default"].max_output_tokens == 4096
    assert limits["embedding-default"].dimensions == 512


def test_a_malformed_dimensionality_is_rejected_rather_than_guessed():
    with pytest.raises(
        ConfigurationError, match="EMBEDDINGS_DIMENSIONS must be a positive integer"
    ):
        ModelRegistry.from_env(environment(EMBEDDINGS_DIMENSIONS="plenty"))


@pytest.mark.parametrize(
    "variable", ["TEXT_LLM_CONTEXT_WINDOW", "TEXT_LLM_MAX_OUTPUT_TOKENS"]
)
def test_a_negative_text_limit_is_rejected(variable):
    with pytest.raises(ConfigurationError, match=f"{variable} must not be negative"):
        ModelRegistry.from_env(environment(**{variable: "-1"}))


def test_a_non_positive_embedding_dimension_is_rejected():
    with pytest.raises(
        ConfigurationError, match="EMBEDDINGS_DIMENSIONS must be a positive integer, got '0'"
    ):
        ModelRegistry.from_env(environment(EMBEDDINGS_DIMENSIONS="0"))


def test_from_env_builds_the_registry_from_the_process_environment(monkeypatch):
    for variable, value in environment(
        TEXT_LLM_CONTEXT_WINDOW="1048576",
        TEXT_LLM_MAX_OUTPUT_TOKENS="131072",
        EMBEDDINGS_DIMENSIONS="1024",
    ).items():
        monkeypatch.setenv(variable, value)

    models = ModelRegistry.from_env()

    assert [
        (info.id, info.context_window, info.max_output_tokens, info.dimensions)
        for info in models.list_models()
    ] == [
        ("text-default", 1048576, 131072, None),
        ("image-default", None, None, None),
        ("embedding-default", None, None, 1024),
    ]


def test_a_registry_built_from_routes_reads_the_process_environment(monkeypatch):
    """The text limits are process configuration; the embeddings limit is not.

    Only ``TEXT_LLM_CONTEXT_WINDOW`` and ``TEXT_LLM_MAX_OUTPUT_TOKENS`` remain
    environment configuration the registry reads from the process environment
    when it is built from routes alone. The embeddings dimensionality is a
    property of the primary route, so a value exported into the process
    environment but absent from the routes the registry was given advertises
    nothing.
    """
    monkeypatch.setenv("TEXT_LLM_CONTEXT_WINDOW", "128000")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "384")

    models = ModelRegistry(parse_routes(environment()))

    limits = {info.id: info for info in models.list_models()}

    assert limits["text-default"].context_window == 128000
    assert limits["embedding-default"].dimensions is None


# ---------------------------------------------------------------------------
# Resolving one public model ID
# ---------------------------------------------------------------------------


def test_get_raises_for_an_unknown_model_id():
    models = registry(block())

    with pytest.raises(ModelNotFoundError) as error:
        models.get("gpt-sunburst")

    assert error.value.reason is ErrorReason.MODEL_NOT_FOUND
    assert error.value.http_status == 404


def test_get_raises_for_a_public_model_id_that_is_not_an_alias():
    models = registry(block())

    for alias in ("primary-model", "openai/primary-model", "text", ""):
        with pytest.raises(ModelNotFoundError):
            models.get(alias)


def test_get_raises_for_a_public_id_whose_operation_is_unconfigured():
    models = registry(block())

    with pytest.raises(ModelNotFoundError):
        models.get("image-default")

    with pytest.raises(ModelNotFoundError):
        models.get("embedding-default")


def test_the_unknown_model_error_names_the_configured_public_ids():
    models = registry({**block(), **embeddings_block()})

    with pytest.raises(ModelNotFoundError) as error:
        models.get("no-such-model")

    message = str(error.value)

    assert "text-default" in message
    assert "embedding-default" in message


def test_the_unknown_model_error_leaks_no_route_configuration():
    models = registry(
        {
            **block(TEXT_LLM_API_KEY=SECRET_KEY),
            **routes_json(entry(model_id="fallback-upstream", api_key=SECRET_KEY)),
        }
    )

    with pytest.raises(ModelNotFoundError) as error:
        models.get("no-such-model")

    message = str(error.value)
    for secret in (
        "primary.example",
        "fallback.example",
        "primary-model",
        "fallback-upstream",
        "fallback-model",
        SECRET_KEY,
    ):
        assert secret not in message


# ---------------------------------------------------------------------------
# The routes behind one public model ID
# ---------------------------------------------------------------------------


def test_routes_for_returns_the_primary_route_first():
    routes = parse_routes(
        {**block(), **routes_json(entry(), entry(model_id="fallback-2"))},
        resolver=public_resolver,
    )
    models = ModelRegistry(routes)

    assert [(route.index, route.model_id) for route in models.routes_for("text-default")] == [
        (0, "primary-model"),
        (1, "fallback-model"),
        (2, "fallback-2"),
    ]


def test_routes_for_returns_the_primary_route_when_it_is_the_only_route():
    models = registry(block())

    (route,) = models.routes_for("text-default")

    assert route.index == 0
    assert route.model_id == "primary-model"


def test_routes_for_never_returns_a_route_of_another_operation():
    models = registry(
        {
            **block(),
            "EMBEDDINGS_ENDPOINT": "https://primary.example/v1",
            "EMBEDDINGS_MODEL_ID": "primary-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    assert [route.operation for route in models.routes_for("text-default")] == [Operation.TEXT]
    assert [route.operation for route in models.routes_for("embedding-default")] == [
        Operation.EMBEDDINGS
    ]


def test_routes_for_raises_for_an_unknown_model_id():
    models = registry(block())

    with pytest.raises(ModelNotFoundError):
        models.routes_for("no-such-model")


def test_routes_for_raises_for_an_unconfigured_operation_and_returns_no_route():
    models = registry(block())

    with pytest.raises(ModelNotFoundError):
        models.routes_for("image-default")

    assert [route.operation for route in models.routes_for("text-default")] == [Operation.TEXT]


# ---------------------------------------------------------------------------
# The routes behind one operation, resolved without a model ID
# ---------------------------------------------------------------------------


def test_routes_for_operation_returns_the_primary_route_first():
    routes = parse_routes(
        {**block(), **routes_json(entry(), entry(model_id="fallback-2"))},
        resolver=public_resolver,
    )
    models = ModelRegistry(routes)

    assert [
        (route.index, route.model_id) for route in models.routes_for_operation(Operation.TEXT)
    ] == [
        (0, "primary-model"),
        (1, "fallback-model"),
        (2, "fallback-2"),
    ]


def test_routes_for_operation_returns_the_same_routes_as_routes_for():
    models = registry(
        {
            **block(),
            "EMBEDDINGS_ENDPOINT": "https://primary.example/v1",
            "EMBEDDINGS_MODEL_ID": "primary-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    assert models.routes_for_operation(Operation.TEXT) == models.routes_for("text-default")
    assert models.routes_for_operation(Operation.EMBEDDINGS) == models.routes_for(
        "embedding-default"
    )


def test_routes_for_operation_never_returns_a_route_of_another_operation():
    models = registry(
        {
            **block(),
            "EMBEDDINGS_ENDPOINT": "https://primary.example/v1",
            "EMBEDDINGS_MODEL_ID": "primary-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    assert [route.operation for route in models.routes_for_operation(Operation.TEXT)] == [
        Operation.TEXT
    ]
    assert [
        route.operation for route in models.routes_for_operation(Operation.EMBEDDINGS)
    ] == [Operation.EMBEDDINGS]


def test_routes_for_operation_raises_for_an_unconfigured_operation():
    models = registry(block())

    with pytest.raises(ModelNotFoundError) as error:
        models.routes_for_operation(Operation.IMAGE)

    assert "image-default" in error.value.message


def test_routes_for_operation_raises_model_not_found_for_grounded_search():
    """An operation the contract reserves but publishes no model ID for is a model-not-found.

    The inference layer resolves the operation from the transport and looks its
    routes up by operation. Grounded search is in the contract's vocabulary — the
    proto enum reserves it — but the gateway exposes no public model ID for it, so
    the lookup must fail with the gateway's own model-not-found error. A KeyError
    here would be an implementation accident escaping as a 500: the operation is
    a known member of the enum, it simply resolves to nothing.
    """
    models = registry(block())

    with pytest.raises(ModelNotFoundError) as error:
        models.routes_for_operation(Operation.GROUNDED_SEARCH)

    assert error.value.reason is ErrorReason.MODEL_NOT_FOUND
    assert "grounded-search" in error.value.message


# ---------------------------------------------------------------------------
# The model info itself
# ---------------------------------------------------------------------------


def test_model_info_is_frozen_and_hashable():
    info = ModelInfo(
        id="text-default",
        operation=Operation.TEXT,
        capabilities=frozenset({Capability.CHAT}),
        context_window=1048576,
        max_output_tokens=131072,
        dimensions=None,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        info.context_window = 5
    assert len({info}) == 1


def test_model_info_distinguishes_a_configured_limit_from_an_unconfigured_one():
    configured = ModelInfo(
        id="text-default",
        operation=Operation.TEXT,
        capabilities=frozenset({Capability.CHAT}),
        context_window=1048576,
        max_output_tokens=131072,
        dimensions=None,
    )
    unconfigured = ModelInfo(
        id="text-default",
        operation=Operation.TEXT,
        capabilities=frozenset({Capability.CHAT}),
        context_window=None,
        max_output_tokens=None,
        dimensions=None,
    )

    assert configured != unconfigured
