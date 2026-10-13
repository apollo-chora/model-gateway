"""Route parsing and startup validation (PRD sections 6, 7, 8, 10, 11, 18)."""

import dataclasses
import json

import pytest

from model_gateway.capabilities import Capability
from model_gateway.config import ConfigurationError
from model_gateway.config.routes import Operation, Route, parse_routes

PUBLIC_ADDRESS = "93.184.216.34"
METADATA_ADDRESS = "169.254.169.254"

SUPPORTED_CAPABILITIES = "'chat', 'tools', 'vision', 'embeddings', 'generation', 'editing'"

ENTRY_KEYS = (
    "'allow_absolute_path_url', 'api_key', 'capabilities', 'endpoint', 'extra_headers', "
    "'format', 'model_id', 'path'"
)


def public_resolver(host: str) -> list[str]:
    """Stand-in for DNS: every hostname resolves to one public address."""
    return [PUBLIC_ADDRESS]


def resolver_for(addresses: dict[str, list[str]]):
    def resolve(host: str) -> list[str]:
        return addresses[host]

    return resolve


PUBLIC_HOSTS = {
    "primary.example": [PUBLIC_ADDRESS],
    "fallback.example": [PUBLIC_ADDRESS],
    "metadata.example": [METADATA_ADDRESS],
}


def parse(env: dict[str, str], resolver=public_resolver) -> tuple[Route, ...]:
    """Parse routes without touching DNS."""
    return parse_routes(env, resolver=resolver)


def message(env: dict[str, str], resolver=public_resolver) -> str:
    """The error a deployment would see for ``env``."""
    with pytest.raises(ConfigurationError) as error:
        parse(env, resolver)
    return str(error.value)


def block(**overrides: str) -> dict[str, str]:
    """A valid TEXT_LLM singleton block with ``overrides`` applied."""
    env = {
        "TEXT_LLM_ENDPOINT": "https://primary.example/v1",
        "TEXT_LLM_MODEL_ID": "primary-model",
        "TEXT_LLM_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def entry(**fields) -> dict[str, object]:
    """A self-contained ROUTES_JSON entry with ``fields`` applied."""
    route = {
        "endpoint": "https://fallback.example/v1",
        "model_id": "fallback-model",
        "format": "openai",
    }
    route.update(fields)
    return route


def routes_json(*entries: dict[str, object]) -> dict[str, str]:
    return {"TEXT_LLM_ROUTES_JSON": json.dumps(list(entries))}


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


def test_operation_values_match_the_prd():
    assert Operation.TEXT == "text"
    assert Operation.IMAGE == "image"
    assert Operation.EMBEDDINGS == "embeddings"


def test_public_model_ids_are_the_fixed_public_contract():
    assert Operation.TEXT.public_model_id == "text-default"
    assert Operation.IMAGE.public_model_id == "image-default"
    assert Operation.EMBEDDINGS.public_model_id == "embedding-default"


# ---------------------------------------------------------------------------
# The singleton block
# ---------------------------------------------------------------------------


def test_no_configuration_yields_no_routes():
    assert parse({}) == ()


def test_unrelated_variables_do_not_create_routes():
    assert parse({"GATEWAY_AUTH_ENABLED": "true", "CHORA_HTTP_PORT": "8080"}) == ()


def test_singleton_block_defines_the_primary_route():
    routes = parse(block(TEXT_LLM_API_KEY="secret"))

    assert routes == (
        Route(
            operation=Operation.TEXT,
            index=0,
            endpoint="https://primary.example/v1",
            base_url="https://primary.example/v1",
            model_id="primary-model",
            format="openai",
            api_key="secret",
            capabilities=frozenset({Capability.CHAT}),
            extra_headers={},
            allow_absolute_path_url=False,
        ),
    )


def test_primary_route_fields_are_readable_individually():
    (route,) = parse(block())

    assert route.operation is Operation.TEXT
    assert route.index == 0
    assert route.endpoint == "https://primary.example/v1"
    assert route.base_url == "https://primary.example/v1"
    assert route.model_id == "primary-model"
    assert route.format == "openai"
    assert route.api_key == ""
    assert route.capabilities == frozenset({Capability.CHAT})
    assert route.extra_headers == {}
    assert route.allow_absolute_path_url is False


def test_litellm_model_is_the_provider_prefixed_model_string():
    (route,) = parse(block(TEXT_LLM_MODEL_ID="gpt-sunburst", TEXT_LLM_FORMAT="openai"))

    assert route.litellm_model == "openai/gpt-sunburst"


def test_configured_values_are_stripped():
    (route,) = parse(
        block(
            TEXT_LLM_ENDPOINT="  https://primary.example/v1  ",
            TEXT_LLM_MODEL_ID=" primary-model ",
            TEXT_LLM_FORMAT=" openai ",
            TEXT_LLM_API_KEY=" secret ",
        )
    )

    assert route.endpoint == "https://primary.example/v1"
    assert route.model_id == "primary-model"
    assert route.format == "openai"
    assert route.api_key == "secret"


@pytest.mark.parametrize(
    "dialect",
    ["openai", "anthropic", "vertex_ai", "gemini", "ollama", "azure"],
)
def test_any_provider_dialect_is_accepted_as_a_format(dialect):
    (route,) = parse(block(TEXT_LLM_FORMAT=dialect))

    assert route.format == dialect
    assert route.litellm_model == f"{dialect}/primary-model"


def test_an_unlisted_provider_dialect_is_accepted():
    (route,) = parse(block(TEXT_LLM_FORMAT="openrouter"))

    assert route.format == "openrouter"


def test_api_key_is_optional_because_some_routes_use_ambient_credentials():
    (route,) = parse(block(TEXT_LLM_FORMAT="vertex_ai"))

    assert route.api_key == ""


def test_route_is_frozen():
    (route,) = parse(block())

    with pytest.raises(dataclasses.FrozenInstanceError):
        route.model_id = "other"


# ---------------------------------------------------------------------------
# One operation block per operation
# ---------------------------------------------------------------------------


def test_every_operation_block_is_parsed_independently():
    routes = parse(
        {
            **block(),
            "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
            "IMAGE_GEN_MODEL_ID": "image-model",
            "IMAGE_GEN_FORMAT": "azure",
            "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
            "EMBEDDINGS_MODEL_ID": "embedding-model",
            "EMBEDDINGS_FORMAT": "ollama",
        }
    )

    assert [route.operation for route in routes] == [
        Operation.TEXT,
        Operation.IMAGE,
        Operation.EMBEDDINGS,
    ]
    assert [route.index for route in routes] == [0, 0, 0]
    assert [route.model_id for route in routes] == [
        "primary-model",
        "image-model",
        "embedding-model",
    ]
    assert [route.litellm_model for route in routes] == [
        "openai/primary-model",
        "azure/image-model",
        "ollama/embedding-model",
    ]


def test_an_unconfigured_operation_contributes_no_routes():
    routes = parse(block())

    assert len(routes) == 1
    assert routes[0].operation is Operation.TEXT


def test_image_generation_defaults_to_the_generation_capability():
    (route,) = parse(
        {
            "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
            "IMAGE_GEN_MODEL_ID": "image-model",
            "IMAGE_GEN_FORMAT": "openai",
        }
    )

    assert route.operation is Operation.IMAGE
    assert route.capabilities == frozenset({Capability.GENERATION})


def test_embeddings_defaults_to_the_embeddings_capability():
    (route,) = parse(
        {
            "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
            "EMBEDDINGS_MODEL_ID": "embedding-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    assert route.operation is Operation.EMBEDDINGS
    assert route.capabilities == frozenset({Capability.EMBEDDINGS})


# ---------------------------------------------------------------------------
# ROUTES_JSON
# ---------------------------------------------------------------------------


def test_routes_json_appends_routes_in_array_order():
    routes = parse(
        {
            **block(),
            **routes_json(
                entry(endpoint="https://fallback-1.example/v1", model_id="fallback-1"),
                entry(endpoint="https://fallback-2.example/v1", model_id="fallback-2"),
            ),
        }
    )

    assert [route.index for route in routes] == [0, 1, 2]
    assert [route.model_id for route in routes] == [
        "primary-model",
        "fallback-1",
        "fallback-2",
    ]
    assert [route.base_url for route in routes] == [
        "https://primary.example/v1",
        "https://fallback-1.example/v1",
        "https://fallback-2.example/v1",
    ]


def test_an_empty_routes_json_array_means_one_route():
    routes = parse({**block(), "TEXT_LLM_ROUTES_JSON": "[]"})

    assert len(routes) == 1
    assert routes[0].index == 0


def test_a_blank_routes_json_is_absent():
    routes = parse({**block(), "TEXT_LLM_ROUTES_JSON": "   "})

    assert len(routes) == 1


def test_entry_fields_are_never_inherited_from_the_singleton_block():
    routes = parse(
        {
            **block(
                TEXT_LLM_API_KEY="primary-secret",
                TEXT_LLM_PATH="/custom",
                TEXT_LLM_CAPABILITIES="chat,tools",
                TEXT_LLM_EXTRA_HEADERS_JSON='{"x-organization": "acme"}',
                TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL="false",
            ),
            **routes_json(entry()),
        }
    )

    primary, fallback = routes

    assert primary.api_key == "primary-secret"
    assert primary.base_url == "https://primary.example/v1/custom"
    assert primary.capabilities == frozenset({Capability.CHAT, Capability.TOOLS})
    assert primary.extra_headers == {"x-organization": "acme"}

    assert fallback.api_key == ""
    assert fallback.base_url == "https://fallback.example/v1"
    assert fallback.capabilities == frozenset({Capability.CHAT})
    assert fallback.extra_headers == {}


def test_an_entry_declares_its_own_capabilities():
    routes = parse(
        {**block(TEXT_LLM_CAPABILITIES="chat"), **routes_json(entry(capabilities=["vision"]))}
    )

    assert routes[0].capabilities == frozenset({Capability.CHAT})
    assert routes[1].capabilities == frozenset({Capability.VISION})


def test_an_entry_declares_its_own_credentials_and_headers():
    routes = parse(
        {
            **block(),
            **routes_json(
                entry(
                    api_key="fallback-secret",
                    extra_headers={"x-organization": "fallback-org"},
                )
            ),
        }
    )

    assert routes[1].api_key == "fallback-secret"
    assert routes[1].extra_headers == {"x-organization": "fallback-org"}


def test_each_operation_appends_its_own_routes_json():
    routes = parse(
        {
            **block(),
            **routes_json(entry(model_id="text-fallback")),
            "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
            "EMBEDDINGS_MODEL_ID": "embedding-model",
            "EMBEDDINGS_FORMAT": "openai",
            "EMBEDDINGS_ROUTES_JSON": json.dumps(
                [entry(endpoint="https://embeddings-2.example/v1", model_id="embedding-2")]
            ),
        }
    )

    assert [(route.operation, route.index) for route in routes] == [
        (Operation.TEXT, 0),
        (Operation.TEXT, 1),
        (Operation.EMBEDDINGS, 0),
        (Operation.EMBEDDINGS, 1),
    ]
    assert [route.model_id for route in routes] == [
        "primary-model",
        "text-fallback",
        "embedding-model",
        "embedding-2",
    ]


def test_image_routes_json_is_supported():
    routes = parse(
        {
            "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
            "IMAGE_GEN_MODEL_ID": "image-model",
            "IMAGE_GEN_FORMAT": "openai",
            "IMAGE_GEN_ROUTES_JSON": json.dumps(
                [entry(endpoint="https://images-2.example/v1", model_id="image-2", format="azure")]
            ),
        }
    )

    assert [(route.operation, route.index) for route in routes] == [
        (Operation.IMAGE, 0),
        (Operation.IMAGE, 1),
    ]
    assert routes[1].litellm_model == "azure/image-2"


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {"TEXT_LLM_ENDPOINT": ""}, "TEXT_LLM_ENDPOINT must not be empty", id="endpoint"
        ),
        pytest.param(
            {"TEXT_LLM_ENDPOINT": "   "},
            "TEXT_LLM_ENDPOINT must not be empty",
            id="blank-endpoint",
        ),
        pytest.param(
            {"TEXT_LLM_MODEL_ID": ""}, "TEXT_LLM_MODEL_ID must not be empty", id="model-id"
        ),
        pytest.param({"TEXT_LLM_FORMAT": ""}, "TEXT_LLM_FORMAT must not be empty", id="format"),
    ],
)
def test_a_partially_configured_block_is_rejected(overrides, expected):
    assert message(block(**overrides)) == expected


@pytest.mark.parametrize(
    "present",
    ["TEXT_LLM_ENDPOINT", "TEXT_LLM_MODEL_ID", "TEXT_LLM_FORMAT"],
)
def test_a_block_with_one_of_the_required_fields_is_rejected(present):
    env = {present: "https://only.example/v1" if "ENDPOINT" in present else "only"}

    assert message(env).endswith("must not be empty")


def test_a_block_with_only_inert_variables_contributes_no_routes():
    env = {
        "TEXT_LLM_API_KEY": "secret",
        "TEXT_LLM_AUTH_TYPE": "bearer",
        "TEXT_LLM_EXTRA_HEADERS_JSON": "{}",
        "TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL": "false",
    }

    assert parse(env) == ()


def test_a_disabled_block_does_not_prevent_other_operations_from_operating():
    env = {
        "TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL": "false",
        "TEXT_LLM_EXTRA_HEADERS_JSON": "{}",
        "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_FORMAT": "openai",
    }

    routes = parse(env)

    assert [route.operation for route in routes] == [Operation.EMBEDDINGS]


def test_embeddings_block_without_a_format_is_rejected():
    env = {
        "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
    }

    assert message(env) == "EMBEDDINGS_FORMAT must not be empty"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_are_parsed_from_a_comma_separated_list():
    (route,) = parse(block(TEXT_LLM_CAPABILITIES="chat,tools,vision"))

    assert route.capabilities == frozenset({Capability.CHAT, Capability.TOOLS, Capability.VISION})


def test_capability_parsing_tolerates_spacing_case_and_duplicates():
    (route,) = parse(block(TEXT_LLM_CAPABILITIES=" chat , TOOLS ,, chat ,Vision"))

    assert route.capabilities == frozenset({Capability.CHAT, Capability.TOOLS, Capability.VISION})


def test_a_blank_capabilities_list_falls_back_to_the_operation_default():
    (route,) = parse(block(TEXT_LLM_CAPABILITIES="   "))

    assert route.capabilities == frozenset({Capability.CHAT})


def test_an_unknown_capability_is_rejected():
    assert message(block(TEXT_LLM_CAPABILITIES="chat,telepathy")) == (
        f"TEXT_LLM_CAPABILITIES must name capabilities from {SUPPORTED_CAPABILITIES}; "
        "got 'telepathy'"
    )


def test_an_unknown_image_capability_is_rejected():
    env = {
        "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_FORMAT": "openai",
        "IMAGE_GEN_CAPABILITIES": "editing,animation",
    }

    assert message(env) == (
        f"IMAGE_GEN_CAPABILITIES must name capabilities from {SUPPORTED_CAPABILITIES}; "
        "got 'animation'"
    )


def test_image_editing_must_be_declared_explicitly():
    env = {
        "IMAGE_GEN_ENDPOINT": "https://images.example/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_FORMAT": "openai",
        "IMAGE_GEN_CAPABILITIES": "generation,editing",
    }

    (route,) = parse(env)

    assert route.capabilities == frozenset({Capability.GENERATION, Capability.EDITING})


def test_an_unknown_capability_in_an_entry_is_rejected():
    env = {**block(), **routes_json(entry(capabilities=["chat", "telepathy"]))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'capabilities' must name capabilities from "
        f"{SUPPORTED_CAPABILITIES}; got 'telepathy'"
    )


# ---------------------------------------------------------------------------
# Extra headers
# ---------------------------------------------------------------------------


def test_non_authentication_extra_headers_are_accepted():
    (route,) = parse(
        block(
            TEXT_LLM_EXTRA_HEADERS_JSON=json.dumps(
                {"x-organization": "acme", "anthropic-beta": "prompt-caching-2024-07-31"}
            )
        )
    )

    assert route.extra_headers == {
        "x-organization": "acme",
        "anthropic-beta": "prompt-caching-2024-07-31",
    }


def test_blank_extra_headers_are_absent():
    (route,) = parse(block(TEXT_LLM_EXTRA_HEADERS_JSON="  "))

    assert route.extra_headers == {}


@pytest.mark.parametrize("header", ["Authorization", "authorization", "x-api-key", "API-KEY"])
def test_reserved_authentication_headers_are_rejected(header):
    env = block(TEXT_LLM_EXTRA_HEADERS_JSON=json.dumps({header: "secret"}))

    assert message(env) == (
        f"TEXT_LLM_EXTRA_HEADERS_JSON must not set the reserved authentication header "
        f"{header!r}; the route's own authentication configuration owns that header"
    )


def test_a_reserved_header_in_an_entry_is_rejected():
    env = {**block(), **routes_json(entry(extra_headers={"x-api-key": "secret"}))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'extra_headers' must not set the reserved authentication "
        "header 'x-api-key'; the route's own authentication configuration owns that header"
    )


def test_malformed_extra_headers_json_is_rejected():
    assert message(block(TEXT_LLM_EXTRA_HEADERS_JSON="{")).startswith(
        "TEXT_LLM_EXTRA_HEADERS_JSON must be a JSON object of string values"
    )


def test_extra_headers_must_be_a_json_object():
    assert message(block(TEXT_LLM_EXTRA_HEADERS_JSON='["x-organization"]')) == (
        "TEXT_LLM_EXTRA_HEADERS_JSON must be a JSON object of string values, got an array"
    )


def test_extra_header_values_must_be_strings():
    assert message(block(TEXT_LLM_EXTRA_HEADERS_JSON='{"x-retries": 3}')) == (
        "TEXT_LLM_EXTRA_HEADERS_JSON header 'x-retries' must have a string value, got a number"
    )


def test_entry_extra_headers_must_be_a_json_object():
    env = {**block(), **routes_json(entry(extra_headers=["x-organization"]))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'extra_headers' must be a JSON object of string values, "
        "got an array"
    )


def test_entry_extra_header_values_must_be_strings():
    env = {**block(), **routes_json(entry(extra_headers={"x-retries": None}))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'extra_headers' header 'x-retries' must have a string "
        "value, got null"
    )


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def test_a_relative_path_is_appended_to_the_base_path():
    (route,) = parse(block(TEXT_LLM_PATH="/custom/generate"))

    assert route.endpoint == "https://primary.example/v1"
    assert route.base_url == "https://primary.example/v1/custom/generate"


def test_repeated_path_components_are_preserved_literally():
    (route,) = parse(block(TEXT_LLM_PATH="/v1/custom"))

    assert route.base_url == "https://primary.example/v1/v1/custom"


def test_an_absolute_path_url_is_rejected_without_the_route_opt_in():
    assert message(block(TEXT_LLM_PATH="https://override.example/v1")) == (
        "TEXT_LLM_PATH: path setting is an absolute URL; the route's absolute-path-URL opt-in "
        "is required"
    )


def test_an_absolute_path_url_is_rejected_without_the_route_opt_in_in_an_entry():
    env = {
        **block(),
        **routes_json(entry(path="https://override.example/v1")),
    }

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'path': path setting is an absolute URL; the route's "
        "absolute-path-URL opt-in is required"
    )


@pytest.mark.parametrize("opt_in", ["true", "1", "yes", "on", "TRUE"])
def test_the_absolute_path_url_opt_in_is_honoured(opt_in):
    (route,) = parse(
        block(
            TEXT_LLM_PATH="https://override.example/v1",
            TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL=opt_in,
        )
    )

    assert route.allow_absolute_path_url is True
    assert route.endpoint == "https://primary.example/v1"
    assert route.base_url == "https://override.example/v1"


@pytest.mark.parametrize("opt_in", ["false", "0", "no", "off"])
def test_the_opt_in_is_off_by_default(opt_in):
    (route,) = parse(block(TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL=opt_in))

    assert route.allow_absolute_path_url is False


def test_a_non_boolean_opt_in_is_rejected():
    assert message(block(TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL="perhaps")) == (
        "TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL must be a boolean, got 'perhaps'; expected one of "
        "'0', '1', 'false', 'no', 'off', 'on', 'true', 'yes'"
    )


def test_an_entry_opt_in_enables_its_own_absolute_path_url():
    env = {
        **block(),
        **routes_json(
            entry(path="https://override.example/v1", allow_absolute_path_url=True),
        ),
    }

    routes = parse(env)

    assert routes[1].allow_absolute_path_url is True
    assert routes[1].base_url == "https://override.example/v1"


def test_an_entry_opt_in_must_be_a_boolean():
    env = {**block(), **routes_json(entry(allow_absolute_path_url="true"))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'allow_absolute_path_url' must be a boolean, got a string"
    )


def test_an_endpoint_that_resolves_to_a_blocked_address_is_rejected():
    env = block(TEXT_LLM_ENDPOINT="https://metadata.example/latest")

    assert message(env, resolver_for(PUBLIC_HOSTS)) == (
        "TEXT_LLM_ENDPOINT: host 'metadata.example' resolves to a blocked address: "
        "169.254.169.254 (link-local)"
    )


def test_an_entry_endpoint_that_resolves_to_a_blocked_address_is_rejected():
    env = {**block(), **routes_json(entry(endpoint="https://metadata.example/latest"))}

    assert message(env, resolver_for(PUBLIC_HOSTS)) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'endpoint': host 'metadata.example' resolves to a blocked "
        "address: 169.254.169.254 (link-local)"
    )


def test_plain_http_is_rejected():
    assert message(block(TEXT_LLM_ENDPOINT="http://primary.example/v1")) == (
        "TEXT_LLM_ENDPOINT: plain http is only allowed when insecure transport is explicitly "
        "enabled"
    )


def test_an_endpoint_with_embedded_credentials_is_rejected():
    assert message(block(TEXT_LLM_ENDPOINT="https://user:secret@primary.example/v1")) == (
        "TEXT_LLM_ENDPOINT: endpoint must not embed credentials; supply them through route "
        "configuration"
    )


def test_an_endpoint_without_a_scheme_is_rejected():
    assert message(block(TEXT_LLM_ENDPOINT="primary.example/v1")) == (
        "TEXT_LLM_ENDPOINT: endpoint must be an absolute URL with a scheme and host"
    )


# ---------------------------------------------------------------------------
# ROUTES_JSON without a singleton block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "variable"),
    [
        ("TEXT_LLM", "TEXT_LLM_ROUTES_JSON"),
        ("IMAGE_GEN", "IMAGE_GEN_ROUTES_JSON"),
        ("EMBEDDINGS", "EMBEDDINGS_ROUTES_JSON"),
    ],
)
def test_routes_json_without_the_singleton_block_is_rejected(prefix, variable):
    env = {variable: json.dumps([entry()])}

    assert message(env) == (
        f"{variable} is set but the {prefix}_* block is unset; a capability is enabled only when "
        f"its singleton block is configured, so define {prefix}_ENDPOINT, {prefix}_MODEL_ID and "
        f"{prefix}_FORMAT, or remove {variable}"
    )


@pytest.mark.parametrize("value", ["[]", '[{"endpoint": "https://x.example"}]', "not json"])
def test_any_routes_json_without_the_singleton_block_is_rejected(value):
    assert message({"TEXT_LLM_ROUTES_JSON": value}) == (
        "TEXT_LLM_ROUTES_JSON is set but the TEXT_LLM_* block is unset; a capability is enabled "
        "only when its singleton block is configured, so define TEXT_LLM_ENDPOINT, "
        "TEXT_LLM_MODEL_ID and TEXT_LLM_FORMAT, or remove TEXT_LLM_ROUTES_JSON"
    )


def test_routes_json_without_the_singleton_block_is_rejected_before_other_errors():
    env = {"TEXT_LLM_ROUTES_JSON": "not json", "TEXT_LLM_ENDPOINT": ""}

    assert message(env).startswith("TEXT_LLM_ROUTES_JSON is set but the TEXT_LLM_* block is unset")


# ---------------------------------------------------------------------------
# Malformed ROUTES_JSON
# ---------------------------------------------------------------------------


def test_malformed_routes_json_is_rejected():
    assert message({**block(), "TEXT_LLM_ROUTES_JSON": "["}).startswith(
        "TEXT_LLM_ROUTES_JSON must be a JSON array of route objects"
    )


def test_routes_json_must_be_an_array():
    assert message({**block(), "TEXT_LLM_ROUTES_JSON": '{"endpoint": "https://x.example"}'}) == (
        "TEXT_LLM_ROUTES_JSON must be a JSON array of route objects, got an object"
    )


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ('"https://fallback.example/v1"', "a string"),
        ("3", "a number"),
        ("null", "null"),
        ("true", "a boolean"),
    ],
)
def test_a_routes_json_entry_must_be_an_object(value, kind):
    assert message({**block(), "TEXT_LLM_ROUTES_JSON": f"[{value}]"}) == (
        f"TEXT_LLM_ROUTES_JSON entry 1 must be a JSON object, got {kind}"
    )


def test_the_second_entry_is_reported_by_its_index():
    env = {**block(), **routes_json(entry(), "not an object")}

    assert message(env) == "TEXT_LLM_ROUTES_JSON entry 2 must be a JSON object, got a string"


@pytest.mark.parametrize("key", ["endpoint", "model_id", "format"])
def test_an_entry_must_specify_every_required_field(key):
    incomplete = entry()
    del incomplete[key]

    assert message({**block(), **routes_json(incomplete)}) == (
        f"TEXT_LLM_ROUTES_JSON entry 1 must specify a non-empty {key!r}"
    )


@pytest.mark.parametrize("key", ["endpoint", "model_id", "format"])
def test_an_entry_required_field_must_be_a_non_empty_string(key):
    env = {**block(), **routes_json(entry(**{key: "   "}))}

    assert message(env) == f"TEXT_LLM_ROUTES_JSON entry 1 must specify a non-empty {key!r}"


def test_an_entry_required_field_must_not_be_a_number():
    env = {**block(), **routes_json(entry(model_id=5))}

    assert message(env) == "TEXT_LLM_ROUTES_JSON entry 1 must specify a non-empty 'model_id'"


def test_an_entry_rejects_an_unsupported_key():
    env = {**block(), **routes_json(entry(size="1024x1024"))}

    assert message(env) == (
        f"TEXT_LLM_ROUTES_JSON entry 1 has unsupported key 'size'; supported keys are {ENTRY_KEYS}"
    )


def test_an_entry_optional_field_must_be_a_string():
    env = {**block(), **routes_json(entry(path=5))}

    assert message(env) == "TEXT_LLM_ROUTES_JSON entry 1 'path' must be a string, got a number"


def test_an_entry_capabilities_field_must_be_a_list_or_a_string():
    env = {**block(), **routes_json(entry(capabilities=7))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'capabilities' must be an array of capability names or a "
        "comma-separated string, got a number"
    )


def test_an_entry_capabilities_array_must_hold_strings():
    env = {**block(), **routes_json(entry(capabilities=["chat", 7]))}

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 'capabilities' must be an array of capability names, got a "
        "number in the array"
    )


def test_an_entry_capabilities_string_is_split_on_commas():
    env = {**block(), **routes_json(entry(capabilities="tools, vision"))}

    assert parse(env)[1].capabilities == frozenset({Capability.TOOLS, Capability.VISION})


# ---------------------------------------------------------------------------
# Ambiguity between routes
# ---------------------------------------------------------------------------


def test_an_entry_may_not_repeat_the_primary_route():
    env = {
        **block(),
        **routes_json(entry(endpoint="https://primary.example/v1", model_id="primary-model")),
    }

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 1 names the same endpoint and model ID as the TEXT_LLM_* "
        "block; every route of the TEXT_LLM operation must name a distinct endpoint and model"
    )


def test_two_entries_may_not_name_the_same_route():
    env = {
        **block(),
        **routes_json(entry(), entry(endpoint="https://fallback.example/v1/")),
    }

    assert message(env) == (
        "TEXT_LLM_ROUTES_JSON entry 2 names the same endpoint and model ID as "
        "TEXT_LLM_ROUTES_JSON entry 1; every route of the TEXT_LLM operation must name a "
        "distinct endpoint and model"
    )


def test_the_same_endpoint_with_a_different_model_is_allowed():
    env = {**block(), **routes_json(entry(model_id="other-model"))}

    assert len(parse(env)) == 2


def test_the_same_model_on_a_different_endpoint_is_allowed():
    env = {**block(), **routes_json(entry(endpoint="https://fallback.example/v2"))}

    assert len(parse(env)) == 2


def test_the_same_endpoint_and_model_may_serve_two_operations():
    env = {
        **block(),
        "EMBEDDINGS_ENDPOINT": "https://primary.example/v1",
        "EMBEDDINGS_MODEL_ID": "primary-model",
        "EMBEDDINGS_FORMAT": "openai",
    }

    assert len(parse(env)) == 2


# ---------------------------------------------------------------------------
# The indexed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "prefix"),
    [
        ("TEXT_LLM_1_ENDPOINT", "TEXT_LLM"),
        ("TEXT_LLM_2_MODEL_ID", "TEXT_LLM"),
        ("TEXT_LLM_10_FORMAT", "TEXT_LLM"),
        ("IMAGE_GEN_1_ENDPOINT", "IMAGE_GEN"),
        ("EMBEDDINGS_2_ENDPOINT", "EMBEDDINGS"),
    ],
)
def test_the_indexed_route_form_is_rejected(variable, prefix):
    assert message({variable: "https://indexed.example/v1"}) == (
        f"{variable}: the indexed route form ({prefix}_1_*, {prefix}_2_*, ...) is not supported; "
        f"define the primary route in the {prefix}_* block and additional routes in "
        f"{prefix}_ROUTES_JSON"
    )


def test_the_indexed_form_is_rejected_even_when_the_block_is_configured():
    env = {**block(), "TEXT_LLM_1_ENDPOINT": "https://indexed.example/v1"}

    assert message(env).startswith("TEXT_LLM_1_ENDPOINT: the indexed route form")


def test_the_indexed_form_is_rejected_before_the_singleton_block_is_read():
    env = {"TEXT_LLM_1_ENDPOINT": "https://indexed.example/v1", "TEXT_LLM_ENDPOINT": ""}

    assert message(env).startswith("TEXT_LLM_1_ENDPOINT: the indexed route form")


# ---------------------------------------------------------------------------
# Environment source
# ---------------------------------------------------------------------------


def test_parse_routes_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENDPOINT", f"https://{PUBLIC_ADDRESS}/v1")
    monkeypatch.setenv("TEXT_LLM_MODEL_ID", "primary-model")
    monkeypatch.setenv("TEXT_LLM_FORMAT", "openai")
    monkeypatch.delenv("TEXT_LLM_ROUTES_JSON", raising=False)

    routes = parse_routes()

    assert len(routes) == 1
    assert routes[0].base_url == f"https://{PUBLIC_ADDRESS}/v1"


def test_parse_routes_ignores_the_process_environment_when_given_a_mapping(monkeypatch):
    monkeypatch.setenv("TEXT_LLM_ENDPOINT", f"https://{PUBLIC_ADDRESS}/v1")
    monkeypatch.setenv("TEXT_LLM_MODEL_ID", "primary-model")
    monkeypatch.setenv("TEXT_LLM_FORMAT", "openai")

    assert parse_routes({}) == ()


def test_routes_are_returned_in_a_stable_order():
    env = {
        "EMBEDDINGS_ENDPOINT": "https://embeddings.example/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_FORMAT": "openai",
        **block(),
        **routes_json(entry(), entry(endpoint="https://fallback-2.example/v1")),
    }

    assert parse(env) == parse(dict(reversed(list(env.items()))))


# ---------------------------------------------------------------------------
# The PRD's own examples
# ---------------------------------------------------------------------------


def test_the_prd_fallback_example_is_accepted():
    routes = parse(
        {
            "TEXT_LLM_ENDPOINT": "https://primary.example/v1",
            "TEXT_LLM_MODEL_ID": "primary-model",
            "TEXT_LLM_API_KEY": "",
            "TEXT_LLM_FORMAT": "openai",
            "TEXT_LLM_ROUTES_JSON": (
                '[{"endpoint":"https://fallback.example/v1","model_id":"fallback-model",'
                '"api_key":"","format":"openai"}]'
            ),
        }
    )

    assert [(route.index, route.endpoint, route.litellm_model) for route in routes] == [
        (0, "https://primary.example/v1", "openai/primary-model"),
        (1, "https://fallback.example/v1", "openai/fallback-model"),
    ]


def test_the_prd_complete_environment_example_is_accepted():
    env = {
        "CHORA_HTTP_PORT": "8080",
        "CHORA_ENVIRONMENT": "production",
        "TEXT_LLM_ENDPOINT": "https://api.example.com/v1",
        "TEXT_LLM_MODEL_ID": "text-model",
        "TEXT_LLM_API_KEY": "",
        "TEXT_LLM_FORMAT": "openai",
        "TEXT_LLM_AUTH_TYPE": "bearer",
        "TEXT_LLM_PATH": "",
        "TEXT_LLM_EXTRA_HEADERS_JSON": "{}",
        "TEXT_LLM_CAPABILITIES": "chat,tools,vision",
        "TEXT_LLM_CONTEXT_WINDOW": "",
        "TEXT_LLM_MAX_OUTPUT_TOKENS": "",
        "TEXT_LLM_ROUTES_JSON": "[]",
        "IMAGE_GEN_ENDPOINT": "https://images.example.com/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_API_KEY": "",
        "IMAGE_GEN_FORMAT": "openai",
        "IMAGE_GEN_AUTH_TYPE": "bearer",
        "IMAGE_GEN_PATH": "",
        "IMAGE_GEN_EXTRA_HEADERS_JSON": "{}",
        "IMAGE_GEN_CAPABILITIES": "generation",
        "IMAGE_GEN_ROUTES_JSON": "[]",
        "EMBEDDINGS_ENDPOINT": "https://embeddings.example.com/v1",
        "EMBEDDINGS_MODEL_ID": "embedding-model",
        "EMBEDDINGS_API_KEY": "",
        "EMBEDDINGS_FORMAT": "openai",
        "EMBEDDINGS_AUTH_TYPE": "bearer",
        "EMBEDDINGS_PATH": "",
        "EMBEDDINGS_DIMENSIONS": "1024",
        "EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM": "false",
        "EMBEDDINGS_ROUTES_JSON": "[]",
    }

    routes = parse(env)

    assert [route.operation for route in routes] == [
        Operation.TEXT,
        Operation.IMAGE,
        Operation.EMBEDDINGS,
    ]
    assert [route.index for route in routes] == [0, 0, 0]
    assert [route.base_url for route in routes] == [
        "https://api.example.com/v1",
        "https://images.example.com/v1",
        "https://embeddings.example.com/v1",
    ]
    assert routes[0].capabilities == frozenset(
        {Capability.CHAT, Capability.TOOLS, Capability.VISION}
    )
    assert routes[1].capabilities == frozenset({Capability.GENERATION})
    assert routes[2].capabilities == frozenset({Capability.EMBEDDINGS})
