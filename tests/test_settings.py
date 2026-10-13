"""Settings loading and startup validation (PRD sections 6, 7, 18)."""

import dataclasses
import importlib

import pytest

from model_gateway.config import ConfigurationError, Environment, Settings, UsageStorage

BOOLEAN_VARIABLES = [
    "GATEWAY_AUTH_ENABLED",
    "GATEWAY_ALLOW_ANONYMOUS_HEALTH",
    "USAGE_TRACKING_ENABLED",
    "COST_TRACKING_ENABLED",
    "OTEL_ENABLED",
]

PORT_VARIABLES = {
    "CHORA_HTTP_PORT": "http_port",
    "CHORA_GRPC_PORT": "grpc_port",
}

DECIMAL_VARIABLES = [
    "PROVIDER_TIMEOUT_SECONDS",
    "PROVIDER_CONNECT_TIMEOUT_SECONDS",
    "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS",
]

NON_NEGATIVE_INTEGER_VARIABLES = {
    "PROVIDER_RETRY_COUNT": "provider_retry_count",
    "PROVIDER_RETRY_BACKOFF_MS": "provider_retry_backoff_ms",
}

STRICTLY_POSITIVE_VARIABLES = [
    "PROVIDER_MAX_CONCURRENT_REQUESTS",
    *DECIMAL_VARIABLES,
]


def test_settings_is_importable_from_the_config_package():
    config = importlib.import_module("model_gateway.config")
    assert config.Settings is Settings
    assert Settings.__module__ == "model_gateway.config.settings"


def test_defaults_match_the_prd_example_environment():
    settings = Settings.from_env({})

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.http_port == 8080
    assert settings.grpc_port == 9090
    assert settings.log_level == "INFO"
    assert settings.auth_enabled is False
    assert settings.api_keys == ()
    assert settings.allow_anonymous_health is True
    assert settings.provider_timeout_seconds == 110.0
    assert settings.provider_connect_timeout_seconds == 10.0
    assert settings.provider_retry_count == 2
    assert settings.provider_retry_backoff_ms == 500
    assert settings.provider_max_concurrent_requests == 100
    assert settings.shutdown_timeout_seconds == 30.0
    assert settings.usage_tracking_enabled is True
    assert settings.cost_tracking_enabled is False
    assert settings.usage_storage is UsageStorage.MEMORY
    assert settings.otel_enabled is False
    assert settings.otel_exporter_otlp_endpoint == ""


def test_from_env_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv("CHORA_HTTP_PORT", "9000")
    monkeypatch.setenv("GATEWAY_AUTH_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_API_KEYS", "key-a,key-b")

    settings = Settings.from_env()

    assert settings.http_port == 9000
    assert settings.auth_enabled is True
    assert settings.api_keys == ("key-a", "key-b")


def test_every_deployment_variable_is_loaded():
    settings = Settings.from_env(
        {
            "CHORA_HTTP_PORT": "9080",
            "CHORA_GRPC_PORT": "9091",
            "CHORA_ENVIRONMENT": "production",
            "LOG_LEVEL": "debug",
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": "alpha, beta",
            "GATEWAY_ALLOW_ANONYMOUS_HEALTH": "false",
            "PROVIDER_TIMEOUT_SECONDS": "12.5",
            "PROVIDER_CONNECT_TIMEOUT_SECONDS": "3",
            "PROVIDER_RETRY_COUNT": "5",
            "PROVIDER_RETRY_BACKOFF_MS": "250",
            "PROVIDER_MAX_CONCURRENT_REQUESTS": "7",
            "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS": "45",
            "USAGE_TRACKING_ENABLED": "false",
            "COST_TRACKING_ENABLED": "true",
            "USAGE_STORAGE": "sqlite",
            "OTEL_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.http_port == 9080
    assert settings.grpc_port == 9091
    assert settings.log_level == "DEBUG"
    assert settings.auth_enabled is True
    assert settings.api_keys == ("alpha", "beta")
    assert settings.allow_anonymous_health is False
    assert settings.provider_timeout_seconds == 12.5
    assert settings.provider_connect_timeout_seconds == 3.0
    assert settings.provider_retry_count == 5
    assert settings.provider_retry_backoff_ms == 250
    assert settings.provider_max_concurrent_requests == 7
    assert settings.shutdown_timeout_seconds == 45.0
    assert settings.usage_tracking_enabled is False
    assert settings.cost_tracking_enabled is True
    assert settings.usage_storage is UsageStorage.SQLITE
    assert settings.otel_enabled is True
    assert settings.otel_exporter_otlp_endpoint == "http://collector:4317"


@pytest.mark.parametrize("value", ["debug", "Debug", "INFO", "warning", "Error", "critical"])
def test_log_level_accepts_supported_levels_case_insensitively(value):
    settings = Settings.from_env({"LOG_LEVEL": value})

    assert settings.log_level == value.upper()


@pytest.mark.parametrize("value", ["potato", "verbose", "warn", "10", "info!"])
def test_log_level_rejects_an_unknown_level(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"LOG_LEVEL": value})

    assert str(error.value) == (
        "LOG_LEVEL must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', "
        f"got {value!r}"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("development", Environment.DEVELOPMENT),
        ("Development", Environment.DEVELOPMENT),
        ("DEVELOPMENT", Environment.DEVELOPMENT),
        ("test", Environment.TEST),
        ("TEST", Environment.TEST),
        ("production", Environment.PRODUCTION),
        ("Production", Environment.PRODUCTION),
        ("PRODUCTION", Environment.PRODUCTION),
    ],
)
def test_environment_accepts_supported_values_case_insensitively(value, expected):
    settings = Settings.from_env(
        {
            "CHORA_ENVIRONMENT": value,
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": "key-a",
        }
    )

    assert settings.environment is expected


@pytest.mark.parametrize("value", ["prodution", "PROD", "banana", "staging", "production-1"])
def test_environment_rejects_unknown_values(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"CHORA_ENVIRONMENT": value})

    assert str(error.value) == (
        f"CHORA_ENVIRONMENT must be one of 'development', 'test', 'production', got {value!r}"
    )


def test_environment_must_not_be_empty():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"CHORA_ENVIRONMENT": "  "})

    assert str(error.value) == "CHORA_ENVIRONMENT must not be empty"


def test_environment_is_validated_on_direct_construction():
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError, match="CHORA_ENVIRONMENT must be one of"):
        dataclasses.replace(settings, environment="banana")

    production = dataclasses.replace(
        settings,
        environment="PRODUCTION",
        auth_enabled=True,
        api_keys=("key-a",),
    )

    assert production.environment is Environment.PRODUCTION


@pytest.mark.parametrize("value", ["development", "test", "DEVELOPMENT", "Test"])
def test_non_production_environments_allow_authentication_to_be_disabled(value):
    settings = Settings.from_env({"CHORA_ENVIRONMENT": value, "GATEWAY_AUTH_ENABLED": "false"})

    assert settings.auth_enabled is False


def test_production_requires_authentication():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"CHORA_ENVIRONMENT": "production"})

    assert str(error.value) == (
        "CHORA_ENVIRONMENT is production but GATEWAY_AUTH_ENABLED is false; "
        "production must not serve unauthenticated inference. Set "
        "GATEWAY_AUTH_ENABLED=true with GATEWAY_API_KEYS, or set "
        "CHORA_ENVIRONMENT=development"
    )


def test_production_requires_authentication_even_when_it_is_disabled_explicitly():
    with pytest.raises(ConfigurationError, match="CHORA_ENVIRONMENT is production"):
        Settings.from_env(
            {"CHORA_ENVIRONMENT": "production", "GATEWAY_AUTH_ENABLED": "false"}
        )


def test_the_production_invariant_is_enforced_on_direct_construction():
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError, match="CHORA_ENVIRONMENT is production"):
        dataclasses.replace(settings, environment="production")


def test_production_with_authentication_is_accepted():
    settings = Settings.from_env(
        {
            "CHORA_ENVIRONMENT": "production",
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": "key-a",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.auth_enabled is True


def test_production_with_authentication_but_no_keys_reports_the_missing_keys():
    with pytest.raises(ConfigurationError, match="GATEWAY_API_KEYS is empty"):
        Settings.from_env({"CHORA_ENVIRONMENT": "production", "GATEWAY_AUTH_ENABLED": "true"})


@pytest.mark.parametrize("name", list(PORT_VARIABLES))
@pytest.mark.parametrize("value", ["1", "8080", "65535"])
def test_port_variables_accept_valid_ports(name, value):
    settings = Settings.from_env({name: value})

    assert getattr(settings, PORT_VARIABLES[name]) == int(value)


@pytest.mark.parametrize("name", list(PORT_VARIABLES))
@pytest.mark.parametrize("value", ["0", "-1", "65536", "999999"])
def test_port_variables_reject_out_of_range_ports(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == (
        f"{name} must be a port number between 1 and 65535, got {value!r}"
    )


@pytest.mark.parametrize("name", list(PORT_VARIABLES))
@pytest.mark.parametrize("value", ["abc", "80.5", "", "http"])
def test_port_variables_reject_malformed_values(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == (
        f"{name} must be a port number between 1 and 65535, got {value!r}"
    )


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on", " true "])
def test_boolean_variables_accept_recognised_true_values(value):
    settings = Settings.from_env({"USAGE_TRACKING_ENABLED": value})

    assert settings.usage_tracking_enabled is True


@pytest.mark.parametrize("value", ["false", "FALSE", "False", "0", "no", "off", " off "])
def test_boolean_variables_accept_recognised_false_values(value):
    settings = Settings.from_env({"COST_TRACKING_ENABLED": value})

    assert settings.cost_tracking_enabled is False


@pytest.mark.parametrize("value", ["maybe", "", "2", "truthy"])
def test_boolean_variables_reject_unrecognised_values(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"COST_TRACKING_ENABLED": value})

    assert str(error.value) == (
        f"COST_TRACKING_ENABLED must be a boolean, got {value!r}; expected one of "
        "'0', '1', 'false', 'no', 'off', 'on', 'true', 'yes'"
    )


@pytest.mark.parametrize("name", BOOLEAN_VARIABLES)
def test_every_boolean_variable_is_validated(name):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({name: "perhaps"})


def test_numeric_variables_accept_integers_and_decimals():
    settings = Settings.from_env(
        {
            "PROVIDER_TIMEOUT_SECONDS": "2.5",
            "PROVIDER_RETRY_COUNT": "3",
        }
    )

    assert settings.provider_timeout_seconds == 2.5
    assert settings.provider_retry_count == 3


@pytest.mark.parametrize("name", STRICTLY_POSITIVE_VARIABLES)
def test_strictly_positive_variables_reject_zero(name):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({name: "0"})


@pytest.mark.parametrize("value", ["abc", "2.5", "", "10s"])
def test_an_integer_variable_rejects_a_malformed_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_MAX_CONCURRENT_REQUESTS": value})

    assert str(error.value) == (
        f"PROVIDER_MAX_CONCURRENT_REQUESTS must be a positive integer, got {value!r}"
    )


def test_an_integer_variable_rejects_a_negative_value():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_MAX_CONCURRENT_REQUESTS": "-1"})

    assert str(error.value) == (
        "PROVIDER_MAX_CONCURRENT_REQUESTS must be a positive integer, got '-1'"
    )


@pytest.mark.parametrize("name", list(NON_NEGATIVE_INTEGER_VARIABLES))
@pytest.mark.parametrize("value", ["0", "1", "5"])
def test_non_negative_integer_variables_accept_zero_and_above(name, value):
    settings = Settings.from_env({name: value})

    assert getattr(settings, NON_NEGATIVE_INTEGER_VARIABLES[name]) == int(value)


@pytest.mark.parametrize("name", list(NON_NEGATIVE_INTEGER_VARIABLES))
@pytest.mark.parametrize("value", ["-1", "-100"])
def test_non_negative_integer_variables_reject_negative_values(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == f"{name} must be a non-negative integer, got {value!r}"


@pytest.mark.parametrize("name", list(NON_NEGATIVE_INTEGER_VARIABLES))
@pytest.mark.parametrize("value", ["abc", "2.5", "", "3 retries"])
def test_non_negative_integer_variables_reject_malformed_values(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == f"{name} must be a non-negative integer, got {value!r}"


@pytest.mark.parametrize("name", DECIMAL_VARIABLES)
@pytest.mark.parametrize("value", ["soon", "", "12,5"])
def test_a_decimal_variable_rejects_a_malformed_value(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == f"{name} must be a positive number, got {value!r}"


@pytest.mark.parametrize("name", DECIMAL_VARIABLES)
@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_a_decimal_variable_rejects_a_non_positive_value(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == f"{name} must be a positive number, got {value!r}"


@pytest.mark.parametrize("name", DECIMAL_VARIABLES)
@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "Infinity", "1e999"])
def test_a_decimal_variable_rejects_a_non_finite_value(name, value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({name: value})

    assert str(error.value) == f"{name} must be a finite number, got {value!r}"


def test_a_decimal_variable_accepts_a_finite_value():
    settings = Settings.from_env(
        {"PROVIDER_TIMEOUT_SECONDS": "0.5", "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS": "1e3"}
    )

    assert settings.provider_timeout_seconds == 0.5
    assert settings.shutdown_timeout_seconds == 1000.0


def test_api_keys_are_split_on_commas_and_trimmed():
    settings = Settings.from_env({"GATEWAY_API_KEYS": " key-a ,, key-b , "})

    assert settings.api_keys == ("key-a", "key-b")


def test_auth_enabled_without_keys_fails():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"GATEWAY_AUTH_ENABLED": "true"})

    assert str(error.value) == (
        "GATEWAY_AUTH_ENABLED is true but GATEWAY_API_KEYS is empty; set one or more "
        "comma-separated keys, or set GATEWAY_AUTH_ENABLED=false"
    )


@pytest.mark.parametrize("keys", ["", " , ", ","])
def test_auth_enabled_with_only_blank_keys_fails(keys):
    with pytest.raises(ConfigurationError, match="GATEWAY_API_KEYS is empty"):
        Settings.from_env({"GATEWAY_AUTH_ENABLED": "true", "GATEWAY_API_KEYS": keys})


def test_auth_enabled_with_keys_is_accepted():
    settings = Settings.from_env({"GATEWAY_AUTH_ENABLED": "true", "GATEWAY_API_KEYS": "key-a"})

    assert settings.auth_enabled is True
    assert settings.api_keys == ("key-a",)


def test_auth_disabled_without_keys_is_accepted():
    settings = Settings.from_env({"GATEWAY_AUTH_ENABLED": "false", "GATEWAY_API_KEYS": ""})

    assert settings.auth_enabled is False
    assert settings.api_keys == ()


@pytest.mark.parametrize("value", ["none", "memory", "sqlite", "postgres"])
def test_supported_usage_storage_modes_are_accepted(value):
    settings = Settings.from_env({"USAGE_STORAGE": value})

    assert settings.usage_storage is UsageStorage(value)


def test_unknown_usage_storage_fails():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"USAGE_STORAGE": "redis"})

    assert str(error.value) == (
        "USAGE_STORAGE must be one of 'none', 'memory', 'sqlite', 'postgres', got 'redis'"
    )


def test_usage_storage_is_validated_on_direct_construction():
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError, match="USAGE_STORAGE must be one of"):
        dataclasses.replace(settings, usage_storage="redis")

    sqlite = dataclasses.replace(settings, usage_storage="sqlite")

    assert sqlite.usage_storage is UsageStorage.SQLITE


def test_auth_is_validated_on_direct_construction():
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError, match="GATEWAY_AUTH_ENABLED is true"):
        dataclasses.replace(settings, auth_enabled=True)
