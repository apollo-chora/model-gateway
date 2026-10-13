"""Settings loading and startup validation (PRD sections 6, 7, 18)."""

import dataclasses
import importlib

import pytest

from model_gateway.config import ConfigurationError, Settings, UsageStorage

BOOLEAN_VARIABLES = [
    "GATEWAY_AUTH_ENABLED",
    "GATEWAY_ALLOW_ANONYMOUS_HEALTH",
    "USAGE_TRACKING_ENABLED",
    "COST_TRACKING_ENABLED",
    "OTEL_ENABLED",
]

POSITIVE_NUMERIC_VARIABLES = [
    "CHORA_HTTP_PORT",
    "CHORA_GRPC_PORT",
    "PROVIDER_TIMEOUT_SECONDS",
    "PROVIDER_CONNECT_TIMEOUT_SECONDS",
    "PROVIDER_RETRY_BACKOFF_MS",
    "PROVIDER_MAX_CONCURRENT_REQUESTS",
    "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS",
]


def test_settings_is_importable_from_the_config_package():
    config = importlib.import_module("model_gateway.config")
    assert config.Settings is Settings
    assert Settings.__module__ == "model_gateway.config.settings"


def test_defaults_match_the_prd_example_environment():
    settings = Settings.from_env({})

    assert settings.environment == "development"
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

    assert settings.environment == "production"
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


def test_log_level_is_normalised_to_upper_case():
    assert Settings.from_env({"LOG_LEVEL": "warning"}).log_level == "WARNING"


def test_environment_must_not_be_empty():
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"CHORA_ENVIRONMENT": "  "})

    assert str(error.value) == "CHORA_ENVIRONMENT must not be empty"


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


@pytest.mark.parametrize("value", ["abc", "2.5", "", "10s"])
def test_an_integer_variable_rejects_a_malformed_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_MAX_CONCURRENT_REQUESTS": value})

    assert str(error.value) == (
        f"PROVIDER_MAX_CONCURRENT_REQUESTS must be a positive integer, got {value!r}"
    )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_an_integer_variable_rejects_a_non_positive_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_MAX_CONCURRENT_REQUESTS": value})

    assert str(error.value) == (
        f"PROVIDER_MAX_CONCURRENT_REQUESTS must be a positive integer, got {value!r}"
    )


@pytest.mark.parametrize("value", ["0", "1", "5"])
def test_provider_retry_count_accepts_zero_and_above(value):
    settings = Settings.from_env({"PROVIDER_RETRY_COUNT": value})

    assert settings.provider_retry_count == int(value)


@pytest.mark.parametrize("value", ["-1", "-100"])
def test_provider_retry_count_rejects_a_negative_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_RETRY_COUNT": value})

    assert str(error.value) == f"PROVIDER_RETRY_COUNT must be a non-negative integer, got {value!r}"


@pytest.mark.parametrize("value", ["abc", "2.5", "", "3 retries"])
def test_provider_retry_count_rejects_a_malformed_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_RETRY_COUNT": value})

    assert str(error.value) == f"PROVIDER_RETRY_COUNT must be a non-negative integer, got {value!r}"


@pytest.mark.parametrize("value", ["soon", "", "12,5"])
def test_a_decimal_variable_rejects_a_malformed_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_TIMEOUT_SECONDS": value})

    assert str(error.value) == f"PROVIDER_TIMEOUT_SECONDS must be a positive number, got {value!r}"


@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_a_decimal_variable_rejects_a_non_positive_value(value):
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({"PROVIDER_TIMEOUT_SECONDS": value})

    assert str(error.value) == f"PROVIDER_TIMEOUT_SECONDS must be a positive number, got {value!r}"


@pytest.mark.parametrize("name", POSITIVE_NUMERIC_VARIABLES)
def test_every_strictly_positive_numeric_variable_rejects_zero(name):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({name: "0"})


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
