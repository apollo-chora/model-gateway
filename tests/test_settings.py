"""Settings loading and startup validation."""

import dataclasses
import importlib

import pytest

from model_gateway.config import (
    ConfigurationError,
    Environment,
    GovernanceMode,
    Settings,
    UsageStorage,
)

BOOLEAN_VARIABLES = [
    "GATEWAY_AUTH_ENABLED",
    "GATEWAY_ALLOW_ANONYMOUS_HEALTH",
    "USAGE_TRACKING_ENABLED",
    "COST_TRACKING_ENABLED",
    "OTEL_ENABLED",
    "CHORA_LLM_BUDGET_REQUIRED",
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

BOOLEAN_EXPECTATION = "expected one of '0', '1', 'false', 'no', 'off', 'on', 'true', 'yes'"

PRODUCTION_REQUIRES_AUTH = (
    "CHORA_ENVIRONMENT is production but GATEWAY_AUTH_ENABLED is false; "
    "production must not serve unauthenticated inference. Set "
    "GATEWAY_AUTH_ENABLED=true with GATEWAY_API_KEYS, or set "
    "CHORA_ENVIRONMENT=development"
)

AUTH_KEYS_REQUIRED = (
    "GATEWAY_AUTH_ENABLED is true but GATEWAY_API_KEYS is empty; set one or more "
    "comma-separated keys, or set GATEWAY_AUTH_ENABLED=false"
)


def _direct(**overrides) -> Settings:
    """Build settings by direct construction from valid defaults plus overrides."""
    fields = dataclasses.asdict(Settings.from_env({}))
    fields.update(overrides)
    return Settings(**fields)


def _replaced(**overrides) -> Settings:
    """Build settings by replacing fields on a valid instance."""
    return dataclasses.replace(Settings.from_env({}), **overrides)


INVALID_STATES = [
    pytest.param(
        {"http_port": 999999},
        "CHORA_HTTP_PORT must be a port number between 1 and 65535, got '999999'",
        id="http-port-above-range",
    ),
    pytest.param(
        {"http_port": 0},
        "CHORA_HTTP_PORT must be a port number between 1 and 65535, got '0'",
        id="http-port-zero",
    ),
    pytest.param(
        {"grpc_port": 65536},
        "CHORA_GRPC_PORT must be a port number between 1 and 65535, got '65536'",
        id="grpc-port-above-range",
    ),
    pytest.param(
        {"grpc_port": -1},
        "CHORA_GRPC_PORT must be a port number between 1 and 65535, got '-1'",
        id="grpc-port-negative",
    ),
    pytest.param(
        {"log_level": "POTATO"},
        "LOG_LEVEL must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', "
        "got 'POTATO'",
        id="log-level-unknown",
    ),
    pytest.param({"log_level": ""}, "LOG_LEVEL must not be empty", id="log-level-empty"),
    pytest.param(
        {"provider_timeout_seconds": float("inf")},
        "PROVIDER_TIMEOUT_SECONDS must be a finite number, got 'inf'",
        id="timeout-infinite",
    ),
    pytest.param(
        {"provider_timeout_seconds": float("nan")},
        "PROVIDER_TIMEOUT_SECONDS must be a finite number, got 'nan'",
        id="timeout-nan",
    ),
    pytest.param(
        {"provider_connect_timeout_seconds": 0},
        "PROVIDER_CONNECT_TIMEOUT_SECONDS must be a positive number, got '0'",
        id="connect-timeout-zero",
    ),
    pytest.param(
        {"shutdown_timeout_seconds": -0.5},
        "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS must be a positive number, got '-0.5'",
        id="shutdown-timeout-negative",
    ),
    pytest.param(
        {"provider_retry_count": -1},
        "PROVIDER_RETRY_COUNT must be a non-negative integer, got '-1'",
        id="retry-count-negative",
    ),
    pytest.param(
        {"provider_retry_count": 2.5},
        "PROVIDER_RETRY_COUNT must be a non-negative integer, got '2.5'",
        id="retry-count-not-an-integer",
    ),
    pytest.param(
        {"provider_retry_backoff_ms": -1},
        "PROVIDER_RETRY_BACKOFF_MS must be a non-negative integer, got '-1'",
        id="backoff-negative",
    ),
    pytest.param(
        {"provider_max_concurrent_requests": 0},
        "PROVIDER_MAX_CONCURRENT_REQUESTS must be a positive integer, got '0'",
        id="max-concurrent-zero",
    ),
    pytest.param(
        {"rate_limit_requests": -1},
        "GATEWAY_RATE_LIMIT_REQUESTS must be a non-negative integer, got '-1'",
        id="rate-limit-negative",
    ),
    pytest.param(
        {"rate_limit_window_seconds": 0},
        "GATEWAY_RATE_LIMIT_WINDOW_SECONDS must be a positive number, got '0'",
        id="rate-limit-window-zero",
    ),
    pytest.param(
        {"tenant_max_concurrent_requests": -1},
        "GATEWAY_TENANT_MAX_CONCURRENT_REQUESTS must be a non-negative integer, got '-1'",
        id="tenant-concurrency-negative",
    ),
    pytest.param(
        {"environment": "banana"},
        "CHORA_ENVIRONMENT must be one of 'development', 'test', 'production', got 'banana'",
        id="environment-unknown",
    ),
    pytest.param(
        {"environment": "production", "auth_enabled": False},
        PRODUCTION_REQUIRES_AUTH,
        id="production-without-authentication",
    ),
    pytest.param(
        {"usage_storage": "redis"},
        "USAGE_STORAGE must be one of 'none', 'memory', 'sqlite', 'postgres', got 'redis'",
        id="usage-storage-unknown",
    ),
    pytest.param(
        {"auth_enabled": "maybe"},
        f"GATEWAY_AUTH_ENABLED must be a boolean, got 'maybe'; {BOOLEAN_EXPECTATION}",
        id="auth-enabled-not-a-boolean",
    ),
    pytest.param(
        {"allow_anonymous_health": "maybe"},
        f"GATEWAY_ALLOW_ANONYMOUS_HEALTH must be a boolean, got 'maybe'; {BOOLEAN_EXPECTATION}",
        id="anonymous-health-not-a-boolean",
    ),
    pytest.param(
        {"usage_tracking_enabled": "maybe"},
        f"USAGE_TRACKING_ENABLED must be a boolean, got 'maybe'; {BOOLEAN_EXPECTATION}",
        id="usage-tracking-not-a-boolean",
    ),
    pytest.param(
        {"cost_tracking_enabled": "maybe"},
        f"COST_TRACKING_ENABLED must be a boolean, got 'maybe'; {BOOLEAN_EXPECTATION}",
        id="cost-tracking-not-a-boolean",
    ),
    pytest.param(
        {"otel_enabled": "perhaps"},
        f"OTEL_ENABLED must be a boolean, got 'perhaps'; {BOOLEAN_EXPECTATION}",
        id="otel-enabled-not-a-boolean",
    ),
    pytest.param(
        {"api_keys": 5},
        "GATEWAY_API_KEYS must be a comma-separated list of keys, got 5",
        id="api-keys-not-a-list",
    ),
    pytest.param(
        {"auth_enabled": True, "api_keys": " , "},
        AUTH_KEYS_REQUIRED,
        id="authentication-without-keys",
    ),
]


def test_settings_is_importable_from_the_config_package():
    config = importlib.import_module("model_gateway.config")
    assert config.Settings is Settings
    assert Settings.__module__ == "model_gateway.config.settings"


def test_defaults_match_the_example_environment():
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
    assert settings.rate_limit_requests == 0
    assert settings.rate_limit_window_seconds == 60.0
    assert settings.tenant_max_concurrent_requests == 0
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
            "GATEWAY_RATE_LIMIT_REQUESTS": "120",
            "GATEWAY_RATE_LIMIT_WINDOW_SECONDS": "30",
            "GATEWAY_TENANT_MAX_CONCURRENT_REQUESTS": "4",
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
    assert settings.rate_limit_requests == 120
    assert settings.rate_limit_window_seconds == 30.0
    assert settings.tenant_max_concurrent_requests == 4
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

    assert str(error.value) == PRODUCTION_REQUIRES_AUTH


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
        f"COST_TRACKING_ENABLED must be a boolean, got {value!r}; {BOOLEAN_EXPECTATION}"
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


@pytest.mark.parametrize("build", [_direct, _replaced], ids=["direct", "replace"])
@pytest.mark.parametrize(("overrides", "message"), INVALID_STATES)
def test_every_construction_path_rejects_invalid_state(build, overrides, message):
    with pytest.raises(ConfigurationError) as error:
        build(**overrides)

    assert str(error.value) == message


def test_direct_construction_normalises_valid_values():
    settings = _direct(
        environment="PRODUCTION",
        http_port="8080",
        log_level="warning",
        auth_enabled=True,
        api_keys="key-a, key-b",
        provider_timeout_seconds=30,
        provider_retry_count=0,
        usage_storage="sqlite",
        otel_enabled="yes",
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.http_port == 8080
    assert settings.log_level == "WARNING"
    assert settings.auth_enabled is True
    assert settings.api_keys == ("key-a", "key-b")
    assert settings.provider_timeout_seconds == 30.0
    assert settings.provider_retry_count == 0
    assert settings.usage_storage is UsageStorage.SQLITE
    assert settings.otel_enabled is True


def test_direct_construction_matches_from_env_for_the_same_input():
    assert _direct() == Settings.from_env({})


def test_valid_settings_survive_direct_construction_round_trip():
    settings = Settings.from_env({})

    assert Settings(**dataclasses.asdict(settings)) == settings


def test_valid_settings_survive_replace_round_trip():
    settings = Settings.from_env({})

    assert dataclasses.replace(settings) == settings
    assert dataclasses.replace(settings, http_port=9090).http_port == 9090


def test_replace_accepts_already_normalised_values():
    settings = Settings.from_env({})

    production = dataclasses.replace(
        settings,
        environment=Environment.PRODUCTION,
        auth_enabled=True,
        api_keys=("key-a",),
    )

    assert production.environment is Environment.PRODUCTION
    assert production.usage_storage is UsageStorage.MEMORY


def test_zero_retry_and_zero_backoff_survive_every_construction_path():
    assert _direct(provider_retry_count=0).provider_retry_count == 0
    assert _replaced(provider_retry_count=0).provider_retry_count == 0
    assert _direct(provider_retry_backoff_ms=0).provider_retry_backoff_ms == 0
    assert _replaced(provider_retry_backoff_ms=0).provider_retry_backoff_ms == 0


class TestGovernanceSettings:
    """The governance settings: a DSN that is required for governance, and the
    budget flags that make enforcement fail-closed."""

    def test_observability_dsn_defaults_to_empty(self):
        assert Settings.from_env({}).observability_dsn == ""

    def test_observability_dsn_accepts_a_connection_string(self):
        dsn = "postgresql://user:pass@db.example.com:5432/chora_observability"
        assert Settings.from_env({"CHORA_OBSERVABILITY_DSN": dsn}).observability_dsn == dsn

    def test_identity_grpc_addr_defaults_to_empty(self):
        assert Settings.from_env({}).identity_grpc_addr == ""

    def test_identity_grpc_addr_accepts_an_address(self):
        addr = "chora-identity.chora.local:9090"
        assert Settings.from_env({"CHORA_IDENTITY_GRPC_ADDR": addr}).identity_grpc_addr == addr

    def test_governance_mode_defaults_to_chora(self):
        assert Settings.from_env({}).governance_mode is GovernanceMode.CHORA

    def test_governance_mode_accepts_standalone(self):
        settings = Settings.from_env({"CHORA_GOVERNANCE_MODE": "standalone"})
        assert settings.governance_mode is GovernanceMode.STANDALONE

    def test_governance_mode_accepts_an_uppercase_spelling(self):
        settings = Settings.from_env({"CHORA_GOVERNANCE_MODE": "CHORA"})
        assert settings.governance_mode is GovernanceMode.CHORA

    def test_governance_mode_rejects_an_unknown_mode(self):
        with pytest.raises(
            ConfigurationError,
            match="CHORA_GOVERNANCE_MODE must be one of 'chora', 'standalone', got 'hybrid'",
        ):
            Settings.from_env({"CHORA_GOVERNANCE_MODE": "hybrid"})

    def test_budget_required_defaults_to_false(self):
        assert Settings.from_env({}).llm_budget_required is False

    def test_budget_required_tenants_defaults_to_empty(self):
        assert Settings.from_env({}).llm_budget_required_tenants == ()

    def test_budget_required_tenants_are_split_on_commas_and_trimmed(self):
        settings = Settings.from_env(
            {"CHORA_LLM_BUDGET_REQUIRED_TENANTS": " 123e4567-e89b-42d3-a456-426614174000 ,"}
        )

        assert settings.llm_budget_required_tenants == (
            "123e4567-e89b-42d3-a456-426614174000",
        )

    def test_budget_required_tenants_ignore_blank_entries(self):
        settings = Settings.from_env(
            {"CHORA_LLM_BUDGET_REQUIRED_TENANTS": ",, 123e4567-e89b-42d3-a456-426614174000 ,, "}
        )

        assert settings.llm_budget_required_tenants == (
            "123e4567-e89b-42d3-a456-426614174000",
        )

    def test_governance_defaults_survive_direct_construction_round_trip(self):
        assert _direct().observability_dsn == ""
        assert _direct().identity_grpc_addr == ""
        assert _direct().governance_mode is GovernanceMode.CHORA
        assert _direct().llm_budget_required is False
        assert _direct().llm_budget_required_tenants == ()
