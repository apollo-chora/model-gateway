"""Deployment settings loaded and validated at startup (PRD sections 6, 7, 15, 18).

Environment variables are the authoritative deployment configuration. Every
value a deployment can set is read and validated here, so an unusable
configuration fails when the process starts rather than at the first inference
request.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_MIN_PORT = 1
_MAX_PORT = 65535


class Environment(StrEnum):
    """Deployment environment, which selects the security policy (PRD section 7)."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class UsageStorage(StrEnum):
    """Where usage and cost records are persisted (PRD section 13)."""

    NONE = "none"
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class ConfigurationError(ValueError):
    """Raised when the deployment environment cannot be used as configured."""


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_BOOLEAN_VALUES = tuple(sorted(_TRUE_VALUES | _FALSE_VALUES))


def _raw(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, default).strip()


def _read_text(
    env: Mapping[str, str],
    name: str,
    default: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = _raw(env, name, default)
    if not value and not allow_empty:
        raise ConfigurationError(f"{name} must not be empty")
    return value


def _read_bool(env: Mapping[str, str], name: str, default: str) -> bool:
    value = _raw(env, name, default).lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    expected = ", ".join(repr(option) for option in _BOOLEAN_VALUES)
    raise ConfigurationError(
        f"{name} must be a boolean, got {value!r}; expected one of {expected}"
    )


def _read_port(env: Mapping[str, str], name: str, default: str) -> int:
    value = _raw(env, name, default)
    try:
        port = int(value)
    except ValueError as exc:
        raise _port_error(name, value) from exc
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise _port_error(name, value)
    return port


def _port_error(name: str, value: str) -> ConfigurationError:
    return ConfigurationError(
        f"{name} must be a port number between {_MIN_PORT} and {_MAX_PORT}, got {value!r}"
    )


def _read_positive_int(env: Mapping[str, str], name: str, default: str) -> int:
    value = _raw(env, name, default)
    try:
        number = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}") from exc
    if number <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}")
    return number


def _read_non_negative_int(env: Mapping[str, str], name: str, default: str) -> int:
    value = _raw(env, name, default)
    try:
        number = int(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from exc
    if number < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer, got {value!r}")
    return number


def _read_positive_float(env: Mapping[str, str], name: str, default: str) -> float:
    value = _raw(env, name, default)
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be a finite number, got {value!r}")
    if number <= 0:
        raise ConfigurationError(f"{name} must be a positive number, got {value!r}")
    return number


def _read_list(env: Mapping[str, str], name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in _raw(env, name, default).split(",") if item.strip())


def _read_log_level(env: Mapping[str, str], name: str, default: str) -> str:
    value = _read_text(env, name, default)
    level = value.upper()
    if level not in LOG_LEVELS:
        supported = ", ".join(repr(option) for option in LOG_LEVELS)
        raise ConfigurationError(f"{name} must be one of {supported}, got {value!r}")
    return level


def _read_environment(env: Mapping[str, str], name: str, default: str) -> Environment:
    value = _read_text(env, name, default)
    try:
        return Environment(value.lower())
    except ValueError as exc:
        raise _environment_error(value) from exc


def _environment_error(value: object) -> ConfigurationError:
    supported = ", ".join(repr(member.value) for member in Environment)
    return ConfigurationError(f"CHORA_ENVIRONMENT must be one of {supported}, got {value!r}")


def _read_usage_storage(env: Mapping[str, str], name: str, default: str) -> UsageStorage:
    value = _read_text(env, name, default, allow_empty=True)
    try:
        return UsageStorage(value)
    except ValueError as exc:
        raise _usage_storage_error(name, value) from exc


def _usage_storage_error(name: str, value: object) -> ConfigurationError:
    supported = ", ".join(repr(member.value) for member in UsageStorage)
    return ConfigurationError(f"{name} must be one of {supported}, got {value!r}")


@dataclass(frozen=True)
class Settings:
    """Validated settings for one gateway process."""

    environment: Environment
    http_port: int
    grpc_port: int
    log_level: str

    auth_enabled: bool
    api_keys: tuple[str, ...]
    allow_anonymous_health: bool

    provider_timeout_seconds: float
    provider_connect_timeout_seconds: float
    provider_retry_count: int
    provider_retry_backoff_ms: int
    provider_max_concurrent_requests: int
    shutdown_timeout_seconds: float

    usage_tracking_enabled: bool
    cost_tracking_enabled: bool
    usage_storage: UsageStorage

    otel_enabled: bool
    otel_exporter_otlp_endpoint: str

    def __post_init__(self) -> None:
        environment = self.environment
        if not isinstance(environment, Environment):
            try:
                environment = Environment(str(environment).lower())
            except ValueError as exc:
                raise _environment_error(self.environment) from exc
            object.__setattr__(self, "environment", environment)

        storage = self.usage_storage
        if not isinstance(storage, UsageStorage):
            try:
                storage = UsageStorage(storage)
            except ValueError as exc:
                raise _usage_storage_error("USAGE_STORAGE", self.usage_storage) from exc
            object.__setattr__(self, "usage_storage", storage)

        if environment is Environment.PRODUCTION and not self.auth_enabled:
            raise ConfigurationError(
                "CHORA_ENVIRONMENT is production but GATEWAY_AUTH_ENABLED is false; "
                "production must not serve unauthenticated inference. Set "
                "GATEWAY_AUTH_ENABLED=true with GATEWAY_API_KEYS, or set "
                "CHORA_ENVIRONMENT=development"
            )

        if self.auth_enabled and not self.api_keys:
            raise ConfigurationError(
                "GATEWAY_AUTH_ENABLED is true but GATEWAY_API_KEYS is empty; "
                "set one or more comma-separated keys, or set GATEWAY_AUTH_ENABLED=false"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Read settings from ``env``, defaulting to the process environment."""
        source = os.environ if env is None else env
        return cls(
            environment=_read_environment(source, "CHORA_ENVIRONMENT", "development"),
            http_port=_read_port(source, "CHORA_HTTP_PORT", "8080"),
            grpc_port=_read_port(source, "CHORA_GRPC_PORT", "9090"),
            log_level=_read_log_level(source, "LOG_LEVEL", "info"),
            auth_enabled=_read_bool(source, "GATEWAY_AUTH_ENABLED", "false"),
            api_keys=_read_list(source, "GATEWAY_API_KEYS"),
            allow_anonymous_health=_read_bool(source, "GATEWAY_ALLOW_ANONYMOUS_HEALTH", "true"),
            provider_timeout_seconds=_read_positive_float(
                source, "PROVIDER_TIMEOUT_SECONDS", "110"
            ),
            provider_connect_timeout_seconds=_read_positive_float(
                source, "PROVIDER_CONNECT_TIMEOUT_SECONDS", "10"
            ),
            provider_retry_count=_read_non_negative_int(source, "PROVIDER_RETRY_COUNT", "2"),
            provider_retry_backoff_ms=_read_non_negative_int(
                source, "PROVIDER_RETRY_BACKOFF_MS", "500"
            ),
            provider_max_concurrent_requests=_read_positive_int(
                source, "PROVIDER_MAX_CONCURRENT_REQUESTS", "100"
            ),
            shutdown_timeout_seconds=_read_positive_float(
                source, "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS", "30"
            ),
            usage_tracking_enabled=_read_bool(source, "USAGE_TRACKING_ENABLED", "true"),
            cost_tracking_enabled=_read_bool(source, "COST_TRACKING_ENABLED", "false"),
            usage_storage=_read_usage_storage(source, "USAGE_STORAGE", "memory"),
            otel_enabled=_read_bool(source, "OTEL_ENABLED", "false"),
            otel_exporter_otlp_endpoint=_read_text(
                source, "OTEL_EXPORTER_OTLP_ENDPOINT", "", allow_empty=True
            ),
        )
