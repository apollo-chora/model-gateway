"""Deployment settings loaded and validated at startup (PRD sections 6, 7, 15, 18).

Environment variables are the authoritative deployment configuration. Every
value a deployment can set is described once, in ``_FIELDS``, and parsed by the
parser named there. Parsing happens in ``Settings.__post_init__``, so every
construction path — ``from_env``, direct construction and
``dataclasses.replace`` — either produces a valid ``Settings`` or raises
``ConfigurationError``; an instance cannot hold a value the gateway could not
run with.
"""

import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import NamedTuple, Self

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


Parser = Callable[[str, object], object]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_BOOLEAN_VALUES = tuple(sorted(_TRUE_VALUES | _FALSE_VALUES))


def _raw(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, default).strip()


def _text(value: object) -> str:
    return str(value).strip()


def _parse_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    text = _text(value)
    if not text and not allow_empty:
        raise ConfigurationError(f"{name} must not be empty")
    return text


_parse_optional_text = partial(_parse_text, allow_empty=True)


def _parse_bool(name: str, value: object) -> bool:
    text = _text(value).lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    expected = ", ".join(repr(option) for option in _BOOLEAN_VALUES)
    raise ConfigurationError(
        f"{name} must be a boolean, got {text!r}; expected one of {expected}"
    )


def _parse_port(name: str, value: object) -> int:
    text = _text(value)
    try:
        port = int(text)
    except ValueError as exc:
        raise _port_error(name, text) from exc
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise _port_error(name, text)
    return port


def _port_error(name: str, value: str) -> ConfigurationError:
    return ConfigurationError(
        f"{name} must be a port number between {_MIN_PORT} and {_MAX_PORT}, got {value!r}"
    )


def _parse_positive_int(name: str, value: object) -> int:
    text = _text(value)
    try:
        number = int(text)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer, got {text!r}") from exc
    if number <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {text!r}")
    return number


def _parse_non_negative_int(name: str, value: object) -> int:
    text = _text(value)
    try:
        number = int(text)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a non-negative integer, got {text!r}"
        ) from exc
    if number < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer, got {text!r}")
    return number


def _parse_positive_float(name: str, value: object) -> float:
    text = _text(value)
    try:
        number = float(text)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive number, got {text!r}") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be a finite number, got {text!r}")
    if number <= 0:
        raise ConfigurationError(f"{name} must be a positive number, got {text!r}")
    return number


def _parse_log_level(name: str, value: object) -> str:
    text = _parse_text(name, value)
    level = text.upper()
    if level not in LOG_LEVELS:
        supported = ", ".join(repr(option) for option in LOG_LEVELS)
        raise ConfigurationError(f"{name} must be one of {supported}, got {text!r}")
    return level


def _parse_environment(name: str, value: object) -> Environment:
    text = _parse_text(name, value)
    try:
        return Environment(text.lower())
    except ValueError as exc:
        raise _environment_error(name, text) from exc


def _environment_error(name: str, value: object) -> ConfigurationError:
    supported = ", ".join(repr(member.value) for member in Environment)
    return ConfigurationError(f"{name} must be one of {supported}, got {value!r}")


def _parse_usage_storage(name: str, value: object) -> UsageStorage:
    text = _text(value)
    try:
        return UsageStorage(text)
    except ValueError as exc:
        raise _usage_storage_error(name, text) from exc


def _usage_storage_error(name: str, value: object) -> ConfigurationError:
    supported = ", ".join(repr(member.value) for member in UsageStorage)
    return ConfigurationError(f"{name} must be one of {supported}, got {value!r}")


def _parse_api_keys(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        items: Iterable[object] = value.split(",")
    elif isinstance(value, Iterable):
        items = value
    else:
        raise ConfigurationError(
            f"{name} must be a comma-separated list of keys, got {value!r}"
        )
    keys = (str(item).strip() for item in items)
    return tuple(key for key in keys if key)


class _Field(NamedTuple):
    """One deployment variable: the attribute it fills, its name, default and parser."""

    attribute: str
    variable: str
    default: str
    parse: Parser


_FIELDS: tuple[_Field, ...] = (
    _Field("environment", "CHORA_ENVIRONMENT", "development", _parse_environment),
    _Field("http_port", "CHORA_HTTP_PORT", "8080", _parse_port),
    _Field("grpc_port", "CHORA_GRPC_PORT", "9090", _parse_port),
    _Field("log_level", "LOG_LEVEL", "info", _parse_log_level),
    _Field("auth_enabled", "GATEWAY_AUTH_ENABLED", "false", _parse_bool),
    _Field("api_keys", "GATEWAY_API_KEYS", "", _parse_api_keys),
    _Field("allow_anonymous_health", "GATEWAY_ALLOW_ANONYMOUS_HEALTH", "true", _parse_bool),
    _Field("provider_timeout_seconds", "PROVIDER_TIMEOUT_SECONDS", "110", _parse_positive_float),
    _Field(
        "provider_connect_timeout_seconds",
        "PROVIDER_CONNECT_TIMEOUT_SECONDS",
        "10",
        _parse_positive_float,
    ),
    _Field("provider_retry_count", "PROVIDER_RETRY_COUNT", "2", _parse_non_negative_int),
    _Field(
        "provider_retry_backoff_ms",
        "PROVIDER_RETRY_BACKOFF_MS",
        "500",
        _parse_non_negative_int,
    ),
    _Field(
        "provider_max_concurrent_requests",
        "PROVIDER_MAX_CONCURRENT_REQUESTS",
        "100",
        _parse_positive_int,
    ),
    _Field("rate_limit_requests", "GATEWAY_RATE_LIMIT_REQUESTS", "0", _parse_non_negative_int),
    _Field(
        "rate_limit_window_seconds",
        "GATEWAY_RATE_LIMIT_WINDOW_SECONDS",
        "60",
        _parse_positive_float,
    ),
    _Field(
        "tenant_max_concurrent_requests",
        "GATEWAY_TENANT_MAX_CONCURRENT_REQUESTS",
        "0",
        _parse_non_negative_int,
    ),
    _Field(
        "max_request_body_bytes",
        "GATEWAY_MAX_REQUEST_BODY_BYTES",
        "16777216",
        _parse_positive_int,
    ),
    _Field(
        "shutdown_timeout_seconds",
        "GATEWAY_SHUTDOWN_TIMEOUT_SECONDS",
        "30",
        _parse_positive_float,
    ),
    _Field("usage_tracking_enabled", "USAGE_TRACKING_ENABLED", "true", _parse_bool),
    _Field("cost_tracking_enabled", "COST_TRACKING_ENABLED", "false", _parse_bool),
    _Field("usage_storage", "USAGE_STORAGE", "memory", _parse_usage_storage),
    _Field("otel_enabled", "OTEL_ENABLED", "false", _parse_bool),
    _Field(
        "otel_exporter_otlp_endpoint",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "",
        _parse_optional_text,
    ),
)


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
    rate_limit_requests: int
    rate_limit_window_seconds: float
    tenant_max_concurrent_requests: int
    max_request_body_bytes: int
    shutdown_timeout_seconds: float

    usage_tracking_enabled: bool
    cost_tracking_enabled: bool
    usage_storage: UsageStorage

    otel_enabled: bool
    otel_exporter_otlp_endpoint: str

    def __post_init__(self) -> None:
        for entry in _FIELDS:
            current = getattr(self, entry.attribute)
            object.__setattr__(self, entry.attribute, entry.parse(entry.variable, current))

        if self.environment is Environment.PRODUCTION and not self.auth_enabled:
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
        values: dict[str, object] = {}
        for entry in _FIELDS:
            raw = _raw(source, entry.variable, entry.default)
            values[entry.attribute] = entry.parse(entry.variable, raw)
        return cls(**values)
