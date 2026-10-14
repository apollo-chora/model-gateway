"""Configured inference routes: one operation block per capability (PRD sections 6, 7, 8, 11, 18).

Each environment block — ``TEXT_LLM_*``, ``IMAGE_GEN_*``, ``EMBEDDINGS_*`` —
describes one operation. The block's singleton variables define the primary
route, and ``<BLOCK>_ROUTES_JSON`` appends additional routes in array order.
Route order is fallback order (PRD section 11.2).

A ``<BLOCK>_ROUTES_JSON`` entry is self-contained: no field is inherited from
the singleton block, because a fallback route may point at a different origin
and must carry its own credentials (PRD section 7). Precedence between routes is
positional, never field-merged.

Configuration that cannot be read unambiguously is rejected rather than resolved
by guessing (PRD section 11.1): a partially configured block, a setting left
behind in a block that has no endpoint, model ID or format, an unknown
capability or authentication type, an explicitly supplied but empty capability
list, a reserved authentication header supplied as an extra header, an absolute
URL in ``*_PATH`` without the route's opt-in, a malformed ``*_ROUTES_JSON``, an
entry missing a required field, two routes naming the same endpoint, format and
model, and the unsupported indexed form (``TEXT_LLM_1_*``).

Which authentication modes require a static credential, and how a public model
ID maps onto an operation, are deliberately not this module's business: a route
records the credential and authentication type it was configured with, and the
public contract belongs to the northbound layer.
"""

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, NoReturn, Self

from model_gateway.auth import AuthType
from model_gateway.capabilities import Capability
from model_gateway.config.settings import (
    ConfigurationError,
    _parse_bool,
    _parse_text,
)
from model_gateway.security.url_policy import (
    Resolver,
    UrlPolicyError,
    is_reserved_auth_header,
    resolve_endpoint,
    system_resolver,
)

__all__ = ["Operation", "Route", "parse_routes"]


class Operation(StrEnum):
    """The upstream operation a configured route performs (PRD sections 6 and 10)."""

    TEXT = "text"
    IMAGE = "image"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True)
class Route:
    """One configured inference target, validated at startup (PRD sections 6, 7, 8, 11).

    ``endpoint`` is the base URL exactly as configured and ``base_url`` is that
    endpoint resolved under the URL policy, which is the value handed to the
    inference binding as its base URL (PRD sections 8 and 9.6). ``api_key`` is the
    route's credential and ``auth_type`` its authentication override: ``None``
    means the provider's native authentication, which is the default and the only
    honest reading of a route such as ``vertex_ai`` that authenticates through
    ambient credentials, while an explicit member means the route was configured
    with a non-native header (PRD section 7). ``extra_headers`` holds only
    non-authentication headers, because a reserved authentication header is
    refused at parse time, and it is immutable so that a validated route cannot
    acquire one afterwards.
    """

    operation: Operation
    index: int
    endpoint: str
    base_url: str
    model_id: str
    format: str
    api_key: str
    capabilities: frozenset[Capability]
    extra_headers: Mapping[str, str]
    allow_absolute_path_url: bool
    auth_type: AuthType | None = None

    @property
    def litellm_model(self) -> str:
        """The provider-prefixed model string handed to LiteLLM (PRD section 9.6)."""
        return f"{self.format}/{self.model_id}"


class _ImmutableHeaders(dict[str, str]):
    """A route's extra headers, frozen once the route has been validated (PRD section 7).

    A plain ``dict`` inside a frozen dataclass is only a shallow freeze: assigning
    to it would let a validated route acquire a reserved authentication header the
    parser had refused. Every ordinary mutator raises instead, which is the same
    strength of freeze the dataclass itself provides. Being a ``dict`` subclass
    keeps ``dataclasses.asdict``, equality and JSON serialisation working; the
    ``__reduce__`` hook keeps the frozen type across copying and pickling.
    """

    __slots__ = ()

    def __setitem__(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("a route's extra headers are immutable")

    __delitem__ = __setitem__
    __ior__ = __setitem__
    clear = __setitem__
    pop = __setitem__
    popitem = __setitem__
    setdefault = __setitem__
    update = __setitem__

    def __reduce__(self) -> tuple[type[Self], tuple[dict[str, str]]]:
        return (type(self), (dict(self),))


class _Block(NamedTuple):
    """One operation's environment block."""

    operation: Operation
    prefix: str
    default_capability: Capability


_BLOCKS: tuple[_Block, ...] = (
    _Block(Operation.TEXT, "TEXT_LLM", Capability.CHAT),
    _Block(Operation.IMAGE, "IMAGE_GEN", Capability.GENERATION),
    _Block(Operation.EMBEDDINGS, "EMBEDDINGS", Capability.EMBEDDINGS),
)

_ENDPOINT = "ENDPOINT"
_MODEL_ID = "MODEL_ID"
_FORMAT = "FORMAT"
_API_KEY = "API_KEY"
_AUTH_TYPE = "AUTH_TYPE"
_PATH = "PATH"
_EXTRA_HEADERS_JSON = "EXTRA_HEADERS_JSON"
_CAPABILITIES = "CAPABILITIES"
_ALLOW_ABSOLUTE_PATH_URL = "ALLOW_ABSOLUTE_PATH_URL"
_CONTEXT_WINDOW = "CONTEXT_WINDOW"
_MAX_OUTPUT_TOKENS = "MAX_OUTPUT_TOKENS"
_DIMENSIONS = "DIMENSIONS"
_SUPPORTS_DIMENSIONS_PARAM = "SUPPORTS_DIMENSIONS_PARAM"
_ROUTES_JSON = "ROUTES_JSON"

_IDENTITY_SUFFIXES = (_ENDPOINT, _MODEL_ID, _FORMAT)

_ENTRY_ENDPOINT = "endpoint"
_ENTRY_MODEL_ID = "model_id"
_ENTRY_FORMAT = "format"
_ENTRY_API_KEY = "api_key"
_ENTRY_AUTH_TYPE = "auth_type"
_ENTRY_PATH = "path"
_ENTRY_EXTRA_HEADERS = "extra_headers"
_ENTRY_CAPABILITIES = "capabilities"
_ENTRY_ALLOW_ABSOLUTE_PATH_URL = "allow_absolute_path_url"

_ENTRY_KEYS: frozenset[str] = frozenset(
    {
        _ENTRY_ENDPOINT,
        _ENTRY_MODEL_ID,
        _ENTRY_FORMAT,
        _ENTRY_API_KEY,
        _ENTRY_AUTH_TYPE,
        _ENTRY_PATH,
        _ENTRY_EXTRA_HEADERS,
        _ENTRY_CAPABILITIES,
        _ENTRY_ALLOW_ABSOLUTE_PATH_URL,
    }
)

_INDEXED_FORM: Mapping[str, re.Pattern[str]] = {
    block.prefix: re.compile(rf"^{re.escape(block.prefix)}_\d+(?:_|$)") for block in _BLOCKS
}


def parse_routes(
    env: Mapping[str, str] | None = None,
    *,
    resolver: Resolver = system_resolver,
) -> tuple[Route, ...]:
    """Read every configured route from ``env``, defaulting to the process environment.

    Routes come back in a deterministic order: by operation (text, image,
    embeddings) and, within an operation, by route index — the singleton block's
    primary route first, then ``<BLOCK>_ROUTES_JSON`` entries in array order.
    That order is the fallback order of PRD section 11.2.

    ``resolver`` resolves hostnames for the URL policy and exists so that tests
    can validate routes without DNS; deployments use the system resolver.
    """
    source = os.environ if env is None else env
    _reject_indexed_form(source)

    routes: list[Route] = []
    for block in _BLOCKS:
        routes.extend(_parse_block(block, source, resolver=resolver))
    return tuple(routes)


def _parse_block(block: _Block, env: Mapping[str, str], *, resolver: Resolver) -> list[Route]:
    routes_variable = _variable(block, _ROUTES_JSON)
    routes_json = _read(env, routes_variable)

    if not any(_read(env, _variable(block, suffix)) for suffix in _IDENTITY_SUFFIXES):
        orphans = _orphan_configuration(block, env)
        if orphans:
            raise ConfigurationError(
                f"{orphans[0]} is set but the {block.prefix} block has no endpoint, model ID or "
                f"format; a capability is enabled only when its singleton block is configured, so "
                f"define {block.prefix}_ENDPOINT, {block.prefix}_MODEL_ID and "
                f"{block.prefix}_FORMAT, or remove {orphans[0]}"
            )
        if routes_json:
            raise ConfigurationError(
                f"{routes_variable} is set but the {block.prefix}_* block is unset; a capability "
                f"is enabled only when its singleton block is configured, so define "
                f"{block.prefix}_ENDPOINT, {block.prefix}_MODEL_ID and {block.prefix}_FORMAT, or "
                f"remove {routes_variable}"
            )
        return []

    endpoint = _parse_text(_variable(block, _ENDPOINT), _read(env, _variable(block, _ENDPOINT)))
    model_id = _parse_text(_variable(block, _MODEL_ID), _read(env, _variable(block, _MODEL_ID)))
    format = _parse_text(_variable(block, _FORMAT), _read(env, _variable(block, _FORMAT)))
    path = _read(env, _variable(block, _PATH))
    api_key = _read(env, _variable(block, _API_KEY))
    auth_type = _parse_auth_type(
        _variable(block, _AUTH_TYPE), _read(env, _variable(block, _AUTH_TYPE))
    )

    opt_in_variable = _variable(block, _ALLOW_ABSOLUTE_PATH_URL)
    opt_in = _read(env, opt_in_variable)
    allow_absolute_path_url = _parse_bool(opt_in_variable, opt_in) if opt_in else False
    capabilities = _singleton_capabilities(block, _read(env, _variable(block, _CAPABILITIES)))
    extra_headers = _singleton_extra_headers(
        block, _read(env, _variable(block, _EXTRA_HEADERS_JSON))
    )

    primary = Route(
        operation=block.operation,
        index=0,
        endpoint=endpoint,
        base_url=_resolve_base_url(
            endpoint,
            path,
            allow_absolute_path_url=allow_absolute_path_url,
            endpoint_variable=_variable(block, _ENDPOINT),
            path_variable=_variable(block, _PATH),
            resolver=resolver,
        ),
        model_id=model_id,
        format=format,
        api_key=api_key,
        auth_type=auth_type,
        capabilities=capabilities,
        extra_headers=extra_headers,
        allow_absolute_path_url=allow_absolute_path_url,
    )

    sources: list[tuple[str, Route]] = [(f"the {block.prefix}_* block", primary)]
    for route in _parse_entries(block, routes_variable, routes_json, resolver=resolver):
        sources.append((f"{routes_variable} entry {route.index}", route))
    _reject_duplicate_routes(block, sources)
    return [route for _, route in sources]


def _orphan_configuration(block: _Block, env: Mapping[str, str]) -> list[str]:
    """The block's non-identity settings that hold a value the gateway cannot act on.

    A block with no endpoint, model ID or format defines no route, so any other
    value it carries is configuration the gateway would otherwise discard in
    silence — a non-empty API key above all.
    """
    orphans: list[str] = []
    for suffix, is_inert in _INERT_SETTINGS:
        variable = _variable(block, suffix)
        if not is_inert(variable, _read(env, variable)):
            orphans.append(variable)
    return sorted(orphans)


def _inert_text(variable: str, raw: str) -> bool:
    return not raw


def _inert_object(variable: str, raw: str) -> bool:
    if not raw:
        return True
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and not parsed


def _inert_flag(variable: str, raw: str) -> bool:
    if not raw:
        return True
    return not _parse_bool(variable, raw)


_INERT_SETTINGS: tuple[tuple[str, Callable[[str, str], bool]], ...] = (
    (_API_KEY, _inert_text),
    (_AUTH_TYPE, _inert_text),
    (_PATH, _inert_text),
    (_EXTRA_HEADERS_JSON, _inert_object),
    (_CAPABILITIES, _inert_text),
    (_ALLOW_ABSOLUTE_PATH_URL, _inert_flag),
    (_CONTEXT_WINDOW, _inert_text),
    (_MAX_OUTPUT_TOKENS, _inert_text),
    (_DIMENSIONS, _inert_text),
    (_SUPPORTS_DIMENSIONS_PARAM, _inert_flag),
)


def _parse_entries(
    block: _Block,
    variable: str,
    raw: str,
    *,
    resolver: Resolver,
) -> list[Route]:
    if not raw:
        return []

    parsed = _parse_json(variable, raw, "a JSON array of route objects")
    if not isinstance(parsed, list):
        raise ConfigurationError(
            f"{variable} must be a JSON array of route objects, got {_json_type(parsed)}"
        )
    return [
        _parse_entry(block, variable, index, entry, resolver=resolver)
        for index, entry in enumerate(parsed, start=1)
    ]


def _parse_entry(
    block: _Block,
    variable: str,
    index: int,
    entry: object,
    *,
    resolver: Resolver,
) -> Route:
    where = f"{variable} entry {index}"
    if not isinstance(entry, dict):
        raise ConfigurationError(f"{where} must be a JSON object, got {_json_type(entry)}")

    unsupported = sorted(set(entry) - _ENTRY_KEYS)
    if unsupported:
        supported = ", ".join(repr(key) for key in sorted(_ENTRY_KEYS))
        raise ConfigurationError(
            f"{where} has unsupported key {unsupported[0]!r}; supported keys are {supported}"
        )

    endpoint = _entry_required_text(where, entry, _ENTRY_ENDPOINT)
    model_id = _entry_required_text(where, entry, _ENTRY_MODEL_ID)
    format = _entry_required_text(where, entry, _ENTRY_FORMAT)
    path = _entry_optional_text(where, entry, _ENTRY_PATH)
    allow_absolute_path_url = _entry_flag(where, entry)

    return Route(
        operation=block.operation,
        index=index,
        endpoint=endpoint,
        base_url=_resolve_base_url(
            endpoint,
            path,
            allow_absolute_path_url=allow_absolute_path_url,
            endpoint_variable=f"{where} {_ENTRY_ENDPOINT!r}",
            path_variable=f"{where} {_ENTRY_PATH!r}",
            resolver=resolver,
        ),
        model_id=model_id,
        format=format,
        api_key=_entry_optional_text(where, entry, _ENTRY_API_KEY),
        auth_type=_entry_auth_type(where, entry.get(_ENTRY_AUTH_TYPE)),
        capabilities=_entry_capabilities(block, where, entry.get(_ENTRY_CAPABILITIES)),
        extra_headers=_entry_extra_headers(where, entry.get(_ENTRY_EXTRA_HEADERS)),
        allow_absolute_path_url=allow_absolute_path_url,
    )


def _reject_duplicate_routes(block: _Block, sources: Sequence[tuple[str, Route]]) -> None:
    seen: dict[tuple[str, str, str], str] = {}
    for label, route in sources:
        identity = (route.base_url, route.format, route.model_id)
        first = seen.get(identity)
        if first is not None:
            raise ConfigurationError(
                f"{label} names the same endpoint, format and model ID as {first}; every route of "
                f"the {block.prefix} operation must name a distinct endpoint, format and model"
            )
        seen[identity] = label


def _reject_indexed_form(env: Mapping[str, str]) -> None:
    for block in _BLOCKS:
        pattern = _INDEXED_FORM[block.prefix]
        offenders = sorted(name for name in env if pattern.match(name))
        if offenders:
            raise ConfigurationError(
                f"{offenders[0]}: the indexed route form ({block.prefix}_1_*, {block.prefix}_2_*, "
                f"...) is not supported; define the primary route in the {block.prefix}_* block "
                f"and additional routes in {block.prefix}_ROUTES_JSON"
            )


def _resolve_base_url(
    endpoint: str,
    path: str,
    *,
    allow_absolute_path_url: bool,
    endpoint_variable: str,
    path_variable: str,
    resolver: Resolver,
) -> str:
    try:
        base_url = resolve_endpoint(endpoint, resolver=resolver)
    except UrlPolicyError as error:
        raise ConfigurationError(f"{endpoint_variable}: {error}") from error

    if not path:
        return base_url

    try:
        return resolve_endpoint(
            endpoint,
            path,
            allow_absolute_path_url=allow_absolute_path_url,
            resolver=resolver,
        )
    except UrlPolicyError as error:
        raise ConfigurationError(f"{path_variable}: {error}") from error


def _singleton_capabilities(block: _Block, raw: str) -> frozenset[Capability]:
    if not raw:
        return frozenset({block.default_capability})
    return _capabilities_from(_variable(block, _CAPABILITIES), raw, raw.split(","))


def _entry_capabilities(block: _Block, where: str, value: object) -> frozenset[Capability]:
    label = f"{where} {_ENTRY_CAPABILITIES!r}"
    if value is None:
        return frozenset({block.default_capability})
    if isinstance(value, str):
        return _capabilities_from(label, value, value.split(","))
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                raise ConfigurationError(
                    f"{label} must be an array of capability names, got {_json_type(item)} in "
                    f"the array"
                )
        return _capabilities_from(label, value, value)
    raise ConfigurationError(
        f"{label} must be an array of capability names or a comma-separated string, got "
        f"{_json_type(value)}"
    )


def _capabilities_from(where: str, raw: object, tokens: Iterable[str]) -> frozenset[Capability]:
    capabilities = frozenset(_capability(where, token) for token in tokens if token.strip())
    if not capabilities:
        raise ConfigurationError(f"{where} must name at least one capability, got {raw!r}")
    return capabilities


def _capability(where: str, token: str) -> Capability:
    name = token.strip().lower()
    try:
        return Capability(name)
    except ValueError as error:
        supported = ", ".join(repr(member.value) for member in Capability)
        raise ConfigurationError(
            f"{where} must name capabilities from {supported}; got {name!r}"
        ) from error


def _parse_auth_type(where: str, raw: str) -> AuthType | None:
    """Parse an authentication override; ``None`` means the provider's native scheme.

    An absent value is not the same as ``bearer``: PRD section 7 makes native
    provider authentication the default and scopes ``*_AUTH_TYPE`` to custom
    endpoints, so collapsing the two would tell the credential rules that a
    ``vertex_ai`` route presents a bearer token when it actually uses ADC.
    """
    if not raw:
        return None
    try:
        return AuthType(raw.lower())
    except ValueError as error:
        supported = ", ".join(repr(member.value) for member in AuthType)
        raise ConfigurationError(
            f"{where} must be one of {supported}, got {raw!r}; omit it to use the provider's "
            f"native authentication"
        ) from error


def _entry_auth_type(where: str, value: object) -> AuthType | None:
    label = f"{where} {_ENTRY_AUTH_TYPE!r}"
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a string, got {_json_type(value)}")
    return _parse_auth_type(label, value.strip())


def _singleton_extra_headers(block: _Block, raw: str) -> _ImmutableHeaders:
    variable = _variable(block, _EXTRA_HEADERS_JSON)
    if not raw:
        return _ImmutableHeaders()

    parsed = _parse_json(variable, raw, "a JSON object of string values")
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"{variable} must be a JSON object of string values, got {_json_type(parsed)}"
        )
    return _headers_from(variable, parsed)


def _entry_extra_headers(where: str, value: object) -> _ImmutableHeaders:
    label = f"{where} {_ENTRY_EXTRA_HEADERS!r}"
    if value is None:
        return _ImmutableHeaders()
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"{label} must be a JSON object of string values, got {_json_type(value)}"
        )
    return _headers_from(label, value)


def _headers_from(where: str, parsed: Mapping[str, object]) -> _ImmutableHeaders:
    headers: dict[str, str] = {}
    for name, value in parsed.items():
        if not isinstance(value, str):
            raise ConfigurationError(
                f"{where} header {name!r} must have a string value, got {_json_type(value)}"
            )
        if is_reserved_auth_header(name):
            raise ConfigurationError(
                f"{where} must not set the reserved authentication header {name!r}; the route's "
                f"own authentication configuration owns that header"
            )
        headers[name] = value
    return _ImmutableHeaders(headers)


def _entry_required_text(where: str, entry: Mapping[str, object], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{where} must specify a non-empty {key!r}")
    return value.strip()


def _entry_optional_text(where: str, entry: Mapping[str, object], key: str) -> str:
    value = entry.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigurationError(f"{where} {key!r} must be a string, got {_json_type(value)}")
    return value.strip()


def _entry_flag(where: str, entry: Mapping[str, object]) -> bool:
    value = entry.get(_ENTRY_ALLOW_ABSOLUTE_PATH_URL)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"{where} {_ENTRY_ALLOW_ABSOLUTE_PATH_URL!r} must be a boolean, got {_json_type(value)}"
        )
    return value


def _parse_json(where: str, raw: str, expectation: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{where} must be {expectation}: {error}") from error


def _json_type(value: object) -> str:
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, dict):
        return "an object"
    return "null"


def _variable(block: _Block, suffix: str) -> str:
    return f"{block.prefix}_{suffix}"


def _read(env: Mapping[str, str], variable: str) -> str:
    return str(env.get(variable, "")).strip()
