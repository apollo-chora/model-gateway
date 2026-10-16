"""Tests for the inbound HTTP authentication layer.

These tests exercise authentication against the real HTTP surface — middleware,
handler and error mapping — so a request travels the whole path and the
assertions are on what a caller sees: which endpoints a credential opens, how
the health exemption behaves both ways, how a body ``tenant_id`` is treated
when it matches, disagrees and is absent, and that the presented key reaches
neither a response body nor a log line.

The key set is a double. The authentication layer is what these tests are
about, so the double stands in for the configured keys and lets a test bind a
key to a tenant of its choosing — which is what the tenant rule needs
exercised. The application-factory tests at the end use the real key set.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from blacksheep import Application, JSONContent, Request
from blacksheep.server.routing import Router
from blacksheep.server.routing import router as default_router
from blacksheep.testing import TestClient

from model_gateway.app import create_app
from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import FinishReason, InvocationResponse, TextPart, Usage
from model_gateway.errors import UnauthenticatedError
from model_gateway.health import healthz, readyz
from model_gateway.http.auth import (
    create_auth_middleware,
    extract_credential,
    get_identity,
)
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.models import register_model_routes
from model_gateway.inference import InferenceService
from model_gateway.registry import ModelRegistry
from model_gateway.security.identity import Identity

API_KEY = "sk-gateway-test-key"
WRONG_KEY = "sk-gateway-wrong-key"
TENANT = "tenant-a"
# A tenant no configured key is bound to, so a body claiming it disagrees with
# whatever tenant the real key set derives for the factory tests below.
UNBOUND_TENANT = "tenant-no-key-is-bound-to"
UPSTREAM_ENDPOINT = "https://upstream.internal.example/v1"
UPSTREAM_KEY = "sk-upstream-secret-key"

CHAT_BODY = {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}


def auth_headers(key: str = API_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class FakeTextProvider:
    """A ``TextProvider`` double: records the invocations, answers with a canned response."""

    def __init__(self) -> None:
        self.calls: list[tuple[Route, object]] = []

    async def complete(self, route: Route, request, context) -> InvocationResponse:
        self.calls.append((route, request))
        return InvocationResponse(
            model="text-default",
            upstream_model="upstream-text",
            content=(TextPart("Recursion is a technique..."),),
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=Usage(input_tokens=25, output_tokens=42, total_tokens=67),
        )


def make_registry() -> ModelRegistry:
    return ModelRegistry(
        (
            Route(
                operation=Operation.TEXT,
                index=0,
                endpoint=UPSTREAM_ENDPOINT,
                base_url=UPSTREAM_ENDPOINT,
                model_id="upstream-text",
                format="openai",
                api_key=UPSTREAM_KEY,
                capabilities=frozenset({Capability.CHAT}),
                extra_headers={},
                allow_absolute_path_url=False,
            ),
        )
    )


def make_settings(**overrides: str) -> Settings:
    values = {
        "GATEWAY_AUTH_ENABLED": "true",
        "GATEWAY_API_KEYS": API_KEY,
        "GATEWAY_ALLOW_ANONYMOUS_HEALTH": "true",
    }
    values.update(overrides)
    return Settings.from_env(values)


class StubKeySet:
    """An ``ApiKeySet`` double binding each presented key to a tenant of the test's choosing."""

    def __init__(self, bindings: dict[str, str | None]) -> None:
        self._bindings = dict(bindings)

    @classmethod
    def from_settings(cls, settings: Settings) -> "StubKeySet":
        return cls({key: TENANT for key in settings.api_keys})

    @property
    def enabled(self) -> bool:
        return bool(self._bindings)

    def authenticate(self, presented: str | None) -> Identity:
        if presented is None:
            raise UnauthenticatedError("no credential was presented")
        if presented not in self._bindings:
            raise UnauthenticatedError("the credential was rejected", reason="invalid_credentials")
        return Identity(
            authenticated=True,
            tenant=self._bindings[presented],
            credential_id="stub-credential-id",
            key_label="stub-key",
        )

    def anonymous(self) -> Identity:
        return Identity(authenticated=False, tenant=None, credential_id=None, key_label=None)


def build_app(
    key_set: StubKeySet,
    *,
    auth_enabled: bool = True,
    allow_anonymous_health: bool = True,
    provider: FakeTextProvider | None = None,
) -> Application:
    registry = make_registry()
    app = Application(router=Router())
    register_error_handlers(app)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    register_model_routes(app.router, registry)
    register_chat_routes(app.router, InferenceService(registry, provider or FakeTextProvider()))
    app.middlewares.append(
        create_auth_middleware(
            key_set,
            auth_enabled=auth_enabled,
            allow_anonymous_health=allow_anonymous_health,
        )
    )
    return app


@asynccontextmanager
async def running(app: Application) -> AsyncIterator[TestClient]:
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


@pytest.fixture
def clear_route_cache():
    """Drop the default router's cached route matches around a test.

    BlackSheep caches route matches on the router instance, and the default
    router is shared by every application built without an explicit router —
    which is what ``create_app`` builds. Two applications that differ in
    behavior (here: authentication on and off) would otherwise serve each
    other's cached matches, so a test that varies them clears the cache
    before and after running.
    """
    default_router.get_match_by_method_and_path.cache_clear()
    yield
    default_router.get_match_by_method_and_path.cache_clear()


@pytest.fixture
async def provider():
    return FakeTextProvider()


@pytest.fixture
async def client(provider):
    """A client for an application with authentication enabled and a tenant-bound key."""
    app = build_app(StubKeySet({API_KEY: TENANT}), provider=provider)
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


class LogCapture(logging.Handler):
    """Collects the rendered message of every record the root logger emits."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def log_capture():
    root = logging.root
    capture = LogCapture()
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        yield capture
    finally:
        root.removeHandler(capture)
        root.setLevel(previous_level)


# --------------------------------------------------------------------------------------
# Credential extraction
# --------------------------------------------------------------------------------------


def test_a_bearer_token_is_extracted():
    request = Request("GET", b"/", [(b"authorization", f"Bearer {API_KEY}".encode())])

    assert extract_credential(request) == API_KEY


def test_the_bearer_scheme_is_case_insensitive():
    request = Request("GET", b"/", [(b"authorization", f"bearer {API_KEY}".encode())])

    assert extract_credential(request) == API_KEY


def test_header_names_are_case_insensitive():
    request = Request("GET", b"/", [(b"Authorization", f"Bearer {API_KEY}".encode())])

    assert extract_credential(request) == API_KEY


def test_x_api_key_is_extracted():
    request = Request("GET", b"/", [(b"x-api-key", API_KEY.encode())])

    assert extract_credential(request) == API_KEY


def test_the_two_headers_agreeing_present_one_credential():
    request = Request(
        "GET",
        b"/",
        [
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"x-api-key", API_KEY.encode()),
        ],
    )

    assert extract_credential(request) == API_KEY


def test_conflicting_headers_are_rejected():
    request = Request(
        "GET",
        b"/",
        [
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"x-api-key", WRONG_KEY.encode()),
        ],
    )

    with pytest.raises(UnauthenticatedError):
        extract_credential(request)


def test_a_repeated_authorization_header_is_refused():
    """Several Authorization headers are an ambiguous presentation, not the
    first one of them: the request-smuggling boundary refuses them outright."""
    request = Request(
        "GET",
        b"/",
        [
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"authorization", f"Bearer {WRONG_KEY}".encode()),
        ],
    )

    with pytest.raises(UnauthenticatedError):
        extract_credential(request)


def test_a_repeated_x_api_key_header_is_refused():
    request = Request(
        "GET",
        b"/",
        [
            (b"x-api-key", API_KEY.encode()),
            (b"x-api-key", WRONG_KEY.encode()),
        ],
    )

    with pytest.raises(UnauthenticatedError):
        extract_credential(request)


def test_both_headers_repeated_are_refused():
    """A repeated header is refused even when every copy carries the same value:
    the ambiguity is in the presentation, not in which value wins."""
    request = Request(
        "GET",
        b"/",
        [
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"authorization", f"Bearer {API_KEY}".encode()),
            (b"x-api-key", API_KEY.encode()),
            (b"x-api-key", API_KEY.encode()),
        ],
    )

    with pytest.raises(UnauthenticatedError):
        extract_credential(request)


@pytest.mark.parametrize(
    "value",
    ["Basic xyz", "Bearer", "Bearer ", "Bearer a b", "Bearer \t", ""],
)
def test_a_malformed_authorization_header_is_unauthenticated(value):
    """Not a 400: the caller's credential is wrong, not its request's shape."""
    request = Request("GET", b"/", [(b"authorization", value.encode())])

    with pytest.raises(UnauthenticatedError):
        extract_credential(request)


def test_no_credential_at_all_is_none():
    assert extract_credential(Request("GET", b"/", None)) is None


def test_a_blank_header_is_absent():
    assert extract_credential(Request("GET", b"/", [(b"x-api-key", b"   ")])) is None


# --------------------------------------------------------------------------------------
# Scope: which endpoints a credential opens
# --------------------------------------------------------------------------------------


async def test_a_valid_key_opens_every_v1_endpoint(client):
    responses = [
        await client.get("/v1/models", headers=auth_headers()),
        await client.get("/v1/models/text-default", headers=auth_headers()),
        await client.post(
            "/v1/chat/completions",
            content=JSONContent(CHAT_BODY),
            headers=auth_headers(),
        ),
    ]

    assert [response.status for response in responses] == [200, 200, 200]


async def test_a_missing_credential_is_refused_on_the_model_listing(client):
    response = await client.get("/v1/models")

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_missing_credential_is_refused_on_the_chat_endpoint(client):
    response = await client.post("/v1/chat/completions", content=JSONContent(CHAT_BODY))

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_wrong_credential_is_refused(client):
    response = await client.get("/v1/models", headers=auth_headers(WRONG_KEY))

    assert response.status == 401
    assert (await response.json())["reason"] == "invalid_credentials"


async def test_a_malformed_authorization_header_is_refused_as_unauthenticated(client):
    response = await client.get("/v1/models", headers={"Authorization": "Basic xyz"})

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_an_empty_bearer_token_is_refused_as_unauthenticated(client):
    response = await client.get("/v1/models", headers={"Authorization": "Bearer"})

    assert response.status == 401


async def test_x_api_key_is_accepted(client):
    response = await client.get("/v1/models", headers={"x-api-key": API_KEY})

    assert response.status == 200


async def test_conflicting_headers_are_refused(client):
    response = await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}", "x-api-key": WRONG_KEY},
    )

    assert response.status == 401


async def test_a_repeated_authorization_header_is_refused_on_the_wire(client):
    """The refusal holds on the wire, not just on the extraction function: the
    middleware must never resolve several Authorization headers by position."""
    response = await client.get(
        "/v1/models",
        headers=[
            (b"Authorization", f"Bearer {API_KEY}".encode()),
            (b"Authorization", f"Bearer {WRONG_KEY}".encode()),
        ],
    )

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_repeated_x_api_key_header_is_refused_on_the_wire(client):
    response = await client.get(
        "/v1/models",
        headers=[(b"x-api-key", API_KEY.encode()), (b"x-api-key", WRONG_KEY.encode())],
    )

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_both_headers_repeated_are_refused_on_the_wire(client):
    response = await client.get(
        "/v1/models",
        headers=[
            (b"Authorization", f"Bearer {API_KEY}".encode()),
            (b"Authorization", f"Bearer {WRONG_KEY}".encode()),
            (b"x-api-key", API_KEY.encode()),
            (b"x-api-key", WRONG_KEY.encode()),
        ],
    )

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_repeated_header_is_refused_even_on_an_exempt_path(client):
    """An exempt path still validates a presented credential — and a presented
    credential that is ambiguous is refused like any other."""
    response = await client.get(
        "/healthz",
        headers=[
            (b"Authorization", f"Bearer {API_KEY}".encode()),
            (b"Authorization", f"Bearer {WRONG_KEY}".encode()),
        ],
    )

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_rejected_key_is_refused_even_on_an_exempt_path(client):
    response = await client.get("/healthz", headers=auth_headers(WRONG_KEY))

    assert response.status == 401
    assert (await response.json())["reason"] == "invalid_credentials"


async def test_a_missing_credential_is_refused_on_the_favicon_path(client):
    """Authentication is a property of the middleware, not of the ``/v1``
    routes: a path no route claims is still authenticated."""
    response = await client.get("/favicon.ico")

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_missing_credential_is_refused_on_an_unknown_path(client):
    response = await client.get("/no/such/path")

    assert response.status == 401
    assert (await response.json())["reason"] == "authentication_required"


async def test_a_valid_credential_carries_an_unknown_path_past_authentication(client):
    """Routing decides only after authentication has let the request through,
    so an unknown path answers 404 rather than 401 once the credential is valid."""
    response = await client.get("/no/such/path", headers=auth_headers())

    assert response.status == 404


# --------------------------------------------------------------------------------------
# The health exemption, both ways
# --------------------------------------------------------------------------------------


async def test_health_is_open_without_a_credential_when_anonymous_health_is_allowed(client):
    responses = [await client.get("/healthz"), await client.get("/readyz")]

    assert [response.status for response in responses] == [200, 200]


async def test_health_requires_a_credential_when_anonymous_health_is_not_allowed(provider):
    app = build_app(
        StubKeySet({API_KEY: TENANT}),
        allow_anonymous_health=False,
        provider=provider,
    )
    async with running(app) as client:
        anonymous = await client.get("/healthz")
        authenticated = await client.get("/healthz", headers=auth_headers())
        wrong = await client.get("/healthz", headers=auth_headers(WRONG_KEY))

    assert anonymous.status == 401
    assert authenticated.status == 200
    assert wrong.status == 401


async def test_authentication_disabled_lets_everything_through(provider):
    app = build_app(StubKeySet({}), auth_enabled=False, provider=provider)
    async with running(app) as client:
        responses = [
            await client.get("/healthz"),
            await client.get("/v1/models"),
            await client.post("/v1/chat/completions", content=JSONContent(CHAT_BODY)),
        ]

    assert [response.status for response in responses] == [200, 200, 200]


# --------------------------------------------------------------------------------------
# The tenant rule
# --------------------------------------------------------------------------------------


async def test_a_matching_tenant_id_is_accepted(client, provider):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent({**CHAT_BODY, "tenant_id": TENANT}),
        headers=auth_headers(),
    )

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.tenant_id == TENANT


async def test_a_disagreeing_tenant_id_is_refused(client, provider):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent({**CHAT_BODY, "tenant_id": "tenant-other"}),
        headers=auth_headers(),
    )

    assert response.status == 403
    assert (await response.json())["reason"] == "permission_denied"
    assert provider.calls == []


async def test_an_absent_tenant_id_is_accepted(client, provider):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent(CHAT_BODY),
        headers=auth_headers(),
    )

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.tenant_id == TENANT


async def test_a_tenant_claimed_for_a_tenant_less_key_is_refused(provider):
    app = build_app(StubKeySet({API_KEY: None}), provider=provider)
    async with running(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": TENANT}),
            headers=auth_headers(),
        )

    assert response.status == 403
    assert provider.calls == []


async def test_an_unauthenticated_body_tenant_is_not_propagated(provider):
    """Authentication disabled: the body's tenant_id is not authoritative, so it is discarded."""
    app = build_app(StubKeySet({}), auth_enabled=False, provider=provider)
    async with running(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": TENANT}),
        )

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.tenant_id is None


async def test_the_handler_sees_the_identity_the_middleware_assigned(client, provider):
    response = await client.post(
        "/v1/chat/completions",
        content=JSONContent(CHAT_BODY),
        headers=auth_headers(),
    )

    assert response.status == 200
    _, invocation = provider.calls[0]
    assert invocation.tenant_id == TENANT


def test_get_identity_returns_none_when_the_middleware_did_not_run():
    assert get_identity(Request("GET", b"/", None)) is None


# --------------------------------------------------------------------------------------
# No secret leakage
# --------------------------------------------------------------------------------------


async def test_the_presented_key_appears_in_no_response_and_no_log(client, log_capture):
    responses = [
        await client.post(
            "/v1/chat/completions",
            content=JSONContent(CHAT_BODY),
            headers=auth_headers(),
        ),
        await client.get("/v1/models"),
        await client.get("/v1/models", headers=auth_headers(WRONG_KEY)),
        await client.get("/v1/models", headers={"Authorization": "Basic xyz"}),
        await client.get("/v1/models", headers={"Authorization": "Bearer"}),
        await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": "tenant-other"}),
            headers=auth_headers(),
        ),
    ]

    for response in responses:
        body = await response.text()
        assert API_KEY not in body
        assert WRONG_KEY not in body
    for message in log_capture.messages:
        assert API_KEY not in message
        assert WRONG_KEY not in message


# --------------------------------------------------------------------------------------
# The application factory wires the layer in
# --------------------------------------------------------------------------------------


async def test_the_factory_requires_a_credential_when_authentication_is_enabled(
    provider, clear_route_cache
):
    app = create_app(
        make_settings(),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(app) as client:
        anonymous = await client.get("/v1/models")
        authenticated = await client.get("/v1/models", headers=auth_headers())
        chat = await client.post(
            "/v1/chat/completions",
            content=JSONContent(CHAT_BODY),
            headers=auth_headers(),
        )
        health = await client.get("/healthz")

    assert anonymous.status == 401
    assert authenticated.status == 200
    assert chat.status == 200
    assert health.status == 200


async def test_the_factory_serves_openly_when_authentication_is_disabled(
    provider, clear_route_cache
):
    app = create_app(
        make_settings(GATEWAY_AUTH_ENABLED="false"),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(app) as client:
        responses = [
            await client.get("/v1/models"),
            await client.post("/v1/chat/completions", content=JSONContent(CHAT_BODY)),
        ]

    assert [response.status for response in responses] == [200, 200]


async def test_the_factory_refuses_a_body_tenant_that_disagrees_with_the_credential(
    provider, clear_route_cache
):
    app = create_app(
        make_settings(),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": UNBOUND_TENANT}),
            headers=auth_headers(),
        )

    assert response.status == 403
    assert (await response.json())["reason"] == "permission_denied"


async def test_the_factory_binds_a_key_to_its_configured_tenant(provider, clear_route_cache):
    """A ``<key>:<tenant>`` entry makes the credential's tenant the trusted one."""
    app = create_app(
        make_settings(GATEWAY_API_KEYS=f"{API_KEY}:{TENANT}"),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(app) as client:
        matching = await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": TENANT}),
            headers=auth_headers(),
        )
        disagreeing = await client.post(
            "/v1/chat/completions",
            content=JSONContent({**CHAT_BODY, "tenant_id": UNBOUND_TENANT}),
            headers=auth_headers(),
        )

    assert matching.status == 200
    assert disagreeing.status == 403
    assert (await disagreeing.json())["reason"] == "permission_denied"
    _, matching_invocation = provider.calls[0]
    assert matching_invocation.tenant_id == TENANT


async def test_the_factory_never_publishes_the_presented_key(
    provider, log_capture, clear_route_cache
):
    app = create_app(
        make_settings(),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(app) as client:
        responses = [
            await client.get("/v1/models", headers=auth_headers()),
            await client.get("/v1/models", headers=auth_headers(WRONG_KEY)),
            await client.get("/v1/models", headers={"Authorization": "Basic xyz"}),
            await client.post(
                "/v1/chat/completions",
                content=JSONContent({**CHAT_BODY, "tenant_id": UNBOUND_TENANT}),
                headers=auth_headers(),
            ),
        ]

    for response in responses:
        assert API_KEY not in await response.text()
        assert WRONG_KEY not in await response.text()
    for message in log_capture.messages:
        assert API_KEY not in message
        assert WRONG_KEY not in message


async def test_the_factory_health_exemption_follows_the_setting(provider, clear_route_cache):
    open_app = create_app(
        make_settings(GATEWAY_ALLOW_ANONYMOUS_HEALTH="true"),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(open_app) as open_client:
        open_health = await open_client.get("/healthz")

    default_router.get_match_by_method_and_path.cache_clear()

    closed_app = create_app(
        make_settings(GATEWAY_ALLOW_ANONYMOUS_HEALTH="false"),
        registry=make_registry(),
        service=InferenceService(make_registry(), provider),
    )
    async with running(closed_app) as closed_client:
        closed_health = await closed_client.get("/healthz")
        closed_health_authenticated = await closed_client.get("/healthz", headers=auth_headers())

    assert open_health.status == 200
    assert closed_health.status == 401
    assert closed_health_authenticated.status == 200
