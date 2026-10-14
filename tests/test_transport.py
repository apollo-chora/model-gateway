"""Tests for the outbound transport that pins a connection and disables redirects.

The transport's job is to refuse what the URL policy blocks and to reach nothing
else, so most of these tests are refusals and the proof has to land at the socket
layer. The pinning is a transport-neutral helper, :func:`pin_request`, so it is
tested once and directly: every refusal it raises and every field of the
rewritten request it returns is asserted on the helper's own result. Each
transport is a thin delegation over that helper, and two seams make the
delegation provable without an external network:

* The inner transport is injectable, so a test sees exactly what the pinning
  layer produced — the address it will connect to, the ``Host`` header, the TLS
  server name — and can assert that a refused request never reached it at all.
* The resolver is injectable, so a hostname resolves to anything a test wants,
  including a blocked address, without touching DNS.

One test per client factory opens a real socket: a local HTTP server on loopback
that counts the requests it receives. The policy blocks loopback by design, so a
count that stays at zero is connection-level evidence that the refusal happened
before anything was sent — and because the resolver points that hostname at the
running server, it is evidence that the policy, not an unreachable host, is
what stopped it.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from model_gateway.security.transport import (
    PinnedAddressTransport,
    SyncPinnedAddressTransport,
    attempt_timeout,
    build_async_client,
    build_sync_client,
    pin_request,
)
from model_gateway.security.url_policy import Resolver, UrlPolicyError

PUBLIC_ADDRESS = "93.184.216.34"
PUBLIC_ADDRESS_V6 = "2606:2800:220:1::1"
METADATA_ADDRESS = "169.254.169.254"


class RecordingTransport(httpx.AsyncBaseTransport):
    """An inner transport that answers without a socket and records what it was sent."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.closed = False
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._response is not None:
            return self._response
        return httpx.Response(200, json={"ok": True}, request=request)

    async def aclose(self) -> None:
        self.closed = True


class RecordingSyncTransport(httpx.BaseTransport):
    """The synchronous twin of :class:`RecordingTransport`."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.closed = False
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._response is not None:
            return self._response
        return httpx.Response(200, json={"ok": True}, request=request)

    def close(self) -> None:
        self.closed = True


def public_resolver(host: str) -> list[str]:
    """Stand-in for DNS: every hostname resolves to one public address."""
    return [PUBLIC_ADDRESS]


def resolver_for(addresses: dict[str, list[str]]) -> Resolver:
    def resolve(host: str) -> list[str]:
        return addresses[host]

    return resolve


def unreachable_resolver(host: str) -> list[str]:
    raise AssertionError(f"the resolver must not be called for {host}")


def recording_resolver(calls: list[str]) -> Resolver:
    def resolve(host: str) -> list[str]:
        calls.append(host)
        return [PUBLIC_ADDRESS]

    return resolve


@pytest.fixture
def local_server():
    """A real HTTP server on loopback that records the paths it was asked for."""
    received: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


# ------------------------------------------------------------------ real socket


def test_the_local_server_receives_a_request_that_is_sent_directly(local_server):
    port, received = local_server

    assert httpx.get(f"http://127.0.0.1:{port}/v1").status_code == 200
    assert received == ["/v1"]


async def test_a_blocked_address_is_refused_before_any_connection(local_server):
    port, received = local_server
    inner = RecordingTransport()

    async with build_async_client(
        budget=5.0,
        resolver=resolver_for({"provider.example": ["127.0.0.1"]}),
        allow_insecure_transport=True,
        inner=inner,
    ) as client:
        with pytest.raises(UrlPolicyError) as raised:
            await client.get(f"http://provider.example:{port}/v1")

    assert "blocked address" in str(raised.value)
    assert received == []
    assert inner.requests == []


def test_the_sync_client_refuses_a_blocked_address_before_any_connection(local_server):
    port, received = local_server
    inner = RecordingSyncTransport()

    with build_sync_client(
        budget=5.0,
        resolver=resolver_for({"provider.example": ["127.0.0.1"]}),
        allow_insecure_transport=True,
        inner=inner,
    ) as client, pytest.raises(UrlPolicyError) as raised:
        client.get(f"http://provider.example:{port}/v1")

    assert "blocked address" in str(raised.value)
    assert received == []
    assert inner.requests == []


# ------------------------------------------------- the pinning helper, direct


def test_a_hostname_resolving_to_a_public_and_a_blocked_address_is_refused():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "https://mixed.example/v1"),
            resolver=resolver_for({"mixed.example": [PUBLIC_ADDRESS, "127.0.0.1"]}),
        )

    assert "127.0.0.1" in str(raised.value)
    assert "blocked address" in str(raised.value)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "169.254.169.254",
        "::1",
        "fd00:ec2::254",
        "0.0.0.0",
    ],
)
def test_a_literal_blocked_address_is_refused(address):
    host = f"[{address}]" if ":" in address else address

    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", f"https://{host}/v1"), resolver=unreachable_resolver
        )

    assert "blocked address" in str(raised.value)


def test_a_hostname_that_resolves_to_nothing_is_refused():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "https://empty.example/v1"),
            resolver=resolver_for({"empty.example": []}),
        )

    assert "did not resolve" in str(raised.value)


def test_a_resolver_answer_that_is_not_an_address_is_refused():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "https://broken.example/v1"),
            resolver=resolver_for({"broken.example": ["not-an-ip"]}),
        )

    assert "not an IP address" in str(raised.value)


def test_plain_http_is_refused_unless_insecure_transport_is_enabled():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "http://provider.example/v1"), resolver=public_resolver
        )

    assert "insecure transport" in str(raised.value)


def test_plain_http_is_sent_when_insecure_transport_is_enabled():
    pinned = pin_request(
        httpx.Request("GET", "http://provider.example/v1"),
        resolver=public_resolver,
        allow_insecure_transport=True,
    )

    assert str(pinned.url) == f"http://{PUBLIC_ADDRESS}/v1"


def test_an_unsupported_scheme_is_refused():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "ftp://provider.example/v1"), resolver=public_resolver
        )

    assert "https is required" in str(raised.value)


def test_a_refusal_names_no_part_of_the_url():
    with pytest.raises(UrlPolicyError) as raised:
        pin_request(
            httpx.Request("GET", "https://metadata.internal/private/secret-token"),
            resolver=resolver_for({"metadata.internal": [METADATA_ADDRESS]}),
        )

    message = str(raised.value)
    assert "metadata.internal" in message
    assert "secret-token" not in message


def test_the_request_is_pinned_to_the_validated_address():
    pinned = pin_request(
        httpx.Request("GET", "https://provider.example/v1/models"), resolver=public_resolver
    )

    assert pinned.url.host == PUBLIC_ADDRESS


def test_the_pinned_url_keeps_the_scheme_port_path_and_query():
    pinned = pin_request(
        httpx.Request(
            "POST",
            "https://provider.example:8443/v1/chat/completions?api-version=2024-10-21",
        ),
        resolver=public_resolver,
    )

    assert (
        str(pinned.url)
        == f"https://{PUBLIC_ADDRESS}:8443/v1/chat/completions?api-version=2024-10-21"
    )


@pytest.mark.parametrize(
    ("url", "authority"),
    [
        ("https://provider.example/v1", "provider.example"),
        ("https://provider.example:8443/v1", "provider.example:8443"),
        ("https://provider.example:443/v1", "provider.example"),
    ],
)
def test_the_original_hostname_is_preserved_in_the_host_header(url, authority):
    pinned = pin_request(httpx.Request("GET", url), resolver=public_resolver)

    assert pinned.headers["host"] == authority


def test_the_original_hostname_is_used_for_tls():
    """``sni_hostname`` is what httpcore hands to TLS as ``server_hostname``."""
    pinned = pin_request(
        httpx.Request("GET", "https://provider.example/v1/models"), resolver=public_resolver
    )

    assert pinned.url.host == PUBLIC_ADDRESS
    assert pinned.extensions["sni_hostname"] == "provider.example"


def test_a_request_without_a_host_header_gets_the_original_authority():
    request = httpx.Request(
        "GET", "https://provider.example:8443/v1", stream=httpx.ByteStream(b"")
    )
    assert "host" not in request.headers

    pinned = pin_request(request, resolver=public_resolver)

    assert pinned.headers["host"] == "provider.example:8443"


def test_a_literal_public_address_is_pinned_without_being_resolved():
    pinned = pin_request(
        httpx.Request("GET", f"https://{PUBLIC_ADDRESS}/v1"),
        resolver=unreachable_resolver,
    )

    assert pinned.url.host == PUBLIC_ADDRESS


def test_an_ipv6_literal_is_pinned_with_its_brackets():
    pinned = pin_request(
        httpx.Request("GET", f"https://[{PUBLIC_ADDRESS_V6}]/v1"),
        resolver=unreachable_resolver,
    )

    assert str(pinned.url) == f"https://[{PUBLIC_ADDRESS_V6}]/v1"
    assert pinned.url.host == PUBLIC_ADDRESS_V6
    assert pinned.headers["host"] == f"[{PUBLIC_ADDRESS_V6}]"
    assert pinned.extensions["sni_hostname"] == PUBLIC_ADDRESS_V6


def test_the_request_body_and_headers_survive_the_pinning():
    pinned = pin_request(
        httpx.Request(
            "POST",
            "https://provider.example/v1/chat/completions",
            json={"prompt": "hello"},
            headers={"authorization": "Bearer example-secret"},
        ),
        resolver=public_resolver,
    )

    assert pinned.headers["authorization"] == "Bearer example-secret"
    assert pinned.headers["content-type"] == "application/json"
    assert pinned.headers["content-length"] == "18"
    pinned.read()
    assert pinned.content == b'{"prompt":"hello"}'


def test_the_request_extensions_survive_the_pinning():
    pinned = pin_request(
        httpx.Request(
            "GET", "https://provider.example/v1", extensions={"timeout": {"read": 7.0}}
        ),
        resolver=public_resolver,
    )

    assert pinned.extensions["timeout"]["read"] == 7.0


def test_the_injected_resolver_is_asked_for_the_hostname():
    calls: list[str] = []

    pin_request(
        httpx.Request("GET", "https://provider.example/v1"),
        resolver=recording_resolver(calls),
    )

    assert calls == ["provider.example"]


def test_a_hostname_the_system_cannot_resolve_is_reachable_through_the_resolver():
    pinned = pin_request(
        httpx.Request("GET", "https://provider.invalid/v1"),
        resolver=resolver_for({"provider.invalid": [PUBLIC_ADDRESS]}),
    )

    assert pinned.url.host == PUBLIC_ADDRESS


# ------------------------------------------------------------------ transports


def test_the_async_transport_is_an_httpx_transport():
    assert isinstance(
        PinnedAddressTransport(inner=RecordingTransport()), httpx.AsyncBaseTransport
    )


async def test_the_async_transport_delegates_the_pinned_request_to_the_inner_transport():
    inner = RecordingTransport()
    transport = PinnedAddressTransport(resolver=public_resolver, inner=inner)

    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get("https://provider.example/v1/models")).status_code == 200

    (request,) = inner.requests
    assert request.url.host == PUBLIC_ADDRESS
    assert request.headers["host"] == "provider.example"
    assert request.extensions["sni_hostname"] == "provider.example"


async def test_the_async_transport_refuses_before_the_inner_transport_is_reached():
    inner = RecordingTransport()
    transport = PinnedAddressTransport(resolver=public_resolver, inner=inner)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UrlPolicyError):
            await client.get("http://provider.example/v1")

    assert inner.requests == []


async def test_the_async_transport_closes_its_inner_transport():
    inner = RecordingTransport()
    transport = PinnedAddressTransport(resolver=public_resolver, inner=inner)

    await transport.aclose()

    assert inner.closed is True


def test_the_sync_transport_is_an_httpx_transport():
    assert isinstance(
        SyncPinnedAddressTransport(inner=RecordingSyncTransport()), httpx.BaseTransport
    )


def test_the_sync_transport_delegates_the_pinned_request_to_the_inner_transport():
    inner = RecordingSyncTransport()
    transport = SyncPinnedAddressTransport(resolver=public_resolver, inner=inner)

    with httpx.Client(transport=transport) as client:
        assert (
            client.post(
                "https://provider.example/v1/images/generations", json={"prompt": "hi"}
            ).status_code
            == 200
        )

    (request,) = inner.requests
    assert request.url.host == PUBLIC_ADDRESS
    assert request.headers["host"] == "provider.example"
    assert request.extensions["sni_hostname"] == "provider.example"


def test_the_sync_transport_refuses_before_the_inner_transport_is_reached():
    inner = RecordingSyncTransport()
    transport = SyncPinnedAddressTransport(resolver=public_resolver, inner=inner)

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(UrlPolicyError),
    ):
        client.post("https://127.0.0.1/v1/images/generations")

    assert inner.requests == []


def test_the_sync_transport_refuses_plain_http_unless_insecure_transport_is_enabled():
    inner = RecordingSyncTransport()
    transport = SyncPinnedAddressTransport(resolver=public_resolver, inner=inner)

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(UrlPolicyError),
    ):
        client.post("http://provider.example/v1/images/generations")

    assert inner.requests == []


def test_the_sync_transport_closes_its_inner_transport_with_itself():
    inner = RecordingSyncTransport()
    transport = SyncPinnedAddressTransport(resolver=public_resolver, inner=inner)

    transport.close()

    assert inner.closed


def test_the_sync_transport_enters_and_exits_with_its_inner_transport():
    inner = RecordingSyncTransport()
    transport = SyncPinnedAddressTransport(resolver=public_resolver, inner=inner)

    with transport:
        pass

    assert inner.closed


# ------------------------------------------------------------------ client factory


async def test_the_client_does_not_follow_redirects():
    client = build_async_client(budget=5.0, resolver=public_resolver, inner=RecordingTransport())
    try:
        assert client.follow_redirects is False
    finally:
        await client.aclose()


async def test_a_redirect_surfaces_to_the_caller_and_is_not_resolved():
    calls: list[str] = []
    inner = RecordingTransport(
        httpx.Response(302, headers={"location": "https://other.example/v1/models"})
    )

    async with build_async_client(
        budget=5.0, resolver=recording_resolver(calls), inner=inner
    ) as client:
        response = await client.post(
            "https://provider.example/v1/chat/completions", json={"prompt": "hello"}
        )

    assert response.status_code == 302
    assert response.headers["location"] == "https://other.example/v1/models"
    assert response.history == []
    assert len(inner.requests) == 1
    assert calls == ["provider.example"]


async def test_the_client_does_not_read_proxies_from_the_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    calls: list[str] = []
    inner = RecordingTransport()

    async with build_async_client(
        budget=5.0, resolver=recording_resolver(calls), inner=inner
    ) as client:
        await client.get("https://provider.example/v1")

    assert calls == ["provider.example"]
    assert str(inner.requests[0].url) == f"https://{PUBLIC_ADDRESS}/v1"


async def test_the_client_uses_the_pinned_transport():
    inner = RecordingTransport()

    async with build_async_client(
        budget=5.0,
        resolver=resolver_for({"metadata.internal": [METADATA_ADDRESS]}),
        inner=inner,
    ) as client:
        with pytest.raises(UrlPolicyError):
            await client.get("https://metadata.internal/v1")

    assert inner.requests == []


async def test_closing_the_client_closes_the_inner_transport():
    inner = RecordingTransport()
    client = build_async_client(budget=5.0, resolver=public_resolver, inner=inner)

    await client.aclose()

    assert inner.closed is True


async def test_the_client_carries_the_attempt_timeout():
    client = build_async_client(
        budget=12.0,
        connect_timeout=4.0,
        resolver=public_resolver,
        inner=RecordingTransport(),
    )
    try:
        assert client.timeout.connect == 4.0
        assert client.timeout.read == 12.0
    finally:
        await client.aclose()


def test_the_sync_client_does_not_follow_redirects():
    client = build_sync_client(
        budget=5.0, resolver=public_resolver, inner=RecordingSyncTransport()
    )
    try:
        assert client.follow_redirects is False
        assert client.trust_env is False
    finally:
        client.close()


def test_a_redirect_surfaces_through_the_sync_client_and_is_not_resolved():
    calls: list[str] = []
    inner = RecordingSyncTransport(
        httpx.Response(302, headers={"location": "https://other.example/v1/models"})
    )

    with build_sync_client(
        budget=5.0, resolver=recording_resolver(calls), inner=inner
    ) as client:
        response = client.post(
            "https://provider.example/v1/chat/completions", json={"prompt": "hello"}
        )

    assert response.status_code == 302
    assert response.headers["location"] == "https://other.example/v1/models"
    assert len(inner.requests) == 1
    assert calls == ["provider.example"]


def test_the_sync_client_does_not_read_proxies_from_the_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    calls: list[str] = []
    inner = RecordingSyncTransport()

    with build_sync_client(
        budget=5.0, resolver=recording_resolver(calls), inner=inner
    ) as client:
        client.get("https://provider.example/v1")

    assert calls == ["provider.example"]
    assert str(inner.requests[0].url) == f"https://{PUBLIC_ADDRESS}/v1"


def test_the_sync_client_uses_the_pinned_transport():
    inner = RecordingSyncTransport()

    with build_sync_client(
        budget=5.0,
        resolver=resolver_for({"metadata.internal": [METADATA_ADDRESS]}),
        inner=inner,
    ) as client, pytest.raises(UrlPolicyError):
        client.get("https://metadata.internal/v1")

    assert inner.requests == []


def test_closing_the_sync_client_closes_the_inner_transport():
    inner = RecordingSyncTransport()
    client = build_sync_client(budget=5.0, resolver=public_resolver, inner=inner)

    client.close()

    assert inner.closed is True


def test_the_sync_client_carries_the_attempt_timeout():
    client = build_sync_client(
        budget=12.0,
        connect_timeout=4.0,
        resolver=public_resolver,
        inner=RecordingSyncTransport(),
    )
    try:
        assert client.timeout.connect == 4.0
        assert client.timeout.read == 12.0
    finally:
        client.close()


# ------------------------------------------------------------------ timeout


def test_the_attempt_budget_covers_every_phase():
    timeout = attempt_timeout(30.0)

    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (30.0, 30.0, 30.0, 30.0)


def test_the_connect_timeout_caps_the_connect_phase():
    timeout = attempt_timeout(30.0, connect_timeout=10.0)

    assert timeout.connect == 10.0
    assert timeout.read == 30.0


def test_the_attempt_budget_caps_the_connect_timeout():
    timeout = attempt_timeout(3.0, connect_timeout=10.0)

    assert timeout.connect == 3.0
    assert timeout.read == 3.0


def test_an_absent_budget_leaves_the_client_without_a_deadline():
    timeout = attempt_timeout(None)

    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (None, None, None, None)


def test_an_absent_budget_still_applies_a_configured_connect_timeout():
    timeout = attempt_timeout(None, connect_timeout=10.0)

    assert timeout.connect == 10.0
    assert timeout.read is None


@pytest.mark.parametrize("budget", [0, -1, 0.0, float("inf"), float("nan"), True, "30"])
def test_a_budget_that_is_not_a_usable_number_of_seconds_is_refused(budget):
    with pytest.raises(ValueError):
        attempt_timeout(budget)


@pytest.mark.parametrize("connect_timeout", [0, -1, float("inf"), True, "10"])
def test_a_connect_timeout_that_is_not_a_usable_number_of_seconds_is_refused(connect_timeout):
    with pytest.raises(ValueError):
        attempt_timeout(30.0, connect_timeout=connect_timeout)
