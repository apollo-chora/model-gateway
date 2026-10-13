import pytest

from model_gateway.security import (
    RESERVED_AUTH_HEADERS,
    Resolver,
    UrlPolicyError,
    evaluate_redirect,
    is_blocked_address,
    is_reserved_auth_header,
    join_endpoint,
    resolve_endpoint,
    system_resolver,
    validate_upstream_url,
)

PUBLIC_ADDRESS = "93.184.216.34"
PUBLIC_ADDRESS_V6 = "2606:2800:220:1::1"
METADATA_ADDRESS = "169.254.169.254"


def public_resolver(host: str) -> list[str]:
    """Stand-in for DNS: every hostname resolves to one public address."""
    return [PUBLIC_ADDRESS]


def resolver_for(addresses: dict[str, list[str]]) -> Resolver:
    def resolve(host: str) -> list[str]:
        return addresses[host]

    return resolve


def unreachable_resolver(host: str) -> list[str]:
    raise AssertionError(f"the resolver must not be called for {host}")


def test_join_returns_the_base_url_when_no_path_is_configured():
    assert join_endpoint("https://provider.example/v1") == "https://provider.example/v1"


@pytest.mark.parametrize("path", [None, "", "   "])
def test_join_treats_a_blank_path_as_absent(path):
    assert join_endpoint("https://provider.example/v1", path) == "https://provider.example/v1"


def test_join_appends_a_configured_path_to_the_base_path():
    assert (
        join_endpoint("https://provider.example/v1", "/custom")
        == "https://provider.example/v1/custom"
    )


def test_join_appends_a_path_to_a_bare_base_url():
    assert join_endpoint("https://provider.example", "/custom") == "https://provider.example/custom"


def test_join_accepts_a_path_without_a_leading_slash():
    assert (
        join_endpoint("https://provider.example/v1", "custom")
        == "https://provider.example/v1/custom"
    )


@pytest.mark.parametrize(
    ("base_url", "path", "expected"),
    [
        ("https://provider.example/v1", "/v1", "https://provider.example/v1/v1"),
        (
            "https://provider.example/v1",
            "/v1/custom",
            "https://provider.example/v1/v1/custom",
        ),
        (
            "https://provider.example/v1",
            "/v1/chat/completions",
            "https://provider.example/v1/v1/chat/completions",
        ),
        (
            "https://provider.example/v1/chat/completions",
            "/chat/completions",
            "https://provider.example/v1/chat/completions/chat/completions",
        ),
        (
            "https://provider.example/v1/chat",
            "/chat/completions",
            "https://provider.example/v1/chat/chat/completions",
        ),
        (
            "https://provider.example/api/models",
            "/models/generate",
            "https://provider.example/api/models/models/generate",
        ),
    ],
)
def test_join_concatenates_path_segments_literally(base_url, path, expected):
    assert join_endpoint(base_url, path) == expected


def test_join_keeps_every_base_path_component():
    assert (
        join_endpoint("https://provider.example/openai/v1", "/custom")
        == "https://provider.example/openai/v1/custom"
    )


@pytest.mark.parametrize(
    ("base_url", "path", "expected"),
    [
        ("https://provider.example/", None, "https://provider.example"),
        ("https://provider.example/v1/", None, "https://provider.example/v1"),
        ("https://provider.example/v1/", "/custom/", "https://provider.example/v1/custom"),
        ("https://provider.example/v1", "/custom/", "https://provider.example/v1/custom"),
        ("https://provider.example/v1//", "/custom", "https://provider.example/v1/custom"),
    ],
)
def test_join_normalises_trailing_and_repeated_slashes(base_url, path, expected):
    assert join_endpoint(base_url, path) == expected


def test_join_preserves_the_port_and_does_not_apply_scheme_policy():
    assert join_endpoint("http://localhost:8080/v1", "/custom") == "http://localhost:8080/v1/custom"


def test_join_is_pure_and_deterministic():
    base_url = "https://provider.example/v1/"
    path = "/v1/custom/"
    first = join_endpoint(base_url, path)
    second = join_endpoint(base_url, path)
    assert first == second == "https://provider.example/v1/v1/custom"
    assert base_url == "https://provider.example/v1/"
    assert path == "/v1/custom/"


@pytest.mark.parametrize(
    "absolute_path",
    [
        "https://override.example/v1",
        "http://override.example/v1",
        "//override.example/v1",
        "//custom",
    ],
)
def test_join_refuses_an_absolute_url_in_the_path_by_default(absolute_path):
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint("https://provider.example/v1", absolute_path)
    assert "absolute" in str(raised.value)


def test_join_allows_an_absolute_url_in_the_path_only_with_the_opt_in():
    assert (
        join_endpoint(
            "https://provider.example/v1",
            "https://override.example/v2",
            allow_absolute_path_url=True,
        )
        == "https://override.example/v2"
    )


def test_join_still_requires_an_absolute_override_to_be_a_full_url():
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint(
            "https://provider.example/v1",
            "//override.example/v1",
            allow_absolute_path_url=True,
        )
    assert "absolute" in str(raised.value)


@pytest.mark.parametrize("base_url", ["", "   ", "/v1", "provider.example/v1"])
def test_join_rejects_a_base_url_that_is_not_absolute(base_url):
    with pytest.raises(UrlPolicyError):
        join_endpoint(base_url)


@pytest.mark.parametrize(
    ("base_url", "path"),
    [
        ("https://provider.example/v1?api-version=1", None),
        ("https://provider.example/v1#fragment", None),
        ("https://provider.example/v1", "/custom?api-version=1"),
        ("https://provider.example/v1", "/custom#fragment"),
    ],
)
def test_join_rejects_a_query_string_or_fragment(base_url, path):
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint(base_url, path)
    assert "query string or fragment" in str(raised.value)


@pytest.mark.parametrize(
    ("base_url", "path"),
    [
        ("https://provider.example/v1/../v2", None),
        ("https://provider.example/v1/./v2", None),
        ("https://provider.example/v1", "/custom/../v2"),
    ],
)
def test_join_rejects_dot_segments(base_url, path):
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint(base_url, path)
    assert "'.'" in str(raised.value) or "'..'" in str(raised.value)


def test_join_does_not_echo_a_secret_from_the_path_setting():
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint("https://provider.example/v1", "/custom?token=supersecret")
    assert "supersecret" not in str(raised.value)


def test_resolve_endpoint_joins_and_validates_in_one_step():
    assert (
        resolve_endpoint("https://provider.example/v1", "/custom", resolver=public_resolver)
        == "https://provider.example/v1/custom"
    )


def test_resolve_endpoint_refuses_plain_http_unless_insecure_transport_is_enabled():
    with pytest.raises(UrlPolicyError) as raised:
        resolve_endpoint("http://provider.example/v1", resolver=public_resolver)
    assert "insecure transport" in str(raised.value)

    assert (
        resolve_endpoint(
            "http://provider.example/v1",
            allow_insecure_transport=True,
            resolver=public_resolver,
        )
        == "http://provider.example/v1"
    )


def test_resolve_endpoint_revalidates_the_joined_endpoint():
    with pytest.raises(UrlPolicyError) as raised:
        resolve_endpoint(
            "https://metadata.internal/v1",
            "/custom",
            resolver=resolver_for({"metadata.internal": [METADATA_ADDRESS]}),
        )
    assert "blocked address" in str(raised.value)


def test_validate_accepts_https():
    assert validate_upstream_url("https://provider.example/v1", resolver=public_resolver) is None


def test_validate_accepts_an_upper_case_https_scheme():
    assert validate_upstream_url("HTTPS://provider.example/v1", resolver=public_resolver) is None


def test_validate_accepts_plain_http_only_with_the_insecure_opt_in():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url("http://provider.example/v1", resolver=public_resolver)
    assert "insecure transport" in str(raised.value)

    assert (
        validate_upstream_url(
            "http://provider.example/v1",
            allow_insecure_transport=True,
            resolver=public_resolver,
        )
        is None
    )


@pytest.mark.parametrize(
    ("url", "scheme"),
    [("ftp://provider.example/v1", "ftp"), ("ws://provider.example/v1", "ws")],
)
def test_validate_rejects_every_other_scheme(url, scheme):
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(url)
    assert "https is required" in str(raised.value)
    assert scheme in str(raised.value)


def test_validate_rejects_a_url_without_a_scheme_or_host():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url("file:///etc/passwd")
    assert "absolute URL" in str(raised.value)


def test_validate_rejects_an_invalid_port():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url("https://provider.example:notaport/v1")
    assert "port" in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://leakyuser@provider.example/v1",
        "https://:leakysecret@provider.example/v1",
        "https://leakyuser:leakysecret@provider.example/v1",
    ],
)
def test_validate_rejects_url_userinfo_without_echoing_it(url):
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(url, resolver=public_resolver)
    message = str(raised.value)
    assert "credentials" in message
    assert "leakyuser" not in message
    assert "leakysecret" not in message


def test_validate_rejects_userinfo_in_an_absolute_path_override():
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint(
            "https://provider.example/v1",
            "https://leakyuser:leakysecret@other.example/v1",
            allow_absolute_path_url=True,
        )
    message = str(raised.value)
    assert "credentials" in message
    assert "leakysecret" not in message


def test_join_rejects_a_base_url_with_embedded_credentials():
    with pytest.raises(UrlPolicyError) as raised:
        join_endpoint("https://leakyuser:leakysecret@provider.example/v1")
    message = str(raised.value)
    assert "credentials" in message
    assert "leakysecret" not in message


def test_scheme_error_does_not_echo_the_url():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            "http://provider.example/private/secret-token",
            resolver=public_resolver,
        )
    message = str(raised.value)
    assert "insecure transport" in message
    assert "secret-token" not in message


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.1.2.3",
        "::1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.10",
        "169.254.1.1",
        "169.254.169.254",
        "fd00:ec2::254",
        "fe80::1",
        "239.1.1.1",
        "ff02::1",
        "240.0.0.1",
        "0.0.0.0",
        "::",
        "100.64.0.1",
        "192.0.2.1",
        "::ffff:127.0.0.1",
    ],
)
def test_validate_blocks_every_reserved_address_class(address):
    host = f"[{address}]" if ":" in address else address
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(f"https://{host}/v1", resolver=unreachable_resolver)
    assert "blocked address" in str(raised.value)


def test_validate_names_the_cloud_metadata_address_class():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(f"https://{METADATA_ADDRESS}/v1", resolver=unreachable_resolver)
    message = str(raised.value)
    assert METADATA_ADDRESS in message
    assert "link-local" in message


def test_validate_resolves_a_hostname_through_the_injected_resolver():
    calls: list[str] = []

    def resolve(host: str) -> list[str]:
        calls.append(host)
        return [PUBLIC_ADDRESS]

    assert validate_upstream_url("https://provider.example/v1", resolver=resolve) is None
    assert calls == ["provider.example"]


def test_validate_blocks_a_hostname_that_resolves_to_a_blocked_address():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            "https://metadata.internal/v1",
            resolver=resolver_for({"metadata.internal": [METADATA_ADDRESS]}),
        )
    message = str(raised.value)
    assert "metadata.internal" in message
    assert "link-local" in message


def test_validate_checks_every_resolved_address():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            "https://mixed.example/v1",
            resolver=resolver_for({"mixed.example": [PUBLIC_ADDRESS, "127.0.0.1"]}),
        )
    assert "127.0.0.1" in str(raised.value)


def test_validate_accepts_a_hostname_that_resolves_only_to_public_addresses():
    assert (
        validate_upstream_url(
            "https://public.example/v1",
            resolver=resolver_for({"public.example": [PUBLIC_ADDRESS, PUBLIC_ADDRESS_V6]}),
        )
        is None
    )


def test_validate_never_resolves_a_literal_address():
    assert (
        validate_upstream_url(f"https://{PUBLIC_ADDRESS}/v1", resolver=unreachable_resolver) is None
    )
    assert (
        validate_upstream_url(f"https://[{PUBLIC_ADDRESS_V6}]/v1", resolver=unreachable_resolver)
        is None
    )


def test_validate_rejects_a_host_that_resolves_to_nothing():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            "https://empty.example/v1",
            resolver=resolver_for({"empty.example": []}),
        )
    assert "did not resolve" in str(raised.value)


def test_validate_rejects_a_resolver_result_that_is_not_an_address():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            "https://broken.example/v1",
            resolver=resolver_for({"broken.example": ["not-an-ip"]}),
        )
    assert "not an IP address" in str(raised.value)


def test_validate_rejects_a_missing_host():
    with pytest.raises(UrlPolicyError):
        validate_upstream_url("https:///v1")


def test_ssrf_error_names_the_host_without_echoing_the_url():
    with pytest.raises(UrlPolicyError) as raised:
        validate_upstream_url(
            f"https://{METADATA_ADDRESS}/private/secret-token",
            resolver=unreachable_resolver,
        )
    message = str(raised.value)
    assert METADATA_ADDRESS in message
    assert "secret-token" not in message


def test_system_resolver_returns_a_literal_address_without_network_access():
    assert system_resolver("127.0.0.1") == ("127.0.0.1",)


@pytest.mark.parametrize(
    ("address", "blocked"),
    [
        ("127.0.0.1", True),
        ("10.1.2.3", True),
        ("169.254.169.254", True),
        ("::1", True),
        ("fc00::1", True),
        ("224.0.0.1", True),
        ("0.0.0.0", True),
        ("93.184.216.34", False),
        ("2606:2800:220:1::1", False),
    ],
)
def test_is_blocked_address_classifies_an_address(address, blocked):
    assert is_blocked_address(address) is blocked


def test_is_blocked_address_rejects_a_non_address():
    with pytest.raises(UrlPolicyError):
        is_blocked_address("not-an-ip")


def test_redirect_within_one_origin_forwards_credentials():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "https://provider.example/v1/models",
        resolver=public_resolver,
    )
    assert decision.allowed is True
    assert decision.forward_credentials is True


def test_redirect_to_an_explicit_default_port_is_still_the_same_origin():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "https://provider.example:443/v1/models",
        resolver=public_resolver,
    )
    assert decision.forward_credentials is True


@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/v1/models",
        "https://provider.example:8443/v1/models",
        "http://provider.example/v1/models",
    ],
)
def test_redirect_across_an_origin_change_drops_credentials(target):
    decision = evaluate_redirect(
        "https://provider.example/v1",
        target,
        allow_insecure_transport=True,
        resolver=public_resolver,
    )
    assert decision.allowed is True
    assert decision.forward_credentials is False
    assert "origin changed" in decision.reason


def test_redirect_refuses_an_insecure_downgrade():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "http://provider.example/v1/models",
        resolver=public_resolver,
    )
    assert decision.allowed is False
    assert decision.forward_credentials is False


def test_redirect_revalidates_the_destination_host():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        f"https://{METADATA_ADDRESS}/v1",
        resolver=unreachable_resolver,
    )
    assert decision.allowed is False
    assert decision.forward_credentials is False
    assert "blocked address" in decision.reason


def test_redirect_revalidates_the_destination_host_through_the_resolver():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "https://metadata.internal/v1",
        resolver=resolver_for({"metadata.internal": ["10.0.0.5"]}),
    )
    assert decision.allowed is False
    assert "blocked address" in decision.reason


def test_redirect_refuses_a_relative_destination():
    decision = evaluate_redirect("https://provider.example/v1", "/v1/models")
    assert decision.allowed is False
    assert "absolute URL" in decision.reason


def test_redirect_refuses_a_destination_carrying_a_query_string():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "https://provider.example/v1?x=1",
        resolver=public_resolver,
    )
    assert decision.allowed is False


def test_redirect_refuses_a_destination_with_embedded_credentials():
    decision = evaluate_redirect(
        "https://provider.example/v1",
        "https://leakyuser:leakysecret@provider.example/v1/models",
        resolver=public_resolver,
    )
    assert decision.allowed is False
    assert decision.forward_credentials is False
    assert "credentials" in decision.reason
    assert "leakysecret" not in decision.reason


def test_redirect_requires_a_valid_source_url():
    with pytest.raises(UrlPolicyError):
        evaluate_redirect("/v1", "https://provider.example/v1")


def test_reserved_auth_headers_are_exposed():
    assert {"authorization", "x-api-key", "api-key"} == RESERVED_AUTH_HEADERS


@pytest.mark.parametrize(
    "name",
    ["Authorization", "authorization", "AUTHORIZATION", "x-api-key", "X-Api-Key", "api-key"],
)
def test_reserved_auth_header_names_are_recognised_case_insensitively(name):
    assert is_reserved_auth_header(name) is True


@pytest.mark.parametrize("name", ["content-type", "x-request-id", "openai-organization", ""])
def test_ordinary_header_names_are_not_reserved(name):
    assert is_reserved_auth_header(name) is False
