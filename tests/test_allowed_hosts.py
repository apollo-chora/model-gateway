"""The optional upstream hostname allowlist (PRD section 14).

Section 14 lists an upstream hostname allowlist among the controls the gateway must
implement, with configuring one optional. Two properties make it worth having and
both are asserted here: it **restricts what may be configured**, so a route added
later that reaches an unlisted provider fails the start rather than being discovered
in a bill; and it **narrows and never widens**, so a host on the allowlist still has
to pass the address screen and naming one cannot make a blocked address reachable.

The second property is the one that matters for safety. An allowlist that could
exempt a host from the address screen would be a second security policy with
exceptions, which is a worse thing to have than no allowlist at all.
"""

import pytest

from model_gateway.config import ConfigurationError
from model_gateway.config.routes import parse_routes
from model_gateway.security.url_policy import (
    ALLOWED_HOSTS_VARIABLE,
    UrlPolicyError,
    check_allowed_host,
    parse_allowed_hosts,
    resolve_endpoint,
)

UPSTREAM = "93.184.216.34"
"""A literal public address, so the address screen needs no DNS to accept it."""

ENDPOINT = f"https://{UPSTREAM}/v1"

TEXT_BLOCK = {
    "TEXT_LLM_ENDPOINT": ENDPOINT,
    "TEXT_LLM_MODEL_ID": "upstream-text",
    "TEXT_LLM_API_KEY": "sk-upstream-secret",
    "TEXT_LLM_FORMAT": "openai",
}


# --------------------------------------------------------------------------- #
# Reading the setting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, frozenset()),
        ("", frozenset()),
        ("   ", frozenset()),
        ("one.example", frozenset({"one.example"})),
        ("one.example,two.example", frozenset({"one.example", "two.example"})),
        (" One.Example , TWO.example ", frozenset({"one.example", "two.example"})),
        ("one.example,,two.example", frozenset({"one.example", "two.example"})),
    ],
)
def test_a_hostname_list_is_read_into_the_hosts_it_names(raw, expected):
    """Empty is "no allowlist", not "allow nothing".

    A deployment that never set the variable must keep the behaviour it had before
    the variable existed, so an empty value is the absence of the restriction rather
    than a restriction that refuses everything.
    """
    assert parse_allowed_hosts(raw) == expected


@pytest.mark.parametrize(
    "entry",
    [
        "https://one.example",
        "one.example:443",
        "one.example/path",
        "user@one.example",
        "one.example two.example",
        "[2001:db8::1]",
        "2001:db8::1",
    ],
)
def test_an_entry_that_is_not_a_hostname_is_refused(entry):
    """An entry the gateway silently ignored would be an operator misled either way.

    An entry is a hostname, so a scheme, a port, a path, userinfo and an IPv6 literal
    are refused rather than kept as text that can never match a host.
    """
    with pytest.raises(UrlPolicyError, match=ALLOWED_HOSTS_VARIABLE):
        parse_allowed_hosts(entry)


# --------------------------------------------------------------------------- #
# Applying it
# --------------------------------------------------------------------------- #


def test_no_allowlist_permits_every_host_the_rest_of_the_policy_permits():
    check_allowed_host("anything.example", frozenset())


@pytest.mark.parametrize(
    "entry",
    [
        "-leading.example",
        "trailing-.example",
        "under_score.example",
        "two..dots",
        "a" * 64 + ".example",
        "x" * 254,
    ],
)
def test_an_entry_outside_the_hostname_grammar_is_refused(entry):
    """A typo kept in the list looks like a list that works, so it is refused.

    The check above catches an authority written where a hostname belongs; this
    catches a value that is authority-shaped but not a hostname at all. Either way
    the operator is told at startup rather than discovering it when a route they
    believed was permitted is refused.
    """
    with pytest.raises(UrlPolicyError, match=ALLOWED_HOSTS_VARIABLE):
        parse_allowed_hosts(entry)


@pytest.mark.parametrize("entry", ["one.example", "localhost", "93.184.216.34", "a-b.c-d.example"])
def test_an_entry_inside_the_hostname_grammar_is_accepted(entry):
    assert parse_allowed_hosts(entry) == frozenset({entry})


def test_a_named_host_is_permitted_and_matching_ignores_case():
    check_allowed_host("One.Example", frozenset({"one.example"}))


def test_an_unnamed_host_is_refused_and_the_refusal_names_it():
    with pytest.raises(UrlPolicyError) as refusal:
        check_allowed_host("unlisted.example", frozenset({"one.example", "two.example"}))

    assert "unlisted.example" in str(refusal.value)


def test_a_listed_host_still_has_to_pass_the_address_screen():
    """The property that makes the allowlist safe: it narrows, it never widens.

    ``127.0.0.1`` is named by the allowlist and is still refused, because the address
    screen is not the allowlist's to waive. An allowlist that could exempt a host
    from it would be a second security policy with exceptions.
    """
    with pytest.raises(UrlPolicyError, match="blocked address"):
        resolve_endpoint("https://127.0.0.1/v1", allowed_hosts=frozenset({"127.0.0.1"}))


def test_a_listed_endpoint_resolves_and_an_unlisted_one_is_refused():
    assert (
        resolve_endpoint(ENDPOINT, allowed_hosts=frozenset({UPSTREAM})) == ENDPOINT
    )

    with pytest.raises(UrlPolicyError):
        resolve_endpoint(ENDPOINT, allowed_hosts=frozenset({"other.example"}))


# --------------------------------------------------------------------------- #
# Through the route parser, which is where configuration is validated
# --------------------------------------------------------------------------- #


def test_a_route_inside_the_allowlist_is_accepted():
    routes = parse_routes({**TEXT_BLOCK, ALLOWED_HOSTS_VARIABLE: UPSTREAM})

    assert [route.base_url for route in routes] == [ENDPOINT]


def _refusal(environment) -> str:
    """The message a configuration is refused with.

    Asserting on the message rather than only on the exception type is what makes
    these tests about the allowlist: the endpoint variable is named by the wrapper
    around *every* URL-policy refusal, so a test that matched only on it would pass
    for a blocked address just as happily.
    """
    with pytest.raises(ConfigurationError) as refusal:
        parse_routes(environment)
    return str(refusal.value)


def test_a_route_outside_the_allowlist_fails_the_start_naming_its_variable():
    message = _refusal({**TEXT_BLOCK, ALLOWED_HOSTS_VARIABLE: "other.example"})

    assert ALLOWED_HOSTS_VARIABLE in message
    assert "TEXT_LLM_ENDPOINT" in message


def test_no_allowlist_leaves_the_configuration_as_it_was():
    assert [route.base_url for route in parse_routes(dict(TEXT_BLOCK))] == [ENDPOINT]


def test_an_additional_route_outside_the_allowlist_fails_the_start_naming_the_entry():
    """The allowlist covers every route, including the ones JSON adds.

    A restriction that applied only to the primary route would be one an operator
    could add a fallback around without noticing.
    """
    environment = {
        **TEXT_BLOCK,
        ALLOWED_HOSTS_VARIABLE: UPSTREAM,
        "TEXT_LLM_ROUTES_JSON": f'[{{"endpoint": "https://{UPSTREAM}:8443/v1", '
        f'"model_id": "fallback", "api_key": "sk-x", "format": "openai"}}]',
    }

    # The authority is the endpoint's host, so a different port is the same host and
    # is permitted: the allowlist names hosts, not authorities.
    assert len(parse_routes(environment)) == 2

    environment["TEXT_LLM_ROUTES_JSON"] = (
        '[{"endpoint": "https://93.184.216.35/v1", "model_id": "fallback", '
        '"api_key": "sk-x", "format": "openai"}]'
    )
    message = _refusal(environment)

    assert ALLOWED_HOSTS_VARIABLE in message
    assert "TEXT_LLM_ROUTES_JSON" in message


def test_an_absolute_path_override_outside_the_allowlist_fails_the_start():
    """The override is checked against the allowlist, because it is checked as a URL.

    An override is the one way a route's authority can differ from its configured
    endpoint, so an allowlist that missed it would be a documented control with a
    documented way around it.
    """
    message = _refusal(
        {
            **TEXT_BLOCK,
            ALLOWED_HOSTS_VARIABLE: UPSTREAM,
            "TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL": "true",
            "TEXT_LLM_PATH": "https://93.184.216.35/v1/generate",
        }
    )

    assert ALLOWED_HOSTS_VARIABLE in message
    assert "TEXT_LLM_PATH" in message


def test_an_absolute_path_override_inside_the_allowlist_is_accepted():
    environment = {
        **TEXT_BLOCK,
        ALLOWED_HOSTS_VARIABLE: f"{UPSTREAM},93.184.216.35",
        "TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL": "true",
        "TEXT_LLM_PATH": "https://93.184.216.35/v1/generate",
    }

    assert [route.base_url for route in parse_routes(environment)] == [
        "https://93.184.216.35/v1/generate"
    ]


def test_a_malformed_allowlist_fails_the_start():
    with pytest.raises(UrlPolicyError, match=ALLOWED_HOSTS_VARIABLE):
        parse_routes({**TEXT_BLOCK, ALLOWED_HOSTS_VARIABLE: "https://one.example"})
