"""Security policy for upstream traffic (PRD sections 7, 8, 9.2 and 14).

Endpoint joining, scheme and SSRF validation, redirect and credential rules,
and the reserved-authentication-header rule live here so that configuration
validation and the LiteLLM binding apply exactly one policy.

These checks screen configuration; they are not by themselves a guarantee of
SSRF prevention. The transport that performs the outbound request must pin the
validated address and disable automatic redirect following, so that every hop
passes through ``evaluate_redirect``. See the transport contract in
``model_gateway.security.url_policy``.
"""

from model_gateway.security.url_policy import (
    RESERVED_AUTH_HEADERS,
    RedirectDecision,
    Resolver,
    UrlPolicyError,
    check_host,
    evaluate_redirect,
    is_blocked_address,
    is_reserved_auth_header,
    join_endpoint,
    resolve_endpoint,
    system_resolver,
    validate_upstream_url,
)

__all__ = [
    "RESERVED_AUTH_HEADERS",
    "RedirectDecision",
    "Resolver",
    "UrlPolicyError",
    "check_host",
    "evaluate_redirect",
    "is_blocked_address",
    "is_reserved_auth_header",
    "join_endpoint",
    "resolve_endpoint",
    "system_resolver",
    "validate_upstream_url",
]
