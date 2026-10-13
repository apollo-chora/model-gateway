"""Gateway error taxonomy: categories, payloads and sanitisation (PRD 5.4, 9.3, 15)."""

import ast
import re
from pathlib import Path

import pytest

from model_gateway import errors as errors_module
from model_gateway.errors import (
    CapabilityError,
    DeadlineExceededError,
    ErrorReason,
    FailedPreconditionError,
    GatewayError,
    GrpcStatus,
    InternalError,
    InvalidArgumentError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
    normalise_exception,
    sanitise_text,
)

# Every category in PRD section 5.4, plus the capability category of PRD 9.3.
CATEGORY_TABLE = (
    (InvalidArgumentError, 400, GrpcStatus.INVALID_ARGUMENT, ErrorReason.INVALID_REQUEST),
    (UnauthenticatedError, 401, GrpcStatus.UNAUTHENTICATED, ErrorReason.AUTHENTICATION_REQUIRED),
    (PermissionDeniedError, 403, GrpcStatus.PERMISSION_DENIED, ErrorReason.PERMISSION_DENIED),
    (NotFoundError, 404, GrpcStatus.NOT_FOUND, ErrorReason.RESOURCE_NOT_FOUND),
    (ModelNotFoundError, 404, GrpcStatus.NOT_FOUND, ErrorReason.MODEL_NOT_FOUND),
    (ResourceExhaustedError, 429, GrpcStatus.RESOURCE_EXHAUSTED, ErrorReason.RATE_LIMITED),
    (FailedPreconditionError, 412, GrpcStatus.FAILED_PRECONDITION, ErrorReason.FAILED_PRECONDITION),
    (DeadlineExceededError, 504, GrpcStatus.DEADLINE_EXCEEDED, ErrorReason.DEADLINE_EXCEEDED),
    (UnavailableError, 503, GrpcStatus.UNAVAILABLE, ErrorReason.PROVIDER_UNAVAILABLE),
    (InternalError, 500, GrpcStatus.INTERNAL, ErrorReason.INTERNAL_ERROR),
    (CapabilityError, 412, GrpcStatus.FAILED_PRECONDITION, ErrorReason.UNSUPPORTED_CAPABILITY),
)

CATEGORY_IDS = [error_class.__name__ for error_class, *_ in CATEGORY_TABLE]

# The published reason vocabulary. Pinned so that a rename is a test failure
# rather than a silent break of the machine-readable contract.
STABLE_REASON_VALUES = frozenset(
    {
        "invalid_request",
        "authentication_required",
        "invalid_credentials",
        "permission_denied",
        "model_not_found",
        "resource_not_found",
        "rate_limited",
        "concurrency_limit_exceeded",
        "failed_precondition",
        "deadline_exceeded",
        "provider_unavailable",
        "internal_error",
        "unsupported_capability",
    }
)

GRPC_STATUS_NAMES = frozenset(
    {
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "DEADLINE_EXCEEDED",
        "UNAVAILABLE",
        "INTERNAL",
    }
)

UPSTREAM_URL = "https://api.provider.example/v1/chat/completions"
CREDENTIAL = "sk-live-9f8e7d6c5b4a"
STACK_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/usr/lib/python3.13/site-packages/litellm/main.py", line 42, in completion\n'
    "    return await provider.acompletion(**kwargs)\n"
    "litellm.exceptions.APIError: upstream refused the request"
)


class ProviderFailure(Exception):
    """Stands in for a provider-shaped exception raised by the inference binding."""


def _all_subclasses(cls):
    found = set()
    for subclass in cls.__subclasses__():
        found.add(subclass)
        found |= _all_subclasses(subclass)
    return found


# --- categories ------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "http_status", "grpc_code", "reason"),
    CATEGORY_TABLE,
    ids=CATEGORY_IDS,
)
def test_every_category_maps_to_its_transport_status(
    error_class, http_status, grpc_code, reason
):
    error = error_class("the request was rejected")

    assert error.http_status == http_status
    assert error.grpc_code is grpc_code
    assert error.grpc_code == grpc_code.value
    assert error.reason is reason


@pytest.mark.parametrize(("error_class", "http_status", "grpc_code", "reason"), CATEGORY_TABLE,
                         ids=CATEGORY_IDS)
def test_every_category_subclasses_the_base_error(error_class, http_status, grpc_code, reason):
    error = error_class("the request was rejected")

    assert isinstance(error, GatewayError)
    assert isinstance(error, Exception)
    assert issubclass(error_class, GatewayError)


@pytest.mark.parametrize(("error_class", "http_status", "grpc_code", "reason"), CATEGORY_TABLE,
                         ids=CATEGORY_IDS)
def test_every_category_can_be_raised_and_caught_as_a_gateway_error(
    error_class, http_status, grpc_code, reason
):
    with pytest.raises(GatewayError):
        raise error_class("the request was rejected")


@pytest.mark.parametrize(("error_class", "http_status", "grpc_code", "reason"), CATEGORY_TABLE,
                         ids=CATEGORY_IDS)
def test_every_category_has_a_distinct_reason(error_class, http_status, grpc_code, reason):
    reasons = [entry[3] for entry in CATEGORY_TABLE]

    assert len(reasons) == len(set(reasons))
    assert error_class("x").reason is reason


def test_every_gateway_error_subclass_is_covered_by_the_category_table():
    assert _all_subclasses(GatewayError) == {error_class for error_class, *_ in CATEGORY_TABLE}


def test_the_base_error_fails_closed_onto_internal():
    error = GatewayError("something went wrong")

    assert error.http_status == 500
    assert error.grpc_code is GrpcStatus.INTERNAL
    assert error.reason is ErrorReason.INTERNAL_ERROR


def test_every_http_status_is_a_plausible_status_code():
    for _, http_status, _, _ in CATEGORY_TABLE:
        assert 400 <= http_status <= 599


def test_every_category_uses_one_of_the_standard_grpc_codes():
    assert {grpc_code for _, _, grpc_code, _ in CATEGORY_TABLE} == set(GrpcStatus)


def test_grpc_codes_are_the_standard_names():
    assert {status.value for status in GrpcStatus} == GRPC_STATUS_NAMES
    assert {status.name for status in GrpcStatus} == GRPC_STATUS_NAMES


def test_grpc_status_values_are_upper_snake_case():
    for status in GrpcStatus:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", status.value)


def test_the_module_declares_no_third_party_dependency():
    source = Path(errors_module.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "grpc" not in imported
    assert imported <= {"__future__", "collections", "enum", "re", "types", "typing"}


# --- reasons ---------------------------------------------------------------


def test_reason_values_are_stable():
    assert {reason.value for reason in ErrorReason} == STABLE_REASON_VALUES
    assert {reason.name for reason in ErrorReason} == {
        value.upper() for value in STABLE_REASON_VALUES
    }


def test_reasons_are_unique_and_alias_free():
    values = [reason.value for reason in ErrorReason]

    assert len(values) == len(set(values))
    assert len(list(ErrorReason)) == len(ErrorReason.__members__)


def test_reason_values_are_machine_readable_tokens():
    for reason in ErrorReason:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", reason.value)


def test_reasons_are_not_free_form_sentences():
    for reason in ErrorReason:
        assert " " not in reason.value
        assert reason.value == reason.value.lower()


def test_a_reason_narrows_the_category_it_was_raised_in():
    error = UnauthenticatedError(
        "the presented key is not recognised", reason=ErrorReason.INVALID_CREDENTIALS
    )

    assert error.reason is ErrorReason.INVALID_CREDENTIALS
    assert error.http_status == 401
    assert error.to_payload()["reason"] == "invalid_credentials"


def test_an_unknown_reason_is_rejected():
    with pytest.raises(ValueError):
        InternalError("boom", reason="not_a_reason")


def test_reason_defaults_come_from_the_class():
    assert InvalidArgumentError("x").reason is ErrorReason.INVALID_REQUEST
    assert ModelNotFoundError("x").reason is ErrorReason.MODEL_NOT_FOUND
    assert ModelNotFoundError("x").reason is not NotFoundError("x").reason


# --- payload shape ---------------------------------------------------------


def test_payload_carries_reason_and_message_only():
    error = ResourceExhaustedError(
        "rate limit exceeded for tenant 'acme'", details={"retry_after_seconds": "5"}
    )

    payload = error.to_payload()

    assert set(payload) == {"reason", "message"}
    assert payload["reason"] == "rate_limited"
    assert payload["message"] == "rate limit exceeded for tenant 'acme'"
    assert payload["reason"] == error.reason.value
    assert payload["message"] == error.message


def test_payload_is_json_serialisable_plain_data():
    payload = InternalError("boom", details={"stage": "routing"}).to_payload(
        include_details=True
    )

    assert isinstance(payload["reason"], str)
    assert isinstance(payload["message"], str)
    assert isinstance(payload["details"], dict)


def test_payload_includes_details_only_when_requested():
    error = CapabilityError(
        "the selected route does not support tool calling",
        details={"capability": "tools"},
    )

    assert "details" not in error.to_payload()
    assert error.to_payload(include_details=True)["details"] == {"capability": "tools"}


def test_payload_omits_an_empty_details_mapping_even_when_requested():
    assert "details" not in InternalError("boom").to_payload(include_details=True)


def test_a_payload_cannot_mutate_the_error():
    error = CapabilityError("unsupported", details={"capability": "tools"})
    payload = error.to_payload(include_details=True)
    payload["details"]["capability"] = "vision"

    assert error.details["capability"] == "tools"


# --- immutability ----------------------------------------------------------


def test_details_are_read_only():
    error = CapabilityError("unsupported", details={"capability": "tools"})

    with pytest.raises(TypeError):
        error.details["capability"] = "vision"


def test_details_are_copied_from_the_caller_mapping():
    source = {"capability": "tools"}
    error = CapabilityError("unsupported", details=source)
    source["capability"] = "vision"

    assert error.details == {"capability": "tools"}


def test_errors_constructed_without_details_do_not_share_state():
    first = InternalError("one")
    second = InternalError("two")

    assert first.details == {}
    assert second.details == {}
    assert first.details is not second.details


def test_statuses_are_read_only_properties():
    assert isinstance(GatewayError.http_status, property)
    assert isinstance(GatewayError.grpc_code, property)
    assert isinstance(GatewayError.reason, property)

    for error_class, *_ in CATEGORY_TABLE:
        assert "http_status" not in error_class.__dict__
        assert "grpc_code" not in error_class.__dict__


# --- message and repr ------------------------------------------------------


def test_str_returns_the_sanitised_message():
    error = InternalError(f"failed to reach {UPSTREAM_URL}")

    assert str(error) == error.message
    assert "api.provider.example" not in str(error)


def test_exception_args_carry_the_sanitised_message():
    error = InternalError(f"failed to reach {UPSTREAM_URL}")

    assert error.args == (error.message,)


def test_repr_names_the_reason():
    rendered = repr(InternalError("boom"))

    assert rendered.startswith("InternalError(")
    assert "reason='internal_error'" in rendered
    assert "message='boom'" in rendered


# --- sanitisation ----------------------------------------------------------


def test_sanitise_text_strips_an_upstream_url():
    cleaned = sanitise_text(f"the upstream call to {UPSTREAM_URL} failed")

    assert "api.provider.example" not in cleaned
    assert "https://" not in cleaned
    assert cleaned == "the upstream call to [redacted] failed"


def test_sanitise_text_strips_a_bearer_credential():
    cleaned = sanitise_text(f"upstream rejected Authorization: Bearer {CREDENTIAL}")

    assert CREDENTIAL not in cleaned
    assert "Bearer" not in cleaned
    assert "Authorization" not in cleaned


def test_sanitise_text_strips_an_assigned_credential():
    for text in (
        f"api_key={CREDENTIAL}",
        f"API_KEY: {CREDENTIAL}",
        f"x-api-key={CREDENTIAL}",
        f"password={CREDENTIAL}",
        f"access_token={CREDENTIAL}",
    ):
        cleaned = sanitise_text(f"request failed with {text}")

        assert CREDENTIAL not in cleaned
        assert cleaned == "request failed with [redacted]"


def test_sanitise_text_strips_a_provider_stack_trace():
    cleaned = sanitise_text(f"upstream call failed\n{STACK_TRACE}")

    assert "Traceback" not in cleaned
    assert "litellm" not in cleaned
    assert "site-packages" not in cleaned
    assert "APIError" not in cleaned
    assert cleaned == "upstream call failed [redacted]"


def test_sanitise_text_strips_bare_stack_frames():
    cleaned = sanitise_text('failed\n  File "/app/main.py", line 10, in handler\n  boom')

    assert "main.py" not in cleaned
    assert cleaned == "failed [redacted]"


def test_sanitise_text_collapses_to_a_single_line():
    assert "\n" not in sanitise_text("first line\nsecond line")


def test_sanitise_text_keeps_gateway_vocabulary_intact():
    message = "model 'text-default' is not one of the exposed public model IDs"

    assert sanitise_text(message) == message


def test_sanitise_text_handles_an_empty_message():
    assert sanitise_text("") == ""


def test_a_message_is_sanitised_when_the_error_is_constructed():
    error = InternalError(
        f"upstream {UPSTREAM_URL} replied with Authorization: Bearer {CREDENTIAL}\n{STACK_TRACE}"
    )

    payload = error.to_payload()

    assert UPSTREAM_URL not in payload["message"]
    assert CREDENTIAL not in payload["message"]
    assert "Traceback" not in payload["message"]
    assert "litellm" not in payload["message"]
    assert "Bearer" not in payload["message"]
    assert "api.provider.example" not in str(error)
    assert "api.provider.example" not in repr(error)


def test_details_are_sanitised():
    error = CapabilityError(
        "the route cannot serve this request",
        details={"endpoint": UPSTREAM_URL, "credential": f"Bearer {CREDENTIAL}"},
    )

    details = error.to_payload(include_details=True)["details"]

    assert details["endpoint"] == "[redacted]"
    assert CREDENTIAL not in details["credential"]
    assert "api.provider.example" not in str(details)


# --- wrapping arbitrary exceptions -----------------------------------------


def test_wrapping_an_unknown_exception_yields_an_internal_error():
    original = ProviderFailure(
        f"provider returned 500 for {UPSTREAM_URL} with Authorization: Bearer {CREDENTIAL}\n"
        f"{STACK_TRACE}"
    )

    wrapped = normalise_exception(original)

    assert isinstance(wrapped, InternalError)
    assert wrapped.reason is ErrorReason.INTERNAL_ERROR
    assert wrapped.http_status == 500
    assert wrapped.grpc_code is GrpcStatus.INTERNAL


def test_wrapping_an_unknown_exception_does_not_echo_its_message():
    original = ProviderFailure(
        f"provider returned 500 for {UPSTREAM_URL} with Authorization: Bearer {CREDENTIAL}\n"
        f"{STACK_TRACE}"
    )

    payload = normalise_exception(original).to_payload()

    assert set(payload) == {"reason", "message"}
    assert UPSTREAM_URL not in payload["message"]
    assert CREDENTIAL not in payload["message"]
    assert "Traceback" not in payload["message"]
    assert "litellm" not in payload["message"]
    assert "ProviderFailure" not in payload["message"]
    assert str(original) not in payload["message"]
    assert payload["message"] == "the gateway could not complete the request"


def test_wrapping_a_raw_upstream_body_keeps_it_out_of_the_payload():
    body = '{"error": {"message": "invalid api key", "code": "invalid_api_key"}}'

    payload = normalise_exception(ProviderFailure(body)).to_payload()

    assert "invalid api key" not in payload["message"]
    assert "invalid_api_key" not in payload["message"]
    assert body not in payload["message"]


def test_wrapping_keeps_the_original_available_for_logs():
    original = ProviderFailure("upstream exploded at the socket layer")

    wrapped = normalise_exception(original)

    assert wrapped.__cause__ is original
    assert "upstream exploded" in repr(wrapped.__cause__)


def test_wrapping_can_produce_an_unavailable_error():
    original = ProviderFailure("connection reset by peer")

    wrapped = normalise_exception(original, unavailable=True)

    assert isinstance(wrapped, UnavailableError)
    assert wrapped.reason is ErrorReason.PROVIDER_UNAVAILABLE
    assert wrapped.http_status == 503
    assert wrapped.grpc_code is GrpcStatus.UNAVAILABLE
    assert "connection reset" not in wrapped.to_payload()["message"]
    assert wrapped.__cause__ is original


def test_a_caller_supplied_wrapper_message_is_still_sanitised():
    wrapped = normalise_exception(
        ProviderFailure("boom"), message=f"the call to {UPSTREAM_URL} failed"
    )

    assert "api.provider.example" not in wrapped.to_payload()["message"]
    assert wrapped.to_payload()["message"] == "the call to [redacted] failed"


def test_an_already_classified_error_is_not_reclassified():
    classified = PermissionDeniedError("tenant 'acme' may not use this model")

    assert normalise_exception(classified) is classified
    assert normalise_exception(classified, unavailable=True) is classified


# --- capability errors -----------------------------------------------------


def test_a_capability_error_is_its_own_category():
    error = CapabilityError(
        "the selected route does not support tool calling", details={"capability": "tools"}
    )

    assert not isinstance(error, InvalidArgumentError)
    assert not isinstance(error, FailedPreconditionError)
    assert not issubclass(CapabilityError, InvalidArgumentError)
    assert not issubclass(CapabilityError, FailedPreconditionError)


def test_a_capability_error_is_distinct_from_an_invalid_argument_error():
    capability = CapabilityError("the route cannot serve images")
    invalid = InvalidArgumentError("the request cannot be served")

    assert capability.reason is not invalid.reason
    assert capability.reason is ErrorReason.UNSUPPORTED_CAPABILITY
    assert capability.reason is not ErrorReason.INVALID_REQUEST
    assert capability.http_status != invalid.http_status
    assert capability.grpc_code is not invalid.grpc_code
    assert capability.to_payload() != invalid.to_payload()


def test_a_capability_error_reports_the_unsupported_feature():
    error = CapabilityError(
        "the route does not support structured output",
        details={"capability": "structured_output", "model": "text-default"},
    )

    payload = error.to_payload(include_details=True)

    assert payload["reason"] == "unsupported_capability"
    assert payload["details"] == {
        "capability": "structured_output",
        "model": "text-default",
    }
