"""Gateway error taxonomy: categories, payloads and sanitisation (PRD 5.4, 9.5, 15)."""

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

# Every category in PRD section 5.4, plus the capability category of PRD 9.5.
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
    (CapabilityError, 422, GrpcStatus.FAILED_PRECONDITION, ErrorReason.UNSUPPORTED_CAPABILITY),
)

CATEGORY_IDS = [error_class.__name__ for error_class, *_ in CATEGORY_TABLE]

# The reasons each category owns. A category accepts exactly these and rejects
# every other reason, so a public error can never contradict the status code it
# arrived with. UnauthenticatedError and ResourceExhaustedError each permit two
# reasons; every other category permits one.
CATEGORY_REASONS = {
    InvalidArgumentError: (ErrorReason.INVALID_REQUEST,),
    UnauthenticatedError: (ErrorReason.AUTHENTICATION_REQUIRED, ErrorReason.INVALID_CREDENTIALS),
    PermissionDeniedError: (ErrorReason.PERMISSION_DENIED,),
    NotFoundError: (ErrorReason.RESOURCE_NOT_FOUND,),
    ModelNotFoundError: (ErrorReason.MODEL_NOT_FOUND,),
    ResourceExhaustedError: (ErrorReason.RATE_LIMITED, ErrorReason.CONCURRENCY_LIMIT_EXCEEDED),
    FailedPreconditionError: (ErrorReason.FAILED_PRECONDITION,),
    DeadlineExceededError: (ErrorReason.DEADLINE_EXCEEDED,),
    UnavailableError: (ErrorReason.PROVIDER_UNAVAILABLE,),
    InternalError: (ErrorReason.INTERNAL_ERROR,),
    CapabilityError: (ErrorReason.UNSUPPORTED_CAPABILITY,),
}

REASON_IDS = [error_class.__name__ for error_class in CATEGORY_REASONS]
REASON_CATEGORIES = list(CATEGORY_REASONS)

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


def test_the_reason_table_covers_every_category():
    assert set(CATEGORY_REASONS) == {error_class for error_class, *_ in CATEGORY_TABLE}


def test_every_published_reason_belongs_to_some_category():
    permitted = set().union(*CATEGORY_REASONS.values())

    assert permitted == set(ErrorReason)


@pytest.mark.parametrize("error_class", REASON_CATEGORIES, ids=REASON_IDS)
def test_each_category_declares_exactly_the_reasons_the_table_names(error_class):
    assert error_class.allowed_reasons == frozenset(CATEGORY_REASONS[error_class])


@pytest.mark.parametrize("error_class", REASON_CATEGORIES, ids=REASON_IDS)
def test_a_category_accepts_every_reason_it_permits(error_class):
    for reason in CATEGORY_REASONS[error_class]:
        error = error_class("the request was rejected", reason=reason)

        assert error.reason is reason
        assert error.to_payload()["reason"] == reason.value


@pytest.mark.parametrize("error_class", REASON_CATEGORIES, ids=REASON_IDS)
def test_a_category_rejects_a_reason_belonging_to_another_category(error_class):
    permitted = frozenset(CATEGORY_REASONS[error_class])

    for reason in ErrorReason:
        if reason in permitted:
            continue
        with pytest.raises(ValueError, match="does not permit reason"):
            error_class("the request was rejected", reason=reason)


@pytest.mark.parametrize("error_class", REASON_CATEGORIES, ids=REASON_IDS)
def test_each_category_permits_the_reason_it_defaults_to(error_class):
    assert error_class.default_reason in error_class.allowed_reasons


def test_the_base_error_permits_only_the_internal_reason():
    assert GatewayError.allowed_reasons == frozenset({ErrorReason.INTERNAL_ERROR})
    assert GatewayError("boom").reason is ErrorReason.INTERNAL_ERROR

    with pytest.raises(ValueError, match="does not permit reason"):
        GatewayError("boom", reason=ErrorReason.INVALID_REQUEST)


def test_a_capability_error_rejects_an_authentication_reason():
    with pytest.raises(ValueError, match="does not permit reason 'invalid_credentials'"):
        CapabilityError("the route cannot serve images", reason=ErrorReason.INVALID_CREDENTIALS)


def test_a_not_found_error_rejects_a_rate_limit_reason():
    with pytest.raises(ValueError, match="does not permit reason 'rate_limited'"):
        NotFoundError("no such route", reason=ErrorReason.RATE_LIMITED)


def test_an_unauthenticated_error_distinguishes_its_two_reasons():
    missing = UnauthenticatedError("no credential was presented")
    rejected = UnauthenticatedError(
        "the presented key is not recognised", reason=ErrorReason.INVALID_CREDENTIALS
    )

    assert missing.reason is ErrorReason.AUTHENTICATION_REQUIRED
    assert rejected.reason is ErrorReason.INVALID_CREDENTIALS
    assert missing.http_status == rejected.http_status == 401
    assert missing.to_payload() != rejected.to_payload()


def test_a_resource_exhausted_error_distinguishes_its_two_reasons():
    rate = ResourceExhaustedError("request rate exceeded")
    concurrency = ResourceExhaustedError(
        "too many requests in flight", reason=ErrorReason.CONCURRENCY_LIMIT_EXCEEDED
    )

    assert rate.reason is ErrorReason.RATE_LIMITED
    assert concurrency.reason is ErrorReason.CONCURRENCY_LIMIT_EXCEEDED
    assert rate.http_status == concurrency.http_status == 429
    assert rate.grpc_code is concurrency.grpc_code is GrpcStatus.RESOURCE_EXHAUSTED


def test_a_subclass_declares_its_own_reason_set():
    assert ModelNotFoundError.allowed_reasons == frozenset({ErrorReason.MODEL_NOT_FOUND})
    assert ErrorReason.RESOURCE_NOT_FOUND in NotFoundError.allowed_reasons
    assert ModelNotFoundError.allowed_reasons.isdisjoint(NotFoundError.allowed_reasons)

    with pytest.raises(ValueError, match="does not permit reason"):
        ModelNotFoundError("text-default", reason=ErrorReason.RESOURCE_NOT_FOUND)


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


# --- the limits of sanitisation --------------------------------------------
#
# Sanitisation is a backstop, not a guarantee: it removes the shapes its
# patterns recognise and nothing else. These tests pin the limitation so that
# it stays a documented property rather than an assumption, and so that nobody
# treats a call to sanitise_text as licence to forward upstream text.


def test_sanitise_text_does_not_scrub_a_bare_unrecognised_secret():
    for message in (
        f"the upstream rejected the token {CREDENTIAL} for this tenant",
        "the session cookie session=8f14e45fceea167a5a36dedd4bea2543 was rejected",
        "the upstream expects its own authorization scheme: Token abc123def456",
    ):
        assert sanitise_text(message) == message


def test_an_error_cannot_rely_on_sanitisation_to_remove_an_unrecognised_secret():
    message = f"the upstream rejected the token {CREDENTIAL} for this tenant"

    error = InternalError(message, details={"hint": message})

    assert error.to_payload()["message"] == message
    assert error.to_payload(include_details=True)["details"] == {"hint": message}


def test_sanitisation_is_documented_as_a_backstop():
    documented = errors_module.__doc__ or ""

    assert "backstop" in documented
    assert "not a guarantee" in documented
    assert "session cookie" in documented


def test_the_retained_cause_carries_text_that_must_not_be_logged_by_default():
    original = ProviderFailure(f"rejected Bearer {CREDENTIAL} for {UPSTREAM_URL}")

    wrapped = normalise_exception(original)

    assert wrapped.__cause__ is original
    assert CREDENTIAL in str(wrapped.__cause__)
    assert CREDENTIAL not in wrapped.to_payload()["message"]


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


def test_a_capability_error_is_unprocessable_content():
    error = CapabilityError("the selected route does not support tool calling")

    assert error.http_status == 422
    assert error.grpc_code is GrpcStatus.FAILED_PRECONDITION


def test_the_http_and_grpc_mappings_of_a_capability_error_need_not_be_namesakes():
    """422 and FAILED_PRECONDITION are different concepts, so they may diverge.

    412 Precondition Failed carries a conditional-request meaning in RFC 9110
    that a capability failure does not have, which is why HTTP moves to 422
    Unprocessable Content. gRPC has no 422, and FAILED_PRECONDITION is its code
    for a valid request the system cannot operate on, so it stays. The two
    mappings are decided by their own specifications, not by each other.
    """
    precondition = FailedPreconditionError("the route is not ready")

    assert precondition.http_status == 412
    assert precondition.grpc_code is GrpcStatus.FAILED_PRECONDITION
    assert CapabilityError("unsupported").grpc_code is precondition.grpc_code
    assert CapabilityError("unsupported").http_status == 422
    assert CapabilityError("unsupported").http_status != precondition.http_status


def test_only_a_capability_error_uses_the_unprocessable_content_status():
    unprocessable = [
        error_class.__name__
        for error_class, *_ in CATEGORY_TABLE
        if error_class("x").http_status == 422
    ]

    assert unprocessable == ["CapabilityError"]
