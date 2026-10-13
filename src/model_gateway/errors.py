"""Gateway error taxonomy shared by the HTTP and gRPC layers (PRD 5.4, 9.3, 14, 15).

Every failure the gateway reports to a caller is a :class:`GatewayError`. One
carries three things and nothing else:

``reason``
    A stable, machine-readable identifier from :class:`ErrorReason`. Callers
    branch on this value; it is never a free-form sentence, and it never changes
    meaning once published.
``message``
    A human-readable explanation, sanitised when the error is constructed.
``details``
    Optional structured extras, each value sanitised when the error is
    constructed. The mapping is read-only.

Each category of failure in PRD section 5.4 is a class with a fixed
``http_status`` and ``grpc_code``, so the two northbound transports translate a
failure by reading those properties rather than by matching exception types::

    except GatewayError as error:
        return Response(error.http_status, error.to_payload())

gRPC status names travel as :class:`GrpcStatus` string constants holding the
standard names from the gRPC status code list. This module imports no gRPC
package and adds no third-party dependency; the gRPC layer maps the name onto
its own status type, for example ``getattr(grpc.StatusCode, error.grpc_code)``.

Sanitisation
------------
Error messages are sanitised and carry stable machine-readable reasons (PRD
5.4). Nothing derived from an upstream provider may reach a caller: not a URL,
not a credential, not a provider stack trace, not a raw upstream body.
:func:`sanitise_text` enforces that for every string this module publishes, and
:func:`normalise_exception` converts an arbitrary exception into an internal or
unavailable gateway error while discarding the original text from the public
payload entirely. The original exception is attached as ``__cause__``, so a log
record that formats the exception chain still sees the provider's own text while
the caller does not.

Sanitisation is a backstop, not a licence to forward upstream text. Build
messages from gateway vocabulary and pass upstream text only through
:func:`normalise_exception`, which does not echo it at all.
"""

import re
from collections.abc import Mapping
from enum import StrEnum, unique
from types import MappingProxyType
from typing import ClassVar

__all__ = [
    "CapabilityError",
    "DeadlineExceededError",
    "ErrorReason",
    "FailedPreconditionError",
    "GatewayError",
    "GrpcStatus",
    "InternalError",
    "InvalidArgumentError",
    "ModelNotFoundError",
    "NotFoundError",
    "PermissionDeniedError",
    "ResourceExhaustedError",
    "UnauthenticatedError",
    "UnavailableError",
    "normalise_exception",
    "sanitise_text",
]

_REDACTED = "[redacted]"

_TRACEBACK_START = re.compile(r"(?im)^[ \t]*traceback \(most recent call last\)[ \t]*:?[ \t]*$")

_STACK_FRAME_LINE = re.compile(r'(?im)^[ \t]*(?:file "[^"]*", line \d+.*|at [\w$.<>]+\([^)]*\).*)$')

_STACK_MARKERS = (_TRACEBACK_START, _STACK_FRAME_LINE)

_BEARER = re.compile(r"(?i)\bbearer[ \t]+\S+")

_CREDENTIAL = re.compile(
    r"(?i)\b(?:authorization|proxy-authorization|x-api-key|api[-_]?key|access[-_]?token"
    r"|auth[-_]?token|bearer|password|secret)\b[ \t]*[:=][ \t]*\S+"
)

_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s'\"<>()]+", re.IGNORECASE)


@unique
class ErrorReason(StrEnum):
    """Stable machine-readable reasons a caller can switch on (PRD 5.4).

    These values are part of the public contract: a value is never renamed and
    never reused for a different meaning. Reasons are finer grained than
    categories, so several reasons can share one transport status
    (``MODEL_NOT_FOUND`` and ``RESOURCE_NOT_FOUND`` are both ``NOT_FOUND``), and
    a category may be raised with any reason it can justify.
    """

    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_REQUIRED = "authentication_required"
    INVALID_CREDENTIALS = "invalid_credentials"
    PERMISSION_DENIED = "permission_denied"
    MODEL_NOT_FOUND = "model_not_found"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RATE_LIMITED = "rate_limited"
    CONCURRENCY_LIMIT_EXCEEDED = "concurrency_limit_exceeded"
    FAILED_PRECONDITION = "failed_precondition"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INTERNAL_ERROR = "internal_error"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@unique
class GrpcStatus(StrEnum):
    """The standard gRPC status names this gateway uses (PRD 5.4).

    Values are the status names exactly as gRPC defines them, so the gRPC
    servicer can map a name onto its own status type without a lookup table.
    No gRPC package is imported by this module.
    """

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


def sanitise_text(text: str) -> str:
    """Strip upstream-identifying detail from text a caller may see.

    Three things are removed: everything from the first provider stack marker
    onwards, credential material, and URLs. Whitespace is then collapsed, so the
    result is always a single line. Text carrying none of those is returned
    unchanged.
    """
    if not text:
        return text

    cleaned = text
    markers = [match.start() for pattern in _STACK_MARKERS if (match := pattern.search(cleaned))]
    if markers:
        cleaned = f"{cleaned[: min(markers)]} {_REDACTED}"
    cleaned = _BEARER.sub(_REDACTED, cleaned)
    cleaned = _CREDENTIAL.sub(_REDACTED, cleaned)
    cleaned = _URL.sub(_REDACTED, cleaned)
    return " ".join(cleaned.split())


class GatewayError(Exception):
    """Base class for every error the gateway reports to a caller (PRD 5.4).

    The default category is internal, so a subclass that forgets to declare a
    transport mapping fails closed onto 500/``INTERNAL`` rather than onto
    something that suggests caller error.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.INTERNAL_ERROR
    _http_status: ClassVar[int] = 500
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        reason: ErrorReason | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        """Build an error from a gateway-authored message and optional details.

        ``reason`` defaults to the class's own reason; pass one explicitly to
        narrow it within the category (``INVALID_CREDENTIALS`` instead of
        ``AUTHENTICATION_REQUIRED``, for example). ``details`` keys are
        gateway-authored field names; every value is sanitised. The message and
        the details are sanitised here, so no construction path can publish an
        upstream URL, a credential or a stack trace.
        """
        self._reason = self.default_reason if reason is None else ErrorReason(reason)
        self._message = sanitise_text(message)
        self._details: Mapping[str, str] = MappingProxyType(
            {key: sanitise_text(str(value)) for key, value in (details or {}).items()}
        )
        super().__init__(self._message)

    @property
    def reason(self) -> ErrorReason:
        """The stable machine-readable reason a caller switches on."""
        return self._reason

    @property
    def message(self) -> str:
        """The sanitised, human-readable explanation."""
        return self._message

    @property
    def details(self) -> Mapping[str, str]:
        """Sanitised structured extras; read-only, and empty when none were given."""
        return self._details

    @property
    def http_status(self) -> int:
        """The HTTP status code the HTTP layer returns for this category."""
        return self._http_status

    @property
    def grpc_code(self) -> GrpcStatus:
        """The gRPC status name the gRPC layer maps for this category."""
        return self._grpc_code

    def to_payload(self, *, include_details: bool = False) -> dict[str, object]:
        """Return the safe public representation of this error.

        The default payload has exactly two keys, ``reason`` and ``message``,
        both sanitised. Nothing upstream-derived is ever present: no endpoint,
        no credential, no provider text, no raw response body.

        ``include_details`` adds the sanitised ``details`` mapping under a third
        key. It is off by default so that the documented public shape stays
        reason-and-message only, and it is safe to turn on because details are
        gateway-authored and sanitised at construction.
        """
        payload: dict[str, object] = {"reason": self._reason.value, "message": self._message}
        if include_details and self._details:
            payload["details"] = dict(self._details)
        return payload

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reason={self._reason.value!r}, message={self._message!r})"


class InvalidArgumentError(GatewayError):
    """The request is malformed or fails gateway validation: 400 / INVALID_ARGUMENT."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.INVALID_REQUEST
    _http_status: ClassVar[int] = 400
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.INVALID_ARGUMENT


class UnauthenticatedError(GatewayError):
    """The caller is missing or has presented a bad credential: 401 / UNAUTHENTICATED."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.AUTHENTICATION_REQUIRED
    _http_status: ClassVar[int] = 401
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.UNAUTHENTICATED


class PermissionDeniedError(GatewayError):
    """The caller is authenticated but not allowed to do this: 403 / PERMISSION_DENIED."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.PERMISSION_DENIED
    _http_status: ClassVar[int] = 403
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.PERMISSION_DENIED


class NotFoundError(GatewayError):
    """The addressed resource does not exist: 404 / NOT_FOUND."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.RESOURCE_NOT_FOUND
    _http_status: ClassVar[int] = 404
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.NOT_FOUND


class ModelNotFoundError(NotFoundError):
    """The requested public model ID is not one the gateway exposes (PRD 10)."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.MODEL_NOT_FOUND


class ResourceExhaustedError(GatewayError):
    """A rate, concurrency or quota limit was reached: 429 / RESOURCE_EXHAUSTED."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.RATE_LIMITED
    _http_status: ClassVar[int] = 429
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.RESOURCE_EXHAUSTED


class FailedPreconditionError(GatewayError):
    """The system state required to serve the request does not hold: 412 / FAILED_PRECONDITION."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.FAILED_PRECONDITION
    _http_status: ClassVar[int] = 412
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.FAILED_PRECONDITION


class DeadlineExceededError(GatewayError):
    """The invocation deadline expired before completion: 504 / DEADLINE_EXCEEDED (PRD 15)."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.DEADLINE_EXCEEDED
    _http_status: ClassVar[int] = 504
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.DEADLINE_EXCEEDED


class UnavailableError(GatewayError):
    """No provider could serve the request, or the gateway is shutting down: 503 / UNAVAILABLE."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.PROVIDER_UNAVAILABLE
    _http_status: ClassVar[int] = 503
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.UNAVAILABLE


class InternalError(GatewayError):
    """A gateway defect or an unclassified upstream failure: 500 / INTERNAL."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.INTERNAL_ERROR
    _http_status: ClassVar[int] = 500
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.INTERNAL


class CapabilityError(GatewayError):
    """A provider cannot serve a requested capability: 412 / FAILED_PRECONDITION (PRD 9.3).

    Raised when the selected route cannot honour tools, images, system
    instructions, structured-output requirements or streaming. Silently dropping
    such a parameter is forbidden, so this is a refusal and never a downgrade.

    It is its own category rather than an invalid-argument error: the request is
    well-formed, and the failure is a precondition of the route that was chosen.
    It is deliberately not a subclass of :class:`FailedPreconditionError`
    either, so that a handler matching on type cannot conflate a missing
    feature with an unmet state precondition. Carry the offending feature in
    ``details`` under ``capability``.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.UNSUPPORTED_CAPABILITY
    _http_status: ClassVar[int] = 412
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.FAILED_PRECONDITION


def normalise_exception(
    error: BaseException,
    *,
    unavailable: bool = False,
    message: str | None = None,
) -> GatewayError:
    """Convert an arbitrary exception into a gateway error for the public boundary.

    This is the only path by which a failure raised outside the gateway reaches
    a caller. The original text is discarded from the public payload entirely:
    the returned error carries a fixed, gateway-authored message, so the
    provider's message, type name and traceback cannot leak. The original
    exception is attached as ``__cause__``, so a log record that formats the
    exception chain still sees everything.

    Set ``unavailable`` for a failure that means no provider could serve the
    request, giving 503 / ``UNAVAILABLE``; the default is the internal case,
    500 / ``INTERNAL``. Those are the two categories an unclassified failure can
    honestly claim. Classifying anything further — a caller error, a rate limit,
    an expiry — is policy, and policy belongs to the code that raised it, which
    should raise the category directly instead.

    An error that is already a :class:`GatewayError` is returned unchanged: it
    has already been classified and sanitised, and re-classifying it would
    demote a precise reason to ``INTERNAL_ERROR``. ``message`` overrides the
    default text and is sanitised like any other message.
    """
    if isinstance(error, GatewayError):
        return error

    gateway_error: GatewayError
    if unavailable:
        gateway_error = UnavailableError(message or "the upstream provider is unavailable")
    else:
        gateway_error = InternalError(message or "the gateway could not complete the request")

    gateway_error.__cause__ = error
    gateway_error.__suppress_context__ = True
    return gateway_error
