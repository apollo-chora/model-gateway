"""Gateway error taxonomy shared by the HTTP and gRPC layers (PRD 5.4, 9.5, 14, 15).

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

Each category also declares the reasons it permits in ``allowed_reasons``, and
the constructor rejects anything outside that set. A category therefore cannot
publish a reason it does not own, and the reason a caller switches on always
agrees with the transport status it arrived with.

gRPC status names travel as :class:`GrpcStatus` string constants holding the
standard names from the gRPC status code list. This module imports no gRPC
package and adds no third-party dependency; the gRPC layer maps the name onto
its own status type, for example ``getattr(grpc.StatusCode, error.grpc_code)``.

Sanitisation
------------
Error messages are sanitised and carry stable machine-readable reasons (PRD
5.4). :func:`sanitise_text` removes the upstream-identifying shapes it knows —
provider stack markers, credentials in an assigned or bearer form, and URLs —
and it runs on every string this module publishes. :func:`normalise_exception`
converts an arbitrary exception into an internal or unavailable gateway error
while discarding the original text from the public payload entirely.

Sanitisation is a backstop, not a guarantee, and never a licence to forward
upstream text. A regular expression cannot recognise every secret form: a bare
unprefixed token, a session cookie and a provider-specific authorization format
all pass through unchanged, and no pattern list can be shown to be complete.
The only guarantee on offer is that the known shapes are removed, which is a
safety net for text that leaked past the call sites — not permission to send
upstream text to a caller. Build messages from gateway vocabulary, and pass
upstream text only through :func:`normalise_exception`, which does not echo it
at all.

That places a requirement on the observability work of PRD section 17.
:func:`normalise_exception` retains the original exception as ``__cause__``, so
the provider's own message — and any prompt, response or credential embedded in
it — stays reachable through the exception chain. PRD section 17 forbids
logging prompts, responses and credentials by default, so the logging layer must
not format exception chains by default: it logs the gateway error's own
``reason`` and sanitised ``message``, and the chain only behind an explicit
opt-in that accepts the risk.
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
    "IdempotencyConflictError",
    "InternalError",
    "InvalidArgumentError",
    "MalformedResponseError",
    "ModelNotFoundError",
    "NotFoundError",
    "PermissionDeniedError",
    "ResourceExhaustedError",
    "UnauthenticatedError",
    "UnavailableError",
    "normalise_exception",
    "sanitise_text",
    "redact_credential_shapes",
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
    (``MODEL_NOT_FOUND`` and ``RESOURCE_NOT_FOUND`` are both ``NOT_FOUND``), but
    a reason belongs to the category that can justify it: each category declares
    the reasons it permits in ``allowed_reasons`` and rejects the rest.
    """

    INVALID_REQUEST = "invalid_request"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
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
    MALFORMED_RESPONSE = "malformed_response"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@unique
class GrpcStatus(StrEnum):
    """The standard gRPC status names this gateway uses (PRD 5.4).

    Values are the status names exactly as gRPC defines them, so the gRPC
    servicer can map a name onto its own status type without a lookup table.
    No gRPC package is imported by this module.
    """

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"
    UNKNOWN = "UNKNOWN"


def redact_credential_shapes(text: str) -> str:
    """Credential material in an assigned or bearer form, replaced.

    The narrow half of :func:`sanitise_text`, exposed because the log redaction
    wants exactly this and not the rest: collapsing whitespace and removing URLs
    are right for a message a caller reads and wrong for a log value an operator is
    reading for a reason. What it recognises is what :func:`sanitise_text`
    recognises and no more, so the limits stated there apply here too.
    """
    if not text:
        return text
    return _CREDENTIAL.sub(_REDACTED, _BEARER.sub(_REDACTED, text))


def sanitise_text(text: str) -> str:
    """Strip the upstream-identifying shapes this module recognises from text.

    Three things are removed: everything from the first provider stack marker
    onwards, credential material in an assigned or bearer form, and URLs.
    Whitespace is then collapsed, so the result is always a single line. Text
    carrying none of those is returned unchanged.

    Recognising only known shapes is the whole of the guarantee. A bare
    unprefixed token, a session cookie and a provider-specific authorization
    format all pass through, so treat the result as a backstop for accidental
    leakage rather than as evidence that text is safe to forward upstream.
    """
    if not text:
        return text

    cleaned = text
    markers = [match.start() for pattern in _STACK_MARKERS if (match := pattern.search(cleaned))]
    if markers:
        cleaned = f"{cleaned[: min(markers)]} {_REDACTED}"
    cleaned = redact_credential_shapes(cleaned)
    cleaned = _URL.sub(_REDACTED, cleaned)
    return " ".join(cleaned.split())


class GatewayError(Exception):
    """Base class for every error the gateway reports to a caller (PRD 5.4).

    The default category is internal, so a subclass that forgets to declare a
    transport mapping fails closed onto 500/``INTERNAL`` rather than onto
    something that suggests caller error.

    ``allowed_reasons`` is the set of reasons the category owns. Each subclass
    declares its own set, and it is not inherited: the declaration in a class is
    exactly the set of reasons that class accepts, so a subclass can neither
    widen nor silently inherit what its parent permits.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.INTERNAL_ERROR
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.INTERNAL_ERROR})
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
        ``AUTHENTICATION_REQUIRED``, for example). Only a reason in the class's
        ``allowed_reasons`` is accepted, so an error can never carry a reason
        that contradicts its transport status; anything else raises
        :exc:`ValueError`.

        ``details`` keys are gateway-authored field names; every value is
        sanitised. The message and the details are sanitised here, so the
        upstream-identifying shapes :func:`sanitise_text` knows about cannot be
        published from this constructor. That is a backstop and not a licence to
        pass upstream text in: see the module docstring.
        """
        chosen = self.default_reason if reason is None else ErrorReason(reason)
        if chosen not in self.allowed_reasons:
            permitted = ", ".join(sorted(item.value for item in self.allowed_reasons))
            raise ValueError(
                f"{type(self).__name__} does not permit reason {chosen.value!r}; "
                f"it permits: {permitted}"
            )
        self._reason = chosen
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
        both sanitised. Every field is gateway-authored: the message and the
        details are sanitised at construction, and this module's own errors never
        carry upstream text in the first place. Sanitisation is a backstop rather
        than a guarantee, so that is a statement about how the fields are built
        and not a claim that arbitrary text would come out clean.

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
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.INVALID_REQUEST})
    _http_status: ClassVar[int] = 400
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.INVALID_ARGUMENT


class IdempotencyConflictError(GatewayError):
    """An idempotency key was reused for a different request: 409 / ALREADY_EXISTS.

    A key names one operation. A repeat carrying the same fingerprint is a replay and
    is answered from the record; one carrying a different fingerprint is a different
    operation and is refused rather than executed, because executing it would both
    run a request the caller never made and overwrite the record the key already
    holds — so one key could oscillate between unrelated payloads indefinitely.

    409 Conflict is HTTP's own code for a request that conflicts with the current
    state of the resource it addresses, which is exactly this: the key already
    identifies another request. ``ALREADY_EXISTS`` is gRPC's code for a resource
    that already exists, which is what the key does.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.IDEMPOTENCY_CONFLICT
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset(
        {ErrorReason.IDEMPOTENCY_CONFLICT}
    )
    _http_status: ClassVar[int] = 409
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.ALREADY_EXISTS


class UnauthenticatedError(GatewayError):
    """The caller is missing or has presented a bad credential: 401 / UNAUTHENTICATED.

    Two reasons are permitted: ``AUTHENTICATION_REQUIRED`` when no credential
    arrived at all, and ``INVALID_CREDENTIALS`` when one arrived and was
    rejected. The distinction is what tells a caller whether to obtain a
    credential or to replace the one it sent.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.AUTHENTICATION_REQUIRED
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset(
        {ErrorReason.AUTHENTICATION_REQUIRED, ErrorReason.INVALID_CREDENTIALS}
    )
    _http_status: ClassVar[int] = 401
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.UNAUTHENTICATED


class PermissionDeniedError(GatewayError):
    """The caller is authenticated but not allowed to do this: 403 / PERMISSION_DENIED."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.PERMISSION_DENIED
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.PERMISSION_DENIED})
    _http_status: ClassVar[int] = 403
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.PERMISSION_DENIED


class NotFoundError(GatewayError):
    """The addressed resource does not exist: 404 / NOT_FOUND."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.RESOURCE_NOT_FOUND
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.RESOURCE_NOT_FOUND})
    _http_status: ClassVar[int] = 404
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.NOT_FOUND


class ModelNotFoundError(NotFoundError):
    """The requested public model ID is not one the gateway exposes (PRD 10).

    The reason set is declared fresh as ``MODEL_NOT_FOUND`` rather than
    inherited: a caller can tell a missing public model ID from a missing
    resource, so the subclass does not accept its parent's ``RESOURCE_NOT_FOUND``
    either.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.MODEL_NOT_FOUND
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.MODEL_NOT_FOUND})


class ResourceExhaustedError(GatewayError):
    """A rate, concurrency or quota limit was reached: 429 / RESOURCE_EXHAUSTED.

    Two reasons are permitted: ``RATE_LIMITED`` for a request-rate or quota
    rejection, and ``CONCURRENCY_LIMIT_EXCEEDED`` for a rejection caused by the
    in-flight request count. They share a transport status because the caller's
    remedy is the same shape in both cases — wait and retry — while the reason
    keeps the two measurable separately (PRD 17).
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.RATE_LIMITED
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset(
        {ErrorReason.RATE_LIMITED, ErrorReason.CONCURRENCY_LIMIT_EXCEEDED}
    )
    _http_status: ClassVar[int] = 429
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.RESOURCE_EXHAUSTED


class FailedPreconditionError(GatewayError):
    """The system state required to serve the request does not hold: 412 / FAILED_PRECONDITION."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.FAILED_PRECONDITION
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.FAILED_PRECONDITION})
    _http_status: ClassVar[int] = 412
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.FAILED_PRECONDITION


class DeadlineExceededError(GatewayError):
    """The invocation deadline expired before completion: 504 / DEADLINE_EXCEEDED (PRD 15)."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.DEADLINE_EXCEEDED
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.DEADLINE_EXCEEDED})
    _http_status: ClassVar[int] = 504
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.DEADLINE_EXCEEDED


class UnavailableError(GatewayError):
    """No provider could serve the request, or the gateway is shutting down: 503 / UNAVAILABLE."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.PROVIDER_UNAVAILABLE
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset(
        {ErrorReason.PROVIDER_UNAVAILABLE}
    )
    _http_status: ClassVar[int] = 503
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.UNAVAILABLE


class InternalError(GatewayError):
    """A gateway defect or an unclassified upstream failure: 500 / INTERNAL."""

    default_reason: ClassVar[ErrorReason] = ErrorReason.INTERNAL_ERROR
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.INTERNAL_ERROR})
    _http_status: ClassVar[int] = 500
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.INTERNAL


class MalformedResponseError(GatewayError):
    """An upstream answered, and its answer cannot serve the request: 502 / UNKNOWN.

    Raised when a provider's response is structurally wrong for the request the
    gateway made: the wrong number of vectors, a missing or duplicated input
    position, a vector of the wrong dimensionality, or a record the canonical
    representation cannot carry. The request itself was valid and the route was
    reachable — the fault lies in that particular upstream's *answer*, which is
    what separates this category from :class:`InternalError`. A gateway
    programming bug is a defect in this process and must terminate the
    invocation; a malformed answer is a property of one provider's response, so
    another eligible route may well answer correctly. The reliability layer
    classifies this accordingly: not retried on the route that just answered
    wrongly, but eligible for a fallback (PRD section 11.2).

    It is its own category rather than a reason on :class:`InternalError`,
    because widening that category would make every gateway defect
    fallback-eligible, and a defect is not something another route can fix.
    It is also deliberately not an :class:`UnavailableError`: the provider was
    reachable and did answer — the gateway could not use what it said.

    502 Bad Gateway is HTTP's own code for exactly this: a gateway that
    received an invalid response from an upstream server. gRPC has no 522-style
    code; ``UNKNOWN`` is its standard name for an error that could not be
    classified into a more specific code, which is what an undecodable upstream
    answer is. As with :class:`CapabilityError`, the two transports map the
    concept by their own specifications and only the numeric status is HTTP's
    to decide.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.MALFORMED_RESPONSE
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset({ErrorReason.MALFORMED_RESPONSE})
    _http_status: ClassVar[int] = 502
    _grpc_code: ClassVar[GrpcStatus] = GrpcStatus.UNKNOWN


class CapabilityError(GatewayError):
    """A provider cannot serve a requested capability: 422 / FAILED_PRECONDITION (PRD 9.5).

    Raised when the selected route cannot honour tools, images, system
    instructions, structured-output requirements or streaming. Silently dropping
    such a parameter is forbidden, so this is a refusal and never a downgrade.

    It is its own category rather than an invalid-argument error: the request is
    well-formed, and the failure is a property of the route that was chosen. It
    is deliberately not a subclass of :class:`FailedPreconditionError` either, so
    that a handler matching on type cannot conflate a missing feature with an
    unmet state precondition. Carry the offending feature in ``details`` under
    ``capability``.

    The HTTP and gRPC mappings are not conceptual namesakes, and they need not
    be. 412 Precondition Failed has a specific conditional-request meaning in
    RFC 9110 — the ``If-Match`` family of headers — and a capability failure is
    not that, so HTTP uses 422 Unprocessable Content instead: the request is
    syntactically valid and correctly formed, but cannot be processed against
    the selected route. gRPC has no 422 and no conditional-request concept, and
    ``FAILED_PRECONDITION`` is its documented code for a request that is valid
    yet rejected because the system cannot operate as required, so it keeps that
    code. The two transports share a code where the concept lines up and diverge
    where it does not; only the numeric status is HTTP's to decide.
    """

    default_reason: ClassVar[ErrorReason] = ErrorReason.UNSUPPORTED_CAPABILITY
    allowed_reasons: ClassVar[frozenset[ErrorReason]] = frozenset(
        {ErrorReason.UNSUPPORTED_CAPABILITY}
    )
    _http_status: ClassVar[int] = 422
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
    exception chain still sees everything — which is exactly why PRD section 17
    forbids logging prompts, responses and credentials by default, and why the
    logging layer must not format exception chains by default either. Log the
    gateway error's own ``reason`` and sanitised ``message``; reach the chain
    only behind an explicit opt-in that accepts the risk.

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
