"""Tracing: configuration, propagation and the span an invocation leaves (PRD §17).

Section 17 asks for W3C Trace Context and OpenTelemetry support. Three things are
worth proving separately: that a deployment which did not ask for tracing gets
none, that one which asked and gave no endpoint is refused rather than silently
dropping spans, and that an inbound ``traceparent`` really is the parent of the
span the gateway produces — because "supports W3C Trace Context" means continuing
a caller's trace, not merely emitting spans of the gateway's own.
"""

import pytest
from blacksheep import Application, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.config.settings import ConfigurationError
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    TextPart,
    Usage,
)
from model_gateway.grpc.tracing import continue_trace
from model_gateway.http.auth import create_auth_middleware
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.tracing import create_tracing_middleware
from model_gateway.inference import InferenceService
from model_gateway.observability import (
    SERVICE_NAME,
    _traces_endpoint,
    configure_tracing,
    extract_trace_context,
    inject_trace_context,
)
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
"""The W3C specification's own example header, so the format is theirs and not ours."""

_PARENT_TRACE_ID = 0x4BF92F3577B34DA6A3CE929D0E0E4736


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_a_deployment_that_did_not_ask_for_tracing_gets_none():
    assert configure_tracing(Settings.from_env({"OTEL_ENABLED": "false"})) is None


def test_tracing_with_nowhere_to_send_is_refused_at_startup():
    """A deployment that asked for spans and gave them nowhere to go is misconfigured."""
    settings = Settings.from_env({"OTEL_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": ""})

    with pytest.raises(ConfigurationError):
        configure_tracing(settings)


def test_the_configured_provider_names_the_service():
    provider = configure_tracing(
        Settings.from_env(
            {"OTEL_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"}
        )
    )

    assert provider is not None
    assert provider.resource.attributes["service.name"] == SERVICE_NAME
    provider.shutdown()


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://collector:4318", "http://collector:4318/v1/traces"),
        ("http://collector:4318/", "http://collector:4318/v1/traces"),
    ],
)
def test_the_signal_path_is_appended_to_the_configured_base(base, expected):
    """The exporter takes a complete URL when one is passed, so the path is added here."""
    assert _traces_endpoint(base) == expected


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


def test_injecting_a_context_writes_the_w3c_headers():
    exporter, tracer = _recording_tracer()

    carrier: dict[str, str] = {}
    with tracer.start_as_current_span("work"):
        inject_trace_context(carrier)

    assert carrier["traceparent"].startswith("00-")
    assert len(carrier["traceparent"].split("-")) == 4


def test_injecting_nothing_writes_nothing():
    """A deployment that does not trace sends no tracing headers, not an empty one."""
    carrier: dict[str, str] = {}

    inject_trace_context(carrier)

    assert carrier == {}


def test_an_absent_traceparent_is_an_empty_context_rather_than_a_failure():
    """A caller that does not trace starts a trace of its own."""
    assert extract_trace_context({}) is not None


# --------------------------------------------------------------------------- #
# The span an invocation leaves
# --------------------------------------------------------------------------- #


def _recording_tracer() -> tuple[InMemorySpanExporter, object]:
    """A tracer whose spans a test can read, without a collector and without globals."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("model-gateway-test")


def _route(model_id: str = "upstream-text", **overrides) -> Route:
    values: dict[str, object] = {
        "operation": Operation.TEXT,
        "index": 0,
        "endpoint": "https://upstream.internal.example/v1",
        "base_url": "https://upstream.internal.example/v1",
        "model_id": model_id,
        "format": "openai",
        "api_key": "sk-upstream-secret",
        "capabilities": frozenset({Capability.CHAT}),
        "extra_headers": {},
        "allow_absolute_path_url": False,
    }
    values.update(overrides)
    return Route(**values)  # type: ignore[arg-type]


def _request(**overrides) -> InvocationRequest:
    values: dict[str, object] = {
        "model": "text-default",
        "messages": (Message(role=Role.USER, content=(TextPart("hello"),)),),
        "idempotency_key": None,
    }
    values.update(overrides)
    return InvocationRequest(**values)  # type: ignore[arg-type]


class _EchoProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        return InvocationResponse(
            model="text-default",
            upstream_model="upstream-text",
            invocation_id="inv-trace-1",
            content=(TextPart("hello"),),
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def _service(tracer, **overrides) -> InferenceService:
    return InferenceService(
        ModelRegistry((_route(),)), _EchoProvider(), tracer=tracer, **overrides
    )


async def test_an_invocation_leaves_one_span_named_for_the_operation():
    exporter, tracer = _recording_tracer()
    service = _service(tracer)

    await service.invoke(_request())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "invoke"
    assert spans[0].attributes["gateway.model"] == "text-default"


async def test_a_deployment_that_does_not_trace_still_serves():
    """No tracer is not a degraded mode: the API is a no-op, and the answer is the answer."""
    service = _service(None)

    answer = await service.invoke(_request())

    assert answer.text == "hello"


async def test_an_inbound_traceparent_is_the_parent_of_the_gateways_span():
    """Continuing a caller's trace is what supporting W3C Trace Context means."""
    exporter, tracer = _recording_tracer()
    service = _service(tracer)

    token = attach(extract_trace_context({"traceparent": TRACEPARENT}))
    try:
        await service.invoke(_request())
    finally:
        detach(token)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].context.trace_id == _PARENT_TRACE_ID
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == 0x00F067AA0BA902B7


async def test_each_operation_names_its_own_span():
    """A trace backend groups by span name, so four operations sharing one would be unreadable."""
    exporter, tracer = _recording_tracer()
    service = _service(tracer)

    await service.invoke(_request())

    assert [span.name for span in exporter.get_finished_spans()] == ["invoke"]


# --------------------------------------------------------------------------- #
# The inbound context reaches the service through the transport
# --------------------------------------------------------------------------- #

_KEY = "sk-gateway-key"


def _traced_chat_app(tracer) -> Application:
    """A chat application whose service traces into the given tracer."""
    settings = Settings.from_env(
        {"GATEWAY_AUTH_ENABLED": "true", "GATEWAY_API_KEYS": f"{_KEY}:tenant-a"}
    )
    service = InferenceService(ModelRegistry((_route(),)), _EchoProvider(), tracer=tracer)
    app = Application(router=Router())
    register_error_handlers(app)
    register_chat_routes(app.router, service)
    app.middlewares.append(create_tracing_middleware())
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


async def test_an_inbound_traceparent_reaches_the_service_through_the_http_surface():
    """The middleware's contribution, which no service-level test can show.

    A service test attaches the context itself, so it would keep passing if the
    transport stopped extracting one — and a gateway that does not extract is a
    gateway whose caller's trace stops at it.
    """
    exporter, tracer = _recording_tracer()
    app = _traced_chat_app(tracer)

    await app.start()
    try:
        await TestClient(app).post(
            "/v1/chat/completions",
            content=JSONContent(
                {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
            ),
            headers={"Authorization": f"Bearer {_KEY}", "traceparent": TRACEPARENT},
        )
    finally:
        await app.stop()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke"]
    assert spans[0].context.trace_id == _PARENT_TRACE_ID
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == 0x00F067AA0BA902B7


async def test_a_request_without_a_traceparent_starts_its_own_trace():
    """A caller that does not trace gets a span attributed to its own request, not to nothing."""
    exporter, tracer = _recording_tracer()
    app = _traced_chat_app(tracer)

    await app.start()
    try:
        await TestClient(app).post(
            "/v1/chat/completions",
            content=JSONContent(
                {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
            ),
            headers={"Authorization": f"Bearer {_KEY}"},
        )
    finally:
        await app.stop()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].parent is None


# --------------------------------------------------------------------------- #
# The same over gRPC, whose carrier is the call's metadata
# --------------------------------------------------------------------------- #


class _CallContext:
    """A gRPC context double carrying only the metadata the helper reads."""

    def __init__(self, metadata: tuple[tuple[str, str], ...]) -> None:
        self._metadata = metadata

    def invocation_metadata(self):
        return self._metadata


async def test_a_calls_metadata_carries_the_callers_trace():
    """gRPC propagation is the same carrier shape as HTTP, so one trace spans both."""
    exporter, tracer = _recording_tracer()

    async with continue_trace(_CallContext((("traceparent", TRACEPARENT),))):
        with tracer.start_as_current_span("Invoke"):
            pass

    spans = exporter.get_finished_spans()
    assert spans[0].context.trace_id == _PARENT_TRACE_ID
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == 0x00F067AA0BA902B7


async def test_a_call_without_metadata_starts_its_own_trace():
    exporter, tracer = _recording_tracer()

    async with continue_trace(_CallContext(())):
        with tracer.start_as_current_span("Invoke"):
            pass

    assert exporter.get_finished_spans()[0].parent is None


async def test_the_context_is_detached_even_when_the_call_aborts():
    """A context left attached would make the next call on this worker a child of this one."""
    exporter, tracer = _recording_tracer()

    with pytest.raises(RuntimeError):
        async with continue_trace(_CallContext((("traceparent", TRACEPARENT),))):
            raise RuntimeError("the handler aborted")

    with tracer.start_as_current_span("after"):
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["after"]
    assert spans[0].parent is None
