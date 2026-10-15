"""Distributed tracing.

The gateway supports **W3C Trace Context** and **OpenTelemetry**, and it names the
propagation format rather than leaving it to the library's default: the contract is
that an inbound ``traceparent`` continues into the gateway's spans and onward to
the upstream call, so a dependency changing its default would change that contract
without anything here changing. Naming it costs three lines and makes the format a
statement of this module rather than an assumption about a package.

**Tracing is configured into existence.** ``OTEL_ENABLED`` off means no provider is
built and the OpenTelemetry API stays in its no-op state, which is what the API is
designed for: a span is cheap and non-recording rather than absent, so the code
that starts spans needs no branch of its own. ``OTEL_ENABLED`` on with no endpoint
is a **configuration error at startup** rather than a gateway that quietly drops
every span: a deployment that asked for tracing and gave it nowhere to send is
misconfigured, and the same reasoning that makes a malformed price a startup error
applies.

**The endpoint is a base and the signal path is appended here.** The OTLP/HTTP
exporter takes a complete signal URL when one is passed to it and appends the path
only when it reads the environment variable itself, so the gateway appends it: the
setting is the base endpoint the deployment configured, which is the value the
exporter's own environment variable would hold.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from model_gateway.config.settings import ConfigurationError, Settings

__all__ = [
    "SERVICE_NAME",
    "configure_tracing",
    "extract_trace_context",
    "inject_trace_context",
]

SERVICE_NAME = "model-gateway"
"""The ``service.name`` every span carries, so a backend can tell the gateway apart."""

_TRACES_PATH = "v1/traces"


def configure_tracing(settings: Settings) -> TracerProvider | None:
    """The tracer provider the deployment configured, or ``None`` when tracing is off.

    ``None`` is not a degraded mode: with no provider the OpenTelemetry API records
    nothing and costs almost nothing, so a deployment that turned tracing off pays
    for a dictionary lookup per span rather than for a branch in every call site.

    Raises :class:`~model_gateway.config.settings.ConfigurationError` when tracing
    is enabled and the endpoint is empty, because that deployment asked for spans
    and gave them nowhere to go.
    """
    if not settings.otel_enabled:
        return None
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        raise ConfigurationError(
            "OTEL_ENABLED is true but OTEL_EXPORTER_OTLP_ENDPOINT is empty; tracing has "
            "nowhere to send its spans. Set the endpoint, or set OTEL_ENABLED=false"
        )
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    # Installed as the process's provider, not just handed back. Without this the
    # OpenTelemetry API keeps handing out its default tracer, which records
    # nothing: a span would still be started and would still carry a correct
    # parent, but it would never reach the exporter, so a deployment that asked
    # for tracing would receive nothing and would have no way to tell. Keeping
    # the provider returned as well lets a caller shut it down explicitly.
    trace.set_tracer_provider(provider)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=_traces_endpoint(endpoint)))
    )
    set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
    return provider


def _traces_endpoint(base: str) -> str:
    """The complete signal URL the OTLP/HTTP exporter takes when one is passed."""
    return f"{base.removesuffix('/')}/{_TRACES_PATH}"


def extract_trace_context(carrier: Mapping[str, Any]) -> Context:
    """The parent context an inbound request carries, or an empty one.

    An empty context is not a failure: a request that carries no ``traceparent``
    starts a trace of its own, which is what a caller that does not trace should
    get rather than a span attributed to nothing.
    """
    return extract(carrier)


def inject_trace_context(carrier: dict[str, str]) -> dict[str, str]:
    """Add the current context's propagation headers to a carrier, and return it.

    The carrier is mutated and returned so a caller can build one inline. When
    nothing is recording, the propagator writes nothing, so an outbound request
    from a deployment that does not trace carries no tracing headers rather than
    an empty ``traceparent`` a backend would have to reject.
    """
    inject(carrier)
    return carrier
