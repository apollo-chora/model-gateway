"""The operational metrics and their exposition (PRD section 17).

The metric *names* are the contract, so one test checks this registry against
section 17's list line for line rather than against a copy of itself. The
exposition *format* is the other contract — a scraper reads the wire text, not the
objects — so the rendering is pinned with golden expectations rather than by
inspecting values, because a value assertion would keep passing while the text
around it went wrong.
"""

import pytest
from blacksheep import Application, JSONContent
from blacksheep.server.routing import Router
from blacksheep.testing import TestClient

from model_gateway.capabilities import Capability
from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Message,
    Role,
    TextPart,
    Usage,
)
from model_gateway.http.auth import create_auth_middleware
from model_gateway.http.chat import register_chat_routes
from model_gateway.http.errors import register_error_handlers
from model_gateway.http.metrics import register_metrics_route
from model_gateway.inference import InferenceService
from model_gateway.metrics import (
    METRIC_NAMES,
    Counter,
    Gauge,
    Histogram,
    Metrics,
    _render_metric,
    render_prometheus,
)
from model_gateway.registry import ModelRegistry
from model_gateway.security.api_keys import ApiKeySet

REQUIRED_METRICS = (
    "gateway_requests_total",
    "gateway_request_duration_seconds",
    "gateway_upstream_requests_total",
    "gateway_upstream_duration_seconds",
    "gateway_tokens_input_total",
    "gateway_tokens_output_total",
    "gateway_retries_total",
    "gateway_fallbacks_total",
    "gateway_errors_total",
    "gateway_active_requests",
    "gateway_rate_limit_rejections_total",
)
"""Section 17's list, copied here on purpose.

Copied rather than imported from the module under test: a test that reads the
names out of the implementation agrees with the implementation by construction and
would not notice a name being changed.
"""


def test_the_registry_declares_exactly_the_metrics_section_17_requires():
    assert METRIC_NAMES == REQUIRED_METRICS
    assert [metric.name for metric in Metrics().all()] == list(REQUIRED_METRICS)


def test_a_fresh_registry_reports_every_metric_as_zero():
    """A deployment that never reaches a metric shows it at zero rather than hiding it."""
    text = render_prometheus(Metrics())

    for name in REQUIRED_METRICS:
        assert f"# HELP {name} " in text
        assert f"# TYPE {name} " in text


# --------------------------------------------------------------------------- #
# The three instrument kinds
# --------------------------------------------------------------------------- #


def test_a_counter_only_goes_up():
    counter = Counter("c", "help")

    counter.inc()
    counter.inc(2.5)

    assert counter.value == 3.5
    with pytest.raises(ValueError):
        counter.inc(-1)


def test_a_gauge_goes_both_ways_and_can_be_set():
    gauge = Gauge("g", "help")

    gauge.inc(2)
    gauge.dec()
    assert gauge.value == 1.0

    gauge.set(7)
    assert gauge.value == 7.0


def test_a_histogram_counts_cumulatively_and_keeps_its_sum():
    """``le`` is "at or below", so each bucket counts every observation up to it."""
    histogram = Histogram("h", "help", buckets=(1.0, 2.0))

    histogram.observe(0.5)
    histogram.observe(1.5)
    histogram.observe(5.0)

    assert histogram.counts == [1, 2]
    assert histogram.count == 3
    assert histogram.total == pytest.approx(7.0)


def test_a_histogram_rejects_buckets_that_are_not_buckets():
    with pytest.raises(ValueError):
        Histogram("h", "help", buckets=())
    with pytest.raises(ValueError):
        Histogram("h", "help", buckets=(2.0, 1.0))


# --------------------------------------------------------------------------- #
# The exposition format
# --------------------------------------------------------------------------- #


def test_a_counter_renders_as_help_a_type_and_a_sample():
    """The golden text, because the wire format is what a scraper reads."""
    counter = Counter("gateway_errors_total", "Invocations that failed.")
    counter.inc(3)

    assert _render_metric(counter) == [
        "# HELP gateway_errors_total Invocations that failed.",
        "# TYPE gateway_errors_total counter",
        "gateway_errors_total 3",
    ]


def test_a_gauge_renders_as_a_gauge():
    assert _render_metric(Gauge("g", "help")) == [
        "# HELP g help",
        "# TYPE g gauge",
        "g 0",
    ]


def test_a_histogram_renders_cumulative_buckets_a_sum_and_a_count():
    histogram = Histogram("h", "help", buckets=(1.0, 2.0))
    histogram.observe(0.5)
    histogram.observe(1.5)
    histogram.observe(5.0)

    assert _render_metric(histogram) == [
        "# HELP h help",
        "# TYPE h histogram",
        'h_bucket{le="1"} 1',
        'h_bucket{le="2"} 2',
        'h_bucket{le="+Inf"} 3',
        "h_sum 7",
        "h_count 3",
    ]


def test_a_whole_number_is_rendered_without_a_decimal_point():
    """The format's own spelling: ``1`` and not ``1.0``, which is what clients emit."""
    counter = Counter("c", "help")
    counter.inc(1)
    counter.inc(0.5)

    assert _render_metric(counter)[-1] == "c 1.5"


def test_a_help_string_escapes_the_two_characters_the_format_defines():
    counter = Counter("c", "a \\ backslash and a \n newline")

    assert _render_metric(counter)[0] == "# HELP c a \\\\ backslash and a \\n newline"


def test_the_exposition_ends_with_a_newline():
    assert render_prometheus(Metrics()).endswith("\n")


# --------------------------------------------------------------------------- #
# The service counts into the registry
# --------------------------------------------------------------------------- #


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


def _answer(**overrides) -> InvocationResponse:
    values: dict[str, object] = {
        "model": "text-default",
        "upstream_model": "upstream-text",
        "invocation_id": "inv-metrics-1",
        "content": (TextPart("hello"),),
        "finish_reason": FinishReason.STOP,
        "usage": Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    }
    values.update(overrides)
    return InvocationResponse(**values)  # type: ignore[arg-type]


class _EchoProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[Route, object, object]] = []

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        return _answer()


class _FailingOnceProvider(_EchoProvider):
    """Fails the first attempt with an unavailable provider, then answers."""

    async def complete(self, route, request, context):
        self.calls.append((route, request, context))
        if len(self.calls) == 1:
            from model_gateway.errors import UnavailableError

            raise UnavailableError("the route is down")
        return _answer()


def _service(provider, *, metrics, routes=None, **overrides) -> InferenceService:
    return InferenceService(
        ModelRegistry(routes or (_route(),)), provider, metrics=metrics, **overrides
    )


async def test_one_invocation_counts_its_arrival_its_duration_and_its_tokens():
    metrics = Metrics()
    service = _service(_EchoProvider(), metrics=metrics)

    await service.invoke(_request())

    assert metrics.requests_total.value == 1
    assert metrics.active_requests.value == 0
    assert metrics.request_duration_seconds.count == 1
    assert metrics.upstream_requests_total.value == 1
    assert metrics.upstream_duration_seconds.count == 1
    assert metrics.tokens_input_total.value == 10
    assert metrics.tokens_output_total.value == 5
    assert metrics.errors_total.value == 0
    assert metrics.fallbacks_total.value == 0


async def test_an_invocation_the_gateway_refuses_is_counted_as_an_error():
    """A refusal is an invocation the gateway declined to serve, not an absence of one."""
    from model_gateway.errors import ModelNotFoundError

    metrics = Metrics()
    service = _service(_EchoProvider(), metrics=metrics)

    with pytest.raises(ModelNotFoundError):
        await service.invoke(_request(model="not-a-model"))

    assert metrics.requests_total.value == 1
    assert metrics.errors_total.value == 1
    assert metrics.active_requests.value == 0
    assert metrics.upstream_requests_total.value == 0


async def test_a_retry_and_a_fallback_are_counted_where_they_happen():
    metrics = Metrics()
    provider = _FailingOnceProvider()
    service = _service(
        provider,
        metrics=metrics,
        routes=(_route("primary"), _route("fallback")),
    )

    await service.invoke(_request())

    # The first attempt fails with an unavailable provider, which is fallback
    # eligible but not retried on the same route, so the invocation walks: one
    # retry count stays zero and one fallback is counted.
    assert metrics.retries_total.value == 0
    assert metrics.fallbacks_total.value == 1
    assert metrics.upstream_requests_total.value == 2


async def test_a_gateway_limit_refusal_is_counted_as_a_limit_rejection():
    """The gateway's own refusal, told apart from an upstream's rate limit."""
    from model_gateway.errors import ResourceExhaustedError
    from model_gateway.limits import RateLimiter

    metrics = Metrics()
    service = _service(
        _EchoProvider(),
        metrics=metrics,
        rate_limiter=RateLimiter(requests=1, window_seconds=60),
    )

    await service.invoke(_request())
    with pytest.raises(ResourceExhaustedError):
        await service.invoke(_request())

    assert metrics.rate_limit_rejections_total.value == 1
    assert metrics.errors_total.value == 1
    assert metrics.requests_total.value == 2


async def test_an_upstream_rate_limit_is_not_counted_as_the_gateways_own():
    """An upstream's 429 is the provider reporting, not the gateway refusing."""
    from model_gateway.errors import ErrorReason, ResourceExhaustedError

    class _RateLimitedProvider(_EchoProvider):
        async def complete(self, route, request, context):
            self.calls.append((route, request, context))
            raise ResourceExhaustedError(
                "the upstream is rate limiting", reason=ErrorReason.RATE_LIMITED
            )

    metrics = Metrics()
    service = _service(_RateLimitedProvider(), metrics=metrics)

    with pytest.raises(ResourceExhaustedError):
        await service.invoke(_request())

    assert metrics.rate_limit_rejections_total.value == 0
    assert metrics.errors_total.value == 1


async def test_a_count_the_provider_never_reported_is_not_added_to_the_total():
    """The totals are sums of what the gateway knows, not of what it assumed."""
    metrics = Metrics()
    service = _service(
        _EchoProvider(),
        metrics=metrics,
    )

    class _SilentProvider(_EchoProvider):
        async def complete(self, route, request, context):
            self.calls.append((route, request, context))
            return _answer(usage=Usage(input_tokens=10))

    service = _service(_SilentProvider(), metrics=metrics)
    await service.invoke(_request())

    assert metrics.tokens_input_total.value == 10
    assert metrics.tokens_output_total.value == 0


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #


def _metrics_app(metrics: Metrics, provider) -> Application:
    settings = Settings.from_env(
        {
            "GATEWAY_AUTH_ENABLED": "true",
            "GATEWAY_API_KEYS": f"{_KEY}:tenant-a",
        }
    )
    service = InferenceService(
        ModelRegistry((_route(),)), provider, metrics=metrics
    )
    app = Application(router=Router())
    register_error_handlers(app)
    register_metrics_route(app.router, metrics)
    register_chat_routes(app.router, service)
    app.middlewares.append(
        create_auth_middleware(
            ApiKeySet.from_settings(settings),
            auth_enabled=True,
            allow_anonymous_health=True,
        )
    )
    return app


_KEY = "sk-gateway-key"


async def test_the_metrics_endpoint_serves_the_exposition_format():
    """The content type is the format's own, version included, because a scraper reads it."""
    metrics = Metrics()
    app = _metrics_app(metrics, _EchoProvider())

    await app.start()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {_KEY}"}
        body = {"model": "text-default", "messages": [{"role": "user", "content": "hi"}]}
        await client.post("/v1/chat/completions", content=JSONContent(body), headers=headers)

        response = await client.get("/metrics", headers=headers)
        text_body = await response.text()
    finally:
        await app.stop()

    assert response.status == 200
    assert response.content_type() == b"text/plain; version=0.0.4; charset=utf-8"
    assert "# TYPE gateway_requests_total counter" in text_body
    assert "gateway_requests_total 1" in text_body


async def test_the_metrics_endpoint_requires_a_credential_like_every_other_path():
    """Section 14.2 exempts two paths, and a scrape is not one of them."""
    metrics = Metrics()
    app = _metrics_app(metrics, _EchoProvider())

    await app.start()
    try:
        response = await TestClient(app).get("/metrics")
    finally:
        await app.stop()

    assert response.status == 401
