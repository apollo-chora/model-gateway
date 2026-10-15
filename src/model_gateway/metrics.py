"""The gateway's operational metrics (PRD section 17).

Eleven metrics, named by the PRD, rendered in the Prometheus text exposition
format. The gateway carries its own small registry rather than taking a metrics
client dependency, and the reason is the format rather than the arithmetic: the
PRD fixes the metric names, so what has to be exactly right is the wire text —
the ``# HELP`` and ``# TYPE`` lines, the escaping rules and the histogram's
bucket series — and that is a stable public format this module pins with golden
tests rather than a library's version-sensitive API.

**A metric is a number the gateway maintains, not a number it reports.** Each one
is incremented at the place the fact is known — a request arriving, an attempt
being made, a token count being recorded — and read once, when the exposition is
rendered. Nothing here samples, aggregates or interprets: a counter that is never
incremented is a counter that has nothing to say, which is why every metric the
PRD names is wired where its fact is decided and the exposition shows zero for the
ones a given deployment never reaches.

**Labels are deliberately absent.** The PRD names each metric without labels, and
a label set is a claim about which dimensions the gateway publishes and therefore
about the cardinality an operator's metrics backend must hold. Adding one is a
contract decision the PRD does not settle, so this registry renders one series per
metric; a revision that settles the dimensions adds them here and to the golden
tests together.

**Reading is not atomic across metrics.** The exposition is a point-in-time
snapshot assembled from independent counters, so a scrape taken while requests are
in flight can show a request counted and its duration not yet observed. That is
inherent to exposition rather than a defect: the alternative would be a lock held
across every increment, which would put the metrics on the request path's critical
section to make a scrape perfectly consistent — a trade no operator would take.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "METRIC_NAMES",
    "Counter",
    "Gauge",
    "Histogram",
    "Metrics",
    "render_prometheus",
]

_DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)
"""The durations a histogram counts into, in seconds.

The set spans the intervals a gateway invocation actually occupies — a fast
upstream, a retry with backoff, a streamed answer — and it is the same shape the
Prometheus client's own default uses, so a dashboard written against either reads
the same buckets.
"""


def _escape_help(text: str) -> str:
    """Escape a help string for the exposition format: backslash and newline only.

    The format defines exactly two escapes in a help string, so escaping anything
    else would change the text an operator reads.
    """
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _format_value(value: float) -> str:
    """A sample value in the format's own spelling.

    A whole number is rendered without a decimal point, because the format's
    examples and every Prometheus client render ``1`` and not ``1.0``; a fraction
    keeps the shortest representation that round-trips.
    """
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


@dataclass
class Counter:
    """A value that only ever goes up."""

    name: str
    help: str
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Add ``amount``, which must not be negative."""
        if amount < 0:
            raise ValueError(f"{self.name} is a counter and cannot decrease, got {amount!r}")
        self.value += amount


@dataclass
class Gauge:
    """A value that goes up and down, such as the requests in flight."""

    name: str
    help: str
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount

    def set(self, value: float) -> None:
        self.value = value


@dataclass
class Histogram:
    """A distribution counted into cumulative buckets, with a sum and a count.

    The buckets are cumulative, as the format requires: ``le`` is "less than or
    equal to this bound", and each series counts every observation at or below it.
    An observation above the last bound is counted by the ``+Inf`` series alone,
    which is why the count and the sum are kept beside the buckets rather than
    derived from them.
    """

    name: str
    help: str
    buckets: tuple[float, ...] = _DEFAULT_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError(f"{self.name} must have at least one bucket")
        ordered = tuple(sorted(self.buckets))
        if list(ordered) != list(self.buckets):
            raise ValueError(f"{self.name} buckets must be ascending, got {self.buckets!r}")
        if not self.counts:
            self.counts = [0] * len(ordered)

    def observe(self, value: float) -> None:
        """Record one observation."""
        self.total += value
        self.count += 1
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1


class Metrics:
    """The gateway's metrics, one of each the PRD names.

    Every field is a metric with the name the PRD gives it, so a reader can match
    this class against section 17's list line for line. The names are attributes
    rather than a dictionary because they are fixed: a deployment cannot add,
    rename or remove one, and an operator's dashboard is written against them.
    """

    def __init__(self, *, duration_buckets: tuple[float, ...] = _DEFAULT_BUCKETS) -> None:
        self.requests_total = Counter(
            "gateway_requests_total",
            "Invocations the gateway was asked to serve, one count per invocation.",
        )
        self.request_duration_seconds = Histogram(
            "gateway_request_duration_seconds",
            "Elapsed time of an invocation, from its start to its answer or failure.",
            buckets=duration_buckets,
        )
        self.upstream_requests_total = Counter(
            "gateway_upstream_requests_total",
            "Attempts the gateway made against an upstream provider.",
        )
        self.upstream_duration_seconds = Histogram(
            "gateway_upstream_duration_seconds",
            "Elapsed time of one upstream attempt.",
            buckets=duration_buckets,
        )
        self.tokens_input_total = Counter(
            "gateway_tokens_input_total",
            "Input tokens the gateway recorded, as reported or estimated.",
        )
        self.tokens_output_total = Counter(
            "gateway_tokens_output_total",
            "Output tokens the gateway recorded, as reported or estimated.",
        )
        self.retries_total = Counter(
            "gateway_retries_total",
            "Attempts made after a retryable failure on the same route.",
        )
        self.fallbacks_total = Counter(
            "gateway_fallbacks_total",
            "Invocations served by a route other than the primary one.",
        )
        self.errors_total = Counter(
            "gateway_errors_total",
            "Invocations that ended in a gateway error rather than an answer.",
        )
        self.active_requests = Gauge(
            "gateway_active_requests",
            "Invocations admitted and not yet finished.",
        )
        self.rate_limit_rejections_total = Counter(
            "gateway_rate_limit_rejections_total",
            "Requests refused by a gateway limit rather than by an upstream.",
        )

    def all(self) -> tuple[Counter | Gauge | Histogram, ...]:
        """Every metric, in the order the PRD lists them."""
        return (
            self.requests_total,
            self.request_duration_seconds,
            self.upstream_requests_total,
            self.upstream_duration_seconds,
            self.tokens_input_total,
            self.tokens_output_total,
            self.retries_total,
            self.fallbacks_total,
            self.errors_total,
            self.active_requests,
            self.rate_limit_rejections_total,
        )


METRIC_NAMES: tuple[str, ...] = tuple(metric.name for metric in Metrics().all())
"""The metric names, as section 17 lists them, for tests and operators to check against."""


def _render_metric(metric: Counter | Gauge | Histogram) -> list[str]:
    """One metric as its exposition lines."""
    kind = "histogram" if isinstance(metric, Histogram) else (
        "gauge" if isinstance(metric, Gauge) else "counter"
    )
    lines = [f"# HELP {metric.name} {_escape_help(metric.help)}", f"# TYPE {metric.name} {kind}"]
    if isinstance(metric, Histogram):
        for bound, count in zip(metric.buckets, metric.counts, strict=True):
            lines.append(f'{metric.name}_bucket{{le="{_format_value(bound)}"}} {count}')
        lines.append(f'{metric.name}_bucket{{le="+Inf"}} {metric.count}')
        lines.append(f"{metric.name}_sum {_format_value(metric.total)}")
        lines.append(f"{metric.name}_count {metric.count}")
        return lines
    lines.append(f"{metric.name} {_format_value(metric.value)}")
    return lines


def render_prometheus(metrics: Metrics) -> str:
    """The whole registry in the Prometheus text exposition format.

    The result ends with a newline, which the format's own examples do and which
    every scraper tolerates; a body without one is still valid, so the choice is
    made once here rather than left to whichever caller assembles it.
    """
    lines: list[str] = []
    for metric in metrics.all():
        lines.extend(_render_metric(metric))
    return "\n".join(lines) + "\n"
