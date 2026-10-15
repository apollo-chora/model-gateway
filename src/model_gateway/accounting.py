"""Usage and cost accounting (PRD section 13).

Every invocation leaves a record naming what was asked, what answered, how long it
took and what it cost. The record is the gateway's own: the public payloads
deliberately omit it, because none of the wire shapes has a field for it (§5.2,
§13).

Two rules shape the module.

**A token count is either reported or estimated, and the difference is carried.**
The canonical :class:`~model_gateway.domain.Usage` already types its counts as
optional and records which fields it derived rather than read, and a UsageRecord
copies both. Rendering an unreported count as zero — which the HTTP and gRPC usage
shapes must do, because they type their counters as integers — would turn an
unknown quantity into an asserted one, and a caller reading a usage object could
not tell a provider that reported no tokens from one that reported none at all.
The wire layers make that trade-off and state it; the record does not. A count the
provider never reported is not priced: `estimated_cost` is an estimate of *known*
usage, and `estimated_tokens` preserves the estimate rather than letting it vanish
behind zeros a caller cannot distinguish from a real measurement.

**Cost comes from configured pricing, never from a vendor name.** A route names an
upstream and a dialect, but what a million tokens costs is the operator's
configuration, not a property of the provider. A deployment that does not
configure prices records usage and no cost; a hardcoded per-vendor price table
would embed a claim the gateway cannot stand behind and go stale the moment a
provider changes its prices.

Storage is in memory by default, which is the honest scope for a single-instance
deployment: records are lost when the process stops, and a deployment that needs
them to survive restarts or be shared configures durable storage, which is a
separate concern requiring configured persistence (§13).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import orjson
import structlog

from model_gateway.config.routes import Operation, Route
from model_gateway.config.settings import ConfigurationError
from model_gateway.domain import InvocationRequest, InvocationResponse, Usage
from model_gateway.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from model_gateway.domain.images import ImageRequest, ImageResponse
from model_gateway.domain.serialization import as_int

logger = structlog.get_logger(__name__)

__all__ = [
    "AccountingRecorder",
    "AccountingSink",
    "MemoryAccountingSink",
    "NullAccountingSink",
    "PricingTable",
    "UsageRecord",
    "pricing_from_env",
]

_PRICE_SCALE = 1_000_000
"""Token prices are per million, the unit providers quote."""

_PRICING_VARIABLE = "COST_PRICING_JSON"
_DEFAULT_CAPACITY = 100_000

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class ModelPrice:
    """One model's token prices and the currency they are quoted in."""

    currency: str
    tokens: Mapping[str, float]


@dataclass(frozen=True)
class PricingTable:
    """What one million tokens of each kind costs, per public model ID.

    ``prices`` maps a public model ID to a currency and up to three prices: input
    tokens, output tokens and cached input tokens. A model absent from the table
    has no configured price, so its usage is recorded with no cost rather than
    with an invented one.
    """

    prices: Mapping[str, ModelPrice] = field(default_factory=dict)

    def estimate(self, usage: Usage, model: str) -> tuple[float | None, str | None]:
        """The cost of one usage report, or ``(None, None)`` when none is configured.

        A count the provider never reported is not priced: multiplying zero would
        report a cost the caller did not incur, and the estimate is only ever as
        good as the counts behind it.
        """
        configured = self.prices.get(model)
        if configured is None:
            return None, None
        total = 0.0
        for kind, count in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cached", usage.cached_tokens),
        ):
            if count is None:
                continue
            total += configured.tokens.get(kind, 0.0) * count / _PRICE_SCALE
        return total, configured.currency


def pricing_from_env(env: Mapping[str, str] | None = None) -> PricingTable:
    """The pricing table the deployment configured, empty when it configured none.

    The value is a JSON object keyed by public model ID::

        COST_PRICING_JSON={"text-default": {"currency": "USD", "tokens":
        {"input": 3.0, "output": 15.0, "cached": 0.3}}}

    Prices are per million tokens, the unit providers quote. A value that is not
    valid JSON, or whose shapes do not match, is refused as a configuration error
    rather than partially applied: a deployment that misconfigures its prices
    should learn at startup, not bill wrong amounts for a month.
    """
    import os

    raw = dict(env if env is not None else os.environ).get(_PRICING_VARIABLE)
    if not raw:
        return PricingTable()
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as error:
        raise ConfigurationError(f"{_PRICING_VARIABLE} is not valid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigurationError(
            f"{_PRICING_VARIABLE} must be a JSON object keyed by public model ID"
        )
    prices: dict[str, ModelPrice] = {}
    for model, value in payload.items():
        try:
            currency = value["currency"]
            tokens = value["tokens"]
        except (TypeError, KeyError) as error:
            raise ConfigurationError(
                f"{_PRICING_VARIABLE} entry {model!r} must carry a currency and a "
                f"tokens object"
            ) from error
        if not isinstance(tokens, Mapping):
            raise ConfigurationError(
                f"{_PRICING_VARIABLE} entry {model!r} tokens must be an object"
            )
        prices[model] = ModelPrice(currency=currency, tokens=dict(tokens))
    return PricingTable(prices=prices)


@dataclass(frozen=True)
class UsageRecord:
    """One invocation's facts: what was asked, what answered, what it cost.

    ``status`` is ``completed`` when the invocation produced an answer and
    ``failed`` when it did not; ``error_reason`` names the gateway reason for a
    failure, and is ``None`` for a completion.

    ``namespace`` is the trusted principal the invocation acted under, derived from
    the authenticated credential. It is ``""`` when authentication was disabled,
    which is the shared anonymous namespace.

    ``capabilities`` are the serving route's advertised capabilities and
    ``operation`` the route's operation, so a record can be attributed to a feature
    of the route rather than only to a model. They are read from the route that
    actually served the invocation, never from the plan's primary: a fallback
    answer recorded against the primary would credit a route that never produced it
    and attribute the answer to capabilities the answering route may not have.

    ``route_model_id`` is the serving route's own model ID and ``upstream_model``
    the model the answer reports having come from. They coincide today, because the
    gateway sends the route's model ID upstream and stamps the answer with it, but
    they are separate facts — the route is the deployment's configured entry, the
    upstream model is what answered — and §13 asks for both. Keeping them apart
    means a route that later names its upstream differently does not silently
    change what either field means.
    """

    invocation_id: str
    timestamp: datetime
    namespace: str
    tenant: str | None
    public_model: str
    route_model_id: str
    operation: Operation
    capabilities: frozenset[str]
    upstream_model: str
    status: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    estimated_tokens: frozenset[str]
    latency_ms: float
    attempts: int
    fallback_used: bool
    error_reason: str | None
    estimated_cost: float | None
    cost_currency: str | None

    def to_dict(self) -> dict[str, object]:
        """The record as a JSON-ready mapping, for tests and operators."""
        return {
            "invocation_id": self.invocation_id,
            "timestamp": self.timestamp.isoformat(),
            "namespace": self.namespace,
            "tenant": self.tenant,
            "public_model": self.public_model,
            "operation": self.operation.value,
            "capabilities": sorted(self.capabilities),
            "upstream_model": self.upstream_model,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated_tokens": sorted(self.estimated_tokens),
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
            "error_reason": self.error_reason,
            "estimated_cost": self.estimated_cost,
            "cost_currency": self.cost_currency,
        }


class AccountingSink(Protocol):
    """Where a completed invocation's record goes.

    Recording is not a part of serving the request: a sink that fails must not
    fail the invocation, because the answer has already been produced and the
    caller is waiting on it. The failure is surfaced to the operator instead, so a
    broken store cannot silently turn into a month of unaccounted usage.
    """

    def record(self, record: UsageRecord) -> None:
        """Store one record."""
        ...


class NullAccountingSink:
    """A sink that discards records, for deployments that do not track usage."""

    def record(self, record: UsageRecord) -> None:
        """Drop the record."""

    def __len__(self) -> int:
        """Nothing is held, so the count is zero."""
        return 0


class MemoryAccountingSink:
    """An in-memory sink, sufficient for a single-instance deployment.

    Bounded so a deployment that never reads its records cannot grow without
    limit. Eviction is by insertion order rather than by last use, because an LRU
    costs a timestamp write on every record and the bound's purpose is to stop
    unbounded growth, not to keep the hottest records — a deployment that needs
    that configures durable storage.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = as_int("accounting sink 'capacity'", capacity, minimum=1)
        self._records: list[UsageRecord] = []

    def record(self, record: UsageRecord) -> None:
        """Append the record, evicting the oldest once the sink is full."""
        self._records.append(record)
        while len(self._records) > self._capacity:
            self._records.pop(0)

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        """The records held, oldest first, for tests and operators."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"MemoryAccountingSink(records={len(self._records)}, capacity={self._capacity})"


class AccountingRecorder:
    """Build one record from an invocation and hand it to the sink.

    The recorder holds the sink and the pricing table, so a caller supplies the
    facts it alone has — the request, the response, the route and how long the
    invocation took — and the recorder derives the rest. Deriving them here rather
    than at the call site keeps the cost of an invocation's record identical
    whatever route served it.
    """

    def __init__(
        self,
        sink: AccountingSink | None,
        pricing: PricingTable | None,
    ) -> None:
        self._sink = sink
        self._pricing = pricing

    def record(
        self,
        request: InvocationRequest | EmbeddingRequest | ImageRequest,
        response: InvocationResponse | EmbeddingResponse | ImageResponse,
        route: Route,
        *,
        namespace: str,
        latency_ms: float,
        attempts: int,
        fallback_used: bool,
    ) -> UsageRecord:
        """Record one completed invocation, whatever operation produced it.

        ``route`` is the route that *served* the invocation, which is where the
        operation and the advertised capabilities come from — the request carries
        a public model ID, not a route, and the route that answered may be a
        fallback rather than the primary one.

        ``attempts`` and ``fallback_used`` are supplied by the caller rather than
        read off ``response``, because only the text answer carries them: the
        canonical embeddings and image answers have no field for either (PRD
        section 5.2), while the record §13 asks for does. They come from the same
        walk state the text response is stamped from, so the two agree wherever
        both exist.

        ``tenant`` is the tenant the request stated, and only the text request can
        state one: the canonical embeddings and image requests carry no tenant
        field, because the tenant an invocation acts under is the credential's
        (PRD section 5.1). It is therefore ``None`` for those operations rather
        than borrowed from somewhere it was never stated.
        """
        usage = response.usage
        cost, currency = (
            self._pricing.estimate(usage, request.model)
            if self._pricing is not None
            else (None, None)
        )
        record = UsageRecord(
            invocation_id=response.invocation_id or "",
            timestamp=datetime.now(UTC),
            namespace=namespace,
            tenant=request.tenant_id if isinstance(request, InvocationRequest) else None,
            public_model=request.model,
            route_model_id=route.model_id,
            operation=route.operation,
            capabilities=frozenset(capability.value for capability in route.capabilities),
            upstream_model=response.upstream_model,
            status=_STATUS_COMPLETED,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            estimated_tokens=frozenset(usage.estimated_fields),
            latency_ms=latency_ms,
            attempts=attempts,
            fallback_used=fallback_used,
            error_reason=None,
            estimated_cost=cost,
            cost_currency=currency,
        )
        try:
            self._sink.record(record)
        except Exception as error:
            # A store the operator misconfigured is not a reason to withhold an
            # answer that has already been produced, but it must not fail the
            # invocation silently either.
            logger.error(
                "usage record could not be stored",
                error_type=type(error).__name__,
                invocation_id=record.invocation_id,
            )
        return record
