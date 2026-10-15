"""Usage and cost accounting.

Every invocation that produces an answer leaves a record naming what was asked,
what answered, how long it took and what it cost. The record is the gateway's own:
the public payloads deliberately omit it, because none of the wire shapes has a
field for it.

Three rules shape the module.

**A token count is either reported or estimated, and the difference is carried.**
The canonical :class:`~model_gateway.domain.Usage` already types its counts as
optional and records which fields it derived rather than read, and a UsageRecord
copies both. Rendering an unreported count as zero — which the HTTP and gRPC usage
shapes must do, because they type their counters as integers — would turn an
unknown quantity into an asserted one, and a caller reading a usage object could
not tell a provider that reported no tokens from one that reported none at all.
The wire layers make that trade-off and state it; the record does not.

**Cost comes from configured pricing, never from a vendor name.** A route names an
upstream and a dialect, but what a million tokens costs is the operator's
configuration, not a property of the provider. A deployment that does not
configure prices records usage and no cost; a hardcoded per-vendor price table
would embed a claim the gateway cannot stand behind and go stale the moment a
provider changes its prices.

**A cost is reported only when it is complete.** The deployment's price entry
declares which token categories it covers, and a category whose count the provider
never reported leaves a charge the gateway cannot see: reporting the remaining sum
as *the* cost would understate spend while looking like a finished figure, which
is the one failure a cost estimate must not have. `estimated_cost` is therefore
`None` whenever any priced category is unknown, and the price itself is validated
where it is read — a negative, non-finite, boolean or non-numeric amount, an
unknown token category and an unknown entry key are all refused at startup rather
than multiplied into a number the gateway would then stand behind.

Storage is in memory by default, which is the honest scope for a single-instance
deployment: records are lost when the process stops, and a deployment that needs
them to survive restarts or be shared configures durable storage, which is a
separate concern requiring configured persistence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from numbers import Real
from typing import Protocol

import orjson
import structlog

from model_gateway.config.routes import Operation, Route
from model_gateway.config.settings import ConfigurationError
from model_gateway.domain import InvocationRequest, Usage
from model_gateway.domain.embeddings import EmbeddingRequest
from model_gateway.domain.images import ImageRequest
from model_gateway.domain.serialization import as_int

logger = structlog.get_logger(__name__)

__all__ = [
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_INCOMPLETE",
    "AccountingRecorder",
    "AccountingSink",
    "InvocationOutcome",
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

STATUS_COMPLETED = "completed"
"""The invocation reached its answer's finish."""

STATUS_FAILED = "failed"
"""The invocation produced output and then failed."""

STATUS_INCOMPLETE = "incomplete"
"""The invocation produced output and then stopped without reaching a terminal event.

A caller that went away, a transport that was lost, or an upstream that ended
without reporting a finish reason. It is deliberately not ``failed``: the
stream's own contract draws that distinction for the caller, and a record that
collapsed the two would report a failure the stream never declared.
"""

_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_INCOMPLETE})
"""Every status a record may carry; a value outside this set is a programming defect."""

_PRICED_CATEGORIES = frozenset({"input", "output", "cached"})
"""The token categories a price entry may quote.

``total_tokens`` is deliberately not one of them. A provider's total is a fact
about the answer and is recorded, because the canonical usage model keeps it and
the record must not silently drop a count a provider reported — but it is not
priced: charging it beside the input and output counts would bill the same tokens
twice.
"""

_PRICE_ENTRY_KEYS = frozenset({"currency", "tokens"})
"""The only keys a price entry may carry; an unknown one is a misconfiguration."""


@dataclass(frozen=True)
class ModelPrice:
    """One model's token prices, the currency they are quoted in, and what they cover.

    ``tokens`` declares *which* categories the model is priced for, and that
    declaration is load-bearing rather than cosmetic. A category the operator did
    not price is not a category priced at zero, and
    :meth:`PricingTable.estimate` refuses to report a number when a category the
    entry *does* price has a count the provider never reported. Reading a missing
    category as free — the obvious consequence of defaulting it to zero — would
    under-report spend while looking like a complete figure, which is the one
    failure mode a cost estimate must not have.

    The value object validates itself rather than trusting its caller, because a
    price is configuration: a negative, non-finite, boolean or non-numeric price
    is a mistake to refuse where it is read, not one to multiply into a cost the
    gateway then stands behind.
    """

    currency: str
    tokens: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError(
                f"a price entry must carry a non-empty currency, got {self.currency!r}"
            )
        if not isinstance(self.tokens, Mapping):
            raise ValueError(
                "a price entry's tokens must be a mapping from token category to price"
            )
        if not self.tokens:
            raise ValueError(
                "a price entry must price at least one token category; an entry that "
                "prices none says nothing about what the model costs"
            )
        for kind, price in self.tokens.items():
            if kind not in _PRICED_CATEGORIES:
                raise ValueError(
                    f"{kind!r} is not a priced token category; the priced categories are "
                    f"{', '.join(sorted(_PRICED_CATEGORIES))}"
                )
            _require_price(kind, price)


def _require_price(kind: str, price: object) -> None:
    """Refuse a price that is not a finite, non-negative real number.

    A boolean is refused explicitly: ``True`` is an ``int`` in Python, so a price
    written as ``true`` would otherwise be read as a price of one. A string price
    would raise deep inside a later multiplication rather than here, and a
    negative or non-finite price would produce a cost the gateway cannot stand
    behind.
    """
    if isinstance(price, bool) or not isinstance(price, Real):
        raise ValueError(f"the {kind} price must be a number, got {price!r}")
    if not isfinite(price):
        raise ValueError(f"the {kind} price must be finite, got {price!r}")
    if price < 0:
        raise ValueError(f"the {kind} price must not be negative, got {price!r}")


@dataclass(frozen=True)
class PricingTable:
    """What one million tokens of each kind costs, per public model ID.

    ``prices`` maps a public model ID to a currency and the token categories that
    model is priced for: input tokens, output tokens and cached input tokens. A
    model absent from the table has no configured price, so its usage is recorded
    with no cost rather than with an invented one.

    The table validates itself, so every instance is a table the gateway can
    price against and not only the one the environment produced. The configuration
    parser is a caller like any other: a table built by hand — by an operator's
    own wiring, or by a test — must not be able to carry a key that is not a model
    ID or a value that is not a price, because :meth:`estimate` would then reach a
    shape it cannot price and raise inside an invocation's accounting rather than
    where the mistake was made.
    """

    prices: Mapping[str, ModelPrice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prices, Mapping):
            raise ValueError("a pricing table's prices must be a mapping from model ID to price")
        for model, price in self.prices.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"a pricing table's keys must be non-empty public model IDs, got {model!r}"
                )
            if not isinstance(price, ModelPrice):
                raise ValueError(f"the price for {model!r} must be a ModelPrice, got {price!r}")

    def estimate(self, usage: Usage, model: str) -> tuple[float | None, str | None]:
        """The cost of one usage report, or ``(None, None)`` when none can be stated.

        ``(None, None)`` answers three situations, and they are deliberately one
        answer because a caller can act on none of the differences: the model has
        no configured price; the price covers a category whose count the provider
        never reported, so a sum would silently omit a charge the caller did
        incur; and no billable count is known at all, where a total of zero would
        read as evidence of a free invocation. Reporting a number in any of them
        would present an incomplete estimate as a cost, and under-reporting spend
        is the one failure a cost estimate must not have.

        A category the entry does not price contributes nothing and is not a
        reason to refuse: an operator who priced only input tokens has said what
        input tokens cost, and the estimate is complete with respect to what they
        priced.
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
            if kind not in configured.tokens:
                continue
            if count is None:
                return None, None
            total += configured.tokens[kind] * count / _PRICE_SCALE
        return total, configured.currency


def pricing_from_env(env: Mapping[str, str] | None = None) -> PricingTable:
    """The pricing table the deployment configured, empty when it configured none.

    The value is a JSON object keyed by public model ID::

        COST_PRICING_JSON={"text-default": {"currency": "USD", "tokens":
        {"input": 3.0, "output": 15.0, "cached": 0.3}}}

    Prices are per million tokens, the unit providers quote. The whole value is
    validated here and refused as a configuration error rather than partially
    applied: a deployment that misconfigures its prices should learn at startup,
    not bill wrong amounts for a month. Each rejection below is a value that would
    otherwise reach a cost calculation and produce a number the gateway could not
    stand behind — an unknown category silently ignored, a string or boolean price
    raising deep inside a multiplication, a negative or non-finite price producing
    a negative or non-finite cost.

    A currency must be a non-empty string and is not otherwise checked: the
    gateway does not validate it against a currency registry, so an operator who
    writes a code no one uses gets a cost denominated in it rather than a refusal.
    That limit is stated rather than hidden, because inventing a registry the
    gateway does not consult would be a stronger claim than the code supports.
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
    return PricingTable(
        prices={model: _price_from_entry(model, value) for model, value in payload.items()}
    )


def _price_from_entry(model: object, value: object) -> ModelPrice:
    """One price entry, validated, or a configuration error naming what was wrong.

    ``model`` is validated here rather than in :class:`ModelPrice`, because it is
    the table's key rather than a field of the price and so never reaches the
    value object.
    """
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError(
            f"{_PRICING_VARIABLE} keys must be non-empty public model IDs, got {model!r}"
        )
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{_PRICING_VARIABLE} entry {model!r} must be an object carrying a currency "
            f"and a tokens object"
        )
    unknown = set(value) - _PRICE_ENTRY_KEYS
    if unknown:
        names = ", ".join(sorted(map(repr, unknown)))
        known = ", ".join(sorted(_PRICE_ENTRY_KEYS))
        raise ConfigurationError(
            f"{_PRICING_VARIABLE} entry {model!r} carries unknown {names}; an entry holds "
            f"only {known}"
        )
    missing = _PRICE_ENTRY_KEYS - set(value)
    if missing:
        raise ConfigurationError(
            f"{_PRICING_VARIABLE} entry {model!r} is missing "
            f"{', '.join(sorted(missing))}"
        )
    try:
        return ModelPrice(currency=value["currency"], tokens=value["tokens"])
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{_PRICING_VARIABLE} entry {model!r}: {error}") from error


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
    upstream model is what answered — and the record asks for both. Keeping them apart
    means a route that later names its upstream differently does not silently
    change what either field means.

    ``input_tokens``, ``output_tokens``, ``cached_tokens`` and ``total_tokens``
    are the canonical usage model's four counts, each carried as the provider
    reported it or as the gateway derived it, with ``None`` for a count that was
    never reported and never a zero. ``total_tokens`` is carried although it is
    not priced: the canonical model keeps a provider's own total distinct from a
    sum the gateway could compute, and a record that dropped the count could not
    preserve that distinction.

    ``status`` names how the invocation ended, and the three values are the ones
    :class:`InvocationOutcome` defines: ``completed`` for an answer that reached
    its finish, ``failed`` for one that produced output and then failed, and
    ``incomplete`` for one that produced output and then stopped without a
    terminal event. ``error_reason`` carries the gateway reason for a ``failed``
    outcome and is ``None`` for the other two. A non-streaming invocation that
    produced an answer is always ``completed``: a failure there produces no answer
    to record, so the other two values are the streaming walk's.
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
    total_tokens: int | None
    estimated_tokens: frozenset[str]
    latency_ms: float
    attempts: int
    fallback_used: bool
    error_reason: str | None
    estimated_cost: float | None
    cost_currency: str | None

    def to_dict(self) -> dict[str, object]:
        """The record as a JSON-ready mapping, for tests and operators.

        The projection is exhaustive: every field of the record appears under its
        own name, so a field added to the record cannot be silently absent from
        what an operator exports. Three fields are rendered rather than copied —
        ``timestamp`` as an ISO-8601 string, ``operation`` as its value and the
        two frozensets as sorted lists — because those are the JSON-ready forms of
        the same facts, and sorting them makes the projection stable.
        """
        return {
            "invocation_id": self.invocation_id,
            "timestamp": self.timestamp.isoformat(),
            "namespace": self.namespace,
            "tenant": self.tenant,
            "public_model": self.public_model,
            "route_model_id": self.route_model_id,
            "operation": self.operation.value,
            "capabilities": sorted(self.capabilities),
            "upstream_model": self.upstream_model,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
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

    The bound is a ``deque`` with a ``maxlen``, so an insertion into a full sink
    drops the oldest record in constant time. The obvious list-and-``pop(0)``
    implementation shifts every remaining record on each insert, which turns a
    sustained-traffic accounting path into quadratic work while looking correct to
    every test a single-process deployment would write.

    The sink is **not thread-safe**. It is written from the event loop, where a
    synchronous ``record`` call cannot interleave with another, and read the same
    way; a deployment that records from worker threads needs a sink of its own
    rather than this one.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = as_int("accounting sink 'capacity'", capacity, minimum=1)
        self._records: deque[UsageRecord] = deque(maxlen=self._capacity)

    def record(self, record: UsageRecord) -> None:
        """Append the record, dropping the oldest once the sink is at capacity."""
        self._records.append(record)

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        """The records held, oldest first, for tests and operators."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"MemoryAccountingSink(records={len(self._records)}, capacity={self._capacity})"


_OPERATION_FOR_REQUEST: Mapping[type, Operation] = {
    InvocationRequest: Operation.TEXT,
    EmbeddingRequest: Operation.EMBEDDINGS,
    ImageRequest: Operation.IMAGE,
}
"""The operation each canonical request type is an invocation of."""


def _require_matching_operation(
    request: InvocationRequest | EmbeddingRequest | ImageRequest, route: Route
) -> None:
    """Refuse a request and a route that describe different operations.

    The route is an audit fact: the record states which configured route served
    the invocation and what that route advertises. A caller that handed the
    recorder the wrong route — an embeddings route for a text answer, say — would
    produce a record that looks valid and is not, and nothing downstream compares
    the two, so the mistake would never surface. The pairing is known only here,
    and the check is cheap, so it is made here rather than trusted.

    A mismatch is a programming defect rather than a caller error, so it raises
    :exc:`ValueError`. The invocation itself is unaffected: the service records
    best-effort and logs what it could not record, so a defect in the recorder
    cannot withhold an answer the caller is waiting on.
    """
    for request_type, operation in _OPERATION_FOR_REQUEST.items():
        if isinstance(request, request_type):
            if route.operation is not operation:
                raise ValueError(
                    f"a {request_type.__name__} is a {operation.value} invocation, but the "
                    f"route handed to the recorder is a {route.operation.value} route"
                )
            return


@dataclass(frozen=True)
class InvocationOutcome:
    """What one invocation's answer is, whichever operation produced it.

    The three canonical answers have different shapes, and a streamed completion
    has no single object at all: its usage arrives on events of its own and its
    terminal event is what says it ended. The facts a record needs are the
    intersection of those shapes, so they are stated once here rather than
    re-derived per operation, and the recorder no longer depends on the concrete
    answer type. A streamed answer therefore reaches the sink through the same
    path as a completed one instead of through a second, parallel one.

    ``status`` names exactly what happened rather than collapsing two cases into
    one: ``completed`` for an answer that reached its
    finish, ``failed`` for one that produced output and then failed, and
    ``incomplete`` for one that produced output and then stopped without ever
    reaching a terminal event — a caller that went away, a transport that was
    lost, or an upstream that ended without reporting a finish reason.
    ``error_reason`` is the gateway reason for a ``failed`` outcome and ``None``
    for the other two.
    """

    invocation_id: str
    upstream_model: str
    usage: Usage
    status: str = STATUS_COMPLETED
    error_reason: str | None = None

    def __post_init__(self) -> None:
        """Refuse a status and a reason that cannot both be true.

        The vocabulary is only meaningful if the two fields agree, and a value
        object that let a caller state ``failed`` with no reason, or a reason
        beside a status that declared no failure, would put a record in front of
        an operator that says less than it appears to. The service produces
        consistent pairs, but the outcome is a public value of this module and the
        invariant belongs to it rather than to the call sites that happen to
        honour it.
        """
        if self.status not in _STATUSES:
            raise ValueError(
                f"an invocation outcome's status must be one of "
                f"{', '.join(sorted(_STATUSES))}, got {self.status!r}"
            )
        if self.status == STATUS_FAILED and not self.error_reason:
            raise ValueError(
                "a failed outcome must carry the gateway reason it failed under; a failure "
                "with no reason is a status the record cannot stand behind"
            )
        if self.status != STATUS_FAILED and self.error_reason is not None:
            raise ValueError(
                f"only a failed outcome carries an error reason; a {self.status} outcome "
                f"declared no failure, so it cannot carry {self.error_reason!r}"
            )


class AccountingRecorder:
    """Build one record from an invocation and hand it to the sink.

    The recorder holds the sink and the pricing table, so a caller supplies the
    facts it alone has — the request, the answer, the route that served it, the
    walk's execution facts and how long the invocation took — and the recorder
    derives the rest. Deriving them here rather than at the call site keeps the
    cost of an invocation's record identical whatever route served it.
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
        outcome: InvocationOutcome,
        route: Route,
        *,
        namespace: str,
        latency_ms: float,
        attempts: int,
        fallback_used: bool,
    ) -> UsageRecord:
        """Record one invocation's answer, whatever operation produced it.

        ``route`` is the route that *served* the invocation, which is where the
        operation and the advertised capabilities come from — the request carries
        a public model ID, not a route, and the route that answered may be a
        fallback rather than the primary one. It must be the route of the same
        operation as the request, which :func:`_require_matching_operation`
        enforces rather than assumes.

        ``outcome`` is the answer's own facts, stated by whichever operation
        produced it. It is what lets a streamed answer be recorded here rather
        than by a second entry point: a stream has no response object to read, so
        the facts the record needs are passed in instead of derived from a type.

        ``attempts`` and ``fallback_used`` are supplied by the caller rather than
        read off the answer, because only the text answer carries them: the
        canonical embeddings and image answers have no field for either, while
        the record asks for both. They come from the same
        walk state the text response is stamped from, so the two agree wherever
        both exist.

        ``tenant`` is the tenant the request stated, and only the text request can
        state one: the canonical embeddings and image requests carry no tenant
        field, because the tenant an invocation acts under is the credential's. It
        is therefore ``None`` for those operations rather than borrowed from
        somewhere it was never stated.

        ``latency_ms`` is the invocation's elapsed time as the service measured
        it, from the call that began the invocation to the moment this record was
        built. It therefore includes route walking, retries, fallbacks and any
        wait for a concurrency slot — it is *not* time spent inside the provider
        alone, and the field is named for what it measures rather than for the
        narrower quantity the record's field list calls provider latency.
        """
        _require_matching_operation(request, route)
        usage = outcome.usage
        cost, currency = (
            self._pricing.estimate(usage, request.model)
            if self._pricing is not None
            else (None, None)
        )
        record = UsageRecord(
            invocation_id=outcome.invocation_id,
            timestamp=datetime.now(UTC),
            namespace=namespace,
            tenant=request.tenant_id if isinstance(request, InvocationRequest) else None,
            public_model=request.model,
            route_model_id=route.model_id,
            operation=route.operation,
            capabilities=frozenset(capability.value for capability in route.capabilities),
            upstream_model=outcome.upstream_model,
            status=outcome.status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            total_tokens=usage.total_tokens,
            estimated_tokens=frozenset(usage.estimated_fields),
            latency_ms=latency_ms,
            attempts=attempts,
            fallback_used=fallback_used,
            error_reason=outcome.error_reason,
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
