"""Application factory.

One call builds everything a serving process needs: validated settings, logging,
the JSON encoder, the model registry read from the environment, the inbound
authentication middleware, the health endpoints, the OpenAI-compatible HTTP
surface and the error mapping that turns a gateway failure into a status and a
sanitised body.

The registry is built here, before the application is handed to the server, so
that configuration the gateway cannot run with fails at startup rather than on
the first request. Nothing is resolved lazily: a
deployment whose routes are malformed does not start.

When no service is injected, the factory builds the production one — the
LiteLLM binding over the registry's routes, with the attempt
budget from ``Settings`` — so a process started with configuration alone
actually calls upstream. An injected service is still used verbatim: tests and
callers bind the provider they want, naming the deterministic
:class:`~model_gateway.providers.MockProvider` when they need the stand-in.
There is no configuration flag that enables the mock: an
operator cannot turn fake inference on by accident, and cannot turn it on on
purpose either.

The gateway core never names a provider, so the binding is swappable here without
touching routing, policy or the public contract.
"""

from dataclasses import dataclass

import grpc
import orjson
import structlog
from blacksheep import Application
from blacksheep.server.routing import Router
from blacksheep.settings.json import json_settings

from model_gateway.accounting import (
    AccountingRecorder,
    AccountingSink,
    MemoryAccountingSink,
    NullAccountingSink,
    pricing_from_env,
)
from model_gateway.app_logging import configure_logging
from model_gateway.capabilities import Capability
from model_gateway.config import ConfigurationError, GovernanceMode, Settings
from model_gateway.config.routes import Operation
from model_gateway.config.settings import UsageStorage
from model_gateway.errors import FailedPreconditionError
from model_gateway.governance import (
    BudgetManager,
    Database,
    EgressGate,
    GovernanceManager,
    GrpcManaClient,
    IdempotencyClaims,
    ManaMeter,
    Outbox,
    SuspensionChecker,
)
from model_gateway.grpc.server import GrpcServer
from model_gateway.health import healthz, readyz
from model_gateway.http import (
    register_chat_routes,
    register_completions_routes,
    register_embeddings_routes,
    register_error_handlers,
    register_image_routes,
    register_model_routes,
    register_responses_routes,
)
from model_gateway.http.auth import create_auth_middleware
from model_gateway.http.body_limit import create_body_limit_middleware
from model_gateway.http.metrics import register_metrics_route
from model_gateway.http.tracing import create_tracing_middleware
from model_gateway.inference import InferenceService
from model_gateway.limits import RateLimiter, TenantConcurrencyLimiter
from model_gateway.metrics import Metrics
from model_gateway.observability import SERVICE_NAME, configure_tracing
from model_gateway.providers.litellm_image import ensure_supported_image_dialect
from model_gateway.providers.litellm_provider import (
    LiteLLMProvider,
    ensure_supported_dialect,
)
from model_gateway.registry import ModelRegistry
from model_gateway.reliability import ConcurrencyLimiter
from model_gateway.security.api_keys import ApiKeySet

logger = structlog.get_logger(__name__)

_BINDING_OPERATIONS: frozenset[Operation] = frozenset(
    {Operation.TEXT, Operation.IMAGE, Operation.EMBEDDINGS}
)
"""The operations the production binding runs, and so the ones its start validates.

The binding has one entry point per operation: the text completion, the
embedding call and the image call. An operation it does not run has no route
this process would serve, so a dialect it could not keep behind the transport
controls is not a reason to refuse to start — that route is inert here. The
image operation was absent from this set only because it had no binding, and a
binding for it now exists, so its routes are checked like the others: an image
dialect whose HTTP path has not been verified is refused at startup rather
than on the first caller.
"""


def _ensure_binding_can_run(registry: ModelRegistry) -> None:
    """Refuse to start when a configured route names a dialect the binding cannot run.

    The binding refuses such a route at the attempt too, and this is not a
    duplicate of that check: a deployment whose configuration the transport
    controls cannot survive should not reach the point of serving requests at
    all. Every operation the binding runs is
    checked — the text, embeddings and image operations today — so no route
    this process would actually serve is unusable at startup and discoverable
    only when a caller arrives. An operation with no binding is not checked,
    because this process would never run its routes. The check runs only when
    this factory builds the binding, since an injected service brings its own
    provider.
    """
    for model in registry.list_models():
        if model.operation not in _BINDING_OPERATIONS:
            continue
        for route in registry.routes_for(model.id):
            try:
                if model.operation is Operation.IMAGE:
                    ensure_supported_image_dialect(route.format)
                else:
                    ensure_supported_dialect(route.format)
            except FailedPreconditionError as error:
                raise ConfigurationError(
                    f"{model.id}: {error}. The gateway does not start with a route it cannot "
                    f"run behind its transport controls"
                ) from error


def _ensure_editing_is_not_advertised(registry: ModelRegistry) -> None:
    """Refuse to start when a configured image route advertises ``editing``.

    The gateway's canonical image request carries no field for an image input,
    so an edit cannot be forwarded to any route however its capabilities read,
    and an edit answered without it would report an image edited that never
    left the gateway. The gateway therefore categorically cannot perform
    editing: the endpoint is absent rather than present-and-always-failing,
    and a route declaring the ``editing`` capability advertises an operation
    this process will never serve. The refusal says editing is not
    implemented, rather than that this route cannot serve it, because no
    route can.
    """
    for model in registry.list_models():
        if model.operation is not Operation.IMAGE:
            continue
        for route in registry.routes_for(model.id):
            if Capability.EDITING in route.capabilities:
                raise ConfigurationError(
                    f"{model.id}: the 'editing' capability is not implemented: the gateway's "
                    f"canonical image request carries no field for an image input, so an edit "
                    f"cannot be forwarded to any route; remove 'editing' from the route's "
                    f"declared capabilities"
                )


_DEFAULT_CAPACITY = 100_000


def sink_for(storage: UsageStorage) -> AccountingSink:
    """The sink a storage mode names.

    ``none`` discards records: a deployment that does not want usage or cost
    accounting gets a sink that does not hold anything, rather than an
    in-memory store no one reads. ``memory`` holds them in process, which is
    the default standalone deployment and is sufficient for one instance.

    The durable modes are named by the settings vocabulary but not
    implemented here: they need configured persistence, which this slice
    does not presume. An unrecognised mode is a configuration error, not a
    silent fall-back to memory, because a deployment that asks for durable
    records and loses them on restart is worse than one that refuses to start.
    """
    if storage is UsageStorage.NONE:
        return NullAccountingSink()
    if storage is UsageStorage.MEMORY:
        return MemoryAccountingSink()
    raise ConfigurationError(
        f"USAGE_STORAGE is {storage.value!r}, which requires configured persistence "
        "that this deployment does not provide; use memory or none"
    )


def _accounting(settings: Settings) -> AccountingRecorder | None:
    """The accounting the deployment configured, or none when it tracks nothing.

    Both settings must be on for records to exist: usage tracking with no
    pricing data would record counts no one can cost, and cost tracking with
    no usage data would price nothing. A deployment that wants usage but not
    cost — or the reverse — says so by turning one off, and the other is
    ignored rather than silently dropped.

    An unconfigured pricing table is not an error: usage is still recorded,
    with no cost, which is the honest state of a deployment that has not said
    what its models cost. Cost estimation is optional; usage is not.
    """
    if not settings.usage_tracking_enabled:
        return None
    pricing = pricing_from_env() if settings.cost_tracking_enabled else None
    return AccountingRecorder(sink_for(settings.usage_storage), pricing)


def _limits(
    settings: Settings,
) -> tuple[RateLimiter | None, TenantConcurrencyLimiter | None]:
    """The admission decisions the deployment configured, either of which may be none.

    A limit is built only when it was configured with something that bounds: a
    request rate of zero and a per-tenant bound of zero are the deployment saying
    it has not decided what its callers are entitled to, and building a limiter
    that admits everything would be machinery no configuration asked for.

    They are returned separately rather than as one value because they are keyed
    differently — the rate per caller, the concurrency per tenant — and a
    deployment may reasonably want one and not the other.
    """
    rate = (
        RateLimiter(
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_requests > 0
        else None
    )
    tenant = (
        TenantConcurrencyLimiter(maximum=settings.tenant_max_concurrent_requests)
        if settings.tenant_max_concurrent_requests > 0
        else None
    )
    return rate, tenant


@dataclass(frozen=True)
class GovernanceWiring:
    """The governance chain and the database it shares, for one process.

    The database travels with the chain because the process owns both
    lifecycles: the pool opens at startup and closes at shutdown, and the
    chain is what every invocation asks for a decision. The mana channel
    travels with them for the same reason: it is opened with the chain and
    closed with the process, so an unclosed channel does not keep the event
    loop from exiting cleanly.
    """

    db: Database
    manager: GovernanceManager
    mana_channel: grpc.aio.Channel | None = None


def _ensure_governance_mode(settings: Settings) -> None:
    """Refuse to start a governed deployment whose mana meter is not configured.

    In the default ``chora`` mode the deployment is part of the chora platform:
    a governance database without the identity service's address would run the
    chain un-metered — every metered action silently unpriced — so the start
    fails rather than let a governed deployment drift into an un-metered one.
    ``standalone`` mode is the explicit opt-out for an operator who runs the
    gateway apart from the platform's mana ledger, and allows a DSN with no
    identity address. A deployment with no DSN at all is unaffected: there is
    no chain to wire and nothing to meter.
    """
    if (
        settings.governance_mode is GovernanceMode.CHORA
        and settings.observability_dsn
        and not settings.identity_grpc_addr
    ):
        raise ConfigurationError(
            "CHORA_IDENTITY_GRPC_ADDR is required when CHORA_OBSERVABILITY_DSN is set — "
            "governance requires mana metering; set CHORA_GOVERNANCE_MODE=standalone to "
            "run the chain un-metered"
        )


def _governance(settings: Settings) -> GovernanceWiring | None:
    """The governance chain the deployment configured, or none when it did not.

    Governance is opt-in at the settings layer: a deployment that configures
    no ``CHORA_OBSERVABILITY_DSN`` gets no database and no chain, and the
    invocation flow runs exactly as it did before governance existed. The
    mana meter is wired when the deployment also names chora-identity's
    gRPC address: the channel is built here and closed with the process, so
    a deployment that configures no ``CHORA_IDENTITY_GRPC_ADDR`` runs the
    chain un-metered, exactly as it runs without a budget window. In the
    default ``chora`` mode the identity address is required whenever the DSN
    is set, so a governed deployment cannot silently run un-metered; only an
    explicit ``CHORA_GOVERNANCE_MODE=standalone`` allows the pair to be split.
    """
    _ensure_governance_mode(settings)
    if not settings.observability_dsn:
        return None
    db = Database(settings.observability_dsn)
    channel: grpc.aio.Channel | None = None
    mana: ManaMeter | None = None
    if settings.identity_grpc_addr:
        # The mesh terminates the service-to-service mTLS the identity
        # contract declares, so the channel itself is plaintext: the gateway
        # connects to its sidecar, and the sidecar holds the identity.
        channel = grpc.aio.insecure_channel(settings.identity_grpc_addr)
        mana = ManaMeter(GrpcManaClient(channel))
    return GovernanceWiring(
        db,
        GovernanceManager(
            db,
            BudgetManager(db),
            Outbox(db),
            IdempotencyClaims(db),
            SuspensionChecker(db),
            EgressGate(db),
            mana=mana,
        ),
        mana_channel=channel,
    )


def _orjson_dumps(obj) -> str:
    return orjson.dumps(obj).decode("utf-8")


def configure_json() -> None:
    json_settings.use(dumps=_orjson_dumps)


def create_app(
    settings: Settings | None = None,
    *,
    registry: ModelRegistry | None = None,
    service: InferenceService | None = None,
    serve_grpc: bool = False,
    router: Router | None = None,
) -> Application:
    """Build the application, ready to be started by the server.

    ``registry`` and ``service`` are keyword-only. An injected ``service`` is
    used verbatim — tests and callers pass
    ``InferenceService(registry, provider)`` naming the provider this process
    should call upstream with. When no service is injected the factory builds
    the production one:
    :class:`~model_gateway.providers.litellm_provider.LiteLLMProvider`
    over the registry's routes, with the deployment's configured reliability
    policy — ``PROVIDER_TIMEOUT_SECONDS`` as both the invocation deadline and
    the per-attempt ceiling (one configured value plays both roles today; they
    are logically distinct bounds and separating them is a future refinement),
    ``PROVIDER_RETRY_COUNT`` and ``PROVIDER_RETRY_BACKOFF_MS`` as the retry
    schedule, and a limiter built from ``PROVIDER_MAX_CONCURRENT_REQUESTS``
    . The service anchors one deadline per invocation and
    hands each attempt the smaller of the configured timeout and what is left
    of that deadline, so a retry can never outlive the caller's budget.

    Building that binding also validates the routes it would run: a route of any
    operation the binding serves — text or embeddings — whose dialect it cannot
    keep behind the transport controls fails the start, and
    an image route that declares the ``editing`` capability fails it too,
    because the gateway's canonical image request carries no field for an image
    input and editing is therefore not implemented at all. An injected service is
    not validated, because it brings its own provider and this factory cannot know
    what that provider can run.

    ``router`` is the application's own router, or ``None`` for one built here. It
    exists for a caller that wants to supply a router of its own; the default is a
    fresh one rather than the framework's shared one, because the framework keys
    applications by their router's identity: two built on its default router are
    taken for one application being reloaded, and the second resets the first's
    routes. A deployment builds one application and is unaffected either way, so
    the default is the safe one.

    Governance is wired when the deployment configures
    ``CHORA_OBSERVABILITY_DSN``: the factory builds the chain — budget, outbox,
    idempotency, containment and egress over that database — passes it to the
    inference service it builds and to the gRPC listener's grounded search, and
    opens and closes the database with the process. When the deployment also
    configures ``CHORA_IDENTITY_GRPC_ADDR``, the mana meter is wired too: a
    channel to chora-identity is built with the chain and closed with the
    process, so every metered action quotes and debits against the learner's
    wallet. In the default ``chora`` governance mode the identity address is
    required whenever the DSN is set, so a governed deployment cannot silently
    run un-metered; only an explicit ``CHORA_GOVERNANCE_MODE=standalone`` allows
    the pair to be split. A deployment that configures no DSN gets no chain and
    no database, and the invocation flow runs exactly as it did before
    governance existed.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    configure_json()

    registry = registry or ModelRegistry.from_env()
    # One registry for the process, shared by the service that counts into it and
    # the endpoint that publishes it: two would be two different sets of numbers.
    metrics = Metrics()
    # Governance is built before the service, because the service is one of the
    # two things it gates: a deployment that configures a DSN gets the chain
    # wired into the inference service it builds and into the gRPC listener's
    # grounded search, and the database's lifetime is this process's — opened
    # at start, closed at stop. A deployment that configures none gets no
    # database and no chain, and every invocation runs exactly as it did
    # before governance existed.
    wiring = _governance(settings)
    manager = wiring.manager if wiring is not None else None
    db = wiring.db if wiring is not None else None
    # Tracing is configured before the service is built, so a deployment that asked
    # for it and gave it nowhere to send fails at startup rather than dropping
    # every span it produces.
    trace_provider = configure_tracing(settings)
    tracer = (
        trace_provider.get_tracer(SERVICE_NAME) if trace_provider is not None else None
    )

    if service is None:
        _ensure_binding_can_run(registry)
        _ensure_editing_is_not_advertised(registry)
        rate_limiter, tenant_limiter = _limits(settings)
        service = InferenceService(
            registry,
            LiteLLMProvider(),
            attempt_timeout=settings.provider_timeout_seconds,
            retry_count=settings.provider_retry_count,
            retry_backoff_ms=settings.provider_retry_backoff_ms,
            limiter=ConcurrencyLimiter(settings.provider_max_concurrent_requests),
            accounting=_accounting(settings),
            rate_limiter=rate_limiter,
            tenant_limiter=tenant_limiter,
            metrics=metrics,
            tracer=tracer,
            governance=manager,
        )

    api_keys = ApiKeySet.from_settings(settings)

    # The gRPC listener is built only when asked for, so a test
    # that builds an application never binds a port it does not intend to serve.
    # The production entrypoint asks for it; the factory's own default is the
    # HTTP-only application the tests drive. It starts and stops with the
    # application, because the application is the process's unit of lifetime.
    grpc_server = (
        GrpcServer(
            service=service,
            registry=registry,
            settings=settings,
            api_keys=api_keys,
            governance=manager,
        )
        if serve_grpc
        else None
    )

    # Every application gets its own router unless a caller supplies one. The
    # framework's default router is a module-level singleton and the framework keys
    # applications by their router's identity, so two built on it are taken for one
    # application being reloaded and the second resets the first's routes — a later
    # request is then served by an earlier application's handlers, silently. A
    # deployment builds one application and is unaffected either way, so the
    # default is the safe one rather than the shared one.
    app = Application(router=router if router is not None else Router())
    # Appended before the auth middleware so it is the outer one: a refusal raised
    # by authentication is then still inside the caller's trace, which is what an
    # operator wants to see when the refusals are the thing under investigation.
    app.middlewares.append(create_tracing_middleware())
    app.middlewares.append(
        create_auth_middleware(
            api_keys,
            auth_enabled=settings.auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
    )
    # Appended after authentication so an unidentified caller is told that it is
    # unidentified rather than something about this deployment's configuration.
    # Neither check reads the body, so the order costs nothing.
    app.middlewares.append(create_body_limit_middleware(settings.max_request_body_bytes))
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    register_metrics_route(app.router, metrics)
    register_model_routes(app.router, registry)
    register_chat_routes(app.router, service)
    register_completions_routes(app.router, service)
    register_responses_routes(app.router, service)
    register_embeddings_routes(app.router, service)
    register_image_routes(app.router, service)
    register_error_handlers(app)

    @app.on_start
    async def _on_start(app: Application) -> None:
        logger.info(
            "model-gateway starting",
            environment=settings.environment,
            http_port=settings.http_port,
            models=[info.id for info in registry.list_models()],
        )
        if db is not None:
            # The governance pool opens with the process, so a deployment that
            # configured a DSN and cannot reach its database fails at startup
            # rather than on the first invocation — and the pool is closed at
            # stop, because an unclosed pool keeps the event loop from exiting
            # cleanly.
            await db.connect()
        if grpc_server is not None:
            await grpc_server.start()

    @app.on_stop
    async def _on_stop(app: Application) -> None:
        logger.info("model-gateway stopped")
        if grpc_server is not None:
            await grpc_server.stop()
        if db is not None:
            await db.close()
        if wiring is not None and wiring.mana_channel is not None:
            # The mana channel opened with the chain, so it closes with the
            # process: an unclosed channel keeps the event loop from exiting
            # cleanly, the same reason the database pool does.
            await wiring.mana_channel.close()
        if trace_provider is not None:
            # A batch processor holds spans until it exports them, so a process that
            # stops without this loses its last batch — which is the one covering
            # the shutdown an operator is most likely to be looking at.
            trace_provider.shutdown()

    return app
