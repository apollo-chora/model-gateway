"""The inference service: gateway-owned policy around one provider call.

The service owns policy; the provider owns translation. That split is PRD
section 9's boundary, and this module is the policy side of it:

* **Routing.** The public model ID in the request is resolved to the routes
  configured for its operation by the registry. An unknown ID is a
  :class:`~model_gateway.errors.ModelNotFoundError` raised here, before any
  provider is called, so a caller-supplied model name cannot reach an
  unconfigured upstream endpoint (PRD section 10).
* **Capability enforcement.** The request is validated against the primary
  route's capability contract before anything is selected. The registry
  advertises the primary route's capabilities as the public model's own (PRD
  section 4.3), so a request the primary cannot honour is refused with a
  :class:`~model_gateway.errors.CapabilityError` even when a later route declares
  what the request needs (PRD section 9.5): serving it would answer a request the
  public contract never advertised for that model. Silently dropping a tool, an
  image or a structured-output requirement is forbidden, and so is obtaining one
  through a fallback. Only then are the remaining routes filtered, each by the
  same test, into the fallbacks that could serve the request unchanged, so a
  fallback may preserve the advertised capabilities but can never expand them.
* **Attempt execution.** The provider is handed one resolved route, the request
  and an :class:`~model_gateway.providers.base.AttemptContext` carrying the
  attempt's own facts: the invocation it belongs to, which attempt it is and the
  budget that attempt may use (PRD section 9.6). The budget is the gateway's, so
  a provider never reads a deadline out of configuration it does not own, and the
  core never wraps the call in a timeout the provider cannot see.
* **Normalisation.** The provider's answer comes back as upstream data and is
  re-normalised at this boundary: the public model that was asked for, the
  upstream model that actually served it and the invocation's own reliability
  facts are the gateway's to state, because the gateway chose the route and made
  the attempts (PRD sections 5.2 and 9).

The provider performs exactly one call and owns protocol translation and
execution and nothing else (:mod:`model_gateway.providers.base`).

**No retries, no fallbacks and no deadline policy yet.** Those are a later
slice's, and PRD section 11.3 gives the gateway core the only retry loop. The
call is structured so that slice can be added without reshaping it:
:func:`_select_routes` decides the primary route and the fallbacks that could
serve the request, :meth:`InferenceService._attempt` is a single provider call
against a single route, :meth:`InferenceService._invoke_on` owns the route order,
:class:`InvocationState` carries the reliability facts the response reports and
:class:`~model_gateway.providers.base.AttemptContext` is where a computed
deadline will travel. A retry wraps ``_attempt`` and advances
``InvocationState``; a fallback walks the fallbacks selection returns.
``fallback_used`` is already truthful without either, because it reports an
actual fallback path rather than any route other than the first configured one.
Adding a retry or a fallback must not change what ``invoke`` guarantees today.
"""

from dataclasses import dataclass, replace

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Route
from model_gateway.domain import (
    ImagePart,
    InvocationRequest,
    InvocationResponse,
    ResponseFormatMode,
)
from model_gateway.errors import CapabilityError
from model_gateway.providers.base import AttemptContext, TextProvider, _is_usable_budget
from model_gateway.registry import ModelRegistry

__all__ = ["InferenceService", "InvocationState"]


@dataclass(frozen=True)
class InvocationState:
    """What the gateway has done for one invocation so far (PRD sections 5.2 and 13).

    The response reports facts about the gateway's own execution — how many
    provider attempts were made and whether a fallback path was taken — and those
    are facts about the invocation rather than about any one answer a provider
    gave. They live here so that ``_normalise`` states them instead of inventing
    them: a hardcoded ``attempt_count`` of one is a claim about the reliability
    loop, which is exactly what this object can answer truthfully.

    ``attempt_count`` starts at zero, because a state exists before the first
    call, and counts the attempts that were made — including ones that raised,
    because a failed attempt is work the gateway did (PRD section 13).

    ``fallback_used`` records that the invocation took a *fallback path*: a route
    other than the primary one served it because the normal primary path could
    not (PRD section 5.2). It is deliberately narrower than "a route other than
    the first configured one was used", which was also true whenever capability
    routing passed the primary over. The primary route is the one the public
    model advertises, and a request it cannot honour is refused rather than
    routed elsewhere, so capability routing can never set this flag. With that
    definition, ``attempt_count=1`` and ``fallback_used=False`` says the primary
    route served the invocation on its first attempt, rather than merely that no
    later route was configured.

    Nothing here implements the reliability loop. :meth:`with_attempt` is the one
    place the state advances, so the retry and fallback slice has a single seam
    to move and no field to add in a second place.
    """

    attempt_count: int = 0
    fallback_used: bool = False

    def with_attempt(self, *, on_fallback: bool = False) -> "InvocationState":
        """The state after recording that one more provider attempt is made.

        The attempt is recorded before the call, so an attempt that raises counts
        too: the count describes what the gateway did, and a failure is not less
        of an attempt than a success (PRD section 13).

        ``on_fallback`` states that this attempt is made on a route other than
        the primary one, after the primary path could not serve the invocation.
        That is the one thing that makes ``fallback_used`` true (PRD section
        5.2), and it is sticky, because one such route is enough for the
        invocation to have fallen back. It defaults to false because an attempt
        on the primary route is never a fallback, however many routes are
        configured behind it.
        """
        return replace(
            self,
            attempt_count=self.attempt_count + 1,
            fallback_used=self.fallback_used or on_fallback,
        )


class InferenceService:
    """Policy for a text invocation: resolve, select, call, normalise."""

    def __init__(
        self,
        registry: ModelRegistry,
        provider: TextProvider,
        *,
        attempt_timeout: float | None = None,
    ) -> None:
        """Wire the service to the registry and the provider it calls through.

        ``attempt_timeout`` is the budget in seconds one attempt may use, handed
        to the provider on every attempt context (PRD section 9.6). It stands in
        for the per-invocation deadline the reliability slice will compute from
        the caller's deadline; ``None`` means the gateway sets no budget for an
        attempt, which is not the same as a budget of zero. The gateway core owns
        this value, and a value no attempt could use is refused here rather than
        on the first request.
        """
        if attempt_timeout is not None and not _is_usable_budget(attempt_timeout):
            raise ValueError(
                f"attempt_timeout must be a positive number of seconds or None, got "
                f"{attempt_timeout!r}"
            )
        self._registry = registry
        self._provider = provider
        self._attempt_timeout = attempt_timeout

    async def invoke(self, request: InvocationRequest) -> InvocationResponse:
        """Invoke the public model named by ``request.model``.

        Resolves the ID to its configured routes, validates the request against
        the primary route's capability contract — the contract the registry
        advertises for the model — makes one provider call and returns the
        normalised response. Raises
        :class:`~model_gateway.errors.ModelNotFoundError` for an unknown ID and
        :class:`~model_gateway.errors.CapabilityError` when the primary route
        cannot honour the request.
        """
        routes = self._registry.routes_for(request.model)
        return await self._invoke_on(request, routes)

    async def _invoke_on(
        self, request: InvocationRequest, routes: tuple[Route, ...]
    ) -> InvocationResponse:
        """Serve one invocation from the configured route order, primary first.

        Selection decides which routes may serve the request: the primary route
        — the one the public model advertises, which the request has already been
        validated against — and, behind it, the later routes that could serve it
        unchanged. Execution then makes one attempt against the primary route.
        Today that is one attempt deep: there are no retries and no failure
        fallbacks yet, so the primary route is the route and ``fallback_used``
        stays false, because no non-primary route served anything. The later
        reliability slice walks ``plan.fallbacks`` from here and wraps
        :meth:`_attempt`; it does not reshape what ``invoke`` returns.
        """
        plan = _select_routes(routes, request)
        state = InvocationState().with_attempt()
        response = await self._attempt(plan.primary, request, state)
        return _normalise(plan.primary, request, response, state)

    async def _attempt(
        self, route: Route, request: InvocationRequest, state: InvocationState
    ) -> InvocationResponse:
        """One provider call against one route.

        This is the unit the retry and fallback slice wraps: it holds no policy
        of its own beyond performing the call the gateway already decided on, and
        it hands the provider the attempt's own facts — which invocation, which
        attempt, and the budget that attempt may use (PRD section 9.6).
        """
        context = AttemptContext(
            invocation_id=request.invocation_id,
            attempt_number=state.attempt_count,
            timeout=self._attempt_timeout,
        )
        return await self._provider.complete(route, request, context)


@dataclass(frozen=True)
class _RoutePlan:
    """Which routes may serve one invocation, decided before any call is made.

    ``primary`` is the route the public model advertises — the first configured
    route — and the one every invocation is attempted on. ``fallbacks`` are the
    later routes that could serve the same request unchanged, in configured
    order: the candidates a genuine failure fallback may walk (PRD section 11.2).

    Empty ``fallbacks`` is not a refusal. It says only that the invocation has
    nowhere to go if the primary path fails, which is the ordinary case for a
    model with one configured route.
    """

    primary: Route
    fallbacks: tuple[Route, ...]


def _select_routes(routes: tuple[Route, ...], request: InvocationRequest) -> _RoutePlan:
    """The primary route that must serve ``request``, and the fallbacks that may.

    Capability enforcement happens here, in this order, because the registry
    advertises the *primary* route's capabilities as the public model's own
    (:class:`~model_gateway.registry.ModelInfo`). The primary route is therefore
    validated first: a request it cannot honour is refused with a
    :class:`~model_gateway.errors.CapabilityError` even when a later route
    declares what the request needs, because serving it would answer a request
    the public model never advertised (PRD sections 4.3 and 9.5).

    Only then are the remaining routes filtered, each by the same test, into the
    fallbacks. A route that does not declare everything the request needs is
    skipped rather than attempted, so a capability gap in a fallback cannot make
    an invocation fatal that the primary route could have served (PRD section
    11.2); and because the request was already validated against the primary,
    filtering afterwards is what keeps a fallback from expanding the advertised
    capabilities. A fallback may preserve them, never add to them.

    This is not a fallback walk: nothing is retried and no route is abandoned for
    having failed, so the plan is a decision about eligibility alone.

    Raises :class:`~model_gateway.errors.CapabilityError` when the primary route
    cannot serve the request.
    """
    if not routes:
        raise _no_eligible_route(request, None)
    primary = routes[0]
    missing = _first_missing_capability(primary, request)
    if missing is not None:
        raise _no_eligible_route(request, missing)
    return _RoutePlan(
        primary=primary,
        fallbacks=tuple(
            route for route in routes[1:] if _first_missing_capability(route, request) is None
        ),
    )


def _first_missing_capability(route: Route, request: InvocationRequest) -> Capability | None:
    """The first capability ``route`` lacks for ``request``, or ``None`` if it has them all.

    The order is the order :func:`_required_capabilities` reports, so the
    capability a refusal names is deterministic.
    """
    for capability in _required_capabilities(request):
        if capability not in route.capabilities:
            return capability
    return None


def _no_eligible_route(
    request: InvocationRequest, missing: Capability | None
) -> CapabilityError:
    """The refusal for a request the advertised model cannot serve (PRD section 9.5).

    ``missing`` is the first capability the primary route lacks for this request,
    which is the capability the public model does not advertise. It is ``None``
    only when the operation has no routes at all, which the registry refuses
    before selection runs.

    A later route declaring the capability does not change the answer: the
    public contract is the primary route's, and a fallback may preserve it but
    never expand it.
    """
    if missing is None:
        return CapabilityError(
            f"no route is configured for {request.model!r}, so this invocation cannot be served"
        )
    return CapabilityError(
        f"model {request.model!r} does not advertise {missing.value!r}: its primary route does "
        f"not declare that capability, so this invocation cannot be served unchanged",
        details={"capability": missing.value},
    )


def _required_capabilities(request: InvocationRequest) -> tuple[Capability, ...]:
    """The capabilities a text invocation asks of the route that serves it.

    A chat invocation needs the chat capability; declaring tools needs the tools
    capability, because a provider that cannot call tools cannot be made to; an
    image in the conversation needs the vision capability, because dropping it
    would change what the caller said (PRD sections 9.4 and 9.5); and a
    generation config carrying a non-text response format needs the
    structured-output capability, because a route that cannot honour the schema
    would otherwise answer with free text while the caller believed its
    requirement had been met (PRD section 9.5). A text-mode response format asks
    for nothing beyond chat: it is the provider's own default.

    A system instruction asks for nothing beyond chat. It is baseline chat
    semantics that every chat binding must map or refuse (PRD sections 4.4, 9.1
    and 9.5), so a route declaring chat already claims to handle it, and a
    capability of its own would move that obligation somewhere it could be left
    unset (:class:`~model_gateway.capabilities.Capability`).

    The first missing capability is reported, in this order, so the refusal is
    deterministic.
    """
    required: list[Capability] = [Capability.CHAT]
    if request.tools:
        required.append(Capability.TOOLS)
    if _carries_images(request):
        required.append(Capability.VISION)
    if _requires_structured_output(request):
        required.append(Capability.STRUCTURED_OUTPUT)
    return tuple(required)


def _requires_structured_output(request: InvocationRequest) -> bool:
    """Whether the generation config asks for an output shape a provider must honour."""
    response_format = request.generation_config.response_format
    return response_format is not None and response_format.mode is not ResponseFormatMode.TEXT


def _carries_images(request: InvocationRequest) -> bool:
    """Whether any message in the conversation carries an image part."""
    return any(
        isinstance(part, ImagePart)
        for message in request.messages
        for part in message.content
    )


def _normalise(
    route: Route,
    request: InvocationRequest,
    response: InvocationResponse,
    state: InvocationState,
) -> InvocationResponse:
    """Stamp the gateway's own facts onto a provider's answer.

    A provider's response is upstream data, never forwarded unchanged (PRD
    section 9). The gateway chose the route, so it states which public model was
    asked for and which upstream model answered; and the gateway made the
    attempts, so ``attempt_count`` and ``fallback_used`` are read from ``state``
    rather than assumed. A provider cannot report how many times the gateway
    called, and neither may this function invent the number.

    ``invocation_id`` is always the gateway's own, never the provider's. The
    gateway mints the identifier before the call, and PRD section 5.2 describes
    the response field as the gateway invocation identifier, so a value a
    provider supplies is discarded: a buggy or hostile provider must not be able
    to rename the invocation in the caller's own record of it. A provider's own
    request or response identifier is a different fact and must not be published
    here; it belongs in a field of its own — an upstream request ID reported
    beside ``upstream_model`` — and until such a field exists it is not reported
    at all, rather than passed off as the invocation's identity.
    """
    return replace(
        response,
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=request.invocation_id,
        attempt_count=state.attempt_count,
        fallback_used=state.fallback_used,
    )
