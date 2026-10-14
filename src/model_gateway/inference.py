"""The inference service: gateway-owned policy around one provider call.

The service owns policy; the provider owns translation. That split is PRD
section 9's boundary, and this module is the policy side of it:

* **Routing.** The public model ID in the request is resolved to the routes
  configured for its operation by the registry. An unknown ID is a
  :class:`~model_gateway.errors.ModelNotFoundError` raised here, before any
  provider is called, so a caller-supplied model name cannot reach an
  unconfigured upstream endpoint (PRD section 10).
* **Capability enforcement.** A route may serve the request only when it
  declares every capability the request needs, or the invocation is refused with
  a :class:`~model_gateway.errors.CapabilityError` (PRD section 9.5). Silently
  dropping a tool, an image or a structured-output requirement is forbidden, so
  an unsupported request is a refusal and never a smaller call. The check belongs
  to selection rather than to execution: an ineligible route is passed over for
  the next eligible one, so a capability gap in a fallback route cannot make an
  invocation fatal that another configured route could have served.
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
:func:`_select_route` picks the route, :meth:`InferenceService._attempt` is a
single provider call against a single route, :meth:`InferenceService._invoke_on`
owns the route order, :class:`InvocationState` carries the reliability facts the
response reports and :class:`~model_gateway.providers.base.AttemptContext` is
where a computed deadline will travel. A retry wraps ``_attempt`` and advances
``InvocationState``; a fallback walks the routes selection returns. Adding either
must not change what ``invoke`` guarantees today.
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
    provider attempts were made and whether a route other than the first
    configured one was used — and those are facts about the invocation rather
    than about any one answer a provider gave. They live here so that
    ``_normalise`` states them instead of inventing them: a hardcoded
    ``attempt_count`` of one is a claim about the reliability loop, which is
    exactly what this object can answer truthfully.

    ``attempt_count`` starts at zero, because a state exists before the first
    call, and counts the attempts that were made — including ones that raised,
    because a failed attempt is work the gateway did (PRD section 13).
    ``fallback_used`` records whether any route other than the first configured
    one was used (PRD section 5.2).

    Nothing here implements the reliability loop. :meth:`with_attempt` is the one
    place the state advances, so the retry and fallback slice has a single seam
    to move and no field to add in a second place.
    """

    attempt_count: int = 0
    fallback_used: bool = False

    def with_attempt(self, *, fallback_used: bool = False) -> "InvocationState":
        """The state after recording that one more provider attempt is made.

        The attempt is recorded before the call, so an attempt that raises counts
        too: the count describes what the gateway did, and a failure is not less
        of an attempt than a success (PRD section 13). ``fallback_used`` is
        sticky, because one route other than the first configured one is enough
        for the invocation to have fallen back.
        """
        return replace(
            self,
            attempt_count=self.attempt_count + 1,
            fallback_used=self.fallback_used or fallback_used,
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

        Resolves the ID to its configured routes, selects the first route that
        can honour the request, makes one provider call and returns the
        normalised response. Raises
        :class:`~model_gateway.errors.ModelNotFoundError` for an unknown ID and
        :class:`~model_gateway.errors.CapabilityError` when no configured route
        can honour the request.
        """
        routes = self._registry.routes_for(request.model)
        return await self._invoke_on(request, routes)

    async def _invoke_on(
        self, request: InvocationRequest, routes: tuple[Route, ...]
    ) -> InvocationResponse:
        """Serve one invocation from the configured route order, primary first.

        Selection decides which route may serve the request; execution then makes
        one attempt against it. Today that is one attempt deep: there are no
        retries and no fallbacks yet, so the first eligible route is the route.
        The later reliability slice walks ``routes`` from here and wraps
        :meth:`_attempt`; it does not reshape what ``invoke`` returns.
        """
        route, is_fallback = _select_route(routes, request)
        state = InvocationState().with_attempt(fallback_used=is_fallback)
        response = await self._attempt(route, request, state)
        return _normalise(route, request, response, state)

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


def _select_route(routes: tuple[Route, ...], request: InvocationRequest) -> tuple[Route, bool]:
    """The first configured route that can serve ``request``, and whether it is a fallback.

    Route order is fallback order (PRD section 11.2), so eligibility is decided
    here rather than after a route has been chosen: a route that cannot honour
    the request is passed over for the next one, and the primary route is still
    tried first because it is first in the order. This is not a fallback walk —
    nothing is retried and no route is abandoned for having failed — it only
    keeps an ineligible route from being attempted at all, where it could produce
    the refusal the whole invocation would otherwise die of.

    The second element is ``True`` when the chosen route is not the first
    configured one, which is what ``fallback_used`` reports (PRD section 5.2).

    Raises :class:`~model_gateway.errors.CapabilityError` when no configured
    route can serve the request.
    """
    missing: Capability | None = None
    for index, route in enumerate(routes):
        absent = _first_missing_capability(route, request)
        if absent is None:
            return route, index > 0
        if missing is None:
            missing = absent
    raise _no_eligible_route(request, missing)


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
    """The refusal for a request no configured route can serve (PRD section 9.5).

    ``missing`` is the first capability the first configured route lacks — the
    reason selection started walking — so the refusal is deterministic rather
    than whichever route happened to be checked last. It is ``None`` only when
    the operation has no routes at all, which the registry refuses before
    selection runs.
    """
    if missing is None:
        return CapabilityError(
            f"no route is configured for {request.model!r}, so this invocation cannot be served"
        )
    return CapabilityError(
        f"no route serving {request.model!r} supports {missing.value!r}, so this invocation "
        f"cannot be sent to any configured route unchanged",
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
