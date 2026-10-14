"""The inference service: gateway-owned policy around one provider call.

The service owns policy; the provider owns translation. That split is PRD
section 9's boundary, and this module is the policy side of it:

* **Routing.** The public model ID in the request is resolved to the routes
  configured for its operation by the registry. An unknown ID is a
  :class:`~model_gateway.errors.ModelNotFoundError` raised here, before any
  provider is called, so a caller-supplied model name cannot reach an
  unconfigured upstream endpoint (PRD section 10).
* **Capability enforcement.** The route the gateway chose must declare the
  capability the request needs, or the invocation is refused with a
  :class:`~model_gateway.errors.CapabilityError` (PRD section 9.5). Silently
  dropping a tool or an image is forbidden, so an unsupported request is a
  refusal and never a smaller call.
* **Normalisation.** The provider's answer comes back as upstream data and is
  re-normalised at this boundary: the public model that was asked for and the
  upstream model that actually served it are the gateway's to state, because
  the gateway chose the route (PRD sections 5.2 and 9).

The provider is handed exactly one resolved route and performs exactly one call.
It owns protocol translation and execution and nothing else
(:mod:`model_gateway.providers.base`).

**No retries and no fallbacks yet.** Those are a later slice's, and PRD section
11.3 gives the gateway core the only retry loop. The call is structured so that
slice can be added without reshaping it: :meth:`InferenceService._attempt` is a
single provider call against a single route, :meth:`InferenceService._invoke_on`
owns the route order and :meth:`InferenceService.invoke` owns the request. A
retry wraps ``_attempt``; a fallback walks the routes ``_invoke_on`` is given.
Adding either must not change what ``invoke`` guarantees today.
"""

from dataclasses import replace

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Route
from model_gateway.domain import ImagePart, InvocationRequest, InvocationResponse
from model_gateway.errors import CapabilityError
from model_gateway.providers.base import TextProvider
from model_gateway.registry import ModelRegistry

__all__ = ["InferenceService"]


class InferenceService:
    """Policy for a text invocation: resolve, enforce, call, normalise."""

    def __init__(self, registry: ModelRegistry, provider: TextProvider) -> None:
        self._registry = registry
        self._provider = provider

    async def invoke(self, request: InvocationRequest) -> InvocationResponse:
        """Invoke the public model named by ``request.model``.

        Resolves the ID to its configured routes, enforces the capabilities the
        request needs against the primary route, makes one provider call and
        returns the normalised response. Raises
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

        Today that order is walked one route deep: there are no retries and no
        fallbacks yet, so the primary route is the route. The later reliability
        slice walks this method's ``routes`` and wraps :meth:`_attempt`; it does
        not reshape what ``invoke`` returns.
        """
        route = routes[0]
        _enforce_capabilities(route, request)
        response = await self._attempt(route, request)
        return _normalise(route, request, response)

    async def _attempt(self, route: Route, request: InvocationRequest) -> InvocationResponse:
        """One provider call against one route.

        This is the unit the retry and fallback slice wraps: it holds no policy
        of its own beyond performing the call the gateway already decided on.
        """
        return await self._provider.complete(route, request)


def _enforce_capabilities(route: Route, request: InvocationRequest) -> None:
    """Refuse a request the chosen route cannot honour (PRD section 9.5).

    A chat invocation needs the chat capability; declaring tools needs the tools
    capability, because a provider that cannot call tools cannot be made to; and
    an image in the conversation needs the vision capability, because dropping
    it would change what the caller said (PRD sections 9.4 and 9.5). The first
    missing capability is reported, in that order, so the refusal is
    deterministic.
    """
    for capability in _required_capabilities(request):
        if capability not in route.capabilities:
            raise CapabilityError(
                f"the route serving {request.model!r} does not support "
                f"{capability.value!r}, so this invocation cannot be sent to it "
                f"unchanged",
                details={"capability": capability.value},
            )


def _required_capabilities(request: InvocationRequest) -> tuple[Capability, ...]:
    """The capabilities a text invocation asks of the route that serves it."""
    required: list[Capability] = [Capability.CHAT]
    if request.tools:
        required.append(Capability.TOOLS)
    if _carries_images(request):
        required.append(Capability.VISION)
    return tuple(required)


def _carries_images(request: InvocationRequest) -> bool:
    """Whether any message in the conversation carries an image part."""
    return any(
        isinstance(part, ImagePart)
        for message in request.messages
        for part in message.content
    )


def _normalise(
    route: Route, request: InvocationRequest, response: InvocationResponse
) -> InvocationResponse:
    """Stamp the gateway's own facts onto a provider's answer.

    A provider's response is upstream data, never forwarded unchanged (PRD
    section 9). The gateway chose the route, so it states which public model was
    asked for and which upstream model answered; and with no retries and no
    fallbacks yet, one attempt on the primary route is the whole of the
    reliability record. A provider that did not echo the invocation identifier
    is given the one the caller supplied, so a response always identifies the
    invocation it answers (PRD section 5.2).
    """
    invocation_id = response.invocation_id
    if invocation_id is None:
        invocation_id = request.invocation_id
    return replace(
        response,
        model=request.model,
        upstream_model=route.model_id,
        attempt_count=1,
        fallback_used=False,
        invocation_id=invocation_id,
    )
