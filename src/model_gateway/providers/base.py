"""The provider protocol: the whole of the gateway's upstream boundary (PRD section 9).

A provider performs upstream protocol translation and execution, and nothing
else. Given a route and a canonical request, it calls the upstream endpoint the
route names, translates between the provider's wire format and the canonical
domain, and returns the canonical response. That is the entire job.

The boundary is deliberate, and it is what makes the gateway vendor-neutral
(PRD section 9 and principle 1): a provider speaks one vendor's protocol, the
gateway core speaks the canonical model, and adding a provider means adding a
translation rather than editing gateway business logic. LiteLLM is the first
implementation of this protocol; :mod:`model_gateway.providers.mock` is a
deterministic one for tests and the vertical slice.

What a provider must not own (PRD section 9.2):

* **Routing.** The gateway hands the provider exactly one resolved route. A
  provider never chooses between routes, and never sees the fallback list.
* **Authentication decisions.** The route carries the credential and
  authentication type the gateway chose (PRD section 7); a provider constructs
  the provider's native header from what it is given and decides nothing.
* **Retry and fallback policy.** The gateway core owns the retry loop and the
  single invocation deadline (PRD sections 11.3 and 15), so a provider performs
  one call and returns one result. A provider that retried on its own would be a
  second retry loop stacked on the gateway's, which PRD section 11.3 prohibits.
* **Idempotency and accounting.** Deduplication, token counts and cost are
  gateway records (PRD sections 12 and 13); a provider reports what the upstream
  said and the gateway decides what that means.
* **Error normalisation.** A provider may raise provider-shaped exceptions. The
  gateway classifies them into its own taxonomy and sanitises them before a
  caller sees them (PRD section 5.4), so a provider never writes a message a
  caller will read.
* **The public contract.** A provider returns canonical types and knows nothing
  about HTTP status codes, gRPC names or the OpenAI wire shape.
"""

from typing import Protocol

from model_gateway.config.routes import Route
from model_gateway.domain import InvocationRequest, InvocationResponse

__all__ = ["TextProvider"]


class TextProvider(Protocol):
    """One upstream text provider: translation and execution, nothing more (PRD section 9).

    Implementations are structural — any object with a ``complete`` method of this
    shape satisfies the protocol, which is what lets a deployment bind LiteLLM,
    a test bind :class:`~model_gateway.providers.mock.MockProvider`, and the
    gateway core bind neither.

    ``route`` is the resolved route: its endpoint, credential, authentication
    type, headers, provider format and upstream model. ``request`` is the canonical
    invocation. The provider returns the canonical response for that call.

    A provider that cannot honour the request raises; it does not degrade the
    request into something smaller. Dropping a tool, an image or a generation
    parameter because no canonical target exists upstream changes what the caller
    asked for, and PRD section 9.5 requires a structured capability error instead.
    """

    async def complete(self, route: Route, request: InvocationRequest) -> InvocationResponse:
        """Call the upstream named by ``route`` and return the canonical response.

        Performs exactly one upstream call for exactly one route. It does not
        retry, does not fall back to another route, and does not choose between
        routes: those are the gateway core's (PRD sections 11.2 and 11.3).
        """
        ...
