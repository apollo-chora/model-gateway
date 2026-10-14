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
  The deadline reaches the provider as the budget on the attempt it is handed
  (:class:`AttemptContext`); a provider neither computes a deadline nor reads one
  out of configuration it does not own.
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

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from model_gateway.config.routes import Route
from model_gateway.domain import InvocationRequest, InvocationResponse

__all__ = ["AttemptContext", "TextProvider"]


@dataclass(frozen=True)
class AttemptContext:
    """The execution facts of one provider attempt (PRD sections 9.6 and 15).

    One attempt, not one invocation. This is what the provider needs to perform
    the single call it is given, and nothing about how the invocation was routed.

    ``invocation_id`` is the gateway's identifier for the invocation the attempt
    belongs to, so a provider can correlate its own upstream call with the
    gateway's record of it (PRD section 5.2). ``attempt_number`` is 1-based and
    counts the attempts the gateway has made for that invocation, including
    retries of the same route. ``timeout`` is the budget in seconds this attempt
    may use, drawn from the invocation deadline (PRD section 9.6) — the value a
    binding hands LiteLLM as its ``timeout``. ``None`` means the gateway set no
    budget for the attempt, which is not a budget of zero.

    It carries execution facts only, and never routing policy: which route to
    call, which credential to present, whether to retry and where to fall back
    are all decided by the gateway core before the attempt is made (PRD sections
    9.2 and 11.3), and the route arrives beside this object rather than inside
    it. The object exists so that the facts a later slice needs — a deadline to
    cancel against, a span to attribute a trace to — travel through one seam
    instead of one parameter per caller.
    """

    invocation_id: str | None
    attempt_number: int
    timeout: float | None

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError(
                f"attempt_number must be at least 1, got {self.attempt_number}; the gateway "
                f"counts the first attempt of an invocation as attempt 1"
            )
        if self.timeout is not None and not _is_usable_budget(self.timeout):
            raise ValueError(
                f"timeout must be a positive number of seconds or None, got {self.timeout!r}"
            )


class TextProvider(Protocol):
    """One upstream text provider: translation and execution, nothing more (PRD section 9).

    Implementations are structural — any object with a ``complete`` method of this
    shape satisfies the protocol, which is what lets a deployment bind LiteLLM,
    a test bind :class:`~model_gateway.providers.mock.MockProvider`, and the
    gateway core bind neither.

    ``route`` is the resolved route: its endpoint, credential, authentication
    type, headers, provider format and upstream model. ``request`` is the canonical
    invocation. ``context`` is this attempt's own execution facts — the
    invocation it belongs to, which attempt it is and the budget it may use
    (:class:`AttemptContext`). The provider returns the canonical response for
    that call.

    A provider that cannot honour the request raises; it does not degrade the
    request into something smaller. Dropping a tool, an image or a generation
    parameter because no canonical target exists upstream changes what the caller
    asked for, and PRD section 9.5 requires a structured capability error instead.
    """

    async def complete(
        self,
        route: Route,
        request: InvocationRequest,
        context: AttemptContext,
    ) -> InvocationResponse:
        """Call the upstream named by ``route`` and return the canonical response.

        Performs exactly one upstream call for exactly one route. It does not
        retry, does not fall back to another route, and does not choose between
        routes: those are the gateway core's (PRD sections 11.2 and 11.3). It
        applies the budget on ``context`` to the call it makes, so the upstream
        client is given the gateway's deadline rather than one of its own
        choosing (PRD section 9.6).
        """
        ...


def _is_usable_budget(value: float) -> bool:
    """Whether ``value`` is a number of seconds an attempt could actually be given."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return isfinite(value) and value > 0

