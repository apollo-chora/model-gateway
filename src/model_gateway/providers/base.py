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

Streaming is a separate capability (:class:`StreamingTextProvider`), not a mode
of :class:`TextProvider`: a provider that cannot stream is refused rather than
asked to fake token-by-token output from a whole answer (PRD sections 5.3, 16
and 9.5). Embeddings are separate in the same way
(:class:`EmbeddingProvider`): the operation is a different upstream call with a
different answer, so a provider that cannot embed is refused rather than handed
an embeddings request it would answer with something else (PRD sections 4.7,
6.3 and 9.6).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from model_gateway.config.routes import Route
from model_gateway.domain import InvocationRequest, InvocationResponse
from model_gateway.domain.embeddings import EmbeddingRequest, EmbeddingResponse
from model_gateway.domain.streaming import StreamEvent

__all__ = [
    "AttemptContext",
    "EmbeddingProvider",
    "StreamingTextProvider",
    "TextProvider",
]


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

        Performs one gateway attempt for exactly one route. It does not
        retry, does not fall back to another route, and does not choose between
        routes: those are the gateway core's (PRD sections 11.2 and 11.3). It
        applies the budget on ``context`` to the call it makes, so the upstream
        client is given the gateway's deadline rather than one of its own
        choosing (PRD section 9.6).

        One attempt is not the same as one upstream request. An implementation
        may reach a library that retries internally without the gateway's
        knowledge, so the wire-request count can exceed the attempt count and
        must never be reported as it.
        """
        ...


@runtime_checkable
class StreamingTextProvider(Protocol):
    """A text provider that can also stream its answer (PRD sections 5.3 and 16).

    Streaming is a separate capability from :class:`TextProvider`, not a mode of
    it, because a provider that cannot stream must be *refused* rather than
    faked: answering a ``stream: true`` request with a complete, non-streamed
    body would tell the caller it received tokens as they were produced when it
    did not (PRD sections 16 and 9.5). A provider either produces a real
    upstream stream or the gateway rejects the request with a capability error
    naming ``streaming``; there is no third option.

    The protocol is structural and runtime-checkable, so the gateway core can
    ask ``isinstance(provider, StreamingTextProvider)`` before it promises a
    caller a stream and raise the refusal naming the operation when the answer
    is no. A provider that implements :meth:`stream` satisfies it; one that does
    not is simply not a streaming provider, and no attribute is faked onto it.

    ``route``, ``request`` and ``context`` carry the same facts
    :class:`TextProvider` is handed — the resolved route, the canonical request
    and this attempt's own execution facts (:class:`AttemptContext`). The result
    is an async iterator of canonical :class:`~model_gateway.domain.streaming.StreamEvent`
    objects: the provider translates each upstream chunk and numbers the events
    it emits, 0-based and strictly increasing within the invocation, and the
    gateway core applies its own policy — capability enforcement, the one
    deadline, the concurrency limiter, the no-fallback-after-first-output rule —
    around the iteration.
    """

    def stream(
        self,
        route: Route,
        request: InvocationRequest,
        context: AttemptContext,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the upstream named by ``route`` as canonical stream events.

        Performs one gateway attempt for exactly one route, under the same
        policy as :meth:`TextProvider.complete`: it does not retry, does not
        fall back to another route, and does not choose between routes (PRD
        sections 9.2 and 11.3). It applies the budget on ``context`` to the
        call it makes and translates each chunk the upstream produces into
        canonical events, numbering them from 0 in the order the answer reads.

        A provider that cannot stream does not implement this method at all;
        the capability refusal is the gateway core's, raised before the first
        event is promised to a caller.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A provider that can turn text into vectors (PRD sections 4.7, 6.3 and 9.6).

    Embedding is a capability of its own, not a mode of :class:`TextProvider`: it
    is a different upstream operation with a different answer — vectors instead
    of an answer in words — so a provider that cannot embed must be *refused*
    rather than handed the request and left to improvise. There is no honest
    degradation here: a chat provider asked for embeddings could only invent a
    vector or fail, and either would tell the caller its request was served when
    it was not (PRD section 9.5).

    The protocol is structural and runtime-checkable, so the gateway core can ask
    ``isinstance(provider, EmbeddingProvider)`` before it accepts an embeddings
    request and raise a capability error naming ``embeddings`` when the answer is
    no. A provider that implements :meth:`embed` satisfies it; one that does not
    is simply not an embedding provider, and no attribute is faked onto it.

    ``route`` is the resolved route — its endpoint, credential, authentication
    type, headers, provider dialect and upstream model — and it is the embeddings
    route the registry selected for the public model, never a chat route reached
    through the same provider. ``request`` is the canonical embeddings request,
    whose ``input`` is the texts to embed in the caller's order. ``context`` is
    this attempt's own execution facts (:class:`AttemptContext`), carrying the
    budget the attempt may use exactly as it does for
    :meth:`TextProvider.complete`.

    What the provider returns is upstream data translated into the canonical
    shape: one vector per input text, each carrying the position it reported, the
    usage the upstream reported, and the invocation the attempt belongs to. The
    gateway core imposes the caller's ordering and checks the vectors against the
    dimensionality the route declares, so a provider that returned a
    wrong-shaped answer is refused there rather than passing it on.
    """

    async def embed(
        self,
        route: Route,
        request: EmbeddingRequest,
        context: AttemptContext,
    ) -> EmbeddingResponse:
        """Call the upstream named by ``route`` and return the canonical vectors.

        Performs one gateway attempt for exactly one route, under the same policy
        as :meth:`TextProvider.complete`: it does not retry, does not fall back to
        another route, and does not choose between routes (PRD sections 9.2 and
        11.3). It applies the budget on ``context`` to the call it makes, so the
        upstream client is given the gateway's deadline rather than one of its
        own choosing (PRD section 9.6).

        The vectors come back in the order the upstream reported them, each
        carrying its own index. Ordering them against the caller's input is the
        gateway core's work, not the provider's: the provider reports what the
        upstream said, and a provider that silently reordered or renumbered the
        answer would hide a difference the gateway is required to notice.

        One attempt is not the same as one upstream request. An implementation
        may reach a library that retries internally without the gateway's
        knowledge, so the wire-request count can exceed the attempt count and
        must never be reported as it.
        """
        ...


def _is_usable_budget(value: float) -> bool:
    """Whether ``value`` is a number of seconds an attempt could actually be given."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return isfinite(value) and value > 0

