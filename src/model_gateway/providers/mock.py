"""A deterministic text provider for tests and the vertical slice (PRD sections 9 and 20).

:class:`MockProvider` satisfies :class:`~model_gateway.providers.base.TextProvider`
without touching a network, a vendor SDK or LiteLLM. It exists for two reasons.

For tests, it is the stand-in PRD section 20 asks for: a provider whose behaviour
is a pure function of its inputs, so a test can assert exactly what a route
received and exactly what it returned, and so failure paths (a provider that
raises) are testable without a failing server somewhere.

For the vertical slice it is the provider an application factory can wire before
a real binding exists: the gateway's own routing, policy, capability enforcement
and error taxonomy can be exercised end to end with no upstream account.

What it echoes is the point. The response carries the public model ID the caller
asked for, the upstream model ID resolved from the route, and the last user
message text, so a test can prove the round trip happened — that the request
reached the provider and the response came back — without parsing a vendor's wire
format. Nothing about the response is random, time-dependent or environment
dependent, so two calls with the same route and request produce equal responses.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from model_gateway.config.routes import Route
from model_gateway.domain import (
    FinishReason,
    InvocationRequest,
    InvocationResponse,
    Role,
    TextPart,
    ToolCall,
    Usage,
)
from model_gateway.errors import GatewayError
from model_gateway.providers.base import AttemptContext

__all__ = ["MockProvider", "MockSpec"]


@dataclass(frozen=True)
class MockSpec:
    """What one mocked call returns.

    ``text`` is the text part of the answer. ``None`` — the default — echoes the
    last user message's text, which needs no fixture to prove the round trip.

    ``tool_calls`` are returned after the text, in order, so a test can prove a
    tool call survives with its identifier and arguments (PRD section 9.3).

    ``finish_reason`` classifies why generation stopped and ``raw_finish_reason``
    records the provider's own reason verbatim beside it. When ``raw_finish_reason``
    is ``None`` the classification's own value is recorded, which is always a
    consistent pair; pass a raw reason to model a provider whose vocabulary the
    gateway has not been taught (PRD section 5.2).

    ``usage`` is the token record reported for the call, all counts unknown by
    default so an unreported count is never presented as zero.

    ``error`` makes the call raise that gateway error instead of returning, so a
    failure path is testable with the same determinism as a success. The error is
    raised unchanged: it has already been classified, and re-classifying it would
    demote a precise reason to an internal one.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    raw_finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    error: GatewayError | None = None


class MockProvider:
    """A deterministic :class:`~model_gateway.providers.base.TextProvider`.

    One spec serves every call; a sequence of specs serves one per call in order
    and then repeats the last, so a test can script a failure followed by a
    recovery — or the reverse — and still know exactly what the third call will
    get.

    Every call is recorded as a ``(route, request, context)`` triple in
    :attr:`calls`, which is how a test asserts the gateway handed the provider
    the route it resolved, the request it was given and the attempt's own facts.
    """

    def __init__(self, spec: MockSpec | Sequence[MockSpec] | None = None) -> None:
        chosen = MockSpec() if spec is None else spec
        specs = (chosen,) if isinstance(chosen, MockSpec) else tuple(chosen)
        if not specs:
            raise ValueError("a mock provider needs at least one response to return")
        self._specs = specs
        self.calls: list[tuple[Route, InvocationRequest, AttemptContext]] = []

    async def complete(
        self, route: Route, request: InvocationRequest, context: AttemptContext
    ) -> InvocationResponse:
        """Return the scripted response for ``route`` and ``request``.

        The result is a pure function of the scripted spec, the route and the
        request: no network, no clock, no randomness, no environment. The
        attempt's own facts are recorded beside the call and change nothing about
        the answer, so a scripted response is the same response whatever budget
        the gateway set for the attempt.
        """
        self.calls.append((route, request, context))
        spec = self._specs[min(len(self.calls), len(self._specs)) - 1]
        if spec.error is not None:
            raise spec.error
        return _response(route, request, spec)


def _response(route: Route, request: InvocationRequest, spec: MockSpec) -> InvocationResponse:
    """Build the canonical response for one mocked call."""
    raw = spec.raw_finish_reason if spec.raw_finish_reason is not None else spec.finish_reason.value
    text = spec.text if spec.text is not None else _echo(request)
    return InvocationResponse(
        model=request.model,
        upstream_model=route.model_id,
        invocation_id=request.invocation_id,
        content=(TextPart(text=text), *spec.tool_calls),
        finish_reason=spec.finish_reason,
        raw_finish_reason=raw,
        usage=spec.usage,
    )


def _echo(request: InvocationRequest) -> str:
    """The last user message's text, the mock's default answer.

    A multi-turn conversation is echoed at its newest user turn, which is the
    part of the request a reader would expect an answer to. A conversation with
    no user message echoes nothing.
    """
    for message in reversed(request.messages):
        if message.role is Role.USER:
            return "".join(part.text for part in message.content if isinstance(part, TextPart))
    return ""
