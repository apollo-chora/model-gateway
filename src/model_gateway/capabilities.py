"""Capabilities a configured route can offer."""

from enum import StrEnum


class Capability(StrEnum):
    """A behaviour a provider route may declare support for.

    Values match the vocabulary used in the environment configuration so that
    ``TEXT_LLM_CAPABILITIES`` and friends parse without translation.

    ``STRUCTURED_OUTPUT`` is required of a route whenever the request's
    generation config asks for a non-text response format. Structured-output
    requirements are among the semantics the gateway must never discard, so a
    route that cannot honour a schema has to refuse the invocation rather than
    answer with free text while the caller believes its requirement was met.
    Without a member for it the requirement could not be declared at all, and
    the gateway would report having enforced capabilities it never checked.

    A system instruction deliberately has no member of its own; it is baseline
    chat semantics. An instruction is part of what a chat endpoint accepts, it
    is among the semantics that must not be silently discarded, and every
    binding must map each semantically relevant element of the request or
    reject the invocation as unsupported. A per-route ``system_instructions``
    capability would restate an
    obligation the chat binding already carries, and restating it weakens it: a
    route could declare ``chat`` and still be handed an instruction it drops,
    because the check that mattered would be the one nobody configured. Chat is
    therefore the capability that carries it, and a binding that cannot map an
    instruction refuses the invocation instead of degrading it.
    """

    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    STRUCTURED_OUTPUT = "structured-output"
    EMBEDDINGS = "embeddings"
    GENERATION = "generation"
    EDITING = "editing"
