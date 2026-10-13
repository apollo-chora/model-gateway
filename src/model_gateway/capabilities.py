"""Capabilities a configured route can offer (PRD sections 6, 9, 10)."""

from enum import StrEnum


class Capability(StrEnum):
    """A behaviour a provider route may declare support for.

    Values match the vocabulary used in the PRD's environment configuration so
    that ``TEXT_LLM_CAPABILITIES`` and friends parse without translation.
    """

    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    GENERATION = "generation"
    EDITING = "editing"
