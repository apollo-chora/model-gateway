"""Upstream wire formats (PRD sections 6.1-6.4).

A format identifies the request/response protocol, not the company operating
the endpoint, so the same format can serve any compatible provider.
"""

from enum import StrEnum


class TextFormat(StrEnum):
    """Wire formats usable for text generation."""

    CHAT_COMPLETIONS = "chat-completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    VERTEX = "vertex"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ImageFormat(StrEnum):
    """Wire formats usable for image generation or editing."""

    IMAGES = "images"
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat-completions"
    VERTEX = "vertex"
    GEMINI = "gemini"


class EmbeddingFormat(StrEnum):
    """Wire formats usable for embeddings."""

    EMBEDDINGS = "embeddings"


class SearchFormat(StrEnum):
    """Wire formats usable for grounded search."""

    EXA = "exa"
    BRAVE = "brave"
    TAVILY = "tavily"
    SERPER = "serper"
    CUSTOM_HTTP = "custom-http"
