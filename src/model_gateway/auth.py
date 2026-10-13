"""How the gateway presents a configured key or header to an upstream route.

Provider-specific authentication — including Google ADC for Vertex routes —
is handled by LiteLLM itself; these values only describe the transport of a
credential the gateway has already been given (PRD section 7).
"""

from enum import StrEnum


class AuthType(StrEnum):
    """How the gateway authenticates to an upstream provider."""

    BEARER = "bearer"
    X_API_KEY = "x-api-key"
    API_KEY = "api-key"
    NONE = "none"
    CUSTOM_HEADER = "custom-header"
