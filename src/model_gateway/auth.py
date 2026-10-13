"""Authentication mechanisms for upstream calls and inbound clients (PRD section 7)."""

from enum import StrEnum


class AuthType(StrEnum):
    """How the gateway authenticates to an upstream provider."""

    BEARER = "bearer"
    X_API_KEY = "x-api-key"
    API_KEY = "api-key"
    NONE = "none"
    GOOGLE_ADC = "google-adc"
    CUSTOM_HEADER = "custom-header"
