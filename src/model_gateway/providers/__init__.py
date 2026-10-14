"""Upstream providers: the protocol a binding implements and the deterministic stand-in.

:mod:`model_gateway.providers.base` is the protocol — the whole of the gateway's
upstream boundary (PRD section 9). :mod:`model_gateway.providers.mock` is the
deterministic implementation for tests and the vertical slice.

Neither module knows about HTTP, gRPC, retry or accounting: a provider translates
and executes one call against one route, and the gateway core owns everything
around it (PRD sections 9.2 and 23).
"""

from model_gateway.providers.base import TextProvider
from model_gateway.providers.mock import MockProvider, MockSpec

__all__ = ["MockProvider", "MockSpec", "TextProvider"]
