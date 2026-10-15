"""Upstream providers: the protocol a binding implements and the deterministic stand-in.

:mod:`model_gateway.providers.base` is the protocol — the whole of the gateway's
upstream boundary. :mod:`model_gateway.providers.mock` is the deterministic
implementation for tests and the vertical slice.

Neither module knows about HTTP, gRPC, retry or accounting: a provider translates
and executes one call against one route, and the gateway core owns everything
around it.
"""

from model_gateway.providers.base import AttemptContext, TextProvider
from model_gateway.providers.mock import MockProvider, MockSpec

__all__ = ["AttemptContext", "MockProvider", "MockSpec", "TextProvider"]
