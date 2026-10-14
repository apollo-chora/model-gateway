"""The LiteLLM text provider: one upstream call through the pinned binding (PRD section 9).

This is the production :class:`~model_gateway.providers.base.TextProvider`. It
adds execution to the transport-agnostic translation of
:mod:`model_gateway.providers.litellm_translation`: build the call's parameters,
perform exactly one upstream request through ``litellm.acompletion`` with the
safe outbound client injected, and normalise the answer into the canonical
response. What it owns is PRD section 9's boundary and nothing more — routing,
retry policy, accounting, identity and the public contract stay in the gateway
core (:mod:`model_gateway.providers.base`).

Controls, and why each is there
-------------------------------
``num_retries=0``
    Set by :func:`~model_gateway.providers.litellm_translation.to_litellm_params`
    and never overridden here. It is mandatory, not defensive: it suppresses both
    the LiteLLM-level retry loop and the OpenAI-SDK retry loop, which is what
    leaves the gateway core as the only retry owner (PRD section 11.3). The
    injected client is *not* a substitute: passing
    ``client=AsyncOpenAI(max_retries=0)`` without ``num_retries=0`` silently
    resets the SDK's ``max_retries`` to 2, so retries return unless the
    parameter travels beside the client.

The injected client
    A per-call ``client=`` kwarg, built by the transport's factory
    (:func:`model_gateway.security.transport.build_async_client`). The connection
    is the binding's to make, so it is the binding's to make safe (PRD section
    14.1): the client pins the connection to the address the gateway validated
    and disables automatic redirect following, which LiteLLM otherwise forces
    on (``follow_redirects=True`` is hardcoded in its OpenAI path). A per-call
    ``transport=`` kwarg cannot do this — LiteLLM treats it as a body
    parameter — so client injection is the only mechanism that reaches the
    wire. One client per attempt, closed when the attempt ends: the binding
    owns no client lifecycle beyond the call it is making.

The attempt timeout
    ``context.timeout`` is passed through as the call's ``timeout`` — the
    budget the gateway set for this attempt (PRD section 9.6). ``None`` means
    the gateway set no budget, and the binding passes the absence through
    rather than inventing one of its own.

Error mapping
    Every exception the call raises is mapped by
    :func:`~model_gateway.providers.litellm_translation.map_litellm_error` and
    re-raised, so a caller never sees a provider-shaped error and no upstream
    message, URL or credential reaches the public payload (PRD sections 5.4
    and 9). A refusal raised by the transport itself — a host that resolves to
    a blocked address at connect time — arrives as a connection failure whose
    cause chain carries the
    :class:`~model_gateway.security.url_policy.UrlPolicyError`, and is mapped as
    the connection failure it was reported as. Cancellation is the deliberate
    boundary of that rule:
    :exc:`asyncio.CancelledError` and the other ``BaseException``s propagate
    unmapped, because PRD section 15 requires the gateway to propagate
    cancellation rather than report it as a gateway failure.

Residual limitation
-------------------
``num_retries=0`` does not buy exactly one upstream request. Two retry layers
inside the binding cannot be disabled by any knob, both verified against the
pinned ``litellm-core``:

* the message-reformat loop, which rewrites the request and retries specific
  400 responses, and
* the generic-path retry on connection errors, which rebuilds a fresh
  *default* client — bypassing this provider's injected transport, so a retry
  on that path also loses the pinning and redirect controls of PRD section
  14.1 — and tries again.

Exactly one upstream request per gateway attempt is therefore guaranteed only
when the upstream triggers neither loop. The consequence for attempt
accounting is stated rather than hidden: the gateway counts attempts at its own
seam (:meth:`model_gateway.inference.InferenceService._attempt`) and cannot
see requests the binding makes internally, so ``attempt_count`` may understate
the number of requests that actually reached the upstream — and a request the
binding retried internally may have been partially processed, or billed,
while the gateway records a single attempt.

The ``.env`` guard
-----------------
Importing ``litellm`` calls ``dotenv.load_dotenv()`` at import time while
``LITELLM_MODE`` is ``DEV``, its default: any ``.env`` in the working directory
is loaded into ``os.environ``, and
:meth:`model_gateway.config.settings.Settings.from_env` would then read
configuration the operator never set for this process. This module pins the
mode before importing the binding — the earliest point any importer of the
binding can reach — so the process environment stays the only configuration
source (PRD section 6). ``setdefault`` respects an explicit operator choice;
the value otherwise only selects LiteLLM's proxy CLI modes, which this gateway
never runs. The test session sets the same guard (:file:`tests/conftest.py`).
"""

import os
from collections.abc import Callable
from typing import Any

# The .env guard above: this must run before the first ``import litellm`` in
# the process, and every import below pulls the binding in, so each one
# carries its E402 exemption.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import litellm  # noqa: E402

from model_gateway.config.routes import Route  # noqa: E402
from model_gateway.domain import InvocationRequest, InvocationResponse  # noqa: E402
from model_gateway.providers.base import AttemptContext  # noqa: E402
from model_gateway.providers.litellm_translation import (  # noqa: E402
    from_litellm_response,
    map_litellm_error,
    to_litellm_params,
)
from model_gateway.security.transport import build_async_client  # noqa: E402

__all__ = ["LiteLLMProvider"]


class LiteLLMProvider:
    """The production text provider: translation plus one execution (PRD section 9).

    One attempt in, one canonical answer out. The provider performs exactly one
    upstream request for the route it is handed and applies the attempt's own
    budget to it; it does not retry, does not choose between routes and knows
    nothing about fallbacks — those are the gateway core's (PRD sections 9.2
    and 11.3). Every control the call carries — ``num_retries=0``, the injected
    safe client, the attempt timeout, the error mapping — is documented in the
    module docstring, where the reasons for each live.

    The client factory is injectable so a caller can stand in for the
    transport's pinned client; the default is
    :func:`model_gateway.security.transport.build_async_client`, which is what
    makes the connection satisfy PRD section 14.1.
    """

    def __init__(self, *, client_factory: Callable[..., Any] = build_async_client) -> None:
        """Build the provider over the transport's client factory.

        ``client_factory`` is called once per attempt with the attempt's budget
        as ``budget=`` and returns the client that attempt's request is
        injected with. It is the single seam between this binding and the
        transport obligations of PRD section 14.1: a deployment that needs a
        different client constructs its own factory; tests inject a local stub
        and never touch a network.
        """
        self._client_factory = client_factory

    async def complete(
        self,
        route: Route,
        request: InvocationRequest,
        context: AttemptContext,
    ) -> InvocationResponse:
        """Perform the one upstream call for this attempt (PRD section 9.6).

        The call's parameters are the translation of ``route`` and ``request``
        (:func:`~model_gateway.providers.litellm_translation.to_litellm_params`)
        plus this attempt's own controls: the timeout ``context`` carries and
        the client the transport's factory builds for this attempt's budget,
        closed when the attempt ends. Translation errors are already gateway
        errors and propagate unchanged; the mapping covers the call itself,
        whose exceptions are provider-shaped. The answer is normalised through
        :func:`~model_gateway.providers.litellm_translation.from_litellm_response`
        and returned as the canonical response.
        """
        params = to_litellm_params(route, request)
        params["timeout"] = context.timeout
        try:
            async with self._client_factory(budget=context.timeout) as client:
                response = await litellm.acompletion(**params, client=client)
        except Exception as exc:
            raise map_litellm_error(exc) from exc
        return from_litellm_response(route, request, response)
