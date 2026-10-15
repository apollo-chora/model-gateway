"""The optional live-provider smoke test.

The end-to-end requirements close with a request that no other test in this tree
can satisfy: at least one optional live-provider smoke test should demonstrate
switching providers through environment configuration alone. Every other suite
stands in for the upstream, which is what makes it deterministic and what keeps it
from needing an account; this module is the one that does not, so it is the one that
is **skipped by default**.

**Why it exists as a skip rather than not at all.** The gap between "the gateway
works against a stand-in" and "the gateway works against a provider" is exactly the
gap a smoke test covers, and a suite that cannot express it leaves the claim
untested *and* unstated. Written this way, the capability exists, the command to use
it is documented, and its absence from a normal run is a deliberate skip with a
reason rather than a silence.

**Running it.** Configure one provider in the environment the gateway itself reads —
the same variables a deployment sets, because the point is that nothing else is
needed — and run:

    GATEWAY_LIVE_TEXT_ENDPOINT=https://... \\
    GATEWAY_LIVE_TEXT_MODEL_ID=... \\
    GATEWAY_LIVE_TEXT_API_KEY=... \\
    GATEWAY_LIVE_TEXT_FORMAT=openai \\
    uv run pytest -m live

A second provider may be configured the same way with a `_2` suffix, and the second
test then demonstrates the switch. Both are read from the process environment rather
than from a fixture, because a fixture would be a second configuration path and the
claim under test is that there is only one.

**What it does not do.** It does not run in CI, and it is not evidence that the
gateway is correct — the deterministic suite is that. It is evidence that the
configuration path reaches a real provider, which is a different question and the
one an operator asks first.
"""

import os

import pytest
from blacksheep import JSONContent
from blacksheep.testing import TestClient

from model_gateway.app import create_app

_PROMPT = {
    "model": "text-default",
    "messages": [{"role": "user", "content": "Reply with the word ok."}],
}

#: The variables a provider is configured through. Read from the process
#: environment, because that is the configuration path the test is about.
_FIRST = {
    "TEXT_LLM_ENDPOINT": "GATEWAY_LIVE_TEXT_ENDPOINT",
    "TEXT_LLM_MODEL_ID": "GATEWAY_LIVE_TEXT_MODEL_ID",
    "TEXT_LLM_API_KEY": "GATEWAY_LIVE_TEXT_API_KEY",
    "TEXT_LLM_FORMAT": "GATEWAY_LIVE_TEXT_FORMAT",
}

_SECOND = {variable: f"{live}_2" for variable, live in _FIRST.items()}

_MISSING = sorted(live for live in _FIRST.values() if not os.environ.get(live))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=(
            "a live provider is not fully configured: set "
            f"{', '.join(_MISSING)} to run this (see this module's docstring)"
        ),
    ),
]
"""Skipped unless the **whole** first provider is configured.

The group matters rather than the endpoint alone: a run with an endpoint and no
credential is a misconfiguration, and reporting it as a test that ran and failed
would be worse than reporting it as the incomplete setup it is. The second
provider is checked by the test that needs it, because requiring both here would
hide the one-provider check behind the two-provider one.
"""


def _configure(monkeypatch: pytest.MonkeyPatch, source: dict[str, str], *, what: str) -> None:
    """Point the gateway's own variables at one live provider."""
    missing = sorted(live for live in source.values() if not os.environ.get(live))
    if missing:
        pytest.skip(f"{', '.join(missing)} is not set, so {what} is not exercised")
    for variable, live in source.items():
        monkeypatch.setenv(variable, os.environ[live])


async def _one_answer() -> tuple[int, str]:
    """Start the application the environment describes and ask it one question.

    Nothing is injected: the registry, the binding, the service and the application
    are the ones ``create_app`` builds from configuration, which is the path a
    deployment takes and the path a stand-in cannot exercise.
    """
    app = create_app()
    await app.start()
    try:
        response = await TestClient(app).post("/v1/chat/completions", content=JSONContent(_PROMPT))
        return response.status, await response.text()
    finally:
        await app.stop()


async def test_a_configured_provider_answers_a_real_request(monkeypatch):
    """One provider, configured by environment alone, reached through the public API."""
    _configure(monkeypatch, _FIRST, what="the configured provider")

    status, body = await _one_answer()

    assert status == 200, body
    assert body


async def test_a_second_provider_configured_the_same_way_answers(monkeypatch):
    """Switching providers is an environment change, and this is that claim.

    The second provider is reached by changing the variables the gateway reads and
    building the application again — no code, no fixture, no injection — which is
    what "switched without recompilation" means when it is done for real rather than
    asserted about a mapping.
    """
    _configure(monkeypatch, _SECOND, what="the switch between two providers")

    status, body = await _one_answer()

    assert status == 200, body
    assert body
