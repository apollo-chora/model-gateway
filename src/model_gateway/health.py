"""Liveness and readiness endpoints for Kubernetes probes (PRD section 4)."""

import inspect
from collections.abc import Awaitable, Callable

from blacksheep.server.responses import json, text

Check = Callable[[], bool | Awaitable[bool]]

_checks: dict[str, Check] = {}


def register_check(name: str, check: Check) -> None:
    """Register a readiness check; called by capability wiring at startup."""
    _checks[name] = check


async def healthz():
    return text("ok")


async def readyz():
    results: dict[str, str] = {}
    for name, check in _checks.items():
        try:
            outcome = check()
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception:
            results[name] = "error"
            continue
        results[name] = "ok" if outcome else "fail"

    ready = all(result == "ok" for result in results.values())
    return json(
        {"status": "ready" if ready else "not_ready", "checks": results},
        status=200 if ready else 503,
    )
