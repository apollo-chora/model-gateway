"""Shared pytest setup for the gateway test suite.

Importing LiteLLM loads the project's ``.env`` into the process environment —
its DEV-mode default calls ``dotenv.load_dotenv()`` at import time. That
side effect would leak the deployment's configured endpoints and credentials
into ``os.environ`` for the whole test session, polluting tests that read the
process environment (for example the route parser's default-environment
test).

The production import path is guarded at :mod:`model_gateway.providers._litellm`,
the seam every binding module imports the library through: it forces
``LITELLM_MODE`` to ``PRODUCTION`` before the first ``import litellm``, so no
ambient value — not even an explicit ``LITELLM_MODE=DEV`` — can re-enable the
load. This file must stay the test session's own copy of that guard because
the binding tests import ``litellm`` and its fixture types directly, to build
upstream shapes from the library's own types, and pytest imports this file
before any test module: without it, the session's first direct import would
run before any seam. The mode is gateway-owned implementation state, not
operator configuration, so it is forced here exactly as the seam forces it.
"""

import os

os.environ["LITELLM_MODE"] = "PRODUCTION"

# The Postgres integration fixtures live in their own module so the unit-test
# session stays free of the testcontainers import; loading it as a plugin makes
# its fixtures discoverable without importing them into the test modules (which
# would shadow the fixture names against the test parameters).
pytest_plugins = ["conftest_governance"]
