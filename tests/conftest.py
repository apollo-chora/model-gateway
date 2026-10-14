"""Shared pytest setup for the gateway test suite.

Importing LiteLLM loads the project's ``.env`` into the process environment —
its DEV-mode default calls ``dotenv.load_dotenv()`` at import time. That
side effect would leak the deployment's configured endpoints and credentials
into ``os.environ`` for the whole test session, polluting tests that read the
process environment (for example the route parser's default-environment
test). Setting ``LITELLM_MODE`` to a non-DEV value before the first LiteLLM
import keeps the session's environment clean. The mode otherwise only affects
the proxy CLI, which these tests never run.
"""

import os

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
