"""The one seam through which the binding reaches LiteLLM (PRD section 6).

Importing ``litellm`` calls ``dotenv.load_dotenv()`` at import time while
``LITELLM_MODE`` is ``DEV``, its default: any ``.env`` in the working directory
is loaded into ``os.environ``, and
:meth:`model_gateway.config.settings.Settings.from_env` would then read
configuration the operator never set for this process. An environment variable
that is not part of the gateway's public configuration contract must not be
able to disable that invariant, so this module *forces* the mode to
``PRODUCTION`` — an assignment, not a default: ``LITELLM_MODE`` is
gateway-owned implementation state, not operator configuration. Its only other
effect is on LiteLLM's proxy CLI, which this gateway never runs.

Every binding module imports the library through here rather than importing it
directly, so every production import path crosses this seam before any
``import litellm`` and the process environment stays the only configuration
source. The test session's own copy of the guard lives in
:file:`tests/conftest.py`, because the binding tests import ``litellm`` and its
fixture types directly and pytest imports that file before any test module.
"""

import os

os.environ["LITELLM_MODE"] = "PRODUCTION"

import litellm  # noqa: E402
import openai  # noqa: E402
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler  # noqa: E402
from litellm.types.utils import (  # noqa: E402
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.types.utils import Message as LiteLLMMessage  # noqa: E402
from litellm.types.utils import Usage as LiteLLMUsage  # noqa: E402

__all__ = [
    "AsyncHTTPHandler",
    "ChatCompletionMessageToolCall",
    "Choices",
    "Delta",
    "LiteLLMMessage",
    "LiteLLMUsage",
    "ModelResponse",
    "ModelResponseStream",
    "StreamingChoices",
    "litellm",
    "openai",
]
