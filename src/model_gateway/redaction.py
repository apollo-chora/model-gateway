"""Redaction for what the gateway logs.

Prompts, responses, credentials and private tool arguments are not logged by
default. This module is the *enforcement* of that rule rather
than a convention the code happens to follow: today nothing logs those things, and
nothing stopped the next change from doing so. A processor in the log chain means
a value reaching a log call under one of these names is replaced before it is
rendered, whatever the call site believed it was passing.

Two layers, because one leaves an obvious hole.

**By key.** Any mapping key naming one of those things has its value replaced, at
any depth, so a payload logged as a nested object is redacted as thoroughly as one
logged field by field.

**By shape.** Any string carrying a credential in an assigned or bearer form is
redacted wherever it appears, so a credential cannot reach a log line through a key
this module does not know about. The shapes are the ones
:func:`~model_gateway.errors.redact_credential_shapes` recognises, and its limits
apply: a bare unprefixed token, a session cookie and a provider-specific
credential format all pass through. This is a backstop against accident, not
evidence that a string is safe to log.

**What is deliberately not redacted.** The gateway's own vocabulary — a public
model ID, a route's upstream model ID, an error reason, an attempt count, a
credential's derived label — is what an operator reads logs for, and redacting it
would leave a log line that says nothing. The rule is about the caller's content
and the operator's secrets, not about the gateway's own facts.

**Nested values keep their shape.** A mapping stays a mapping and a sequence stays
a sequence, with the redaction applied inside, because a log line whose types
change with its content is one an operator cannot write a query against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from model_gateway.errors import redact_credential_shapes

__all__ = ["SENSITIVE_KEYS", "REDACTED", "redact_event", "redact_value"]

REDACTED = "[redacted]"
"""What replaces a value the gateway must not log.

A marker rather than an omission: a log line that dropped the key entirely would
leave an operator unable to tell "this invocation carried no prompt" from "this
invocation's prompt was withheld", and the second is the fact worth knowing.
"""

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        # Credentials and the material that carries them.
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "x-api-key",
        # Prompts and the conversation they belong to.
        "prompt",
        "prompts",
        "messages",
        "system_instruction",
        "input",
        # Answers.
        "response",
        "responses",
        "completion",
        "content",
        "text",
        "output",
        # Tool arguments, which are caller content by another name.
        "arguments",
        "arguments_json",
        "tool_arguments",
        "tool_calls",
    }
)
"""The key names whose values are never logged, matched case-insensitively.

An explicit set rather than a pattern over suffixes: a suffix rule would catch
``input_tokens`` — a count the gateway reports on purpose — along with the
``api_key`` it was meant for, and a redaction rule that fires on the wrong field is
one an operator learns to route around.
"""


def _is_sensitive(key: object) -> bool:
    return isinstance(key, str) and key.lower() in SENSITIVE_KEYS


def redact_value(value: Any) -> Any:
    """The value as it may be logged, with anything sensitive inside it replaced.

    Recurses through mappings and sequences so a payload logged whole is redacted
    as thoroughly as one logged field by field. A string is passed through the
    credential-shape scan, because that is the one thing a key cannot catch: a
    credential embedded in text that is not itself named for one.
    """
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_sensitive(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return redact_credential_shapes(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        redacted = [redact_value(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


def redact_event(logger: object, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """A structlog processor: the event as it may be written.

    The signature is structlog's, and the unused arguments are part of it rather
    than oversights: the processor chain calls every processor with the logger, the
    method name and the event, and a processor that dropped one would not be
    callable in the chain.
    """
    return redact_value(event_dict)
