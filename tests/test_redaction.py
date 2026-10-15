"""Log redaction: what the gateway must not write down.

Prompts, responses, credentials and private tool arguments
are not logged by default. These tests prove the rule is *enforced* rather than
merely observed — today nothing logs those things, and the point of the processor
is that the next change cannot start to.

The last test drives the configured chain rather than the processor alone, because
a processor that is correct and not installed is a processor that redacts nothing.
"""

import json
import logging

import structlog

from model_gateway.app_logging import configure_logging
from model_gateway.redaction import REDACTED, SENSITIVE_KEYS, redact_event, redact_value

EVENT = "an invocation finished"


def test_a_sensitive_key_is_replaced():
    event = redact_event(None, "info", {"event": EVENT, "prompt": "tell me a secret"})

    assert event["prompt"] == REDACTED
    assert event["event"] == EVENT


def test_a_sensitive_key_is_matched_whatever_its_case():
    event = redact_event(None, "info", {"API_KEY": "sk-live-1", "Authorization": "Bearer x"})

    assert event["API_KEY"] == REDACTED
    assert event["Authorization"] == REDACTED


def test_a_nested_mapping_is_redacted_at_any_depth():
    """A payload logged whole must be redacted as thoroughly as one logged field by field."""
    event = redact_event(
        None,
        "info",
        {"event": EVENT, "request": {"model": "text-default", "prompt": "tell me a secret"}},
    )

    assert event["request"]["model"] == "text-default"
    assert event["request"]["prompt"] == REDACTED


def test_a_sensitive_key_takes_its_whole_value_with_it():
    """A conversation is caller content whether it is logged as one field or many."""
    event = redact_event(
        None,
        "info",
        {"event": EVENT, "messages": [{"role": "user", "content": "hello"}]},
    )

    assert event["messages"] == REDACTED


def test_a_sequence_keeps_its_type_and_is_redacted_inside():
    """A log line whose types change with its content is one no query can be written against."""
    event = redact_event(
        None,
        "info",
        {
            "event": EVENT,
            "items": [{"prompt": "one"}, {"prompt": "two"}],
            "ids": ("a", "b"),
        },
    )

    assert event["items"] == [{"prompt": REDACTED}, {"prompt": REDACTED}]
    assert event["ids"] == ("a", "b")


def test_a_credential_embedded_in_an_unrelated_key_is_redacted_by_shape():
    """A key cannot catch a credential in text that is not itself named for one."""
    event = redact_event(None, "info", {"event": EVENT, "note": "sent Bearer sk-live-abcdef"})

    assert "sk-live-abcdef" not in event["note"]
    assert REDACTED in event["note"]


def test_the_gateways_own_vocabulary_is_not_redacted():
    """What an operator reads logs for must survive, or the redaction has gone too far."""
    event = redact_event(
        None,
        "info",
        {
            "event": EVENT,
            "model": "text-default",
            "upstream_model": "upstream-text",
            "reason": "provider_unavailable",
            "attempts": 3,
            "key_label": "5f3a46a7dc6a",
        },
    )

    assert event["model"] == "text-default"
    assert event["upstream_model"] == "upstream-text"
    assert event["reason"] == "provider_unavailable"
    assert event["attempts"] == 3
    assert event["key_label"] == "5f3a46a7dc6a"


def test_the_keys_the_rule_names_are_all_in_the_set():
    """The four sensitive categories, each with a key that names it."""
    named = (
        "prompt",
        "messages",
        "response",
        "content",
        "api_key",
        "authorization",
        "arguments",
    )

    for key in named:
        assert key in SENSITIVE_KEYS


def test_redact_value_leaves_a_plain_value_alone():
    assert redact_value(3) == 3
    assert redact_value(None) is None
    assert redact_value(b"bytes stay bytes") == b"bytes stay bytes"


def test_the_configured_chain_redacts_what_it_writes(capsys):
    """The processor installed is the processor that runs: a chain is the only proof."""
    configure_logging(logging.INFO)

    structlog.get_logger("test").info("an invocation finished", prompt="tell me a secret")

    rendered = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert rendered["event"] == "an invocation finished"
    assert rendered["prompt"] == REDACTED
    assert "tell me a secret" not in json.dumps(rendered)
