"""Every acceptance criterion, and the test that proves it (PRD section 21).

Section 21 states nineteen criteria and says the product is accepted when they hold.
Nothing in this tree said *where* each one is proved, so "the criteria are covered"
was a claim a reader had to take on trust — and a claim no test could contradict when
a covering test was renamed, weakened or deleted.

This module is that mapping, and it is checked rather than asserted. Each criterion
names the tests that prove it, and a test here fails if any of them stops existing.
That is the difference between a traceability document and a traceability *check*:
prose about coverage decays silently, and a node ID that no longer resolves does not.

**What the check can and cannot do.** It proves that every criterion points at tests
that exist; it cannot prove that a named test proves the criterion, which is a
reading a person does once and this module then holds to. What it prevents is the
decay: deleting or renaming a covering test fails here, so a criterion cannot lose
its evidence without someone deciding to.

**Where the mapping came from.** Each entry was established by reading the candidate
tests and confirming they assert the criterion rather than something adjacent. Two
criteria had no test at all when this map was written — text inference through four
providers, and vendor neutrality — and the tests that now prove them were written for
it, which is the useful thing a map does: it shows where the evidence is missing
rather than where it is plentiful.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _defines(nodeid: str) -> bool:
    """Whether the test module defines the function or method ``nodeid`` names, exactly once.

    Once, rather than at least once, because a duplicate definition shadows an
    earlier one and pytest collects the later: a name defined twice is ambiguous
    evidence rather than evidence, and this check must not report a criterion as
    covered on the strength of a definition the runner would not reach. The same
    reasoning refuses a malformed structural path — a class method named where the
    class is not, or a nested definition — because a name this module cannot place is
    a name it cannot vouch for.
    """
    relative, _, name = nodeid.partition("::")
    path = ROOT / relative
    if not name or not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            matches += 1
        if isinstance(node, ast.ClassDef):
            matches += sum(
                1
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                and f"{node.name}::{child.name}" == name
            )
    return matches == 1


CRITERIA: dict[int, tuple[str, tuple[str, ...]]] = {
    1: (
        "The gateway starts independently without external application services.",
        (
            "tests/test_grpc_server.py::test_the_listener_binds_the_configured_port",
            "tests/test_http_app.py::test_the_default_path_serves_an_application_with_the_real_binding",
        ),
    ),
    2: (
        "Text inference works through at least four configured LiteLLM providers.",
        (
            "tests/test_conformance.py::test_text_inference_is_served_through_each_configured_provider",
            "tests/test_conformance.py::test_the_configured_providers_are_all_reachable_in_one_walk",
            "tests/test_routes.py::test_routes_json_appends_routes_in_array_order",
        ),
    ),
    3: (
        "Text providers can be switched without recompilation.",
        (
            "tests/test_conformance.py::test_text_inference_is_served_through_each_configured_provider",
            "tests/test_routes.py::test_parse_routes_reads_the_process_environment_by_default",
            "tests/test_registry.py::test_from_env_builds_the_registry_from_the_process_environment",
        ),
    ),
    4: (
        "Image generation uses independently configured credentials and endpoints.",
        (
            "tests/test_routes.py::test_the_prd_complete_environment_example_is_accepted",
            "tests/test_images.py::test_image_default_is_not_served_by_another_operations_routes",
            "tests/test_litellm_image.py::TestTheOpenAISdkPath::test_the_sdk_is_given_a_retry_pinned_sdk_client_over_the_pinned_httpx_client",
        ),
    ),
    5: (
        "Embeddings use independently configured credentials and endpoints.",
        (
            "tests/test_routes.py::test_the_prd_complete_environment_example_is_accepted",
            "tests/test_embeddings.py::test_embedding_default_is_not_served_by_another_operations_routes",
            "tests/test_embeddings_provider.py::TestTheCall::test_the_parameters_are_the_route_the_texts_and_the_attempts_controls",
        ),
    ),
    6: (
        "HTTP and gRPC share canonical application logic.",
        (
            "tests/test_grpc_invoke.py::test_the_provider_is_handed_the_canonical_request",
            "tests/test_http_chat.py::test_every_message_role_reaches_the_provider_as_canonical_content",
            "tests/test_http_dto.py::TestChatRequestParameters::test_builds_the_canonical_invocation",
            "tests/test_http_app.py::test_an_injected_service_is_used_verbatim",
        ),
    ),
    7: (
        "Tool calls round-trip without losing arguments or identifiers.",
        (
            "tests/test_grpc_invoke.py::test_an_answer_carrying_a_tool_call_is_translated",
            "tests/test_grpc_invoke.py::test_a_tool_calls_integer_arguments_survive_as_integers",
            "tests/test_grpc_mapping.py::test_tool_call_arguments_as_an_object_round_trip",
            "tests/test_http_dto.py::TestChatRequestMessages::test_parses_an_assistant_tool_call_with_json_arguments",
        ),
    ),
    8: (
        "Multimodal inputs preserve supported content.",
        (
            "tests/test_http_dto.py::TestChatRequestMessages::test_parses_content_parts",
            "tests/test_http_dto.py::TestChatRequestMessages::test_parses_a_base64_data_uri_into_inline_image_content",
            "tests/test_http_responses.py::test_an_input_image_part_becomes_an_image_part",
            "tests/test_grpc_invoke.py::test_an_answer_carrying_an_image_is_translated",
        ),
    ),
    9: (
        "Unsupported capabilities produce explicit errors.",
        (
            "tests/test_errors.py::test_a_capability_error_reports_the_unsupported_feature",
            "tests/test_errors.py::test_a_capability_error_is_unprocessable_content",
            "tests/test_http_dto.py::TestChatRequestMessages::test_refuses_a_remote_image_url_the_gateway_cannot_screen",
            "tests/test_http_responses.py::test_an_input_image_requires_the_vision_capability",
        ),
    ),
    10: (
        "Streaming works for compatible providers.",
        (
            "tests/test_inference_streaming.py::TestTheHappyPath::test_a_simple_text_stream_is_served_in_order",
            "tests/test_grpc_streaming.py::TestTheAnswerStream::test_a_streamed_answer_arrives_as_fragments_numbered_from_zero",
            "tests/test_conformance.py::test_a_streamed_chat_completion_is_served_by_the_real_binding",
        ),
    ),
    11: (
        "Timeouts, retries, and fallbacks behave deterministically.",
        (
            "tests/test_reliability.py::TestBackoff::test_the_injected_source_makes_the_schedule_reproducible",
            "tests/test_reliability.py::TestNonRetryableFailures::test_a_deterministic_failure_is_neither_retried_nor_fallen_back_from",
            "tests/test_inference.py::test_a_route_is_retried_before_a_fallback_is_tried",
            "tests/test_inference.py::test_every_attempt_draws_its_budget_from_the_one_deadline",
        ),
    ),
    12: (
        "No provider credentials are exposed through public APIs.",
        (
            "tests/test_conformance.py::test_no_answer_carries_the_provider_credential_or_the_upstream_model",
            "tests/test_http_app.py::test_no_public_response_carries_upstream_configuration",
            "tests/test_registry.py::test_the_unknown_model_error_leaks_no_route_configuration",
        ),
    ),
    13: (
        "Unknown public model IDs cannot reach unconfigured upstream endpoints.",
        (
            "tests/test_conformance.py::test_an_unknown_model_reaches_no_upstream",
            "tests/test_registry.py::test_get_raises_for_an_unknown_model_id",
            "tests/test_grpc_invoke.py::test_an_unknown_model_is_refused_and_the_provider_is_never_called",
        ),
    ),
    14: (
        "Usage accounting distinguishes actual and estimated token counts.",
        (
            "tests/test_domain.py::test_usage_records_provenance_for_each_count",
            "tests/test_accounting.py::test_a_derived_count_is_marked_estimated_in_the_record",
            "tests/test_grpc_invoke.py::test_an_estimated_count_is_named_as_one_on_the_wire",
        ),
    ),
    15: (
        "The service can run without a database.",
        (
            "tests/test_http_app.py::test_the_default_path_serves_an_application_with_the_real_binding",
            "tests/test_accounting.py::test_cost_tracking_without_usage_records_usage_and_no_cost",
            "tests/test_accounting.py::test_a_durable_storage_mode_without_persistence_is_refused",
        ),
    ),
    16: (
        "Health and readiness checks function independently.",
        (
            "tests/test_health.py::test_healthz_returns_ok",
            "tests/test_health.py::test_readyz_returns_ready",
            "tests/test_health.py::test_readyz_returns_503_when_a_check_fails",
            "tests/test_grpc_health.py::test_the_overall_service_reports_serving_once_the_listener_starts",
        ),
    ),
    17: (
        "Configuration errors are detected at startup.",
        (
            "tests/test_litellm_client_injection.py::TestTheStartupRefusal::test_a_text_route_with_an_unverified_dialect_fails_the_start",
            "tests/test_images_http.py::TestTheStartupCheck::test_an_image_route_advertising_editing_refuses_the_start",
            "tests/test_routes.py::test_a_partially_configured_block_is_rejected",
            "tests/test_allowed_hosts.py::test_a_route_outside_the_allowlist_fails_the_start_naming_its_variable",
            "tests/test_observability.py::test_tracing_with_nowhere_to_send_is_refused_at_startup",
        ),
    ),
    18: (
        "Automated tests cover all public endpoints.",
        (
            "tests/test_http_models.py::test_list_returns_an_openai_list_envelope",
            "tests/test_http_chat.py::test_happy_path_returns_the_documented_response_shape",
            "tests/test_http_completions.py::test_happy_path_returns_the_documented_response_shape",
            "tests/test_http_responses.py::test_happy_path_returns_the_documented_response_shape",
            "tests/test_embeddings_http.py::TestTheHappyPath::test_a_list_of_texts_is_embedded",
            "tests/test_images_http.py::TestTheHappyPath::test_a_generation_request_is_served",
            "tests/test_grpc_invoke.py::test_a_successful_invocation_round_trips",
            "tests/test_grpc_embed.py::test_embed_answers_with_the_vectors_the_provider_produced",
            "tests/test_grpc_list_models.py::test_list_models_lists_every_configured_public_model",
            "tests/test_metrics.py::test_the_metrics_endpoint_serves_the_exposition_format",
        ),
    ),
    19: (
        (
            "The implementation contains no vendor-specific business logic outside the "
            "LiteLLM binding."
        ),
        (
            "tests/test_vendor_neutrality.py::test_no_core_module_imports_a_vendor_library",
            "tests/test_vendor_neutrality.py::test_no_core_module_names_a_vendor_or_a_dialect",
            "tests/test_vendor_neutrality.py::test_only_the_composition_root_reaches_into_a_vendor_binding",
        ),
    ),
}
"""Criterion number to its text and the tests that prove it.

The text is section 21's, verbatim, so a criterion this map does not know about is a
criterion section 21 gained and this module has not caught up with.
"""


def test_every_criterion_in_section_21_is_mapped():
    """Nineteen criteria, numbered from one, with none skipped and none invented."""
    assert sorted(CRITERIA) == list(range(1, len(CRITERIA) + 1))


def test_every_criterion_names_at_least_one_test():
    unmapped = sorted(number for number, (_, evidence) in CRITERIA.items() if not evidence)

    assert unmapped == []


def test_every_named_test_exists():
    """The check that makes this a mapping rather than a document.

    A named test that was renamed, deleted or moved fails here, and so does a name
    that is defined twice, because a duplicate shadows the definition pytest would not
    reach. The names are read from the test modules themselves rather than from a
    collected session, so this module is meaningful when it is run on its own and does
    not depend on the rest of the suite having been collected first. What it proves is
    that the test *is defined*, exactly once, at the structural path named; it does
    not reproduce pytest's collection rules and does not prove that the runner would
    collect and execute it.
    """
    missing = sorted(
        f"criterion {number} names {nodeid}, which no test module defines"
        for number, (_, evidence) in CRITERIA.items()
        for nodeid in evidence
        if not _defines(nodeid)
    )

    assert missing == []


@pytest.mark.parametrize("number", sorted(CRITERIA), ids=lambda number: f"criterion-{number}")
def test_a_criterion_is_a_sentence_and_its_evidence_is_not_empty(number):
    """The shape of an entry, so a malformed one is a failure and not a silent gap."""
    text, evidence = CRITERIA[number]

    assert text.endswith(".")
    assert len(text) > 20
    assert all(nodeid.startswith("tests/") and "::" in nodeid for nodeid in evidence)
