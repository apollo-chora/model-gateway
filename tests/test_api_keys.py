"""Inbound API-key parsing, verification and identity (PRD section 7).

Every test is hermetic: settings are built from an explicit environment
mapping, no real environment is read. The secrets used here are meaningless
placeholders, and several tests additionally assert that the implementation
never puts key material into messages or reprs.
"""

import dataclasses
import hashlib
import hmac
import re

import pytest

import model_gateway.security.api_keys as api_keys_module
from model_gateway.config import ConfigurationError, Settings
from model_gateway.errors import ErrorReason, UnauthenticatedError
from model_gateway.security.api_keys import ApiKeySet
from model_gateway.security.identity import Identity

HEX_LABEL = re.compile(r"[0-9a-f]{12}")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def make_settings(**env: str) -> Settings:
    return Settings.from_env(dict(env))


def make_keys(**env: str) -> ApiKeySet:
    return ApiKeySet.from_settings(make_settings(**env))


class TestKeyFormat:
    def test_plain_entry_authenticates_with_no_tenant(self):
        keys = make_keys(GATEWAY_API_KEYS="solo-key")

        identity = keys.authenticate("solo-key")

        assert identity.authenticated is True
        assert identity.tenant is None

    def test_tenant_bound_entry_authenticates_with_its_tenant(self):
        keys = make_keys(GATEWAY_API_KEYS="prod-key:tenant-a")

        identity = keys.authenticate("prod-key")

        assert identity.authenticated is True
        assert identity.tenant == "tenant-a"

    def test_whitespace_around_entries_and_halves_is_trimmed(self):
        keys = make_keys(GATEWAY_API_KEYS="  key-a  :  tenant-a  ,   key-b   ")

        assert keys.authenticate("key-a").tenant == "tenant-a"
        assert keys.authenticate("key-b").tenant is None

    def test_blank_entries_are_dropped(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a, ,   ,,key-b")

        assert keys.authenticate("key-a").tenant is None
        assert keys.authenticate("key-b").tenant is None

    def test_mixed_entries_keep_their_own_bindings(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a:tenant-a,key-b,key-c:tenant-c")

        assert keys.authenticate("key-a").tenant == "tenant-a"
        assert keys.authenticate("key-b").tenant is None
        assert keys.authenticate("key-c").tenant == "tenant-c"


class TestStartupRejection:
    @pytest.mark.parametrize("entry", [":tenant", " : tenant", ":,key-a"])
    def test_empty_key_part_is_rejected(self, entry: str):
        with pytest.raises(ConfigurationError):
            make_keys(GATEWAY_API_KEYS=entry)

    @pytest.mark.parametrize("entry", ["key-a:", "key-a : ", "key-a:,"])
    def test_empty_tenant_part_is_rejected(self, entry: str):
        with pytest.raises(ConfigurationError):
            make_keys(GATEWAY_API_KEYS=entry)

    def test_more_than_one_separator_is_rejected(self):
        with pytest.raises(ConfigurationError):
            make_keys(GATEWAY_API_KEYS="key-a:tenant-a:extra")

    @pytest.mark.parametrize(
        "entries",
        ["key-a,key-a", "key-a:tenant-a,key-a:tenant-b", "key-a,key-a:tenant-a"],
    )
    def test_duplicate_key_is_rejected(self, entries: str):
        with pytest.raises(ConfigurationError):
            make_keys(GATEWAY_API_KEYS=entries)

    def test_auth_enabled_without_keys_is_rejected(self):
        with pytest.raises(ConfigurationError):
            make_keys(GATEWAY_AUTH_ENABLED="true")

    @pytest.mark.parametrize(
        "entry", ["key-a:", "key-a: ,key-b", "key-a:tenant-a:extra", "key-a,key-a"]
    )
    def test_rejection_messages_never_echo_the_key(self, entry: str):
        with pytest.raises(ConfigurationError) as excinfo:
            make_keys(GATEWAY_API_KEYS=entry)

        assert "key-a" not in str(excinfo.value)


class TestVerification:
    @pytest.mark.parametrize("presented", [None, ""])
    def test_missing_credential_is_rejected(self, presented: str | None):
        keys = make_keys(GATEWAY_API_KEYS="key-a")

        with pytest.raises(UnauthenticatedError) as excinfo:
            keys.authenticate(presented)

        assert excinfo.value.reason is ErrorReason.AUTHENTICATION_REQUIRED

    def test_wrong_key_of_the_same_length_is_rejected(self):
        keys = make_keys(GATEWAY_API_KEYS="aaaa-bbbb-cccc")

        with pytest.raises(UnauthenticatedError) as excinfo:
            keys.authenticate("aaaa-bbbb-xxxy")

        assert excinfo.value.reason is ErrorReason.INVALID_CREDENTIALS

    def test_wrong_key_of_different_length_is_rejected(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a")

        with pytest.raises(UnauthenticatedError) as excinfo:
            keys.authenticate("much-longer-wrong-key")

        assert excinfo.value.reason is ErrorReason.INVALID_CREDENTIALS

    def test_comparison_runs_against_every_key_when_one_matches(self, monkeypatch):
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b,key-c")
        calls = _count_comparisons(monkeypatch)

        keys.authenticate("key-b")

        assert len(calls) == 3

    def test_comparison_runs_against_every_key_when_none_matches(self, monkeypatch):
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b,key-c")
        calls = _count_comparisons(monkeypatch)

        with pytest.raises(UnauthenticatedError):
            keys.authenticate("key-z")

        assert len(calls) == 3

    def test_every_comparison_is_between_equal_length_digests(self, monkeypatch):
        """The compared operands are fixed-length digests, not the keys themselves."""
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b")
        calls = _count_comparisons(monkeypatch)

        with pytest.raises(UnauthenticatedError):
            keys.authenticate("a-much-longer-presented-key-value")

        assert len(calls) == 2
        assert all(len(left) == 64 and len(right) == 64 for left, right in calls)

    def test_authentication_verifies_even_when_disabled(self):
        keys = make_keys(GATEWAY_AUTH_ENABLED="false", GATEWAY_API_KEYS="key-a")

        identity = keys.authenticate("key-a")

        assert identity.authenticated is True

    @pytest.mark.parametrize("presented", [None, "", "definitely-not-the-key"])
    def test_error_messages_never_echo_the_presented_key(self, presented):
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b")

        with pytest.raises(UnauthenticatedError) as excinfo:
            keys.authenticate(presented)

        if presented:
            assert presented not in excinfo.value.message
            assert presented not in str(excinfo.value)
            assert presented not in excinfo.value.to_payload()["message"]


def _count_comparisons(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Spy on ``hmac.compare_digest`` so tests can count its calls and see operands."""
    calls: list[tuple[str, str]] = []
    real_compare = hmac.compare_digest

    def counting_compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(api_keys_module.hmac, "compare_digest", counting_compare)
    return calls


class TestKeyLabel:
    def test_key_label_is_a_short_hex_label_and_not_the_key(self):
        keys = make_keys(GATEWAY_API_KEYS="a-distinctive-key-value")

        key_label = keys.authenticate("a-distinctive-key-value").key_label

        assert key_label is not None
        assert HEX_LABEL.fullmatch(key_label)
        assert "a-distinctive-key-value" not in key_label

    def test_key_label_is_stable_per_key_and_distinguishes_keys(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b")

        first = keys.authenticate("key-a").key_label
        second = keys.authenticate("key-a").key_label
        other = keys.authenticate("key-b").key_label

        assert first == second
        assert first != other

    def test_key_label_is_the_truncated_digest(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a")

        key_label = keys.authenticate("key-a").key_label

        assert key_label == hashlib.sha256(b"key-a").hexdigest()[:12]


class TestCredentialId:
    def test_credential_id_is_the_full_length_digest(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a")

        credential_id = keys.authenticate("key-a").credential_id

        assert credential_id == hashlib.sha256(b"key-a").hexdigest()
        assert credential_id is not None
        assert HEX_DIGEST.fullmatch(credential_id)

    def test_credential_id_is_stable_and_distinguishes_keys(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a,key-b")

        first = keys.authenticate("key-a").credential_id
        second = keys.authenticate("key-a").credential_id
        other = keys.authenticate("key-b").credential_id

        assert first == second
        assert first != other

    def test_key_label_is_a_prefix_of_credential_id(self):
        keys = make_keys(GATEWAY_API_KEYS="key-a")

        identity = keys.authenticate("key-a")

        assert identity.credential_id is not None
        assert identity.key_label == identity.credential_id[:12]

    def test_credential_id_does_not_expose_the_key(self):
        keys = make_keys(GATEWAY_API_KEYS="a-distinctive-key-value")

        credential_id = keys.authenticate("a-distinctive-key-value").credential_id

        assert credential_id is not None
        assert "a-distinctive-key-value" not in credential_id


class TestEnabled:
    @pytest.mark.parametrize("value", ["true", "false"])
    def test_enabled_reflects_the_setting(self, value: str):
        keys = make_keys(GATEWAY_AUTH_ENABLED=value, GATEWAY_API_KEYS="key-a")

        assert keys.enabled is (value == "true")


class TestAnonymous:
    def test_anonymous_identity_has_no_tenant_credential_id_or_label(self):
        keys = make_keys(GATEWAY_AUTH_ENABLED="false", GATEWAY_API_KEYS="key-a")

        assert keys.anonymous() == Identity(
            authenticated=False, tenant=None, credential_id=None, key_label=None
        )

    def test_anonymous_identity_is_the_same_when_authentication_is_enabled(self):
        keys = make_keys(GATEWAY_AUTH_ENABLED="true", GATEWAY_API_KEYS="key-a")

        assert keys.anonymous() == keys.anonymous()


class TestSecretHygiene:
    def test_repr_of_the_store_does_not_disclose_keys(self):
        keys = make_keys(GATEWAY_API_KEYS="super-secret-a,super-secret-b:tenant-a")

        rendered = repr(keys)

        assert "super-secret-a" not in rendered
        assert "super-secret-b" not in rendered

    def test_the_store_keeps_a_digest_and_not_the_plaintext(self):
        keys = make_keys(GATEWAY_API_KEYS="super-secret-a")

        binding = keys._bindings[0]

        assert not hasattr(binding, "key")
        assert binding.digest == hashlib.sha256(b"super-secret-a").hexdigest()

    def test_identity_is_frozen(self):
        identity = Identity(
            authenticated=True, tenant="tenant-a", credential_id="a" * 64, key_label="abc123"
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            identity.authenticated = False  # type: ignore[misc]
