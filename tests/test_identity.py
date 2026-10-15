"""Unit tests for the trusted-identity invariants.

:class:`Identity` is the transport-neutral principal both the HTTP and gRPC
layers construct, so the two states its fields can hold are enforced here at
the type itself: an authenticated identity must carry the credential
identifier and key label of the matched credential, and an anonymous identity
must carry no tenant, no credential identifier and no key label. Every
rejected combination is refused by ``__post_init__`` with a message naming the
contradiction.
"""

from typing import Any

import pytest

from model_gateway.security.identity import Identity

CREDENTIAL_ID = "a" * 64
KEY_LABEL = "abc123abc123"


def authenticated(
    *,
    tenant: str | None = None,
    credential_id: str | None = CREDENTIAL_ID,
    key_label: str | None = KEY_LABEL,
) -> Identity:
    return Identity(
        authenticated=True, tenant=tenant, credential_id=credential_id, key_label=key_label
    )


class TestValidForms:
    def test_an_authenticated_identity_may_carry_a_tenant(self):
        identity = authenticated(tenant="tenant-a")

        assert identity.authenticated is True
        assert identity.tenant == "tenant-a"
        assert identity.credential_id == CREDENTIAL_ID
        assert identity.key_label == KEY_LABEL

    def test_an_authenticated_identity_may_carry_no_tenant(self):
        identity = authenticated(tenant=None)

        assert identity.authenticated is True
        assert identity.tenant is None

    def test_the_anonymous_identity_carries_nothing(self):
        identity = Identity(authenticated=False, tenant=None, credential_id=None, key_label=None)

        assert identity.authenticated is False
        assert identity.tenant is None
        assert identity.credential_id is None
        assert identity.key_label is None


class TestAuthenticatedRequiresItsCredential:
    def test_an_authenticated_identity_without_a_credential_id_is_refused(self):
        with pytest.raises(ValueError, match="authenticated identity must carry"):
            authenticated(credential_id=None)

    def test_an_authenticated_identity_without_a_key_label_is_refused(self):
        with pytest.raises(ValueError, match="authenticated identity must carry"):
            authenticated(key_label=None)

    def test_an_authenticated_identity_with_neither_is_refused(self):
        with pytest.raises(ValueError, match="authenticated identity must carry"):
            authenticated(credential_id=None, key_label=None)

    def test_the_refusal_names_the_contradiction(self):
        with pytest.raises(ValueError, match="authenticated caller with no credential"):
            authenticated(credential_id=None)


class TestAnonymousCarriesNothing:
    @pytest.mark.parametrize("field", ["tenant", "credential_id", "key_label"])
    def test_an_anonymous_identity_with_a_credential_property_is_refused(self, field: str):
        values: dict[str, Any] = {
            "authenticated": False,
            "tenant": None,
            "credential_id": None,
            "key_label": None,
        }
        values[field] = "value"

        with pytest.raises(ValueError, match="anonymous identity carries no tenant"):
            Identity(**values)

    def test_the_refusal_names_the_contradiction(self):
        with pytest.raises(ValueError, match="unauthenticated caller with a credential"):
            Identity(authenticated=False, tenant="tenant-a", credential_id=None, key_label=None)
