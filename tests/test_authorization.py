"""Unit tests for the transport-independent tenant reconciliation (PRD section 5.1).

:func:`reconcile_identity` is the policy both transports call, so it is tested
here on its own rather than through one wire format. The HTTP surface's use of
it is covered end to end in ``test_http_auth.py``.
"""

import pytest

from model_gateway.errors import PermissionDeniedError
from model_gateway.security.authorization import reconcile_identity
from model_gateway.security.identity import Identity

TENANT = "tenant-a"
CREDENTIAL_ID = "a" * 64
KEY_LABEL = "abc123abc123"


def authenticated(tenant: str | None) -> Identity:
    return Identity(
        authenticated=True,
        tenant=tenant,
        credential_id=CREDENTIAL_ID,
        key_label=KEY_LABEL,
    )


def anonymous() -> Identity:
    return Identity(authenticated=False, tenant=None, credential_id=None, key_label=None)


class TestAuthenticatedCaller:
    def test_a_matching_claim_is_accepted(self):
        reconcile_identity(authenticated(TENANT), TENANT)

    def test_an_absent_claim_is_accepted(self):
        reconcile_identity(authenticated(TENANT), None)

    def test_a_disagreeing_claim_is_refused(self):
        with pytest.raises(PermissionDeniedError) as error:
            reconcile_identity(authenticated(TENANT), "tenant-other")

        assert error.value.reason.value == "permission_denied"

    def test_a_claim_for_a_tenant_less_credential_is_refused(self):
        """An authenticated key with no tenant cannot be given one by a claim."""
        with pytest.raises(PermissionDeniedError):
            reconcile_identity(authenticated(None), TENANT)

    def test_a_tenant_less_credential_with_no_claim_is_accepted(self):
        reconcile_identity(authenticated(None), None)

    def test_the_refusal_names_the_label_and_not_the_credential_id(self):
        with pytest.raises(PermissionDeniedError) as error:
            reconcile_identity(authenticated(TENANT), "tenant-other")

        assert error.value.details["key_label"] == KEY_LABEL
        assert CREDENTIAL_ID not in str(error.value)


class TestUnauthenticatedCaller:
    def test_an_unauthenticated_claim_is_discarded(self):
        """A claim from an unauthenticated caller must never become identity."""
        reconcile_identity(anonymous(), TENANT)

    def test_a_missing_identity_is_discarded(self):
        reconcile_identity(None, TENANT)

    def test_a_missing_identity_with_no_claim_is_accepted(self):
        reconcile_identity(None, None)
