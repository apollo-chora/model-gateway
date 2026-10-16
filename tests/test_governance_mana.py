"""Tests for mana metering against a stub client.

Tested against a stub :class:`ManaClient`: what is under test is the policy the
meter owns — a shortfall denies the quote with the two balances, an unpriced
action is un-metered rather than a failure, and a meter that raised fails OPEN
rather than denying service — and that the meter hands the client the learner,
the tenant, the action and the idempotency key it was given. No gRPC connection
is involved, which is the point of the seam.
"""

from __future__ import annotations

import pytest

from model_gateway.governance.mana import (
    ManaClient,
    ManaDebit,
    ManaDebitResult,
    ManaMeter,
    ManaQuote,
    ManaResult,
    UnknownAction,
)

_GCID = "123e4567-e89b-42d3-a456-426614174000"
_TENANT_ID = "123e4567-e89b-42d3-a456-426614174001"
_ACTION_CODE = "companion_chat_turn_standard"
_IDEM_KEY = "0192a4b6-7f3e-7c1a-9d2e-5f6a7b8c9d0e"


class StubManaClient:
    """A :class:`ManaClient` that records its calls and replays a scripted outcome."""

    def __init__(
        self,
        *,
        quote_result: ManaResult | None = None,
        debit_result: ManaResult | None = None,
        quote_error: Exception | None = None,
        debit_error: Exception | None = None,
    ) -> None:
        self.quote_result = quote_result or ManaResult(success=True)
        self.debit_result = debit_result or ManaResult(success=True)
        self.quote_error = quote_error
        self.debit_error = debit_error
        self.quote_calls: list[tuple[str, str, str]] = []
        self.debit_calls: list[tuple[str, str, str, str]] = []

    async def quote(self, gcid: str, tenant_id: str, action_code: str) -> ManaResult:
        self.quote_calls.append((gcid, tenant_id, action_code))
        if self.quote_error is not None:
            raise self.quote_error
        return self.quote_result

    async def debit(self, gcid: str, tenant_id: str, action_code: str, idem_key: str) -> ManaResult:
        self.debit_calls.append((gcid, tenant_id, action_code, idem_key))
        if self.debit_error is not None:
            raise self.debit_error
        return self.debit_result


def _meter(**kwargs: object) -> tuple[ManaMeter, StubManaClient]:
    client = StubManaClient(**kwargs)  # type: ignore[arg-type]
    return ManaMeter(client), client


class TestQuote:
    async def test_an_affordable_action_is_allowed(self) -> None:
        meter, _ = _meter(quote_result=ManaResult(success=True, balance_after_units=100))

        assert await meter.quote(_GCID, _TENANT_ID, _ACTION_CODE) == ManaQuote(
            affordable=True, available=100
        )

    async def test_a_shortfall_denies_with_both_balances(self) -> None:
        meter, _ = _meter(
            quote_result=ManaResult(success=False, required_units=50, current_balance_units=20)
        )

        assert await meter.quote(_GCID, _TENANT_ID, _ACTION_CODE) == ManaQuote(
            affordable=False, required=50, available=20
        )

    async def test_an_unpriced_action_is_un_metered(self) -> None:
        meter, _ = _meter(quote_error=UnknownAction(_ACTION_CODE))

        assert await meter.quote(_GCID, _TENANT_ID, _ACTION_CODE) == ManaQuote(
            affordable=True, unknown_action=True
        )

    async def test_a_meter_error_fails_open(self) -> None:
        meter, _ = _meter(quote_error=RuntimeError("identity unreachable"))

        assert await meter.quote(_GCID, _TENANT_ID, _ACTION_CODE) == ManaQuote(
            affordable=True, error="identity unreachable"
        )

    async def test_the_client_is_given_the_learner_the_tenant_and_the_action(self) -> None:
        meter, client = _meter()

        await meter.quote(_GCID, _TENANT_ID, _ACTION_CODE)

        assert client.quote_calls == [(_GCID, _TENANT_ID, _ACTION_CODE)]
        assert client.debit_calls == []


class TestDebit:
    async def test_a_successful_debit_is_ok(self) -> None:
        meter, _ = _meter(debit_result=ManaResult(success=True, balance_after_units=80))

        assert await meter.debit(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY) == ManaDebit(
            result=ManaDebitResult.OK, available=80
        )

    async def test_a_shortfall_is_reported_not_raised(self) -> None:
        meter, _ = _meter(
            debit_result=ManaResult(success=False, required_units=50, current_balance_units=20)
        )

        assert await meter.debit(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY) == ManaDebit(
            result=ManaDebitResult.SHORTFALL, required=50, available=20
        )

    async def test_an_unpriced_action_is_un_metered(self) -> None:
        meter, _ = _meter(debit_error=UnknownAction(_ACTION_CODE))

        assert await meter.debit(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY) == ManaDebit(
            result=ManaDebitResult.UN_METERED
        )

    async def test_a_meter_error_is_an_error_result(self) -> None:
        meter, _ = _meter(debit_error=RuntimeError("identity unreachable"))

        assert await meter.debit(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY) == ManaDebit(
            result=ManaDebitResult.ERROR, error="identity unreachable"
        )

    async def test_the_client_is_given_the_idempotency_key_and_the_tenant(self) -> None:
        meter, client = _meter()

        await meter.debit(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY)

        assert client.debit_calls == [(_GCID, _TENANT_ID, _ACTION_CODE, _IDEM_KEY)]
        assert client.quote_calls == []


class TestManaResult:
    @pytest.mark.parametrize(
        "field",
        ["required_units", "current_balance_units", "balance_after_units"],
    )
    def test_refuses_a_negative_unit_count(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            ManaResult(success=True, **{field: -1})  # type: ignore[arg-type]

    def test_a_field_the_service_did_not_set_is_zero(self) -> None:
        assert ManaResult(success=True) == ManaResult(
            success=True, required_units=0, current_balance_units=0, balance_after_units=0
        )


def test_the_stub_client_satisfies_the_protocol() -> None:
    assert isinstance(StubManaClient(), ManaClient)
