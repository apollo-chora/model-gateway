"""Per-GCID mana metering against chora-identity's ManaService.

Every LLM call the gateway serves draws down one learner's mana wallet, priced
by action code in the identity catalogue. This module is the gateway's half of
that contract: a dry-run quote that checks affordability before any compute is
spent, and a debit after a call has actually produced output.

**The meter is a policy layer over a client seam.** :class:`ManaClient` is the
port to chora-identity's ``ManaService`` — ``quote`` runs
``DeductMana(dry_run=True, units=0)`` and ``debit`` runs
``DeductMana(dry_run=False, units=0)``, both idempotent on the caller's key —
and the ``grpc.aio`` wiring lands with the generated stubs. What the meter owns
is the decision policy the seam cannot: an action code that is not in the
pricing catalogue is un-metered rather than a failure, and a meter that cannot
be reached fails OPEN, because a metering outage must not deny service.

**Fail-open is deliberate and logged.** A quote or debit that raises is
returned as allowed-with-error and logged, never raised: the gateway's
resilience rule is that a dependent control's outage is not a reason to refuse a
call the caller is entitled to. The error rides on the result so the caller can
count it, and the log carries the failure's type and the action it was pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import structlog

from model_gateway.domain.serialization import as_int

__all__ = [
    "ManaClient",
    "ManaDebit",
    "ManaDebitResult",
    "ManaMeter",
    "ManaQuote",
    "ManaResult",
    "UnknownAction",
]

logger = structlog.get_logger(__name__)


class UnknownAction(Exception):
    """The action code is not in the identity pricing catalogue.

    Raised by a :class:`ManaClient` when the service refuses the action code as
    unpriced. It is not a failure: an action the catalogue does not price is
    one the gateway does not meter, and the meter reports it as un-metered
    rather than as an error.
    """


@dataclass(frozen=True)
class ManaResult:
    """One ``DeductMana`` outcome, as the client reads it.

    ``success`` is the service's own verdict; on a shortfall it is ``False``
    with the required and current balances set. The units are the service's
    integers, never floats, and a field the service did not set is zero.
    """

    success: bool
    required_units: int = 0
    current_balance_units: int = 0
    balance_after_units: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_units",
            as_int("mana result 'required_units'", self.required_units, minimum=0),
        )
        object.__setattr__(
            self,
            "current_balance_units",
            as_int(
                "mana result 'current_balance_units'", self.current_balance_units, minimum=0
            ),
        )
        object.__setattr__(
            self,
            "balance_after_units",
            as_int("mana result 'balance_after_units'", self.balance_after_units, minimum=0),
        )


@runtime_checkable
class ManaClient(Protocol):
    """The slice of chora-identity's ``ManaService`` the meter calls.

    ``quote`` runs the dry-run affordability check and ``debit`` charges; both
    return the service's raw outcome and raise :class:`UnknownAction` when the
    action code is unpriced, or any other error when the meter could not be
    read. The real implementation speaks ``grpc.aio``; a test injects a stub,
    which is why the meter never names a channel of its own.
    """

    async def quote(self, gcid: str, action_code: str) -> ManaResult: ...

    async def debit(self, gcid: str, action_code: str, idem_key: str) -> ManaResult: ...


@dataclass(frozen=True)
class ManaQuote:
    """The dry-run affordability verdict for one action code.

    ``affordable`` is the gate. ``required`` and ``available`` are the
    shortfall's two sides, set only when the wallet could not cover the
    action. ``unknown_action`` marks an action the catalogue does not price,
    which the gateway serves un-metered; ``error`` marks a meter that could not
    be read, which the gateway also serves, because metering must not deny
    service.
    """

    affordable: bool
    required: int = 0
    available: int = 0
    unknown_action: bool = False
    error: str = ""


class ManaDebitResult(StrEnum):
    """The outcome of one real debit.

    ``OK`` charged the wallet; ``SHORTFALL`` could not, and carries the two
    balances; ``UN_METERED`` names an action the catalogue does not price, so
    nothing was charged and nothing was owed; ``ERROR`` names a meter that
    could not be reached, so the charge is unrecorded rather than failed.
    """

    OK = "ok"
    SHORTFALL = "shortfall"
    UN_METERED = "un_metered"
    ERROR = "error"


@dataclass(frozen=True)
class ManaDebit:
    """The outcome of one real debit.

    ``result`` is the whole answer; ``required`` and ``available`` are set on a
    shortfall, and ``error`` on a meter that could not be read.
    """

    result: ManaDebitResult
    required: int = 0
    available: int = 0
    error: str = ""


class ManaMeter:
    """Mana metering: a dry-run quote and a post-success debit.

    The meter holds the policy and the client holds the transport, so a
    deployment swaps the identity's address without touching the rules here and
    a test injects a stub without a connection.
    """

    def __init__(self, client: ManaClient) -> None:
        self._client = client

    async def quote(self, gcid: str, action_code: str) -> ManaQuote:
        """Whether the learner can afford one action, without charging it.

        A shortfall denies with the required and available balances, which is
        the caller's upsell context. An unpriced action code is un-metered, not
        a failure. A meter that raised is allowed through with the error on the
        result and a warning in the log — the fail-open half of the contract,
        because a metering outage must not deny a call the caller is entitled
        to.
        """
        try:
            result = await self._client.quote(gcid, action_code)
        except UnknownAction:
            return ManaQuote(affordable=True, unknown_action=True)
        except Exception as error:
            logger.warning(
                "mana quote unavailable; failing open",
                error_type=type(error).__name__,
                error=str(error),
                gcid=gcid,
                action_code=action_code,
            )
            return ManaQuote(affordable=True, error=str(error))
        if not result.success:
            return ManaQuote(
                affordable=False,
                required=result.required_units,
                available=result.current_balance_units,
            )
        return ManaQuote(affordable=True, available=result.balance_after_units)

    async def debit(self, gcid: str, action_code: str, idem_key: str) -> ManaDebit:
        """Charge one action against the learner's wallet, idempotently.

        ``idem_key`` is the caller's own key, so a retried debit collapses onto
        the ledger row the first attempt wrote. A shortfall is reported, not
        raised: the call has already been served, so the charge is a
        bookkeeping outcome the caller accounts rather than a reason to un-serve
        it. An unpriced action code is un-metered and a meter that raised is an
        error result, both logged rather than raised, for the same fail-open
        reason the quote documents.
        """
        try:
            result = await self._client.debit(gcid, action_code, idem_key)
        except UnknownAction:
            return ManaDebit(result=ManaDebitResult.UN_METERED)
        except Exception as error:
            logger.warning(
                "mana debit unavailable; failing open",
                error_type=type(error).__name__,
                error=str(error),
                gcid=gcid,
                action_code=action_code,
            )
            return ManaDebit(result=ManaDebitResult.ERROR, error=str(error))
        if not result.success:
            return ManaDebit(
                result=ManaDebitResult.SHORTFALL,
                required=result.required_units,
                available=result.current_balance_units,
            )
        return ManaDebit(result=ManaDebitResult.OK, available=result.balance_after_units)
