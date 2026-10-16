"""The real mana client: chora-identity's ``ManaService`` over ``grpc.aio``.

:class:`ManaMeter` in :mod:`model_gateway.governance.mana` is the policy layer
over the client seam; this module is the transport half of that seam — the
``ManaClient`` protocol's production implementation, speaking the generated
stubs of chora-identity's ``ManaService`` (checked in as
``proto/identity/v1/mana_service.proto``, source of truth in chora-contracts).

**The two calls are the service's one RPC, dry-run and real.** ``quote`` runs
``DeductMana`` with ``dry_run=True, units=0`` and ``debit`` with
``dry_run=False, units=0``, so the service resolves the action's price from
its own pricing catalogue and the gateway never prices an action itself. Both
pass ``units=0`` — the caller-priced form is the legacy ad-hoc path — and both
carry an idempotency key: the debit carries the caller's key so a retried
charge collapses onto the ledger row the first attempt wrote, and the quote
carries a fresh one, because the service requires a non-empty key even for a
dry run and a dry run writes no ledger row for the key to dedup. Both also
carry the tenant the action is priced under: the service selects the subsidy
slice a debit draws down by ``tenant_id``, so a learner holding allocations
from several tenants is charged the one the call's tenant names — a request
that dropped it would be priced and debited against the wrong wallet.

**An unpriced action code is the service's ``INVALID_ARGUMENT``.** chora-identity
maps an action its catalogue does not price onto that status rather than a
server fault, so the gateway can distinguish "unpriced, serve un-metered" from
"the meter could not be read, fail open" — the client raises
:class:`~model_gateway.governance.mana.UnknownAction` on exactly that status
and lets every other failure propagate, which the meter turns into its own
fail-open error result.

**The RPC carries a deadline.** The quote runs on the critical path of every
fail-closed invoke, so a hung identity service must fail the quote open after
a bounded wait rather than hang the call it was gating; the deadline is a
module constant because it is a transport concern, not a policy one.
"""

from __future__ import annotations

import uuid

import grpc

from model_gateway.governance.identity.v1 import mana_service_pb2, mana_service_pb2_grpc
from model_gateway.governance.mana import ManaResult, UnknownAction

__all__ = ["GrpcManaClient"]

_RPC_TIMEOUT_SECONDS = 5.0
"""The deadline on every ``DeductMana`` call.

The quote is a pre-flight gate on the fail-closed path, so the wait for the
identity service is bounded: past the deadline the call raises, the meter
fails the quote open, and the invoke proceeds — a metering outage must not
deny service, and it must not hang it either.
"""


class GrpcManaClient:
    """A :class:`~model_gateway.governance.mana.ManaClient` over one gRPC channel.

    The channel is built by the deployment, so the client owns no address and
    no credentials: a deployment names the identity service's address at
    wiring time, and a test injects a channel to a stub server it controls.
    """

    def __init__(self, channel: grpc.aio.Channel) -> None:
        self._channel = channel
        self._stub = mana_service_pb2_grpc.ManaServiceStub(channel)

    async def close(self) -> None:
        """Close the underlying channel with the process that built it."""
        await self._channel.close()

    async def quote(self, gcid: str, tenant_id: str, action_code: str) -> ManaResult:
        """The dry-run affordability verdict for one action, without charging it.

        A fresh idempotency key names the quote: a dry run resolves the
        price and checks affordability but writes no ledger row, so the key
        records no dedup state and the next quote is an independent check.
        ``tenant_id`` is carried so the wallet the service checks is the one
        the caller's tenant feeds.
        """
        return await self._deduct(
            gcid, tenant_id, action_code, dry_run=True, idem_key=uuid.uuid4().hex
        )

    async def debit(
        self, gcid: str, tenant_id: str, action_code: str, idem_key: str
    ) -> ManaResult:
        """Charge one action against the learner's wallet, idempotently.

        ``idem_key`` is the caller's own key, so a retried debit collapses
        onto the ledger row the first attempt wrote. ``tenant_id`` is carried
        so the subsidy slice the debit draws down is the one the caller's
        tenant's allocations feed.
        """
        return await self._deduct(gcid, tenant_id, action_code, dry_run=False, idem_key=idem_key)

    async def _deduct(
        self, gcid: str, tenant_id: str, action_code: str, *, dry_run: bool, idem_key: str
    ) -> ManaResult:
        """Run one ``DeductMana`` and read the service's own outcome.

        An action code the service does not price arrives as
        ``INVALID_ARGUMENT`` and is raised as :class:`UnknownAction` — the
        signal the meter reads as "un-metered", never as a failure. Any other
        gRPC failure propagates: the meter fails the call open and logs it.
        """
        request = mana_service_pb2.DeductManaRequest(
            gcid=gcid,
            action_code=action_code,
            units=0,
            idempotency_key=idem_key,
            tenant_id=tenant_id,
            dry_run=dry_run,
        )
        try:
            response = await self._stub.DeductMana(request, timeout=_RPC_TIMEOUT_SECONDS)
        except grpc.aio.AioRpcError as error:
            if error.code() is grpc.StatusCode.INVALID_ARGUMENT:
                raise UnknownAction(action_code) from error
            raise
        return ManaResult(
            success=response.success,
            required_units=response.required_units,
            current_balance_units=response.current_balance_units,
            balance_after_units=response.balance_after_units,
        )
