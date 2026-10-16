"""The published gRPC contract's identity, and that the Go stubs agree.

The gateway publishes one contract and generates two implementations of it:
the Python stubs the server imports, and the Go stubs callers import. Nothing
else ties those two together, so a rename on one side compiles and passes on
that side while every caller breaks on the wire.

These assertions pin the identity that must not change silently:

* the protobuf package and the service's full name, which are what the gRPC
  method path is built from;
* the RPC names, so a renamed method is caught here rather than by a caller;
* a serialize/parse round trip, so the descriptor still describes a usable
  message rather than only a well-named one;
* the Go stubs' own ``FullMethodName`` constants, read from the generated
  source, so the two generated artefacts cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from model_gateway.grpc import gateway_pb2

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_GRPC_STUBS = REPO_ROOT / "gen" / "go" / "gateway" / "v1" / "gateway_grpc.pb.go"

#: The service's full name, and the RPCs it serves. The plan keeps the
#: ``gateway.v1`` package rather than reusing ``chora.services.model_gateway.v1``,
#: which is already taken by the proto generated in chora-contracts for the
#: retiring Go service.
PACKAGE = "gateway.v1"
SERVICE = f"{PACKAGE}.ModelGatewayService"
RPCS = ("Invoke", "InvokeStream", "Embed", "GroundedSearch", "ListModels")


def test_the_descriptor_publishes_the_expected_package_and_service() -> None:
    assert gateway_pb2.DESCRIPTOR.package == PACKAGE
    service = gateway_pb2.DESCRIPTOR.services_by_name["ModelGatewayService"]
    assert service.full_name == SERVICE


def test_every_rpc_the_gateway_serves_is_present() -> None:
    service = gateway_pb2.DESCRIPTOR.services_by_name["ModelGatewayService"]
    assert tuple(m.name for m in service.methods) == RPCS


def test_a_message_still_serializes_and_parses() -> None:
    """The descriptor must describe a usable message, not just a named one."""
    sent = gateway_pb2.InvokeRequest(invocation_id="inv-1", tenant_id="tenant-1")
    parsed = gateway_pb2.InvokeRequest.FromString(sent.SerializeToString())
    assert parsed.invocation_id == "inv-1"
    assert parsed.tenant_id == "tenant-1"


@pytest.mark.skipif(not GO_GRPC_STUBS.exists(), reason="Go stubs are generated, not vendored")
def test_the_go_stubs_publish_the_same_method_paths() -> None:
    """The Go stubs must carry the same full method names as the descriptor.

    Read from the generated source rather than compiled, so this runs without
    a Go toolchain and fails on drift at generation time rather than at a
    caller's first RPC.
    """
    source = GO_GRPC_STUBS.read_text(encoding="utf-8")
    constants = dict(re.findall(r"(\w+)_FullMethodName\s+=\s+\"([^\"]+)\"", source))

    assert set(constants) == {f"ModelGatewayService_{rpc}" for rpc in RPCS}
    for rpc in RPCS:
        assert constants[f"ModelGatewayService_{rpc}"] == f"/{SERVICE}/{rpc}"
