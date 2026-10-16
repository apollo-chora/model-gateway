# Code generation for model-gateway.
#
# The .proto files under proto/ are the source of truth. Everything else is
# generated; never hand-edit a generated file.
#
# `make` is not assumed to be present everywhere, so each target is also a
# plain command you can paste. The recipes are the ones documented in
# docs/ARCHITECTURE.md.

PY ?= uv run python

.PHONY: gen gen-go gen-python

## gen: regenerate every published stub
gen: gen-python gen-go

## gen-python: regenerate the Python stubs the server imports
gen-python:
	mkdir -p .gen
	$(PY) -m grpc_tools.protoc -I proto --python_out=.gen --grpc_python_out=.gen proto/gateway/v1/gateway.proto
	mv .gen/gateway/v1/gateway_pb2.py src/model_gateway/grpc/gateway_pb2.py
	mv .gen/gateway/v1/gateway_pb2_grpc.py src/model_gateway/grpc/gateway_pb2_grpc.py
	rm -rf .gen
	mkdir -p .gen
	$(PY) -m grpc_tools.protoc -I proto --python_out=.gen --grpc_python_out=.gen proto/identity/v1/mana_service.proto
	mv .gen/identity/v1/mana_service_pb2.py src/model_gateway/governance/identity/v1/mana_service_pb2.py
	mv .gen/identity/v1/mana_service_pb2_grpc.py src/model_gateway/governance/identity/v1/mana_service_pb2_grpc.py
	rm -rf .gen
	# grpcio-tools emits the package-qualified import, which does not resolve
	# inside the destination packages. See docs/ARCHITECTURE.md.
	sed -i 's/^from gateway\.v1 import gateway_pb2/from . import gateway_pb2/' src/model_gateway/grpc/gateway_pb2_grpc.py
	sed -i 's/^from identity\.v1 import mana_service_pb2/from . import mana_service_pb2/' src/model_gateway/governance/identity/v1/mana_service_pb2_grpc.py

## gen-go: regenerate the Go stubs Go callers import
#
# Requires buf, protoc-gen-go and protoc-gen-go-grpc on PATH.
gen-go:
	buf generate --path proto/gateway/v1/gateway.proto
