// Generated module for the model-gateway contract's Go stubs.
//
// Module path matches the `M` mapping in buf.gen.yaml, which is what gives
// the generated code its import path without a `go_package` option in the
// .proto source. Files are produced by `buf generate` and committed.
//
// Go callers add this module to their go.mod and a `replace` pointing at
// their local checkout of the Python repository:
//
//	require github.com/apollo-chora/model-gateway v0.0.0
//	replace github.com/apollo-chora/model-gateway => ../model-gateway
module github.com/apollo-chora/model-gateway

go 1.26.1

require (
	google.golang.org/grpc v1.83.0
	google.golang.org/protobuf v1.36.12
)

require (
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/text v0.37.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260526163538-3dc84a4a5aaa // indirect
)
