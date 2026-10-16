# Architecture

The gateway is one process with two listeners — HTTP on `CHORA_HTTP_PORT` and
gRPC on `CHORA_GRPC_PORT`, started and stopped together — in front of whichever
upstreams the environment configured. An application talks to the gateway; which
provider answers is an environment change rather than a code change.

## Component boundaries

Three layers, with a one-way dependency rule: the core never names a provider,
and the binding is the only module that does.

**Northbound transports.** The public HTTP `/v1` surface and the versioned gRPC
service `gateway.v1.ModelGatewayService`. Each transport only translates: it
authenticates the caller, maps the wire payload onto the canonical domain, calls
the one inference service, and renders the canonical answer in its own wire
shape. Both transports reach identical policy.

**Gateway core.** Everything that is policy: configuration loading and
validation, the model registry, request validation, capability checking,
admission limits, the retry and fallback walk, idempotency, usage and cost
accounting, the public error contract, network security, and what may be logged.

**Inference binding.** The pinned LiteLLM library behind one interface that
owns upstream protocol translation — request encoding, network execution,
response decoding — and nothing else. The library's own retry and fallback are
disabled (`num_retries=0` on every call), so exactly one retry loop and one
deadline exist and both are the gateway's. Swapping the library changes the
binding, not the public contract, routing, policy or accounting.

## The gRPC contract

The service is defined in `proto/gateway/v1/gateway.proto`: package
`gateway.v1`, four RPCs — `Invoke`, `InvokeStream`, `Embed`, `ListModels` — plus
the standard gRPC health checking protocol on the same port. The package is
versioned, field numbers are stable, and a field is never reused or renumbered
after publication.

The contract is deliberately the canonical model rather than a provider dialect,
which draws one line every field falls on: a field the canonical model types is
typed protobuf (a model ID, a generation bound, a modality, a usage count), and
a field the canonical model holds as arbitrary JSON is canonical-domain JSON
bytes (request metadata, a tool's parameters, a JSON Schema, a tool call's
arguments). The bytes form exists because the only JSON-ish well-known protobuf
types represent every number as a double, which would change both the type and
the value of a JSON integer.

The generated Python stubs are checked in and excluded from linting, because
they are the compiler's output. Regenerating them is a command, not an edit:

```sh
mkdir -p .gen
uv run python -m grpc_tools.protoc -I proto --python_out=.gen --grpc_python_out=.gen proto/gateway/v1/gateway.proto
mv .gen/gateway/v1/gateway_pb2.py src/model_gateway/grpc/gateway_pb2.py
mv .gen/gateway/v1/gateway_pb2_grpc.py src/model_gateway/grpc/gateway_pb2_grpc.py
rm -rf .gen
```

The stub's import of the pb2 module must then be rewritten from
`from gateway.v1 import gateway_pb2` to the relative `from . import gateway_pb2`:
grpcio-tools emits the package-qualified form, which does not resolve inside
`model_gateway/grpc/`, and regeneration silently breaks the package without the
fixup.

The identity service's ManaService stubs are regenerated the same way, from
`proto/identity/v1/mana_service.proto` (source of truth: chora-contracts'
`services/identity/v1/mana_service.proto`):

```sh
mkdir -p .gen
uv run python -m grpc_tools.protoc -I proto --python_out=.gen --grpc_python_out=.gen proto/identity/v1/mana_service.proto
mv .gen/identity/v1/mana_service_pb2.py src/model_gateway/governance/identity/v1/mana_service_pb2.py
mv .gen/identity/v1/mana_service_pb2_grpc.py src/model_gateway/governance/identity/v1/mana_service_pb2_grpc.py
rm -rf .gen
```

The same relative-import fixup applies (`from identity.v1 import mana_service_pb2`
to `from . import mana_service_pb2`).

### Go stubs

Go callers need the same contract as Go code. `buf generate` produces it into
`gen/go/`, and the `go.mod` at the repository root makes that tree an
importable module:

```sh
buf generate --path proto/gateway/v1/gateway.proto
```

The `--path` is required: the module also holds the governance peer protos
(`identity/v1`, `observability/v1`), which have no Go import path and exist
for the Python side only.

The Go import path comes from the `M` mapping in `buf.gen.yaml`, not from a
`go_package` option in the `.proto`. Adding one would change the descriptor
the Python stubs are generated from, for no benefit to them.

A caller imports it as:

```go
import gatewayv1 "github.com/apollo-chora/model-gateway/gen/go/gateway/v1"
```

and adds to its own `go.mod`:

```
require github.com/apollo-chora/model-gateway v0.0.0
replace github.com/apollo-chora/model-gateway => ../model-gateway
```

The `replace` is what makes a local checkout work before the module is
tagged and published.

The gRPC health protocol is served without authentication — it is the
infrastructure protocol, and a prober has no credential to present. The HTTP
health endpoints follow the deployment's configured auth policy.

## Request lifecycle

One inference service decides what may happen on every request, whichever
transport it arrives on. In order:

1. **Authentication.** The credential is validated against the configured key
   list. Trusted tenant identity derives solely from the validated credential —
   never from the request body: a body `tenant_id` that disagrees with the
   authenticated tenant is refused with a permission error, and `actor_id` /
   `agent_id` are attribution labels that never become principals.
2. **Body bound.** A request stating an oversized `Content-Length` is refused
   without reading a byte of it; a chunked body that states none is measured;
   the same configured value bounds the gRPC received message.
3. **Operation resolution.** The operation is resolved from the transport the
   call arrived on — the HTTP endpoint or the gRPC method — and the registry
   resolves it onto that operation's configured routes. An operation with no
   configured route is a model-not-found error — even when an idempotency
   record exists for the key, and even when the caller is also over its rate.
   A model ID the caller may state is carried for the caller's own correlation
   and ignored for routing, so an unknown ID is refused nothing: it is simply
   not what selects the routes.
4. **Admission.** The per-client request-rate bucket and the per-tenant
   concurrency bound are checked. A refusal is raised here, before anything else
   the operation checks, and never from inside the retry loop.
5. **Capability check.** The request is validated against the primary route's
   advertised capabilities. A fallback route may preserve those capabilities but
   never expand them.
6. **Streaming eligibility.** For a streaming request, the bound provider's
   streaming support is confirmed before any provider is called.
7. **Idempotency.** The record is consulted after the operation's routes are
   resolved and before any provider is called.
8. **The walk.** The primary route is attempted; retryable failures retry the
   same route within its budget; fallback-eligible failures move to the next
   capability-eligible route.
9. **Binding execution.** The binding translates the canonical call, injects
   the pinned transport, and performs the upstream request.
10. **Rendering.** The canonical answer is rendered in the transport's wire
    shape — the documented OpenAI-compatible subset, refusing what it cannot
    represent rather than approximating it.
11. **Accounting.** The usage record is written at the service boundary —
    before the concurrency slot is released, and after the answer exists.
12. **Observability.** The invocation's counts are taken at the same service
    boundary, so an HTTP request and a gRPC call for one invocation are counted
    once and counted the same way.

## Retries and fallback

One invocation has one retry loop, one attempt count and one deadline, all the
gateway core's. The inference library is called with retries disabled, and the
gateway never relies on the library's own fallback parameters.

Every failure is classified into the gateway's own error taxonomy first, and
two separate questions are asked of it: *may another attempt on this route
help?* (same-route retry) and *may another route help?* (fallback). The answers
differ per category:

- **Retryable on the same route and fallback-eligible:** an unavailable
  provider, and a deadline that still has budget left.
- **Fallback-eligible but not retried on the same route:** an upstream rate
  limit — the limit belongs to the route that answered it, the taxonomy carries
  no retry-after to wait, and a blind retry would add load to the provider that
  just refused — and a structurally malformed upstream response, where the
  fault lies in that upstream's answer rather than in the request.
- **Neither:** the deterministic request and caller failures (invalid argument,
  authentication, permission, missing resource or model, capability gap, failed
  precondition), internal errors, and the gateway's own concurrency refusal,
  which every route shares.

The retry delay is exponential from `PROVIDER_RETRY_BACKOFF_MS` with a route-local
exponent — a fallback's first retry backs off one base exactly as the primary's
did — extended by jitter that only ever lengthens the wait, capped, and refused
rather than slept away when it would consume the invocation's remaining budget.
The wait is spent outside the concurrency slot, and only when a retry will
actually follow it.

One deadline is anchored when the invocation begins and covers the whole
invocation: route walking, retries, fallbacks and any wait for a concurrency
slot. Each attempt is handed the smaller of the configured provider timeout and
what the deadline then has left; an attempt whose budget is exhausted is never
started.

**Once an event has been delivered, no fallback may occur.** Before the first
delivered event the ordinary rules apply unchanged. After it, the gateway makes
no further attempt at all — not even a retry of the route that was streaming —
and the failure is surfaced as the stream's terminal `error` event; the partial
output the caller already received stands as the answer.

When no route could serve, the failure surfaced is the last attempt's, because
that is the most informative one. `attempt_count` counts gateway attempts, never
upstream requests: two internal library retry paths cannot be switched off, so
one gateway attempt can carry more than one wire request, and the gateway claims
no exactly-once upstream execution.

Embedding vectors are interchangeable only within one embedding space, and that
is a property of the configuration the gateway cannot verify: two routes can
declare the same dimensionality and produce incompatible vectors. The gateway
enforces each answer's shape and each route's capability; every route behind the
embedding model must serve the same embedding space.

## Idempotency

Every invocation may carry an idempotency key. A repeat of the same request is
answered from the gateway's own record of the first answer rather than by
asking the provider again.

- **The fingerprint covers the payload, not the key.** It is taken over a
  canonical JSON encoding of the model, conversation, instruction, generation
  config, tools, tool choice, modality, permitted fallbacks and tenant. The key
  itself and the gateway-minted invocation identifier are excluded, so a retry
  carrying a freshly minted identifier is still the same request.
- **The key is namespaced by the trusted principal** — the authenticated
  credential's identifier and tenant — so two tenants using the same key are two
  independent operations. With authentication disabled, keyed requests share one
  anonymous namespace, which is the honest consequence of removing caller
  isolation.
- **Reuse is a conflict, not a re-run.** A repeat carrying a different
  fingerprint is refused (`idempotency_conflict`, 409 Conflict / gRPC
  `ALREADY_EXISTS`) before any provider is called, leaving the stored record
  intact.
- **Only a completed answer is recorded.** A failure is not cached, so a retry
  after a transient failure may still reach the provider.
- **Concurrent identical requests are coalesced.** The first caller claims the
  key; every later caller with the same fingerprint waits on that claim's
  outcome instead of starting a second invocation. Cancellation belongs to the
  caller who owns it.
- **The store is in memory and therefore single-instance**, bounded, and lost
  when the process stops. Durable or distributed idempotency requires configured
  persistence, which the gateway does not presume.

Idempotency applies to the non-streaming invocation only: a replayed stream
would be a synthetic stream of an answer never produced token by token in this
call.

## Accounting

Every invocation that produces an answer leaves a record, built in the inference
service — the only layer that knows which route actually answered, so a
fallback answer is attributed to the route that served it rather than to the
plan's primary. The record names the invocation identifier and timestamp, the
serving route and its advertised capabilities, the public model, the upstream
model, the operation, the token counts with their provenance, the invocation's
elapsed latency (route walking, retries, fallbacks and slot waits included),
the attempt count, whether a fallback was used, and the estimated cost. A
record's status has three values: `completed`, `failed` (output produced, then a
failure) and `incomplete` (output produced, then no terminal event — a caller
that went away, a lost transport, or an upstream that never reported a finish
reason).

A token count reaches a record as the provider reported it or as the gateway
derived it, and the difference travels with the number: an unreported count is
never rendered as zero in a record, because that would turn an unknown quantity
into an asserted one. Cost comes only from `COST_PRICING_JSON`, per million
tokens keyed by public model ID, and is reported only when it is complete — a
category the pricing covers whose count the provider never reported leaves a
charge the gateway cannot see, so no number is reported at all rather than the
sum of the known categories.

Recording is not part of serving the request: a sink that fails is logged for
the operator rather than allowed to fail the invocation, and a coalesced
idempotency waiter is not recorded at all because it did not reach the
provider.

## Security invariants

**The environment is the only configuration path.** The gateway forces
`LITELLM_MODE` before the first import of the inference library so the library's
import-time `.env` load never runs; configuration cannot arrive from the working
directory.

**Outbound connections are screened at two layers.** Configuration time
validates the URL — `https` required, no embedded credentials, query or
fragment, no `.`/`..` segments — and the hostname's resolved addresses against
the blocked classes: loopback, private, link-local (including the cloud metadata
address), unique-local, multicast, reserved, unspecified and non-routable.
Screening alone is not SSRF prevention — the request is performed later, by a
transport that resolves the hostname again — so the binding injects a pinned
client into every upstream call: the connection terminates at the validated
address while the original hostname is preserved for TLS SNI and the `Host`
header, and redirect following is disabled, so a redirect surfaces as an error
rather than a silent hop. A dialect whose HTTP path has not been verified to
honour the injected client is refused at startup and again at the attempt rather
than run with the controls lost. A network-level egress restriction in front of
the deployment is a recommended second layer, not a substitute for either
control.

**The upstream allowlist narrows and never widens.** A host named by
`GATEWAY_ALLOWED_UPSTREAM_HOSTS` still has to pass the address screen; a route
outside the allowlist is a startup failure, not a per-request refusal.

**Provider credentials arrive as environment variables and are never accepted
from a request body.** They do not appear in a public response, in a log line
or in an error payload; logs carry a derived, non-secret key label. A route's
reserved authentication headers cannot be supplied as extra headers, and its
extra headers are immutable once validated.

**Tenant identity derives solely from the validated credential.** Caller-
supplied identity fields are attribution labels, never principals for quotas or
authorization.

**Errors are sanitised.** Public payloads carry stable, machine-readable reasons
and nothing internal: no configuration a refusal was made under, no credential
material, and no `Retry-After` header — a deliberate usability trade-off, since
a concurrency slot's availability cannot be stated precisely.

**Logs are redacted by construction, not by convention.** A processor redacts
by key at any depth and scans every string for a credential in an assigned or
bearer form; a redacted value is a marker rather than an omission, so an operator
can tell "this carried none" from "this was withheld". The gateway's own
vocabulary — model IDs, error reasons, attempt counts, key labels — is
deliberately not redacted, and the key set is explicit rather than a suffix
pattern. The scan recognises shapes and only shapes: a bare unprefixed token or
a session cookie passes through.

**Production must not serve unauthenticated inference.**
`CHORA_ENVIRONMENT=production` refuses to start with authentication disabled.

**A caller-supplied model ID cannot reach an unconfigured upstream endpoint.**
The registry maps only the three fixed public model IDs onto configured routes,
and the operation is resolved from the transport, so a stated model — known or
not — is never what selects the routes.

**The gateway terminates no TLS.** A deployment that needs it terminates it in
front of both ports, which is also where an operator's own access control
belongs.

## Observability

`GET /metrics` is authenticated like every other path and renders eleven
metrics in the Prometheus text exposition format, with the format's own media
type. The gateway carries its own small registry rather than a metrics client,
because the names are fixed and what must be exactly right is the wire text.

| Metric | Type | Meaning |
| --- | --- | --- |
| `gateway_requests_total` | counter | Invocations the gateway was asked to serve, one per invocation. |
| `gateway_request_duration_seconds` | histogram | Elapsed time of an invocation, from its start to its answer or failure. |
| `gateway_upstream_requests_total` | counter | Attempts the gateway made against an upstream provider. |
| `gateway_upstream_duration_seconds` | histogram | Elapsed time of one upstream attempt. |
| `gateway_tokens_input_total` | counter | Input tokens recorded, as reported or estimated. |
| `gateway_tokens_output_total` | counter | Output tokens recorded, as reported or estimated. |
| `gateway_retries_total` | counter | Attempts made after a retryable failure on the same route. |
| `gateway_fallbacks_total` | counter | Invocations served by a route other than the primary one. |
| `gateway_errors_total` | counter | Invocations that ended in a gateway error rather than an answer. |
| `gateway_active_requests` | gauge | Invocations admitted and not yet finished. |
| `gateway_rate_limit_rejections_total` | counter | Requests refused by a gateway limit rather than by an upstream. |

There are no labels: a label set is a claim about which dimensions the gateway
publishes, and therefore about the cardinality an operator's backend must hold.
Counts are taken at the service boundary, so both transports count one
invocation once and identically; a refusal from the gateway's own limits is
counted where it is raised, because that is the only place that can tell it from
an upstream's rate limit; and a count the provider never reported is not added
to a token total.

Tracing covers the inbound direction. A caller's `traceparent` continues into
the gateway's spans; one span is produced per invocation, named for its
operation, covering a streamed answer's whole life. The inbound context is
extracted from the HTTP headers or the gRPC metadata and detached on every
exit, including a refusal or an abort, so a context never leaks into whatever
the event loop runs next. The outbound leg is not traced: an upstream call does
not carry the invocation's context, and the transport seam is the pinned,
security-critical one — a stated gap rather than a silent shortfall.
