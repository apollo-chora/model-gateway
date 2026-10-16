# Configuration

The gateway is configured entirely by environment variables: the process
environment is the only configuration path. There is no configuration file the
gateway reads on its own, no database and no administrative API. Which upstreams
it calls, over which protocols, with which credentials, under which limits, is
all environment, so changing a provider is an environment change rather than a
build.

Every variable is read, parsed and validated once at startup
(`src/model_gateway/config/settings.py` for the server, authentication,
reliability, limits, usage and observability variables;
`src/model_gateway/config/routes.py` for the operation blocks). The gateway
refuses to start on a configuration it cannot serve — a malformed route, a
dialect the binding has not verified, an endpoint whose host resolves to an
address the URL policy forbids, a price the gateway cannot read — so a
misconfiguration is a failed start rather than a request that fails later or a
deployment that serves wrong behaviour for a month.

`.env.example` and `.env.production.example` are worked examples covering the
whole surface; this document is the reference. `docs/DEPLOYMENT.md` covers
running and operating the image.

## How values are parsed

- **Booleans** accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`,
  case-insensitively; anything else is refused.
- **Lists** are comma-separated; surrounding whitespace is trimmed and empty
  entries are dropped.
- **JSON** must parse and carry the stated shape.
- **Numbers** are refused rather than guessed at: a port is an integer from 1 to
  65535; a positive setting is finite and greater than zero; a non-negative
  setting is an integer of zero or more.

A configuration the gateway cannot read unambiguously is rejected at startup
rather than resolved by guessing.

## Server

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `CHORA_ENVIRONMENT` | `development` | The deployment environment; selects the security policy. One of `development`, `test`, `production` (case-insensitive). `production` refuses to start unless `GATEWAY_AUTH_ENABLED` is `true` with at least one `GATEWAY_API_KEYS` entry. |
| `CHORA_HTTP_PORT` | `8080` | The HTTP listener port. An integer from 1 to 65535. |
| `CHORA_GRPC_PORT` | `9090` | The gRPC listener port, started and stopped with the HTTP one. An integer from 1 to 65535. The listener binds before startup completes, so a port that cannot be taken fails the start rather than producing a process that answers a probe on a port it is not serving. |
| `CHORA_BIND_ADDRESS` | `0.0.0.0` | The address both listeners bind. An IP address or a resolvable host; an unusable value fails the start with a configuration error. The default exposes the service on every interface — set it to `127.0.0.1` only when the service must not be reachable from other containers or hosts. |
| `LOG_LEVEL` | `info` | The structured log level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive). |

## Authentication

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `GATEWAY_AUTH_ENABLED` | `false` | Whether the `/v1/*` endpoints and `/metrics` require a valid credential. Boolean. With authentication disabled, every request proceeds with the anonymous identity and all callers share one rate bucket and one concurrency bound. |
| `GATEWAY_API_KEYS` | *(empty)* | Comma-separated client credentials. Each entry is `<key>` or `<key>:<tenant>`, binding the key to a trusted tenant. Required non-empty when `GATEWAY_AUTH_ENABLED=true`. A credential is accepted from `Authorization: Bearer` or `x-api-key`; conflicting values in the two are refused. Keys are never logged or echoed — logs carry a derived, non-secret key label. |
| `GATEWAY_ALLOW_ANONYMOUS_HEALTH` | `true` | Whether `/healthz` and `/readyz` answer without a credential. Boolean. A presented credential is still validated on those paths, so a bad credential is never waved through. |

## Reliability

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `PROVIDER_TIMEOUT_SECONDS` | `110` | The per-attempt upstream timeout, budgeted inside the invocation's single deadline. A positive, finite number. |
| `PROVIDER_CONNECT_TIMEOUT_SECONDS` | `10` | The upstream connect timeout. A positive, finite number. |
| `PROVIDER_RETRY_COUNT` | `2` | How many times one route may be retried after its first attempt, so a route is attempted at most `PROVIDER_RETRY_COUNT + 1` times. A non-negative integer. |
| `PROVIDER_RETRY_BACKOFF_MS` | `500` | The base delay before a same-route retry. The delay grows exponentially with a route-local exponent and is extended — never shortened — by jitter, is capped, and is refused rather than slept away when it would consume the invocation's remaining budget. A non-negative integer. |
| `PROVIDER_MAX_CONCURRENT_REQUESTS` | `100` | The cap on simultaneous upstream attempts across all callers. Waiting callers are bounded separately; a caller that arrives at a full queue is refused immediately rather than queued. A positive integer. |

## Limits

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `GATEWAY_RATE_LIMIT_REQUESTS` | `0` | The per-client request-rate bound: a token bucket keyed on the credential, so a caller may burst up to the configured number and is then held to the rate. `0` disables the rate limiter. A non-negative integer. |
| `GATEWAY_RATE_LIMIT_WINDOW_SECONDS` | `60` | The bucket's refill window. A positive number. |
| `GATEWAY_TENANT_MAX_CONCURRENT_REQUESTS` | `0` | The per-tenant concurrency bound, which shares the upstream pool between tenants; a credential bound to no tenant is its own principal for this bound. `0` disables the limiter. A non-negative integer. |
| `GATEWAY_MAX_REQUEST_BODY_BYTES` | `16777216` | The request-body bound for both transport surfaces in each one's own unit (HTTP request bytes, gRPC received-message bytes). A positive integer. The bound cannot be removed by configuration, only tuned: a deployment that never set one is the one a limit protects. |
| `GATEWAY_SHUTDOWN_TIMEOUT_SECONDS` | `30` | How long the server waits for a worker that has been asked to stop before killing it, on `SIGTERM` or `SIGINT`. A positive number. |

## Operation blocks

Three environment blocks configure the three operations independently:

| Block | Operation | Base capability |
| --- | --- | --- |
| `TEXT_LLM_*` | chat and text generation | `chat` |
| `IMAGE_GEN_*` | image generation | `generation` |
| `EMBEDDINGS_*` | embeddings | `embeddings` |

Each block's singleton variables define the primary route, and
`<BLOCK>_ROUTES_JSON` appends additional routes as a JSON array, in array order.
Route order is fallback order: the primary route is tried first, then each
configured fallback in turn. The gateway exposes one fixed public model ID per
operation (`text-default`, `image-default`, `embedding-default`); a caller-
supplied model name can never reach an unconfigured upstream endpoint.

### Block variables

| Variable | Required | Meaning and validation |
| --- | --- | --- |
| `<BLOCK>_ENDPOINT` | yes* | The upstream base URL. Must be an absolute URL with a scheme and host; `https` is required; it must not embed credentials or carry a query string or fragment; its host must resolve to no blocked address (loopback, private, link-local, metadata, unique-local, multicast, reserved, unspecified) and must be named by `GATEWAY_ALLOWED_UPSTREAM_HOSTS` when that allowlist is set. |
| `<BLOCK>_MODEL_ID` | yes* | The upstream model identifier. Non-empty. |
| `<BLOCK>_FORMAT` | yes* | The inference binding's provider dialect for this route — the name the binding understands, such as `openai`, `anthropic`, `openrouter` or `vertex_ai`; not the company operating the endpoint. The dialect must be one the binding has verified, or the start fails. |
| `<BLOCK>_AUTH_TYPE` | no | The route's authentication override: one of `bearer`, `x-api-key`, `api-key`, `none` or `custom-header`. Empty — the default — means the provider's own authentication. Unknown values are refused. |
| `<BLOCK>_API_KEY` | no | The route's static credential. Required only when the route's authentication mechanism requires one: not for ambient-credential dialects such as `vertex_ai`, and not for `AUTH_TYPE=none`. |
| `<BLOCK>_PATH` | no | A path appended to the endpoint's path, never replacing it. Segments are joined literally — none is dropped, deduplicated or repaired; a configured repetition survives. A query string, a fragment, or a `.`/`..` segment is refused. An absolute URL here is refused unless `<BLOCK>_ALLOW_ABSOLUTE_PATH_URL=true`. |
| `<BLOCK>_ALLOW_ABSOLUTE_PATH_URL` | no | Boolean, default `false`. Opts the route into accepting an absolute URL in `<BLOCK>_PATH`, which then replaces the base URL entirely. |
| `<BLOCK>_EXTRA_HEADERS_JSON` | no | A JSON object of string values: extra non-authentication headers for this route. The reserved authentication headers (`authorization`, `x-api-key`, `api-key`) are refused — the route's own authentication configuration owns those. Every value must be a string. |
| `<BLOCK>_CAPABILITIES` | no | Comma-separated capability names the route declares, from one vocabulary: `chat`, `tools`, `vision`, `structured-output`, `generation`, `editing`, `embeddings`. Defaults to the block's base capability when unset. An explicitly empty list and unknown names are refused. A declared capability is a claim the gateway relies on when it selects a route. |

\* The three identity variables (`ENDPOINT`, `MODEL_ID`, `FORMAT`) are required
together: a block holding none of them defines no route, and any other value
left in such a block — a capability, an API key — is refused at startup as
configuration the gateway would otherwise discard in silence.

### Advertised limits

Two text settings are advertised limits rather than route fields; the model
registry reads them from the environment. A blank value means the deployment did
not configure the limit, which is different from zero.

| Variable | Meaning and validation |
| --- | --- |
| `TEXT_LLM_CONTEXT_WINDOW` | The text operation's advertised context window. A non-negative integer or blank. |
| `TEXT_LLM_MAX_OUTPUT_TOKENS` | The text operation's advertised output bound. A non-negative integer or blank. |

The embeddings operation's dimensionality is a route property instead:
`EMBEDDINGS_DIMENSIONS` (a positive integer or blank — blank declares no
dimensionality and checks none) and `EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM`
(boolean, default `false`; whether the route accepts a caller-selectable
`dimensions`). Both may also appear on embeddings route entries.

### Route entries: `<BLOCK>_ROUTES_JSON`

A JSON array of additional routes appended after the singleton primary route.
Each entry is self-contained: nothing is inherited from the singleton block,
because a fallback route may point at a different origin and must carry its own
credentials. Precedence between routes is positional, never field-merged.

| Key | Required | Meaning |
| --- | --- | --- |
| `endpoint` | yes | As `<BLOCK>_ENDPOINT`. |
| `model_id` | yes | As `<BLOCK>_MODEL_ID`. |
| `format` | yes | As `<BLOCK>_FORMAT`. |
| `api_key` | no | As `<BLOCK>_API_KEY`. |
| `auth_type` | no | As `<BLOCK>_AUTH_TYPE`. |
| `path` | no | As `<BLOCK>_PATH`. |
| `extra_headers` | no | As `<BLOCK>_EXTRA_HEADERS_JSON`, already a JSON object. |
| `capabilities` | no | An array of capability names, or a comma-separated string; defaults to the block's base capability when absent. |
| `allow_absolute_path_url` | no | Boolean, default `false`. |
| `dimensions` | no | A positive integer or absent; embeddings entries only. |
| `supports_dimensions_param` | no | Boolean, default `false`; embeddings entries only. |

Rejected at startup: a malformed JSON array; an entry that is not an object;
an entry missing a required key; an unsupported key; two routes of one operation
naming the same endpoint, format and model ID; a `<BLOCK>_ROUTES_JSON` set while
the singleton block is unset; and the indexed route form (`TEXT_LLM_1_*`,
`TEXT_LLM_2_*`, …), which does not exist.

## Upstream allowlist

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `GATEWAY_ALLOWED_UPSTREAM_HOSTS` | *(empty)* | Comma-separated hostnames the deployment permits as upstream endpoints. Empty means no allowlist: every host the URL policy permits may be configured. An entry is a hostname and not an authority — a scheme, port, path, userinfo or IPv6 literal is refused rather than ignored — and must satisfy the hostname grammar (dot-separated labels of letters, digits and hyphens, a label not beginning or ending with a hyphen, at most 253 characters). Matching is case-insensitive. The allowlist narrows and never widens: a host it names still has to pass the address screen. A route outside the allowlist fails the start, naming the variable that produced it. |

## Usage and cost

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `USAGE_TRACKING_ENABLED` | `true` | Whether invocations leave usage and cost records. Boolean. |
| `COST_TRACKING_ENABLED` | `false` | Whether the gateway estimates cost from the configured pricing. Boolean. |
| `USAGE_STORAGE` | `memory` | Where usage and cost records are persisted: `none` (records discarded), `memory` (a bounded in-process store — the default single-instance deployment), `sqlite` or `postgres`. The durable modes are named by the vocabulary but require configured persistence this deployment does not provide: setting one fails the start rather than silently falling back to memory. The in-memory store holds 100,000 records and drops the oldest once full. |
| `COST_PRICING_JSON` | *(empty)* | A JSON object keyed by public model ID, e.g. `{"text-default": {"currency": "USD", "tokens": {"input": 3.0, "output": 15.0, "cached": 0.3}}}`. Prices are per million tokens, the unit providers quote. The whole value is validated at startup rather than partially applied: a currency must be a non-empty string (not checked against a currency registry); the token categories must be from `input`, `output` and `cached`, at least one of them required; each price must be a finite, non-negative number; unknown keys are refused. A deployment that configures no pricing records usage and no cost. |

## Observability

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `OTEL_ENABLED` | `false` | Whether distributed tracing is configured. Boolean. With it off, no tracer provider is built and the tracing calls stay no-ops. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(empty)* | The base endpoint for the OTLP/HTTP exporter; the signal path is appended to it. Required when `OTEL_ENABLED=true` — tracing enabled with nowhere to send spans fails the start rather than quietly dropping every span. |

## Governance

| Variable | Default | Meaning and validation |
| --- | --- | --- |
| `CHORA_OBSERVABILITY_DSN` | *(empty)* | Postgres DSN for the governance mechanisms (budget admission/settlement, token-usage outbox, invocation idempotency, egress policy). Required for governance to function; leave empty to run without a database. Connect as the observability app role (NOBYPASSRLS) — row-level security keys on the tenant GUCs set inside each transaction. |
| `CHORA_IDENTITY_GRPC_ADDR` | *(empty)* | gRPC address of chora-identity's `ManaService` (`host:port`). When set alongside `CHORA_OBSERVABILITY_DSN`, every metered action quotes affordability before fail-closed compute and debits the learner's mana wallet after a served call. Required in the default `chora` governance mode whenever the DSN is set — a governed deployment cannot silently run un-metered; set `CHORA_GOVERNANCE_MODE=standalone` to run the chain un-metered by explicit choice. The service mesh terminates the mTLS the identity contract declares, so the address is plaintext. |
| `CHORA_GOVERNANCE_MODE` | `chora` | The deployment's intent. `chora` (the default) is part of the platform: `CHORA_IDENTITY_GRPC_ADDR` is required whenever `CHORA_OBSERVABILITY_DSN` is set, and the start fails without it. `standalone` runs the governance chain without the identity service, un-metered. |
| `CHORA_LLM_BUDGET_REQUIRED` | `false` | Make budget enforcement fail-closed: a tenant with no active budget window is blocked instead of allowed through. Boolean. |
| `CHORA_LLM_BUDGET_REQUIRED_TENANTS` | *(empty)* | Comma-separated tenant IDs the budget requirement applies to; an empty list with the flag on applies it to every tenant. |

## Test-only variables

`GATEWAY_LIVE_TEXT_ENDPOINT`, `GATEWAY_LIVE_TEXT_MODEL_ID`,
`GATEWAY_LIVE_TEXT_API_KEY` and `GATEWAY_LIVE_TEXT_FORMAT`, plus the same four
with a `_2` suffix, are read by the optional live-provider smoke test — see
`docs/DEPLOYMENT.md`. The gateway itself never reads them.

## Implementation state, not operator configuration

`LITELLM_MODE` is forced to `PRODUCTION` by the gateway before the first import
of the inference library, so the library's import-time `.env` load never runs
and configuration cannot arrive from the working directory. An ambient
`LITELLM_MODE=DEV` does not re-enable the load. The variable is gateway-owned
implementation state and is not part of the public configuration surface.
