# model-gateway

A vendor-neutral gateway in front of AI model providers. It exposes one stable OpenAI-compatible
HTTP surface and one gRPC service, and routes each request to whichever upstream the deployment
configured — so an application talks to the gateway, and *which provider answers* is an
environment change rather than a code change.

```text
application ──▶ model-gateway ──▶ provider A
   HTTP /v1        routing,         provider B
   gRPC gateway.v1 policy,          provider C
                   accounting
```

Every command, port, endpoint and variable below was checked against the source in this
repository and against a running `chora-stack` — including a live chat completion.

## What it is

**This repository is a Python service.** The `go.mod` at the root is not the gateway: it is a
generated Go module holding contract stubs so Go callers can talk to the gRPC surface. Running
`go build` here produces nothing; `uv sync` is the way in. The Dockerfile builds a Python image.

Inference itself is performed by [LiteLLM] (`litellm-core`), pinned and isolated behind one
binding that owns upstream protocol translation and nothing else. The gateway core owns
everything that is policy:

| The core owns | The binding owns |
| --- | --- |
| operation → route resolution | upstream protocol translation |
| caller authentication, tenant isolation | the provider's wire format |
| capability enforcement, request validation, size bounds | — |
| deadline, cancellation, retry and fallback walk | — |
| idempotency, usage and cost accounting | — |
| the public error contract, network security, log redaction | — |

LiteLLM's own retry and fallback are **disabled**, so exactly one retry loop and one deadline
exist and both are the gateway's. That separation is what makes the provider swappable: the core
names no provider, and the binding is the only module that does.

### What the stack runs

| Feature | State in the stack | Consequence |
| --- | --- | --- |
| Text/chat | **live** | a real completion returns through the configured provider |
| Image generation | **configured** | `image-default` is listed and routable |
| Embeddings | **live, 1024 dims** | `embedding-default` reports `dimensions: 1024` |
| Exa search + synthesis | **configured** | requires `EXA_API_KEY`; the stack refuses to start without it |
| Inbound auth | **disabled** | `GATEWAY_AUTH_ENABLED=false` — legal only because `CHORA_ENVIRONMENT=development` |
| Accounting | **in memory only** | usage is recorded to `MemoryAccountingSink`; records are lost on restart, and no database sink is implemented |
| Governance | **wired** | `CHORA_OBSERVABILITY_DSN` + `CHORA_IDENTITY_GRPC_ADDR` enable the budget/mana chain against `chora_observability` |
| Tracing | **enabled** | `OTEL_ENABLED=true` to `otel-collector:4317` |

Accounting and governance are **different mechanisms** and it is worth not conflating them:
governance is the budget/mana enforcement chain and it does use the database; usage/cost
accounting goes to an in-process sink and survives nothing. See [Events](#events).

Auth being off is deliberate and commented in `compose.yml`: no caller in the stack presents a
gateway credential, and production must turn it on. See [Troubleshooting](#troubleshooting) — the
process refuses to start if you set `CHORA_ENVIRONMENT=production` with auth still off.

## Prerequisites

| Route | What you need |
| --- | --- |
| In the stack | A working `chora-stack` deployment. PostgreSQL, `observability-migrate` and `identity` arrive from it. |
| Standalone (container) | Docker, and an `.env` with at least the text block. |
| Standalone (source) | Python **3.13+** and [uv]. `requires-python = ">=3.13"` in `pyproject.toml`. |

A container with **no** configuration starts fine and serves an empty model list. That is the trap
this service is built around: a missing `.env` does not look like a crash, and it does not look
like a working deployment either. `GET /healthz` returning `ok` says the process is the process —
it says nothing about whether any provider is configured. `GET /v1/models` is the real check.

## Credentials

All provider credentials are plain environment variables. There are three independent blocks and
one search provider.

| Credential | Needed for | Where it comes from |
| --- | --- | --- |
| `TEXT_LLM_API_KEY` | chat, completions, responses | Your text provider. **Gemini:** `https://aistudio.google.com/app/apikey` (docs: `ai.google.dev/gemini-api/docs/api-key`). |
| `IMAGE_GEN_API_KEY` | `/v1/images/generations` | Your image provider; may be the same key as the text block if one provider serves both. |
| `EMBEDDINGS_API_KEY` | `/v1/embeddings` | Your embedding provider. Gemini embeddings use the same AI Studio key. |
| `EXA_API_KEY` | Exa search and synthesis | `https://dashboard.exa.ai/api-keys`. **Required by `chora-stack/compose.yml`**, which uses `${EXA_API_KEY:?…}` — omit it and compose refuses to start the service. The Python settings layer itself treats it as optional (default empty), so a standalone run without it boots and fails only when a search is attempted. |
| `GATEWAY_API_KEYS` | inbound caller auth | You generate these: comma-separated opaque strings, e.g. `openssl rand -hex 32`. Only needed when `GATEWAY_AUTH_ENABLED=true`. |

A model block is enabled by **three** fields — `<BLOCK>_ENDPOINT`, `<BLOCK>_MODEL_ID` and
`<BLOCK>_FORMAT`. The API key is optional: the parser reads it without requiring it, so an
unauthenticated upstream (a local inference server, an in-cluster vLLM) works with the three
identity fields alone. Each block is self-contained, so a deployment can mix providers freely and
nothing is inherited between blocks.

## Run it in the stack

The Compose service is `model-gateway`, and it takes a **distinct image name** on purpose:

```yaml
image: chora-model-gateway-python
```

The comment explains why: a plain `compose up` would otherwise reuse a stale `chora-model-gateway`
image left over from the decommissioned Go service of the same name.

```sh
docker compose up -d model-gateway
```

It builds from this repository, depends on `postgres` (healthy), `observability-migrate`
(completed) and `identity` (`service_started` — the identity service has no healthcheck, and the
mana meter is fail-open on a metering error, so a slow identity costs a temporary unmetered window
rather than availability). It exposes `8080` and `9090`, with no host port.

The stack's environment, in the order `compose.yml` sets it:

```text
CHORA_HTTP_PORT                8080
CHORA_GRPC_PORT                9090
CHORA_ENVIRONMENT              development
LOG_LEVEL                      info
CHORA_OBSERVABILITY_DSN        postgres://chora_observability_app_rw:…@postgres:5432/chora_observability
CHORA_IDENTITY_GRPC_ADDR       ${SVC_IDENTITY_GRPC_URL:-identity:9090}
EXA_API_KEY                    (required, no default)

EXA_SYNTHESIS_MODEL_ID         ${TEXT_LLM_FORMAT}/${TEXT_LLM_MODEL}
EXA_SYNTHESIS_ENDPOINT         ${TEXT_LLM_BASE_URL}
EXA_SYNTHESIS_API_KEY          ${TEXT_LLM_API_KEY}

TEXT_LLM_ENDPOINT              ${TEXT_LLM_BASE_URL}
TEXT_LLM_MODEL_ID              ${TEXT_LLM_MODEL}
TEXT_LLM_API_KEY               ${TEXT_LLM_API_KEY}
TEXT_LLM_AUTH_TYPE             ${TEXT_LLM_AUTH_TYPE:-bearer}
TEXT_LLM_FORMAT                ${TEXT_LLM_FORMAT:-anthropic}
TEXT_LLM_DISCARD_REASONING     true
TEXT_LLM_DISCARD_USAGE_DETAILS true

IMAGE_GEN_ENDPOINT             ${IMAGE_LLM_BASE_URL}
IMAGE_GEN_MODEL_ID             ${IMAGE_LLM_MODEL}
IMAGE_GEN_API_KEY              ${IMAGE_LLM_API_KEY}
IMAGE_GEN_FORMAT               ${IMAGE_GEN_FORMAT:-openai}

EMBEDDINGS_ENDPOINT            ${EMBEDDING_LLM_BASE_URL}
EMBEDDINGS_MODEL_ID            ${EMBEDDING_LLM_MODEL}
EMBEDDINGS_API_KEY             ${EMBEDDING_LLM_API_KEY}
EMBEDDINGS_FORMAT              openai
EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM  ${EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM:-false}
EMBEDDINGS_DIMENSIONS          ${EMBEDDINGS_DIMENSIONS:-}

GATEWAY_AUTH_ENABLED           ${GATEWAY_AUTH_ENABLED:-false}
GATEWAY_API_KEYS               ${GATEWAY_API_KEYS:-}
OTEL_ENABLED                   true
OTEL_EXPORTER_OTLP_ENDPOINT    http://otel-collector:4317
```

Three of those defaults are worth understanding rather than copying:

- `TEXT_LLM_AUTH_TYPE=bearer` because the bundled provider endpoint expects
  `Authorization: Bearer` and not the `x-api-key` header LiteLLM's anthropic handler sends by
  default.
- `TEXT_LLM_DISCARD_REASONING=true` because the provider reports a reasoning trace beside every
  answer and offers no parameter to disable it; the route declares the trace is not part of the
  answer and the gateway drops it instead of failing every call.
- `EXA_SYNTHESIS_MODEL_ID` derives from `TEXT_LLM_FORMAT`/`TEXT_LLM_MODEL` by nested default.
  Compose substitutes those, so you rarely set it — but the resulting value **must be
  provider-prefixed** (`openai/gemini-3.8-flash`), because a bare Gemini id makes LiteLLM infer
  the `vertex_ai` provider whose `google.auth` dependency this image does not ship.

## Run it standalone

Container, with your own `.env`:

```sh
docker build -t model-gateway:local .
docker run --rm -p 8080:8080 -p 9090:9090 --env-file .env model-gateway:local
```

From source:

```sh
uv sync
uv run model-gateway
```

The entry point is `model_gateway.__main__:main`. The minimum to serve inference is one complete
block — endpoint, model id, format and key:

```sh
export TEXT_LLM_ENDPOINT='https://generativelanguage.googleapis.com/v1beta/openai'
export TEXT_LLM_MODEL_ID='gemini-3.8-flash'
export TEXT_LLM_FORMAT='openai'
export TEXT_LLM_API_KEY='…'
uv run model-gateway
```

With nothing set it still starts, serves `/v1/models` with an empty list, and fails every
inference call. That is the designed behaviour, not a bug.

## Configuration

Configuration is entirely environment variables, in five families. `docs/CONFIGURATION.md` is the
complete reference and `docs/DEPLOYMENT.md` covers operating it; this table is the operator's
working subset.

**Operation blocks** — one per capability, each self-contained:

| Suffix | Meaning |
| --- | --- |
| `<BLOCK>_ENDPOINT` | upstream base URL — **required to enable the block** |
| `<BLOCK>_MODEL_ID` | model id sent upstream — **required to enable the block** |
| `<BLOCK>_FORMAT` | upstream dialect — **required to enable the block** |
| `<BLOCK>_API_KEY` | credential for that upstream — optional |
| `<BLOCK>_AUTH_TYPE` | authentication mode |
| `<BLOCK>_PATH` | request path override |
| `<BLOCK>_ROUTES_JSON` | **additional routes, in array order = fallback order** |

where `<BLOCK>` is `TEXT_LLM` (chat), `IMAGE_GEN` (generation) or `EMBEDDINGS` (embeddings). A
`*_ROUTES_JSON` entry is self-contained — no field is inherited from the singleton block, because
a fallback route may point at a different origin and must carry its own credentials. Precedence
between routes is positional, never field-merged. The indexed form (`TEXT_LLM_1_*`) is
**rejected**, not supported.

**Inbound policy** (`GATEWAY_*`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_AUTH_ENABLED` | `false` | require a caller credential |
| `GATEWAY_API_KEYS` | empty | comma-separated accepted keys |
| `GATEWAY_MAX_REQUEST_BODY_BYTES` | — | body size ceiling |
| `GATEWAY_RATE_LIMIT_REQUESTS` / `…_WINDOW_SECONDS` | — | per-window request limit |
| `GATEWAY_TENANT_MAX_CONCURRENT_REQUESTS` | — | per-tenant concurrency cap |
| `GATEWAY_ALLOWED_UPSTREAM_HOSTS` | — | upstream host allowlist |
| `GATEWAY_ALLOW_ANONYMOUS_HEALTH` | — | serve health without a credential |
| `GATEWAY_SHUTDOWN_TIMEOUT_SECONDS` | `30` | graceful shutdown budget |

**Accounting** (`USAGE_*`, `COST_*`) — the family that decides whether usage records exist at all:

| Variable | Default | Purpose |
| --- | --- | --- |
| `USAGE_TRACKING_ENABLED` | **`true`** | record every invocation; turning this off disables accounting entirely |
| `COST_TRACKING_ENABLED` | **`false`** | attach a price to each record (needs a pricing table) |
| `USAGE_STORAGE` | **`memory`** | where records go: `none`, `memory`, `sqlite` or `postgres` |

Both tracking flags must be on for a record to carry a cost, and pricing is optional while usage
is not: an unpriced deployment still records counts, just with no cost attached.

**The `USAGE_STORAGE` value is the trap.** This build implements only two sinks —
`none` (`NullAccountingSink`) and `memory` (`MemoryAccountingSink`). `sqlite` and `postgres` are
accepted by the enum but rejected at wiring time:

```text
USAGE_STORAGE is 'postgres', which requires configured persistence that this deployment
does not provide; use memory or none
```

So there is no database sink in this repository, and the default `memory` means usage records are
lost on every restart. Do not set `USAGE_STORAGE=postgres` expecting persistence — it fails loud
rather than silently dropping records, which is the honest behaviour but not what the name
suggests.

**Reliability** (`PROVIDER_*`):

| Variable | Purpose |
| --- | --- |
| `PROVIDER_TIMEOUT_SECONDS` | per-attempt deadline |
| `PROVIDER_CONNECT_TIMEOUT_SECONDS` | connection deadline |
| `PROVIDER_RETRY_COUNT` | retries before the fallback walk |
| `PROVIDER_RETRY_BACKOFF_MS` | backoff between retries |
| `PROVIDER_MAX_CONCURRENT_REQUESTS` | outbound concurrency ceiling |

**Platform** (`CHORA_*`) and search (`EXA_*`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHORA_HTTP_PORT` | `8080` | HTTP listen port |
| `CHORA_GRPC_PORT` | `9090` | gRPC listen port |
| `CHORA_BIND_ADDRESS` | — | bind address |
| `CHORA_ENVIRONMENT` | `development` | **selects the security policy** (see below) |
| `CHORA_IDENTITY_GRPC_ADDR` | unset | identity gRPC for the mana meter |
| `CHORA_OBSERVABILITY_DSN` | unset | governance/accounting database |
| `CHORA_GOVERNANCE_MODE` | — | governance enforcement mode |
| `CHORA_LLM_BUDGET_REQUIRED` / `…_TENANTS` | — | require a budget for listed tenants |
| `EXA_API_KEY` | **required by compose** | Exa search credential |
| `EXA_SEARCH_ENDPOINT` | Exa default | search API base |
| `EXA_SYNTHESIS_MODEL_ID` / `…_ENDPOINT` / `…_API_KEY` | derived from the text block | synthesis leg |
| `OTEL_ENABLED` | — | tracing on/off; setting the endpoint alone does **not** enable it |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP/HTTP collector endpoint |

### Three configurations that refuse to boot

The settings object validates on construction and fails loud rather than serving something
half-configured. Three rules will catch you:

```text
CHORA_ENVIRONMENT is production but GATEWAY_AUTH_ENABLED is false; production must not serve
unauthenticated inference. Set GATEWAY_AUTH_ENABLED=true with GATEWAY_API_KEYS, or set
CHORA_ENVIRONMENT=development

GATEWAY_AUTH_ENABLED is true but GATEWAY_API_KEYS is empty; set one or more comma-separated keys,
or set GATEWAY_AUTH_ENABLED=false

CHORA_IDENTITY_GRPC_ADDR is required when CHORA_OBSERVABILITY_DSN is set — governance requires
mana metering; set CHORA_GOVERNANCE_MODE=standalone to run the chain un-metered
```

`CHORA_ENVIRONMENT` is therefore not a label — it selects the security policy. The stack runs
`development`, which is the only reason auth-off is legal there.

The third rule explains the compose dependency on `identity`: governance meters mana through
chora-identity, so a DSN without an identity address is refused rather than run unmetered.
`CHORA_GOVERNANCE_MODE=standalone` is the explicit opt-out for running the gateway apart from the
platform's mana ledger.

## Verify it works

### The process is up

```sh
docker compose exec -T nats wget -qO- http://model-gateway:8080/healthz
docker compose exec -T nats wget -qO- http://model-gateway:8080/readyz
```

Observed against the running stack:

```text
ok                                             ← /healthz is PLAIN TEXT, not JSON
{"status":"ready","checks":{}}
```

`/healthz` returning `ok` as text is deliberate; a JSON parser expecting an object will choke on
it. Neither probe says anything about provider configuration.

### The real check: configured models

```sh
docker compose exec -T nats wget -qO- http://model-gateway:8080/v1/models
```

```json
{"object":"list","data":[
  {"id":"text-default","object":"model","created":0,"owned_by":"gateway","capabilities":["chat"]},
  {"id":"image-default","object":"model","created":0,"owned_by":"gateway","capabilities":["generation"]},
  {"id":"embedding-default","object":"model","created":0,"owned_by":"gateway","capabilities":["embeddings"],"dimensions":1024}
]}
```

Three logical models is a fully configured deployment. An **empty** `data` array means the process
is healthy and every inference call will fail — that is the failure mode to watch for.

The `dimensions: 1024` on the embedding entry is the operator's confirmation that
`EMBEDDINGS_DIMENSIONS` reached the gateway and downstream consumers can size their vectors.

### A real completion crosses the provider boundary

```sh
docker compose exec -T nats wget -qO- \
  --post-data='{"model":"text-default","messages":[{"role":"user","content":"Reply with exactly: PONG"}]}' \
  --header='Content-Type: application/json' \
  http://model-gateway:8080/v1/chat/completions
```

Observed:

```json
{"id":"chatcmpl-e3bca8600e33447d997f7cf9e41b184f","object":"chat.completion",
 "created":1792563449,"model":"text-default","idempotency_status":"not_applicable",
 "choices":[{"index":0,"message":{"role":"assistant","content":"PONG"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":106}}
```

Two things in that response are gateway-specific and worth recognising:

- **`idempotency_status`** is an extra field this gateway adds to the OpenAI shape.
- **`total_tokens` (106) is not `prompt + completion` (9).** With
  `TEXT_LLM_DISCARD_USAGE_DETAILS=true` the provider's usage detail is discarded, so the total is
  what the gateway records rather than a sum. Do not assert on `prompt_tokens + completion_tokens
  == total_tokens` against this stack.

### Metrics are exported

```sh
docker compose exec -T nats wget -qO- http://model-gateway:8080/metrics | head -5
```

```text
# HELP gateway_requests_total Invocations the gateway was asked to serve, one count per invocation.
# TYPE gateway_requests_total counter
gateway_requests_total 10
```

`gateway_requests_total` incrementing across a call is the cheapest proof that traffic reached the
inference service rather than dying at the router.

## Troubleshooting

- **`/healthz` is green but every inference call fails** — no route is configured. Check
  `/v1/models`: an empty `data` array means the environment blocks never reached the process. This
  is the service's primary silent failure.
- **The container exits with a configuration error mentioning production and auth** —
  `CHORA_ENVIRONMENT=production` with `GATEWAY_AUTH_ENABLED=false`. Either enable auth with keys,
  or set the environment back to `development`. This check is intentional and must not be
  bypassed by weakening the code.
- **It exits saying auth is enabled but keys are empty** — set `GATEWAY_API_KEYS` to one or more
  comma-separated keys, or turn auth off.
- **`TEXT_LLM_1_*` has no effect** — the indexed form is explicitly rejected, not supported. Use
  `TEXT_LLM_ROUTES_JSON` for additional routes; order is fallback order.
- **A fallback route silently ignores the primary's settings** — by design: `*_ROUTES_JSON`
  entries inherit nothing and must carry their own endpoint, model, format and key.
- **Exa calls fail with a Vertex/LiteLLM provider error** — `EXA_SYNTHESIS_MODEL_ID` is a bare
  model id. It must be provider-prefixed (`openai/…`) or LiteLLM infers `vertex_ai`, whose
  `google.auth` dependency this image does not ship.
- **Every answer has an empty reasoning section** — `TEXT_LLM_DISCARD_REASONING=true` drops the
  provider's reasoning trace because it cannot be disabled and is not part of the answer.
- **Traces never appear although the endpoint is set** — `OTEL_ENABLED` is the switch. The compose
  comment is explicit that setting the endpoint alone does not enable tracing.
- **`docker compose up` runs an old Go gateway instead of this one** — the image name is pinned to
  `chora-model-gateway-python` precisely to avoid reusing the decommissioned `chora-model-gateway`
  image. Do not remove that line.
- **`go build ./...` does nothing useful** — correct. This is a Python service; the root `go.mod`
  only publishes generated contract stubs under `gen/go/`.
- **Usage rows are gone after a restart** — expected with `USAGE_STORAGE=memory`, the default.
  This build implements only the `none` and `memory` sinks; `sqlite` and `postgres` are rejected
  at wiring time. Usage records are not written to `chora_observability` — only *governance* uses
  that database.
- **Governance/budget enforcement does nothing** — `CHORA_OBSERVABILITY_DSN` is unset, so no
  chain was wired and the invocation flow runs as if governance never existed. If the DSN is set
  but `CHORA_IDENTITY_GRPC_ADDR` is not, the process refuses to start instead.
- **Governance tables are missing** — the schema belongs to `chora-observability`; check that the
  `observability-migrate` one-shot completed. There is no migration here to run.

## API surface

Two transports, identical policy.

| Surface | Endpoints |
| --- | --- |
| HTTP inference | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, `POST /v1/images/generations` |
| HTTP discovery | `GET /v1/models` |
| HTTP operations | `GET /healthz` (plain text), `GET /readyz` (JSON), `GET /metrics` (Prometheus) |
| gRPC | `gateway.v1.ModelGatewayService`: `Invoke`, `InvokeStream`, `Embed`, `ListModels`, plus the standard gRPC health protocol |

Both transports reach the same inference service — one decides what may happen, and each
transport only translates. The gRPC package is versioned and its field numbers are stable; see
`proto/gateway/v1/gateway.proto`.

The public model ids (`text-default`, `image-default`, `embedding-default` in the stack) are
gateway-level names, not provider model names. Mapping a public id onto an operation is the
northbound layer's job; the provider model id never appears in a client request.

Errors follow one public contract across both transports; upstream provider errors are translated
rather than leaked.

## Architecture

```text
src/model_gateway/
  __main__.py      entry point (model_gateway.__main__:main)
  app.py           application assembly
  registry.py      operation → route resolution
  inference.py     the single inference service both transports call
  auth.py          caller authentication and auth types
  capabilities.py  what a route declared it can do
  limits.py        body size, rate, concurrency
  reliability.py   deadline, retry, fallback walk
  idempotency.py   request de-duplication
  accounting.py    usage and cost accounting
  governance/      budget enforcement against the observability database
  providers/       the ONLY provider-aware module (LiteLLM binding)
  http/            OpenAI-compatible surface (blacksheep)
  grpc/            gRPC surface
  security/        upstream host allowlist, network policy
  redaction.py     what may be logged
  metrics.py       Prometheus
  observability.py tracing
  health.py        healthz / readyz
  config/
    settings.py    environment → typed settings, with fail-loud validation
    routes.py      TEXT_LLM_* / IMAGE_GEN_* / EMBEDDINGS_* blocks and routes JSON
```

Runtime is **granian** (not uvicorn) serving the **blacksheep** ASGI app; `orjson` handles
serialisation and `structlog` handles logging. Dependencies are pinned exactly in `pyproject.toml`
and locked in `uv.lock`.

The layering rule is the maintainable part: `providers/` is the only module that knows a provider
exists. Everything else is policy and would survive swapping LiteLLM for something else.

## Events

**This service neither publishes nor subscribes to domain events.** It is request/response over
HTTP and gRPC; there is no NATS client and no outbox.

What it *does* emit is three other kinds of telemetry:

| Output | Mechanism |
| --- | --- |
| Metrics | Prometheus exposition on `GET /metrics` (`gateway_requests_total`, duration histograms) |
| Traces | OTLP/HTTP to `otel-collector:4317`, gated by `OTEL_ENABLED` |
| Usage records | `AccountingRecorder` into `NullAccountingSink` or `MemoryAccountingSink` — **in process only**, lost on restart |
| Governance writes | the budget/mana chain against `chora_observability`, when `CHORA_OBSERVABILITY_DSN` is set |
| Logs | `structlog`, with a redaction module deciding what may be written |

If you are looking for the gateway's contribution to the platform event stream, there isn't one —
usage and cost land in the observability database, and `chora-observability` owns what happens
next.

## Migrations

**None in this repository.** The database schema the gateway writes to belongs to
`chora-observability`: `CHORA_OBSERVABILITY_DSN` points at `chora_observability` as the
`chora_observability_app_rw` role, and the tables are created by that service's migrations.

That is why the compose block depends on `observability-migrate` completing, and why the comment
notes the gateway reuses *the same observability database the Go gateway used, as the NOBYPASSRLS
app role the migrations create*. Nothing to run here.

[LiteLLM]: https://github.com/BerriAI/litellm
[uv]: https://docs.astral.sh/uv/
