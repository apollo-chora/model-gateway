# Product Requirements Document
# Vendor-Neutral AI Model Gateway

**Version:** 1.1  
**Status:** Implementation-ready specification  
**Implementation language:** Python  
**Architecture:** Standalone, self-hostable, multi-provider AI gateway  
**Protocols:** gRPC, HTTP/JSON  
**Configuration:** Environment variables  
**Primary deployment:** Containerized service  
**Supported capabilities:** Text generation, chat, tool calling, vision, image generation, embeddings

**Revision history:**

- **1.1** — Removes grounded search and configurable model aliases; delegates all upstream inference to the LiteLLM library.
- **1.0** — Initial specification.

---

## 1. Executive Summary

The AI Model Gateway is a standalone service that provides a unified interface for interacting with artificial intelligence models from any compatible provider.

Applications communicate with the gateway using standardized gRPC or HTTP endpoints. The gateway handles model selection, request translation, provider authentication, inference execution, retries, fallbacks, usage tracking, and response normalization.

The gateway must be **vendor-neutral, configurable through environment variables, and independently deployable**.

A provider is defined by its endpoint, model identifier, authentication configuration, supported capabilities, and LiteLLM provider name.

The gateway delegates upstream protocol handling (request encoding, wire-format translation, and response decoding) to the LiteLLM library.

The gateway must not require a specific cloud provider, proprietary SDK, managed database, external orchestration service, or commercial AI vendor.

### Core design principles

1. **Vendor neutrality:** Provider selection is based on configured LiteLLM providers and capabilities, not vendor names.
2. **Protocol abstraction:** Applications use one consistent API regardless of the upstream protocol.
3. **Environment-driven configuration:** All deployment-specific settings, provider endpoints, model identifiers, credentials, and routing rules are configurable through environment variables.
4. **Standalone operation:** The gateway can operate without external application services.
5. **Optional infrastructure:** Persistence, caching, distributed coordination, and observability integrations are configurable.
6. **Security by default:** Authentication, credential protection, request validation, and network restrictions are first-class requirements.
7. **Deterministic behavior:** Retries, routing, errors, and fallback behavior must be predictable and testable.
8. **Extensibility:** Additional upstream providers are supported through LiteLLM without modifying the gateway's core business logic.

---

## 2. Product Goals

The gateway must provide:

- A unified gRPC API for model inference.
- An OpenAI-compatible HTTP API.
- Configurable text, image, and embedding providers.
- Multi-provider inference through the LiteLLM library.
- Provider-independent request and response representations.
- Support for multi-turn conversations.
- Tool/function calling with structured arguments.
- Multimodal input and output.
- Model aliases and fallback routing.
- Token usage and cost accounting.
- Configurable rate limits and concurrency limits.
- Optional API authentication and tenant isolation.
- Health checks and operational metrics.
- Container-ready deployment.
- Graceful startup, shutdown, and error handling.

### Non-goals for version 1

The following are not required for the initial release:

- Model training or fine-tuning.
- Hosting model weights or running inference engines internally.
- A graphical administration dashboard.
- A proprietary agent orchestration framework.
- Persistent conversation history.
- User billing or payment processing.
- Automatic model discovery across every provider.
- Arbitrary code execution from model-generated tool calls.
- Web search and grounded retrieval; applications integrate a search provider directly.

The gateway may integrate with external inference engines, but it does not implement the inference engine itself.

---

## 3. System Architecture

```text
                       CLIENT APPLICATIONS
                    ┌───────────┴───────────┐
                    │                       │
                gRPC API                HTTP API
                 :9090                    :8080
                    │                       │
                    └───────────┬───────────┘
                                │
                         API Translation
                                │
                         Canonical Request
                                │
                      Authentication / Policy
                                │
                        Model Route Resolver
                                │
                      Capability Dispatcher
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
          Text/Vision         Images           Embeddings
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                              LiteLLM
                                │
                         Upstream APIs
```

All HTTP and gRPC operations must reuse common application services rather than duplicate inference logic.

---

# 4. Public HTTP API

The gateway must expose the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness |
| GET | `/v1/models` | List configured models |
| GET | `/v1/models/{model_id}` | Get model metadata |
| POST | `/v1/chat/completions` | Chat/text generation |
| POST | `/v1/completions` | Legacy text completion |
| POST | `/v1/responses` | Unified response generation |
| POST | `/v1/images/generations` | Image generation |
| POST | `/v1/images/edits` | Image editing |
| POST | `/v1/embeddings` | Text embeddings |

The inference endpoints should follow their commonly used OpenAI-compatible response shapes where applicable.

## 4.1 Health

`GET /healthz`

Returns HTTP 200 with:

```text
ok
```

Liveness must not depend on external provider availability.

## 4.2 Readiness

`GET /readyz`

Returns HTTP 200 when configuration is valid and the gateway can accept requests.

Returns HTTP 503 when mandatory internal dependencies are unavailable.

Provider outages should not automatically make the entire gateway unready when other configured routes remain usable.

## 4.3 Model Listing

`GET /v1/models`

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "text-default",
      "object": "model",
      "created": 0,
      "owned_by": "gateway",
      "context_window": 1048576,
      "max_output_tokens": 131072,
      "capabilities": ["chat", "tools", "vision"]
    },
    {
      "id": "embedding-default",
      "object": "model",
      "created": 0,
      "owned_by": "gateway",
      "capabilities": ["embeddings"],
      "dimensions": 1024
    }
  ]
}
```

The gateway must not expose upstream API keys, internal URLs, or other sensitive configuration.

## 4.4 Chat Completions

`POST /v1/chat/completions`

Request:

```json
{
  "model": "text-default",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain recursion."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

Response:

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "created": 0,
  "model": "text-default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Recursion is a technique..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 42,
    "total_tokens": 67
  }
}
```

The endpoint must support system, developer, user, assistant, and tool messages where the selected provider supports their semantics.

## 4.5 Responses API

`POST /v1/responses`

Supports:

- `model`
- `input`
- `instructions`
- `tools`
- `temperature`
- `max_output_tokens`
- `stream`

The gateway must normalize requests and responses into the Responses API representation.

A Responses-compatible upstream is not mandatory; a different LiteLLM provider may serve this endpoint when the requested semantics can be preserved.

Unsupported features must produce explicit errors rather than silently disappearing.

## 4.6 Images

`POST /v1/images/generations`

Example:

```json
{
  "model": "image-default",
  "prompt": "A futuristic city at sunset",
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

Response:

```json
{
  "created": 0,
  "data": [
    {
      "b64_json": "<base64-encoded-image>"
    }
  ]
}
```

`POST /v1/images/edits` must accept image inputs using multipart form data or another explicitly documented supported representation.

Image editing must be rejected with a capability error if the configured provider supports generation but not editing.

## 4.7 Embeddings

`POST /v1/embeddings`

```json
{
  "model": "embedding-default",
  "input": [
    "First document",
    "Second document"
  ]
}
```

Response:

```json
{
  "object": "list",
  "model": "embedding-default",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.12, -0.34, 0.56]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

The gateway must preserve input ordering and verify output dimensionality.

---

# 5. gRPC API

Default port: `9090`.

The gateway must define its own versioned protobuf contract.

Package:

```protobuf
package gateway.v1;
```

Service:

```protobuf
service ModelGatewayService {
  rpc Invoke(InvokeRequest) returns (InvokeResponse);
  rpc InvokeStream(InvokeRequest) returns (stream InvokeChunk);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}
```

The service must also implement the standard gRPC Health Checking Protocol.

## 5.1 InvokeRequest

The canonical gRPC request must represent:

| Field | Type | Purpose |
|---|---|---|
| `invocation_id` | string | Unique invocation identifier |
| `tenant_id` | string | Optional tenant identity |
| `actor_id` | string | Optional caller identity |
| `agent_id` | string | Agent/service identifier |
| `model` | string | Logical model alias |
| `messages` | repeated message | Conversation |
| `system_prompt` | string | System instruction |
| `generation_config` | structured object | Model parameters |
| `tools` | repeated message | Tool declarations |
| `response_modality` | enum | Text, image, multimodal |
| `fallback_models` | repeated string | Allowed fallback routes |
| `idempotency_key` | string | Request deduplication |
| `metadata` | map | Non-sensitive request metadata |

The protobuf schema must assign stable field numbers and preserve backward compatibility after publication.

## 5.2 InvokeResponse

Required concepts:

- Invocation ID.
- Logical model ID.
- Resolved upstream model ID.
- Text content.
- Tool calls.
- Optional image content.
- Finish reason.
- Token usage.
- Estimated cost.
- Provider attempt count.
- Whether a fallback was used.
- Idempotency status.

## 5.3 Streaming

`InvokeStream` must use server-side streaming.

Chunks must include:

- Invocation ID.
- Sequence number.
- Content delta or tool-call delta.
- Optional usage information.
- Terminal finish reason.
- Error details where appropriate.

Streaming is optional at the upstream level only if the gateway can explicitly reject it. The gateway must not pretend a non-streaming provider response is token-by-token streaming.

## 5.4 Error codes

Use standard gRPC status codes:

- `INVALID_ARGUMENT`
- `UNAUTHENTICATED`
- `PERMISSION_DENIED`
- `NOT_FOUND`
- `RESOURCE_EXHAUSTED`
- `FAILED_PRECONDITION`
- `DEADLINE_EXCEEDED`
- `UNAVAILABLE`
- `INTERNAL`

Error messages must be sanitized and contain stable machine-readable reasons.

---

# 6. Environment Configuration

Environment variables are the authoritative deployment configuration.

All variable names are uppercase and use underscores.

Configuration is loaded and validated at startup.

An absent optional capability is disabled rather than preventing unrelated capabilities from operating.

## 6.1 Text generation

```dotenv
TEXT_LLM_ENDPOINT=https://api.openai.com/v1
TEXT_LLM_MODEL_ID=gpt-sunburst
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_PATH=
TEXT_LLM_EXTRA_HEADERS_JSON={}
TEXT_LLM_CAPABILITIES=chat,tools,vision
TEXT_LLM_CONTEXT_WINDOW=
TEXT_LLM_MAX_OUTPUT_TOKENS=
```

`TEXT_LLM_MODEL_ID` is the bare upstream model name; the gateway composes `<format>/<model_id>` when calling LiteLLM.

Supported initial formats (LiteLLM provider names):

| Format | Provider |
|---|---|
| `openai` | OpenAI and OpenAI-compatible endpoints |
| `anthropic` | Anthropic Messages API |
| `vertex_ai` | Vertex AI |
| `gemini` | Gemini API |
| `ollama` | Ollama |
| `azure` | Azure OpenAI |

The format is a LiteLLM provider name identifying the upstream API dialect, not the company operating the endpoint.

## 6.2 Image generation

```dotenv
IMAGE_GEN_ENDPOINT=
IMAGE_GEN_MODEL_ID=
IMAGE_GEN_API_KEY=
IMAGE_GEN_FORMAT=openai
IMAGE_GEN_AUTH_TYPE=bearer
IMAGE_GEN_PATH=
IMAGE_GEN_EXTRA_HEADERS_JSON={}
IMAGE_GEN_CAPABILITIES=generation
```

`IMAGE_GEN_FORMAT` is a LiteLLM provider name, and `IMAGE_GEN_MODEL_ID` is the bare upstream model name.

Possible image formats:

- `openai`
- `vertex_ai`
- `gemini`
- `azure`

Image editing is enabled only when explicitly supported.

## 6.3 Embeddings

```dotenv
EMBEDDINGS_ENDPOINT=
EMBEDDINGS_MODEL_ID=
EMBEDDINGS_API_KEY=
EMBEDDINGS_FORMAT=openai
EMBEDDINGS_AUTH_TYPE=bearer
EMBEDDINGS_PATH=
EMBEDDINGS_DIMENSIONS=
EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM=false
EMBEDDINGS_EXTRA_HEADERS_JSON={}
```

`EMBEDDINGS_FORMAT` is a LiteLLM provider name (`openai`, `vertex_ai`, `gemini`, `azure`, `ollama`), and `EMBEDDINGS_MODEL_ID` is the bare upstream model name.

The gateway must support both fixed-dimension and configurable-dimension models.

---

# 7. Authentication

Supported upstream authentication mechanisms:

| Value | Behavior |
|---|---|
| `bearer` | Authorization bearer token |
| `x-api-key` | `x-api-key` header |
| `api-key` | `api-key` header |
| `none` | No upstream authentication |
| `custom-header` | Configurable authentication header |

For `vertex_ai` routes, upstream Google credentials are resolved by LiteLLM through Google Application Default Credentials, and the gateway passes no static key. ADC is ambient: on GCE or GKE the workload's attached service account is inherited automatically, so operators must scope that service account deliberately.

For example:

```dotenv
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_API_KEY=example-secret
```

Produces:

```http
Authorization: Bearer example-secret
```

The gateway must separately configure inbound client authentication. Upstream credentials must never be accepted from ordinary inference request bodies.

Recommended inbound configuration:

```dotenv
GATEWAY_AUTH_ENABLED=true
GATEWAY_API_KEYS=
GATEWAY_ALLOW_ANONYMOUS_HEALTH=true
```

Production must reject unauthenticated inference requests when authentication is enabled.

---

# 8. Provider URL Resolution

The gateway must support both plain base URLs and base URLs that include an explicit path.

Example:

```dotenv
TEXT_LLM_ENDPOINT=https://provider.example/v1
TEXT_LLM_FORMAT=openai
```

Resolved base URL handed to LiteLLM:

```text
https://provider.example/v1
```

With an explicit path:

```dotenv
TEXT_LLM_PATH=/custom
```

Resolved base URL handed to LiteLLM:

```text
https://provider.example/custom
```

A full URL in `TEXT_LLM_PATH` may override the base URL when explicitly permitted by configuration.

URL joining must be deterministic.

The gateway must not duplicate `/v1`, append the same path component twice, or silently strip meaningful path components.

LiteLLM appends operation paths (for example `/chat/completions`) to the resolved base URL; those operation paths are LiteLLM's responsibility, not the gateway's.

---

# 9. LiteLLM Binding

The gateway binds to LiteLLM as its vendor-neutral inference layer. Each configured route supplies a LiteLLM provider name, an upstream model name, a base URL, and an API key, and the gateway calls LiteLLM with a provider-prefixed model string.

LiteLLM owns provider-specific request encoding, network execution, response decoding, error normalization, and translation for text, vision, tool calling, image generation, and embeddings. This satisfies the extensibility principle: additional upstream providers are supported by LiteLLM without changing the gateway's core business logic.

The binding must not own:

- Tenant authorization.
- Global routing.
- Billing policy.
- Request idempotency.
- Global retry policy.
- Audit persistence.

These responsibilities belong to the gateway core.

## 9.1 Tool calling

The canonical representation must support:

- Function name.
- JSON-schema parameters.
- Tool-call identifier.
- Tool arguments.
- Tool result.
- Tool-choice policy.

LiteLLM translates tool definitions and calls to and from the upstream provider's format.

**The gateway must not execute arbitrary model-requested tools.** The caller or an explicitly registered trusted tool executor remains responsible for execution.

## 9.2 Vision

The canonical content representation must support text and images in the same message.

Image inputs may be supplied as:

- Base64 content.
- Authorized remote URLs.
- Supported binary payloads.

Vision content is passed through to LiteLLM, which handles provider-specific encoding.

Remote image fetching must follow SSRF and size-limit protections.

## 9.3 Unsupported features

When a provider does not support a requested capability, the gateway must return a structured capability error.

It must not silently discard tools, images, system instructions, structured-output requirements, or other semantically important parameters.

---

# 10. Model Routing

The gateway must support logical model aliases.

Fixed logical model IDs:

```text
text-default
image-default
embedding-default
```

Each logical model ID maps to a configured capability route.

An unknown model alias must return a model-not-found error.

Caller-supplied model names must not allow access to unconfigured upstream endpoints.

---

# 11. Multiple Providers and Fallbacks

The gateway should support multiple routes for a capability.

Example:

```dotenv
TEXT_LLM_1_ENDPOINT=
TEXT_LLM_1_MODEL_ID=
TEXT_LLM_1_API_KEY=
TEXT_LLM_1_FORMAT=

TEXT_LLM_2_ENDPOINT=
TEXT_LLM_2_MODEL_ID=
TEXT_LLM_2_API_KEY=
TEXT_LLM_2_FORMAT=
```

A structured alternative is:

```dotenv
TEXT_LLM_ROUTES_JSON=[]
```

If both indexed and structured route configuration are supplied, startup must reject the ambiguity unless an explicit precedence rule is documented.

Fallback behavior:

```text
Primary
   ↓ retryable failure
Primary retry
   ↓ exhausted
Fallback 1
   ↓ failure
Fallback 2
```

Retries may be triggered by:

- Connection failures.
- Timeouts.
- HTTP 429.
- HTTP 500–599.
- Equivalent transient provider errors.

Retries must be bounded.

Requests that have already produced visible streaming output must not silently switch providers.

LiteLLM exposes its own fallback support. The gateway may use it or keep fallback selection in its core; either way, the deterministic retry and fallback semantics defined in this section govern.

---

# 12. Idempotency

Every invocation may carry an idempotency key.

The gateway must prevent duplicate accounting for equivalent repeated requests.

Idempotency records must include a request fingerprint to prevent accidental reuse of the same key for different payloads.

In-memory idempotency is sufficient for single-instance deployments.

Durable or distributed idempotency requires configured persistence.

The gateway must not claim exactly-once upstream execution when the provider lacks idempotency support and a timeout leaves execution outcome unknown.

---

# 13. Usage and Cost Accounting

Record:

- Invocation ID.
- Timestamp.
- Route and logical model.
- Actual upstream model.
- Capability.
- Input tokens.
- Output tokens.
- Cached tokens when reported.
- Provider latency.
- Request status.
- Retry/fallback count.
- Estimated cost.

Token counts must distinguish provider-reported values from estimates.

Cost estimation must use configured pricing data, not hardcoded vendor-name assumptions.

Example:

```dotenv
USAGE_TRACKING_ENABLED=true
COST_TRACKING_ENABLED=true
USAGE_STORAGE=memory
```

Supported storage modes may include:

- `none`
- `memory`
- `sqlite`
- `postgres`

The default standalone deployment must not require an external database.

---

# 14. Security

The gateway must implement:

- TLS support.
- Inbound authentication.
- Secret redaction.
- Request-body size limits.
- Provider timeouts.
- SSRF protection.
- Redirect restrictions.
- Host and scheme validation.
- Optional upstream hostname allowlists.
- Safe error normalization.
- Tenant-aware authorization where enabled.
- Per-client rate limits.
- Per-tenant concurrency limits.

Development may permit plain HTTP and unauthenticated local operation when explicitly configured.

Production must not silently fall back to insecure behavior.

---

# 15. Reliability

Recommended environment variables:

```dotenv
PROVIDER_TIMEOUT_SECONDS=110
PROVIDER_CONNECT_TIMEOUT_SECONDS=10
PROVIDER_RETRY_COUNT=2
PROVIDER_RETRY_BACKOFF_MS=500
PROVIDER_MAX_CONCURRENT_REQUESTS=100
GATEWAY_SHUTDOWN_TIMEOUT_SECONDS=30
```

The gateway must:

- Propagate cancellation.
- Enforce request deadlines.
- Bound retry duration.
- Respect upstream rate limits.
- Shut down gracefully.
- Avoid leaking connections.
- Prevent unbounded request queues.
- Return deterministic error categories.

A retry must not exceed the caller's remaining deadline.

---

# 16. Streaming

HTTP streaming must use Server-Sent Events where the endpoint's protocol defines SSE.

gRPC streaming must use server-streaming responses.

Streaming through LiteLLM must support:

- Incremental text.
- Tool-call argument deltas.
- Completion status.
- Cancellation.
- Terminal errors.
- Usage reporting where available.

The gateway must preserve event ordering.

If an upstream cannot stream, the gateway may reject the streaming request with a clear unsupported-capability error.

---

# 17. Observability

Required metrics:

```text
gateway_requests_total
gateway_request_duration_seconds
gateway_upstream_requests_total
gateway_upstream_duration_seconds
gateway_tokens_input_total
gateway_tokens_output_total
gateway_retries_total
gateway_fallbacks_total
gateway_errors_total
gateway_active_requests
gateway_rate_limit_rejections_total
```

Distributed tracing should support W3C Trace Context and OpenTelemetry.

Logs must be structured.

Prompts, responses, credentials, and private tool arguments must not be logged by default.

---

# 18. Complete Environment Example

```dotenv
# ======================================
# SERVER
# ======================================
CHORA_HTTP_PORT=8080
CHORA_GRPC_PORT=9090
CHORA_ENVIRONMENT=production

# ======================================
# TEXT GENERATION
# ======================================
TEXT_LLM_ENDPOINT=https://api.example.com/v1
TEXT_LLM_MODEL_ID=text-model
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_PATH=
TEXT_LLM_EXTRA_HEADERS_JSON={}
TEXT_LLM_CAPABILITIES=chat,tools,vision
TEXT_LLM_CONTEXT_WINDOW=
TEXT_LLM_MAX_OUTPUT_TOKENS=

# ======================================
# IMAGE GENERATION
# ======================================
IMAGE_GEN_ENDPOINT=https://images.example.com/v1
IMAGE_GEN_MODEL_ID=image-model
IMAGE_GEN_API_KEY=
IMAGE_GEN_FORMAT=openai
IMAGE_GEN_AUTH_TYPE=bearer
IMAGE_GEN_PATH=
IMAGE_GEN_EXTRA_HEADERS_JSON={}
IMAGE_GEN_CAPABILITIES=generation

# ======================================
# EMBEDDINGS
# ======================================
EMBEDDINGS_ENDPOINT=https://embeddings.example.com/v1
EMBEDDINGS_MODEL_ID=embedding-model
EMBEDDINGS_API_KEY=
EMBEDDINGS_FORMAT=openai
EMBEDDINGS_AUTH_TYPE=bearer
EMBEDDINGS_PATH=
EMBEDDINGS_DIMENSIONS=1024
EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM=false

# ======================================
# AUTHENTICATION
# ======================================
GATEWAY_AUTH_ENABLED=true
GATEWAY_API_KEYS=

# ======================================
# RELIABILITY
# ======================================
PROVIDER_TIMEOUT_SECONDS=110
PROVIDER_CONNECT_TIMEOUT_SECONDS=10
PROVIDER_RETRY_COUNT=2
PROVIDER_RETRY_BACKOFF_MS=500
PROVIDER_MAX_CONCURRENT_REQUESTS=100

# ======================================
# USAGE
# ======================================
USAGE_TRACKING_ENABLED=true
COST_TRACKING_ENABLED=false
USAGE_STORAGE=memory

# ======================================
# OBSERVABILITY
# ======================================
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=
LOG_LEVEL=info
```

---

# 19. Deployment

The gateway must support running as a single standalone process.

Container deployment must expose:

```text
8080/tcp  HTTP
9090/tcp  gRPC
```

The process must:

1. Load environment configuration.
2. Validate enabled capabilities.
3. Initialize the LiteLLM binding.
4. Initialize optional persistence.
5. Initialize observability.
6. Start HTTP and gRPC listeners.
7. Publish readiness.
8. Handle shutdown signals gracefully.

The service must not require Kubernetes, a service mesh, or a cloud-managed runtime.

---

# 20. Testing Requirements

## Unit tests

Cover:

- Configuration parsing.
- Environment validation.
- URL resolution.
- Authentication headers.
- Request normalization.
- Response normalization.
- Model aliases.
- Retry classification.
- Error mapping.
- Tool-call translation.
- Embedding dimensions.

## Integration tests

Use deterministic mock upstream servers implementing each supported LiteLLM provider protocol.

Test:

- OpenAI.
- Anthropic.
- Vertex.
- Gemini.
- Ollama.
- Images.
- Embeddings.
- Streaming.
- Provider failures.
- Cancellation.
- Authentication.
- Timeouts.
- Fallbacks.

## End-to-end tests

Start the gateway with environment variables and mock providers.

Verify every HTTP and gRPC endpoint without requiring a real commercial provider account.

At least one optional live-provider smoke test should demonstrate switching providers through environment configuration alone.

---

# 21. Acceptance Criteria

The product is accepted when:

1. The gateway starts independently without external application services.
2. Text inference works through at least four configured LiteLLM providers.
3. Text providers can be switched without recompilation.
4. Image generation uses independently configured credentials and endpoints.
5. Embeddings use independently configured credentials and endpoints.
6. HTTP and gRPC share canonical application logic.
7. Tool calls round-trip without losing arguments or identifiers.
8. Multimodal inputs preserve supported content.
9. Unsupported capabilities produce explicit errors.
10. Streaming works for compatible providers.
11. Timeouts, retries, and fallbacks behave deterministically.
12. No provider credentials are exposed through public APIs.
13. Unknown model aliases cannot access arbitrary upstream endpoints.
14. Usage accounting distinguishes actual and estimated token counts.
15. The service can run without a database.
16. Health and readiness checks function independently.
17. Configuration errors are detected at startup.
18. Automated tests cover all public endpoints.
19. The implementation contains no vendor-specific business logic outside the LiteLLM binding.

---

# 22. Implementation Milestones

**Phase 1 — Core gateway**

- Configuration loader.
- HTTP and gRPC servers.
- Canonical request/response types.
- Model routing.
- LiteLLM binding for text inference.
- Authentication and health checks.

**Phase 2 — Multi-provider support**

- Additional LiteLLM providers (Anthropic, Vertex, Gemini, Ollama, Azure).
- Tool calling.
- Vision support.

**Phase 3 — Additional capabilities**

- Image generation.
- Image editing.
- Embeddings.

**Phase 4 — Production hardening**

- Streaming.
- Retries and fallbacks.
- Rate limiting.
- Concurrency limits.
- Usage and cost accounting.
- Metrics and tracing.
- Security testing.

**Phase 5 — Conformance and release**

- Protocol conformance suites.
- Mock-provider integration tests.
- Deployment documentation.
- Example environment files.
- Container image.
- Versioned API and protobuf contracts.

---

# 23. Final Architectural Requirements

The gateway must maintain three independent layers.

**Northbound interfaces**

Stable HTTP and gRPC contracts consumed by applications.

**Gateway core**

Vendor-independent routing, validation, governance, reliability, and accounting.

**LiteLLM binding**

Protocol-specific request encoding, authentication, network execution, and response decoding, delegated to the LiteLLM library.

No commercial vendor identity may be required by the gateway core.

A provider change must normally require only environment configuration changes.

The gateway must be usable as an independent infrastructure component, regardless of which application, programming language, deployment platform, or AI provider is connected to it.