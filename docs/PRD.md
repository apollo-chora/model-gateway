# Product Requirements Document
# Vendor-Neutral AI Model Gateway

**Version:** 1.4  
**Status:** Implementation-ready specification  
**Implementation language:** Python  
**Architecture:** Standalone, self-hostable, multi-provider AI gateway  
**Protocols:** gRPC, HTTP/JSON  
**Configuration:** Environment variables  
**Primary deployment:** Containerized service  
**Supported capabilities:** Text generation, chat, tool calling, vision, image generation, embeddings

**Revision history:**

- **1.4** — Documents additive path joining in §8 (`*_PATH` is appended to the base path and never replaces it, so a provider prefix is never dropped) and sets `OTEL_ENABLED=false` in the §18 example to match the shipped environment templates.
- **1.3** — Corrections from architecture review: additional routes now carry their own authentication configuration plus only the credentials that configuration requires; drops the impossible duplicate-route-index rejection while keeping duplicate route identity and identical endpoint+model rejections; allows `*_EXTRA_HEADERS_JSON` on any route and replaces the unenforceable credential-injection wording with a reserved-authentication-header rule enforced at startup; tightens the §9 extensibility wording to "can be added through LiteLLM".
- **1.2** — Makes the LiteLLM boundary and ownership split explicit; collapses multi-route configuration to one schema (a singleton primary block plus `<CAPABILITY>_ROUTES_JSON`); assigns retry and fallback ownership solely to the gateway core with LiteLLM retry and fallback disabled; restricts an absolute-URL override in `*_PATH` to explicit opt-in.
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
8. **Extensibility:** Additional upstream providers can be added through LiteLLM without modifying the gateway's core business logic.

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
- Fixed logical model IDs with explicitly configured route fallback.
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
| `model` | string | Public model ID |
| `messages` | repeated message | Conversation |
| `system_prompt` | string | System instruction |
| `generation_config` | structured object | Model parameters |
| `tools` | repeated message | Tool declarations |
| `response_modality` | enum | Text, image, multimodal |
| `fallback_models` | repeated string | Permitted fallback routes |
| `idempotency_key` | string | Request deduplication |
| `metadata` | map | Non-sensitive request metadata |

The protobuf schema must assign stable field numbers and preserve backward compatibility after publication.

## 5.2 InvokeResponse

Required concepts:

- Invocation ID.
- Public model ID.
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

`*_FORMAT` keeps its name in every capability block. Its value is the provider dialect name understood by the inference binding — a LiteLLM provider name such as `openai` or `vertex_ai`, not the company operating the endpoint. The operation performed (chat, embedding, image generation) is determined by which capability block the variable belongs to, not by the variable name: `TEXT_LLM_FORMAT` selects the dialect for chat and text, `EMBEDDINGS_FORMAT` for embeddings, and `IMAGE_GEN_FORMAT` for image generation.

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

`TEXT_LLM_FORMAT` is the provider dialect name for chat and text understood by the inference binding (a LiteLLM provider name), not the company operating the endpoint.

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

`IMAGE_GEN_FORMAT` is the provider dialect name for image generation, and `IMAGE_GEN_MODEL_ID` is the bare upstream model name.

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

`EMBEDDINGS_FORMAT` is the provider dialect name for embeddings (`openai`, `vertex_ai`, `gemini`, `azure`, `ollama`), and `EMBEDDINGS_MODEL_ID` is the bare upstream model name.

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

The gateway decides which credential configuration applies to a route and supplies it to the binding; LiteLLM builds the provider's native authentication header from what it is given. Precedence:

1. **Native provider authentication is the default.** The gateway supplies `api_key` for the route and normally lets LiteLLM construct the provider's native header — for example the OpenAI-compatible `Authorization: Bearer` header.
2. **`*_AUTH_TYPE` applies only when a custom endpoint needs a non-native header.** It exists for endpoints that are not the provider's native endpoint and require `x-api-key`, `api-key`, or a named custom header. A value that matches the provider's native scheme (for example `bearer` for an OpenAI-compatible endpoint) is a no-op and equivalent to leaving the variable unset.
3. **`*_EXTRA_HEADERS_JSON` may be used on any route.** It carries non-authentication headers a route legitimately needs — provider routing headers, organisation or project identifiers, proxy-required headers, and feature or version headers — and is independent of the route's authentication configuration.

Extra headers merge with the headers LiteLLM generates for the request. The gateway's configured headers are applied last, so a configured header name wins over a LiteLLM-generated header with the same name, and LiteLLM's remaining generated headers are preserved.

Reserved authentication headers must not be supplied through `*_EXTRA_HEADERS_JSON`: `Authorization`, `x-api-key`, `api-key`, and any header the route's own authentication configuration owns. Supplying a reserved header requires an explicit custom-auth configuration for the route; otherwise startup must reject the configuration. The reserved-header rule is the enforceable form of the constraint, because the headers a route's authentication owns follow from its `*_AUTH_TYPE` and credential configuration and can therefore be checked at startup, whereas "must not inject a credential" describes an intent that no configuration check can decide.

Credentials are bound to one origin. The gateway must never forward a route's credentials to a different origin, and must never re-send them across a redirect that changes scheme, host, or port (§14).

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

Resolved base URL handed to the inference binding:

```text
https://provider.example/v1
```

With an explicit path:

```dotenv
TEXT_LLM_PATH=/custom/generate
```

Resolved base URL handed to the inference binding:

```text
https://provider.example/v1/custom/generate
```

A relative path in `*_PATH` is appended to the base path and never replaces it, so a configured provider prefix such as `/v1` is never silently dropped. That is what satisfies the prohibition below on silently stripping meaningful path components, and it is what makes `*_PATH` useful now that the operation path is appended by the inference binding.

A full URL in `TEXT_LLM_PATH` must not override the base URL by default. Deterministic joining governs: the value is treated as a path and appended to the base URL. An absolute-URL override requires an explicit opt-in for the route, for example `TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL=true`; without that opt-in, an absolute URL in `*_PATH` is invalid configuration and startup must reject it rather than resolve it silently.

URL joining must be deterministic.

The gateway must not duplicate `/v1`, append the same path component twice, or silently strip meaningful path components.

The resolved value is handed to the inference binding as its base URL. LiteLLM appends operation paths (for example `/chat/completions`) to that base URL; those operation paths are LiteLLM's responsibility, not the gateway's.

---

# 9. LiteLLM Binding

The gateway binds to LiteLLM as its vendor-neutral inference layer. Each configured route supplies a LiteLLM provider name, an upstream model name, a base URL, and an API key, and the gateway calls LiteLLM with a provider-prefixed model string.

LiteLLM owns provider-specific request encoding, network execution, and response decoding for text, vision, tool calling, image generation, and embeddings. That is the whole of the boundary: **LiteLLM owns upstream protocol translation and execution, and nothing else.**

The gateway owns policy, lifecycle, validation, identity, reliability policy, and the public contract. It also re-normalises LiteLLM's output at its own boundary: a LiteLLM result is treated as upstream data, mapped into the canonical representation, and checked against the fields the gateway depends on, never forwarded to callers unchanged.

In particular, LiteLLM does not own error normalisation or authentication decisions. It may raise provider-shaped exceptions, and it may construct a provider-native authentication header from credentials the gateway supplies (§7), but the gateway decides which error category a caller sees and which credential configuration applies to a route.

This satisfies the extensibility principle: additional upstream providers can be added through LiteLLM without changing the gateway's core business logic.

The binding must not own:

- Tenant authorization.
- Global routing.
- Billing policy.
- Request idempotency.
- Global retry policy.
- Audit persistence.
- Error normalisation.
- Authentication decisions.
- The public response contract.

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

## 9.4 Route-to-invocation mapping

Each capability resolves to exactly one LiteLLM entry point. The gateway passes the resolved base URL as the endpoint, the route's API key, and a provider-prefixed model string (`<format>/<model_id>`). The endpoint is always passed as a base URL; operation paths are appended by LiteLLM (§8).

| Capability | LiteLLM entry point | Parameters passed |
|---|---|---|
| Chat / text | `litellm.completion` (streaming: `stream=True`; async: `litellm.acompletion`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `messages`, generation parameters, `timeout` drawn from the invocation deadline, `num_retries=0` |
| Embeddings | `litellm.embedding` (async: `litellm.aembedding`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `input`, `dimensions` only when the route enables it, `num_retries=0` |
| Image generation | `litellm.image_generation` (async: `litellm.aimage_generation`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `prompt`, only the image parameters the route advertises (§9.5), `num_retries=0` |

Retries are disabled on every call because the gateway core owns the retry loop (§11.3).

**"Supported in this gateway" is a narrower claim than "supported somewhere in LiteLLM".** A provider appearing in LiteLLM's catalogue is not acceptance evidence for this gateway. A provider is supported here only when a configured route names it *and* the gateway's own tests cover its operation, custom endpoint, authentication, and response semantics. LiteLLM may know how to call a provider this gateway has never validated, and the gateway must not advertise or route to one.

## 9.5 Image generation constraints

The supported upstream operation is `POST {endpoint}/images/generations`. A route advertises image generation only when that operation is what its endpoint provides.

Image support must not be overstated. For an image model that LiteLLM does not recognise, passing `response_format` (and possibly `size`) makes LiteLLM raise `UnsupportedParamsError` instead of ignoring the parameter. Routes for such models must therefore drop those parameters before the call, or not advertise them at all; the gateway must not promise OpenAI-style image parameters it cannot pass through.

---

# 10. Model Routing

The gateway exposes a fixed set of public model IDs. They are part of the public contract and are not configurable aliases:

```text
text-default
image-default
embedding-default
```

Each public model ID maps to a configured capability route and to that route's configured fallbacks (§11).

An unknown public model ID must return a model-not-found error.

Caller-supplied model names must not allow access to unconfigured upstream endpoints.

The word "alias" is reserved for configurable mappings. The gateway defines none: a public model ID is a fixed identifier, and callers cannot introduce, rename, or remap one.

---

# 11. Multiple Providers and Fallbacks

The gateway must support multiple routes for a capability. Fallbacks are required: a capability may declare a primary route and any number of additional routes.

## 11.1 Configuration schema

There is exactly one schema. For each capability, the singleton block defines the primary route and a `<CAPABILITY>_ROUTES_JSON` variable supplies the additional routes as a JSON array.

```dotenv
TEXT_LLM_ENDPOINT=https://primary.example/v1
TEXT_LLM_MODEL_ID=primary-model
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai

TEXT_LLM_ROUTES_JSON=[{"endpoint":"https://fallback.example/v1","model_id":"fallback-model","api_key":"","format":"openai"}]
```

The same pattern applies to every capability: `IMAGE_GEN_*` with `IMAGE_GEN_ROUTES_JSON`, and `EMBEDDINGS_*` with `EMBEDDINGS_ROUTES_JSON`.

How the singleton block and `*_ROUTES_JSON` combine:

- The singleton block always defines the primary route (index 0). It is the only place the primary route is defined.
- `*_ROUTES_JSON` entries are additional routes, appended in array order (index 1, 2, ...).
- An absent or empty array means the capability has exactly one route.
- Precedence is positional, not field-merged. An entry never overrides a field of the singleton block, and no field is inherited from one route to another: each entry is self-contained and must specify its own endpoint, model ID, format, and authentication configuration, plus any credentials that configuration requires. Credentials are never carried across routes, because a fallback route may point at a different origin (§7).
- A capability is enabled only when its singleton block is configured. `*_ROUTES_JSON` alone, with the singleton block unset, is ambiguous and must be rejected at startup.

The indexed form (`TEXT_LLM_1_*`, `TEXT_LLM_2_*`) does not exist. It is not a supported configuration, and its presence must be rejected at startup.

Ambiguous configuration must be rejected at startup rather than resolved by guessing. This includes a malformed JSON array, an entry missing a required field, duplicate route identity (two entries presenting themselves as the same route), and two entries naming the same endpoint and model.

## 11.2 Fallback selection

Route order is fallback order. The gateway walks the configured routes in order, retrying the current route within its retry budget and moving to the next route only when that route is exhausted or fails with a retryable error.

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

## 11.3 Retry and fallback ownership

The gateway core owns retries and fallbacks. LiteLLM's own retry and fallback support is disabled, so that there is exactly one retry loop, one deadline, and one attempt count for an invocation.

- LiteLLM is called with retries disabled (`num_retries=0`). The gateway must not rely on LiteLLM's `fallbacks` or `context_window_fallbacks` parameters.
- The gateway enforces a single total deadline per invocation (§15). Provider timeouts and connect timeouts are budgeted inside that deadline, and no retry or fallback may extend it.
- Retries must never switch providers after visible streaming output has been emitted to the caller.

Stacked retry loops are prohibited: neither the gateway core nor the LiteLLM binding may wrap a second retry layer around a call that the other already governs.

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
- Route and public model ID.
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

The gateway enforces a single total deadline per invocation. Provider timeouts, connect timeouts, retries, and fallbacks all draw from that one budget. LiteLLM's retry and fallback support is disabled so that no second loop and no second deadline exists (§11.3).

---

# 16. Streaming

HTTP streaming must use Server-Sent Events where the endpoint's protocol defines SSE.

gRPC streaming must use server-streaming responses.

Streaming is delivered through the LiteLLM binding. The gateway maintains no per-provider streaming adapters: it consumes the stream the binding produces for the configured provider and re-normalises each chunk at the gateway boundary.

Streaming through the LiteLLM binding must support:

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
TEXT_LLM_ROUTES_JSON=[]

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
IMAGE_GEN_ROUTES_JSON=[]

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
EMBEDDINGS_ROUTES_JSON=[]

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
OTEL_ENABLED=false
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
- Public model ID resolution.
- Retry classification.
- Error mapping.
- Tool-call translation.
- Embedding dimensions.

## Integration tests

Use deterministic mock upstream servers that implement the provider protocol the binding speaks for each configured format. The mocks stand in for the upstream provider, never for the gateway: the gateway itself maintains no per-provider protocol implementation.

The public gateway contract must stay covered. Mock providers substitute for the upstream only, and the gateway's own HTTP and gRPC surface is exercised against them.

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
13. Unknown public model IDs cannot reach unconfigured upstream endpoints.
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

Protocol-specific request encoding, network execution, and response decoding, delegated to the LiteLLM library.

Vendor neutrality is a property of the gateway, not of its inference dependency. The gateway core holds no provider-specific business logic, and the provider is chosen entirely by configuration. That neutrality is bounded by the LiteLLM dependency: the gateway is vendor-neutral to the extent that LiteLLM can speak the target provider's protocol. The dependency is pinned, isolated behind the binding, and replaceable — swapping it changes the binding, not the gateway's public contract, routing, policy, or accounting.

No commercial vendor identity may be required by the gateway core.

A provider change must normally require only environment configuration changes.

The gateway must be usable as an independent infrastructure component, regardless of which application, programming language, deployment platform, or AI provider is connected to it.