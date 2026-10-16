# model-gateway

A vendor-neutral gateway in front of AI model providers. It exposes one stable
OpenAI-compatible HTTP surface and one gRPC service, and it routes each request to
whichever upstream the deployment configured — so an application talks to the
gateway, and which provider answers is an environment change rather than a code
change.

```
application ──▶ model-gateway ──▶ provider A
   HTTP /v1         routing,          provider B
   gRPC gateway.v1  policy,           provider C
                    accounting
```

## What the gateway owns, and what it delegates

Inference is performed by [LiteLLM], pinned and isolated behind one binding that
owns upstream protocol translation and nothing else. The gateway core owns
everything that is policy: resolving the operation a request arrived on to its
routes, authenticating
the caller and isolating tenants, enforcing the capabilities a route declared,
validating the request and bounding its size, the deadline and cancellation, the
retry and fallback walk, idempotency, usage and cost accounting, the public error
contract, network security, and what may be logged.

The library's own retry and fallback are disabled, so exactly one retry loop and
one deadline exist and both are the gateway's. That separation is what makes the
provider swappable: the core names no provider, and the binding is the only module
that does.

## Running it

```sh
docker build -t model-gateway:local .
docker run --rm -p 8080:8080 -p 9090:9090 --env-file .env model-gateway:local
```

A container with no configuration runs and serves an empty model list, and cannot
serve inference — a missing environment file does not look like a crash, and it does
not look like a working deployment either. A green `/healthz` says the process is the
process and nothing about whether providers are configured; check `/v1/models`.

Configuration is entirely environment variables: `TEXT_LLM_*` for the chat
operation, `IMAGE_GEN_*` for images, `EMBEDDINGS_*` for embeddings, `GATEWAY_*` for
inbound authentication and limits, `PROVIDER_*` for reliability. Each operation's
block names its own endpoint, model, credential, dialect and capabilities, so the
three are configured independently.

[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) covers running, probing and operating it,
and [docs/CONFIGURATION.md](docs/CONFIGURATION.md) is the complete list of settings.

## Endpoints

| Surface | Endpoints |
| --- | --- |
| HTTP | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, `POST /v1/images/generations` |
| Operations | `GET /healthz`, `GET /readyz`, `GET /metrics` |
| gRPC | `gateway.v1.ModelGatewayService`: `Invoke`, `InvokeStream`, `Embed`, `ListModels`, plus the standard health protocol |

Both transports reach identical policy: one inference service decides what may
happen, and each transport only translates. The gRPC package is versioned and its
field numbers are stable — see `proto/gateway/v1/gateway.proto`.

## Development

```sh
uv sync
uv run pytest -q
uv run ruff check src tests
```

The suite runs against deterministic stand-ins and opens no socket to any provider,
so it needs no provider account and no network. The code, the public API and the
protobuf schema are the contracts: the tests verify behaviour against them, and the
documentation explains intent. Where a document and the implementation disagree,
the document is revised rather than the disagreement left standing.

[LiteLLM]: https://github.com/BerriAI/litellm
