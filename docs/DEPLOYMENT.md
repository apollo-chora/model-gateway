# Deployment

The gateway is one process with two listeners: HTTP on `8080` and gRPC on `9090`,
both started and stopped together, configured entirely by environment variables. It
requires no Kubernetes, no service mesh and no cloud-managed runtime — a container
runtime and a set of environment variables are enough (PRD sections 18 and 19).

Nothing about a provider is baked into the image. Which upstreams it calls, over
which protocols, with which credentials, is configuration; changing a provider is
an environment change rather than a build (PRD section 23).

## Running the container

```sh
docker build -t model-gateway:local .
docker run --rm \
  -p 8080:8080 -p 9090:9090 \
  --env-file .env \
  model-gateway:local
```

Both images the build uses are pinned by digest, and both stages use the same base,
because a virtual environment records the interpreter it was built for. The runtime
image carries the virtual environment and not the source tree, and it runs as an
unprivileged user: the process writes nothing to its own filesystem.

`--env-file` takes the `KEY=VALUE` form the example files use, comments included.
Copy `.env.example` for development or `.env.production.example` for a real
deployment and fill in the credentials; PRD section 18 is the complete list of
settings with the defaults they take.

**A container started with no configuration at all is a working gateway with no
providers.** It starts, answers `/healthz` and `/readyz`, serves `/v1/models` with an
empty list, and refuses every inference request as a model it does not know. That is
the honest state of a deployment that has not said what it can reach, and it is
worth knowing because it means a missing environment file does not look like a
crash.

## What happens at startup

The process loads the environment, validates it, and refuses to start on a
configuration it cannot serve — a malformed route, an endpoint whose host resolves
to an address the URL policy forbids, a dialect the binding has not verified, an
image route advertising an editing capability that is not implemented, a malformed
price, a request-body bound that is not a positive whole number, tracing enabled
with nowhere to send spans. Each of those is a deployment that would otherwise fail
on a caller's request instead, so none of them is deferred to the first request
(PRD sections 6 and 21).

It then builds the model registry and the inference binding, wires the
authentication, the limiters, the accounting and the tracing the environment asked
for, starts both listeners, and publishes readiness. The gRPC listener is bound
before either starts, so a port it cannot take fails the start rather than
producing a process that answers a probe on a port it is not serving.

## Health, readiness and metrics

| Path | What it answers | Credential |
| --- | --- | --- |
| `GET /healthz` | the process is the process | none when `GATEWAY_ALLOW_ANONYMOUS_HEALTH=true` |
| `GET /readyz` | the process is ready to serve | same exemption |
| `GET /metrics` | the eleven counters and histograms of PRD section 17 | **required** |

Probe liveness on `/healthz` and route on `/readyz`; they are different questions.
`/metrics` is authenticated like every other path, because the only two exemptions
are the health endpoints and a scraper carries a credential the way any other
client does.

The image declares a `HEALTHCHECK` against `/healthz`, reading `CHORA_HTTP_PORT`
so a deployment that moved the port is still probed. It deliberately does not probe
readiness: a container that reports unhealthy because a provider is slow is a
container a runtime may kill for no reason.

## Shutdown

The process stops on `SIGTERM` or `SIGINT`. The application's own stop hooks run
before it exits: the gRPC listener is taken down and the tracer provider is shut
down, which flushes the span batch it is holding — a process that exits without that
loses the last batch, which is the one covering the shutdown an operator is most
likely to be reading. `GATEWAY_SHUTDOWN_TIMEOUT_SECONDS` is how long the server waits
for a worker that has been asked to stop before killing it.

## The contracts

Both transports are versioned and both are stable.

- **HTTP** — the `/v1` surface: `/v1/models`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/images/generations`.
- **gRPC** — `gateway.v1.ModelGatewayService` in `proto/gateway/v1/gateway.proto`:
  `Invoke`, `InvokeStream`, `Embed`, `ListModels`, plus the standard gRPC health
  protocol on the same port. The package is versioned, field numbers are stable, and
  a field is never reused or renumbered.

Both reach identical policy: one inference service decides what may happen, and each
transport only translates (PRD section 5.2).

## What a deployment must provide

The gateway terminates no TLS. A deployment that needs it terminates it in front,
for both ports, which is also where an operator's own access control belongs.

The URL policy pins every outbound connection to an address it has validated and
refuses redirects, so the gateway cannot be pointed at loopback, private,
link-local, metadata or unique-local addresses by configuration. A deployment that
wants a second layer — or that enables a path the binding has not verified, which
the binding refuses rather than runs — should enforce it at the network egress.
The gateway's own controls are not a substitute for a deployment's (PRD section
14.1).

Provider credentials arrive as environment variables and are never accepted from an
inference request body. They do not appear in a public response, in a log line, or
in an error payload.

## Verifying a deployment

```sh
curl -fsS http://localhost:8080/healthz          # ok
curl -fsS http://localhost:8080/readyz           # {"status":"ready","checks":{}}
curl -fsS -H "Authorization: Bearer $KEY" http://localhost:8080/v1/models
```

The gRPC surface answers the standard health protocol on the same port, and
`/metrics` requires the same credential as the rest of the API.
