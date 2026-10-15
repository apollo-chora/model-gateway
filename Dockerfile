# The gateway's container image (PRD section 19).
#
# One image, one process, two listeners: HTTP on 8080 and gRPC on 9090, both
# started and stopped by the application, which is the process's unit of lifetime
# (PRD section 18). Nothing about a provider is baked in — every setting arrives as
# an environment variable and the process refuses to start on a configuration it
# cannot serve — so one image runs against any provider by changing its
# environment alone (PRD section 23).
#
# Both stages use the same base image, and that is load-bearing rather than tidy:
# a virtual environment records the interpreter it was built for, so copying it
# into a different base would leave it pointing at an interpreter that is not
# there.
#
# Both base images are pinned by digest rather than by tag. A tag is a moving
# reference, and a deployment that wants the same bytes twice needs the digest.

FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641 AS build

# The lockfile's own resolver, copied from its image rather than installed by a
# script that would itself need pinning. uv is a build-time tool: it is not in the
# runtime image.
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /uvx /bin/

# Byte-compile at install time; copy rather than hard-link, because this
# environment is about to be moved into another stage; and never download an
# interpreter, because the base image's own Python 3.13 is the one the lockfile
# resolved for.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, from the lockfile and nothing else, so this layer survives a
# change to the source and the build cannot resolve a version the lock does not
# name. The gateway itself is left for the next layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
# A regular install rather than an editable one: the runtime image carries the
# environment and not the source tree.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641 AS runtime

LABEL org.opencontainers.image.title="model-gateway" \
      org.opencontainers.image.description="Vendor-neutral AI model gateway" \
      org.opencontainers.image.source="https://github.com/apollo-chora/model-gateway"

# The two ports the deployment must publish (PRD section 19). These are the
# defaults: the ports are CHORA_HTTP_PORT and CHORA_GRPC_PORT, so a deployment that
# moves them publishes those instead.
EXPOSE 8080 9090

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# The process writes nothing to its own filesystem, so it is given no privileges
# to write with.
RUN groupadd --system --gid 1001 gateway \
    && useradd --system --uid 1001 --gid gateway --no-create-home gateway

USER gateway
WORKDIR /app

# Liveness only, and deliberately /healthz rather than /readyz: readiness is the
# deployment's question and is what a load balancer should route on, while this
# answers whether the process is still the process. The port is read from the
# environment so a deployment that moved it is still probed.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('CHORA_HTTP_PORT', '8080')).read()"]

ENTRYPOINT ["model-gateway"]
