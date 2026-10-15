"""The public model registry: the fixed public model IDs resolved onto configured routes.

The gateway exposes exactly three public model IDs:

    text-default
    image-default
    embedding-default

They are part of the public contract and are not configurable aliases. Each names
one operation, and each operation resolves to the routes configured for it — the
singleton block's primary route first, then ``<BLOCK>_ROUTES_JSON`` entries in
array order, which is the fallback order.

The registry is the only thing that maps a public ID onto a route, and it maps an
ID onto *configured* routes only. A caller-supplied model name therefore cannot
reach an unconfigured upstream endpoint:
the ID either resolves to a route this deployment configured, or it does not
resolve at all. Nothing about a route — its endpoint, its credential, its upstream
model, its extra headers — is reachable from a model ID, and the error an unknown
ID raises carries none of it.

A :class:`ModelInfo` is derived from the operation's routes. Its capabilities are
the capabilities the *primary* route declares, because that is the route that
would serve the model; a fallback route the primary does not need is not evidence
the primary can do more. That single rule covers every operation, and it is the
rule that lets a caller read an image model's contract off the listing: an image
route declares ``generation``, and ``editing`` only when the operator stated it
, so the advertised set is what tells a caller whether the model
can be asked to edit an image as well as to produce one — and an image route that
declares ``generation`` alone is refused an edit request rather than handed one.

Its advertised limits are the ones the deployment configured for the operation:
``TEXT_LLM_CONTEXT_WINDOW`` and
``TEXT_LLM_MAX_OUTPUT_TOKENS`` for the text operation, which are environment
configuration the route model does not carry and are read from the environment
the registry was built from, and the declared embedding dimensionality for the
embeddings operation, which is a property of the *primary route*
(``EMBEDDINGS_DIMENSIONS``, or a ``dimensions`` entry on it), because the
inference path checks every answer against the dimensionality of the route that
produced it rather than against one operation-wide setting. The image operation
advertises no limit: an ``IMAGE_GEN_*`` block configures neither a context window
nor a dimensionality, and an unconfigured limit is not a limit of zero.

An operation with no configured route is not listed at all, which is what makes
an absent optional capability disabled rather than fatal to the others. Image
generation is one of those optional capabilities: a deployment
that configures no ``IMAGE_GEN_*`` block serves text and embeddings and publishes
no image model, and ``image-default`` resolves to nothing at all.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation, Route, parse_routes
from model_gateway.config.settings import ConfigurationError
from model_gateway.errors import ModelNotFoundError

__all__ = ["PUBLIC_MODEL_IDS", "ModelInfo", "ModelRegistry"]


_CONTEXT_WINDOW = "TEXT_LLM_CONTEXT_WINDOW"
_MAX_OUTPUT_TOKENS = "TEXT_LLM_MAX_OUTPUT_TOKENS"

PUBLIC_MODEL_IDS: Mapping[Operation, str] = MappingProxyType(
    {
        Operation.TEXT: "text-default",
        Operation.IMAGE: "image-default",
        Operation.EMBEDDINGS: "embedding-default",
    }
)
"""The gateway's fixed public model IDs, one per operation.

The mapping is read-only: the IDs are part of the public contract, so neither a
deployment nor a caller may introduce, rename or remap one.
"""

_MODEL_BY_ID: Mapping[str, Operation] = MappingProxyType(
    {public_id: operation for operation, public_id in PUBLIC_MODEL_IDS.items()}
)


@dataclass(frozen=True)
class ModelInfo:
    """One public model as the northbound layer is allowed to see it.

    ``id`` is the public model ID and ``operation`` the operation it resolves to.

    ``capabilities`` are the capabilities the primary route declares — the route
    that would actually serve this model. They are advertised so a caller can
    choose a model and know what it can ask for, and they are the
    set the inference service enforces against the route. The
    reading differs per operation because the capability vocabulary does: a text
    model advertises ``chat`` and whatever else its route declares, an embeddings
    model ``embeddings``, and an image model ``generation`` — with ``editing``
    beside it only when the operator declared that too, which is
    how a caller learns whether the model accepts an edit request or refuses one.

    ``context_window`` and ``max_output_tokens`` are the text operation's
    advertised limits and ``dimensions`` the embeddings operation's. Each is
    ``None`` when the deployment did not configure it.
    An unconfigured limit is not a limit of zero: publishing zero would be a
    claim about a model that no configuration made.

    ``dimensions`` is the dimensionality the primary route declares for the
    embeddings operation, because the route is what an answer is checked
    against; it is ``None`` when no route of the operation declares one, or when
    the deployment declared no embeddings dimensionality.
    """

    id: str
    operation: Operation
    capabilities: frozenset[Capability]
    context_window: int | None
    max_output_tokens: int | None
    dimensions: int | None


@dataclass(frozen=True)
class _Limits:
    """The text operation's advertised limits.

    These are deployment settings the route model does not carry, so they are
    read from the environment rather than derived from a route. The embeddings
    operation's dimensionality is not one of them: it is a route property.
    """

    context_window: int | None = None
    max_output_tokens: int | None = None


class ModelRegistry:
    """The fixed public model IDs resolved onto the routes configured for each operation.

    The registry holds routes grouped by operation, in route order: the primary
    route of an operation is its first configured route, and the rest are that
    operation's fallbacks.

    Construction from routes alone reads the text limits from the process
    environment, which is where a deployment's configuration lives; pass
    :meth:`from_env` an environment mapping to read both the routes and the text
    limits from it instead. A malformed limit is rejected rather than guessed at,
    because a value the gateway cannot read is a configuration error, and
    configuration errors are detected at startup.
    """

    def __init__(self, routes: tuple[Route, ...]) -> None:
        grouped: dict[Operation, list[Route]] = {}
        for route in routes:
            grouped.setdefault(route.operation, []).append(route)
        self._by_operation = {operation: tuple(items) for operation, items in grouped.items()}
        self._limits = _limits_from(os.environ)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ModelRegistry":
        """Build the registry from ``env``, defaulting to the process environment.

        Routes come from :func:`~model_gateway.config.routes.parse_routes`, and
        the text operation's advertised limits come from the same environment
        they were configured in: those limits are deployment configuration
        , not route fields, so when an environment mapping is
        supplied both are read from it rather than from the process environment.
        The embeddings operation's advertised dimensionality is not read here at
        all — it is a property of the parsed routes' primary route.
        """
        source = os.environ if env is None else env
        registry = cls(parse_routes(source))
        if env is not None:
            registry._limits = _limits_from(env)
        return registry

    def list_models(self) -> tuple[ModelInfo, ...]:
        """One entry per configured operation, in the gateway's canonical operation order.

        An operation with no configured route contributes no entry: an absent
        optional capability is disabled rather than preventing unrelated
        capabilities from operating).
        """
        return tuple(
            self._info(operation, routes) for operation, routes in self._by_operation.items()
        )

    def get(self, model_id: str) -> ModelInfo:
        """The info for one public model ID.

        Raises :class:`~model_gateway.errors.ModelNotFoundError` when ``model_id``
        is not one of the gateway's public model IDs, or names one whose operation
        is not configured.
        """
        operation = self._operation_for(model_id)
        return self._info(operation, self._routes_of(operation))

    def routes_for(self, model_id: str) -> tuple[Route, ...]:
        """The routes serving one public model ID, primary first.

        Raises :class:`~model_gateway.errors.ModelNotFoundError` for an unknown ID,
        so a caller-supplied model name cannot be used to reach an unconfigured
        upstream endpoint.
        """
        operation = self._operation_for(model_id)
        return self._routes_of(operation)

    def _operation_for(self, model_id: str) -> Operation:
        operation = _MODEL_BY_ID.get(model_id)
        if operation is None:
            raise _unknown_model_error(model_id, self.list_models())
        return operation

    def _routes_of(self, operation: Operation) -> tuple[Route, ...]:
        routes = self._by_operation.get(operation)
        if routes is None:
            raise _unknown_model_error(PUBLIC_MODEL_IDS[operation], self.list_models())
        return routes

    def _info(self, operation: Operation, routes: tuple[Route, ...]) -> ModelInfo:
        limits = self._limits
        return ModelInfo(
            id=PUBLIC_MODEL_IDS[operation],
            operation=operation,
            capabilities=routes[0].capabilities,
            context_window=limits.context_window if operation is Operation.TEXT else None,
            max_output_tokens=limits.max_output_tokens if operation is Operation.TEXT else None,
            dimensions=routes[0].dimensions if operation is Operation.EMBEDDINGS else None,
        )


def _limits_from(env: Mapping[str, str]) -> _Limits:
    """Read the text operation's advertised limits from ``env``.

    A blank value means the deployment did not configure the limit, which is
    ``None`` and never zero. A value that is not a non-negative integer is a
    configuration error rather than something to guess at.
    """
    return _Limits(
        context_window=_read_limit(env, _CONTEXT_WINDOW),
        max_output_tokens=_read_limit(env, _MAX_OUTPUT_TOKENS),
    )


def _read_limit(env: Mapping[str, str], variable: str) -> int | None:
    raw = str(env.get(variable, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ConfigurationError(f"{variable} must be an integer, got {raw!r}") from None
    if value < 0:
        raise ConfigurationError(f"{variable} must not be negative, got {raw!r}")
    return value


def _unknown_model_error(model_id: str, configured: tuple[ModelInfo, ...]) -> ModelNotFoundError:
    """A model-not-found error that carries nothing about any route.

    Only public model IDs appear in the message. An upstream endpoint, model ID or
    credential is configuration the registry never publishes, so it cannot leak
    through the error a caller sees.
    """
    if configured:
        known = ", ".join(info.id for info in configured)
        message = (
            f"model {model_id!r} is not a public model ID this gateway exposes; "
            f"configured models are: {known}"
        )
    else:
        message = (
            f"model {model_id!r} is not a public model ID this gateway exposes; "
            f"no model is configured"
        )
    return ModelNotFoundError(message)
