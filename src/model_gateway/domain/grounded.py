"""Canonical grounded search request, result, citations and usage.

Grounded search is the gateway's controlled web-egress operation: a caller
asks a directive, the gateway searches the web, and the answer it returns is
grounded only in the sources the search found. The types here are the canonical
representation of that exchange — independent of the Exa search API, of the
OpenAI-compatible synthesis call and of any wire format, exactly as the text
operation's types are independent of a provider's dialect.

Two properties follow the rest of the domain package.

**Immutable.** Each type is a frozen dataclass holding tuples rather than
lists, so a validated value cannot be mutated afterwards.

**Serialisable.** Every type round-trips through ``to_dict`` and
``from_dict``, and ``to_dict`` always emits every field so the shape is pinned
rather than implied.

The one type with a vocabulary is :class:`GroundedFinishReason`. The grounded
answer's finish reason is the grounded operation's own — ``COMPLETE`` when the
synthesis produced an answer and ``UNSPECIFIED`` when it produced none — rather
than the text operation's, whose members describe how a provider's generation
stopped.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from model_gateway.domain.serialization import (
    as_int,
    as_object,
    as_optional_float,
    as_required_text,
    as_text,
    as_text_tuple,
    reject_unknown_keys,
)

__all__ = [
    "GroundedCitation",
    "GroundedFinishReason",
    "GroundedSearchRequest",
    "GroundedSearchResult",
    "GroundedUsage",
]

_REQUEST_KEYS = frozenset(
    {
        "invocation_id",
        "tenant_id",
        "gcid",
        "agent_id",
        "crew_kind",
        "directive",
        "max_results",
        "action_code",
        "traceparent",
        "tracestate",
    }
)
_RESULT_KEYS = frozenset(
    {
        "invocation_id",
        "citations",
        "grounded_answer",
        "search_entry_point_html",
        "web_search_queries",
        "usage",
        "vendor",
        "model_version",
        "finish_reason",
        "finish_detail",
        "completed_at",
    }
)
_CITATION_KEYS = frozenset({"uri", "title", "domain", "snippet", "confidence"})
_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "grounding_units", "cost_micros"}
)


class GroundedFinishReason(StrEnum):
    """Why one grounded search finished.

    ``COMPLETE`` is the happy path: the synthesis produced an answer over the
    cited sources. ``UNSPECIFIED`` is the in-band anomaly — the search found no
    renderable citation, so no synthesis ran and there is no answer to report.
    It is a legitimate outcome the caller hedges on, not a failure: the response
    carries it with empty citations rather than raising.
    """

    UNSPECIFIED = "unspecified"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GroundedCitation:
    """One renderable source the grounded answer rests on.

    ``uri`` and ``title`` are the citation surface — a hit counts toward the
    citation mandate only when both are set, which is why both are required
    here rather than defaulted. ``domain`` is the publisher host, ``snippet`` the
    evidence segment the citation was grounded in, and ``confidence`` the search
    score, ``0.0`` when the search reported none.
    """

    uri: str
    title: str
    domain: str = ""
    snippet: str = ""
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", as_required_text("citation 'uri'", self.uri))
        object.__setattr__(self, "title", as_required_text("citation 'title'", self.title))
        object.__setattr__(self, "domain", as_text("citation 'domain'", self.domain))
        object.__setattr__(self, "snippet", as_text("citation 'snippet'", self.snippet))
        confidence = as_optional_float("citation 'confidence'", self.confidence)
        if confidence is None or not math.isfinite(confidence):
            raise ValueError("citation 'confidence' must be a finite number")
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "title": self.title,
            "domain": self.domain,
            "snippet": self.snippet,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "citation"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _CITATION_KEYS)
        return cls(
            uri=as_required_text(f"{where} 'uri'", payload.get("uri")),
            title=as_required_text(f"{where} 'title'", payload.get("title")),
            domain=as_text(f"{where} 'domain'", payload.get("domain")),
            snippet=as_text(f"{where} 'snippet'", payload.get("snippet")),
            confidence=as_optional_float(f"{where} 'confidence'", payload.get("confidence")) or 0.0,
        )


@dataclass(frozen=True)
class GroundedUsage:
    """Token and cost accounting for one grounded search.

    Every field is a count the gateway can state: the synthesis call's input
    and output tokens, their total, the number of grounding units (one per
    search), and the total cost in micros — the flat per-search surcharge plus
    the synthesis leg's own token cost. A field is never absent because the
    gateway accounts the call itself: even a search that found nothing renderable
    billed the per-search surcharge, and a count the gateway could not state
    would hide a cost the caller owes.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    grounding_units: int = 0
    cost_micros: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens", "grounding_units"):
            object.__setattr__(self, name, as_int(name, getattr(self, name), minimum=0))
        object.__setattr__(self, "cost_micros", as_int("cost_micros", self.cost_micros, minimum=0))

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "grounding_units": self.grounding_units,
            "cost_micros": self.cost_micros,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "grounded usage"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _USAGE_KEYS)
        return cls(
            input_tokens=as_int(f"{where} 'input_tokens'", payload.get("input_tokens")),
            output_tokens=as_int(f"{where} 'output_tokens'", payload.get("output_tokens")),
            total_tokens=as_int(f"{where} 'total_tokens'", payload.get("total_tokens")),
            grounding_units=as_int(f"{where} 'grounding_units'", payload.get("grounding_units")),
            cost_micros=as_int(f"{where} 'cost_micros'", payload.get("cost_micros")),
        )


@dataclass(frozen=True)
class GroundedSearchRequest:
    """The canonical grounded search a caller asks for.

    The identity fields — ``tenant_id``, ``gcid``, ``agent_id`` — are
    caller-supplied labels the way the invocation's are, and the transport
    reconciles the tenant against the credential before the request is built.
    Which of them a search requires is the handler's validation, not this
    type's: the type carries what the caller stated and the handler refuses a
    request that states too little.
    """

    invocation_id: str = ""
    tenant_id: str = ""
    gcid: str = ""
    agent_id: str = ""
    crew_kind: str = ""
    directive: str = ""
    max_results: int = 0
    action_code: str = ""
    traceparent: str = ""
    tracestate: str = ""

    def __post_init__(self) -> None:
        for name in (
            "invocation_id",
            "tenant_id",
            "gcid",
            "agent_id",
            "crew_kind",
            "directive",
            "action_code",
            "traceparent",
            "tracestate",
        ):
            object.__setattr__(self, name, as_text(name, getattr(self, name)))
        object.__setattr__(self, "max_results", as_int("max_results", self.max_results, minimum=0))

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation_id": self.invocation_id,
            "tenant_id": self.tenant_id,
            "gcid": self.gcid,
            "agent_id": self.agent_id,
            "crew_kind": self.crew_kind,
            "directive": self.directive,
            "max_results": self.max_results,
            "action_code": self.action_code,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "grounded search request"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _REQUEST_KEYS)
        return cls(
            invocation_id=as_text(f"{where} 'invocation_id'", payload.get("invocation_id")),
            tenant_id=as_text(f"{where} 'tenant_id'", payload.get("tenant_id")),
            gcid=as_text(f"{where} 'gcid'", payload.get("gcid")),
            agent_id=as_text(f"{where} 'agent_id'", payload.get("agent_id")),
            crew_kind=as_text(f"{where} 'crew_kind'", payload.get("crew_kind")),
            directive=as_text(f"{where} 'directive'", payload.get("directive")),
            max_results=as_int(f"{where} 'max_results'", payload.get("max_results")),
            action_code=as_text(f"{where} 'action_code'", payload.get("action_code")),
            traceparent=as_text(f"{where} 'traceparent'", payload.get("traceparent")),
            tracestate=as_text(f"{where} 'tracestate'", payload.get("tracestate")),
        )


@dataclass(frozen=True)
class GroundedSearchResult:
    """The canonical answer to one grounded search.

    ``citations`` are the sources the answer is grounded in; empty means the
    search found nothing renderable and the caller must hedge rather than
    present web content as knowledge. ``grounded_answer`` is empty on that same
    path, because no synthesis ran. ``web_search_queries`` records the queries
    the gateway issued for transparency. ``vendor`` names the search backend,
    ``model_version`` the synthesis model, and ``completed_at`` an ISO 8601
    timestamp of when the gateway emitted the response.
    """

    invocation_id: str = ""
    citations: tuple[GroundedCitation, ...] = ()
    grounded_answer: str = ""
    search_entry_point_html: str = ""
    web_search_queries: tuple[str, ...] = ()
    usage: GroundedUsage = GroundedUsage()
    vendor: str = ""
    model_version: str = ""
    finish_reason: GroundedFinishReason = GroundedFinishReason.UNSPECIFIED
    finish_detail: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", _citations("citations", self.citations))
        object.__setattr__(
            self, "web_search_queries", as_text_tuple("web_search_queries", self.web_search_queries)
        )
        for name in (
            "invocation_id",
            "grounded_answer",
            "search_entry_point_html",
            "vendor",
            "model_version",
            "finish_detail",
            "completed_at",
        ):
            object.__setattr__(self, name, as_text(name, getattr(self, name)))
        object.__setattr__(self, "usage", as_usage("usage", self.usage))
        object.__setattr__(
            self, "finish_reason", _finish_reason("finish_reason", self.finish_reason)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation_id": self.invocation_id,
            "citations": [citation.to_dict() for citation in self.citations],
            "grounded_answer": self.grounded_answer,
            "search_entry_point_html": self.search_entry_point_html,
            "web_search_queries": list(self.web_search_queries),
            "usage": self.usage.to_dict(),
            "vendor": self.vendor,
            "model_version": self.model_version,
            "finish_reason": self.finish_reason.value,
            "finish_detail": self.finish_detail,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        where = "grounded search result"
        payload = as_object(where, data)
        reject_unknown_keys(where, payload, _RESULT_KEYS)
        return cls(
            invocation_id=as_text(f"{where} 'invocation_id'", payload.get("invocation_id")),
            citations=_citations(f"{where} 'citations'", payload.get("citations")),
            grounded_answer=as_text(f"{where} 'grounded_answer'", payload.get("grounded_answer")),
            search_entry_point_html=as_text(
                f"{where} 'search_entry_point_html'", payload.get("search_entry_point_html")
            ),
            web_search_queries=as_text_tuple(
                f"{where} 'web_search_queries'", payload.get("web_search_queries")
            ),
            usage=as_usage(f"{where} 'usage'", payload.get("usage")),
            vendor=as_text(f"{where} 'vendor'", payload.get("vendor")),
            model_version=as_text(f"{where} 'model_version'", payload.get("model_version")),
            finish_reason=_finish_reason(f"{where} 'finish_reason'", payload.get("finish_reason")),
            finish_detail=as_text(f"{where} 'finish_detail'", payload.get("finish_detail")),
            completed_at=as_text(f"{where} 'completed_at'", payload.get("completed_at")),
        )


def _citations(where: str, value: object) -> tuple[GroundedCitation, ...]:
    """The citations of a grounded result, refusing anything else."""
    if value is None:
        return ()
    if isinstance(value, GroundedCitation):
        return (value,)
    if not isinstance(value, tuple | list):
        raise ValueError(f"{where} must be an array of citations, got {type(value).__name__}")
    return tuple(
        item if isinstance(item, GroundedCitation) else GroundedCitation.from_dict(item)
        for item in value
    )


def _finish_reason(where: str, value: object) -> GroundedFinishReason:
    """A grounded finish reason, refusing a value outside the vocabulary."""
    member = GroundedFinishReason(value) if isinstance(value, str) else value
    if not isinstance(member, GroundedFinishReason):
        supported = ", ".join(repr(item.value) for item in GroundedFinishReason)
        raise ValueError(f"{where} must be one of {supported}, got {value!r}")
    return member


def as_usage(where: str, value: object) -> GroundedUsage:
    """A grounded usage record, refusing anything else."""
    if isinstance(value, GroundedUsage):
        return value
    if isinstance(value, Mapping):
        return GroundedUsage.from_dict(value)
    raise ValueError(f"{where} must be a GroundedUsage, got {type(value).__name__}")
