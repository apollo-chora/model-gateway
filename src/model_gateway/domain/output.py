"""Canonical model output as an ordered sequence of items (PRD 5.2, 9.1, 9.3 and 9.4).

A model emits its answer in order, and the order carries meaning: text, then an
image, then more text is not the same response as both texts followed by the
image. The canonical response therefore stores its output as a sequence of items
rather than as separate ``text`` and ``images`` fields, which cannot express an
interleaving and silently concatenate it away.

An item is a content part or a tool call, because a model interleaves those too:
it may say what it is about to do, call a tool, and then explain what came back.
Keeping tool calls in the same sequence is what preserves that ordering, and it
is why this union is wider than ``ContentPart`` alone.

An item whose type the gateway does not know is refused rather than skipped: a
dropped item changes the answer the caller receives (PRD section 9.1).
"""

from model_gateway.domain.json_value import JsonDocument
from model_gateway.domain.parts import (
    CONTENT_PART_TYPES,
    ContentPart,
    ImagePart,
    RefusalPart,
    TextPart,
    content_part_from_dict,
)
from model_gateway.domain.serialization import as_object, as_sequence, describe
from model_gateway.domain.tools import ToolCall

__all__ = ["OutputItem", "as_output_items", "output_item_from_dict", "output_item_to_dict"]

_TYPE = "type"
_TOOL_CALL = "tool_call"

_OUTPUT_ITEM_TYPES: frozenset[str] = frozenset({*CONTENT_PART_TYPES, _TOOL_CALL})

type OutputItem = ContentPart | ToolCall


def output_item_to_dict(item: OutputItem) -> dict[str, JsonDocument]:
    """Serialise one output item, giving a tool call the discriminator it needs.

    A content part carries its own ``type`` already. A tool call does not, because
    inside a message's ``tool_calls`` array every entry is a call; in an ordered
    output sequence it shares the array with other kinds of item, so the
    discriminator is added here to keep it unambiguous.
    """
    payload = item.to_dict()
    if isinstance(item, ToolCall):
        return {_TYPE: _TOOL_CALL, **payload}
    return payload


def output_item_from_dict(data: object) -> OutputItem:
    """Parse one output item, dispatching on its ``type`` discriminator.

    A tool call's discriminator is stripped before the call itself is parsed: the
    discriminator belongs to the enclosing sequence, and ``ToolCall`` defines the
    call's own fields only.
    """
    where = "output item"
    payload = as_object(where, data)
    kind = payload.get(_TYPE)
    if kind == _TOOL_CALL:
        return ToolCall.from_dict({key: value for key, value in payload.items() if key != _TYPE})
    if isinstance(kind, str) and kind in CONTENT_PART_TYPES:
        return content_part_from_dict(payload)
    if not isinstance(kind, str):
        raise ValueError(
            f"{where} must carry a string {_TYPE!r} discriminator, got {describe(kind)}"
        )
    supported = ", ".join(repr(item) for item in sorted(_OUTPUT_ITEM_TYPES))
    raise ValueError(
        f"{where} has type {kind!r}, which the canonical output representation does not "
        f"define; supported types are {supported} (PRD section 9.1)"
    )


def as_output_items(where: str, value: object) -> tuple[OutputItem, ...]:
    """Coerce a sequence of output items, refusing anything else."""
    items = as_sequence(where, value)
    for index, item in enumerate(items):
        if not isinstance(item, TextPart | ImagePart | RefusalPart | ToolCall):
            raise ValueError(
                f"{where}[{index}] must be a TextPart, an ImagePart, a RefusalPart or a "
                f"ToolCall, got {describe(item)}"
            )
    return items
