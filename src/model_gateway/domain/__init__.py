"""Canonical, vendor-neutral domain types (PRD sections 4, 5, 9.3, 9.4 and 13).

This package is the gateway's own representation of an inference invocation: the
request, the response, the conversation they carry, and the accounting they
produce. It is deliberately independent of every provider wire format — no field
here is named after a vendor, and nothing here knows how a provider spells a
role, a content block or a finish reason. Translating between this
representation and a provider's is the LiteLLM binding's work (PRD section 9),
and this package must stay out of it.

Three properties hold for every type here.

**Immutable.** Each type is a frozen dataclass, and every collection it holds is
a tuple, a frozenset or an immutable JSON object rather than a list, a set or a
dict. A frozen dataclass holding a plain dict is only shallowly frozen: the
attribute cannot be reassigned, but the object it points at can still be mutated
in place, so a value that passed validation can change afterwards. The route
model already made that mistake once, and the domain types do not repeat it.
Values are normalised in ``__post_init__``, so a constructed value cannot hold
something the gateway could not represent, and the types are hashable wherever
their fields allow it.

**Serialisable.** Every type round-trips through ``to_dict`` and ``from_dict``.
The serialised form is canonical: ``to_dict`` always emits every field, so the
shape is pinned rather than implied, and ``from_dict`` accepts exactly that
shape. The fixtures under ``tests/fixtures/canonical/`` pin a representative
request and response to that shape.

**Explicit about what it cannot represent (PRD section 9.1).** A key the
canonical type does not define, a value of the wrong JSON type, a content part
with an unknown discriminator, a message role outside the canonical vocabulary,
an image part carrying neither data nor a URL, a structured-output mode without
the schema it requires, a value JSON cannot carry at all — each raises
``ValueError`` instead of being ignored. The rule is the PRD's: *a feature the
gateway cannot represent must fail explicitly rather than being silently
dropped.* A parameter that quietly disappears between the caller and the
provider is worse than a rejected request, because the caller is told the request
succeeded.

**Nothing upstream is discarded.** Where a provider's own vocabulary does not
line up with the gateway's, the canonical type keeps both: the classification
beside the raw finish reason, and each token count's provenance beside its value.
A canonical state that cannot tell "the provider said nothing" from "the gateway
could not classify what it said" has already lost information.

The one deliberate exception is ``FinishReason``: an unrecognised provider reason
maps to ``FinishReason.UNKNOWN`` rather than failing, because the field must
always carry a value and ``UNKNOWN`` states plainly that the gateway did not
recognise the reason (PRD sections 5.2 and 9.1). The raw reason is preserved
beside it, so ``UNKNOWN`` never means the gateway threw the reason away.

The domain layer raises builtin ``ValueError`` and defines no error type of its
own; deciding which error category a caller sees belongs to the gateway core.
"""

from model_gateway.domain.content import Message, Role
from model_gateway.domain.generation import (
    FinishReason,
    GenerationConfig,
    ResponseFormat,
    ResponseFormatMode,
    Usage,
    UsageSource,
)
from model_gateway.domain.invocation import (
    FORWARDABLE_METADATA_KEYS,
    IdempotencyStatus,
    InvocationRequest,
    InvocationResponse,
    ResponseModality,
)
from model_gateway.domain.json_value import (
    FrozenJsonObject,
    JsonDocument,
    JsonValue,
    freeze_json,
    thaw_json,
)
from model_gateway.domain.output import OutputItem, output_item_from_dict
from model_gateway.domain.parts import (
    ContentPart,
    ImagePart,
    RefusalPart,
    TextPart,
    content_part_from_dict,
)
from model_gateway.domain.streaming import (
    StreamEvent,
    StreamEventKind,
    ToolCallDelta,
)
from model_gateway.domain.tools import (
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResult,
)

__all__ = [
    "ContentPart",
    "FORWARDABLE_METADATA_KEYS",
    "FinishReason",
    "FrozenJsonObject",
    "GenerationConfig",
    "IdempotencyStatus",
    "ImagePart",
    "InvocationRequest",
    "InvocationResponse",
    "JsonDocument",
    "JsonValue",
    "Message",
    "OutputItem",
    "RefusalPart",
    "ResponseFormat",
    "ResponseFormatMode",
    "ResponseModality",
    "Role",
    "StreamEvent",
    "StreamEventKind",
    "TextPart",
    "ToolCall",
    "ToolCallDelta",
    "ToolChoice",
    "ToolChoiceMode",
    "ToolDefinition",
    "ToolResult",
    "Usage",
    "UsageSource",
    "content_part_from_dict",
    "freeze_json",
    "output_item_from_dict",
    "thaw_json",
]
