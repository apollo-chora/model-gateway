"""Vendor neutrality: the core names no provider.

The gateway core holds no provider-specific business logic, and the provider is
chosen entirely by configuration. That is the property the whole design turns on —
it is what makes a provider swap an environment change — and until this module
existed nothing asserted it. Every other suite tests behaviour; a suite that only
tests behaviour cannot tell whether the core grew a branch on a vendor name, because
a branch on a vendor name can pass every behavioural test there is.

**What is asserted, and why each half is needed.** A rule enforced by one check is a
rule the other can be broken without. The first is that no core module imports a
vendor's library, which is the blunt form: whatever vendor logic exists must live
where the vendor's own types are. The second is that no core module names a provider
dialect in a string literal, which catches the form the first cannot — a dialect
compared as a string, a default chosen per provider, a mapping keyed by vendor name.
The third is that only the composition root reaches into a vendor binding, because
the core's legitimate dependency on the binding layer is its *protocol* — the shape a
provider must have — and not its implementation.

**The dialect names are read from the binding rather than listed here.** Adding a
provider to the binding therefore extends what the core may not name, which is the
direction the rule should move: the binding is where the vendor list lives, and the
core is not allowed to know it. The vendor *library* names are listed here, because
they are the two the binding imports and a library name is not something the binding
discovers at runtime.

**What this proves, and what it does not.** It prevents the forms of vendor coupling
that are reviewable in source — an import, a literal, an implementation reached from
the wrong module — and it makes the intended dependency boundary executable rather
than a convention. It does **not** prove the absence of all vendor-specific logic,
and no static check could: a core module could hold provider-specific behaviour
behind a neutral name, or reach a vendor through an alias assembled at runtime, and
neither is something a scanner can decide. A future violation of that kind should
motivate a targeted rule rather than an arms race against string assembly, and
saying so is more useful than implying this module closes the claim.

The literal check reads every string constant, docstrings included, so a core module
that *explained* a provider dialect in prose would fail it. That is stricter than
preventing an executable branch, and it is deliberate: the architectural policy is
that vendor names live in the binding, and a rule with an exception for prose is a
rule whose exception is where the next coupling hides.
"""

import ast
from pathlib import Path

from model_gateway.providers.litellm_image import IMAGE_DIALECTS
from model_gateway.providers.litellm_provider import SUPPORTED_DIALECTS

SOURCE = Path(__file__).resolve().parent.parent / "src" / "model_gateway"
BINDING = SOURCE / "providers"

VENDOR_LIBRARIES = frozenset({"litellm", "openai"})
"""The vendor libraries the binding is allowed to import and the core is not."""

DIALECTS = frozenset(SUPPORTED_DIALECTS) | frozenset(IMAGE_DIALECTS)
"""Every provider dialect the binding verifies, for either operation."""

VENDOR_NAMES = VENDOR_LIBRARIES | DIALECTS
"""Every name the core may not carry, as a library or as a dialect."""

VENDOR_BINDINGS = frozenset(
    {"model_gateway.providers.litellm_provider", "model_gateway.providers.litellm_image"}
)
"""The binding modules that name a vendor, reachable only from the composition root."""

COMPOSITION_ROOT = "app.py"
"""The one module whose job is to choose the binding, so the one allowed to name it."""


def _core_modules() -> list[Path]:
    """Every module of the gateway core: the package without the binding package."""
    return sorted(
        path
        for path in SOURCE.rglob("*.py")
        if BINDING not in path.parents and "__pycache__" not in path.parts
    )


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> set[str]:
    """Every dotted module name the module imports, from either import form."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _string_literals(tree: ast.Module) -> set[str]:
    """Every string constant in the module, docstrings included."""
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_no_core_module_imports_a_vendor_library():
    """The vendor's own types exist only where the vendor's protocol is spoken."""
    violations = sorted(
        f"{path.relative_to(SOURCE)} imports {module}"
        for path in _core_modules()
        for module in _imported_modules(_tree(path))
        if module.split(".")[0] in VENDOR_LIBRARIES
    )

    assert violations == []


def test_no_core_module_names_a_vendor_or_a_dialect():
    """A vendor's name is configuration the binding interprets, never a name the core knows.

    This is the half an import check cannot make, in two ways. A core that branched
    on ``"anthropic"`` would import nothing a vendor owns and would still have
    encoded one vendor's semantics into the gateway; and a core that imported a
    vendor by name rather than by statement — ``__import__("litellm")`` — would be
    invisible to a check that reads import statements. Both are literals, so both
    are caught here.
    """
    violations = sorted(
        f"{path.relative_to(SOURCE)} names {literal!r}"
        for path in _core_modules()
        for literal in _string_literals(_tree(path))
        if literal in VENDOR_NAMES
    )

    assert violations == []


def test_only_the_composition_root_reaches_into_a_vendor_binding():
    """The core depends on the binding's protocol, not on its implementation.

    ``inference`` imports ``providers.base`` — the shape a provider must have — and
    that is the dependency the architecture wants: the core is written against the
    protocol and the binding satisfies it. What no other module may do is import the
    modules that name a vendor, because a second such importer is a second place the
    provider is chosen, and the provider is chosen by configuration in one place.
    """
    importers = sorted(
        path.name
        for path in _core_modules()
        if _imported_modules(_tree(path)) & VENDOR_BINDINGS
    )

    assert importers == [COMPOSITION_ROOT]
