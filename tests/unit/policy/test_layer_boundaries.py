"""Static dependency checks for the policy layer (Phase 10).

The policy layer sits after the uncertainty layer, and it is the last layer that decides
*whether* to act. What it must not import is anything that measures what happened next -
outcome, feedback, evaluation, or API. The edge that matters most is the feedback layer:
a policy that reads its own observed response would be selecting on the outcome of its own
previous choices, which is the feedback architecture arriving through the back door and
without its version, its observation gate, or its refusal to retrain.

The check is static rather than behavioural, because a late-layer import that only appears
on a rarely-taken branch will not be exercised by any test, and a test that depends on
reaching a line to notice a violation is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Layers the policy layer is permitted to depend on.
ALLOWED_DEPENDENCIES = frozenset(
    {
        "focus_engine.configuration",
        "focus_engine.events",
        "focus_engine.schemas",
        "focus_engine.uncertainty",
        "focus_engine.utils",
    }
)

#: Internal edges. ``engine`` composes the other two, and ``__init__`` re-exports all three.
SELF_DEPENDENCIES = {
    "engine": frozenset({"history", "models"}),
    "history": frozenset(),
    "models": frozenset(),
    "__init__": frozenset({"engine", "history", "models"}),
}

#: Later layers, named so the failure message says which phase was reached into.
#:
#: ``focus_engine.outcomes`` is Phase 11 and is a *different module* from
#: ``focus_engine.uncertainty.outcomes``, which this layer is allowed to import. The
#: prefix check below therefore matches the full dotted path rather than a bare
#: ``outcomes``, so the permitted import is not mistaken for the forbidden one.
LATER_LAYERS = {
    "focus_engine.outcomes": "Phase 11 outcome engine",
    "focus_engine.feedback": "Phase 12 feedback architecture",
    "focus_engine.evaluation": "Phase 13 evaluation framework",
    "focus_engine.api": "Phase 14 API",
}


def _imported_modules(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    return re.findall(r"^\s*(?:from|import)\s+(focus_engine[\w.]*)", text, re.MULTILINE)


def _module_source(module: str) -> Path:
    """Locate a policy module *without importing the package*.

    Importing it to find its path looks harmless but is exactly the failure this check
    exists to catch. A forbidden import of a module that does not exist yet is an
    ``ImportError`` at collection time, so every test in the file errors out with a stack
    trace naming the missing module instead of one test failing with a sentence naming
    the layer boundary that was crossed. The path is derived from this file's own
    location, so the check reads the source as text and nothing has to be executable.
    """
    source = (
        Path(__file__).resolve().parents[3] / "src" / "focus_engine" / "policy" / f"{module}.py"
    )
    assert source.is_file(), f"expected the policy layer to contain {module}.py"
    return source


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_no_policy_module_imports_a_later_layer(module: str) -> None:
    for imported in _imported_modules(_module_source(module)):
        for later, phase in LATER_LAYERS.items():
            assert not (imported == later or imported.startswith(f"{later}.")), (
                f"policy/{module}.py imports {imported}, reaching into the {phase}"
            )


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_policy_modules_import_only_permitted_layers(module: str) -> None:
    permitted_siblings = SELF_DEPENDENCIES[module]
    for imported in _imported_modules(_module_source(module)):
        if imported.startswith("focus_engine.policy"):
            if imported == "focus_engine.policy":
                continue
            suffix = imported.removeprefix("focus_engine.policy.")
            assert suffix in permitted_siblings, (
                f"policy/{module}.py imports {imported}; "
                f"the declared internal dependencies are "
                f"{sorted(permitted_siblings) or 'none'}"
            )
            continue
        assert any(
            imported == base or imported.startswith(f"{base}.") for base in ALLOWED_DEPENDENCIES
        ), f"policy/{module}.py imports {imported}, which is not a permitted dependency"


def test_the_policy_package_exports_only_known_names() -> None:
    import focus_engine.policy as package

    for name in package.__all__:
        assert hasattr(package, name), (
            f"policy/__init__.py lists {name} in __all__ but does not bind it"
        )
    assert len(set(package.__all__)) == len(package.__all__), (
        "policy/__init__.py lists a name in __all__ twice"
    )


def test_the_package_exports_its_composition_point_and_its_evidence_type() -> None:
    """A caller must be able to build a policy and the history it reasons over.

    ``__init__`` originally re-exported only ``models``, so the two names a caller
    actually needs - the policy and the history it is given - were reachable only by
    importing a private module path. A public surface that omits its own entry point
    invites callers to reach past it, and past the package is where the layer boundary
    stops being enforced by anything.
    """
    import focus_engine.policy as package

    assert {"InterventionPolicy", "InterventionHistory", "PolicyDecision"} <= set(package.__all__)


def test_the_uncertainty_boundary_test_names_the_policy_layer() -> None:
    """The boundary has to be declared from both sides to be a boundary.

    Phase 9's check forbids the uncertainty layer from importing the policy. If the
    policy's own check did not exist, the edge would be guarded in one direction only, and
    the unguarded direction is the one that gets taken.
    """
    boundary = Path("tests/unit/uncertainty/test_layer_boundaries.py").resolve()
    assert boundary.exists()
    text = boundary.read_text(encoding="utf-8")
    assert "focus_engine.policy" in text
    assert "Phase 10" in text
