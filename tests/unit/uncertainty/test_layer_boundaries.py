"""Static dependency checks for the uncertainty layer (Phase 9).

The uncertainty layer sits after the model layer and before the policy layer, and it
imports from both earlier layers and from the model layer itself. What it must *not* import
is anything later - policy, outcome, feedback, evaluation, or API. The check is static
rather than behavioural, because a late-layer import that only appears on a rarely-taken
branch will not be exercised by any test, and a test that depends on reaching a line to
notice a violation is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Layers the uncertainty layer is permitted to depend on.
ALLOWED_DEPENDENCIES = frozenset(
    {
        "focus_engine.baseline",
        "focus_engine.configuration",
        "focus_engine.features",
        "focus_engine.models",
        "focus_engine.schemas",
        "focus_engine.utils",
    }
)

#: Internal edges. ``engine`` depends on every sibling, and ``__init__`` re-exports them.
SELF_DEPENDENCIES = {
    "engine": frozenset({"calibration", "evidence", "explanation", "outcomes"}),
    "calibration": frozenset(),
    "evidence": frozenset(),
    "explanation": frozenset(),
    "outcomes": frozenset({"explanation"}),
}

#: Later layers, named so the failure message says which phase was reached into.
LATER_LAYERS = {
    "focus_engine.policy": "Phase 10 intervention policy",
    "focus_engine.outcomes": "Phase 11 outcome engine",
    "focus_engine.feedback": "Phase 12 feedback architecture",
    "focus_engine.evaluation": "Phase 13 evaluation framework",
    "focus_engine.api": "Phase 14 API",
}


def _imported_modules(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    return re.findall(r"^\s*(?:from|import)\s+(focus_engine[\w.]*)", text, re.MULTILINE)


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_no_uncertainty_module_imports_a_later_layer(module: str) -> None:
    import focus_engine.uncertainty as package

    root = package.__file__
    assert root is not None
    source = Path(root).parent / f"{module}.py"
    assert source.is_file()

    for imported in _imported_modules(source):
        for later, phase in LATER_LAYERS.items():
            assert not (imported == later or imported.startswith(f"{later}.")), (
                f"uncertainty/{module}.py imports {imported}, reaching into the {phase}"
            )


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_uncertainty_modules_import_only_permitted_layers(module: str) -> None:
    import focus_engine.uncertainty as package

    root = package.__file__
    assert root is not None
    source = Path(root).parent / f"{module}.py"
    assert source.is_file()

    permitted_siblings = SELF_DEPENDENCIES[module]
    for imported in _imported_modules(source):
        if imported == "focus_engine" or imported.startswith("focus_engine.uncertainty"):
            if imported == "focus_engine.uncertainty":
                continue
            suffix = imported.removeprefix("focus_engine.uncertainty.")
            assert suffix in permitted_siblings, (
                f"uncertainty/{module}.py imports {imported}; "
                f"the declared internal dependencies are "
                f"{sorted(permitted_siblings) or 'none'}"
            )
            continue
        assert any(
            imported == base or imported.startswith(f"{base}.") for base in ALLOWED_DEPENDENCIES
        ), f"uncertainty/{module}.py imports {imported}, which is not a permitted dependency"


def test_the_uncertainty_package_exports_only_known_names() -> None:
    import focus_engine.uncertainty as package

    for name in package.__all__:
        assert hasattr(package, name), (
            f"uncertainty/__init__.py lists {name} in __all__ but does not bind it"
        )
    assert len(set(package.__all__)) == len(package.__all__), (
        "uncertainty/__init__.py lists a name in __all__ twice"
    )


def test_the_models_boundary_test_names_the_uncertainty_layer() -> None:
    """Phase 8's boundary test must forbid importing the uncertainty layer, and vice versa."""
    boundary = Path("tests/unit/models/test_layer_boundaries.py").resolve()
    assert boundary.exists()
    text = boundary.read_text(encoding="utf-8")
    assert "focus_engine.uncertainty" in text
    assert "Phase 9" in text
