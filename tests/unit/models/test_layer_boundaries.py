"""Static dependency checks for the model layer (Phase 8).

The model layer is the last layer before Phase 9 takes over, so its boundary is the one
most worth pinning. A model module reaching into `predictors` is not a style complaint: it
means the artifact, the training path, and the serving path can each acquire a private
definition of the same quantity, and the disagreement shows up as a model that trains
correctly and then scores differently from itself.

The check is static rather than behavioural because the boundary is easier to break than to
notice. An import that only appears on a rarely-taken branch will not be exercised by any
test, and a test suite that depends on reaching a line to notice a violation is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Layers the model layer is permitted to depend on. The feature layer supplies the named
#: vectors; the schema layer supplies the record, status, and primitive vocabulary; the
#: utility layer supplies clock and digest. The simulator and every later layer are absent
#: by design, so that a model module cannot reach back to the data that produced its inputs.
ALLOWED_DEPENDENCIES = frozenset(
    {
        "focus_engine.features",
        "focus_engine.schemas",
        "focus_engine.utils",
    }
)

#: Internal edges, read off the source rather than assumed. ``artifacts`` depends on
#: ``encoding`` for the schema digest and the refusal type, and the deferred imports inside
#: its functions are real dependencies: a check that only read module-level imports would
#: pass on a module that had been thoroughly rewired.
SELF_DEPENDENCIES = {
    "artifacts": frozenset({"encoding"}),
    "encoding": frozenset(),
    "predictors": frozenset({"artifacts", "encoding"}),
    "registry": frozenset(),
    "trainers": frozenset({"artifacts", "encoding"}),
}

#: Later layers. Importing any of these is a boundary violation regardless of how the
#: import is spelled, so they are listed explicitly to make the failure message name the
#: phase that would be reached into.
LATER_LAYERS = {
    "focus_engine.simulator": "Phase 4 simulator",
    "focus_engine.temporal": "Phase 7 temporal engine",
    "focus_engine.uncertainty": "Phase 9 prediction and uncertainty",
    "focus_engine.policy": "Phase 10 intervention policy",
    "focus_engine.outcomes": "Phase 11 outcome engine",
    "focus_engine.feedback": "Phase 12 feedback architecture",
    "focus_engine.evaluation": "Phase 13 evaluation framework",
    "focus_engine.api": "Phase 14 API",
}


def _imported_modules(source: Path) -> list[str]:
    """Return every ``focus_engine`` module the file imports at any depth."""
    text = source.read_text(encoding="utf-8")
    return re.findall(r"^\s*(?:from|import)\s+(focus_engine[\w.]*)", text, re.MULTILINE)


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_no_model_module_imports_a_later_layer(module: str) -> None:
    """No model module may import a layer that comes after Phase 8."""
    import focus_engine.models as package

    root = package.__file__
    assert root is not None
    source = Path(root).parent / f"{module}.py"
    assert source.is_file()

    for imported in _imported_modules(source):
        for later, phase in LATER_LAYERS.items():
            assert not (imported == later or imported.startswith(f"{later}.")), (
                f"models/{module}.py imports {imported}, reaching into the {phase}"
            )


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_model_modules_import_only_permitted_layers(module: str) -> None:
    """Every model-module import resolves to a permitted earlier or sibling layer."""
    import focus_engine.models as package

    root = package.__file__
    assert root is not None
    source = Path(root).parent / f"{module}.py"
    assert source.is_file()

    permitted_siblings = SELF_DEPENDENCIES[module]
    for imported in _imported_modules(source):
        if imported == "focus_engine" or imported.startswith("focus_engine.models"):
            if imported == "focus_engine.models":
                continue
            suffix = imported.removeprefix("focus_engine.models.")
            assert suffix in permitted_siblings, (
                f"models/{module}.py imports {imported}; "
                f"the declared internal dependencies are "
                f"{sorted(permitted_siblings) or 'none'}"
            )
            continue
        assert any(
            imported == base or imported.startswith(f"{base}.") for base in ALLOWED_DEPENDENCIES
        ), f"models/{module}.py imports {imported}, which is not a permitted dependency"


def test_the_models_package_exports_only_known_names() -> None:
    """``__all__`` is a contract, so an unexported or misspelled name is a real defect."""
    import focus_engine.models as package

    for name in package.__all__:
        assert hasattr(package, name), (
            f"models/__init__.py lists {name} in __all__ but does not bind it"
        )
    assert len(set(package.__all__)) == len(package.__all__), (
        "models/__init__.py lists a name in __all__ twice"
    )
