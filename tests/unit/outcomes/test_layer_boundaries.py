"""Static dependency checks for the outcome layer (Phase 11).

The outcome layer is the last layer that *measures* before the project stops measuring and
starts learning. What it must not import is anything that concludes: feedback, evaluation, or
API. The edge that matters most is the feedback layer. An outcome record is a before/after
measurement of an event stream, and the only thing standing between it and a causal claim is
that the claim is a Phase 12 decision made on the other side of a wall. A feedback import
reaching into this layer would put the learning inside the measurement, where it would be
invisible in review because the measurement code would still look like measurement code.

The policy edge is the one that is *allowed* and therefore the easiest to overstate. The
outcome layer reads the policy layer's vocabulary - the outcome-class labels and the
version of the rules that chose the intervention - and nothing else. It must not reach into
``policy.engine`` and ask what the policy would have done, because a measurement computed
under the counterfactual of a decision that was not made is a different claim, and a
plausible-sounding one.

The check is static rather than behavioural, because an import that only appears on a
rarely-taken branch will not be exercised by any test, and a test suite that depends on
reaching a line to notice a violation is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Layers the outcome layer is permitted to depend on. The configuration layer supplies the
#: window widths and the sample floor; the events layer supplies the stream being read; the
#: policy layer supplies the response-class vocabulary and the policy version; the schemas
#: layer supplies provenance and version primitives; the temporal layer supplies the state
#: the trajectory reading reuses rather than reimplements; the utility layer supplies the
#: clock and the content digest. Every later layer is absent by design.
ALLOWED_DEPENDENCIES = frozenset(
    {
        "focus_engine.configuration",
        "focus_engine.events",
        "focus_engine.policy",
        "focus_engine.schemas",
        "focus_engine.temporal",
        "focus_engine.utils",
    }
)

#: Internal edges, read off the source rather than assumed. ``engine`` composes the measures
#: and the models, ``measures`` depends on the models it constructs, and ``models`` depends
#: on nothing inside the layer: the record shape is the bottom of the stack.
SELF_DEPENDENCIES = {
    "engine": frozenset({"measures", "models"}),
    "measures": frozenset({"models"}),
    "models": frozenset(),
    "__init__": frozenset({"engine", "measures", "models"}),
}

#: Later layers, named so the failure message says which phase was reached into.
LATER_LAYERS = {
    "focus_engine.feedback": "Phase 12 feedback architecture",
    "focus_engine.evaluation": "Phase 13 evaluation framework",
    "focus_engine.api": "Phase 14 API",
}

#: The policy submodules this layer may import. ``history`` supplies the delivery record and
#: the outcome-class vocabulary; ``models`` supplies the policy version constant. The policy
#: *engine* is absent from this set deliberately, and that absence is the interesting part:
#: this layer measures what happened after a delivery, and asking the policy engine what it
#: would have done instead is the counterfactual claim the layer refuses to make.
ALLOWED_POLICY_SUBMODULES = frozenset({"history", "models"})


def _imported_modules(source: Path) -> list[str]:
    """Return every ``focus_engine`` module the file imports at any depth.

    Args:
        source: The file to read.

    Returns:
        The imported dotted module paths, in the order they appear.
    """
    text = source.read_text(encoding="utf-8")
    return re.findall(r"^\s*(?:from|import)\s+(focus_engine[\w.]*)", text, re.MULTILINE)


def _module_source(module: str) -> Path:
    """Locate an outcome module *without importing the package*.

    Importing it to find its path looks harmless but is exactly the failure this check
    exists to catch. A forbidden import of a module that does not exist yet is an
    ``ImportError`` at collection time, so every test in the file errors out with a stack
    trace naming the missing module instead of one test failing with a sentence naming the
    layer boundary that was crossed. The path is derived from this file's own location, so
    the check reads the source as text and nothing has to be executable.

    Args:
        module: The module file name, without the ``.py``.

    Returns:
        The path to the module.
    """
    source = (
        Path(__file__).resolve().parents[3] / "src" / "focus_engine" / "outcomes" / f"{module}.py"
    )
    assert source.is_file(), f"expected the outcome layer to contain {module}.py"
    return source


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_no_outcome_module_imports_a_later_layer(module: str) -> None:
    """No outcome module may import feedback, evaluation, or API."""
    for imported in _imported_modules(_module_source(module)):
        for later, phase in LATER_LAYERS.items():
            assert not (imported == later or imported.startswith(f"{later}.")), (
                f"outcomes/{module}.py imports {imported}, reaching into the {phase}"
            )


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_outcome_modules_import_only_permitted_layers(module: str) -> None:
    """Every outcome-module import resolves to a permitted earlier or sibling layer."""
    permitted_siblings = SELF_DEPENDENCIES[module]
    for imported in _imported_modules(_module_source(module)):
        if imported == "focus_engine" or imported.startswith("focus_engine.outcomes"):
            if imported == "focus_engine.outcomes":
                continue
            suffix = imported.removeprefix("focus_engine.outcomes.")
            assert suffix in permitted_siblings, (
                f"outcomes/{module}.py imports {imported}; "
                f"the declared internal dependencies are "
                f"{sorted(permitted_siblings) or 'none'}"
            )
            continue
        assert any(
            imported == base or imported.startswith(f"{base}.") for base in ALLOWED_DEPENDENCIES
        ), f"outcomes/{module}.py imports {imported}, which is not a permitted dependency"


@pytest.mark.parametrize("module", sorted(SELF_DEPENDENCIES))
def test_no_outcome_module_asks_the_policy_engine_what_it_would_have_done(module: str) -> None:
    """The policy engine is not a permitted dependency, however it is spelled.

    The policy layer is on the allowed list, so a bare "no later layer" check passes a file
    that imports ``policy.engine`` freely. The counterfactual this layer would be reading is
    the one claim it does not make: what would have happened had the policy not intervened.
    """
    for imported in _imported_modules(_module_source(module)):
        if not imported.startswith("focus_engine.policy"):
            continue
        suffix = imported.removeprefix("focus_engine.policy.").rstrip(".")
        assert suffix in ALLOWED_POLICY_SUBMODULES, (
            f"outcomes/{module}.py imports {imported}. Only "
            f"{sorted(ALLOWED_POLICY_SUBMODULES)} are readable from the policy layer; the "
            "policy engine computes decisions and counterfactuals, and a measurement that "
            "reads one is making a claim the events do not support."
        )


def test_the_outcomes_package_exports_only_known_names() -> None:
    """``__all__`` is a contract, so an unexported or misspelled name is a real defect."""
    import focus_engine.outcomes as package

    for name in package.__all__:
        assert hasattr(package, name), (
            f"outcomes/__init__.py lists {name} in __all__ but does not bind it"
        )
    assert len(set(package.__all__)) == len(package.__all__), (
        "outcomes/__init__.py lists a name in __all__ twice"
    )


def test_the_package_exports_its_composition_point_and_its_evidence_type() -> None:
    """A caller must be able to build an engine and read the record it produces.

    ``__init__`` must export ``OutcomeEngine`` and ``OutcomeRecord`` together. An engine
    reachable only by a private module path is an invitation to reach past the package, and
    past the package is where the layer boundary stops being enforced by anything.
    """
    import focus_engine.outcomes as package

    assert {"OutcomeEngine", "OutcomeRecord", "Measurement", "OutcomeMeasure"} <= set(
        package.__all__
    )


def test_the_outcome_layer_never_claims_a_causal_provenance() -> None:
    """The record's provenance field is constrained to ``OBSERVED`` at the type.

    The wall between an observed measurement and a causal claim is enforced by the
    constructor rather than by convention, and this checks the source says so. A reader who
    wants to know whether the boundary is real should be able to find the refusal, and a
    model that had quietly grown the ability to be labelled ``ground_truth`` would otherwise
    look identical from the outside.
    """
    source = _module_source("models").read_text(encoding="utf-8")

    assert "is not Provenance.OBSERVED" in source
    assert "causal claim" in source


def test_the_policy_boundary_test_names_the_outcome_layer() -> None:
    """The boundary has to be declared from both sides to be a boundary.

    Phase 10's check forbids the policy layer from importing the outcome engine. If the
    outcome layer's own check did not exist, the edge would be guarded in one direction only,
    and the unguarded direction is the one that gets taken.
    """
    boundary = Path("tests/unit/policy/test_layer_boundaries.py").resolve()
    assert boundary.exists()
    text = boundary.read_text(encoding="utf-8")
    assert "focus_engine.outcomes" in text
    assert "Phase 11" in text
