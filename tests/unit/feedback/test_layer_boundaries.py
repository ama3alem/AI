"""Static boundary and import safety tests for the feedback layer."""

from __future__ import annotations

import ast
from pathlib import Path

FEEDBACK_SRC = Path(__file__).parents[3] / "src" / "focus_engine" / "feedback"


def _get_imports(file_path: Path) -> list[str]:
    with file_path.open(encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=str(file_path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                imports.append(imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_feedback_does_not_import_evaluation() -> None:
    """Feedback layer must not import the terminal evaluation layer."""
    for py_file in FEEDBACK_SRC.glob("*.py"):
        imports = _get_imports(py_file)
        for imported in imports:
            assert not imported.startswith("focus_engine.evaluation"), (
                f"{py_file.name} illegally imports evaluation layer: {imported}"
            )


def test_feedback_does_not_import_api_or_adapters() -> None:
    """Feedback engine must not import external api, lab, or foreign packages."""
    for py_file in FEEDBACK_SRC.glob("*.py"):
        imports = _get_imports(py_file)
        for imported in imports:
            assert not imported.startswith("api"), (
                f"{py_file.name} illegally imports api: {imported}"
            )
            assert not imported.startswith("eduflow"), (
                f"{py_file.name} illegally imports eduflow: {imported}"
            )
