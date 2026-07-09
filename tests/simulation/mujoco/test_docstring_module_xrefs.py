"""Docstrings in the MuJoCo backend package must cross-reference sibling code by
module, never by source filename.

Referencing a source file (``scene_ops.py``, ``simulation.py``, ...) in a
docstring is documentation archaeology: the name breaks silently the moment a
file is renamed or split, and it points a reader at a path instead of an
importable symbol. The project convention (see the cosmos3 policy package) is
to use Sphinx cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` - that
name the actual API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring under
``strands_robots.simulation.mujoco`` and fails if any of them embeds a
``<something>.py`` filename token.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.simulation.mujoco as mujoco_pkg

# A bare source-filename token such as ``scene_ops.py`` or ``run_agent.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(mujoco_pkg.__file__).parent


def _docstrings_with_offenders() -> dict[str, list[str]]:
    """Map ``module.py::qualname`` -> filename tokens found in that docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            hits = _FILENAME_RE.findall(doc)
            if hits:
                qualname = getattr(node, "name", "<module>")
                offenders[f"{source_file.name}::{qualname}"] = hits
    return offenders


def test_mujoco_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "MuJoCo backend docstrings must cross-reference siblings by module "
        "(:mod:`scene_ops`) not source filename (``scene_ops.py``). Offending "
        f"docstrings: {offenders}"
    )
