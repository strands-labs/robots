"""Top-level mesh modules must cross-reference *sibling* code by module, never by
source filename.

Citing a sibling source file (``core.py``, ``security.py``, ...) in a docstring
is documentation archaeology: the name breaks silently the moment a file is
renamed or split, and it points a reader at a path instead of an importable
symbol. This package already carried a *dead* example - an ``_ProcessAuditState``
docstring pointed at ``mesh/security.py::_ProcessSecurityState``, a class that no
longer exists - which is exactly the silent-breakage this rule prevents. The
project convention (already guarded for the policy package by
``tests/policies/test_docstring_module_xrefs.py``, the trainer package by
``tests/training/test_docstring_module_xrefs.py``, and the MuJoCo backend by
``tests/simulation/mujoco/test_docstring_module_xrefs.py``) is to use Sphinx
cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` - that name the actual
API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring in the *top-level*
``strands_robots.mesh`` modules (``core.py``, ``security.py``, ``audit.py``, ...)
and fails if any embeds a ``<name>.py`` token that names an actual sibling
module. It is intentionally sibling-aware rather than flagging every ``.py``
token: mesh docstrings legitimately cite *test* modules by filename (the guarding
``test_*.py`` files that pin a behavior), which live under ``tests/`` and are not
importable siblings of the package.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.mesh as mesh_pkg

# A bare source-filename token such as ``core.py`` or ``security.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(mesh_pkg.__file__).parent
_SIBLING_MODULES = {p.name for p in _PACKAGE_DIR.glob("*.py")}


def _docstrings_with_sibling_filename_refs() -> dict[str, list[str]]:
    """Map ``module.py::qualname`` -> sibling-filename tokens found in its docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            hits = [tok for tok in _FILENAME_RE.findall(doc) if tok in _SIBLING_MODULES]
            if hits:
                qualname = getattr(node, "name", "<module>")
                offenders[f"{source_file.name}::{qualname}"] = hits
    return offenders


def test_top_level_mesh_modules_scanned() -> None:
    """Guard: the scan actually walked the top-level mesh modules."""
    assert {"core.py", "security.py", "audit.py"} <= _SIBLING_MODULES


def test_mesh_docstrings_reference_modules_not_sibling_filenames() -> None:
    offenders = _docstrings_with_sibling_filename_refs()
    assert not offenders, (
        "Top-level mesh docstrings must cross-reference siblings by module "
        "(:func:`~strands_robots.mesh.core._parse_positive_float_env`) not source "
        f"filename (``core.py``). Offending docstrings: {offenders}"
    )
