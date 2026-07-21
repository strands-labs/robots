"""Isaac backend modules must cross-reference sibling code by module, never by
source filename.

Referencing an internal sibling by its source file (``procedural.py``,
``loaders.py``, ...) in a docstring is documentation archaeology: the name
breaks silently the moment a file is renamed or split, and it points a reader
at a path instead of an importable symbol. The project convention (see the
top-level simulation guard, ``tests/simulation/test_docstring_module_xrefs.py``,
and the MuJoCo backend guard) is to use Sphinx cross-reference roles -
``:mod:``, ``:class:``, ``:func:`` - that name the actual API object, so the
reference is checkable and survives refactors.

This guard walks every module/class/function docstring in the
``strands_robots.simulation.isaac`` package modules and fails if any embeds a
``<name>.py`` token that names a real sibling module in the package directory.
The scan is intentionally restricted to sibling filenames: Isaac modules
legitimately cite *upstream* reference scripts by filename (Isaac Lab / Isaac
Sim standalone scripts), which name real files in other repositories and are
not internal siblings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.simulation.isaac as isaac_pkg

# A bare source-filename token such as ``procedural.py`` or ``loaders.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(isaac_pkg.__file__).parent

# The set of real sibling modules. Only filename tokens naming one of these are
# internal archaeology; anything else is an external upstream file.
_SIBLING_MODULES = {p.name for p in _PACKAGE_DIR.glob("*.py")}


def _docstrings_with_offenders() -> dict[str, list[str]]:
    """Map ``module.py::qualname`` -> sibling filename tokens found in that docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            hits = [h for h in _FILENAME_RE.findall(doc) if h in _SIBLING_MODULES]
            if hits:
                qualname = getattr(node, "name", "<module>")
                offenders[f"{source_file.name}::{qualname}"] = hits
    return offenders


def test_isaac_modules_scanned() -> None:
    """Guard: the scan actually walked the Isaac package modules."""
    assert {"loaders.py", "procedural.py", "simulation.py"} <= _SIBLING_MODULES


def test_isaac_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "Isaac backend docstrings must cross-reference siblings by module "
        "(:mod:`strands_robots.simulation.isaac.procedural`) not source filename "
        f"(``procedural.py``). Offending docstrings: {offenders}"
    )
