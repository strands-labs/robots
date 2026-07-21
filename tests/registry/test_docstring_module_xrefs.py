"""Registry modules must cross-reference sibling code by module, never by
source filename.

Referencing a source file (``loader.py``, ``robots.py``, ...) in a docstring is
documentation archaeology: the name breaks silently the moment a file is
renamed or split, and it points a reader at a path instead of an importable
symbol. The project convention (shared with
:mod:`strands_robots.policies` and the MuJoCo backend, guarded by
``tests/policies/test_docstring_module_xrefs.py`` and
``tests/simulation/mujoco/test_docstring_module_xrefs.py``) is to use Sphinx
cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` - that name the
actual API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring in the top-level
``strands_robots.registry`` modules (``__init__.py``, ``loader.py``,
``robots.py``, ``policies.py``, ``discovery.py``, ``user_registry.py``) and
fails if any embeds a ``<something>.py`` filename token. Data files
(``robots.json``, ``policies.json``) are exempt: they are payload, not
importable modules, and cannot be named with a ``:mod:`` role.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.registry as registry_pkg

# A bare source-filename token such as ``loader.py`` or ``robots.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(registry_pkg.__file__).parent


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


def test_registry_modules_scanned() -> None:
    """Guard: the scan actually walked the top-level registry modules."""
    scanned = {p.name for p in _PACKAGE_DIR.glob("*.py")}
    assert {"__init__.py", "loader.py", "robots.py", "policies.py"} <= scanned


def test_registry_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "Registry docstrings must cross-reference siblings by module "
        "(:mod:`~strands_robots.registry.loader`) not source filename "
        f"(``loader.py``). Offending docstrings: {offenders}"
    )
