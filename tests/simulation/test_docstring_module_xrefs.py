"""Top-level simulation modules must cross-reference sibling code by module,
never by source filename.

Referencing an internal sibling by its source file (``base.py``, ``factory.py``,
...) in a docstring is documentation archaeology: the name breaks silently the
moment a file is renamed or split, and it points a reader at a path instead of
an importable symbol. The project convention (see the MuJoCo backend guard,
``tests/simulation/mujoco/test_docstring_module_xrefs.py``, and the policy and
training guards) is to use Sphinx cross-reference roles - ``:mod:``, ``:class:``,
``:func:`` - that name the actual API object, so the reference is checkable and
survives refactors.

This guard walks every module/class/function docstring in the *top-level*
``strands_robots.simulation`` modules and fails if any embeds a ``<name>.py``
token that names a real sibling module in the package directory. The scan is
intentionally restricted to sibling filenames: top-level modules legitimately
cite *upstream* reference scripts by filename (LIBERO's ``bddl_base_domain.py``,
GR00T's ``standalone_inference_script.py``), which name real files in other
repositories and are not internal siblings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.simulation as simulation_pkg

# A bare source-filename token such as ``base.py`` or ``factory.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(simulation_pkg.__file__).parent

# The set of real top-level sibling modules. Only filename tokens naming one of
# these are internal archaeology; anything else is an external upstream file.
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


def test_top_level_simulation_modules_scanned() -> None:
    """Guard: the scan actually walked the top-level simulation modules."""
    assert {"base.py", "factory.py", "models.py"} <= _SIBLING_MODULES


def test_simulation_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "Top-level simulation docstrings must cross-reference siblings by module "
        "(:class:`~strands_robots.simulation.base.SimEngine`) not source filename "
        f"(``base.py``). Offending docstrings: {offenders}"
    )
