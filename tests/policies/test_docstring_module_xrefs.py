"""Top-level policy modules must cross-reference sibling code by module, never by
source filename.

Referencing a source file (``mock.py``, ``factory.py``, ...) in a docstring is
documentation archaeology: the name breaks silently the moment a file is
renamed or split, and it points a reader at a path instead of an importable
symbol. The project convention (see the cosmos3 policy package and the MuJoCo
backend, guarded by ``tests/simulation/mujoco/test_docstring_module_xrefs.py``)
is to use Sphinx cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` -
that name the actual API object, so the reference is checkable and survives
refactors.

This guard walks every module/class/function docstring in the *top-level*
``strands_robots.policies`` modules (``base.py``, ``factory.py``, ``mock.py``,
``composite.py``, ``registry.py``, ...) and fails if any embeds a
``<something>.py`` filename token. It is intentionally non-recursive: the
provider subpackages (``wbc/``, ``lerobot_local/``, ``cosmos3/``,
``motionbricks/``, ...) legitimately cite *upstream* reference scripts by
filename (``run_mujoco_gear_wbc.py``, ``launch_finetune.py``), which name real
files in other repositories and are not internal siblings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.policies as policies_pkg

# A bare source-filename token such as ``mock.py`` or ``factory.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(policies_pkg.__file__).parent


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


def test_top_level_policy_modules_scanned() -> None:
    """Guard: the scan actually walked the top-level policy modules."""
    scanned = {p.name for p in _PACKAGE_DIR.glob("*.py")}
    assert {"base.py", "factory.py", "mock.py"} <= scanned


def test_policy_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "Top-level policy docstrings must cross-reference siblings by module "
        "(:class:`~strands_robots.policies.mock.MockPolicy`) not source filename "
        f"(``mock.py``). Offending docstrings: {offenders}"
    )
