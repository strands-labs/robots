"""Top-level trainer modules must cross-reference *sibling* code by module, never
by source filename.

Citing a sibling source file (``mock.py``, ``factory.py``, ...) in a docstring is
documentation archaeology: the name breaks silently the moment a file is renamed
or split, and it points a reader at a path instead of an importable symbol. The
project convention (already guarded for the policy package by
``tests/policies/test_docstring_module_xrefs.py`` and for the MuJoCo backend by
``tests/simulation/mujoco/test_docstring_module_xrefs.py``) is to use Sphinx
cross-reference roles - ``:mod:``, ``:class:``, ``:func:`` - that name the actual
API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring in the *top-level*
``strands_robots.training`` modules (``base.py``, ``factory.py``, ``mock.py``,
``reward.py``, ...) and fails if any embeds a ``<name>.py`` token that names an
actual sibling module. It is intentionally sibling-aware rather than flagging
every ``.py`` token: the provider trainers legitimately cite *upstream*
reference scripts by filename (``launch_finetune.py`` from GR00T, ``train.py``
from the Cosmos framework, ``parser.py`` from lerobot), which name real files in
other repositories and are not internal siblings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.training as training_pkg

# A bare source-filename token such as ``mock.py`` or ``factory.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(training_pkg.__file__).parent
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


def test_top_level_training_modules_scanned() -> None:
    """Guard: the scan actually walked the top-level training modules."""
    assert {"base.py", "factory.py", "mock.py"} <= _SIBLING_MODULES


def test_training_docstrings_reference_modules_not_sibling_filenames() -> None:
    offenders = _docstrings_with_sibling_filename_refs()
    assert not offenders, (
        "Top-level training docstrings must cross-reference siblings by module "
        "(:class:`~strands_robots.training.mock.MockTrainer`) not source filename "
        f"(``mock.py``). Offending docstrings: {offenders}"
    )
