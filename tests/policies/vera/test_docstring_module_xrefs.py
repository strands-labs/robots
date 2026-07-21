"""VERA provider modules must cross-reference sibling code by module, never by a
bare source filename.

Referencing an internal sibling by its source file (``sim_ik.py``,
``client.py``, ...) in a docstring is documentation archaeology: the name breaks
silently the moment a file is renamed or split, and it points a reader at a path
instead of an importable symbol. It is also ambiguous here - the VERA provider
mirrors several ``cosmos3`` siblings that share a basename (both packages ship a
``sim_ik.py`` and a ``_msgpack_numpy.py``), so a bare ``sim_ik.py`` cannot tell
the reader whether the VERA or the cosmos3 module is meant. The project
convention (see :mod:`tests.simulation.mujoco.test_docstring_module_xrefs` and
the Isaac backend guard) is to use Sphinx cross-reference roles - ``:mod:``,
``:class:``, ``:func:`` - that name the actual API object, so the reference is
checkable and survives refactors.

This guard walks every module/class/function docstring in the
``strands_robots.policies.vera`` package and fails if any embeds a *bare*
``<name>.py`` token that names a real sibling module in the package directory.

The scan is deliberately restricted twice over:

* **Sibling filenames only.** VERA modules legitimately cite *upstream*
  reference scripts by filename (openpi / DreamZero server scripts) that name
  real files in other repositories and are not internal siblings.
* **Bare tokens only.** A path-qualified reference such as
  ``vera/server/protocol/_msgpack_numpy.py`` names the external VERA server file
  and is unambiguous even though its basename collides with a local sibling, so
  it is not flagged.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.policies.vera as vera_pkg

# A bare source-filename token (``sim_ik.py``) - but NOT one that is part of a
# longer path (``.../protocol/_msgpack_numpy.py``) or a dotted attribute, which
# the negative lookbehind for ``/``, ``.`` and word chars excludes.
_FILENAME_RE = re.compile(r"(?<![\w/.])[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_DIR = Path(vera_pkg.__file__).parent

# Real sibling modules. Only bare filename tokens naming one of these are
# internal archaeology; anything else is an external upstream file.
_SIBLING_MODULES = {p.name for p in _PACKAGE_DIR.glob("*.py")}


def _docstrings_with_offenders() -> dict[str, list[str]]:
    """Map ``module.py::qualname`` -> bare sibling filename tokens in that docstring."""
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


def test_vera_modules_scanned() -> None:
    """Guard: the scan actually walked the VERA package modules."""
    assert {"sim_ik.py", "client.py", "provider.py"} <= _SIBLING_MODULES


def test_path_qualified_upstream_reference_is_not_flagged() -> None:
    """A path-qualified external ref keeps its basename off the offender list."""
    doc = "Mirrors ``vera/server/protocol/_msgpack_numpy.py`` (openpi port)."
    hits = [h for h in _FILENAME_RE.findall(doc) if h in _SIBLING_MODULES]
    assert hits == []


def test_bare_sibling_reference_would_be_flagged() -> None:
    """A bare sibling token is exactly what the guard is meant to catch."""
    doc = "Copied from the cosmos3 ``sim_ik.py``."
    hits = [h for h in _FILENAME_RE.findall(doc) if h in _SIBLING_MODULES]
    assert hits == ["sim_ik.py"]


def test_vera_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders()
    assert not offenders, (
        "VERA docstrings must cross-reference siblings by module "
        "(:mod:`~strands_robots.policies.cosmos3.sim_ik`) not a bare source "
        f"filename (``sim_ik.py``). Offending docstrings: {offenders}"
    )
