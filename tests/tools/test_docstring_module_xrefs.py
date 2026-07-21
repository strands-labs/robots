"""Agent-tool modules must cross-reference internal sibling code by module,
never by source filename.

Referencing an internal source file (``core.py``, ``factory.py``, ...) in a
docstring is documentation archaeology: the name breaks silently the moment a
file is renamed or split, and it points a reader at a path instead of an
importable symbol. The project convention (shared with
:mod:`strands_robots.policies`, :mod:`strands_robots.registry`, and
:mod:`strands_robots.assets`, guarded by their respective
``test_docstring_module_xrefs`` modules) is to use Sphinx cross-reference
roles - ``:mod:``, ``:class:``, ``:meth:``, ``:func:`` - that name the actual
API object, so the reference is checkable and survives refactors.

This guard walks every module/class/function docstring in the top-level
:mod:`strands_robots.tools` modules and fails if any embeds a
``<something>.py`` filename token *that names a real internal module*. Unlike
the sibling guards, it is deliberately scoped to internal stems: the
:mod:`strands_robots.tools.gr00t_inference` tool documents the upstream
Isaac-GR00T launcher scripts it shells out to (``inference_service.py``,
``embodiment_tags.py``) by filename. Those name real files in another
repository - not internal siblings - so a checkable module xref does not
exist for them and they are correctly left untouched.

It would have failed while ``tools/robot_mesh.py`` still cited the mesh
command handler as ``core.py:_on_cmd`` instead of
:meth:`~strands_robots.mesh.core.Mesh._on_cmd`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots
import strands_robots.tools as tools_pkg

# A bare source-filename token such as ``core.py`` or ``factory.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent
_TOOLS_DIR = Path(tools_pkg.__file__).resolve().parent


def _internal_module_stems() -> set[str]:
    """Stems (``core`` for ``core.py``) of every module inside strands_robots.

    A filename token in a docstring is only an internal-sibling reference - and
    thus an offender - when a module with that stem actually ships in the
    package. Tokens like ``inference_service.py`` that name upstream external
    scripts have no internal stem and are left alone.
    """
    return {p.stem for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts}


def _docstrings_with_offenders(internal_stems: set[str]) -> dict[str, list[str]]:
    """Map ``module.py::qualname`` -> internal filename tokens in that docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_TOOLS_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            hits = [tok for tok in _FILENAME_RE.findall(doc) if tok.removesuffix(".py") in internal_stems]
            if hits:
                qualname = getattr(node, "name", "<module>")
                offenders[f"{source_file.name}::{qualname}"] = hits
    return offenders


def test_tool_modules_scanned() -> None:
    """Guard the guard: the scan walked the top-level tool modules."""
    scanned = {p.name for p in _TOOLS_DIR.glob("*.py")}
    assert {"robot_mesh.py", "gr00t_inference.py", "run_policy.py"} <= scanned


def test_internal_stem_set_is_populated() -> None:
    """Sanity: the internal-stem allowlist basis is non-trivial and includes core."""
    stems = _internal_module_stems()
    assert "core" in stems  # strands_robots/mesh/core.py
    assert len(stems) > 50


def test_external_launcher_scripts_are_not_flagged() -> None:
    """Upstream Isaac-GR00T script filenames have no internal stem, so the
    internal-only rule must not flag them even though they end in ``.py``."""
    internal = _internal_module_stems()
    assert "inference_service" not in internal
    assert "embodiment_tags" not in internal


def test_tool_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders(_internal_module_stems())
    assert not offenders, (
        "Tool docstrings must cross-reference internal siblings by module "
        "(:meth:`~strands_robots.mesh.core.Mesh._on_cmd`) not source filename "
        f"(``core.py``). Offending docstrings: {offenders}"
    )
