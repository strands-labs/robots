"""Benchmark-adapter modules must cross-reference internal sibling code by
module, never by source filename.

Referencing an internal source file (``rendering.py``, ``core.py``, ...) in a
docstring is documentation archaeology: the name breaks silently the moment a
file is renamed or split, and it points a reader at a path instead of an
importable symbol. The project convention (shared with
:mod:`strands_robots.tools`, :mod:`strands_robots.policies`,
:mod:`strands_robots.registry`, and :mod:`strands_robots.assets`, guarded by
their respective ``test_docstring_module_xrefs`` modules) is to use Sphinx
cross-reference roles - ``:mod:``, ``:class:``, ``:meth:``, ``:func:`` - that
name the actual API object, so the reference is checkable and survives
refactors.

This guard walks every module/class/function docstring under
:mod:`strands_robots.benchmarks` (recursively) and fails if any embeds a
``<something>.py`` filename token *that names a real internal module*. Like the
sibling :mod:`strands_robots.tools` guard it is deliberately scoped to internal
stems: the LIBERO adapter documents the upstream RoboSuite / LIBERO scripts it
integrates with (``libero_mujoco.py``, ``single_arm.py``, ``binding_utils.py``,
``env_libero.py``, ``rollout_policy.py``, ``coverage_support.py``) by filename.
Those name real files in other repositories - not internal siblings - so a
checkable module xref does not exist for them and they are correctly left
untouched.

``__init__`` is excluded from the offender stems: a package always ships an
``__init__.py`` so the stem is "internal", but a docstring never cross-refers
the dunder module itself (you reference the package, e.g.
:mod:`strands_robots.benchmarks.libero`). The LIBERO suite loader legitimately
cites the *upstream* ``libero/__init__.py`` install layout by path, and that
external reference must not be flagged.

It would have failed while ``benchmarks/libero/adapter.py`` still cited the
MuJoCo render path as ``simulation/mujoco/rendering.py`` instead of
:mod:`strands_robots.simulation.mujoco.rendering`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots
import strands_robots.benchmarks as benchmarks_pkg

# A bare source-filename token such as ``rendering.py`` or ``adapter.py``.
_FILENAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b")

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent
_BENCHMARKS_DIR = Path(benchmarks_pkg.__file__).resolve().parent


def _internal_module_stems() -> set[str]:
    """Stems (``rendering`` for ``rendering.py``) of every module in the package.

    A filename token in a docstring is only an internal-sibling reference - and
    thus an offender - when a module with that stem actually ships in
    ``strands_robots``. Tokens like ``libero_mujoco.py`` that name upstream
    external scripts have no internal stem and are left alone. ``__init__`` is
    dropped: it is technically an internal stem but is never a meaningful xref
    target, and upstream package install layouts are cited by ``__init__.py``.
    """
    stems = {p.stem for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts}
    stems.discard("__init__")
    return stems


def _docstrings_with_offenders(internal_stems: set[str]) -> dict[str, list[str]]:
    """Map ``<relpath>::qualname`` -> internal filename tokens in that docstring."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_BENCHMARKS_DIR.rglob("*.py")):
        if "__pycache__" in source_file.parts:
            continue
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
                rel = source_file.relative_to(_BENCHMARKS_DIR)
                offenders[f"{rel}::{qualname}"] = hits
    return offenders


def test_benchmark_modules_scanned() -> None:
    """Guard the guard: the recursive scan reached the LIBERO adapter + suite."""
    scanned = {str(p.relative_to(_BENCHMARKS_DIR)) for p in _BENCHMARKS_DIR.rglob("*.py")}
    assert "libero/adapter.py" in scanned
    assert "libero/suite.py" in scanned


def test_internal_stem_set_excludes_init_and_includes_rendering() -> None:
    """Sanity: the offender basis includes a real sibling module but not ``__init__``."""
    stems = _internal_module_stems()
    assert "rendering" in stems  # strands_robots/simulation/mujoco/rendering.py
    assert "__init__" not in stems
    assert len(stems) > 50


def test_upstream_libero_scripts_are_not_flagged() -> None:
    """Upstream RoboSuite / LIBERO script filenames have no internal stem, so
    the internal-only rule must not flag them even though they end in ``.py``."""
    internal = _internal_module_stems()
    for external in ("libero_mujoco", "single_arm", "binding_utils", "env_libero", "rollout_policy"):
        assert external not in internal


def test_benchmark_docstrings_reference_modules_not_filenames() -> None:
    offenders = _docstrings_with_offenders(_internal_module_stems())
    assert not offenders, (
        "Benchmark docstrings must cross-reference internal siblings by module "
        "(:mod:`strands_robots.simulation.mujoco.rendering`) not source filename "
        f"(``rendering.py``). Offending docstrings: {offenders}"
    )
