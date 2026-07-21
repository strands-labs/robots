"""Regression: no runtime string may point the reader at an internal source file
by ``strands_robots/<path>.py`` path.

A user- or agent-facing message (an ``ImportError`` install hint, a help line, a
docstring "see also") should cite the importable *module* by its dotted path -
``strands_robots.policies.vera`` - not the source *file* that happens to define
it. A file path is a dead end: in a ``pip install`` the source lives buried in
``site-packages`` and cannot be opened by dotted import, whereas the module path
is exactly what the reader types to ``import`` it or read its ``__doc__``. This
mirrors the docstring-xref rule ("reference module paths, not filenames") but
for string *literals*, which reach the reader at runtime.

The scan walks every module under the ``strands_robots`` package, parses it, and
inspects string constants only (via AST, so ``# strands_robots/__init__.py``
maintainer comments about the install layout are exempt - they are not string
literals and never reach a user). It would have failed while the VERA server
runner's ``_require_vera_installed`` ImportError told the reader to
"See strands_robots/policies/vera/__init__.py for the full quickstart" instead
of citing the ``strands_robots.policies.vera`` module.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots

# An internal source-file path: the package name, a ``/``-separated module path,
# and a ``.py`` suffix. Dotted module references (``strands_robots.policies.vera``)
# carry no ``/`` or ``.py`` and are deliberately not matched, and upstream file
# names without the ``strands_robots/`` prefix (``run_mujoco_gear_wbc.py``) are
# out of scope - only self-references to this package's own sources are rejected.
_INTERNAL_FILE_PATH = re.compile(r"strands_robots/[A-Za-z0-9_/]+\.py\b")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _string_literals(path: Path) -> list[str]:
    """Every string constant in a module (docstrings and runtime strings alike)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_package_sources_discovered() -> None:
    """Guard: the scan actually walked the whole package, not one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "policies"} <= rel_dirs


def test_no_string_literal_cites_an_internal_source_file() -> None:
    """No runtime string may point the reader at a ``strands_robots/<path>.py`` file."""
    offenders: list[str] = []
    for path in _python_sources():
        for literal in _string_literals(path):
            for match in _INTERNAL_FILE_PATH.finditer(literal):
                offenders.append(f"{path.relative_to(_PACKAGE_DIR.parent)}: {match.group(0)}")
    assert not offenders, (
        "A string literal cites an internal source file by path. Cite the "
        "importable module by its dotted path (strands_robots.policies.vera), "
        "not the source file that defines it - a file path is a dead end for a "
        "reader who can only ``import`` the module:\n" + "\n".join(offenders)
    )


def test_internal_file_path_pattern_is_matched() -> None:
    """The pattern flags an internal source-file path in a runtime string."""
    assert _INTERNAL_FILE_PATH.search("See strands_robots/policies/vera/__init__.py for the quickstart.")
    assert _INTERNAL_FILE_PATH.search("strands_robots/simulation/mujoco/rendering.py")


def test_module_and_upstream_references_are_not_matched() -> None:
    """Dotted module paths and non-package file names must not trip the guard."""
    allowed = [
        "See the ``strands_robots.policies.vera`` module docstring.",  # dotted module ref
        "run_mujoco_gear_wbc.py:47-50",  # upstream file, no strands_robots/ prefix
        "import strands_robots.simulation.mujoco.rendering",  # dotted import, no .py
        "strands_robots.tools.robot_mesh",  # dotted, no slash / suffix
    ]
    for text in allowed:
        assert not _INTERNAL_FILE_PATH.search(text), f"should not be flagged: {text!r}"
