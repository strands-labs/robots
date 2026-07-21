"""Guard: fully-qualified ``strands_robots`` Sphinx cross-references must resolve.

A docstring role such as :func:`~strands_robots.simulation.predicates.base_below_z`
promises the reader an importable API object at that exact dotted path. When the
path is wrong - a private implementation renamed public, a symbol moved or split,
or, as happened here, a *registered predicate name* dressed up as an importable
function that never existed - the pointer is a silent dead end: Sphinx renders a
plain-text token, IDEs cannot jump to it, and the reader chases a path that
does not import.

The sibling filename guards (for example
:mod:`tests.mesh.test_docstring_module_xrefs`) already forbid citing a source
*file* by name; this guard closes the complementary gap for the *recommended*
form - the ``:mod:`` / ``:class:`` / ``:func:`` / ``:meth:`` roles that name a
dotted API path - by verifying that every fully-qualified ``strands_robots.*``
target actually resolves to a real module or attribute.

Scope is deliberately conservative: only targets that start with
``strands_robots.`` are checked. Unqualified roles (``:func:`reset```) and roles
into third-party packages resolve against Sphinx's current-module context or an
intersphinx inventory that is not available here, so flagging them would produce
false positives without catching the rot this guard exists to prevent.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

# Sphinx cross-reference roles naming a dotted Python target, optionally with a
# leading ``~`` (display-shortening) tilde.
_ROLE_RE = re.compile(r":(?:mod|class|func|meth|attr|data|obj|exc):`~?([A-Za-z_][\w.]*)`")


def _resolves(target: str) -> bool:
    """True if ``target`` names a real ``strands_robots`` module or attribute.

    Imports the longest importable module prefix, then walks the remaining
    dotted components as attributes (so ``pkg.mod.Class.method`` resolves).
    """
    parts = target.split(".")
    module = None
    consumed = 0
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            consumed = i
            break
        except Exception:
            continue
    if module is None:
        return False
    obj = module
    for attr in parts[consumed:]:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True


def _unresolved_xref_roles() -> dict[str, list[str]]:
    """Map ``relpath::qualname`` -> list of unresolved ``strands_robots.*`` roles."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PKG_ROOT.rglob("*.py")):
        if "__pycache__" in source_file.parts:
            continue
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            bad = [t for t in _ROLE_RE.findall(doc) if t.startswith("strands_robots.") and not _resolves(t)]
            if bad:
                qualname = getattr(node, "name", "<module>")
                rel = source_file.relative_to(_PKG_ROOT)
                offenders[f"{rel}::{qualname}"] = bad
    return offenders


def test_qualified_strands_robots_xref_roles_resolve() -> None:
    offenders = _unresolved_xref_roles()
    assert not offenders, (
        "docstring cross-reference roles must name a real importable object. "
        "Cite a registered predicate/backend by its literal name (``base_below_z``) "
        "rather than a :func:`...` role, and reference actual API objects with "
        ":mod:/:class:/:func:/:meth:. Offending docstrings: " + repr(offenders)
    )


def test_guard_resolver_accepts_real_symbol_and_rejects_bogus() -> None:
    """The resolver walks module.attr chains and rejects nonexistent paths."""
    assert _resolves("strands_robots.simulation.base.SimEngine.get_observation")
    assert not _resolves("strands_robots.simulation.predicates.base_below_z")
