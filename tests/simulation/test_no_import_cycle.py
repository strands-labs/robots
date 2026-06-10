"""Regression: no RUNTIME import cycles inside strands_robots.

Before: /tmp/ast-analysis/DEEPER_FINDINGS.md hazard A flagged
`simulation.base ↔ simulation.policy_runner` - papered over by three
inline lazy imports inside SimEngine methods. These were removed in
the concurrency-audit pass and the imports hoisted to module level,
exploiting the fact that policy_runner only imports SimEngine under
TYPE_CHECKING (so the cycle is a compile-time artifact, not runtime).

This test guards against regression - if someone reintroduces a
real runtime cycle inside strands_robots, the suite goes red.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]
else:
    nx = pytest.importorskip("networkx")  # dev-only dep; skip cleanly when absent

PKG = Path(__file__).resolve().parents[2] / "strands_robots"


def _is_in_type_checking(tree: ast.AST, target: ast.AST) -> bool:
    """True if target_node is inside an `if TYPE_CHECKING:` block."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            ):
                for child in ast.walk(node):
                    if child is target:
                        return True
    return False


def _is_inside_function(tree: ast.Module, target: ast.AST) -> bool:
    """True if target_node is inside a function or method body (lazy import).

    Imports inside function/method bodies are deferred — they execute only
    when the function is called, not at module import time. These cannot
    cause import-time cycles and should not be flagged.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


def _build_import_graph(root: Path) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        mod = ".".join(p.relative_to(root.parent).with_suffix("").parts)
        G.add_node(mod)
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("strands_robots"):
                if _is_in_type_checking(tree, n):
                    continue
                if _is_inside_function(tree, n):
                    continue
                G.add_edge(mod, n.module)
            elif isinstance(n, ast.Import):
                if _is_in_type_checking(tree, n):
                    continue
                if _is_inside_function(tree, n):
                    continue
                for alias in n.names:
                    if alias.name.startswith("strands_robots"):
                        G.add_edge(mod, alias.name)
    return G


def test_no_runtime_import_cycles():
    """Zero runtime import-time cycles.

    Only module-level imports are considered. Imports inside function/method
    bodies (lazy imports) and TYPE_CHECKING blocks are excluded since they
    cannot cause import-time circular dependency failures.
    """
    G = _build_import_graph(PKG)
    cycles = list(nx.simple_cycles(G))
    assert cycles == [], "runtime cycles detected:\n" + "\n".join("  " + " -> ".join(c) + " -> " + c[0] for c in cycles)


def test_base_does_not_lazy_import_policy_runner():
    """The three inline lazy imports were the symptom of the prior cycle.
    They've been hoisted to module level; don't let them sneak back in."""
    base_src = (PKG / "simulation/base.py").read_text()
    # Count occurrences of the lazy pattern
    lazy_pattern = "from strands_robots.simulation.policy_runner import"
    # Module-level import: 1 occurrence. Any >1 would be a lazy reintroduction.
    n = base_src.count(lazy_pattern)
    assert n == 1, (
        f"expected exactly 1 module-level import of policy_runner in base.py, got {n}. "
        "Did someone reintroduce an inline lazy import?"
    )
