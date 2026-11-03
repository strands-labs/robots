"""The package layout contracts of 'one concept, one package' (part 3.5, #4749 decision 14).

Every source file is small and single-purpose; an ``__init__.py`` only re-exports; there are no banner
comments; and a robot's name never appears as a string literal in package code (robot facts are
registry data). These are mechanical rules a reviewer should not have to police by eye.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "strands_robots"
_BANNER = re.compile(r"^\s*# .*-{3,}|^    # [a-z][a-z ,]+$")

# A driver binds to named vendor hardware, so a robot/model name is a legitimate dispatch key there:
# ``_DRIVERS`` (microduck) and unitree's ``_IDL`` (g1/go2/h1). Nowhere else may a name be a literal.
_DRIVER_NAME_KEY_FILES = {
    PKG / "robot" / "drivers" / "driver.py",
    PKG / "robot" / "drivers" / "unitree.py",
}


def _py_files():
    return sorted(p for p in PKG.rglob("*.py"))


def _robot_names():
    names = set()
    for fname in ("robots.json", "urdf_robots.json"):
        data = json.loads((PKG / "registry" / fname).read_text())["robots"]
        for key, entry in data.items():
            names.add(key)
            for alias in entry.get("aliases", ()) if isinstance(entry, dict) else ():
                names.add(alias.strip().lower().replace("-", "_"))
    return names


def test_no_source_file_exceeds_250_lines():
    big = {p.relative_to(PKG): len(p.read_text().splitlines()) for p in _py_files()}
    over = {p: n for p, n in big.items() if n > 250}
    assert not over, f"files over 250 lines: {over}"


def test_no_banner_comments():
    bad = {}
    for p in _py_files():
        hits = [i + 1 for i, line in enumerate(p.read_text().splitlines()) if _BANNER.match(line)]
        if hits:
            bad[p.relative_to(PKG)] = hits
    assert not bad, f"banner comments found: {bad}"


def test_every_init_only_re_exports():
    for p in _py_files():
        if p.name != "__init__.py":
            continue
        tree = ast.parse(p.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # docstring / bare string
            if isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.startswith("__") and t.id.endswith("__") for t in node.targets
            ):
                continue  # __all__, __version__ and other dunder constants
            raise AssertionError(f"{p.relative_to(PKG)} has logic in __init__: {ast.dump(node)[:80]}")


def test_no_robot_name_string_literal_in_package_code():
    names = _robot_names()
    bad = []
    for p in _py_files():
        if p in _DRIVER_NAME_KEY_FILES:
            continue
        tree = ast.parse(p.read_text())
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value.strip().lower().replace("-", "_") in names:
                    bad.append((str(p.relative_to(PKG)), node.lineno, node.value))
    assert not bad, f"robot-name string literals in package code: {bad}"
