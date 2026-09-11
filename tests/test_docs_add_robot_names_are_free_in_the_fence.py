"""A documented ``add_robot`` must not reuse a name the fence already took.

``Robot("so100")`` registers the arm it builds under the name ``"so100"``
(``strands_robots/robot.py`` forwards ``name=`` to ``SimEngine.add_robot``), and
``add_robot`` refuses a second robot under a taken name with ``status=error``
rather than raising. A fence that then calls ``add_robot(name="so100", ...)`` for
a "second arm" therefore builds a one-arm world, and every line below it in the
example runs against a scene the reader was not shown.

This grades every ``python`` fence in ``docs/**/*.md`` and ``README.md``: the
names ``Robot(...)`` and ``add_robot(...)`` register inside one fence must be
pairwise distinct. Only string-literal names are graded; a name computed at
runtime is not something this file can resolve.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _documentation_files() -> list[Path]:
    files = sorted((_REPO_ROOT / "docs").rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    return [*files, readme] if readme.is_file() else files


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _factory_name(call: ast.Call) -> str | None:
    """The name a sim-mode ``Robot(...)`` registers its robot under, if literal.

    A ``mode="real"`` factory builds no world, so nothing can be added to it.
    """
    mode = next((kw.value for kw in call.keywords if kw.arg == "mode"), None)
    if mode is not None and _literal(mode) != "sim":
        return None
    for keyword in call.keywords:
        if keyword.arg == "name":
            return _literal(keyword.value)
    return _literal(call.args[0]) if call.args else None


def _added_name(call: ast.Call) -> str | None:
    """The name an ``add_robot(...)`` call asks for, if literal (``None`` auto-numbers)."""
    for keyword in call.keywords:
        if keyword.arg == "name":
            return _literal(keyword.value)
    return _literal(call.args[0]) if call.args else None


def _duplicate_names(source: str) -> list[str]:
    """``add_robot`` names already registered on the same engine variable.

    ``sim = Robot("so100")`` binds ``sim`` to a world holding ``"so100"``; a later
    ``sim.add_robot(...)`` may not ask for that name again, nor one an earlier
    ``sim.add_robot`` took. A different variable is a different world.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # a fence with ``...`` placeholders is prose, not code
    taken: dict[str, set[str]] = {}
    duplicates: list[str] = []
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callee_name(node.value) == "Robot" and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = _factory_name(node.value)
                taken[node.targets[0].id] = {name} if name else set()
        elif isinstance(node, ast.Call) and _callee_name(node) == "add_robot":
            func = node.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
                continue
            name = _added_name(node)
            if name is None:
                continue
            names = taken.setdefault(func.value.id, set())
            if name in names:
                duplicates.append(name)
            names.add(name)
    return duplicates


def test_no_fence_adds_a_robot_under_a_name_it_already_took() -> None:
    offenders: list[str] = []
    for path in _documentation_files():
        text = path.read_text(encoding="utf-8")
        for fence in _PYTHON_FENCE.finditer(text):
            for name in _duplicate_names(fence.group(1)):
                line = text.count("\n", 0, fence.start()) + 1
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line} adds a second robot named {name!r}")
    assert not offenders, (
        "add_robot refuses a name the fence already registered (status=error, not an "
        "exception), so the example silently continues on a one-arm world:\n  " + "\n  ".join(offenders)
    )


def test_the_grader_sees_a_factory_name_reused_by_add_robot() -> None:
    fence = 'sim = Robot("so100")\nsim.add_robot(name="so100", position=[0.0, 0.5, 0.0])\n'
    assert _duplicate_names(fence) == ["so100"]
    assert _duplicate_names('sim = Robot("so100")\nsim.add_robot(name="arm2", data_config="so100")\n') == []
