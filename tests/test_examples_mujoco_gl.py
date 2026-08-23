"""No tracked example, notebook, or test may hard-default MUJOCO_GL to a windowed backend.

MuJoCo has two *windowed* GL backends -- ``cgl`` (macOS) and ``glfw`` (which
needs a window server) -- and two offscreen ones, ``egl`` and ``osmesa``. On a
headless host (CI, cloud GPUs, Jetson) a windowed default cannot render, and
the two fail differently:

* ``cgl`` is rejected at ``import mujoco`` with ``RuntimeError: invalid value
  for environment variable MUJOCO_GL: cgl`` -- loud, and it names the cause;
* ``glfw`` is a *valid* value everywhere, so the import succeeds. The render
  probe then fails and the backend warns that rendering is unavailable, quoting
  ``GLFWError: X11: The DISPLAY environment variable is missing`` -- it names
  the missing display, but not the ``MUJOCO_GL`` value that asked for a
  windowed backend -- and camera observations are skipped from there on. The
  *failure* the caller finally gets is whatever it was doing with those frames:
  a camera recording reports it as a *dataset feature mismatch* listing the
  camera keys, several frames from the setting that caused it.

The platform-appropriate default -- the form all five notebooks and every
guarded example already use -- is::

    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

so a windowed backend is only selected where one exists, an offscreen backend
is used everywhere else, and a user-exported ``MUJOCO_GL`` always wins.

Two rules are enforced over the notebooks' code cells plus every tracked
``.py`` under ``tests/``, ``tests_integ/`` and ``examples/``:

1. **No unguarded** ``"cgl"`` **on any line** (:func:`test_no_unguarded_cgl_default`).
   Line-scoped and deliberately blunt: ``cgl`` cannot be a working default off
   macOS in any scope.
2. **No unguarded windowed backend in a module-scope default**
   (:func:`test_no_module_scope_windowed_gl_default`). Scope matters because
   ``MUJOCO_GL`` is read once, at ``import mujoco``: a module-scope
   ``setdefault`` runs at import and therefore selects the backend for the
   whole file, while one inside a test function usually runs *after* the module
   has already imported mujoco and cannot change anything. Rule 2 is therefore
   AST-scoped to module level, which also excludes by construction the sites
   where a backend name is the value *under test* -- a ``monkeypatch.setenv``
   or an assertion about what the resolver did.

Rule 2 does not grade ``egl``/``osmesa`` defaults: an unguarded ``"egl"``
raises on macOS, but that is a macOS-hostile default rather than a windowed
one, and converging the tree's offscreen spellings is a separate question.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOKS_DIR = _REPO_ROOT / "examples" / "notebooks"

# A line assigning the "cgl" string literal as the MUJOCO_GL value, e.g.
#   os.environ.setdefault("MUJOCO_GL", "cgl")
#   os.environ["MUJOCO_GL"] = "cgl"
# The guarded form keeps "cgl" but also names the platform (darwin / sys.platform).
_CGL_VALUE_RE = re.compile(r'MUJOCO_GL"[^\n]*?"cgl"')


def _is_guarded(line: str) -> bool:
    return "darwin" in line or "sys.platform" in line


def _lines_of(text: str) -> list[str]:
    return text.splitlines()


def _scan_py(path: Path) -> list[str]:
    """Return unguarded-cgl lines from a .py file (empty if clean)."""
    return [
        ln.strip()
        for ln in _lines_of(path.read_text(encoding="utf-8"))
        if _CGL_VALUE_RE.search(ln) and not _is_guarded(ln)
    ]


def _scan_notebook(path: Path) -> list[str]:
    """Return unguarded-cgl lines from a notebook's code cells (empty if clean)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    offending: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for ln in _lines_of("".join(cell.get("source", []))):
            if _CGL_VALUE_RE.search(ln) and not _is_guarded(ln):
                offending.append(ln.strip())
    return offending


def _tracked_py() -> list[Path]:
    # Exclude this scanner file: its docstring/regex mention the pattern verbatim.
    self_path = Path(__file__).resolve()
    files: list[Path] = []
    for base in ("tests", "tests_integ", "examples"):
        root = _REPO_ROOT / base
        if root.is_dir():
            files.extend(p for p in sorted(root.rglob("*.py")) if p.resolve() != self_path)
    return files


def _notebooks() -> list[Path]:
    return sorted(_NOTEBOOKS_DIR.glob("*.ipynb")) if _NOTEBOOKS_DIR.is_dir() else []


def _count_cgl_sites() -> int:
    """Total MUJOCO_GL="cgl" references (guarded or not) - scanner liveness."""
    n = 0
    for p in _tracked_py():
        n += sum(1 for ln in _lines_of(p.read_text(encoding="utf-8")) if _CGL_VALUE_RE.search(ln))
    for p in _notebooks():
        nb = json.loads(p.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            n += sum(1 for ln in _lines_of("".join(cell.get("source", []))) if _CGL_VALUE_RE.search(ln))
    return n


def test_no_unguarded_cgl_default():
    """No tracked example/notebook/test may default MUJOCO_GL to cgl unconditionally."""
    offenders: dict[str, list[str]] = {}
    for p in _tracked_py():
        bad = _scan_py(p)
        if bad:
            offenders[str(p.relative_to(_REPO_ROOT))] = bad
    for p in _notebooks():
        bad = _scan_notebook(p)
        if bad:
            offenders[str(p.relative_to(_REPO_ROOT))] = bad
    assert not offenders, (
        "MUJOCO_GL defaulted to the macOS-only 'cgl' without a platform guard "
        "(breaks headless Linux). Use "
        '\'os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")\'. '
        f"Offending sites: {offenders}"
    )


def test_scanner_sees_cgl_usage():
    """Sanity: the scanner reaches the (now guarded) cgl sites, so the guard can't pass vacuously."""
    assert _count_cgl_sites() >= 4, (
        "expected to find the guarded MUJOCO_GL='cgl' notebook/test sites; "
        "scanner found none - a path/glob regression may make the guard vacuous."
    )


#: MuJoCo's *windowed* GL backends. Neither can render on a host with no
#: window server, so neither is a working unconditional default. The offscreen
#: pair MuJoCo also accepts is ``egl`` / ``osmesa``.
_WINDOWED_BACKENDS = ("cgl", "glfw")

# Fallback for a notebook cell that does not parse (a ``%``/``!`` magic makes the
# cell invalid Python on its own): report any line naming a windowed backend as
# a MUJOCO_GL value, so an unparseable cell is never silently skipped.
_WINDOWED_VALUE_RE = re.compile(rf'MUJOCO_GL"[^\n]*?"({"|".join(_WINDOWED_BACKENDS)})"')


def _gl_default_value(node: ast.AST) -> ast.expr | None:
    """The value expression if ``node`` sets a ``MUJOCO_GL`` default, else ``None``.

    Recognises both spellings the tree uses:
    ``os.environ.setdefault("MUJOCO_GL", V)`` and ``os.environ["MUJOCO_GL"] = V``.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "setdefault" and len(node.args) == 2:
            key = node.args[0]
            if isinstance(key, ast.Constant) and key.value == "MUJOCO_GL":
                return node.args[1]
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "MUJOCO_GL"
            ):
                return node.value
    return None


def _module_scope_gl_defaults(source: str) -> list[tuple[int, str]]:
    """``(line, value-expression)`` for every module-scope ``MUJOCO_GL`` default.

    Function and class bodies are not descended into: those run after the module
    has already imported mujoco, so they cannot change the backend, and a
    backend name appearing there is typically the value *under test*. A default
    nested in a module-level ``if`` / ``try`` / ``with`` still runs at import and
    is therefore in scope.
    """
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                continue
            value = _gl_default_value(child)
            if value is not None:
                found.append((getattr(child, "lineno", 0), ast.unparse(value)))
            visit(child)

    visit(ast.parse(source))
    return found


def _is_guarded_expr(expr_src: str) -> bool:
    """Does this value expression choose per platform, so a windowed name is fine?"""
    return "darwin" in expr_src or "sys.platform" in expr_src


def _names_windowed(expr_src: str) -> bool:
    return any(f'"{backend}"' in expr_src or f"'{backend}'" in expr_src for backend in _WINDOWED_BACKENDS)


def _unguarded_windowed_defaults(source: str) -> list[str]:
    """Module-scope ``MUJOCO_GL`` defaults naming a windowed backend, unguarded."""
    return [
        f"line {line}: {expr}"
        for line, expr in _module_scope_gl_defaults(source)
        if _names_windowed(expr) and not _is_guarded_expr(expr)
    ]


def _unguarded_windowed_in_notebook(path: Path) -> list[str]:
    """Same rule over a notebook's code cells (a cell's top level is module scope)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    offending: list[str] = []
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            bad = _unguarded_windowed_defaults(source)
        except SyntaxError:
            bad = [ln.strip() for ln in _lines_of(source) if _WINDOWED_VALUE_RE.search(ln) and not _is_guarded_expr(ln)]
        offending.extend(f"cell {index} {entry}" for entry in bad)
    return offending


def test_no_module_scope_windowed_gl_default():
    """A module-scope MUJOCO_GL default must not name a windowed backend.

    It runs at import and therefore selects the backend for the whole file, so a
    windowed name there is what a headless host is left with when the operator
    exported nothing.
    """
    offenders: dict[str, list[str]] = {}
    for path in _tracked_py():
        bad = _unguarded_windowed_defaults(path.read_text(encoding="utf-8"))
        if bad:
            offenders[str(path.relative_to(_REPO_ROOT))] = bad
    for path in _notebooks():
        bad = _unguarded_windowed_in_notebook(path)
        if bad:
            offenders[str(path.relative_to(_REPO_ROOT))] = bad
    assert not offenders, (
        "a module-scope MUJOCO_GL default names a windowed GL backend "
        f"({', '.join(_WINDOWED_BACKENDS)}), which cannot render on a headless host. "
        'Use \'os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")\'. '
        f"Offending sites: {offenders}"
    )


def test_scan_reaches_the_module_scope_defaults():
    """Sanity: the AST scan finds the tree's module-scope defaults.

    Without this a path/glob regression, or a walker that descended into nothing,
    would make the rule above pass by reaching no source at all.
    """
    total = sum(len(_module_scope_gl_defaults(p.read_text(encoding="utf-8"))) for p in _tracked_py())
    assert total >= 20, (
        f"the AST scan found only {total} module-scope MUJOCO_GL defaults across {_tracked_py()[:1]}...; "
        "the tree has far more, so the scan is not reaching the sources."
    )


class TestTheRuleIsScopedToWhatSelectsTheBackend:
    """Planted sources, both directions: what the module-scope rule does and does not report."""

    def test_a_module_scope_glfw_default_is_reported(self):
        source = 'import os\nos.environ.setdefault("MUJOCO_GL", "glfw")\n'
        assert _unguarded_windowed_defaults(source) == ["line 2: 'glfw'"]

    def test_a_subscript_assignment_is_a_default_too(self):
        source = 'import os\nos.environ["MUJOCO_GL"] = "glfw"\n'
        assert _unguarded_windowed_defaults(source) == ["line 2: 'glfw'"]

    def test_the_guarded_form_is_accepted(self):
        source = (
            'import os\nimport sys\nos.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")\n'
        )
        assert _unguarded_windowed_defaults(source) == []

    def test_an_offscreen_default_is_accepted(self):
        source = 'import os\nos.environ.setdefault("MUJOCO_GL", "egl")\n'
        assert _unguarded_windowed_defaults(source) == []

    def test_a_default_inside_a_test_function_is_out_of_scope(self):
        # Runs after the module imported mujoco, so it cannot select the backend.
        source = 'import os\n\n\ndef test_x():\n    os.environ.setdefault("MUJOCO_GL", "glfw")\n'
        assert _unguarded_windowed_defaults(source) == []

    def test_a_backend_name_under_test_is_not_a_default(self):
        # The resolver's own tests set and assert backend names; neither is a default.
        source = (
            "import os\n\n\n"
            "def test_respects_user_mujoco_gl(monkeypatch):\n"
            '    monkeypatch.setenv("MUJOCO_GL", "glfw")\n'
            '    assert os.environ["MUJOCO_GL"] == "glfw"\n'
        )
        assert _unguarded_windowed_defaults(source) == []

    def test_a_module_level_conditional_default_is_still_module_scope(self):
        # Nested in a module-level ``if``, so it still runs at import.
        source = (
            'import os\nimport sys\nif sys.version_info >= (3, 12):\n    os.environ.setdefault("MUJOCO_GL", "glfw")\n'
        )
        assert _unguarded_windowed_defaults(source) == ["line 4: 'glfw'"]
