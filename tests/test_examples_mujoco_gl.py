"""No tracked example, notebook, or test may hard-default MUJOCO_GL to "cgl".

``cgl`` is the macOS-only offscreen GL backend; on headless Linux (CI, cloud
GPUs, Jetson) ``MUJOCO_GL=cgl`` makes the first offscreen render die with
``RuntimeError: invalid value for environment variable MUJOCO_GL: cgl``. The
platform-appropriate default (matching the ``.py`` examples) is::

    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

so ``cgl`` is only selected on macOS and a user-exported ``MUJOCO_GL`` always
wins. This scans the notebooks' code cells plus every tracked ``.py`` under
``tests/`` and ``examples/`` and flags any line that assigns the ``"cgl"``
literal to ``MUJOCO_GL`` without a ``sys.platform`` / ``darwin`` guard.
"""

from __future__ import annotations

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
    for base in ("tests", "examples"):
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
