"""fastapi/starlette/uvicorn are imported only by the dashboard's server modules; nothing else imports the dashboard."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import strands_robots

PKG = Path(strands_robots.__file__).parent
WEB = {"fastapi", "starlette", "uvicorn"}
WEB_FILES = {"dashboard/app.py", "dashboard/guard.py", "dashboard/server.py"}


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_web_frameworks_only_in_the_dashboard_server_modules() -> None:
    offenders = {
        str(p.relative_to(PKG)): sorted(_imports(p) & WEB)
        for p in PKG.rglob("*.py")
        if _imports(p) & WEB and str(p.relative_to(PKG)) not in WEB_FILES
    }
    assert offenders == {}


def test_no_module_outside_the_dashboard_imports_it() -> None:
    users = []
    for p in PKG.rglob("*.py"):
        rel = str(p.relative_to(PKG))
        if rel.startswith("dashboard/"):
            continue
        text = p.read_text()
        if "strands_robots.dashboard" in text or "from .dashboard" in text or "from ..dashboard" in text:
            users.append(rel)
    assert users == []


def test_importing_the_dashboard_does_not_start_a_web_framework() -> None:
    code = (
        "import sys, strands_robots.dashboard; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'fastapi', 'starlette', 'uvicorn', 'strands'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
