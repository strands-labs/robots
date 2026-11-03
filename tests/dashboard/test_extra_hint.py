"""Without the web stack, every way into the dashboard says which extra to install, never a bare module name."""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

import pytest

import strands_robots.dashboard
from strands_robots import Simulation
from strands_robots.dashboard import Dashboard


class _NoWeb(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):  # noqa: ANN001
        if name.split(".")[0] in ("fastapi", "starlette", "uvicorn"):
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
        return None


@pytest.mark.parametrize("entry", ["app", "start", "serve"])
def test_a_missing_web_stack_names_the_dashboard_extra(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for mod in [m for m in sys.modules if m.split(".")[0] in ("fastapi", "starlette", "uvicorn")]:
        monkeypatch.delitem(sys.modules, mod)
    for name in ("app", "server", "guard"):  # a fresh process: neither in sys.modules nor bound on the package
        monkeypatch.delitem(sys.modules, f"strands_robots.dashboard.{name}", raising=False)
        monkeypatch.delattr(strands_robots.dashboard, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoWeb(), *sys.meta_path])
    dash = Dashboard(Simulation.__new__(Simulation))
    with pytest.raises(ImportError, match=r"strands-robots\[dashboard\]"):
        getattr(dash, entry) if entry == "app" else getattr(dash, entry)(port=8091)
