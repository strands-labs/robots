"""Every module of the package imports alone: no module-level import cycle (the test behind R-V19)."""

import json
import subprocess
import sys

SCRIPT = r"""
import importlib, json, pkgutil, sys
import strands_robots
names = [m.name for m in pkgutil.walk_packages(strands_robots.__path__, "strands_robots.")]
out = {"ok": [], "skipped": {}, "failed": {}}
for name in names:
    for k in [k for k in sys.modules if k == "strands_robots" or k.startswith("strands_robots.")]:
        del sys.modules[k]  # a fresh package state: this module is the first one imported
    try:
        importlib.import_module(name)
        out["ok"].append(name)
    except ImportError as e:  # an optional extra (mujoco, fastapi, zenoh, awsiot, isaacsim ...) is not a cycle
        text = str(e)
        if "strands_robots" in text and "circular" in text:
            out["failed"][name] = text
        else:
            out["skipped"][name] = text
    except Exception as e:  # noqa: BLE001 — reported by name below
        out["failed"][name] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""


def test_every_module_imports_alone() -> None:
    run = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert not out["failed"], json.dumps(out["failed"], indent=1)
    assert len(out["ok"]) > 50, out
    assert "strands_robots.robot.run.loop" in out["ok"] and "strands_robots.dashboard.dashboard" in out["ok"]
