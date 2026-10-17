"""Policies from the page: what ``Policy(spec)`` can build here, start one on a robot, read its task, stop it.

A run on a real body needs the operator's yes (``confirmed``) and its first action goes through the robot's own
safety chain before the loop starts, so the page can never start motion the chain would refuse.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from ..policy.factory import FORMS


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _is_lerobot_config(path: Path) -> bool:
    """A LeRobot policy config has ``type`` (act, smolvla, pi0, ...) and no transformers ``architectures``."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and "type" in doc and "architectures" not in doc


def policies_json() -> dict[str, Any]:
    out: list[dict[str, Any]] = [
        {"name": "hold", "kind": "builtin", "robots": [], "ready": True, "hz": 30, "description": "hold the pose"}
    ]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for d in sorted(hub.glob("models--*--*")) if hub.is_dir() else []:
        owner, _, name = d.name[len("models--") :].partition("--")
        cfgs = sorted((d / "snapshots").glob("*/config.json"))
        if cfgs and _is_lerobot_config(cfgs[-1]):
            out.append(
                {
                    "name": f"{owner}/{name}",
                    "kind": "lerobot",
                    "robots": [],
                    "ready": _have("lerobot") and _have("torch"),
                    "hz": 30,
                    "description": "LeRobot checkpoint in the local Hub cache",
                }
            )
    return {"policies": out, "forms": list(FORMS)}


def _hold(obs: Any) -> dict[str, float]:
    return {}  # an empty action keeps every target where it is


def start(robot: Any, spec: Any, *, instruction: Any, seconds: Any, hz: Any, confirmed: bool) -> dict[str, Any]:
    from ..policy.factory import make

    if not isinstance(spec, str) or not spec:
        raise ValueError('body must be {"policy": "<spec>", "seconds": <float|null>, "hz": <float|null>}')
    if seconds is not None and float(seconds) <= 0 or hz is not None and float(hz) <= 0:
        raise ValueError("hz and seconds must be > 0")
    if robot._task is not None and robot._task.running:
        raise RuntimeError(f"{robot.id} already runs a task; POST /api/robots/{robot.id}/stop first")
    pol: Any = _hold if spec == "hold" else make(spec)
    if robot.is_real:
        if not confirmed:
            raise PermissionError(f"{robot.id} is a real robot: run it from this page with header X-Confirm: yes")
        first = pol(robot.observe(images=False)) if callable(pol) else pol.act(robot.observe(images=False))
        robot.gate(first, confirmed=True)  # the whole chain on the first action, before anything moves
        if hasattr(pol, "reset"):
            pol.reset()
    kw: dict[str, Any] = {"instruction": instruction, "realtime": True}
    if hz is not None:
        kw["hz"] = float(hz)
    if seconds is None:
        kw["stop_when"] = lambda obs: False  # until POST /stop
    else:
        kw["seconds"] = float(seconds)
    task = robot.start(pol, **kw)
    return {"robot": robot.id, "policy": spec, "status": task.status, "seconds": seconds}


def status(robot: Any) -> dict[str, Any]:
    t = robot._task
    if t is None:
        return {"robot": robot.id, "status": "idle", "result": None}
    if t.running:
        return {"robot": robot.id, "status": "running", "result": None}
    try:
        r = t.result
    except Exception as exc:  # noqa: BLE001 — the task's own error is its status
        return {"robot": robot.id, "status": "error", "error": f"{type(exc).__name__}: {exc}", "result": None}
    return {"robot": robot.id, "status": "done", "result": _result(r)}


def stop(robot: Any) -> dict[str, Any]:
    if robot._task is None:
        return {"robot": robot.id, "status": "idle", "result": None}
    was = robot._task.running
    robot._task.halt()
    out = status(robot)
    out["status"] = "stopped" if was else out["status"]
    return out


def _result(r: Any) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "steps": r.steps,
        "seconds": round(r.seconds, 3),
        "hz_achieved": round(r.hz_achieved, 2),
        "stopped_by": r.stopped_by,
        "fault": r.fault,
    }
