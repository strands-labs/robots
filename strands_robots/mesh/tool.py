"""``use_mesh``: every device on the mesh as one agent tool, generated from their manifests (strands imported here).

Actions: list · describe · observe · act · run · stop · estop · resume. ``act`` goes through the device's safety
chain and operator gate on the device itself; ``run`` drives a device through ``connect`` with the device's own
gate on every write; ``resume`` is listed so a model learns why it cannot: clearing an e-stop is the operator's,
never a model's, even on a mesh that holds the operator's key. The description lists the devices present when the
tool is built; ``list`` always answers with the devices present now.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import Observation
from .device import Device

if TYPE_CHECKING:
    from .mesh import Mesh

ACTIONS = ("list", "describe", "observe", "act", "run", "stop", "estop", "resume")


def _json(x: Any) -> Any:
    if isinstance(x, Observation):
        return {"t": x.t, "joints": x.joints.as_dict(), "images": {k: list(v.shape) for k, v in x.images.items()}}
    if isinstance(x, Device):
        return {"id": x.id, "device_type": x.device_type, "labels": x.labels, "last_seen": x.last_seen}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Mapping):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def spec(mesh: Mesh) -> dict[str, Any]:
    """The tool spec for the devices present right now: their ids, labels, rpcs and joints come from manifests."""
    devices = mesh.discover()
    joints: dict[str, Any] = {}
    lines = []
    for d in devices:
        for c in d.manifest.get("controls") or ():
            key = str(c.get("name", "")).removesuffix(".pos")
            rng = c.get("range")
            joints[key] = {"type": "number", "description": f"{c.get('unit', '')} {rng}".strip()}
        labels = ",".join(f"{k}:{v}" for k, v in sorted(d.labels.items()))
        rpcs = [r.get("name") for r in d.manifest.get("rpcs") or ()]
        lines.append(f"- {d.id} ({d.device_type}; {labels or 'no labels'}) rpcs={rpcs}")
    about = f"Every device on the mesh (tenant {mesh.tenant}). Selectors: device(*), device(role:arm,...).\n"
    props: dict[str, Any] = {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "device_id": {"type": "string", "description": "the device, for describe/observe/act/run/stop/resume"},
        "selector": {"type": "string", "description": "for list and estop; default device(*)"},
        "act": {"type": "object", "properties": joints, "description": "joint targets, radians"},
        "run": {
            "type": "object",
            "properties": {
                "policy": {"type": "string"},
                "steps": {"type": "integer"},
                "seconds": {"type": "number"},
                "instruction": {"type": "string"},
            },
            "required": ["policy"],
        },
        "approved_by": {"type": "string", "description": "resume is refused: the operator clears an e-stop"},
    }
    return {
        "name": "use_mesh",
        "description": about + ("\n".join(lines) or "(no devices present)"),
        "inputSchema": {"json": {"type": "object", "properties": props, "required": ["action"]}},
    }


def dispatch(mesh: Mesh, inp: Mapping[str, Any]) -> Any:
    action, did = inp.get("action"), inp.get("device_id")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    extra = sorted(set(inp) - {"action", "device_id", "selector", "act", "run", "approved_by"})
    if extra:
        raise TypeError(f"use_mesh does not take {extra}")
    if action == "list":
        return mesh.discover(str(inp.get("selector") or "device(*)"))
    if action == "estop":
        return mesh.broadcast("estop", selector=str(inp.get("selector") or "device(*)"))
    if not isinstance(did, str):
        raise ValueError(f"{action} needs device_id")
    if action == "act":
        return mesh.invoke(did, "act", action=dict(inp.get("act") or {}))
    if action == "run":
        args = dict(inp.get("run") or {})
        prev = mesh.tasks.get(did)
        if prev is not None and prev.running:
            raise RuntimeError(f"robot {did!r} already has a running task; call stop() first")
        task = mesh.connect(did).start(args.pop("policy"), **args)
        mesh.tasks[did] = task
        return {"started": did, "status": getattr(task, "status", None)}
    if action == "stop":
        task = mesh.tasks.pop(did, None)
        if task is not None:
            task.stop()
        return mesh.invoke(did, "stop")
    if action == "resume":
        raise PermissionError(
            f"resume is the operator's, not an agent's: the operator clears {did} with "
            "mesh.invoke(id, 'resume', approved_by=...) or the dashboard"
        )
    return mesh.invoke(did, action)


def use_mesh(mesh: Mesh) -> Any:
    from strands.tools.tools import PythonAgentTool

    def run(tool_use: Any, **_state: Any) -> Any:
        tid = tool_use["toolUseId"]
        try:
            result = dispatch(mesh, dict(tool_use.get("input") or {}))
        except Exception as exc:  # noqa: BLE001 - the model reads the refusal and corrects its next call
            return {"toolUseId": tid, "status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
        return {"toolUseId": tid, "status": "success", "content": [{"text": json.dumps(_json(result))}]}

    return PythonAgentTool("use_mesh", spec(mesh), run)  # type: ignore[arg-type]
