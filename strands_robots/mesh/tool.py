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
from .record import _LABEL

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


RUN_ARGS = ("policy", "steps", "seconds", "hz", "instruction")  # the run keys a model may set


def _token(x: Any) -> str | None:
    """``x`` if it is a selector token (what a label may be), else ``None``: any peer in the tenant writes these."""
    return x if isinstance(x, str) and _LABEL.match(x) else None


def spec(mesh: Mesh) -> dict[str, Any]:
    """The tool spec for the devices present right now: their ids, labels, rpcs and joints come from manifests.

    Every manifest field comes from a peer, so only selector tokens and numbers reach the description, one JSON
    line per device: a label or type cannot write prose into the model's instructions."""
    devices = mesh.discover()
    joints: dict[str, Any] = {}
    lines = []
    for d in devices:
        for c in d.manifest.get("controls") or ():
            key = _token(str(c.get("name", "")).removesuffix(".pos")) if isinstance(c, Mapping) else None
            rng = c.get("range") if key else None
            rng = (
                [float(v) for v in rng]
                if isinstance(rng, list) and all(isinstance(v, (int, float)) for v in rng)
                else None
            )
            if key:
                desc = " ".join(str(p) for p in (_token(c.get("unit")), rng) if p)
                joints[key] = {"type": "number", "description": desc}
        labels = {k: v for k, v in sorted(d.labels.items()) if _token(k) and _token(v)}
        rpcs = [r.get("name") for r in d.manifest.get("rpcs") or () if isinstance(r, Mapping) and _token(r.get("name"))]
        line = {"id": _token(d.id), "type": _token(d.device_type) or "?", "labels": labels, "rpcs": rpcs}
        if line["id"]:
            lines.append("- " + json.dumps(line))
    about = f"Every device on the mesh (tenant {mesh.tenant}). Selectors: device(*), device(role:arm,...).\n"
    props: dict[str, Any] = {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "device_id": {"type": "string", "description": "the device, for describe/observe/act/run/stop/resume"},
        "selector": {"type": "string", "description": "for list and estop; default device(*)"},
        "act": {"type": "object", "properties": joints, "description": "joint targets, radians"},
        "seconds": {"type": "number", "description": "act: reach the targets over this long; without it, one step"},
        "run": {
            "type": "object",
            "properties": {
                "policy": {"type": "string"},
                "steps": {"type": "integer"},
                "seconds": {"type": "number"},
                "hz": {"type": "number"},
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


def dispatch(mesh: Mesh, inp: Mapping[str, Any], *, approve: Any = None) -> Any:
    """One ``use_mesh`` call. ``approve`` (``callable(RunPlan) -> bool``) is the operator's answer for an agent's run
    on a device: without it such a run is refused, as for any real robot."""
    action, did = inp.get("action"), inp.get("device_id")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    extra = sorted(set(inp) - {"action", "device_id", "selector", "act", "seconds", "run", "approved_by"})
    if extra:
        raise TypeError(f"use_mesh does not take {extra}")
    if action == "list":
        return mesh.discover(str(inp.get("selector") or "device(*)"))
    if action == "estop":
        return mesh.broadcast("estop", selector=str(inp.get("selector") or "device(*)"))
    if not isinstance(did, str):
        raise ValueError(f"{action} needs device_id")
    if action == "act":
        reach = {} if inp.get("seconds") is None else {"seconds": inp["seconds"]}
        return mesh.invoke(did, "act", action=dict(inp.get("act") or {}), **reach)
    if action == "run":
        args = dict(inp.get("run") or {})
        extra = sorted(set(args) - set(RUN_ARGS))  # record=/push= are the developer's: nothing leaves on a model's say
        if extra:
            raise TypeError(f"use_mesh run takes {list(RUN_ARGS)}, not {extra}")
        if isinstance(args.get("policy"), str) and "://" in args["policy"]:  # its frames would stream there
            raise PermissionError(
                f"a policy at a network address is the developer's to pass, not an agent's: {args['policy']}"
            )
        prev = mesh.tasks.get(did)
        if prev is not None and prev.running:
            raise RuntimeError(f"robot {did!r} already has a running task; call stop() first")
        robot = mesh.connect(did)
        if robot.is_real and approve is None:  # said now, not from the run's thread after "started"
            raise PermissionError(
                f"{did}: an agent's run on a device needs the operator's yes: build the tool with "
                "use_mesh(mesh, approve=<callable(RunPlan) -> bool>)"
            )
        robot.approve = approve
        task = robot.start(args.pop("policy"), **args)
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


def use_mesh(mesh: Mesh, *, approve: Any = None) -> Any:
    """The tool. Every call runs as an agent's (``from_agent``); ``approve`` answers its runs (see ``dispatch``)."""
    from strands.tools.tools import PythonAgentTool

    def run(tool_use: Any, **_state: Any) -> Any:
        tid = tool_use["toolUseId"]
        from ..robot.run.source import from_agent

        try:
            with from_agent():  # an agent's call: a real robot's run asks the operator, as through every other tool
                result = dispatch(mesh, dict(tool_use.get("input") or {}), approve=approve)
        except Exception as exc:  # noqa: BLE001 - the model reads the refusal and corrects its next call
            return {"toolUseId": tid, "status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
        return {"toolUseId": tid, "status": "success", "content": [{"text": json.dumps(_json(result))}]}

    return PythonAgentTool("use_mesh", spec(mesh), run)  # type: ignore[arg-type]
