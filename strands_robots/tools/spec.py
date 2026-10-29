"""A device's tool spec: one ``rpc`` selector plus one parameter object per rpc of its manifest."""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from collections.abc import Callable
from typing import Any

from .schema import parameters

# Only the operator may call these; a model never sees them: ``resume`` after an e-stop, and ``calibrate``
# (on a real bus it turns torque off and rewrites the servos' offsets).
OPERATOR_ONLY = frozenset({"resume", "calibrate"})
# Arguments only the developer's own code passes: a model never decides what is recorded or uploaded.
DEVELOPER_ONLY_ARGS = frozenset({"record", "push"})


def rpc_names(device: Any) -> tuple[str, ...]:
    """The rpcs the device's manifest declares, minus the operator-only ones."""
    return tuple(r.name for r in device.manifest.rpcs if r.name not in OPERATOR_ONLY)


def handler(device: Any, rpc: str) -> Callable[..., Any]:
    fn = getattr(device, rpc, None)
    if not callable(fn):
        raise AttributeError(f"{device.id}: the manifest declares rpc {rpc!r} but the device has no such method")
    return fn


def _plain(text: str) -> str:
    return " ".join(text.replace("``", "").split())


def summary(fn: Callable[..., Any]) -> str:
    """The first paragraph of ``fn``'s docstring on one line, RST markup dropped: what a model reads per rpc."""
    return _plain((inspect.getdoc(fn) or "").split("\n\n")[0])


def details(fn: Callable[..., Any]) -> str:
    """The rest of ``fn``'s docstring, the same way: the description of the rpc's argument object."""
    return _plain((inspect.getdoc(fn) or "").partition("\n\n")[2])


def joints(device: Any) -> dict[str, dict[str, Any]]:
    """Action key -> its control's unit and range from the manifest, rounded inward so every bound advertised is one the
    gate accepts; no bounds for a joint without a hard limit."""
    out: dict[str, dict[str, Any]] = {}
    for k in getattr(device, "action_keys", ()):
        c = next((c for c in getattr(device.manifest, "controls", ()) if c.name in (f"{k}.pos", k)), None)
        bounds = (
            {}
            if c is None or c.range is None
            else {"minimum": math.ceil(c.range[0] * 1e3) / 1e3, "maximum": math.floor(c.range[1] * 1e3) / 1e3}
        )
        out[k] = {} if c is None else {"description": c.unit, **bounds}
    return out


def tool_name(prefix: str, ident: str) -> str:
    """``<prefix>_<id>`` in the tool-name alphabet; past 64 chars, a prefix plus a hash of the whole id."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{prefix}_{ident}")
    if len(name) <= 64:
        return name
    return f"{name[:55]}_{hashlib.sha1(ident.encode()).hexdigest()[:8]}"


def tool_spec(device: Any, *, prefix: str = "robot", policies: tuple[str, ...] = ()) -> dict[str, Any]:
    """The JSON-safe tool spec of ``device``, generated from its rpcs and their signatures; ``policies`` are the
    registered handles ``run`` takes by name (with exactly one, ``policy`` is optional)."""
    keys = joints(device)
    robot = getattr(device.manifest, "name", None)
    names = rpc_names(device)
    fns = {n: handler(device, n) for n in names}
    props: dict[str, Any] = {
        "rpc": {
            "type": "string",
            "enum": list(names),
            "description": " | ".join(f"{n}: {summary(f)}" for n, f in fns.items()),
        }
    }
    for n, f in fns.items():
        params = parameters(f, joints=keys, exclude=DEVELOPER_ONLY_ARGS)
        if more := details(f):
            params["description"] = more
        pol = params["properties"].get("policy") if n == "run" else None
        if pol is not None and isinstance(robot, str):  # the presets that drive THIS robot, not every shipped one
            from ..policy.factory import spec_help

            pol["description"] = spec_help(robot)
        if pol is not None and policies:
            pol["description"] = (pol.get("description") or "").rstrip(". ") + (
                f". Registered policies, by name: {', '.join(policies)}"
                + (" (the only one: may be left out)" if len(policies) == 1 else "")
            )
            if len(policies) == 1:
                params["required"] = [r for r in params.get("required", []) if r != "policy"]
        if params["properties"]:
            props[n] = params
    about = f"{device.id}: call one rpc; its arguments go under the property named after it."
    if keys:
        about += f" Joints: {list(keys)}."
    return {
        "name": tool_name(prefix, device.id),
        "description": about,
        "inputSchema": {"json": {"type": "object", "properties": props, "required": ["rpc"]}},
    }
