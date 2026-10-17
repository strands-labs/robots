"""A device's tool spec: one ``rpc`` selector plus one parameter object per rpc of its manifest."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable
from typing import Any

from .schema import parameters

# Only the operator may call these; a model never sees them: ``resume`` after an e-stop, and ``calibrate``
# (on a real bus it turns torque off and rewrites the servos' offsets).
OPERATOR_ONLY = frozenset({"resume", "calibrate"})


def rpc_names(device: Any) -> tuple[str, ...]:
    """The rpcs the device's manifest declares, minus the operator-only ones."""
    return tuple(r.name for r in device.manifest.rpcs if r.name not in OPERATOR_ONLY)


def handler(device: Any, rpc: str) -> Callable[..., Any]:
    fn = getattr(device, rpc, None)
    if not callable(fn):
        raise AttributeError(f"{device.id}: the manifest declares rpc {rpc!r} but the device has no such method")
    return fn


def summary(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.splitlines()[0] if doc else ""


def tool_name(prefix: str, ident: str) -> str:
    """``<prefix>_<id>`` in the tool-name alphabet; past 64 chars, a prefix plus a hash of the whole id."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{prefix}_{ident}")
    if len(name) <= 64:
        return name
    return f"{name[:55]}_{hashlib.sha1(ident.encode()).hexdigest()[:8]}"


def tool_spec(device: Any, *, prefix: str = "robot") -> dict[str, Any]:
    """The JSON-safe tool spec of ``device``, generated from its rpcs and their signatures."""
    joints = tuple(getattr(device, "action_keys", ()))
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
        params = parameters(f, joints=joints)
        if params["properties"]:
            props[n] = params
    about = f"{device.id}: call one rpc; its arguments go under the property named after it."
    if joints:
        about += f" Joints: {list(joints)}."
    return {
        "name": tool_name(prefix, device.id),
        "description": about,
        "inputSchema": {"json": {"type": "object", "properties": props, "required": ["rpc"]}},
    }
