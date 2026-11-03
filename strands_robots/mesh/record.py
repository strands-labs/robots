"""What a mounted device tells the mesh about itself, and the defaults a Mesh starts from."""

from __future__ import annotations

import getpass
import os
import re
import socket
from collections.abc import Mapping
from typing import Any

from ..robot.safety import EStopped, RequestRefused

RAISE: dict[str, type[Exception]] = {
    "RequestRefused": RequestRefused,
    "EStopped": EStopped,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "LookupError": LookupError,
    "KeyError": LookupError,
}
"""Remote error types re-raised as themselves at the caller; anything else is a RuntimeError."""


_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MAX_LABELS = 16

_HOST_ONLY = ("model",)
"""Manifest fields that only make sense on the device's own machine (paths); never sent."""


def record(
    robot: Any, device_id: str, labels: Mapping[str, str], tenant: str, served: tuple[str, ...]
) -> dict[str, Any]:
    """The device's record: identity and manifest only, as the mesh knows it (its id, tenant, labels, the rpcs it
    answers over the mesh). Live status (estop, task) travels on state and events."""
    manifest = dict(robot.manifest.to_dict() if hasattr(robot, "manifest") else robot.describe())
    labels = {str(k): str(v) for k, v in labels.items()}
    bad = [f"{k}:{v}" for k, v in labels.items() if not (_LABEL.match(k) and _LABEL.match(v))]
    if bad or len(labels) > MAX_LABELS:
        raise ValueError(
            f"labels must be at most {MAX_LABELS} pairs of 1-64 characters from [A-Za-z0-9_.:-] (what a selector "
            f"can match); refused: {bad or f'{len(labels)} labels'}"
        )
    manifest.update(device_id=device_id, tenant=tenant, labels=labels)
    for k in _HOST_ONLY:
        manifest.pop(k, None)
    if isinstance(manifest.get("description"), Mapping):
        manifest["description"] = {k: v for k, v in manifest["description"].items() if k not in _HOST_ONLY}
    if "rpcs" in manifest:
        manifest["rpcs"] = [r for r in manifest["rpcs"] if isinstance(r, Mapping) and r.get("name") in served]
    device_type = str(manifest.get("device_type") or manifest.get("name") or type(robot).__name__)
    return {"id": device_id, "device_type": device_type, "tenant": tenant, "labels": labels, "manifest": manifest}


def state_rate(manifest: Mapping[str, Any]) -> float:
    for e in manifest.get("events") or ():
        if isinstance(e, Mapping) and e.get("name") == "state" and e.get("hz"):
            return float(e["hz"])
    return 10.0


def _segment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "-", text) or "anonymous"


def default_tenant() -> str:
    """The user's own tenant (``STRANDS_TENANT`` or the login name): a LAN of strands-robots users is not one room."""
    return _segment(os.environ.get("STRANDS_TENANT") or getpass.getuser())


def default_operator() -> str:
    """Who this process is when it stops or resumes a device: ``user@host``."""
    return f"{getpass.getuser()}@{socket.gethostname()}"


def check_kill_switch() -> None:
    """``STRANDS_MESH=false`` (or 0/no/off) forces the mesh off, even for an explicit ``Mesh(...)``."""
    if os.environ.get("STRANDS_MESH", "").strip().lower() in ("0", "false", "no", "off"):
        raise RuntimeError("the mesh is switched off on this machine (STRANDS_MESH=false)")
