"""Connected policies: what a policy endpoint is as a ``Policy(spec)`` string, and what it says about itself."""

from __future__ import annotations

from typing import Any

KINDS = ("lerobot-async", "onnx", "lerobot", "dataset")
"""``lerobot-async``: a ``lerobot-policy-server`` over zenoh · ``onnx``: a preset or .onnx spec, run here ·
``lerobot``: a Hub id or checkpoint dir, run here · ``dataset``: a recorded LeRobot dataset replayed."""


def policy_spec(endpoint: str, kind: str, deployment: str | None) -> str:
    """The ``Policy(spec)`` string for an endpoint of ``kind`` (see ``KINDS``)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    e = endpoint.strip()
    if not e:
        raise ValueError("endpoint is empty")
    if kind != "lerobot-async":
        return e
    if e.startswith("zenoh"):
        return e if deployment is None else f"{e.rstrip('/')}/{deployment}"
    if not deployment:
        raise ValueError("a lerobot-async endpoint 'host:port' needs deployment= (the server's deployment name)")
    return f"zenoh://{e.removeprefix('tcp/')}/{deployment}"


def probe(spec: str, kind: str) -> dict[str, Any]:
    """What the endpoint says about itself before any robot runs it; a lerobot server is asked for its descriptor."""
    from ..policy.factory import make

    if kind in ("lerobot", "dataset"):
        return {"summary": f"{kind} {spec} (loaded when a run starts)"}
    pol = make(spec)
    if kind != "lerobot-async":
        return {"summary": repr(pol)}
    d = pol.describe()
    caps = d.get("capabilities") or {}
    feats = caps.get("features") or []
    interval = caps.get("action_interval")
    out: dict[str, Any] = {
        "summary": f"lerobot-policy-server {pol.deployment} @ {pol.endpoint}",
        "semantics": d.get("semantics"),
        "features": [f.get("name") for f in feats if isinstance(f, dict) and f.get("name")],
        "modes": list(caps.get("modes") or ()),
        "tasks": list(d.get("tasks") or ())[:20],
        "artifact": d.get("artifact_identity"),
        "session": d.get("session_status"),
    }
    if interval:
        out["hz"] = 1.0 / float(interval)
    return out
