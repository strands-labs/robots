"""Observations as mesh payloads: plain dicts with arrays, images only when asked for."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import JointState, Observation, Pose

_POSE = ("position", "orientation", "linear_velocity", "angular_velocity")


def observation_to_dict(obs: Observation, *, images: bool = False) -> dict[str, Any]:
    js = obs.joints
    out: dict[str, Any] = {
        "t": float(obs.t),
        "names": list(js.names),
        "positions": np.asarray(js.positions, dtype=float),
        "velocities": np.asarray(js.velocities, dtype=float),
        "efforts": np.asarray(js.efforts, dtype=float),
        "pose": None if obs.pose is None else {k: getattr(obs.pose, k) for k in _POSE},
    }
    if images:
        out["images"] = dict(obs.images)
    return out


def observation_from_dict(d: dict[str, Any]) -> Observation:
    js = JointState(
        tuple(d["names"]),
        np.asarray(d["positions"], dtype=float),
        np.asarray(d["velocities"], dtype=float),
        np.asarray(d["efforts"], dtype=float),
    )
    p = d.get("pose")
    pose = None
    if p is not None:
        lin, ang = p.get("linear_velocity"), p.get("angular_velocity")
        pose = Pose(
            np.asarray(p["position"], float),
            np.asarray(p["orientation"], float),
            None if lin is None else np.asarray(lin, float),
            None if ang is None else np.asarray(ang, float),
        )
    return Observation(t=float(d["t"]), joints=js, pose=pose, images=d.get("images") or {})
