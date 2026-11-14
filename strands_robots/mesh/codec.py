"""Observations as mesh payloads: plain dicts with arrays, images only when asked for."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..types import JointState, Observation, Pose

_POSE = ("position", "orientation", "linear_velocity", "angular_velocity")


def unmeasured_as_null(values: Any) -> Any:
    """Velocities or efforts for the wire: the array as is when every joint reports one, else a list with ``None``
    where a joint has no reading.

    A joint with no velocity or effort interface reports NaN (ros2_control does for every position-only joint), and
    the wire refuses NaN, so the reply to a move that happened came back as an error and a caller could retry a move
    that already ran. Absent is a measurement state, not a fault: it travels as JSON ``null`` and decodes back to NaN
    (``observation_from_dict``). Positions are not passed through here: a joint with no position is a fault, and the
    wire's refusal of it stands, as does its refusal of a non-finite command."""
    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).all():
        return arr
    return [float(v) if np.isfinite(v) else None for v in arr.tolist()]


def observation_to_dict(obs: Observation, *, images: bool = False) -> dict[str, Any]:
    js = obs.joints
    out: dict[str, Any] = {
        "t": float(obs.t),
        "names": list(js.names),
        "positions": np.asarray(js.positions, dtype=float),
        "velocities": unmeasured_as_null(js.velocities),
        "efforts": unmeasured_as_null(js.efforts),
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


_RENDER: dict[int, ThreadPoolExecutor] = {}


def _render_thread(robot: Any) -> ThreadPoolExecutor:
    """One thread per robot does every image observe: a GL context is bound to the thread that first rendered with
    it (measured on macOS: a render from a second thread never returns)."""
    pool = _RENDER.get(id(robot))
    if pool is None:
        pool = _RENDER[id(robot)] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-render")
    return pool


def observe_dict(robot: Any, images: bool) -> Any:
    """``robot.observe`` as a mesh payload; an image observe runs on the robot's own render thread."""
    if not images:
        return observation_to_dict(robot.observe(images=False))
    obs = _render_thread(robot).submit(robot.observe, images=True).result()
    return observation_to_dict(obs, images=True)
