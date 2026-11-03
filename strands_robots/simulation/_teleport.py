"""The arithmetic behind ``Simulation.set_pose``: where a body is now, where it is asked to go, and which record
holds that (the compiled model, or the scene entry when the body is not built yet or is rebuilt from the scene).
Called under the world lock."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import moved_frame, vec
from ..types import Pose


def _requested(cur: Pose, position: Any, orientation: Any) -> Pose:
    return Pose(
        cur.position if position is None else np.asarray(vec(position, 3, "position")),
        cur.orientation if orientation is None else np.asarray(vec(orientation, 4, "orientation (w, x, y, z)")),
    )


def _pending(ops: Any, cur: Pose, built: Pose | None, entry: Any) -> Pose:
    """Where the body IS for the next partial request: ``cur`` outside a block; inside one the backend is not
    rebuilt, so the pose an earlier call in the block asked for lives only in the scene entry — a ``free=False``
    object's entry is that pose, a fixed base's entry is a spawn frame, so its pose is the entry composed onto the
    frame ``cur`` was compiled from (``entry * built^-1 * cur``; ``cur`` itself while nothing has moved)."""
    if not ops._editing:
        return cur
    entry_pose = Pose(np.asarray(entry.position, float), np.asarray(entry.orientation, float))
    return entry_pose if built is None else moved_frame(cur, built, entry_pose)


def _entry(ops: Any, name: str) -> Any:
    """The scene record of a robot (by id) or an object (by name)."""
    if name in ops._robots:
        return ops.scene.robot(name)
    return next(o for o in ops.scene.objects if o.name == name)


def teleport(ops: Any, name: str, position: Any, orientation: Any) -> Pose:
    """Move ``name`` to the requested pose, velocities zeroed; returns where it is."""
    try:
        cur = ops.backend.body_pose(name)
    except KeyError:
        if not ops._editing:
            raise
        entry = _entry(ops, name)  # added in this block, not built yet: the spawn frame is where it is
        pose = _requested(
            Pose(np.asarray(entry.position, float), np.asarray(entry.orientation, float)), position, orientation
        )
        entry.position, entry.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())
        ops._reload()  # the block's one rebuild places it from the scene
        return pose
    fixed = name in ops._robots and not ops._robots[name].description.floating
    if fixed:
        sr = ops.scene.robot(name)  # a fixed base is rebuilt from the scene, so the scene moves first
        built = ops._built.get(name, (sr.position, sr.orientation))  # the frame ``cur`` was compiled from: inside
        built_pose = Pose(np.asarray(built[0], float), np.asarray(built[1], float))  # a block the entry may have moved
        pose = _requested(_pending(ops, cur, built_pose, sr), position, orientation)
        f = moved_frame(built_pose, cur, pose)
        sr.position, sr.orientation = tuple(f.position.tolist()), tuple(f.orientation.tolist())
    else:
        pose = _requested(cur, position, orientation)
    for o in ops.scene.objects:
        if o.name == name and not o.free:
            pose = _requested(_pending(ops, cur, None, o), position, orientation)
            o.position, o.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())
            fixed = True
    if fixed and ops._editing:  # the scene holds the pose: the block's one rebuild places it, not a second
        ops._reload()
        return pose
    ops.backend.set_pose(name, pose, zero_velocity=True)
    if name in ops._robots and not ops._robots[name].description.floating:
        sr = ops.scene.robot(name)  # the backend now stands on this frame: the next deferred delta composes onto it
        ops._built[name] = (tuple(sr.position), tuple(sr.orientation))
    ops.revision += 1
    return ops.backend.body_pose(name)
