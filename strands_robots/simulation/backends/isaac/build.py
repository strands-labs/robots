"""Scene pieces as Isaac prims: primitive objects, cameras, the dof map and a robot's keyframe."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ...scene import Scene, SceneCamera, SceneObject, SceneRobot
from .usd import convert_to_usd, demangle, object_model

_KINDS = {"box": "Cuboid", "sphere": "Sphere", "cylinder": "Cylinder", "capsule": "Capsule"}


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(w, x, y, z) for a camera whose +X looks at ``target`` with +Z up (``camera_axes="world"``)."""
    f = target - pos
    n = float(np.linalg.norm(f))
    f = f / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    yaw, pitch = math.atan2(f[1], f[0]), -math.asin(max(-1.0, min(1.0, f[2])))
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def refuse(scene: Scene, objects: Any, cameras: Any) -> None:
    """Every scene change Isaac cannot build, refused by name before ``load`` touches the stage."""
    if scene.friction_scale != 1.0 or scene.mass_scale != 1.0 or scene.scales or scene.camera_jitter:
        raise NotImplementedError(
            "Simulation('isaac') cannot randomize friction, mass or cameras yet (PhysX materials / body masses are not "
            "wired); randomize(lighting=) works, and Simulation('mujoco') randomizes them all"
        )
    for c in scene.cameras:
        if c.name not in cameras and c.attach is not None:
            raise NotImplementedError(
                f"Simulation('isaac'): camera {c.name!r} attach={c.attach!r} is not wired yet; use a world camera"
            )


def add_ground() -> Any:
    from isaacsim.core.api.objects import GroundPlane

    return GroundPlane(prim_path="/World/ground", size=50.0, z_position=0.0, color=np.array([0.55, 0.58, 0.62]))


def add_object(world: Any, o: SceneObject) -> Any:
    if o.path is not None:
        return add_file_object(world, o)
    from isaacsim.core.api import objects

    kind = _KINDS[o.shape or "box"]
    cls = getattr(objects, ("Dynamic" if o.free else "Fixed") + kind)
    kw: dict[str, Any] = {
        "prim_path": f"/World/objects/{o.name}",
        "name": o.name,
        "position": np.asarray(o.position),
        "orientation": np.asarray(o.orientation),
        "color": np.asarray(o.color[:3]),
    }
    if kind == "Cuboid":  # scene sizes are half-extents (as in MuJoCo); Isaac scales a unit cube
        kw["size"] = 1.0
        kw["scale"] = np.asarray([2 * s for s in (list(o.size) + [o.size[0]] * 3)[:3]])
    elif kind == "Sphere":
        kw["radius"] = float(o.size[0])
    else:
        kw["radius"], kw["height"] = float(o.size[0]), float(2 * (o.size[1] if len(o.size) > 1 else o.size[0]))
    if o.mass is not None and o.free:
        kw["mass"] = o.mass
    prim = cls(**kw)
    world.scene.add(prim)
    return prim


def add_file_object(world: Any, o: SceneObject) -> Any:
    """A model file as one body under ``/World/objects/<name>``: a ``free=False`` file is scenery, placed by its root
    xform; a ``free=True`` file is the rigid body Isaac's importer made of it (its root xform never moves)."""
    from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdPhysics

    assert o.path is not None
    path = f"/World/objects/{o.name}"
    model, fmt = object_model(Path(o.path).expanduser().resolve(), free=o.free)
    add_reference_to_stage(usd_path=str(convert_to_usd(model, fmt, fix_base=not o.free)), prim_path=path)
    pos, quat = np.asarray(o.position, dtype=float), np.asarray(o.orientation, dtype=float)
    if not o.free:
        prim = SingleXFormPrim(path, name=o.name, position=pos, orientation=quat)
    else:
        SingleXFormPrim(path, name=f"{o.name}_root", position=pos, orientation=quat)  # places the reference, once
        root = world.stage.GetPrimAtPath(path)
        body = next((p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
        if body is None:
            raise ValueError(f"{o.path}: Isaac's importer made no rigid body of it, so it cannot be a free object")
        prim = SingleRigidPrim(str(body.GetPath()), name=o.name)
    world.scene.add(prim)
    return prim


def sweep(new_robots: list[SceneRobot], new_objects: list[SceneObject], robots: Any, objects: Any) -> None:
    """Delete the reference prim of every body a failed ``load`` left half-added (an import that raised)."""
    from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

    half = [(r.id, robots, f"/World/{r.id}") for r in new_robots]
    half += [(o.name, objects, f"/World/objects/{o.name}") for o in new_objects]
    for name, held, path in half:
        if name not in held and is_prim_path_valid(path):
            delete_prim(path)


def add_camera(c: SceneCamera) -> Any:
    from isaacsim.sensors.camera import Camera

    pos = np.asarray(c.position, dtype=float)
    if c.orientation is not None:
        quat = np.asarray(c.orientation, dtype=float)
    else:
        quat = look_at_quat(pos, np.asarray(c.target if c.target is not None else (0, 0, 0), dtype=float))
    cam = Camera(prim_path=f"/World/cameras/{c.name}", position=pos, frequency=30, resolution=tuple(c.size))
    cam.set_world_pose(pos, quat, camera_axes="world")
    cam.set_focal_length(1.0 / math.tan(math.radians(c.fov) / 2))  # Isaac's 2 cm horizontal aperture
    cam.initialize()
    return cam


def map_dofs(r: SceneRobot, dof_names: list[str]) -> list[int]:
    """Isaac dof index per action key, by joint name."""
    out = []
    for key in r.description.action_keys:
        model = r.description.unlabel(key)
        hits = [i for i, n in enumerate(dof_names) if demangle(n.split("/")[-1]) == model]
        if not hits:
            raise KeyError(f"{r.id}: joint {model!r} (action key {key!r}) not among Isaac dofs {dof_names}")
        out.append(hits[0])
    return out


def keyframe(r: SceneRobot) -> tuple[np.ndarray, np.ndarray | None] | None:
    """(joint targets in action-key order, free-joint qpos or None) of the robot's keyframe, from its MJCF."""
    key = r.keyframe or (r.description.keyframes[0] if r.description.keyframes else None)
    if key is None or r.description.format != "mjcf":
        return None
    try:
        import mujoco
    except ImportError:
        return None
    m = mujoco.MjModel.from_xml_path(str(r.description.model_path))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        return None
    q = np.zeros(len(r.description.action_keys))
    for i, k in enumerate(r.description.action_keys):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, r.description.unlabel(k))
        if j >= 0:
            q[i] = m.key_qpos[kid, m.jnt_qposadr[j]]
    free = None
    if r.description.floating and m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        a = int(m.jnt_qposadr[0])
        free = np.asarray(m.key_qpos[kid, a : a + 7], dtype=float)
    return q, free


def add_lights(stage: Any, scale: float) -> None:
    """A dome (ambient) and a sun, like the MuJoCo world's headlight + key light; an Isaac stage starts dark."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome")).CreateIntensityAttr().Set(400.0 * scale)
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/sun"))
    sun.CreateIntensityAttr().Set(1500.0 * scale)
    xf = UsdGeom.Xformable(sun.GetPrim())
    if not xf.GetOrderedXformOps():
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))
