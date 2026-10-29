"""Scene -> MjSpec -> MjModel: the functions that assemble the MuJoCo world and apply keyframes."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ...._gl import ensure_gl
from ....geometry import quat_mul, quat_rotate
from ...scene import Scene, SceneCamera, SceneObject, SceneRobot

ensure_gl()
import mujoco  # noqa: E402

_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


@lru_cache(maxsize=64)
def robot_spec_xml(path: str) -> str:
    """Load a robot MJCF/URDF once, absolutise its asset paths and keep it as XML (specs are consumed by attach)."""
    spec = mujoco.MjSpec.from_file(path)
    base = Path(spec.modelfiledir or Path(path).parent)
    for assets, sub in ((spec.meshes, spec.meshdir), (spec.hfields, spec.meshdir), (spec.textures, spec.texturedir)):
        for a in assets:
            if a.file and not Path(a.file).is_absolute():
                cand = base / (sub or "") / a.file
                if cand.is_file():
                    a.file = str(cand)
    return spec.to_xml()


def look_at(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """``xyaxes`` for a camera at ``pos`` looking at ``target`` with +z world as up."""
    fwd = target - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x.tolist(), *y.tolist()]


def geom_size(shape: str, size: tuple[float, ...]) -> list[float]:
    s = list(size) + [0.0] * (3 - len(size))
    if shape == "sphere":
        return [s[0], 0, 0]
    if shape in ("cylinder", "capsule"):
        return [s[0], s[1] if len(size) > 1 else s[0], 0]
    return s[:3] if len(size) >= 3 else [s[0]] * 3


def attach_robot(spec: mujoco.MjSpec, r: SceneRobot) -> None:
    rs = mujoco.MjSpec.from_string(robot_spec_xml(str(r.description.model_path)))
    for g in [g for g in rs.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:
        rs.delete(g)  # the world owns the floor
    for light in [li for li in rs.lights if li.parent.name == "world"]:
        rs.delete(light)  # and the lighting: a robot's scene light would stack on the world's and wash it out
    frame = spec.worldbody.add_frame(pos=list(r.position), quat=list(r.orientation))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "attach conflict: integrator" — the world's option wins
        spec.attach(rs, prefix=f"{r.id}/", frame=frame)


def add_object(spec: mujoco.MjSpec, o: SceneObject) -> None:
    body = spec.worldbody.add_body(name=o.name, pos=list(o.position), quat=list(o.orientation))
    if o.free:
        body.add_freejoint(name=f"{o.name}/free")
    if o.path is not None:
        sub = mujoco.MjSpec.from_string(robot_spec_xml(str(Path(o.path).expanduser().resolve())))
        frame = body.add_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.attach(sub, prefix=f"{o.name}/", frame=frame)
        return
    kw: dict[str, Any] = {
        "name": f"{o.name}/geom",
        "type": _GEOM[o.shape or "box"],
        "size": geom_size(o.shape or "box", o.size),
        "rgba": list(o.color),
    }
    if o.mass is not None:
        kw["mass"] = o.mass
    body.add_geom(**kw)


def add_camera(spec: mujoco.MjSpec, c: SceneCamera, scene: Scene) -> None:
    parent = spec.worldbody
    pos = np.asarray(c.position, dtype=float) + np.asarray(scene.camera_jitter.get(c.name, (0.0, 0.0, 0.0)))
    if c.attach is not None and "/" in c.attach:  # "<robot>/<link>": the link carries it (a wrist camera)
        parent = spec.body(c.attach)
        if parent is None:
            raise KeyError(f"camera {c.name!r}: no link {c.attach!r}; describe()['robots'][<id>]['links'] lists them")
    elif c.attach is not None:
        parent = spec.body(root_body_name(spec, c.attach))
    kw: dict[str, Any] = {"name": c.name, "pos": pos.tolist(), "fovy": c.fov}
    if c.orientation is not None:
        kw["quat"] = list(c.orientation)
    else:
        tgt = np.asarray(c.target if c.target is not None else (0, 0, 0), dtype=float)
        kw["xyaxes"] = look_at(pos, tgt)
    parent.add_camera(**kw)


def scale_bodies(model: mujoco.MjModel, scene: Scene) -> None:
    """``randomize``: every body by the world's factors, then one robot's or object's subtree by its own."""
    model.geom_friction[:, 0] *= scene.friction_scale
    model.body_mass[:] *= scene.mass_scale
    model.body_inertia[:] *= scene.mass_scale
    for name, (friction, mass) in scene.scales.items():
        own = [
            b for b in range(1, model.nbody) if model.body(b).name == name or model.body(b).name.startswith(f"{name}/")
        ]
        bodies = np.isin(model.body_rootid, [b for b in own if model.body_parentid[b] == 0])
        model.geom_friction[bodies[model.geom_bodyid], 0] *= friction
        model.body_mass[bodies] *= mass
        model.body_inertia[bodies] *= mass


def link_names(model: mujoco.MjModel, root: int) -> list[str]:
    """The bodies of one attached robot (the root and its subtree), without the ``<id>/`` prefix."""
    names = (model.body(b).name for b in range(model.nbody) if model.body_rootid[b] == root)
    return [n.split("/", 1)[1] for n in names if "/" in n]  # an unnamed MJCF body attaches as ""


def root_body_name(spec: mujoco.MjSpec, rid: str) -> str:
    for b in spec.worldbody.bodies:
        if b.name.startswith(f"{rid}/"):
            return b.name
    raise KeyError(f"no attached root body for {rid!r}")


def apply_keyframes(model: mujoco.MjModel, data: mujoco.MjData, scene: Scene, joint_ids: dict[str, list[int]]) -> None:
    m, d = model, data
    for r in scene.robots:
        keys = [r.keyframe] if r.keyframe else list(r.description.keyframes[:1])
        for key in keys:
            kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, f"{r.id}/{key}")
            if kid < 0:
                continue
            for j in joint_ids[r.id]:
                adr, n = m.jnt_qposadr[j], (7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1)
                d.qpos[adr : adr + n] = m.key_qpos[kid, adr : adr + n]
                if n == 7:  # attach leaves a keyframe's base pose in the robot frame: move it to the spawn frame
                    fq = np.asarray(r.orientation, dtype=float)
                    d.qpos[adr : adr + 3] = np.asarray(r.position) + quat_rotate(fq, m.key_qpos[kid, adr : adr + 3])
                    d.qpos[adr + 3 : adr + 7] = quat_mul(fq, m.key_qpos[kid, adr + 3 : adr + 7])
            if m.nu:
                for a in _robot_actuator_ids(m, r):
                    d.ctrl[a] = m.key_ctrl[kid, a]


def _robot_actuator_ids(model: mujoco.MjModel, r: SceneRobot) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{r.id}/{a}") for a in r.description.actuators]


def add_ground(spec: mujoco.MjSpec, ground: bool | str) -> None:
    """A ground plane in the look ``ground`` names: a checker texture (True/"checker") so a camera sees edges
    not white on white, a flat grey plane ("plain"), or none (False)."""
    if not ground:
        return
    geom: dict[str, Any] = {"name": "ground", "type": mujoco.mjtGeom.mjGEOM_PLANE, "size": [0, 0, 0.05]}
    if ground == "plain":
        geom["rgba"] = [0.75, 0.78, 0.82, 1]
    else:
        spec.add_texture(
            name="groundtex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.3, 0.4, 0.5],
        )
        mat = spec.add_material(name="groundmat", texrepeat=[4, 4], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundtex"
        geom["material"] = "groundmat"
    spec.worldbody.add_geom(**geom)


def outline(model: mujoco.MjModel, data: mujoco.MjData, roots: list[int]) -> np.ndarray:
    """The world corners of every visible geom's box (groups 0-2, what a render draws) in the trees under ``roots``."""
    g = np.flatnonzero(np.isin(model.body_rootid[model.geom_bodyid], roots) & (model.geom_group < 3))
    g = g[model.geom_rgba[g, 3] > 0]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    local = model.geom_aabb[g, None, :3] + signs[None] * model.geom_aabb[g, None, 3:]  # n x 8 x 3, geom frame
    world = data.geom_xpos[g, None] + np.einsum("nij,nkj->nki", data.geom_xmat[g].reshape(-1, 3, 3), local)
    return world.reshape(-1, 3)
