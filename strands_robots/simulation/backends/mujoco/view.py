"""What a MuJoCo world shows rather than simulates: the lights a scene adds, a camera moved in place (no rebuild),
and how a robot's farthest link moves per radian of each joint (``Simulation.describe()['robots'][id]['reach']``)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ...scene import Scene, SceneCamera
from .build import look_at


def add_lights(spec: mujoco.MjSpec, scene: Scene) -> None:
    """The headlight, the world's key light and every ``Simulation.light()``, each scaled by ``light_scale``."""
    k = scene.light_scale
    spec.visual.headlight.diffuse, spec.visual.headlight.ambient = [0.5 * k] * 3, [0.15 * k] * 3
    spec.worldbody.add_light(pos=[0.5, -1.0, 3.0], dir=[-0.2, 0.4, -1.0], diffuse=[0.8 * k] * 3)
    for li in scene.lights:
        kind = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if li.directional else mujoco.mjtLightType.mjLIGHT_SPOT
        spec.worldbody.add_light(
            name=li.name, pos=list(li.position), dir=list(li.direction), diffuse=[c * k for c in li.color], type=kind
        )


class View:
    """The MuJoCo backend's camera-in-place and reach readings (mixed into it)."""

    model: mujoco.MjModel | None
    data: mujoco.MjData | None
    _scene: Scene
    _index: dict[str, Any]

    def move_camera(self, old: SceneCamera, cam: SceneCamera) -> bool:
        """Re-aim a built camera by writing the compiled model; False (rebuild instead) if its parent or size moved."""
        m, d = self.model, self.data
        i = -1 if m is None else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam.name)
        if m is None or d is None or i < 0 or (old.attach, old.size) != (cam.attach, cam.size):
            return False
        pos = np.asarray(cam.position, dtype=float) + np.asarray(self._scene.camera_jitter.get(cam.name, (0, 0, 0)))
        quat = np.asarray(cam.orientation if cam.orientation is not None else (1.0, 0.0, 0.0, 0.0), dtype=float)
        if cam.orientation is None:  # the look-at frame build.add_camera gives the spec, as a quaternion
            xy = np.asarray(look_at(pos, np.asarray(cam.target if cam.target is not None else (0, 0, 0), float)))
            mat = np.column_stack([xy[:3], xy[3:], np.cross(xy[:3], xy[3:])])
            mujoco.mju_mat2Quat(quat, mat.flatten())
        m.cam_pos[i], m.cam_quat[i], m.cam_fovy[i] = pos, quat / np.linalg.norm(quat), cam.fov
        mujoco.mj_camlight(m, d)
        return True

    def reach(self, body: str) -> dict[str, Any]:
        """The robot's farthest link now, and how it moves (world m) per +1 rad of each driven joint."""
        m, d, idx = self.model, self.data, self._index.get(body)
        if m is None or d is None or idx is None:
            return {}
        mujoco.mj_kinematics(m, d)  # position-derived quantities only: no state changes
        mujoco.mj_comPos(m, d)
        own = [b for b in range(1, m.nbody) if m.body_rootid[b] == idx.root_body]
        leaves = [b for b in own if not any(m.body_parentid[c] == b for c in own)]
        tip = max(leaves, key=lambda b: float(np.linalg.norm(d.xpos[b] - d.xpos[idx.root_body])))
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jac, None, tip)
        keys = self._scene.robot(body).description.action_keys
        per = {k: np.round(jac[:, a], 4).tolist() for k, a in zip(keys, idx.joint_dadr, strict=True) if a >= 0}
        name = m.body(tip).name
        return {"link": name.split("/", 1)[-1], "position": np.round(d.xpos[tip], 4).tolist(), "per_rad": per}
