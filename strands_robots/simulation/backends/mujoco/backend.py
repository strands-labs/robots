"""The MuJoCo world: one MjSpec built from a Scene, recompiled on every change with state carried over.

Robots are attached with ``spec.attach(robot_spec, prefix="<id>/", frame=...)`` so MuJoCo namespaces
bodies, joints, actuators and keyframes for us. One per-robot index (actuator ids, joint addresses,
root body id) is built in ``load`` so the hot paths — ``act`` / ``observe`` — never scan the model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...._extras import hint

try:
    import mujoco
except ImportError as exc:  # the extra the first snippet needs, so the error says how to get it
    raise ImportError(f"Simulation() needs MuJoCo: {hint('mujoco')} ({exc})") from exc

from ...._gl import close_on_gl_thread_at_exit, gl_in_effect, gl_thread, on_gl_thread, release_before_gl_teardown
from ....types import JointState, Pose
from ...backend import Backend
from ...physics import Contact, Physics
from ...scene import Scene, SceneRobot
from . import build
from .state import place, restore, snapshot
from .view import View, add_lights


@dataclass
class _RobotIndex:
    """Where one robot's state lives in the compiled model, built once per ``load``."""

    actuator_ids: list[int]  # mj actuator id per action key (model order), -1 if absent
    joint_qadr: list[int]  # qpos address of the driven joint per action key, -1 if the key has no joint
    joint_dadr: list[int]  # dof address of the driven joint per action key, -1 if the key has no joint
    joint_ids: list[int]  # every joint id of this robot (for keyframes)
    root_body: int  # the robot's root body id


class MuJoCoBackend(View, Backend):
    """Builds the world from a Scene; every ``load`` recompiles and carries joint state by name."""

    def __init__(self, **cfg: Any) -> None:
        super().__init__(**cfg)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._scene: Scene = Scene()
        self._index: dict[str, _RobotIndex] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._gl_failed: str | None = None
        close_on_gl_thread_at_exit(self._close_renderers)

    def load(self, scene: Scene) -> None:
        carried = snapshot(self.model, self.data)
        t = self.data.time if self.data is not None else 0.0
        spec = mujoco.MjSpec()
        spec.option.timestep = scene.timestep
        spec.option.gravity = list(scene.gravity)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.nkey = sum(len(r.description.keyframes) for r in scene.robots)  # else attach warns "nkey conflict"
        spec.visual.global_.offwidth = max(self.render_size[0], *(c.size[0] for c in scene.cameras), 640)
        spec.visual.global_.offheight = max(self.render_size[1], *(c.size[1] for c in scene.cameras), 480)
        build.add_ground(spec, scene.ground)
        add_lights(spec, scene)
        for r in scene.robots:
            build.attach_robot(spec, r)
        for o in scene.objects:
            build.add_object(spec, o)
        for c in scene.cameras:
            build.add_camera(spec, c, scene)
        if hasattr(spec, "dedup_assets"):  # MuJoCo 3.17+: N copies of a robot share one mesh, not one each
            spec.dedup_assets()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = spec.compile()
        build.scale_bodies(model, scene)
        data = mujoco.MjData(model)
        self.model, self.data = model, data  # the spec is not kept: it holds every mesh again (~250 MB per so101)
        self._scene = scene
        self._index = {r.id: self._make_index(model, r) for r in scene.robots}
        self._close_renderers()  # built for the old model; freed on the GL thread, never by the collector elsewhere
        build.apply_keyframes(model, data, scene, {rid: idx.joint_ids for rid, idx in self._index.items()})
        restore(model, data, carried)
        data.time = t  # a scene change is not a reset: world time carries over like joint state
        mujoco.mj_forward(model, data)

    @staticmethod
    def _make_index(model: mujoco.MjModel, r: SceneRobot) -> _RobotIndex:
        m, rid, desc = model, r.id, r.description
        act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}/{a}") for a in desc.actuators]
        jq, jd = [], []
        for key in desc.action_keys:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{rid}/{desc.unlabel(key)}")
            jq.append(int(m.jnt_qposadr[j]) if j >= 0 else -1)
            jd.append(int(m.jnt_dofadr[j]) if j >= 0 else -1)
        joint_ids = [j for j in range(m.njnt) if m.joint(j).name.startswith(f"{rid}/")]
        root = next(b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and m.body(b).name.startswith(f"{rid}/"))
        return _RobotIndex(act_ids, jq, jd, joint_ids, root)

    def links(self, body: str) -> list[str]:
        if self.model is None or body not in self._index:  # not built yet (inside sim.editing())
            return []
        return build.link_names(self.model, self._index[body].root_body)

    def outline(self) -> np.ndarray | None:
        roots = [i.root_body for i in self._index.values()]
        return build.outline(self.model, self.data, roots) if self.model is not None and roots else None

    def _body_id(self, name: str) -> int:
        if name in self._index:
            return self._index[name].root_body
        assert self.model is not None
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise KeyError(f"no body {name!r} in the world")
        return b

    def step(self, n: int) -> None:
        assert self.model is not None and self.data is not None
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def reset(self, keyframe: str | None) -> None:
        assert self.model is not None and self.data is not None
        mujoco.mj_resetData(self.model, self.data)
        scene = self._scene
        if keyframe is not None:
            scene = Scene(
                **{**vars(scene), "robots": [SceneRobot(**{**vars(r), "keyframe": keyframe}) for r in scene.robots]}
            )
        build.apply_keyframes(self.model, self.data, scene, {rid: idx.joint_ids for rid, idx in self._index.items()})
        mujoco.mj_forward(self.model, self.data)

    def time(self) -> float:
        assert self.data is not None
        return float(self.data.time)

    def joint_state(self, body: str) -> JointState:
        assert self.model is not None and self.data is not None
        m, d, desc, idx = self.model, self.data, self._scene.robot(body).description, self._index[body]
        pos, vel, eff = (np.empty(len(desc.action_keys)) for _ in range(3))
        for i, key in enumerate(desc.action_keys):
            qa, da, a = idx.joint_qadr[i], idx.joint_dadr[i], idx.actuator_ids[i]
            if qa >= 0:
                pos[i], vel[i], eff[i] = d.qpos[qa], d.qvel[da], d.qfrc_actuator[da]
            elif a >= 0:  # an actuator driving a tendon or site (panda's gripper): hold its transmission
                gain, bias = m.actuator_gainprm[a, 0], m.actuator_biasprm[a]
                servo = m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and gain != 0 and bias[1] != 0
                pos[i] = -(bias[0] + bias[1] * d.actuator_length[a]) / gain if servo else d.ctrl[a]
                vel[i], eff[i] = d.actuator_velocity[a], d.actuator_force[a]
            else:
                raise KeyError(f"{body}: no joint or actuator for action key {key!r} in the compiled world")
        return JointState(desc.action_keys, pos, vel, eff)

    def set_targets(self, body: str, targets: np.ndarray) -> None:
        assert self.data is not None
        ids = self._index[body].actuator_ids
        if len(targets) != len(ids):
            raise ValueError(f"{body}: {len(targets)} targets for {len(ids)} actuators")
        self.data.ctrl[ids] = targets

    def body_pose(self, body: str) -> Pose:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        b = self._body_id(body)
        j = int(m.body_jntadr[b])
        if m.body_jntnum[b] > 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            va = m.jnt_dofadr[j]  # free joint: linear velocity in the world frame, angular in the body frame
            return Pose(d.xpos[b].copy(), d.xquat[b].copy(), d.qvel[va : va + 3].copy(), d.qvel[va + 3 : va + 6].copy())
        return Pose(d.xpos[b].copy(), d.xquat[b].copy())

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        """Make this robot's position actuators the PD law ``tau = kp·(target − q) − kd·q̇`` (no recompile)."""
        assert self.model is not None
        ids = self._index[body].actuator_ids
        kp, kd = np.asarray(kp, dtype=float).reshape(-1), np.asarray(kd, dtype=float).reshape(-1)
        if kp.shape != (len(ids),) or kd.shape != (len(ids),):
            raise ValueError(f"{body}: kp {kp.shape} / kd {kd.shape} for {len(ids)} actuators")
        m = self.model
        for i, p, dgain in zip(ids, kp, kd, strict=True):
            m.actuator_gainprm[i, 0] = p
            m.actuator_biasprm[i, :3] = (0.0, -p, -dgain)

    def set_joints(self, body: str, positions: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        idx = self._index[body]
        place(self.model, self.data, idx.joint_qadr, idx.joint_dadr, positions)

    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        b = self._body_id(body)
        j = int(m.body_jntadr[b])
        if m.body_jntnum[b] > 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[qa : qa + 3] = pose.position
            d.qpos[qa + 3 : qa + 7] = pose.orientation
            if zero_velocity:
                d.qvel[va : va + 6] = 0.0
        else:  # fixed base: the caller has recorded the new pose in the scene; rebuild it the way every change is
            self.load(self._scene)
        mujoco.mj_forward(self.model, self.data)

    @gl_thread  # a GL context lives on one thread: macOS hangs, EGL fails, on any other
    def render(self, camera: str) -> np.ndarray:
        assert self.model is not None and self.data is not None
        if self._gl_failed:
            raise RuntimeError(f"rendering unavailable: {self._gl_failed}")
        cam = next((c for c in self._scene.cameras if c.name == camera), None)
        size = cam.size if cam else self.render_size
        r = self._renderers.get(size)
        if r is None:
            try:
                r = mujoco.Renderer(self.model, height=size[1], width=size[0])
            except Exception as exc:  # noqa: BLE001 — no GL on this host
                self._gl_failed = f"{type(exc).__name__}: {exc} ({gl_in_effect()})"
                raise RuntimeError(f"rendering unavailable: {self._gl_failed}") from exc
            self._renderers[size] = r
            release_before_gl_teardown()
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def contacts(self) -> list[Contact]:
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        out: list[Contact] = []
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f)
            body1, body2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
            out.append(Contact(body1, body2, c.pos.copy(), c.frame[:3].copy(), float(f[0])))
        return out

    def physics(self) -> Physics:
        assert self.model is not None
        return Physics("mujoco", float(self.model.opt.timestep), 1)

    def _close_renderers(self) -> None:
        if self._renderers:
            on_gl_thread(lambda: [self._renderers.pop(k).close() for k in list(self._renderers)])

    def close(self) -> None:
        self._close_renderers()
        self.model = self.data = None
