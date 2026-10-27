"""Isaac Sim as a world: one PhysX ``World`` on the process's one ``SimulationApp``.

Robots are converted to USD once (``usd.convert_to_usd``), referenced at ``/World/<id>`` and driven through
``SingleArticulation``. ``load(scene)`` is incremental: prims are added and removed to match the Scene, and
joint state is carried by name across the ``world.reset()`` that initialises new articulations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....types import JointState, Pose
from ...backend import Backend
from ...physics import Contact, Physics
from ...scene import Scene, SceneObject, SceneRobot
from . import app, build
from .usd import convert_to_usd


class IsaacBackend(Backend):
    deterministic = False  # a PhysX world behind the Backend contract; bit-exact replay is not promised

    def __init__(self, **cfg: Any) -> None:
        super().__init__(**cfg)
        if self.physics_name != "native":
            raise NotImplementedError(
                f"Simulation('isaac', physics={self.physics_name!r}) is not supported: Isaac steps PhysX. "
                "For GPU-batched physics use Simulation('mujoco', physics='newton', num_envs=N)."
            )
        if self.num_envs != 1:
            raise ValueError("Simulation('isaac') steps one world; num_envs > 1 needs physics='newton' on mujoco")
        app.simulation_app(self.headless)
        app.claim(self)
        from isaacsim.core.api import World

        World.clear_instance()
        self.world = World(physics_dt=0.002, rendering_dt=1 / 60, stage_units_in_meters=1.0, device="cpu")
        self._scene = Scene()
        self._robots: dict[str, Any] = {}
        self._objects: dict[str, Any] = {}
        self._cameras: dict[str, Any] = {}
        self._dofs: dict[str, np.ndarray] = {}
        self._ground: Any = None
        self._rendered = False
        self._lit: float | None = None
        self._time_base = 0.0

    def load(self, scene: Scene) -> None:
        build.refuse(scene, self._objects, self._cameras)  # before the first mutation: a refusal changes nothing
        carried = self._snapshot()
        pc = self.world.get_physics_context()
        if abs(pc.get_physics_dt() - scene.timestep) > 1e-12:
            pc.set_physics_dt(scene.timestep)
        pc.set_gravity(float(scene.gravity[2]))
        if scene.ground and self._ground is None:
            self._ground = build.add_ground()
        if self._lit != scene.light_scale:
            build.add_lights(self.world.stage, scene.light_scale)
            self._lit = scene.light_scale
        t = self.time()
        removed = self._remove_gone(scene)
        new_robots = [r for r in scene.robots if r.id not in self._robots]
        new_objects = [o for o in scene.objects if o.name not in self._objects]
        rebuild = bool(removed or new_robots or new_objects)
        if rebuild and self.world.is_playing():
            self.world.stop()  # PhysX views are rebuilt with the timeline stopped; deleting a prim invalidates them
        try:
            for r in new_robots:
                self._robots[r.id] = self._add_robot(r)
            for o in new_objects:
                self._objects[o.name] = build.add_object(self.world, o)
            for c in scene.cameras:
                if c.name not in self._cameras:
                    self._cameras[c.name] = build.add_camera(c)
            if rebuild or not self.world.is_playing():
                self.world.reset()  # initialises the physics views (and zeroes state; restored below)
                for r in new_robots:
                    self._dofs[r.id] = np.asarray(build.map_dofs(r, list(self._robots[r.id].dof_names)))
                    self._apply_keyframe(r)
            self._scene = scene
        except BaseException:
            self._undo(new_robots, new_objects)  # drop what this load added (self._scene is the old world), restart
            raise
        finally:
            self._restore(carried)
            self._time_base = t - float(self.world.current_time)  # a scene change is not a reset: time carries over

    def _undo(self, new_robots: list[SceneRobot], new_objects: list[SceneObject]) -> None:
        if self.world.is_playing():
            self.world.stop()
        self._remove_gone(self._scene)
        build.sweep(new_robots, new_objects, self._robots, self._objects)  # a reference an import left half-added
        self.world.reset()

    def _add_robot(self, r: SceneRobot) -> Any:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage

        usd = convert_to_usd(r.description.model_path, r.description.format, fix_base=not r.description.floating)
        add_reference_to_stage(usd_path=str(usd), prim_path=f"/World/{r.id}")
        art = SingleArticulation(
            prim_path=f"/World/{r.id}",
            name=r.id,
            position=np.asarray(r.position),
            orientation=np.asarray(r.orientation),
        )
        self.world.scene.add(art)
        return art

    def _remove_gone(self, scene: Scene) -> bool:
        from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

        keep = (
            ({r.id for r in scene.robots}, self._robots, "/World/{}"),
            ({o.name for o in scene.objects}, self._objects, "/World/objects/{}"),
            ({c.name for c in scene.cameras}, self._cameras, "/World/cameras/{}"),
        )
        removed = False
        for names, held, path in keep:
            for name in [n for n in held if n not in names]:
                if self.world.scene.object_exists(name):
                    self.world.scene.remove_object(name)  # deletes the handle's prim and forgets the registry entry
                if is_prim_path_valid(path.format(name)):  # a free file object's handle is a child: its root remains
                    delete_prim(path.format(name))
                held.pop(name)
                self._dofs.pop(name, None)
                removed = True
        return removed

    def _apply_keyframe(self, r: SceneRobot) -> None:
        kf = build.keyframe(r)
        if kf is None:
            return
        q, free = kf
        art, idx = self._robots[r.id], self._dofs[r.id]
        art.set_joint_positions(q, joint_indices=idx)
        art.set_joint_velocities(np.zeros_like(q), joint_indices=idx)
        self._drive(r.id, q)
        if free is not None:
            art.set_world_pose(position=np.asarray(r.position) + free[:3], orientation=free[3:])

    def _snapshot(self) -> dict[str, tuple[np.ndarray, ...]]:
        out: dict[str, tuple[np.ndarray, ...]] = {}
        for rid, art in self._robots.items():
            if rid in self._dofs:
                idx = self._dofs[rid]
                pos, quat = art.get_world_pose()
                q, v = np.asarray(art.get_joint_positions())[idx], np.asarray(art.get_joint_velocities())[idx]
                out[rid] = (q.copy(), v.copy(), np.asarray(pos), np.asarray(quat))
        return out

    def _restore(self, carried: dict[str, tuple[np.ndarray, ...]]) -> None:
        for rid, (q, v, pos, quat) in carried.items():
            art = self._robots.get(rid)
            if art is None:
                continue
            if self._scene.robot(rid).description.floating:
                art.set_world_pose(position=pos, orientation=quat)
            art.set_joint_positions(q, joint_indices=self._dofs[rid])
            art.set_joint_velocities(v, joint_indices=self._dofs[rid])
            self._drive(rid, q)

    def step(self, n: int) -> None:
        for _ in range(n):
            self.world.step(render=False)

    def reset(self, keyframe: str | None) -> None:
        self.world.reset()
        for r in self._scene.robots:
            self._apply_keyframe(SceneRobot(r.id, r.description, r.position, r.orientation, keyframe or r.keyframe))
        self._time_base = -float(self.world.current_time)  # world.reset() itself steps PhysX; reset means t = 0

    def time(self) -> float:
        return self._time_base + float(self.world.current_time)

    def joint_state(self, body: str) -> JointState:
        art, idx = self._robots[body], self._dofs[body]
        q = np.asarray(art.get_joint_positions())[idx]
        v = np.asarray(art.get_joint_velocities())[idx]
        try:
            e = np.asarray(art.get_measured_joint_efforts())[idx]
        except Exception:  # noqa: BLE001 — the effort view only exists after the first physics step
            e = np.zeros_like(q)
        return JointState(self._scene.robot(body).description.action_keys, q, v, e)

    def _drive(self, body: str, targets: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        action = ArticulationAction(joint_positions=np.asarray(targets, dtype=float), joint_indices=self._dofs[body])
        self._robots[body].apply_action(action)

    def set_targets(self, body: str, targets: np.ndarray) -> None:
        if len(targets) != len(self._dofs[body]):
            raise ValueError(f"{body}: {len(targets)} targets for {len(self._dofs[body])} actuators")
        self._drive(body, targets)

    def _prim(self, body: str) -> Any:
        if (prim := self._robots.get(body) or self._objects.get(body)) is None:
            raise KeyError(f"no body {body!r} in the world")
        return prim

    def body_pose(self, body: str) -> Pose:
        pos, quat = self._prim(body).get_world_pose()
        return Pose(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))

    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None:
        prim = self._prim(body)
        pos, quat = np.asarray(pose.position), np.asarray(pose.orientation)
        prim.set_world_pose(position=pos, orientation=quat)
        if body in self._robots and not self._scene.robot(body).description.floating:
            prim.set_default_state(position=pos, orientation=quat)  # a fixed base stays where it lands across reloads
        if not zero_velocity:
            return
        if body in self._robots:
            prim.set_joint_velocities(np.zeros(prim.num_dof))
        elif hasattr(prim, "set_linear_velocity"):
            prim.set_linear_velocity(np.zeros(3))
            prim.set_angular_velocity(np.zeros(3))

    def render(self, camera: str) -> np.ndarray:
        cam = self._cameras[camera]
        for _ in range(2 if self._rendered else 6):  # RTX needs a few frames to converge after stage changes
            self.world.render()
        self._rendered = True
        rgba = cam.get_rgba()
        if rgba is None or np.asarray(rgba).size == 0:
            raise RuntimeError(f"camera {camera!r} returned no frame yet")
        return np.ascontiguousarray(np.asarray(rgba)[..., :3]).astype(np.uint8)

    def contacts(self) -> list[Contact]:
        return []  # PhysX contact reports need a per-prim contact sensor; not wired yet

    def physics(self) -> Physics:
        return Physics("physx", float(self.world.get_physics_dt()), 1)

    def close(self) -> None:
        try:
            self.world.clear()
        finally:
            for held in (self._robots, self._objects, self._cameras, self._dofs):
                held.clear()
            self._ground = None
            self._lit = None
            app.release(self)
