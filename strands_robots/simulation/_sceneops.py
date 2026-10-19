"""Scene-editing verbs of a Simulation: add/remove a body, add an object, teleport, cameras, randomize.

These are methods of ``Simulation`` (it inherits this private base); they live here so ``simulation.py``
holds the world's lifecycle and time, and no file crosses 250 lines. Every method takes the world lock,
so scene, backend and registry change together and a locked reader never sees them disagree.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

import numpy as np

from ..geometry import moved_frame, vec
from ..types import Pose
from .backend import Backend
from .member import Member
from .object import Object
from .physics import Contact
from .scene import _IDENTITY_QUAT, Scene, SceneCamera, SceneObject, SceneRobot


class _SceneOps:
    """The scene-editing verbs of a Simulation (mixed into it)."""

    scene: Scene
    backend: Backend
    lock: threading.RLock
    revision: int
    render_size: tuple[int, int]
    backend_name: str
    _closed: bool
    _robots: dict[str, Member]

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError(f"this {self.backend_name} Simulation is closed; make a new one")

    def unique_id(self, want: str) -> str:
        if want not in self._robots:
            return want
        n = 2
        while f"{want}_{n}" in self._robots:
            n += 1
        return f"{want}_{n}"

    def add(
        self,
        robot: Member,
        *,
        position: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        keyframe: str | None = None,
    ) -> Member:
        """Put ``robot`` into this world (called by ``Robot.__init__``). Surviving bodies keep their state."""
        self._open()
        if robot.id in self._robots:
            raise ValueError(f"robot id {robot.id!r} is already in this world")
        pos = vec(robot.description.spawn_position if position is None else position, 3, "position")
        quat = vec(_IDENTITY_QUAT if orientation is None else orientation, 4, "orientation (w, x, y, z)")
        if keyframe is not None and keyframe not in robot.description.keyframes:
            raise ValueError(f"keyframe {keyframe!r} not in {robot.description.name}: {robot.description.keyframes}")
        with self.lock:  # scene, backend and registry change together: a locked reader never sees them disagree
            self._grow(self.scene.robots, SceneRobot(robot.id, robot.description, pos, quat, keyframe))  # type: ignore[arg-type]
            self._robots[robot.id] = robot
        return robot

    def _reload(self) -> None:
        with self.lock:
            self._open()  # camera(), randomize() and render() never rebuild a closed world
            self.backend.load(self.scene)
            self.revision += 1

    def _grow(self, items: list[Any], item: Any) -> None:
        """Append ``item`` and rebuild; a rebuild that fails leaves the scene as it was."""
        with self.lock:
            items.append(item)
            try:
                self._reload()
            except Exception:
                items.pop()
                raise

    def remove(self, robot: Member | str) -> None:
        """Take a robot, object or camera out of this world, by name (to move a camera: remove it, add it again)."""
        rid = robot if isinstance(robot, str) else robot.id
        if rid not in self._robots:
            with self.lock:
                objs = [o for o in self.scene.objects if o.name != rid]
                cams = [c for c in self.scene.cameras if c.name != rid]
                if (objs, cams) == (self.scene.objects, self.scene.cameras):
                    raise KeyError(f"nothing named {rid!r} in this world; robots are {list(self._robots)}")
                self.scene.objects, self.scene.cameras = objs, cams
                self._reload()
            return
        with self.lock:
            self.scene.robots = [r for r in self.scene.robots if r.id != rid]
            self.scene.cameras = [c for c in self.scene.cameras if c.attach != rid]
            self._reload()
            gone = self._robots.pop(rid)
        gone.detached()

    def add_object(
        self,
        name: str,
        *,
        shape: Literal["box", "sphere", "cylinder", "capsule"] | None = None,
        path: str | None = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.5),
        orientation: tuple[float, float, float, float] | None = None,
        size: float | tuple[float, ...] = (0.05, 0.05, 0.05),
        mass: float | None = None,
        color: tuple[float, float, float, float] | None = None,
        free: bool = True,
    ) -> Object:
        """Add a shape, or a mesh/MJCF file (path); position m, orientation quaternion w,x,y,z, color RGBA 0-1."""
        self._open()
        if (shape is None) == (path is None):
            raise ValueError("add_object needs exactly one of shape= or path=")
        if shape is not None and shape not in ("box", "sphere", "cylinder", "capsule"):
            raise ValueError(f"shape={shape!r}; choose from box, sphere, cylinder, capsule")
        if any(o.name == name for o in self.scene.objects) or name in self._robots:
            raise ValueError(f"name {name!r} is already used in this world")
        obj = SceneObject(
            name=name,
            shape=shape,
            path=path,
            size=tuple(float(x) for x in np.atleast_1d(size)),
            position=vec(position, 3, "position"),  # type: ignore[arg-type]
            orientation=vec(orientation or _IDENTITY_QUAT, 4, "orientation (w, x, y, z)"),  # type: ignore[arg-type]
            mass=mass,
            color=vec((0.6, 0.6, 0.6, 1.0) if color is None else color, 4, "color (RGBA 0-1)"),  # type: ignore[arg-type]
            free=free,
        )
        self._grow(self.scene.objects, obj)
        return Object(self, obj)  # type: ignore[arg-type]

    def set_pose(
        self,
        body: Member | Object | str,
        position: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ) -> Pose:
        """Teleport a robot or object (position m, quaternion w,x,y,z), velocities zeroed; returns where it is.

        A fixed base is rebuilt where it lands and stays there. A floating base keeps its spawn
        frame, so ``reset()`` stands it back on its keyframe instead of adding the keyframe's
        height on top of the teleport.
        """
        name = body if isinstance(body, str) else (body.id if hasattr(body, "id") else body.name)
        if name not in self._robots and not any(o.name == name for o in self.scene.objects):
            raise KeyError(f"no body {name!r} in this world")
        with self.lock:
            cur = self.backend.body_pose(name)
            pose = Pose(
                cur.position if position is None else np.asarray(vec(position, 3, "position")),
                cur.orientation if orientation is None else np.asarray(vec(orientation, 4, "orientation (w, x, y, z)")),
            )
            if name in self._robots and not self._robots[name].description.floating:
                sr = self.scene.robot(name)  # a fixed base is rebuilt from the scene, so the scene moves first
                f = moved_frame(Pose(np.asarray(sr.position), np.asarray(sr.orientation)), cur, pose)
                sr.position, sr.orientation = tuple(f.position.tolist()), tuple(f.orientation.tolist())  # type: ignore[assignment]
            for o in self.scene.objects:
                if o.name == name and not o.free:
                    o.position, o.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())  # type: ignore[assignment]
            self.backend.set_pose(name, pose, zero_velocity=True)
            self.revision += 1
            return self.backend.body_pose(name)

    def camera(
        self,
        name: str,
        *,
        position: tuple[float, float, float] = (1.5, -1.5, 1.2),
        target: tuple[float, float, float] = (0.0, 0.0, 0.3),
        orientation: tuple[float, float, float, float] | None = None,
        fov: float = 45.0,
        size: tuple[int, int] | None = None,
        attach: Member | str | None = None,
    ) -> SceneCamera:
        """Add a camera at position (m) looking at target, or along orientation (quaternion w,x,y,z); size W,H."""
        if any(c.name == name for c in self.scene.cameras):
            raise ValueError(f"camera {name!r} already exists")
        att = attach if (attach is None or isinstance(attach, str)) else attach.id
        if att is not None and att not in self._robots:
            raise KeyError(f"attach={att!r} is not a robot in this world")
        cam = SceneCamera(
            name=name,
            position=vec(position, 3, "position"),  # type: ignore[arg-type]
            target=None if orientation is not None else vec(target, 3, "target"),  # type: ignore[arg-type]
            orientation=None if orientation is None else vec(orientation, 4, "orientation (w, x, y, z)"),  # type: ignore[arg-type]
            fov=float(fov),
            size=tuple(size) if size else self.render_size,  # type: ignore[arg-type]
            attach=att,
        )
        self._grow(self.scene.cameras, cam)
        return cam

    def render(self, camera: str | SceneCamera | None = None) -> np.ndarray:
        """HxWx3 uint8 from a named camera, or from an auto-framed overview camera when none is given."""
        if camera is None:  # the auto-framed overview, created on first use (user cameras never replace it)
            if not any(c.name == "overview" for c in self.scene.cameras):
                self.camera("overview", **self._frame_all())
            name = "overview"
        else:
            name = camera if isinstance(camera, str) else camera.name
        assert name is not None
        if not any(c.name == name for c in self.scene.cameras):
            raise KeyError(f"no camera {name!r}; cameras are {[c.name for c in self.scene.cameras]}")
        with self.lock:
            return self.backend.render(name)

    def _frame_all(self) -> dict[str, Any]:
        """A camera position that sees every robot: look at their centroid from the -y side."""
        pts = np.array([r.position for r in self.scene.robots] or [(0.0, 0.0, 0.0)])
        center = pts.mean(axis=0) + np.array([0.0, 0.0, 0.4])
        spread = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
        dist = max(1.8, 1.4 * spread + 1.6)
        return {"position": tuple(center + np.array([0.0, -dist, 0.65 * dist])), "target": tuple(center)}

    def randomize(
        self,
        *,
        friction: float | tuple[float, float] | None = None,
        mass: float | tuple[float, float] | None = None,
        lighting: float | tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Scale friction / mass / light intensity (a factor or a (lo, hi) range to sample), rebuild; the scales."""
        rng = np.random.default_rng(seed)

        def pick(v: float | tuple[float, float] | None, current: float) -> float:
            return current if v is None else float(v) if isinstance(v, int | float) else float(rng.uniform(*v))

        with self.lock:
            sc = self.scene
            was = (sc.friction_scale, sc.mass_scale, sc.light_scale)
            sc.friction_scale, sc.mass_scale = pick(friction, was[0]), pick(mass, was[1])
            sc.light_scale = pick(lighting, was[2])
            try:
                self._reload()
            except Exception:  # an engine that cannot randomize this leaves the world as it was
                sc.friction_scale, sc.mass_scale, sc.light_scale = was
                raise
            return {"friction_scale": sc.friction_scale, "mass_scale": sc.mass_scale, "light_scale": sc.light_scale}

    def contacts(self) -> list[Contact]:
        with self.lock:  # a GPU solver mirrors its state into the CPU world on this read
            return self.backend.contacts()
