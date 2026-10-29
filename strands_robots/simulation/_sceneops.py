"""Scene-editing verbs of a Simulation (its private base): add/remove a body, add an object, teleport, cameras,
lights, randomize. Every verb takes the world lock, so scene, backend and registry change together and a locked reader
never sees them disagree; ``simulation.py`` keeps the world's lifecycle and time."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ..geometry import vec
from ..types import Pose
from . import _randomize, _teleport, _view
from .backend import Backend
from .lock import WorldLock
from .member import Member
from .object import Object
from .physics import Contact
from .scene import _IDENTITY_QUAT, Scene, SceneCamera, SceneLight, SceneObject, SceneRobot


class _SceneOps:
    """The scene-editing verbs of a Simulation (mixed into it)."""

    scene: Scene
    backend: Backend
    lock: WorldLock
    revision: int
    render_size: tuple[int, int]
    backend_name: str
    _closed: bool
    _robots: dict[str, Member]
    _editing: int
    _gone: list[Member]
    _built: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    _reload: Callable[[], None]  # Simulation's (rebuild now, or when ``with sim.editing():`` exits); so is _flush
    _flush: Callable[[], None]

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
            self._commit(self.scene.robots, SceneRobot(robot.id, robot.description, pos, quat, keyframe))  # type: ignore[arg-type]
            self._robots[robot.id] = robot
        return robot

    def _commit(self, items: list[Any], item: Any) -> None:
        """Append ``item`` to a scene list and rebuild; a rebuild that fails leaves the scene as it was."""
        with self.lock:
            items.append(item)
            try:
                self._reload()
            except Exception:
                items.pop()
                raise

    def remove(self, robot: Member | str) -> None:
        """Take a robot, object, camera or light out of this world, by name (``camera(name, ...)`` moves a camera)."""
        rid = robot if isinstance(robot, str) else robot.id
        if rid not in self._robots:
            with self.lock:
                s = self.scene
                kept = [[x for x in xs if x.name != rid] for xs in (s.objects, s.cameras, s.lights)]
                if kept == [s.objects, s.cameras, s.lights]:
                    raise KeyError(f"nothing named {rid!r} in this world; robots are {list(self._robots)}")
                s.objects, s.cameras, s.lights = kept  # type: ignore[assignment]
                self.scene.scales.pop(rid, None)  # a body re-added under this name inherits no draw
                self.scene.camera_jitter.pop(rid, None)
                self._reload()
            return
        with self.lock:
            self.scene.robots = [r for r in self.scene.robots if r.id != rid]
            for c in [c for c in self.scene.cameras if c.robot == rid]:
                self.scene.camera_jitter.pop(c.name, None)
            self.scene.cameras = [c for c in self.scene.cameras if c.robot != rid]
            self.scene.scales.pop(rid, None)
            self._reload()
            gone = self._robots.pop(rid)
            if self._editing:  # a failed block restores it, so it stays live until the block commits
                self._gone.append(gone)
                return
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
        self._commit(self.scene.objects, obj)
        return Object(self, obj)  # type: ignore[arg-type]

    def set_pose(
        self,
        body: Member | Object | str,
        position: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ) -> Pose:
        """Teleport a robot or object (position m, quaternion w,x,y,z), velocities zeroed; returns where it is.
        A fixed base is rebuilt where it lands and stays there; a floating base keeps its spawn frame, so
        ``reset()`` stands it back on its keyframe instead of adding the keyframe's height on top of the teleport."""
        name = body if isinstance(body, str) else (body.id if hasattr(body, "id") else body.name)
        if name not in self._robots and not any(o.name == name for o in self.scene.objects):
            raise KeyError(f"no body {name!r} in this world")
        with self.lock:
            self._flush()
            return _teleport.teleport(self, name, position, orientation)

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
        """Add a camera at position (m) looking at target, or along orientation (quaternion w,x,y,z); size W,H.
        Naming an existing camera re-aims it to these arguments, size kept unless given (on MuJoCo: no rebuild)."""
        with self.lock:  # the name lookup and the add (or re-aim) are one step under concurrent callers
            old = next((c for c in self.scene.cameras if c.name == name), None)
            att = attach if (attach is None or isinstance(attach, str)) else attach.id
            if att is not None and att.split("/")[0] not in self._robots:  # "<robot>" (its base) or "<robot>/<link>"
                raise KeyError(f"attach={att!r} is not a robot in this world (or '<robot>/<link>')")
            cam = SceneCamera(
                name=name,
                position=vec(position, 3, "position"),  # type: ignore[arg-type]
                target=None if orientation is not None else vec(target, 3, "target"),  # type: ignore[arg-type]
                orientation=None if orientation is None else vec(orientation, 4, "orientation (w, x, y, z)"),  # type: ignore[arg-type]
                fov=float(fov),
                size=tuple(size) if size else (old.size if old else self.render_size),  # type: ignore[arg-type]
                attach=att,
            )
            _view.put_camera(self, old, cam)  # a new name is added; a known one is re-aimed
            return cam

    def light(
        self,
        name: str,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 2.0),
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        directional: bool = False,
    ) -> SceneLight:
        """Add a light: a spot at position (m) shining along direction, or a sun (directional=True); color RGB 0-1.
        Naming a light that exists moves or recolors it; ``remove(name)`` takes it out."""
        self._open()
        return _view.light(self, SceneLight(name, *_view.light_vectors(position, direction, color), directional))

    def render(self, camera: str | SceneCamera | None = None) -> np.ndarray:
        """HxWx3 uint8 from a named camera, or from an auto-framed overview camera when none is given."""
        with self.lock:  # check and draw are one step: a concurrent edit cannot land between them
            if camera is None:  # the auto-framed overview, created on first use (user cameras never replace it)
                if not any(c.name == "overview" for c in self.scene.cameras):
                    self._flush()
                    w, h = self.render_size
                    self.camera("overview", **self.scene.overview(self.backend.outline(), w / h))
                name = "overview"
            else:
                name = camera if isinstance(camera, str) else camera.name
            assert name is not None
            if not any(c.name == name for c in self.scene.cameras):
                raise KeyError(f"no camera {name!r}; cameras are {[c.name for c in self.scene.cameras]}")
            self._flush()
            return self.backend.render(name)

    def randomize(
        self,
        *,
        friction: float | tuple[float, float] | None = None,
        mass: float | tuple[float, float] | None = None,
        lighting: float | tuple[float, float] | None = None,
        camera: float | None = None,
        target: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Scale friction / mass / light (a factor or a (lo, hi) range to sample), rebuild; returns what was drawn:
        the world-wide factors, ``scales[<target>]`` per targeted body and ``camera_jitter`` per camera.
        ``target`` scales one robot or object only; ``camera`` moves every camera up to that many metres per axis."""
        _randomize.check(self.scene, self._robots, target, lighting)
        with self.lock:
            was = _randomize.draw(
                self.scene, friction=friction, mass=mass, lighting=lighting, camera=camera, target=target, seed=seed
            )
            try:
                self._reload()
            except Exception:  # an engine that cannot randomize this leaves the world as it was
                _randomize.restore(self.scene, was)
                raise
            return _randomize.scales(self.scene)

    def contacts(self) -> list[Contact]:
        with self.lock:  # a GPU solver mirrors its state into the CPU world on this read
            self._flush()
            return self.backend.contacts()
