"""Dashboard: one Simulation served to a browser twin — scene, state frames, act.

    Dashboard(sim).serve(port=8090)

The dashboard holds a reference to the world, never a copy: every read takes ``sim.lock`` and reads the
world's own Scene and backend, so a robot added by a script shows up on the page with nothing to sync.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Any

from .._extras import hint
from . import confirm
from .auth import configure
from .connect import Connections
from .twin_scene import SceneXml, robot_cards, scene_to_mjcf

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..robot import Robot
    from ..simulation import Simulation
    from .server import Server


class Dashboard:
    """A browser twin of ``sim``; ``robots=`` are hardware robots (their twins get a real card); ``model=`` chats;
    ``voice=`` talks (``"openai"`` | ``"nova"`` | ``"auto"`` | a BidiModel factory, see ``dashboard.voice``); ``auth=``
    is passkey sign-in (``True``; ``self.auth.token()`` signs a script in; ``False`` is refused with a real robot);
    ``object_dir=`` is the one folder ``POST /api/world`` may load object files from (none by default)."""

    def __init__(
        self,
        sim: Simulation,
        *,
        hz: float = 30.0,
        robots: Iterable[Robot] = (),
        model: Any = None,
        agent: Any = None,
        mesh: Any = None,
        voice: Any = None,
        auth: Any = True,
        unauthenticated_hardware: bool = False,
        object_dir: Any = None,
    ) -> None:
        if hz <= 0:
            raise ValueError(f"hz must be > 0, got {hz}")
        self.sim = sim
        self.hz = hz
        self.real: dict[str, Robot] = {r.id: r for r in robots}
        self.auth = configure(auth, hardware=bool(self.real), insist=unauthenticated_hardware)  # dashboard/auth/
        self._scene_cache: tuple[int, SceneXml] | None = None
        self.object_dir = confirm.object_dir(object_dir)
        self._app: Any = None
        self.model = model
        self.mesh = mesh  # a strands_robots.mesh.Mesh: its devices are listed (and e-stoppable) next to the world's
        self._agent = agent
        if isinstance(voice, str) and voice not in ("openai", "nova", "auto"):  # voice.PROVIDERS; strands loads later
            raise ValueError(f"voice must be 'openai', 'nova', 'auto' or a BidiModel factory, got {voice!r}")
        self.voice = voice  # a fresh BidiModel per /ws/voice conversation
        self._agent_revision: int | None = None

    def scene_xml(self) -> SceneXml:
        with self.sim.lock:
            rev = self.sim.revision
            if self._scene_cache is None or self._scene_cache[0] != rev:
                self._scene_cache = (rev, scene_to_mjcf(self.sim.scene))
            return self._scene_cache[1]

    def real_cards(self) -> dict[str, Robot]:
        """Card id -> the hardware robot it stands for: a hardware robot's twin, or itself."""
        out: dict[str, Robot] = {r.twin.id: r for r in self.real.values() if r.twin is not None}
        out.update({rid: r for rid, r in self.real.items() if rid in self.sim.robots})
        return out

    def asset_file(self, path: str) -> str:
        file = self.scene_xml().assets.get(f"assets/{path}")
        if file is None:
            raise KeyError(f"no asset {path!r} in the current scene")
        return file

    def scene_json(self) -> dict[str, Any]:
        with self.sim.lock:
            sx = self.scene_xml()
            p = self.sim.physics
            return {
                "xml": sx.xml,
                "assets": [{"path": k, "url": f"/api/assets/{k[len('assets/') :]}"} for k in sx.assets],
                "robots": robot_cards(self.sim.scene, sx.roots, self.real_cards()),
                "objects": [
                    {"name": o.name, "shape": o.shape or o.path, "root": sx.roots.get(o.name)}
                    for o in self.sim.scene.objects
                ],
                "cameras": self.cameras_json(),
                "physics": dict(name=p.name, timestep=p.timestep, substeps=p.substeps, backend=self.sim.backend_name),
                "time": self.sim.time,
                "revision": self.sim.revision,
                "stats": sx.stats,
            }

    def cameras_json(self) -> list[dict[str, Any]]:
        """The world's cameras; each one's picture is ``GET /api/cameras/<name>.png``, what a policy sees."""
        with self.sim.lock:
            return [
                {"name": c.name, "size": list(c.size), "url": f"/api/cameras/{c.name}.png"}
                for c in self.sim.scene.cameras
            ]

    def world(self, call: Any, *, confirmed: bool = False) -> Any:
        """One ``sim.tool`` call from the page (``{"rpc": "add_object", "add_object": {...}}``): ``confirm.world``."""
        return confirm.world(self.sim, call, confirmed=confirmed, objects=self.object_dir)

    def frame(self) -> dict[str, Any]:
        """One state frame: joint values by model joint name, root poses by the browser's root body."""
        with self.sim.lock:
            sx = self.scene_xml()
            backend = self.sim.backend
            bodies: dict[str, Any] = {}
            for sr in self.sim.scene.robots:
                d = sr.description
                js = backend.joint_state(sr.id)
                pose = backend.body_pose(sr.id)
                bodies[sr.id] = {
                    "q": {f"{sr.id}/{d.unlabel(n)}": float(v) for n, v in zip(js.names, js.positions, strict=True)},
                    "root": sx.roots.get(sr.id),
                    "pose": [*pose.position.tolist(), *pose.orientation.tolist()],
                }
            for o in self.sim.scene.objects:
                if o.free:
                    pose = backend.body_pose(o.name)
                    bodies[o.name] = {
                        "q": {},
                        "root": sx.roots.get(o.name),
                        "pose": [*pose.position.tolist(), *pose.orientation.tolist()],
                    }
            cards = self.real_cards()  # a hardware card is its twin: report the REAL latch, as estop/resume resolve it
            estop = {rid: bool(getattr(cards.get(rid, r), "estopped", False)) for rid, r in self.sim.robots.items()}
            return {"t": self.sim.time, "revision": self.sim.revision, "bodies": bodies, "estop": estop}

    def _robot(self, rid: str) -> Robot:
        robots: dict[str, Any] = {**self.sim.robots, **self.real_cards(), **self.real}  # hardware wins over its twin
        if rid in robots:
            return robots[rid]  # type: ignore[no-any-return]
        raise KeyError(f"no robot {rid!r}; robots are {list(self.sim.robots) + list(self.real)}")

    def act(self, rid: str, targets: Any, *, confirmed: bool) -> dict[str, Any]:
        """One command from the page. A card that drives hardware needs the operator's ``confirmed``."""
        robot = self._robot(rid)
        if not isinstance(targets, dict) or not targets:
            raise ValueError('body must be {"targets": {joint: value, ...}}')
        unknown = [k for k in targets if k not in robot.action_keys]
        if unknown:
            raise ValueError(f"{rid} has no joints {unknown}; action keys are {list(robot.action_keys)}")
        try:  # the robot's own safety chain decides; X-Confirm is the operator's yes for the last step
            obs = robot.confirm_act({k: float(v) for k, v in targets.items()}, confirmed=confirmed)
        except PermissionError as exc:
            if robot.is_real and not confirmed and not getattr(robot, "estopped", False):
                raise type(exc)(f"{exc} (from this page: confirm with header X-Confirm: yes)") from exc
            raise
        return {"t": obs.t, "joints": dict(zip(obs.joints.names, obs.joints.positions.tolist(), strict=True))}

    def add_robot(self, name: Any, *, id: str | None = None, position: Any = None) -> dict[str, Any]:
        """Spawn a registry robot into THIS world; the page recompiles on the next frame (revision bumps)."""
        from ..robot import Robot

        if not isinstance(name, str) or not name:
            raise ValueError('body must be {"name": "<registry name>", "id": <str|null>, "position": [x, y, z]}')
        robot = Robot(name, sim=self.sim, id=id, position=position)
        return {"id": robot.id, "name": robot.name, "action_keys": [*robot.action_keys], "revision": self.sim.revision}

    def estop(self, rid: str) -> dict[str, Any]:
        """The page's E-STOP: bypasses the safety chain and latches (real or simulated, no confirm needed)."""
        robot = self._robot(rid)
        robot.estop()
        return {"id": rid, "estop": robot.estopped}

    def resume(self, rid: str, *, approved_by: Any, confirmed: bool = False) -> dict[str, Any]:
        """Release the latch; refused (a 403) without the operator's name, and for a real body without X-Confirm."""
        robot = self._robot(rid)
        if robot.is_real and not confirmed:
            raise PermissionError(f"{rid} is a real robot: resume it from this page with header X-Confirm: yes")
        robot.resume(approved_by=approved_by if isinstance(approved_by, str) else None)
        log.warning("dashboard: %s e-stop released, approved_by=%r", rid, robot.estop_latch.resumed_by)
        return {"id": rid, "estop": robot.estopped, "resumed_by": robot.estop_latch.resumed_by}

    def mesh_json(self) -> dict[str, Any]:
        """``mesh.discover()`` now: every device present, with the page's own world marked (``local``)."""
        if self.mesh is None:
            return {"tenant": None, "devices": []}
        devices = [d.to_dict() for d in self.mesh.discover("device(*)")]
        for d in devices:
            d["local"] = d["id"] in self.sim.robots or d["id"] in self.real
        return {"tenant": getattr(self.mesh, "tenant", None), "devices": devices}

    def mesh_estop(self, device_id: str) -> Any:
        """E-stop one mesh device by id (stopping is always allowed; resume stays with that device's authority)."""
        if self.mesh is None:
            raise KeyError("this dashboard has no mesh (Dashboard(sim, mesh=Mesh(...)))")
        return self.mesh.invoke(device_id, "estop")

    @cached_property
    def connections(self) -> Any:  # the Connect panel's meshes, remote robots and policies (dashboard/connect.py)
        return Connections(self)

    @property
    def agent(self) -> Any:
        """The chat agent over ``sim.tool`` + every ``robot.tool``; its tools follow the world's robots."""
        from . import chat

        if self._agent is None:
            if self.model is None:
                raise RuntimeError(chat.NO_CHAT)
            self._agent = chat.build_agent(self.sim, self.model, self.real.values())
        elif self._agent_revision != self.sim.revision:
            chat.refresh_tools(self._agent, self.sim, self.real.values())
        self._agent_revision = self.sim.revision
        return self._agent

    @property
    def app(self) -> Any:
        """The ASGI app (FastAPI is imported here, on first use)."""
        if self._app is None:
            self._app = _web()[0](self)
        return self._app

    def start(self, *, host: str = "127.0.0.1", port: int = 0) -> Server:
        """Serve in a background thread; ``.url`` and ``.stop()`` on the returned handle."""
        return self.auth.announce(_web()[1](self.app, host=host, port=port)).start()

    def serve(self, *, host: str = "127.0.0.1", port: int = 8090) -> None:
        """Serve in this thread until interrupted. Loopback only."""
        self.auth.announce(_web()[1](self.app, host=host, port=port)).run()


def _web() -> tuple[Any, type[Server]]:
    """The web stack (fastapi + uvicorn), imported on first use; missing, the error names the extra."""
    try:
        from . import app, server
    except ImportError as e:
        raise ImportError(f"the dashboard needs its web server: {hint('dashboard')} ({e})") from e
    return app.build_app, server.Server
