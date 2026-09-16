"""A simulated robot stepping in this process, owned by one thread.

``SimSession`` wraps the engine ``Robot(name, mode="sim")`` returns. One worker
thread creates the engine, steps it in real time, renders frames and applies
queued commands; every other thread reads an immutable snapshot. This is not a
style choice: the renderer's GL context is bound to the thread that made it, and
MuJoCo's ``MjData`` is not safe to step and read from two threads at once.

Freezing is the e-stop. A frozen session stops stepping but keeps rendering
and keeps answering telemetry, so an operator sees the robot exactly where it
stopped. A queued command that would move the robot is refused when the worker
reaches it rather than applied - the refusal is that command's own result, so
the caller is told, never silently dropped. A command the worker was already
inside when the freeze landed cannot be recalled: that write completes, and the
route answers the request 423 with the lockout still latched.

Nothing here knows about the mesh, hardware, or HTTP.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_SESSIONS = 4

#: Commands that move the robot, and so are refused while a session is frozen.
#: A read (``state``) is still answered, because answering it moves nothing.
MOTION_COMMANDS = frozenset({"reset", "set_joints", "step"})
_RENDER_FPS = 15.0
_RENDER_SIZE = (512, 384)  # width, height: a live view, not a dataset frame
_TELEMETRY_HZ = 30.0


@dataclass(frozen=True)
class Snapshot:
    """What every reader gets: the sim as of one instant, on this dashboard's clock."""

    id: str
    robot: str
    state: str  # starting | running | frozen | error | stopped
    sim_time: float
    steps: int
    joint_names: tuple[str, ...]
    qpos: tuple[float, ...]
    fps: float
    cameras: tuple[str, ...]
    error: str | None = None
    model_path: str | None = None
    created: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """The snapshot as JSON-ready fields, floats rounded for the wire."""
        return {
            "id": self.id,
            "robot": self.robot,
            "state": self.state,
            "sim_time": round(self.sim_time, 4),
            "steps": self.steps,
            "joint_names": list(self.joint_names),
            "qpos": [round(q, 5) for q in self.qpos],
            "fps": round(self.fps, 1),
            "cameras": list(self.cameras),
            "error": self.error,
            "model_path": self.model_path,
            "created": self.created,
        }


@dataclass
class _Command:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class SimSession:
    """One simulated robot in one worker thread. See the module docstring."""

    def __init__(self, robot: str, *, engine_factory: Callable[[str], Any] | None = None, realtime: bool = True):
        self.id = uuid.uuid4().hex[:8]
        self.robot = robot
        self._factory = engine_factory or _default_factory
        self._realtime = realtime
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._frozen = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._snapshot = Snapshot(self.id, robot, "starting", 0.0, 0, (), (), 0.0, (), created=time.time())
        self._thread = threading.Thread(target=self._run, name=f"sim-{robot}-{self.id}", daemon=True)
        self._thread.start()

    # -- readers (any thread) ------------------------------------------------

    @property
    def snapshot(self) -> Snapshot:
        """The latest immutable snapshot."""
        with self._lock:
            return self._snapshot

    def latest_frame(self) -> np.ndarray | None:
        """The most recent rendered RGB frame, for :func:`mjpeg_frames`."""
        with self._lock:
            return self._frame

    @property
    def frozen(self) -> bool:
        """Whether the e-stop has this session stopped."""
        return self._frozen.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the engine exists (or failed). Tests and the create route use it."""
        return self._ready.wait(timeout)

    # -- controls (any thread) -----------------------------------------------

    def freeze(self) -> None:
        """Stop stepping; keep rendering and answering. The e-stop."""
        self._frozen.set()
        self._publish(state="frozen")

    def thaw(self) -> None:
        """Resume stepping after :meth:`freeze`."""
        self._frozen.clear()
        self._publish(state="running")

    def stop(self, timeout: float = 5.0) -> None:
        """End the worker and close the engine."""
        self._stop.set()
        self._thread.join(timeout)
        self._publish(state="stopped")

    def command(self, kind: str, timeout: float = 5.0, **payload: Any) -> dict[str, Any]:
        """Run one engine call on the worker thread and return its result.

        Raises:
            RuntimeError: when the session is not running, or the call timed out.
        """
        if self._stop.is_set() or self.snapshot.state == "error":
            raise RuntimeError(f"session {self.id} is not running")
        cmd = _Command(kind, payload)
        self._commands.put(cmd)
        if not cmd.done.wait(timeout):
            raise RuntimeError(f"{kind} timed out after {timeout:.0f}s")
        assert cmd.result is not None
        return cmd.result

    # -- worker --------------------------------------------------------------

    def _publish(self, **changes: Any) -> None:
        with self._lock:
            current = self._snapshot.__dict__ | changes
            self._snapshot = Snapshot(**current)

    def _run(self) -> None:
        try:
            engine = self._factory(self.robot)
        except Exception as exc:
            logger.warning("sim %s (%s) failed to start: %s", self.id, self.robot, exc)
            self._publish(state="error", error=f"{type(exc).__name__}: {exc}")
            self._ready.set()
            return

        names = tuple(str(n) for n in engine.robot_joint_names(self.robot))
        cameras = tuple(engine.list_cameras())
        dt = float(engine.mj_model.opt.timestep)
        self._publish(
            # An e-stop that arrived while the engine was building already set
            # the flag; this first publish reports that, not "running".
            state="frozen" if self._frozen.is_set() else "running",
            joint_names=names,
            cameras=cameras,
            model_path=_model_path(self.robot),
        )
        self._ready.set()

        last_render = 0.0
        last_wall = time.monotonic()
        frames = 0
        fps = 0.0
        fps_window = time.monotonic()
        steps = 0
        try:
            while not self._stop.is_set():
                self._drain(engine)
                now = time.monotonic()
                if not self._frozen.is_set():
                    # Real time: step as many physics ticks as wall time asks for, capped
                    # so a stall never turns into a burst.
                    n = min(int((now - last_wall) / dt), 200) if self._realtime else 1
                    if n > 0:
                        engine.step(n)
                        steps += n
                        last_wall = now
                else:
                    last_wall = now
                if now - last_render >= 1.0 / _RENDER_FPS:
                    rgb, _ = engine.get_frame(width=_RENDER_SIZE[0], height=_RENDER_SIZE[1])
                    with self._lock:
                        self._frame = rgb
                    last_render = now
                    frames += 1
                    if now - fps_window >= 1.0:
                        fps = frames / (now - fps_window)
                        frames = 0
                        fps_window = now
                self._publish(
                    sim_time=float(engine.mj_data.time),
                    steps=steps,
                    qpos=tuple(float(q) for q in engine.mj_data.qpos),
                    fps=fps,
                    state="frozen" if self._frozen.is_set() else "running",
                )
                time.sleep(1.0 / _TELEMETRY_HZ)
        except Exception as exc:
            logger.exception("sim %s (%s) died", self.id, self.robot)
            self._publish(state="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("engine close failed", exc_info=True)

    def _drain(self, engine: Any) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if self._frozen.is_set() and cmd.kind in MOTION_COMMANDS:
                    cmd.result = {
                        "status": "error",
                        "content": [{"text": f"{cmd.kind}: refused, this session is frozen by an e-stop"}],
                    }
                else:
                    cmd.result = self._apply(engine, cmd)
            except Exception as exc:
                cmd.result = {"status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
            finally:
                cmd.done.set()

    def _apply(self, engine: Any, cmd: _Command) -> dict[str, Any]:
        if cmd.kind == "reset":
            return dict(engine.reset())
        if cmd.kind == "set_joints":
            return dict(engine.set_joint_positions(cmd.payload["positions"], robot_name=self.robot))
        if cmd.kind == "state":
            return dict(engine.get_robot_state(self.robot))
        if cmd.kind == "step":
            return dict(engine.step(int(cmd.payload.get("n", 1))))
        return {"status": "error", "content": [{"text": f"unknown command {cmd.kind}"}]}


def _default_factory(robot: str) -> Any:
    from strands_robots import Robot

    return Robot(robot, mode="sim")


def _model_path(robot: str) -> str | None:
    try:
        from strands_robots.simulation.model_registry import resolve_model

        return resolve_model(robot)
    except Exception:
        return None


class SessionStore:
    """The sessions this process is running, capped so a page cannot fork the machine."""

    def __init__(self, limit: int = MAX_SESSIONS):
        self.limit = limit
        self._sessions: dict[str, SimSession] = {}
        self._lock = threading.Lock()

    def create(self, robot: str, **kwargs: Any) -> SimSession:
        """Start a session, or raise ``RuntimeError`` when the cap is reached."""
        with self._lock:
            live = [s for s in self._sessions.values() if s.snapshot.state not in ("stopped", "error")]
            if len(live) >= self.limit:
                raise RuntimeError(f"{self.limit} sessions already running; stop one first")
            session = SimSession(robot, **kwargs)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> SimSession | None:
        """The session with this id, or None."""
        with self._lock:
            return self._sessions.get(session_id)

    def all(self) -> list[SimSession]:
        """Every session, including stopped ones not yet removed."""
        with self._lock:
            return list(self._sessions.values())

    def remove(self, session_id: str) -> bool:
        """Stop and forget a session. False when there was none."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop()
        return True

    def freeze_all(self) -> list[str]:
        """Freeze every session that can still step; returns the ids frozen.

        A session in ``starting`` is included: its engine is still being built,
        so it has not stepped yet and will begin the moment the build returns.
        Selecting on ``running`` alone would leave that one stepping after an
        e-stop, which is the window the e-stop exists for.
        """
        ids = []
        for s in self.all():
            if s.snapshot.state not in ("stopped", "error"):
                s.freeze()
                ids.append(s.id)
        return ids

    def thaw_all(self) -> list[str]:
        """Thaw every frozen session; returns the ids thawed."""
        ids = []
        for s in self.all():
            if s.snapshot.state == "frozen":
                s.thaw()
                ids.append(s.id)
        return ids

    def shutdown(self) -> None:
        """Stop every session. Called at app shutdown."""
        for s in self.all():
            s.stop()
