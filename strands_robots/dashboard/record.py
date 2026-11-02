"""Recording from the page: pick robots and cameras, name the task, then save or discard episodes as you go.

Each episode is written by the package's own ``recording.Episode`` (LeRobot's ``LeRobotDataset`` underneath), so a
dataset recorded here is the dataset ``robot.run(record=...)`` writes. The recorder only observes: it samples every
chosen robot at ``fps`` on the wall clock, whatever moves it (a slider, a policy run, the chat agent), and records
the last targets written as the action (the measured joints before any write). It never writes to a body.

A LeRobot dataset holds one robot, so several robots record into one dataset each: ``target/<robot id>`` for a
directory, ``hf://owner/name-<robot id>`` for a Hub name. Episodes stay in step: save and discard act on every
robot's current episode at once. A Hub name is only a name here: nothing is uploaded from the page.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from ..recording import Episode
from . import takes
from .takes import State, Track, targets_for

if TYPE_CHECKING:
    from .dashboard import Dashboard

log = logging.getLogger(__name__)
__all__ = ["Recorder", "targets_for"]


class Recorder:
    """The dashboard's one recording: ``start`` / ``save`` / ``discard`` / ``stop`` / ``status``."""

    def __init__(self, dash: Dashboard) -> None:
        self.dash = dash
        self._lock = threading.RLock()  # one transition at a time; the sampler takes it per tick
        self._s = State()
        self._tracks: list[Track] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warm: threading.Thread | None = None

    def warm(self) -> None:
        """Import LeRobot in the background (torch makes it take tens of seconds), once: the page asks this when
        the Record panel loads, so pressing REC does not wait for an import."""
        if self._warm is None:
            from ..recording.lerobot_v3 import require_lerobot

            def load() -> None:
                try:
                    require_lerobot()
                except ImportError:
                    pass  # start() raises it with the install line

            self._warm = threading.Thread(target=load, name="dashboard-record-warm", daemon=True)
            self._warm.start()

    def start(
        self, *, robots: Any, cameras: Any, target: Any, task: Any, fps: Any, units: Any = "auto"
    ) -> dict[str, Any]:
        with self._lock:
            if self._s.state in ("recording", "saving"):
                raise RuntimeError("a recording is running; POST /api/record/stop first")
            tracks = takes.plan(self.dash, robots, cameras, target, fps, task, units)
            self._s = State(
                state="recording", target=str(target).strip(), task=(task or "").strip() or None, fps=int(fps)
            )
            self._tracks = tracks
            try:
                self._open_episodes()
            except Exception:
                self._close_all()
                self._s = State()
                self._tracks = []
                raise
            self._s.started = time.monotonic()
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample, name="dashboard-record", daemon=True)
            self._thread.start()
            log.info("dashboard: recording %s to %s at %d fps", robots, target, fps)
            return self.status()

    def save(self) -> dict[str, Any]:
        """Close the current episode of every robot into its dataset and open the next one."""
        with self._lock:
            self._need_recording()
            self._s.state = "saving"
            try:
                self._finish_episodes()
                self._open_episodes()
                self._s.state = "recording"
            except Exception as exc:
                self._fail(exc)
                raise
            return self.status()

    def discard(self) -> dict[str, Any]:
        """Drop the current episode of every robot and start it again."""
        with self._lock:
            self._need_recording()
            for t in self._tracks:
                if t.episode is not None:
                    t.episode.abort()
                    t.episode = None
            try:
                self._open_episodes()
            except Exception as exc:
                self._fail(exc)
                raise
            return self.status()

    def stop(self, *, save: Any = True) -> dict[str, Any]:
        """End the recording; its current episode is saved (``save``, default) or dropped."""
        if not isinstance(save, bool):
            raise ValueError('"save" is true or false')
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        with self._lock:
            if self._s.state == "idle":
                return self.status()
            failed = self._s.state == "error"
            try:
                if save and not failed and self._s.frames:
                    self._s.state = "saving"
                    self._finish_episodes()
            finally:
                self._close_all()
                self._s.state = "idle"
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return takes.status_json(self._s, self._tracks)

    def options(self) -> dict[str, Any]:
        """What the page can record: every robot with the cameras it carries, every world camera, a default target."""
        self.warm()
        return takes.options(self.dash)

    def close(self) -> None:
        if self._s.state in ("recording", "saving"):
            self.stop(save=False)

    def _need_recording(self) -> None:
        if self._s.state != "recording":
            raise RuntimeError(f"nothing is recording (state {self._s.state}); POST /api/record/start first")

    def _open_episodes(self) -> None:
        """A new Episode per robot, opened on one observation (creates or resumes its dataset, or refuses it)."""
        for t in self._tracks:
            t.episode = Episode(
                t.target,
                description=t.robot.description,
                hz=float(self._s.fps),
                instruction=self._s.task,
                units=t.units,
            )
            t.episode.open(lambda t=t: takes.observe(self.dash, t))  # type: ignore[misc]
        self._s.frames = 0
        self._s.episode_started = time.monotonic()

    def _finish_episodes(self) -> None:
        seconds = round(time.monotonic() - self._s.episode_started, 2)
        frames = self._s.frames
        for t in self._tracks:
            ep, t.episode = t.episode, None
            if ep is None:
                continue
            path = ep.finish()
            if path is None:
                continue
            t.path, t.saved = path, t.saved + 1
            self._s.episodes.append(
                {
                    "index": self._s.episode,
                    "robot": t.robot.id,
                    "frames": frames,
                    "seconds": seconds,
                    "dataset": t.target,
                    "path": path,
                }
            )
        if frames:
            self._s.episode += 1
        self._s.frames = 0

    def _close_all(self) -> None:
        for t in self._tracks:
            if t.episode is not None:
                try:
                    t.episode.abort()
                except Exception:  # noqa: BLE001 — closing after a failure must close the others too
                    log.exception("dashboard: closing the %s recording", t.robot.id)
                t.episode = None

    def _fail(self, exc: BaseException) -> None:
        self._s.error = f"{type(exc).__name__}: {exc}"
        self._s.state = "error"
        self._stop.set()
        self._close_all()
        log.error("dashboard: recording stopped: %s", self._s.error)

    def _sample(self) -> None:
        period = 1.0 / self._s.fps
        nxt = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if self._s.state == "recording":
                    try:
                        for t in self._tracks:
                            obs = takes.observe(self.dash, t)
                            assert t.episode is not None
                            t.episode.add(obs, takes.action(t, obs))
                        self._s.frames += 1
                        now = time.monotonic()
                        self._s.ticks = [x for x in self._s.ticks if now - x <= 2.0] + [now]
                    except Exception as exc:  # noqa: BLE001 — a failed frame ends the recording with its reason
                        self._fail(exc)
                        return
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:  # fell behind (a save encoding video): re-phase instead of bursting
                nxt = time.monotonic()
                delay = 0.0
            self._stop.wait(max(0.0, delay))
