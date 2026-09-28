"""The record session state machine and its control loop (leader drives follower, frames are recorded)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from strands_robots.dashboard import record_motion

logger = logging.getLogger(__name__)

THUMB_MAX_WIDTH = 160


class TeleopBackend(Protocol):
    """What the worker needs from hardware. One leader, one follower."""

    #: distinct camera names in the follower observation (schema order)
    camera_keys: list[str]

    def leader_action(self) -> dict[str, float]:
        """Read the leader arm - the action to mirror and record."""
        ...

    def follower_apply(self, action: dict[str, float]) -> dict[str, float]:
        """Send the action to the follower; returns the action as sent."""
        ...

    def follower_observation(self) -> dict[str, Any]:
        """Follower joints + camera frames (name -> float | ndarray)."""
        ...

    def close(self) -> None:
        """Release both arms."""
        ...


def achieved_fps(frames: int, duration_s: float) -> float | None:
    """The rate frames were REALLY captured at, or None when unknowable."""
    if frames < 2 or duration_s <= 0:
        return None
    return round(frames / duration_s, 2)


FPS_DRIFT_TOLERANCE = 0.10


def fps_verdict(declared: int, measured: float | None) -> dict[str, Any] | None:
    """What the operator must know when the dataset's fps is not its rate."""
    if measured is None or declared <= 0:
        return None
    if abs(measured - declared) <= FPS_DRIFT_TOLERANCE * declared:
        return None
    slower = measured < declared
    ratio = (declared / measured) if slower else (measured / declared)
    return {
        "declared_fps": declared,
        "measured_fps": measured,
        "ratio": round(ratio, 2),
        "slower": slower,
        "detail": (
            f"frames are being captured at {measured:g}/s but this dataset declares "
            f"{declared}/s, so every timestamp in it will be {round(ratio, 2)}x "
            + ("closer together than reality" if slower else "further apart than reality")
            + ". A policy trained on it learns the wrong control period, and replay runs "
            "at the wrong speed. Stop, re-open the session at "
            f"{max(1, int(measured))} fps, and the episodes will be honest."
        ),
    }


class EpisodeState:
    """Bookkeeping for one episode; ``to_dict`` speaks the wire contract."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.frames = 0
        self.started_at = 0.0
        self.duration_s = 0.0
        self.thumbnails: dict[str, str] = {}
        self.discarded = False

    def to_dict(self) -> dict[str, Any]:
        """The episode as the frontend reads it."""
        return {
            "index": self.index,
            "frames": self.frames,
            "duration_s": round(self.duration_s, 1),
            # The rate this episode was really captured at, next to the frame
            # count it is derived from - so a short episode's rate can be judged
            # by the reader instead of trusted.
            "fps_achieved": achieved_fps(self.frames, self.duration_s),
            "thumbnails": dict(self.thumbnails),
            "discarded": self.discarded,
        }


def _save_thumbnail(frame: Any, path: Path) -> bool:
    """Write a small JPEG for the episode strip; best-effort, never raises."""
    try:
        import numpy as np

        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return False
        if arr.shape[1] > THUMB_MAX_WIDTH:
            step = max(1, arr.shape[1] // THUMB_MAX_WIDTH)
            arr = arr[::step, ::step]
        try:
            import cv2

            path.parent.mkdir(parents=True, exist_ok=True)
            # observations are RGB; cv2 writes BGR
            return bool(cv2.imwrite(str(path), arr[:, :, 2::-1]))
        except ImportError:
            from PIL import Image

            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr[:, :, :3].astype("uint8")).save(path)
            return True
    except Exception as exc:  # noqa: BLE001 - decoration only
        logger.debug("thumbnail skipped: %r", exc)
        return False


def upload_verdict(
    *,
    asked_repo_id: str | None,
    dataset: str,
    push: Callable[[], Any],
) -> dict[str, Any]:
    """Run the push and say what really happened; "pushed" only on a recognised success."""
    if asked_repo_id and asked_repo_id != dataset:
        return {
            "ok": False,
            "detail": (
                f"NOT uploaded: a dataset publishes under the name it was recorded with "
                f"({dataset}), and this asked for {asked_repo_id}. The episodes are saved "
                f"locally; to publish as {asked_repo_id}, record a session under that name"
            ),
        }
    try:
        result = push()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"saved locally, upload FAILED: {exc}"}
    status = (result or {}).get("status") if isinstance(result, dict) else None
    if status == "success":
        return {"ok": True, "detail": f"pushed to {(result or {}).get('repo_id') or dataset}"}
    if status == "error":
        # The recorder's own message names the cause (empty dataset, 403, no token).
        return {
            "ok": False,
            "detail": f"saved locally, upload REFUSED: {(result or {}).get('message') or 'no reason given'}",
        }
    # An unrecognised answer is not evidence of a push. Saying "pushed" here is
    # the exact lie this function exists to remove.
    return {
        "ok": False,
        "detail": (
            "saved locally; the upload returned an answer this dashboard cannot read "
            f"({result!r}), so whether it published is UNKNOWN - check the Hub"
        ),
    }


class RecordWorker:
    """The record session state machine + control loop. Phases: ``idle`` (between episodes) and
    ``recording``.
    """

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        leader: str,
        follower: str,
        target_episodes: int,
        fps: int,
        backend: TeleopBackend,
        recorder_factory: Callable[..., Any],
        thumb_dir: Path | str,
        clock: Callable[[], float] = time.monotonic,
        autostart_loop: bool = True,
    ) -> None:
        if not dataset or not dataset.strip():
            raise ValueError("dataset name is required")
        if not task or not task.strip():
            raise ValueError("task is required - it is written on every frame")
        if leader == follower:
            raise ValueError("leader and follower must be different arms")
        fps = int(fps)
        if fps < 1 or fps > 120:
            raise ValueError("fps must be between 1 and 120")
        target = int(target_episodes)
        if target < 1:
            raise ValueError("target_episodes must be at least 1")

        self.dataset = dataset.strip()
        self.task = task.strip()
        self.leader = leader
        self.follower = follower
        self.target_episodes = target
        self.fps = fps
        self._backend = backend
        self._recorder = recorder_factory(repo_id=self.dataset, fps=fps, task=self.task)
        self._thumb_dir = Path(thumb_dir)
        self._clock = clock

        self._lock = threading.RLock()
        self._phase = "idle"
        self._episodes: list[EpisodeState] = []
        self._current: EpisodeState | None = None
        self._closed = False
        self._last_error: str | None = None
        # One (timestamp, joint positions) sample per RECORDED frame, pruned to record_motion's
        # window.
        self._motion: list[tuple[float, dict[str, float]]] = []
        # The verdict for the episode in flight, kept after it stops so the panel can still say "that
        # one never moved" while the operator reads it in ``idle`` - the same reason _measured_fps
        # falls back to the last episode.
        self._motion_notice: dict[str, Any] | None = None

        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        if autostart_loop:
            self._thread = threading.Thread(target=self._loop, name="record-loop", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------- session

    def session(self) -> dict[str, Any]:
        """The live session view, declared and measured fps side by side."""
        with self._lock:
            return {
                "dataset": None if self._closed else self.dataset,
                "task": self.task,
                "leader": self.leader,
                "follower": self.follower,
                "target_episodes": self.target_episodes,
                "fps": self.fps,
                "phase": self._phase,
                "episodes": [e.to_dict() for e in self._episodes]
                + ([self._current.to_dict()] if self._current else []),
                "error": self._last_error,
                # The DECLARED fps above is what lands in the dataset; this is
                # what the loop actually managed. Both, always, because the gap
                # between them is invisible in the artifact.
                "fps_achieved": self._measured_fps(),
                "fps_notice": fps_verdict(self.fps, self._measured_fps()),
                "motion_notice": self._motion_verdict_locked(),
                # None unless a requested camera is missing: absent cameras
                # must be visible BEFORE 10 episodes are collected blind.
                "camera_notice": getattr(self._backend, "camera_notice", None),
            }

    # ------------------------------------------------------------ commands

    def start_episode(self) -> dict[str, Any]:
        """Open a new episode; idempotent while one is recording."""
        with self._lock:
            self._require_open()
            if self._phase == "recording":
                return self.session()  # idempotent, per contract
            self._current = EpisodeState(index=self._next_index())
            self._current.started_at = self._clock()
            self._last_error = None
            # A fresh episode starts with no motion history: carrying the previous one's samples over
            # would let a long pause between episodes - which is exactly when the operator lines the arms
            # up - be reported as this episode holding still.
            self._motion = []
            self._motion_notice = None
            self._phase = "recording"
        return self.session()

    def stop_episode(self) -> dict[str, Any]:
        """Judge the episode's motion, save it, and return to idle."""
        with self._lock:
            self._require_open()
            if self._phase != "recording" or self._current is None:
                return self.session()
            ep = self._current
            # Judge the episode ON STOP, not only when someone polls: computing this solely in session()
            # made the verdict depend on whether anyone was WATCHING - an unattended collection run would
            # keep the frames and lose the finding.
            self._motion_notice = (
                record_motion.motion_verdict(self._motion, now=self._clock(), frames=ep.frames) or self._motion_notice
            )
            self._phase = "idle"
            self._current = None
            if ep.frames == 0:
                # An empty episode cannot be saved (LeRobot rejects empty
                # buffers) and keeping a 0-frame entry would lie about the
                # dataset's contents. Surface it instead of half-keeping it.
                self._last_error = "episode had 0 frames - nothing was captured, check teleop"
                return self.session()
            info = self._recorder.save_episode()
            if isinstance(info, dict) and info.get("status") == "error":
                self._last_error = str(info.get("message", "save_episode failed"))
                return self.session()
            ep.duration_s = self._clock() - ep.started_at
            self._episodes.append(ep)
        return self.session()

    def redo_episode(self) -> dict[str, Any]:
        """Throw the episode in flight away and return to idle."""
        with self._lock:
            self._require_open()
            if self._phase != "recording":
                return self.session()
            self._recorder.clear_episode_buffer()
            self._current = None
            self._phase = "idle"
        return self.session()

    def discard(self, index: int) -> dict[str, Any]:
        """Mark a saved episode discarded; raises KeyError for an unknown index."""
        with self._lock:
            self._require_open()
            for ep in self._episodes:
                if ep.index == int(index):
                    ep.discarded = True
                    break
            else:
                raise KeyError(f"no saved episode with index {index}")
        return self.session()

    def close(self, *, upload: bool = False, repo_id: str | None = None) -> dict[str, Any]:
        """Stop the loop, finalize the dataset, optionally push it, and release the arms."""
        with self._lock:
            if self._closed:
                return {"ok": True, "detail": "session already closed"}
            if self._phase == "recording":
                # closing mid-episode keeps nothing half-written
                self._recorder.clear_episode_buffer()
                self._current = None
                self._phase = "idle"
            self._closed = True
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        kept = [e for e in self._episodes if not e.discarded]
        dropped = [e for e in self._episodes if e.discarded]
        detail = f"{len(kept)} episode(s) kept"
        if dropped:
            # LeRobot v3 has no cheap in-place episode deletion; discarded
            # episodes are excluded from training via the metadata note the
            # controller writes. Saying so beats silently pretending.
            detail += f", {len(dropped)} discarded (excluded via metadata)"
        try:
            self._recorder.finalize()
        except Exception as exc:  # noqa: BLE001
            self._backend.close()
            return {"ok": False, "detail": f"finalize failed: {exc}"}
        if upload:
            wanted = (repo_id or "").strip() or None
            verdict = upload_verdict(
                asked_repo_id=wanted,
                dataset=self.dataset,
                push=lambda: self._recorder.push_to_hub(),
            )
            detail += f", {verdict['detail']}"
            if not verdict["ok"]:
                self._backend.close()
                return {"ok": False, "detail": detail}
        # The finished dataset's own completion sentence must carry the defect it was born with.
        notice = getattr(self._backend, "camera_notice", None)
        self._backend.close()
        result = {"ok": True, "detail": detail, "discarded_indices": [e.index for e in dropped]}
        # Where the dataset now LIVES - the receipt the training screen can be handed.
        # Without this, the operator must re-find their minutes-old dataset by memory.
        result["dataset"] = self.dataset
        result["episodes_kept"] = len(kept)
        try:
            from strands_robots.dataset_recorder import resolve_dataset_dir

            result["root"] = str(resolve_dataset_dir(self.dataset))
        except Exception:  # noqa: BLE001 - the receipt is still useful without the path
            pass
        if notice:
            missing = ", ".join(notice.get("missing") or ())
            result["camera_notice"] = notice
            result["detail"] = f"{detail} - WITHOUT camera(s) {missing}: " + (
                "no image channel at all, so this dataset cannot train a visual policy"
                if not notice.get("present")
                else "those image channels are missing from every episode"
            )
        return result

    # ------------------------------------------------------------- helpers

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def _next_index(self) -> int:
        return (self._episodes[-1].index + 1) if self._episodes else 0

    # ---------------------------------------------------------------- loop

    def _loop(self) -> None:
        from strands_robots.mesh.pacing import Ticker

        with Ticker(1.0 / self.fps, self._stop_evt) as ticker:
            while not ticker.wait():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - loop survives a bad read
                    with self._lock:
                        self._last_error = f"control step failed: {exc}"
                    logger.warning("record tick failed: %r", exc)

    def _motion_verdict_locked(self) -> dict[str, Any] | None:
        """Whether the follower is holding one pose. Caller holds ``self._lock``."""
        if self._phase == "recording" and self._current is not None:
            verdict = record_motion.motion_verdict(self._motion, now=self._clock(), frames=self._current.frames)
            if verdict is not None:
                self._motion_notice = verdict
            return verdict or self._motion_notice
        return self._motion_notice

    def _measured_fps(self) -> float | None:
        """Capture rate of the episode in flight, else the last real one. Caller holds ``self._lock``."""
        if self._current is not None:
            live = achieved_fps(self._current.frames, self._current.duration_s)
            if live is not None:
                return live
        for ep in reversed(self._episodes):
            if not ep.discarded:
                measured = achieved_fps(ep.frames, ep.duration_s)
                if measured is not None:
                    return measured
        return None

    def tick(self) -> bool:
        """One control step. Teleop runs in EVERY phase (the operator lines the arms up between episodes);
        frames are recorded only while ``recording``.
        """
        action = self._backend.leader_action()
        sent = self._backend.follower_apply(action)
        with self._lock:
            if self._phase != "recording" or self._current is None:
                return False
            obs = self._backend.follower_observation()
            self._recorder.add_frame(obs, sent, task=self.task)
            ep = self._current
            ep.frames += 1
            now = self._clock()
            ep.duration_s = now - ep.started_at
            self._motion.append((now, record_motion.joint_positions(obs)))
            # Kept WIDER than record_motion's window on purpose: the verdict needs a sample older than the
            # window to know it has a full window of history at all (see record_motion.motion_verdict).
            self._motion = record_motion.prune(  # type: ignore[assignment]
                self._motion, now, record_motion.WINDOW_S * 2
            )
            if ep.frames == 1:
                for cam in self._backend.camera_keys:
                    if cam in obs:
                        # The camera name is client-chosen (spawn config). validate_cameras already
                        # closes its charset; this is the same reduction the read route applies, so
                        # the file written here is the file /thumb/{episode}/{camera} looks for and
                        # no name can be a path.
                        safe = thumb_name(cam)
                        if not safe:
                            continue
                        p = self._thumb_dir / f"{ep.index}_{safe}.jpg"
                        if _save_thumbnail(obs[cam], p):
                            ep.thumbnails[cam] = f"/api/record/thumb/{ep.index}/{cam}"
            return True


def thumb_name(camera: str) -> str:
    """The bare file-name form of a camera name: alphanumerics, ``-`` and ``_`` only.

    Shared contract with ``record_api``'s ``/thumb/{episode}/{camera}`` route, which reduces
    the URL segment the same way, so the writer and the reader agree on the file.
    """
    return "".join(c for c in str(camera) if c.isalnum() or c in "-_")


def camera_verdict(requested: Any, present: Any) -> dict[str, Any] | None:
    """What the operator must be told about cameras BEFORE they collect."""
    req = sorted(str(c) for c in (requested or ()))
    got = sorted(str(c) for c in (present or ()))
    missing = [c for c in req if c not in got]
    if not missing:
        return None
    consequence = (
        "the dataset will have NO image channel and cannot train a visual policy"
        if not got
        else "the dataset will be missing those image channels"
    )
    return {
        "requested": req,
        "present": got,
        "missing": missing,
        "message": (
            f"{len(missing)} of {len(req)} requested cameras did not open "
            f"({', '.join(missing)}) - {consequence}. The follower's own log says "
            "why it dropped them; on macOS a server started by a background "
            "daemon can never be granted camera access."
        ),
    }


def hardware_backend(
    *,
    follower_name: str,
    follower_port: str,
    leader_type: str,
    leader_port: str,
    cameras: dict[str, Any] | None = None,
) -> TeleopBackend:
    """The real thing: strands_robots follower + lerobot leader.

    Imported lazily so the state machine stays testable on machines without
    lerobot or serial hardware.
    """
    from strands_robots.robot import Robot
    from strands_robots.teleoperator import Teleoperator

    class _Hardware:
        def __init__(self) -> None:
            self._robot = Robot(
                follower_name,
                mode="real",
                port=follower_port,
                cameras=cameras or {},
            )
            # HardwareRobot connects lazily on the first task; recording reads observations directly, so
            # connect eagerly the same way the fleet spawner does (calibrate=False - the arm was
            # calibrated when its peer was set up).
            inner = getattr(self._robot, "robot", None)
            if inner is not None and not getattr(inner, "is_connected", False):
                inner.connect(False)
            self._leader = Teleoperator(leader_type, port=leader_port)
            if hasattr(self._leader, "connect"):
                try:
                    self._leader.connect(calibrate=False)
                except TypeError:  # keyboards/gamepads take no calibrate arg
                    self._leader.connect()
                if not getattr(self._leader, "is_calibrated", True):
                    fpath = getattr(self._leader, "calibration_fpath", "its calibration file")
                    self._leader.disconnect()
                    raise ValueError(
                        f"leader arm at {leader_port} has no usable calibration "
                        f"({fpath} is missing or does not match the motors). "
                        "Calibrate it once from a terminal (lerobot-calibrate "
                        f"--teleop.type={leader_type} --teleop.port={leader_port}) "
                        "or copy an existing calibration json to that path - "
                        "a headless server cannot run the interactive wizard."
                    )
            obs = cast(Any, self._robot).get_observation()
            self.camera_keys = sorted(k for k, v in obs.items() if getattr(v, "ndim", 0) == 3)
            self._camera_dims = {k: (obs[k].shape[0], obs[k].shape[1]) for k in self.camera_keys}
            # Requested vs actually-present, judged at OPEN time: the caller's
            # ``cameras`` dict is the intent, the first observation is reality.
            self.camera_notice = camera_verdict(cameras or {}, self.camera_keys)
            self._robot_type = follower_name

        def recorder_kwargs(self) -> dict[str, Any]:
            """What DatasetRecorder.create needs beyond repo_id/fps/task -
            derived from the live robot so the schema cannot drift from
            what the observations actually contain."""
            return {
                "robot_type": self._robot_type,
                "robot_features": getattr(self._robot, "observation_features", None),
                "action_features": getattr(self._robot, "action_features", None),
                "camera_keys": list(self.camera_keys),
                "camera_dims": dict(self._camera_dims),
            }

        def leader_action(self) -> dict[str, float]:
            return dict(self._leader.get_action())

        def follower_apply(self, action: dict[str, float]) -> dict[str, float]:
            sent = self._robot.send_action(action)
            return dict(sent) if isinstance(sent, dict) else dict(action)

        def follower_observation(self) -> dict[str, Any]:
            return dict(cast(Any, self._robot).get_observation())

        def close(self) -> None:
            for dev in (self._leader, self._robot):
                try:
                    if hasattr(dev, "disconnect"):
                        dev.disconnect()
                    elif hasattr(dev, "close"):
                        dev.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("device close failed: %r", exc)

    return _Hardware()
