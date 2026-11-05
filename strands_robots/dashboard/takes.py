"""The parts of a page recording: one robot's track, the recording's state, where each dataset goes, its units."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..recording import Episode, is_repo_id

if TYPE_CHECKING:
    from ..robot import Robot
    from ..units import Units
    from .dashboard import Dashboard

MAX_FPS = 60


def recordings_dir() -> Path:
    """Where the page may write datasets: ``$STRANDS_ROBOTS_RECORDINGS``, default ``~/strands-robots/recordings``."""
    return Path(
        os.environ.get("STRANDS_ROBOTS_RECORDINGS") or Path.home() / "strands-robots" / "recordings"
    ).expanduser()


def inside(target: str) -> str:
    """A page target as the recorder takes it: ``hf://owner/name`` as it is, a relative name under
    :func:`recordings_dir`, an absolute path only inside it. A dataset is created or resumed (and an aborted take
    deleted) at that path, so the page never names one elsewhere (``..`` and symlinks resolved first)."""
    if is_repo_id(target):
        return target
    base = recordings_dir().resolve()
    path = (base / Path(target).expanduser()).resolve()  # an absolute target replaces base in the join
    if path == base or not path.is_relative_to(base):
        raise PermissionError(f"the page records only inside {base} (STRANDS_ROBOTS_RECORDINGS), not {target!r}")
    return str(path)


@dataclass
class Track:
    """One robot's share of the recording: its dataset target, cameras, units and the episode being written."""

    robot: Robot
    target: str
    cameras: tuple[str, ...]
    units: Units | None
    episode: Episode | None = None
    path: str | None = None
    saved: int = 0


@dataclass
class State:
    state: str = "idle"  # idle | recording | saving | error
    target: str | None = None
    task: str | None = None
    fps: int = 30
    started: float = 0.0
    episode_started: float = 0.0
    episode: int = 0
    frames: int = 0
    ticks: list[float] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def targets_for(target: str, ids: list[str]) -> dict[str, str]:
    """Where each robot's dataset goes: the target itself for one robot, one per robot id for several."""
    if len(ids) == 1:
        return {ids[0]: target}
    if is_repo_id(target):
        return {rid: f"{target}-{rid.replace('/', '-')}" for rid in ids}
    return {rid: str(Path(target).expanduser() / rid.replace("/", "-")) for rid in ids}


def units_of(robot: Robot, name: Any) -> Units | None:
    """``"auto"``: LeRobot's units for a robot LeRobot knows (``hardware.lerobot_type``), else radians."""
    from ..units import Units

    if name in (None, "rad"):
        return None
    if name == "auto":
        if not robot.description.hardware.lerobot_type:
            return None
        try:
            return Units.for_robot(robot.description, "lerobot")
        except (KeyError, ValueError):
            return None
    if not isinstance(name, str):
        raise ValueError(f"units must be 'auto', 'rad' or a scheme name, got {name!r}")
    return Units.for_robot(robot.description, name)


def options(dash: Dashboard) -> dict[str, Any]:
    """What the page can record: every robot with the cameras it carries, every world camera, a default target."""
    sim = dash.sim
    with sim.lock:
        cams = [
            {"name": c.name, "size": list(c.size), "url": f"/api/cameras/{c.name}.png", "robot": c.robot}
            for c in sim.scene.cameras
        ]
        ids = list(sim.robots)
    robots, seen = [], set()
    for rid in ids + [r for r in dash.real if r not in ids]:
        robot = dash._robot(rid)  # a hardware robot's twin card resolves to the hardware robot: listed once
        if id(robot) in seen:
            continue
        seen.add(id(robot))
        robots.append(
            {
                "id": rid,
                "name": robot.name,
                "is_real": bool(robot.is_real),
                "cameras": [c["name"] for c in cams if c["robot"] == rid],
                "units": "rad" if units_of(robot, "auto") is None else "lerobot",
            }
        )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = str(recordings_dir() / f"recording-{stamp}")
    return {"robots": robots, "cameras": cams, "target": target, "max_fps": MAX_FPS}


def observe(dash: Dashboard, t: Track) -> Any:
    robot = t.robot
    obs = robot.observe(images=robot.sim is None and bool(t.cameras))
    images: dict[str, np.ndarray] = {}
    for cam in t.cameras:
        img = obs.images.get(cam) if robot.sim is None else None
        images[cam] = img if img is not None else dash.sim.render(cam)
    return replace(obs, images=images)


def action(t: Track, obs: Any) -> np.ndarray:
    held = t.robot._held
    return np.asarray(obs.joints.positions if held is None else held, dtype=float)


def plan(dash: Dashboard, robots: Any, cameras: Any, target: Any, fps: Any, task: Any, units: Any) -> list[Track]:
    """A start body checked (robots, cameras, target, fps, task) into one Track per robot, or the reason it is not."""
    if not isinstance(robots, list) or not robots or not all(isinstance(r, str) for r in robots):
        raise ValueError('body must have "robots": ["<robot id>", ...] (at least one)')
    if len(set(robots)) != len(robots):
        raise ValueError(f"robots {robots} name a robot twice")
    if not isinstance(target, str) or not target.strip():
        raise ValueError('body must have "target": "<directory>" or "hf://owner/name"')
    if cameras is None:
        cameras = {}
    if not isinstance(cameras, dict):
        raise ValueError('"cameras" is {"<robot id>": ["<camera name>", ...]}')
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not 1 <= fps <= MAX_FPS or fps != int(fps):
        raise ValueError(f"fps must be a whole number from 1 to {MAX_FPS}, got {fps!r}")
    if task is not None and not isinstance(task, str):
        raise ValueError('"task" is the episode instruction, a string')
    extra = [k for k in cameras if k not in robots]
    if extra:
        raise ValueError(f"cameras name robots {extra} that are not recorded ({robots})")
    known = {c.name for c in dash.sim.scene.cameras}
    where = {rid: inside(t) for rid, t in targets_for(target.strip(), robots).items()}
    tracks = []
    for rid in robots:
        robot = dash._robot(rid)
        cams = cameras.get(rid) or []
        if not isinstance(cams, list) or not all(isinstance(c, str) for c in cams):
            raise ValueError(f"cameras for {rid} must be a list of camera names, got {cams!r}")
        unknown = [c for c in cams if c not in known and c not in _hw_cameras(robot)]
        if unknown:
            raise KeyError(f"no camera {unknown} for {rid}; cameras are {sorted(known)}")
        tracks.append(Track(robot, where[rid], tuple(cams), units_of(robot, units)))
    return tracks


def _hw_cameras(robot: Robot) -> set[str]:
    if robot.sim is not None:
        return set()
    try:
        return set(robot.observe(images=True).images)
    except Exception:  # noqa: BLE001 — a body that cannot be read has no cameras to offer
        return set()


def status_json(s: State, tracks: list[Track]) -> dict[str, Any]:
    """What ``GET /api/record/status`` answers."""
    now = time.monotonic()
    recent = [t for t in s.ticks if now - t <= 2.0]
    achieved = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0.0
    live = s.state in ("recording", "saving")
    # a slow tick (camera renders, a save encoding video) is a missed frame: the dataset keeps a fixed fps, so say so
    due = int((now - s.episode_started) * s.fps) + 1 if s.state == "recording" else 0
    return {
        "state": s.state,
        "target": s.target,
        "task": s.task,
        "fps": s.fps,
        "fps_achieved": round(achieved, 1) if live else 0.0,
        "elapsed": round(now - s.started, 2) if live else 0.0,
        "episode": s.episode,
        "frames": s.frames,
        "dropped": max(0, due - s.frames),
        "episode_seconds": round(now - s.episode_started, 2) if live else 0.0,
        "robots": {
            t.robot.id: {
                "dataset": t.target,
                "path": t.path,
                "cameras": list(t.cameras),
                "units": "rad" if t.units is None else t.units.name,
                "saved": t.saved,
            }
            for t in tracks
        },
        "episodes": list(s.episodes),
        "error": s.error,
    }
