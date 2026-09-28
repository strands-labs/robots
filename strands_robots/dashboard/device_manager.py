"""USB device + camera discovery and robot process lifecycle. Detects: * Serial servo buses
(Feetech/Dynamixel USB adapters).
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from strands_robots.dashboard import bus_claim
from strands_robots.simulation.models import registered, registry_entry

from ..utils import non_negative_whole_number_error, refusal_repr, refusal_str
from . import arm_roles, camera_liveness
from . import cameras as camera_facts

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 200  # ring buffer per managed robot


def _settle_seconds(raw: str | None, default: float = 5.0) -> float:
    """``STRANDS_DASHBOARD_SPAWN_SETTLE_S`` as a finite non-negative span, else *default*.

    ``float`` accepts ``"nan"``, ``"inf"`` and ``"1e999"``; a spawn that waits
    ``nan`` seconds answers before the child has said anything, and one that
    waits ``inf`` never answers. Tested through the shared
    :func:`strands_robots.utils.finite_number_error` domain.
    """
    from strands_robots.utils import finite_number_error

    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    if finite_number_error(value, "STRANDS_DASHBOARD_SPAWN_SETTLE_S", "dashboard") is not None or value < 0:
        return default
    return value


# : How long /api/devices/spawn watches a new child before answering.
SPAWN_SETTLE_S = _settle_seconds(os.environ.get("STRANDS_DASHBOARD_SPAWN_SETTLE_S"))

# Where USB device profiles live.
DEFAULT_PROFILES_PATH = os.path.join(Path.home(), ".strands_dashboard", "profiles.json")

# Auto-spawn watcher cadence and unplug debounce.
AUTOSPAWN_POLL_S = 2.0
AUTOSPAWN_MISSING_POLLS = 2

# Which serial ports are a servo bus is decided in one place,
# ``strands_robots._serial_discovery.matches_servo_bus`` (vendor ids, name
# keywords, and the disqualifiers), so the dashboard and the CLI cannot drift.


@dataclass
class ManagedRobot:
    """One robot process this dashboard spawned, with its config and log ring buffer."""

    peer_id: str
    robot_name: str
    mode: str  # "real" | "sim"
    port: str | None = None
    # The lerobot calibration id this child was started with.
    robot_id: str | None = None
    cameras: dict[str, Any] = field(default_factory=dict)
    process: subprocess.Popen | None = None
    started_at: float = 0.0
    # Last N lines of child output. The pipe MUST be drained:
    # with stdout=PIPE and no reader, the child blocks in print() once the
    # OS pipe buffer (~64KB) fills and the peer silently freezes.
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    joint_problem_seen: dict[str, Any] | None = None
    # What this process was started to DO ({"repo_id","episode"} for a replay, {"dataset_root"}
    # for a collect).
    job: dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        """True while the child process is running."""
        return self.process is not None and self.process.poll() is None


# : lerobot's calibration store.
_CALIBRATION_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
_CALIB_CACHE: dict[str, Any] = {"at": 0.0, "value": {}}


def _calibrations_on_disk(ttl: float = 30.0) -> dict[str, list[str]]:
    """``{"robots/so101_follower": ["follower", ...]}`` - the ids that HAVE a calibration file."""
    now = time.time()
    if now - float(_CALIB_CACHE["at"]) < ttl:
        return dict(_CALIB_CACHE["value"])
    found: dict[str, list[str]] = {}
    try:
        for group in ("robots", "teleoperators"):
            base = _CALIBRATION_ROOT / group
            if not base.is_dir():
                continue
            for kind_dir in sorted(base.iterdir()):
                if not kind_dir.is_dir():
                    continue
                ids = sorted(f.stem for f in kind_dir.glob("*.json"))
                if ids:
                    found[f"{group}/{kind_dir.name}"] = ids
    except OSError:
        found = {}
    _CALIB_CACHE.update(at=now, value=found)
    return dict(found)


def crash_reason(lines: Iterable[str]) -> str | None:
    """The line that explains why a child died, or None if nothing does."""
    cleaned: list[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line:
            continue
        body = _strip_log_prefixes(line)
        low = body.lower()
        if low.startswith("during handling of the above") or low.startswith("the above exception was the direct cause"):
            # A chained block starts here: everything after it is fallout.
            break
        if "cleanup error" in low:
            continue
        cleaned.append(body)

    in_traceback = False
    for body in cleaned:
        if body.lower().startswith("traceback (most recent"):
            in_traceback = True
            continue
        if in_traceback:
            # Frames are indented; the exception line is not.
            if body != body.lstrip():
                continue
            if _looks_like_a_fault(body):
                return body[:400]
            continue
        if _looks_like_a_fault(body):
            return body[:400]

    if in_traceback:
        # A traceback whose tail never arrived (ring buffer cut it, child was
        # killed mid-print). Say that rather than quoting a frame.
        return "the process died with a traceback (see the log)"
    return None


def _strip_log_prefixes(line: str) -> str:
    """Drop the drain thread's ``HH:MM:SS`` stamp and any ``LEVEL:logger:``."""
    body = line
    if len(body) > 9 and body[2] == ":" and body[5] == ":" and body[8] == " ":
        body = body[9:]
    for level in ("ERROR:", "CRITICAL:", "WARNING:"):
        if body.lstrip().startswith(level):
            rest = body.lstrip()[len(level) :]
            body = rest.split(":", 1)[1].strip() if ":" in rest else rest.strip()
            break
    return body


def _looks_like_a_fault(text: str) -> bool:
    """True when a line reads like the reason a process stopped."""
    if not text:
        return False
    head = text.split(":", 1)[0].strip()
    if head.endswith(("Error", "Exception", "Interrupt")) and " " not in head:
        return True  # "ValueError: ..." / "ConnectionError: ..."
    low = text.lower()
    return low.startswith(("traceback (most recent", "fatal", "error:", "usage:"))


def _drain(proc: subprocess.Popen, logs: deque[str], peer_id: str) -> None:
    """Continuously read child stdout so the pipe never fills."""
    try:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if line:
                logs.append(f"{time.strftime('%H:%M:%S')} {line}")
    except Exception as e:  # reader must never take the server down
        logs.append(f"[drain error: {e}]")
    finally:
        try:
            code: int | None = proc.wait(timeout=5)  # EOF usually precedes reaping by a tick
        except Exception:
            code = proc.poll()
        logs.append(f"[process exited code={code}]")
        logger.info("robot %s child exited (code=%s)", peer_id, code)


def scan_serial_ports() -> list[dict[str, Any]]:
    """Enumerate candidate servo-bus serial ports."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    from strands_robots._serial_discovery import WCH_CH34X_VID, matches_servo_bus

    out: list[dict[str, Any]] = []
    for p in serial.tools.list_ports.comports():
        if not matches_servo_bus(p):
            continue
        # prefer /dev/cu.* on macOS (call-up device, non-blocking open)
        device = p.device
        if sys.platform == "darwin" and device.startswith("/dev/tty."):
            cu = device.replace("/dev/tty.", "/dev/cu.")
            if os.path.exists(cu):
                device = cu
        out.append(
            {
                "device": device,
                "description": p.description,
                "vid": f"{p.vid:04x}" if p.vid else None,
                "pid": f"{p.pid:04x}" if p.pid else None,
                "serial_number": getattr(p, "serial_number", None),
                # The WCH CH34x adapter (the first id the owner lists) is what
                # ships on the SO-10x arms; FTDI boards could be any bus.
                "likely_robot": "so101" if (p.vid == WCH_CH34X_VID) else None,
            }
        )
    return out


def scan_cameras(max_index: int = 4, skip: set[int] | None = None) -> list[dict[str, Any]]:
    """Probe OpenCV camera indices. Cheap open/read/release per index."""
    cams, _ = scan_cameras_with_failures(max_index=max_index, skip=skip)
    return cams


def scan_cameras_with_failures(
    max_index: int = 4,
    skip: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Like :func:`scan_cameras`, but also says WHY each index failed."""
    try:
        import cv2
    except ImportError:
        return [], {}
    cams: list[dict[str, Any]] = []
    failed: list[int] = []
    for i in range(max_index):
        if skip and i in skip:
            continue
        cap = cv2.VideoCapture(i)
        try:
            got = False
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    cams.append(
                        {
                            "index": i,
                            "width": w,
                            "height": h,
                            "fps": round(fps, 1) if fps and fps > 0 else None,
                        }
                    )
                    got = True
        finally:
            cap.release()
        if not got:
            failed.append(i)
    return cams, diagnose_camera_indices(failed)


#: Re-probe one index and let OpenCV print its own complaint to stderr.
_DIAGNOSE_SRC = (
    "import sys;import cv2;"
    "cap=cv2.VideoCapture(int(sys.argv[1]));"
    "ok=cap.isOpened();"
    "r=cap.read()[0] if ok else False;"
    "cap.release();"
    "print('opened' if r else 'failed')"
)


def diagnose_camera_indices(indices: Sequence[int], timeout: float = 12.0) -> dict[int, str]:
    """index -> the stderr OpenCV produced while failing to open it."""
    out: dict[int, str] = {}
    for index in indices:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DIAGNOSE_SRC, str(index)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            out[index] = proc.stderr or ""
        except subprocess.TimeoutExpired:
            # A probe that hangs is a real symptom of a camera in a bad state -
            # report it as such instead of dropping the index.
            out[index] = "device or resource busy (the probe never returned)"
        except Exception as e:  # noqa: BLE001 - diagnosis is decoration, never fatal
            logger.debug("camera diagnosis for index %s failed: %r", index, e)
            out[index] = ""
    return out


#: Read Present_Voltage from each servo id on a Feetech bus. Register 62, one
#: byte, read-only - this script physically cannot move an arm.
_BUS_VOLTAGE_SRC = """
import json, sys
port, model, ids = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3].split(',') if i]
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
motors = {f'm{i}': Motor(i, model, MotorNormMode.RANGE_M100_100) for i in ids}
bus = FeetechMotorsBus(port=port, motors=motors)
out = {}
try:
    bus.connect(handshake=False)
    for name in motors:
        try:
            out[name] = float(bus.read('Present_Voltage', name, normalize=False)) / 10.0
        except Exception as e:
            print(f'{name}: {e}', file=sys.stderr)
finally:
    try:
        bus.disconnect()
    except Exception:
        pass
print(json.dumps(out))
"""

#: Resolutions worth asking a USB camera about - the ones lerobot configs
#: actually use. The camera's answer, not this list, is what gets offered.
CAMERA_MODE_CANDIDATES: tuple[tuple[int, int], ...] = (
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
)
CAMERA_FPS_CANDIDATES: tuple[int, ...] = (15, 30, 60)


def modes_from_readbacks(
    native: Mapping[str, Any],
    readbacks: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Distill set/read-back probes into the modes a camera really has."""
    keep: dict[tuple[int, int, int], dict[str, Any]] = {}

    def _add(w: Any, h: Any, fps: Any) -> None:
        try:
            w, h, fps = int(w), int(h), int(round(float(fps)))
        except (TypeError, ValueError):
            return
        if w <= 0 or h <= 0 or fps <= 0:
            return
        keep.setdefault((w, h, fps), {"width": w, "height": h, "fps": fps})

    _add(native.get("width"), native.get("height"), native.get("fps"))
    for rb in readbacks:
        req, got = rb.get("requested") or {}, rb.get("got") or {}
        try:
            if (
                int(got.get("width", -1)) == int(req.get("width", -2))
                and int(got.get("height", -1)) == int(req.get("height", -2))
                and abs(float(got.get("fps", -99)) - float(req.get("fps", -1))) <= 1.0
            ):
                _add(req.get("width"), req.get("height"), req.get("fps"))
        except (TypeError, ValueError):
            continue
    return sorted(keep.values(), key=lambda m: (m["width"] * m["height"], m["fps"]))


def scan_camera_names() -> list[dict[str, Any]]:
    """Human names of the attached cameras, best effort per platform. macOS: parsed from ffmpeg's
    AVFoundation device listing (if ffmpeg is installed).
    """
    names: list[dict[str, Any]] = []
    if sys.platform == "darwin":
        import shutil

        ffmpeg = shutil.which("ffmpeg") or next(
            (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)),
            None,
        )
        if not ffmpeg:
            return names
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            ).stderr
        except Exception as e:  # noqa: BLE001 - enumeration is decoration, never fatal
            logger.debug("camera name scan failed: %r", e)
            return names
        in_video = False
        for line in out.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if in_video:
                m = re.search(r"\[(\d+)\]\s+(.+)\Z", line)
                if m:
                    names.append({"listing_index": int(m.group(1)), "name": m.group(2).strip()})
    elif sys.platform.startswith("linux"):
        import glob

        for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            try:
                idx = int(path.split("video")[-1].split("/")[0])
                with open(path, encoding="utf-8") as f:
                    names.append({"listing_index": idx, "name": f.read().strip()})
            except (OSError, ValueError):
                continue
    return names


_SPAWNER = r"""
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot

if cfg["mode"] == "real":
    kwargs = {}
    if cfg.get("robot_id"):
        kwargs["id"] = cfg["robot_id"]  # lerobot calibration identity
    robot = Robot(
        cfg["robot_name"], mode="real", port=cfg["port"],
        cameras=cfg.get("cameras") or None,
        mesh=True, peer_id=cfg["peer_id"], **kwargs,
    )
    # Connect eagerly so the mesh publishes joints + camera frames right away
    # (HardwareRobot otherwise connects lazily on the first task). Goes through
    # connect_eagerly so a camera this machine will not open costs the camera
    # and not the whole arm -- lerobot's own connect() gates is_connected on
    # every camera, so one refusal reported a healthy arm as dead.
    try:
        ok, degraded, err = robot.connect_eagerly()
        if ok and degraded:
            for cam_name, reason in degraded.items():
                print(f"camera {cam_name!r} unavailable, dropped: {reason}", flush=True)
            print(f"hardware connected WITHOUT camera(s): {', '.join(degraded)}", flush=True)
        elif ok:
            print("hardware connected", flush=True)
        else:
            print(f"eager connect failed (will retry on first task): {err}", flush=True)
    except Exception as e:
        print(f"eager connect failed (will retry on first task): {e}", flush=True)
    print(f"{cfg['peer_id']} (real @ {cfg['port']}) online", flush=True)
    while True:
        time.sleep(1)
else:
    sim = Robot(cfg["robot_name"], mesh=True, peer_id=cfg["peer_id"])
    n = cfg["robot_name"]
    sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
    try:
        sim.add_camera(name=f"{n}/wrist", position=[0.02, 0.0, 0.05], target=[0.2, 0.0, 0.0], parent_body=f"{n}/gripper")
    except Exception:
        pass
    print(f"{cfg['peer_id']} (sim) online", flush=True)
    while True:
        sim.step(5)
        time.sleep(0.2)
"""

_COLLECT_SPAWNER = r"""
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot
from strands_robots.tools.run_policy import run_policy

robot_name = cfg.get("robot_name") or "so101"
sim = Robot(robot_name, mesh=True, peer_id=cfg["peer_id"])
n = robot_name
sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
try:
    sim.add_camera(name=f"{n}/wrist", position=[0.02, 0.0, 0.05], target=[0.2, 0.0, 0.0], parent_body=f"{n}/gripper")
except Exception:
    pass
print(f"{cfg['peer_id']} (collect sim) online - {cfg['n_episodes']} episodes", flush=True)

# run_policy drives the N-episode loop deterministically and reports
# parquet-truth (meta/info.json total_episodes) - no self-reporting.
# run_policy takes n_steps (not duration): steps = seconds * control Hz.
control_hz = 30.0
n_steps = max(1, int(float(cfg.get("duration", 10.0)) * control_hz))
res = run_policy(
    simulation=sim,
    robot_name=robot_name,
    policy_provider=cfg.get("policy_provider", "mock"),
    policy_config=cfg.get("policy_config") or None,
    instruction=cfg.get("instruction", ""),
    n_episodes=int(cfg.get("n_episodes", 5)),
    n_steps=n_steps,
    control_frequency=control_hz,
    dataset_root=cfg["dataset_root"],
    dataset_repo_id=cfg.get("dataset_repo_id", "local/collected"),
    dataset_task=cfg.get("instruction", ""),
    dataset_fps=int(cfg.get("fps", 30)),
)
status = res.get("status")
for block in res.get("content", []):
    if isinstance(block, dict) and block.get("json"):
        print(f"collect: {status}: {json.dumps(block['json'])[:400]}", flush=True)
        break
    if isinstance(block, dict) and block.get("text"):
        print(f"collect: {status}: {block['text'][:300]}", flush=True)
time.sleep(3)
os._exit(0)
"""

_REPLAY_SPAWNER = r"""
import os, sys, time, json
cfg = json.loads(sys.argv[1])
os.environ.setdefault("STRANDS_ROBOTS_NO_DYLD_SHIM", "1")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", os.environ.get("STRANDS_MESH_LOCAL_DEV", "1"))
os.environ.setdefault("STRANDS_MESH_MULTICAST", "true")
os.environ.setdefault("STRANDS_MESH", "true")
os.environ.setdefault("STRANDS_MESH_CAMERA_HZ", os.environ.get("STRANDS_MESH_CAMERA_HZ", "5"))

from strands_robots import Robot

robot_name = cfg.get("robot_name") or "so101"
sim = Robot(robot_name, mesh=True, peer_id=cfg["peer_id"])
n = robot_name
sim.add_camera(name=f"{n}/front", position=[0.6, 0.4, 0.5], target=[0.0, 0.0, 0.15])
print(f"{cfg['peer_id']} (replay sim) online", flush=True)

res = sim.replay_episode(
    cfg["repo_id"],
    episode=int(cfg.get("episode", 0)),
    root=cfg.get("root"),
    speed=float(cfg.get("speed", 1.0)),
)
status = res.get("status")
for block in res.get("content", []):
    if isinstance(block, dict) and block.get("text"):
        print(f"replay: {status}: {block['text'][:300]}", flush=True)
        break
# linger briefly so the last camera frames reach subscribers, then exit.
# os._exit: the mesh session runs non-daemon threads that would otherwise
# keep this one-shot process alive forever after the script body returns.
time.sleep(3)
os._exit(0)
"""


def profile_key(port: dict[str, Any]) -> str:
    """Stable identity for a detected serial board. The USB serial number survives replug and port
    renumbering, so it is the key.
    """
    serial = port.get("serial_number")
    if serial:
        return str(serial)
    return str(port.get("device") or "")


def remembered_spawn(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """The spawn payload a saved USB profile describes, or {} when there is none."""
    if not profile:
        return {}
    peer_id = str(profile.get("peer_id") or "").strip()
    if not peer_id:
        return {}
    cams = profile.get("cameras")
    names: list[str] = []
    if isinstance(cams, Mapping):
        names = [str(k) for k in cams]
    out: dict[str, Any] = {
        "peer_id": peer_id,
        "robot_name": profile.get("robot_name") or None,
        "mode": profile.get("mode") or None,
        "cameras": names,
        "saved_at": profile.get("saved_at"),
    }
    # The lerobot calibration id travels with the arm, so a wrong one moves a real arm with another
    # arm's zero points. It is shown when remembered - never defaulted, never guessed from the name.
    if profile.get("robot_id"):
        out["robot_id"] = str(profile["robot_id"])
    return out


def remembered_camera_health(
    cameras: Mapping[str, Any] | None,
    camera_rows: Sequence[Mapping[str, Any]],
    peer_id: str = "",
) -> dict[str, Any]:
    """Are the cameras this board remembers usable RIGHT NOW?"""
    if not isinstance(cameras, Mapping) or not cameras:
        return {}
    by_index = {int(r["index"]): r for r in camera_rows if isinstance(r.get("index"), int)}
    entries: list[dict[str, Any]] = []
    for name, cfg in cameras.items():
        entry: dict[str, Any] = {"name": str(name)}
        target = cfg.get("index_or_path") if isinstance(cfg, Mapping) else cfg
        if isinstance(target, bool) or not isinstance(target, int):
            # A device PATH (or junk). We cannot judge a path from an index scan, and pretending to
            # would be the confident-but-untested claim this dashboard keeps deleting.
            entry.update(state="unchecked", reason="configured by path, not by index - not checked here")
            entries.append(entry)
            continue
        entry["index"] = target
        row = by_index.get(target)
        if row is None:
            entry.update(state="absent", reason=f"no camera at index {target} in the latest scan")
            entries.append(entry)
            continue
        state = str(row.get("state") or "unknown")
        owner = str(row.get("claimed_by") or "")
        if state in {"in_use", "assigned"} and owner and owner == peer_id:
            state, reason = "ready", f"index {target} was this peer's own camera"
        else:
            reason = str(row.get("reason") or "")
        entry.update(state=state, reason=reason)
        if row.get("remedy"):
            entry["remedy"] = str(row["remedy"])
        entries.append(entry)

    trouble = [e for e in entries if e["state"] not in {"ready", "unchecked"}]
    if not trouble:
        return {}
    names = ", ".join(f"{e['name']} (index {e.get('index', '?')})" for e in trouble)
    return {
        "cameras": entries,
        "ok": False,
        # The consequence, not just the state: an arm that cannot open a camera still starts, still
        # streams joints and still records - into a dataset with no pictures in it.
        "text": (
            f"the saved config names {names}, which {'is' if len(trouble) == 1 else 'are'} not "
            f"available right now: {trouble[0]['reason']}"
            + (f" (+{len(trouble) - 1} more)" if len(trouble) > 1 else "")
            + ". Spawning anyway works - the arm drops the camera it cannot open and comes up "
            "streaming joints only, which looks healthy and records episodes with no pictures in them"
        ),
    }


def respawn_payload(profile: Mapping[str, Any] | None, port: str) -> dict[str, Any]:
    """A spawn payload for ``port`` rebuilt from its remembered profile, or an error body."""
    if not profile:
        return {
            "error": (
                "no saved profile for this board, so there is nothing to bring back - spawn it once "
                "with the form above and it will be remembered by its USB serial"
            )
        }
    peer_id = str(profile.get("peer_id") or profile.get("name") or "").strip()
    robot_name = str(profile.get("robot_name") or "").strip()
    if not peer_id or not robot_name:
        return {
            "error": (
                "the saved profile for this board is incomplete (no "
                + ("peer name" if not peer_id else "robot family")
                + "), so it cannot be spawned as it stands - use the form above"
            )
        }
    if (bad_port := validate_port(port)) is not None:
        return {"error": bad_port}
    if (bad_rid := validate_robot_id(profile.get("robot_id") or None)) is not None:
        return {"error": f"the saved profile for this board carries a bad calibration id: {bad_rid}"}
    payload: dict[str, Any] = {
        "robot_name": robot_name,
        "mode": str(profile.get("mode") or "real"),
        "peer_id": peer_id,
        "port": port,
        "cameras": profile.get("cameras") if isinstance(profile.get("cameras"), Mapping) else None,
        "robot_id": profile.get("robot_id") or None,
    }
    saved_port = str(profile.get("port") or "")
    if saved_port and saved_port != port:
        payload["port_moved"] = {"was": saved_port, "now": port}
    return payload


class ProfileStore:
    """USB device profiles on disk: serial number -> saved spawn payload."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("STRANDS_DASHBOARD_PROFILES") or DEFAULT_PROFILES_PATH
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except Exception as e:
            # A corrupt or unreadable store must be visible, not silently
            # treated as "no profiles remembered".
            logger.warning("device profiles at %s unreadable (%r); starting empty", self.path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("device profiles at %s are not a JSON object; starting empty", self.path)
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def all(self) -> dict[str, dict[str, Any]]:
        """Every remembered profile, keyed by USB serial number."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def get(self, key: str) -> dict[str, Any] | None:
        """One profile by key, or None when that board was never spawned."""
        with self._lock:
            entry = self._data.get(key)
            return dict(entry) if entry is not None else None

    def record_role(self, key: str, verdict: Mapping[str, Any]) -> dict[str, Any] | None:
        """Remember a measured leader/follower verdict for a board serial."""
        role = verdict.get("role")
        if role not in ("leader", "follower"):
            return None
        with self._lock:
            entry = dict(self._data.get(key) or {})
            entry.update(
                {
                    "serial_number": key,
                    "role": role,
                    "role_volts": verdict.get("volts"),
                    "role_source": "measured",
                    "role_measured_at": time.time(),
                }
            )
            entry.setdefault("name", key)
            self._data[key] = entry
            snapshot = {k: dict(v) for k, v in self._data.items()}
        self._persist(snapshot, key)
        return dict(entry)

    # : Facts about the BOARD that a spawn payload knows nothing about, so they : survive being
    # re-saved.
    MEASURED_FIELDS = ("role", "role_volts", "role_source", "role_measured_at")

    # : Things the OPERATOR chose, which a spawn payload may simply not mention.
    REMEMBERED_FIELDS = ("cameras",)

    def save(self, key: str, payload: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """Remember ``payload`` as the way to spawn the board at ``key``."""
        entry = dict(payload)
        with self._lock:
            previous = dict(self._data.get(key) or {})
        for fname in self.MEASURED_FIELDS:
            if fname not in entry and fname in previous:
                entry[fname] = previous[fname]
        for fname in self.REMEMBERED_FIELDS:
            # None counts as unstated here (unlike MEASURED_FIELDS): the spawn payload always
            # carries the key, so "absent" alone would never fire and the memory would still be lost.
            if entry.get(fname) is None and previous.get(fname) is not None:
                entry[fname] = previous[fname]
        entry["name"] = name or entry.get("name") or entry.get("peer_id") or key
        entry["serial_number"] = key
        entry["saved_at"] = time.time()
        with self._lock:
            self._data[key] = entry
            snapshot = {k: dict(v) for k, v in self._data.items()}
        self._persist(snapshot, key)
        return dict(entry)

    def _persist(self, snapshot: dict[str, dict[str, Any]], key: str) -> None:
        """Atomic write of the whole store (tmp + os.replace)."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            # Kept in memory either way, but the operator has to know the
            # profile will not survive a restart.
            logger.warning("could not persist device profile %s to %s: %r", key, self.path, e)


def autospawn_veto(env: Mapping[str, str]) -> str | None:
    """Why USB auto-spawn must NOT bring boards up in this process, or None when it may."""
    raw = str(env.get("STRANDS_DASHBOARD_AUTOSPAWN", "")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return "STRANDS_DASHBOARD_AUTOSPAWN is off"
    override = raw in ("1", "true", "yes", "on", "force")
    if override:
        return None
    test_marker = env.get("PYTEST_CURRENT_TEST") or env.get("PYTEST_VERSION")
    if test_marker:
        marker = str(test_marker).split("::")[0]
        return (
            f"this process is a pytest run ({marker}): a test must never take a real serial port. "
            "Set STRANDS_DASHBOARD_AUTOSPAWN=1 if you truly mean to spawn hardware from a test."
        )
    if str(env.get("STRANDS_MESH", "")).strip().lower() in ("0", "false", "no", "off"):
        return "STRANDS_MESH is off (the hard kill switch): a robot child must not be started"
    return None


class AutoSpawnWatcher:
    """Brings known USB boards up (and unplugged ones down) on its own."""

    def __init__(
        self,
        manager: DeviceManager,
        list_ports: Callable[[], list[dict[str, Any]]] | None = None,
        peer_ids: Callable[[], Iterable[str]] | None = None,
        missing_polls: int = AUTOSPAWN_MISSING_POLLS,
    ) -> None:
        self.manager = manager
        self.list_ports = list_ports or scan_serial_ports
        self.peer_ids = peer_ids or (lambda: ())
        self.missing_polls = max(1, int(missing_polls))
        # key -> peer_id we auto-spawned for it
        self.adopted: dict[str, str] = {}
        self._missing: dict[str, int] = {}
        self._stop = threading.Event()
        # While True the watcher observes but never spawns/despawns.
        self.suspended = False

    @staticmethod
    def enabled() -> bool:
        """False when STRANDS_DASHBOARD_AUTOSPAWN is set to a falsey value."""
        raw = os.environ.get("STRANDS_DASHBOARD_AUTOSPAWN", "1").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def _claimed(self, port: dict[str, Any], profile: dict[str, Any]) -> str | None:
        """Reason this board must not be spawned, or None when it is free."""
        device = port.get("device")
        for m in self.manager.robots.values():
            if not m.alive():
                continue
            if device and m.port == device:
                return f"port {device} already claimed by managed robot {m.peer_id}"
        peer_id = profile.get("peer_id")
        if peer_id:
            managed = registry_entry(self.manager.robots, peer_id)
            if managed is not None and managed.alive():
                return f"peer {peer_id} already running locally"
            try:
                if peer_id in set(self.peer_ids()):
                    return f"peer {peer_id} already present on the mesh"
            except Exception as e:
                # Not knowing the mesh is a reason to hold back, not to guess.
                logger.warning("autospawn: mesh peer lookup failed (%r); skipping %s", e, peer_id)
                return f"mesh peer list unavailable, refusing to spawn {peer_id}"
        return None

    def poll(self) -> dict[str, Any]:
        """One appear/disappear pass. Returns what it did, for tests and logs."""
        if not self.enabled():
            return {"skipped": "autospawn disabled"}
        if self.suspended:
            return {"skipped": "autospawn suspended (record session owns the ports)"}
        ports = {profile_key(p): p for p in self.list_ports() if profile_key(p)}
        spawned: list[str] = []
        despawned: list[str] = []
        ignored: list[str] = []

        for key, port in ports.items():
            self._missing.pop(key, None)
            if key in self.adopted:
                continue
            profile = self.manager.profiles.get(key)
            if profile is None:
                ignored.append(key)
                continue
            reason = self._claimed(port, profile)
            if reason is not None:
                logger.debug("autospawn: %s", reason)
                ignored.append(key)
                continue
            res = self._spawn_from_profile(port, profile)
            peer_id = res.get("peer_id")
            if res.get("error") or not peer_id:
                logger.warning("autospawn: spawning %s failed: %s", key, res.get("error"))
                continue
            self.adopted[key] = peer_id
            spawned.append(peer_id)
            logger.info("autospawn: %s appeared, spawned %s", key, peer_id)

        for key, peer_id in list(self.adopted.items()):
            if key in ports:
                continue
            misses = self._missing.get(key, 0) + 1
            self._missing[key] = misses
            if misses < self.missing_polls:
                continue
            self._missing.pop(key, None)
            self.adopted.pop(key, None)
            self.manager.despawn(peer_id)
            despawned.append(peer_id)
            logger.info("autospawn: %s unplugged, stopped %s", key, peer_id)

        return {"spawned": spawned, "despawned": despawned, "detected_unknown": ignored}

    def _spawn_from_profile(self, port: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        return self.manager.spawn(
            robot_name=profile.get("robot_name") or "so101",
            mode=profile.get("mode") or "real",
            peer_id=profile.get("peer_id"),
            # The board may have re-enumerated on a new path; the live path
            # wins over whatever the profile remembered.
            port=port.get("device") or profile.get("port"),
            cameras=profile.get("cameras"),
            robot_id=profile.get("robot_id"),
            remember=False,
        )

    def run_forever(self, interval: float = AUTOSPAWN_POLL_S) -> None:
        """Poll until :meth:`stop`. For a thread; the server uses asyncio.

        Paced by :class:`strands_robots.mesh.pacing.Ticker` so a slow
        ``poll()`` does not stretch the period by its own duration.
        """
        from strands_robots.mesh.pacing import Ticker

        with Ticker(interval, self._stop) as ticker:
            while not ticker.wait():
                try:
                    self.poll()
                except Exception as e:  # a watcher crash must not be silent
                    logger.warning("autospawn poll failed: %r", e)

    def stop(self) -> None:
        """Ask :meth:`run_forever` to return after the current sleep."""
        self._stop.set()


# : The only modes the dashboard can spawn.
SPAWNABLE_MODES = ("sim", "real")

# : What a caller-chosen peer_id may look like.
_PEER_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}\Z")


def validate_peer_id(peer_id: Any) -> str | None:
    """Refusal reason for a caller-supplied peer_id, or None if acceptable. Only judges ids a CALLER
    chose (``None`` -- "generate one for me" -- is fine).
    """
    if peer_id is None:
        return None
    if not isinstance(peer_id, str):
        return f"peer_id must be a string, got {type(peer_id).__name__}"
    if not _PEER_ID_RE.match(peer_id):
        return (
            f"peer_id {refusal_repr(peer_id)} refused: it becomes a zenoh key segment, so it must "
            f"match [A-Za-z0-9._:-]{{1,64}} - '*' or '/' there rewrites the fleet's "
            f"key space rather than naming a peer"
        )
    return None


#: A serial device path, as the calibration wizard already requires it: under ``/dev/``, no
#: whitespace, no ``..`` segment, and only path characters. A port is the highest-privilege
#: input here - it names the device a child process opens and WRITES handshake bytes to, and it
#: reaches ``lsof`` argv - so its shape is checked before any subprocess or bus probe.
_PORT_RE = re.compile(r"^/dev/[A-Za-z0-9._/-]{1,200}\Z")

#: One path segment (letters, digits, . _ : -), no leading dot: the same ``_SEGMENT`` the
#: calibration wizard applies to a lerobot id, because a robot_id IS a file name in the child
#: (``calibration/robots/<type>/<id>.json``) and an ``is_file`` probe on this side.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}\Z")

#: A lerobot motor-table name (``sts3215``, ``scs0009``, ``xl330-m288``...): it is handed to a
#: child on argv and looked up in a table, so it is a closed charset, not free text.
_MOTOR_MODEL_RE = re.compile(r"^[a-z0-9_-]{1,32}\Z")

#: What a camera NAME may look like. It becomes an observation key, a dataset feature key and
#: a thumbnail FILE NAME (``<episode>_<camera>.jpg``), so a slash or ``..`` in it is a write
#: outside the thumbnail directory, not a camera.
_CAMERA_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}\Z")


def validate_port(port: Any) -> str | None:
    """Refusal reason for a caller-supplied serial port path, or None if it is one."""
    if not isinstance(port, str):
        return f"port must be a string, got {type(port).__name__}"
    text = port.strip()
    if not text or text != port or any(c.isspace() for c in text):
        return f"port {refusal_repr(port)} refused: a serial device path has no whitespace"
    if not _PORT_RE.match(text) or "/../" in text or text.endswith("/..") or text.startswith("/dev/-"):
        return (
            f"port {refusal_repr(port)} refused: it names the device a robot process opens and writes to, "
            f"so it must be a path under /dev/ (letters, digits, . _ / -), like /dev/ttyACM0 or "
            f"/dev/serial/by-id/usb-..."
        )
    return None


def validate_robot_id(robot_id: Any) -> str | None:
    """Refusal reason for a caller-supplied lerobot calibration id, or None if acceptable."""
    if robot_id is None:
        return None
    if not isinstance(robot_id, str):
        return f"robot_id must be a string, got {type(robot_id).__name__}"
    if not _SEGMENT_RE.match(robot_id) or robot_id.startswith("."):
        return (
            f"robot_id {refusal_repr(robot_id)} refused: lerobot uses it as a FILE NAME "
            f"(calibration/robots/<type>/<id>.json), so it must be one path segment "
            f"(letters, digits, . _ : -) that does not start with a dot"
        )
    return None


def validate_motor_model(model: Any) -> str | None:
    """Refusal reason for a caller-supplied motor model name, or None if acceptable."""
    if not isinstance(model, str) or not _MOTOR_MODEL_RE.match(model):
        return (
            f"model {refusal_repr(model)} refused: a lerobot motor table name is lowercase letters, "
            f"digits, _ or - (sts3215, scs0009, xl330-m288...)"
        )
    return None


def validate_spawn(robot_name: Any, mode: Any) -> tuple[str, str] | dict[str, str]:
    """Normalise and check what a spawn was asked for, BEFORE any process exists. Returns
    ``(robot_name, mode)`` or an ``{"error": ...}`` dict.
    """
    mode_s = mode.strip().lower() if isinstance(mode, str) else mode
    if mode_s not in SPAWNABLE_MODES:
        shown = mode if isinstance(mode, str) else type(mode).__name__
        extra = (
            " 'auto' is an SDK-side detection that resolves inside the child, so"
            " the dashboard cannot label the card honestly - say which you mean."
            if mode_s == "auto"
            else ""
        )
        return {"error": f"mode must be one of {', '.join(SPAWNABLE_MODES)} (got {shown!r}).{extra}"}

    name_s = robot_name.strip() if isinstance(robot_name, str) else robot_name
    if not name_s or not isinstance(name_s, str):
        return {"error": "robot_name required"}
    try:
        from strands_robots.robot import _validate_known_robot, resolve_name

        canonical = resolve_name(name_s)
        _validate_known_robot(canonical, name_s, None)
    except ValueError as exc:
        # The SDK's message already names the registry and the resolution.
        return {"error": str(exc)}
    except Exception as exc:  # importing the SDK must never be the thing that fails
        logger.warning("robot-name validation unavailable (%s); accepting %r", exc, name_s)
        return (name_s, mode_s)
    return (canonical, mode_s)


def validate_replay(repo_id: Any, episode: Any, root: Any = None, speed: Any = 1.0) -> dict[str, str] | None:
    """Refusal reason for a replay request, or None - judged BEFORE any process."""
    # The episode domain is the SHARED rule, not a second copy of it.
    episode_error = non_negative_whole_number_error(episode, "episode", "replay")
    if episode_error:
        return {"error": episode_error}
    try:
        speed_f = float(speed)
    except (TypeError, ValueError):
        return {"error": f"speed must be a number, got {type(speed).__name__}"}
    if not math.isfinite(speed_f) or speed_f <= 0:
        return {"error": f"speed must be a finite positive number (got {refusal_str(speed)})"}
    if not isinstance(repo_id, str) or not repo_id.strip():
        return {"error": "repo_id required"}
    rid = repo_id.strip()
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)?\Z", rid):
        return {"error": f"repo_id {rid!r} does not look like a dataset id (org/name) or a local dataset name"}
    if root is not None:
        if not isinstance(root, str) or not root.strip():
            return {"error": f"root must be a path string, got {type(root).__name__}"}
        # Containment before existence, and one sentence either way: a caller who names a
        # directory outside the dataset home learns nothing about whether it is there.
        from strands_robots.dataset_source import _lerobot_home

        home = os.path.realpath(os.path.expanduser(str(_lerobot_home())))
        candidate = os.path.normpath(os.path.join(home, os.path.expanduser(root.strip())))
        if not candidate.startswith(home + os.sep):
            return {"error": "root must live under the dataset home ($HF_LEROBOT_HOME)"}
        # A link under the home that points outside it is outside it.
        if not os.path.realpath(candidate).startswith(home + os.sep):
            return {"error": "root must live under the dataset home ($HF_LEROBOT_HOME)"}
        if not os.path.isdir(candidate):
            return {"error": "root does not exist under the dataset home - a replay from it can only fail"}
    return None


# : The camera options lerobot's OpenCVCameraConfig declares, as a fallback for when lerobot
# cannot be : imported (the dashboard must validate a config on a machine with no robot stack
# installed).
_CAMERA_OPTION_FIELDS = (
    "backend",
    "color_mode",
    "fourcc",
    "fps",
    "height",
    "index_or_path",
    "rotation",
    "warmup_s",
    "width",
)


def _camera_option_names() -> tuple[str, ...]:
    try:
        import dataclasses

        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    except Exception:  # noqa: BLE001 - no lerobot here: the frozen list is the best truth available
        return _CAMERA_OPTION_FIELDS
    return tuple(sorted(f.name for f in dataclasses.fields(OpenCVCameraConfig)))


# : The ENUMERATED camera option values lerobot publishes, as a fallback for when lerobot
# cannot be : imported (same contract as _CAMERA_OPTION_FIELDS above, and a test asserts it
# matches the real : enums wherever lerobot IS importable).
_CAMERA_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "color_mode": ("bgr", "rgb"),
    "rotation": ("-90", "0", "90", "180"),
}


def _camera_option_values() -> dict[str, tuple[str, ...]]:
    """The admitted spellings per enumerated camera option, read from lerobot's own enums."""
    try:
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
    except Exception:  # noqa: BLE001 - no lerobot here: the frozen list is the best truth available
        return dict(_CAMERA_ENUM_VALUES)
    return {
        "color_mode": tuple(sorted(str(m.value) for m in ColorMode)),
        "rotation": tuple(str(r.value) for r in Cv2Rotation),
    }


def camera_option_value_problem(
    option: str, value: Any, admitted: dict[str, tuple[str, ...]] | None = None
) -> str | None:
    """Why this ENUMERATED option's value cannot work, or None."""
    table = _camera_option_values() if admitted is None else admitted
    allowed = table.get(option)
    if not allowed:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return (
            f"{option} must be one of {', '.join(allowed)}, "
            f"got {type(value).__name__} - lerobot reads this option as an enumerated value"
        )
    given = str(value).strip()
    if given in allowed:
        return None
    close = difflib.get_close_matches(given.lower(), [a.lower() for a in allowed], n=1, cutoff=0.6)
    # Compared against the spelling AS GIVEN, not its lowercase form: a case-only mistake ("RGB")
    # is the likeliest one an operator makes here, and folding the case first made close[0] equal
    # the normalised input, which silently swallowed the hint for exactly that case (caught by its
    # test).
    hint = f" Did you mean {close[0]!r}?" if close and close[0] != given else ""
    return (
        f"{option}={refusal_repr(value)} is not one of {', '.join(allowed)}.{hint} "
        f"lerobot refuses the spelling itself, so the child would die at connect"
    )


def requested_camera_names(cameras: Any) -> list[str]:
    """The camera names a spawn ASKED for, sorted, or [] if it asked for none."""
    if not isinstance(cameras, dict):
        return []
    return sorted(str(name) for name in cameras if name)


def indices_beyond_roster(cameras: Any, roster_size: int) -> dict[str, int]:
    """Requested camera indices this machine cannot possibly have, as {name: index}."""
    out: dict[str, int] = {}
    if not isinstance(cameras, dict) or roster_size <= 0:
        return out
    for name, cfg in cameras.items():
        idx = cfg.get("index_or_path") if isinstance(cfg, dict) else cfg
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx >= roster_size or idx < 0:
            out[str(name)] = idx
    return out


def validate_cameras(cameras: Any) -> dict[str, str] | None:
    """Refusal reason for a spawn/reconfigure camera config, or None."""
    if cameras is None:
        return None
    if not isinstance(cameras, dict):
        return {"error": f"cameras must be a mapping of name -> config, got {type(cameras).__name__}"}
    for name, cfg in cameras.items():
        if not isinstance(name, str) or not name.strip():
            return {"error": f"camera name {name!r} must be a non-empty string (top/wrist/main...)"}
        if not _CAMERA_NAME_RE.match(name):
            return {
                "error": (
                    f"camera name {refusal_repr(name)} refused: it becomes an observation key and a "
                    f"thumbnail file name, so it must match [A-Za-z0-9._-]{{1,64}} (top, wrist_left, cam0...)"
                )
            }
        if not isinstance(cfg, dict):
            return {
                "error": (
                    f"camera {name!r} config must be a mapping like "
                    f'{{"index_or_path": 0, "fps": 30}}, got {type(cfg).__name__}: {cfg!r}'
                )
            }
        iop = cfg.get("index_or_path")
        if iop is None:
            return {"error": f"camera {name!r} needs index_or_path (an OpenCV index or a device path)"}
        if isinstance(iop, bool) or not isinstance(iop, (int, str)):
            return {"error": f"camera {name!r}: index_or_path must be an integer index or a path string"}
        if isinstance(iop, int) and iop < 0:
            return {"error": f"camera {name!r}: index_or_path {iop} - an OpenCV index is not negative"}
        for fname, lo, hi in (("fps", 1, 240), ("width", 16, 7680), ("height", 16, 4320)):
            v = cfg.get(fname)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int):
                return {"error": f"camera {name!r}: {fname} must be an integer, got {type(v).__name__}"}
            if not lo <= v <= hi:
                return {"error": f"camera {name!r}: {fname}={v} is outside {lo}..{hi}"}
        # An UNKNOWN option is refused HERE, because the child refuses it too -- and by the time the
        # child speaks, reconfigure_cameras has already despawned the arm that was working.
        for key in camera_liveness.ANNOTATION_KEYS:
            if key in cfg and not isinstance(cfg[key], str):
                return {
                    "error": (
                        f"camera {name!r}: {key} is the dashboard's note of which device this index "
                        f"was, so it must be a string, got {type(cfg[key]).__name__}"
                    )
                }
        accepted = _camera_option_names()
        unknown = sorted(
            k for k in cfg if k not in accepted and k != "type" and k not in camera_liveness.ANNOTATION_KEYS
        )
        if unknown:
            hints = []
            for key in unknown:
                close = difflib.get_close_matches(str(key), list(accepted), n=1, cutoff=0.7)
                if close:
                    hints.append(f"{key!r} -> {close[0]!r}")
            hint = f" Did you mean {', '.join(hints)}?" if hints else ""
            return {
                "error": (
                    f"camera {name!r}: unknown option(s) {unknown}.{hint} "
                    f"Accepted: {', '.join(accepted)} (plus 'type' to choose the backend). "
                    f"Refused before anything is stopped - a reconfigure despawns the robot first."
                )
            }
        # ...and an option whose NAME is right but whose VALUE is not in lerobot's enumeration dies in
        # the same place, after the despawn.
        for option in sorted(cfg):
            problem = camera_option_value_problem(option, cfg[option])
            if problem:
                return {"error": f"camera {name!r}: {problem}"}
    return None


class DeviceManager:
    """Owns local device discovery + robot child processes."""

    CAMERA_CACHE_TTL_S = 30.0  # don't re-open /dev cameras on every request
    # : How long a port -> serial mapping is trusted. /api/fleet is polled about : once a second;
    # enumerating the USB bus that often to answer "which arm is : the leader" would be absurd,
    # and the answer only changes on a replug.
    PORT_SERIAL_TTL_S = 10.0

    def __init__(self, profiles_path: str | None = None) -> None:
        self.robots: dict[str, ManagedRobot] = {}
        self._lock = threading.Lock()
        # Concurrency contract: ``robots`` is mutated (spawn/despawn) under ``_lock`` on worker
        # threads; readers on the event loop (bridge snapshot, /api/fleet) iterate a GIL-atomic
        # ``list(...)`` copy so a despawn mid-poll cannot raise "dictionary changed size".
        self._camera_cache: list[dict[str, Any]] = []
        self._camera_cache_t = 0.0
        # : index -> stderr from the last failed probe, and index -> last known : geometry.
        self._camera_failures: dict[int, str] = {}
        self._camera_memory: dict[int, dict[str, Any]] = {}
        self._camera_names_cache: list[dict[str, Any]] = []
        # Each /api/devices call runs in its own worker thread, so probes are concurrent by default:
        # separate locks because a name scan is cheap metadata while a probe OPENS the devices (see
        # cameras.probe_needed).
        self._camera_probe_lock = threading.Lock()
        self._camera_names_lock = threading.Lock()
        self._camera_names_cache_t = 0.0  # a time.monotonic() stamp: compared, never shown
        self._port_serial_cache: dict[str, str] = {}
        self._port_serial_cache_t = 0.0
        # One preview at a time: two concurrent opens of the same device
        # wedge some UVC cameras until replug.
        self._preview_lock = threading.Lock()
        self.profiles = ProfileStore(profiles_path)
        self.autospawn: AutoSpawnWatcher | None = None

    def _claimed_camera_indices(self) -> dict[int, str]:
        """OpenCV indices owned by LIVE managed robots -> peer_id."""
        claimed: dict[int, str] = {}
        for m in list(self.robots.values()):
            if not m.alive():
                continue
            for cfg in (m.cameras or {}).values():
                iop = cfg.get("index_or_path") if isinstance(cfg, dict) else None
                if isinstance(iop, bool):
                    continue
                if isinstance(iop, int):
                    claimed[iop] = m.peer_id
                elif isinstance(iop, str) and iop.isdigit():
                    claimed[int(iop)] = m.peer_id
        return claimed

    def _streaming_indices(self, live_cameras: Mapping[str, Iterable[str]] | None) -> set[int] | None:
        """Which claimed indices are provably publishing frames. ``live_cameras`` is peer_id -> camera
        NAMES the mesh has actually seen frames for.
        """
        if live_cameras is None:
            return None
        streaming: set[int] = set()
        for peer_id, names in live_cameras.items():
            managed = registry_entry(self.robots, peer_id)
            if managed is None:
                continue
            wanted = {str(n).split("/")[-1] for n in names}
            for cam_name, cfg in (managed.cameras or {}).items():
                if str(cam_name).split("/")[-1] not in wanted:
                    continue
                iop = cfg.get("index_or_path") if isinstance(cfg, dict) else None
                if isinstance(iop, bool):
                    continue
                if isinstance(iop, int):
                    streaming.add(iop)
                elif isinstance(iop, str) and iop.isdigit():
                    streaming.add(int(iop))
        return streaming

    def _cameras(
        self,
        refresh: bool = False,
        live_cameras: Mapping[str, Iterable[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Every camera this machine could have, each with its state and WHY."""
        claimed = self._claimed_camera_indices()
        requested_at = time.time()
        # One probe at a time.
        if camera_facts.probe_needed(
            refresh=refresh,
            requested_at=requested_at,
            cache_t=self._camera_cache_t,
            ttl_s=self.CAMERA_CACHE_TTL_S,
            now=requested_at,
        ):
            with self._camera_probe_lock:
                now = time.time()
                if camera_facts.probe_needed(
                    refresh=refresh,
                    requested_at=requested_at,
                    cache_t=self._camera_cache_t,
                    ttl_s=self.CAMERA_CACHE_TTL_S,
                    now=now,
                ):
                    probed, failures = scan_cameras_with_failures(skip=set(claimed))
                    self._camera_cache = probed
                    self._camera_failures = failures
                    self._camera_cache_t = time.time()
                    for cam in probed:
                        # What we know about an index survives someone claiming it.
                        self._camera_memory[int(cam["index"])] = dict(cam)
        return camera_facts.merge_cameras(
            probed=[c for c in self._camera_cache if c["index"] not in claimed],
            claimed=claimed,
            roster=self._camera_names(),
            remembered=self._camera_memory,
            failures={i: t for i, t in (self._camera_failures or {}).items() if i not in claimed},
            streaming=self._streaming_indices(live_cameras),
        )

    def managed_children(self) -> list[dict[str, Any]]:
        """The managed table, cheaply: no serial scan, no camera probe. devices() rescans the USB ports and
        can probe cameras, so it must never be called from /api/fleet (~1Hz).
        """
        with self._lock:
            return [
                {
                    "peer_id": m.peer_id,
                    "robot_name": m.robot_name,
                    "mode": m.mode,
                    "alive": m.alive(),
                    "returncode": m.process.poll() if m.process is not None else None,
                    "started_at": m.started_at,
                }
                for m in list(self.robots.values())
            ]

    def devices(
        self,
        refresh: bool = False,
        live_cameras: Mapping[str, Iterable[str]] | None = None,
    ) -> dict[str, Any]:
        """Serial ports, cameras and managed robots on this machine, with in-use evidence."""
        cams = self._cameras(refresh=refresh, live_cameras=live_cameras)
        # ONE serial scan per call, shared below: profile_for_port() rescans on
        # every lookup, so per-child role resolution would have re-enumerated the
        # USB bus once per managed robot.
        ports = scan_serial_ports()
        roles = {
            p["device"]: self._role_fields(self.profiles.get(str(p["serial_number"])))
            for p in ports
            if p.get("device") and p.get("serial_number")
        }
        remembered: dict[str, dict[str, Any]] = {}
        for p in ports:
            dev = p.get("device")
            if not dev:
                continue
            profile = self.profiles.get(profile_key(p))
            mem = remembered_spawn(profile)
            if not mem:
                continue
            health = remembered_camera_health(
                (profile or {}).get("cameras"),
                cams,
                str(mem.get("peer_id") or ""),
            )
            if health:
                mem["camera_health"] = health
            remembered[dev] = mem
        return {
            "serial_ports": [
                {
                    **e,
                    **roles.get(e.get("device"), {}),
                    # Absent stays absent: a board nobody configured carries no `remembered` key at
                    # all, rather than an empty object a screen would have to special-case.
                    **({"remembered": r} if (r := remembered.get(cast(str, e.get("device")))) else {}),
                }
                for e in ports
            ],
            "cameras": cams,
            # One loud line when the whole machine is blocked, instead of the
            # same reason repeated on every row (and missed on all of them).
            "camera_problem": camera_facts.blocked_verdict(cams),
            "camera_names": self._camera_names(refresh=refresh),
            # The keys here are PEER IDS, not pids.
            "managed": {
                peer_id: {
                    "peer_id": m.peer_id,
                    "robot_name": m.robot_name,
                    "mode": m.mode,
                    "port": m.port,
                    "alive": m.alive(),
                    "started_at": m.started_at,
                    # None only when the child was never started; a pid for a dead
                    # process is still useful (it is what the logs refer to), so it
                    # is reported alongside alive=False rather than blanked.
                    "pid": m.process.pid if m.process is not None else None,
                    "returncode": m.process.poll() if m.process is not None else None,
                    "log_tail": list(m.logs)[-20:],
                    "cameras": dict(m.cameras or {}),
                    **roles.get(m.port, {}),
                }
                for peer_id, m in list(self.robots.items())
            },
        }

    def _port_serials(self, refresh: bool = False) -> dict[str, str]:
        """/dev path -> USB serial, cached (see PORT_SERIAL_TTL_S)."""
        now = time.time()
        if refresh or (now - self._port_serial_cache_t) > self.PORT_SERIAL_TTL_S:
            try:
                self._port_serial_cache = {
                    str(p["device"]): str(p["serial_number"])
                    for p in scan_serial_ports()
                    if p.get("device") and p.get("serial_number")
                }
                self._port_serial_cache_t = now
            except Exception as e:
                # Keep the previous map rather than forgetting every role because
                # one scan failed: a stale answer beats "no arm has a role".
                logger.warning("serial rescan for roles failed (%r); keeping the last map", e)
        return self._port_serial_cache

    def annotations_by_peer(self) -> dict[str, dict[str, Any]]:
        """Everything the DASHBOARD knows about a managed peer that the mesh cannot say."""
        out: dict[str, dict[str, Any]] = {pid: dict(f) for pid, f in self.roles_by_peer().items()}
        # joint_silence is the mesh lane's module (log-line readings of a silent joint probe).
        # Imported here, not at module top, so the device roster stands on its own.
        from . import joint_silence

        for peer_id, m in list(self.robots.items()):
            names = requested_camera_names(m.cameras)
            if names:
                out.setdefault(peer_id, {})["cameras_requested"] = names
            lines = list(m.logs)
            problem = joint_silence.classify(
                lines,
                _calibrations_on_disk(),
                robot_name=m.robot_name,
                robot_id=m.robot_id,
            )
            if problem:
                m.joint_problem_seen = problem
            elif joint_silence.recovered(lines):
                # The child SAID it recovered: that is a real clearing rail, unlike the reason
                # merely scrolling out of the window.
                m.joint_problem_seen = None
            if m.joint_problem_seen:
                out.setdefault(peer_id, {})["joint_problem"] = m.joint_problem_seen
        return out

    def roles_by_peer(self) -> dict[str, dict[str, Any]]:
        """Leader/follower role facts for every managed peer, keyed by peer id."""
        serials = self._port_serials()
        out: dict[str, dict[str, Any]] = {}
        for peer_id, m in list(self.robots.items()):
            serial = serials.get(m.port or "")
            fields = self._role_fields(self.profiles.get(serial) if serial else None)
            if fields:
                out[peer_id] = fields
        return out

    @staticmethod
    def _role_fields(profile: Mapping[str, Any] | None) -> dict[str, Any]:
        if not profile:
            return {}
        return {f: profile[f] for f in ("role", "role_volts", "role_source") if profile.get(f) is not None}

    #: A roster older than this is not evidence of anything. Same number and same reason as the
    #: record gate's ROSTER_MAX_AGE_S: this morning's scan can easily omit a camera plugged in since.
    ROSTER_MAX_AGE_S = 300.0

    def _roster_for_stamp(self) -> list[dict[str, Any]]:
        """The camera roster this manager ALREADY has, or [] - never a fresh scan."""
        try:
            roster = self._camera_names_cache
            taken_at = float(self._camera_names_cache_t or 0.0)
            if not roster or taken_at <= 0 or time.monotonic() - taken_at > self.ROSTER_MAX_AGE_S:
                return []
            return list(roster)
        except Exception:  # noqa: BLE001 - a note is never worth breaking a spawn for
            logger.debug("[devices] could not read the camera roster for the identity stamp")
            return []

    def _camera_names(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Cached roster of camera names (see scan_camera_names on ordering)."""
        requested_at = time.monotonic()
        if camera_facts.probe_needed(
            refresh=refresh,
            requested_at=requested_at,
            cache_t=self._camera_names_cache_t,
            ttl_s=self.CAMERA_CACHE_TTL_S,
            now=requested_at,
        ):
            with self._camera_names_lock:
                if camera_facts.probe_needed(
                    refresh=refresh,
                    requested_at=requested_at,
                    cache_t=self._camera_names_cache_t,
                    ttl_s=self.CAMERA_CACHE_TTL_S,
                    now=time.monotonic(),
                ):
                    self._camera_names_cache = scan_camera_names()
                    self._camera_names_cache_t = time.monotonic()
        return self._camera_names_cache

    def preview_frame(
        self,
        index: int,
        live_cameras: Mapping[str, Iterable[str]] | None = None,
    ) -> bytes:
        """One JPEG frame from a camera nobody is streaming."""
        claimed = self._claimed_camera_indices()
        streaming = self._streaming_indices(live_cameras)
        if index in claimed and (streaming is None or index in streaming):
            raise PermissionError(
                f"camera index {index} is streaming for {claimed[index]} - watch it on that robot's card instead"
            )
        import cv2

        with self._preview_lock:
            cap = cv2.VideoCapture(index)
            try:
                if not cap.isOpened():
                    raise self._camera_fault(index)
                # A couple of warm-up reads: first frames from a cold sensor
                # are often black or half-exposed.
                frame = None
                for _ in range(3):
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                        break
                if frame is None:
                    raise self._camera_fault(index)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                return bytes(buf.tobytes())
            finally:
                cap.release()

    def probe_modes(
        self,
        index: int,
        live_cameras: Mapping[str, Iterable[str]] | None = None,
    ) -> dict[str, Any]:
        """Resolutions and frame rates an unclaimed camera index actually delivers."""
        claimed = self._claimed_camera_indices()
        streaming = self._streaming_indices(live_cameras)
        if index in claimed and (streaming is None or index in streaming):
            raise PermissionError(
                f"camera index {index} is streaming for {claimed[index]} - "
                f"stop that robot before probing its camera's modes"
            )
        import cv2

        readbacks: list[dict[str, Any]] = []
        with self._preview_lock:
            cap = cv2.VideoCapture(index)
            try:
                if not cap.isOpened():
                    raise self._camera_fault(index)
                # Native mode first: what the camera does when nobody asks.
                native = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                }
                for w, h in CAMERA_MODE_CANDIDATES:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    for fps in CAMERA_FPS_CANDIDATES:
                        cap.set(cv2.CAP_PROP_FPS, fps)
                        readbacks.append(
                            {
                                "requested": {"width": w, "height": h, "fps": fps},
                                "got": {
                                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                    "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                                },
                            }
                        )
            finally:
                cap.release()
        return {"index": index, "native": native, "modes": modes_from_readbacks(native, readbacks)}

    def port_owner(self, port: str) -> str | None:
        """peer_id of the LIVE managed child holding this serial port, if any."""
        for m in list(self.robots.values()):
            if m.alive() and m.port and str(m.port) == str(port):
                return m.peer_id
        return None

    def profile_for_port(self, port: str) -> dict[str, Any] | None:
        """The remembered profile for the board at ``port``. Profiles are keyed by USB SERIAL NUMBER, not
        by port - a /dev name is reassigned by the OS, a serial is the board.
        """
        for entry in scan_serial_ports():
            if entry.get("device") == port and entry.get("serial_number"):
                return self.profiles.get(str(entry["serial_number"]))
        return None

    def read_bus_role(
        self,
        port: str,
        motor_model: str = "sts3215",
        ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        timeout: float = 25.0,
    ) -> dict[str, Any]:
        """Measure a Feetech bus's supply voltage and say which arm role it is. READS ONLY -
        Present_Voltage (register 62, one byte, read-only in lerobot's Feetech table).
        """
        # Shape first: both strings reach a child's argv and the port is the path that child
        # opens and transmits read-request packets to.
        if (bad := validate_port(port)) is not None:
            raise ValueError(bad)
        if (bad := validate_motor_model(motor_model)) is not None:
            raise ValueError(bad)
        owner = self.port_owner(port)
        if owner is not None:
            raise PermissionError(
                f"{port} is held by {owner} - despawn it first, then read the voltage "
                f"(the arm has to release the bus; nothing can share a servo port)"
            )
        code = _BUS_VOLTAGE_SRC
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, port, motor_model, ",".join(str(i) for i in ids)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "port": port,
                **arm_roles.role_verdict({}),
                "reason": f"the bus did not answer within {timeout:.0f}s",
                "remedy": "unplug and replug the arm's USB cable, then retry",
            }
        readings: dict[str, float | None] = {}
        try:
            readings = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            pass  # no readings: role_verdict({}) below says the bus answered nothing usable
        verdict = arm_roles.role_verdict(readings)
        verdict["port"] = port
        if verdict["role"] == "unknown" and proc.stderr:
            # The SDK's own words beat a generic failure line.
            tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-1:]
            if tail:
                verdict["detail"] = tail[0][:300]
        return verdict

    def measure_arm_role(self, port: str, model: str = "sts3215") -> dict[str, Any]:
        """Read the role off the bus AND remember it against the board's serial."""
        verdict = self.read_bus_role(port, model)
        serial = None
        for entry in scan_serial_ports():
            if entry.get("device") == port:
                serial = entry.get("serial_number")
                break
        verdict["serial_number"] = serial
        previous = self.profiles.get(str(serial)) if serial else None
        verdict["mismatch"] = arm_roles.disagreement((previous or {}).get("role"), verdict)
        if serial:
            saved = self.profiles.record_role(str(serial), verdict)
            verdict["remembered"] = bool(saved)
        else:
            verdict["remembered"] = False
            verdict["remember_problem"] = (
                f"{port} reports no USB serial number, so the role cannot be remembered "
                f"for this board (a /dev name is reassigned by the OS)"
            )
        return verdict

    def _camera_fault(self, index: int) -> camera_facts.CameraUnavailable:
        """Ask OpenCV why, in a child process, and answer in the operator's terms."""
        stderr = diagnose_camera_indices([index]).get(index, "")
        state, reason, remedy = camera_facts.classify_probe_stderr(stderr)
        if state == "absent" and not stderr.strip():
            reason = "it would not open, and OpenCV gave no reason"
            remedy = "check the cable, then rescan"
        return camera_facts.CameraUnavailable(index, state, reason, remedy)

    def logs(self, peer_id: str) -> dict[str, Any]:
        """Full ring buffer for one managed robot."""
        m = registry_entry(self.robots, peer_id)
        if m is None:
            return {"error": f"unknown peer {peer_id}"}
        return {"peer_id": peer_id, "alive": m.alive(), "lines": list(m.logs)}

    def _unique_peer_id(self, base: str) -> str:
        """A peer id that is not already tracked, starting from ``base``."""
        if not registered(self.robots, base):
            return base
        for n in range(2, 1000):
            candidate = f"{base}-{n}"
            if not registered(self.robots, candidate):
                return candidate
        return f"{base}-{uuid.uuid4().hex[:6]}"  # pathological; still unique

    def _running_job(self, mode: str, **fields: Any) -> ManagedRobot | None:
        """A live process of ``mode`` whose job matches every given field.

        Caller must hold ``self._lock``.
        """
        for managed in self.robots.values():
            if managed.mode != mode or not managed.alive():
                continue
            if all(managed.job.get(k) == v for k, v in fields.items()):
                return managed
        return None

    def spawn(
        self,
        robot_name: str,
        mode: str = "sim",
        peer_id: str | None = None,
        port: str | None = None,
        cameras: dict[str, Any] | None = None,
        robot_id: str | None = None,
        remember: bool = True,
    ) -> dict[str, Any]:
        """Start a robot child process for ``robot_name`` and register it as a managed peer."""
        import json as _json

        # Refuse before a process exists: a pid reported for a child that is already raising is worse
        # than a refusal, because the fleet grid shows a card for it and the operator waits out the
        # settle window to find out.
        checked = validate_spawn(robot_name, mode)
        if isinstance(checked, dict):
            return checked
        robot_name, mode = checked

        # A caller-chosen peer_id becomes a zenoh key segment (see
        # validate_peer_id): '*' there is a fleet-wide wildcard, not a name.
        bad_id = validate_peer_id(peer_id)
        if bad_id:
            return {"error": bad_id}

        bad_cams = validate_cameras(cameras)
        if bad_cams:
            return bad_cams

        if mode == "real" and not port:
            return {"error": "port required for mode=real"}
        # The port reaches lsof argv (bus_claim) and is the path the child opens and writes
        # handshake bytes to; robot_id is a file name in the child. Both are shape-checked
        # here, before any subprocess, the way calibration_run.cli_args already does.
        if port is not None and (bad_port := validate_port(port)) is not None:
            return {"error": bad_port}
        if (bad_rid := validate_robot_id(robot_id)) is not None:
            return {"error": bad_rid}
        peer_id = peer_id or self._unique_peer_id(f"{robot_name}-{mode}-{int(time.time()) % 10000}")
        with self._lock:
            if (_live := registry_entry(self.robots, peer_id)) is not None and _live.alive():
                return {"error": f"peer {peer_id} already running"}
            if mode == "real" and port:
                tracked = {
                    r.process.pid: pid_key for pid_key, r in self.robots.items() if r.process is not None and r.alive()
                }
                conflict = bus_claim.bus_conflict(port, bus_claim.bus_holders(port), tracked)
                if conflict:
                    return {"error": conflict}
            # Stamp each numeric camera index with the device that answers it RIGHT NOW, while the
            # operator's choice and the machine's numbering are still known to agree.
            cameras = cast(
                "dict[str, Any] | None",
                camera_liveness.stamp_device_names(cameras, self._roster_for_stamp()),
            )
            cfg = {
                "robot_name": robot_name,
                "mode": mode,
                "peer_id": peer_id,
                "port": port,
                "cameras": cameras,
                "robot_id": robot_id,
            }
            proc = subprocess.Popen(
                # The child gets the camera config with the dashboard's own notes REMOVED:
                # _build_camera_config refuses any key OpenCVCameraConfig does not declare, so a forwarded
                # device_name would not degrade one camera, it would kill every camera on the arm.
                [
                    sys.executable,
                    "-c",
                    _SPAWNER,
                    _json.dumps({**cfg, "cameras": camera_liveness.without_annotations(cameras)}),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            managed = ManagedRobot(
                peer_id=peer_id,
                robot_name=robot_name,
                mode=mode,
                port=port,
                robot_id=robot_id,
                cameras=cameras or {},
                process=proc,
                started_at=time.time(),
            )
            self.robots[peer_id] = managed
            # Drain stdout forever - daemon thread, ring buffer.
            threading.Thread(
                target=_drain,
                args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}",
                daemon=True,
            ).start()
        logger.info("spawned %s (%s, pid=%s)", peer_id, mode, proc.pid)
        if remember and mode == "real" and port:
            self.remember_profile(cfg)
        out = {"peer_id": peer_id, "pid": proc.pid, "mode": mode}
        if mode == "real" and robot_id:
            try:
                from strands_robots.dashboard.calibration import robot_calibration_gap

                gap = robot_calibration_gap(robot_name, robot_id)
            except Exception:  # a diagnostic must never be able to break a spawn
                gap = None
            if gap:
                logger.warning("%s: %s", peer_id, gap)
                out["calibration_warning"] = gap
        return out

    def settle(
        self,
        peer_id: str,
        *,
        timeout: float = SPAWN_SETTLE_S,
        is_up: Callable[[str], bool] | None = None,
        poll: float = 0.1,
        sleep: Callable[[float], None] | None = None,
        now: Callable[[], float] | None = None,
    ) -> dict[str, Any]:
        """Watch a just-spawned peer long enough to answer honestly.

        ``spawn()`` returns as soon as ``Popen`` hands back a pid; a child that
        dies a second later still got one, so this watches the gap and answers
        ``running`` / ``failed`` / ``starting`` / ``gone``.

        Args:
            peer_id: The peer returned by :meth:`spawn`.
            timeout: Seconds to watch before answering ``starting``.
            is_up: Optional presence check -- given the peer id, True when the
                mesh has heard from it. Without one, an alive child at the
                deadline is ``starting`` (a pid alone is not evidence).
            poll: Seconds between checks.
            sleep: Injected for tests.
            now: Injected for tests.

        Returns:
            A dict with ``status`` plus whatever that status carries.
        """
        _sleep = sleep or time.sleep
        _now = now or time.monotonic
        deadline = _now() + max(0.0, timeout)
        while True:
            managed = registry_entry(self.robots, peer_id)
            if managed is None:
                return {"status": "gone"}
            if not managed.alive():
                proc = managed.process
                code = proc.poll() if proc is not None else None
                reason = crash_reason(list(managed.logs))
                out: dict[str, Any] = {"status": "failed", "exit_code": code}
                # The log tail travels with the failure: a reason the operator
                # can act on beats "exit code 1", and they should not have to
                # go find a second endpoint while the card is still on screen.
                out["reason"] = reason or (
                    f"the process exited with code {code}" if code is not None else "the process is gone"
                )
                out["log_tail"] = list(managed.logs)[-12:]
                return out
            if is_up is not None:
                try:
                    if is_up(peer_id):
                        return {"status": "running"}
                except Exception:  # noqa: BLE001 - a broken probe must not fail a good spawn
                    logger.debug("settle: presence probe raised for %s", peer_id, exc_info=True)
            if _now() >= deadline:
                # Alive but unannounced. Say exactly that.
                return {"status": "starting", "waited_s": round(max(0.0, timeout), 2)}
            _sleep(min(poll, max(0.0, deadline - _now())))

    def remember_profile(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Save this spawn payload as the profile for the board behind its port. Keyed by the port's USB
        serial number so a replug on a different ``/dev/cu.*`` path still finds it.
        """
        port = payload.get("port")
        if not port:
            return None
        try:
            detected = {p.get("device"): p for p in scan_serial_ports()}
        except Exception as e:
            logger.warning("could not rescan serial ports to save a profile: %r", e)
            return None
        info = detected.get(port)
        if info is None:
            # The port is not in the scan at all, so there is no board to remember.
            logger.warning(
                "not saving a profile for %s: it is not in the serial scan, so there is no board to key it to",
                port,
            )
            return None
        key = profile_key(info)
        if not key:
            return None
        return self.profiles.save(key, payload)

    def start_autospawn(
        self,
        list_ports: Callable[[], list[dict[str, Any]]] | None = None,
        peer_ids: Callable[[], Iterable[str]] | None = None,
    ) -> AutoSpawnWatcher | None:
        """Create the USB auto-spawn watcher, or None when the environment forbids it."""
        veto = autospawn_veto(os.environ)
        if veto is not None:
            logger.warning("USB auto-spawn refused: %s", veto)
            return None
        self.autospawn = AutoSpawnWatcher(self, list_ports=list_ports, peer_ids=peer_ids)
        return self.autospawn

    def replay(
        self,
        repo_id: str,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        robot_name: str = "so101",
    ) -> dict[str, Any]:
        """One-shot replay sim: spawns a mesh sim peer, replays the episode with real physics + cameras
        streaming on the mesh, exits when done.
        """
        import json as _json

        # Keyword-explicit so the forward is legible as one: this is the only
        # call in the method that submits `episode` to the shared rule.
        bad = validate_replay(repo_id, episode=episode, root=root, speed=speed)
        if bad:
            return bad
        # Now that the shared rule is the judge, an accepted `episode` may be a float or a numpy
        # scalar with an integral value.
        episode = int(episode)

        with self._lock:
            # Clicking Run twice is one click too many: a second sim of the same episode fights the first
            # for the same mesh peer name and doubles the physics load for nothing.
            running = self._running_job("replay", repo_id=repo_id, episode=episode)
            if running is not None:
                return {
                    "error": (
                        f"episode {episode} of {repo_id} is already replaying as "
                        f"{running.peer_id} - watch that card instead"
                    ),
                    "peer_id": running.peer_id,
                    "already_running": True,
                }
            peer_id = self._unique_peer_id(f"replay-{int(time.time()) % 100000}")
            cfg = {
                "peer_id": peer_id,
                "repo_id": repo_id,
                "episode": episode,
                "root": root,
                "speed": speed,
                "robot_name": robot_name,
            }
            proc = subprocess.Popen(
                [sys.executable, "-c", _REPLAY_SPAWNER, _json.dumps(cfg)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            managed = ManagedRobot(
                peer_id=peer_id,
                robot_name=robot_name,
                mode="replay",
                process=proc,
                started_at=time.time(),
                job={"repo_id": repo_id, "episode": episode},
            )
            self.robots[peer_id] = managed
            threading.Thread(
                target=_drain,
                args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}",
                daemon=True,
            ).start()
        logger.info("replay %s ep%d as %s (pid=%s)", repo_id, episode, peer_id, proc.pid)
        return {"peer_id": peer_id, "pid": proc.pid, "repo_id": repo_id, "episode": episode}

    def collect(
        self,
        dataset_root: str,
        dataset_repo_id: str = "local/collected",
        robot_name: str = "so101",
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 5,
        duration: float = 10.0,
        fps: int = 30,
    ) -> dict[str, Any]:
        """One-shot data collection: spawn a mesh sim, roll out a policy for N recorded episodes
        (parquet-truth verified by the run_policy tool), exit.
        """
        import json as _json

        with self._lock:
            # Two recorders writing one dataset directory interleave episodes
            # into each other's files -- this guard protects DATA, not just CPU.
            running = self._running_job("collect", dataset_root=dataset_root)
            if running is not None:
                return {
                    "error": (
                        f"a recording session is already writing to {dataset_root} "
                        f"as {running.peer_id} - stop it before starting another"
                    ),
                    "peer_id": running.peer_id,
                    "already_running": True,
                }
            peer_id = self._unique_peer_id(f"collect-{int(time.time()) % 100000}")
            # The id is reserved immediately, under the SAME lock as the guard: releasing it here would
            # let two concurrent recorders both pass the check and both mint the same id -- the bug this
            # method is fixing.
            self.robots[peer_id] = ManagedRobot(
                peer_id=peer_id,
                robot_name=robot_name,
                mode="collect",
                started_at=time.time(),
                job={"dataset_root": dataset_root},
            )
        cfg = {
            "peer_id": peer_id,
            "robot_name": robot_name,
            "policy_provider": policy_provider,
            "policy_config": policy_config,
            "instruction": instruction,
            "n_episodes": n_episodes,
            "duration": duration,
            "dataset_root": dataset_root,
            "dataset_repo_id": dataset_repo_id,
            "fps": fps,
        }
        with self._lock:
            proc = subprocess.Popen(
                [sys.executable, "-c", _COLLECT_SPAWNER, _json.dumps(cfg)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            managed = self.robots[peer_id]  # the reservation made above
            managed.process = proc
            threading.Thread(
                target=_drain,
                args=(proc, managed.logs, peer_id),
                name=f"drain-{peer_id}",
                daemon=True,
            ).start()
        logger.info("collect %d eps -> %s as %s (pid=%s)", n_episodes, dataset_root, peer_id, proc.pid)
        return {"peer_id": peer_id, "pid": proc.pid, "dataset_root": dataset_root, "n_episodes": n_episodes}

    def despawn(self, peer_id: str) -> dict[str, Any]:
        """Stop a managed peer's child process and drop it from the roster."""
        with self._lock:
            m = registry_entry(self.robots, peer_id)
            if m is None:
                return {"error": f"unknown peer {peer_id}"}
            if m.process is not None and m.alive():
                m.process.terminate()
                try:
                    m.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    m.process.kill()
            del self.robots[peer_id]
        return {"peer_id": peer_id, "stopped": True}

    def reconfigure_cameras(self, peer_id: str, cameras: dict[str, Any] | None) -> dict[str, Any]:
        """Respawn a managed peer with a new camera mapping (None detaches all)."""
        bad = validate_cameras(cameras)
        if bad:
            return bad
        # A camera that CANNOT exist is refused before the despawn, not after the respawn: the arm
        # that is streaming right now is the thing at stake, and "index 7 of 3 cameras" is knowable
        # without opening anything (see indices_beyond_roster - count only, no probe, silent when
        # enumeration itself failed).
        roster = self._camera_names(refresh=True)
        impossible = indices_beyond_roster(cameras, len(roster))
        if impossible:
            listed = ", ".join(f"{n!r} -> index {i}" for n, i in sorted(impossible.items()))
            return {
                "error": (
                    f"{listed}: this machine enumerates {len(roster)} capture device(s), so that index "
                    f"cannot exist - {peer_id} was left running and untouched. Rescan the devices screen "
                    f"(a camera may have been unplugged, which renumbers the rest) and pick again."
                )
            }
        with self._lock:
            m = registry_entry(self.robots, peer_id)
            if m is None:
                return {
                    "error": (
                        f"unknown managed peer {peer_id} - only robots this dashboard "
                        f"spawned can be respawned with new cameras"
                    )
                }
            # Everything about the old spawn EXCEPT the cameras.
            robot_name, mode, port = m.robot_name, m.mode, m.port
        if mode not in SPAWNABLE_MODES:
            return {"error": f"{peer_id} is a {mode} job, not a respawnable robot"}
        # robot_id is the lerobot CALIBRATION identity - ManagedRobot does not
        # carry it, but the port profile does. Dropping it here would respawn
        # an arm that silently forgot its calibration.
        robot_id = None
        if port:
            profile = self.profile_for_port(port)
            if profile:
                robot_id = profile.get("robot_id")
        stopped = self.despawn(peer_id)
        if "error" in stopped:
            return stopped
        result = self.spawn(
            robot_name,
            mode,
            peer_id=peer_id,
            port=port,
            cameras=cameras,
            robot_id=robot_id,
            remember=True,
        )
        result["reconfigured"] = "error" not in result
        return result

    def shutdown(self) -> None:
        """Stop every managed child and the auto-spawn watcher."""
        for pid in list(self.robots):
            self.despawn(pid)
