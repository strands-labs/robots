#!/usr/bin/env python3
"""
Reachy Mini Tool — Full-featured control for Pollen Reachy Mini robots.

Dual transport: Zenoh (real-time joints/IMU/commands) + REST API (daemon, moves, camera).

Multi-robot ready: each robot identified by host + prefix.
Connection pooling: one Zenoh session per (host, port) pair, reused across calls.
"""

import atexit
import json
import logging
import math
import socket
import threading
import time
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Connection pool — ONE Zenoh session per robot, reused
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SESSIONS: Dict[str, Any] = {}  # "ip:port" -> zenoh session
_SESSIONS_LOCK = threading.Lock()


def _resolve_host(host: str) -> str:
    """Resolve hostname to IP address for Zenoh connection."""
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return host  # Already an IP or unresolvable


def _get_zenoh_session(host: str, zenoh_port: int = 7447):
    """Get or create a pooled Zenoh session for a specific robot."""
    ip = _resolve_host(host)
    key = f"{ip}:{zenoh_port}"

    with _SESSIONS_LOCK:
        if key in _SESSIONS:
            return _SESSIONS[key]

    try:
        import importlib

        zenoh = importlib.import_module("zenoh")

        try:
            config = zenoh.Config.from_json5(
                json.dumps(
                    {
                        "mode": "peer",
                        "connect": {"endpoints": [f"tcp/{ip}:{zenoh_port}"]},
                        "scouting": {"multicast": {"enabled": True}, "gossip": {"enabled": True}},
                    }
                )
            )
        except AttributeError:
            config = zenoh.Config()
            config.insert_json5("connect/endpoints", json.dumps([f"tcp/{ip}:{zenoh_port}"]))

        session = zenoh.open(config)

        with _SESSIONS_LOCK:
            _SESSIONS[key] = session

        logger.info(f"Zenoh session opened: {key}")
        return session

    except ImportError:
        logger.warning("eclipse-zenoh not installed")
        return None
    except Exception as e:
        logger.error(f"Zenoh connect failed ({key}): {e}")
        return None


def _close_all_sessions():
    """Close all pooled sessions on exit."""
    with _SESSIONS_LOCK:
        for key, session in _SESSIONS.items():
            try:
                session.close()
            except Exception:
                pass
        _SESSIONS.clear()


atexit.register(_close_all_sessions)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transport helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _api(host: str, port: int, path: str, method: str = "GET", data: dict = None) -> dict:
    """Call Reachy Mini daemon REST API."""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode() if data else None
    try:
        with urllib.request.urlopen(req, body, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "code": e.code}
    except Exception as e:
        return {"error": str(e)}


def _zenoh_put(host: str, prefix: str, topic: str, data: dict, zenoh_port: int = 7447):
    """Publish a command via Zenoh (uses pooled session)."""
    session = _get_zenoh_session(host, zenoh_port)
    if session is None:
        return {"error": "zenoh unavailable"}
    try:
        session.put(f"{prefix}/{topic}", json.dumps(data).encode())
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


def _zenoh_cmd(host: str, prefix: str, cmd: dict, zenoh_port: int = 7447):
    """Send a command to the robot via Zenoh."""
    return _zenoh_put(host, prefix, "command", cmd, zenoh_port)


def _zenoh_sub(host: str, prefix: str, topic: str, duration: float = 0.5, zenoh_port: int = 7447) -> list:
    """Subscribe briefly to collect messages (uses pooled session)."""
    session = _get_zenoh_session(host, zenoh_port)
    if session is None:
        return [("error", {"error": "zenoh unavailable"})]
    msgs = []
    try:

        def handler(sample):
            try:
                msgs.append((str(sample.key_expr), json.loads(sample.payload.to_bytes().decode())))
            except Exception:
                pass

        sub = session.declare_subscriber(f"{prefix}/{topic}", handler)
        time.sleep(duration)
        sub.undeclare()
        return msgs
    except Exception as e:
        return [("error", {"error": str(e)})]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pose math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _rpy_to_pose(
    pitch_deg: float, roll_deg: float, yaw_deg: float, x_mm: float = 0, y_mm: float = 0, z_mm: float = 0
) -> list:
    """Convert RPY (degrees) + XYZ (mm) to 4x4 pose matrix."""
    p, r, y = math.radians(pitch_deg), math.radians(roll_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x_mm / 1000],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y_mm / 1000],
        [-sp, cp * sr, cp * cr, z_mm / 1000],
        [0, 0, 0, 1],
    ]


def _identity_pose() -> list:
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@tool
def reachy_mini(
    action: str,
    host: str = "reachy-mini.local",
    prefix: str = "reachy_mini",
    api_port: int = 8000,
    zenoh_port: int = 7447,
    # Head (degrees / mm)
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    # Antennas (degrees)
    left_antenna: float = 0.0,
    right_antenna: float = 0.0,
    # Body
    body_yaw: float = 0.0,
    # Motion
    duration: float = 1.0,
    # Moves
    move_name: str = "",
    library: str = "emotions",
    # Recording / files
    save_path: str = "",
    # Motor
    motor_ids: str = "",
    # Audio
    sound_file: str = "",
    # Toggle
    enabled: bool = True,
) -> Dict[str, Any]:
    """Control Reachy Mini — head, body, antennas, camera, audio, IMU, moves, recording.

    Supports multiple robots: each identified by host + prefix.
    Connection pooled: Zenoh sessions reused across calls.

    Args:
        action: What to do:
            **Status & Daemon:**
            - "status": Daemon state, motors, control freq
            - "state": Full robot state (joints + head pose + IMU)
            - "daemon_start" / "daemon_stop" / "daemon_restart"

            **Movement (Zenoh):**
            - "look": Set head pose instantly (pitch/roll/yaw °, x/y/z mm)
            - "goto_pose": Smooth interpolated move (pitch/roll/yaw °, x/y/z mm, duration)
            - "antennas": Set antenna angles (degrees)
            - "body": Set body yaw (degrees)
            - "auto_body_yaw": Toggle automatic body yaw tracking

            **Gaze (REST):**
            - "look_at_world": Look at 3D coords (x/y/z in meters)
            - "look_at_image": Look at pixel coords (x=u, y=v)

            **Sensors (Zenoh):**
            - "joints" / "head_pose" / "imu"

            **Camera (REST):**
            - "camera": Capture JPEG (save_path optional)

            **Audio (REST):**
            - "play_sound" (sound_file) / "record_audio" (duration)

            **Motors (Zenoh):**
            - "enable_motors" / "disable_motors" (motor_ids optional)
            - "gravity_compensation" / "stiff"
            - "stop": Emergency stop (REST)

            **Move Libraries (REST, HuggingFace):**
            - "list_moves" (library: "emotions"|"dance")
            - "play_move" (move_name + library)

            **Recording (Zenoh):**
            - "start_recording" / "stop_recording" (save_path optional)

            **Expressions (Zenoh):**
            - "wake_up" / "sleep" (REST)
            - "nod" / "shake" / "happy"

        host: Robot hostname or IP. Each robot has its own. Default: reachy-mini.local
        prefix: Zenoh topic prefix. Each robot has its own. Default: reachy_mini
        api_port: REST API port. Default: 8000
        zenoh_port: Zenoh port. Default: 7447

    Examples:
        # Single robot (defaults)
        reachy_mini(action="status")
        reachy_mini(action="look", pitch=-15, yaw=30)

        # Multiple robots on the network
        reachy_mini(action="look", pitch=10, host="reachy-mini-1.local", prefix="reachy_mini_1")
        reachy_mini(action="nod", host="reachy-mini-2.local", prefix="reachy_mini_2")
        reachy_mini(action="happy", host="192.168.1.50", prefix="reachy_mini")

        # Custom ports (rare)
        reachy_mini(action="joints", host="10.0.0.5", zenoh_port=7448, prefix="robot_a")
    """
    try:
        # ── Status & Daemon ──────────────────────────────────
        if action == "status":
            r = _api(host, api_port, "/api/daemon/status")
            bs = r.get("backend_status") or {}
            cs = bs.get("control_loop_stats") or {}
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"🤖 Reachy Mini @ {host}\n  State: {r.get('state')}\n  Version: {r.get('version')}\n"
                        f"  IP: {r.get('wlan_ip')}\n  Motors: {bs.get('motor_control_mode')}\n"
                        f"  Freq: {cs.get('mean_control_loop_frequency', 0):.1f}Hz\n"
                        f"  Prefix: {prefix}  Zenoh: {zenoh_port}  API: {api_port}"
                    }
                ],
            }

        elif action == "state":
            joints = _zenoh_sub(host, prefix, "joint_positions", 0.3, zenoh_port)
            pose = _zenoh_sub(host, prefix, "head_pose", 0.3, zenoh_port)
            imu = _zenoh_sub(host, prefix, "imu_data", 0.3, zenoh_port)
            text = f"🤖 State @ {host} (prefix={prefix}):\n"
            if joints and joints[0][0] != "error":
                d = joints[-1][1]
                text += f"  Head: {[round(math.degrees(j), 1) for j in d.get('head_joint_positions', [])]}\n"
                text += f"  Antennas: {[round(math.degrees(j), 1) for j in d.get('antennas_joint_positions', [])]}\n"
            if pose and pose[0][0] != "error":
                text += f"  Pose: {str(pose[-1][1].get('head_pose', '?'))[:120]}\n"
            if imu and imu[0][0] != "error":
                d = imu[-1][1]
                text += f"  Accel: {[round(v, 2) for v in d.get('accelerometer', [])]}\n"
                text += f"  Temp: {d.get('temperature', '?')}°C\n"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "daemon_start":
            return {
                "status": "success",
                "content": [{"text": f"▶️ {_api(host, api_port, '/api/daemon/start?wake_up=true', 'POST')}"}],
            }
        elif action == "daemon_stop":
            return {
                "status": "success",
                "content": [{"text": f"⏹️ {_api(host, api_port, '/api/daemon/stop', 'POST')}"}],
            }
        elif action == "daemon_restart":
            return {
                "status": "success",
                "content": [{"text": f"🔄 {_api(host, api_port, '/api/daemon/restart', 'POST')}"}],
            }

        # ── Movement (Zenoh) ─────────────────────────────────
        elif action == "look":
            _zenoh_cmd(host, prefix, {"head_pose": _rpy_to_pose(pitch, roll, yaw, x, y, z)}, zenoh_port)
            return {
                "status": "success",
                "content": [{"text": f"👀 pitch={pitch}° roll={roll}° yaw={yaw}° xyz=({x},{y},{z})mm"}],
            }

        elif action == "goto_pose":
            payload = {
                "target": {
                    "x": x / 1000,
                    "y": y / 1000,
                    "z": z / 1000,
                    "roll": math.radians(roll),
                    "pitch": math.radians(pitch),
                    "yaw": math.radians(yaw),
                },
                "duration": duration,
            }
            r = _api(host, api_port, "/api/move/goto", "POST", payload)
            return {"status": "success", "content": [{"text": f"🎯 Goto (dur={duration}s): {r}"}]}

        elif action == "antennas":
            _zenoh_cmd(
                host,
                prefix,
                {"antennas_joint_positions": [math.radians(left_antenna), math.radians(right_antenna)]},
                zenoh_port,
            )
            return {"status": "success", "content": [{"text": f"📡 L={left_antenna}° R={right_antenna}°"}]}

        elif action == "body":
            _zenoh_cmd(host, prefix, {"body_yaw": math.radians(body_yaw)}, zenoh_port)
            return {"status": "success", "content": [{"text": f"🔄 Body={body_yaw}°"}]}

        elif action == "auto_body_yaw":
            _zenoh_cmd(host, prefix, {"automatic_body_yaw": enabled}, zenoh_port)
            return {"status": "success", "content": [{"text": f"🎪 Auto body yaw: {'ON' if enabled else 'OFF'}"}]}

        # ── Gaze (REST) ──────────────────────────────────────
        elif action == "look_at_world":
            r = _api(
                host,
                api_port,
                "/api/move/look_at_world",
                "POST",
                {"target": {"x": x, "y": y, "z": z}, "duration": duration, "perform_movement": True},
            )
            return {"status": "success", "content": [{"text": f"👁️ World ({x},{y},{z})m: {r}"}]}

        elif action == "look_at_image":
            r = _api(
                host,
                api_port,
                "/api/move/look_at_image",
                "POST",
                {"u": int(x), "v": int(y), "duration": duration, "perform_movement": True},
            )
            return {"status": "success", "content": [{"text": f"👁️ Pixel ({int(x)},{int(y)}): {r}"}]}

        # ── Sensors (Zenoh) ──────────────────────────────────
        elif action == "joints":
            msgs = _zenoh_sub(host, prefix, "joint_positions", 0.5, zenoh_port)
            if msgs and msgs[0][0] != "error":
                d = msgs[-1][1]
                head = d.get("head_joint_positions", [])
                ant = d.get("antennas_joint_positions", [])
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"🦾 Head: {[f'{math.degrees(j):.1f}°' for j in head]}  "
                                f"Ant: {[f'{math.degrees(j):.1f}°' for j in ant]}"
                            )
                        },
                        {"json": {"head": head, "antennas": ant}},
                    ],
                }
            return {"status": "error", "content": [{"text": "No joint data"}]}

        elif action == "head_pose":
            msgs = _zenoh_sub(host, prefix, "head_pose", 0.5, zenoh_port)
            if msgs and msgs[0][0] != "error":
                return {
                    "status": "success",
                    "content": [{"text": f"🎯 {json.dumps(msgs[-1][1].get('head_pose', []), indent=2)[:300]}"}],
                }
            return {"status": "error", "content": [{"text": "No pose data"}]}

        elif action == "imu":
            msgs = _zenoh_sub(host, prefix, "imu_data", 0.5, zenoh_port)
            if msgs and msgs[0][0] != "error":
                d = msgs[-1][1]
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"📐 Accel: {d.get('accelerometer')}\n  Gyro: {d.get('gyroscope')}\n"
                            f"  Quat: {d.get('quaternion')}\n  Temp: {d.get('temperature')}°C"
                        }
                    ],
                }
            return {"status": "error", "content": [{"text": "No IMU data"}]}

        # ── Camera (REST) ─────────────────────────────────────
        elif action == "camera":
            import urllib.request

            try:
                with urllib.request.urlopen(f"http://{host}:{api_port}/api/camera/snapshot", timeout=5) as resp:
                    fb = resp.read()
                    if save_path:
                        with open(save_path, "wb") as f:
                            f.write(fb)
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"📷 {len(fb)} bytes" + (f" → {save_path}" if save_path else "")},
                            {"image": {"format": "jpeg", "source": {"bytes": fb}}},
                        ],
                    }
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Camera: {e}"}]}

        # ── Audio (REST) ──────────────────────────────────────
        elif action == "play_sound":
            if not sound_file:
                return {"status": "error", "content": [{"text": "sound_file required"}]}
            return {
                "status": "success",
                "content": [
                    {"text": f"🔊 {_api(host, api_port, '/api/media/play_sound', 'POST', {'file': sound_file})}"}
                ],
            }

        elif action == "record_audio":
            return {
                "status": "success",
                "content": [
                    {"text": f"🎤 {_api(host, api_port, '/api/media/record', 'POST', {'duration': duration})}"}
                ],
            }

        # ── Motors (Zenoh) ────────────────────────────────────
        elif action == "enable_motors":
            ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
            _zenoh_cmd(host, prefix, {"torque": True, "ids": ids}, zenoh_port)
            return {"status": "success", "content": [{"text": f"⚡ Enabled{f' ({motor_ids})' if motor_ids else ''}"}]}

        elif action == "disable_motors":
            ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
            _zenoh_cmd(host, prefix, {"torque": False, "ids": ids}, zenoh_port)
            return {"status": "success", "content": [{"text": f"💤 Disabled{f' ({motor_ids})' if motor_ids else ''}"}]}

        elif action == "gravity_compensation":
            _zenoh_cmd(host, prefix, {"gravity_compensation": True}, zenoh_port)
            return {"status": "success", "content": [{"text": "🎯 Gravity comp ON (compliant)"}]}

        elif action == "stiff":
            _zenoh_cmd(host, prefix, {"gravity_compensation": False}, zenoh_port)
            return {"status": "success", "content": [{"text": "🔒 Stiff"}]}

        elif action == "stop":
            return {"status": "success", "content": [{"text": f"🛑 {_api(host, api_port, '/api/move/stop', 'POST')}"}]}

        # ── Move Libraries (REST) ─────────────────────────────
        elif action == "list_moves":
            ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
            r = _api(host, api_port, f"/api/move/recorded-move-datasets/list/{ds}")
            if isinstance(r, list):
                return {
                    "status": "success",
                    "content": [{"text": f"📚 {library} ({len(r)}):\n" + "\n".join(f"  • {m}" for m in r)}],
                }
            return {"status": "success", "content": [{"text": f"📚 {r}"}]}

        elif action == "play_move":
            if not move_name:
                return {"status": "error", "content": [{"text": "move_name required"}]}
            ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
            r = _api(host, api_port, f"/api/move/play/recorded-move-dataset/{ds}/{move_name}", "POST")
            return {"status": "success", "content": [{"text": f"🎭 '{move_name}' ({library}): {r}"}]}

        # ── Recording (Zenoh) ─────────────────────────────────
        elif action == "start_recording":
            _zenoh_cmd(host, prefix, {"start_recording": True}, zenoh_port)
            return {"status": "success", "content": [{"text": "📹 Recording started"}]}

        elif action == "stop_recording":
            _zenoh_cmd(host, prefix, {"stop_recording": True}, zenoh_port)
            msgs = _zenoh_sub(host, prefix, "recorded_data", 3.0, zenoh_port)
            frames = 0
            if msgs and msgs[0][0] != "error":
                data = msgs[-1][1]
                frames = len(data) if isinstance(data, list) else 0
                if save_path and data:
                    with open(save_path, "w") as f:
                        json.dump(data, f, indent=2)
            return {
                "status": "success",
                "content": [{"text": f"📹 Stopped: {frames} frames" + (f" → {save_path}" if save_path else "")}],
            }

        # ── Expressions ───────────────────────────────────────
        elif action == "wake_up":
            return {
                "status": "success",
                "content": [{"text": f"☀️ {_api(host, api_port, '/api/move/play/wake_up', 'POST')}"}],
            }

        elif action == "sleep":
            return {
                "status": "success",
                "content": [{"text": f"😴 {_api(host, api_port, '/api/move/play/goto_sleep', 'POST')}"}],
            }

        elif action == "nod":
            for _ in range(3):
                _zenoh_cmd(host, prefix, {"head_pose": _rpy_to_pose(15, 0, 0)}, zenoh_port)
                time.sleep(0.25)
                _zenoh_cmd(host, prefix, {"head_pose": _rpy_to_pose(-10, 0, 0)}, zenoh_port)
                time.sleep(0.25)
            _zenoh_cmd(host, prefix, {"head_pose": _identity_pose()}, zenoh_port)
            return {"status": "success", "content": [{"text": "🤖 *nods*"}]}

        elif action == "shake":
            for _ in range(3):
                _zenoh_cmd(host, prefix, {"head_pose": _rpy_to_pose(0, 0, 25)}, zenoh_port)
                time.sleep(0.2)
                _zenoh_cmd(host, prefix, {"head_pose": _rpy_to_pose(0, 0, -25)}, zenoh_port)
                time.sleep(0.2)
            _zenoh_cmd(host, prefix, {"head_pose": _identity_pose()}, zenoh_port)
            return {"status": "success", "content": [{"text": "🤖 *shakes head*"}]}

        elif action == "happy":
            for _ in range(4):
                _zenoh_cmd(
                    host, prefix, {"antennas_joint_positions": [math.radians(60), math.radians(-60)]}, zenoh_port
                )
                time.sleep(0.2)
                _zenoh_cmd(
                    host, prefix, {"antennas_joint_positions": [math.radians(-60), math.radians(60)]}, zenoh_port
                )
                time.sleep(0.2)
            _zenoh_cmd(host, prefix, {"antennas_joint_positions": [0, 0]}, zenoh_port)
            return {"status": "success", "content": [{"text": "🤖 *happy wiggle*"}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown: {action}. Valid: status, state, daemon_start/stop/restart, "
                        f"look, goto_pose, antennas, body, auto_body_yaw, look_at_world, look_at_image, "
                        f"joints, head_pose, imu, camera, play_sound, record_audio, "
                        f"enable_motors, disable_motors, gravity_compensation, stiff, stop, "
                        f"list_moves, play_move, start_recording, stop_recording, "
                        f"wake_up, sleep, nod, shake, happy"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"reachy_mini @ {host}: {e}")
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
