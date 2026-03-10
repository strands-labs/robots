#!/usr/bin/env python3
"""
Native Teleoperator Tool — Uses LeRobot Teleoperator ABC directly.

Uses LeRobot's Teleoperator ABC directly for in-process teleop and recording.

Unlike lerobot_teleoperate.py (subprocess wrapper), this tool uses
LeRobot's Teleoperator and Robot classes directly in-process.
Supports teleop, teleop+record, and calibration in one tool.

Usage via Strands Agent:
    teleoperator(action="teleop",
                 robot_type="so100_follower", robot_port="/dev/ttyACM0",
                 teleop_type="so100_leader", teleop_port="/dev/ttyACM1")

    teleoperator(action="record",
                 robot_type="so100_follower", robot_port="/dev/ttyACM0",
                 teleop_type="so100_leader", teleop_port="/dev/ttyACM1",
                 repo_id="user/my_data", task="pick up cube", num_episodes=10)

    teleoperator(action="stop")
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Global session state (one active session at a time)
_ACTIVE_SESSION = {
    "running": False,
    "thread": None,
    "robot": None,
    "teleop": None,
    "record_session": None,
    "mode": None,
    "stats": {},
}


def _make_robot(robot_type: str, robot_port: str = None, robot_id: str = None, cameras: Dict = None, **kwargs):
    """Create a LeRobot Robot from type string using native factory."""
    # Dynamically resolve config class
    import importlib
    import pkgutil

    import lerobot.robots as lr_robots
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config

    robot_type_lower = robot_type.lower().replace("-", "_")

    # Scan lerobot.robots for matching config
    ConfigClass = None
    for importer, modname, ispkg in pkgutil.iter_modules(lr_robots.__path__):
        if modname in ("config", "robot", "utils"):
            continue
        try:
            mod = importlib.import_module(f"lerobot.robots.{modname}")
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and attr_name.endswith("Config")
                    and issubclass(obj, RobotConfig)
                    and obj is not RobotConfig
                ):
                    if attr_name.lower().replace("_", "") == robot_type_lower.replace("_", "") + "config":
                        ConfigClass = obj
                        break
            if ConfigClass:
                break
        except Exception:
            continue

    if not ConfigClass:
        raise ValueError(f"Robot config not found for '{robot_type}'. Check lerobot.robots.")

    # Build camera configs
    camera_configs = {}
    if cameras:
        for name, cfg in cameras.items():
            camera_configs[name] = OpenCVCameraConfig(
                index_or_path=cfg.get("index_or_path", 0),
                fps=cfg.get("fps", 30),
                width=cfg.get("width", 640),
                height=cfg.get("height", 480),
            )

    # Build config
    config_data = {"cameras": camera_configs}
    if robot_id:
        config_data["id"] = robot_id
    if robot_port:
        config_data["port"] = robot_port

    # Forward any extra kwargs
    if hasattr(ConfigClass, "__dataclass_fields__"):
        for f in ConfigClass.__dataclass_fields__:
            if f in kwargs and f not in config_data:
                config_data[f] = kwargs[f]

    config = ConfigClass(**config_data)
    return make_robot_from_config(config)


def _make_teleop(teleop_type: str, teleop_port: str = None, teleop_id: str = None, **kwargs):
    """Create a LeRobot Teleoperator from type string using native factory."""
    import importlib
    import pkgutil

    import lerobot.teleoperators as lr_teleops
    from lerobot.teleoperators.config import TeleoperatorConfig

    teleop_type_lower = teleop_type.lower().replace("-", "_")

    ConfigClass = None
    for importer, modname, ispkg in pkgutil.iter_modules(lr_teleops.__path__):
        if modname in ("config", "teleoperator", "utils"):
            continue
        try:
            mod = importlib.import_module(f"lerobot.teleoperators.{modname}")
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and attr_name.endswith("Config")
                    and issubclass(obj, TeleoperatorConfig)
                    and obj is not TeleoperatorConfig
                ):
                    if attr_name.lower().replace("_", "") == teleop_type_lower.replace("_", "") + "config":
                        ConfigClass = obj
                        break
            if ConfigClass:
                break
        except Exception:
            continue

    if not ConfigClass:
        raise ValueError(f"Teleoperator config not found for '{teleop_type}'.")

    config_data = {}
    if teleop_id:
        config_data["id"] = teleop_id
    if teleop_port:
        config_data["port"] = teleop_port

    if hasattr(ConfigClass, "__dataclass_fields__"):
        for f in ConfigClass.__dataclass_fields__:
            if f in kwargs and f not in config_data:
                config_data[f] = kwargs[f]

    from lerobot.teleoperators.utils import make_teleoperator_from_config

    config = ConfigClass(**config_data)
    return make_teleoperator_from_config(config)


def _resolve_teleoperator(device_spec: str):
    """Resolve a teleoperator from a device specification string.

    Used by lerobot_dataset for recording with a teleop device.

    Args:
        device_spec: Either a type name (e.g. "so100_leader") or
                    "type:port" (e.g. "so100_leader:/dev/ttyUSB1").

    Returns:
        A Teleoperator instance (not yet connected).
    """
    parts = device_spec.split(":", 1)
    teleop_type = parts[0]
    teleop_port = parts[1] if len(parts) > 1 else None
    return _make_teleop(teleop_type, teleop_port)


def _run_teleop_loop(robot, teleop, fps: int = 60, duration: float = None):
    """Run teleop control loop in background thread."""
    frame_interval = 1.0 / fps
    start = time.time()
    step = 0

    while _ACTIVE_SESSION["running"]:
        step_start = time.time()

        if duration and (time.time() - start) > duration:
            break

        try:
            action = teleop.get_action()
            robot.send_action(action)
            step += 1
            _ACTIVE_SESSION["stats"]["steps"] = step
            _ACTIVE_SESSION["stats"]["duration"] = time.time() - start
        except Exception as e:
            logger.error(f"Teleop step error: {e}")
            break

        elapsed = time.time() - step_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    _ACTIVE_SESSION["running"] = False
    logger.info(f"Teleop loop ended: {step} steps")


def _run_record_loop(robot, teleop, record_session, num_episodes: int):
    """Run teleop+record loop in background thread."""
    from strands_robots.record import RecordMode

    for ep in range(num_episodes):
        if not _ACTIVE_SESSION["running"]:
            break

        logger.info(f"Recording episode {ep + 1}/{num_episodes}")
        _ACTIVE_SESSION["stats"]["current_episode"] = ep + 1

        record_session.record_episode(mode=RecordMode.TELEOP)
        _ACTIVE_SESSION["stats"]["episodes_completed"] = ep + 1
        _ACTIVE_SESSION["stats"]["total_frames"] = sum(e.frames for e in record_session._episodes if not e.discarded)

    # Finalize
    result = record_session.save_and_push()
    _ACTIVE_SESSION["stats"]["final_result"] = result
    _ACTIVE_SESSION["running"] = False
    logger.info(f"Recording complete: {result}")


@tool
def teleoperator(
    action: str = "status",
    # Robot config
    robot_type: str = "so100_follower",
    robot_port: Optional[str] = None,
    robot_id: Optional[str] = None,
    cameras: Optional[Dict[str, Any]] = None,
    # Teleop config
    teleop_type: Optional[str] = "so100_leader",
    teleop_port: Optional[str] = None,
    teleop_id: Optional[str] = None,
    # Recording config
    repo_id: Optional[str] = None,
    task: str = "",
    num_episodes: int = 10,
    episode_time_s: float = 60.0,
    fps: int = 30,
    push_to_hub: bool = False,
    vcodec: str = "libsvtav1",
    # Control
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Native LeRobot teleoperator — uses Teleoperator ABC directly (no subprocess).

    Actions:
        teleop: Start teleoperation (leader → follower)
        record: Start teleop + recording to LeRobotDataset
        stop: Stop active session
        status: Get session status
        discard: Discard current episode (during record)

    Uses LeRobot's native Robot + Teleoperator classes in-process.
    For recording, uses strands_robots.record.RecordSession which writes
    directly to LeRobotDataset (parquet + mp4).

    Args:
        action: teleop, record, stop, status, discard
        robot_type: LeRobot robot type (so100_follower, koch_follower, etc.)
        robot_port: Serial port for robot
        robot_id: Robot instance ID
        cameras: Camera config dict
        teleop_type: Teleoperator type (so100_leader, gamepad, phone, etc.)
        teleop_port: Serial port for teleop device
        teleop_id: Teleop instance ID
        repo_id: HuggingFace dataset repo ID (for record action)
        task: Task description for recording
        num_episodes: Number of episodes to record
        episode_time_s: Episode duration in seconds
        fps: Control/recording frequency
        push_to_hub: Push dataset to HuggingFace Hub
        vcodec: Video codec for recording
        duration: Max teleop duration in seconds (None = unlimited)

    Returns:
        Dict with status and session info
    """

    try:
        if action == "teleop":
            if _ACTIVE_SESSION["running"]:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Session already running. Use action='stop' first."}],
                }

            # Create robot and teleop
            robot = _make_robot(robot_type, robot_port, robot_id, cameras or {})
            teleop_dev = _make_teleop(teleop_type, teleop_port, teleop_id)

            # Connect
            robot.connect()
            teleop_dev.connect()

            # Start background loop
            _ACTIVE_SESSION.update(
                {
                    "running": True,
                    "robot": robot,
                    "teleop": teleop_dev,
                    "mode": "teleop",
                    "stats": {"steps": 0, "duration": 0, "start_time": time.time()},
                }
            )

            thread = threading.Thread(
                target=_run_teleop_loop,
                args=(robot, teleop_dev, fps, duration),
                daemon=True,
            )
            _ACTIVE_SESSION["thread"] = thread
            thread.start()

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🕹️ Teleoperation started\n"
                            f"🤖 Robot: {robot_type} ({robot_port})\n"
                            f"🎮 Teleop: {teleop_type} ({teleop_port})\n"
                            f"⏱️ FPS: {fps}\n"
                            f"💡 Use action='stop' to end, action='status' to check"
                        )
                    }
                ],
            }

        elif action == "record":
            if _ACTIVE_SESSION["running"]:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Session already running. Use action='stop' first."}],
                }

            if not repo_id:
                repo_id = f"local/teleop_{int(time.time())}"

            # Create robot and teleop
            robot = _make_robot(robot_type, robot_port, robot_id, cameras or {})
            teleop_dev = _make_teleop(teleop_type, teleop_port, teleop_id)

            # Create record session
            from strands_robots.record import RecordSession

            record_session = RecordSession(
                robot=robot,
                teleop=teleop_dev,
                repo_id=repo_id,
                task=task,
                fps=fps,
                push_to_hub=push_to_hub,
                num_episodes=num_episodes,
                episode_time_s=episode_time_s,
                vcodec=vcodec,
            )
            record_session.connect()

            _ACTIVE_SESSION.update(
                {
                    "running": True,
                    "robot": robot,
                    "teleop": teleop_dev,
                    "record_session": record_session,
                    "mode": "record",
                    "stats": {
                        "current_episode": 0,
                        "episodes_completed": 0,
                        "total_frames": 0,
                        "start_time": time.time(),
                    },
                }
            )

            thread = threading.Thread(
                target=_run_record_loop,
                args=(robot, teleop_dev, record_session, num_episodes),
                daemon=True,
            )
            _ACTIVE_SESSION["thread"] = thread
            thread.start()

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🔴 Recording started\n"
                            f"🤖 Robot: {robot_type} ({robot_port})\n"
                            f"🎮 Teleop: {teleop_type} ({teleop_port})\n"
                            f"📁 Dataset: {repo_id}\n"
                            f"🎯 Task: '{task}'\n"
                            f"📊 Episodes: 0/{num_episodes} @ {fps}fps\n"
                            f"💡 Use action='stop' to end, action='discard' to drop episode"
                        )
                    }
                ],
            }

        elif action == "stop":
            if not _ACTIVE_SESSION["running"]:
                return {"status": "success", "content": [{"text": "💤 No active session to stop"}]}

            _ACTIVE_SESSION["running"] = False

            # Wait for thread to finish
            if _ACTIVE_SESSION["thread"]:
                _ACTIVE_SESSION["thread"].join(timeout=5)

            # Stop recording if active
            final_result = None
            if _ACTIVE_SESSION["record_session"]:
                try:
                    _ACTIVE_SESSION["record_session"].stop()
                    final_result = _ACTIVE_SESSION["record_session"].save_and_push()
                except Exception as e:
                    logger.error(f"Recording finalization error: {e}")

            # Disconnect
            try:
                if _ACTIVE_SESSION["teleop"]:
                    _ACTIVE_SESSION["teleop"].disconnect()
                if _ACTIVE_SESSION["robot"]:
                    _ACTIVE_SESSION["robot"].disconnect()
            except Exception as e:
                logger.error(f"Disconnect error: {e}")

            stats = _ACTIVE_SESSION["stats"]
            mode = _ACTIVE_SESSION["mode"]
            _ACTIVE_SESSION.update(
                {
                    "running": False,
                    "thread": None,
                    "robot": None,
                    "teleop": None,
                    "record_session": None,
                    "mode": None,
                }
            )

            text = f"🛑 Session stopped ({mode})\n"
            if mode == "record" and final_result:
                text += (
                    f"📊 Episodes: {final_result.get('episodes', 0)}\n"
                    f"🎞️ Total frames: {final_result.get('total_frames', 0)}\n"
                    f"📁 Saved to: {final_result.get('root', 'unknown')}\n"
                )
            elif mode == "teleop":
                text += f"📊 Steps: {stats.get('steps', 0)}\n" f"⏱️ Duration: {stats.get('duration', 0):.1f}s\n"

            return {"status": "success", "content": [{"text": text}]}

        elif action == "discard":
            if not _ACTIVE_SESSION["running"] or not _ACTIVE_SESSION["record_session"]:
                return {"status": "error", "content": [{"text": "❌ No active recording session"}]}
            _ACTIVE_SESSION["record_session"].discard_episode()
            return {"status": "success", "content": [{"text": "🗑️ Current episode discarded"}]}

        elif action == "status":
            if not _ACTIVE_SESSION["running"]:
                return {"status": "success", "content": [{"text": "💤 No active session"}]}

            mode = _ACTIVE_SESSION["mode"]
            stats = _ACTIVE_SESSION["stats"]
            uptime = time.time() - stats.get("start_time", time.time())

            text = f"📊 Active session: {mode}\n⏱️ Uptime: {uptime:.0f}s\n"
            if mode == "teleop":
                text += f"📊 Steps: {stats.get('steps', 0)}\n"
            elif mode == "record":
                text += (
                    f"🔴 Episode: {stats.get('current_episode', 0)}/{num_episodes}\n"
                    f"📊 Completed: {stats.get('episodes_completed', 0)}\n"
                    f"🎞️ Frames: {stats.get('total_frames', 0)}\n"
                )

            return {"status": "success", "content": [{"text": text}]}

        else:
            return {
                "status": "error",
                "content": [{"text": f"❌ Unknown action: {action}. Valid: teleop, record, stop, status, discard"}],
            }

    except Exception as e:
        logger.error(f"Teleoperator tool error: {e}")
        return {"status": "error", "content": [{"text": f"❌ Error: {e}"}]}
