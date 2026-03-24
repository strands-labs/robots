#!/usr/bin/env python3
"""
LeRobot teleoperation tool with recording capabilities for robot training data collection.

This tool integrates teleoperation and recording functionality from lerobot, allowing users to:
- Control robots through teleoperation devices
- Record demonstrations for training machine learning models
- Replay recorded episodes
- Manage multiple teleoperation sessions
"""

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
from strands import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage directory
SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


class SessionManager:
    """Manage teleoperation sessions with persistence."""

    def __init__(self):
        self.sessions_file = SESSION_DIR / "active_sessions.json"

    def _load_sessions(self) -> dict[str, Any]:
        """Load active sessions from disk."""
        if not self.sessions_file.exists():
            return {}

        try:
            with open(self.sessions_file) as f:
                sessions = json.load(f)

            # Check if processes are still running and clean up dead sessions
            active_sessions = {}
            for name, info in sessions.items():
                pid = info.get("pid")
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            active_sessions[name] = info
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            # Update sessions file with only active sessions
            if len(active_sessions) != len(sessions):
                self._save_sessions(active_sessions)

            return active_sessions

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sessions: {e}")
            return {}

    def _save_sessions(self, sessions: dict[str, Any]):
        """Save sessions to disk."""
        try:
            with open(self.sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except OSError as e:
            logger.error(f"Error saving sessions: {e}")

    def add_session(self, name: str, info: dict[str, Any]):
        """Add a new session."""
        sessions = self._load_sessions()
        sessions[name] = info
        self._save_sessions(sessions)

    def remove_session(self, name: str):
        """Remove a session."""
        sessions = self._load_sessions()
        if name in sessions:
            del sessions[name]
            self._save_sessions(sessions)

    def get_session(self, name: str) -> dict[str, Any] | None:
        """Get session info."""
        sessions = self._load_sessions()
        return sessions.get(name)

    def list_sessions(self) -> dict[str, Any]:
        """List all active sessions."""
        return self._load_sessions()


def build_lerobot_command(
    action: str,
    robot_type: str,
    robot_port: str | None = None,
    robot_id: str | None = None,
    robot_cameras: dict[str, Any] | None = None,
    robot_left_arm_port: str | None = None,
    robot_right_arm_port: str | None = None,
    teleop_type: str | None = None,
    teleop_port: str | None = None,
    teleop_id: str | None = None,
    teleop_left_arm_port: str | None = None,
    teleop_right_arm_port: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_single_task: str | None = None,
    dataset_num_episodes: int = 50,
    dataset_fps: int = 30,
    dataset_episode_time_s: int = 60,
    dataset_reset_time_s: int = 60,
    dataset_root: str | None = None,
    dataset_video: bool = True,
    dataset_push_to_hub: bool = False,
    replay_episode: int = 0,
    display_data: bool = False,
    fps: int = 60,
    teleop_time_s: float | None = None,
    play_sounds: bool = True,
    **kwargs,
) -> list[str]:
    """Build the lerobot command based on action and parameters."""

    if action == "replay":
        # Build replay command
        if not dataset_repo_id:
            raise ValueError("dataset_repo_id is required for replay action")
        cmd = [
            "python",
            "-m",
            "lerobot.scripts.lerobot_replay",
            "--robot-path",
            robot_type,
            "--policy-path",
            dataset_repo_id,
            "--episode",
            str(replay_episode),
        ]

        if robot_port:
            cmd.extend(["--robot-port", robot_port])
        if robot_left_arm_port:
            cmd.extend(["--robot-left-arm-port", robot_left_arm_port])
        if robot_right_arm_port:
            cmd.extend(["--robot-right-arm-port", robot_right_arm_port])
        if display_data:
            cmd.append("--display-data")

        return cmd

    elif action == "start":
        # Determine the base command based on whether we're recording
        if dataset_repo_id:
            # Recording mode
            cmd = [
                "python",
                "-m",
                "lerobot.scripts.lerobot_record",
                "--robot-path",
                robot_type,
                "--robot-port",
                robot_port or "/dev/ttyACM0",
                "--fps",
                str(fps),
                "--repo-id",
                dataset_repo_id,
                "--num-episodes",
                str(dataset_num_episodes),
                "--episode-time-s",
                str(dataset_episode_time_s),
                "--reset-time-s",
                str(dataset_reset_time_s),
            ]

            if dataset_single_task:
                cmd.extend(["--single-task", dataset_single_task])
            if dataset_root:
                cmd.extend(["--root", dataset_root])
            if dataset_push_to_hub:
                cmd.append("--push-to-hub")
            if not dataset_video:
                cmd.append("--no-video")
        else:
            # Simple teleoperation mode
            cmd = ["python", "-m", "lerobot.scripts.lerobot_teleoperate", "--robot.type", robot_type, "--fps", str(fps)]

            if teleop_time_s:
                cmd.extend(["--teleop_time_s", str(teleop_time_s)])

        # Add robot configuration
        if robot_port:
            cmd.extend(["--robot.port", robot_port])
        if robot_id:
            cmd.extend(["--robot.id", robot_id])
        if robot_left_arm_port:
            cmd.extend(["--robot.left_arm_port", robot_left_arm_port])
        if robot_right_arm_port:
            cmd.extend(["--robot.right_arm_port", robot_right_arm_port])

        # Add teleoperator configuration
        if teleop_type:
            cmd.extend(["--teleop.type", teleop_type])
        if teleop_id:
            cmd.extend(["--teleop.id", teleop_id])
        if teleop_port:
            cmd.extend(["--teleop.port", teleop_port])
        if teleop_left_arm_port:
            cmd.extend(["--teleop.left_arm_port", teleop_left_arm_port])
        if teleop_right_arm_port:
            cmd.extend(["--teleop.right_arm_port", teleop_right_arm_port])

        # Add camera configuration
        if robot_cameras:
            for cam_name, cam_config in robot_cameras.items():
                cam_type = cam_config.get("type", "opencv")
                cam_path = str(cam_config.get("index_or_path", 0))
                fps_val = cam_config.get("fps", 30)
                width = cam_config.get("width", 640)
                height = cam_config.get("height", 480)

                cmd.extend(["--camera-config", f"{cam_name}={cam_type}:{cam_path}:{fps_val}:{width}x{height}"])

        # Add common options
        if display_data:
            cmd.extend(["--display_data", "true"])
        # Note: play_sounds option may not exist in lerobot_teleoperate

        return cmd

    else:
        raise ValueError(f"Unknown action: {action}")


@tool
def lerobot_teleoperate(
    action: str = "start",
    session_name: str | None = None,
    background: bool = True,
    # Robot configuration
    robot_type: str = "so101_follower",
    robot_port: str | None = "/dev/ttyACM0",
    robot_id: str | None = None,
    robot_cameras: dict[str, Any] | None = None,
    robot_left_arm_port: str | None = None,
    robot_right_arm_port: str | None = None,
    # Teleoperator configuration
    teleop_type: str | None = "so101_leader",
    teleop_port: str | None = "/dev/ttyACM1",
    teleop_id: str | None = None,
    teleop_left_arm_port: str | None = None,
    teleop_right_arm_port: str | None = None,
    # Dataset configuration (for recording)
    dataset_repo_id: str | None = None,
    dataset_single_task: str | None = None,
    dataset_num_episodes: int = 50,
    dataset_fps: int = 30,
    dataset_episode_time_s: int = 60,
    dataset_reset_time_s: int = 60,
    dataset_root: str | None = None,
    dataset_video: bool = True,
    dataset_push_to_hub: bool = False,
    # Replay configuration
    replay_episode: int = 0,
    # Common options
    display_data: bool = False,
    fps: int = 60,
    teleop_time_s: float | None = None,
    play_sounds: bool = True,
    auto_accept_calibration: bool = True,
) -> dict[str, Any]:
    """
    Advanced LeRobot teleoperation tool with recording capabilities for robot training data collection.

    This tool integrates teleoperation and recording functionality from lerobot, allowing users to:
    - Control robots through teleoperation devices
    - Record demonstrations for training machine learning models
    - Replay recorded episodes
    - Manage multiple teleoperation sessions

    Features:
    - Session Management: Start, stop, list, and monitor teleoperation sessions
    - Background Execution: Run teleoperation in background with logging
    - Recording Mode: Automatically record demonstrations when dataset configuration is provided
    - Multi-Robot Support: Support for single-arm and bimanual robots
    - Camera Integration: Multi-camera support with configurable settings
    - Replay Capability: Replay recorded episodes on physical robots
    - Safety Features: Graceful shutdown and process management

    Actions:
        start: Start a new teleoperation session
            - Simple teleoperation (just teleop_type specified)
            - Recording mode (dataset_repo_id specified)
            - Background or foreground execution

        stop: Stop a running session by name

        list: List all active teleoperation sessions

        status: Get detailed status of a specific session
            - Process information, uptime, logs

        replay: Replay a recorded episode on the robot
            - Requires dataset_repo_id and replay_episode

    Robot Types:
        - so101_follower: Single-arm SO-101 robot
        - bi_so100_follower: Dual-arm SO-100 robot
        - koch_follower: Koch robot
        - hope_jr: HOPE Jr robot

    Teleoperator Types:
        - so101_leader: SO-101 leader device
        - bi_so100_leader: Dual SO-100 leader devices
        - koch_leader: Koch leader device
        - gamepad: Gamepad controller
        - homunculus: Homunculus teleoperator

    Camera Configuration Format:
        {
            "camera_name": {
                "type": "opencv",  # or "realsense"
                "index_or_path": 0,  # camera index or device path
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }

    Examples:
        # Simple teleoperation
        lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM1"
        )

        # Recording demonstrations
        lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM1",
            dataset_repo_id="my_user/cube_picking",
            dataset_single_task="Pick up the red cube and place it in the box",
            dataset_num_episodes=25,
            robot_cameras={
                "front": {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30}
            }
        )

        # Bimanual robot teleoperation
        lerobot_teleoperate(
            action="start",
            robot_type="bi_so100_follower",
            robot_left_arm_port="/dev/ttyACM0",
            robot_right_arm_port="/dev/ttyACM1",
            teleop_type="bi_so100_leader",
            teleop_left_arm_port="/dev/ttyACM2",
            teleop_right_arm_port="/dev/ttyACM3"
        )

        # List sessions
        lerobot_teleoperate(action="list")

        # Stop session
        lerobot_teleoperate(action="stop", session_name="teleop_1234567890")

        # Replay episode
        lerobot_teleoperate(
            action="replay",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            dataset_repo_id="my_user/cube_picking",
            replay_episode=5
        )

    Calibration Management:
        For calibration management (list, view, backup, etc.), use the separate
        lerobot_calibrate tool:

        # List available calibrations
        lerobot_calibrate(action="list")

        # View specific calibration
        lerobot_calibrate(action="view", device_type="robots",
                         device_model="so101_follower", device_id="orange_arm")

    Args:
        action: Action to perform (start, stop, list, status, replay)
        session_name: Session identifier (auto-generated for start, required for stop/status)
        background: Run session in background with logging (default: True)

        robot_type: Robot type identifier
        robot_port: Serial port for single-arm robots
        robot_id: Robot instance identifier
        robot_cameras: Camera configuration dictionary
        robot_left_arm_port: Left arm port for bimanual robots
        robot_right_arm_port: Right arm port for bimanual robots

        teleop_type: Teleoperator type identifier
        teleop_port: Serial port for single-arm teleoperators
        teleop_id: Teleoperator instance identifier
        teleop_left_arm_port: Left arm port for bimanual teleoperators
        teleop_right_arm_port: Right arm port for bimanual teleoperators

        dataset_repo_id: HuggingFace dataset repository ID (enables recording mode)
        dataset_single_task: Task description for recordings
        dataset_num_episodes: Number of episodes to record
        dataset_fps: Recording frame rate
        dataset_episode_time_s: Episode duration in seconds
        dataset_reset_time_s: Reset time between episodes
        dataset_root: Local dataset storage directory
        dataset_video: Enable video encoding
        dataset_push_to_hub: Upload dataset to HuggingFace Hub

        replay_episode: Episode number to replay

        display_data: Show live camera feeds and telemetry
        fps: Teleoperation control loop frequency
        teleop_time_s: Session duration limit
        play_sounds: Enable audio feedback

    Returns:
        Dict with operation status and results:
        {
            "status": "success|error",
            "content": [{"text": "Description of operation"}],
            "session_name": "session_id",  # for start action
            "pid": 12345,  # process ID for background sessions
            "command": "full_command_executed",
            "log_file": "/tmp/session.log",  # for background sessions
            "sessions": {...},  # for list action
            "uptime": 123.45,  # session uptime in seconds
            "is_running": true  # for status action
        }
    """

    session_manager = SessionManager()

    try:
        if action == "start":
            # Generate session name if not provided
            if not session_name:
                session_name = f"teleop_{int(time.time())}"

            # Check if session already exists
            if session_manager.get_session(session_name):
                return {"status": "error", "content": [{"text": f"❌ Session '{session_name}' already exists"}]}

            # Build command
            try:
                cmd = build_lerobot_command(
                    action=action,
                    robot_type=robot_type,
                    robot_port=robot_port,
                    robot_id=robot_id,
                    robot_cameras=robot_cameras,
                    robot_left_arm_port=robot_left_arm_port,
                    robot_right_arm_port=robot_right_arm_port,
                    teleop_type=teleop_type,
                    teleop_port=teleop_port,
                    teleop_id=teleop_id,
                    teleop_left_arm_port=teleop_left_arm_port,
                    teleop_right_arm_port=teleop_right_arm_port,
                    dataset_repo_id=dataset_repo_id,
                    dataset_single_task=dataset_single_task,
                    dataset_num_episodes=dataset_num_episodes,
                    dataset_fps=dataset_fps,
                    dataset_episode_time_s=dataset_episode_time_s,
                    dataset_reset_time_s=dataset_reset_time_s,
                    dataset_root=dataset_root,
                    dataset_video=dataset_video,
                    dataset_push_to_hub=dataset_push_to_hub,
                    replay_episode=replay_episode,
                    display_data=display_data,
                    fps=fps,
                    teleop_time_s=teleop_time_s,
                    play_sounds=play_sounds,
                )
            except Exception as e:
                return {"status": "error", "content": [{"text": f"❌ Command build failed: {str(e)}"}]}

            if background:
                # Start in background
                log_file = SESSION_DIR / f"{session_name}.log"

                if auto_accept_calibration:
                    # Start process with stdin for automatic calibration acceptance
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            text=True,
                            start_new_session=True,
                        )

                    # Send automatic "ENTER" to accept existing calibrations
                    # We'll do this in a separate thread to not block
                    import threading

                    def auto_respond():
                        try:
                            time.sleep(2)  # Allow process to initialize before writing to stdin
                            proc.stdin.write("\n")  # Send ENTER
                            proc.stdin.flush()
                            time.sleep(1)
                            proc.stdin.write("\n")  # Send another ENTER (for robot calibration)
                            proc.stdin.flush()
                            proc.stdin.close()  # Close stdin after sending responses
                        except Exception:
                            pass  # Ignore errors if process has already finished

                    threading.Thread(target=auto_respond, daemon=True).start()
                else:
                    # Start normally without stdin handling
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(
                            cmd, stdout=f, stderr=subprocess.STDOUT, text=True, start_new_session=True
                        )

                # Store session info
                session_info = {
                    "action": "teleoperate" if not dataset_repo_id else "record",
                    "pid": proc.pid,
                    "command": " ".join(cmd),
                    "log_file": str(log_file),
                    "start_time": time.time(),
                    "background": True,
                    "robot_type": robot_type,
                    "teleop_type": teleop_type,
                    "dataset_repo_id": dataset_repo_id,
                }
                session_manager.add_session(session_name, session_info)

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"🚀 **Teleoperation Session Started**\n"
                            f"📝 Session: `{session_name}`\n"
                            f"🆔 Process ID: {proc.pid}\n"
                            f"📁 Command: `{' '.join(cmd)}`\n"
                            f"📋 Log file: `{log_file}`\n"
                            f"🔄 Running in background"
                        }
                    ],
                    "session_name": session_name,
                    "pid": proc.pid,
                    "command": " ".join(cmd),
                    "log_file": str(log_file),
                    "background": True,
                }
            else:
                # Start in foreground
                result = subprocess.run(cmd, capture_output=True, text=True)

                return {
                    "status": "success" if result.returncode == 0 else "error",
                    "content": [
                        {
                            "text": f"🖥️ **Foreground Execution Complete**\n"
                            f"↩️ Return code: {result.returncode}\n"
                            f"📁 Command: `{' '.join(cmd)}`\n\n"
                            f"📤 **Output:**\n```\n{result.stdout}\n```\n\n"
                            f"⚠️ **Errors:**\n```\n{result.stderr}\n```"
                        }
                    ],
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        elif action == "stop":
            if not session_name:
                return {"status": "error", "content": [{"text": "❌ Session name required for stop action"}]}

            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"❌ Session '{session_name}' not found"}]}

            pid = session_info.get("pid")
            if not pid:
                return {"status": "error", "content": [{"text": f"❌ No PID found for session '{session_name}'"}]}

            pid_int = int(pid)
            try:
                # Try graceful termination first
                os.kill(pid_int, signal.SIGTERM)
                time.sleep(2)  # Grace period for process to flush buffers and exit cleanly

                # Force kill if still running after grace period
                if psutil.pid_exists(pid_int):
                    os.kill(pid_int, signal.SIGKILL)

                session_manager.remove_session(session_name)

                return {
                    "status": "success",
                    "content": [{"text": f"🛑 **Session Stopped**\n📝 Session: `{session_name}`\n🆔 PID: {pid}"}],
                    "session_name": session_name,
                    "session_info": session_info,
                }

            except ProcessLookupError:
                # Process already dead
                session_manager.remove_session(session_name)
                return {
                    "status": "success",
                    "content": [{"text": f"✅ Session '{session_name}' was already stopped"}],
                    "session_name": session_name,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Failed to stop session '{session_name}': {str(e)}"}],
                }

        elif action == "list":
            sessions = session_manager.list_sessions()

            content_lines = [f"📋 **Active Teleoperation Sessions** ({len(sessions)})", ""]

            if sessions:
                for name, info in sessions.items():
                    uptime = time.time() - info.get("start_time", 0)
                    uptime_min = uptime / 60
                    pid = info.get("pid")
                    is_running = pid and psutil.pid_exists(pid)

                    content_lines.extend(
                        [
                            f"🎮 **{name}**",
                            f"   - Action: {info.get('action', 'Unknown')}",
                            f"   - PID: {pid}",
                            f"   - Uptime: {uptime_min:.1f} min",
                            f"   - Status: {'🟢 Running' if is_running else '🔴 Stopped'}",
                            f"   - Robot: {info.get('robot_type', 'Unknown')}",
                            f"   - Teleop: {info.get('teleop_type', 'Unknown')}",
                            "",
                        ]
                    )
            else:
                content_lines.append("✨ No active sessions")

            return {
                "status": "success",
                "content": [{"text": "\n".join(content_lines)}],
                "sessions": sessions,
                "count": len(sessions),
            }

        elif action == "status":
            if not session_name:
                return {"status": "error", "content": [{"text": "❌ Session name required for status action"}]}

            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"❌ Session '{session_name}' not found"}]}

            pid = session_info.get("pid")
            start_time: float = float(session_info.get("start_time") or 0)
            uptime = time.time() - start_time
            uptime_min = uptime / 60
            is_running = pid and psutil.pid_exists(int(pid))

            content_lines = [
                f"📊 **Session Status: `{session_name}`**",
                f"🆔 PID: {pid}",
                f"🔧 Action: {session_info.get('action', 'Unknown')}",
                f"⏱️ Uptime: {uptime_min:.1f} min",
                f"📈 Status: {'🟢 Running' if is_running else '🔴 Stopped'}",
                f"🤖 Robot: {session_info.get('robot_type', 'Unknown')}",
                f"🕹️ Teleop: {session_info.get('teleop_type', 'Unknown')}",
            ]

            # Add log tail if available
            log_file_path = session_info.get("log_file")
            if log_file_path and Path(str(log_file_path)).exists():
                content_lines.append(f"📋 Log file: `{log_file_path}`")

                try:
                    with open(str(log_file_path)) as f:
                        lines = f.readlines()
                        if lines:
                            tail_lines = lines[-10:]  # Last 10 lines
                            content_lines.extend(
                                ["", "📖 **Recent Log Output:**", "```", "".join(tail_lines).strip(), "```"]
                            )
                except Exception as e:
                    content_lines.append(f"⚠️ Error reading log: {str(e)}")

            return {
                "status": "success",
                "content": [{"text": "\n".join(content_lines)}],
                "session_name": session_name,
                "pid": pid,
                "uptime": uptime,
                "is_running": is_running,
                **session_info,
            }

        elif action == "replay":
            if not dataset_repo_id:
                return {"status": "error", "content": [{"text": "❌ dataset_repo_id required for replay action"}]}

            try:
                cmd = build_lerobot_command(
                    action="replay",
                    robot_type=robot_type,
                    robot_port=robot_port,
                    robot_id=robot_id,
                    robot_left_arm_port=robot_left_arm_port,
                    robot_right_arm_port=robot_right_arm_port,
                    dataset_repo_id=dataset_repo_id,
                    replay_episode=replay_episode,
                    display_data=display_data,
                )
            except Exception as e:
                return {"status": "error", "content": [{"text": f"❌ Replay command build failed: {str(e)}"}]}

            # Execute replay
            result = subprocess.run(cmd, capture_output=True, text=True)

            content_lines = [
                "🔄 **Episode Replay Complete**",
                f"↩️ Return code: {result.returncode}",
                f"📁 Command: `{' '.join(cmd)}`",
            ]

            if result.stdout:
                content_lines.extend(["", "📤 **Output:**", "```", result.stdout, "```"])

            if result.stderr:
                content_lines.extend(["", "⚠️ **Errors:**", "```", result.stderr, "```"])

            return {
                "status": "success" if result.returncode == 0 else "error",
                "content": [{"text": "\n".join(content_lines)}],
                "command": " ".join(cmd),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        else:
            return {"status": "error", "content": [{"text": f"❌ Unknown action: {action}"}]}

    except Exception as e:
        logger.error(f"LeRobot teleoperate error: {e}")
        return {"status": "error", "content": [{"text": f"❌ Tool execution failed: {str(e)}"}]}
