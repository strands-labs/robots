"""Tests for strands_robots/tools/lerobot_teleoperate.py — subprocess-based teleop.

Comprehensive coverage of:
- build_lerobot_command() — all action/flag combinations
- SessionManager — persistence, loading, cleaning dead sessions
- lerobot_teleoperate() — all actions (start, stop, list, status, replay) and edge cases

All subprocess calls and process management are mocked. CPU-only.
"""

import json
import os
import signal
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# CROSS_PR_SKIP: Tests require extended lerobot_teleoperate API (cross-PR)
try:
    _cross_pr_check = hasattr(
        __import__("strands_robots.tools.lerobot_teleoperate", fromlist=["_get_psutil"]), "_get_psutil"
    )
    if not _cross_pr_check:
        raise AttributeError
except (ImportError, AttributeError, FileNotFoundError, OSError):
    import pytest as _skip_pytest

    _skip_pytest.skip("Tests require extended lerobot_teleoperate API (cross-PR)", allow_module_level=True)


# psutil is lazy-imported in lerobot_teleoperate via _get_psutil().
# Tests mock _get_psutil() so real psutil is NOT needed at runtime.
# We only need the exception classes (NoSuchProcess, AccessDenied) for mock setup.
try:
    import psutil
except ImportError:
    import types as _types

    psutil = _types.ModuleType("psutil")  # type: ignore[assignment]

    class _NoSuchProcess(Exception):
        def __init__(self, pid=None, name=None, msg=None):
            self.pid = pid
            super().__init__(msg or f"process {pid} does not exist")

    class _AccessDenied(Exception):
        def __init__(self, pid=None, name=None, msg=None):
            self.pid = pid
            super().__init__(msg or f"access denied (pid={pid})")

    psutil.NoSuchProcess = _NoSuchProcess  # type: ignore[attr-defined]
    psutil.AccessDenied = _AccessDenied  # type: ignore[attr-defined]
    psutil.pid_exists = lambda pid: False  # type: ignore[attr-defined]
    psutil.Process = lambda pid: None  # type: ignore[attr-defined]
    sys.modules["psutil"] = psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock strands if not installed so tests can run without the full SDK
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f  # @tool decorator becomes identity
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

try:
    from strands_robots.tools.lerobot_teleoperate import (
        SessionManager,
        build_lerobot_command,
        lerobot_teleoperate,
    )
except (ImportError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #11 (agent-tools)", allow_module_level=True)

_requires = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


# ═════════════════════════════════════════════════════════════════════════════
# build_lerobot_command Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildCommand:
    """Test command line building for all action modes."""

    # --- Replay commands ---

    def test_replay_basic(self):
        cmd = build_lerobot_command(
            action="replay",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            replay_episode=3,
        )
        assert "lerobot.scripts.lerobot_replay" in " ".join(cmd)
        assert "--robot-path" in cmd
        assert "so100_follower" in cmd
        assert "--policy-path" in cmd
        assert "user/data" in cmd
        assert "--episode" in cmd
        assert "3" in cmd

    def test_replay_with_ports(self):
        cmd = build_lerobot_command(
            action="replay",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            robot_left_arm_port="/dev/ttyACM1",
            robot_right_arm_port="/dev/ttyACM2",
            dataset_repo_id="user/data",
        )
        assert "--robot-port" in cmd
        assert "/dev/ttyACM0" in cmd
        assert "--robot-left-arm-port" in cmd
        assert "/dev/ttyACM1" in cmd
        assert "--robot-right-arm-port" in cmd
        assert "/dev/ttyACM2" in cmd

    def test_replay_with_display_data(self):
        cmd = build_lerobot_command(
            action="replay",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            display_data=True,
        )
        assert "--display-data" in cmd

    def test_replay_without_optional_flags(self):
        cmd = build_lerobot_command(
            action="replay",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
        )
        assert "--robot-port" not in cmd
        assert "--robot-left-arm-port" not in cmd
        assert "--robot-right-arm-port" not in cmd
        assert "--display-data" not in cmd

    # --- Start (teleoperation) commands ---

    def test_start_teleop_simple(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
        )
        assert "lerobot.scripts.lerobot_teleoperate" in " ".join(cmd)
        assert "--robot.type" in cmd
        assert "so100_follower" in cmd
        assert "--fps" in cmd

    def test_start_teleop_with_time_limit(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            teleop_time_s=120.0,
        )
        assert "--teleop_time_s" in cmd
        assert "120.0" in cmd

    def test_start_teleop_with_all_robot_config(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            robot_id="arm_1",
            robot_left_arm_port="/dev/ttyACM1",
            robot_right_arm_port="/dev/ttyACM2",
        )
        assert "--robot.port" in cmd
        assert "--robot.id" in cmd
        assert "arm_1" in cmd
        assert "--robot.left_arm_port" in cmd
        assert "--robot.right_arm_port" in cmd

    def test_start_teleop_with_all_teleop_config(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            teleop_type="so100_leader",
            teleop_port="/dev/ttyACM1",
            teleop_id="leader_1",
            teleop_left_arm_port="/dev/ttyACM3",
            teleop_right_arm_port="/dev/ttyACM4",
        )
        assert "--teleop.type" in cmd
        assert "so100_leader" in cmd
        assert "--teleop.port" in cmd
        assert "--teleop.id" in cmd
        assert "leader_1" in cmd
        assert "--teleop.left_arm_port" in cmd
        assert "--teleop.right_arm_port" in cmd

    def test_start_teleop_with_cameras(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            robot_cameras={
                "front": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "fps": 30,
                    "width": 1920,
                    "height": 1080,
                },
                "wrist": {
                    "type": "realsense",
                    "index_or_path": 2,
                    "fps": 60,
                    "width": 640,
                    "height": 480,
                },
            },
        )
        cmd_str = " ".join(cmd)
        assert "--camera-config" in cmd_str
        assert "front=opencv:0:30:1920x1080" in cmd_str
        assert "wrist=realsense:2:60:640x480" in cmd_str

    def test_start_teleop_with_display_data(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            display_data=True,
        )
        assert "--display_data" in cmd
        assert "true" in cmd

    # --- Start (recording) commands ---

    def test_start_record_basic(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            dataset_repo_id="user/data",
            dataset_num_episodes=25,
            dataset_episode_time_s=30,
            dataset_reset_time_s=10,
        )
        assert "lerobot.scripts.lerobot_record" in " ".join(cmd)
        assert "--repo-id" in cmd
        assert "user/data" in cmd
        assert "--num-episodes" in cmd
        assert "25" in cmd
        assert "--episode-time-s" in cmd
        assert "30" in cmd
        assert "--reset-time-s" in cmd
        assert "10" in cmd

    def test_start_record_with_task(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            dataset_single_task="pick up red cube",
        )
        assert "--single-task" in cmd
        assert "pick up red cube" in cmd

    def test_start_record_with_root(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            dataset_root="/tmp/data",
        )
        assert "--root" in cmd
        assert "/tmp/data" in cmd

    def test_start_record_push_to_hub(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            dataset_push_to_hub=True,
        )
        assert "--push-to-hub" in cmd

    def test_start_record_no_video(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            dataset_repo_id="user/data",
            dataset_video=False,
        )
        assert "--no-video" in cmd

    def test_start_record_default_port(self):
        cmd = build_lerobot_command(
            action="start",
            robot_type="so100_follower",
            robot_port=None,
            dataset_repo_id="user/data",
        )
        assert "/dev/ttyACM0" in cmd

    # --- Error handling ---

    def test_unknown_action(self):
        with pytest.raises(ValueError, match="Unknown action"):
            build_lerobot_command(action="invalid", robot_type="so100_follower")


# ═════════════════════════════════════════════════════════════════════════════
# SessionManager Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSessionManager:
    """Test the SessionManager persistence layer."""

    def test_init(self):
        mgr = SessionManager()
        assert mgr.sessions_file.name == "active_sessions.json"

    def test_load_sessions_no_file(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"
        sessions = mgr._load_sessions()
        assert sessions == {}

    def test_load_sessions_dead_process(self, tmp_path):
        """Sessions with dead PIDs get cleaned up."""
        sessions_file = tmp_path / "sessions.json"
        data = {"session_1": {"pid": 99999999, "status": "running"}}
        sessions_file.write_text(json.dumps(data))
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        mock_psutil = MagicMock()
        mock_psutil.pid_exists.return_value = False
        # Must have real exception classes for except clause
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.AccessDenied = psutil.AccessDenied

        with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil):
            sessions = mgr._load_sessions()
        assert sessions == {}

    def test_load_sessions_running_process(self, tmp_path):
        """Sessions with running PIDs are kept."""
        sessions_file = tmp_path / "sessions.json"
        data = {"session_1": {"pid": 12345, "status": "running"}}
        sessions_file.write_text(json.dumps(data))
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        mock_psutil = MagicMock()
        mock_psutil.pid_exists.return_value = True
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.AccessDenied = psutil.AccessDenied
        mock_proc = MagicMock()
        mock_proc.is_running.return_value = True
        mock_psutil.Process.return_value = mock_proc

        with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil):
            sessions = mgr._load_sessions()
        assert "session_1" in sessions

    def test_load_sessions_process_access_denied(self, tmp_path):
        """Sessions where process check raises AccessDenied are cleaned up."""
        sessions_file = tmp_path / "sessions.json"
        data = {"session_1": {"pid": 12345, "status": "running"}}
        sessions_file.write_text(json.dumps(data))
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        mock_psutil = MagicMock()
        mock_psutil.pid_exists.return_value = True
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.AccessDenied = psutil.AccessDenied
        mock_psutil.Process.side_effect = psutil.AccessDenied(pid=12345)

        with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil):
            sessions = mgr._load_sessions()
        assert sessions == {}

    def test_load_sessions_no_such_process(self, tmp_path):
        """Sessions where process disappeared are cleaned up."""
        sessions_file = tmp_path / "sessions.json"
        data = {"session_1": {"pid": 12345, "status": "running"}}
        sessions_file.write_text(json.dumps(data))
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        mock_psutil = MagicMock()
        mock_psutil.pid_exists.return_value = True
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.AccessDenied = psutil.AccessDenied
        mock_psutil.Process.side_effect = psutil.NoSuchProcess(pid=12345)

        with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil):
            sessions = mgr._load_sessions()
        assert sessions == {}

    def test_load_sessions_corrupt_file(self, tmp_path):
        """Corrupt JSON file returns empty dict."""
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text("not valid json{{{")
        mgr = SessionManager()
        mgr.sessions_file = sessions_file
        sessions = mgr._load_sessions()
        assert sessions == {}

    def test_save_sessions(self, tmp_path):
        sessions_file = tmp_path / "sessions.json"
        mgr = SessionManager()
        mgr.sessions_file = sessions_file
        mgr._save_sessions({"test": {"pid": 1, "status": "ok"}})
        assert sessions_file.exists()
        loaded = json.loads(sessions_file.read_text())
        assert "test" in loaded

    def test_save_sessions_io_error(self, tmp_path):
        """Save handles IOError gracefully."""
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "nonexistent_dir" / "deep" / "sessions.json"
        mgr._save_sessions({"test": {"pid": 1}})

    def test_add_session(self, tmp_path):
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text("{}")
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        with patch.object(mgr, "_load_sessions", return_value={}):
            with patch.object(mgr, "_save_sessions") as mock_save:
                mgr.add_session("new_session", {"pid": 42})
                mock_save.assert_called_once()
                saved = mock_save.call_args[0][0]
                assert "new_session" in saved
                assert saved["new_session"]["pid"] == 42

    def test_remove_session(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"

        with patch.object(mgr, "_load_sessions", return_value={"s1": {"pid": 1}, "s2": {"pid": 2}}):
            with patch.object(mgr, "_save_sessions") as mock_save:
                mgr.remove_session("s1")
                saved = mock_save.call_args[0][0]
                assert "s1" not in saved
                assert "s2" in saved

    def test_remove_nonexistent_session(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"

        with patch.object(mgr, "_load_sessions", return_value={"s1": {"pid": 1}}):
            with patch.object(mgr, "_save_sessions") as mock_save:
                mgr.remove_session("nonexistent")
                mock_save.assert_not_called()

    def test_get_session_exists(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"

        with patch.object(mgr, "_load_sessions", return_value={"s1": {"pid": 1}}):
            result = mgr.get_session("s1")
            assert result == {"pid": 1}

    def test_get_session_not_exists(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"

        with patch.object(mgr, "_load_sessions", return_value={}):
            result = mgr.get_session("nonexistent")
            assert result is None

    def test_list_sessions(self, tmp_path):
        mgr = SessionManager()
        mgr.sessions_file = tmp_path / "sessions.json"

        expected = {"s1": {"pid": 1}, "s2": {"pid": 2}}
        with patch.object(mgr, "_load_sessions", return_value=expected):
            result = mgr.list_sessions()
            assert result == expected

    def test_load_sessions_process_not_running(self, tmp_path):
        """Sessions where process exists but is_running returns False are cleaned."""
        sessions_file = tmp_path / "sessions.json"
        data = {"session_1": {"pid": 12345, "status": "running"}}
        sessions_file.write_text(json.dumps(data))
        mgr = SessionManager()
        mgr.sessions_file = sessions_file

        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = True
        mock_psutil_mod.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil_mod.AccessDenied = psutil.AccessDenied
        mock_proc = MagicMock()
        mock_proc.is_running.return_value = False  # Process exists but not running
        mock_psutil_mod.Process.return_value = mock_proc

        with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
            sessions = mgr._load_sessions()
        assert sessions == {}


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_teleoperate — Start Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStartAction:
    """Test start teleop action."""

    @patch("subprocess.Popen")
    def test_start_background_with_auto_calibration(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so100_leader",
            teleop_port="/dev/ttyACM1",
            background=True,
            auto_accept_calibration=True,
        )
        assert result["status"] == "success"
        assert result["pid"] == 12345
        assert result["background"] is True
        assert "session_name" in result

    @patch("subprocess.Popen")
    def test_start_background_without_auto_calibration(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 12346
        mock_popen.return_value = mock_proc

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            background=True,
            auto_accept_calibration=False,
        )
        assert result["status"] == "success"

    @patch("subprocess.run")
    def test_start_foreground(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Teleop running..."
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            background=False,
        )
        assert result["status"] == "success"
        assert "Foreground Execution Complete" in result["content"][0]["text"]
        assert result["return_code"] == 0

    @patch("subprocess.run")
    def test_start_foreground_error(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: device not found"
        mock_run.return_value = mock_result

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
            background=False,
        )
        assert result["status"] == "error"
        assert result["return_code"] == 1

    @patch("subprocess.Popen")
    def test_start_with_recording(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 12347
        mock_popen.return_value = mock_proc

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so100_leader",
            teleop_port="/dev/ttyACM1",
            dataset_repo_id="user/test_data",
            dataset_single_task="pick cube",
        )
        assert result["status"] == "success"

    @patch("subprocess.Popen")
    def test_start_auto_generates_session_name(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        result = lerobot_teleoperate(
            action="start",
            robot_type="so100_follower",
        )
        assert result["status"] == "success"
        assert result["session_name"].startswith("teleop_")

    @patch("subprocess.Popen")
    def test_start_custom_session_name(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_popen.return_value = mock_proc

        result = lerobot_teleoperate(
            action="start",
            session_name="my_custom_session",
            robot_type="so100_follower",
        )
        assert result["status"] == "success"
        assert result["session_name"] == "my_custom_session"

    def test_start_session_already_exists(self):
        with patch.object(SessionManager, "get_session", return_value={"pid": 123}):
            result = lerobot_teleoperate(
                action="start",
                session_name="existing_session",
            )
            assert result["status"] == "error"
            assert "already exists" in result["content"][0]["text"]

    def test_start_command_build_failure(self):
        with patch(
            "strands_robots.tools.lerobot_teleoperate.build_lerobot_command", side_effect=ValueError("bad config")
        ):
            result = lerobot_teleoperate(
                action="start",
                robot_type="so100_follower",
            )
            assert result["status"] == "error"
            assert "Command build failed" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_teleoperate — Stop Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStopAction:
    """Test stop teleop action."""

    def test_stop_no_session_name(self):
        result = lerobot_teleoperate(action="stop")
        assert result["status"] == "error"
        assert "Session name required" in result["content"][0]["text"]

    def test_stop_session_not_found(self):
        with patch.object(SessionManager, "get_session", return_value=None):
            result = lerobot_teleoperate(action="stop", session_name="nonexistent")
            assert result["status"] == "error"
            assert "not found" in result["content"][0]["text"]

    def test_stop_no_pid_in_session(self):
        with patch.object(SessionManager, "get_session", return_value={"status": "running"}):
            result = lerobot_teleoperate(action="stop", session_name="broken_session")
            assert result["status"] == "error"
            assert "No PID found" in result["content"][0]["text"]

    @patch("os.kill")
    def test_stop_graceful_termination(self, mock_kill):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = False

        session_data = {"pid": 12345, "start_time": time.time() - 60}

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch.object(SessionManager, "remove_session") as mock_remove:
                with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                    with patch("time.sleep"):
                        result = lerobot_teleoperate(action="stop", session_name="my_session")

        assert result["status"] == "success"
        mock_kill.assert_called_with(12345, signal.SIGTERM)
        mock_remove.assert_called_with("my_session")

    @patch("os.kill")
    def test_stop_force_kill(self, mock_kill):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = True

        session_data = {"pid": 12345, "start_time": time.time() - 60}

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch.object(SessionManager, "remove_session"):
                with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                    with patch("time.sleep"):
                        result = lerobot_teleoperate(action="stop", session_name="my_session")

        assert result["status"] == "success"
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        mock_kill.assert_any_call(12345, signal.SIGKILL)

    @patch("os.kill")
    def test_stop_process_already_dead(self, mock_kill):
        mock_kill.side_effect = ProcessLookupError("No such process")

        session_data = {"pid": 12345, "start_time": time.time()}

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch.object(SessionManager, "remove_session") as mock_remove:
                result = lerobot_teleoperate(action="stop", session_name="dead_session")

        assert result["status"] == "success"
        assert "already stopped" in result["content"][0]["text"]
        mock_remove.assert_called_with("dead_session")

    @patch("os.kill")
    def test_stop_unexpected_error(self, mock_kill):
        mock_kill.side_effect = PermissionError("Permission denied")

        session_data = {"pid": 12345, "start_time": time.time()}

        with patch.object(SessionManager, "get_session", return_value=session_data):
            result = lerobot_teleoperate(action="stop", session_name="perm_error")

        assert result["status"] == "error"
        assert "Failed to stop" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_teleoperate — List Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestListAction:
    """Test list sessions action."""

    def test_list_empty(self):
        with patch.object(SessionManager, "list_sessions", return_value={}):
            result = lerobot_teleoperate(action="list")
            assert result["status"] == "success"
            assert result["count"] == 0
            assert "No active sessions" in result["content"][0]["text"]

    def test_list_with_sessions(self):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = True

        sessions = {
            "session_1": {
                "action": "teleoperate",
                "pid": 1234,
                "start_time": time.time() - 300,
                "robot_type": "so100_follower",
                "teleop_type": "so100_leader",
            },
            "session_2": {
                "action": "record",
                "pid": 5678,
                "start_time": time.time() - 60,
                "robot_type": "koch_follower",
                "teleop_type": "gamepad",
            },
        }

        with patch.object(SessionManager, "list_sessions", return_value=sessions):
            with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                result = lerobot_teleoperate(action="list")

        assert result["status"] == "success"
        assert result["count"] == 2
        text = result["content"][0]["text"]
        assert "session_1" in text
        assert "session_2" in text

    def test_list_with_stopped_session(self):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = False

        sessions = {
            "dead_session": {
                "action": "teleoperate",
                "pid": 99999,
                "start_time": time.time() - 3600,
                "robot_type": "so100_follower",
                "teleop_type": "so100_leader",
            },
        }

        with patch.object(SessionManager, "list_sessions", return_value=sessions):
            with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                result = lerobot_teleoperate(action="list")

        assert result["status"] == "success"
        assert "Stopped" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_teleoperate — Status Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStatusAction:
    """Test status action."""

    def test_status_no_session_name(self):
        result = lerobot_teleoperate(action="status")
        assert result["status"] == "error"
        assert "Session name required" in result["content"][0]["text"]

    def test_status_session_not_found(self):
        with patch.object(SessionManager, "get_session", return_value=None):
            result = lerobot_teleoperate(action="status", session_name="nonexistent")
            assert result["status"] == "error"
            assert "not found" in result["content"][0]["text"]

    def test_status_running_session(self):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = True

        session_data = {
            "action": "teleoperate",
            "pid": 12345,
            "start_time": time.time() - 120,
            "robot_type": "so100_follower",
            "teleop_type": "so100_leader",
        }

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                result = lerobot_teleoperate(action="status", session_name="my_session")

        assert result["status"] == "success"
        assert result["is_running"] is True
        assert result["pid"] == 12345
        text = result["content"][0]["text"]
        assert "Running" in text

    def test_status_with_log_file(self, tmp_path):
        log_file = tmp_path / "session.log"
        log_lines = [f"Line {i}\n" for i in range(20)]
        log_file.write_text("".join(log_lines))

        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = True

        session_data = {
            "action": "teleoperate",
            "pid": 12345,
            "start_time": time.time() - 60,
            "robot_type": "so100_follower",
            "teleop_type": "so100_leader",
            "log_file": str(log_file),
        }

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                result = lerobot_teleoperate(action="status", session_name="my_session")

        text = result["content"][0]["text"]
        assert "Recent Log Output" in text
        assert "Line 19" in text

    def test_status_with_log_read_error(self, tmp_path):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.pid_exists.return_value = False

        log_path = tmp_path / "fake.log"
        log_path.mkdir()

        session_data = {
            "action": "teleoperate",
            "pid": 12345,
            "start_time": time.time() - 60,
            "robot_type": "so100_follower",
            "teleop_type": "so100_leader",
            "log_file": str(log_path),
        }

        with patch.object(SessionManager, "get_session", return_value=session_data):
            with patch("strands_robots.tools.lerobot_teleoperate._get_psutil", return_value=mock_psutil_mod):
                result = lerobot_teleoperate(action="status", session_name="my_session")

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Error reading log" in text


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_teleoperate — Replay Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestReplayAction:
    """Test replay action."""

    def test_replay_no_dataset(self):
        result = lerobot_teleoperate(
            action="replay",
            dataset_repo_id=None,
        )
        assert result["status"] == "error"
        assert "dataset_repo_id required" in result["content"][0]["text"]

    @patch("subprocess.run")
    def test_replay_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Replay complete"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = lerobot_teleoperate(
            action="replay",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            dataset_repo_id="user/data",
            replay_episode=5,
        )
        assert result["status"] == "success"
        assert "Replay Complete" in result["content"][0]["text"]
        assert result["return_code"] == 0

    @patch("subprocess.run")
    def test_replay_with_output_and_errors(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Replaying episode 5..."
        mock_result.stderr = "Warning: low battery"
        mock_run.return_value = mock_result

        result = lerobot_teleoperate(
            action="replay",
            dataset_repo_id="user/data",
        )
        text = result["content"][0]["text"]
        assert "Replaying episode 5" in text
        assert "Warning: low battery" in text

    @patch("subprocess.run")
    def test_replay_empty_output(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = lerobot_teleoperate(action="replay", dataset_repo_id="user/data")
        assert result["status"] == "success"

    def test_replay_command_build_failure(self):
        with patch(
            "strands_robots.tools.lerobot_teleoperate.build_lerobot_command", side_effect=ValueError("bad args")
        ):
            result = lerobot_teleoperate(action="replay", dataset_repo_id="user/data")
            assert result["status"] == "error"
            assert "Replay command build failed" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Error Handling / Edge Cases
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_unknown_action(self):
        result = lerobot_teleoperate(action="invalid_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_general_exception_handler(self):
        with patch.object(SessionManager, "list_sessions", side_effect=Exception("DB crashed")):
            result = lerobot_teleoperate(action="list")
            assert result["status"] == "error"
            assert "Tool execution failed" in result["content"][0]["text"]

    def test_psutil_lazy_import(self):
        from strands_robots.tools.lerobot_teleoperate import _get_psutil

        try:
            ps = _get_psutil()
            assert ps is not None
            assert hasattr(ps, "pid_exists")
        except ImportError:
            pytest.skip("psutil not installed — lazy import correctly raises")
