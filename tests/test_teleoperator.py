"""Tests for strands_robots/tools/teleoperator.py — native LeRobot teleoperator.

All LeRobot robot/teleop hardware is mocked. CPU-only.
Targets: _make_robot, _make_teleop, _run_teleop_loop, _run_record_loop,
         and all teleoperator() action branches including edge cases.
"""

import os
import sys
import threading
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

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

from strands_robots.tools.teleoperator import (
    _ACTIVE_SESSION,
    _make_robot,
    _make_teleop,
    _run_record_loop,
    _run_teleop_loop,
    teleoperator,
)

_requires = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


@pytest.fixture(autouse=True)
def reset_session():
    """Reset global session state before each test."""
    _ACTIVE_SESSION.update(
        {
            "running": False,
            "thread": None,
            "robot": None,
            "teleop": None,
            "record_session": None,
            "mode": None,
            "stats": {},
        }
    )
    yield
    _ACTIVE_SESSION.update(
        {
            "running": False,
            "thread": None,
            "robot": None,
            "teleop": None,
            "record_session": None,
            "mode": None,
            "stats": {},
        }
    )


def _build_lerobot_package_mocks():
    """Build a complete mock lerobot package hierarchy for sys.modules patching.

    Returns dict of module mocks suitable for patch.dict("sys.modules", ...).
    """

    # Base config class — must be a real class for issubclass()
    class RobotConfig:
        pass

    class TeleoperatorConfig:
        pass

    mocks = {}

    # Top-level lerobot package
    mock_lerobot = ModuleType("lerobot")
    mocks["lerobot"] = mock_lerobot

    # lerobot.robots
    mock_lr_robots = MagicMock()
    mock_lr_robots.__path__ = ["/fake/lerobot/robots"]
    mocks["lerobot.robots"] = mock_lr_robots

    # lerobot.robots.config
    mock_robots_config = MagicMock()
    mock_robots_config.RobotConfig = RobotConfig
    mocks["lerobot.robots.config"] = mock_robots_config

    # lerobot.robots.utils
    mock_robots_utils = MagicMock()
    mocks["lerobot.robots.utils"] = mock_robots_utils

    # lerobot.cameras
    mock_cameras = MagicMock()
    mocks["lerobot.cameras"] = mock_cameras
    mocks["lerobot.cameras.opencv"] = MagicMock()
    mock_opencv_config = MagicMock()
    mocks["lerobot.cameras.opencv.configuration_opencv"] = mock_opencv_config

    # lerobot.teleoperators
    mock_lr_teleops = MagicMock()
    mock_lr_teleops.__path__ = ["/fake/lerobot/teleoperators"]
    mocks["lerobot.teleoperators"] = mock_lr_teleops

    # lerobot.teleoperators.config
    mock_teleops_config = MagicMock()
    mock_teleops_config.TeleoperatorConfig = TeleoperatorConfig
    mocks["lerobot.teleoperators.config"] = mock_teleops_config

    # lerobot.teleoperators.utils
    mock_teleops_utils = MagicMock()
    mocks["lerobot.teleoperators.utils"] = mock_teleops_utils

    return mocks, RobotConfig, TeleoperatorConfig


# ═════════════════════════════════════════════════════════════════════════════
# _make_robot Tests — comprehensive mock of LeRobot discovery
# ═════════════════════════════════════════════════════════════════════════════


class TestMakeRobot:
    """Test the _make_robot factory function with full mock chain."""

    def test_make_robot_import_error(self):
        """_make_robot fails if lerobot not installed at all."""
        # Block lerobot reimport by setting None in sys.modules
        block = {k: None for k in list(sys.modules) if k.startswith("lerobot")}
        block["lerobot"] = None
        with patch.dict("sys.modules", block):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                _make_robot("so100_follower")

    def test_make_robot_not_found(self):
        """_make_robot raises ValueError when no matching config is found."""
        mocks, RobotConfig, _ = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[]):
                with pytest.raises(ValueError, match="Robot config not found"):
                    _make_robot("nonexistent_robot_type")

    def test_make_robot_skips_reserved_modules(self):
        """_make_robot skips 'config', 'robot', 'utils' module names."""
        mocks, RobotConfig, _ = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch(
                "pkgutil.iter_modules",
                return_value=[
                    (None, "config", False),
                    (None, "robot", False),
                    (None, "utils", False),
                ],
            ):
                with pytest.raises(ValueError, match="Robot config not found"):
                    _make_robot("so100_follower")

    def test_make_robot_continues_on_import_exception(self):
        """_make_robot continues scanning if a module import fails."""
        mocks, RobotConfig, _ = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch(
                "pkgutil.iter_modules",
                return_value=[
                    (None, "broken_module", False),
                    (None, "another_broken", False),
                ],
            ):
                with patch("importlib.import_module", side_effect=RuntimeError("broken")):
                    with pytest.raises(ValueError, match="Robot config not found"):
                        _make_robot("so100_follower")

    def test_make_robot_success_full_chain(self):
        """_make_robot discovers config, builds cameras, and creates robot."""
        mocks, RobotConfig, _ = _build_lerobot_package_mocks()

        # Create config class that is a subclass of RobotConfig
        class So100FollowerConfig(RobotConfig):
            __dataclass_fields__ = {"cameras": None, "port": None, "id": None, "extra_param": None}

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Build the module containing our config
        mock_robot_mod = ModuleType("lerobot.robots.so100_follower")
        mock_robot_mod.So100FollowerConfig = So100FollowerConfig
        mocks["lerobot.robots.so100_follower"] = mock_robot_mod

        mock_make_robot = MagicMock(return_value=MagicMock())
        mocks["lerobot.robots.utils"].make_robot_from_config = mock_make_robot

        mock_opencv_config_cls = MagicMock()
        mocks["lerobot.cameras.opencv.configuration_opencv"].OpenCVCameraConfig = mock_opencv_config_cls

        with patch.dict("sys.modules", mocks):
            with patch(
                "pkgutil.iter_modules",
                return_value=[
                    (None, "so100_follower", False),
                ],
            ):
                with patch("importlib.import_module", return_value=mock_robot_mod):
                    result = _make_robot(
                        "so100_follower",
                        robot_port="/dev/ttyACM0",
                        robot_id="arm_1",
                        cameras={"front": {"index_or_path": 0, "fps": 30, "width": 640, "height": 480}},
                        extra_param="custom_value",
                    )

        mock_make_robot.assert_called_once()
        mock_opencv_config_cls.assert_called_once()
        assert result is not None

    def test_make_robot_no_cameras(self):
        """_make_robot works without cameras."""
        mocks, RobotConfig, _ = _build_lerobot_package_mocks()

        class So100FollowerConfig(RobotConfig):
            __dataclass_fields__ = {"cameras": None, "port": None}

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_robot_mod = ModuleType("lerobot.robots.so100_follower")
        mock_robot_mod.So100FollowerConfig = So100FollowerConfig
        mocks["lerobot.robots.so100_follower"] = mock_robot_mod
        mock_make_robot = MagicMock(return_value=MagicMock())
        mocks["lerobot.robots.utils"].make_robot_from_config = mock_make_robot

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[(None, "so100_follower", False)]):
                with patch("importlib.import_module", return_value=mock_robot_mod):
                    _make_robot("so100_follower", robot_port="/dev/ttyACM0", cameras=None)

        mock_make_robot.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# _make_teleop Tests — comprehensive mock of LeRobot teleop discovery
# ═════════════════════════════════════════════════════════════════════════════


class TestMakeTeleop:
    """Test the _make_teleop factory function with full mock chain."""

    def test_make_teleop_import_error(self):
        """_make_teleop fails if lerobot not installed at all."""
        block = {k: None for k in list(sys.modules) if k.startswith("lerobot")}
        block["lerobot"] = None
        with patch.dict("sys.modules", block):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                _make_teleop("so100_leader")

    def test_make_teleop_not_found(self):
        mocks, _, TeleoperatorConfig = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[]):
                with pytest.raises(ValueError, match="Teleoperator config not found"):
                    _make_teleop("nonexistent_type")

    def test_make_teleop_skips_reserved_modules(self):
        mocks, _, TeleoperatorConfig = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch(
                "pkgutil.iter_modules",
                return_value=[
                    (None, "config", False),
                    (None, "teleoperator", False),
                    (None, "utils", False),
                ],
            ):
                with pytest.raises(ValueError, match="Teleoperator config not found"):
                    _make_teleop("so100_leader")

    def test_make_teleop_continues_on_import_exception(self):
        mocks, _, TeleoperatorConfig = _build_lerobot_package_mocks()

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[(None, "broken_mod", False)]):
                with patch("importlib.import_module", side_effect=RuntimeError("broken")):
                    with pytest.raises(ValueError, match="Teleoperator config not found"):
                        _make_teleop("so100_leader")

    def test_make_teleop_success(self):
        """_make_teleop creates teleop config with port and id."""
        mocks, _, TeleoperatorConfig = _build_lerobot_package_mocks()

        class So100LeaderConfig(TeleoperatorConfig):
            __dataclass_fields__ = {"port": None, "id": None, "extra_field": None}

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_teleop_mod = ModuleType("lerobot.teleoperators.so100_leader")
        mock_teleop_mod.So100LeaderConfig = So100LeaderConfig
        mocks["lerobot.teleoperators.so100_leader"] = mock_teleop_mod

        mock_make_teleop = MagicMock(return_value=MagicMock())
        mocks["lerobot.teleoperators.utils"].make_teleoperator_from_config = mock_make_teleop

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[(None, "so100_leader", False)]):
                with patch("importlib.import_module", return_value=mock_teleop_mod):
                    result = _make_teleop(
                        "so100_leader",
                        teleop_port="/dev/ttyACM1",
                        teleop_id="leader_1",
                        extra_field="custom",
                    )

        mock_make_teleop.assert_called_once()
        assert result is not None

    def test_make_teleop_no_dataclass_fields(self):
        """_make_teleop handles config without __dataclass_fields__."""
        mocks, _, TeleoperatorConfig = _build_lerobot_package_mocks()

        class So100LeaderConfig(TeleoperatorConfig):
            # Intentionally no __dataclass_fields__
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_teleop_mod = ModuleType("lerobot.teleoperators.so100_leader")
        mock_teleop_mod.So100LeaderConfig = So100LeaderConfig
        mocks["lerobot.teleoperators.so100_leader"] = mock_teleop_mod

        mock_make_teleop = MagicMock(return_value=MagicMock())
        mocks["lerobot.teleoperators.utils"].make_teleoperator_from_config = mock_make_teleop

        with patch.dict("sys.modules", mocks):
            with patch("pkgutil.iter_modules", return_value=[(None, "so100_leader", False)]):
                with patch("importlib.import_module", return_value=mock_teleop_mod):
                    _make_teleop("so100_leader", teleop_port="/dev/ttyACM1")

        mock_make_teleop.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# Action Dispatch / Status Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestActionDispatch:
    """Test action routing."""

    def test_unknown_action(self):
        result = teleoperator(action="bogus")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_status_no_session(self):
        result = teleoperator(action="status")
        assert result["status"] == "success"
        assert "No active" in result["content"][0]["text"]

    def test_status_active_teleop(self):
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "mode": "teleop",
                "stats": {"steps": 42, "start_time": time.time() - 10},
            }
        )
        result = teleoperator(action="status")
        assert result["status"] == "success"
        assert "teleop" in result["content"][0]["text"]
        assert "42" in result["content"][0]["text"]

    def test_status_active_record(self):
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "mode": "record",
                "stats": {
                    "current_episode": 3,
                    "episodes_completed": 2,
                    "total_frames": 500,
                    "start_time": time.time() - 30,
                },
            }
        )
        result = teleoperator(action="status")
        assert result["status"] == "success"
        assert "record" in result["content"][0]["text"].lower()

    def test_status_shows_uptime(self):
        """Status should show uptime calculation."""
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "mode": "teleop",
                "stats": {"steps": 10, "start_time": time.time() - 60},
            }
        )
        result = teleoperator(action="status")
        text = result["content"][0]["text"]
        assert "Uptime" in text

    def test_default_action_is_status(self):
        """Default action parameter should be 'status'."""
        result = teleoperator()
        assert result["status"] == "success"
        assert "No active" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Stop Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStopAction:
    """Test stop action."""

    def test_stop_no_session(self):
        result = teleoperator(action="stop")
        assert result["status"] == "success"
        assert "No active" in result["content"][0]["text"]

    def test_stop_teleop_session(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_thread = MagicMock()

        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": mock_robot,
                "teleop": mock_teleop,
                "record_session": None,
                "mode": "teleop",
                "stats": {"steps": 100, "duration": 10.0},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"
        assert "stopped" in result["content"][0]["text"].lower()
        mock_robot.disconnect.assert_called_once()
        mock_teleop.disconnect.assert_called_once()
        assert _ACTIVE_SESSION["running"] is False

    def test_stop_teleop_shows_stats(self):
        """Stop teleop should show steps and duration."""
        mock_thread = MagicMock()
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": MagicMock(),
                "teleop": MagicMock(),
                "record_session": None,
                "mode": "teleop",
                "stats": {"steps": 500, "duration": 25.3},
            }
        )

        result = teleoperator(action="stop")
        text = result["content"][0]["text"]
        assert "500" in text
        assert "25.3" in text

    def test_stop_record_session(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_record = MagicMock()
        mock_record.save_and_push.return_value = {"episodes": 5, "total_frames": 1500, "root": "/tmp/ds"}
        mock_thread = MagicMock()

        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": mock_robot,
                "teleop": mock_teleop,
                "record_session": mock_record,
                "mode": "record",
                "stats": {"episodes_completed": 5},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"
        mock_record.stop.assert_called_once()
        mock_record.save_and_push.assert_called_once()
        assert "Episodes" in result["content"][0]["text"]

    def test_stop_record_finalization_error(self):
        """Record finalization error is handled gracefully."""
        mock_record = MagicMock()
        mock_record.stop.side_effect = RuntimeError("DB error")
        mock_thread = MagicMock()

        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": MagicMock(),
                "teleop": MagicMock(),
                "record_session": mock_record,
                "mode": "record",
                "stats": {},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"

    def test_stop_disconnect_error_handled(self):
        mock_robot = MagicMock()
        mock_robot.disconnect.side_effect = RuntimeError("USB error")
        mock_teleop = MagicMock()
        mock_thread = MagicMock()

        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": mock_robot,
                "teleop": mock_teleop,
                "record_session": None,
                "mode": "teleop",
                "stats": {},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"

    def test_stop_clears_session_state(self):
        mock_thread = MagicMock()
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": MagicMock(),
                "teleop": MagicMock(),
                "record_session": None,
                "mode": "teleop",
                "stats": {"steps": 100},
            }
        )

        teleoperator(action="stop")
        assert _ACTIVE_SESSION["running"] is False
        assert _ACTIVE_SESSION["thread"] is None
        assert _ACTIVE_SESSION["robot"] is None
        assert _ACTIVE_SESSION["teleop"] is None
        assert _ACTIVE_SESSION["record_session"] is None
        assert _ACTIVE_SESSION["mode"] is None

    def test_stop_with_null_robot_teleop(self):
        mock_thread = MagicMock()
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": None,
                "teleop": None,
                "record_session": None,
                "mode": "teleop",
                "stats": {},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"

    def test_stop_thread_join_timeout(self):
        """Stop handles thread that doesn't join in time."""
        mock_thread = MagicMock()
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "thread": mock_thread,
                "robot": MagicMock(),
                "teleop": MagicMock(),
                "record_session": None,
                "mode": "teleop",
                "stats": {},
            }
        )

        result = teleoperator(action="stop")
        assert result["status"] == "success"
        mock_thread.join.assert_called_once_with(timeout=5)


# ═════════════════════════════════════════════════════════════════════════════
# Discard Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDiscardAction:
    """Test discard action."""

    def test_discard_no_session(self):
        result = teleoperator(action="discard")
        assert result["status"] == "error"

    def test_discard_no_record_session(self):
        _ACTIVE_SESSION["running"] = True
        _ACTIVE_SESSION["record_session"] = None
        result = teleoperator(action="discard")
        assert result["status"] == "error"

    def test_discard_success(self):
        mock_record = MagicMock()
        _ACTIVE_SESSION.update(
            {
                "running": True,
                "record_session": mock_record,
            }
        )
        result = teleoperator(action="discard")
        assert result["status"] == "success"
        mock_record.discard_episode.assert_called_once()
        assert "discarded" in result["content"][0]["text"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# Teleop Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTeleopAction:
    """Test teleop start action."""

    def test_teleop_already_running(self):
        _ACTIVE_SESSION["running"] = True
        result = teleoperator(action="teleop")
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"].lower()

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_teleop_start(self, mock_make_robot, mock_make_teleop):
        mock_robot = MagicMock()
        mock_teleop_dev = MagicMock()
        mock_make_robot.return_value = mock_robot
        mock_make_teleop.return_value = mock_teleop_dev

        result = teleoperator(
            action="teleop",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so100_leader",
            teleop_port="/dev/ttyACM1",
            duration=0.01,
        )

        assert result["status"] == "success"
        assert "Teleoperation started" in result["content"][0]["text"]
        mock_robot.connect.assert_called_once()
        mock_teleop_dev.connect.assert_called_once()
        assert _ACTIVE_SESSION["running"] is True
        assert _ACTIVE_SESSION["mode"] == "teleop"
        time.sleep(0.1)

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_teleop_start_shows_config(self, mock_make_robot, mock_make_teleop):
        mock_make_robot.return_value = MagicMock()
        mock_make_teleop.return_value = MagicMock()

        result = teleoperator(
            action="teleop",
            robot_type="so100_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so100_leader",
            teleop_port="/dev/ttyACM1",
            fps=60,
            duration=0.01,
        )

        text = result["content"][0]["text"]
        assert "so100_follower" in text
        assert "so100_leader" in text
        assert "60" in text
        time.sleep(0.1)

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_teleop_make_robot_error(self, mock_make_robot, mock_make_teleop):
        mock_make_robot.side_effect = ValueError("Robot config not found")
        result = teleoperator(action="teleop")
        assert result["status"] == "error"
        assert "Robot config not found" in result["content"][0]["text"]

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_teleop_make_teleop_error(self, mock_make_robot, mock_make_teleop):
        mock_make_robot.return_value = MagicMock()
        mock_make_teleop.side_effect = ValueError("Teleop config not found")
        result = teleoperator(action="teleop")
        assert result["status"] == "error"
        assert "Teleop config not found" in result["content"][0]["text"]

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_teleop_connect_error(self, mock_make_robot, mock_make_teleop):
        mock_robot = MagicMock()
        mock_robot.connect.side_effect = RuntimeError("USB not found")
        mock_make_robot.return_value = mock_robot
        mock_make_teleop.return_value = MagicMock()
        result = teleoperator(action="teleop")
        assert result["status"] == "error"
        assert "USB not found" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Record Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordAction:
    """Test record start action."""

    def test_record_already_running(self):
        _ACTIVE_SESSION["running"] = True
        result = teleoperator(action="record")
        assert result["status"] == "error"

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_record_start(self, mock_make_robot, mock_make_teleop):
        mock_robot = MagicMock()
        mock_teleop_dev = MagicMock()
        mock_make_robot.return_value = mock_robot
        mock_make_teleop.return_value = mock_teleop_dev

        mock_record_session = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "strands_robots.record": MagicMock(RecordSession=MagicMock(return_value=mock_record_session)),
            },
        ):
            with patch(
                "strands_robots.tools.teleoperator.RecordSession",
                MagicMock(return_value=mock_record_session),
                create=True,
            ):
                result = teleoperator(
                    action="record",
                    robot_type="so100_follower",
                    teleop_type="so100_leader",
                    repo_id="user/test_data",
                    task="pick cube",
                    num_episodes=1,
                )

                assert result["status"] == "success"
                assert "Recording started" in result["content"][0]["text"]

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_record_auto_generates_repo_id(self, mock_make_robot, mock_make_teleop):
        mock_make_robot.return_value = MagicMock()
        mock_make_teleop.return_value = MagicMock()

        mock_record_session = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "strands_robots.record": MagicMock(RecordSession=MagicMock(return_value=mock_record_session)),
            },
        ):
            with patch(
                "strands_robots.tools.teleoperator.RecordSession",
                MagicMock(return_value=mock_record_session),
                create=True,
            ):
                result = teleoperator(action="record", repo_id=None, num_episodes=1)
                assert result["status"] == "success"
                text = result["content"][0]["text"]
                assert "local/teleop_" in text

    @patch("strands_robots.tools.teleoperator._make_teleop")
    @patch("strands_robots.tools.teleoperator._make_robot")
    def test_record_make_robot_error(self, mock_make_robot, mock_make_teleop):
        mock_make_robot.side_effect = ValueError("No robot")
        result = teleoperator(action="record")
        assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# _run_teleop_loop Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRunTeleopLoop:
    """Test the background teleop loop."""

    def test_loop_runs_and_stops(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_teleop.get_action.return_value = {"j0": 0.1}

        _ACTIVE_SESSION["running"] = True

        def stop_after_delay():
            time.sleep(0.05)
            _ACTIVE_SESSION["running"] = False

        stopper = threading.Thread(target=stop_after_delay, daemon=True)
        stopper.start()

        _run_teleop_loop(mock_robot, mock_teleop, fps=100, duration=None)

        assert mock_teleop.get_action.call_count > 0
        assert mock_robot.send_action.call_count > 0

    def test_loop_duration_limit(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_teleop.get_action.return_value = {"j0": 0.0}

        _ACTIVE_SESSION["running"] = True

        _run_teleop_loop(mock_robot, mock_teleop, fps=100, duration=0.02)

        assert _ACTIVE_SESSION["running"] is False

    def test_loop_handles_exception(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_teleop.get_action.side_effect = RuntimeError("Connection lost")

        _ACTIVE_SESSION["running"] = True

        _run_teleop_loop(mock_robot, mock_teleop, fps=100, duration=None)

        assert _ACTIVE_SESSION["running"] is False

    def test_loop_updates_stats(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_teleop.get_action.return_value = {"j0": 0.0}

        _ACTIVE_SESSION["running"] = True
        _ACTIVE_SESSION["stats"] = {}

        _run_teleop_loop(mock_robot, mock_teleop, fps=200, duration=0.05)

        assert _ACTIVE_SESSION["stats"]["steps"] > 0
        assert _ACTIVE_SESSION["stats"]["duration"] > 0

    def test_loop_sends_action_from_teleop(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        action_data = {"j0": 1.5, "j1": -0.3}
        mock_teleop.get_action.return_value = action_data

        _ACTIVE_SESSION["running"] = True

        _run_teleop_loop(mock_robot, mock_teleop, fps=100, duration=0.02)

        sent_actions = [c.args[0] for c in mock_robot.send_action.call_args_list]
        assert all(a == action_data for a in sent_actions)


# ═════════════════════════════════════════════════════════════════════════════
# _run_record_loop Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRunRecordLoop:
    """Test the background record loop."""

    def test_record_loop_runs_episodes(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_record_session = MagicMock()
        mock_record_session.record_episode.return_value = {}

        mock_ep1 = MagicMock(frames=100, discarded=False)
        mock_ep2 = MagicMock(frames=150, discarded=False)
        mock_record_session._episodes = [mock_ep1, mock_ep2]
        mock_record_session.save_and_push.return_value = {"episodes": 2}

        _ACTIVE_SESSION["running"] = True
        _ACTIVE_SESSION["stats"] = {}

        mock_record_mod = MagicMock()
        mock_record_mod.RecordMode.TELEOP = "teleop"

        with patch.dict("sys.modules", {"strands_robots.record": mock_record_mod}):
            _run_record_loop(mock_robot, mock_teleop, mock_record_session, num_episodes=2)

        assert mock_record_session.record_episode.call_count == 2
        assert _ACTIVE_SESSION["stats"]["episodes_completed"] == 2
        assert _ACTIVE_SESSION["running"] is False

    def test_record_loop_stops_early(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_record_session = MagicMock()

        def stop_on_second_call(*args, **kwargs):
            if mock_record_session.record_episode.call_count >= 1:
                _ACTIVE_SESSION["running"] = False
            return {}

        mock_record_session.record_episode.side_effect = stop_on_second_call
        mock_record_session._episodes = []
        mock_record_session.save_and_push.return_value = {}

        _ACTIVE_SESSION["running"] = True
        _ACTIVE_SESSION["stats"] = {}

        mock_record_mod = MagicMock()
        mock_record_mod.RecordMode.TELEOP = "teleop"

        with patch.dict("sys.modules", {"strands_robots.record": mock_record_mod}):
            _run_record_loop(mock_robot, mock_teleop, mock_record_session, num_episodes=10)

        assert mock_record_session.record_episode.call_count < 10

    def test_record_loop_tracks_total_frames(self):
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        mock_record_session = MagicMock()
        mock_record_session.record_episode.return_value = {}

        mock_ep1 = MagicMock(frames=100, discarded=False)
        mock_ep2 = MagicMock(frames=200, discarded=True)
        mock_ep3 = MagicMock(frames=150, discarded=False)
        mock_record_session._episodes = [mock_ep1, mock_ep2, mock_ep3]
        mock_record_session.save_and_push.return_value = {}

        _ACTIVE_SESSION["running"] = True
        _ACTIVE_SESSION["stats"] = {}

        mock_record_mod = MagicMock()
        mock_record_mod.RecordMode.TELEOP = "teleop"

        with patch.dict("sys.modules", {"strands_robots.record": mock_record_mod}):
            _run_record_loop(mock_robot, mock_teleop, mock_record_session, num_episodes=3)

        assert _ACTIVE_SESSION["stats"]["total_frames"] == 250


# ═════════════════════════════════════════════════════════════════════════════
# Edge Cases / Error Handling
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_teleoperator_general_exception(self):
        with patch("strands_robots.tools.teleoperator._make_robot", side_effect=Exception("Unexpected!")):
            result = teleoperator(action="teleop")
            assert result["status"] == "error"
            assert "Unexpected!" in result["content"][0]["text"]
