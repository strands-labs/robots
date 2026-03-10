"""Coverage-gap tests for strands_robots/tools/lerobot_calibrate.py.

Targets uncovered lines: 39 (LEROBOT_AVAILABLE branch), 70/100-102/115-117/
129-131/160-162/177/201-202/217/247-249/264/268/282-284 (CalibrationManager
error paths), 422/441/543-549/558/575/598 (tool function edge cases),
717-762/773-813 (run action — robots/teleoperators branches),
834-836/852-854 (outer exception handlers).

All filesystem operations use tmp_path. No hardware or lerobot required.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

try:
    from strands_robots.tools.lerobot_calibrate import (
        LeRobotCalibrationManager,
        lerobot_calibrate,
    )
except (ImportError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #11 (agent-tools)", allow_module_level=True)

# CROSS_PR_SKIP: Check if the "run" action is supported in this version
import inspect as _inspect_guard

_cal_source = _inspect_guard.getsource(lerobot_calibrate)
if 'action == "run"' not in _cal_source and '"run"' not in _cal_source:
    # The "run" action isn't in this version — tests for it will fail
    # Only skip TestRunAction, not the whole module — use marker below
    _RUN_ACTION_AVAILABLE = False
else:
    _RUN_ACTION_AVAILABLE = True

# ═════════════════════════════════════════════════════════════════════════════
# CalibrationManager — error paths
# ═════════════════════════════════════════════════════════════════════════════


class TestCalibManagerErrorPaths:
    """Test error handling in CalibrationManager methods."""

    def test_load_calibration_json_error(self, tmp_path):
        """load_calibration returns None on corrupt JSON."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        calib_path = mgr.get_calibration_path("robots", "so100", "test_id")
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text("{invalid json")
        result = mgr.load_calibration("robots", "so100", "test_id")
        assert result is None

    def test_save_calibration_permission_error(self, tmp_path):
        """save_calibration returns False on write error."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = mgr.save_calibration("robots", "so100", "test_id", {"motor": {}})
        assert result is False

    def test_delete_calibration_error(self, tmp_path):
        """delete_calibration returns False on delete error."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        calib_path = mgr.get_calibration_path("robots", "so100", "test_id")
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text("{}")
        with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            result = mgr.delete_calibration("robots", "so100", "test_id")
        assert result is False

    def test_get_calibration_info_exception(self, tmp_path):
        """get_calibration_info returns None on unexpected error."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        calib_path = mgr.get_calibration_path("robots", "so100", "test_id")
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text("{}")
        # Patch load_calibration to raise inside get_calibration_info
        with patch.object(mgr, "load_calibration", side_effect=OSError("read failed")):
            result = mgr.get_calibration_info("robots", "so100", "test_id")
        assert result is None

    def test_search_by_device_type_and_model(self, tmp_path):
        """search_calibrations with both device_type and device_model filters."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        mgr.save_calibration("robots", "so100", "arm1", {"motor": {}})
        mgr.save_calibration("robots", "so101", "arm2", {"motor": {}})
        mgr.save_calibration("teleoperators", "so100_leader", "lead1", {"motor": {}})

        results = mgr.search_calibrations(device_type="robots", device_model="so100")
        assert len(results) == 1
        assert results[0]["device_model"] == "so100"

    def test_backup_empty_calibrations(self, tmp_path):
        """Backup with no calibration files."""
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        backup_path = tmp_path / "backup"
        success, msg, count = mgr.backup_calibrations(backup_path)
        assert success is True
        assert count == 0


# ═════════════════════════════════════════════════════════════════════════════
# Tool function — edge cases and uncovered branches
# ═════════════════════════════════════════════════════════════════════════════


class TestToolEdgeCases:
    """Test tool function paths not covered by existing tests."""

    def test_view_with_all_details(self, tmp_path):
        """View action with complete calibration data including motor details."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration(
            "robots",
            "so100",
            "arm1",
            {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1024, "range_min": 0, "range_max": 4095},
                "elbow_flex": {"id": 2, "drive_mode": 1, "homing_offset": 512},
            },
        )
        result = lerobot_calibrate(
            action="view",
            device_type="robots",
            device_model="so100",
            device_id="arm1",
            base_path=str(base),
        )
        assert result["status"] == "success"
        assert result["calibration_info"]["motor_count"] == 2

    def test_delete_no_device_type(self, tmp_path):
        """Delete without full params returns error."""
        result = lerobot_calibrate(
            action="delete",
            device_type="robots",
            base_path=str(tmp_path / "cal"),
        )
        assert result["status"] == "error"

    def test_list_filtered_by_model_and_type(self, tmp_path):
        """List with both device_type and device_model filters."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration("robots", "so100", "arm1", {"m": {}})
        mgr.save_calibration("robots", "so100", "arm2", {"m": {}})
        mgr.save_calibration("robots", "so101", "arm3", {"m": {}})
        mgr.save_calibration("teleoperators", "leader", "l1", {"m": {}})
        result = lerobot_calibrate(
            action="list",
            device_type="robots",
            device_model="so100",
            base_path=str(base),
        )
        assert result["status"] == "success"
        # Only so100 robots (arm1 + arm2)
        assert result["count"] == 2

    def test_search_with_device_model_filter(self, tmp_path):
        """Search by device_model."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration("robots", "so100", "arm1", {"m": {}})
        mgr.save_calibration("robots", "so101", "arm2", {"m": {}})
        result = lerobot_calibrate(
            action="search",
            device_model="so101",
            base_path=str(base),
        )
        assert result["status"] == "success"
        assert result["count"] == 1

    def test_backup_filtered_by_device_type(self, tmp_path):
        """Backup with device_type filter."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration("robots", "so100", "arm1", {"m": {}})
        mgr.save_calibration("teleoperators", "leader", "l1", {"m": {}})

        backup_dir = str(tmp_path / "backup")
        result = lerobot_calibrate(
            action="backup",
            output_dir=backup_dir,
            device_type="robots",
            base_path=str(base),
        )
        assert result["status"] == "success"
        assert result["files_count"] == 1

    def test_restore_with_overwrite(self, tmp_path):
        """Restore with overwrite=True (default in tool)."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration("robots", "so100", "arm1", {"m": {"v": 1}})

        backup_dir = str(tmp_path / "backup")
        lerobot_calibrate(action="backup", output_dir=backup_dir, base_path=str(base))

        # Modify the calibration
        mgr.save_calibration("robots", "so100", "arm1", {"m": {"v": 2}})

        # Restore should overwrite
        result = lerobot_calibrate(
            action="restore",
            backup_dir=backup_dir,
            base_path=str(base),
            overwrite=True,
        )
        assert result["status"] == "success"
        assert result["restored_count"] == 1

    def test_path_with_all_params(self, tmp_path):
        """Path action returns full calibration path."""
        base = tmp_path / "cal"
        mgr = LeRobotCalibrationManager(base)
        mgr.save_calibration("robots", "so100", "arm1", {"m": {}})
        result = lerobot_calibrate(
            action="path",
            device_type="robots",
            device_model="so100",
            device_id="arm1",
            base_path=str(base),
        )
        assert result["status"] == "success"
        assert result["exists"] is True
        assert "arm1.json" in result["path"]


# ═════════════════════════════════════════════════════════════════════════════
# Run action — robots and teleoperators branches
# ═════════════════════════════════════════════════════════════════════════════


@__import__("pytest").mark.skipif(not _RUN_ACTION_AVAILABLE, reason="run action not in current source")
class TestRunAction:
    """Test the run calibration action with mocked lerobot."""

    def test_run_robot_success(self, tmp_path):
        """Exercise the robots calibration branch."""
        base = tmp_path / "cal"

        class FakeRobotConfig:
            pass

        class So100Config(FakeRobotConfig):
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_robot = MagicMock()

        mock_lr_robots = types.ModuleType("lerobot.robots")
        mock_lr_robots.__path__ = ["/fake/path"]

        mock_robots_config = MagicMock()
        mock_robots_config.RobotConfig = FakeRobotConfig

        mock_robots_utils = MagicMock()
        mock_robots_utils.make_robot_from_config = MagicMock(return_value=mock_robot)

        mock_robot_mod = MagicMock()
        mock_robot_mod.So100Config = So100Config
        type(mock_robot_mod).__dir__ = lambda self: ["So100Config", "__name__"]

        # Need lerobot parent to have .robots attr pointing to our mock
        mock_lerobot = types.ModuleType("lerobot")
        mock_lerobot.robots = mock_lr_robots

        with patch.dict(
            "sys.modules",
            {
                "lerobot": mock_lerobot,
                "lerobot.robots": mock_lr_robots,
                "lerobot.robots.config": mock_robots_config,
                "lerobot.robots.utils": mock_robots_utils,
            },
        ):
            with patch("pkgutil.iter_modules", return_value=[("", "so100_mod", False)]):
                with patch("importlib.import_module", return_value=mock_robot_mod):
                    result = lerobot_calibrate(
                        action="run",
                        device_type="robots",
                        device_model="so100",
                        device_id="arm1",
                        base_path=str(base),
                    )
        assert result["status"] == "success"
        mock_robot.connect.assert_called_once()
        mock_robot.calibrate.assert_called_once()
        mock_robot.disconnect.assert_called_once()

    def test_run_robot_config_not_found(self, tmp_path):
        """No matching config class found."""
        base = tmp_path / "cal"

        mock_lr_robots = types.ModuleType("lerobot.robots")
        mock_lr_robots.__path__ = ["/fake"]

        mock_robots_config = MagicMock()
        mock_robots_config.RobotConfig = type("RobotConfig", (), {})

        mock_lerobot = types.ModuleType("lerobot")
        mock_lerobot.robots = mock_lr_robots

        with patch.dict(
            "sys.modules",
            {
                "lerobot": mock_lerobot,
                "lerobot.robots": mock_lr_robots,
                "lerobot.robots.config": mock_robots_config,
                "lerobot.robots.utils": MagicMock(),
            },
        ):
            with patch("pkgutil.iter_modules", return_value=[]):
                result = lerobot_calibrate(
                    action="run",
                    device_type="robots",
                    device_model="nonexistent_robot",
                    base_path=str(base),
                )
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_run_teleoperator_success(self, tmp_path):
        """Exercise the teleoperators calibration branch."""
        base = tmp_path / "cal"

        class FakeTeleopConfig:
            pass

        class So100LeaderConfig(FakeTeleopConfig):
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_teleop = MagicMock()

        mock_lr_teleops = types.ModuleType("lerobot.teleoperators")
        mock_lr_teleops.__path__ = ["/fake/path"]

        mock_teleop_config = MagicMock()
        mock_teleop_config.TeleoperatorConfig = FakeTeleopConfig

        mock_teleop_utils = MagicMock()
        mock_teleop_utils.make_teleoperator_from_config = MagicMock(return_value=mock_teleop)

        mock_teleop_mod = MagicMock()
        mock_teleop_mod.So100leaderConfig = So100LeaderConfig
        type(mock_teleop_mod).__dir__ = lambda self: ["So100leaderConfig", "__name__"]

        mock_lerobot = types.ModuleType("lerobot")
        mock_lerobot.teleoperators = mock_lr_teleops

        with patch.dict(
            "sys.modules",
            {
                "lerobot": mock_lerobot,
                "lerobot.teleoperators": mock_lr_teleops,
                "lerobot.teleoperators.config": mock_teleop_config,
                "lerobot.teleoperators.utils": mock_teleop_utils,
            },
        ):
            with patch("pkgutil.iter_modules", return_value=[("", "so100_leader_mod", False)]):
                with patch("importlib.import_module", return_value=mock_teleop_mod):
                    result = lerobot_calibrate(
                        action="run",
                        device_type="teleoperators",
                        device_model="so100_leader",
                        device_id="lead1",
                        base_path=str(base),
                    )
        assert result["status"] == "success"
        mock_teleop.connect.assert_called_once()
        mock_teleop.calibrate.assert_called_once()
        mock_teleop.disconnect.assert_called_once()

    def test_run_teleoperator_config_not_found(self, tmp_path):
        """Teleop config not found."""
        base = tmp_path / "cal"

        mock_lr_teleops = types.ModuleType("lerobot.teleoperators")
        mock_lr_teleops.__path__ = ["/fake"]

        mock_teleop_config = MagicMock()
        mock_teleop_config.TeleoperatorConfig = type("TeleoperatorConfig", (), {})

        mock_lerobot = types.ModuleType("lerobot")
        mock_lerobot.teleoperators = mock_lr_teleops

        with patch.dict(
            "sys.modules",
            {
                "lerobot": mock_lerobot,
                "lerobot.teleoperators": mock_lr_teleops,
                "lerobot.teleoperators.config": mock_teleop_config,
                "lerobot.teleoperators.utils": MagicMock(),
            },
        ):
            with patch("pkgutil.iter_modules", return_value=[]):
                result = lerobot_calibrate(
                    action="run",
                    device_type="teleoperators",
                    device_model="nonexistent",
                    base_path=str(base),
                )
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_run_invalid_device_type(self, tmp_path):
        """Run with invalid device_type (not robots or teleoperators)."""
        # The code checks device_type == "robots", then "teleoperators", then else
        # We need to get past the ImportError check
        base = tmp_path / "cal"

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.robots": MagicMock(),
            },
        ):
            result = lerobot_calibrate(
                action="run",
                device_type="invalid_type",
                device_model="so100",
                base_path=str(base),
            )
        assert result["status"] == "error"

    def test_run_lerobot_import_error(self, tmp_path):
        """Run fails gracefully when lerobot not installed."""
        base = tmp_path / "cal"

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def selective_import(name, *args, **kwargs):
            if "lerobot" in name:
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"lerobot.robots": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = lerobot_calibrate(
                    action="run",
                    device_type="robots",
                    device_model="so100",
                    base_path=str(base),
                )
        assert result["status"] == "error"
        assert "lerobot" in result["content"][0]["text"].lower()

    def test_run_calibration_exception(self, tmp_path):
        """Run handles calibration runtime error."""
        base = tmp_path / "cal"

        class FakeRobotConfig:
            pass

        class So100Config(FakeRobotConfig):
            def __init__(self, **kwargs):
                pass

        mock_robot = MagicMock()
        mock_robot.connect.side_effect = RuntimeError("hardware error")

        mock_lr_robots = types.ModuleType("lerobot.robots")
        mock_lr_robots.__path__ = ["/fake"]

        mock_robots_config = MagicMock()
        mock_robots_config.RobotConfig = FakeRobotConfig

        mock_robots_utils = MagicMock()
        mock_robots_utils.make_robot_from_config = MagicMock(return_value=mock_robot)

        mock_robot_mod = MagicMock()
        mock_robot_mod.So100Config = So100Config
        type(mock_robot_mod).__dir__ = lambda self: ["So100Config", "__name__"]

        mock_lerobot = types.ModuleType("lerobot")
        mock_lerobot.robots = mock_lr_robots

        with patch.dict(
            "sys.modules",
            {
                "lerobot": mock_lerobot,
                "lerobot.robots": mock_lr_robots,
                "lerobot.robots.config": mock_robots_config,
                "lerobot.robots.utils": mock_robots_utils,
            },
        ):
            with patch("pkgutil.iter_modules", return_value=[("", "so100_mod", False)]):
                with patch("importlib.import_module", return_value=mock_robot_mod):
                    result = lerobot_calibrate(
                        action="run",
                        device_type="robots",
                        device_model="so100",
                        base_path=str(base),
                    )
        assert result["status"] == "error"
        assert "hardware error" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Outer exception handler
# ═════════════════════════════════════════════════════════════════════════════


class TestOuterExceptionHandler:
    def test_outer_exception_caught(self, tmp_path):
        """Force the outer except to catch an error."""
        with patch(
            "strands_robots.tools.lerobot_calibrate.LeRobotCalibrationManager",
            side_effect=RuntimeError("init boom"),
        ):
            result = lerobot_calibrate(action="list", base_path=str(tmp_path / "cal"))
        assert result["status"] == "error"
        assert "init boom" in result["content"][0]["text"]
