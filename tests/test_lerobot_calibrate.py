"""Tests for strands_robots/tools/lerobot_calibrate.py — calibration management.

All filesystem operations use tmp_path. No hardware or lerobot required.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

# CROSS_PR_SKIP: Tests require extended lerobot_calibrate API (cross-PR)
try:
    _cross_pr_check = hasattr(
        __import__("strands_robots.tools.lerobot_calibrate", fromlist=["_detect_port"]), "_detect_port"
    )
    if not _cross_pr_check:
        raise AttributeError
except (ImportError, AttributeError, FileNotFoundError, OSError):
    import pytest as _skip_pytest

    _skip_pytest.skip("Tests require extended lerobot_calibrate API (cross-PR)", allow_module_level=True)


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
    from strands_robots.tools.lerobot_calibrate import (
        LeRobotCalibrationManager,
        lerobot_calibrate,
    )
except (ImportError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #11 (agent-tools)", allow_module_level=True)

_requires = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


@pytest.fixture
def calib_dir(tmp_path):
    """Create a realistic calibration directory structure."""
    base = tmp_path / "calibrations"
    # Create robot calibrations
    robot_dir = base / "robots" / "so101_follower"
    robot_dir.mkdir(parents=True)
    (robot_dir / "orange_arm.json").write_text(
        json.dumps(
            {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1024, "range_min": 0, "range_max": 4095},
                "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 0},
                "elbow_flex": {"id": 3, "drive_mode": 1, "homing_offset": 512},
                "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 0},
                "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0},
                "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0},
            }
        )
    )
    (robot_dir / "blue_arm.json").write_text(
        json.dumps(
            {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0},
                "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0},
            }
        )
    )

    # Create teleop calibrations
    teleop_dir = base / "teleoperators" / "so101_leader"
    teleop_dir.mkdir(parents=True)
    (teleop_dir / "leader_1.json").write_text(
        json.dumps(
            {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 100},
            }
        )
    )

    return base


# ═════════════════════════════════════════════════════════════════════════════
# CalibrationManager Unit Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCalibrationManager:
    """Test the LeRobotCalibrationManager class directly."""

    def test_init_creates_dirs(self, tmp_path):
        LeRobotCalibrationManager(tmp_path / "cal")
        assert (tmp_path / "cal").exists()
        assert (tmp_path / "cal" / "teleoperators").exists()
        assert (tmp_path / "cal" / "robots").exists()

    def test_get_structure_empty(self, tmp_path):
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        structure = mgr.get_calibration_structure()
        assert structure == {"teleoperators": {}, "robots": {}}

    def test_get_structure_with_data(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        structure = mgr.get_calibration_structure()
        assert "so101_follower" in structure["robots"]
        assert sorted(structure["robots"]["so101_follower"]) == ["blue_arm", "orange_arm"]
        assert "so101_leader" in structure["teleoperators"]

    def test_calibration_exists(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        assert mgr.calibration_exists("robots", "so101_follower", "orange_arm")
        assert not mgr.calibration_exists("robots", "so101_follower", "nonexistent")

    def test_load_calibration(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        data = mgr.load_calibration("robots", "so101_follower", "orange_arm")
        assert data is not None
        assert "shoulder_pan" in data
        assert data["shoulder_pan"]["id"] == 1

    def test_load_calibration_not_found(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        data = mgr.load_calibration("robots", "so101_follower", "nonexistent")
        assert data is None

    def test_save_calibration(self, tmp_path):
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        data = {"motor_1": {"id": 1, "offset": 0}}
        success = mgr.save_calibration("robots", "test_robot", "test_id", data)
        assert success
        # Verify file
        path = mgr.get_calibration_path("robots", "test_robot", "test_id")
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_delete_calibration(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        assert mgr.calibration_exists("robots", "so101_follower", "blue_arm")
        success = mgr.delete_calibration("robots", "so101_follower", "blue_arm")
        assert success
        assert not mgr.calibration_exists("robots", "so101_follower", "blue_arm")

    def test_delete_nonexistent(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        success = mgr.delete_calibration("robots", "so101_follower", "nope")
        assert not success

    def test_get_calibration_info(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        info = mgr.get_calibration_info("robots", "so101_follower", "orange_arm")
        assert info is not None
        assert info["device_type"] == "robots"
        assert info["device_model"] == "so101_follower"
        assert info["device_id"] == "orange_arm"
        assert info["motor_count"] == 6
        assert "shoulder_pan" in info["motor_names"]

    def test_get_calibration_info_not_found(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        info = mgr.get_calibration_info("robots", "so101_follower", "nope")
        assert info is None

    def test_search_calibrations_all(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        results = mgr.search_calibrations()
        assert len(results) == 3  # orange_arm, blue_arm, leader_1

    def test_search_by_query(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        results = mgr.search_calibrations("orange")
        assert len(results) == 1
        assert results[0]["device_id"] == "orange_arm"

    def test_search_by_device_type(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        results = mgr.search_calibrations(device_type="teleoperators")
        assert len(results) == 1
        assert results[0]["device_type"] == "teleoperators"

    def test_search_no_results(self, calib_dir):
        mgr = LeRobotCalibrationManager(calib_dir)
        results = mgr.search_calibrations("nonexistent_query_xyz")
        assert len(results) == 0

    def test_backup_calibrations(self, calib_dir, tmp_path):
        mgr = LeRobotCalibrationManager(calib_dir)
        backup_path = tmp_path / "my_backup"
        success, message, count = mgr.backup_calibrations(backup_path)
        assert success
        assert count == 3
        # Verify manifest
        manifest = json.loads((backup_path / "backup_manifest.json").read_text())
        assert manifest["files_count"] == 3

    def test_backup_filtered(self, calib_dir, tmp_path):
        mgr = LeRobotCalibrationManager(calib_dir)
        backup_path = tmp_path / "filtered_backup"
        success, _, count = mgr.backup_calibrations(backup_path, device_type="robots", device_id="orange_arm")
        assert success
        assert count == 1

    def test_restore_calibrations(self, calib_dir, tmp_path):
        mgr = LeRobotCalibrationManager(calib_dir)
        # First backup
        backup_path = tmp_path / "backup"
        mgr.backup_calibrations(backup_path)

        # Create a fresh manager (different base)
        new_base = tmp_path / "restored"
        new_mgr = LeRobotCalibrationManager(new_base)
        success, msg, count = new_mgr.restore_calibrations(backup_path)
        assert success
        assert count == 3

    def test_restore_no_overwrite(self, calib_dir, tmp_path):
        mgr = LeRobotCalibrationManager(calib_dir)
        backup_path = tmp_path / "backup"
        mgr.backup_calibrations(backup_path)
        # Restore to same base — should skip existing files
        success, msg, count = mgr.restore_calibrations(backup_path, overwrite=False)
        assert success
        assert count == 0  # All files exist, none overwritten

    def test_restore_with_overwrite(self, calib_dir, tmp_path):
        mgr = LeRobotCalibrationManager(calib_dir)
        backup_path = tmp_path / "backup"
        mgr.backup_calibrations(backup_path)
        success, msg, count = mgr.restore_calibrations(backup_path, overwrite=True)
        assert success
        assert count == 3

    def test_restore_nonexistent_dir(self, tmp_path):
        mgr = LeRobotCalibrationManager(tmp_path / "cal")
        success, msg, count = mgr.restore_calibrations(tmp_path / "nonexistent_backup")
        assert not success
        assert count == 0


# ═════════════════════════════════════════════════════════════════════════════
# lerobot_calibrate Tool Tests — Action Dispatch
# ═════════════════════════════════════════════════════════════════════════════


class TestToolActionDispatch:
    """Test the @tool function action routing."""

    def test_unknown_action(self, tmp_path):
        result = lerobot_calibrate(action="bogus", base_path=str(tmp_path / "cal"))
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_list_empty(self, tmp_path):
        result = lerobot_calibrate(action="list", base_path=str(tmp_path / "cal"))
        assert result["status"] == "success"
        assert result["count"] == 0
        assert "No calibration" in result["content"][0]["text"]

    def test_list_with_data(self, calib_dir):
        result = lerobot_calibrate(action="list", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] == 3

    def test_list_filtered_by_type(self, calib_dir):
        result = lerobot_calibrate(action="list", device_type="robots", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] == 2  # orange_arm + blue_arm

    def test_list_filtered_by_model(self, calib_dir):
        result = lerobot_calibrate(action="list", device_model="so101_leader", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] == 1


class TestToolViewAction:
    """Test view action."""

    def test_view_missing_params(self, calib_dir):
        result = lerobot_calibrate(action="view", base_path=str(calib_dir))
        assert result["status"] == "error"
        assert "requires" in result["content"][0]["text"]

    def test_view_not_found(self, calib_dir):
        result = lerobot_calibrate(
            action="view",
            device_type="robots",
            device_model="so101_follower",
            device_id="nonexistent",
            base_path=str(calib_dir),
        )
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_view_success(self, calib_dir):
        result = lerobot_calibrate(
            action="view",
            device_type="robots",
            device_model="so101_follower",
            device_id="orange_arm",
            base_path=str(calib_dir),
        )
        assert result["status"] == "success"
        assert "shoulder_pan" in result["content"][0]["text"]
        assert result["calibration_info"]["motor_count"] == 6


class TestToolSearchAction:
    """Test search action."""

    def test_search_all(self, calib_dir):
        result = lerobot_calibrate(action="search", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] == 3

    def test_search_by_query(self, calib_dir):
        result = lerobot_calibrate(action="search", query="leader", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] >= 1

    def test_search_no_results(self, calib_dir):
        result = lerobot_calibrate(action="search", query="xyz123", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["count"] == 0


class TestToolBackupRestore:
    """Test backup and restore actions."""

    def test_backup(self, calib_dir, tmp_path):
        backup_dir = str(tmp_path / "backup")
        result = lerobot_calibrate(action="backup", output_dir=backup_dir, base_path=str(calib_dir))
        assert result["status"] == "success"
        assert result["files_count"] == 3

    def test_restore_no_dir(self, calib_dir):
        result = lerobot_calibrate(action="restore", base_path=str(calib_dir))
        assert result["status"] == "error"
        assert "requires" in result["content"][0]["text"]

    def test_restore(self, calib_dir, tmp_path):
        # Backup first
        backup_dir = str(tmp_path / "backup")
        lerobot_calibrate(action="backup", output_dir=backup_dir, base_path=str(calib_dir))

        # Restore to new location
        new_base = str(tmp_path / "new_cal")
        result = lerobot_calibrate(action="restore", backup_dir=backup_dir, base_path=new_base)
        assert result["status"] == "success"
        assert result["restored_count"] == 3


class TestToolDeleteAction:
    """Test delete action."""

    def test_delete_missing_params(self, calib_dir):
        result = lerobot_calibrate(action="delete", base_path=str(calib_dir))
        assert result["status"] == "error"

    def test_delete_not_found(self, calib_dir):
        result = lerobot_calibrate(
            action="delete",
            device_type="robots",
            device_model="so101_follower",
            device_id="nope",
            base_path=str(calib_dir),
        )
        assert result["status"] == "error"

    def test_delete_success(self, calib_dir):
        result = lerobot_calibrate(
            action="delete",
            device_type="robots",
            device_model="so101_follower",
            device_id="blue_arm",
            base_path=str(calib_dir),
        )
        assert result["status"] == "success"
        assert "deleted" in result["content"][0]["text"].lower()
        # Verify file is gone
        assert not (calib_dir / "robots" / "so101_follower" / "blue_arm.json").exists()


class TestToolAnalyzeAction:
    """Test analyze action."""

    def test_analyze_empty(self, tmp_path):
        result = lerobot_calibrate(action="analyze", base_path=str(tmp_path / "cal"))
        assert result["status"] == "success"
        assert "No calibrations" in result["content"][0]["text"]

    def test_analyze_with_data(self, calib_dir):
        result = lerobot_calibrate(action="analyze", base_path=str(calib_dir))
        assert result["status"] == "success"
        analysis = result["analysis"]
        assert analysis["total_calibrations"] == 3
        assert analysis["device_counts"]["robots"] == 2
        assert analysis["device_counts"]["teleoperators"] == 1


class TestToolPathAction:
    """Test path action."""

    def test_path_base(self, calib_dir):
        result = lerobot_calibrate(action="path", base_path=str(calib_dir))
        assert result["status"] == "success"
        assert "base_path" in result

    def test_path_specific_exists(self, calib_dir):
        result = lerobot_calibrate(
            action="path",
            device_type="robots",
            device_model="so101_follower",
            device_id="orange_arm",
            base_path=str(calib_dir),
        )
        assert result["status"] == "success"
        assert result["exists"] is True

    def test_path_specific_not_exists(self, calib_dir):
        result = lerobot_calibrate(
            action="path",
            device_type="robots",
            device_model="so101_follower",
            device_id="nope",
            base_path=str(calib_dir),
        )
        assert result["status"] == "success"
        assert result["exists"] is False


class TestToolRunAction:
    """Test run calibration action (requires mocked lerobot)."""

    def test_run_missing_params(self, calib_dir):
        result = lerobot_calibrate(action="run", base_path=str(calib_dir))
        assert result["status"] == "error"
        assert "requires" in result["content"][0]["text"]

    def test_run_invalid_device_type(self, calib_dir):
        result = lerobot_calibrate(action="run", device_type="invalid", device_model="test", base_path=str(calib_dir))
        assert result["status"] == "error"

    def test_run_lerobot_not_installed(self, calib_dir):
        with patch.dict("sys.modules", {"lerobot.robots": None}):
            with patch("builtins.__import__", side_effect=ImportError("no lerobot")):
                result = lerobot_calibrate(
                    action="run", device_type="robots", device_model="so101_follower", base_path=str(calib_dir)
                )
                assert result["status"] == "error"
                assert "lerobot" in result["content"][0]["text"].lower()
