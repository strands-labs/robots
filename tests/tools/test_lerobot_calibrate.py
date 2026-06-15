"""Behavior tests for the ``lerobot_calibrate`` agent tool.

Covers the calibration-management contract end to end against a real
temporary calibration tree (no hardware, no network): the
:class:`LeRobotCalibrationManager` filesystem operations and every
documented tool action (``list``, ``view``, ``search``, ``backup``,
``restore``, ``delete``, ``analyze``, ``path``) plus the error paths.

All tests drive the public ``base_path`` parameter at a ``tmp_path`` so
they assert on the tool's documented behavior (status, counts, returned
structures, round-trips) rather than implementation details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.lerobot_calibrate import (
    LeRobotCalibrationManager,
    lerobot_calibrate,
)


def _motor(idx: int) -> dict[str, int]:
    """Return a single LeRobot motor calibration record."""
    return {
        "id": idx,
        "drive_mode": 0,
        "homing_offset": 100 + idx,
        "range_min": 0,
        "range_max": 4095,
    }


def _calib(n_motors: int = 3) -> dict[str, dict[str, int]]:
    """Return a calibration payload with ``n_motors`` named motors."""
    names = ["shoulder", "elbow", "wrist", "gripper", "pan", "tilt"]
    return {names[i]: _motor(i + 1) for i in range(n_motors)}


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    """A calibration tree with one teleoperator and two robot calibrations."""
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("teleoperators", "so101_leader", "blue_arm", _calib(6))
    mgr.save_calibration("robots", "so101_follower", "orange_arm", _calib(6))
    mgr.save_calibration("robots", "so101_follower", "green_arm", _calib(3))
    return tmp_path


# --- LeRobotCalibrationManager filesystem behavior -------------------------


def test_manager_creates_directory_tree(tmp_path: Path) -> None:
    """Constructing the manager materializes base/teleop/robot directories."""
    mgr = LeRobotCalibrationManager(tmp_path / "calib")
    assert mgr.teleop_path.is_dir()
    assert mgr.robot_path.is_dir()


def test_save_load_round_trip(tmp_path: Path) -> None:
    """A saved calibration loads back identically."""
    mgr = LeRobotCalibrationManager(tmp_path)
    payload = _calib(4)
    assert mgr.save_calibration("robots", "koch_follower", "k1", payload) is True
    assert mgr.load_calibration("robots", "koch_follower", "k1") == payload


def test_load_missing_returns_none(tmp_path: Path) -> None:
    """Loading an absent calibration yields None rather than raising."""
    mgr = LeRobotCalibrationManager(tmp_path)
    assert mgr.load_calibration("robots", "nope", "missing") is None


def test_load_corrupt_json_returns_none(tmp_path: Path) -> None:
    """A calibration file with invalid JSON loads as None, not an exception."""
    mgr = LeRobotCalibrationManager(tmp_path)
    path = mgr.get_calibration_path("robots", "so101_follower", "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert mgr.load_calibration("robots", "so101_follower", "broken") is None


def test_exists_and_delete(tmp_path: Path) -> None:
    """exists() tracks presence and delete() removes the file once."""
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("robots", "so101_follower", "x", _calib(2))
    assert mgr.calibration_exists("robots", "so101_follower", "x") is True
    assert mgr.delete_calibration("robots", "so101_follower", "x") is True
    assert mgr.calibration_exists("robots", "so101_follower", "x") is False
    # Deleting a now-absent file reports failure.
    assert mgr.delete_calibration("robots", "so101_follower", "x") is False


def test_get_calibration_info_includes_motor_metadata(tmp_path: Path) -> None:
    """info() surfaces motor count/names and the parsed data."""
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("robots", "so101_follower", "arm", _calib(3))
    info = mgr.get_calibration_info("robots", "so101_follower", "arm")
    assert info is not None
    assert info["motor_count"] == 3
    assert info["motor_names"] == ["shoulder", "elbow", "wrist"]
    assert info["size_bytes"] > 0


def test_get_calibration_info_missing_returns_none(tmp_path: Path) -> None:
    """info() for an absent calibration returns None."""
    mgr = LeRobotCalibrationManager(tmp_path)
    assert mgr.get_calibration_info("robots", "m", "absent") is None


def test_structure_groups_by_type_and_model(populated: Path) -> None:
    """The structure groups calibration ids under type/model, sorted."""
    mgr = LeRobotCalibrationManager(populated)
    structure = mgr.get_calibration_structure()
    assert structure["teleoperators"]["so101_leader"] == ["blue_arm"]
    assert structure["robots"]["so101_follower"] == ["green_arm", "orange_arm"]


def test_search_matches_substring_and_filters(populated: Path) -> None:
    """search() matches a case-insensitive substring and honors filters."""
    mgr = LeRobotCalibrationManager(populated)
    hits = mgr.search_calibrations(query="orange")
    assert [h["device_id"] for h in hits] == ["orange_arm"]
    # An empty query with a device_type filter returns only that type.
    only_teleop = mgr.search_calibrations(query="", device_type="teleoperators")
    assert {h["device_type"] for h in only_teleop} == {"teleoperators"}
    # A device_model filter narrows further.
    followers = mgr.search_calibrations(query="", device_model="so101_follower")
    assert {h["device_id"] for h in followers} == {"green_arm", "orange_arm"}


# --- backup / restore round-trip -------------------------------------------


def test_backup_then_restore_round_trip(populated: Path, tmp_path: Path) -> None:
    """Backup copies files + manifest; restore rebuilds them in a clean tree."""
    src = LeRobotCalibrationManager(populated)
    out = tmp_path / "bk"
    ok, location, count = src.backup_calibrations(output_dir=out)
    assert ok is True
    assert count == 3
    assert (Path(location) / "backup_manifest.json").is_file()

    dest = LeRobotCalibrationManager(tmp_path / "fresh")
    ok2, _msg, restored = dest.restore_calibrations(Path(location))
    assert ok2 is True
    assert restored == 3
    assert dest.load_calibration("robots", "so101_follower", "orange_arm") == _calib(6)


def test_backup_filters_by_device_type(populated: Path, tmp_path: Path) -> None:
    """A device_type filter only backs up matching calibrations."""
    mgr = LeRobotCalibrationManager(populated)
    ok, _location, count = mgr.backup_calibrations(output_dir=tmp_path / "robots_only", device_type="robots")
    assert ok is True
    assert count == 2


def test_restore_missing_directory_reports_error(tmp_path: Path) -> None:
    """Restoring from a non-existent backup dir reports failure, not a crash."""
    mgr = LeRobotCalibrationManager(tmp_path)
    ok, message, count = mgr.restore_calibrations(tmp_path / "does_not_exist")
    assert ok is False
    assert count == 0
    assert "not found" in message.lower()


def test_restore_skips_existing_without_overwrite(populated: Path, tmp_path: Path) -> None:
    """Restore leaves existing calibrations untouched unless overwrite=True."""
    src = LeRobotCalibrationManager(populated)
    _ok, location, _count = src.backup_calibrations(output_dir=tmp_path / "bk")

    # Pre-seed the destination with a different payload for one id.
    dest = LeRobotCalibrationManager(tmp_path / "dest")
    dest.save_calibration("robots", "so101_follower", "orange_arm", _calib(1))

    _ok2, _msg, restored = dest.restore_calibrations(Path(location), overwrite=False)
    # The pre-existing orange_arm is skipped; the other two are restored.
    assert restored == 2
    assert dest.load_calibration("robots", "so101_follower", "orange_arm") == _calib(1)

    _ok3, _msg3, restored_ow = dest.restore_calibrations(Path(location), overwrite=True)
    assert restored_ow == 3
    assert dest.load_calibration("robots", "so101_follower", "orange_arm") == _calib(6)


# --- tool action contract --------------------------------------------------


def test_list_empty_tree_reports_zero(tmp_path: Path) -> None:
    """``list`` on an empty tree succeeds with count 0."""
    result = lerobot_calibrate(action="list", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert result["count"] == 0


def test_list_populated_counts_all(populated: Path) -> None:
    """``list`` enumerates every calibration across both device types."""
    result = lerobot_calibrate(action="list", base_path=str(populated))
    assert result["status"] == "success"
    assert result["count"] == 3
    assert "so101_follower" in result["content"][0]["text"]


def test_list_filtered_by_device_type(populated: Path) -> None:
    """``list`` with device_type='robots' counts only robot calibrations."""
    result = lerobot_calibrate(action="list", device_type="robots", base_path=str(populated))
    assert result["count"] == 2


def test_view_returns_motor_details(populated: Path) -> None:
    """``view`` surfaces the per-motor configuration for a calibration."""
    result = lerobot_calibrate(
        action="view",
        device_type="robots",
        device_model="so101_follower",
        device_id="orange_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    assert result["calibration_info"]["motor_count"] == 6
    assert "shoulder" in result["content"][0]["text"]


def test_view_requires_full_identifier(populated: Path) -> None:
    """``view`` without all three identifiers errors with guidance."""
    result = lerobot_calibrate(action="view", device_type="robots", base_path=str(populated))
    assert result["status"] == "error"
    assert "device_id" in result["content"][0]["text"]


def test_view_missing_calibration_errors(populated: Path) -> None:
    """``view`` of an absent calibration returns an error status."""
    result = lerobot_calibrate(
        action="view",
        device_type="robots",
        device_model="so101_follower",
        device_id="ghost",
        base_path=str(populated),
    )
    assert result["status"] == "error"


def test_search_action_returns_matches(populated: Path) -> None:
    """``search`` returns the matching calibration records."""
    result = lerobot_calibrate(action="search", query="green", base_path=str(populated))
    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["device_id"] == "green_arm"


def test_search_no_match_reports_zero(populated: Path) -> None:
    """``search`` with no hits succeeds with an empty result set."""
    result = lerobot_calibrate(action="search", query="zzz", base_path=str(populated))
    assert result["status"] == "success"
    assert result["count"] == 0


def test_backup_action_reports_file_count(populated: Path, tmp_path: Path) -> None:
    """``backup`` copies all calibrations and reports the count."""
    result = lerobot_calibrate(action="backup", output_dir=str(tmp_path / "out"), base_path=str(populated))
    assert result["status"] == "success"
    assert result["files_count"] == 3


def test_restore_action_round_trips(populated: Path, tmp_path: Path) -> None:
    """``restore`` rebuilds calibrations from a prior ``backup``."""
    backup = lerobot_calibrate(action="backup", output_dir=str(tmp_path / "out"), base_path=str(populated))
    dest = tmp_path / "dest"
    result = lerobot_calibrate(action="restore", backup_dir=backup["backup_path"], base_path=str(dest))
    assert result["status"] == "success"
    assert result["restored_count"] == 3


def test_restore_action_requires_backup_dir(tmp_path: Path) -> None:
    """``restore`` without backup_dir errors."""
    result = lerobot_calibrate(action="restore", base_path=str(tmp_path))
    assert result["status"] == "error"
    assert "backup_dir" in result["content"][0]["text"]


def test_delete_action_removes_calibration(populated: Path) -> None:
    """``delete`` removes an existing calibration and reports success."""
    result = lerobot_calibrate(
        action="delete",
        device_type="robots",
        device_model="so101_follower",
        device_id="green_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    assert not LeRobotCalibrationManager(populated).calibration_exists("robots", "so101_follower", "green_arm")


def test_delete_action_missing_errors(populated: Path) -> None:
    """``delete`` of an absent calibration returns an error."""
    result = lerobot_calibrate(
        action="delete",
        device_type="robots",
        device_model="so101_follower",
        device_id="ghost",
        base_path=str(populated),
    )
    assert result["status"] == "error"


def test_delete_action_requires_full_identifier(populated: Path) -> None:
    """``delete`` without the full identifier errors."""
    result = lerobot_calibrate(action="delete", device_type="robots", base_path=str(populated))
    assert result["status"] == "error"


def test_analyze_action_summarizes_statistics(populated: Path) -> None:
    """``analyze`` aggregates counts and per-model motor statistics."""
    result = lerobot_calibrate(action="analyze", base_path=str(populated))
    assert result["status"] == "success"
    analysis: dict[str, Any] = result["analysis"]
    assert analysis["total_calibrations"] == 3
    assert analysis["device_counts"] == {"teleoperators": 1, "robots": 2}
    assert analysis["motor_stats"]["robots/so101_follower"]["max"] == 6


def test_analyze_empty_tree(tmp_path: Path) -> None:
    """``analyze`` on an empty tree succeeds with an empty analysis."""
    result = lerobot_calibrate(action="analyze", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert result["analysis"] == {}


def test_path_action_specific_calibration(populated: Path) -> None:
    """``path`` for a known calibration reports its path and existence."""
    result = lerobot_calibrate(
        action="path",
        device_type="robots",
        device_model="so101_follower",
        device_id="orange_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    assert result["exists"] is True
    assert result["path"].endswith("orange_arm.json")


def test_path_action_base_paths(tmp_path: Path) -> None:
    """``path`` without an identifier reports the base/teleop/robot paths."""
    result = lerobot_calibrate(action="path", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert result["teleop_path"].endswith("teleoperators")
    assert result["robot_path"].endswith("robots")


def test_unknown_action_errors(tmp_path: Path) -> None:
    """An unrecognized action returns an error listing valid actions."""
    result = lerobot_calibrate(action="frobnicate", base_path=str(tmp_path))
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_backup_action_echoes_applied_filters(populated: Path, tmp_path: Path) -> None:
    """``backup`` with filters echoes them back in the summary text."""
    result = lerobot_calibrate(
        action="backup",
        output_dir=str(tmp_path / "filtered"),
        device_type="robots",
        device_model="so101_follower",
        device_id="orange_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "Filters applied" in text
    assert "robots" in text and "so101_follower" in text and "orange_arm" in text


def test_tool_surfaces_validation_errors(tmp_path: Path) -> None:
    """A path-traversal output_dir is rejected and reported as a tool error."""
    result = lerobot_calibrate(action="backup", output_dir="../escape", base_path=str(tmp_path))
    assert result["status"] == "error"
    assert "failed" in result["content"][0]["text"].lower()
