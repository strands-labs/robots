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
from tests.tool_result_contract import tool_json


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


def test_structure_skips_absent_device_type_directory(tmp_path: Path) -> None:
    """Discovery tolerates a missing top-level device-type directory.

    The manager materializes ``teleoperators`` and ``robots`` on construction,
    but a hand-managed tree may lack one. ``get_calibration_structure`` must
    skip the absent directory and still report the other, rather than raising.
    """
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("robots", "so101_follower", "arm1", _calib(3))
    # Remove the (empty) teleoperators directory so the branch that skips a
    # non-existent device-type path is exercised.
    mgr.teleop_path.rmdir()

    structure = mgr.get_calibration_structure()
    assert structure["teleoperators"] == {}
    assert structure["robots"] == {"so101_follower": ["arm1"]}


def test_backup_default_location_is_timestamped_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backup with no explicit output_dir writes to a timestamped default dir."""
    monkeypatch.setattr("strands_robots.tools.lerobot_calibrate.BACKUP_DIR", tmp_path / "default_backups")
    mgr = LeRobotCalibrationManager(tmp_path / "src")
    mgr.save_calibration("robots", "so101_follower", "arm1", _calib(2))

    ok, location, count = mgr.backup_calibrations()
    assert ok is True
    assert count == 1
    dest = Path(location)
    assert dest.parent == tmp_path / "default_backups"
    assert dest.name.startswith("backup_")
    assert (dest / "backup_manifest.json").is_file()


def test_backup_filters_by_device_model(populated: Path, tmp_path: Path) -> None:
    """A device_model filter backs up only matching models across every type."""
    mgr = LeRobotCalibrationManager(populated)
    ok, _location, count = mgr.backup_calibrations(
        output_dir=tmp_path / "followers_only", device_model="so101_follower"
    )
    assert ok is True
    # so101_follower has two calibrations; the so101_leader teleoperator is skipped.
    assert count == 2


def test_restore_ignores_nondirectory_entries(tmp_path: Path) -> None:
    """Restore skips stray files at the model level and restores real dirs."""
    backup = tmp_path / "backup_src"
    (backup / "robots" / "so101_follower").mkdir(parents=True)
    (backup / "robots" / "so101_follower" / "arm1.json").write_text('{"shoulder": {"id": 1}}')
    # A regular file sitting where a model directory would be must be ignored.
    (backup / "robots" / "stray.txt").write_text("not a model directory")

    mgr = LeRobotCalibrationManager(tmp_path / "live")
    ok, _message, restored = mgr.restore_calibrations(backup)
    assert ok is True
    assert restored == 1
    assert mgr.calibration_exists("robots", "so101_follower", "arm1")


# --- tool action contract --------------------------------------------------


def test_list_empty_tree_reports_zero(tmp_path: Path) -> None:
    """``list`` on an empty tree succeeds with count 0."""
    result = lerobot_calibrate(action="list", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert tool_json(result)["count"] == 0


def test_list_populated_counts_all(populated: Path) -> None:
    """``list`` enumerates every calibration across both device types."""
    result = lerobot_calibrate(action="list", base_path=str(populated))
    assert result["status"] == "success"
    assert tool_json(result)["count"] == 3
    assert "so101_follower" in result["content"][0]["text"]


def test_list_filtered_by_device_type(populated: Path) -> None:
    """``list`` with device_type='robots' counts only robot calibrations."""
    result = lerobot_calibrate(action="list", device_type="robots", base_path=str(populated))
    assert tool_json(result)["count"] == 2


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
    assert tool_json(result)["calibration_info"]["motor_count"] == 6
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
    assert tool_json(result)["count"] == 1
    assert tool_json(result)["results"][0]["device_id"] == "green_arm"


def test_search_no_match_reports_zero(populated: Path) -> None:
    """``search`` with no hits succeeds with an empty result set."""
    result = lerobot_calibrate(action="search", query="zzz", base_path=str(populated))
    assert result["status"] == "success"
    assert tool_json(result)["count"] == 0


def test_backup_action_reports_file_count(populated: Path, tmp_path: Path) -> None:
    """``backup`` copies all calibrations and reports the count."""
    result = lerobot_calibrate(action="backup", output_dir=str(tmp_path / "out"), base_path=str(populated))
    assert result["status"] == "success"
    assert tool_json(result)["files_count"] == 3


def test_restore_action_round_trips(populated: Path, tmp_path: Path) -> None:
    """``restore`` rebuilds calibrations from a prior ``backup``."""
    backup = lerobot_calibrate(action="backup", output_dir=str(tmp_path / "out"), base_path=str(populated))
    dest = tmp_path / "dest"
    result = lerobot_calibrate(action="restore", backup_dir=tool_json(backup)["backup_path"], base_path=str(dest))
    assert result["status"] == "success"
    assert tool_json(result)["restored_count"] == 3


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
    analysis: dict[str, Any] = tool_json(result)["analysis"]
    assert analysis["total_calibrations"] == 3
    assert analysis["device_counts"] == {"teleoperators": 1, "robots": 2}
    assert analysis["motor_stats"]["robots/so101_follower"]["max"] == 6


def test_analyze_empty_tree(tmp_path: Path) -> None:
    """``analyze`` on an empty tree succeeds with an empty analysis."""
    result = lerobot_calibrate(action="analyze", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert tool_json(result)["analysis"] == {}


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
    assert tool_json(result)["exists"] is True
    assert tool_json(result)["path"].endswith("orange_arm.json")


def test_path_action_base_paths(tmp_path: Path) -> None:
    """``path`` without an identifier reports the base/teleop/robot paths."""
    result = lerobot_calibrate(action="path", base_path=str(tmp_path))
    assert result["status"] == "success"
    assert tool_json(result)["teleop_path"].endswith("teleoperators")
    assert tool_json(result)["robot_path"].endswith("robots")


# --- get_calibration_path security contract (path-traversal guard) ---------


def test_get_calibration_path_allows_valid_components(tmp_path: Path) -> None:
    """A well-formed (type, model, id) triple maps to a path under base_path."""
    mgr = LeRobotCalibrationManager(tmp_path)
    path = mgr.get_calibration_path("teleoperators", "so101_leader", "blue_arm")
    assert path == tmp_path / "teleoperators" / "so101_leader" / "blue_arm.json"


def test_get_calibration_path_rejects_unknown_device_type(tmp_path: Path) -> None:
    """device_type outside {teleoperators, robots} is refused, not silently used."""
    mgr = LeRobotCalibrationManager(tmp_path)
    with pytest.raises(ValueError, match="device_type must be one of"):
        mgr.get_calibration_path("weapons", "so101", "arm")


@pytest.mark.parametrize(
    "bad",
    [
        "..",  # parent-dir escape
        "so101/../evil",  # embedded traversal
        "a/b",  # path separator
        ".hidden",  # leading dot (relative/hidden segment)
        "arm id",  # whitespace
        "arm;rm",  # shell metacharacter
        "",  # empty component
    ],
)
def test_get_calibration_path_rejects_unsafe_device_model(tmp_path: Path, bad: str) -> None:
    """A device_model that could escape base_path is rejected with a clear error."""
    mgr = LeRobotCalibrationManager(tmp_path)
    with pytest.raises(ValueError, match="device_model contains invalid characters"):
        mgr.get_calibration_path("robots", bad, "arm")


@pytest.mark.parametrize("bad", ["..", "../evil", "a/b", ".hidden", "arm id", ""])
def test_get_calibration_path_rejects_unsafe_device_id(tmp_path: Path, bad: str) -> None:
    """A device_id that could escape base_path is rejected with a clear error."""
    mgr = LeRobotCalibrationManager(tmp_path)
    with pytest.raises(ValueError, match="device_id contains invalid characters"):
        mgr.get_calibration_path("robots", "so101", bad)


def test_get_calibration_path_keeps_resolved_path_inside_base(tmp_path: Path) -> None:
    """Every accepted path stays within base_path (no traversal slips through)."""
    mgr = LeRobotCalibrationManager(tmp_path)
    path = mgr.get_calibration_path("robots", "so101_follower", "orange_arm")
    assert tmp_path.resolve() in path.resolve().parents


def test_path_action_surfaces_traversal_as_tool_error(tmp_path: Path) -> None:
    """A traversal identifier reaches the tool as a clean error dict, not a crash."""
    result = lerobot_calibrate(
        action="path",
        device_type="robots",
        device_model="..",
        device_id="arm",
        base_path=str(tmp_path),
    )
    assert result["status"] == "error"
    assert "invalid characters" in result["content"][0]["text"]


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


# --- ASCII-only output contract -------------------------------------------
# AGENTS.md forbids emojis (and orphan U+FE0F variation selectors left over
# from emoji sweeps) anywhere in code/logs/error strings. These tests pin the
# user-facing text of every action that previously carried such characters:
# the empty-list notice, the per-motor view header, and the delete-success
# message. They assert the rendered text is pure ASCII so the violation cannot
# silently reappear.


def _assert_ascii(text: str) -> None:
    """Fail with the offending code points if ``text`` is not pure ASCII."""
    offenders = [(i, hex(ord(c))) for i, c in enumerate(text) if ord(c) > 127]
    assert not offenders, f"non-ASCII in tool output at {offenders}: {text!r}"


def test_list_empty_notice_is_ascii(tmp_path: Path) -> None:
    """The empty-tree ``list`` notice carries no emoji/variation-selector."""
    result = lerobot_calibrate(action="list", base_path=str(tmp_path))
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    _assert_ascii(text)
    assert "No calibration files found." in text


def test_view_motor_header_is_ascii(populated: Path) -> None:
    """The per-motor ``view`` header carries no orphan variation selector."""
    result = lerobot_calibrate(
        action="view",
        device_type="robots",
        device_model="so101_follower",
        device_id="orange_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    _assert_ascii(text)
    assert "shoulder" in text


def test_delete_success_message_is_ascii(populated: Path) -> None:
    """The ``delete`` success message carries no orphan variation selector."""
    result = lerobot_calibrate(
        action="delete",
        device_type="robots",
        device_model="so101_follower",
        device_id="green_arm",
        base_path=str(populated),
    )
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    _assert_ascii(text)
    assert "Successfully deleted:" in text


# --- Fail-soft error branches (filesystem faults must not raise) -----------


def test_save_calibration_returns_false_when_path_is_a_directory(tmp_path: Path) -> None:
    """A blocked write target makes save_calibration report failure, not raise."""
    mgr = LeRobotCalibrationManager(tmp_path)
    # Occupy the target file path with a directory so open(..., "w") fails.
    mgr.get_calibration_path("robots", "so101", "arm1").mkdir(parents=True)
    assert mgr.save_calibration("robots", "so101", "arm1", _calib(2)) is False


def test_delete_calibration_returns_false_when_unlink_fails(tmp_path: Path) -> None:
    """delete_calibration fails soft (False) when the entry cannot be unlinked."""
    mgr = LeRobotCalibrationManager(tmp_path)
    # A directory at the calibration path exists() True but unlink() raises.
    mgr.get_calibration_path("robots", "so101", "arm1").mkdir(parents=True)
    assert mgr.calibration_exists("robots", "so101", "arm1") is True
    assert mgr.delete_calibration("robots", "so101", "arm1") is False


def test_get_calibration_info_returns_none_when_read_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read fault while gathering info yields None rather than propagating."""
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("robots", "so101", "arm1", _calib(2))

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated read failure")

    monkeypatch.setattr(mgr, "load_calibration", _boom)
    assert mgr.get_calibration_info("robots", "so101", "arm1") is None


def test_backup_returns_failure_when_destination_blocked(tmp_path: Path) -> None:
    """backup_calibrations reports (False, error, 0) when the dest tree is blocked."""
    mgr = LeRobotCalibrationManager(tmp_path)
    mgr.save_calibration("teleoperators", "so101_leader", "blue_arm", _calib(3))
    out = tmp_path / "backup_dest"
    out.mkdir()
    # A file where the per-type subdir must be created forces mkdir to fail.
    (out / "teleoperators").write_text("not a directory")

    success, message, count = mgr.backup_calibrations(output_dir=out)
    assert success is False
    assert count == 0
    assert message  # carries the underlying OS error text


def test_restore_returns_failure_when_destination_blocked(tmp_path: Path) -> None:
    """restore_calibrations reports failure when a destination path is occupied."""
    backup = tmp_path / "backup_src"
    (backup / "robots" / "m1").mkdir(parents=True)
    (backup / "robots" / "m1" / "arm.json").write_text('{"shoulder": {"id": 1}}')

    mgr = LeRobotCalibrationManager(tmp_path / "live")
    # Block the destination model directory with a regular file.
    (mgr.base_path / "robots" / "m1").write_text("block")

    success, message, count = mgr.restore_calibrations(backup)
    assert success is False
    assert count == 0
    assert message


def test_list_filters_out_nonmatching_model(populated: Path) -> None:
    """A device_model filter on list omits other models and counts only matches."""
    result = lerobot_calibrate(action="list", device_model="so101_follower", base_path=str(populated))
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    # so101_follower has two calibrations; the teleoperator model must be hidden.
    assert "so101_follower" in text
    assert "so101_leader" not in text
    assert tool_json(result)["count"] == 2


def test_list_marks_unreadable_calibration_without_failing(populated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a calibration's info cannot be read, list still succeeds and flags it."""
    monkeypatch.setattr(LeRobotCalibrationManager, "get_calibration_info", lambda *_a, **_k: None)
    result = lerobot_calibrate(action="list", base_path=str(populated))
    assert result["status"] == "success"
    assert "error reading file" in result["content"][0]["text"]
