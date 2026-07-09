"""Behavior tests for hardware Robot ROS 2 telemetry classification and the
pre-0.5 SO-family calibration file auto-migration.

Both paths are exercised on a ``Robot`` built via ``__new__`` + manual attribute
wiring (the same pattern as ``test_hardware_robot_lifecycle``) so no serial/USB
hardware and no lerobot driver is touched.

``_publish_ros_telemetry`` splits an observation dict into ``JointState``
scalars (sorted, deterministic) and per-camera ``(H, W, 3)`` frames, skips
booleans, and never lets a bridge failure interrupt the control loop.

``_migrate_legacy_calibration`` copies a pre-0.5 ``so100_follower/`` /
``so101_follower/`` calibration file to the unified ``so_follower/`` path,
refusing to guess when the source is ambiguous and no-op'ing for non-SO robots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.hardware_robot import Robot as HwRobot


class _RecordingBridge:
    """ROS 2 bridge double that records what telemetry it was asked to publish."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.joint_calls: list[tuple[str, list[str], list[float]]] = []
        self.image_calls: list[tuple[str, str, Any]] = []

    def publish_joint_states(self, robot_name: str, names: list[str], positions: list[float]) -> None:
        if self.fail:
            raise RuntimeError("bridge down")
        self.joint_calls.append((robot_name, names, positions))

    def publish_image(self, robot_name: str, camera: str, frame: Any) -> None:
        self.image_calls.append((robot_name, camera, frame))


class _FakeRobot:
    """Minimal lerobot-robot stand-in exposing only what the paths under test read."""

    def __init__(self, *, name: str = "arm0", calibration_fpath: str | None = None) -> None:
        self.name = name
        self.calibration_fpath = calibration_fpath


def _telemetry_robot(bridge: _RecordingBridge | None, *, robot_name: str = "arm0") -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.robot = _FakeRobot(name=robot_name)
    hw._ros_bridge = bridge
    hw._ros2_domain = 0
    return hw


def _migration_robot(calibration_fpath: str | None) -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.robot = _FakeRobot(calibration_fpath=calibration_fpath)
    return hw


class TestRosTelemetryClassification:
    def test_scalars_sorted_and_images_split_booleans_dropped(self) -> None:
        bridge = _RecordingBridge()
        hw = _telemetry_robot(bridge, robot_name="follower")
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        observation = {
            "j1.pos": 1.0,
            "j0.pos": np.float32(2.0),  # numpy 0-d scalar -> joint
            "estop": True,  # python bool -> skipped
            "wrist": frame,  # (H, W, 3) -> image
        }

        hw._publish_ros_telemetry(observation)

        assert len(bridge.joint_calls) == 1
        robot_name, names, positions = bridge.joint_calls[0]
        assert robot_name == "follower"
        # Sorted by key, booleans excluded from the JointState arrays.
        assert names == ["j0.pos", "j1.pos"]
        assert positions == [2.0, 1.0]
        assert [c[1] for c in bridge.image_calls] == ["wrist"]

    def test_skip_images_publishes_joints_only(self) -> None:
        bridge = _RecordingBridge()
        hw = _telemetry_robot(bridge)
        observation = {"j0.pos": 0.5, "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}

        hw._publish_ros_telemetry(observation, skip_images=True)

        assert bridge.joint_calls and bridge.joint_calls[0][1] == ["j0.pos"]
        assert bridge.image_calls == []

    def test_disabled_bridge_is_noop(self) -> None:
        hw = _telemetry_robot(None)
        # No bridge wired -> silent no-op, must not raise.
        hw._publish_ros_telemetry({"j0.pos": 0.0})

    def test_bridge_failure_never_propagates(self) -> None:
        bridge = _RecordingBridge(fail=True)
        hw = _telemetry_robot(bridge)
        # A publish failure is best-effort and must be swallowed.
        hw._publish_ros_telemetry({"j0.pos": 0.0})
        assert bridge.joint_calls == []


class TestLegacyCalibrationMigration:
    def _seed(self, tmp_path: Path, shared: str, legacy_dirs: list[str], file_name: str = "cal.json") -> Path:
        """Return the new (shared) calibration path; create legacy files under tmp_path."""
        calib_root = tmp_path / "robots"
        new_path = calib_root / shared / file_name
        for i, d in enumerate(legacy_dirs):
            legacy = calib_root / d / file_name
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(f'{{"legacy": {i}}}')
        return new_path

    def test_single_legacy_file_is_copied_to_shared_path(self, tmp_path: Path) -> None:
        new_path = self._seed(tmp_path, "so_follower", ["so100_follower"])
        hw = _migration_robot(str(new_path))

        hw._migrate_legacy_calibration()

        assert new_path.is_file()
        assert new_path.read_text() == '{"legacy": 0}'

    def test_already_calibrated_is_left_untouched(self, tmp_path: Path) -> None:
        new_path = self._seed(tmp_path, "so_follower", ["so100_follower"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text('{"current": true}')
        hw = _migration_robot(str(new_path))

        hw._migrate_legacy_calibration()

        # Existing calibration at the new path wins; no overwrite from a legacy dir.
        assert new_path.read_text() == '{"current": true}'

    def test_ambiguous_legacy_sources_refuse_to_guess(self, tmp_path: Path) -> None:
        new_path = self._seed(tmp_path, "so_leader", ["so100_leader", "so101_leader"])
        hw = _migration_robot(str(new_path))

        hw._migrate_legacy_calibration()

        # Two candidate sources -> refuse to guess, nothing migrated.
        assert not new_path.exists()

    def test_no_legacy_source_is_noop(self, tmp_path: Path) -> None:
        new_path = self._seed(tmp_path, "so_follower", [])
        hw = _migration_robot(str(new_path))

        hw._migrate_legacy_calibration()

        assert not new_path.exists()

    def test_non_so_family_is_skipped(self, tmp_path: Path) -> None:
        # koch_follower is not part of the SO-family rename; a same-role legacy
        # file must not be pulled in.
        new_path = self._seed(tmp_path, "koch_follower", ["koch100_follower"])
        hw = _migration_robot(str(new_path))

        hw._migrate_legacy_calibration()

        assert not new_path.exists()

    def test_missing_calibration_fpath_is_noop(self) -> None:
        hw = _migration_robot(None)
        # robot exposes no calibration path -> nothing to migrate, must not raise.
        hw._migrate_legacy_calibration()
