"""Tests for sim-targeted ``execute`` / ``start`` payload validation (issue #303).

The validator at ``strands_robots.mesh.security.validate_command`` now
admits a small set of sim-peer fields used by ``Mesh._dispatch_sim_policy``:

* ``robot_name`` — disambiguates which robot in a multi-robot sim
* ``target_pose`` / ``target_joints`` / ``world_update`` — issue #300
  well-known per-call kwargs forwarded to planner-style policies
* ``control_frequency`` / ``action_horizon`` / ``fast_mode`` / ``n_steps``
  — sim-side runner controls

Each accepts a happy path and a representative rejection path so a
malicious peer cannot smuggle control-byte robot names, multi-MB
``world_update`` blobs, or NaN frequencies past the wire validator.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec


def _base() -> dict[str, object]:
    """A minimal-valid execute payload to layer optional fields onto."""
    return {
        "action": "execute",
        "instruction": "task",
        "policy_provider": "mock",
    }


# robot_name
class TestRobotName:
    def test_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "robot_name": "so100"})
        assert out["robot_name"] == "so100"

    def test_empty_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="robot_name"):
            sec.validate_command({**_base(), "robot_name": ""})

    def test_non_string_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="robot_name"):
            sec.validate_command({**_base(), "robot_name": 123})

    def test_shell_meta_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="robot_name"):
            sec.validate_command({**_base(), "robot_name": "arm; rm -rf /"})

    def test_slash_rejected(self) -> None:
        # Mirrors the teleop_receive.source_peer_id rule — robot_name is
        # not a path, so '/' is a wire-side red flag.
        with pytest.raises(sec.ValidationError, match="robot_name"):
            sec.validate_command({**_base(), "robot_name": "ns/arm"})

    def test_oversize_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="robot_name"):
            sec.validate_command({**_base(), "robot_name": "a" * 200})


# target_pose
class TestTargetPose:
    def test_happy_path(self) -> None:
        pose = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        out = sec.validate_command({**_base(), "target_pose": pose})
        assert out["target_pose"] == pose

    def test_wrong_length_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_pose"):
            sec.validate_command({**_base(), "target_pose": [0.0, 0.0, 0.0]})

    def test_non_list_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_pose"):
            sec.validate_command({**_base(), "target_pose": {"x": 0.0}})

    def test_nan_component_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_pose"):
            sec.validate_command({**_base(), "target_pose": [float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]})

    def test_inf_component_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_pose"):
            sec.validate_command({**_base(), "target_pose": [float("inf"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]})

    def test_oversize_component_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_pose"):
            sec.validate_command({**_base(), "target_pose": [1e9, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]})


# target_joints
class TestTargetJoints:
    def test_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "target_joints": {"j0": 0.5, "j1": -0.2}})
        assert out["target_joints"] == {"j0": 0.5, "j1": -0.2}

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_joints"):
            sec.validate_command({**_base(), "target_joints": [0.5, -0.2]})

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_joints"):
            sec.validate_command({**_base(), "target_joints": {0: 0.5}})

    def test_shell_meta_key_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_joints"):
            sec.validate_command({**_base(), "target_joints": {"j$0": 0.5}})

    def test_nan_value_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_joints"):
            sec.validate_command({**_base(), "target_joints": {"j0": float("nan")}})

    def test_too_many_joints_rejected(self) -> None:
        joints = {f"j{i}": 0.0 for i in range(sec.MAX_TARGET_JOINTS + 1)}
        with pytest.raises(sec.ValidationError, match="target_joints"):
            sec.validate_command({**_base(), "target_joints": joints})


# world_update
class TestWorldUpdate:
    def test_happy_path(self) -> None:
        update = {"obstacles": [{"name": "cube", "pose": [0.5, 0.0, 0.05]}]}
        out = sec.validate_command({**_base(), "world_update": update})
        assert out["world_update"] == update

    def test_null_accepted(self) -> None:
        out = sec.validate_command({**_base(), "world_update": None})
        assert out["world_update"] is None

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="world_update"):
            sec.validate_command({**_base(), "world_update": "not-a-dict"})

    def test_oversize_rejected(self) -> None:
        # Build a payload that JSON-encodes above MAX_WORLD_UPDATE_BYTES.
        big_string = "x" * (sec.MAX_WORLD_UPDATE_BYTES + 100)
        with pytest.raises(sec.ValidationError, match="world_update"):
            sec.validate_command({**_base(), "world_update": {"blob": big_string}})


# Optional sim-side controls
class TestRunControls:
    def test_control_frequency_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "control_frequency": 30.0})
        assert out["control_frequency"] == 30.0

    def test_control_frequency_nan_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="control_frequency"):
            sec.validate_command({**_base(), "control_frequency": float("nan")})

    def test_control_frequency_oversize_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="control_frequency"):
            sec.validate_command({**_base(), "control_frequency": 1e6})

    def test_action_horizon_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "action_horizon": 8})
        assert out["action_horizon"] == 8

    def test_action_horizon_zero_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="action_horizon"):
            sec.validate_command({**_base(), "action_horizon": 0})

    def test_fast_mode_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "fast_mode": True})
        assert out["fast_mode"] is True

    def test_fast_mode_non_bool_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="fast_mode"):
            sec.validate_command({**_base(), "fast_mode": "yes"})

    def test_n_steps_happy_path(self) -> None:
        out = sec.validate_command({**_base(), "n_steps": 100})
        assert out["n_steps"] == 100

    def test_n_steps_negative_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="n_steps"):
            sec.validate_command({**_base(), "n_steps": -5})
