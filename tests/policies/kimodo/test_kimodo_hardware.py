"""Tests for the Kimodo -> lerobot Unitree G1 action-key bridge.

The bridge is a pure key rename between two vocabularies for the same 29
joints. These tests pin the rename table, the fact that it pairs joints by name
rather than by position, and its failure surfaces -- without a real robot or the
unitree_sdk2 runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import pytest

from strands_robots.policies.kimodo.hardware import (
    _build_joint_map,
    build_lerobot_g1_action_dict,
    get_joint_map,
    kimodo_action_to_lerobot_g1,
)
from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS

# The driver's own spelling of the 29 G1 joints, in the order lerobot's
# G1_29_JointIndex declares them today. Held here as data so a test can permute
# or rename it and drive the map builder without lerobot installed.
DRIVER_JOINT_NAMES: tuple[str, ...] = (
    "kLeftHipPitch",
    "kLeftHipRoll",
    "kLeftHipYaw",
    "kLeftKnee",
    "kLeftAnklePitch",
    "kLeftAnkleRoll",
    "kRightHipPitch",
    "kRightHipRoll",
    "kRightHipYaw",
    "kRightKnee",
    "kRightAnklePitch",
    "kRightAnkleRoll",
    "kWaistYaw",
    "kWaistRoll",
    "kWaistPitch",
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)


def _driver_enum(names: Sequence[str]) -> Any:
    """Build a stand-in for the driver's joint enum from ``names``."""
    return IntEnum("G1_JointIndex", {name: index for index, name in enumerate(names)})


def _make_dummy_kimodo_action() -> dict[str, float]:
    """Build a Kimodo action-dict with a distinct value per joint."""
    return {name: 0.01 * i for i, name in enumerate(KIMODO_G1_JOINTS)}


def test_joint_map_has_all_29_joints():
    """The rename table covers every Kimodo joint exactly once."""
    joint_map = get_joint_map()
    assert len(joint_map) == 29
    assert set(joint_map.keys()) == set(KIMODO_G1_JOINTS)


def test_joint_map_targets_are_lerobot_q_keys():
    """Every mapped target has the lerobot ``kJointName.q`` shape."""
    joint_map = get_joint_map()
    for src, dst in joint_map.items():
        assert dst.startswith("k"), f"{dst} not lerobot-enum-shaped"
        assert dst.endswith(".q"), f"{dst} missing .q suffix"


def test_joint_map_is_cached():
    """The map is built once and returned on subsequent calls."""
    a = get_joint_map()
    b = get_joint_map()
    assert a is b  # identity - same dict object


def test_rename_preserves_values():
    """Values pass through the rename verbatim (radians in, radians out)."""
    kimodo = _make_dummy_kimodo_action()
    lerobot = kimodo_action_to_lerobot_g1(kimodo)
    joint_map = get_joint_map()
    for src, dst in joint_map.items():
        assert lerobot[dst] == pytest.approx(kimodo[src])


def test_rename_produces_29_keys():
    """The output has exactly 29 keys, one per G1 joint."""
    result = kimodo_action_to_lerobot_g1(_make_dummy_kimodo_action())
    assert len(result) == 29


def test_rename_missing_joint_raises_keyerror():
    """A short input surfaces the missing joints in a clear KeyError."""
    kimodo = _make_dummy_kimodo_action()
    del kimodo["left_hip_pitch_joint"]
    with pytest.raises(KeyError) as exc_info:
        kimodo_action_to_lerobot_g1(kimodo)
    assert "left_hip_pitch_joint" in str(exc_info.value)


def test_rename_ignores_unknown_input_keys():
    """Extra input keys (e.g. leftover base-pose entries) are dropped, not forwarded."""
    kimodo = _make_dummy_kimodo_action()
    kimodo["floating_base_joint"] = 999.0
    kimodo["surprise"] = 42.0
    result = kimodo_action_to_lerobot_g1(kimodo)
    assert 999.0 not in result.values()
    assert 42.0 not in result.values()
    assert "surprise" not in result


def test_build_full_action_no_extras():
    """Without ``extra_action_keys``, the wrapper equals the rename."""
    kimodo = _make_dummy_kimodo_action()
    a = build_lerobot_g1_action_dict(kimodo)
    b = kimodo_action_to_lerobot_g1(kimodo)
    assert a == b


def test_build_full_action_merges_extras():
    """``extra_action_keys`` is merged after the rename (overrides win)."""
    kimodo = _make_dummy_kimodo_action()
    extras = {"remote_axis_lx": 0.5, "remote_axis_ly": -0.5}
    result = build_lerobot_g1_action_dict(kimodo, extra_action_keys=extras)
    assert result["remote_axis_lx"] == 0.5
    assert result["remote_axis_ly"] == -0.5
    # The 29 joints are still there
    assert len([k for k in result if k.endswith(".q")]) == 29


def test_extras_override_joint_values():
    """``extra_action_keys`` wins over the rename for shared keys."""
    kimodo = _make_dummy_kimodo_action()
    # Overwrite the first joint via extras
    joint_map = get_joint_map()
    override_key = joint_map["left_hip_pitch_joint"]
    result = build_lerobot_g1_action_dict(kimodo, extra_action_keys={override_key: 3.14})
    assert result[override_key] == 3.14


def test_end_to_end_policy_to_hardware_action():
    """The full KimodoPolicy → hardware-bridge → send_action-ready dict flow.

    Uses a stub motion agent (no diffusers/CUDA/checkpoints) so this test
    runs in CI on any machine — the point is to prove the shapes line up,
    not to sample real motion.
    """
    import asyncio

    import numpy as np

    from strands_robots.policies.kimodo import KimodoConfig, KimodoPolicy

    class _StubAgent:
        def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
            n = 7 + len(KIMODO_G1_JOINTS)
            arr = np.zeros((num_frames, n), dtype=np.float32)
            arr[:, 6] = 1.0  # identity quat
            # distinct per-joint value so we can verify the rename
            for i in range(len(KIMODO_G1_JOINTS)):
                arr[:, 7 + i] = 0.1 * i
            return arr

    policy = KimodoPolicy(
        config=KimodoConfig(num_frames=10, diffusion_steps=5),
        motion_agent=_StubAgent(),
    )
    policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))

    kimodo_actions = asyncio.run(policy.get_actions({}, "walk forward"))
    assert len(kimodo_actions) == 1

    hw_action = build_lerobot_g1_action_dict(kimodo_actions[0])
    # 29 lerobot-shaped keys, values passed through
    assert len(hw_action) == 29
    joint_map = get_joint_map()
    for i, kimodo_key in enumerate(KIMODO_G1_JOINTS):
        lerobot_key = joint_map[kimodo_key]
        assert hw_action[lerobot_key] == pytest.approx(0.1 * i)


class TestTheMapPairsJointsByName:
    """The pairing key is the joint name, never the position in the driver enum.

    The driver applies only the action keys it recognises and silently leaves
    every other motor on its previous command, so a mis-paired or unrecognised
    key raises nothing at all -- the robot just moves wrong. Position is
    therefore not usable as the pairing key.
    """

    def test_a_reordered_driver_enum_still_pairs_every_joint_with_itself(self):
        """Reversing the driver's declaration order must not move a single target."""
        forward = _build_joint_map(_driver_enum(DRIVER_JOINT_NAMES))
        reversed_order = _build_joint_map(_driver_enum(tuple(reversed(DRIVER_JOINT_NAMES))))
        assert reversed_order == forward
        assert forward["left_hip_pitch_joint"] == "kLeftHipPitch.q"
        assert reversed_order["left_hip_pitch_joint"] == "kLeftHipPitch.q"

    def test_moving_the_waist_block_does_not_move_the_leg_targets(self):
        """A block reorder is the realistic drift; leg targets must be unaffected."""
        names = list(DRIVER_JOINT_NAMES)
        waist = [name for name in names if "Waist" in name]
        for name in waist:
            names.remove(name)
        reordered = _build_joint_map(_driver_enum(tuple(waist + names)))
        assert reordered == _build_joint_map(_driver_enum(DRIVER_JOINT_NAMES))
        assert reordered["right_knee_joint"] == "kRightKnee.q"
        assert reordered["waist_yaw_joint"] == "kWaistYaw.q"

    def test_a_renamed_driver_joint_is_refused_naming_both_sides(self):
        """A rename keeps the joint count, so only a name check can see it.

        Refusing is the safe answer: the bridge cannot confirm that the new
        spelling is the same joint, and naming both spellings is what lets a
        human decide instead of the map being taken on trust.
        """
        renamed = tuple("kTorsoYaw" if name == "kWaistYaw" else name for name in DRIVER_JOINT_NAMES)
        assert len(renamed) == len(DRIVER_JOINT_NAMES)
        with pytest.raises(RuntimeError) as exc_info:
            _build_joint_map(_driver_enum(renamed))
        message = str(exc_info.value)
        assert "waist_yaw_joint" in message
        assert "kTorsoYaw.q" in message

    def test_a_dropped_driver_joint_is_refused_naming_the_joint(self):
        """A shrunk DOF set must not yield a map that silently omits a joint."""
        with pytest.raises(RuntimeError) as exc_info:
            _build_joint_map(_driver_enum(DRIVER_JOINT_NAMES[:-1]))
        assert "right_wrist_yaw_joint" in str(exc_info.value)

    def test_two_driver_joints_naming_one_joint_are_refused(self):
        """An ambiguous pairing is reported, not resolved by declaration order."""
        with pytest.raises(RuntimeError) as exc_info:
            _build_joint_map(_driver_enum((*DRIVER_JOINT_NAMES, "WaistYaw")))
        message = str(exc_info.value)
        assert "waist_yaw_joint" in message
        assert "kWaistYaw.q" in message
        assert "WaistYaw.q" in message


def test_the_installed_driver_enum_still_names_the_canonical_joint_set():
    """The live lerobot enum agrees with the joint spellings pinned above.

    A lerobot rename or DOF change surfaces here as a diff against
    ``DRIVER_JOINT_NAMES`` rather than as a wrong target on a real robot.
    """
    g1_utils = pytest.importorskip("lerobot.robots.unitree_g1.g1_utils")
    installed = tuple(joint.name for joint in g1_utils.G1_29_JointIndex)
    assert set(installed) == set(DRIVER_JOINT_NAMES)
    assert get_joint_map() == _build_joint_map(_driver_enum(DRIVER_JOINT_NAMES))
