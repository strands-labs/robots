"""The teleop input envelope is sized in the unit the frames actually carry.

``validate_input_frame`` bounds how far one teleop command may reach
(:data:`~strands_robots.mesh.security.DEFAULT_INPUT_VALUE_ABS`) and
:func:`~strands_robots.mesh.security.input_frame_slew_violation` bounds how fast
a joint may be commanded to travel
(:data:`~strands_robots.mesh.security.DEFAULT_INPUT_SLEW_ABS`). Both are pure
numbers compared against whatever the leader driver puts on the wire, so the
unit of that wire value is part of the contract even though no signature states
it.

lerobot's SO leader/follower default to ``use_degrees=True``, and their gripper
is ``MotorNormMode.RANGE_0_100`` whatever ``use_degrees`` says, so a shipped
SO-100 class arm streams degrees and a 0-100 gripper. Sizing the envelope in
radians instead leaves a usable reach of ~12.6 units on a joint whose own travel
is measured in hundreds, and a slew bound an order of magnitude *below* the
leader's own servo speed - the module docstring of the slew tests states the
opposite property ("a physical leader arm at full speed is never refused"), and
its own arithmetic ("roughly 4x the no-load speed ... ~6.5 rad/s") only holds if
the frames are radians.

These tests pin the unit, not a magnitude: the policy the constants encode (two
full turns of reach, that envelope traversed once per second, comfortably above
the leader's own servos) is asserted against the unit the frames carry, so a
future retune must keep the relation rather than silently re-enter the mismatch.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.mesh import security
from strands_robots.mesh.security import (
    DEFAULT_INPUT_SLEW_ABS,
    DEFAULT_INPUT_VALUE_ABS,
    input_frame_slew_violation,
    validate_input_frame,
)

#: One full revolution, in the degrees an SO leader publishes.
FULL_TURN_DEG = 360.0

#: Feetech STS3215 no-load speed at 12 V, the figure the slew bound's own
#: docstring reasons from. Converted here into the unit the frames carry.
STS3215_NO_LOAD_RAD_S = 6.5
STS3215_NO_LOAD_DEG_S = STS3215_NO_LOAD_RAD_S * 180.0 / math.pi  # ~372 deg/s

#: A 50 Hz frame period - the rate ``InputPublisher`` streams at by default.
FRAME_S = 1.0 / 50.0


def _baseline(pose: dict[str, float], t: float = 0.0) -> dict[str, tuple[float, float]]:
    return {k: (v, t) for k, v in pose.items()}


class TestTheWireUnitIsDegrees:
    """The unit is lerobot's, so read it off lerobot rather than restating it."""

    def test_the_so_driver_publishes_degrees_by_default(self):
        pytest.importorskip("lerobot")
        import dataclasses

        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        defaults = {f.name: f.default for f in dataclasses.fields(SOFollowerRobotConfig)}
        assert defaults["use_degrees"] is True, "premise: the envelope is sized for the unit the SO driver defaults to"


class TestAFullRangeLeaderFrameIsAdmitted:
    def test_a_modest_reach_is_admitted(self):
        # 45 deg of pan is an ordinary teleop pose, not an outlier.
        validate_input_frame({"shoulder_pan.pos": 45.0})

    def test_a_full_joint_sweep_is_admitted(self):
        frame = {
            "shoulder_pan.pos": 110.0,
            "shoulder_lift.pos": -95.0,
            "elbow_flex.pos": 120.0,
            "wrist_flex.pos": 75.0,
            "wrist_roll.pos": 180.0,
        }
        validate_input_frame(frame)

    def test_a_multi_turn_wrist_is_admitted(self):
        # wrist_roll is the multi-turn joint the reach headroom exists for.
        validate_input_frame({"wrist_roll.pos": FULL_TURN_DEG + 90.0})

    def test_a_fully_open_gripper_is_admitted(self):
        # The gripper is RANGE_0_100 whatever use_degrees says, so 100 is its
        # fully-open command and no configuration of the arm makes it smaller.
        validate_input_frame({"gripper.pos": 100.0})


class TestAPhysicalLeaderAtFullSpeedIsNotRefused:
    """The property the slew bound's own test module says the design rests on."""

    def test_a_no_load_sweep_is_not_refused(self):
        step = STS3215_NO_LOAD_DEG_S * FRAME_S
        assert step > 0.0, "premise: the sweep must actually move"
        reason = input_frame_slew_violation(
            {"shoulder_pan.pos": step}, _baseline({"shoulder_pan.pos": 0.0}), FRAME_S, FRAME_S
        )
        assert reason is None, f"a physical leader at its servo limit was refused: {reason}"

    def test_a_brisk_human_sweep_is_not_refused(self):
        step = STS3215_NO_LOAD_DEG_S * FRAME_S / 2
        reason = input_frame_slew_violation(
            {"shoulder_pan.pos": step}, _baseline({"shoulder_pan.pos": 0.0}), FRAME_S, FRAME_S
        )
        assert reason is None, f"half the servo limit was refused: {reason}"


class TestTheEncodedPolicyIsStatedInTheFrameUnit:
    def test_reach_is_two_full_turns(self):
        assert DEFAULT_INPUT_VALUE_ABS == pytest.approx(2 * FULL_TURN_DEG)

    def test_slew_traverses_the_reach_envelope_once_per_second(self):
        # The relation the slew docstring states, independent of the unit.
        assert DEFAULT_INPUT_SLEW_ABS == pytest.approx(2 * DEFAULT_INPUT_VALUE_ABS)

    def test_slew_clears_the_leader_servo_speed_in_the_frame_unit(self):
        assert DEFAULT_INPUT_SLEW_ABS > STS3215_NO_LOAD_DEG_S
        # The docstring's own margin: "roughly 4x the no-load speed".
        assert DEFAULT_INPUT_SLEW_ABS / STS3215_NO_LOAD_DEG_S == pytest.approx(3.87, abs=0.1)


class TestTheBoundStillRefusesARunaway:
    """Widening for the real unit must not stop the envelope protecting."""

    def test_a_runaway_reach_is_still_refused(self):
        with pytest.raises(security.ValidationError):
            validate_input_frame({"shoulder_pan.pos": 1_000_000.0})

    def test_a_full_scale_reversal_is_still_refused(self):
        reason = input_frame_slew_violation(
            {"shoulder_pan.pos": -DEFAULT_INPUT_VALUE_ABS},
            _baseline({"shoulder_pan.pos": DEFAULT_INPUT_VALUE_ABS}),
            FRAME_S,
            FRAME_S,
        )
        assert reason is not None
        assert "shoulder_pan.pos" in reason

    def test_an_operator_can_still_narrow_the_envelope(self, monkeypatch):
        # A radian-valued or normalized fleet narrows rather than widens.
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "3.2")
        assert security._input_value_abs() == pytest.approx(3.2)
