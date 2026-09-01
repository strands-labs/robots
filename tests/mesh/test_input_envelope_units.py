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

The datasheet is a floor on what a real stream contains, not a ceiling: a leader
arm is back-driven by hand, so it outruns its own servos' no-load speed. The last
class states the margin against recorded teleop instead, and why the per-joint
reach table below is not reused on that axis.

One frame carries more than one unit, which is the other half: the same arm that
streams degrees for its five arm joints streams 0-100 percent for its gripper, so
a single scalar bound has to be loose enough for the widest joint and is 7.2x full
scale on the narrowest. The last classes pin the per-joint envelope
:func:`~strands_robots.mesh.security.input_value_abs_by_key` builds from the
units the RECEIVING robot declares.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.bus_access import motor_norm_modes
from strands_robots.mesh import security
from strands_robots.mesh.input import InputReceiver
from strands_robots.mesh.security import (
    DEFAULT_INPUT_SLEW_ABS,
    DEFAULT_INPUT_VALUE_ABS,
    INPUT_ENVELOPE_FULL_SCALES,
    INPUT_FULL_SCALE_BY_NORM_MODE,
    input_frame_slew_violation,
    input_value_abs_by_key,
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

#: The fastest single step recorded teleop commands, per unit, in the unit that
#: joint's frames carry: ``max |a[t+1] - a[t]| * fps`` within an episode, over
#: 176,293 action frames / 295 episodes of SO-100 and SO-101 leader-follower
#: teleop (``lerobot/svla_so100_pickplace``, ``lerobot/svla_so101_pickplace``,
#: ``HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1``). The degree figure is
#: 2.4x :data:`STS3215_NO_LOAD_DEG_S`, which is what makes the datasheet a floor
#: on a hand-driven stream rather than a ceiling.
TELEOP_FASTEST_DEGREE_STEP = 899.3  # shoulder_lift, deg/s
TELEOP_FASTEST_PERCENT_STEP = 405.7  # gripper, percent/s

#: The largest magnitude those same frames command, per unit - the reach axis's
#: counterpart to the two speeds above, measured the same way (``max |a[t]|``).
TELEOP_FARTHEST_DEGREE_REACH = 137.8  # wrist_roll, deg
TELEOP_FARTHEST_PERCENT_REACH = 46.7  # gripper, percent

#: What a shipped SO-100 class arm declares, motor by motor. Restated here so
#: the per-joint cells run with lerobot absent;
#: :meth:`TestTheWireUnitIsDegrees.test_the_so_arm_declares_a_mixed_unit_frame`
#: is what holds it to lerobot's own declaration.
SO_ARM_NORM_MODES = {
    "shoulder_pan": "DEGREES",
    "shoulder_lift": "DEGREES",
    "elbow_flex": "DEGREES",
    "wrist_flex": "DEGREES",
    "wrist_roll": "DEGREES",
    "gripper": "RANGE_0_100",
}


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

    def test_the_so_arm_declares_a_mixed_unit_frame(self):
        """:data:`SO_ARM_NORM_MODES` is lerobot's declaration, not our guess."""
        pytest.importorskip("lerobot")
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        arm = SOFollower(SOFollowerRobotConfig(port="/dev/null", id="unit-probe"))
        assert motor_norm_modes(arm) == SO_ARM_NORM_MODES

    def test_every_bounded_mode_is_one_lerobot_declares(self):
        """The table may not carry a row for a mode no bus can declare.

        The other direction is deliberately not pinned: a mode lerobot adds
        later is absent from the table, and an absent mode keeps the scalar
        envelope, which is the safe verdict rather than a broken one.
        """
        pytest.importorskip("lerobot")
        from lerobot.motors import MotorNormMode

        declarable = {mode.name for mode in MotorNormMode}
        unknown = set(INPUT_FULL_SCALE_BY_NORM_MODE) - declarable
        assert not unknown, f"bounded modes no bus can declare: {sorted(unknown)}"


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


class TestTheSpeedAxisIsSizedOnRecordedTeleop:
    """What the bound clears, and why full scale does not size it.

    Reach and speed are different axes on the same frame.
    :func:`~strands_robots.mesh.security.input_value_abs_by_key` bounds reach per
    declared unit because full scale is where reach stops being addressable -
    lerobot clamps a command past it. A speed has no such ceiling in full scale:
    a frame-unit speed converts to servo travel through the joint's *calibrated*
    range, so the same rule applied here lands under real motion on the narrowest
    unit. The slew bound therefore stays scalar, sized on what recorded teleop
    commands.
    """

    @staticmethod
    def _verdict(key: str, speed: float, max_slew: float | None = None) -> str | None:
        """Whether one joint moving at *speed* frame units/second is refused."""
        return input_frame_slew_violation({key: speed * FRAME_S}, _baseline({key: 0.0}), FRAME_S, FRAME_S, max_slew)

    def test_a_hand_driven_leader_outruns_its_own_servo_datasheet(self):
        """The premise the measured constants exist for."""
        assert TELEOP_FASTEST_DEGREE_STEP > STS3215_NO_LOAD_DEG_S
        assert TELEOP_FASTEST_DEGREE_STEP / STS3215_NO_LOAD_DEG_S == pytest.approx(2.4, abs=0.1)

    @pytest.mark.parametrize(
        ("key", "speed"),
        [
            ("shoulder_lift.pos", TELEOP_FASTEST_DEGREE_STEP),
            ("gripper.pos", TELEOP_FASTEST_PERCENT_STEP),
        ],
    )
    def test_the_bound_admits_the_fastest_recorded_step_in_every_unit(self, key, speed):
        assert self._verdict(key, speed) is None, "recorded teleop was refused"

    def test_the_real_margin_is_smaller_than_the_datasheet_reading(self):
        """1.6x, not 3.9x - what a retune of the default actually has to spend."""
        assert DEFAULT_INPUT_SLEW_ABS / TELEOP_FASTEST_DEGREE_STEP == pytest.approx(1.6, abs=0.1)

    def test_the_reach_rule_applied_to_speed_would_refuse_recorded_teleop(self):
        """Why :data:`INPUT_FULL_SCALE_BY_NORM_MODE` is not reused on this axis.

        Reach grants each unit the same multiple of its own full scale. Granting
        speed the same way gives a percent gripper 400 units/s - under the ~406
        it is actually commanded at - while leaving the degree joints untouched,
        so the tighter number is bought by refusing the motion the bound exists
        to carry, on the one joint a hand traverses full scale fastest.
        """
        slew_per_reach = DEFAULT_INPUT_SLEW_ABS / DEFAULT_INPUT_VALUE_ABS
        reach = input_value_abs_by_key(SO_ARM_NORM_MODES)
        percent_row = reach["gripper.pos"] * slew_per_reach
        degree_row = reach["shoulder_lift.pos"] * slew_per_reach

        assert degree_row == pytest.approx(DEFAULT_INPUT_SLEW_ABS), "the degree joints see no change"
        assert percent_row < TELEOP_FASTEST_PERCENT_STEP, "premise: the row lands under real motion"
        assert self._verdict("gripper.pos", TELEOP_FASTEST_PERCENT_STEP, percent_row) is not None
        assert self._verdict("gripper.pos", TELEOP_FASTEST_PERCENT_STEP) is None

    def test_the_receiver_applies_a_percent_gripper_at_the_fastest_recorded_speed(self, monkeypatch):
        """End to end, on a follower that declares the percent unit.

        This is the cell a per-unit speed row turns red: the receiver reads the
        gripper's declared unit for reach already, so a row keyed on that same
        declaration would refuse the second frame here.

        With the apply-rate cap off the interval charged to a move is the nominal
        publish period, so two back-to-back frames are exactly one 50 Hz frame
        apart in the bound's terms rather than microseconds apart.
        """
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _receiver(SO_ARM_NORM_MODES)
        step = TELEOP_FASTEST_PERCENT_STEP * FRAME_S
        _send(recv, {"gripper.pos": 0.0})
        _send(recv, {"gripper.pos": step})
        assert applied == [{"gripper.pos": 0.0}, {"gripper.pos": step}]
        assert recv.stats["slew_rejected"] == 0

    @pytest.mark.parametrize(
        ("key", "reach"),
        [
            ("wrist_roll.pos", TELEOP_FARTHEST_DEGREE_REACH),
            ("gripper.pos", TELEOP_FARTHEST_PERCENT_REACH),
        ],
    )
    def test_the_per_joint_reach_bound_admits_every_recorded_magnitude(self, key, reach):
        """The axis that *is* bounded per unit still clears the same frames."""
        bounds = input_value_abs_by_key(SO_ARM_NORM_MODES)
        assert validate_input_frame({key: reach}, bounds) == {key: reach}
        assert bounds[key] / reach > 3.0, "and not marginally"


class TestTheEncodedPolicyIsStatedInTheFrameUnit:
    def test_reach_is_two_full_turns(self):
        assert DEFAULT_INPUT_VALUE_ABS == pytest.approx(2 * FULL_TURN_DEG)

    def test_slew_traverses_the_reach_envelope_once_per_second(self):
        # The relation the slew docstring states, independent of the unit.
        assert DEFAULT_INPUT_SLEW_ABS == pytest.approx(2 * DEFAULT_INPUT_VALUE_ABS)

    def test_slew_clears_the_leader_servo_speed_in_the_frame_unit(self):
        assert DEFAULT_INPUT_SLEW_ABS > STS3215_NO_LOAD_DEG_S
        # ~3.9x the datasheet speed. That ratio is not the margin against a real
        # stream, which a hand outruns; TestTheSpeedAxisIsSizedOnRecordedTeleop
        # pins the measured one.
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


class TestEachJointIsBoundedInItsOwnDeclaredUnit:
    """A degree shoulder and a percent gripper do not share one scalar bound."""

    @pytest.mark.parametrize(
        ("key", "value", "admitted"),
        [
            # The degree joints keep the envelope they were sized for.
            ("shoulder_pan.pos", 110.0, True),
            ("wrist_roll.pos", FULL_TURN_DEG + 90.0, True),
            ("wrist_roll.pos", 3 * FULL_TURN_DEG, False),
            # The gripper is RANGE_0_100: 100 is fully open, and the same
            # 2x-full-scale margin the degree row carries admits 200.
            ("gripper.pos", 100.0, True),
            ("gripper.pos", 200.0, True),
            # 300 is 3x a fully-open gripper. The scalar envelope admits it.
            ("gripper.pos", 300.0, False),
        ],
    )
    def test_a_mixed_unit_frame_is_bounded_per_joint(self, key, value, admitted):
        bounds = input_value_abs_by_key(SO_ARM_NORM_MODES)
        if admitted:
            assert validate_input_frame({key: value}, bounds) == {key: value}
        else:
            with pytest.raises(security.ValidationError):
                validate_input_frame({key: value}, bounds)

    def test_the_bound_is_the_same_multiple_of_full_scale_in_every_unit(self):
        """One rule, applied per unit - not a second hand-picked number.

        The degree row is the scalar default restated through that rule, so the
        joints the scalar was sized for are bounded exactly as before and a
        retune of :data:`DEFAULT_INPUT_VALUE_ABS` moves every unit together.
        """
        bounds = input_value_abs_by_key(SO_ARM_NORM_MODES)
        assert bounds["shoulder_pan.pos"] == pytest.approx(DEFAULT_INPUT_VALUE_ABS)
        for mode, full_scale in INPUT_FULL_SCALE_BY_NORM_MODE.items():
            per_joint = input_value_abs_by_key({"j": mode})["j"]
            assert per_joint / full_scale == pytest.approx(INPUT_ENVELOPE_FULL_SCALES)

    def test_the_percent_gripper_stops_being_bounded_at_7x_its_full_scale(self):
        """The slack this exists to remove, measured in both directions."""
        full_scale = INPUT_FULL_SCALE_BY_NORM_MODE["RANGE_0_100"]
        scalar = security._input_value_abs()
        assert scalar / full_scale == pytest.approx(7.2)
        per_joint = input_value_abs_by_key(SO_ARM_NORM_MODES)["gripper.pos"]
        assert per_joint / full_scale == pytest.approx(2.0)

    @pytest.mark.parametrize("key", ["gripper", "gripper.pos"])
    def test_both_spellings_of_one_motor_are_bounded(self, key):
        """A frame keys a motor bare or as lerobot's ``<motor>.pos`` action."""
        with pytest.raises(security.ValidationError):
            validate_input_frame({key: 300.0}, input_value_abs_by_key(SO_ARM_NORM_MODES))

    def test_an_unrecognised_mode_keeps_the_scalar_envelope(self):
        """An unknown declaration must not widen anything, nor narrow blindly."""
        bounds = input_value_abs_by_key({"gripper": "RANGE_M42_42"})
        assert bounds == {}
        validate_input_frame({"gripper.pos": DEFAULT_INPUT_VALUE_ABS}, bounds)
        with pytest.raises(security.ValidationError):
            validate_input_frame({"gripper.pos": DEFAULT_INPUT_VALUE_ABS * 1.5}, bounds)

    def test_a_narrowed_operator_envelope_is_not_widened_by_a_declared_unit(self, monkeypatch):
        """The per-joint row only ever tightens the scalar envelope.

        A fleet whose actuators use a smaller unit narrows
        ``STRANDS_MESH_INPUT_VALUE_ABS``; a degree row of 720 must not hand the
        envelope back.
        """
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "3.2")
        bounds = input_value_abs_by_key(SO_ARM_NORM_MODES)
        validate_input_frame({"shoulder_pan.pos": 3.0}, bounds)
        with pytest.raises(security.ValidationError):
            validate_input_frame({"shoulder_pan.pos": 45.0}, bounds)

    def test_the_module_bounds_units_without_importing_lerobot(self):
        """``mesh.security`` stays importable and testable with lerobot absent.

        The mode spellings are plain strings for this reason, so the guard is on
        the import graph rather than on the strings.
        """
        import ast

        tree = ast.parse(Path(security.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        assert "lerobot" not in imported


def _receiver(norm_modes):
    """A receiver whose follower declares *norm_modes*, and the frames it applies."""

    class _Bus:
        motors = {name: SimpleNamespace(norm_mode=mode) for name, mode in norm_modes.items()}

        def sync_read(self, *a, **k):  # what makes this the joint-read source
            return {}

    applied: list[dict] = []
    recv = InputReceiver(
        mesh=SimpleNamespace(
            peer_id="follower-1",
            subscribe=lambda *a, **k: "sub",
            unsubscribe=lambda *a, **k: None,
        ),
        robot=SimpleNamespace(bus=_Bus()),
        source_peer_id="leader-1",
        apply_fn=lambda robot, action: applied.append(action),
    )
    recv._running = True
    return recv, applied


def _send(recv, frame, **extra):
    recv._on_input(recv.topic, {"action": frame, "seq": 0, "t": time.time(), **extra})


class TestTheUnitComesFromTheReceiverNotTheFrame:
    """The bound that constrains a sender must not be chosen by that sender."""

    def test_the_follower_declaration_decides_the_verdict(self):
        """One frame, two followers: the percent gripper is the one refused."""
        percent, applied_percent = _receiver(SO_ARM_NORM_MODES)
        _send(percent, {"gripper.pos": 300.0})
        assert applied_percent == []
        assert percent._rejected == 1

        degrees, applied_degrees = _receiver({"gripper": "DEGREES"})
        _send(degrees, {"gripper.pos": 300.0})
        assert applied_degrees == [{"gripper.pos": 300.0}]

    def test_a_unit_claimed_by_the_frame_cannot_widen_the_bound(self):
        """A sender-supplied declaration is not consulted, so it changes nothing."""
        recv, applied = _receiver(SO_ARM_NORM_MODES)
        _send(recv, {"gripper.pos": 300.0}, norm_modes={"gripper": "DEGREES"}, unit="deg")
        assert applied == []
        assert recv._rejected == 1

    def test_a_robot_that_declares_nothing_keeps_the_scalar_envelope(self):
        """No bus to read is "no per-joint knowledge", not a refusal."""
        recv, applied = _receiver({})
        _send(recv, {"gripper.pos": 300.0})
        assert applied == [{"gripper.pos": 300.0}]
        assert recv._value_abs_by_key == {}
