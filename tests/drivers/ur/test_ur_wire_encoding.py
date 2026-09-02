"""An action dict becomes one ordered ``servoJ`` vector, or a named refusal.

:func:`~strands_robots.drivers.ur.targets_from_action` is the whole wire
encoding as a pure function, which is what lets it be graded without a
controller: the joint order, the held value for an omitted joint, and the two
motion gates are all decided here.

The gate worth the most is the *step* gate. A UR controller does not reject an
out-of-range setpoint the way a servo bus rejects a bad register write - it
accepts the register and tracks what it can, so a policy emitting a step larger
than the joint can travel in one control period gets partial motion and a
success envelope. Refusing it here is the difference between a caller learning
their cadence is wrong and a caller watching the arm lag its commands.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.drivers.ur import (
    FALLBACK_MODEL,
    JOINT_LIMIT_RAD,
    JOINT_NAMES,
    MAX_JOINT_SPEED_RAD_S,
    SUPPORTED_ROBOTS,
    speed_limits,
    targets_from_action,
)

#: A measured pose with a distinct value per joint, so a transposition in the
#: encoded vector cannot pass.
CURRENT = [0.1, -1.2, 1.4, -0.3, 0.7, 0.05]

#: A control period long enough that a small step clears the gate on either arm.
PERIOD = 0.05


class TestTheEncodedVector:
    """What reaches the wire, for an action the gates admit."""

    def test_every_joint_named_lands_in_wire_order(self) -> None:
        action = dict(zip(JOINT_NAMES, [0.2, -1.1, 1.3, -0.2, 0.6, 0.1], strict=True))
        targets, reason = targets_from_action(action, CURRENT, model="ur5e", control_period=PERIOD)
        assert reason is None, reason
        assert targets == [0.2, -1.1, 1.3, -0.2, 0.6, 0.1]

    def test_an_omitted_joint_holds_its_measured_value(self) -> None:
        """``servoJ`` takes a whole-arm setpoint, so a gap cannot default to zero.

        A policy naming only the wrist would otherwise command the shoulder and
        elbow to fold - the arm would move where nothing asked it to.
        """
        targets, reason = targets_from_action({"wrist_3_joint": 0.10}, CURRENT, model="ur5e", control_period=PERIOD)
        assert reason is None, reason
        assert targets == [0.1, -1.2, 1.4, -0.3, 0.7, 0.10]

    def test_the_vector_is_always_six_long(self) -> None:
        targets, reason = targets_from_action({"elbow_joint": 1.41}, CURRENT, model="ur5e")
        assert reason is None, reason
        assert len(targets) == len(JOINT_NAMES) == 6


class TestTheGatesRefuseAndSayWhy:
    """Each refusal names the joint, the value and the limit it exceeded."""

    def test_a_key_that_is_not_a_ur_joint_is_refused(self) -> None:
        _, reason = targets_from_action({"gripper": 0.5}, CURRENT)
        assert reason is not None
        assert "gripper" in reason
        assert "shoulder_pan_joint" in reason, "the refusal must name the accepted keys"

    def test_an_empty_action_commands_nothing_rather_than_holding(self) -> None:
        """A no-op write would report success for a command that was never made."""
        _, reason = targets_from_action({}, CURRENT)
        assert reason is not None
        assert "nothing to command" in reason

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.3", None])
    def test_a_value_that_is_not_a_finite_number_is_refused(self, value: object) -> None:
        _, reason = targets_from_action({"elbow_joint": value}, CURRENT)
        assert reason is not None
        assert "elbow_joint" in reason

    def test_a_target_outside_the_joints_travel_is_refused(self) -> None:
        beyond = JOINT_LIMIT_RAD + 0.1
        _, reason = targets_from_action({"wrist_1_joint": beyond}, CURRENT)
        assert reason is not None
        assert "wrist_1_joint" in reason
        assert "travel" in reason

    def test_a_current_vector_of_the_wrong_length_is_refused(self) -> None:
        """A five-axis read would silently mis-index every joint after the gap."""
        _, reason = targets_from_action({"elbow_joint": 1.4}, CURRENT[:5])
        assert reason is not None
        assert "5" in reason and "6" in reason

    def test_a_step_larger_than_the_joint_can_travel_is_refused(self) -> None:
        period = 0.008
        allowed = MAX_JOINT_SPEED_RAD_S["ur5e"][0] * period
        action = {"shoulder_pan_joint": CURRENT[0] + allowed * 2}
        _, reason = targets_from_action(action, CURRENT, model="ur5e", control_period=period)
        assert reason is not None
        assert "shoulder_pan_joint" in reason
        assert "rad/s ceiling" in reason

    def test_a_step_at_the_ceiling_is_admitted(self) -> None:
        """The gate is a ceiling, not a margin: the reachable step must pass."""
        period = 0.008
        allowed = MAX_JOINT_SPEED_RAD_S["ur5e"][0] * period
        action = {"shoulder_pan_joint": CURRENT[0] + allowed}
        targets, reason = targets_from_action(action, CURRENT, model="ur5e", control_period=period)
        assert reason is None, reason
        assert targets[0] == pytest.approx(CURRENT[0] + allowed)

    @pytest.mark.parametrize("period", [float("nan"), float("inf"), 0, -0.02, True, "0.02"])
    def test_a_control_period_off_the_domain_is_refused(self, period: object) -> None:
        """An unusable period must not silently disable the gate it sizes.

        ``nan`` is the dangerous one: ``allowed`` becomes ``nan``, every
        ``step > allowed`` comparison is then false, and the speed gate vanishes
        inside a success envelope - the one failure a caller could not see,
        because the envelope would report the setpoint as sent. Explicit
        ``None`` is the documented way to ask for no gate.
        """
        action = {"shoulder_pan_joint": CURRENT[0] + 5.0}
        targets, reason = targets_from_action(action, CURRENT, model="ur5e", control_period=period)  # type: ignore[arg-type]
        assert reason is not None, f"period {period!r} admitted a 5 rad step"
        assert "control_period" in reason
        assert targets == []

    def test_no_control_period_means_no_step_gate(self) -> None:
        """A single setpoint has no cadence to size a step against."""
        action = {"shoulder_pan_joint": CURRENT[0] + 1.0}
        _, reason = targets_from_action(action, CURRENT, model="ur5e", control_period=None)
        assert reason is None, reason


class TestTheModelDecidesTheCeiling:
    """One driver serves both arms, so the speed table must discriminate."""

    def test_a_step_the_ur5e_admits_the_ur10e_refuses(self) -> None:
        """The UR10e's proximal joints are held to 120 deg/s, the UR5e's to 180.

        This is the whole reason the driver carries a per-model table rather than
        one number: the same policy cadence is safe on one arm and asks the other
        for a motion it will not perform.
        """
        period = 0.02
        step = MAX_JOINT_SPEED_RAD_S["ur10e"][0] * period * 1.2
        action = {"shoulder_pan_joint": CURRENT[0] + step}

        _, ur5e_reason = targets_from_action(action, CURRENT, model="ur5e", control_period=period)
        _, ur10e_reason = targets_from_action(action, CURRENT, model="ur10e", control_period=period)

        assert ur5e_reason is None, ur5e_reason
        assert ur10e_reason is not None
        assert "ur10e" in ur10e_reason

    def test_the_table_covers_every_robot_the_driver_serves(self) -> None:
        assert set(MAX_JOINT_SPEED_RAD_S) == set(SUPPORTED_ROBOTS)

    def test_every_row_gives_one_ceiling_per_joint(self) -> None:
        for model, limits in MAX_JOINT_SPEED_RAD_S.items():
            assert len(limits) == len(JOINT_NAMES), model
            assert all(limit > 0 for limit in limits), model

    def test_the_ceilings_are_the_datasheet_numbers(self) -> None:
        """Degrees per second is how the datasheets state them; radians is the wire."""
        assert speed_limits("ur5e") == pytest.approx([math.radians(180)] * 6)
        assert speed_limits("ur10e") == pytest.approx([math.radians(120)] * 3 + [math.radians(180)] * 3)

    def test_an_unknown_model_gets_the_slower_arms_ceilings(self) -> None:
        """A name off the table is held to the tighter limits, never the looser.

        A renamed mesh peer given the faster budget would have its steps admitted
        here and dropped by the controller, which is the failure the gate exists
        to prevent - so the fallback errs toward refusing.
        """
        assert speed_limits("not_a_ur_model") == MAX_JOINT_SPEED_RAD_S[FALLBACK_MODEL]
        assert MAX_JOINT_SPEED_RAD_S[FALLBACK_MODEL][0] < MAX_JOINT_SPEED_RAD_S["ur5e"][0]
