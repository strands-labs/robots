"""An unflyable setpoint is refused by name, never clamped and never sent.

``cflib`` imposes no ceiling on a setpoint and the firmware attempts whatever
arrives, so the flight envelope is this driver's to enforce. It refuses rather
than clamping, and the distinction is the whole point of the file: an operator
who asked for 5 m/s and silently got 1 m/s plans the next command around a speed
the aircraft never flew, and the error only surfaces as a position discrepancy
metres later. A refusal naming the bound leaves the caller's model correct.

The rows cover the two ways a number is unusable - outside the envelope, and not
a number at all - because they fail differently: an over-speed is a valid float
the firmware would honour, while ``nan`` serialises into a CRTP packet as a
perfectly well-formed float64 that poisons the controller's state.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.drivers.crazyflie import (
    MAX_HEIGHT,
    MAX_HORIZONTAL_SPEED,
    MAX_VERTICAL_SPEED,
    MAX_YAW_RATE,
    MIN_HEIGHT,
    action_to_setpoint,
    twist_envelope,
    twist_error,
)


class TestTheEnvelopeIsReportedAsItIsEnforced:
    """The discovery surface and the check cannot disagree."""

    def test_every_enforced_bound_is_reported(self) -> None:
        assert twist_envelope() == {
            "max_horizontal_speed": MAX_HORIZONTAL_SPEED,
            "max_vertical_speed": MAX_VERTICAL_SPEED,
            "max_yaw_rate": MAX_YAW_RATE,
            "min_height": MIN_HEIGHT,
            "max_height": MAX_HEIGHT,
        }

    def test_the_hover_band_is_a_band(self) -> None:
        """Non-vacuity: a min above a max would accept nothing and pass silently."""
        assert 0.0 < MIN_HEIGHT < MAX_HEIGHT


class TestAValueOnTheBoundIsFlyable:
    """The bound is inclusive, so the reported ceiling is reachable."""

    @pytest.mark.parametrize(
        "twist",
        [
            {"vx": MAX_HORIZONTAL_SPEED},
            {"vx": -MAX_HORIZONTAL_SPEED},
            {"vy": MAX_HORIZONTAL_SPEED},
            {"vz": MAX_VERTICAL_SPEED},
            {"wz": MAX_YAW_RATE},
            {"wz": -MAX_YAW_RATE},
            {"z": MIN_HEIGHT},
            {"z": MAX_HEIGHT},
        ],
    )
    def test_the_reported_ceiling_is_accepted(self, twist: dict[str, float]) -> None:
        assert not isinstance(action_to_setpoint(twist), str), (
            f"{twist} sits exactly on a bound twist_envelope() advertises; refusing it would "
            "make the advertised ceiling unreachable"
        )


class TestAValueOutsideTheEnvelopeIsRefusedByName:
    """Every refusal names the parameter, the value and the bound."""

    @pytest.mark.parametrize(
        ("twist", "param"),
        [
            ({"vx": 5.0}, "vx"),
            ({"vx": -5.0}, "vx"),
            ({"vy": 1.01}, "vy"),
            ({"vz": 0.6}, "vz"),
            ({"wz": 10.0}, "wz"),
            ({"z": 0.01}, "z"),
            ({"z": 3.0}, "z"),
        ],
    )
    def test_the_reason_names_the_parameter_and_a_bound(self, twist: dict[str, float], param: str) -> None:
        reason = action_to_setpoint(twist)
        assert isinstance(reason, str), f"{twist} is outside the envelope and must be refused"
        assert param in reason
        assert "twist_envelope()" in reason, "a refused caller needs the discovery surface"

    @pytest.mark.parametrize(
        ("twist", "param"),
        [
            ({"vx": float("nan")}, "vx"),
            ({"vx": float("inf")}, "vx"),
            ({"wz": float("-inf")}, "wz"),
            ({"vy": "0.5"}, "vy"),
            ({"vz": None}, "vz"),
            ({"z": float("nan")}, "z"),
            ({"z": -0.5}, "z"),
        ],
    )
    def test_a_value_that_is_not_a_usable_number_is_refused(self, twist: dict[str, Any], param: str) -> None:
        """``nan`` is a valid float64 on the wire, so the door is the only guard."""
        reason = action_to_setpoint(twist)
        assert isinstance(reason, str), f"{twist} is not a usable number and must be refused"
        assert param in reason

    def test_the_yaw_rate_bound_is_quoted_in_radians(self) -> None:
        """The refusal must speak the caller's unit, not the wire's.

        Quoting 143 deg/s at a caller who passed rad/s hands them a number they
        then have to convert back to check their own command.
        """
        reason = twist_error(0.0, 0.0, 99.0, context="set_twist")
        assert reason is not None
        assert "rad/s" in reason
        assert str(MAX_YAW_RATE) in reason
        assert "deg" not in reason


class TestNothingIsClamped:
    """A refusal, not a silently reduced command."""

    def test_an_over_speed_command_produces_no_setpoint_at_all(self) -> None:
        translated = action_to_setpoint({"vx": 5.0, "z": 0.5})
        assert isinstance(translated, str), (
            "clamping to the ceiling would return a setpoint here, and the caller would fly "
            "1 m/s while believing they commanded 5"
        )

    def test_the_first_offender_is_the_one_reported(self) -> None:
        """One reason at a time, in a stated order, so the message stays readable."""
        reason = action_to_setpoint({"vx": 5.0, "wz": 99.0})
        assert isinstance(reason, str)
        assert "vx" in reason
