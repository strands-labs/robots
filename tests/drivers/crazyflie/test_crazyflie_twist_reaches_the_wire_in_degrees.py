"""A twist in rad/s reaches ``cflib`` in deg/s, in the right argument slot.

The Crazyflie is the first robot in this package whose SDK does not speak SI.
Every twist here - the mesh's, the G1 driver's, the simulation's - carries ``wz``
in rad/s, and ``cflib``'s ``Commander`` takes ``yawrate`` in **degrees** per
second. A driver that forwards the number commands 1/57th of the requested yaw:
not a crash, not an error, just an aircraft that appears not to respond.

The second half of the same hazard is the argument *order*. The two setpoint
kinds disagree about where the yaw rate goes -
``send_hover_setpoint(vx, vy, yawrate, zdistance)`` puts it third and
``send_velocity_world_setpoint(vx, vy, vz, yawrate)`` puts it fourth - so
forwarding a positional tuple built for one to the other commands a height as a
yaw rate.

Both are pinned against the SI value, not against the driver's own arithmetic:
the expected degrees are written out per row.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.drivers.crazyflie import (
    HOVER_SETPOINT,
    VELOCITY_SETPOINT,
    action_to_setpoint,
    twist_to_setpoint,
)


class TestTheYawRateIsConvertedToDegrees:
    """The one unit boundary in the driver, pinned against hand-written degrees."""

    @pytest.mark.parametrize(
        ("wz_rad", "expected_deg"),
        [
            (0.0, 0.0),
            (1.0, 57.29577951308232),
            (-1.0, -57.29577951308232),
            (math.pi, 180.0),
            (math.pi / 2, 90.0),
            (math.radians(72.0), 72.0),  # cflib MotionCommander's own default rate
        ],
    )
    def test_a_hover_setpoint_carries_degrees_per_second(self, wz_rad: float, expected_deg: float) -> None:
        method, args = twist_to_setpoint(0.0, 0.0, wz_rad, 0.5)
        assert method == HOVER_SETPOINT
        assert args[2] == pytest.approx(expected_deg), (
            f"wz={wz_rad} rad/s must reach the wire as {expected_deg} deg/s; forwarding the "
            "radians commands 1/57th of the requested yaw"
        )

    @pytest.mark.parametrize(
        ("wz_rad", "expected_deg"),
        [(0.0, 0.0), (1.0, 57.29577951308232), (-2.0, -114.59155902616465)],
    )
    def test_a_world_velocity_setpoint_carries_degrees_per_second(self, wz_rad: float, expected_deg: float) -> None:
        method, args = twist_to_setpoint(0.0, 0.0, wz_rad)
        assert method == VELOCITY_SETPOINT
        assert args[3] == pytest.approx(expected_deg)

    def test_the_two_kinds_agree_on_the_yaw_they_command(self) -> None:
        """Whichever setpoint carries it, one rad/s is one rad/s."""
        _, hover = twist_to_setpoint(0.0, 0.0, 1.25, 0.4)
        _, world = twist_to_setpoint(0.0, 0.0, 1.25)
        assert hover[2] == pytest.approx(world[3])


class TestTheHeightDecidesWhichSetpointCarriesTheTwist:
    """``z`` given is an altitude-holding hover; ``z`` omitted is a velocity."""

    def test_a_height_selects_the_hover_setpoint_with_the_height_last(self) -> None:
        method, args = twist_to_setpoint(0.3, -0.2, 0.0, 0.75)
        assert (method, args) == (HOVER_SETPOINT, (0.3, -0.2, 0.0, 0.75))

    def test_no_height_selects_the_world_velocity_setpoint_with_vz_third(self) -> None:
        method, args = twist_to_setpoint(0.3, -0.2, 0.0, None, 0.25)
        assert (method, args) == (VELOCITY_SETPOINT, (0.3, -0.2, 0.25, 0.0))

    def test_a_hover_setpoint_ignores_vz_rather_than_smuggling_it_into_a_slot(self) -> None:
        """A hover holds a height; there is no rate slot for ``vz`` to occupy.

        The failure this refuses is the tempting one: pack ``vz`` into the
        4-tuple anyway and it lands in the ``zdistance`` slot, commanding a
        0.25 m hover for a 0.25 m/s climb.
        """
        method, args = twist_to_setpoint(0.0, 0.0, 0.0, 0.75, 0.25)
        assert (method, args) == (HOVER_SETPOINT, (0.0, 0.0, 0.0, 0.75))
        assert 0.25 not in args


class TestAnActionIsTranslatedTheSameWay:
    """The dict door reaches the same translation, with absent keys resting."""

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ({"vx": 0.5}, (VELOCITY_SETPOINT, (0.5, 0.0, 0.0, 0.0))),
            ({"wz": 1.0}, (VELOCITY_SETPOINT, (0.0, 0.0, 0.0, 57.29577951308232))),
            ({"z": 0.5}, (HOVER_SETPOINT, (0.0, 0.0, 0.0, 0.5))),
            ({"vx": 0.2, "z": 0.5}, (HOVER_SETPOINT, (0.2, 0.0, 0.0, 0.5))),
            ({"vz": 0.3}, (VELOCITY_SETPOINT, (0.0, 0.0, 0.3, 0.0))),
        ],
    )
    def test_absent_keys_rest_at_zero(self, action: dict[str, float], expected: tuple[str, tuple[float, ...]]) -> None:
        translated = action_to_setpoint(action)
        assert not isinstance(translated, str), translated
        method, args = translated
        assert method == expected[0]
        assert args == pytest.approx(expected[1])

    def test_an_action_naming_no_flight_key_is_refused_rather_than_resting(self) -> None:
        """An all-zero setpoint would latch a hover the caller never asked for."""
        reason = action_to_setpoint({"gripper": 1.0, "joint_0": 0.5})
        assert isinstance(reason, str)
        assert "no flight command" in reason
        assert "gripper" in reason, "the refusal must show what was actually passed"
        for key in ("vx", "vy", "vz", "wz", "z"):
            assert key in reason, f"the refusal must name {key!r} as an accepted alternative"
