# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`strands_robots.drivers.feetech.bus.FeetechBus`.

The bus is the layer that turns a caller's degrees into the bytes a servo
latches, and back. Both directions are graded against the wire: a write is
decoded out of the frame the bus produced, and a read is decoded from a frame
:class:`~tests.drivers.conftest.FakeServoPort` built the way a servo does.

No serial port is opened. ``connect()`` is the only method that imports
:mod:`serial`, and the module-load pin in :mod:`test_feetech_module_load`
covers the fact that importing the package does not.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers.feetech.bus import (
    READABLE_REGISTERS,
    SO_ARM_MOTORS,
    FeetechBus,
    MotorSpec,
)
from tests.drivers.conftest import FakeServoPort


def _open_bus(port: FakeServoPort, **kwargs: object) -> FeetechBus:
    """A bus already holding ``port``, skipping the real ``connect()``."""
    bus = FeetechBus(port="/dev/fake", **kwargs)  # type: ignore[arg-type]
    bus._conn = port
    return bus


# ============================================================================
# Unit conversion.
# ============================================================================


class TestUnits:
    """Degrees in, counts out, and back - the conversion a wrong motion hides in."""

    @pytest.mark.parametrize(
        ("joint", "value", "counts"),
        [
            # Each joint's two ends and midpoint. The gripper is percent, not
            # degrees, which is the one asymmetry in the map.
            ("shoulder_pan", -180.0, 0),
            ("shoulder_pan", 0.0, 2048),
            ("shoulder_pan", 180.0, 4095),
            ("shoulder_lift", -90.0, 0),
            ("shoulder_lift", 90.0, 4095),
            ("elbow_flex", -150.0, 0),
            ("elbow_flex", 150.0, 4095),
            ("wrist_roll", 0.0, 2048),
            ("gripper", 0.0, 0),
            ("gripper", 50.0, 2048),
            ("gripper", 100.0, 4095),
        ],
    )
    def test_a_target_encodes_to_the_counts_the_servo_expects(self, joint: str, value: float, counts: int) -> None:
        assert SO_ARM_MOTORS[joint].to_counts(value) == counts

    @pytest.mark.parametrize("joint", sorted(SO_ARM_MOTORS))
    def test_encoding_round_trips_within_one_count(self, joint: str) -> None:
        """A value survives degrees -> counts -> degrees.

        One count of 4095 across the joint's span is the encoder's own
        resolution, so that is the tolerance; anything looser would hide a
        scale or offset error.
        """
        spec = SO_ARM_MOTORS[joint]
        tolerance = (spec.high - spec.low) / spec.resolution
        for value in (spec.low, (spec.low + spec.high) / 2, spec.high):
            assert spec.to_value(spec.to_counts(value)) == pytest.approx(value, abs=tolerance)

    def test_a_target_outside_the_range_is_refused_not_clamped(self) -> None:
        """A clamp would report success for a motion it did not make.

        This is the whole reason the bus validates: silently turning a
        400-degree command into a 90-degree one moves the arm somewhere the
        caller never asked for and tells them it worked.
        """
        with pytest.raises(ValueError, match="outside range"):
            SO_ARM_MOTORS["shoulder_lift"].to_counts(400.0)


# ============================================================================
# Reads.
# ============================================================================


class TestReads:
    """``sync_read`` is the method the mesh reaches for joint telemetry."""

    def test_positions_come_back_in_the_joints_own_unit(self, servo_port: FakeServoPort) -> None:
        """Midpoint counts decode to the middle of every joint's range."""
        bus = _open_bus(servo_port)
        reading = bus.sync_read("Present_Position")
        assert set(reading) == set(SO_ARM_MOTORS)
        for joint, value in reading.items():
            spec = SO_ARM_MOTORS[joint]
            assert value == pytest.approx((spec.low + spec.high) / 2, abs=0.1)

    def test_a_reply_behind_echoed_bytes_is_still_read(self) -> None:
        """Leading noise offsets a frame; it does not corrupt it.

        The bus is half-duplex, so the host's own transmission can land in
        front of the answer. Indexing at a fixed offset would report a joint
        ninety degrees from where it is - a number that looks like a
        measurement - so the frame is located instead.
        """
        port = FakeServoPort({1: 2048}, leading_noise=b"\x00\xfe")
        bus = _open_bus(port, motors={"shoulder_pan": SO_ARM_MOTORS["shoulder_pan"]})
        assert bus.sync_read("Present_Position")["shoulder_pan"] == pytest.approx(0.0, abs=0.1)

    def test_a_motor_that_does_not_answer_is_absent_not_guessed(self) -> None:
        """A mute servo is omitted, so no caller reads a fabricated angle."""
        port = FakeServoPort({1: 2048})  # ids 2..6 answer nothing
        bus = _open_bus(port)
        assert list(bus.sync_read("Present_Position")) == ["shoulder_pan"]

    @pytest.mark.parametrize(
        ("encoded", "expected"),
        [
            (0, 0),
            (100, 100),
            (32768, 0),  # bit 15 set, magnitude 0 - the servo is stopped
            (32868, -100),  # bit 15 set, magnitude 100 - reverse
        ],
    )
    def test_velocity_is_sign_magnitude_not_twos_complement(self, encoded: int, expected: int) -> None:
        """Bit 15 is direction. Reading it as magnitude reports a stopped
        joint as moving at full speed, which is the defect this pins."""
        port = FakeServoPort({1: encoded})
        bus = _open_bus(port, motors={"shoulder_pan": SO_ARM_MOTORS["shoulder_pan"]})
        assert bus.sync_read("Present_Velocity")["shoulder_pan"] == expected

    def test_an_unreadable_register_is_refused_by_name(self) -> None:
        """``Present_Current``'s sign encoding is not established here, so it
        is refused rather than decoded by guess."""
        bus = _open_bus(FakeServoPort())
        with pytest.raises(ValueError, match="Present_Current"):
            bus.sync_read("Present_Current")
        assert "Present_Current" not in READABLE_REGISTERS


# ============================================================================
# Writes.
# ============================================================================


class TestWrites:
    """A write is graded by decoding the frame the bus put on the wire."""

    def test_the_whole_arm_moves_in_one_sync_write_frame(self, servo_port: FakeServoPort) -> None:
        """One frame, not six.

        Six separate writes would start the joints at six different times,
        smearing a coordinated move over the bus latency.
        """
        bus = _open_bus(servo_port)
        bus.write_goal_positions({name: 0.0 for name in ("shoulder_pan", "wrist_roll")})
        (frame,) = servo_port.writes
        assert frame[:2] == b"\xff\xff"
        assert frame[2] == 0xFE  # broadcast: every servo reads its own slice
        assert frame[4] == 0x83  # SYNC_WRITE
        assert frame[5] == 0x2A  # Goal_Position

    def test_the_frame_carries_the_counts_each_motor_was_commanded(self, servo_port: FakeServoPort) -> None:
        """Decode the payload back into (id, counts) pairs."""
        bus = _open_bus(servo_port)
        bus.write_goal_positions({"shoulder_pan": 180.0, "gripper": 0.0})
        (frame,) = servo_port.writes
        payload = frame[7:-1]  # after header/id/len/instr/address/width, before checksum
        pairs = {payload[i]: payload[i + 1] | (payload[i + 2] << 8) for i in range(0, len(payload), 3)}
        assert pairs == {1: 4095, 6: 0}

    @pytest.mark.parametrize(
        ("targets", "match"),
        [
            ({}, "no targets"),
            ({"nope": 0.0}, "unknown motor"),
            ({"gripper": float("nan")}, "must be finite"),
            ({"gripper": float("inf")}, "must be finite"),
            ({"shoulder_lift": 400.0}, "outside range"),
        ],
    )
    def test_a_write_the_arm_cannot_honour_is_refused(self, targets: dict[str, float], match: str) -> None:
        bus = _open_bus(FakeServoPort())
        with pytest.raises(ValueError, match=match):
            bus.write_goal_positions(targets)

    def test_torque_release_writes_every_motor(self, servo_port: FakeServoPort) -> None:
        """Every joint is attempted, so a partial release cannot read as done."""
        bus = _open_bus(servo_port)
        assert bus.set_torque(False) == []
        assert len(servo_port.writes) == len(SO_ARM_MOTORS)
        for frame in servo_port.writes:
            assert frame[4] == 0x03  # WRITE
            assert frame[5] == 0x28  # Torque_Enable
            assert frame[6] == 0


# ============================================================================
# Lifecycle.
# ============================================================================


class TestLifecycle:
    """Open / closed is a fact a consumer reads to tell live from stale."""

    def test_a_bus_with_no_port_refuses_to_connect(self) -> None:
        """Refused by name rather than opening some default device."""
        with pytest.raises(ValueError, match="no port configured"):
            FeetechBus(port=None).connect()

    @pytest.mark.parametrize("method", ["sync_read", "write_goal_positions", "set_torque"])
    def test_every_bus_operation_needs_an_open_port(self, method: str) -> None:
        """A closed bus raises naming what was attempted, rather than
        returning an empty reading that looks like a stopped arm."""
        bus = FeetechBus(port="/dev/fake")
        argument: object = {"gripper": 0.0} if method == "write_goal_positions" else False
        with pytest.raises(RuntimeError, match="needs an open bus"):
            getattr(bus, method)() if method == "sync_read" else getattr(bus, method)(argument)

    def test_disconnect_closes_the_port_and_is_idempotent(self, servo_port: FakeServoPort) -> None:
        bus = _open_bus(servo_port)
        assert bus.is_connected
        bus.disconnect()
        assert not bus.is_connected
        assert not servo_port.is_open
        bus.disconnect()  # a second call must not raise

    def test_motors_may_be_narrowed_to_a_subset_of_the_arm(self) -> None:
        """A bus carrying two servos reads and writes only those two."""
        bus = _open_bus(FakeServoPort({6: 4095}), motors={"gripper": MotorSpec(6, 0, 100)})
        assert bus.sync_read("Present_Position") == {"gripper": pytest.approx(100.0)}
