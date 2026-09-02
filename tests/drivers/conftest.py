# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A hardware-shaped Feetech servo bus, shared by the bus and driver tests.

:class:`FakeServoPort` stands in for ``serial.Serial``. It is deliberately
*shaped like the hardware* rather than a bare mock: it answers a READ with a
real status frame it builds itself (header, length, error byte, little-endian
parameters, checksum), so a driver reading through it exercises the same codec
path a physical arm does. A mock returning a canned dict would pass while the
framing was wrong, which is the one thing worth grading here.

The frame it builds is the one
:func:`strands_robots.drivers.feetech.protocol.parse_status_packet` verifies -
they are written from the same datasheet layout, and if they ever disagree the
read tests fail, which is the point.
"""

from __future__ import annotations

import pytest


class FakeServoPort:
    """A serial port with servos behind it.

    Args:
        counts: Motor ID -> the raw register value that motor answers with.
            A motor absent from this map answers nothing, which is how a
            missing or mute servo is simulated.
        leading_noise: Bytes to put in front of the next reply, simulating the
            host's own echo on a half-duplex bus. The parser must skip these.
    """

    def __init__(self, counts: dict[int, int] | None = None, leading_noise: bytes = b"") -> None:
        self.counts = dict(counts or {})
        self.leading_noise = leading_noise
        self.is_open = True
        #: Every frame written, in order, for a test to decode.
        self.writes: list[bytes] = []
        self._pending = b""

    # -- the serial.Serial surface the bus uses ---------------------------- #

    def write(self, data: bytes) -> int:
        """Record the frame and, for a READ, queue the servo's answer."""
        self.writes.append(bytes(data))
        instruction = data[4]
        if instruction == 0x02:  # READ
            motor_id = data[2]
            if motor_id in self.counts:
                self._pending = self.leading_noise + self._status_frame(motor_id, self.counts[motor_id])
        return len(data)

    def read(self, size: int) -> bytes:
        """Return the queued reply once, then nothing."""
        pending, self._pending = self._pending, b""
        return pending[:size]

    def close(self) -> None:
        self.is_open = False

    # -- frame construction ------------------------------------------------ #

    @staticmethod
    def _status_frame(motor_id: int, value: int) -> bytes:
        """Build the reply a servo sends for a two-byte register read."""
        params = bytes([value & 0xFF, (value >> 8) & 0xFF])
        body = bytes([motor_id, len(params) + 2, 0x00]) + params
        return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


#: Raw counts putting every SO-arm joint at a value a test can name.
#: 0 is the low end of a joint's range, 4095 the high end, 2048 the middle.
MIDPOINT_COUNTS: dict[int, int] = dict.fromkeys((1, 2, 3, 4, 5, 6), 2048)


@pytest.fixture
def servo_port() -> FakeServoPort:
    """A six-servo SO-arm sitting at the middle of every joint's range."""
    return FakeServoPort(MIDPOINT_COUNTS)
