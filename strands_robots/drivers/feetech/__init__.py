"""Feetech STS/SMS-series native driver package.

Three layers, each usable alone: the wire codec (:mod:`.protocol`), the serial
bus that puts those frames on a port and converts units (:mod:`.bus`), and the
driver the robot factory builds (:mod:`.driver`), so
``Robot("so101", mode="real", driver="strands")`` commands the arm. Nothing
imported from this module opens a serial port - or even imports
:mod:`serial` - so the codec stays gradeable on a box with no serial stack.

:class:`~strands_robots.drivers.feetech.protocol.ProtocolError` is exported
alongside the codec because every parser docstring names it as the class a
caller catches to separate a wire fault from a caller bug; a handler the
package will not hand out is a contract the caller cannot write.
"""

from __future__ import annotations

from strands_robots.drivers.feetech.bus import (
    READABLE_REGISTERS,
    SO_ARM_MOTORS,
    FeetechBus,
    MotorSpec,
)
from strands_robots.drivers.feetech.driver import FeetechDriver
from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_GOAL_POSITION,
    MAX_UNICAST_ID,
    WORD_LENGTH,
    Instruction,
    ProtocolError,
    build_packet,
    decode_word,
    encode_word,
    parse_status_packet,
    ping_packet,
    read_packet,
    sync_write_packet,
    write_packet,
)

__all__ = [
    "BROADCAST_ID",
    "FeetechBus",
    "FeetechDriver",
    "HEADER",
    "Instruction",
    "MAX_GOAL_POSITION",
    "MAX_UNICAST_ID",
    "WORD_LENGTH",
    "MotorSpec",
    "ProtocolError",
    "READABLE_REGISTERS",
    "SO_ARM_MOTORS",
    "build_packet",
    "decode_word",
    "encode_word",
    "parse_status_packet",
    "ping_packet",
    "read_packet",
    "sync_write_packet",
    "write_packet",
]
