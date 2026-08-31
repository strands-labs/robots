"""Native Modbus TCP driver for the Robotiq 2F-85 adaptive gripper.

``Robot("robotiq_2f85", mode="real", port="192.168.1.11")`` builds
:class:`RobotiqDriver` and the fingers move: unlike the servo-bus drivers whose
transport is still deferred, the command path here is wired end to end, because
Modbus TCP needs a socket and a six-byte register map and nothing else.

Two modules, the same split the sibling driver packages use:

* :mod:`strands_robots.drivers.robotiq.protocol` - the register map and Modbus
  TCP framing, pure functions, no I/O, importable on any host.
* :mod:`strands_robots.drivers.robotiq.driver` - :class:`RobotiqDriver`, which
  owns the socket and the power-on activation sequence.

The 2F-140 and the Hand-E answer the same registers with a different stroke,
which is why :func:`~strands_robots.drivers.robotiq.protocol.counts_to_aperture_mm`
and the driver both take ``stroke_mm``; only the 2F-85 entries are registered,
because those are the ones the package registry knows.
"""

from strands_robots.drivers.robotiq.driver import SUPPORTED_ROBOTS, RobotiqDriver
from strands_robots.drivers.robotiq.protocol import (
    DEFAULT_TCP_PORT,
    DEFAULT_UNIT_ID,
    INPUT_BASE,
    OUTPUT_BASE,
    REGISTER_COUNT,
    STROKE_MM,
    ActivationStatus,
    Fault,
    FunctionCode,
    ObjectStatus,
    ProtocolError,
    aperture_mm_to_counts,
    closed_fraction_to_counts,
    command_payload,
    command_registers,
    counts_to_aperture_mm,
    parse_status,
    read_input_registers_frame,
    read_registers_payload,
    write_registers_frame,
)

__all__ = [
    "DEFAULT_TCP_PORT",
    "DEFAULT_UNIT_ID",
    "INPUT_BASE",
    "OUTPUT_BASE",
    "REGISTER_COUNT",
    "STROKE_MM",
    "SUPPORTED_ROBOTS",
    "ActivationStatus",
    "Fault",
    "FunctionCode",
    "ObjectStatus",
    "ProtocolError",
    "RobotiqDriver",
    "aperture_mm_to_counts",
    "closed_fraction_to_counts",
    "command_payload",
    "command_registers",
    "counts_to_aperture_mm",
    "parse_status",
    "read_input_registers_frame",
    "read_registers_payload",
    "write_registers_frame",
]
