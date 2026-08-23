"""Sole owner of the servo-bus recognition rule.

A USB port path -- ``/dev/ttyACM0`` on Linux, ``/dev/cu.usbmodem58FA0826181`` on
macOS -- is a POSITION on the bus, not an identity. Unplug an arm and plug it
back in and the kernel may hand it a different number; plug a second arm in
first and it may take the number the first one had. The USB serial number does
not move, so a report that names a port should name the serial number beside it.

Two callers in different layers need the same answer to "which of this host's
serial devices look like a robot's motor bus":

- :func:`strands_robots.robot._auto_detect_mode`, deciding sim vs real for
  ``mode="auto"``.
- :meth:`strands_robots.hardware_robot.Robot._create_minimal_config`, naming the
  candidates when the caller did not supply the ``port`` its robot requires.

:mod:`strands_robots.robot` defers importing :mod:`strands_robots.hardware_robot`
(it is the heavy hardware layer, and a sim-only caller must not pay for it), so
neither can own the rule for the other. It lives here, once, as
:data:`SERVO_BUS_VIDS` plus :func:`matches_servo_bus`.

``pyserial`` is not a declared dependency of this package -- it arrives with
lerobot -- so enumeration is best-effort throughout: :func:`scan_serial_devices`
answers with an empty list rather than raising, and every caller treats "no
devices" and "cannot enumerate" the same way.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Servo-bus USB bridge vendor IDs. Feetech/SO-10x controller boards carry WCH
#: CH34x chips that enumerate with the generic description ``USB Single Serial``
#: (observed on macOS with SO-101, vid ``0x1a86`` pid ``0x55d3``), so keyword
#: matching alone misses them entirely and ``mode="auto"`` silently falls back
#: to sim with hardware attached.
SERVO_BUS_VIDS: frozenset[int] = frozenset({0x1A86, 0x0403})  # WCH CH34x, FTDI

#: Substrings that identify a servo bus by name, matched against the device
#: description and manufacturer.
SERVO_BUS_KEYWORDS: tuple[str, ...] = (
    "feetech",
    "dynamixel",
    "sts3215",
    "xl430",
    "xl330",
    "ch340",
    "ch343",
)

#: Substrings that disqualify a device however it matched: an internal UART, a
#: debug port or a Bluetooth bridge is not a robot even on a matching vid.
NON_SERVO_BUS_KEYWORDS: tuple[str, ...] = (
    "bluetooth",
    "internal",
    "debug",
    "apple",
    "modem",
)


@dataclasses.dataclass(frozen=True)
class SerialCandidate:
    """One serial device on this host, as discovery sees it.

    Attributes:
        port: The device path, e.g. ``/dev/ttyACM0``. This is a position on the
            bus and may differ the next time the same hardware is plugged in.
        stable_id: The USB serial number when the device reports one, else a
            ``vid:pid`` (optionally ``:location``) fallback, else ``None`` for a
            device that carries no USB identity at all (an on-board UART). Two
            devices of the same model report different serial numbers, so this
            is what survives a replug.
        likely_servo_bus: Whether :func:`matches_servo_bus` recognised it as a
            robot's motor bus.
    """

    port: str
    stable_id: str | None
    likely_servo_bus: bool


def matches_servo_bus(port_info: Any) -> bool:
    """Return whether ``port_info`` looks like a robot's motor bus.

    Recognised by name (:data:`SERVO_BUS_KEYWORDS` against the description and
    manufacturer) or by vendor id (:data:`SERVO_BUS_VIDS`), and disqualified by
    :data:`NON_SERVO_BUS_KEYWORDS` however it matched.

    Args:
        port_info: A ``serial.tools.list_ports_common.ListPortInfo``, or
            anything exposing the same ``description`` / ``manufacturer`` /
            ``vid`` attributes. Read with ``getattr`` defaults so a partial
            stand-in is answered rather than raising.

    Returns:
        ``True`` when the device is a servo-bus candidate.
    """
    description = (getattr(port_info, "description", None) or "").lower()
    manufacturer = (getattr(port_info, "manufacturer", None) or "").lower()
    named = any(keyword in description + manufacturer for keyword in SERVO_BUS_KEYWORDS)
    by_vid = getattr(port_info, "vid", None) in SERVO_BUS_VIDS
    if not (named or by_vid):
        return False
    return not any(keyword in description for keyword in NON_SERVO_BUS_KEYWORDS)


def stable_device_id(port_info: Any) -> str | None:
    """Return the identity of ``port_info`` that survives a replug.

    The USB serial number when the device reports one. Devices that report none
    fall back to ``vid:pid`` plus the bus ``location`` when it is available:
    that still distinguishes two different models, and distinguishes two
    identical models on different bus positions, but it is a weaker identity
    than a serial number because it moves when the cable does.

    Args:
        port_info: A ``ListPortInfo``, or anything exposing ``serial_number`` /
            ``vid`` / ``pid`` / ``location``.

    Returns:
        The identity, or ``None`` for a device with no USB identity at all
        (an on-board UART reports neither a serial number nor a vid).
    """
    serial_number = getattr(port_info, "serial_number", None)
    if serial_number:
        return str(serial_number)
    vid = getattr(port_info, "vid", None)
    pid = getattr(port_info, "pid", None)
    if vid is None or pid is None:
        return None
    fallback = f"{vid:04x}:{pid:04x}"
    location = getattr(port_info, "location", None)
    return f"{fallback}:{location}" if location else fallback


def scan_serial_devices() -> list[SerialCandidate]:
    """Enumerate this host's serial devices, best-effort.

    Returns:
        One :class:`SerialCandidate` per device, in the order pyserial reports
        them. An empty list when there are no devices AND when enumeration
        could not be performed at all: ``pyserial`` is not a declared
        dependency of this package, and USB enumeration fails for reasons that
        are none of the caller's business (a permission error on the device
        node, a libusb hub glitch). Callers treat both the same way, so this
        never raises -- the failure is logged at debug for diagnosis.
    """
    try:
        import serial.tools.list_ports

        ports = list(serial.tools.list_ports.comports())
    except Exception as exc:  # noqa: BLE001 - see Returns: enumeration is best-effort
        # pyserial usually raises OSError (incl. PermissionError,
        # SerialException) but libusb backends have been observed to raise
        # RuntimeError on hub glitches, and the import itself raises
        # ImportError when pyserial is absent.
        logger.debug("USB enumeration failed (%s: %s); reporting no devices", type(exc).__name__, exc)
        return []
    return [
        SerialCandidate(
            port=port_info.device,
            stable_id=stable_device_id(port_info),
            likely_servo_bus=matches_servo_bus(port_info),
        )
        for port_info in ports
    ]


def describe_serial_candidates(devices: list[SerialCandidate]) -> str:
    """Describe ``devices`` for a refusal that could not name a port.

    Args:
        devices: The result of :func:`scan_serial_devices`.

    Returns:
        One sentence naming the servo-bus candidates and their stable ids, or
        naming what was present instead when none of them looks like a robot,
        or saying that nothing is present. Never empty, so a refusal can append
        it unconditionally.
    """
    servo = [device for device in devices if device.likely_servo_bus]
    if servo:
        listed = ", ".join(
            f"{device.port} (usb id {device.stable_id})" if device.stable_id else device.port for device in servo
        )
        return f"Candidate servo-bus device(s) on this host: {listed}."
    if devices:
        return (
            f"None of this host's {len(devices)} serial device(s) looks like a servo bus: "
            f"{', '.join(device.port for device in devices)}."
        )
    return "No serial devices are present on this host."
