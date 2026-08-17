"""Raw serial-bus access: port discovery, byte-level I/O, and Feetech servo writes.

Every numeric option an action consumes is checked here, before the port is
opened, because none of them is refused by the layer that finally consumes it.

The three Feetech register fields are encoded into fixed-width bytes of the
outgoing packet with a mask -- ``motor_id`` is the frame's single ID byte, and
``position`` / ``velocity`` are written as two little-endian bytes -- so a value
outside a field's range is not rejected on the wire. It is silently truncated
into a different, reachable command: ``position=70000`` encodes as 4464 and
``position=-1`` as 65535, the largest value the field can hold, while the
success message still quotes the number the caller supplied. Bounding the field
is what makes that message true.

``baudrate`` and ``read_bytes`` are coerced rather than checked by pyserial
(``2.7`` becomes 2 baud, and a non-positive ``read_bytes`` reads nothing and
reports success, which is indistinguishable from a timed-out read on a healthy
port). A non-finite ``timeout`` either waits no time at all -- ``nan``, whose
every comparison is false -- or fails with an ``OverflowError`` naming neither
the tool nor the option, for ``inf``.

:mod:`~strands_robots.tools.pose_tool` writes the same ``Goal_Position``
register through the same mask and needs no bound of its own: it clamps to each
motor's declared range before encoding, so its mask can only ever see a value
that already fits.
"""

import time
from collections.abc import Callable
from typing import Any

import serial
import serial.tools.list_ports
from strands import tool

from strands_robots.utils import (
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
)

# Inclusive bounds and the reason for each ceiling, keyed by the parameter that
# carries the field. The floor and the type are delegated to the shared count
# domains so an off-type or negative value is reported in the words every other
# surface uses; only the ceiling is decided here, because it is a property of
# the packet field this module encodes into.
_REGISTER_FIELDS: dict[str, tuple[int, int, str]] = {
    "motor_id": (1, 254, "the packet carries the ID in one byte, and 255 is the frame header"),
    "position": (
        0,
        4095,
        "Goal_Position is a 12-bit register, the same full scale the reported angle divides by",
    ),
    "velocity": (0, 65535, "Goal_Velocity is written as two bytes"),
}

# The options each action consumes. An action absent from this map reads none of
# them and is never refused here: ``list_ports`` returns before the port is even
# opened, so a value it never looks at must not turn a working call into an
# error.
_OPTIONS_BY_ACTION: dict[str, tuple[str, ...]] = {
    "send": ("baudrate", "timeout"),
    "read": ("baudrate", "timeout", "read_bytes"),
    "send_read": ("baudrate", "timeout", "read_bytes"),
    "feetech_position": ("baudrate", "timeout", "motor_id", "position"),
    "feetech_velocity": ("baudrate", "timeout", "motor_id", "velocity"),
    "feetech_ping": ("baudrate", "timeout", "motor_id"),
    "monitor": ("baudrate", "timeout"),
}


def _register_field_error(value: Any, param: str, action: str) -> str | None:
    """Error text when ``value`` cannot be encoded into its Feetech register field.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from; selects the field bounds.
        action: The requested action, used as the message prefix.

    Returns:
        An error message, or ``None`` when the value fits the field.
    """
    low, high, why = _REGISTER_FIELDS[param]
    floor_error: Callable[[Any, str, str], str | None] = positive_count_error if low > 0 else non_negative_count_error
    if error := floor_error(value, param, action):
        return error
    if value > high:
        return f"{action}: {param} must be at most {high} ({why}), got {value}."
    return None


def _read_timeout_error(value: Any, param: str, action: str) -> str | None:
    """Error text when ``value`` is not a usable serial read budget in seconds.

    The floor is ``0`` rather than the ``> 0`` of the shared
    :func:`~strands_robots.utils.positive_finite_number_error` because pyserial
    documents ``timeout=0`` as non-blocking mode: ``read`` then returns whatever
    is already buffered instead of waiting, which is a real request rather than a
    degenerate one. Finiteness, numeric-ness and ``bool`` are delegated to
    :func:`~strands_robots.utils.finite_number_error`, so only the floor is
    decided here.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        action: The requested action, used as the message prefix.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if error := finite_number_error(value, param, action):
        return error
    if value < 0:
        return f"{action}: {param} must be at least 0 seconds, got {value}."
    return None


# The domain each option is checked against, in the order errors are reported.
_OPTION_DOMAINS: tuple[tuple[str, Callable[[Any, str, str], str | None]], ...] = (
    ("baudrate", positive_count_error),
    ("timeout", _read_timeout_error),
    ("read_bytes", positive_count_error),
    ("motor_id", _register_field_error),
    ("position", _register_field_error),
    ("velocity", _register_field_error),
)


def _option_error(action: str, supplied: dict[str, Any]) -> str | None:
    """Error text for the first numeric option ``action`` reads but cannot honor.

    Only the options ``action`` consumes are checked, so a caller is never
    refused for a value the requested action never looks at. A register field
    left unset stays the concern of the action's own "required" check, which
    reports the whole missing pair at once.

    Args:
        action: The requested action; decides which options are effective.
        supplied: The numeric options as supplied, keyed by parameter name.

    Returns:
        An error message naming the action and the option, or ``None`` when
        every option this action reads is usable.
    """
    consumed = _OPTIONS_BY_ACTION.get(action, ())
    for param, check in _OPTION_DOMAINS:
        if param not in consumed:
            continue
        value = supplied[param]
        if value is None and param in _REGISTER_FIELDS:
            continue
        if error := check(value, param, action):
            return error
    return None


@tool
def serial_tool(
    action: str,
    port: str | None = None,
    baudrate: int = 9600,
    timeout: float = 1.0,
    data: str | None = None,
    hex_data: str | None = None,
    motor_id: int | None = None,
    position: int | None = None,
    velocity: int | None = None,
    read_bytes: int = 1024,
) -> dict[str, Any]:
    """Advanced serial communication tool for robot control and device communication.

    Actions:
        - "list_ports": Discover available serial ports
        - "send": Send data to serial port
        - "read": Read data from serial port
        - "send_read": Send data and read response
        - "feetech_position": Control Feetech servo position
        - "feetech_velocity": Control Feetech servo velocity
        - "feetech_ping": Ping Feetech servo motor
        - "monitor": Monitor serial port (continuous read)

    Args:
        action: Action to perform
        port: Serial port path (e.g., "/dev/ttyACM0", "COM3")
        baudrate: Communication speed in baud; a positive integer (default: 9600)
        timeout: Read timeout in seconds; a finite number >= 0, where 0 is
            pyserial's non-blocking mode (return what is already buffered)
        data: String data to send
        hex_data: Hex string data to send (e.g., "FF FF 01 04 03 00 64 92")
        motor_id: Motor ID for Feetech commands; an integer in [1, 254]
        position: Target position for Feetech motors; an integer in [0, 4095]
        velocity: Target velocity for Feetech motors; an integer in [0, 65535]
        read_bytes: Number of bytes to read; a positive integer

    Validation:
        A numeric option the requested action consumes is checked before the
        port is opened, so a value that cannot be honored is reported instead
        of being masked into a different servo command, silently coerced by
        pyserial, or read as no wait at all. An option the action ignores is
        never checked.

    Returns:
        Dict containing status and response content
    """

    def list_serial_ports() -> list[dict]:
        """List all available serial ports."""
        ports = []
        for port_info in serial.tools.list_ports.comports():
            ports.append(
                {
                    "device": port_info.device,
                    "name": port_info.name,
                    "description": port_info.description,
                    "manufacturer": port_info.manufacturer,
                    "vid": port_info.vid,
                    "pid": port_info.pid,
                    "serial_number": port_info.serial_number,
                }
            )
        return ports

    def build_feetech_packet(motor_id: int, instruction: int, params: list[int]) -> bytes:
        """Build Feetech servo protocol packet."""
        packet = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + params
        checksum = ~sum(packet[2:]) & 0xFF
        packet.append(checksum)
        return bytes(packet)

    try:
        if action == "list_ports":
            ports = list_serial_ports()
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Found {len(ports)} serial ports:\n"
                        + "\n".join([f"- {p['device']} - {p['description']}" for p in ports])
                    },
                    {"json": {"ports": ports}},
                ],
            }

        if not port:
            return {"status": "error", "content": [{"text": "Port parameter required for this action"}]}

        supplied = {
            "baudrate": baudrate,
            "timeout": timeout,
            "read_bytes": read_bytes,
            "motor_id": motor_id,
            "position": position,
            "velocity": velocity,
        }
        if option_error := _option_error(action, supplied):
            return {"status": "error", "content": [{"text": option_error}]}

        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=timeout)

        if action == "send":
            if hex_data:
                # Parse hex string (e.g., "FF FF 01 04" -> [0xFF, 0xFF, 0x01, 0x04])
                hex_bytes = bytes.fromhex(hex_data.replace(" ", ""))
                ser.write(hex_bytes)
                response_text = f"Sent hex data: {hex_data}"
            elif data:
                ser.write(data.encode())
                response_text = f"Sent string data: {data}"
            else:
                ser.close()
                return {"status": "error", "content": [{"text": "No data or hex_data provided"}]}

            ser.close()
            return {"status": "success", "content": [{"text": response_text}]}

        elif action == "read":
            read_data = ser.read(read_bytes)
            ser.close()

            # Format response as both hex and ASCII
            hex_str = " ".join([f"{b:02X}" for b in read_data])
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in read_data])

            return {
                "status": "success",
                "content": [
                    {"text": f"Read {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"},
                    {"json": {"raw_data": read_data.hex(), "length": len(read_data)}},
                ],
            }

        elif action == "send_read":
            # Send data first
            if hex_data:
                hex_bytes = bytes.fromhex(hex_data.replace(" ", ""))
                ser.write(hex_bytes)
                sent_text = f"Sent hex: {hex_data}"
            elif data:
                ser.write(data.encode())
                sent_text = f"Sent string: {data}"
            else:
                ser.close()
                return {"status": "error", "content": [{"text": "No data to send"}]}

            # Small delay then read response
            time.sleep(0.1)
            read_data = ser.read(read_bytes)
            ser.close()

            hex_str = " ".join([f"{b:02X}" for b in read_data])
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in read_data])

            return {
                "status": "success",
                "content": [{"text": f"{sent_text}\nRead {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"}],
            }

        elif action == "feetech_position":
            if motor_id is None or position is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id and position required"}]}

            # Feetech position command: INST_WRITE (0x03), Goal_Position address (0x2A)
            params = [0x2A, position & 0xFF, (position >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"Feetech Motor {motor_id} -> Position {position} ({position / 4095 * 360:.1f} deg)"}
                ],
            }

        elif action == "feetech_velocity":
            if motor_id is None or velocity is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id and velocity required"}]}

            # Feetech velocity command: Goal_Velocity address (0x2E)
            params = [0x2E, velocity & 0xFF, (velocity >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {"status": "success", "content": [{"text": f"Feetech Motor {motor_id} -> Velocity {velocity}"}]}

        elif action == "feetech_ping":
            if motor_id is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id required"}]}

            # Feetech ping command
            packet = build_feetech_packet(motor_id, 0x01, [])  # INST_PING
            ser.write(packet)

            time.sleep(0.1)
            response = ser.read(10)
            ser.close()

            if len(response) >= 6:
                return {
                    "status": "success",
                    "content": [{"text": f"Feetech Motor {motor_id} responded: {response.hex().upper()}"}],
                }
            else:
                return {"status": "error", "content": [{"text": f"Feetech Motor {motor_id} no response"}]}

        elif action == "monitor":
            # Continuous monitoring (limited time for safety)
            monitor_data = []
            # The safety window is a duration, so it is measured on
            # time.monotonic(); each record's ``timestamp`` below stays on the
            # wall clock because that is an absolute stamp a reader correlates
            # with other logs.
            start_time = time.monotonic()

            while time.monotonic() - start_time < 5.0:  # 5 second limit
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    monitor_data.append(
                        {
                            "timestamp": time.time(),
                            "data": chunk.hex(),
                            "ascii": "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in chunk]),
                        }
                    )
                time.sleep(0.1)

            ser.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"Monitored {len(monitor_data)} data chunks in 5 seconds"},
                    {"json": {"monitor_data": monitor_data}},
                ],
            }

        else:
            ser.close()
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}\n"
                        "Available: list_ports, send, read, send_read,"
                        " feetech_position, feetech_velocity, feetech_ping, monitor"
                    }
                ],
            }

    except serial.SerialException as e:
        return {"status": "error", "content": [{"text": f"Serial error: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
