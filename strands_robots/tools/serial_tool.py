"""Raw serial-bus access: port discovery, byte-level I/O, and STS/SMS servo writes.

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

``Goal_Velocity`` takes more than the byte width to bound, because the STS/SMS
series encodes it as sign-magnitude rather than plain unsigned: bit 15 carries
the direction and bits 0-14 the magnitude. A magnitude that overflows into bit
15 is therefore not truncated but *reinterpreted*. ``velocity=65535`` put those
exact two bytes on the wire and the servo read them as full speed in the
opposite direction, and ``velocity=32768`` read as magnitude zero -- stopping a
servo the caller had just asked to run -- both reported as success quoting the
number supplied.

``motor_id`` takes more than the byte width too, because the ID byte carries
one address that is not a servo: ``0xfe`` is the broadcast, which every servo
on the bus receives and, for an instruction that expects a reply, every servo
answers at once. On a half-duplex bus those replies collide, so the bytes
``action="feetech_ping"`` reads back belong to no single servo -- and it
reported them as one, quoting an ID no servo holds. A reply-less write to the
broadcast means exactly what it says and stays accepted; the highest address a
servo may hold is ``0xfd``, which is what a reply-expecting action is held to.

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

from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    MAX_GOAL_POSITION,
    MAX_UNICAST_ID,
    encode_word,
)
from strands_robots.utils import (
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
)

# Bit index carrying the direction in the two STS/SMS registers this module
# writes. ``Goal_Position`` (0x2A) and ``Goal_Velocity`` (0x2E) are both
# sign-magnitude, so neither ceiling below is the two-byte maximum: a magnitude
# reaching this bit is read by the servo as a command in the opposite
# direction, which is a different command rather than a truncated one.
_DIRECTION_BIT = 15

# Largest magnitude either register carries with ``_DIRECTION_BIT`` still clear.
_MAX_MAGNITUDE = (1 << _DIRECTION_BIT) - 1

# Inclusive bounds and the reason for each ceiling, keyed by the parameter that
# carries the field. The floor and the type are delegated to the shared count
# domains so an off-type or negative value is reported in the words every other
# surface uses; only the ceiling is decided here, because it is a property of
# the packet field this module encodes into.
_REGISTER_FIELDS: dict[str, tuple[int, int, str]] = {
    "motor_id": (
        1,
        BROADCAST_ID,
        f"the packet carries the ID in one byte, of which {MAX_UNICAST_ID:#x} is the highest a servo "
        f"may hold and {BROADCAST_ID:#x} is the broadcast, while 0xff is the frame header",
    ),
    "position": (
        0,
        MAX_GOAL_POSITION,
        "Goal_Position is 12-bit on the STS/SMS series, the same full scale the reported angle "
        "divides by; the SCS series is 10-bit and this module does not address it",
    ),
    "velocity": (
        0,
        _MAX_MAGNITUDE,
        "Goal_Velocity is sign-magnitude with bit 15 the direction bit, so a larger "
        "magnitude sets that bit and commands the opposite direction",
    ),
}

# Actions that read a status packet back after writing. A unicast is answered by
# one servo; the broadcast is answered by every servo at once, and on a
# half-duplex bus those replies collide, so the bytes read back belong to no
# single servo. :func:`~strands_robots.drivers.feetech.protocol.build_packet`
# refuses the broadcast for exactly these instructions -- its ``allow_broadcast``
# defaults to ``False`` -- and this is that rule at the tool's own boundary,
# before the port is opened. A reply-less write is absent from this set, so
# addressing the whole bus with one stays available where it means what it says.
_REPLY_EXPECTING_ACTIONS: frozenset[str] = frozenset({"feetech_ping"})

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


def _motor_id_error(value: Any, param: str, action: str) -> str | None:
    """Error text when ``value`` cannot address the servo ``action`` expects to hear from.

    The frame's ID byte carries the broadcast alongside every unicast address, so
    the field bound alone accepts it. An action that reads a status packet back
    needs one servo to answer, which the broadcast cannot give it.

    Args:
        value: The caller-supplied motor ID.
        param: The parameter it came from; always ``"motor_id"``.
        action: The requested action; decides whether a reply is expected.

    Returns:
        An error message, or ``None`` when the ID is one this action can use.
    """
    if error := _register_field_error(value, param, action):
        return error
    if value == BROADCAST_ID and action in _REPLY_EXPECTING_ACTIONS:
        return (
            f"{action}: {param} {BROADCAST_ID:#x} is the broadcast, which every servo on the bus "
            f"answers at once, so no single reply can be read back; address one servo in "
            f"[1, {MAX_UNICAST_ID}], got {value}."
        )
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
    ("motor_id", _motor_id_error),
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
        - "feetech_position": Control STS/SMS servo position
        - "feetech_velocity": Control STS/SMS servo velocity
        - "feetech_ping": Ping a Feetech servo motor
        - "monitor": Monitor serial port (continuous read)

    Args:
        action: Action to perform
        port: Serial port path (e.g., "/dev/ttyACM0", "COM3")
        baudrate: Communication speed in baud; a positive integer (default: 9600)
        timeout: Read timeout in seconds; a finite number >= 0, where 0 is
            pyserial's non-blocking mode (return what is already buffered)
        data: String data to send
        hex_data: Hex string data to send (e.g., "FF FF 01 04 03 00 64 92")
        motor_id: Motor ID for Feetech commands; an integer in [1, 254], of
            which 254 (0xfe) is the broadcast every servo receives. An action
            that reads a reply back accepts only a single servo, [1, 253]
        position: Target position for STS/SMS-series motors; an integer in
            [0, 4095]. That full scale and the two-byte order this tool encodes
            into are both STS/SMS properties: the SCS series is 10-bit and reads
            the same two bytes in the opposite order, so an SCS-series servo is
            not addressed by this action at all
        velocity: Target velocity for STS/SMS-series motors; an integer in
            [0, 32767]. Goal_Velocity is sign-magnitude on that series, so a
            magnitude reaching bit 15 commands the opposite direction instead of
            a faster move
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
            params = [0x2A, *encode_word(position)]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Feetech Motor {motor_id} -> Position {position} "
                        f"({position / MAX_GOAL_POSITION * 360:.1f} deg)"
                    }
                ],
            }

        elif action == "feetech_velocity":
            if motor_id is None or velocity is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id and velocity required"}]}

            # Feetech velocity command: Goal_Velocity address (0x2E)
            params = [0x2E, *encode_word(velocity)]
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
