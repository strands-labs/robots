import os
import time
from typing import Any

import serial
import serial.tools.list_ports
from strands import tool

import re as _re

# --- Security: port allowlist + velocity bounds ---
# Configurable via environment. Default permits only standard serial device paths.
_SERIAL_PORT_ALLOW_PATTERNS = [
    r'^/dev/tty(USB|ACM|S|AMA)\d+$',    # Linux serial devices
    r'^/dev/cu\.(usbserial|usbmodem)',    # macOS serial
    r'^COM\d+$',                          # Windows COM ports
]

_SERIAL_PORT_ALLOW_ENV = 'STRANDS_SERIAL_PORT_ALLOW'

# Feetech STS3215 safe velocity range (0 = stop, max rated ~2400 steps/s)
_FEETECH_VELOCITY_MAX = int(os.environ.get('STRANDS_SERIAL_VELOCITY_MAX', '2400'))
_FEETECH_MOTOR_ID_MAX = 253  # 254 = broadcast, disallowed by default


def _validate_port(port: str) -> str | None:
    """Return an error message if port is not in the allowlist, else None."""
    # Check env override first (comma-separated regex patterns)
    env_patterns = os.environ.get(_SERIAL_PORT_ALLOW_ENV)
    patterns = (
        [p.strip() for p in env_patterns.split(',') if p.strip()]
        if env_patterns
        else _SERIAL_PORT_ALLOW_PATTERNS
    )
    for pat in patterns:
        if _re.match(pat, port):
            return None
    return (
        f"Port {port!r} is not in the allowed serial port patterns. "
        f"Permitted: {patterns}. Set {_SERIAL_PORT_ALLOW_ENV} to override."
    )


def _validate_velocity(velocity: int) -> str | None:
    """Return an error message if velocity exceeds safe bounds."""
    if velocity < 0 or velocity > _FEETECH_VELOCITY_MAX:
        return (
            f"Velocity {velocity} is outside safe range [0, {_FEETECH_VELOCITY_MAX}]. "
            f"Set STRANDS_SERIAL_VELOCITY_MAX to adjust the limit."
        )
    return None


def _validate_motor_id(motor_id: int) -> str | None:
    """Return an error message if motor_id is invalid."""
    if motor_id < 1 or motor_id > _FEETECH_MOTOR_ID_MAX:
        return f"motor_id {motor_id} is outside valid range [1, {_FEETECH_MOTOR_ID_MAX}]."
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
        baudrate: Communication speed (default: 9600)
        timeout: Read timeout in seconds
        data: String data to send
        hex_data: Hex string data to send (e.g., "FF FF 01 04 03 00 64 92")
        motor_id: Motor ID for Feetech commands (1-254)
        position: Target position for Feetech motors (0-4095)
        velocity: Target velocity for Feetech motors
        read_bytes: Number of bytes to read

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

        # Validate port against allowlist
        port_err = _validate_port(port)
        if port_err:
            return {"status": "error", "content": [{"text": port_err}]}

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

            # Validate motor_id and position bounds
            mid_err = _validate_motor_id(motor_id)
            if mid_err:
                ser.close()
                return {"status": "error", "content": [{"text": mid_err}]}
            if position < 0 or position > 4095:
                ser.close()
                return {"status": "error", "content": [{"text": f"Position {position} outside valid range [0, 4095]"}]}

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

            # Validate motor_id and velocity bounds
            mid_err = _validate_motor_id(motor_id)
            if mid_err:
                ser.close()
                return {"status": "error", "content": [{"text": mid_err}]}
            vel_err = _validate_velocity(velocity)
            if vel_err:
                ser.close()
                return {"status": "error", "content": [{"text": vel_err}]}

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
            start_time = time.time()

            while time.time() - start_time < 5.0:  # 5 second limit
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
