import time
from typing import Any

import serial
import serial.tools.list_ports
from strands import tool


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

    def send_serial_data(ser: serial.Serial, data_to_send: str | bytes) -> None:
        """Send data over serial connection."""
        if isinstance(data_to_send, str):
            ser.write(data_to_send.encode())
        else:
            ser.write(data_to_send)

    try:
        if action == "list_ports":
            ports = list_serial_ports()
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"📡 Found {len(ports)} serial ports:\n"
                        + "\n".join([f"• {p['device']} - {p['description']}" for p in ports])
                    }
                ],
                "ports": ports,
            }

        if not port:
            return {"status": "error", "content": [{"text": "❌ Port parameter required for this action"}]}

        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=timeout)

        if action == "send":
            if hex_data:
                # Parse hex string (e.g., "FF FF 01 04" -> [0xFF, 0xFF, 0x01, 0x04])
                hex_bytes = bytes.fromhex(hex_data.replace(" ", ""))
                ser.write(hex_bytes)
                response_text = f"✅ Sent hex data: {hex_data}"
            elif data:
                ser.write(data.encode())
                response_text = f"✅ Sent string data: {data}"
            else:
                ser.close()
                return {"status": "error", "content": [{"text": "❌ No data or hex_data provided"}]}

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
                "content": [{"text": f"📥 Read {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"}],
                "raw_data": read_data.hex(),
                "length": len(read_data),
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
                return {"status": "error", "content": [{"text": "❌ No data to send"}]}

            # Small delay then read response
            time.sleep(0.1)
            read_data = ser.read(read_bytes)
            ser.close()

            hex_str = " ".join([f"{b:02X}" for b in read_data])
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in read_data])

            return {
                "status": "success",
                "content": [
                    {"text": f"📤 {sent_text}\n📥 Read {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"}
                ],
            }

        elif action == "feetech_position":
            if motor_id is None or position is None:
                ser.close()
                return {"status": "error", "content": [{"text": "❌ motor_id and position required"}]}

            # Feetech position command: INST_WRITE (0x03), Goal_Position address (0x2A)
            params = [0x2A, position & 0xFF, (position >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"🤖 Feetech Motor {motor_id} → Position {position} ({position / 4095 * 360:.1f}°)"}
                ],
            }

        elif action == "feetech_velocity":
            if motor_id is None or velocity is None:
                ser.close()
                return {"status": "error", "content": [{"text": "❌ motor_id and velocity required"}]}

            # Feetech velocity command: Goal_Velocity address (0x2E)
            params = [0x2E, velocity & 0xFF, (velocity >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {"status": "success", "content": [{"text": f"🤖 Feetech Motor {motor_id} → Velocity {velocity}"}]}

        elif action == "feetech_ping":
            if motor_id is None:
                ser.close()
                return {"status": "error", "content": [{"text": "❌ motor_id required"}]}

            # Feetech ping command
            packet = build_feetech_packet(motor_id, 0x01, [])  # INST_PING
            ser.write(packet)

            time.sleep(0.1)
            response = ser.read(10)
            ser.close()

            if len(response) >= 6:
                return {
                    "status": "success",
                    "content": [{"text": f"🟢 Feetech Motor {motor_id} responded: {response.hex().upper()}"}],
                }
            else:
                return {"status": "error", "content": [{"text": f"🔴 Feetech Motor {motor_id} no response"}]}

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
                "content": [{"text": f"📊 Monitored {len(monitor_data)} data chunks in 5 seconds"}],
                "monitor_data": monitor_data,
            }

        else:
            ser.close()
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Unknown action: {action}\n"
                        "Available: list_ports, send, read, send_read,"
                        " feetech_position, feetech_velocity, feetech_ping, monitor"
                    }
                ],
            }

    except serial.SerialException as e:
        return {"status": "error", "content": [{"text": f"❌ Serial error: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Error: {e}"}]}
