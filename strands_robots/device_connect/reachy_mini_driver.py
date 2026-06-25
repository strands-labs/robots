"""ReachyMiniDriver - Device Connect DeviceDriver for Pollen Reachy Mini robots.

Auto-detects hardware variant via the daemon's ``wireless_version`` flag:
- **Wireless** (has onboard Pi): uses Zenoh transport for real-time I/O.
- **Lite** (USB-only, no Pi): uses WebSocket to the daemon directly.

REST API calls go through reachy_transport.api() for daemon/move operations.
"""

import asyncio
import logging
import math
import re
from typing import Any

from device_connect_edge.drivers import DeviceDriver, emit, get_rpc_source_device, on, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

from strands_robots.device_connect._authz import authz_error, is_authorized_caller
from strands_robots.device_connect.reachy_transport import (
    WebSocketLink,
    ZenohLink,
    api,
    identity_pose,
    rpy_to_pose,
)

logger = logging.getLogger(__name__)

# Security hardening: recorded-move names are interpolated into a REST URL
# path, so restrict them to a safe charset to prevent path traversal and
# query/parameter injection into the daemon API.
_MOVE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ReachyMiniDriver(DeviceDriver):
    """Device Connect driver for Pollen Reachy Mini.

    Auto-detects Wireless (Zenoh) vs Lite (WebSocket) via the daemon's
    ``wireless_version`` flag. REST API calls work the same for both.
    """

    device_type = "reachy_mini"

    def __init__(
        self,
        host: str = "reachy-mini.local",
        prefix: str = "reachy_mini",
        api_port: int = 8000,
    ):
        super().__init__()
        self._host = host
        self._prefix = prefix
        self._api_port = api_port
        self._latest_joints: dict[str, Any] | None = None
        self._latest_imu: dict[str, Any] | None = None
        self._hw: WebSocketLink | ZenohLink | None = None

    @property
    def identity(self) -> DeviceIdentity:
        return DeviceIdentity(
            device_type="reachy_mini",
            manufacturer="Pollen Robotics",
            model=f"Reachy Mini @ {self._host}",
            description="Pollen Reachy Mini expressive robot head with antennas",
        )

    @property
    def status(self) -> DeviceStatus:
        return DeviceStatus(availability="idle")

    async def connect(self) -> None:
        """Connect to the Reachy Mini, auto-detecting Wireless vs Lite."""
        try:
            status = await asyncio.to_thread(api, self._host, self._api_port, "/api/daemon/status")
            is_lite = not status.get("wireless_version", True)
        except Exception:
            is_lite = False

        if is_lite:
            self._hw = WebSocketLink(self._host, self._api_port)
            logger.info("Connected to Reachy Mini Lite at %s (WebSocket)", self._host)
        else:
            self._hw = ZenohLink(self.transport, self._prefix)
            logger.info("Connected to Reachy Mini at %s (Zenoh)", self._host)

        await self._hw.start(
            on_joints=lambda d: setattr(self, "_latest_joints", d),
            on_imu=lambda d: setattr(self, "_latest_imu", d),
        )

    async def disconnect(self) -> None:
        """Tear down the hardware link."""
        if self._hw:
            await self._hw.stop()

    # ── Helpers ────────────────────────────────────────────────

    async def _send_cmd(self, cmd: dict[str, Any]) -> None:
        """Send a real-time command via the active hardware link."""
        if self._hw is None:
            raise RuntimeError("Reachy Mini hardware link not connected")
        await self._hw.send_cmd(cmd)

    # ── Movement RPCs (Zenoh via transport) ────────────────────

    @rpc()
    async def look(
        self,
        pitch: float = 0,
        roll: float = 0,
        yaw: float = 0,
        x: float = 0,
        y: float = 0,
        z: float = 0,
    ) -> dict[str, Any]:
        """Set head pose instantly.

        Args:
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees
            yaw: Yaw angle in degrees
            x: X offset in mm
            y: Y offset in mm
            z: Z offset in mm
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "look")
        await self._send_cmd({"head_pose": rpy_to_pose(pitch, roll, yaw, x, y, z)})
        return {"status": "success", "pitch": pitch, "roll": roll, "yaw": yaw}

    @rpc()
    async def antennas(self, left: float = 0, right: float = 0) -> dict[str, Any]:
        """Set antenna angles.

        Args:
            left: Left antenna angle in degrees
            right: Right antenna angle in degrees
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "antennas")
        await self._send_cmd({"antennas_joint_positions": [math.radians(left), math.radians(right)]})
        return {"status": "success", "left": left, "right": right}

    @rpc()
    async def body(self, yaw: float = 0) -> dict[str, Any]:
        """Set body yaw angle.

        Args:
            yaw: Body yaw angle in degrees
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "body")
        await self._send_cmd({"body_yaw": math.radians(yaw)})
        return {"status": "success", "yaw": yaw}

    # ── Sensor RPCs (cached from transport subscription) ───────

    @rpc()
    async def getJoints(self) -> dict[str, Any]:
        """Get current joint positions (head + antennas)."""
        d = self._latest_joints
        if d is not None:
            head = d.get("head_joint_positions", [])
            ant = d.get("antennas_joint_positions", [])
            return {
                "status": "success",
                "head": [math.degrees(j) for j in head],
                "antennas": [math.degrees(j) for j in ant],
            }
        return {"status": "error", "reason": "no joint data"}

    @rpc()
    async def getImu(self) -> dict[str, Any]:
        """Get IMU data (accelerometer, gyroscope, quaternion, temperature)."""
        d = self._latest_imu
        if d is not None:
            return {
                "status": "success",
                "accelerometer": d.get("accelerometer"),
                "gyroscope": d.get("gyroscope"),
                "quaternion": d.get("quaternion"),
                "temperature": d.get("temperature"),
            }
        return {"status": "error", "reason": "no IMU data"}

    # ── Motor RPCs (Zenoh via transport) ───────────────────────

    @rpc()
    async def enableMotors(self, motor_ids: str = "") -> dict[str, Any]:
        """Enable motors (torque on).

        Args:
            motor_ids: Comma-separated motor IDs (empty = all)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "enableMotors")
        ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
        await self._send_cmd({"torque": True, "ids": ids})
        return {"status": "success", "enabled": motor_ids or "all"}

    @rpc()
    async def disableMotors(self, motor_ids: str = "") -> dict[str, Any]:
        """Disable motors (torque off).

        Args:
            motor_ids: Comma-separated motor IDs (empty = all)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "disableMotors")
        ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
        await self._send_cmd({"torque": False, "ids": ids})
        return {"status": "success", "disabled": motor_ids or "all"}

    # ── Move RPCs (REST) ──────────────────────────────────────

    @rpc()
    async def playMove(self, move_name: str, library: str = "emotions") -> dict[str, Any]:
        """Play a recorded move from the HuggingFace library.

        Args:
            move_name: Name of the move to play
            library: Move library (emotions or dance)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "playMove")
        if not _MOVE_NAME_RE.fullmatch(move_name or ""):
            return {"status": "error", "reason": f"invalid move_name: {move_name!r}"}
        ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            f"/api/move/play/recorded-move-dataset/{ds}/{move_name}",
            "POST",
        )
        return {"status": "success", "move": move_name, "result": result}

    @rpc()
    async def listMoves(self, library: str = "emotions") -> dict[str, Any]:
        """List available recorded moves.

        Args:
            library: Move library (emotions or dance)
        """
        ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            f"/api/move/recorded-move-datasets/list/{ds}",
        )
        return {"status": "success", "moves": result}

    # ── Expression RPCs (Zenoh animations via transport) ───────

    @rpc()
    async def nod(self) -> dict[str, Any]:
        """Nod the head (yes gesture)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "nod")
        for _ in range(3):
            await self._send_cmd({"head_pose": rpy_to_pose(15, 0, 0)})
            await asyncio.sleep(0.25)
            await self._send_cmd({"head_pose": rpy_to_pose(-10, 0, 0)})
            await asyncio.sleep(0.25)
        await self._send_cmd({"head_pose": identity_pose()})
        return {"status": "success", "expression": "nod"}

    @rpc()
    async def shake(self) -> dict[str, Any]:
        """Shake the head (no gesture)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "shake")
        for _ in range(3):
            await self._send_cmd({"head_pose": rpy_to_pose(0, 0, 25)})
            await asyncio.sleep(0.2)
            await self._send_cmd({"head_pose": rpy_to_pose(0, 0, -25)})
            await asyncio.sleep(0.2)
        await self._send_cmd({"head_pose": identity_pose()})
        return {"status": "success", "expression": "shake"}

    @rpc()
    async def happy(self) -> dict[str, Any]:
        """Happy antenna wiggle expression."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "happy")
        for _ in range(4):
            await self._send_cmd({"antennas_joint_positions": [math.radians(60), math.radians(-60)]})
            await asyncio.sleep(0.2)
            await self._send_cmd({"antennas_joint_positions": [math.radians(-60), math.radians(60)]})
            await asyncio.sleep(0.2)
        await self._send_cmd({"antennas_joint_positions": [0, 0]})
        return {"status": "success", "expression": "happy"}

    # ── Lifecycle RPCs (REST) ─────────────────────────────────

    @rpc()
    async def wakeUp(self) -> dict[str, Any]:
        """Wake up the robot (enable motors + play wake animation)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "wakeUp")
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/play/wake_up",
            "POST",
        )
        return {"status": "success", "result": result}

    @rpc()
    async def sleep(self) -> dict[str, Any]:
        """Put robot to sleep (play sleep animation + disable motors)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "sleep")
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/play/goto_sleep",
            "POST",
        )
        return {"status": "success", "result": result}

    @rpc()
    async def stopMotion(self) -> dict[str, Any]:
        """Stop all current motion."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "stopMotion")
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/stop",
            "POST",
        )
        return {"status": "success", "result": result}

    @rpc()
    async def getDaemonStatus(self) -> dict[str, Any]:
        """Get daemon status, motor state, and control frequency."""
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/daemon/status",
        )
        return {"status": "success", **result}

    # ── Events ────────────────────────────────────────────────

    @emit()
    async def emergencyStop(self, reason: str = "") -> None:
        """Emitted when this device triggers an emergency stop.

        Args:
            reason: Why the emergency stop was triggered
        """
        pass

    @on(event_name="emergencyStop")
    async def onEmergencyStop(self, device_id: str, event_name: str, payload: dict[str, Any]) -> None:
        """React to emergencyStop from an authorized safety controller.

        Security hardening: only act on emergency-stop events whose source is
        in the emergency-stop allowlist, so a spoofed event from an arbitrary
        device cannot interrupt operations.
        """
        if not is_authorized_caller(device_id, scope="estop"):
            logger.warning("Ignoring emergencyStop from unauthorized source %s", device_id)
            return
        logger.warning("Emergency stop received from %s - disabling motors", device_id)
        await self.stopMotion()
        await self.disableMotors()
