"""Shared transport helpers for Reachy Mini robots.

REST API helpers, pose math, and hardware link abstractions
used by ReachyMiniDriver.
"""

import asyncio
import functools
import json
import logging
import math
import os
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def resolve_host(host: str) -> str:
    """Resolve hostname to IP address."""
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return host


def _daemon_auth_token() -> str | None:
    """Return the Reachy daemon auth token from the environment, if configured.

    Security hardening: the daemon WebSocket/REST interfaces accept commands
    that directly actuate the robot. When ``REACHY_DAEMON_TOKEN`` is set we
    present it as a bearer credential so the daemon can authenticate the
    caller. When it is absent we emit a one-time warning so operators are
    aware the link is unauthenticated (and should be confined to a trusted
    network or fronted by WSS/HTTPS with mutual TLS).
    """
    return os.environ.get("REACHY_DAEMON_TOKEN") or None


@functools.cache
def _emit_unauthenticated_warning(kind: str) -> None:
    """Log the unauthenticated-daemon warning once per transport kind.

    Cached so a reconnect loop logs it only once; the per-kind key surfaces
    each distinct transport (REST / WebSocket). Replaces a module-level
    warn-once flag that CodeQL flagged as an unused global (the ``global``
    rebind is not recognised as a use).
    """
    logger.warning(
        "Reachy daemon %s is unauthenticated (no REACHY_DAEMON_TOKEN set). "
        "Anyone on the same network segment can issue robot commands. "
        "Set REACHY_DAEMON_TOKEN and prefer WSS/HTTPS with mutual TLS.",
        kind,
    )


def _warn_unauthenticated_once(kind: str) -> None:
    if not _daemon_auth_token():
        _emit_unauthenticated_warning(kind)


# ── REST API ─────────────────────────────────────────────────────


def api(host: str, port: int, path: str, method: str = "GET", data: dict | None = None) -> dict:
    """Call Reachy Mini daemon REST API."""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    _token = _daemon_auth_token()
    if _token:
        req.add_header("Authorization", f"Bearer {_token}")
    else:
        _warn_unauthenticated_once("REST API")
    body = json.dumps(data).encode() if data else None
    try:
        with urllib.request.urlopen(req, body, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "code": e.code}
    except Exception as e:
        return {"error": str(e)}


# ── Pose math ────────────────────────────────────────────────────


def rpy_to_pose(
    pitch_deg: float, roll_deg: float, yaw_deg: float, x_mm: float = 0, y_mm: float = 0, z_mm: float = 0
) -> list:
    """Convert RPY (degrees) + XYZ (mm) to 4x4 pose matrix."""
    p, r, y = math.radians(pitch_deg), math.radians(roll_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x_mm / 1000],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y_mm / 1000],
        [-sp, cp * sr, cp * cr, z_mm / 1000],
        [0, 0, 0, 1],
    ]


def identity_pose() -> list:
    """Return a 4x4 identity pose matrix."""
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


# ── Hardware link abstraction ───────────────────────────────────


class HardwareLink(ABC):
    """Abstract interface for real-time I/O with Reachy Mini hardware."""

    @abstractmethod
    async def start(self, on_joints: Callable, on_imu: Callable) -> None:
        """Begin receiving sensor data and enable command sending."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear down the connection."""

    @abstractmethod
    async def send_cmd(self, cmd: dict) -> None:
        """Send a real-time command to the robot."""


class ZenohLink(HardwareLink):
    """Wireless variant — real-time I/O via Device Connect's Zenoh transport."""

    def __init__(self, transport, prefix: str):
        self._transport = transport
        self._prefix = prefix

    async def start(self, on_joints: Callable, on_imu: Callable) -> None:
        async def _on_joints(data: bytes, _reply=None):
            try:
                on_joints(json.loads(data.decode()))
            except Exception:
                pass  # drop malformed/partial frame; keep the subscription alive

        async def _on_imu(data: bytes, _reply=None):
            try:
                on_imu(json.loads(data.decode()))
            except Exception:
                pass  # drop malformed/partial frame; keep the subscription alive

        await self._transport.subscribe(f"{self._prefix}/joint_positions", _on_joints)
        await self._transport.subscribe(f"{self._prefix}/imu_data", _on_imu)

    async def stop(self) -> None:
        pass  # Transport teardown handled by DeviceRuntime

    async def send_cmd(self, cmd: dict) -> None:
        await self._transport.publish(f"{self._prefix}/command", json.dumps(cmd).encode())


class WebSocketLink(HardwareLink):
    """Lite variant — real-time I/O via daemon's WebSocket."""

    _WS_CMD_MAP = {
        "head_pose": lambda c: {"type": "set_target", "head": [v for row in c["head_pose"] for v in row]},
        "antennas_joint_positions": lambda c: {"type": "set_antennas", "antennas": c["antennas_joint_positions"]},
        "body_yaw": lambda c: {"type": "set_body_yaw", "body_yaw": c["body_yaw"]},
        "torque": lambda c: {"type": "set_torque", "on": c["torque"], "ids": c.get("ids")},
    }

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._ws: Any = None
        self._read_task: asyncio.Task | None = None

    async def start(self, on_joints: Callable, on_imu: Callable) -> None:
        import websockets

        # Security hardening: authenticate to the daemon when a token is
        # configured; otherwise warn that the link is unauthenticated.
        _token = _daemon_auth_token()
        _extra_headers = {"Authorization": f"Bearer {_token}"} if _token else None
        if not _token:
            _warn_unauthenticated_once("WebSocket")
        _connect_kwargs = {}
        if _extra_headers:
            # websockets >=12 uses additional_headers; older uses extra_headers.
            try:
                import inspect as _inspect

                _sig = _inspect.signature(websockets.connect)
                _hdr_kw = "additional_headers" if "additional_headers" in _sig.parameters else "extra_headers"
                _connect_kwargs[_hdr_kw] = _extra_headers
            except (ValueError, TypeError):
                _connect_kwargs["extra_headers"] = _extra_headers
        self._ws = await websockets.connect(f"ws://{self._host}:{self._port}/ws/sdk", **_connect_kwargs)
        self._read_task = asyncio.create_task(self._read_loop(on_joints, on_imu))

    async def _read_loop(self, on_joints: Callable, on_imu: Callable) -> None:
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "joint_positions":
                    on_joints(msg)
                elif t == "imu_data":
                    on_imu(msg)
            except Exception:
                pass  # skip malformed frame; keep reading

    async def stop(self) -> None:
        if self._read_task:
            self._read_task.cancel()
        if self._ws:
            await self._ws.close()

    async def send_cmd(self, cmd: dict) -> None:
        if not self._ws:
            return
        for key, fn in self._WS_CMD_MAP.items():
            if key in cmd:
                await self._ws.send(json.dumps(fn(cmd)))
                return
