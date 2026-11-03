"""Pollen Microduck over robotd's JSON-RPC 2.0 (one object per line, unix socket): the READ path.

robotd owns the duck's 50 Hz loop and runs the walking policy on the board; its wire has no per-joint
write (``robot.move``, ``robot.head`` and ``robot.do`` are intents). So this driver reads: ``robot.state``
gives the 14 locomotion joints (robotd's vector is 15 wide with ``mouth`` at index 9, which is dropped),
and IMU, odometry and safety ride in ``extras``. ``write`` raises and says why.

``port=`` is the socket path; without it ``$MICRODUCK_SOCKET``, then ``/run/robotd.sock`` (this code
running on the duck). From a laptop, forward it first:
``ssh -N -L /tmp/robotd.sock:/run/robotd.sock radxa@<duck>``. Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...description import Description
from ...types import JointState, Pose
from .calibration import Calibration

DEFAULT_SOCKET = "/run/robotd.sock"
API_VERSION = (
    31  # duck-ipc-proto API_VERSION this wire was checked against (microduck 0.14.1); a mismatch is not refused
)
HARDWARE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOUTH_INDEX = 9


def line(obj: Mapping[str, Any]) -> bytes:
    """One NDJSON frame, compact like serde_json: what robotd reads."""
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


class RobotdClient:
    """One robotd connection: Hello, ``robot.subscribe``, then a reader thread that keeps the latest state."""

    def __init__(self, path: str, *, timeout: float = 2.0) -> None:
        self.path = path
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(path)
        except OSError as exc:
            self.sock.close()
            raise ConnectionError(
                f"microduck: robotd socket {path!r} did not answer ({exc}); on a laptop forward it with "
                "ssh -N -L /tmp/robotd.sock:/run/robotd.sock radxa@<duck> and pass port='/tmp/robotd.sock'"
            ) from exc
        self.rfile = self.sock.makefile("rb")
        self.latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.hello = self._request("hello", {"api_version": API_VERSION})  # before the reader: nothing else reads yet
        self.sock.settimeout(None)
        self._reader = threading.Thread(target=self._read_loop, name="robotd-reader", daemon=True)
        self._reader.start()
        self._send("robot.subscribe", {})

    def _send(self, method: str, params: Mapping[str, Any]) -> int:
        with self._lock:
            self._next_id += 1
            self.sock.sendall(line({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}))
            return self._next_id

    def _request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        rid = self._send(method, params)
        while True:
            raw = self.rfile.readline()
            if not raw:
                raise ConnectionError(f"microduck: robotd closed the connection during {method!r}")
            obj = json.loads(raw)
            if obj.get("id") == rid:
                if obj.get("error"):
                    raise ConnectionError(f"microduck: robotd refused {method!r}: {obj['error']}")
                return dict(obj.get("result") or {})

    def _read_loop(self) -> None:
        try:
            for raw in self.rfile:
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                if obj.get("method") == "robot.state":
                    self.latest = dict(obj.get("params") or {})
        except (OSError, ValueError):
            pass  # the socket closed under the reader: read() then reports the silence

    def read(self) -> dict[str, Any] | None:
        return self.latest

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already half-closed
        self.sock.close()


class MicroduckDriver:
    """Hardware Body for a Microduck: joints from robotd's ``robot.state``. Observe-only by robotd's design."""

    supports_position_hold = False  # observe-only: nothing of ours writes a goal
    calibration: Calibration | None = None  # the vendor firmware owns it
    resolution: int | None = None

    def __init__(
        self,
        description: Description,
        *,
        port: str | None = None,
        cameras: Mapping[str, int | str] | None = None,
        client: Any = None,
        timeout: float = 2.0,
    ) -> None:
        self.description = description
        self.names = description.action_keys
        unknown = [n for n in self.names if n not in HARDWARE_JOINT_NAMES or n == "mouth"]
        if unknown:
            raise ValueError(f"microduck: joints robotd does not report: {unknown}")
        self.index = [HARDWARE_JOINT_NAMES.index(n) for n in self.names]
        path = port or os.environ.get("MICRODUCK_SOCKET") or DEFAULT_SOCKET
        self.client = client if client is not None else RobotdClient(path, timeout=timeout)
        end = time.monotonic() + timeout
        while self.client.read() is None and time.monotonic() < end:
            time.sleep(0.02)
        if self.client.read() is None:
            self.client.close()
            raise ConnectionError(f"microduck: no robot.state from robotd within {timeout}s on {path!r}")

    def _state(self) -> dict[str, Any]:
        state = self.client.read()
        if state is None:
            raise ConnectionError("microduck: the robot.state stream went silent")
        return state

    def read(self) -> JointState:
        joints = list(self._state().get("joints") or [])
        if len(joints) != len(HARDWARE_JOINT_NAMES):
            raise ValueError(
                f"microduck: robot.state carries {len(joints)} joints, expected {len(HARDWARE_JOINT_NAMES)}"
            )
        q = np.array([float(joints[i]) for i in self.index])
        z = np.zeros(len(self.names))  # robotd reports positions only
        return JointState(self.names, q, z, z.copy())

    def write(self, targets: np.ndarray) -> None:
        raise NotImplementedError(
            "microduck: robotd has no per-joint write (its wire is intents: robot.move, robot.head, robot.do, "
            "and the walking policy runs on the duck); this driver is observe-only"
        )

    def pose(self) -> Pose | None:
        return None  # dead-reckoned odometry is in extras["odom"]; the duck has no world pose

    def images(self) -> Mapping[str, np.ndarray]:
        return {}

    def extras(self) -> Mapping[str, Any]:
        s = self._state()
        return {k: s[k] for k in ("imu", "odom", "safety", "policy", "t") if k in s}

    def hold(self) -> None:
        """Observe-only: this driver never commands a joint, so there is nothing of ours to hold."""

    def estop(self) -> None:
        """Observe-only: nothing of ours is moving the body; the vendor's own stop stays in charge."""

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        raise RuntimeError("microduck: robotd owns the servo calibration; nothing to calibrate from here")

    def close(self) -> None:
        self.client.close()
