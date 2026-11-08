"""A made-up robot that plugs in from OUTSIDE strands_robots, the way any vendor add-on would.

``toy_msgs`` is invented: one sensor frame (``seq``, joint rows by id, an IMU) answered by exactly one command
that echoes ``seq`` and the frame stamp, per-joint kp/kd on the wire, a status message that says when the
low-level API is armed, a gains topic, and a reference command topic for shadow mode. Its codec and its
profile live here, in the tests, and reach the driver only through the public hooks: a profile directory on
``STRANDS_ROBOTS_ROS_PROFILES`` and ``"codec": "<module>:<Class>"``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.robot.drivers.ros import Codec
from strands_robots.ros.convert import from_dict, message_class
from strands_robots.types import JointState

from . import fake_rclpy
from .fake_rclpy import Node, QoSProfile, Reliability, msg_class

SPECS = {
    "toy_msgs/JointRow": {"id": "uint8", "q": "float", "dq": "float", "tau": "float", "stale": "boolean"},
    "toy_msgs/Frame": {
        "stamp": "builtin_interfaces/Time",
        "seq": "uint32",
        "rows": "toy_msgs/JointRow[]",
        "imu": "sensor_msgs/Imu",
    },
    "toy_msgs/JointCmd": {"id": "uint8", "q": "float", "kp": "float", "kd": "float"},
    "toy_msgs/Command": {"frame_stamp": "builtin_interfaces/Time", "seq": "uint32", "cmds": "toy_msgs/JointCmd[]"},
    "toy_msgs/Status": {"mode": "uint8", "api": "boolean"},
    "toy_msgs/Gains": {"ids": "uint8[]", "kp": "float[]", "kd": "float[]"},
}
fake_rclpy.SPECS.update(SPECS)

JOINTS = {"hip": (3, -1.5, 1.5), "knee": (1, 0.0, 2.5), "ankle": (7, -1.0, 1.0), "neck": (2, -1.0, 1.0)}


class ToyCodec(Codec):
    """The whole robot-specific code an add-on writes: read a frame, build a command, read gains."""

    needs_gains = True

    def read(self, state: Any) -> JointState:
        rows = {int(r.id): r for r in state.rows}
        if any(rows[i].stale for i in self.ids if i in rows):
            raise ValueError("stale joints in the frame")
        missing = [i for i in self.ids if i not in rows]
        if missing:
            raise ValueError(f"frame has no joints {missing}")
        col = lambda f: np.array([float(getattr(rows[i], f)) for i in self.ids])  # noqa: E731
        return JointState(self.names, col("q"), col("dq"), col("tau"))

    def build(self, q: np.ndarray, kp: np.ndarray | None, kd: np.ndarray | None, state: Any, sender: str) -> Any:
        if kp is None or kd is None:
            raise ValueError("the toy needs kp/kd per joint")
        cmds = [
            {"id": int(i), "q": float(q[k]), "kp": float(kp[k]), "kd": float(kd[k])} for k, i in enumerate(self.ids)
        ]
        return from_dict(message_class(self.profile.need("command")["type"]), {"cmds": cmds})

    def positions(self, command: Any) -> np.ndarray | None:
        rows = {int(c.id): float(c.q) for c in command.cmds}
        return np.array([rows.get(i, np.nan) for i in self.ids]) if rows else None

    def imu(self, state: Any) -> tuple[np.ndarray, np.ndarray] | None:
        o, g = state.imu.orientation, state.imu.angular_velocity
        return np.array([o.w, o.x, o.y, o.z]), np.array([g.x, g.y, g.z])

    def default_gains(self, info: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if info is None:
            return None
        kp = dict(zip(info.ids, info.kp, strict=True))
        kd = dict(zip(info.ids, info.kd, strict=True))
        return np.array([kp[i] for i in self.ids]), np.array([kd[i] for i in self.ids])


def profile(codec: str = f"{__name__}:ToyCodec") -> dict[str, Any]:
    return {
        "family": "toy",
        "codec": codec,
        "description": "A made-up frame-synced biped for tests.",
        "model": {"topic": "/toy/robot_description", "type": "std_msgs/msg/String", "qos": "latched"},
        "joints": {k: v[0] for k, v in JOINTS.items()},
        "state": {"topic": "/toy/frame", "type": "toy_msgs/msg/Frame", "qos": "sensor", "max_age_ms": 200},
        "command": {
            "topic": "/toy/command",
            "type": "toy_msgs/msg/Command",
            "qos": "reliable",
            "trigger": "state",
            "deadman_ms": 300,
            "target_max_age_ms": 100,
            "echo": {"seq": "seq", "frame_stamp": "stamp"},
        },
        "status": {
            "topic": "/toy/status",
            "type": "toy_msgs/msg/Status",
            "qos": "reliable",
            "armed": {"mode": 3, "api": True},
            "labels": {"mode": {"0": "OFF", "2": "STAND", "3": "LOWLEVEL"}},
        },
        "gains_from": {"topic": "/toy/gains", "type": "toy_msgs/msg/Gains", "qos": "auto"},
        "shadow": {"topic": "/toy/reference_command", "type": "toy_msgs/msg/Command", "qos": "reliable"},
        "stop": [
            {"topic": "/toy/cmd_vel", "type": "geometry_msgs/msg/Twist", "qos": "reliable", "repeat": 2, "fields": {}}
        ],
        "surfaces": ["/toy/command", "/toy/mode_request"],
    }


def write_profile(directory: Path, **kw: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / "toy_biped.json"
    f.write_text(json.dumps(profile(**kw)))
    return f


BE = QoSProfile(depth=10, reliability=Reliability.BEST_EFFORT)
REL = QoSProfile(depth=10)


class FakeToy:
    """The toy on the fake graph: publishes frames on demand, records what it is sent."""

    def __init__(self) -> None:
        from .robots import LATCH, urdf

        self.node = Node("fake_toy")
        self.state = self.node.create_publisher(msg_class("toy_msgs/Frame"), "/toy/frame", BE)
        self.sys = self.node.create_publisher(msg_class("toy_msgs/Status"), "/toy/status", REL)
        self.gains = self.node.create_publisher(msg_class("toy_msgs/Gains"), "/toy/gains", REL)
        self.reference = self.node.create_publisher(msg_class("toy_msgs/Command"), "/toy/reference_command", REL)
        desc = self.node.create_publisher(msg_class("std_msgs/String"), "/toy/robot_description", LATCH)
        self.commands: list[Any] = []
        self.twists: list[Any] = []
        self.node.create_subscription(msg_class("toy_msgs/Command"), "/toy/command", self.commands.append, REL)
        self.node.create_subscription(msg_class("geometry_msgs/Twist"), "/toy/cmd_vel", self.twists.append, REL)
        self._stop = False
        threading.Thread(target=self._drain, daemon=True).start()
        self.seq = 0
        self.q = {i: min(max(0.1 * i, lo), hi) for (i, lo, hi) in JOINTS.values()}
        desc.publish(
            msg_class("std_msgs/String")(data=urdf({k: (lo, hi) for k, (_i, lo, hi) in JOINTS.items()}, "toy"))
        )
        self._gains = msg_class("toy_msgs/Gains")(
            ids=[i for i, _, _ in JOINTS.values()], kp=[20.0 + i for i, _, _ in JOINTS.values()], kd=[1.0] * 4
        )
        self.set_mode(2)

    def _drain(self) -> None:
        while not self._stop:
            try:
                cb, msg = self.node.queue.get(timeout=0.05)
            except Exception:  # noqa: BLE001
                continue
            cb(msg)

    def set_mode(self, mode: int, api: bool = True) -> None:
        self._sys = {"mode": mode, "api": api}
        self.sys.publish(msg_class("toy_msgs/Status")(**self._sys))

    def frame(self) -> int:
        self.sys.publish(msg_class("toy_msgs/Status")(**self._sys))
        self.gains.publish(self._gains)
        self.seq += 1
        row = msg_class("toy_msgs/JointRow")
        f = msg_class("toy_msgs/Frame")(seq=self.seq, rows=[row(id=i, q=q) for i, q in self.q.items()])
        f.imu.orientation.w, f.imu.orientation.z = 0.7071, 0.7071
        f.imu.angular_velocity.z = 0.25
        f.stamp.sec = 7
        self.state.publish(f)
        return self.seq

    def frames(self, n: int, dt: float = 0.004) -> list[int]:
        out = []
        for _ in range(n):
            out.append(self.frame())
            time.sleep(dt)
        return out

    def connect(self, fn: Any) -> Any:
        """Call ``fn()`` while frames flow (a real robot never stops sending), then stop the pump."""
        done = threading.Event()

        def pump() -> None:
            while not done.is_set():
                self.frame()
                time.sleep(0.02)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            return fn()
        finally:
            done.set()
            t.join()

    def close(self) -> None:
        self._stop = True
        self.node.destroy_node()
