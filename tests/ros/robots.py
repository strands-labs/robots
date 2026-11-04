"""Fake robots on the fake graph: a G1 (unitree_hg) and a ros2_control arm (the add-on toy is in toy.py)."""

from __future__ import annotations

import threading
import time
from typing import Any

from .fake_rclpy import Durability, Node, QoSProfile, Reliability, msg_class

BE = QoSProfile(depth=10, reliability=Reliability.BEST_EFFORT)
REL = QoSProfile(depth=10)
LATCH = QoSProfile(depth=1, durability=Durability.TRANSIENT_LOCAL)


def urdf(joints: dict[str, tuple[float, float]], name: str = "robot", floating: bool = False) -> str:
    links = ['<link name="base"><visual><geometry><mesh filename="package://x/base.stl"/></geometry></visual></link>']
    js = []
    for i, (j, (lo, hi)) in enumerate(joints.items()):
        links.append(
            f'<link name="l{i}"><inertial><mass value="0.1"/>'
            f'<inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>'
        )
        js.append(
            f'<joint name="{j}" type="revolute"><parent link="base"/><child link="l{i}"/>'
            f'<axis xyz="0 0 1"/><limit lower="{lo}" upper="{hi}" effort="10" velocity="10"/></joint>'
        )
    return f'<robot name="{name}">{"".join(links)}{"".join(js)}</robot>'


class FakeArm:
    """A ros2_control arm: /joint_states (RELIABLE), /robot_description latched, forward position controller in."""

    JOINTS = {"shoulder": (-2.0, 2.0), "elbow": (-1.5, 1.5), "wrist": (-3.0, 3.0)}

    def __init__(self, ns: str = "") -> None:
        self.node = Node("fake_arm")
        self.ns = ns
        self.js = self.node.create_publisher(msg_class("sensor_msgs/JointState"), f"{ns}/joint_states", REL)
        self.desc = self.node.create_publisher(msg_class("std_msgs/String"), f"{ns}/robot_description", LATCH)
        self.commands: list[Any] = []
        self.node.create_subscription(
            msg_class("std_msgs/Float64MultiArray"),
            f"{ns}/forward_position_controller/commands",
            self.commands.append,
            REL,
        )
        self.desc.publish(msg_class("std_msgs/String")(data=urdf(self.JOINTS, "arm")))
        self.q = {"wrist": 0.3, "shoulder": 0.1, "elbow": -0.2}  # wire order differs from the model's
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop:
            self.js.publish(
                msg_class("sensor_msgs/JointState")(
                    name=list(self.q), position=list(self.q.values()), velocity=[0.0] * 3, effort=[0.0] * 3
                )
            )
            try:
                while True:
                    cb, msg = self.node.queue.get_nowait()
                    cb(msg)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)

    def close(self) -> None:
        self._stop = True
        time.sleep(0.03)
        self.node.destroy_node()


G1_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


class FakeG1:
    def __init__(self) -> None:
        self.node = Node("fake_g1")
        self.ls = self.node.create_publisher(msg_class("unitree_hg/LowState"), "/lowstate", REL)
        self.cmds: list[Any] = []
        self.node.create_subscription(msg_class("unitree_hg/LowCmd"), "/lowcmd", self.cmds.append, REL)
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop:
            s = msg_class("unitree_hg/LowState")(mode_machine=5, tick=1)
            for i, m in enumerate(s.motor_state):
                m.q, m.temperature = 0.01 * i, [30, 31]
            s.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
            s.imu_state.gyroscope = [0.0, 0.0, 0.1]
            self.ls.publish(s)
            try:
                while True:
                    cb, msg = self.node.queue.get_nowait()
                    cb(msg)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.002)

    def close(self) -> None:
        self._stop = True
        time.sleep(0.01)
        self.node.destroy_node()
