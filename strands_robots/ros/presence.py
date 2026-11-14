"""Show any strands Robot on a ROS graph as a native ROS device (the reverse direction of RosDriver).

    RosPresence(mesh.connect("lab-arm"), namespace="/lab_arm").start()   # a mesh peer across the internet
    RosPresence(Robot("so101", port="/dev/ttyACM0")).start()          # a serial arm, now on RViz/Foxglove

Out: ``<ns>/joint_states`` (sensor_msgs/JointState), ``<ns>/robot_description`` (latched URDF/MJCF text),
``<ns>/odom`` when the body reports a pose, ``<ns>/<camera>/image_raw`` (rgb8) when ``images=True``.
In (off by default): ``<ns>/joint_command`` (JointState). Anyone on the domain can publish it, so it needs
``commands=True`` AND ``STRANDS_ROS_OPEN_COMMANDS=1``; every message goes through the robot's safety chain
and operator gate as an agent command (``confirm_act`` under ``from_agent``), whole message or nothing.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from .bus import RosBus
from .convert import from_dict, message_class
from .names import canonical

LOG = logging.getLogger(__name__)
JS = "sensor_msgs/msg/JointState"


class RosPresence:
    def __init__(
        self,
        robot: Any,
        *,
        namespace: str | None = None,
        bus: RosBus | None = None,
        rate_hz: float = 30.0,
        images: bool = False,
        commands: bool = False,
    ) -> None:
        if commands and os.environ.get("STRANDS_ROS_OPEN_COMMANDS") != "1":
            raise PermissionError(
                "commands=True opens <ns>/joint_command to every participant on this ROS domain; "
                "set STRANDS_ROS_OPEN_COMMANDS=1 to say you accept that"
            )
        self.robot, self.bus = robot, bus or RosBus.shared()
        self.ns = canonical(namespace or "/" + str(robot.id).replace("-", "_"))
        self.dt, self.images, self.commands = 1.0 / float(rate_hz), bool(images), bool(commands)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.refused, self.errors, self.last_error = 0, 0, ""

    def start(self) -> RosPresence:
        model = getattr(self.robot.description, "model_path", None)
        if model is not None and os.path.isfile(model):
            msg = from_dict(message_class("std_msgs/msg/String"), {"data": Path(model).read_text()})
            self.bus.publish(f"{self.ns}/robot_description", "std_msgs/msg/String", msg, qos="latched")
        if self.commands:
            self.bus.subscribe(f"{self.ns}/joint_command", JS, qos="reliable", listener=self._on_command)
        self._thread = threading.Thread(target=self._loop, name=f"ros-presence-{self.ns}", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.dt):
            try:
                self.publish_once()
            except Exception as e:  # noqa: BLE001 - a lost peer or a slow camera never kills the presence
                self.errors, self.last_error = self.errors + 1, f"{type(e).__name__}: {e}"
                if self.errors == 1:
                    LOG.warning("ros presence %s: %s", self.ns, self.last_error)

    def publish_once(self) -> None:
        obs = self.robot.observe(images=self.images)
        j = obs.joints
        now = time.time()
        stamp = {"sec": int(now), "nanosec": int((now % 1) * 1e9)}
        self.bus.publish(
            f"{self.ns}/joint_states",
            JS,
            from_dict(
                message_class(JS),
                {
                    "header": {"stamp": stamp},
                    "name": list(j.names),
                    "position": [float(x) for x in j.positions],
                    "velocity": [float(x) for x in j.velocities],
                    "effort": [float(x) for x in j.efforts],
                },
            ),
            qos="sensor",
        )
        if obs.pose is not None:
            p, q = obs.pose.position, obs.pose.orientation
            self.bus.publish(
                f"{self.ns}/odom",
                "nav_msgs/msg/Odometry",
                from_dict(
                    message_class("nav_msgs/msg/Odometry"),
                    {
                        "header": {"stamp": stamp, "frame_id": "odom"},
                        "child_frame_id": "base_link",
                        "pose": {
                            "pose": {
                                "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                                "orientation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])},
                            }
                        },
                    },
                ),
                qos="sensor",
            )
        for cam, img in (obs.images or {}).items():
            h, w = img.shape[:2]
            self.bus.publish(
                f"{self.ns}/{cam}/image_raw",
                "sensor_msgs/msg/Image",
                from_dict(
                    message_class("sensor_msgs/msg/Image"),
                    {
                        "header": {"stamp": stamp, "frame_id": cam},
                        "height": h,
                        "width": w,
                        "encoding": "rgb8",
                        "step": w * 3,
                        "data": img.tobytes(),
                    },
                ),
                qos="sensor",
            )

    def _on_command(self, msg: Any) -> None:
        names, pos = list(msg.name), [float(x) for x in msg.position]
        keys = set(self.robot.action_keys)
        if len(names) != len(pos) or not names or any(n not in keys for n in names) or not all(map(math.isfinite, pos)):
            self.refused += 1
            LOG.warning("%s/joint_command refused: names/positions mismatch, unknown joint or non-finite", self.ns)
            return
        from ..robot.run.source import from_agent

        try:
            with from_agent():  # Robot.confirm_act: the operator gate, and every ramp point checked again
                self.robot.confirm_act(dict(zip(names, pos, strict=True)), source="ros")
        except Exception as e:  # noqa: BLE001 - a refusal is logged, never raised into the executor
            self.refused += 1
            LOG.warning("%s/joint_command refused: %s", self.ns, e)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
