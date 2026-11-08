"""Glue for low-level policies on a ROS body: the robot's joints in wire (hardware) order.

g1 = ros.connect("unitree_g1", released=True, shadow=True)       # shadow first: computes, publishes nothing
pol = MultiOnnxPolicy("walk.onnx", inputs={...}, outputs={...}, joint_order=ros.wire_order(g1))
g1.run(pol, seconds=20)
print(g1.observe().extras["stream"])                              # sent / late / held / shadow_max_err
"""

from __future__ import annotations

from typing import Any


def wire_order(robot: Any) -> list[str]:
    """The robot's action_keys sorted by their wire id (hardware order: what a vendor graph is trained on)."""
    prof = robot.body.profile
    return sorted(robot.action_keys, key=lambda k: prof.wire_id(k))
