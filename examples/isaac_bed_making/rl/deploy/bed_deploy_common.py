"""Shared constants + loaders for the on-robot bed-making deploy scripts.

Single source of truth for the arm-joint set, the conservative clamps (a SAFETY backstop — must not
drift between the reach and the pull), and the robotics-connect module loader. Imported by the
deploy scripts that sit alongside it (they run flat on the robot, so a same-dir import is robust).
"""
from __future__ import annotations

import importlib.util
import os
import sys

LEFT_ARM = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint"]
RIGHT_ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint"]
# 10 arm joints driven via rt/arm_sdk (5 per arm; the EDU lacks wrist pitch/yaw).
ARM_JOINTS = LEFT_ARM + RIGHT_ARM

# Conservative arm clamps (rad) — a backstop; the ladder's rate limit is the primary smoother.
ARM_LIMITS = {
    "left_shoulder_pitch_joint": (-2.6, 2.6), "right_shoulder_pitch_joint": (-2.6, 2.6),
    "left_shoulder_roll_joint": (-1.5, 2.2), "right_shoulder_roll_joint": (-2.2, 1.5),
    "left_shoulder_yaw_joint": (-2.5, 2.5), "right_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-1.0, 2.0), "right_elbow_joint": (-1.0, 2.0),
    "left_wrist_roll_joint": (-1.9, 1.9), "right_wrist_roll_joint": (-1.9, 1.9),
}


def rc_root() -> str:
    """Locate the robotics-connect checkout (the deploy copy first, then the dev box)."""
    env = os.environ.get("ROBOTICS_CONNECT_ROOT")
    for c in ([env] if env else []) + ["/home/unitree/robotics-connect-deploy",
                                        "/home/unitree/robotics-connect",
                                        "/home/aifabric/workspaces/git/robotics-connect"]:
        if c and os.path.exists(os.path.join(c, "lib", "policy_deploy.py")):
            return c
    raise FileNotFoundError("robotics-connect not found — set ROBOTICS_CONNECT_ROOT")


def load(name: str, path: str):
    """Import a module by file path (robotics-connect modules are loaded explicitly, not on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
