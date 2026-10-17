"""Check 3: a real trained LeRobot checkpoint drives the sim SO-101 through ``Robot.run``, in-process.

The checkpoint is a published ACT policy trained on a real SO-101 with one wrist camera; it is run
from the local Hugging Face cache (offline). Graded on what the arm did: every step ran, no fault,
the arm stayed inside its joint limits, and the policy moved it.
"""

from __future__ import annotations

import numpy as np
import pytest
from checks import cached_checkpoint, needs_module

pytest.importorskip("mujoco")
CHECKPOINT = "robotfuel/act_so101_t16b"


def arm_with_wrist_camera():
    from strands_robots import Robot

    arm = Robot("so101")
    arm.sim.camera("wrist", size=(640, 480), attach=arm)
    return arm


def inside_limits(robot, q) -> bool:
    lim = robot.description.limits
    lo = np.array([lim[k][0] for k in robot.action_keys])
    hi = np.array([lim[k][1] for k in robot.action_keys])
    return bool(np.all(np.isfinite(q)) and np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))


def test_a_lerobot_checkpoint_drives_the_sim_arm():
    needs_module("lerobot", "the [lerobot] extra")
    from strands_robots import Policy

    path = cached_checkpoint(CHECKPOINT)
    arm = arm_with_wrist_camera()
    start = arm.observe().joints.positions.copy()
    r = arm.run(Policy(path, device="cpu"), seconds=2)
    assert r.stopped_by == "steps" and r.steps == 60 and r.fault is None, r
    assert inside_limits(arm, r.final.joints.positions)
    assert np.abs(r.final.joints.positions - start).max() > 1e-3  # the policy moved the arm
