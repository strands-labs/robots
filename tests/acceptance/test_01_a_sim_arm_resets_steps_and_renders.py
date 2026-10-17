"""Check 1: a sim SO-101 resets, steps and renders: a joint command moves the joint and the picture."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

JOINT = "shoulder_lift"
TARGET = 0.5  # rad, inside the joint's range


def test_a_sim_arm_resets_steps_and_renders():
    from strands_robots import Robot

    arm = Robot("so101")
    arm.sim.camera("front", size=(640, 480), position=(0.6, -0.6, 0.5), target=(0.0, 0.0, 0.15))
    start = arm.observe().joints.positions.copy()
    before = arm.sim.render("front")
    assert before.shape == (480, 640, 3) and before.dtype == np.uint8 and before.std() > 1.0

    for _ in range(60):
        arm.act({JOINT: TARGET})
    moved = arm.observe()[JOINT]
    after = arm.sim.render("front")
    assert abs(moved - TARGET) < 0.05, moved
    assert np.abs(after.astype(int) - before.astype(int)).mean() > 0.5  # the camera saw the arm move

    arm.sim.reset()
    np.testing.assert_allclose(arm.observe().joints.positions, start, atol=1e-9)
