"""Check 2: two robots share one world; one robot's act leaves the other untouched; both in one picture."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")


def test_two_robots_share_one_world():
    from strands_robots import Robot

    arm, g1 = Robot("so101"), Robot("g1", position=[1.0, 0.0, 0.8])
    assert arm.sim is g1.sim and len(arm.sim.robots) == 2 and arm.id != g1.id
    g1_before = g1.observe().joints.positions.copy()
    t0 = arm.sim.time
    arm.act({"shoulder_pan": 0.3})
    assert arm.sim.time > t0  # one act moved the world's time by one control step
    g1.set_pose([2.0, 0.0, 0.8])
    assert np.allclose(g1.pose.position, [2.0, 0.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(g1.observe().joints.positions, g1_before, atol=5e-2)
    frame = arm.sim.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.std() > 1.0
