"""Check 4: the same checkpoint served as a LeRobot deployment over Zenoh drives the arm in wall time."""

from __future__ import annotations

import time

import numpy as np
import pytest
from checks import cached_checkpoint, free_endpoint, needs_module
from test_03_a_lerobot_checkpoint_drives_the_sim_arm import CHECKPOINT, arm_with_wrist_camera, inside_limits

pytest.importorskip("mujoco")
pytest.importorskip("zenoh")


def test_the_same_checkpoint_drives_the_arm_over_zenoh():
    needs_module("lerobot.remote_inference", "the [lerobot] extra")
    from strands_robots import Description, Policy
    from strands_robots.policy.lerobot import ZenohPolicy, serve, server_config

    path = cached_checkpoint(CHECKPOINT)
    cfg = server_config(
        Description.load("so101"), path, deployment="acceptance", endpoint=free_endpoint(), device="cpu"
    )
    with serve(cfg) as url:
        p = Policy(url)
        assert isinstance(p, ZenohPolicy)
        arm = arm_with_wrist_camera()
        start = arm.observe().joints.positions.copy()
        t = time.perf_counter()
        r = arm.run(p, seconds=2)
        wall = time.perf_counter() - t
    assert r.stopped_by == "seconds" and r.steps == 60 and r.fault is None, r
    assert 1.8 < wall < 6.0, wall  # paced by the deployment's clock, not as fast as the sim can go
    assert inside_limits(arm, r.final.joints.positions)
    assert np.abs(r.final.joints.positions - start).max() > 1e-3
