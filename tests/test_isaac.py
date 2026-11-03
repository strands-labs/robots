"""Simulation("isaac"): two robots in one PhysX world, one SimulationApp per process.

Runs where Isaac Sim is installed (a CUDA box): ``pytest -s tests/test_isaac.py`` (``-s``: SimulationApp
enables faulthandler, which needs a real stderr). Everywhere else every test skips with the reason.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pytest

from strands_robots import Robot, Simulation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("isaacsim") is None, reason="isaacsim not installed (needs a CUDA box with Isaac Sim)"
)


@pytest.fixture(scope="module")
def world():
    t = time.perf_counter()
    sim = Simulation("isaac")
    boot = time.perf_counter() - t
    t = time.perf_counter()
    arm = Robot("so101", sim=sim)
    g1 = Robot("g1", sim=sim, position=[1, 0, 0])
    print(f"\n[measured] isaac boot {boot:.1f} s, two robots loaded {time.perf_counter() - t:.1f} s", flush=True)
    yield sim, arm, g1
    sim.close()


def test_two_robots_share_one_isaac_world(world):
    sim, arm, g1 = world
    assert arm.sim is g1.sim is sim and len(sim.robots) == 2
    assert sim.physics.name == "physx"
    a, g = arm.observe(), g1.observe()
    assert a.joints.names == arm.description.action_keys and len(g.joints.names) == len(g1.description.action_keys)
    key = arm.description.action_keys[0]
    t0 = sim.time
    for _ in range(100):
        arm.act({key: 0.3})
    assert sim.time > t0
    assert arm.observe()[key] == pytest.approx(0.3, abs=0.05)
    g1.set_pose(position=[2, 0, 0.8])
    assert np.allclose(g1.pose.position, [2, 0, 0.8], atol=1e-3)
    n = 1000
    t = time.perf_counter()
    sim.step(n)
    dt = time.perf_counter() - t
    img = sim.render()
    assert img.ndim == 3 and img.shape[2] == 3 and img.std() > 5.0
    from PIL import Image

    out = Path(os.environ.get("STRANDS_ROBOTS_MEDIA", "/tmp")) / "isaac_two_robots.png"
    Image.fromarray(img).save(out)
    print(f"\n[measured] isaac {n} physics steps {dt * 1e3:.0f} ms = {n / dt:,.0f} steps/s, render {img.shape}")


def test_a_second_isaac_world_is_refused_while_one_is_open(world):
    with pytest.raises(RuntimeError, match="already open in this process"):
        Simulation("isaac")


def test_newton_physics_on_isaac_is_refused_with_the_way_out():
    with pytest.raises((NotImplementedError, ValueError), match="newton"):
        Simulation("isaac", physics="newton")


def test_a_policy_runs_through_robot_run_on_isaac(world):
    sim, arm, _ = world
    key = arm.description.action_keys[1]
    t = time.perf_counter()
    res = arm.run(lambda obs: {key: -0.4}, hz=50, steps=100)
    wall = time.perf_counter() - t
    assert res.steps == 100 and res.stopped_by == "steps"
    assert res.final is not None and res.final[key] == pytest.approx(-0.4, abs=0.05)
    print(f"\n[measured] isaac Robot.run callable: 100 steps @50 Hz sim-time {res.seconds:.2f} s in {wall:.2f} s wall")


def test_two_policies_run_in_lockstep_on_isaac(world):
    from strands_robots.policy import SequencePolicy

    sim, arm, g1 = world
    a_key, g_key = arm.description.action_keys[0], g1.description.action_keys[0]
    t0 = sim.time
    out = sim.run({arm: SequencePolicy([{a_key: 0.2}]), g1: lambda obs: {g_key: 0.1}}, hz=50, steps=50)
    assert {r.steps for r in out.values()} == {50}
    assert sim.time == pytest.approx(t0 + 50 * sim.physics.control_dt, rel=1e-3)
    assert arm.observe()[a_key] == pytest.approx(0.2, abs=0.05)
