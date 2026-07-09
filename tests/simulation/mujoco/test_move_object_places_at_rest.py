"""Regression tests: ``move_object`` places a dynamic object AT REST at the new pose.

A dynamic object (``is_static=False``) carries its pose on a freejoint whose
6 velocity DOF (3 linear + 3 angular) live at ``data.qvel[dof_addr : dof_addr+6]``.
Writing ``data.qpos`` to reposition the body does NOT touch that velocity, so a
freejoint retains whatever momentum it had. Without an explicit reset,
``move_object`` teleports the body's *position* while leaving its prior velocity
intact - a settling object shoots off the instant it is "placed", and an
eval/benchmark loop that repositions objects between episodes starts each
episode with the object drifting (silently non-reproducible).

The contract these pin: ``move_object`` places the object at rest at the new
pose, matching ``add_object`` (spawns at rest), ``reset`` (zeroes velocities),
and the Newton backend (rebuilds from the builder at rest). They assert the
observable physical state (``data.qvel`` and the post-step trajectory), not the
internal write.
"""

import math

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_move_object_at_rest_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


def _freejoint_qvel(world, name):
    """Return the 6-DOF freejoint velocity [vx,vy,vz, wx,wy,wz] of an object."""
    m, d = world._model, world._data
    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
    assert jid >= 0, f"freejoint for {name!r} not found"
    adr = m.jnt_dofadr[jid]
    return [float(v) for v in d.qvel[adr : adr + 6]]


def test_position_move_zeroes_velocity(sim):
    """A position move zeroes the object's velocity so it is placed at rest.

    Let a cube fall to build a real downward velocity, then reposition it. Its
    freejoint velocity must be zero afterwards, so it starts falling FROM REST
    rather than continuing at its prior speed.
    """
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.6], is_static=False)
    sim.step(40)
    vel_before = _freejoint_qvel(sim._world, "cube")
    assert abs(vel_before[2]) > 0.3, f"cube should have gained downward velocity, got {vel_before}"

    result = sim.move_object("cube", position=[0.0, 0.0, 0.6])
    assert result["status"] == "success", result

    vel_after = _freejoint_qvel(sim._world, "cube")
    assert vel_after == pytest.approx([0.0] * 6, abs=1e-9), f"expected at rest, got {vel_after}"

    # One step of free fall from rest imparts only g*dt (~0.0196 m/s at dt=2ms),
    # not the ~0.8 m/s the object carried before the move.
    sim.step(1)
    assert abs(_freejoint_qvel(sim._world, "cube")[2]) < 0.1


def test_orientation_move_zeroes_angular_velocity(sim):
    """An orientation move zeroes angular (and linear) velocity too."""
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.6], is_static=False)
    # Inject a spin + drift directly onto the freejoint, then settle one step.
    m, d = sim._world._model, sim._world._data
    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "cube_joint")
    adr = m.jnt_dofadr[jid]
    d.qvel[adr : adr + 6] = [0.5, 0.0, 0.0, 0.0, 0.0, 3.0]
    mj.mj_forward(m, d)
    assert any(abs(v) > 0.1 for v in _freejoint_qvel(sim._world, "cube"))

    quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    result = sim.move_object("cube", orientation=quat)
    assert result["status"] == "success", result

    assert _freejoint_qvel(sim._world, "cube") == pytest.approx([0.0] * 6, abs=1e-9)


def test_no_arg_move_does_not_zero_velocity(sim):
    """A move_object call with neither position nor orientation leaves velocity
    untouched - it is a genuine no-op, not a stealth 'freeze'."""
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.6], is_static=False)
    sim.step(40)
    vel_before = _freejoint_qvel(sim._world, "cube")
    assert abs(vel_before[2]) > 0.3

    result = sim.move_object("cube")
    assert result["status"] == "success", result

    assert _freejoint_qvel(sim._world, "cube")[2] == pytest.approx(vel_before[2], abs=1e-6)
