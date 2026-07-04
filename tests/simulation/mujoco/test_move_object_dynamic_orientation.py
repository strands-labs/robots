"""Regression tests: ``move_object`` must apply ORIENTATION to dynamic objects.

A dynamic object (``is_static=False``) carries its pose on a freejoint, whose
``data.qpos`` slice is laid out as ``[x, y, z, qw, qx, qy, qz]``. ``move_object``
takes the cheap path for these: it writes ``data.qpos`` directly and runs a
forward pass (no recompile). The position half of that path is exercised
elsewhere; these pin the orientation half - writing the quaternion into
``qpos[3:7]`` and mirroring it onto the stored :class:`SimObject`.

The contract these guard (mirroring the static-object tests): ``move_object``
never reports success without the object actually moving. For orientation that
means the *compiled body's* world quaternion (``data.xquat``) must reflect the
requested rotation after the forward pass - asserting the observable physical
effect, not the internal ``qpos`` write.
"""

import math

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_move_dynamic_orientation_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


def _body_xquat(world, name):
    bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found in compiled model"
    return [float(x) for x in world._data.xquat[bid]]


def _body_xpos(world, name):
    bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found in compiled model"
    return [float(x) for x in world._data.xpos[bid]]


def test_move_dynamic_object_orientation_only(sim):
    """An orientation-only move rotates the compiled body and stores the quat.

    A 90 deg rotation about +Z is ``[cos45, 0, 0, sin45]`` in MuJoCo's
    ``[w, x, y, z]`` convention. The body's world quaternion must match after
    the forward pass, and the stored ``SimObject.orientation`` must be updated
    (not left at its spawn identity).
    """
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.5], is_static=False)
    quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]

    result = sim.move_object("cube", orientation=quat)
    assert result["status"] == "success", result

    assert _body_xquat(sim._world, "cube") == pytest.approx(quat, abs=1e-6)
    assert list(sim._world.objects["cube"].orientation) == pytest.approx(quat)


def test_move_dynamic_object_position_and_orientation(sim):
    """Position and orientation supplied together are both applied in one call."""
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.5], is_static=False)
    target_pos = [0.2, 0.1, 0.4]
    identity = [1.0, 0.0, 0.0, 0.0]

    # First rotate away from identity so the reset-to-identity is observable.
    sim.move_object("cube", orientation=[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    result = sim.move_object("cube", position=target_pos, orientation=identity)
    assert result["status"] == "success", result

    assert _body_xpos(sim._world, "cube") == pytest.approx(target_pos, abs=1e-6)
    assert _body_xquat(sim._world, "cube") == pytest.approx(identity, abs=1e-6)
    assert list(sim._world.objects["cube"].position) == pytest.approx(target_pos)
    assert list(sim._world.objects["cube"].orientation) == pytest.approx(identity)
