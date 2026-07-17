"""Regression tests: a floating-base robot spawns SEATED on the local terrain.

``create_world(terrain=...)`` (#1336/#1338/#1339/#1340) lays a heightfield so a
locomotion robot can be spawned and evaluated on non-flat ground -- that is the
feature's stated purpose. But a robot's model spawns its free base at the
flat-ground keyframe height (e.g. the Unitree Go2 base at ``z=0.445``, feet
~``z=0.02``), which on a raised heightfield leaves the feet BELOW the terrain
surface: the robot spawns *buried* in the ground, with penetration that grows
with the curriculum ``difficulty``. Worse, a naive one-shot fix in ``add_robot``
would not survive ``reset()`` (which ``run_policy`` / ``eval_policy`` call before
and between episodes), snapping the base back to the buried flat-ground height.

The engine now seats every floating base on the local terrain -- offsetting its
``z`` by ``_ground_height_at(x, y)`` -- at ``add_robot`` spawn AND on every
``reset()``, so the robot rests on the surface (feet just clear of it) at the
start of every episode. A flat ground plane is a no-op (the height is ``0.0``)
and a fixed-base arm (no free joint) is skipped. These tests are GL-free
(``mesh=False``, no render) so they run in CI without a display.
"""

import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.terrain import terrain_elevation

# A LOW floating base (base at z=0.4, foot sphere bottom at z=0.02) -- like a
# quadruped, its feet sit well below a raised heightfield peak at the flat
# spawn, so the burying is unambiguous. Two variants exercise the NAMED-free-
# joint (humanoid ``floating_base_joint``) and UNNAMED-``<freejoint>`` (mobile
# base, e.g. Go2) detection paths.
_NAMED_LOW_BASE = """
<mujoco model="seat_named">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.4">
      <freejoint name="floating_base_joint"/>
      <geom name="torso" type="box" size="0.1 0.05 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="leg" pos="0 0 -0.35">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="foot" type="sphere" size="0.03" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="knee_act" joint="knee"/></actuator>
</mujoco>
"""

_UNNAMED_LOW_BASE = _NAMED_LOW_BASE.replace('<freejoint name="floating_base_joint"/>', "<freejoint/>").replace(
    "seat_named", "seat_unnamed"
)

# A fixed-base arm (no free joint): the seat must skip it without error.
_FIXED_ARM = """
<mujoco model="seat_fixed_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link0" pos="0 0 0.0">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="j0" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="j0_act" joint="j0"/></actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _lowest_collision_geom_z(sim) -> float:
    """World z of the lowest collidable robot geom (skips the ground plane / hfield)."""
    model, data = sim._world._model, sim._world._data
    zs = []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        if model.geom_type[gid] in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        zs.append(float(data.geom_xpos[gid][2]))
    assert zs, "no collidable robot geoms found"
    return min(zs)


def _base_z(sim) -> float:
    model, data = sim._world._model, sim._world._data
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return float(data.qpos[int(model.jnt_qposadr[j]) + 2])
    raise AssertionError("no free joint")


@pytest.mark.parametrize("xml", [_NAMED_LOW_BASE, _UNNAMED_LOW_BASE], ids=["named_base", "unnamed_base"])
def test_floating_base_spawns_seated_on_terrain(xml):
    """A floating base spawns with its feet ON the terrain, not buried below it."""
    sim = Simulation(tool_name="seat_terrain_spawn", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("floater", urdf_path=_write(xml))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert ground > 0.1  # a genuinely raised plateau (pyramid peak at the centre)
        # The robot's flat-spawn feet (~0.02) would sit BELOW this plateau; the
        # seat must have raised the base so the lowest geom clears the surface.
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6
    finally:
        sim.cleanup()


def test_terrain_seat_survives_reset():
    """The seat is re-applied on reset() -- run_policy/eval_policy reset each episode."""
    sim = Simulation(tool_name="seat_terrain_reset", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6  # seated at spawn
        assert sim.reset()["status"] == "success"
        # After reset the base must still be seated (a one-shot add_robot fix
        # would snap back to the buried flat-ground height here).
        assert _lowest_collision_geom_z(sim) >= ground - 1e-6
    finally:
        sim.cleanup()


def test_seat_offset_scales_with_difficulty():
    """The base is raised by exactly the local terrain height (curriculum knob)."""
    for difficulty in (0.5, 1.0, 2.0):
        sim = Simulation(tool_name="seat_terrain_diff", mesh=False)
        sim.create_world(ground_plane=True, terrain="pyramid", difficulty=difficulty)
        sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
        try:
            ground = sim._ground_height_at(0.0, 0.0)
            assert ground == pytest.approx(terrain_elevation(difficulty), abs=1e-6)
            # base z == flat spawn (0.4) + the local terrain height.
            assert _base_z(sim) == pytest.approx(0.4 + ground, abs=1e-6)
        finally:
            sim.cleanup()


def test_flat_ground_spawn_is_unchanged():
    """No heightfield -> the seat is a no-op (byte-for-byte flat behaviour)."""
    sim = Simulation(tool_name="seat_flat_noop", mesh=False)
    sim.create_world(ground_plane=True)
    sim.add_robot("floater", urdf_path=_write(_NAMED_LOW_BASE))
    try:
        assert sim._ground_height_at(0.0, 0.0) == 0.0
        assert _base_z(sim) == pytest.approx(0.4, abs=1e-9)  # flat keyframe z, no offset
        sim.reset()
        assert _base_z(sim) == pytest.approx(0.4, abs=1e-9)
    finally:
        sim.cleanup()


def test_fixed_base_arm_on_terrain_is_skipped():
    """A fixed-base arm has no free joint -> the seat skips it without error."""
    sim = Simulation(tool_name="seat_fixed_arm", mesh=False)
    sim.create_world(ground_plane=True, terrain="rough", difficulty=1.0)
    result = sim.add_robot("arm", urdf_path=_write(_FIXED_ARM))
    try:
        assert result["status"] == "success"
        assert sim.reset()["status"] == "success"  # no crash on the free-joint-less arm
    finally:
        sim.cleanup()
