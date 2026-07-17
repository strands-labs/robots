"""Regression tests: ``base_below_z`` measures clearance above the LOCAL ground.

``base_below_z`` (#1233) is the height-collapse half of a floating-base fall
termination -- "the torso dropped to the floor, end the episode". It read an
ABSOLUTE world z, which is correct only on a flat ground plane. Once a
locomotion task runs on a raised-terrain heightfield
(``create_world(terrain=...)``, the terrain curriculum) an absolute test
silently MISSES a collapse: a robot that has fallen onto a 0.16 m pyramid
plateau still has an absolute base z (~0.26 m) above the flat-ground collapse
threshold (0.18 m for the Go2), so the episode never terminates.

The predicate now measures the base's height ABOVE THE LOCAL GROUND beneath it
(``base z - _ground_height_at(x, y)``). On flat ground / a backend without a
heightfield the local ground height is ``0.0`` and the behaviour is byte-for-byte
unchanged; on a heightfield it samples the terrain so a collapse on a plateau is
detected. These tests pin both the ``_ground_height_at`` heightfield sampling
(ground truth against a known pyramid) and the terrain-relative predicate, and
guard the flat-ground backward-compatibility. They are GL-free
(``get_observation`` with ``skip_images``) so they run in CI without a display.
"""

import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import make_predicate
from strands_robots.simulation.terrain import terrain_elevation

# A floating base (NAMED free joint) + one actuated hinge; get_observation
# surfaces base_pos for it. No own ground plane: the strands world supplies the
# (flat or terrain) ground.
NAMED_BASE_XML = """
<mujoco model="test_named_base_terrain">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_xy_z(sim, x: float, y: float, z: float) -> None:
    """Place the (only) free joint at world ``(x, y, z)``, upright."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def test_ground_height_at_is_zero_on_flat_ground():
    """A flat ground plane (no heightfield) reports local ground height 0.0."""
    sim = Simulation(tool_name="test_ground_flat", mesh=False)
    sim.create_world(ground_plane=True)
    try:
        assert sim._ground_height_at(0.0, 0.0) == 0.0
        assert sim._ground_height_at(3.0, -2.0) == 0.0
    finally:
        sim.cleanup()


def test_ground_height_at_samples_the_terrain_heightfield():
    """On a pyramid the centre sits at the peak elevation, the edge / off-field at 0."""
    sim = Simulation(tool_name="test_ground_terrain", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    try:
        peak = terrain_elevation(2.0)  # pyramid centre is the highest plateau (1.0)
        assert sim._ground_height_at(0.0, 0.0) == pytest.approx(peak, abs=1e-6)
        # The outer ring / off-field is flush with z=0 (the base slab top).
        assert sim._ground_height_at(4.9, 4.9) == pytest.approx(0.0, abs=1e-6)
        assert sim._ground_height_at(100.0, 0.0) == pytest.approx(0.0, abs=1e-6)
        # Peak scales with the difficulty curriculum knob.
        assert peak == pytest.approx(2.0 * terrain_elevation(1.0), abs=1e-9)
    finally:
        sim.cleanup()


def test_base_below_z_detects_a_collapse_on_raised_terrain():
    """A base collapsed onto a raised plateau trips base_below_z (missed by an absolute z)."""
    sim = Simulation(tool_name="test_collapse_terrain", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert ground > 0.1  # a genuinely raised plateau
        pred = make_predicate("base_below_z", z=0.18)
        # Collapsed onto the plateau: absolute z (ground + 0.1) is ABOVE 0.18 --
        # an absolute test would miss the fall -- but the clearance (0.1) is below.
        _set_base_xy_z(sim, 0.0, 0.0, ground + 0.1)
        assert pred(sim) is True
        # Standing tall on the same plateau is not a collapse.
        _set_base_xy_z(sim, 0.0, 0.0, ground + 0.8)
        assert pred(sim) is False
    finally:
        sim.cleanup()


def test_base_below_z_unchanged_on_flat_ground():
    """Flat-ground behaviour is byte-for-byte the absolute-z test (backward compat)."""
    sim = Simulation(tool_name="test_flat_compat", mesh=False)
    sim.create_world(ground_plane=True)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        pred = make_predicate("base_below_z", z=0.18)
        _set_base_xy_z(sim, 0.0, 0.0, 0.26)  # above threshold on flat ground
        assert pred(sim) is False
        _set_base_xy_z(sim, 0.0, 0.0, 0.10)  # below threshold on flat ground
        assert pred(sim) is True
    finally:
        sim.cleanup()
