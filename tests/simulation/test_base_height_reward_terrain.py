"""Regression tests: ``base_height`` measures clearance above the LOCAL ground.

``base_height`` (the legged_gym / IsaacLab base-height regularizer) is the
anti-crouch companion to ``base_velocity`` in a locomotion ``dense_reward``:
``-weight * (clearance - target) ** 2`` rewards a floating base for holding its
torso/pelvis near a target height and penalises crouching/diving so a policy
cannot cheat the forward-velocity reward by folding down.

It read an ABSOLUTE world z, which is correct only on a flat ground plane. Once
a locomotion task runs on a raised-terrain heightfield
(``create_world(terrain=...)``, the terrain curriculum) an absolute test is
wrong on TWO counts: a robot standing at its proper posture on a raised plateau
has an absolute base z above the target, so it is spuriously penalised; and the
absolute zero-reward pose on the plateau is a deep CROUCH (clearance =
target - terrain height), so the term actively REWARDS crouching on terrain -
inverting the very anti-crouch incentive it exists to provide.

The term now measures the base's height ABOVE THE LOCAL GROUND beneath it
(``(base_z - ground_z) - target``). On flat ground / a backend without a
heightfield the local ground height is ``0.0`` and behaviour is byte-for-byte
unchanged; on a heightfield it samples the terrain so proper posture on a
plateau scores 0 and a crouch is penalised. These tests are GL-free
(``get_observation`` with ``skip_images``) so they run in CI without a display.
This mirrors the ``base_below_z`` terrain fix (#1364) for the reward side.
"""

import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import make_predicate

# A floating base (NAMED free joint) + one actuated hinge; get_observation
# surfaces base_pos for it. No own ground plane: the strands world supplies the
# (flat or terrain) ground.
NAMED_BASE_XML = """
<mujoco model="test_named_base_height_terrain">
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

_TARGET = 0.74  # a G1-pelvis-like target clearance above the ground beneath it


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


def test_base_height_zero_at_proper_posture_on_raised_terrain():
    """Standing at the target clearance ABOVE a plateau scores ~0 (missed by absolute z)."""
    sim = Simulation(tool_name="test_bh_terrain", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        assert ground > 0.1  # a genuinely raised plateau
        term = make_predicate("base_height", target=_TARGET)
        # Proper posture: base sits target metres ABOVE the plateau. The absolute
        # base z (ground + target) is above target, so an absolute test scores
        # -(ground ** 2); the terrain-relative term scores 0.
        _set_base_xy_z(sim, 0.0, 0.0, ground + _TARGET)
        assert term(sim) == pytest.approx(0.0, abs=1e-6)
    finally:
        sim.cleanup()


def test_base_height_penalises_a_crouch_on_terrain_by_local_clearance():
    """A crouch on the plateau is penalised by its local clearance error, not absolute z."""
    sim = Simulation(tool_name="test_bh_crouch", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        term = make_predicate("base_height", target=_TARGET)
        # Crouched 0.1 m below the target CLEARANCE while on the plateau.
        _set_base_xy_z(sim, 0.0, 0.0, ground + _TARGET - 0.1)
        assert term(sim) == pytest.approx(-(0.1**2), abs=1e-6)
        # weight scales the penalty linearly.
        term5 = make_predicate("base_height", target=_TARGET, weight=5.0)
        assert term5(sim) == pytest.approx(-5.0 * (0.1**2), abs=1e-6)
    finally:
        sim.cleanup()


def test_base_height_rewards_proper_posture_over_a_crouch_on_terrain():
    """The anti-crouch incentive holds on terrain: standing tall scores higher than a crouch.

    This is the decisive semantic guard. Under the absolute-z bug the zero-reward
    pose on a plateau is a deep crouch (clearance = target - terrain height), so
    standing at proper posture scores WORSE than crouching - the incentive is
    inverted. The terrain-relative term restores the correct ordering.
    """
    sim = Simulation(tool_name="test_bh_order", mesh=False)
    sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        ground = sim._ground_height_at(0.0, 0.0)
        term = make_predicate("base_height", target=_TARGET)
        _set_base_xy_z(sim, 0.0, 0.0, ground + _TARGET)  # proper posture
        proper = term(sim)
        _set_base_xy_z(sim, 0.0, 0.0, ground + _TARGET - 0.15)  # crouched
        crouch = term(sim)
        assert proper > crouch  # proper posture is rewarded over a crouch on terrain
    finally:
        sim.cleanup()


def test_base_height_unchanged_on_flat_ground():
    """Flat-ground behaviour is byte-for-byte the absolute-z term (backward compat)."""
    sim = Simulation(tool_name="test_bh_flat", mesh=False)
    sim.create_world(ground_plane=True)
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    try:
        term = make_predicate("base_height", target=_TARGET)
        _set_base_xy_z(sim, 0.0, 0.0, _TARGET)  # at target on flat ground
        assert term(sim) == pytest.approx(0.0, abs=1e-6)
        _set_base_xy_z(sim, 0.0, 0.0, 0.50)  # 0.24 m below target on flat ground
        assert term(sim) == pytest.approx(-((0.50 - _TARGET) ** 2), abs=1e-6)
    finally:
        sim.cleanup()
