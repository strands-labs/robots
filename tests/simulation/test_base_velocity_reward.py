"""Regression tests for the ``base_velocity`` velocity-tracking reward term.

The predicate/reward DSL grew a ``base_velocity`` reward term: a floating base's
BODY-frame planar twist ``(vx, vy, wz)`` tracked against a commanded velocity,
the canonical dense locomotion reward. It consumes ``get_observation``'s
floating-base signals - ``base_lin_vel`` (world frame, rotated into the base
frame via ``base_quat``) and ``base_ang_vel`` (already body-frame) - so the
tracked velocity is heading-relative, matching the IsaacLab / legged_gym
locomotion-command convention.

These tests set a KNOWN base pose + twist directly on the sim (a rotated base
with a known world-frame linear velocity), then assert the term computes the
correct body-frame tracking error - and that the world->body rotation actually
fires (the same world velocity yields a different body-frame reward under a
different base orientation). They are GL-free (get_observation with
skip_images) so they run in CI without a display.
"""

import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import (
    _quat_rotate_inverse_wxyz,
    make_predicate,
)

# Floating base with a NAMED free joint (a humanoid's floating_base_joint) plus
# one actuated hinge. get_observation surfaces base_pos/base_quat/base_lin_vel/
# base_ang_vel for this robot.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
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

# Fixed-base arm: no free joint anywhere -> no base twist. base_velocity must
# degrade to 0.0 (and warn) rather than crash or invent a value.
FIXED_ARM_XML = """
<mujoco model="test_fixed">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_velocity", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_free_joint(sim, qpos7, qvel6):
    """Set the robot's (only) free joint pose + twist and forward the model."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 7] = qpos7
    data.qvel[vadr : vadr + 6] = qvel6
    mujoco.mj_forward(model, data)


# 90 deg about world +z. Body +x points along world +y.
_Q_YAW90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
_LIN_WORLD = [1.1, 2.2, 3.3]  # base linear velocity in the WORLD frame
_ANG = [4.4, 5.5, 6.6]  # base angular velocity (body frame); yaw rate = z = 6.6
# World->body rotation of _LIN_WORLD under _Q_YAW90 is [2.2, -1.1, 3.3], so the
# body-frame planar twist the reward tracks is (vx=2.2, vy=-1.1, wz=6.6).
_BODY_VX, _BODY_VY, _BODY_WZ = 2.2, -1.1, 6.6


def test_quat_rotate_inverse_matches_hand_computation():
    """The pure-Python world->body rotation is correct for a 90-deg-about-z base."""
    out = _quat_rotate_inverse_wxyz(_Q_YAW90, _LIN_WORLD)
    assert out == pytest.approx([2.2, -1.1, 3.3], abs=1e-6)
    # Identity quaternion is a no-op.
    assert _quat_rotate_inverse_wxyz([1.0, 0.0, 0.0, 0.0], _LIN_WORLD) == pytest.approx(_LIN_WORLD, abs=1e-9)


def test_base_velocity_is_zero_at_perfect_body_frame_tracking(sim):
    """Reward is 0 when the command equals the base's body-frame twist."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    term = make_predicate("base_velocity", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ)
    assert term(sim) == pytest.approx(0.0, abs=1e-6)


def test_base_velocity_is_negative_l2_error(sim):
    """Reward is -weight * L2(body-frame twist - command) when they differ."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    # Command (0,0,0): error is the full body-frame twist magnitude.
    expected = -((_BODY_VX**2 + _BODY_VY**2 + _BODY_WZ**2) ** 0.5)
    assert make_predicate("base_velocity")(sim) == pytest.approx(expected, abs=1e-5)
    # weight scales the error linearly.
    assert make_predicate("base_velocity", weight=2.0)(sim) == pytest.approx(2.0 * expected, abs=1e-5)


def test_base_velocity_tracks_the_body_frame_not_the_world_frame(sim):
    """The world->body rotation actually fires: the SAME world velocity yields a
    zero reward for a DIFFERENT command depending on the base orientation."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    # Identity orientation: body frame == world frame, so command == world lin.
    _set_free_joint(sim, [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], [*_LIN_WORLD, *_ANG])
    term_world = make_predicate("base_velocity", vx=1.1, vy=2.2, wz=6.6)
    assert term_world(sim) == pytest.approx(0.0, abs=1e-6)
    # Same world velocity, rotated base: the world-frame command is NO LONGER a
    # perfect match (proves the reward is heading-relative, not world-frame).
    _set_free_joint(sim, [0.0, 0.0, 0.8, *_Q_YAW90], [*_LIN_WORLD, *_ANG])
    assert term_world(sim) < -1.0
    # The body-frame command IS a perfect match under the rotated base.
    assert make_predicate("base_velocity", vx=_BODY_VX, vy=_BODY_VY, wz=_BODY_WZ)(sim) == pytest.approx(0.0, abs=1e-6)


def test_base_velocity_degrades_to_zero_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base twist: the term degrades to 0.0 and warns."""
    import logging

    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_velocity", vx=0.5)(sim)
    assert val == 0.0
    assert any("robot base" in r.message or "base" in r.message.lower() for r in caplog.records)
