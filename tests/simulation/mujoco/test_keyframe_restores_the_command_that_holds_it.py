"""A ``<keyframe>`` spawn applies the actuator command that HOLDS the keyed pose.

A MuJoCo ``<key>`` pairs a pose with the actuator command holding it, and
``mj_resetDataKeyframe`` -- MuJoCo's own definition of what a keyframe restores
-- writes ``qpos``, ``qvel``, ``act``, ``ctrl``, the mocap pools and the clock.
``add_robot(keyframe=...)`` read ``key_qpos`` alone, so a gravity-loaded arm
stood at its home configuration with every actuator commanded to the ZERO
configuration: the pose was not self-holding and the first steps drove the robot
off home, and ``reset()`` reproduced exactly the same state at the top of every
episode -- which is what a policy's first inference of each episode then sees.

Measured on the built-in ``panda`` ``home`` keyframe (4 of its 8 keyed ``ctrl``
entries non-zero): ``add_robot(keyframe="home")`` then ``step(400)`` moved joint4
1.5008 rad off home, and ``reset()`` + ``step(400)`` repeated it identically. 28
of the 31 built-in registry robots that ship a ``<keyframe>`` declare a non-zero
``ctrl`` in it.

The keyed command is applied VERBATIM rather than classified by actuator type:
the MJCF author chose those numbers for those actuators, so a servo setpoint, a
motor torque and a stateful actuator's activation each carry across as whatever
quantity their own actuator reads. ``qvel`` is deliberately not applied -- the
robot-scoped reset that runs immediately before the apply zeroes it so that a
freshly added robot starts at rest.

Hermetic: inline MJCFs, gravity on, no mesh download and no render.
"""

from __future__ import annotations

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A gravity-loaded two-hinge arm reaching sideways, so an actuator commanded to
# zero visibly drags it down. Damped servos with enough gain to settle ON the
# setpoint within the step budget, so "holds its pose" is a tight bound rather
# than a race against an oscillation. ``wrist`` is torque-driven and ``clamp``
# carries an activation state, so the keyframe below commands three different
# physical quantities through one ``ctrl`` vector.
_HELD_ARM = """
<mujoco model="held_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.6">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
      <body name="l2" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="1.0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
      </body>
    </body>
    <body name="w" pos="0 0.6 0.6">
      <joint name="wrist" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1"/>
    </body>
    <body name="c" pos="0 -0.6 0.6">
      <joint name="clamp" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="sh_act" joint="shoulder" kp="200"/>
    <position name="el_act" joint="elbow" kp="200"/>
    <motor name="wr_act" joint="wrist"/>
    <general name="cl_act" joint="clamp" dyntype="filter" dynprm="0.4"
             gaintype="fixed" gainprm="8" biastype="none"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="0.7 -1.1 0.55 0.9" act="0.42"/>
  </keyframe>
</mujoco>
"""

# The same arm whose keyframe declares only a pose. MuJoCo fills the omitted
# ``ctrl`` with zeros, which is exactly what a reset writes, so applying it is a
# no-op: these robots must behave byte-for-byte as before.
_POSE_ONLY_ARM = _HELD_ARM.replace(
    '<key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="0.7 -1.1 0.55 0.9" act="0.42"/>',
    '<key name="home" qpos="0.7 -1.1 0.0 0.0"/>',
).replace('model="held_arm"', 'model="pose_only_arm"')

# A keyframe carrying a non-zero ``qvel``, to pin that a robot is added at rest.
_MOVING_ARM = _HELD_ARM.replace(
    '<key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="0.7 -1.1 0.55 0.9" act="0.42"/>',
    '<key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="0.7 -1.1 0.0 0.0" qvel="1.5 0.0 0.0 0.0"/>',
).replace('model="held_arm"', 'model="moving_arm"')

# An unnamed actuator cannot be matched inside the merged model, so its keyed
# command is skipped -- without costing its NAMED neighbours theirs.
_UNNAMED_ACT_ARM = _HELD_ARM.replace(
    '<position name="sh_act" joint="shoulder" kp="200"/>',
    '<position joint="shoulder" kp="200"/>',
).replace('model="held_arm"', 'model="unnamed_act_arm"')

# A keyframe whose servo setpoint deliberately names a DIFFERENT configuration
# than its own pose: the built-in ``stretch3`` ``home`` does exactly this (lift
# ctrl 0.6 against a keyed lift qpos of 0.0). Reproduced, not second-guessed.
_DISAGREEING_ARM = _HELD_ARM.replace(
    '<key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="0.7 -1.1 0.55 0.9" act="0.42"/>',
    '<key name="home" qpos="0.7 -1.1 0.0 0.0" ctrl="-0.4 -1.1 0.0 0.0"/>',
).replace('model="held_arm"', 'model="disagreeing_arm"')

_HOME_POSE = [0.7, -1.1, 0.0, 0.0]
# ``ctrl`` in actuator order: shoulder setpoint, elbow setpoint, wrist torque,
# clamp gain input. Three different quantities through one vector.
_HOME_CTRL = [0.7, -1.1, 0.55, 0.9]
_HOME_ACT = [0.42]
_SETTLE_STEPS = 400


def _xml(tmp_path, name, body):
    p = tmp_path / f"{name}.xml"
    p.write_text(body)
    return str(p)


@pytest.fixture
def held_arm_xml(tmp_path):
    return _xml(tmp_path, "held_arm", _HELD_ARM)


@pytest.fixture
def pose_only_arm_xml(tmp_path):
    return _xml(tmp_path, "pose_only_arm", _POSE_ONLY_ARM)


@pytest.fixture
def moving_arm_xml(tmp_path):
    return _xml(tmp_path, "moving_arm", _MOVING_ARM)


@pytest.fixture
def unnamed_act_arm_xml(tmp_path):
    return _xml(tmp_path, "unnamed_act_arm", _UNNAMED_ACT_ARM)


@pytest.fixture
def disagreeing_arm_xml(tmp_path):
    return _xml(tmp_path, "disagreeing_arm", _DISAGREEING_ARM)


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_keyframe_hold", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _pose(sim, robot="a"):
    """The robot's hinge angles, in the order its actuators drive them."""
    model, data = sim._world._model, sim._world._data
    out = []
    for jn in ("shoulder", "elbow", "wrist", "clamp"):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{robot}/{jn}")
        assert jid >= 0, f"joint {robot}/{jn} missing from the merged model"
        out.append(float(data.qpos[int(model.jnt_qposadr[jid])]))
    return np.array(out)


def _ctrl(sim, robot="a"):
    model, data = sim._world._model, sim._world._data
    out = []
    for an in ("sh_act", "el_act", "wr_act", "cl_act"):
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{robot}/{an}")
        out.append(None if aid < 0 else float(data.ctrl[aid]))
    return out


def _spawn(sim, xml, name="a", **kw):
    res = sim.add_robot(name=name, urdf_path=xml, keyframe="home", **kw)
    assert res["status"] == "success", res["content"][0]["text"]
    return res


class TestASpawnedKeyframePoseIsSelfHolding:
    """The headline: the keyed pose survives stepping, because the keyed command
    that holds it is applied with it."""

    def test_the_keyed_pose_survives_stepping(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        assert _pose(sim) == pytest.approx(_HOME_POSE, abs=1e-6)
        sim.step(_SETTLE_STEPS)
        drift = float(np.max(np.abs(_pose(sim)[:2] - np.array(_HOME_POSE[:2]))))
        assert drift < 0.05, (
            f"the arm left its keyframe home pose by {drift:.4f} rad; the keyed setpoints holding "
            f"it were not applied (ctrl={_ctrl(sim)})"
        )

    def test_the_keyed_command_is_the_live_command(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        assert _ctrl(sim) == pytest.approx(_HOME_CTRL, abs=1e-6)

    def test_a_keyed_activation_is_applied(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        model, data = sim._world._model, sim._world._data
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "a/cl_act")
        adr = int(model.actuator_actadr[aid])
        assert adr >= 0, "premise: cl_act must carry an activation state"
        assert float(data.act[adr]) == pytest.approx(_HOME_ACT[0], abs=1e-6)

    def test_the_command_is_applied_verbatim_for_every_actuator_kind(self, sim, held_arm_xml):
        """A servo setpoint, a motor torque and a stateful actuator's gain input
        all reach their own actuator: the keyframe's author owns the units, so
        nothing here classifies or filters by actuator type."""
        _spawn(sim, held_arm_xml)
        model = sim._world._model
        kinds = {}
        for an in ("sh_act", "el_act", "wr_act", "cl_act"):
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"a/{an}")
            kinds[an] = (
                int(model.actuator_biastype[aid]),
                int(model.actuator_dyntype[aid]),
            )
        # premise: the four actuators really are of different kinds, so a
        # servo-only implementation could not satisfy the assertion below.
        assert len(set(kinds.values())) == 3, f"premise: expected three actuator kinds, got {kinds}"
        assert _ctrl(sim) == pytest.approx(_HOME_CTRL, abs=1e-6)


class TestResetReturnsToAHoldableHome:
    """``reset()`` begins a new rollout, so it must restore the state a rollout
    starts from -- including the command that holds the pose."""

    def test_reset_restores_the_keyed_command(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        sim.send_action({"shoulder": -0.3, "elbow": 0.4}, robot_name="a")
        sim.step(50)
        assert sim.reset()["status"] == "success"
        assert _pose(sim) == pytest.approx(_HOME_POSE, abs=1e-6)
        assert _ctrl(sim) == pytest.approx(_HOME_CTRL, abs=1e-6)

    def test_the_pose_restored_by_reset_survives_stepping(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        sim.step(_SETTLE_STEPS)
        assert sim.reset()["status"] == "success"
        sim.step(_SETTLE_STEPS)
        drift = float(np.max(np.abs(_pose(sim)[:2] - np.array(_HOME_POSE[:2]))))
        assert drift < 0.05, (
            f"the second rollout began sagging off home by {drift:.4f} rad; reset() dropped the "
            f"keyed setpoints (ctrl={_ctrl(sim)})"
        )

    def test_reset_restores_a_keyed_activation(self, sim, held_arm_xml):
        _spawn(sim, held_arm_xml)
        sim.step(50)
        assert sim.reset()["status"] == "success"
        model, data = sim._world._model, sim._world._data
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "a/cl_act")
        adr = int(model.actuator_actadr[aid])
        assert float(data.act[adr]) == pytest.approx(_HOME_ACT[0], abs=1e-6)


class TestNothingElseChanges:
    """Controls: the boundaries this change must not cross."""

    def test_a_keyframe_declaring_no_command_leaves_every_setpoint_at_zero(self, sim, pose_only_arm_xml):
        """MuJoCo fills an omitted ``ctrl`` with zeros, which is what a reset
        writes anyway, so these robots are unaffected in both directions."""
        _spawn(sim, pose_only_arm_xml)
        assert _pose(sim) == pytest.approx(_HOME_POSE, abs=1e-6)
        assert _ctrl(sim) == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)
        assert sim.reset()["status"] == "success"
        assert _ctrl(sim) == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)

    def test_a_robot_spawned_without_a_keyframe_records_nothing_to_restore(self, sim, held_arm_xml):
        res = sim.add_robot(name="a", urdf_path=held_arm_xml)
        assert res["status"] == "success"
        robot = sim._world.robots["a"]
        assert robot.home_qpos == {}
        assert robot.home_actuators == {}
        assert _ctrl(sim) == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)

    def test_a_keyframe_spawn_leaves_another_robots_command_alone(self, sim, held_arm_xml):
        """``add_robot`` promises the scene it joins keeps the pose it is in and
        the setpoints holding it there."""
        _spawn(sim, held_arm_xml, name="first")
        sim.send_action({"shoulder": 0.25, "elbow": -0.35}, robot_name="first")
        before = _ctrl(sim, "first")
        _spawn(sim, held_arm_xml, name="second")
        assert _ctrl(sim, "first") == pytest.approx(before, abs=1e-12)
        assert _ctrl(sim, "second") == pytest.approx(_HOME_CTRL, abs=1e-6)

    def test_a_keyed_velocity_is_not_applied(self, sim, moving_arm_xml):
        """A freshly added robot starts at rest: the robot-scoped reset ahead of
        the keyframe apply zeroes ``qvel`` deliberately."""
        _spawn(sim, moving_arm_xml)
        model, data = sim._world._model, sim._world._data
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "a/shoulder")
        assert float(data.qvel[int(model.jnt_dofadr[jid])]) == pytest.approx(0.0, abs=1e-12)

    def test_an_unnamed_actuator_costs_only_its_own_command(self, sim, unnamed_act_arm_xml):
        """Nothing names it, so it cannot be matched in the merged model. Its
        joint still gets the keyed pose, and its named neighbours still get
        their commands."""
        _spawn(sim, unnamed_act_arm_xml)
        assert _pose(sim) == pytest.approx(_HOME_POSE, abs=1e-6)
        assert _ctrl(sim, "a")[0] is None, "premise: the shoulder actuator must be unnamed"
        assert _ctrl(sim)[1:] == pytest.approx(_HOME_CTRL[1:], abs=1e-6)
        assert "a/sh_act" not in sim._world.robots["a"].home_actuators

    def test_a_command_that_disagrees_with_its_own_pose_is_still_applied(self, sim, disagreeing_arm_xml):
        """The built-in ``stretch3`` ``home`` keyframe commands its lift to
        0.6 m while keying that joint's pose at 0.0. That is the asset author's
        choice and ``mj_resetDataKeyframe`` reproduces it, so this does too
        rather than second-guessing which half of a keyframe to believe."""
        _spawn(sim, disagreeing_arm_xml)
        assert _pose(sim) == pytest.approx(_HOME_POSE, abs=1e-6)
        assert _ctrl(sim)[0] == pytest.approx(-0.4, abs=1e-6)
        sim.step(_SETTLE_STEPS)
        # The shoulder tracks the command it was given, so it ends far nearer the
        # commanded -0.4 than the keyed 0.7 (it settles short of -0.4 by the
        # servo's own droop under the loaded elbow, which is not the point here).
        settled = float(_pose(sim)[0])
        assert abs(settled - (-0.4)) < abs(settled - _HOME_POSE[0]), (
            f"shoulder settled at {settled:.4f}, nearer its keyed pose {_HOME_POSE[0]} than the "
            f"-0.4 its own keyframe commanded"
        )
