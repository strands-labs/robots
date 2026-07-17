"""Regression tests for mobile-base (unnamed-freejoint) observation state.

A mobile manipulator (e.g. LeKiwi) carries its floating base on an UNNAMED
``<freejoint/>`` that is not enumerated in ``robot.joint_names`` (those are the
actuated wheel/arm joints). The floating-base IMU-style signals
(``base_pos`` + ``base_quat`` + ``base_lin_vel`` + ``base_ang_vel``) must still
be surfaced from that base free joint, otherwise the mobile base is silently
observed as a fixed-base arm and a recorded dataset loses all base
position/orientation/velocity state.

The base free joint is recovered from the kinematic tree (walk up from an
actuated joint to the ancestor base body), so a sibling free-jointed task
object (a cube) is never mistaken for the base.
"""

import os
import tempfile

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Mobile base: an UNNAMED free joint on ``base_plate`` (identity orientation),
# an actuated hinge ``shoulder`` on a child arm body, and a SIBLING free-jointed
# ``task_cube`` at a distinct 90-deg-about-z orientation. Mirrors LeKiwi, whose
# base freejoint is unnamed and whose scene carries free-jointed task cubes.
MOBILE_BASE_XML = """
<mujoco model="test_mobile">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base_plate" pos="0 0 0.1" quat="1 0 0 0">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
    <body name="task_cube" pos="1 0 0.05" quat="0.707 0 0 0.707">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: a single hinge, NO free joint anywhere. Must never surface
# base state (guards against a false-positive base pick).
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
    s = Simulation(tool_name="test_mobile_base", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def test_unnamed_base_freejoint_surfaces_base_state(sim):
    """A mobile base whose floating base is an unnamed free joint (not in
    ``robot.joint_names``) still surfaces ``base_quat`` + ``base_ang_vel``, and
    the base is resolved from the kinematic tree - never the sibling free-jointed
    task cube."""
    sim.add_robot("mob", urdf_path=_write(MOBILE_BASE_XML))

    # The unnamed base freejoint is not a named joint.
    assert sim._world.robots["mob"].joint_names == ["shoulder"]

    obs = sim.get_observation(robot_name="mob", skip_images=True)

    assert "base_quat" in obs, "mobile base must surface base_quat"
    assert len(obs["base_quat"]) == 4
    assert all(isinstance(x, float) for x in obs["base_quat"])
    assert "base_ang_vel" in obs, "mobile base must surface base_ang_vel"
    assert len(obs["base_ang_vel"]) == 3
    # Full base kinematics: position (incl. height) + linear velocity too, not
    # just orientation + turn rate (a velocity-tracking / locomotion controller
    # needs base height and base linear velocity).
    assert "base_pos" in obs and len(obs["base_pos"]) == 3
    assert all(isinstance(x, float) for x in obs["base_pos"])
    assert "base_lin_vel" in obs and len(obs["base_lin_vel"]) == 3
    assert all(isinstance(x, float) for x in obs["base_lin_vel"])

    # base_quat is the base_plate (identity), NOT the task_cube (90 deg about z,
    # ~[0.707, 0, 0, 0.707]). If the sibling cube's free joint were picked, w
    # would be ~0.707 instead of ~1.0.
    assert obs["base_quat"][0] == pytest.approx(1.0, abs=1e-3)
    assert obs["base_quat"][3] == pytest.approx(0.0, abs=1e-3)

    # The actuated hinge still gets a scalar .vel; the free joint does not.
    assert "shoulder.vel" in obs and isinstance(obs["shoulder.vel"], float)


def test_base_quat_is_live_read_of_the_base_free_joint(sim):
    """``base_quat`` reflects the base free joint's CURRENT orientation (a live
    read), so a driven/rotated base is observed - it is not a static or
    wrong-joint value."""
    import mujoco

    sim.add_robot("mob", urdf_path=_write(MOBILE_BASE_XML))
    model, data = sim._world._model, sim._world._data

    # Locate the base_plate free joint and rotate it 90 deg about z.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mob/base_plate")
    jadr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jadr = int(model.jnt_qposadr[j])
    assert jadr is not None
    data.qpos[jadr + 3 : jadr + 7] = [0.70710678, 0.0, 0.0, 0.70710678]
    mujoco.mj_forward(model, data)

    obs = sim.get_observation(robot_name="mob", skip_images=True)
    assert obs["base_quat"][0] == pytest.approx(0.7071, abs=1e-3)
    assert obs["base_quat"][3] == pytest.approx(0.7071, abs=1e-3)


def test_fixed_base_arm_has_no_base_state(sim):
    """A fixed-base arm (no free joint) never surfaces base state - the mobile
    detection must not false-positive on a robot that has no floating base."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    obs = sim.get_observation(robot_name="arm", skip_images=True)
    assert "base_quat" not in obs
    assert "base_ang_vel" not in obs
    assert "base_pos" not in obs
    assert "base_lin_vel" not in obs
    assert "j0" in obs and "j0.vel" in obs


# Floating base whose free joint IS named and therefore enumerated in
# ``robot.joint_names`` (like a humanoid's ``floating_base_joint``). This is the
# case where get_robot_state previously misread the 6-DoF free joint as a scalar
# hinge - reporting the base x-coordinate as a joint "position".
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


def _set_free_joint(sim, robot, qpos7, qvel6):
    """Set a robot's (only) free joint pose+twist and forward the model."""
    import mujoco

    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(f"{robot}/") or name in ("floating_base_joint", f"{robot}/floating_base_joint"):
                jid = j
                break
    assert jid >= 0, "expected a free joint on the robot"
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 7] = qpos7
    data.qvel[vadr : vadr + 6] = qvel6
    mujoco.mj_forward(model, data)


def test_get_observation_surfaces_full_base_kinematics(sim):
    """get_observation surfaces the free joint's full 6-DoF pose + twist:
    base_pos (world x,y,z incl. HEIGHT), base_quat, base_lin_vel (m/s) and
    base_ang_vel (rad/s). A locomotion / velocity-tracking / mobile-manip
    controller needs base_pos (height) and base_lin_vel (the tracked quantity),
    not just orientation + turn rate."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    # Distinctive pose + twist so a wrong slice is unmistakable.
    _set_free_joint(sim, "humanoid", [0.3, 0.4, 0.9, 0.70710678, 0.0, 0.0, 0.70710678], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

    obs = sim.get_observation(robot_name="humanoid", skip_images=True)
    assert obs["base_pos"] == pytest.approx([0.3, 0.4, 0.9], abs=1e-3)
    assert obs["base_quat"] == pytest.approx([0.7071, 0.0, 0.0, 0.7071], abs=1e-3)
    assert obs["base_lin_vel"] == pytest.approx([1.1, 2.2, 3.3], abs=1e-3)
    assert obs["base_ang_vel"] == pytest.approx([4.4, 5.5, 6.6], abs=1e-3)
    # The 6-DoF free joint is NOT emitted as a degenerate scalar joint entry
    # (it would report base-x as a joint angle) - its full state is the
    # structured base_* keys above, matching get_robot_state.
    assert "floating_base_joint" not in obs
    assert "floating_base_joint.vel" not in obs


def test_get_robot_state_named_free_base_reports_base_pose_not_scalar(sim):
    """A floating base's NAMED free joint (``floating_base_joint``, enumerated in
    joint_names) must NOT be reported as a scalar joint. Its qpos is [xyz+quat],
    so reading qpos[jnt_qposadr] as a joint "position" reports the base
    x-coordinate and silently drops the orientation. get_robot_state must
    instead surface a structured ``base`` entry with the full 6-DoF pose+twist."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    assert "floating_base_joint" in sim._world.robots["humanoid"].joint_names

    # Distinctive base pose so a scalar misread (base-x as a joint angle) is
    # unmistakable: x=0.3, y=0.4, z=0.9, 90-deg-about-z quat, and a full twist.
    _set_free_joint(sim, "humanoid", [0.3, 0.4, 0.9, 0.70710678, 0.0, 0.0, 0.70710678], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

    result = sim.get_robot_state(robot_name="humanoid")
    assert result["status"] == "success"
    payload = result["content"][1]["json"]

    # The free joint is NOT a scalar joint entry (no misread base-x as "position").
    assert "floating_base_joint" not in payload["state"]
    # The scalar hinge is still reported as usual.
    assert "hip" in payload["state"] and isinstance(payload["state"]["hip"]["position"], float)

    # A structured base entry carries the full 6-DoF pose + twist.
    base = payload["base"]
    assert base["position"] == pytest.approx([0.3, 0.4, 0.9], abs=1e-3)
    assert base["quaternion"] == pytest.approx([0.7071, 0.0, 0.0, 0.7071], abs=1e-3)
    assert base["linear_velocity"] == pytest.approx([1.1, 2.2, 3.3], abs=1e-3)
    assert base["angular_velocity"] == pytest.approx([4.4, 5.5, 6.6], abs=1e-3)


def test_get_robot_state_unnamed_free_base_reports_base(sim):
    """A mobile base whose free joint is UNNAMED (not in joint_names, like
    LeKiwi) still surfaces a ``base`` entry - recovered from the kinematic tree,
    never the sibling free-jointed task cube."""
    sim.add_robot("mob", urdf_path=_write(MOBILE_BASE_XML))
    assert sim._world.robots["mob"].joint_names == ["shoulder"]

    payload = sim.get_robot_state(robot_name="mob")["content"][1]["json"]
    base = payload["base"]
    # base is the base_plate (identity quat), NOT the task_cube (~[0.707,0,0,0.707]).
    assert base["quaternion"][0] == pytest.approx(1.0, abs=1e-3)
    assert base["quaternion"][3] == pytest.approx(0.0, abs=1e-3)
    assert len(base["position"]) == 3 and len(base["linear_velocity"]) == 3


def test_get_robot_state_base_matches_get_observation(sim):
    """The base orientation/angular-velocity from get_robot_state agree with
    get_observation's base_quat / base_ang_vel for the same robot - the two
    state paths are consistent."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    _set_free_joint(sim, "humanoid", [0.1, 0.2, 0.7, 0.70710678, 0.0, 0.0, 0.70710678], [0.5, 0.0, 0.0, 0.0, 0.0, 1.5])

    base = sim.get_robot_state(robot_name="humanoid")["content"][1]["json"]["base"]
    obs = sim.get_observation(robot_name="humanoid", skip_images=True)
    assert base["position"] == obs["base_pos"]
    assert base["quaternion"] == obs["base_quat"]
    assert base["linear_velocity"] == obs["base_lin_vel"]
    assert base["angular_velocity"] == obs["base_ang_vel"]


def test_get_robot_state_fixed_base_arm_has_no_base_entry(sim):
    """A fixed-base arm (no free joint) reports scalar joints and NO ``base``
    entry - the floating-base detection must not false-positive."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    payload = sim.get_robot_state(robot_name="arm")["content"][1]["json"]
    assert "base" not in payload
    assert "j0" in payload["state"] and isinstance(payload["state"]["j0"]["position"], float)
