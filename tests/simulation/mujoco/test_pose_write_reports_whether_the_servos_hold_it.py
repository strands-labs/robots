"""A kinematic pose write reports whether the actuators will hold the pose.

``set_joint_positions`` writes ``qpos`` and runs forward kinematics. On a robot
whose joints are held by position servos that is only half of what "teleport /
set an initial pose" needs: the servos are still commanded to their previous
setpoint, so the first ``mj_step`` drives the pose straight back toward it::

    sim.set_joint_positions(pose)      # -> "Set 6/6 joint positions, FK updated"
    sim.step(150)                      # so101: 2.75 rad away from `pose`, every
                                       #        joint back near zero

The pose read back correct and the call reported success because nothing had
stepped yet. That is the same failure the rest of the backend already guards:
``actuate_robot`` seeds every actuator it adds from its joint's current position
"so the arm doesn't snap to zero on the next step", and ``remove_robot`` carries
``ctrl`` across an eject because dropped setpoints read as zero and "an arm
parked mid-air sags to the floor while ``remove_robot`` reported success". The
one surface whose whole job is writing a pose was the one that neither moved the
setpoints nor said it had not.

It is not a niche shape: 42 of the 62 loadable registry robots drive at least one
of their joints with a position servo (so101, panda, aloha, unitree_g1, spot,
ur5e, kinova_gen3 among them).

These tests pin both halves:

* the default write is still kinematics-only - unchanged, because rendering a
  pose or replaying a planned trajectory frame by frame is a real use - but its
  success text now names the joints whose servo holds a different setpoint and
  quotes the remedy;
* ``hold=True`` moves those setpoints with the pose, which is what makes
  "teleport and stay there" expressible at all: ``send_action`` writes the
  setpoints but never ``qpos``, and always advances at least one step.

Only position servos are moved. A motor takes a torque, so writing a joint angle
into its ``ctrl`` would command a torque numerically equal to an angle in
radians. 14 registry robots are motor-driven throughout (unitree_go2, unitree_h1,
jvrc, cassie) and ``openarm`` carries 2 servos beside 16 motors, so the split has
to be per actuator - which is what
:func:`~strands_robots.simulation.mujoco.scene_ops.joint_drive_map` owns.

A joint a *tendon* couples to one ``ctrl`` is the case that no gain inspection
can classify. Every stock tendon gripper is authored as a ``<position>`` actuator
on its tendon, so it clears all three gain terms a servo is tested by
(``panda/actuator8`` and ``robotiq_2f85/fingers_actuator`` compile to
``biasprm = [0, -100, 0]``): it really is a position servo, just on the tendon
rather than on any joint. Two facts disqualify its ``ctrl`` from carrying a joint
angle - the units are the tendon's (``[0, 255]`` for those grippers, ``[0, 0.52]``
metres for ``stretch3/arm``) and one ``ctrl`` drives 2 joints (4 on the
``stretch3`` telescoping arm), so there is no single angle to write. 7 registry
robots reach this: ``panda``, ``xarm7``, ``robotiq_2f85``, ``robotiq_2f85_v4``,
``shadow_hand``, ``stretch`` and ``stretch3``, 26 joints between them. Those
joints are reported on the same terms as any other non-position drive, and their
setpoints are left alone.
"""

import re

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Gravity is off so a settled servo lands exactly on its setpoint: the only force
# moving a joint here is an actuator, which is the quantity under test. Under
# gravity the same assertions would have to absorb the servo's steady-state droop.
DRIVE_MIX_XML = """
<mujoco model="drive_mix">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link1" pos="0 0 0.4">
      <joint name="served" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link1_geom" type="capsule" size="0.02 0.1" mass="0.4"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="motored" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom name="link2_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
      </body>
    </body>
    <body name="link3" pos="0.5 0 0.4">
      <joint name="undriven" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link3_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
    </body>
    <body name="link4" pos="1.0 0 0.4">
      <joint name="rated" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link4_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
    </body>
    <body name="link5" pos="1.5 0 0.4">
      <joint name="integrated" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link5_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="served_servo" joint="served" kp="60" kv="6"/>
    <motor name="motored_motor" joint="motored"/>
    <!-- Both of these carry mjBIAS_AFFINE, so a classifier reading the bias type
         alone reads them as position servos; neither takes a pose in ctrl. -->
    <velocity name="rated_drive" joint="rated" kv="5"/>
    <intvelocity name="integrated_drive" joint="integrated" kp="20" actrange="-2 2"/>
  </actuator>
</mujoco>
"""

POSE = {"served": 0.5, "motored": 0.6, "undriven": 0.7}

# The stock gripper idiom: one <position> actuator on a <fixed> tendon that wraps
# two joints. It clears every gain term a position servo is tested by, so only the
# transmission tells it apart - and its ctrlrange is in the tendon's units, not the
# joints'. Menagerie authors panda/actuator8, robotiq_2f85/fingers_actuator and
# shadow_hand/lh_A_FFJ0 exactly this way.
TENDON_GRIP_XML = """
<mujoco model="tendon_grip">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="palm" pos="0 0 0.4">
      <geom name="palm_geom" type="box" size="0.05 0.03 0.02" mass="0.5"/>
      <!-- z clears the palm: the capsule spans +/-0.03 about its own origin plus a
           0.008 radius, and the palm box top is at z=0.02, so an origin at 0.065
           leaves the finger out of contact through the whole 0..1.2 range. At the
           original 0.03 both fingers began 0.018 embedded in the palm, and the
           contact force that resolves that penetration - not the tendon servo -
           was what set where the pair came to rest. -->
      <body name="left_finger" pos="0.04 0 0.065">
        <joint name="left_driver" type="hinge" axis="0 1 0" range="0 1.2" damping="0.2" armature="0.005"/>
        <geom name="left_geom" type="capsule" size="0.008 0.03" mass="0.05"/>
      </body>
      <body name="right_finger" pos="-0.04 0 0.065">
        <joint name="right_driver" type="hinge" axis="0 1 0" range="0 1.2" damping="0.2" armature="0.005"/>
        <geom name="right_geom" type="capsule" size="0.008 0.03" mass="0.05"/>
      </body>
    </body>
    <body name="elbow_link" pos="0.4 0 0.4">
      <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="elbow_geom" type="capsule" size="0.02 0.08" mass="0.3"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip">
      <joint joint="left_driver" coef="0.5"/>
      <joint joint="right_driver" coef="0.5"/>
    </fixed>
  </tendon>
  <!-- One tendon over two joints constrains only their weighted sum, so their
       difference is a zero-stiffness mode. robotiq_2f85 pins it with exactly this
       equality (plus the armature above), and without it the pair is free to drift
       apart under an integrator change. With the mode locked and the fingers clear
       of the palm, the only thing left acting on the pair is the tendon servo, so
       where they settle is the setpoint rather than a solver artifact. -->
  <equality>
    <joint joint1="left_driver" joint2="right_driver"/>
  </equality>
  <actuator>
    <position name="grip_actuator" tendon="grip" kp="100" ctrlrange="0 255"/>
    <position name="elbow_servo" joint="elbow" kp="60" kv="6"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def grip_sim():
    s = Simulation(tool_name="test_pose_write_tendon_grip", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(TENDON_GRIP_XML)["status"] == "success"
    yield s
    s.cleanup()


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_pose_write_servo_hold", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(DRIVE_MIX_XML)["status"] == "success"
    yield s
    s.cleanup()


def _qpos(sim, joint: str) -> float:
    model, data = sim._world._model, sim._world._data
    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jnt_id >= 0, joint
    return float(data.qpos[model.jnt_qposadr[jnt_id]])


def _jnt(model, joint: str) -> int:
    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jnt_id >= 0, joint
    return int(jnt_id)


def _act(model, actuator: str) -> int:
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator)
    assert act_id >= 0, actuator
    return int(act_id)


def _ctrl(sim, actuator: str) -> float:
    model, data = sim._world._model, sim._world._data
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator)
    assert act_id >= 0, actuator
    return float(data.ctrl[act_id])


def _ten_length(sim, tendon: str) -> float:
    model, data = sim._world._model, sim._world._data
    ten_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_TENDON, tendon)
    assert ten_id >= 0, tendon
    return float(data.ten_length[ten_id])


def _ncon(sim) -> int:
    return int(sim._world._data.ncon)


def _text(result: dict) -> str:
    assert result["status"] == "success", result
    return str(result["content"][0]["text"])


class TestTheReportNamesAPoseTheNextStepUndoes:
    """The success text distinguishes a pose that survives from one that does not."""

    def test_a_servo_commanded_elsewhere_is_named_with_its_remedy(self, sim):
        """The joint whose servo pulls the pose back is named, and only that one.

        The measurement the report stands on: after the write the pose is exact,
        and stepping moves the served joint by nearly the whole request while the
        joints with no servo of their own stay put.
        """
        text = _text(sim.set_joint_positions(positions=POSE))
        assert _qpos(sim, "served") == pytest.approx(0.5)

        assert sim.step(n_steps=200)["status"] == "success"
        assert abs(_qpos(sim, "served") - 0.5) > 0.4, "premise: the servo pulls the pose back"
        assert _qpos(sim, "undriven") == pytest.approx(0.7, abs=1e-6)

        assert "served" in text, text
        assert "motored" not in text, "a motor's ctrl is not a setpoint the pose can be compared to"
        assert "undriven" not in text, "a joint with no actuator has no setpoint to disagree with"
        assert "hold=True" in text, text

    def test_the_remedy_the_report_quotes_makes_the_pose_survive_stepping(self, sim):
        """Parse the remedy out of the report, apply it, and step.

        Pinning the remedy rather than the wording: this fails both for a report
        that names no remedy and for one that names a remedy that does not work.
        """
        text = _text(sim.set_joint_positions(positions=POSE))
        offered = re.search(r"pass (\w+)=True", text)
        assert offered, f"the report offers no remedy: {text}"

        assert sim.reset()["status"] == "success"
        assert _text(sim.set_joint_positions(positions=POSE, **{offered.group(1): True}))
        assert sim.step(n_steps=200)["status"] == "success"
        assert _qpos(sim, "served") == pytest.approx(0.5, abs=1e-3)

    def test_a_servo_already_commanded_to_the_pose_is_not_reported(self, sim):
        """Writing the pose the servos already hold reports nothing extra.

        The report is about a disagreement, so it must not fire on a write that
        merely re-asserts where the servos are already pointed.
        """
        assert _text(sim.set_joint_positions(positions=POSE, hold=True))
        text = _text(sim.set_joint_positions(positions=POSE))
        assert "hold=True" not in text, text
        assert "served" not in text, text


class TestHoldMovesTheSetpointsWithThePose:
    """``hold=True`` writes the position-servo setpoints, and nothing else."""

    def test_the_servo_setpoint_becomes_the_pose_written(self, sim):
        text = _text(sim.set_joint_positions(positions=POSE, hold=True))
        assert _ctrl(sim, "served_servo") == pytest.approx(0.5)
        assert _qpos(sim, "served") == pytest.approx(0.5)
        assert "1 position-servo setpoint(s) moved" in text, text

    def test_a_motor_keeps_its_command_and_the_report_says_so(self, sim):
        """A joint angle must never land in a motor's ctrl.

        ``ctrl`` on a motor is a torque, so the pose value would be commanded as
        a torque numerically equal to an angle in radians. The report names the
        joints left alone so the caller is not told the whole pose is held.
        """
        text = _text(sim.set_joint_positions(positions=POSE, hold=True))
        assert _ctrl(sim, "motored_motor") == 0.0
        assert "motored" in text, text
        assert "torque" in text, text

    def test_a_velocity_drive_keeps_its_rate_command_and_the_pose_stands(self, sim):
        """A joint angle written into a velocity drive's ctrl is a rate command.

        This is the failure a bias-type-only classification produces on the write
        path: the report names the joint as servo-held, ``hold=True`` writes the
        angle into a rate, and the joint then moves away from the pose the call
        just reported success for (under gravity ``mj_step`` reports
        ``Nan, Inf or huge value in QACC``). Gravity is off in this scene, so a
        rate command of zero is the only thing holding the pose - which makes the
        drift the assertion below measures attributable to the ctrl write alone.
        """
        text = _text(sim.set_joint_positions(positions={"rated": 0.7}, hold=True))
        assert _ctrl(sim, "rated_drive") == 0.0, "a pose must never land in a rate command"
        assert "rated" in text, text
        assert "rate" in text, text

        assert sim.step(n_steps=200)["status"] == "success"
        assert _qpos(sim, "rated") == pytest.approx(0.7, abs=1e-3), "the written pose stands"
        assert np.isfinite(sim._world._data.qacc).all(), "the sim stays stable"

    def test_hold_must_be_a_boolean(self, sim):
        """A flag read by truthiness inverts for the spellings an opt-out uses."""
        result = sim.set_joint_positions(positions=POSE, hold="false")
        assert result["status"] == "error", result
        assert "hold must be a boolean" in result["content"][0]["text"]
        assert _qpos(sim, "served") == pytest.approx(0.0), "a refused flag writes nothing"


class TestTheDefaultWriteIsUnchanged:
    """The kinematic write itself is untouched, so every existing caller is too."""

    def test_no_setpoint_moves_without_hold(self, sim):
        before = sim._world._data.ctrl.copy()
        assert _text(sim.set_joint_positions(positions=POSE))
        assert np.array_equal(sim._world._data.ctrl, before)

    def test_the_whole_pose_still_lands_in_qpos(self, sim):
        assert _text(sim.set_joint_positions(positions=POSE))
        for joint, value in POSE.items():
            assert _qpos(sim, joint) == pytest.approx(value)


class TestTheDriveSplitIsPerActuator:
    """``joint_drive_map`` classifies each actuator, not each robot.

    Imported inside each test so the module still collects against a tree without
    the helper, and the rest of the file reports its own verdict there.
    """

    def test_only_the_actuator_whose_ctrl_is_a_pose_lands_in_servos(self, sim):
        """Exact equality, so any drive misread as a servo fails here.

        The scene carries one of each kind that matters: a position servo, a
        motor, a velocity drive, an integrated-velocity drive and an undriven
        joint.
        """
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = sim._world._model
        servos, other = joint_drive_map(model, mj)
        undriven = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "undriven")

        assert servos == {_jnt(model, "served"): _act(model, "served_servo")}
        assert other == {
            _jnt(model, "motored"): _act(model, "motored_motor"),
            _jnt(model, "rated"): _act(model, "rated_drive"),
            _jnt(model, "integrated"): _act(model, "integrated_drive"),
        }
        assert undriven not in servos and undriven not in other

    def test_an_affine_bias_alone_does_not_make_an_actuator_a_servo(self, sim):
        """The premise: the rejected drives do clear the bias-type term.

        Without this the test above would pass against a classifier that happened
        to exclude them for the wrong reason, and the regression it guards
        (a rate command receiving a joint angle) would be invisible again.
        """
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = sim._world._model
        affine = int(mj.mjtBias.mjBIAS_AFFINE)
        for actuator in ("rated_drive", "integrated_drive"):
            act_id = _act(model, actuator)
            assert int(model.actuator_biastype[act_id]) == affine, actuator

        servos, other = joint_drive_map(model, mj)
        assert _jnt(model, "rated") in other, "a velocity drive commands a rate, not a pose"
        assert _jnt(model, "integrated") in other, "an integrated-velocity drive integrates a rate"

    def test_the_two_maps_never_share_a_joint(self, sim):
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        servos, other = joint_drive_map(sim._world._model, mj)
        assert not set(servos) & set(other)


class TestAJointATendonDrivesIsReportedNotSilent:
    """A tendon-coupled joint is a joint whose written pose will not hold."""

    def test_hold_names_the_tendon_coupled_joints_it_left_alone(self, grip_sim):
        """The whole point of ``hold=True`` is that the pose survives stepping.

        A tendon-coupled joint cannot be held that way, and reporting nothing
        reads exactly like a joint no actuator drives at all - so the caller
        gets a success text that implies the request was honoured.
        """
        text = _text(grip_sim.set_joint_positions({"left_driver": 0.6, "right_driver": 0.6}, hold=True))
        assert "left_driver" in text and "right_driver" in text, text
        assert "left alone" in text, text

    def test_a_tendon_coupled_joint_lands_in_other_drives(self, grip_sim):
        """It is driven, so it belongs in a bucket - and not in *servos*."""
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = grip_sim._world._model
        servos, other = joint_drive_map(model, mj)
        for joint in ("left_driver", "right_driver"):
            assert _jnt(model, joint) in other, joint
            assert _jnt(model, joint) not in servos, joint

    def test_both_coupled_joints_report_the_one_actuator_driving_them(self, grip_sim):
        """One ``ctrl`` for two joints is itself why it cannot carry a pose."""
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = grip_sim._world._model
        _, other = joint_drive_map(model, mj)
        grip = _act(model, "grip_actuator")
        assert other[_jnt(model, "left_driver")] == grip
        assert other[_jnt(model, "right_driver")] == grip

    def test_the_tendon_ctrl_never_receives_a_joint_angle(self, grip_sim):
        """The safety half, and the term a permissive fix would break.

        ``grip_actuator`` clears every gain term a position servo is tested by,
        so resolving the tendon in the *servo* test rather than only in "is this
        joint driven" would write ``0.6`` rad into a ``[0, 255]`` slot. Passes
        both before and after this change: it fails only if that is done.
        """
        before = _ctrl(grip_sim, "grip_actuator")
        assert _text(grip_sim.set_joint_positions({"left_driver": 0.6, "right_driver": 0.6}, hold=True))
        assert _ctrl(grip_sim, "grip_actuator") == before

    def test_the_gain_terms_alone_would_call_it_a_servo(self, grip_sim):
        """Why the transmission has to be a term of its own.

        This asserts the premise the classification rests on rather than the
        classification: the actuator's gains are a position servo's gains, so
        nothing about them separates it from ``elbow_servo``.
        """
        model = grip_sim._world._model
        grip, elbow = _act(model, "grip_actuator"), _act(model, "elbow_servo")
        for act_id in (grip, elbow):
            assert int(model.actuator_biastype[act_id]) == int(mj.mjtBias.mjBIAS_AFFINE)
            assert float(model.actuator_biasprm[act_id, 1]) < 0.0
            assert int(model.actuator_dyntype[act_id]) == int(mj.mjtDyn.mjDYN_NONE)
        assert int(model.actuator_trntype[grip]) == int(mj.mjtTrn.mjTRN_TENDON)
        assert int(model.actuator_trntype[elbow]) == int(mj.mjtTrn.mjTRN_JOINT)

    def test_a_directly_served_joint_in_the_same_scene_is_still_held(self, grip_sim):
        """No overreach: the tendon does not make its neighbours unholdable."""
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = grip_sim._world._model
        assert _text(grip_sim.set_joint_positions({"elbow": 0.5}, hold=True))
        assert _jnt(model, "elbow") in joint_drive_map(model, mj)[0]
        assert _ctrl(grip_sim, "elbow_servo") == pytest.approx(0.5)
        grip_sim.step(300)
        assert _qpos(grip_sim, "elbow") == pytest.approx(0.5, abs=0.02)

    def test_the_default_write_text_is_unchanged_for_a_tendon_joint(self, grip_sim):
        """Without ``hold`` the call is kinematics-only and says so, as before."""
        text = _text(grip_sim.set_joint_positions({"left_driver": 0.6}))
        assert text == "Set 1/1 joint positions, FK updated"
        assert _ctrl(grip_sim, "grip_actuator") == 0.0

    def test_the_fixture_settles_by_the_tendon_and_not_by_contact(self, grip_sim):
        """Premise: nothing but the tendon servo acts on the coupled pair.

        An earlier version of this scene started both fingers 0.018 embedded in
        the palm, so the contact force resolving that penetration - not the
        servo - set where the pair came to rest: about 1.44 rad, outside their
        own ``range="0 1.2"``, at a point that moved with the mujoco build. Any
        settling assertion on such a scene pins a solver artifact rather than the
        behaviour under test, so the premise is asserted directly here and the
        collapse is measured in the test below.
        """
        assert _ncon(grip_sim) == 0, "the fixture starts in contact"
        assert _text(grip_sim.set_joint_positions({"left_driver": 0.6, "right_driver": 0.6}, hold=True))
        for _ in range(8):
            grip_sim.step(50)
            assert _ncon(grip_sim) == 0, "a contact appeared while the pose collapsed"

    def test_the_written_pose_collapses_to_what_the_tendon_servo_commands(self, grip_sim):
        """The report's claim, on the quantity the servo actually controls.

        Asserted against the tendon length returning to its stale setpoint rather
        than against a per-joint landing point: the servo drives the length
        ``0.5*q_left + 0.5*q_right`` toward ``ctrl``, so that is the term whose
        target is build-independent. Each joint separately is pinned only by the
        equality locking their difference, which is why the per-joint half is
        asserted as "far from what was asked for" and not as a landing point.
        """
        assert _text(grip_sim.set_joint_positions({"left_driver": 0.6, "right_driver": 0.6}, hold=True))
        assert _ten_length(grip_sim, "grip") == pytest.approx(0.6, abs=1e-9)
        assert _ctrl(grip_sim, "grip_actuator") == 0.0, "hold must not write the tendon ctrl"
        grip_sim.step(400)
        assert _ten_length(grip_sim, "grip") == pytest.approx(0.0, abs=0.01)
        for joint in ("left_driver", "right_driver"):
            assert abs(_qpos(grip_sim, joint) - 0.6) > 0.3, joint


class TestAStockGripperReportsItsFingerJoints:
    """The shape reaches built-in robots, not just a hand-authored scene."""

    @pytest.mark.parametrize(
        ("robot", "joints"),
        [
            ("panda", ("panda/finger_joint1", "panda/finger_joint2")),
            ("robotiq_2f85", ("robotiq_2f85/left_driver_joint", "robotiq_2f85/right_driver_joint")),
        ],
    )
    def test_hold_names_the_gripper_joints_it_cannot_move(self, robot, joints):
        sim = Simulation(tool_name=f"test_pose_write_{robot}", mesh=False)
        try:
            sim.create_world()
            if sim.add_robot(robot)["status"] != "success":
                pytest.skip(f"{robot} assets unavailable")
            text = _text(sim.set_joint_positions(dict.fromkeys(joints, 0.03), hold=True))
            for joint in joints:
                assert joint in text, text
            assert "left alone" in text, text
        finally:
            sim.cleanup()

    @pytest.mark.parametrize(
        ("robot", "joints", "target"),
        [
            ("panda", ("panda/finger_joint1", "panda/finger_joint2"), 0.04),
            ("robotiq_2f85", ("robotiq_2f85/left_driver_joint", "robotiq_2f85/right_driver_joint"), 0.6),
        ],
    )
    def test_the_pose_it_reports_on_really_does_not_survive_stepping(self, robot, joints, target):
        """The report's claim, measured on the asset it is reporting about.

        Asserted as "far from what was asked for" rather than against a settling
        point: where the tendon takes the joint is the model's business, and only
        that the written pose is not what the next step keeps is this call's.
        """
        sim = Simulation(tool_name=f"test_pose_drift_{robot}", mesh=False)
        try:
            sim.create_world()
            if sim.add_robot(robot)["status"] != "success":
                pytest.skip(f"{robot} assets unavailable")
            assert _text(sim.set_joint_positions(dict.fromkeys(joints, target), hold=True))
            for joint in joints:
                assert _qpos(sim, joint) == pytest.approx(target, abs=1e-9)
            sim.step(400)
            for joint in joints:
                assert abs(_qpos(sim, joint) - target) > target / 2
        finally:
            sim.cleanup()
