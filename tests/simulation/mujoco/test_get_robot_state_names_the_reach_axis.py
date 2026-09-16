"""``get_robot_state`` says where the end effector is FROM THE BASE, and which way the arm extends.

Measured with an agent on a fresh so101 (v0.5.2 devx replay, scenario s06):
the state read ``end_effector … pos=[0.02, -0.38, 0.26]`` and nothing else
said which axis is the robot's front, so the model read "in front of the
base" as +X and placed the cube at ``[0.2, 0, 0.02]`` - beside the arm, whose
whole reach lies along -Y. Only the arm's spare reach saved the move. The
end-effector line now carries the base position, the EE offset from it and
the horizontal axis that offset lies along, and the ``json`` payload the same
three facts, so "in front of" means the same thing to the agent and the person.
"""

from __future__ import annotations

import ast
import importlib.util
import re

import pytest

from strands_robots.simulation.ik import REACH_AXIS_MIN_M, reach_axis, reach_axis_label

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


class TestReachAxis:
    @pytest.mark.parametrize(
        ("offset", "axis"),
        [
            ((0.02, -0.38, 0.26), "-Y"),
            ((0.0, 0.3, 0.1), "+Y"),
            ((0.4, 0.1, 0.0), "+X"),
            ((-0.25, -0.2, 0.5), "-X"),
            ((0.3, 0.3, 0.0), "+X"),  # a tie goes to X, deterministically
        ],
    )
    def test_the_dominant_horizontal_component_names_the_axis(self, offset, axis) -> None:
        assert reach_axis(offset) == axis
        assert reach_axis_label(axis) == f"the arm currently extends along {axis}"

    @pytest.mark.parametrize("offset", [(0.0, 0.0, 0.4), (0.01, -0.01, 0.2), (REACH_AXIS_MIN_M * 0.7, 0.0, 0.0)])
    def test_a_pose_over_the_base_is_not_given_an_axis(self, offset) -> None:
        assert reach_axis(offset) is None
        assert reach_axis_label(None) == "the arm is currently over its base"

    def test_the_threshold_is_the_horizontal_norm(self) -> None:
        d = REACH_AXIS_MIN_M / 2**0.5
        assert reach_axis((d * 1.01, d * 1.01, 0.0)) == "+X"
        assert reach_axis((d * 0.99, d * 0.99, 5.0)) is None


_ARM_XML = """
<mujoco model="reach_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-3 3"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="50"/></actuator>
</mujoco>
"""


@pytest.fixture
def arm_at(tmp_path):
    from strands_robots.simulation import Simulation

    sims = []

    def make(position, xml=_ARM_XML):
        path = tmp_path / f"arm_{len(sims)}.xml"
        path.write_text(xml)
        sim = Simulation()
        sim.create_world(timestep=0.002)
        assert sim.add_robot("arm", urdf_path=str(path), position=position)["status"] == "success"
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.destroy()


@requires_mujoco
class TestGetRobotStateNamesTheReachAxis:
    def test_text_carries_base_offset_and_axis(self, arm_at) -> None:
        text = arm_at([0.0, 0.0, 0.0]).get_robot_state("arm")["content"][0]["text"]
        line = next(ln for ln in text.splitlines() if ln.startswith("end_effector"))
        assert "from base [0.0000, 0.0000, 0.1000]: [+0.4000, +0.0000, +0.0000]" in line
        assert "(the arm currently extends along +X)" in line

    def test_the_offset_is_measured_from_where_the_fixed_base_stands(self, arm_at) -> None:
        res = arm_at([1.0, -2.0, 0.0]).get_robot_state("arm")
        ee = res["content"][1]["json"]["end_effector"]
        # The request was z=0; the model's root body carries pos="0 0 0.1", so
        # the base stands at z=0.1 and the gripper is level with it.
        assert ee["base"] == [1.0, -2.0, 0.1]
        assert ee["position"] == pytest.approx([1.4, -2.0, 0.1], abs=1e-6)
        assert ee["from_base"] == pytest.approx([0.4, 0.0, 0.0], abs=1e-6)
        assert ee["extends_along"] == "+X"

    def test_the_axis_follows_the_arm_as_it_moves(self, arm_at) -> None:
        import math

        sim = arm_at([0.0, 0.0, 0.0])
        # Swing the shoulder 90 deg: the same arm now points along +Y.
        sim.send_action({"shoulder_act": math.pi / 2}, robot_name="arm", n_substeps=400)
        ee = sim.get_robot_state("arm")["content"][1]["json"]["end_effector"]
        assert ee["extends_along"] == "+Y"
        assert ee["from_base"][1] == pytest.approx(0.4, abs=0.05)  # servo still settling; the axis is the point

    def test_the_offset_plus_base_is_a_usable_move_to_target(self, arm_at) -> None:
        sim = arm_at([0.5, 0.5, 0.0])
        ee = sim.get_robot_state("arm")["content"][1]["json"]["end_effector"]
        target = [ee["base"][i] + ee["from_base"][i] for i in range(3)]
        assert target == pytest.approx(ee["position"], abs=1e-9)


_OFFSET_ROOT_XML = _ARM_XML.replace('<body name="base" pos="0 0 0.1">', '<body name="base" pos="0 -0.5 0.1">')

_TWO_ROOT_XML = """
<mujoco model="two_root">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="left_base" pos="0 0.2 0">
      <geom type="cylinder" size="0.05 0.05"/>
      <body name="gripper" pos="0.1 0 0.1"><geom type="box" size="0.02 0.02 0.02"/></body>
    </body>
    <body name="right_base" pos="0 -0.2 0">
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@requires_mujoco
class TestTheBaseIsWhereTheRobotStandsNotWhatWasRequested:
    """``position`` is the attach FRAME: MuJoCo composes it with the model's own
    authored root pose rather than replacing it, so the request names a place the
    robot is not for 30 of the registry's 55 single-root models. Measuring the
    offset from the request flipped the axis this reading exists to name.
    """

    def test_an_authored_root_offset_does_not_flip_the_axis(self, arm_at) -> None:
        # Same arm, root body authored at pos="0 -0.5 0.1": it stands half a
        # metre to -Y and extends +X from there. Measured from the request the
        # dominant component was the BASE's own offset, so the line said the
        # arm extends along -Y - pointing an agent at the base, not the reach.
        res = arm_at([0.0, 0.0, 0.0], _OFFSET_ROOT_XML).get_robot_state("arm")
        ee = res["content"][1]["json"]["end_effector"]
        assert ee["base"] == [0.0, -0.5, 0.1]
        assert ee["from_base"] == pytest.approx([0.4, 0.0, 0.0], abs=1e-6)
        assert ee["extends_along"] == "+X"
        assert "(the arm currently extends along +X)" in res["content"][0]["text"]

    def test_a_model_with_several_roots_reports_the_requested_attach_frame(self, arm_at) -> None:
        # No one root body has a claim to being "the base" (an aloha attaches
        # two arm bases, an rby1 six), so the offset is from the attach frame -
        # the same fallback list_robots labels.
        res = arm_at([0.25, 0.0, 0.0], _TWO_ROOT_XML).get_robot_state("arm")
        ee = res["content"][1]["json"]["end_effector"]
        assert ee["base"] == [0.25, 0.0, 0.0]
        assert ee["from_base"] == pytest.approx([0.1, 0.2, 0.1], abs=1e-6)

    def test_a_mobile_base_that_drove_keeps_a_live_offset(self) -> None:
        # LeKiwi attaches several root bodies (so no single root pose can be
        # measured) AND has a floating base, so the live pose has to answer
        # first: falling through to the request would peg the base at the spawn
        # point and grow from_base by however far the robot drove.
        import mujoco

        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("lekiwi")["status"] == "success"
            before = sim.get_robot_state("lekiwi")["content"][1]["json"]["end_effector"]
            world = sim._world
            assert world is not None
            model, data = world._model, world._data
            assert model is not None and data is not None
            robot = world.robots["lekiwi"]
            free_jnt = sim._robot_base_free_joint(model, robot, robot.namespace or "")
            data.qpos[int(model.jnt_qposadr[free_jnt])] += 1.5  # drive 1.5 m along +X
            mujoco.mj_forward(model, data)
            after = sim.get_robot_state("lekiwi")["content"][1]["json"]["end_effector"]
            assert after["base"][0] == pytest.approx(before["base"][0] + 1.5, abs=1e-6)
            assert after["from_base"] == pytest.approx(before["from_base"], abs=1e-6)
        finally:
            sim.destroy()

    @pytest.mark.parametrize("robot_name", ["xarm7", "leap_hand"])
    def test_the_base_agrees_with_list_robots(self, robot_name) -> None:
        # Two readings of one robot in one session cannot disagree about where
        # its base is. Both models bolt their root above the floor (xarm7 at
        # z=0.12, leap_hand at z=0.1), and both hold their end effector BELOW
        # that base at rest - which the request-relative offset reported as
        # above it.
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot(robot_name)["status"] == "success"
            listed = re.search(r"Position: (\[[^\]]*\])", sim.list_robots_info()["content"][0]["text"])
            assert listed is not None
            ee = sim.get_robot_state(robot_name)["content"][1]["json"]["end_effector"]
            assert ee["base"] == pytest.approx(ast.literal_eval(listed.group(1)), abs=1e-4)
            assert ee["from_base"][2] < 0.0
        finally:
            sim.destroy()


@requires_mujoco
class TestOnTheBundledArms:
    @pytest.mark.parametrize("data_config", ["so100", "so101"])
    def test_the_so_arms_report_their_reach_along_minus_y(self, data_config) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("arm", data_config=data_config)["status"] == "success"
            res = sim.get_robot_state("arm")
            assert "(the arm currently extends along -Y)" in res["content"][0]["text"]
            assert res["content"][1]["json"]["end_effector"]["extends_along"] == "-Y"
        finally:
            sim.destroy()
