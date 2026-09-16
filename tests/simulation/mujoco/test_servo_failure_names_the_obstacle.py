"""A not-reached ``move_to`` / ``rotate_wrist`` names what stopped the servo.

Measured with an agent on the bundled so100 (v0.5.2 devx replay, s09):
three ``move_to`` calls in a row ended in "the pose fights joint
limits/contacts" while ``get_contacts`` showed the actual cause - a
self-collision ``Fixed_Jaw <-> Base``, the target sat too close to the base.
The refusal now reads the engine at the final tick and names the active
contacts on the robot (at most three, nearest first, total reported) and the
commanded joints sitting at a bound, and carries the same facts as
``json.obstruction``. When nothing visible blocks the arm it says so, which
is the "raise max_steps" case.

``rotate_wrist`` servos to a set-point the same way and used to report only
the residual it fell short by: measured on the bundled so100 with the gripper
folded into its own base, the wrist did not turn at all (0.003 of 2.500 rad)
while 15 active contact pairs held it, and the refusal named none of them. It reads
the same obstruction now, scoped to the one joint it commands to a new value.
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.motion_primitives_base import (
    JOINT_LIMIT_MARGIN_FRACTION,
    OBSTRUCTION_MAX_CONTACTS,
    MotionPrimitivesCore,
)

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

_text = MotionPrimitivesCore._obstruction_text


class TestObstructionText:
    def test_engine_did_not_look_keeps_the_legacy_clause(self) -> None:
        assert _text(None) == "The servo may need more steps, or the pose fights joint limits/contacts."

    def test_contacts_are_named_with_distance_and_total(self) -> None:
        text = _text(
            {
                "contacts": [{"a": "so100/Fixed_Jaw", "b": "so100/Base", "dist_m": -0.0021}],
                "contacts_total": 5,
                "joints_at_limit": [],
            }
        )
        assert "the robot is in contact: 'so100/Fixed_Jaw' <-> 'so100/Base' (d=-0.0021 m) and 4 more" in text
        assert text.endswith("Clear what it touches or command a value that avoids it, or loosen tol.")

    def test_joints_at_limit_are_named_with_side_and_bound(self) -> None:
        text = _text(
            {
                "contacts": [],
                "contacts_total": 0,
                "joints_at_limit": [{"joint": "arm/elbow", "pos": 1.5699, "limit": 1.57, "side": "upper"}],
            }
        )
        assert "commanded joint(s) at a limit: 'arm/elbow' at its upper limit (1.5699 vs 1.5700)" in text
        assert "Command a value that joint can reach inside its range, or loosen tol." in text

    def test_nothing_visible_points_at_max_steps(self) -> None:
        text = _text({"contacts": [], "contacts_total": 0, "joints_at_limit": []})
        assert "no contact involved the robot and no commanded joint was at a limit" in text
        assert "raise max_steps" in text

    def test_a_backend_that_did_not_look_publishes_no_obstruction_field(self) -> None:
        # A backend without this reader (Isaac's rotate_wrist) passes None, and
        # a null field would read as "looked, found nothing" to every consumer.
        res = MotionPrimitivesCore._rotate_wrist_result(
            "arm",
            0.02,
            10,
            reached=False,
            steps_used=10,
            wrist_name="wrist",
            target_yaw=1.0,
            final_yaw=0.5,
            yaw_error=0.5,
        )
        assert "obstruction" not in res["content"][1]["json"]
        assert res["content"][0]["text"].endswith(
            "The servo may need more steps, or the pose fights joint limits/contacts."
        )

    def test_backend_that_did_not_read_contacts_does_not_claim_the_robot_is_free(self) -> None:
        text = _text({"contacts": [], "contacts_total": None, "joints_at_limit": []})
        assert "contacts were not read on this backend" in text
        assert "no contact involved" not in text


_YAW_ARM_XML = """
<mujoco model="yaw_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="jaw" type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
    {extra}
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="200" kv="5"/></actuator>
</mujoco>
"""

# A pitch hinge with gravity on and a servo too weak to lift the link: the
# arm rests against its UPPER bound (gravity pushes it there) while the
# target asks it to point up. IK solves the target exactly, so the failure
# reaches the servo path - which is the path that reads the obstruction.
_PITCH_ARM_XML = """
<mujoco model="pitch_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.6">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-0.2 1.0"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="jaw" type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="0.2" kv="0.05"/></actuator>
</mujoco>
"""


@pytest.fixture
def arm_with(tmp_path):
    from strands_robots.simulation import Simulation

    sims = []

    def make(extra: str = "", xml: str = _YAW_ARM_XML):
        path = tmp_path / f"arm_{len(sims)}.xml"
        path.write_text(xml.format(extra=extra))
        sim = Simulation()
        sim.create_world(timestep=0.002)
        assert sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])["status"] == "success"
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.destroy()


@requires_mujoco
class TestMoveToNamesTheJointLimit:
    def test_a_servo_parked_on_a_bound_names_the_joint(self, arm_with) -> None:
        import math

        sim = arm_with(xml=_PITCH_ARM_XML)
        ang = -0.15  # tip slightly UP; gravity holds the weak servo at the +1.0 bound
        target = [0.4 * math.cos(ang), 0.0, 0.6 - 0.4 * math.sin(ang)]
        res = sim.move_to(robot_name="arm", position=target, tol=0.005, max_steps=60)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        obstruction = res["content"][1]["json"]["obstruction"]
        assert obstruction["contacts"] == [] and obstruction["contacts_total"] == 0
        joints = obstruction["joints_at_limit"]
        assert [j["joint"] for j in joints] == ["arm/shoulder"]
        assert joints[0]["side"] == "upper" and joints[0]["limit"] == pytest.approx(1.0)
        assert joints[0]["pos"] >= 1.0 - max(1.2 * JOINT_LIMIT_MARGIN_FRACTION, 1e-3)
        assert "The servo was stopped: commanded joint(s) at a limit: 'arm/shoulder' at its upper limit" in text
        assert "fights joint limits/contacts" not in text

    def test_a_reached_move_carries_no_obstruction(self, arm_with) -> None:
        sim = arm_with()
        res = sim.move_to(robot_name="arm", position=[0.3894, 0.0919, 0.1], tol=0.02, max_steps=200)
        assert res["status"] == "success"
        assert "obstruction" not in res["content"][1]["json"]


_WALL = """
    <body name="wall" pos="0.35 0.12 0.1">
      <geom name="wall_geom" type="box" size="0.01 0.03 0.1"/>
    </body>
    <body name="loose_cube" pos="1.0 1.0 0.02">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
"""


@requires_mujoco
class TestMoveToNamesTheContact:
    def test_a_wall_in_the_sweep_is_named_and_a_far_cube_is_not(self, arm_with) -> None:
        sim = arm_with(extra=_WALL)
        # Swing toward +Y into the wall standing at y=0.12 across the tip's arc;
        # the target sits ON the arc (0.45 rad), so IK solves it and the servo
        # is what gets stopped.
        res = sim.move_to(robot_name="arm", position=[0.3603, 0.1740, 0.1], tol=0.005, max_steps=60)
        assert res["status"] == "error"
        obstruction = res["content"][1]["json"]["obstruction"]
        pairs = {frozenset((c["a"], c["b"])) for c in obstruction["contacts"]}
        assert frozenset(("arm/jaw", "arm/wall_geom")) in pairs, obstruction
        assert all("cube_geom" not in p for p in pairs), "the loose cube's own contacts are not the robot's"
        assert len(obstruction["contacts"]) <= OBSTRUCTION_MAX_CONTACTS
        assert obstruction["contacts_total"] >= len(obstruction["contacts"])
        text = res["content"][0]["text"]
        assert "the robot is in contact:" in text and "'arm/jaw' <-> 'arm/wall_geom'" in text
        # One line per geom pair even when MuJoCo reports several contact points for it.
        assert len(pairs) == len(obstruction["contacts"])
        assert "Clear what it touches or command a value that avoids it" in text

    def test_robot_body_ids_is_the_arm_subtree_only(self, arm_with) -> None:
        import mujoco as mj

        sim = arm_with(extra=_WALL)
        model = sim._world._model
        robot = sim._world.robots["arm"]
        names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in sim._robot_body_ids(model, robot)}
        assert names == {"arm/base", "arm/link1", "arm/gripper"}


@requires_mujoco
class TestOnTheBundledSo100:
    def test_the_replay_target_names_what_the_jaw_touches(self) -> None:
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("arm", data_config="so100")["status"] == "success"
            # s09: a point low and close to the base, well inside IK reach.
            res = sim.move_to(robot_name="arm", position=[0.0, -0.10, 0.03], tol=0.005, max_steps=60)
            assert res["status"] == "error"
            text = res["content"][0]["text"]
            obstruction = res["content"][1]["json"]["obstruction"]
            assert obstruction["contacts"], text
            assert any("Fixed_Jaw" in c["a"] or "Fixed_Jaw" in c["b"] for c in obstruction["contacts"]), text
            assert "The servo was stopped: the robot is in contact:" in text
            assert "fights joint limits/contacts" not in text
        finally:
            sim.destroy()


@requires_mujoco
class TestRotateWristNamesWhatStoppedTheWrist:
    """The wrist servo reads the same engine facts, scoped to the joint it commands."""

    @staticmethod
    def _so100():
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        assert sim.add_robot("arm", data_config="so100")["status"] == "success"
        return sim

    def test_a_gripper_folded_into_the_base_is_named(self) -> None:
        sim = self._so100()
        try:
            # Elbow fully over: the jaw ends up inside the base, so the wrist
            # cannot turn at all - the case that used to report only a residual.
            sim.set_joint_positions(positions={"Pitch": -0.5, "Elbow": 3.0}, robot_name="arm")
            res = sim.rotate_wrist(robot_name="arm", target_yaw=2.5, max_steps=200)
            assert res["status"] == "error"
            text = res["content"][0]["text"]
            obstruction = res["content"][1]["json"]["obstruction"]
            assert obstruction["contacts"], text
            assert obstruction["contacts_total"] >= len(obstruction["contacts"])
            assert any("Fixed_Jaw" in c["a"] or "Fixed_Jaw" in c["b"] for c in obstruction["contacts"]), text
            assert "The servo was stopped: the robot is in contact:" in text
        finally:
            sim.destroy()

    def test_a_wrist_parked_on_its_own_bound_names_that_joint(self) -> None:
        sim = self._so100()
        try:
            # The bound itself is a legal target, so the servo saturates there:
            # the joint at a limit is the one rotate_wrist commanded, and the
            # joints it merely holds are not offered as the cause.
            res = sim.rotate_wrist(robot_name="arm", target_yaw=2.79, tol=1e-5, max_steps=60)
            assert res["status"] == "error"
            obstruction = res["content"][1]["json"]["obstruction"]
            assert [j["joint"] for j in obstruction["joints_at_limit"]] == ["arm/Wrist_Roll"]
            assert obstruction["joints_at_limit"][0]["side"] == "upper"
            assert "commanded joint(s) at a limit: 'arm/Wrist_Roll' at its upper limit" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_an_unobstructed_wrist_that_ran_out_of_steps_says_so(self) -> None:
        sim = self._so100()
        try:
            # Pitch parked ON its upper bound: rotate_wrist only holds that
            # joint where it already was, so it is not what stopped the wrist
            # and is not offered as the cause.
            sim.set_joint_positions(positions={"Pitch": 0.174}, robot_name="arm")
            res = sim.rotate_wrist(robot_name="arm", target_yaw=2.0, max_steps=2)
            assert res["status"] == "error"
            obstruction = res["content"][1]["json"]["obstruction"]
            assert obstruction == {"contacts": [], "contacts_total": 0, "joints_at_limit": []}
            assert "raise max_steps" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_the_robot_subtree_starts_at_its_root_not_its_first_jointed_body(self) -> None:
        import mujoco as mj

        sim = self._so100()
        try:
            model = sim._world._model
            bodies = sim._robot_body_ids(model, sim._world.robots["arm"])
            names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies}
            # The base carries no joint, so reading the first joint's own body
            # would drop it - and the base is what a folded gripper hits.
            assert "arm/Base" in names
            assert names == {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(1, int(model.nbody))}
        finally:
            sim.destroy()

    def test_a_reached_rotation_carries_no_obstruction(self) -> None:
        sim = self._so100()
        try:
            res = sim.rotate_wrist(robot_name="arm", target_yaw=1.0)
            assert res["status"] == "success"
            assert "obstruction" not in res["content"][1]["json"]
        finally:
            sim.destroy()
