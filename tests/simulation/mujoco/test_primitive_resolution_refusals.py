"""Every motion primitive refuses when it cannot resolve what it must drive.

``move_to`` / ``set_gripper`` / ``rotate_wrist`` each begin by resolving the
thing they are about to command - an end-effector frame, the arm's
joint-transmission actuators, the gripper actuators, the wrist joint - from the
compiled model plus the registry. Each resolution can come up empty, and each
primitive then returns a structured error naming what it looked for. Six such
refusals exist and none of them was driven:

===============  ==============================================  =============
primitive        cannot resolve                                  fixture here
===============  ==============================================  =============
``move_to``      an end-effector frame                           bodyless
``move_to``      any joint-transmission actuator                 actuatorless
``move_to``      a NON-gripper actuator                          gripper-only
``set_gripper``  a gripper actuator                              no-gripper
``rotate_wrist`` any joint-transmission actuator                 actuatorless
``rotate_wrist`` a wrist joint                                   gripper-only
===============  ==============================================  =============

``tests/simulation/mujoco/test_motion_primitives.py`` owns the primitives' whole
tool contract, and its own docstring enumerates the guards it pins as "no-world,
unknown-robot, and refuse-while-policy-running" - three guards about the world
and the robot registry. The resolution family is a different question (the robot
IS registered and the world IS live; it simply cannot be driven), and it is
absent from that list: no test in the tree drove any of the six, so a regression
turning one into a silent success, a bare raise, or a message naming the wrong
thing was invisible. The sibling Isaac backend already pins its wrist-resolution
refusal in ``tests/simulation/isaac/test_motion_primitives.py``, so MuJoCo - the
reference backend - was the unpinned one.

Every fixture is an inline MJCF (no asset download) that removes exactly one
thing the resolution needs, and every refusal is reached before any IK solve, so
this module needs neither ``mink`` nor a GL context. Only the over-reach control
- the conventional arm on which all three primitives still succeed - needs the
IK bridge, and it is gated accordingly.

Out of scope, deliberately:

* ``_gripper_setpoint_range``'s refusals, which resolve a gripper actuator's
  set-point BOUNDS rather than the entity to drive.
  ``tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py`` owns
  that family.
* the mid-run aborts (world destroyed, robot removed, policy started) - a
  primitive that resolved everything and then lost it.
  ``tests/simulation/mujoco/test_primitive_teardown_abort.py`` owns those.
"""

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# A robot that contributes no body of its own: the MJCF holds only a worldbody
# geom, so after the namespaced attach there is no site, no hand/tool body and
# no kinematic chain for ``discover_ee_frame`` to walk. This is a real shape -
# a scene fragment or a static prop handed to ``add_robot`` - and ``add_robot``
# accepts it ("Joints: 0, Actuators: 0").
BODYLESS_XML = """
<mujoco model="prim_bodyless">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="pad" type="plane" size="0.4 0.4 0.05"/>
  </worldbody>
</mujoco>
"""

# Hinges and a TCP site but no ``<actuator>`` block: the EE frame resolves, so
# both primitives get past frame discovery and refuse on the actuator map. This
# is what a bare URDF compiles to before ``actuate_robot`` adds servos.
ACTUATORLESS_XML = """
<mujoco model="prim_actuatorless">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
        <site name="ee_site" pos="0.15 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# One actuated joint, and the name heuristic classifies it as a gripper. So the
# actuator map is non-empty (past the no-actuator refusal) and yet nothing is
# left to move the arm with, or to rotate as a wrist.
GRIPPER_ONLY_XML = """
<mujoco model="prim_gripper_only">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="jaw_body" pos="0 0 0.05">
        <joint name="jaw" type="hinge" axis="0 0 1" range="-0.2 1.5" damping="0.1"/>
        <geom type="box" size="0.01 0.01 0.02"/>
        <site name="ee_site" pos="0.02 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="gripper" joint="jaw" kp="20" ctrlrange="-0.2 1.5"/>
  </actuator>
</mujoco>
"""

# Two actuated hinges whose names match no gripper hint, and the second
# actuator is deliberately UNNAMED - ``mj_id2name`` returns ``None`` for it, so
# the refusal's actuator listing exercises ``_short_name``'s empty-name
# fallback while rendering the message.
NO_GRIPPER_XML = """
<mujoco model="prim_no_gripper">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
        <body name="link2" pos="0.15 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-3.0 3.0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.018"/>
          <site name="ee_site" pos="0.1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="30" ctrlrange="-3.14 3.14"/>
    <position joint="j2" kp="30" ctrlrange="-3.0 3.0"/>
  </actuator>
</mujoco>
"""

# (label, xml, primitive, kwargs) for the six refusals, so the shared
# properties below cannot drift from the per-primitive assertions above.
_REFUSALS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("bodyless", BODYLESS_XML, "move_to", {"position": list(REACHABLE)}),
    ("actuatorless", ACTUATORLESS_XML, "move_to", {"position": list(REACHABLE)}),
    ("gripper_only", GRIPPER_ONLY_XML, "move_to", {"position": [0.02, 0.0, 0.06]}),
    ("no_gripper", NO_GRIPPER_XML, "set_gripper", {"state": "open"}),
    ("actuatorless", ACTUATORLESS_XML, "rotate_wrist", {"target_yaw": 0.3}),
    ("gripper_only", GRIPPER_ONLY_XML, "rotate_wrist", {"target_yaw": 0.3}),
]

_ESCAPE_HATCHES = ("action='actuate_robot'", "action='send_action'")


def _text(result: dict[str, Any]) -> str:
    """Every text block of a tool result, joined."""
    return "\n".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def make_sim(tmp_path):
    """Factory: a live world holding one robot built from *xml*, named ``arm``.

    A factory rather than a fixture per shape so a single test can compare two
    shapes, and so the six refusals stay declared in one table.
    """
    made: list[Simulation] = []

    def _make(label: str, xml: str) -> Simulation:
        path = tmp_path / f"{label}.xml"
        path.write_text(xml)
        s = Simulation(tool_name=f"test_prim_resolution_{label}_{len(made)}", mesh=False)
        assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
        assert s.add_robot("arm", urdf_path=str(path))["status"] == "success"
        made.append(s)
        return s

    yield _make
    for s in made:
        s.cleanup(policy_stop_timeout=2.0)


class TestTheFixturesRemoveExactlyOneResolution:
    """Premise: each fixture is missing the one thing its refusal names, and
    nothing else. Without these, a refusal could be firing for an unrelated
    reason and the tests below would pass for the wrong reason."""

    def test_the_bodyless_robot_contributes_no_frame(self, make_sim):
        from strands_robots.simulation.ik import discover_ee_frame

        sim = make_sim("bodyless", BODYLESS_XML)
        assert discover_ee_frame(sim._world._model, "arm/") is None

    def test_the_actuatorless_arm_has_a_frame_but_no_actuators(self, make_sim):
        from strands_robots.simulation.ik import discover_ee_frame

        sim = make_sim("actuatorless", ACTUATORLESS_XML)
        robot = sim._world.robots["arm"]
        assert discover_ee_frame(sim._world._model, "arm/") is not None, (
            "the frame must resolve, or move_to would refuse on the frame instead"
        )
        assert list(robot.joint_ids), "the arm must have joints"
        assert not list(robot.actuator_ids or []), "the arm must have no actuators"

    def test_the_gripper_only_arm_has_one_actuator_and_it_is_a_gripper(self, make_sim):
        sim = make_sim("gripper_only", GRIPPER_ONLY_XML)
        robot = sim._world.robots["arm"]
        assert len(list(robot.actuator_ids or [])) == 1
        with sim._lock:
            grip, meta, err = sim._resolve_gripper_actuators(sim._world._model, robot)
        assert err is None and meta is None, "the heuristic path must resolve without error"
        assert grip == set(int(a) for a in robot.actuator_ids), (
            "every actuator must classify as a gripper, or nothing is left over to refuse about"
        )

    def test_the_no_gripper_arm_has_actuators_and_none_is_a_gripper(self, make_sim):
        sim = make_sim("no_gripper", NO_GRIPPER_XML)
        robot = sim._world.robots["arm"]
        assert len(list(robot.actuator_ids or [])) == 2
        with sim._lock:
            grip, _meta, err = sim._resolve_gripper_actuators(sim._world._model, robot)
        assert err is None
        assert grip == set(), "no actuator or joint name may match a gripper hint"

    def test_the_no_gripper_arm_carries_one_unnamed_actuator(self, make_sim):
        """``_short_name``'s empty-name fallback is only reachable through an
        actuator MuJoCo reports as ``None``; this is the fixture that has one."""
        import mujoco as mj

        sim = make_sim("no_gripper", NO_GRIPPER_XML)
        model = sim._world._model
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        assert None in names, f"expected an unnamed actuator, got {names}"


class TestMoveToRefusesWhatItCannotDrive:
    def test_refuses_when_no_end_effector_frame_resolves(self, make_sim):
        sim = make_sim("bodyless", BODYLESS_XML)
        res = sim.move_to(robot_name="arm", position=list(REACHABLE))
        text = _text(res)
        assert res["status"] == "error", text
        assert "could not auto-discover an end-effector frame" in text
        assert "no TCP-like site or hand/tool body" in text, (
            "the refusal must name what it looked for, since the model is the only thing to fix"
        )

    def test_refuses_when_the_robot_has_no_actuators(self, make_sim):
        sim = make_sim("actuatorless", ACTUATORLESS_XML)
        res = sim.move_to(robot_name="arm", position=list(REACHABLE))
        text = _text(res)
        assert res["status"] == "error", text
        assert "has no joint-transmission actuators to drive" in text
        assert "action='actuate_robot'" in text

    def test_refuses_when_every_actuated_joint_is_a_gripper(self, make_sim):
        sim = make_sim("gripper_only", GRIPPER_ONLY_XML)
        res = sim.move_to(robot_name="arm", position=[0.02, 0.0, 0.06])
        text = _text(res)
        assert res["status"] == "error", text
        assert "no non-gripper joint-transmission" in text
        assert "Nothing can move the end-effector." in text
        assert "gripper" in text and "jaw" in text, "the refusal must name the heuristic it applied"

    def test_the_no_actuator_refusal_is_distinct_from_the_all_gripper_one(self, make_sim):
        """Both shapes leave move_to with nothing to drive, and the two refusals
        must stay distinguishable: one says add servos, the other says the
        servos you have are all gripper drives."""
        bare = _text(make_sim("actuatorless", ACTUATORLESS_XML).move_to(robot_name="arm", position=list(REACHABLE)))
        grip = _text(make_sim("gripper_only", GRIPPER_ONLY_XML).move_to(robot_name="arm", position=[0.02, 0.0, 0.06]))
        assert bare != grip
        assert "non-gripper" not in bare
        assert "action='actuate_robot'" not in grip


class TestSetGripperRefusesWhatItCannotDrive:
    def test_refuses_when_no_name_matches_a_gripper_hint(self, make_sim):
        sim = make_sim("no_gripper", NO_GRIPPER_XML)
        res = sim.set_gripper(robot_name="arm", state="open")
        text = _text(res)
        assert res["status"] == "error", text
        assert "could not resolve a gripper actuator" in text
        assert "registry carries no gripper metadata" in text
        assert "action='send_action'" in text

    def test_the_refusal_lists_an_unnamed_actuator_as_an_empty_name(self, make_sim):
        """The listing is built from ``_short_name``, which answers ``""`` for an
        actuator MuJoCo has no name for. Rendering the refusal must survive one
        rather than raising while explaining itself."""
        sim = make_sim("no_gripper", NO_GRIPPER_XML)
        text = _text(sim.set_gripper(robot_name="arm", state="open"))
        assert "Actuators: ['a1', '']" in text, text


class TestRotateWristRefusesWhatItCannotDrive:
    def test_refuses_when_the_robot_has_no_actuators(self, make_sim):
        sim = make_sim("actuatorless", ACTUATORLESS_XML)
        res = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
        text = _text(res)
        assert res["status"] == "error", text
        assert "has no joint-transmission actuators to drive" in text
        assert "action='actuate_robot'" in text

    def test_refuses_when_every_hinge_is_a_gripper_so_no_wrist_resolves(self, make_sim):
        """The distal-hinge fallback must not reach past the gripper
        classification: with the jaw as the only actuated hinge there is no
        wrist, and picking it would rotate the gripper instead."""
        sim = make_sim("gripper_only", GRIPPER_ONLY_XML)
        res = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
        text = _text(res)
        assert res["status"] == "error", text
        assert "could not resolve a wrist joint" in text
        assert "Actuated hinge joints: ['jaw']" in text
        assert "action='send_action'" in text


class TestPropertiesSharedByEveryResolutionRefusal:
    @pytest.mark.parametrize(
        ("label", "xml", "primitive", "kwargs"),
        _REFUSALS,
        ids=[f"{p}-{lbl}" for lbl, _x, p, _k in _REFUSALS],
    )
    def test_it_is_a_structured_error_naming_the_action_and_the_robot(self, make_sim, label, xml, primitive, kwargs):
        sim = make_sim(label, xml)
        res = getattr(sim, primitive)(robot_name="arm", **kwargs)
        text = _text(res)
        assert res["status"] == "error", text
        assert text.startswith(f"{primitive}:") or f"{primitive}:" in text
        assert "'arm'" in text

    @pytest.mark.parametrize(
        ("label", "xml", "primitive", "kwargs"),
        _REFUSALS,
        ids=[f"{p}-{lbl}" for lbl, _x, p, _k in _REFUSALS],
    )
    def test_it_leaves_the_scene_exactly_as_it_found_it(self, make_sim, label, xml, primitive, kwargs):
        """A resolution refusal happens before any tick, so it must cost no
        physics: the same qpos, the same ctrl, the same clock."""
        import numpy as np

        sim = make_sim(label, xml)
        data = sim._world._data
        qpos_before = np.array(data.qpos, copy=True)
        ctrl_before = np.array(data.ctrl, copy=True)
        time_before = float(data.time)

        res = getattr(sim, primitive)(robot_name="arm", **kwargs)
        assert res["status"] == "error", _text(res)

        assert np.array_equal(np.array(sim._world._data.qpos), qpos_before)
        assert np.array_equal(np.array(sim._world._data.ctrl), ctrl_before)
        assert float(sim._world._data.time) == time_before

    def test_at_least_one_refusal_per_primitive_offers_the_direct_escape_hatch(self, make_sim):
        """Four of the six point the caller at ``send_action`` /
        ``actuate_robot``; the other two describe a model gap that no tool call
        can work around. Every primitive has at least one of the former, so the
        escape hatch is always discoverable from the primitive that failed."""
        with_hatch: set[str] = set()
        for label, xml, primitive, kwargs in _REFUSALS:
            text = _text(getattr(make_sim(label, xml), primitive)(robot_name="arm", **kwargs))
            if any(h in text for h in _ESCAPE_HATCHES):
                with_hatch.add(primitive)
        assert with_hatch == {"move_to", "set_gripper", "rotate_wrist"}


class TestTheConventionalArmStillResolvesEverything:
    """Over-reach control: the fixtures above remove one resolution each, so the
    arm that removes none must still drive all three primitives. Without this,
    every refusal test would also pass on a build that refused unconditionally."""

    @pytest.fixture
    def arm(self, make_sim):
        return make_sim("conventional", ARM_XML)

    def test_move_to_still_reaches_a_reachable_target(self, arm):
        pytest.importorskip("mink")
        res = arm.move_to(robot_name="arm", position=list(REACHABLE), tol=0.02, max_steps=400)
        assert res["status"] == "success", _text(res)

    def test_set_gripper_still_resolves_the_jaw(self, arm):
        res = arm.set_gripper(robot_name="arm", state="open")
        assert res["status"] == "success", _text(res)

    def test_rotate_wrist_still_resolves_the_wrist(self, arm):
        res = arm.rotate_wrist(robot_name="arm", target_yaw=0.3, tol=0.05, max_steps=300)
        assert res["status"] == "success", _text(res)
