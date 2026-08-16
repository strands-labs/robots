"""``move_to`` judges reachability by the joints it actually commands.

``move_to`` solves inverse kinematics with
:class:`strands_robots.simulation.ik.MinkIKBridge` and then drives the arm's
position-servo actuators toward the solution. ``mink`` optimizes over every
degree of freedom in the model it is handed, and the MuJoCo backend hands it the
whole WORLD model, so an unrestricted solve is free to satisfy the Cartesian
task with a degree of freedom the servo loop never sends: a floating/mobile
base, a second robot sharing the world, or the gripper the primitive holds at
its live position.

The consequences are not cosmetic. On the mobile manipulator below, an
unrestricted solve slides the free base 0.91 m and rotates it, leaves both
commanded arm joints at exactly their seed values, and reports an IK residual of
4e-12 m. ``move_to`` accepts that as reachable, servos for the full
``max_steps`` budget, and returns a timeout whose text reads "IK residual was
0.0000 m ... The servo may need more steps" for a point 1.23 m outside the arm's
reach - so the one number the caller has to diagnose with says the target is
solved and blames the servo. More steps never arrive.

The same borrowing contradicts the primitive's own GRASP PRESERVATION contract,
which states that gripper actuators "are excluded from the IK solve". Whenever
the discovered TCP sits on the jaw body, the jaw is kinematically relevant and
an unrestricted solve reaches the target by opening it 40 mm - a solve the
grasp-preserving servo will not carry out.

The fix restricts the solve to the commanded joints (``commanded_dofs``), so
``ik_residual_m`` is the error the servo descent is genuinely left with and the
unreachable refusal fires up front. Because the restriction removes the cheap
answer, the solver works the arm instead: it lands at 0.68 m, better than the
1.25 m the borrowed solution's commanded half is worth. The refusal then
re-solves unrestricted purely to diagnose, so it can distinguish a point outside
the robot's workspace ("choose a closer target") from one that needs uncommanded
motion (name the base, and say to drive it first).

The controls in the last two classes pin that nothing changes where every
kinematically relevant joint is commanded, which is every fixed-base arm in the
suite: the same target is reached, and an out-of-workspace point keeps its
existing advice.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.ik import MinkIKBridge, discover_ee_frame  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE, UNREACHABLE  # noqa: E402

_TOL = 0.01
_REACH_TOL = 0.02
_MAX_STEPS = 400

# A two-joint arm on a free-floating base: a mobile manipulator's kinematics.
# Only the two hinges carry position servos, so the base is exactly the kind of
# degree of freedom move_to cannot command but mink can move. The base free
# joint is NAMED here so the refusal's wording can be pinned; an unnamed
# <freejoint/> (what several registry mobile bases ship) is reported by index.
MOBILE_ARM_XML = """
<mujoco model="mobile_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.15">
      <freejoint name="base_free"/>
      <geom type="box" size="0.12 0.12 0.05"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.03"/>
        <body name="hand" pos="0 0 0.30">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.025"/>
          <site name="attachment_site" pos="0 0 0.25"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder" kp="80" ctrlrange="-1.8 1.8"/>
    <position name="elbow" joint="elbow" kp="60" ctrlrange="-2.2 2.2"/>
  </actuator>
</mujoco>
"""

# A fixed-base arm whose TCP site sits on the JAW body, so the held gripper DOF
# is kinematically relevant to the end-effector task. ARM_XML deliberately puts
# its site upstream of the jaw, which is why the grasp-preservation claim about
# the solve had never been exercised.
GRIPPER_TCP_ARM_XML = """
<mujoco model="gripper_tcp_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body name="hand" pos="0.2 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.018"/>
          <body name="jaw_body" pos="0.1 0 0">
            <joint name="jaw" type="slide" axis="1 0 0" range="0 0.25"/>
            <geom type="box" size="0.01 0.01 0.02" contype="0" conaffinity="0"/>
            <site name="grasp_site" pos="0.02 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder" kp="20" ctrlrange="-3.14 3.14"/>
    <position name="elbow" joint="elbow" kp="20" ctrlrange="-3.14 3.14"/>
    <position name="jaw" joint="jaw" kp="20" ctrlrange="0 0.25"/>
  </actuator>
</mujoco>
"""

# 1.20 m out in +x at the tool frame's home height: unreachable for the 0.55 m
# arm from its home base pose, and trivially reachable if the base drives there.
# Inside the 5 m workspace sanity box, so it reaches the solver, not the guard.
NEEDS_BASE_MOTION = [1.20, 0.0, 0.75]


def _sim_with(tmp_path: Any, name: str, xml: str, *, weightless: bool = False) -> Any:
    """A world holding ``xml`` under the ``name/`` namespace.

    ``weightless`` matches the shared motion-primitive fixture, which zeroes
    world gravity so a servo descent settles on its set-point instead of
    hanging under load - the premise the reach controls below inherit.
    """
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    sim = Simulation(backend="mujoco", mesh=False)
    created = sim.create_world(gravity=[0, 0, 0]) if weightless else sim.create_world()
    assert created["status"] == "success"
    added = sim.add_robot(name=name, urdf_path=str(path))
    assert added["status"] == "success", added
    return sim


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    """The primitive's structured payload."""
    return next(c["json"] for c in result["content"] if "json" in c)


def _text(result: dict[str, Any]) -> str:
    """The primitive's message text."""
    return next(c["text"] for c in result["content"] if "text" in c)


@pytest.fixture
def mobile_sim(tmp_path):
    """The floating-base mobile manipulator."""
    sim = _sim_with(tmp_path, "rover", MOBILE_ARM_XML, weightless=True)
    try:
        yield sim
    finally:
        sim.cleanup()


@pytest.fixture
def gripper_tcp_sim(tmp_path):
    """The fixed-base arm whose TCP rides on the jaw."""
    sim = _sim_with(tmp_path, "grip", GRIPPER_TCP_ARM_XML, weightless=True)
    try:
        yield sim
    finally:
        sim.cleanup()


@pytest.fixture
def fixed_sim(tmp_path):
    """The suite's shared fully actuated fixed-base arm."""
    sim = _sim_with(tmp_path, "arm", ARM_XML, weightless=True)
    try:
        yield sim
    finally:
        sim.cleanup()


class TestAnUncommandedDegreeOfFreedomCannotSolveTheTarget:
    """The floating base is not available to the solve that judges the target."""

    def test_the_target_is_refused_rather_than_accepted_and_servoed(self, mobile_sim):
        """The point 1.23 m out of reach is refused as unreachable, up front.

        Pre-fix the solve slid the base, reported a 4e-12 m residual, and this
        call spent the whole ``max_steps`` budget before returning a servo
        timeout - the wrong failure, with the wrong remedy, at 400x the cost.
        """
        result = mobile_sim.move_to(robot_name="rover", position=NEEDS_BASE_MOTION, tol=_TOL, max_steps=_MAX_STEPS)
        assert result["status"] == "error"
        assert "unreachable" in _text(result)
        assert _json_block(result)["steps"] == 0

    def test_the_reported_residual_is_the_one_the_arm_is_left_with(self, mobile_sim):
        """``ik_residual_m`` measures a configuration the servo can command.

        The target sits 1.23 m beyond the arm's reach, so any residual small
        enough to look solved is measured on borrowed motion.
        """
        payload = _json_block(
            mobile_sim.move_to(robot_name="rover", position=NEEDS_BASE_MOTION, tol=_TOL, max_steps=_MAX_STEPS)
        )
        assert payload["ik_residual_m"] > 0.5

    def test_the_refusal_names_the_borrowed_joint_and_what_it_would_reach(self, mobile_sim):
        """A solvable-but-not-by-this-primitive target says so, and names why.

        This is the difference between "choose a closer target" (useless: the
        point is well inside the robot's workspace once the base drives) and
        "move the base first".
        """
        result = mobile_sim.move_to(robot_name="rover", position=NEEDS_BASE_MOTION, tol=_TOL, max_steps=_MAX_STEPS)
        payload = _json_block(result)
        assert payload["uncommanded_joints_moved"] == ["base_free"]
        assert payload["unrestricted_ik_residual_m"] <= _TOL
        assert "base_free" in _text(result)
        assert "Choose a closer target" not in _text(result)

    def test_the_diagnosis_solve_does_not_write_the_world(self, mobile_sim):
        """The refusal is a pure query: no stepping, no solved pose written.

        The diagnosis re-solve produces a configuration that slides the base
        0.91 m, so it must stay a measurement. The world is weightless, which
        removes gravity as an alternative explanation for base travel: whatever
        moves here was moved by ``move_to``. Pre-fix this call accepted the
        borrowed solve and stepped the servo 400 times instead of refusing, so
        it disturbed the world it had nothing to do in.
        """
        before = np.array(mobile_sim._world._data.qpos[:7], dtype=float, copy=True)
        mobile_sim.move_to(robot_name="rover", position=NEEDS_BASE_MOTION, tol=_TOL, max_steps=_MAX_STEPS)
        np.testing.assert_allclose(mobile_sim._world._data.qpos[:7], before, atol=1e-12)


class TestTheHeldGripperIsExcludedFromTheSolveAsDocumented:
    """The GRASP PRESERVATION contract covers the solve, not only the servo."""

    def test_a_tcp_on_the_jaw_is_not_reached_by_opening_the_jaw(self, gripper_tcp_sim):
        """A target only the jaw's 0.25 m of travel can reach is refused.

        The jaw is held at its live position for the whole descent, so a solve
        that spends jaw travel is a solve the servo will not carry out.
        """
        model = gripper_tcp_sim._world._model
        assert discover_ee_frame(model, "grip/") == ("grip/grasp_site", "site")
        # 0.05 m past the arm's full extension, inside the jaw's travel.
        result = gripper_tcp_sim.move_to(robot_name="grip", position=[0.37, 0.0, 0.05], tol=_TOL, max_steps=_MAX_STEPS)
        assert result["status"] == "error"
        assert "unreachable" in _text(result)
        assert _json_block(result)["uncommanded_joints_moved"] == ["jaw"]


class TestNothingChangesWhereEveryRelevantJointIsCommanded:
    """Controls: the fully actuated fixed-base arm every other test uses."""

    def test_a_reachable_target_is_still_reached(self, fixed_sim):
        """The restriction is a no-op when the arm commands every relevant DOF.

        ``_REACH_TOL`` is the tolerance the shared suite reaches this arm at
        (``test_motion_primitives.py``), so the premise is the suite's, not a
        value tuned here.
        """
        result = fixed_sim.move_to(robot_name="arm", position=REACHABLE, tol=_REACH_TOL, max_steps=_MAX_STEPS)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["ik_residual_m"] <= _REACH_TOL

    def test_an_out_of_workspace_target_keeps_its_existing_advice(self, fixed_sim):
        """No uncommanded DOF would help, so the refusal reads as it always did."""
        result = fixed_sim.move_to(robot_name="arm", position=UNREACHABLE, tol=_TOL, max_steps=_MAX_STEPS)
        assert result["status"] == "error"
        assert "unreachable" in _text(result)
        assert "Choose a closer target or loosen tol." in _text(result)


class TestTheBridgeHonoursItsCommandedDofMask:
    """The solver-level contract the primitive relies on."""

    @staticmethod
    def _bridge_pair(model: Any, commanded: list[int]) -> tuple[Any, Any]:
        """An unrestricted bridge and one restricted to ``commanded``."""
        free = MinkIKBridge(model, "rover/attachment_site", "site", orientation_cost=0.0, max_iters=200)
        restricted = MinkIKBridge(
            model,
            "rover/attachment_site",
            "site",
            orientation_cost=0.0,
            max_iters=200,
            commanded_dofs=commanded,
        )
        return free, restricted

    def test_uncommanded_dofs_hold_their_seed_value_exactly(self, mobile_sim):
        """Every DOF outside the mask comes back bit-identical to the seed.

        Exactness is the point: a solution the caller can only partially apply
        is a solution whose residual means nothing.
        """
        model = mobile_sim._world._model
        shoulder = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rover/shoulder")
        elbow = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rover/elbow")
        commanded_qpos = [int(model.jnt_qposadr[shoulder]), int(model.jnt_qposadr[elbow])]
        free, restricted = self._bridge_pair(model, [int(model.jnt_dofadr[shoulder]), int(model.jnt_dofadr[elbow])])

        q0 = np.array(mobile_sim._world._data.qpos, dtype=float, copy=True)
        target_pose = np.eye(4)
        target_pose[:3, 3] = NEEDS_BASE_MOTION
        target_pose[:3, :3] = free.ee_pose(q0)[:3, :3]

        q_free = free.solve(target_pose, q0)
        q_restricted = restricted.solve(target_pose, q0)

        base = [i for i in range(int(model.nq)) if i not in commanded_qpos]
        assert np.linalg.norm(q_free[base] - q0[base]) > 0.1, "the unrestricted solve borrows the base"
        assert list(q_restricted[base]) == list(q0[base])

    @pytest.mark.parametrize(
        ("commanded", "expected"),
        [
            ([], "empty"),
            ([0, 99], r"outside range\(model\.nv\)"),
            ([-1], r"outside range\(model\.nv\)"),
            ([True], "integer velocity-space indices"),
            ([1.5], "integer velocity-space indices"),
        ],
    )
    def test_a_mask_that_cannot_describe_this_model_is_refused(self, mobile_sim, commanded, expected):
        """A mask built against another model, or one that frees nothing, raises.

        ``True`` is refused explicitly: ``bool`` is an ``int`` subclass that
        would silently act as DOF index 1.
        """
        with pytest.raises(ValueError, match=expected):
            MinkIKBridge(
                mobile_sim._world._model,
                "rover/attachment_site",
                "site",
                commanded_dofs=commanded,
            )
