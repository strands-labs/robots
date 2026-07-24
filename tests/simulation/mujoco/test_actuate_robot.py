"""Regression tests for ``actuate_robot`` and ``zero_dynamics`` (GH #1533, PR 1).

``actuate_robot`` is the supported form of the ``so101_curobo`` example's
private-spec surgery (``sim._world._backend_state["spec"]`` + hand-added
position actuators): it converts an actuator-less URDF arm into a
position-servo arm so ``send_action`` / ``run_policy`` can drive it.
Contracts pinned here:

* an actuator-less URDF arm gains one position actuator per hinge/slide
  joint and tracks ``send_action`` targets,
* ``ctrl`` is initialized to the CURRENT pose (no snap to zero),
* the integrator flips to ``implicitfast`` and the change lives on the spec
  (survives later scene recompiles),
* a ``{joint: kp}`` dict actuates only the listed joints; unknown joints and
  non-positive gains are structured errors,
* double actuation is refused,
* ``zero_dynamics`` clears qvel/qacc world-wide or robot-scoped only.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

MINI_ARM_URDF = """<?xml version="1.0"?>
<robot name="mini_arm">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual>
    <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><mass value="0.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></visual>
    <collision><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></collision>
  </link>
  <link name="link2">
    <inertial><mass value="0.25"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><origin xyz="0 0 0.075"/><geometry><box size="0.02 0.02 0.15"/></geometry></visual>
    <collision><origin xyz="0 0 0.075"/><geometry><box size="0.02 0.02 0.15"/></geometry></collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
</robot>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_actuate_robot_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def arm_sim(sim, tmp_path):
    """A world with an actuator-less 2-joint URDF arm named 'arm'."""
    urdf = tmp_path / "mini_arm.urdf"
    urdf.write_text(MINI_ARM_URDF)
    result = sim.add_robot(name="arm", urdf_path=str(urdf))
    assert result["status"] == "success", result
    assert sim._world._model.nu == 0, "URDF arm must load actuator-less for this test"
    return sim


def _joint_qpos(sim, name):
    m, d = sim._world._model, sim._world._data
    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint {name!r} not found"
    return float(d.qpos[m.jnt_qposadr[jid]])


# A floating-base robot exercising the MULTI-DOF joint path that a hinge-only
# arm never reaches: a free root joint (6 DOFs), a ball joint (3 DOFs), and a
# single hinge (1 DOF). Used to pin how actuate_robot/zero_dynamics treat
# non-hinge/slide joints (humanoids, quadrupeds, mobile bases).
FLOATER_MJCF = """<mujoco model="floater">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
      <body name="head" pos="0 0 0.1">
        <joint name="neck" type="ball"/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.3"/>
        <body name="arm" pos="0 0 0.06">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.02 0.02 0.05" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Same base, but with NO hinge/slide joint at all (free + ball only).
FLOATER_NO_HINGE_MJCF = """<mujoco model="floater_no_hinge">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
      <body name="head" pos="0 0 0.1">
        <joint name="neck" type="ball"/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _add_floating_base(sim, tmp_path, mjcf=FLOATER_MJCF, name="floater"):
    """Add a floating-base robot from an MJCF string and return its handle."""
    model_file = tmp_path / f"{name}.xml"
    model_file.write_text(mjcf)
    result = sim.add_robot(name="fl", urdf_path=str(model_file))
    assert result["status"] == "success", result
    return sim._world.robots["fl"]


class TestActuateRobot:
    def test_send_action_drives_urdf_arm_after_actuation(self, arm_sim):
        """The headline contract: an actuator-less URDF arm becomes drivable."""
        result = arm_sim.actuate_robot("arm", kp=200.0)
        assert result["status"] == "success", result
        assert arm_sim._world._model.nu == 2

        arm_sim.send_action({"shoulder": 0.5, "elbow": -0.4}, robot_name="arm", n_substeps=500)
        assert _joint_qpos(arm_sim, "arm/shoulder") == pytest.approx(0.5, abs=0.05)
        assert _joint_qpos(arm_sim, "arm/elbow") == pytest.approx(-0.4, abs=0.05)

    def test_ctrl_initialized_to_current_pose(self, arm_sim):
        """The arm holds its pose after actuation instead of snapping to zero."""
        m, d = arm_sim._world._model, arm_sim._world._data
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "arm/shoulder")
        d.qpos[m.jnt_qposadr[jid]] = 0.7
        mj.mj_forward(m, d)

        assert arm_sim.actuate_robot("arm", kp=300.0)["status"] == "success"
        arm_sim.step(500)
        assert _joint_qpos(arm_sim, "arm/shoulder") == pytest.approx(0.7, abs=0.05), (
            "actuated arm must hold its pre-actuation pose, not snap to 0"
        )

    def test_integrator_switched_and_survives_recompile(self, arm_sim):
        assert arm_sim.actuate_robot("arm")["status"] == "success"
        assert arm_sim._world._model.opt.integrator == int(mj.mjtIntegrator.mjINT_IMPLICITFAST)
        # Actuation lives on the spec: a later scene recompile keeps nu + integrator.
        assert arm_sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.3, 0, 0.5])["status"] == (
            "success"
        )
        assert arm_sim._world._model.nu == 2
        assert arm_sim._world._model.opt.integrator == int(mj.mjtIntegrator.mjINT_IMPLICITFAST)

    def test_kp_dict_actuates_subset_only(self, arm_sim):
        result = arm_sim.actuate_robot("arm", kp={"shoulder": 250.0})
        assert result["status"] == "success", result
        m = arm_sim._world._model
        assert m.nu == 1
        assert mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, 0) == "arm_act_shoulder"

    def test_kp_dict_unknown_joint_error(self, arm_sim):
        result = arm_sim.actuate_robot("arm", kp={"wrist_flex": 100.0})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "wrist_flex" in text
        assert "shoulder" in text, "the error must name the valid joints"
        assert arm_sim._world._model.nu == 0, "a rejected call must not mutate the scene"

    def test_kp_value_validation(self, arm_sim):
        for bad in (0.0, -5.0, float("inf")):
            result = arm_sim.actuate_robot("arm", kp=bad)
            assert result["status"] == "error", f"kp={bad!r} must be rejected"
        result = arm_sim.actuate_robot("arm", kp={"shoulder": -1.0})
        assert result["status"] == "error"
        assert arm_sim._world._model.nu == 0

    def test_kp_empty_dict_rejected(self, arm_sim):
        """An empty ``{joint: kp}`` dict actuates nothing - a structured error,
        not a silent no-op that reports success while leaving nu unchanged."""
        result = arm_sim.actuate_robot("arm", kp={})
        assert result["status"] == "error"
        assert "empty" in result["content"][0]["text"]
        assert arm_sim._world._model.nu == 0, "a rejected call must not mutate the scene"

    def test_damping_validation(self, arm_sim):
        """``damping`` must be a finite number >= 0; NaN, negative, and
        non-numeric values are rejected before any spec surgery."""
        for bad in (float("nan"), -1.0, "stiff"):
            result = arm_sim.actuate_robot("arm", damping=bad)
            assert result["status"] == "error", f"damping={bad!r} must be rejected"
            assert "damping" in result["content"][0]["text"]
        assert arm_sim._world._model.nu == 0, "a rejected call must not mutate the scene"

    def test_armature_validation(self, arm_sim):
        """``armature`` must be a finite number >= 0; NaN, negative, and
        non-numeric values are rejected before any spec surgery."""
        for bad in (float("nan"), -0.5, None):
            result = arm_sim.actuate_robot("arm", armature=bad)
            assert result["status"] == "error", f"armature={bad!r} must be rejected"
            assert "armature" in result["content"][0]["text"]
        assert arm_sim._world._model.nu == 0, "a rejected call must not mutate the scene"

    def test_double_actuation_refused(self, arm_sim):
        assert arm_sim.actuate_robot("arm")["status"] == "success"
        result = arm_sim.actuate_robot("arm")
        assert result["status"] == "error"
        assert "already has actuators" in result["content"][0]["text"]
        assert arm_sim._world._model.nu == 2, "the refused call must not add duplicates"

    def test_unknown_robot_error(self, sim):
        result = sim.actuate_robot("ghost")
        assert result["status"] == "error"

    def test_disable_self_collision_zeroes_robot_geom_masks(self, arm_sim):
        assert arm_sim.actuate_robot("arm", disable_self_collision=True)["status"] == "success"
        m = arm_sim._world._model
        robot_geoms = 0
        for gid in range(m.ngeom):
            body_name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
            if body_name.startswith("arm/"):
                robot_geoms += 1
                assert int(m.geom_contype[gid]) == 0
                assert int(m.geom_conaffinity[gid]) == 0
        assert robot_geoms > 0, "the arm must contribute geoms to the check"

    def test_dispatch_via_action_router(self, arm_sim):
        result = arm_sim._dispatch_action("actuate_robot", {"robot_name": "arm", "kp": 150.0})
        assert result["status"] == "success", result
        assert arm_sim._world._model.nu == 2

    def test_skips_non_hinge_slide_joints_on_floating_base(self, sim, tmp_path):
        # A free/ball joint carries no position-servo semantics, so a floating
        # base is actuated only on its hinge/slide joints; the free + ball
        # joints are silently skipped (not errored, not actuated).
        _add_floating_base(sim, tmp_path)
        result = sim.actuate_robot(robot_name="fl", kp=100.0)
        assert result["status"] == "success", result
        assert sim._world._model.nu == 1, "only the single hinge must gain an actuator"
        assert "['elbow']" in result["content"][0]["text"]

    def test_no_hinge_slide_joints_is_actionable_error(self, sim, tmp_path):
        # A robot with only free/ball joints has nothing to position-servo:
        # a structured error, not a spec recompile with zero actuators.
        _add_floating_base(sim, tmp_path, mjcf=FLOATER_NO_HINGE_MJCF)
        result = sim.actuate_robot(robot_name="fl", kp=100.0)
        assert result["status"] == "error"
        assert "no hinge/slide joints" in result["content"][0]["text"]
        assert sim._world._model.nu == 0, "a rejected actuate_robot must add no actuators"


class TestZeroDynamics:
    def test_zeroes_all_dofs(self, sim):
        sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0, 0, 0.6])
        sim.step(40)  # build up fall velocity
        m, d = sim._world._model, sim._world._data
        assert any(abs(float(v)) > 0.1 for v in d.qvel), "cube should be falling"

        result = sim.zero_dynamics()
        assert result["status"] == "success", result
        assert [float(v) for v in d.qvel] == pytest.approx([0.0] * m.nv, abs=1e-12)
        assert [float(v) for v in d.qacc_warmstart] == pytest.approx([0.0] * m.nv, abs=1e-12)

    def test_robot_scope_leaves_object_momentum(self, sim, tmp_path):
        urdf = tmp_path / "mini_arm.urdf"
        urdf.write_text(MINI_ARM_URDF)
        assert sim.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"
        sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.5, 0, 0.6])
        m, d = sim._world._model, sim._world._data
        # Give the arm's shoulder a spin and let the cube fall.
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "arm/shoulder")
        d.qvel[m.jnt_dofadr[jid]] = 2.0
        sim.step(30)
        m, d = sim._world._model, sim._world._data
        cube_jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_adr = int(m.jnt_dofadr[cube_jid])
        cube_vz_before = float(d.qvel[cube_adr + 2])
        assert abs(cube_vz_before) > 0.1, "cube should be falling"

        result = sim.zero_dynamics(robot_name="arm")
        assert result["status"] == "success", result
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "arm/shoulder")
        assert float(d.qvel[m.jnt_dofadr[jid]]) == 0.0
        assert float(d.qvel[cube_adr + 2]) == pytest.approx(cube_vz_before, abs=1e-9), (
            "robot-scoped zero_dynamics must not touch the cube's freejoint"
        )

    def test_unknown_robot_error(self, sim):
        result = sim.zero_dynamics(robot_name="ghost")
        assert result["status"] == "error"

    def test_dispatch_via_action_router(self, sim):
        sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0, 0, 0.6])
        sim.step(40)
        result = sim._dispatch_action("zero_dynamics", {})
        assert result["status"] == "success", result

    def test_zeroes_all_dofs_of_free_and_ball_joints(self, sim, tmp_path):
        # zero_dynamics must clear EVERY DOF of a multi-DOF joint, not just the
        # first: a free joint spans 6 DOFs and a ball joint 3. A width-1
        # regression would leave a floating base spinning/translating on 5 of
        # its 6 DOFs after the "anti-explosion" reset.
        robot = _add_floating_base(sim, tmp_path)
        m, d = sim._world._model, sim._world._data

        # Seed a nonzero velocity/acceleration on every DOF the robot owns.
        d.qvel[:] = 1.5
        d.qacc[:] = 2.0

        result = sim.zero_dynamics(robot_name="fl")
        assert result["status"] == "success", result
        # 6 (free) + 3 (ball) + 1 (hinge) = 10 DOFs reported zeroed.
        assert "10 DOFs" in result["content"][0]["text"]

        for jid in robot.joint_ids:
            jtype = int(m.jnt_type[jid])
            width = {int(mj.mjtJoint.mjJNT_FREE): 6, int(mj.mjtJoint.mjJNT_BALL): 3}.get(jtype, 1)
            adr = int(m.jnt_dofadr[jid])
            # qvel and qacc_warmstart are cleared and stay cleared; qacc is
            # re-derived by the trailing mj_forward, so it is not asserted zero
            # (it reflects gravity), matching test_zeroes_all_dofs.
            assert [float(v) for v in d.qvel[adr : adr + width]] == pytest.approx([0.0] * width, abs=1e-12)
            assert [float(v) for v in d.qacc_warmstart[adr : adr + width]] == pytest.approx([0.0] * width, abs=1e-12)
