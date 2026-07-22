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
