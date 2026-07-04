"""Tests for PhysicsMixin - advanced MuJoCo physics features.

Tests: raycasting, jacobians, energy, forces, state checkpointing,
inverse dynamics, sensor readout, body introspection, runtime modification.

Run: uv run pytest tests/test_physics.py -v
"""

import json
import os

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import _full_mass_matrix  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ROBOT_XML = """
<mujoco model="physics_test">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="box1" pos="0 0 0.5">
      <freejoint name="box_free"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
    <body name="arm_base" pos="0.5 0 0">
      <body name="link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_geom" type="capsule" size="0.02 0.1" rgba="0.3 0.3 0.8 1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_geom" type="capsule" size="0.015 0.08" rgba="0.3 0.8 0.3 1"/>
          <site name="end_effector" pos="0 0 0.08"/>
        </body>
      </body>
    </body>
    <camera name="overhead" pos="0 -1 1.5" quat="0.7 0.7 0 0"/>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
    <motor name="elbow_motor" joint="elbow" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
    <jointpos name="elbow_pos" joint="elbow"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def sim():
    """Create a Simulation with the test scene loaded directly.

    Builds a live ``MjSpec`` from the fixture XML so the world satisfies
    the backend contract (every SimWorld has ``_backend_state["spec"]``).
    This is the same contract produced by ``load_scene`` /
    ``_compile_world`` / ``replace_scene_mjcf``.
    """
    from strands_robots.simulation.models import SimStatus, SimWorld

    s = Simulation(tool_name="test_sim", mesh=False)
    s._world = SimWorld()
    spec = mj.MjSpec.from_string(ROBOT_XML)
    s._world._backend_state["spec"] = spec
    s._world._model = spec.compile()
    s._world._data = mj.MjData(s._world._model)
    s._world.status = SimStatus.IDLE
    mj.mj_forward(s._world._model, s._world._data)
    yield s
    s.cleanup()


def _extract_json_block(result, idx=1):
    """Schema-tolerant: accepts both {"json": {...}} (new) and {"text": <json_str>} (legacy).

    The content-block schema is in flux; this helper ensures tests work against either.
    """
    block = result["content"][idx]
    if "json" in block:
        return block["json"]
    return json.loads(block["text"])


class TestRaycasting:
    def test_raycast_hits_ground(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1])
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert data["hit"] is True
        assert data["distance"] is not None
        assert data["distance"] > 0

    def test_raycast_hits_box(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1])
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert data["hit"] is True
        assert data["geom_name"] in ("box_geom", "ground")

    def test_raycast_misses(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, 1])  # shooting up
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert data["hit"] is False

    def test_multi_raycast(self, sim):
        dirs = [[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = sim.multi_raycast(origin=[0, 0, 2], directions=dirs)
        assert result["status"] == "success"
        rays = _extract_json_block(result, 1)["rays"]
        assert len(rays) == 4
        # At least the downward ray should hit
        assert rays[0]["distance"] is not None


class TestJacobians:
    def test_body_jacobian(self, sim):
        result = sim.get_jacobian(body_name="link2")
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert len(data["jacp"]) == 3  # 3×nv
        assert data["nv"] == sim._world._model.nv

    def test_site_jacobian(self, sim):
        result = sim.get_jacobian(site_name="end_effector")
        assert result["status"] == "success"

    def test_geom_jacobian(self, sim):
        result = sim.get_jacobian(geom_name="link2_geom")
        assert result["status"] == "success"

    def test_jacobian_no_target(self, sim):
        result = sim.get_jacobian()
        assert result["status"] == "error"

    def test_jacobian_invalid_body(self, sim):
        result = sim.get_jacobian(body_name="nonexistent")
        assert result["status"] == "error"

    def test_jacobian_reflects_current_configuration(self, sim):
        """get_jacobian must be the Jacobian of the CURRENT qpos.

        Regression: it read data.xpos/site_xpos/subtree_com/cdof left by an
        earlier forward and never re-ran the position pipeline, so after a
        qpos change that did not itself forward (here a direct data.qpos
        write) it returned the OLD configuration's Jacobian while reporting
        success.
        """
        model, data = sim._world._model, sim._world._data
        j_rest = np.array(_extract_json_block(sim.get_jacobian(site_name="end_effector"), 1)["jacp"])

        sh = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")]
        el = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "elbow")]
        data.qpos[sh] = 0.9
        data.qpos[el] = -0.7
        j_after = np.array(_extract_json_block(sim.get_jacobian(site_name="end_effector"), 1)["jacp"])

        # Independent ground truth for the new configuration.
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "end_effector")
        mj.mj_jacSite(model, data, jacp, jacr, sid)

        # fails-before: j_after equalled the stale j_rest, not the new-config truth.
        assert np.linalg.norm(j_after - jacp) < 1e-9
        assert np.linalg.norm(j_after - j_rest) > 1e-3


class TestEnergy:
    def test_get_energy(self, sim):
        result = sim.get_energy()
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert "potential" in data
        assert "kinetic" in data
        assert "total" in data
        # Box at height 0.5 should have nonzero potential energy
        assert data["potential"] != 0 or data["kinetic"] != 0

    def test_energy_changes_after_step(self, sim):
        e1 = _extract_json_block(sim.get_energy(), 1)
        # Step physics to let box fall
        for _ in range(100):
            mj.mj_step(sim._world._model, sim._world._data)
        e2 = _extract_json_block(sim.get_energy(), 1)
        # Kinetic energy should change (box falls)
        assert e1["kinetic"] != e2["kinetic"] or e1["potential"] != e2["potential"]


class TestExternalForces:
    def test_apply_force(self, sim):
        result = sim.apply_force(body_name="box1", force=[0, 0, 100])
        assert result["status"] == "success"
        assert "box1" in result["content"][0]["text"]

    def test_apply_force_invalid_body(self, sim):
        result = sim.apply_force(body_name="nonexistent", force=[0, 0, 10])
        assert result["status"] == "error"

    def test_force_changes_acceleration(self, sim):
        # Get initial state
        data = sim._world._data
        old_qfrc = data.qfrc_applied.copy()
        sim.apply_force(body_name="box1", force=[0, 0, 100])
        # qfrc_applied should change
        assert not np.array_equal(old_qfrc, data.qfrc_applied)


class TestMassMatrix:
    def test_get_mass_matrix(self, sim):
        result = sim.get_mass_matrix()
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        nv = sim._world._model.nv
        assert data["shape"] == [nv, nv]
        assert data["rank"] > 0
        assert data["total_mass"] > 0

    def test_mass_diagonal_positive(self, sim):
        result = sim.get_mass_matrix()
        diag = _extract_json_block(result, 1)["diagonal"]
        assert all(d >= 0 for d in diag)

    def test_mass_matrix_is_symmetric_positive_definite(self, sim):
        # M(q) is symmetric PD for any well-formed model; verifying the actual
        # numbers (not just the shape) guards against a signature fix that
        # silently returns a wrong/zero matrix.
        result = sim.get_mass_matrix()
        data = _extract_json_block(result, 1)
        nv = data["shape"][0]
        M = _full_mass_matrix(mj, sim._world._model, sim._world._data)
        assert M.shape == (nv, nv)
        assert np.allclose(M, M.T), "mass matrix must be symmetric"
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"mass matrix must be PD, got eigvals {eigvals}"


class TestFullMassMatrixSignatureDrift:
    """Regression: ``mj_fullM`` changed its binding signature across MuJoCo
    releases. ``_full_mass_matrix`` must work against every variant rather than
    hard-coding one call order (which crashed the suite under newer MuJoCo).
    """

    def test_helper_matches_native_call(self, sim):
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        M = _full_mass_matrix(mj, model, data)
        assert M.flags["C_CONTIGUOUS"]
        assert M.dtype == np.float64
        # Cross-check against the diagonal MuJoCo reports for this model.
        assert M.shape == (model.nv, model.nv)
        assert np.all(np.diag(M) > 0)

    def test_helper_falls_back_to_legacy_signatures(self, sim):
        # Simulate an older MuJoCo binding whose mj_fullM rejects the modern
        # (model, data, dst) order and expects (model, dst, qM). The helper
        # must transparently fall back and still produce the correct matrix.
        #
        # The legacy emulation must not delegate to the installed mj_fullM in a
        # fixed argument order: the installed binding may itself be the legacy
        # one (mujoco < 3.10), so a hard-coded modern call would raise and make
        # this drift test non-portable across the very signatures it covers.
        # Instead it fills dst from the precomputed reference, asserting only
        # the legacy (model, dst, qM) call contract.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        reference = _full_mass_matrix(mj, model, data)

        class _LegacyShim:
            """Proxy mujoco module exposing only a legacy mj_fullM."""

            def __getattr__(self, attr):
                return getattr(mj, attr)

            @staticmethod
            def mj_fullM(m, a, b):
                # Reject the modern call where the 3rd arg is the dst buffer
                # (i.e. the 2nd arg is MjData), forcing the legacy path.
                import mujoco as _mj

                if isinstance(a, _mj.MjData):
                    raise TypeError("legacy binding: expected (model, dst, qM)")
                # Legacy contract: a is the dense dst buffer, b is the sparse
                # inertia qM (1D or [m, 1]). Validate that contract, then fill
                # dst from the known-correct reference (version-independent, so
                # the emulation works whatever signature the installed mujoco
                # binding actually uses).
                assert isinstance(a, np.ndarray) and a.flags["WRITEABLE"]
                qm = np.asarray(b).reshape(-1)
                assert qm.shape[0] == data.qM.shape[0]
                a[...] = reference

        shim = _LegacyShim()
        M = _full_mass_matrix(shim, model, data)
        assert np.allclose(M, reference)

    def test_helper_returns_empty_for_zero_dof(self):
        # A model with no DoFs must return a well-typed (0, 0) array, never
        # crash in numpy on the empty buffer.
        model = mj.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
        )
        mdata = mj.MjData(model)
        mj.mj_forward(model, mdata)
        assert model.nv == 0
        M = _full_mass_matrix(mj, model, mdata)
        assert M.shape == (0, 0)


class TestStateCheckpointing:
    def test_save_and_load_state(self, sim):
        # Set a known joint position
        sim._world._data.qpos[7] = 1.0  # shoulder
        mj.mj_forward(sim._world._model, sim._world._data)

        # Save
        result = sim.save_state(name="test_checkpoint")
        assert result["status"] == "success"

        # Change state
        sim._world._data.qpos[7] = -1.0
        mj.mj_forward(sim._world._model, sim._world._data)
        assert sim._world._data.qpos[7] == pytest.approx(-1.0)

        # Restore
        result = sim.load_state(name="test_checkpoint")
        assert result["status"] == "success"
        assert sim._world._data.qpos[7] == pytest.approx(1.0)

    def test_load_nonexistent_checkpoint(self, sim):
        result = sim.load_state(name="doesnt_exist")
        assert result["status"] == "error"


class TestInverseDynamics:
    @staticmethod
    def _gravity_compensation(model, data):
        """Ground-truth compensation torques: mj_inverse for zero desired qacc."""
        mj.mj_forward(model, data)
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        return {
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i): float(data.qfrc_inverse[model.jnt_dofadr[i]])
            for i in range(model.njnt)
            if mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        }

    def test_inverse_dynamics(self, sim):
        result = sim.inverse_dynamics()
        assert result["status"] == "success"
        forces = _extract_json_block(result, 1)["qfrc_inverse"]
        assert "shoulder" in forces or "elbow" in forces

    def test_inverse_dynamics_returns_gravity_compensation(self, sim):
        """inverse_dynamics reports the torques that HOLD the current pose.

        Regression: previously it read the stale forward-dynamics ``qacc``
        (the unforced/free-fall acceleration) as the desired acceleration and
        asked ``mj_inverse`` to reproduce free-fall - which needs ~0 force. It
        therefore reported near-zero torques regardless of pose instead of the
        gravity-/bias-compensation torques the query is for.
        """
        model, data = sim._world._model, sim._world._data
        # A gravity-loaded pose (arm tilted away from the vertical).
        sim.set_joint_positions({"shoulder": 0.8, "elbow": -0.5})

        forces = _extract_json_block(sim.inverse_dynamics(), 1)["qfrc_inverse"]

        # Ground truth computed independently AFTER the call (the fixed method
        # forwards + restores qacc, so state is unchanged for this compare).
        expected = self._gravity_compensation(model, data)
        for jn in ("shoulder", "elbow"):
            assert forces[jn] == pytest.approx(expected[jn], abs=1e-9)

        # The shoulder carries a real gravity load in this pose; the buggy
        # free-fall path returned ~0 here, so this discriminates the fix.
        assert abs(forces["shoulder"]) > 1e-2

    def test_inverse_dynamics_ignores_stale_qacc(self, sim):
        """The result must not depend on leftover free-fall qacc.

        Stepping (or a prior forward) leaves ``data.qacc`` holding the
        forward-dynamics acceleration. inverse_dynamics must zero it for the
        solve, so back-to-back calls are identical and independent of that
        buffer.
        """
        sim.set_joint_positions({"shoulder": 0.6, "elbow": 0.4})
        first = _extract_json_block(sim.inverse_dynamics(), 1)["qfrc_inverse"]

        # Perturb the leftover qacc buffer directly; the answer must not move.
        sim._world._data.qacc[:] = 123.4
        second = _extract_json_block(sim.inverse_dynamics(), 1)["qfrc_inverse"]

        for jn in ("shoulder", "elbow"):
            assert first[jn] == pytest.approx(second[jn], abs=1e-9)
        assert abs(first["shoulder"]) > 1e-2


class TestBodyState:
    def test_get_body_state(self, sim):
        result = sim.get_body_state(body_name="box1")
        assert result["status"] == "success"
        state = _extract_json_block(result, 1)
        assert "position" in state
        assert "quaternion" in state
        assert "linear_velocity" in state
        assert "angular_velocity" in state
        assert "mass" in state
        assert len(state["position"]) == 3
        assert len(state["quaternion"]) == 4
        assert state["mass"] == pytest.approx(1.0)

    def test_body_state_invalid(self, sim):
        result = sim.get_body_state(body_name="nonexistent")
        assert result["status"] == "error"

    def test_body_state_pose_reflects_current_qpos(self, sim):
        """get_body_state pose must reflect the current qpos and agree with
        forward_kinematics.

        Regression: it read stale data.xpos without forwarding, so after a
        qpos change (here a direct data.qpos write) it reported the OLD pose
        while its sibling forward_kinematics reported the new one.
        """
        model, data = sim._world._model, sim._world._data
        sh = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")]
        data.qpos[sh] = 1.0

        pos_bs = _extract_json_block(sim.get_body_state(body_name="link2"), 1)["position"]
        pos_fk = _extract_json_block(sim.forward_kinematics(body_name="link2"), 1)["position"]
        # fails-before: get_body_state stale pose != forward_kinematics fresh pose.
        assert pos_bs == pytest.approx(pos_fk, abs=1e-9)

    def test_body_state_velocity_reflects_current_qvel(self, sim):
        """get_body_state 6D velocity must reflect the current qvel.

        Regression: it read data.cvel via mj_objectVelocity without
        forwarding, so a velocity written by set_joint_velocities (which sets
        qvel but does not forward) was reported as the stale ~zero velocity
        while the call reported success.
        """
        sim.set_joint_velocities(velocities={"shoulder": 2.0, "elbow": -1.5})
        state = _extract_json_block(sim.get_body_state(body_name="link2"), 1)
        got = np.array(state["linear_velocity"] + state["angular_velocity"])

        # Independent ground truth (order matches get_body_state: linear then angular).
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        vel = np.zeros(6)
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, bid, vel, 0)
        truth = np.concatenate([vel[3:], vel[:3]])

        assert np.linalg.norm(got - truth) < 1e-9
        # fails-before: stale velocity was ~0 at the freshly-set qvel.
        assert np.linalg.norm(got) > 1e-2


class TestDirectJointControl:
    def test_set_joint_positions(self, sim):
        result = sim.set_joint_positions(positions={"shoulder": 0.5, "elbow": -0.3})
        assert result["status"] == "success"
        assert "2/2" in result["content"][0]["text"]

        # Verify positions were set
        model, data = sim._world._model, sim._world._data
        shoulder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
        qpos_adr = model.jnt_qposadr[shoulder_id]
        assert data.qpos[qpos_adr] == pytest.approx(0.5)

    def test_set_joint_velocities(self, sim):
        result = sim.set_joint_velocities(velocities={"shoulder": 1.0})
        assert result["status"] == "success"


class TestSensors:
    def test_get_all_sensors(self, sim):
        result = sim.get_sensor_data()
        assert result["status"] == "success"
        sensors = _extract_json_block(result, 1)["sensors"]
        assert "shoulder_pos" in sensors
        assert "elbow_pos" in sensors

    def test_get_specific_sensor(self, sim):
        result = sim.get_sensor_data(sensor_name="shoulder_pos")
        assert result["status"] == "success"
        sensors = _extract_json_block(result, 1)["sensors"]
        assert len(sensors) == 1
        assert "shoulder_pos" in sensors

    def test_sensor_values_change(self, sim):
        # Set shoulder position
        sim.set_joint_positions(positions={"shoulder": 1.0})
        result = sim.get_sensor_data(sensor_name="shoulder_pos")
        val = _extract_json_block(result, 1)["sensors"]["shoulder_pos"]["values"]
        assert abs(val - 1.0) < 0.01


class TestRuntimeModification:
    def test_set_body_mass(self, sim):
        result = sim.set_body_properties(body_name="box1", mass=5.0)
        assert result["status"] == "success"
        body_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "box1")
        assert sim._world._model.body_mass[body_id] == pytest.approx(5.0)

    def test_set_body_mass_scales_inertia_with_mass(self, sim):
        """Setting mass scales the inertia tensor by the same ratio.

        A rigid body's inertia tracks its mass at fixed geometry (a uniform
        density change scales I = integral of r^2 dm by the same factor).
        Updating body_mass alone leaves the body physically inconsistent -
        heavy in translation but with the old rotational resistance - and the
        caller has no way to fix it (mass is the only settable property).
        """
        model = sim._world._model
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "box1")
        old_mass = float(model.body_mass[body_id])
        old_inertia = model.body_inertia[body_id].copy()
        assert old_mass > 0 and (old_inertia > 0).all()

        # Scale up by 5x.
        assert sim.set_body_properties(body_name="box1", mass=5.0)["status"] == "success"
        assert model.body_mass[body_id] == pytest.approx(5.0)
        assert model.body_inertia[body_id] == pytest.approx(old_inertia * (5.0 / old_mass))

        # And down again (0.5 kg): inertia tracks the new ratio, not stale.
        cur_mass = float(model.body_mass[body_id])
        cur_inertia = model.body_inertia[body_id].copy()
        assert sim.set_body_properties(body_name="box1", mass=0.5)["status"] == "success"
        assert model.body_inertia[body_id] == pytest.approx(cur_inertia * (0.5 / cur_mass))

    def test_set_body_mass_massless_body_no_crash(self, sim):
        """A massless frame (mass 0, inertia 0) is handled without dividing by zero.

        There is no geometry-derived inertia to scale from a zero prior mass, so
        the inertia stays zero; the mass update still succeeds.
        """
        model = sim._world._model
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm_base")
        assert float(model.body_mass[body_id]) == pytest.approx(0.0)
        result = sim.set_body_properties(body_name="arm_base", mass=2.0)
        assert result["status"] == "success"
        assert model.body_mass[body_id] == pytest.approx(2.0)
        assert model.body_inertia[body_id] == pytest.approx([0.0, 0.0, 0.0])

    def test_set_geom_color(self, sim):
        result = sim.set_geom_properties(geom_name="box_geom", color=[0, 1, 0, 1])
        assert result["status"] == "success"
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        assert sim._world._model.geom_rgba[geom_id][1] == pytest.approx(1.0)

    def test_set_geom_friction(self, sim):
        result = sim.set_geom_properties(geom_name="box_geom", friction=[0.5, 0.01, 0.001])
        assert result["status"] == "success"

    def test_invalid_geom(self, sim):
        result = sim.set_geom_properties(geom_name="nonexistent", color=[1, 0, 0, 1])
        assert result["status"] == "error"

    def test_set_geom_size_resizes_geom(self, sim):
        """set_geom_properties(size=...) writes the new half-extents into the
        live model so the next step / render sees the resized geom (no
        recompile). Only the leading ``min(len(size), 3)`` entries are set,
        matching MuJoCo's per-type geom_size layout."""
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        result = sim.set_geom_properties(geom_name="box_geom", size=[0.25, 0.3, 0.35])
        assert result["status"] == "success"
        assert "size" in result["content"][0]["text"]
        new_size = sim._world._model.geom_size[geom_id]
        assert new_size[0] == pytest.approx(0.25)
        assert new_size[1] == pytest.approx(0.3)
        assert new_size[2] == pytest.approx(0.35)

    def test_set_geom_size_shorter_than_three_leaves_tail_untouched(self, sim):
        """A partial size list updates only the entries provided and leaves the
        remaining half-extents at their compiled value."""
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        original_tail = float(sim._world._model.geom_size[geom_id][2])
        result = sim.set_geom_properties(geom_name="box_geom", size=[0.2])
        assert result["status"] == "success"
        new_size = sim._world._model.geom_size[geom_id]
        assert new_size[0] == pytest.approx(0.2)
        assert float(new_size[2]) == pytest.approx(original_tail)

    def test_set_geom_size_grow_recomputes_rbound_and_aabb(self, sim):
        """Growing a size-defined primitive refreshes its collision bounds.

        ``geom_rbound`` (broadphase) and ``geom_aabb`` (mid-phase) are derived
        from ``geom_size`` at compile time and are not refreshed by the solver.
        A grown geom whose bounds are left stale is silently culled from
        broadphase, so other bodies pass through it. The recompute must bring
        both to the values a fresh compile at the new size would produce.
        """
        model = sim._world._model
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        # box_geom compiles at half-extents 0.1 (rbound ~= 0.1732).
        assert float(model.geom_rbound[gid]) == pytest.approx(np.linalg.norm([0.1, 0.1, 0.1]))

        result = sim.set_geom_properties(geom_name="box_geom", size=[0.25, 0.25, 0.02])
        assert result["status"] == "success"

        expected_half = [0.25, 0.25, 0.02]
        assert float(model.geom_rbound[gid]) == pytest.approx(np.linalg.norm(expected_half))
        assert model.geom_aabb[gid][3:6].tolist() == pytest.approx(expected_half)

    def test_set_geom_size_capsule_recomputes_rbound(self, sim):
        """The recompute uses the correct per-type formula (capsule = r + halflen)."""
        model = sim._world._model
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "link1_geom")
        result = sim.set_geom_properties(geom_name="link1_geom", size=[0.05, 0.2])
        assert result["status"] == "success"
        # capsule rbound = radius + half-length; aabb half = [r, r, r + halflen].
        assert float(model.geom_rbound[gid]) == pytest.approx(0.25)
        assert model.geom_aabb[gid][3:6].tolist() == pytest.approx([0.05, 0.05, 0.25])

    def test_set_geom_size_grow_lets_object_rest_on_it(self, sim):
        """Behavioral: a body rests on a grown static geom instead of falling through.

        A small static platform is grown into a wide table via the public API,
        then a ball offset well beyond the platform's original bounding radius
        is dropped. With stale collision bounds the broadphase culls the pair
        and the ball falls to the floor; after the recompute it lands on the
        grown table.
        """
        from strands_robots.simulation.models import SimStatus, SimWorld

        scene = """
        <mujoco>
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 0.1"/>
            <body name="plat" pos="0 0 0.5">
              <geom name="platg" type="box" size="0.02 0.02 0.02"/>
            </body>
            <body name="ball" pos="0.15 0 0.6">
              <freejoint/>
              <geom name="ballg" type="sphere" size="0.03"/>
            </body>
          </worldbody>
        </mujoco>
        """
        s = Simulation(tool_name="test_grow", mesh=False)
        try:
            s._world = SimWorld()
            spec = mj.MjSpec.from_string(scene)
            s._world._backend_state["spec"] = spec
            s._world._model = spec.compile()
            s._world._data = mj.MjData(s._world._model)
            s._world.status = SimStatus.IDLE
            mj.mj_forward(s._world._model, s._world._data)

            result = s.set_geom_properties(geom_name="platg", size=[0.25, 0.25, 0.02])
            assert result["status"] == "success"

            model, data = s._world._model, s._world._data
            for _ in range(2000):
                mj.mj_step(model, data)
            ball_z = float(data.body("ball").xpos[2])
            # Table top is at z = 0.5 + 0.02 = 0.52; ball (r=0.03) rests ~0.55.
            assert ball_z > 0.5, f"ball fell through the grown table (rest z={ball_z:.4f})"
        finally:
            s.cleanup()


class TestContactForces:
    def test_get_contact_forces_after_settling(self, sim):
        # Let box fall and settle
        for _ in range(500):
            mj.mj_step(sim._world._model, sim._world._data)
        result = sim.get_contact_forces()
        assert result["status"] == "success"
        # Box should be in contact with ground
        contacts = _extract_json_block(result, 1)["contacts"]
        assert len(contacts) > 0
        assert contacts[0]["normal_force"] != 0


class TestForwardKinematics:
    def test_forward_kinematics(self, sim):
        result = sim.forward_kinematics()
        assert result["status"] == "success"
        bodies = _extract_json_block(result, 1)["bodies"]
        assert "box1" in bodies
        assert "link1" in bodies
        assert len(bodies["box1"]["position"]) == 3

    def test_forward_kinematics_single_body_filters_to_that_body(self, sim):
        """forward_kinematics(body_name=X) returns only X's Cartesian pose
        (position + quaternion), not the whole-scene ``bodies`` map."""
        result = sim.forward_kinematics(body_name="box1")
        assert result["status"] == "success"
        payload = _extract_json_block(result, 1)
        assert payload["body"] == "box1"
        assert len(payload["position"]) == 3
        assert len(payload["quaternion"]) == 4
        # Filtered response must not carry the all-bodies map.
        assert "bodies" not in payload

    def test_forward_kinematics_single_body_reflects_moved_joint(self, sim):
        """After driving a joint and re-running FK for one body, the reported
        pose is the freshly recomputed kinematics, not a stale pre-move value."""
        before = _extract_json_block(sim.forward_kinematics(body_name="link2"), 1)["position"]
        sim.set_joint_positions(positions={"shoulder": 1.2})
        after = _extract_json_block(sim.forward_kinematics(body_name="link2"), 1)["position"]
        assert before != after


class TestTotalMass:
    def test_get_total_mass(self, sim):
        result = sim.get_total_mass()
        assert result["status"] == "success"
        data = _extract_json_block(result, 1)
        assert data["total_mass"] > 0
        assert "box1" in data["bodies"]
        assert data["bodies"]["box1"] == pytest.approx(1.0)


class TestExportXML:
    def test_export_xml_string(self, sim):
        result = sim.export_xml()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "mujoco" in text.lower() or "Model XML" in text

    def test_export_xml_file(self, sim, tmp_path):
        path = str(tmp_path / "exported.xml")
        result = sim.export_xml(output_path=path)
        assert result["status"] == "success"
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<mujoco" in content


class TestDirectJointControlListForm:
    """List-form input contract for set_joint_positions / set_joint_velocities.

    The ordered-positional form normalises to a dict using a single robot's
    joint ordering. These cover the documented error contract (no robot,
    ambiguous multi-robot, unknown robot_name, length mismatch, wrong type)
    plus the happy path and the namespace-enumeration fallback.
    """

    @staticmethod
    def _add_robot(sim, name, joint_names, namespace=""):
        from strands_robots.simulation.models import SimRobot

        robot = SimRobot(name=name, urdf_path="", joint_names=list(joint_names), namespace=namespace)
        sim._world.robots[name] = robot
        return robot

    def test_positions_required(self, sim):
        result = sim.set_joint_positions(positions=None)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]

    def test_positions_wrong_type(self, sim):
        result = sim.set_joint_positions(positions=42)
        assert result["status"] == "error"
        assert "must be a dict or list" in result["content"][0]["text"]

    def test_list_form_no_robot(self, sim):
        result = sim.set_joint_positions(positions=[0.1, 0.2])
        assert result["status"] == "error"
        assert "requires a robot" in result["content"][0]["text"]

    def test_list_form_unknown_robot_name(self, sim):
        self._add_robot(sim, "arm", ["shoulder", "elbow"])
        result = sim.set_joint_positions(positions=[0.1, 0.2], robot_name="ghost")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_list_form_ambiguous_multi_robot(self, sim):
        self._add_robot(sim, "arm_a", ["shoulder"])
        self._add_robot(sim, "arm_b", ["elbow"])
        result = sim.set_joint_positions(positions=[0.1])
        assert result["status"] == "error"
        assert "ambiguous" in result["content"][0]["text"]

    def test_list_form_length_mismatch(self, sim):
        self._add_robot(sim, "arm", ["shoulder", "elbow"])
        result = sim.set_joint_positions(positions=[0.1])
        assert result["status"] == "error"
        assert "does not match" in result["content"][0]["text"]

    def test_list_form_success_sets_qpos(self, sim):
        self._add_robot(sim, "arm", ["shoulder", "elbow"])
        result = sim.set_joint_positions(positions=[0.4, -0.2])
        assert result["status"] == "success"
        assert "2/2" in result["content"][0]["text"]
        model, data = sim._world._model, sim._world._data
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
        eid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "elbow")
        assert data.qpos[model.jnt_qposadr[sid]] == pytest.approx(0.4)
        assert data.qpos[model.jnt_qposadr[eid]] == pytest.approx(-0.2)

    def test_list_form_namespace_fallback(self, sim):
        # Robot with no explicit joint_names falls back to enumerating model
        # joints under its namespace ("" matches all joints in the scene).
        self._add_robot(sim, "arm", [], namespace="")
        njnt = sim._world._model.njnt
        result = sim.set_joint_positions(positions=[0.0] * njnt, robot_name="arm")
        assert result["status"] == "success"

    def test_velocities_list_form_success(self, sim):
        self._add_robot(sim, "arm", ["shoulder", "elbow"])
        result = sim.set_joint_velocities(velocities=[1.0, -0.5])
        assert result["status"] == "success"
        model, data = sim._world._model, sim._world._data
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
        assert data.qvel[model.jnt_dofadr[sid]] == pytest.approx(1.0)

    def test_velocities_required(self, sim):
        result = sim.set_joint_velocities(velocities=None)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]

    def test_velocities_wrong_type(self, sim):
        result = sim.set_joint_velocities(velocities="fast")
        assert result["status"] == "error"
        assert "must be a dict or list" in result["content"][0]["text"]

    def test_velocities_list_form_ambiguous(self, sim):
        self._add_robot(sim, "arm_a", ["shoulder"])
        self._add_robot(sim, "arm_b", ["elbow"])
        result = sim.set_joint_velocities(velocities=[1.0])
        assert result["status"] == "error"
        assert "ambiguous" in result["content"][0]["text"]

    def test_velocities_list_form_length_mismatch(self, sim):
        self._add_robot(sim, "arm", ["shoulder", "elbow"])
        result = sim.set_joint_velocities(velocities=[1.0], robot_name="arm")
        assert result["status"] == "error"
        assert "does not match" in result["content"][0]["text"]

    def test_velocities_list_form_namespace_fallback_sets_qvel(self, sim):
        # A robot registered with no explicit ``joint_names`` resolves its
        # positional velocity list against the model joints under its namespace
        # (empty namespace matches every joint in the scene). This mirrors the
        # positions fallback (``test_list_form_namespace_fallback``) and pins the
        # velocity write contract: the list is one entry *per joint* (not per
        # DOF, even when a free joint is present), and each scalar lands on that
        # joint's first qvel slot, in model joint id order.
        self._add_robot(sim, "arm", [], namespace="")
        model, data = sim._world._model, sim._world._data

        joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) for jid in range(model.njnt)]
        # One distinct velocity per joint so a mis-ordered write is caught.
        velocities = [0.1 * (i + 1) for i in range(model.njnt)]

        result = sim.set_joint_velocities(velocities=velocities, robot_name="arm")
        assert result["status"] == "success"

        for jid, expected in enumerate(velocities):
            assert data.qvel[model.jnt_dofadr[jid]] == pytest.approx(expected), joint_names[jid]


class TestMultiRaycast:
    """Batch raycasting: origin validation plus per-ray fail-soft contract.

    A single malformed ray must not abort the whole batch; it produces a
    per-ray error entry while valid rays still resolve.
    """

    def test_multi_raycast_origin_wrong_length(self, sim):
        result = sim.multi_raycast(origin=[0.0, 0.0], directions=[[0, 0, -1]])
        assert result["status"] == "error"
        assert "must be 3 elements" in result["content"][0]["text"]

    def test_multi_raycast_origin_not_iterable(self, sim):
        result = sim.multi_raycast(origin=5, directions=[[0, 0, -1]])
        assert result["status"] == "error"
        assert "list of 3 numbers" in result["content"][0]["text"]

    def test_multi_raycast_per_ray_bad_direction_length(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[[0, 0, -1], [0, 1]])
        assert result["status"] == "success"
        rays = _extract_json_block(result, 1)["rays"]
        assert "must have 3 elements" in rays[1]["error"]

    def test_multi_raycast_per_ray_zero_direction(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[[0, 0, 0]])
        assert result["status"] == "success"
        rays = _extract_json_block(result, 1)["rays"]
        assert "zero-length" in rays[0]["error"]

    def test_multi_raycast_per_ray_direction_not_iterable(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[7])
        assert result["status"] == "success"
        rays = _extract_json_block(result, 1)["rays"]
        assert "list of 3 numbers" in rays[0]["error"]

    def test_multi_raycast_hit_from_above(self, sim):
        # Cast straight down from above the ground plane: expect a hit.
        result = sim.multi_raycast(origin=[0, 0, 2.0], directions=[[0, 0, -1]])
        assert result["status"] == "success"
        rays = _extract_json_block(result, 1)["rays"]
        assert rays[0]["distance"] is not None
        assert "1/1 hits" in result["content"][0]["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRaycastReflectsCurrentPose:
    """raycast / multi_raycast must intersect the CURRENT geom poses.

    ``mj_ray`` reads ``data.geom_xpos``/``geom_xmat`` (world-frame geom poses,
    derived state). MuJoCo does not recompute those on a bare ``qpos`` write --
    a planning/IK loop that pokes ``qpos`` (or a policy thread mid-``mj_step``)
    leaves them stale. The query must refresh kinematics first, exactly like
    ``get_jacobian``/``get_body_state`` do, or it silently reports a hit against
    a geom's previous location while returning ``status=success``.
    """

    @staticmethod
    def _move_box_far(sim):
        """Translate the free box off the +z axis via a direct qpos write, no forward."""
        model, data = sim._world._model, sim._world._data
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "box_free")
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr : adr + 3] = [3.0, 0.0, 0.5]  # move box out of the downward ray at x=0
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        # deliberately NO mj_forward / mj_kinematics here

    def test_raycast_reflects_pose_change_without_forward(self, sim):
        # Baseline: the downward ray hits the box (nearest) at its top face.
        base = _extract_json_block(sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1]), 1)
        assert base["geom_name"] == "box_geom"
        assert base["distance"] == pytest.approx(1.4, abs=1e-3)

        self._move_box_far(sim)

        # After moving the box (no forward), the downward ray at x=0 must miss
        # the box and hit the ground plane at z=0 -> distance 2.0. Pre-fix this
        # reads the stale geom_xpos and still reports box_geom at 1.4.
        after = _extract_json_block(sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1]), 1)
        assert after["geom_name"] == "ground"
        assert after["distance"] == pytest.approx(2.0, abs=1e-3)

    def test_multi_raycast_reflects_pose_change_without_forward(self, sim):
        dirs = [[0, 0, -1]]
        base = _extract_json_block(sim.multi_raycast(origin=[0, 0, 2], directions=dirs), 1)["rays"]
        assert base[0]["distance"] == pytest.approx(1.4, abs=1e-3)  # hits box top

        self._move_box_far(sim)

        after = _extract_json_block(sim.multi_raycast(origin=[0, 0, 2], directions=dirs), 1)["rays"]
        assert after[0]["distance"] == pytest.approx(2.0, abs=1e-3)  # now hits ground
