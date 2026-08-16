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

    def test_energy_reflects_direct_qpos_write(self, sim):
        """get_energy must recompute derived state for the CURRENT qpos.

        mj_energyPos reads position-stage derived state (data.xipos for the
        gravity term). A direct data.qpos write - e.g. a planning/IK loop -
        does not refresh it, so without the defensive forward get_energy
        reports the potential energy of the STALE pose. Regression pin for the
        missing forward: fails-before because e_after equalled e_rest.
        """
        model, data = sim._world._model, sim._world._data
        # box1 has a freejoint: qpos = [x, y, z, qw, qx, qy, qz]; index 2 is z.
        z_adr = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "box_free")] + 2

        e_rest = _extract_json_block(sim.get_energy(), 1)

        # Directly raise the box (no forward) - stale path.
        new_z = float(data.qpos[z_adr]) + 1.5
        data.qpos[z_adr] = new_z
        e_after = _extract_json_block(sim.get_energy(), 1)

        # Independent ground truth on a fresh MjData at the new configuration.
        gt = mj.MjData(model)
        gt.qpos[:] = data.qpos
        gt.qvel[:] = data.qvel
        mj.mj_forward(model, gt)
        mj.mj_energyPos(model, gt)
        truth_potential = float(gt.energy[0])

        # fails-before: e_after["potential"] equalled the stale e_rest value.
        assert abs(e_after["potential"] - truth_potential) < 1e-6
        assert abs(e_after["potential"] - e_rest["potential"]) > 1e-3


class TestExternalForces:
    def test_apply_force(self, sim):
        result = sim.apply_force(body_name="box1", force=[0, 0, 100])
        assert result["status"] == "success"
        assert "box1" in result["content"][0]["text"]

    def test_apply_force_invalid_body(self, sim):
        result = sim.apply_force(body_name="nonexistent", force=[0, 0, 10])
        assert result["status"] == "error"

    def test_force_changes_acceleration(self, sim):
        """A latched force accelerates the body it names.

        Asserts the motion rather than a physics buffer: which buffer carries
        the latch is an implementation choice, whereas a 100 N upward force on
        a 1 kg box under gravity has to lift it.
        """
        data = sim._world._data
        z_before = float(data.qpos[2])

        assert sim.apply_force(body_name="box1", force=[0, 0, 100])["status"] == "success"
        sim.step(50)

        assert float(data.qpos[2]) > z_before, "box1 did not rise under a 100 N upward force"


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


class _LegacyMjData:
    """MjData proxy exposing the pre-3.11 legacy ancestor-walk buffer ``qM``.

    MuJoCo 3.11 removed ``data.qM``, so a test driving the legacy
    ``mj_fullM(model, dst, qM)`` order cannot read that buffer off the
    installed MjData. Presenting one here keeps the drift coverage portable
    across every supported MuJoCo instead of only the builds that still have
    the attribute.
    """

    def __init__(self, data, qm):
        self._data = data
        self.qM = qm

    def __getattr__(self, attr):
        return getattr(self._data, attr)


class _CsrOnlyMjData:
    """MjData proxy exposing the inertia only as the CSR ``M`` (MuJoCo >= 3.11)."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, attr):
        if attr == "qM":
            raise AttributeError(attr)
        return getattr(self._data, attr)


class _NoSym2DenseShim:
    """mujoco proxy of a build with the CSR buffers but no ``mju_sym2dense``.

    MuJoCo exports that conversion only from 3.10, while the CSR inertia and
    its index arrays ship from 3.5, so every build in between reaches the CSR
    rung with nothing to call. ``mj_fullM`` is refused too, which is what puts
    the helper on that rung in the first place.
    """

    def __getattr__(self, attr):
        if attr == "mju_sym2dense":
            raise AttributeError(attr)
        return getattr(mj, attr)

    @staticmethod
    def mj_fullM(m, a, b):
        raise TypeError("mj_fullM unavailable in this binding")


class _RecordingSym2DenseShim:
    """mujoco proxy whose ``mju_sym2dense`` records the call it is handed."""

    def __init__(self, reference, calls):
        self._reference = reference
        self._calls = calls

    def __getattr__(self, attr):
        return getattr(mj, attr)

    @staticmethod
    def mj_fullM(m, a, b):
        raise TypeError("mj_fullM unavailable in this binding")

    def mju_sym2dense(self, dst, values, rownnz, rowadr, colind):
        self._calls.append((dst, values, rownnz, rowadr, colind))
        dst[...] = self._reference


class _NoInertiaMjData:
    """MjData proxy exposing the joint-space inertia under neither name."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, attr):
        if attr in ("qM", "M"):
            raise AttributeError(attr)
        return getattr(self._data, attr)


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
        # BOTH halves of that older build are emulated: the module (a shim
        # mj_fullM) and MjData (a proxy carrying the legacy `qM` buffer, which
        # MuJoCo 3.11 removed). Reading the buffer off the installed MjData
        # instead would pin this drift test to one MuJoCo layout - the exact
        # coupling it exists to catch.
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
        legacy_qm = np.linspace(1.0, 2.0, model.nM)

        class _LegacyShim:
            """Proxy mujoco module exposing only a legacy mj_fullM."""

            def __getattr__(self, attr):
                return getattr(mj, attr)

            @staticmethod
            def mj_fullM(m, a, b):
                # Reject the modern call, whose 2nd arg is MjData rather than
                # the dst buffer, forcing the legacy path.
                if not isinstance(a, np.ndarray):
                    raise TypeError("legacy binding: expected (model, dst, qM)")
                # Legacy contract: a is the dense dst buffer, b is the sparse
                # inertia qM (1D or [m, 1]). Validate that contract, then fill
                # dst from the known-correct reference (version-independent, so
                # the emulation works whatever signature the installed mujoco
                # binding actually uses).
                assert a.flags["WRITEABLE"]
                # The helper must forward the buffer it read off MjData, not a
                # differently-shaped stand-in.
                assert np.array_equal(np.asarray(b).reshape(-1), legacy_qm)
                a[...] = reference

        M = _full_mass_matrix(_LegacyShim(), model, _LegacyMjData(data, legacy_qm))
        assert np.allclose(M, reference)

    def test_helper_falls_back_to_1d_only_legacy_signature(self, sim):
        # Oldest binding: mj_fullM(model, dst, qM) accepts ONLY a raw 1-D sparse
        # buffer and rejects the [m, 1] column form the first legacy attempt
        # passes. The helper must fall through that inner TypeError to the flat
        # 1-D call and still reconstruct the correct matrix. This pins the
        # innermost fallback (the widest-compatibility call) the other drift
        # tests never reach, because their shim accepts both buffer shapes.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        reference = _full_mass_matrix(mj, model, data)
        legacy_qm = np.linspace(1.0, 2.0, model.nM)

        class _OneDLegacyShim:
            """mujoco proxy whose mj_fullM accepts only (model, dst, qM_1d)."""

            def __getattr__(self, attr):
                return getattr(mj, attr)

            @staticmethod
            def mj_fullM(m, a, b):
                # Reject the modern (model, data, dst) order.
                if not isinstance(a, np.ndarray):
                    raise TypeError("legacy binding: expected (model, dst, qM)")
                # Reject the [m, 1] column buffer the first legacy attempt uses:
                # only the raw 1-D sparse form is accepted here.
                arr = np.asarray(b)
                if arr.ndim != 1:
                    raise TypeError("oldest binding: expected a 1-D sparse buffer")
                assert a.flags["WRITEABLE"]
                assert np.array_equal(arr, legacy_qm)
                a[...] = reference

        M = _full_mass_matrix(_OneDLegacyShim(), model, _LegacyMjData(data, legacy_qm))
        assert np.allclose(M, reference)

    def test_helper_reads_csr_inertia_when_the_legacy_buffer_is_gone(self, sim):
        # MuJoCo 3.11 removed data.qM: the joint-space inertia lives only in the
        # CSR data.M. With the modern mj_fullM order rejected there is no legacy
        # buffer left to pass, so the helper must convert the CSR form rather
        # than reach for an attribute that release deleted.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        reference = _full_mass_matrix(mj, model, data)
        if not hasattr(data, "M"):
            pytest.skip("installed mujoco predates the CSR data.M inertia")

        class _ModernOnlyShim:
            """mujoco proxy whose mj_fullM rejects every argument order."""

            def __getattr__(self, attr):
                return getattr(mj, attr)

            @staticmethod
            def mj_fullM(m, a, b):
                raise TypeError("mj_fullM unavailable in this binding")

        M = _full_mass_matrix(_ModernOnlyShim(), model, _CsrOnlyMjData(data))
        # The CSR conversion is exact, not an approximation of mj_fullM.
        assert np.array_equal(M, reference)
        assert M.flags["C_CONTIGUOUS"]
        assert M.dtype == np.float64

    def test_helper_expands_csr_inertia_when_the_conversion_binding_is_absent(self, sim):
        # MuJoCo exports mju_sym2dense only from 3.10, but the CSR inertia and
        # its M_rownnz / M_rowadr / M_colind index arrays ship from 3.5. Every
        # build in between therefore reaches the CSR rung with no conversion to
        # call, and the package supports mujoco>=3.2.0. The helper must expand
        # the stored lower triangle through those index arrays rather than
        # reaching for a symbol that release range does not export.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        reference = _full_mass_matrix(mj, model, data)
        if not hasattr(data, "M"):
            pytest.skip("installed mujoco predates the CSR data.M inertia")

        M = _full_mass_matrix(_NoSym2DenseShim(), model, _CsrOnlyMjData(data))

        # The index-array expansion is exact, not an approximation: it must
        # reproduce mj_fullM bit for bit, both triangles filled.
        assert np.array_equal(M, reference)
        assert np.array_equal(M, M.T)
        assert M.flags["C_CONTIGUOUS"]
        assert M.dtype == np.float64

    def test_helper_prefers_the_native_conversion_where_the_binding_has_it(self, sim):
        # Where MuJoCo does export mju_sym2dense it stays the conversion used,
        # called with the documented argument order (dst, values, rownnz,
        # rowadr, colind), so the index-array expansion is a fallback for the
        # builds that lack it rather than a second implementation shadowing it.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        reference = _full_mass_matrix(mj, model, data)
        if not hasattr(data, "M"):
            pytest.skip("installed mujoco predates the CSR data.M inertia")
        calls: list = []

        M = _full_mass_matrix(_RecordingSym2DenseShim(reference, calls), model, _CsrOnlyMjData(data))

        assert len(calls) == 1
        dst, values, rownnz, rowadr, colind = calls[0]
        assert dst.shape == (model.nv, model.nv)
        assert dst.flags["WRITEABLE"] and dst.flags["C_CONTIGUOUS"]
        assert values.dtype == np.float64 and values.flags["C_CONTIGUOUS"]
        assert np.array_equal(values, np.asarray(data.M, dtype=np.float64))
        assert np.array_equal(np.asarray(rownnz), np.asarray(model.M_rownnz))
        assert np.array_equal(np.asarray(rowadr), np.asarray(model.M_rowadr))
        assert np.array_equal(np.asarray(colind), np.asarray(model.M_colind))
        assert np.array_equal(M, reference)

    def test_helper_names_both_buffers_when_neither_exists(self, sim):
        # Nothing left to read: fail with a message naming both spellings and
        # the installed version, rather than an opaque AttributeError raised by
        # whichever attribute the code happened to touch last.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)

        class _NoFullMShim:
            def __getattr__(self, attr):
                return getattr(mj, attr)

            @staticmethod
            def mj_fullM(m, a, b):
                raise TypeError("mj_fullM unavailable in this binding")

        with pytest.raises(AttributeError) as exc:
            _full_mass_matrix(_NoFullMShim(), model, _NoInertiaMjData(data))
        message = str(exc.value)
        assert "data.qM" in message
        assert "data.M" in message
        assert mj.__version__ in message

    def test_get_mass_matrix_tool_works_without_the_legacy_buffer(self, sim):
        # End-to-end through the agent-facing tool: a build without data.qM must
        # still report a symmetric positive-definite M(q), not an error.
        model, data = sim._world._model, sim._world._data
        mj.mj_forward(model, data)
        if not hasattr(data, "M"):
            pytest.skip("installed mujoco predates the CSR data.M inertia")
        reference = _full_mass_matrix(mj, model, data)
        M = _full_mass_matrix(mj, model, _CsrOnlyMjData(data))
        assert np.allclose(M, reference)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_csr_expansion_is_exact_on_a_branching_tree(self):
        # The chained fixture stores a full lower triangle, so its CSR rows are
        # dense and the index arrays are trivially ordered. A branching tree is
        # the case the expansion can actually get wrong: two sibling limbs do
        # not couple, so rows are shorter than their row index and the column
        # indices skip DoFs. Pin the expansion against mj_fullM there.
        model = mj.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="trunk">
                  <freejoint/>
                  <geom type="box" size="0.1 0.1 0.1"/>
                  <body name="left" pos="0.1 0 0">
                    <joint type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
                    <body name="left_tip" pos="0 0 0.2">
                      <joint type="hinge" axis="1 0 0"/>
                      <geom type="sphere" size="0.04"/>
                    </body>
                  </body>
                  <body name="right" pos="-0.1 0 0">
                    <joint type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
                    <body name="right_tip" pos="0 0 0.2">
                      <joint type="hinge" axis="1 0 0"/>
                      <geom type="sphere" size="0.04"/>
                    </body>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mj.MjData(model)
        data.qpos[7:] = 0.4
        mj.mj_forward(model, data)
        if not hasattr(data, "M"):
            pytest.skip("installed mujoco predates the CSR data.M inertia")
        # The tree really is sparse: fewer stored values than a full triangle.
        assert model.nM < model.nv * (model.nv + 1) // 2

        reference = _full_mass_matrix(mj, model, data)
        M = _full_mass_matrix(_NoSym2DenseShim(), model, _CsrOnlyMjData(data))

        assert np.array_equal(M, reference)
        # Sibling limbs share no inertial coupling, so the expansion must leave
        # those entries zero rather than smear a stored value across the row.
        assert np.count_nonzero(reference) < reference.size

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

    def test_save_load_round_trips_ctrl(self, sim):
        # ctrl (servo targets) MUST survive a checkpoint round-trip. Previously
        # save_state used mjSTATE_FULLPHYSICS, which excludes ctrl/qfrc_applied,
        # so the first step after load_state drove toward the pre-restore
        # targets. Regression for that silent drop.
        sim._world._data.ctrl[0] = 0.8
        mj.mj_forward(sim._world._model, sim._world._data)

        result = sim.save_state(name="ctrl_ckpt")
        assert result["status"] == "success"

        # Clobber ctrl after the checkpoint.
        sim._world._data.ctrl[0] = -0.5
        mj.mj_forward(sim._world._model, sim._world._data)
        assert sim._world._data.ctrl[0] == pytest.approx(-0.5)

        result = sim.load_state(name="ctrl_ckpt")
        assert result["status"] == "success"
        assert sim._world._data.ctrl[0] == pytest.approx(0.8)

    def test_load_state_after_recompile_returns_structured_error(self, sim):
        # A scene recompile that resizes the state vector (add_object inserts a
        # free joint -> nq/nv grow) must invalidate an earlier checkpoint. The
        # stale vector must NOT be applied: previously mj_setState raised a raw
        # ValueError or silently misaligned qpos. Expect a structured error dict.
        result = sim.save_state(name="pre_add")
        assert result["status"] == "success"

        add = sim.add_object(name="dropped_cube", shape="box", size=[0.05, 0.05, 0.05])
        assert add["status"] == "success"

        result = sim.load_state(name="pre_add")
        assert result["status"] == "error"
        assert "stale" in result["content"][0]["text"].lower()

        # The checkpoint saved AFTER the mutation applies cleanly.
        result = sim.save_state(name="post_add")
        assert result["status"] == "success"
        result = sim.load_state(name="post_add")
        assert result["status"] == "success"

    def test_load_state_after_same_shape_recompile_returns_error(self, sim):
        # A same-shape recompile (remove one free-jointed object, add another)
        # leaves nq/nv/na/nu unchanged but the joint addresses now map to
        # different bodies. The recompile-generation stamp must catch this and
        # return a structured error - applying the stale vector would silently
        # teleport the new object into the old objects saved pose/velocity.
        add1 = sim.add_object(name="obj_a", shape="sphere", size=[0.03])
        assert add1["status"] == "success"

        result = sim.save_state(name="with_a")
        assert result["status"] == "success"

        # Remove obj_a, add obj_b - same shape (one free joint each), so
        # nq/nv/na/nu are identical after both mutations.
        rm = sim.remove_object(name="obj_a")
        assert rm["status"] == "success"
        add2 = sim.add_object(name="obj_b", shape="sphere", size=[0.03])
        assert add2["status"] == "success"

        # The fingerprint must detect the stale checkpoint.
        result = sim.load_state(name="with_a")
        assert result["status"] == "error"
        assert "stale" in result["content"][0]["text"].lower()

        # A fresh checkpoint saved after the mutation applies cleanly.
        result = sim.save_state(name="with_b")
        assert result["status"] == "success"
        result = sim.load_state(name="with_b")
        assert result["status"] == "success"


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

    def test_set_body_mass_on_a_massless_body_is_refused(self, sim):
        """A massless frame has no inertial to scale, so the mass is refused.

        ``arm_base`` is a pure kinematic frame: no ``<inertial>`` and no geom of
        its own, so its compiled mass is 0 and there is nothing for a mass change
        to scale. Reporting success here produced a body heavy in translation with
        zero rotational resistance - the same physically inconsistent state
        ``test_set_body_mass_rejects_nonfinite`` exists to prevent, differing only
        in which component is corrupt. It was also a value nothing could keep: the
        mass lives on a body's inertial or on its geoms, and this body has
        neither, so the next scene recompile discarded it.

        The division by zero this test originally guarded is still not reached -
        the refusal happens before the ratio is taken.
        """
        model = sim._world._model
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm_base")
        assert float(model.body_mass[body_id]) == pytest.approx(0.0)
        result = sim.set_body_properties(body_name="arm_base", mass=2.0)
        assert result["status"] == "error"
        assert "no mass of its own" in result["content"][0]["text"]
        assert model.body_mass[body_id] == pytest.approx(0.0)
        assert model.body_inertia[body_id] == pytest.approx([0.0, 0.0, 0.0])

    def test_set_body_mass_rejects_nonfinite(self, sim):
        """A non-finite mass is rejected instead of silently corrupting the model.

        ``float('nan') <= 0`` and ``float('inf') <= 0`` are both ``False``, so a
        bare ``mass <= 0`` guard lets NaN/+Inf slip through: the body's mass and
        (mass-tracking) inertia would be set to NaN/Inf and the next ``mj_step``
        would produce a non-finite ``qacc`` -- a silent physics corruption
        reported as ``status="success"``. The guard must also reject non-finite
        values, matching the finiteness contract already enforced by
        ``set_timestep`` / ``set_gravity``.
        """
        model = sim._world._model
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "box1")
        good_mass = float(model.body_mass[body_id])
        good_inertia = model.body_inertia[body_id].copy()
        assert good_mass > 0 and (good_inertia > 0).all()

        for bad in (float("nan"), float("inf"), -float("inf")):
            result = sim.set_body_properties(body_name="box1", mass=bad)
            assert result["status"] == "error", f"mass={bad!r} was not rejected"
            assert "finite" in result["content"][0]["text"]
            # Neither mass nor inertia was mutated by the rejected call.
            assert model.body_mass[body_id] == pytest.approx(good_mass)
            assert model.body_inertia[body_id] == pytest.approx(good_inertia)
            # And the world remains integrable (no NaN leaked into the model).
            mj.mj_forward(model, sim._world._data)
            import numpy as np

            assert np.all(np.isfinite(sim._world._data.qacc))

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
        recompile). A box defines three half-extents, so all three are set."""
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        result = sim.set_geom_properties(geom_name="box_geom", size=[0.25, 0.3, 0.35])
        assert result["status"] == "success"
        assert "size" in result["content"][0]["text"]
        new_size = sim._world._model.geom_size[geom_id]
        assert new_size[0] == pytest.approx(0.25)
        assert new_size[1] == pytest.approx(0.3)
        assert new_size[2] == pytest.approx(0.35)

    def test_set_geom_size_shorter_than_the_type_defines_is_rejected(self, sim):
        """A size vector shorter than the geom's type defines is refused.

        This previously wrote the components provided and left the rest at their
        compiled value, so ``size=[0.2]`` on a box resized x only and reported
        success for a box the caller never described (0.2 x old_y x old_z). There
        is no meaningful value to invent for the omitted components, so the whole
        write is refused and the compiled half-extents stay intact.
        """
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        original = sim._world._model.geom_size[geom_id].copy()
        result = sim.set_geom_properties(geom_name="box_geom", size=[0.2])
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "exactly 3 component(s)" in text
        assert "box" in text
        assert sim._world._model.geom_size[geom_id] == pytest.approx(original)

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

    _PRIMITIVE_BOUNDS_SCENE = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1"/>
        <body name="b_sphere" pos="0 0 1"><freejoint/>
          <geom name="sphere_g" type="sphere" size="0.1"/></body>
        <body name="b_cylinder" pos="1 0 1"><freejoint/>
          <geom name="cylinder_g" type="cylinder" size="0.1 0.2"/></body>
        <body name="b_ellipsoid" pos="2 0 1"><freejoint/>
          <geom name="ellipsoid_g" type="ellipsoid" size="0.1 0.2 0.3"/></body>
      </worldbody>
    </mujoco>
    """

    def _primitive_bounds_sim(self):
        from strands_robots.simulation.models import SimStatus, SimWorld

        s = Simulation(tool_name="test_primitive_bounds", mesh=False)
        s._world = SimWorld()
        spec = mj.MjSpec.from_string(self._PRIMITIVE_BOUNDS_SCENE)
        s._world._backend_state["spec"] = spec
        s._world._model = spec.compile()
        s._world._data = mj.MjData(s._world._model)
        s._world.status = SimStatus.IDLE
        mj.mj_forward(s._world._model, s._world._data)
        return s

    @pytest.mark.parametrize(
        "geom, new_size, expected_rbound, expected_half",
        [
            # sphere: rbound = radius; aabb half = [r, r, r].
            ("sphere_g", [0.3], 0.3, [0.3, 0.3, 0.3]),
            # cylinder: rbound = hypot(radius, half-length); aabb half = [r, r, hl].
            ("cylinder_g", [0.2, 0.5], float(np.hypot(0.2, 0.5)), [0.2, 0.2, 0.5]),
            # ellipsoid: rbound = max semi-axis; aabb half = the three semi-axes.
            ("ellipsoid_g", [0.15, 0.25, 0.4], 0.4, [0.15, 0.25, 0.4]),
        ],
    )
    def test_set_geom_size_recomputes_bounds_per_primitive_type(self, geom, new_size, expected_rbound, expected_half):
        """Each size-defined primitive gets its own broadphase/mid-phase formula.

        ``geom_rbound`` (broadphase bounding sphere) and ``geom_aabb`` (mid-phase
        box) are compiled from ``geom_size`` and are not refreshed by the solver,
        so a runtime size write must recompute them with the primitive's own
        extent formula - a shared or wrong formula would silently cull the grown
        geom from collision detection. This pins the per-type formula for the
        sphere, cylinder, and ellipsoid branches to a fresh compile's values.
        """
        s = self._primitive_bounds_sim()
        try:
            model = s._world._model
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom)
            result = s.set_geom_properties(geom_name=geom, size=new_size)
            assert result["status"] == "success"
            assert float(model.geom_rbound[gid]) == pytest.approx(expected_rbound)
            assert model.geom_aabb[gid][3:6].tolist() == pytest.approx(expected_half)
        finally:
            s.cleanup()

    def test_set_geom_size_inert_bounds_for_non_size_defined_geom(self):
        """A plane's extent comes from asset/type data, not ``geom_size``.

        For mesh/plane/height-field/SDF geoms the size write still lands in
        ``geom_size`` (callers may rely on the stored value), but the collision
        bounds must be left untouched because recomputing them from ``geom_size``
        would be meaningless. This pins the inert branch: bounds unchanged, size
        written.
        """
        s = self._primitive_bounds_sim()
        try:
            model = s._world._model
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
            rbound_before = float(model.geom_rbound[gid])
            aabb_before = model.geom_aabb[gid].tolist()

            result = s.set_geom_properties(geom_name="ground", size=[8.0, 8.0, 0.1])
            assert result["status"] == "success"

            # Bounds untouched (recompute reported the type as non-size-defined).
            assert float(model.geom_rbound[gid]) == pytest.approx(rbound_before)
            assert model.geom_aabb[gid].tolist() == pytest.approx(aabb_before)
            # ...but the requested size still landed in the live model.
            assert model.geom_size[gid][:2].tolist() == pytest.approx([8.0, 8.0])
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
    """Batch raycasting: origin validation plus the all-or-nothing batch contract.

    A single malformed direction refuses the whole batch, naming its index. The
    batch previously cast the remaining rays and reported ``distance: None`` for
    the malformed one, which is exactly what a genuine miss reports - so a ray
    that was never cast read as free space in the very clearance checks this
    method serves.
    """

    def test_multi_raycast_origin_wrong_length(self, sim):
        result = sim.multi_raycast(origin=[0.0, 0.0], directions=[[0, 0, -1]])
        assert result["status"] == "error"
        assert "must be 3 elements" in result["content"][0]["text"]

    def test_multi_raycast_origin_not_iterable(self, sim):
        result = sim.multi_raycast(origin=5, directions=[[0, 0, -1]])
        assert result["status"] == "error"
        assert "list of 3 numbers" in result["content"][0]["text"]

    def test_multi_raycast_bad_direction_length_refuses_the_batch(self, sim):
        """A 2-component direction refuses the batch and names its index."""
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[[0, 0, -1], [0, 1]])
        assert result["status"] == "error"
        invalid = _extract_json_block(result, 1)["invalid_directions"]
        assert [entry["index"] for entry in invalid] == [1]
        assert "must have exactly 3 component" in invalid[0]["error"]

    def test_multi_raycast_zero_direction_refuses_the_batch(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[[0, 0, 0]])
        assert result["status"] == "error"
        invalid = _extract_json_block(result, 1)["invalid_directions"]
        assert "zero-length" in invalid[0]["error"]

    def test_multi_raycast_direction_not_iterable_refuses_the_batch(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 2], directions=[7])
        assert result["status"] == "error"
        invalid = _extract_json_block(result, 1)["invalid_directions"]
        assert "must be a sequence of numbers" in invalid[0]["error"]

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


class TestContactForcesReflectsCurrentPose:
    """get_contact_forces must reflect the CURRENT qpos, exactly like get_contacts.

    ``mj_contactForce`` reads ``data.contact[]``/``data.ncon`` (collision output)
    and ``data.efc_force`` (constraint solve) -- all recomputed only by
    ``mj_forward``/``mj_step``. A manual ``qpos`` write (planning/IK loop), a
    pose set right after ``reset``/``add_robot``, or a policy thread
    mid-``mj_step`` leaves them stale, so without a forward the method reports
    phantom contacts with fabricated forces while returning ``status=success``.
    ``get_contacts`` already forwards; the two contact queries must agree.
    """

    @staticmethod
    def _box_in_forces(result):
        for block in result["content"]:
            if "json" in block:
                return any("box_geom" in (c["geom1"], c["geom2"]) for c in block["json"].get("contacts", []))
        return False

    @staticmethod
    def _lift_box(sim):
        model, data = sim._world._model, sim._world._data
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "box_free")
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr : adr + 3] = [0.0, 0.0, 3.0]  # lift the box 3m into the air
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        # deliberately NO mj_forward / mj_step here

    def test_get_contact_forces_reflects_pose_change_without_forward(self, sim):
        # Settle the box on the ground -> a real box_geom<->ground contact.
        for _ in range(500):
            mj.mj_step(sim._world._model, sim._world._data)
        base = sim.get_contact_forces()
        assert base["status"] == "success"
        assert self._box_in_forces(base)

        self._lift_box(sim)

        # Query FIRST (before any other call forwards). The box is 3m up with no
        # possible contact, so a correct query no longer reports box_geom.
        # Pre-fix reads the stale contact list + fabricated force for the box.
        after = sim.get_contact_forces()
        assert after["status"] == "success"
        assert not self._box_in_forces(after)

    def test_contact_queries_agree_after_pose_change(self, sim):
        for _ in range(500):
            mj.mj_step(sim._world._model, sim._world._data)
        self._lift_box(sim)
        # get_contact_forces must be queried FIRST; get_contacts forwards and
        # would otherwise clear the staleness for the second call. After the
        # box is lifted, neither query may still report the airborne box.
        forces_has_box = self._box_in_forces(sim.get_contact_forces())
        contacts_text = sim.get_contacts()["content"][0]["text"]
        assert forces_has_box is False
        assert "box_geom" not in contacts_text
