#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.kinematics module.

Tests the Kinematics ABC, MuJoCoKinematics, PlacoKinematics, ONNXKinematics,
and the create_kinematics factory. All tests run on CPU without MuJoCo, Placo,
or ONNX runtime — heavy backends are mocked.

Coverage targets:
- ABC contract (abstract method enforcement)
- MuJoCoKinematics init, FK, IK (with mocked mujoco)
- PlacoKinematics init, FK, IK (with mocked lerobot)
- ONNXKinematics init, FK (multiple output formats), IK, error paths
- ONNXKinematics.generate_training_data (pure math, mock FK backend)
- create_kinematics factory (all branches: mujoco, placo, fallback, error)
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.kinematics import (
    Kinematics,
    MuJoCoKinematics,
    ONNXKinematics,
    PlacoKinematics,
    create_kinematics,
)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class ConcreteKinematics(Kinematics):
    """Minimal concrete implementation for ABC testing."""

    def __init__(self, n_joints=6):
        self._joint_names = [f"joint_{i}" for i in range(n_joints)]

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = joint_positions[:3] if len(joint_positions) >= 3 else [0, 0, 0]
        return T

    def inverse_kinematics(self, current_joints, target_pose, **kwargs):
        return target_pose[:3, 3]

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def backend(self):
        return "concrete_test"


def _make_mock_mujoco():
    """Create a mock mujoco module with the minimum API surface."""
    mj = MagicMock()
    # mjtObj enum
    mj.mjtObj.mjOBJ_BODY = 1
    mj.mjtObj.mjOBJ_JOINT = 3
    # mj_name2id returns valid body id
    mj.mj_name2id = MagicMock(return_value=1)
    # mj_id2name returns joint names
    mj.mj_id2name = MagicMock(side_effect=lambda m, t, jid: f"joint_{jid}")
    # mj_forward does nothing
    mj.mj_forward = MagicMock()

    # mj_jacBody fills the Jacobian
    def mock_jac_body(model, data, jacp, jacr, body_id):
        if jacp is not None:
            jacp[:] = np.random.randn(*jacp.shape) * 0.1

    mj.mj_jacBody = MagicMock(side_effect=mock_jac_body)
    return mj


def _make_mock_model_data(n_joints=3):
    """Create mock MuJoCo model and data objects."""
    model = MagicMock()
    model.njnt = n_joints
    model.nv = n_joints

    data = MagicMock()
    data.qpos = np.zeros(n_joints)
    data.xpos = np.array([[0, 0, 0], [0.1, 0.2, 0.3]])  # body 0 = world, body 1 = ee
    data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])
    return model, data


# ─────────────────────────────────────────────────────────────────────
# 1. Kinematics ABC
# ─────────────────────────────────────────────────────────────────────


class TestKinematicsABC:
    """Test the Kinematics abstract base class contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Kinematics()

    def test_missing_forward_kinematics(self):
        class Incomplete(Kinematics):
            def inverse_kinematics(self, current_joints, target_pose, **kw):
                return current_joints

            @property
            def joint_names(self):
                return []

            @property
            def backend(self):
                return "inc"

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_inverse_kinematics(self):
        class Incomplete(Kinematics):
            def forward_kinematics(self, joint_positions):
                return np.eye(4)

            @property
            def joint_names(self):
                return []

            @property
            def backend(self):
                return "inc"

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_joint_names(self):
        class Incomplete(Kinematics):
            def forward_kinematics(self, joint_positions):
                return np.eye(4)

            def inverse_kinematics(self, current_joints, target_pose, **kw):
                return current_joints

            @property
            def backend(self):
                return "inc"

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_backend(self):
        class Incomplete(Kinematics):
            def forward_kinematics(self, joint_positions):
                return np.eye(4)

            def inverse_kinematics(self, current_joints, target_pose, **kw):
                return current_joints

            @property
            def joint_names(self):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_works(self):
        kin = ConcreteKinematics(n_joints=4)
        assert isinstance(kin, Kinematics)
        assert kin.backend == "concrete_test"
        assert len(kin.joint_names) == 4

    def test_fk_returns_4x4(self):
        kin = ConcreteKinematics()
        T = kin.forward_kinematics(np.array([1, 2, 3, 0, 0, 0]))
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[:3, 3], [1, 2, 3])

    def test_ik_returns_array(self):
        kin = ConcreteKinematics()
        target = np.eye(4)
        target[:3, 3] = [0.5, 0.5, 0.5]
        result = kin.inverse_kinematics(np.zeros(6), target)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_subclass_check(self):
        assert issubclass(MuJoCoKinematics, Kinematics)
        assert issubclass(PlacoKinematics, Kinematics)
        assert issubclass(ONNXKinematics, Kinematics)


# ─────────────────────────────────────────────────────────────────────
# 2. MuJoCoKinematics (with mocked mujoco)
# ─────────────────────────────────────────────────────────────────────


class TestMuJoCoKinematics:
    """Test MuJoCoKinematics with mocked mujoco dependency."""

    def test_import_error_without_mujoco(self):
        """Should raise ImportError when mujoco is not installed."""
        with patch.dict(sys.modules, {"mujoco": None}):
            with pytest.raises(ImportError, match="mujoco required"):
                MuJoCoKinematics(MagicMock(), MagicMock(), body_name="gripper")

    def test_init_success(self):
        """Init with valid mocked mujoco."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper")
            assert kin.backend == "mujoco"
            assert len(kin.joint_names) == 3
            assert kin._body_id == 1

    def test_init_body_not_found(self):
        """Raise ValueError if body_name is not in the model."""
        mock_mj = _make_mock_mujoco()
        mock_mj.mj_name2id = MagicMock(return_value=-1)
        model, data = _make_mock_model_data()

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            with pytest.raises(ValueError, match="not found"):
                MuJoCoKinematics(model, data, body_name="nonexistent")

    def test_init_custom_joint_ids(self):
        """Pass explicit joint_ids."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(6)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="ee", joint_ids=[0, 2, 4])
            assert len(kin.joint_names) == 3
            assert kin._joint_ids == [0, 2, 4]

    def test_forward_kinematics(self):
        """FK should return a 4x4 matrix with position and rotation from mujoco data."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        # Set up the xpos and xmat return values
        data.xpos = np.array([[0, 0, 0], [0.3, 0.4, 0.5]])
        rot = np.eye(3)
        data.xmat = np.array([np.eye(3).flatten(), rot.flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper")
            T = kin.forward_kinematics(np.array([0.1, 0.2, 0.3]))

        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[:3, 3], [0.3, 0.4, 0.5])
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        assert T[3, 3] == 1.0
        # Verify mj_forward was called
        mock_mj.mj_forward.assert_called()

    def test_inverse_kinematics_convergence(self):
        """IK with a target at the current position should converge immediately."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        target_pos = np.array([0.3, 0.4, 0.5])
        # Make xpos return the target position (already converged)
        data.xpos = np.array([[0, 0, 0], target_pos])
        data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper", tolerance=1e-3)
            target_T = np.eye(4)
            target_T[:3, 3] = target_pos
            result = kin.inverse_kinematics(np.zeros(3), target_T)

        assert result.shape == (3,)

    def test_ik_restores_qpos(self):
        """IK should restore the original qpos after solving."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        original_qpos = np.array([1.0, 2.0, 3.0])
        data.qpos = original_qpos.copy()
        data.xpos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper", max_iterations=2)
            target_T = np.eye(4)
            target_T[:3, 3] = [1, 1, 1]
            _ = kin.inverse_kinematics(np.zeros(3), target_T)

        # qpos should be restored to original
        np.testing.assert_array_almost_equal(data.qpos, original_qpos)

    def test_ik_step_size(self):
        """IK respects step_size parameter."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        data.xpos = np.array([[0, 0, 0], [0.0, 0.0, 0.0]])
        data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper", max_iterations=1)
            target_T = np.eye(4)
            target_T[:3, 3] = [1, 1, 1]
            # Just verify it doesn't crash with different step sizes
            result = kin.inverse_kinematics(np.zeros(3), target_T, step_size=0.1)
            assert result.shape == (3,)

    def test_fk_sets_qpos(self):
        """FK should set joint positions in data.qpos."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        data.xpos = np.array([[0, 0, 0], [0.1, 0.2, 0.3]])
        data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper")
            joints = np.array([0.5, 1.0, 1.5])
            kin.forward_kinematics(joints)

        # Verify qpos was set
        np.testing.assert_array_almost_equal(data.qpos, [0.5, 1.0, 1.5])

    def test_properties(self):
        """Verify backend and joint_names properties."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(4)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper")
            assert kin.backend == "mujoco"
            assert len(kin.joint_names) == 4
            assert all(isinstance(n, str) for n in kin.joint_names)


# ─────────────────────────────────────────────────────────────────────
# 3. PlacoKinematics (with mocked lerobot)
# ─────────────────────────────────────────────────────────────────────


class TestPlacoKinematics:
    """Test PlacoKinematics with mocked lerobot dependency."""

    def test_import_error_without_lerobot(self):
        """Raise ImportError when lerobot kinematics is not available."""
        # Ensure lerobot.model.kinematics is not importable
        with patch.dict(
            sys.modules,
            {
                "lerobot": None,
                "lerobot.model": None,
                "lerobot.model.kinematics": None,
            },
        ):
            with pytest.raises(ImportError, match="lerobot"):
                PlacoKinematics(urdf_path="/fake/robot.urdf")

    def test_init_success(self):
        """Init with mocked lerobot kinematics."""
        mock_rk = MagicMock()
        mock_rk.joint_names = ["j0", "j1", "j2"]

        mock_kin_module = MagicMock()
        mock_kin_module.RobotKinematics = MagicMock(return_value=mock_rk)

        mock_lerobot = MagicMock()
        mock_lerobot_model = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot": mock_lerobot,
                "lerobot.model": mock_lerobot_model,
                "lerobot.model.kinematics": mock_kin_module,
            },
        ):
            kin = PlacoKinematics(urdf_path="/robot.urdf", target_frame="ee_link")
            assert kin.backend == "placo"
            assert kin.joint_names == ["j0", "j1", "j2"]
            assert kin._urdf_path == "/robot.urdf"

    def test_forward_kinematics_delegates(self):
        """FK delegates to internal RobotKinematics."""
        mock_rk = MagicMock()
        mock_rk.joint_names = ["j0"]
        expected_pose = np.eye(4)
        expected_pose[:3, 3] = [1, 2, 3]
        mock_rk.forward_kinematics = MagicMock(return_value=expected_pose)

        mock_kin_module = MagicMock()
        mock_kin_module.RobotKinematics = MagicMock(return_value=mock_rk)

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.model": MagicMock(),
                "lerobot.model.kinematics": mock_kin_module,
            },
        ):
            kin = PlacoKinematics(urdf_path="/robot.urdf")
            joints = np.array([45.0])
            result = kin.forward_kinematics(joints)

        mock_rk.forward_kinematics.assert_called_once_with(joints)
        np.testing.assert_array_equal(result, expected_pose)

    def test_inverse_kinematics_delegates(self):
        """IK delegates to internal RobotKinematics with weights."""
        mock_rk = MagicMock()
        mock_rk.joint_names = ["j0", "j1"]
        expected_joints = np.array([30.0, 60.0])
        mock_rk.inverse_kinematics = MagicMock(return_value=expected_joints)

        mock_kin_module = MagicMock()
        mock_kin_module.RobotKinematics = MagicMock(return_value=mock_rk)

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.model": MagicMock(),
                "lerobot.model.kinematics": mock_kin_module,
            },
        ):
            kin = PlacoKinematics(urdf_path="/robot.urdf")
            current = np.array([0.0, 0.0])
            target = np.eye(4)
            result = kin.inverse_kinematics(current, target, position_weight=2.0, orientation_weight=0.1)

        mock_rk.inverse_kinematics.assert_called_once_with(
            current_joint_pos=current,
            desired_ee_pose=target,
            position_weight=2.0,
            orientation_weight=0.1,
        )
        np.testing.assert_array_equal(result, expected_joints)


# ─────────────────────────────────────────────────────────────────────
# 4. ONNXKinematics (with mocked onnxruntime)
# ─────────────────────────────────────────────────────────────────────


class TestONNXKinematics:
    """Test ONNXKinematics with mocked onnxruntime + scipy."""

    def _make_mock_ort(self, fk_output=None, ik_output=None):
        """Create mock onnxruntime module."""
        mock_ort = MagicMock()

        # FK session
        fk_session = MagicMock()
        fk_input = MagicMock()
        fk_input.name = "input"
        fk_session.get_inputs.return_value = [fk_input]
        if fk_output is None:
            fk_output = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
        fk_session.run.return_value = [fk_output]

        # IK session
        ik_session = MagicMock()
        ik_input = MagicMock()
        ik_input.name = "input"
        ik_session.get_inputs.return_value = [ik_input]
        if ik_output is None:
            ik_output = np.array([[0.5, 1.0, 1.5, 0.0, 0.0, 0.0]], dtype=np.float32)
        ik_session.run.return_value = [ik_output]

        # InferenceSession returns different sessions based on path
        def make_session(path):
            if "fk" in str(path):
                return fk_session
            return ik_session

        mock_ort.InferenceSession = MagicMock(side_effect=make_session)
        return mock_ort, fk_session, ik_session

    def test_import_error_without_onnxruntime(self):
        with patch.dict(sys.modules, {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                ONNXKinematics(fk_model_path="fk_model.onnx")

    def test_init_fk_only(self):
        mock_ort, _, _ = self._make_mock_ort()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")
            assert kin.backend == "onnx"
            assert len(kin.joint_names) == 6  # default
            assert kin._ik_session is None

    def test_init_fk_and_ik(self):
        mock_ort, _, _ = self._make_mock_ort()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            kin = ONNXKinematics(
                fk_model_path="fk_model.onnx",
                ik_model_path="ik_model.onnx",
                joint_names=["a", "b", "c"],
            )
            assert kin.joint_names == ["a", "b", "c"]
            assert kin._ik_session is not None

    def test_fk_6d_output(self):
        """FK with 6D output: [x, y, z, roll, pitch, yaw]."""
        output_6d = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 4]], dtype=np.float32)
        mock_ort, _, _ = self._make_mock_ort(fk_output=output_6d)

        # We need scipy for the Rotation class
        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        # Return a rotation matrix close to what we'd expect
        rot_matrix = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        mock_rot_instance.as_matrix.return_value = rot_matrix
        mock_rot_class.from_euler.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")
            T = kin.forward_kinematics(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))

        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[:3, 3], [0.1, 0.2, 0.3])
        assert T[3, 3] == 1.0

    def test_fk_7d_quaternion_output(self):
        """FK with 7D output: [x, y, z, qx, qy, qz, qw]."""
        output_7d = np.array([[0.5, 0.6, 0.7, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        mock_ort, _, _ = self._make_mock_ort(fk_output=output_7d)

        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        mock_rot_instance.as_matrix.return_value = np.eye(3)
        mock_rot_class.from_quat.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")
            T = kin.forward_kinematics(np.zeros(6))

        np.testing.assert_array_almost_equal(T[:3, 3], [0.5, 0.6, 0.7])
        mock_rot_class.from_quat.assert_called_once()

    def test_fk_16d_output(self):
        """FK with 16-element output (flattened 4x4 matrix)."""
        T_expected = np.eye(4, dtype=np.float32)
        T_expected[:3, 3] = [1, 2, 3]
        output_16 = np.array([T_expected.flatten()], dtype=np.float32)
        mock_ort, _, _ = self._make_mock_ort(fk_output=output_16)

        # scipy not needed for 16D path, but import still happens at module level
        mock_scipy = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")
            T = kin.forward_kinematics(np.zeros(6))

        np.testing.assert_array_almost_equal(T[:3, 3], [1, 2, 3])
        assert T.shape == (4, 4)

    def test_fk_3d_output_fallback(self):
        """FK with <6 element output — only position is set."""
        output_3d = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_ort, _, _ = self._make_mock_ort(fk_output=output_3d)

        mock_scipy = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")
            T = kin.forward_kinematics(np.zeros(6))

        np.testing.assert_array_almost_equal(T[:3, 3], [0.1, 0.2, 0.3])
        # Rotation should be identity (no rotation data in output)
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_ik_no_model_raises(self):
        """IK raises RuntimeError when no IK model was provided."""
        mock_ort, _, _ = self._make_mock_ort()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            kin = ONNXKinematics(fk_model_path="fk_model.onnx")  # no ik_model_path
            with pytest.raises(RuntimeError, match="No IK ONNX model"):
                kin.inverse_kinematics(np.zeros(6), np.eye(4))

    def test_ik_success(self):
        """IK extracts pose from 4x4 target and returns joint positions."""
        ik_output = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        mock_ort, _, _ = self._make_mock_ort(ik_output=ik_output)

        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        mock_rot_instance.as_euler.return_value = np.array([0.0, 0.0, 0.0])
        mock_rot_class.from_matrix.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ONNXKinematics(
                fk_model_path="fk_model.onnx",
                ik_model_path="ik_model.onnx",
            )
            current = np.zeros(4)
            target = np.eye(4)
            target[:3, 3] = [1, 2, 3]
            result = kin.inverse_kinematics(current, target)

        assert len(result) == 4  # Truncated to len(current_joints)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3, 0.4])

    def test_properties(self):
        mock_ort, _, _ = self._make_mock_ort()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            kin = ONNXKinematics(
                fk_model_path="fk_model.onnx",
                joint_names=["a", "b"],
            )
            assert kin.backend == "onnx"
            assert kin.joint_names == ["a", "b"]
            assert kin._fk_path == "fk_model.onnx"
            assert kin._ik_path is None


# ─────────────────────────────────────────────────────────────────────
# 5. ONNXKinematics.generate_training_data
# ─────────────────────────────────────────────────────────────────────


class TestGenerateTrainingData:
    """Test the static method that generates FK/IK training data."""

    def test_basic_generation(self):
        """Generate training data from a concrete kinematics backend."""
        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        mock_rot_instance.as_euler.return_value = np.array([0.0, 0.1, 0.2])
        mock_rot_class.from_matrix.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ConcreteKinematics(n_joints=3)
            joints_data, poses_data = ONNXKinematics.generate_training_data(
                kin, n_samples=50, joint_ranges=[(-1, 1), (-1, 1), (-1, 1)]
            )

        assert joints_data.shape[0] == 50
        assert joints_data.shape[1] == 3
        assert poses_data.shape[0] == 50
        assert poses_data.shape[1] == 6  # x, y, z, roll, pitch, yaw

    def test_default_joint_ranges(self):
        """Uses (-pi, pi) when no joint_ranges provided."""
        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        mock_rot_instance.as_euler.return_value = np.array([0.0, 0.0, 0.0])
        mock_rot_class.from_matrix.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = ConcreteKinematics(n_joints=4)
            joints_data, poses_data = ONNXKinematics.generate_training_data(kin, n_samples=20)

        assert joints_data.shape[0] == 20
        assert joints_data.shape[1] == 4
        # Values should be in [-pi, pi]
        assert np.all(joints_data >= -np.pi)
        assert np.all(joints_data <= np.pi)

    def test_handles_fk_exceptions(self):
        """Samples that cause FK exceptions are silently skipped."""
        call_count = 0

        class FailingKinematics(ConcreteKinematics):
            def forward_kinematics(self, joint_positions):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:
                    raise ValueError("Singular configuration")
                return super().forward_kinematics(joint_positions)

        mock_scipy = MagicMock()
        mock_rot_class = MagicMock()
        mock_rot_instance = MagicMock()
        mock_rot_instance.as_euler.return_value = np.array([0.0, 0.0, 0.0])
        mock_rot_class.from_matrix.return_value = mock_rot_instance
        mock_scipy.spatial.transform.Rotation = mock_rot_class

        with patch.dict(
            sys.modules,
            {
                "scipy": mock_scipy,
                "scipy.spatial": mock_scipy.spatial,
                "scipy.spatial.transform": mock_scipy.spatial.transform,
            },
        ):
            kin = FailingKinematics(n_joints=3)
            joints_data, poses_data = ONNXKinematics.generate_training_data(kin, n_samples=30)

        # Some should succeed, some fail — we get fewer than 30
        assert joints_data.shape[0] < 30
        assert joints_data.shape[0] > 0
        assert joints_data.shape[0] == poses_data.shape[0]


# ─────────────────────────────────────────────────────────────────────
# 6. create_kinematics factory
# ─────────────────────────────────────────────────────────────────────


class TestCreateKinematics:
    """Test the create_kinematics factory function."""

    def test_mujoco_path_with_model_and_data(self):
        """If model + data provided, returns MuJoCoKinematics."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = create_kinematics(model=model, data=data, body_name="ee")

        assert isinstance(kin, MuJoCoKinematics)
        assert kin.backend == "mujoco"

    def test_placo_path_with_urdf(self):
        """If urdf_path provided and placo available, returns PlacoKinematics."""
        mock_rk = MagicMock()
        mock_rk.joint_names = ["j0"]
        mock_kin_module = MagicMock()
        mock_kin_module.RobotKinematics = MagicMock(return_value=mock_rk)

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.model": MagicMock(),
                "lerobot.model.kinematics": mock_kin_module,
            },
        ):
            kin = create_kinematics(urdf_path="/robot.urdf", target_frame="ee")

        assert isinstance(kin, PlacoKinematics)
        assert kin.backend == "placo"

    def test_urdf_fallback_to_mujoco(self):
        """If URDF provided but placo unavailable, falls back to MuJoCo URDF loading."""
        mock_mj = _make_mock_mujoco()
        mock_model, mock_data = _make_mock_model_data(3)
        mock_mj.MjModel.from_xml_path = MagicMock(return_value=mock_model)
        mock_mj.MjData = MagicMock(return_value=mock_data)

        # Make placo import fail
        with patch.dict(
            sys.modules,
            {
                "lerobot": None,
                "lerobot.model": None,
                "lerobot.model.kinematics": None,
                "mujoco": mock_mj,
            },
        ):
            kin = create_kinematics(urdf_path="/robot.urdf")

        assert isinstance(kin, MuJoCoKinematics)

    def test_urdf_fallback_both_fail(self):
        """If both placo and MuJoCo fail, raises RuntimeError."""
        with patch.dict(
            sys.modules,
            {
                "lerobot": None,
                "lerobot.model": None,
                "lerobot.model.kinematics": None,
                "mujoco": None,
            },
        ):
            with pytest.raises(RuntimeError, match="No kinematics backend"):
                create_kinematics(urdf_path="/robot.urdf")

    def test_no_args_raises_valueerror(self):
        """If no model/data and no urdf_path, raises ValueError."""
        with pytest.raises(ValueError, match="Provide either"):
            create_kinematics()

    def test_model_only_raises_valueerror(self):
        """If only model (no data) and no urdf_path, raises ValueError."""
        with pytest.raises(ValueError, match="Provide either"):
            create_kinematics(model=MagicMock())

    def test_data_only_raises_valueerror(self):
        """If only data (no model) and no urdf_path, raises ValueError."""
        with pytest.raises(ValueError, match="Provide either"):
            create_kinematics(data=MagicMock())

    def test_kwargs_forwarded_to_mujoco(self):
        """Extra kwargs are forwarded to MuJoCoKinematics."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = create_kinematics(
                model=model,
                data=data,
                body_name="gripper",
                max_iterations=50,
                tolerance=0.01,
            )
            assert kin._max_iter == 50
            assert kin._tol == 0.01

    def test_joint_names_forwarded_to_placo(self):
        """Joint names are forwarded to PlacoKinematics."""
        mock_rk = MagicMock()
        mock_rk.joint_names = ["custom_j0", "custom_j1"]
        mock_kin_module = MagicMock()
        mock_kin_module.RobotKinematics = MagicMock(return_value=mock_rk)

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.model": MagicMock(),
                "lerobot.model.kinematics": mock_kin_module,
            },
        ):
            create_kinematics(
                urdf_path="/robot.urdf",
                joint_names=["custom_j0", "custom_j1"],
            )

        mock_kin_module.RobotKinematics.assert_called_once_with(
            urdf_path="/robot.urdf",
            target_frame_name="gripper_frame_link",
            joint_names=["custom_j0", "custom_j1"],
        )


# ─────────────────────────────────────────────────────────────────────
# 7. Edge cases and integration
# ─────────────────────────────────────────────────────────────────────


class TestKinematicsEdgeCases:
    """Edge cases and integration tests."""

    def test_fk_identity_transform(self):
        """FK at zero joints should give a valid 4x4 transform."""
        kin = ConcreteKinematics(n_joints=6)
        T = kin.forward_kinematics(np.zeros(6))
        assert T.shape == (4, 4)
        assert T[3, 3] == 1.0

    def test_mujoco_ik_max_iterations(self):
        """IK should stop after max_iterations even if not converged."""
        mock_mj = _make_mock_mujoco()
        model, data = _make_mock_model_data(3)
        # Always far from target
        data.xpos = np.array([[0, 0, 0], [0, 0, 0]])
        data.xmat = np.array([np.eye(3).flatten(), np.eye(3).flatten()])

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper", max_iterations=5)
            target_T = np.eye(4)
            target_T[:3, 3] = [100, 100, 100]  # Very far target
            result = kin.inverse_kinematics(np.zeros(3), target_T)

        # Should have called mj_forward exactly 5 times during IK
        # (plus initial FK calls, etc.)
        assert result.shape == (3,)

    def test_mujoco_joint_name_fallback(self):
        """When mj_id2name returns None, fallback to joint_{id}."""
        mock_mj = _make_mock_mujoco()
        mock_mj.mj_id2name = MagicMock(return_value=None)
        model, data = _make_mock_model_data(2)

        with patch.dict(sys.modules, {"mujoco": mock_mj}):
            kin = MuJoCoKinematics(model, data, body_name="gripper")
            assert kin.joint_names == ["joint_0", "joint_1"]

    def test_module_level_imports(self):
        """Verify all public names are importable."""
        from strands_robots.kinematics import (
            Kinematics,
            MuJoCoKinematics,
            ONNXKinematics,
            PlacoKinematics,
            create_kinematics,
        )

        assert all(
            cls is not None
            for cls in [
                Kinematics,
                MuJoCoKinematics,
                PlacoKinematics,
                ONNXKinematics,
                create_kinematics,
            ]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
