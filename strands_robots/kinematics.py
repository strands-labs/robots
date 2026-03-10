"""
Kinematics module for strands-robots.

Provides forward/inverse kinematics with two backends:
1. MuJoCo IK (for simulation) — uses mujoco's built-in IK solver
2. Placo IK (for real hardware) — wraps LeRobot's RobotKinematics

Also provides a unified bridge that auto-selects the right backend.

Usage:
    from strands_robots.kinematics import create_kinematics

    # Auto-detect (MuJoCo if sim, placo if URDF)
    kin = create_kinematics(urdf_path="path/to/robot.urdf")
    pose = kin.forward_kinematics(joint_positions)
    joints = kin.inverse_kinematics(current_joints, target_pose)

    # Explicitly use MuJoCo backend (for sim)
    from strands_robots.kinematics import MuJoCoKinematics
    kin = MuJoCoKinematics(model, data, body_name="gripper")

    # Explicitly use Placo backend (for real hardware)
    from strands_robots.kinematics import PlacoKinematics
    kin = PlacoKinematics(urdf_path="robot.urdf", target_frame="gripper_link")
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Kinematics(ABC):
    """Abstract base class for kinematics solvers."""

    @abstractmethod
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute FK: joint positions → 4x4 end-effector pose.

        Args:
            joint_positions: Joint angles (radians for MuJoCo, degrees for Placo)

        Returns:
            4x4 homogeneous transformation matrix
        """
        pass

    @abstractmethod
    def inverse_kinematics(
        self,
        current_joints: np.ndarray,
        target_pose: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute IK: target pose → joint positions.

        Args:
            current_joints: Current joint positions (initial guess)
            target_pose: Target 4x4 transformation matrix

        Returns:
            Joint positions that achieve the target pose
        """
        pass

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        """Get joint names."""
        pass

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend name ('mujoco' or 'placo')."""
        pass


class MuJoCoKinematics(Kinematics):
    """MuJoCo-based kinematics solver.

    Uses MuJoCo's built-in Jacobian-based IK. Works directly with
    the simulation model and data — no external dependencies.

    Args:
        model: MuJoCo model (mujoco.MjModel)
        data: MuJoCo data (mujoco.MjData)
        body_name: Name of the end-effector body in the model
        joint_ids: Optional list of joint IDs to control (default: all)
        max_iterations: Max IK solver iterations
        tolerance: Position tolerance for IK convergence (meters)
    """

    def __init__(
        self,
        model,
        data,
        body_name: str = "gripper",
        joint_ids: Optional[List[int]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ):
        try:
            import mujoco

            self._mj = mujoco
        except ImportError:
            raise ImportError("mujoco required for MuJoCoKinematics")

        self._model = model
        self._data = data
        self._body_name = body_name
        self._max_iter = max_iterations
        self._tol = tolerance

        # Resolve body ID
        self._body_id = self._mj.mj_name2id(model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        if self._body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")

        # Resolve joint IDs
        if joint_ids is not None:
            self._joint_ids = joint_ids
        else:
            # All joints
            self._joint_ids = list(range(model.njnt))

        # Cache joint names
        self._joint_names = []
        for jid in self._joint_ids:
            name = self._mj.mj_id2name(model, self._mj.mjtObj.mjOBJ_JOINT, jid)
            self._joint_names.append(name or f"joint_{jid}")

        logger.info(f"MuJoCoKinematics: body='{body_name}', {len(self._joint_ids)} joints")

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def backend(self) -> str:
        return "mujoco"

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute FK using MuJoCo.

        Args:
            joint_positions: Joint angles in radians

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Set joint positions
        for i, jid in enumerate(self._joint_ids):
            if i < len(joint_positions):
                self._data.qpos[jid] = joint_positions[i]

        # Forward kinematics
        self._mj.mj_forward(self._model, self._data)

        # Get body position and rotation
        pos = self._data.xpos[self._body_id].copy()
        rot_mat = self._data.xmat[self._body_id].reshape(3, 3).copy()

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T

    def inverse_kinematics(
        self,
        current_joints: np.ndarray,
        target_pose: np.ndarray,
        step_size: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """Compute IK using Jacobian transpose method.

        Args:
            current_joints: Current joint positions (radians)
            target_pose: Target 4x4 transformation matrix
            step_size: Step size for gradient descent

        Returns:
            Joint positions in radians
        """
        mj = self._mj
        model = self._model
        data = self._data
        nj = len(self._joint_ids)

        # Extract target position
        target_pos = target_pose[:3, 3]

        # Save original state
        qpos_save = data.qpos.copy()

        # Set initial joint positions
        for i, jid in enumerate(self._joint_ids):
            if i < len(current_joints):
                data.qpos[jid] = current_joints[i]

        result = current_joints.copy()

        for iteration in range(self._max_iter):
            mj.mj_forward(model, data)

            # Current end-effector position
            ee_pos = data.xpos[self._body_id].copy()

            # Position error
            err = target_pos - ee_pos
            err_norm = np.linalg.norm(err)

            if err_norm < self._tol:
                break

            # Compute Jacobian
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, None, self._body_id)

            # Extract columns for our joints only
            jac = jacp[:, :nj]

            # Jacobian transpose step
            dq = step_size * jac.T @ err

            # Apply
            for i, jid in enumerate(self._joint_ids):
                if i < len(dq):
                    data.qpos[jid] += dq[i]
                    result[i] = data.qpos[jid]

        # Restore original state
        data.qpos[:] = qpos_save
        mj.mj_forward(model, data)

        return result[: len(current_joints)]


class PlacoKinematics(Kinematics):
    """Placo-based kinematics solver (wraps LeRobot's RobotKinematics).

    Uses the placo C++ library for URDF-based FK/IK.
    Works with real hardware where you have a URDF but no MuJoCo model.

    Args:
        urdf_path: Path to robot URDF file
        target_frame: End-effector frame name in the URDF
        joint_names: Optional list of joint names (default: all from URDF)
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame: str = "gripper_frame_link",
        joint_names: Optional[List[str]] = None,
    ):
        try:
            from lerobot.model.kinematics import RobotKinematics

            self._kin = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name=target_frame,
                joint_names=joint_names,
            )
        except ImportError:
            raise ImportError(
                "lerobot with placo required for PlacoKinematics. " "Install with: pip install lerobot[kinematics]"
            )

        self._urdf_path = urdf_path
        logger.info(f"PlacoKinematics: urdf='{urdf_path}', frame='{target_frame}'")

    @property
    def joint_names(self) -> List[str]:
        return self._kin.joint_names

    @property
    def backend(self) -> str:
        return "placo"

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute FK. Joint positions in degrees (Placo convention)."""
        return self._kin.forward_kinematics(joint_positions)

    def inverse_kinematics(
        self,
        current_joints: np.ndarray,
        target_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        """Compute IK. Returns joint positions in degrees."""
        return self._kin.inverse_kinematics(
            current_joint_pos=current_joints,
            desired_ee_pose=target_pose,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )


class ONNXKinematics(Kinematics):
    """ONNX-based neural network kinematics solver.

    Trained from FK/IK data (e.g., from Placo or MuJoCo), then exported
    to ONNX for ~100x faster inference. Useful for real-time control loops.

    Inspired by Reachy Mini's NNKinematics pattern.

    Args:
        fk_model_path: Path to ONNX model for forward kinematics
        ik_model_path: Path to ONNX model for inverse kinematics
        joint_names: List of joint names
    """

    def __init__(
        self,
        fk_model_path: str,
        ik_model_path: Optional[str] = None,
        joint_names: Optional[List[str]] = None,
    ):
        try:
            import onnxruntime as ort

            self._ort = ort
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        self._fk_session = ort.InferenceSession(fk_model_path)
        self._ik_session = ort.InferenceSession(ik_model_path) if ik_model_path else None
        self._joint_names = joint_names or [f"joint_{i}" for i in range(6)]
        self._fk_path = fk_model_path
        self._ik_path = ik_model_path

        logger.info(f"ONNXKinematics: fk={fk_model_path}, ik={ik_model_path}")

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def backend(self) -> str:
        return "onnx"

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute FK via ONNX model.

        Input: joint angles → Output: 4x4 pose (flattened as 6D: x,y,z,roll,pitch,yaw
        then reconstructed to 4x4).
        """
        from scipy.spatial.transform import Rotation as R

        inp = joint_positions.astype(np.float32).reshape(1, -1)
        input_name = self._fk_session.get_inputs()[0].name
        output = self._fk_session.run(None, {input_name: inp})[0][0]

        # Expect output: [x, y, z, roll, pitch, yaw] or [x, y, z, qx, qy, qz, qw]
        T = np.eye(4)
        if len(output) == 6:
            T[:3, 3] = output[:3]
            T[:3, :3] = R.from_euler("xyz", output[3:6]).as_matrix()
        elif len(output) == 7:
            T[:3, 3] = output[:3]
            T[:3, :3] = R.from_quat(output[3:7]).as_matrix()
        elif len(output) == 16:
            T = output.reshape(4, 4)
        else:
            T[:3, 3] = output[:3]

        return T

    def inverse_kinematics(
        self,
        current_joints: np.ndarray,
        target_pose: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute IK via ONNX model."""
        if self._ik_session is None:
            raise RuntimeError("No IK ONNX model provided")

        from scipy.spatial.transform import Rotation as R

        # Extract pose as 6D input
        pos = target_pose[:3, 3]
        euler = R.from_matrix(target_pose[:3, :3]).as_euler("xyz")
        inp = np.concatenate([pos, euler]).astype(np.float32).reshape(1, -1)

        input_name = self._ik_session.get_inputs()[0].name
        joints = self._ik_session.run(None, {input_name: inp})[0][0]

        return joints[: len(current_joints)]

    @staticmethod
    def generate_training_data(
        kinematics: "Kinematics",
        n_samples: int = 10000,
        joint_ranges: Optional[List[tuple]] = None,
    ) -> tuple:
        """Generate FK/IK training data from any other kinematics backend.

        Returns (joint_positions, poses) for training an ONNX model.
        """
        from scipy.spatial.transform import Rotation as R

        n_joints = len(kinematics.joint_names)

        if joint_ranges is None:
            joint_ranges = [(-np.pi, np.pi)] * n_joints

        joints_data = []
        poses_data = []

        for _ in range(n_samples):
            joints = np.array([np.random.uniform(lo, hi) for lo, hi in joint_ranges])
            try:
                pose = kinematics.forward_kinematics(joints)
                pos = pose[:3, 3]
                euler = R.from_matrix(pose[:3, :3]).as_euler("xyz")
                joints_data.append(joints)
                poses_data.append(np.concatenate([pos, euler]))
            except Exception:
                continue

        return np.array(joints_data), np.array(poses_data)


def create_kinematics(
    model=None,
    data=None,
    body_name: str = "gripper",
    urdf_path: str = None,
    target_frame: str = "gripper_frame_link",
    joint_names: Optional[List[str]] = None,
    **kwargs,
) -> Kinematics:
    """Factory: create a kinematics solver with the best available backend.

    Priority:
    1. If model+data provided → MuJoCoKinematics
    2. If urdf_path provided → try Placo, fall back to MuJoCo URDF load
    3. Raise error

    Args:
        model: MuJoCo model (for sim)
        data: MuJoCo data (for sim)
        body_name: End-effector body name (MuJoCo)
        urdf_path: Path to URDF file (for placo)
        target_frame: End-effector frame name (for placo)
        joint_names: Optional joint names

    Returns:
        Kinematics solver instance
    """
    # MuJoCo backend (simulation)
    if model is not None and data is not None:
        return MuJoCoKinematics(model=model, data=data, body_name=body_name, **kwargs)

    # Placo backend (real hardware with URDF)
    if urdf_path:
        try:
            return PlacoKinematics(
                urdf_path=urdf_path,
                target_frame=target_frame,
                joint_names=joint_names,
            )
        except ImportError:
            logger.warning("Placo not available, trying MuJoCo URDF load")

            # Fallback: load URDF into MuJoCo and use MuJoCo IK
            try:
                import mujoco

                model = mujoco.MjModel.from_xml_path(urdf_path)
                data = mujoco.MjData(model)
                return MuJoCoKinematics(model=model, data=data, body_name=body_name, **kwargs)
            except Exception as e:
                raise RuntimeError(f"No kinematics backend available: {e}")

    raise ValueError("Provide either model+data (MuJoCo) or urdf_path (Placo/MuJoCo)")
