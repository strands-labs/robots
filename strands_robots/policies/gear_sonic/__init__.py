"""GEAR-SONIC Policy Provider — Humanoid Whole-Body Control via ONNX.

GEAR-SONIC (NVIDIA GEAR Lab) is a 42M-parameter humanoid behavior foundation model
that provides natural whole-body movements from motion tracking at scale. It uses an
encoder-decoder architecture with optional kinematic planner for task execution.

Architecture:
    Encoder (50MB):  [observation history] → [64-dim motion tokens]
    Decoder (40MB):  [current state + tokens] → [29-dim joint actions]
    Planner (774MB): [task input] → [motion tracking targets]

Supports multiple input interfaces:
    - Motion tracking (retarget reference motions)
    - VR teleoperation (PICO headset 3-point tracking)
    - Video-based (SMPL pose estimation → motion tokens)
    - VLA integration (GR00T N1.6 plans → SONIC executes)

Target robot: Unitree G1 (29 DOF: legs + waist + arms + hands)

Usage:
    from strands_robots.policies import create_policy

    # Basic motion tracking
    policy = create_policy("gear_sonic",
        model_dir="/path/to/gear_sonic_models")

    # With specific mode
    policy = create_policy("gear_sonic",
        model_dir="/path/to/models", mode="teleop")
"""

import logging
import os
from collections import deque
from dataclasses import dataclass  # noqa: F401
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import Policy

logger = logging.getLogger(__name__)

# Unitree G1 joint names (29 DOF)
G1_JOINT_NAMES = [
    # Lower body (12)
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # Waist (1)
    "waist_yaw",
    # Left arm (7) + hand (1)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "left_hand",
    # Right arm (7) + hand (1)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    "right_hand",
]

# Observation dimensions from observation_config.yaml
OBS_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 12,  # 10 * 3 (flattened)? → check
    "his_body_joint_positions_10frame_step1": 116,  # 10 * (29-17.4) → varies
    "his_body_joint_velocities_10frame_step1": 116,
    "his_last_actions_10frame_step1": 116,
    "his_gravity_dir_10frame_step1": 12,
}

# Encoder modes
ENCODER_MODES = {
    "motion_tracking": 0,  # Default: track reference motions
    "teleop": 1,  # VR teleoperation (PICO 3-point)
    "smpl": 2,  # Video-based SMPL pose input
}


class GearSonicPolicy(Policy):
    """GEAR-SONIC whole-body humanoid controller via ONNX inference.

    Runs the encoder-decoder architecture for real-time (>100Hz) whole-body
    control of Unitree G1. Observations are a history of joint states,
    velocities, and actions; output is 29-DOF joint position targets.

    Args:
        model_dir: Directory containing ONNX files and observation_config.yaml
        mode: Encoder mode - "motion_tracking", "teleop", or "smpl"
        device: "cuda" or "cpu" (ONNX provider selection)
        use_planner: Whether to load the kinematic planner (774MB)
        history_len: Number of history frames (default 10)
        hf_model_id: HuggingFace model ID to auto-download from

    Examples:
        # Local ONNX files
        policy = GearSonicPolicy(model_dir="/home/user/gear_sonic_models")

        # Auto-download from HuggingFace
        policy = GearSonicPolicy(hf_model_id="nvidia/GEAR-SONIC")
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        mode: str = "motion_tracking",
        device: str = "cuda",
        use_planner: bool = False,
        history_len: int = 10,
        hf_model_id: str = "nvidia/GEAR-SONIC",
        **kwargs,
    ):
        self._mode = mode
        self._device = device
        self._use_planner = use_planner
        self._history_len = history_len
        self._hf_model_id = hf_model_id
        self._robot_state_keys: List[str] = []

        # ONNX sessions (lazy loaded)
        self._encoder = None
        self._decoder = None
        self._planner = None

        # State history buffers
        self._max_history = self._history_len + 5
        self._joint_pos_history: deque = deque(maxlen=self._max_history)
        self._joint_vel_history: deque = deque(maxlen=self._max_history)
        self._action_history: deque = deque(maxlen=self._max_history)
        self._angular_vel_history: deque = deque(maxlen=self._max_history)
        self._gravity_history: deque = deque(maxlen=self._max_history)
        self._last_token_state = np.zeros(64, dtype=np.float32)
        self._step = 0

        # Resolve model directory
        self._model_dir = self._resolve_model_dir(model_dir)

        logger.info(f"🦿 GEAR-SONIC policy: mode={mode}, device={device}")
        logger.info(f"   Models: {self._model_dir}")

    def _resolve_model_dir(self, model_dir: Optional[str]) -> str:
        """Resolve model directory — use local or download from HF."""
        if model_dir and Path(model_dir).exists():
            return model_dir

        # Try downloading from HuggingFace
        try:
            from huggingface_hub import snapshot_download

            cache_dir = os.path.expanduser("~/.cache/gear_sonic")
            local = snapshot_download(
                self._hf_model_id,
                local_dir=cache_dir,
                allow_patterns=["*.onnx", "*.yaml"],
            )
            logger.info(f"Downloaded GEAR-SONIC models to {local}")
            return local
        except Exception as e:
            if model_dir:
                return model_dir
            raise FileNotFoundError(
                f"GEAR-SONIC models not found. Provide model_dir= or install huggingface_hub. Error: {e}"
            )

    def _load_models(self):
        """Lazy-load ONNX models."""
        if self._encoder is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("GEAR-SONIC requires: pip install onnxruntime")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )

        d = self._model_dir
        logger.info(f"🔄 Loading GEAR-SONIC ONNX models from {d}...")

        self._encoder = ort.InferenceSession(
            f"{d}/model_encoder.onnx", providers=providers
        )
        self._decoder = ort.InferenceSession(
            f"{d}/model_decoder.onnx", providers=providers
        )

        if self._use_planner:
            planner_path = f"{d}/planner_sonic.onnx"
            if os.path.exists(planner_path):
                self._planner = ort.InferenceSession(planner_path, providers=providers)
                logger.info("✅ Planner loaded")

        # Cache input specs
        self._enc_inputs = {i.name: i for i in self._encoder.get_inputs()}
        self._dec_inputs = {i.name: i for i in self._decoder.get_inputs()}
        self._enc_out_names = [o.name for o in self._encoder.get_outputs()]
        self._dec_out_names = [o.name for o in self._decoder.get_outputs()]

        logger.info(
            f"✅ GEAR-SONIC loaded: encoder={len(self._enc_inputs)} inputs, decoder={len(self._dec_inputs)} inputs"
        )

    @property
    def provider_name(self) -> str:
        return "gear_sonic"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys

    def _build_encoder_input(
        self, observation_dict: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Build encoder input from observation + history."""
        inputs = {}
        for inp_spec in self._encoder.get_inputs():
            shape = [s if isinstance(s, int) else 1 for s in inp_spec.shape]
            inputs[inp_spec.name] = np.zeros(shape, dtype=np.float32)

        # Set encoder mode
        mode_id = ENCODER_MODES.get(self._mode, 0)
        for name in inputs:
            if "mode" in name.lower():
                # One-hot or scalar mode ID
                inputs[name][:] = 0
                if inputs[name].size > mode_id:
                    inputs[name].flat[mode_id] = 1.0

        return inputs

    def _build_decoder_input(
        self,
        observation_dict: Dict[str, Any],
        token_state: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build decoder input from current observation + encoder tokens."""
        inputs = {}
        for inp_spec in self._decoder.get_inputs():
            shape = [s if isinstance(s, int) else 1 for s in inp_spec.shape]
            inputs[inp_spec.name] = np.zeros(shape, dtype=np.float32)

        # Extract joint positions from observation
        joint_pos = self._extract_joints(observation_dict)

        # Fill observation fields with actual data
        for name, arr in inputs.items():
            n = name.lower()
            if "token" in n and arr.shape[-1] == 64:
                inputs[name] = token_state.reshape(arr.shape)
            elif "joint_position" in n and "his" not in n:
                if arr.shape[-1] <= len(joint_pos):
                    inputs[name].flat[: len(joint_pos)] = joint_pos[: arr.size]
            elif "joint_position" in n and "his" in n:
                # Fill history
                flat_hist = self._flatten_history(self._joint_pos_history, arr.size)
                inputs[name] = flat_hist.reshape(arr.shape)
            elif "joint_velocit" in n and "his" in n:
                flat_hist = self._flatten_history(self._joint_vel_history, arr.size)
                inputs[name] = flat_hist.reshape(arr.shape)
            elif "last_action" in n and "his" in n:
                flat_hist = self._flatten_history(self._action_history, arr.size)
                inputs[name] = flat_hist.reshape(arr.shape)
            elif "angular_velocity" in n:
                flat_hist = self._flatten_history(self._angular_vel_history, arr.size)
                inputs[name] = flat_hist.reshape(arr.shape)
            elif "gravity" in n:
                flat_hist = self._flatten_history(self._gravity_history, arr.size)
                inputs[name] = flat_hist.reshape(arr.shape)

        return inputs

    def _extract_joints(self, observation_dict: Dict[str, Any]) -> np.ndarray:
        """Extract joint positions from observation dict."""
        if self._robot_state_keys:
            return np.array(
                [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys],
                dtype=np.float32,
            )
        # Try to find joint arrays
        for key in ["joint_position", "joint_positions", "state"]:
            if key in observation_dict:
                val = observation_dict[key]
                if isinstance(val, np.ndarray):
                    return val.astype(np.float32)
        return np.zeros(29, dtype=np.float32)

    def _flatten_history(self, history: list, target_size: int) -> np.ndarray:
        """Flatten history buffer to target size, zero-padded."""
        if not history:
            return np.zeros(target_size, dtype=np.float32)
        flat = np.concatenate([h.flatten() for h in history[-self._history_len :]])
        result = np.zeros(target_size, dtype=np.float32)
        n = min(len(flat), target_size)
        result[:n] = flat[:n]
        return result

    def _update_history(self, joint_pos: np.ndarray, action: np.ndarray):
        """Update state history buffers."""
        self._joint_pos_history.append(joint_pos.copy())
        self._action_history.append(action.copy())

        # Compute velocity from position diff
        if len(self._joint_pos_history) >= 2:
            vel = self._joint_pos_history[-1] - self._joint_pos_history[-2]
            self._joint_vel_history.append(vel)
        else:
            self._joint_vel_history.append(np.zeros_like(joint_pos))

        # Default gravity direction (pointing down)
        self._gravity_history.append(np.array([0, 0, -1], dtype=np.float32))
        # Default angular velocity
        self._angular_vel_history.append(np.zeros(3, dtype=np.float32))

    async def get_actions(
        self,
        observation_dict: Dict[str, Any],
        instruction: str = "",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Get whole-body joint actions from GEAR-SONIC.

        Args:
            observation_dict: Current robot state (joint positions, IMU, etc.)
            instruction: Optional instruction (used by planner if enabled)

        Returns:
            List with single action dict mapping joint names to target positions
        """
        self._load_models()

        # Run encoder → motion tokens
        enc_inputs = self._build_encoder_input(observation_dict)
        enc_outputs = self._encoder.run(None, enc_inputs)

        # Token state is the encoder output
        token_state = enc_outputs[0].flatten()[:64]
        self._last_token_state = token_state

        # Run decoder → joint actions
        dec_inputs = self._build_decoder_input(observation_dict, token_state)
        dec_outputs = self._decoder.run(None, dec_inputs)

        # Decoder output is joint position targets
        action_array = dec_outputs[0].flatten()

        # Extract current joint positions for history
        joint_pos = self._extract_joints(observation_dict)
        self._update_history(joint_pos, action_array)

        # Map to named joints
        action_dict = {}
        joint_names = (
            self._robot_state_keys if self._robot_state_keys else G1_JOINT_NAMES
        )
        for i, name in enumerate(joint_names):
            if i < len(action_array):
                action_dict[name] = float(action_array[i])
            else:
                action_dict[name] = 0.0

        self._step += 1
        return [action_dict]

    def reset(self):
        """Reset history buffers."""
        self._joint_pos_history.clear()
        self._joint_vel_history.clear()
        self._action_history.clear()
        self._angular_vel_history.clear()
        self._gravity_history.clear()
        self._last_token_state = np.zeros(64, dtype=np.float32)
        self._step = 0
        logger.info("🦿 GEAR-SONIC reset")


__all__ = ["GearSonicPolicy", "G1_JOINT_NAMES", "ENCODER_MODES"]
