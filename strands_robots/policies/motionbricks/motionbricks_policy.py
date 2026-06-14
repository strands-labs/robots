"""MotionBricksPolicy - real-time generative locomotion for G1 humanoid.

Wraps the MotionBricks inference pipeline from NVlabs/GR00T-WholeBodyControl.
Three-model architecture (VQVAE + pose + root) generates full-body 29-DOF
qpos at MuJoCo timestep rate. Supports 12+ locomotion styles controlled via
a single `style` parameter.

Unlike WBCPolicy (ONNX, 15-dim torque output, deploy-grade), MotionBricks
outputs full 29-DOF joint positions directly (demo-grade, expressive motion).

Control interface:
    target_velocity: [vx, vy, wz] - movement direction + speed
    target_orientation: float - facing direction (radians, world frame)
    style: str - locomotion style name (see MOTIONBRICKS_STYLES)

Output:
    29-DOF qpos for the G1 humanoid (direct position targets).
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)

# 29 joints of the G1 humanoid in MuJoCo qpos order (after the 7-dim root).
# These are the joints MotionBricks generates targets for.
MOTIONBRICKS_JOINT_NAMES: list[str] = [
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
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# Full G1 qpos dimensionality (7 root + 29 joints = 36 total in MuJoCo)
_N_JOINTS = 29
_QPOS_DIM = 36  # root (3 pos + 4 quat) + 29 joints
_ROOT_DIM = 7
_FPS = 30  # MotionBricks native framerate
_FRAMES_PER_TOKEN = 4
_GENERATE_DT_DEFAULT = 2.0  # default frame generation lookahead


class MotionBricksStyle(StrEnum):
    """Supported locomotion styles for MotionBricks G1."""

    IDLE = "idle"
    SLOW_WALK = "slow_walk"
    WALK = "walk"
    HAND_CRAWLING = "hand_crawling"
    WALK_BOXING = "walk_boxing"
    ELBOW_CRAWLING = "elbow_crawling"
    STEALTH_WALK = "stealth_walk"
    INJURED_WALK = "injured_walk"
    WALK_STEALTH = "walk_stealth"
    WALK_HAPPY_DANCE = "walk_happy_dance"
    WALK_ZOMBIE = "walk_zombie"
    WALK_GUN = "walk_gun"
    WALK_SCARED = "walk_scared"
    WALK_LEFT = "walk_left"
    WALK_RIGHT = "walk_right"


# Convenience list of all style names
MOTIONBRICKS_STYLES: list[str] = [s.value for s in MotionBricksStyle]

# Default standing qpos for G1 (29 joints, radians). Used when no model is loaded.
_DEFAULT_JOINT_POS = np.zeros(_N_JOINTS, dtype=np.float32)

# Velocity threshold below which we select "idle" style automatically
_IDLE_THRESHOLD = 0.05


class MotionBricksPolicy(Policy):
    """Real-time generative locomotion policy for G1 humanoid.

    Wraps the MotionBricks VQVAE + pose + root model pipeline to produce
    expressive full-body motion. Outputs 29-DOF joint positions directly
    (not torques) at each control tick.

    Args:
        checkpoint: Path to checkpoint directory or HuggingFace repo ID.
            Must contain: motionbricks_vqvae/, motionbricks_pose/,
            motionbricks_root/, G1-clip.ckpt, and the G1 skeleton XML.
        style: Default locomotion style. Can be overridden per-call via kwargs.
        speed_scale: Speed scale range [min, max] (default [0.8, 1.2]).
        device: Compute device - "cuda" (default, required for models) or "cpu".
        generate_dt: Frame generation lookahead multiplier (default 2.0).
        allow_missing_models: If True, policy constructs without models
            (for offline testing). If False (default), raises RuntimeError
            when models cannot be loaded.

    Raises:
        RuntimeError: If models cannot be loaded and allow_missing_models is False.
        ImportError: If torch is not installed.
    """

    def __init__(
        self,
        checkpoint: str = "nvidia/MotionBricks-G1",
        style: str = "walk",
        speed_scale: list[float] | None = None,
        device: str = "cuda",
        generate_dt: float = _GENERATE_DT_DEFAULT,
        allow_missing_models: bool = False,
        **kwargs: Any,
    ) -> None:
        self._checkpoint = checkpoint
        self._default_style = style
        self._speed_scale = speed_scale or [0.8, 1.2]
        self._device = device
        self._generate_dt = generate_dt
        self._robot_state_keys: list[str] = []

        # Inference state (populated by _load_models)
        self._agent: Any = None  # full_navigation_agent instance
        self._controller_dt: float = _FRAMES_PER_TOKEN / _FPS

        # Frame buffer for multi-frame generation
        self._frame_buffer: list[np.ndarray] = []
        self._frame_idx: int = 0

        # Load models
        self._load_models(allow_missing=allow_missing_models)

        logger.info(
            "MotionBricksPolicy initialized: checkpoint=%s, style=%s, device=%s",
            checkpoint,
            style,
            device,
        )

    def _load_models(self, allow_missing: bool = False) -> None:
        """Load the MotionBricks model pipeline.

        Requires: torch, motionbricks package (or local checkout).
        Models are loaded from checkpoint path (local dir or HF download).

        Args:
            allow_missing: If True, log warning and continue without models.
                If False, raise RuntimeError on failure.

        Raises:
            RuntimeError: When models fail to load and allow_missing is False.
            ImportError: When torch is not available.
        """
        try:
            import torch  # noqa: F401 - verify availability
        except ImportError as e:
            if allow_missing:
                logger.warning(
                    "PyTorch not available. MotionBricksPolicy will return "
                    "default poses. Install with: pip install torch"
                )
                return
            raise ImportError("PyTorch is required for MotionBricksPolicy. Install with: pip install torch") from e

        # Resolve checkpoint path
        checkpoint_path = self._resolve_checkpoint()
        if checkpoint_path is None:
            if allow_missing:
                logger.warning(
                    "MotionBricks checkpoint not found at '%s'. "
                    "Policy will return default joint positions. "
                    "Clone the checkpoint with: "
                    "git clone https://github.com/NVlabs/GR00T-WholeBodyControl",
                    self._checkpoint,
                )
                return
            raise RuntimeError(
                f"MotionBricks checkpoint not found at '{self._checkpoint}'. "
                f"Download the models first. See: "
                f"https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks"
            )

        # Try loading the motionbricks inference pipeline
        try:
            self._init_inference_pipeline(checkpoint_path)
        except (OSError, ImportError, ValueError, KeyError) as e:
            if allow_missing:
                logger.warning(
                    "Failed to load MotionBricks models from '%s': %s. Policy will return default joint positions.",
                    checkpoint_path,
                    e,
                )
                return
            raise RuntimeError(f"Failed to load MotionBricks models from '{checkpoint_path}': {e}") from e

    def _resolve_checkpoint(self) -> Path | None:
        """Resolve checkpoint path from string (local path or HF repo ID).

        Returns:
            Path to the motionbricks directory if found, None otherwise.
        """
        # Check if it's a local path
        local = Path(self._checkpoint)
        if local.is_dir():
            # Could be the root repo or the motionbricks subdir
            if (local / "motionbricks").is_dir():
                return local / "motionbricks"
            if (local / "motionbricks_vqvae").is_dir():
                return local
            # Check for the standard structure
            if (local / "out" / "G1-clip.ckpt").exists():
                return local
            return local

        # Check common cache locations
        cache_paths = [
            Path.home() / ".cache" / "strands_robots" / "motionbricks",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--MotionBricks-G1",
        ]
        for p in cache_paths:
            if p.is_dir():
                return p

        # Try HuggingFace download
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                self._checkpoint,
                cache_dir=str(Path.home() / ".cache" / "strands_robots" / "motionbricks"),
            )
            return Path(path)
        except (OSError, ImportError, ValueError, ConnectionError):
            pass

        return None

    def _init_inference_pipeline(self, checkpoint_path: Path) -> None:
        """Initialize the MotionBricks inference pipeline from checkpoint.

        This imports and initializes the motionbricks navigation_demo agent.
        Requires the motionbricks package to be importable (either installed
        or the checkout added to sys.path).

        Args:
            checkpoint_path: Resolved path to the motionbricks directory.
        """
        import sys

        import torch

        # Add motionbricks to path if not already importable
        mb_root = checkpoint_path
        if (mb_root / "motionbricks").is_dir() and (mb_root / "setup.py").exists():
            sys.path.insert(0, str(mb_root))

        from types import SimpleNamespace

        from motionbricks.motion_backbone.demo.utils import navigation_demo

        args = SimpleNamespace(
            humanoid_xml=str(mb_root / "assets" / "skeletons" / "g1" / "scene_29dof.xml"),
            result_dir=str(mb_root / "out"),
            data_root=str(mb_root / "datasets"),
            explicit_dataset_folder=str(mb_root / "datasets" / "motionbricks-G1"),
            clips_ckpt=str(mb_root / "out" / "G1-clip.ckpt"),
            reprocess_clips=0,
            controller="random",  # programmatic control, not keyboard
            lookat_movement_direction=0,
            has_viewer=0,
            pre_filter_qpos=1,
            source_root_realignment=1,
            target_root_realignment=1,
            force_canonicalization=1,
            skip_ending_target_cond=0,
            random_speed_scale=0,
            speed_scale=self._speed_scale,
            generate_dt=self._generate_dt,
            max_steps=10000,
            random_seed=1234,
            num_runs=1,
            use_qpos=1,
            planner="default",
            allowed_mode=None,
            clips="G1",
            return_model_configs=True,
            return_dataloader=True,
            recording_dir=None,
            EXP="default",
        )

        # Initialize the demo agent on the configured device
        with torch.device(self._device):
            self._agent = navigation_demo(args)
            self._agent.full_agent.reset()

        logger.info("MotionBricks inference pipeline initialized on %s", self._device)

    @property
    def requires_images(self) -> bool:
        """MotionBricks is state-only (no camera frames needed)."""
        return False

    @property
    def provider_name(self) -> str:
        """Provider name for identification."""
        return "motionbricks"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys.

        Args:
            robot_state_keys: List of joint names the robot expects.
        """
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state.

        Clears the frame buffer and resets the internal motion agent.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self._frame_buffer.clear()
        self._frame_idx = 0
        if self._agent is not None:
            import numpy as np_
            import torch

            if seed is not None:
                np_.random.seed(seed)
                torch.manual_seed(seed)
            self._agent.full_agent.reset()

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generate next action(s) from the MotionBricks model.

        Args:
            observation_dict: Robot observation. Used for current qpos if agent
                is not initialized (fallback mode). When the full pipeline is
                active, the internal agent maintains its own state.
            instruction: Ignored (MotionBricks uses kwargs for control).
            **kwargs: Control parameters:
                - target_velocity: [vx, vy, wz] locomotion command
                - target_orientation: float, facing direction (radians)
                - style: str, override the default locomotion style

        Returns:
            List with one action dict mapping joint names to target positions.
        """
        target_velocity = kwargs.get("target_velocity", [0.0, 0.0, 0.0])
        target_orientation = kwargs.get("target_orientation", 0.0)
        style = kwargs.get("style", self._default_style)

        # If no model loaded, return default positions
        if self._agent is None:
            return self._default_action()

        # Generate qpos from the motion model
        qpos = self._generate_frame(target_velocity, target_orientation, style)

        # Extract joint positions (skip the 7-dim root: 3 pos + 4 quat)
        joint_positions = qpos[_ROOT_DIM : _ROOT_DIM + _N_JOINTS]

        # Map to robot state keys
        action: dict[str, Any] = {}
        if self._robot_state_keys:
            for i, key in enumerate(self._robot_state_keys):
                if i < len(joint_positions):
                    action[key] = float(joint_positions[i])
        else:
            # Use default joint names
            for i, name in enumerate(MOTIONBRICKS_JOINT_NAMES):
                if i < len(joint_positions):
                    action[name] = float(joint_positions[i])

        return [action]

    def _generate_frame(
        self,
        target_velocity: list[float],
        target_orientation: float,
        style: str,
    ) -> np.ndarray:
        """Generate the next qpos frame from the MotionBricks agent.

        Uses the internal frame buffer. When the buffer is empty, generates
        a new batch of frames from the model using the given control signals.

        Args:
            target_velocity: [vx, vy, wz] locomotion command.
            target_orientation: Facing direction in radians.
            style: Locomotion style name.

        Returns:
            Full 36-dim qpos array (7 root + 29 joints).
        """
        import torch

        # If we have buffered frames, consume one
        if self._frame_buffer:
            frame = self._frame_buffer.pop(0)
            return frame

        # Generate new frames
        agent = self._agent.full_agent

        # Get current frame
        qpos = agent.get_next_frame()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32).flatten()

        # Get context for next generation
        context_mujoco_qpos = agent.get_context_mujoco_qpos()

        # Build control signals
        control_signals = self._build_control_signals(target_velocity, target_orientation, style)
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos

        # Generate new frames
        with torch.no_grad():
            agent.generate_new_frames(
                control_signals,
                self._controller_dt * self._generate_dt,
            )

        return qpos

    def _build_control_signals(
        self,
        target_velocity: list[float],
        target_orientation: float,
        style: str,
    ) -> dict[str, Any]:
        """Build control signals dict for the MotionBricks agent.

        Maps our API (target_velocity, orientation, style) to the internal
        control signal format (movement_direction, facing_direction, mode).

        Args:
            target_velocity: [vx, vy, wz] - forward, lateral, yaw rate.
            target_orientation: Desired facing direction (radians).
            style: Locomotion style name.

        Returns:
            Dict of control signals for the full_navigation_agent.
        """
        import torch

        vx = target_velocity[0] if len(target_velocity) > 0 else 0.0
        vy = target_velocity[1] if len(target_velocity) > 1 else 0.0
        wz = target_velocity[2] if len(target_velocity) > 2 else 0.0

        # Compute movement direction (2D unit vector in ground plane)
        speed = np.sqrt(vx * vx + vy * vy)
        if speed > _IDLE_THRESHOLD:
            movement_direction = np.array([vx / speed, vy / speed, 0.0], dtype=np.float32)
        else:
            movement_direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Facing direction incorporates yaw rate as an angular offset
        facing_direction = np.array(
            [np.cos(target_orientation + wz), np.sin(target_orientation + wz), 0.0],
            dtype=np.float32,
        )

        # Resolve style to mode index
        resolved_style = style if speed > _IDLE_THRESHOLD else "idle"
        if resolved_style not in MOTIONBRICKS_STYLES:
            logger.warning(
                "Unknown style '%s', falling back to 'walk'. Available styles: %s",
                resolved_style,
                MOTIONBRICKS_STYLES,
            )
            resolved_style = "walk"
        mode_idx = MOTIONBRICKS_STYLES.index(resolved_style)

        # Build the signals tensor dict
        control_signals: dict[str, Any] = {
            "movement_direction": torch.from_numpy(movement_direction).view(1, -1),
            "facing_direction": torch.from_numpy(facing_direction).view(1, -1),
            "mode": torch.tensor([mode_idx]).view(1, -1),
        }

        # Allowed prediction tokens (style-dependent, use defaults)
        allowed = torch.ones(11, dtype=torch.int).view(1, -1)
        control_signals["allowed_pred_num_tokens"] = allowed

        return control_signals

    def _default_action(self) -> list[dict[str, Any]]:
        """Return default standing pose when no model is loaded.

        Returns:
            List with one action dict of zero joint positions.
        """
        action: dict[str, Any] = {}
        keys = self._robot_state_keys or MOTIONBRICKS_JOINT_NAMES
        for key in keys:
            action[key] = 0.0
        return [action]
