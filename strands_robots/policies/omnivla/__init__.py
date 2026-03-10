#!/usr/bin/env python3
"""
OmniVLA Policy Provider — Omni-Modal Vision-Language-Action for Robot Navigation.

OmniVLA (UC Berkeley + Toyota) is a 7B VLA model built on OpenVLA that generates
2D navigation trajectories from omni-modal inputs: camera images, GPS poses,
goal images, satellite maps, and/or language prompts.

Supports two model variants:
- **omnivla** (full): 7B param OpenVLA backbone + action head + pose projector.
  Requires GPU with ~20GB VRAM. Runs at ~3Hz.
- **omnivla-edge**: Lightweight EfficientNet-B0 + CLIP backbone.
  Runs on edge devices (~1GB VRAM). Faster inference.

Architecture:
    Current Image + Goal (image/GPS/language/satellite)
        → OmniVLA (OpenVLA backbone + L1 Regression Head)
        → 2D Waypoint Trajectory (8 waypoints × 4: dx, dy, cos θ, sin θ)
        → PD Controller → (linear_vel, angular_vel)

Output format: velocity commands (linear, angular) for mobile robot navigation.
Unlike arm policies that output joint positions, OmniVLA outputs navigation
velocity commands — designed for wheeled/legged mobile robots.

Checkpoints (HuggingFace):
    - NHirose/omnivla-original (paper submission)
    - NHirose/omnivla-original-balance (data-balanced training)
    - NHirose/omnivla-finetuned-cast (finetuned on CAST dataset)
    - NHirose/omnivla-edge (lightweight edge model)

Requirements:
    - OmniVLA repo: pip install -e git+https://github.com/cagataycali/OmniVLA.git
    - torch, transformers, utm, Pillow
    - GPU recommended (CPU possible but slow)

Usage:
    from strands_robots.policies import create_policy

    # Full model with language goal
    policy = create_policy(
        "omnivla",
        checkpoint_path="./omnivla-original",
        goal_modality="language",
        instruction="move toward the blue trash bin",
    )

    # Edge model with image goal
    policy = create_policy(
        "omnivla",
        variant="edge",
        checkpoint_path="./omnivla-edge",
        goal_modality="image",
        goal_image_path="./goal.jpg",
    )

    # Get velocity commands
    actions = await policy.get_actions(observation_dict, "go to the door")
    # Returns: [{"linear_vel": 0.2, "angular_vel": -0.1, "waypoints": [...]}]

Reference:
    @misc{hirose2025omnivla,
        title={OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation},
        author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
        year={2025}, eprint={2509.19480}, archivePrefix={arXiv},
    }
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

# OmniVLA constants (from prismatic.vla.constants)
_DEFAULT_ACTION_DIM = 4  # dx, dy, cos(heading), sin(heading)
_DEFAULT_NUM_ACTIONS_CHUNK = 8  # 8 waypoints per prediction
_DEFAULT_POSE_DIM = 4  # relative_y, relative_x, cos(heading_diff), sin(heading_diff)
_DEFAULT_METRIC_WAYPOINT_SPACING = 0.1
_DEFAULT_WAYPOINT_SELECT = 4  # which waypoint in the trajectory to use for velocity
_DEFAULT_MAX_LINEAR_VEL = 0.3
_DEFAULT_MAX_ANGULAR_VEL = 0.3
_DEFAULT_TICK_RATE = 3  # Hz


class OmnivlaPolicy(Policy):
    """OmniVLA — Omni-Modal VLA for Robot Navigation.

    Generates 2D navigation trajectories from multi-modal inputs.
    Outputs (linear_vel, angular_vel) velocity commands for mobile robots.

    Supports 9 input modality combinations:
        0: satellite only
        1: pose + satellite
        2: satellite + image
        3: all (pose + satellite + image)
        4: pose only
        5: pose + image
        6: image only
        7: language only
        8: language + pose

    The modality is selected based on which goals are provided:
    - goal_modality="language" → modality 7
    - goal_modality="image" → modality 6
    - goal_modality="pose" → modality 4
    - Or auto-detected from available observation data.
    """

    def __init__(
        self,
        checkpoint_path: str = "./omnivla-original",
        variant: str = "full",
        goal_modality: str = "language",
        instruction: str = "",
        goal_image_path: Optional[str] = None,
        goal_gps: Optional[Tuple[float, float, float]] = None,
        resume_step: Optional[int] = 120000,
        device: Optional[str] = None,
        waypoint_select: int = _DEFAULT_WAYPOINT_SELECT,
        max_linear_vel: float = _DEFAULT_MAX_LINEAR_VEL,
        max_angular_vel: float = _DEFAULT_MAX_ANGULAR_VEL,
        metric_waypoint_spacing: float = _DEFAULT_METRIC_WAYPOINT_SPACING,
        tick_rate: float = _DEFAULT_TICK_RATE,
        use_lora: bool = True,
        lora_rank: int = 32,
        **kwargs,
    ):
        """Initialize OmniVLA policy.

        .. warning:: Security — trusted paths only
            The checkpoint_path directory may be added to sys.path during edge model
            loading (for utils_policy imports). Only use checkpoint paths from trusted
            sources. See :meth:`_load_edge_model` for details.

        Args:
            checkpoint_path: Path to model checkpoint directory (local or will be downloaded)
            variant: Model variant — "full" (7B) or "edge" (lightweight)
            goal_modality: Default goal type — "language", "image", "pose", "satellite", "all"
            instruction: Default language instruction
            goal_image_path: Path to goal image (for image-based goals)
            goal_gps: Goal GPS as (lat, lon, compass_degrees) tuple
            resume_step: Checkpoint step to load (default: 120000)
            device: Device ("cuda", "cpu", "mps", or auto-detect)
            waypoint_select: Which waypoint index to use for velocity (0-7, default: 4)
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            metric_waypoint_spacing: Spacing between waypoints (meters)
            tick_rate: Control loop frequency (Hz)
            use_lora: Whether to use LoRA (for full model)
            lora_rank: LoRA rank (for full model)
        """
        self._checkpoint_path = checkpoint_path
        self._variant = variant.lower()
        self._goal_modality = goal_modality.lower()
        self._instruction = instruction
        self._goal_image_path = goal_image_path
        self._goal_gps = goal_gps
        self._resume_step = resume_step
        self._requested_device = device
        self._waypoint_select = waypoint_select
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._metric_waypoint_spacing = metric_waypoint_spacing
        self._tick_rate = tick_rate
        self._use_lora = use_lora
        self._lora_rank = lora_rank

        self._robot_state_keys: List[str] = []

        # Model state (lazy loaded)
        self._loaded = False
        self._device = None
        self._step = 0

        # Full model components
        self._vla = None
        self._action_head = None
        self._pose_projector = None
        self._action_tokenizer = None
        self._processor = None
        self._num_patches = None

        # Edge model components
        self._edge_model = None
        self._text_encoder = None

        # Goal state
        self._goal_image_pil = None
        self._goal_utm = None
        self._goal_compass_rad = None

        logger.info(
            f"🧭 OmniVLA policy initialized: variant={self._variant}, "
            f"modality={self._goal_modality}, checkpoint={self._checkpoint_path}"
        )

    @property
    def provider_name(self) -> str:
        return "omnivla"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        logger.info(f"🔧 OmniVLA robot state keys: {self._robot_state_keys}")

    # =========================================================================
    # Lazy Model Loading
    # =========================================================================

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        if self._variant == "edge":
            self._load_edge_model()
        else:
            self._load_full_model()

        # Load goal image if path provided
        if self._goal_image_path and os.path.exists(self._goal_image_path):
            from PIL import Image

            self._goal_image_pil = Image.open(self._goal_image_path).convert("RGB")
            logger.info(f"🧭 Goal image loaded: {self._goal_image_path}")

        # Convert goal GPS to UTM if provided
        if self._goal_gps:
            try:
                import utm

                lat, lon, compass_deg = self._goal_gps
                self._goal_utm = utm.from_latlon(lat, lon)
                self._goal_compass_rad = -float(compass_deg) / 180.0 * math.pi
                logger.info(f"🧭 Goal GPS set: {self._goal_gps}")
            except ImportError:
                logger.warning("utm package not installed — GPS goals unavailable")

        self._loaded = True

    def _load_full_model(self):
        """Load the full 7B OmniVLA model (OpenVLA backbone)."""
        import torch

        logger.info(f"🧭 Loading OmniVLA full model from {self._checkpoint_path}...")

        # Device selection
        if self._requested_device:
            self._device = torch.device(self._requested_device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        try:
            # Import OmniVLA components
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
            from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            from prismatic.models.action_heads import L1RegressionActionHead_idcat
            from prismatic.models.projectors import ProprioProjector
            from prismatic.vla.action_tokenizer import ActionTokenizer
            from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM  # noqa: F401
            from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

            # Register OpenVLA model classes
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

            # Load processor and VLA
            self._processor = AutoProcessor.from_pretrained(self._checkpoint_path, trust_remote_code=True)
            self._vla = AutoModelForVision2Seq.from_pretrained(
                self._checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(self._device)

            self._vla.vision_backbone.set_num_images_in_input(2)
            self._vla.to(dtype=torch.bfloat16, device=self._device)

            # Load pose projector
            self._pose_projector = ProprioProjector(llm_dim=self._vla.llm_dim, proprio_dim=POSE_DIM)
            self._load_checkpoint_module(self._pose_projector, "pose_projector")
            self._pose_projector = self._pose_projector.to(self._device)

            # Load action head
            self._action_head = L1RegressionActionHead_idcat(
                input_dim=self._vla.llm_dim,
                hidden_dim=self._vla.llm_dim,
                action_dim=ACTION_DIM,
            )
            self._load_checkpoint_module(self._action_head, "action_head", to_bf16=True)
            self._action_head = self._action_head.to(self._device)

            # Compute vision patches
            self._num_patches = (
                self._vla.vision_backbone.get_num_patches() * self._vla.vision_backbone.get_num_images_in_input()
                + 1  # +1 for goal pose
            )

            # Action tokenizer
            self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)

            n_params = sum(p.numel() for p in self._vla.parameters())
            logger.info(f"🧭 OmniVLA full model loaded: {n_params/1e6:.0f}M params on {self._device}")

        except ImportError as e:
            raise ImportError(
                f"OmniVLA full model requires the prismatic package. "
                f"Install with: pip install -e git+https://github.com/Embodied-Reasoner/OmniVLA.git\n"
                f"Error: {e}"
            )

    def _load_edge_model(self):
        """Load the lightweight OmniVLA-edge model.

        .. warning:: Security — sys.path modification
            This method inserts ``checkpoint_path`` and its ``inference/`` subdirectory
            into ``sys.path`` so that OmniVLA's ``utils_policy`` module can be imported.
            This affects all subsequent imports in the process. Only use checkpoint paths
            from trusted sources — a malicious checkpoint directory could shadow standard
            library or third-party modules.
        """
        import torch

        logger.info(f"🧭 Loading OmniVLA-edge from {self._checkpoint_path}...")

        # Device selection
        if self._requested_device:
            self._device = torch.device(self._requested_device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        try:
            # Edge model uses a custom loader
            import sys

            # NOTE(security): This inserts user-provided checkpoint_path into sys.path.
            # The checkpoint directory must be trusted — it can contain arbitrary Python
            # modules that will be importable by the entire process. This is inherent to
            # the OmniVLA "load from local path" pattern (requires utils_policy.py).
            omnivla_dir = os.path.dirname(self._checkpoint_path)
            if omnivla_dir not in sys.path:
                sys.path.insert(0, omnivla_dir)
            inference_dir = os.path.join(omnivla_dir, "inference")
            if os.path.isdir(inference_dir) and inference_dir not in sys.path:
                sys.path.insert(0, inference_dir)

            from utils_policy import load_model

            model_params = {
                "model_type": "omnivla-edge",
                "len_traj_pred": 8,
                "learn_angle": True,
                "context_size": 5,
                "obs_encoder": "efficientnet-b0",
                "encoding_size": 256,
                "obs_encoding_size": 1024,
                "goal_encoding_size": 1024,
                "late_fusion": False,
                "mha_num_attention_heads": 4,
                "mha_num_attention_layers": 4,
                "mha_ff_dim_factor": 4,
                "clip_type": "ViT-B/32",
            }

            ckpt_path = os.path.join(self._checkpoint_path, "omnivla-edge.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Edge model weights not found: {ckpt_path}. "
                    f"Download with: git clone https://huggingface.co/NHirose/omnivla-edge"
                )

            self._edge_model, self._text_encoder, _ = load_model(ckpt_path, model_params, self._device)
            self._text_encoder = self._text_encoder.to(self._device).eval()
            self._edge_model = self._edge_model.to(self._device).eval()

            logger.info(f"🧭 OmniVLA-edge loaded on {self._device}")

        except ImportError as e:
            raise ImportError(
                f"OmniVLA-edge requires the OmniVLA package with utils_policy. "
                f"Install with: pip install -e git+https://github.com/Embodied-Reasoner/OmniVLA.git\n"
                f"Error: {e}"
            )

    def _load_checkpoint_module(self, module, name, to_bf16=False):
        """Load a checkpoint for a specific module (pose_projector, action_head)."""
        import torch

        # Try both naming conventions
        for mod_name in [name, name.replace("pose_", "proprio_")]:
            ckpt_path = os.path.join(self._checkpoint_path, f"{mod_name}--{self._resume_step}_checkpoint.pt")
            if os.path.exists(ckpt_path):
                logger.info(f"🧭 Loading checkpoint: {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Remove DDP prefix
                state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
                module.load_state_dict(state_dict)
                if to_bf16:
                    module = module.to(torch.bfloat16)
                return

        logger.warning(f"🧭 Checkpoint not found for {name} at step {self._resume_step}")

    # =========================================================================
    # Goal Modality Resolution
    # =========================================================================

    def _resolve_modality_flags(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        modality_override: Optional[str] = None,
    ) -> Tuple[bool, bool, bool, bool]:
        """Resolve which modality flags to set based on observation + goal config.

        Args:
            observation_dict: Current observations
            instruction: Language instruction
            modality_override: Override for goal modality (from kwargs)

        Returns:
            (pose_goal, satellite, image_goal, lan_prompt) booleans
        """
        modality = modality_override or self._goal_modality

        # Auto-detect from observation if "auto"
        if modality == "auto":
            has_gps = "gps" in observation_dict or "latitude" in observation_dict or self._goal_utm is not None
            has_goal_image = self._goal_image_pil is not None or "goal_image" in observation_dict
            has_instruction = bool(instruction) or bool(self._instruction)
            has_satellite = "satellite" in observation_dict

            return (has_gps, has_satellite, has_goal_image, has_instruction)

        # Explicit modality selection
        modality_map = {
            "language": (False, False, False, True),  # 7
            "image": (False, False, True, False),  # 6
            "pose": (True, False, False, False),  # 4
            "satellite": (False, True, False, False),  # 0
            "pose_image": (True, False, True, False),  # 5
            "pose_satellite": (True, True, False, False),  # 1
            "satellite_image": (False, True, True, False),  # 2
            "all": (True, True, True, False),  # 3
            "language_pose": (True, False, False, True),  # 8
        }

        return modality_map.get(modality, (False, False, False, True))

    def _get_modality_id(
        self,
        pose_goal: bool,
        satellite: bool,
        image_goal: bool,
        lan_prompt: bool,
    ) -> int:
        """Map modality flags to OmniVLA's modality_id integer."""
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            return 0
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            return 1
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            return 2
        elif satellite and not lan_prompt and pose_goal and image_goal:
            return 3
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            return 4
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            return 5
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            return 6
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            return 7
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            return 8
        else:
            # Default to language-only
            return 7

    # =========================================================================
    # Velocity Controller
    # =========================================================================

    def _waypoints_to_velocity(self, waypoints: np.ndarray) -> Tuple[float, float]:
        """Convert waypoint trajectory to (linear_vel, angular_vel) via PD controller.

        Args:
            waypoints: (N, 4) array of [dx, dy, cos(heading), sin(heading)]

        Returns:
            (linear_vel, angular_vel) clipped to max limits
        """
        # Select target waypoint
        idx = min(self._waypoint_select, len(waypoints) - 1)
        chosen = waypoints[idx].copy()
        chosen[:2] *= self._metric_waypoint_spacing

        dx, dy, hx, hy = chosen
        EPS = 1e-8
        DT = 1.0 / self._tick_rate

        # PD controller
        if abs(dx) < EPS and abs(dy) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * self._clip_angle(math.atan2(hy, hx)) / DT
        elif abs(dx) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * np.sign(dy) * math.pi / (2 * DT)
        else:
            linear_vel = dx / DT
            angular_vel = math.atan(dy / dx) / DT

        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -1.0, 1.0)

        # Velocity limiting
        maxv = self._max_linear_vel
        maxw = self._max_angular_vel

        if abs(linear_vel) <= maxv:
            if abs(angular_vel) <= maxw:
                return float(linear_vel), float(angular_vel)
            else:
                rd = linear_vel / angular_vel
                return float(maxw * np.sign(linear_vel) * abs(rd)), float(maxw * np.sign(angular_vel))
        else:
            if abs(angular_vel) <= 0.001:
                return float(maxv * np.sign(linear_vel)), 0.0
            else:
                rd = linear_vel / angular_vel
                if abs(rd) >= maxv / maxw:
                    return float(maxv * np.sign(linear_vel)), float(maxv * np.sign(angular_vel) / abs(rd))
                else:
                    return float(maxw * np.sign(linear_vel) * abs(rd)), float(maxw * np.sign(angular_vel))

    @staticmethod
    def _clip_angle(angle: float) -> float:
        """Clip angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # =========================================================================
    # GPS Utilities
    # =========================================================================

    def _compute_goal_pose_norm(
        self,
        observation_dict: Dict[str, Any],
    ) -> np.ndarray:
        """Compute normalized goal pose relative to current position.

        Returns:
            (4,) array: [relative_y/spacing, -relative_x/spacing, cos(heading_diff), sin(heading_diff)]
        """
        thres_dist = 30.0

        # Current GPS
        cur_lat = observation_dict.get("latitude", observation_dict.get("gps_lat", 0.0))
        cur_lon = observation_dict.get("longitude", observation_dict.get("gps_lon", 0.0))
        cur_compass_deg = observation_dict.get("compass", observation_dict.get("heading", 0.0))

        try:
            import utm

            cur_utm = utm.from_latlon(cur_lat, cur_lon)
            cur_compass_rad = -float(cur_compass_deg) / 180.0 * math.pi

            if self._goal_utm is None:
                return np.zeros(4, dtype=np.float32)

            # Relative position
            delta_x = self._goal_utm[0] - cur_utm[0]
            delta_y = self._goal_utm[1] - cur_utm[1]

            # Rotate to local frame
            rel_x = delta_x * math.cos(cur_compass_rad) + delta_y * math.sin(cur_compass_rad)
            rel_y = -delta_x * math.sin(cur_compass_rad) + delta_y * math.cos(cur_compass_rad)

            # Clip distance
            radius = math.sqrt(rel_x**2 + rel_y**2)
            if radius > thres_dist:
                rel_x *= thres_dist / radius
                rel_y *= thres_dist / radius

            goal_compass = self._goal_compass_rad or 0.0
            return np.array(
                [
                    rel_y / self._metric_waypoint_spacing,
                    -rel_x / self._metric_waypoint_spacing,
                    math.cos(goal_compass - cur_compass_rad),
                    math.sin(goal_compass - cur_compass_rad),
                ],
                dtype=np.float32,
            )

        except ImportError:
            logger.warning("utm not installed — returning zero goal pose")
            return np.zeros(4, dtype=np.float32)

    # =========================================================================
    # Main Inference
    # =========================================================================

    async def get_actions(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Get navigation velocity commands from OmniVLA.

        Args:
            observation_dict: Robot observation containing:
                - Camera image(s) as numpy arrays (any key with "camera" or "image")
                - Optional GPS: "latitude", "longitude", "compass"
                - Optional goal_image: numpy array under "goal_image" key
                - Optional satellite: numpy array under "satellite" key
            instruction: Natural language navigation instruction
            **kwargs: Override goal_modality, waypoint_select, etc.

        Returns:
            List with one action dict containing:
                - linear_vel: forward velocity (m/s)
                - angular_vel: turning velocity (rad/s)
                - waypoints: raw (8, 4) waypoint trajectory
                - modality_id: which input modality was used (0-8)
        """
        self._ensure_loaded()

        # Allow runtime overrides
        goal_modality = kwargs.get("goal_modality", self._goal_modality)
        effective_instruction = instruction or self._instruction

        # Update goal image from observation if provided
        goal_image = self._goal_image_pil
        if "goal_image" in observation_dict:
            from PIL import Image

            gi = observation_dict["goal_image"]
            if isinstance(gi, np.ndarray):
                goal_image = Image.fromarray(gi.astype(np.uint8)).convert("RGB")

        # Extract current camera image
        current_image = self._extract_camera_image(observation_dict)

        # Resolve modality (use runtime override if provided)
        pose_goal, satellite, image_goal, lan_prompt = self._resolve_modality_flags(
            observation_dict, effective_instruction, modality_override=goal_modality
        )
        modality_id = self._get_modality_id(pose_goal, satellite, image_goal, lan_prompt)

        # Compute goal pose if GPS-based
        goal_pose_norm = self._compute_goal_pose_norm(observation_dict)

        # Run inference based on variant
        if self._variant == "edge":
            waypoints = self._infer_edge(
                current_image, goal_image, goal_pose_norm, effective_instruction, modality_id, observation_dict
            )
        else:
            waypoints = self._infer_full(
                current_image,
                goal_image,
                goal_pose_norm,
                effective_instruction,
                modality_id,
                pose_goal,
                satellite,
                image_goal,
                lan_prompt,
            )

        # Convert waypoints to velocity
        linear_vel, angular_vel = self._waypoints_to_velocity(waypoints)

        self._step += 1
        logger.debug(
            f"🧭 OmniVLA step {self._step}: modality={modality_id}, " f"vel=({linear_vel:.3f}, {angular_vel:.3f})"
        )

        return [
            {
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
                "waypoints": waypoints.tolist(),
                "modality_id": modality_id,
            }
        ]

    def _extract_camera_image(self, observation_dict: Dict[str, Any]):
        """Extract the primary camera image from observation dict."""
        from PIL import Image

        for key in sorted(observation_dict.keys()):
            val = observation_dict[key]
            if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
                return Image.fromarray(val[:, :, :3].astype(np.uint8)).convert("RGB")

        # Fallback: black image
        logger.warning("No camera image found in observation — using blank")
        return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _infer_full(
        self,
        current_image,
        goal_image,
        goal_pose_norm: np.ndarray,
        instruction: str,
        modality_id: int,
        pose_goal: bool,
        satellite: bool,
        image_goal: bool,
        lan_prompt: bool,
    ) -> np.ndarray:
        """Run full OmniVLA (7B) inference."""
        import torch
        from PIL import Image
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: F401
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        # Prepare goal image (required for the dual-image input)
        if goal_image is None:
            goal_image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Build data batch
        dummy_actions = np.random.rand(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Language instruction
        lan_inst = instruction if lan_prompt else "xxxx"

        # Transform to dataset format
        batch_data = self._transform_full_data(
            current_image,
            goal_image,
            lan_inst,
            dummy_actions,
            goal_pose_norm,
        )

        # Collate
        batch = self._collate_full([batch_data])

        # Modality tensor
        modality_tensor = torch.as_tensor([modality_id], dtype=torch.float32)

        # Forward pass
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            output = self._vla(
                input_ids=batch["input_ids"].to(self._device),
                attention_mask=batch["attention_mask"].to(self._device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self._device),
                modality_id=modality_tensor.to(torch.bfloat16).to(self._device),
                labels=batch["labels"].to(self._device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self._device),
                proprio_projector=self._pose_projector,
            )

        # Extract action from hidden states
        ground_truth_token_ids = batch["labels"][:, 1:].to(self._device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self._num_patches : -1]

        batch_size = batch["input_ids"].shape[0]
        actions_hidden = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = self._action_head.predict_action(
                actions_hidden, modality_tensor.to(torch.bfloat16).to(self._device)
            )

        return predicted_actions.float().cpu().numpy()[0]

    def _transform_full_data(self, current_image, goal_image, instruction, actions, goal_pose):
        """Transform data for full model input."""
        import torch
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder

        IGNORE_INDEX = -100
        processor = self._processor

        if instruction == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": ""},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {instruction}?"},
                {"from": "gpt", "value": ""},
            ]

        # Build action string via tokenizer
        action_strings = []
        for a in actions:
            action_strings.append(self._action_tokenizer(a))
        action_chunk_string = "".join(action_strings)
        action_chunk_len = len(action_chunk_string)

        conversation[1]["value"] = action_chunk_string

        # Build prompt
        prompt_builder = PurePromptBuilder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        pixel_values_current = processor.image_processor.apply_transform(current_image)
        pixel_values_goal = processor.image_processor.apply_transform(goal_image)

        return {
            "pixel_values_current": pixel_values_current,
            "pixel_values_goal": pixel_values_goal,
            "input_ids": input_ids,
            "labels": labels,
            "actions": torch.as_tensor(actions, dtype=torch.float32),
            "goal_pose": goal_pose.astype(np.float32),
        }

    def _collate_full(self, instances):
        """Collate a batch for full model."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        IGNORE_INDEX = -100
        pad_token_id = self._processor.tokenizer.pad_token_id
        model_max_length = self._processor.tokenizer.model_max_length

        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = pad_sequence(
            [inst["labels"] for inst in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, :model_max_length]
        labels = labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = torch.cat(
            (
                torch.stack([inst["pixel_values_current"] for inst in instances]),
                torch.stack([inst["pixel_values_goal"] for inst in instances]),
            ),
            dim=1,
        )

        actions = torch.stack([inst["actions"] for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "goal_pose": goal_pose,
        }

    def _infer_edge(
        self,
        current_image,
        goal_image,
        goal_pose_norm: np.ndarray,
        instruction: str,
        modality_id: int,
        observation_dict: Dict[str, Any],
    ) -> np.ndarray:
        """Run OmniVLA-edge inference."""
        import clip
        import torch

        imgsize = (96, 96)
        imgsize_clip = (224, 224)

        # Resize images
        current_96 = current_image.resize(imgsize)
        current_224 = current_image.resize(imgsize_clip)

        if goal_image:
            goal_96 = goal_image.resize(imgsize)
        else:
            from PIL import Image

            goal_96 = Image.new("RGB", imgsize, color=(0, 0, 0))

        # Build context queue (replicate current for static context)
        try:
            from utils_policy import transform_images_map, transform_images_PIL
        except ImportError:
            raise ImportError("OmniVLA-edge requires utils_policy from the OmniVLA package")

        context_queue = [current_96] * 6
        obs_images = transform_images_PIL(context_queue)
        obs_images_split = torch.split(obs_images.to(self._device), 3, dim=1)
        obs_image_cur = obs_images_split[-1].to(self._device)
        obs_images_cat = torch.cat(obs_images_split, dim=1).to(self._device)

        cur_large_img = transform_images_PIL(current_224).to(self._device)

        # Dummy satellite images
        from PIL import Image

        sat_cur = Image.new("RGB", (352, 352), color=(0, 0, 0))
        sat_goal = Image.new("RGB", (352, 352), color=(0, 0, 0))
        current_map = transform_images_map(sat_cur)
        goal_map = transform_images_map(sat_goal)
        map_images = torch.cat(
            (
                current_map.to(self._device),
                goal_map.to(self._device),
                obs_image_cur,
            ),
            axis=1,
        )

        # Language encoding
        lan_inst = instruction if modality_id in (7, 8, 9) else "xxxx"
        obj_inst_lan = clip.tokenize(lan_inst, truncate=True).to(self._device)

        # Goal image
        goal_img_tensor = transform_images_PIL(goal_96).to(self._device)

        # Goal pose
        goal_pose_torch = torch.from_numpy(goal_pose_norm).unsqueeze(0).float().to(self._device)

        # Build batch
        batch = {
            "obs_images": obs_images_cat,
            "goal_pose_torch": goal_pose_torch,
            "map_images": map_images,
            "goal_image": goal_img_tensor,
            "obj_inst_lan": obj_inst_lan,
            "cur_large_img": cur_large_img,
        }

        bimg = batch["goal_image"].size(0)
        modality_tensor = torch.tensor([modality_id]).to(self._device)

        with torch.no_grad():
            feat_text_lan = self._text_encoder.encode_text(batch["obj_inst_lan"])
            predicted_actions, _, _ = self._edge_model(
                batch["obs_images"].repeat(bimg, 1, 1, 1),
                batch["goal_pose_torch"].repeat(bimg, 1),
                batch["map_images"].repeat(bimg, 1, 1, 1),
                batch["goal_image"],
                modality_tensor.repeat(bimg),
                feat_text_lan.repeat(bimg, 1),
                batch["cur_large_img"].repeat(bimg, 1, 1, 1),
            )

        return predicted_actions.float().cpu().numpy()[0]

    # =========================================================================
    # Utility
    # =========================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        info = {
            "provider": "omnivla",
            "variant": self._variant,
            "checkpoint_path": self._checkpoint_path,
            "goal_modality": self._goal_modality,
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
            "step": self._step,
            "max_linear_vel": self._max_linear_vel,
            "max_angular_vel": self._max_angular_vel,
        }
        if self._loaded:
            if self._variant == "edge" and self._edge_model:
                info["n_parameters"] = sum(p.numel() for p in self._edge_model.parameters())
            elif self._vla:
                info["n_parameters"] = sum(p.numel() for p in self._vla.parameters())
        return info

    def reset(self):
        """Reset step counter and internal state."""
        self._step = 0
        logger.info("🧭 OmniVLA policy reset")

    def update_goal(
        self,
        instruction: Optional[str] = None,
        goal_image=None,
        goal_gps: Optional[Tuple[float, float, float]] = None,
        goal_modality: Optional[str] = None,
    ):
        """Update the navigation goal at runtime.

        Args:
            instruction: New language instruction
            goal_image: New goal image (PIL Image or numpy array)
            goal_gps: New GPS goal (lat, lon, compass_degrees)
            goal_modality: New modality ("language", "image", "pose", etc.)
        """
        if instruction is not None:
            self._instruction = instruction
        if goal_image is not None:
            from PIL import Image as PILImage

            if isinstance(goal_image, np.ndarray):
                self._goal_image_pil = PILImage.fromarray(goal_image.astype(np.uint8)).convert("RGB")
            else:
                self._goal_image_pil = goal_image
        if goal_gps is not None:
            self._goal_gps = goal_gps
            try:
                import utm

                lat, lon, compass_deg = goal_gps
                self._goal_utm = utm.from_latlon(lat, lon)
                self._goal_compass_rad = -float(compass_deg) / 180.0 * math.pi
            except ImportError:
                pass
        if goal_modality is not None:
            self._goal_modality = goal_modality.lower()

        logger.info(f"🧭 Goal updated: modality={self._goal_modality}, " f"instruction='{self._instruction[:50]}...'")


__all__ = ["OmnivlaPolicy"]
