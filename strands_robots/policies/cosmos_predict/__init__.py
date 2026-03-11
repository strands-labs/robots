"""Cosmos Predict 2.5 Policy Provider — NVIDIA's World Foundation Model as Robot Policy.

Cosmos-Predict2.5 is a flow-based world foundation model that unifies Text2World,
Image2World, and Video2World into a single model. The /robot/policy variant has been
post-trained on LIBERO and RoboCasa benchmarks, achieving 98.5% success on LIBERO-10.

The model operates in latent diffusion space: it encodes camera images + proprioception
into a structured latent sequence, then predicts action chunks via denoising. It can also
jointly predict future camera frames (world model) and value estimates (value function)
— enabling model-based planning and best-of-N action selection.

Available models:
    - nvidia/Cosmos-Predict2.5-2B  (robot/policy — LIBERO, RoboCasa post-trained)
    - nvidia/Cosmos-Predict2.5-14B (base — larger backbone, requires multi-GPU)
    - nvidia/Cosmos-Predict2.5-2B  (robot/action-cond — action-conditioned video gen)

Architecture:
    [Camera Images + Proprio + Language] → VAE Encoder → Latent Sequence
    → Rectified Flow DiT (2B/14B) → Denoised Latent
    → Extract Action Chunk (16-step, 7-DoF) + Future Images + Value

Post-trained checkpoints:
    - nvidia/Cosmos-Policy-LIBERO-Predict2-2B (LIBERO benchmark, 98.5% success)
    - nvidia/Cosmos-Policy-RoboCasa-Predict2-2B (RoboCasa benchmark)

Requirements:
    - cosmos-predict2 package: pip install cosmos-predict2
    - GPU with 16GB+ VRAM (2B model) or multi-GPU (14B model)
    - For policy mode: pre-computed T5/Reason1 text embeddings or online computation

Usage:
    from strands_robots.policies import create_policy

    # Policy mode — direct action prediction (LIBERO)
    policy = create_policy("cosmos_predict",
        model_id="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        mode="policy",
        suite="libero",
        dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    )

    # Action-conditioned world model — video generation from actions
    policy = create_policy("cosmos_predict",
        model_id="nvidia/Cosmos-Predict2.5-2B",
        mode="action_conditioned",
    )

    # Base world model — video prediction from text + image/video
    policy = create_policy("cosmos_predict",
        model_id="nvidia/Cosmos-Predict2.5-14B",
        mode="world_model",
    )

Reference:
    "Cosmos World Foundation Model Platform for Physical AI", arXiv:2511.00062
    GitHub: https://github.com/nvidia-cosmos/cosmos-predict2.5
"""

import logging
import time
import types
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image

logger = logging.getLogger(__name__)

# Default action dimension (7-DoF: x, y, z, roll, pitch, yaw, gripper)
ACTION_DIM = 7
# Standard image size for Cosmos policy models
COSMOS_IMAGE_SIZE = 224
# Temporal compression factor for VAE
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4


class CosmosPredictPolicy(Policy):
    """Cosmos Predict 2.5 — NVIDIA's World Foundation Model as Robot Policy.

    Supports three modes:
    1. **policy**: Direct action prediction via latent diffusion (LIBERO/RoboCasa)
    2. **action_conditioned**: Action-conditioned video generation (world simulation)
    3. **world_model**: Text/image/video → future video prediction

    The policy mode uses a structured latent sequence:
        [blank, proprio, wrist_img, third_person_img, ACTION, future_proprio,
         future_wrist, future_third_person, VALUE]

    Where ACTION and VALUE are predicted via rectified flow denoising, and
    future images are jointly generated as a world model byproduct.
    """

    # Supported evaluation suites and their camera configurations
    SUITE_CONFIGS = {
        "libero": {
            "cameras": ["wrist", "primary"],
            "num_wrist_images": 1,
            "num_third_person_images": 1,
            "state_t": 9,
            "min_conditional_frames": 4,
        },
        "robocasa": {
            "cameras": ["wrist", "primary", "secondary"],
            "num_wrist_images": 1,
            "num_third_person_images": 2,
            "state_t": 11,
            "min_conditional_frames": 5,
        },
        "aloha": {
            "cameras": ["left_wrist", "right_wrist", "primary"],
            "num_wrist_images": 2,
            "num_third_person_images": 1,
            "state_t": 11,
            "min_conditional_frames": 5,
        },
    }

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Predict2.5-2B",
        mode: str = "policy",
        suite: str = "libero",
        device: Optional[str] = None,
        chunk_size: int = 16,
        num_denoising_steps: int = 5,
        dataset_stats_path: Optional[str] = None,
        t5_embeddings_path: Optional[str] = None,
        text_embeddings_kind: str = "t5",
        config_file: Optional[str] = None,
        config_name: Optional[str] = None,
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        normalize_proprio: bool = True,
        unnormalize_actions: bool = True,
        use_jpeg_compression: bool = True,
        flip_images: bool = True,
        trained_with_image_aug: bool = True,
        action_dim: int = ACTION_DIM,
        server_url: Optional[str] = None,
        guidance: float = 7.0,
        num_steps: int = 35,
        **kwargs,
    ):
        """Initialize Cosmos Predict 2.5 policy.

        Args:
            model_id: HuggingFace model ID or local path to checkpoint
            mode: Operating mode — "policy", "action_conditioned", or "world_model"
            suite: Evaluation suite — "libero", "robocasa", or "aloha" (for policy mode)
            device: CUDA device (auto-detected if None)
            chunk_size: Number of actions per chunk (policy mode)
            num_denoising_steps: Denoising steps for action prediction
            dataset_stats_path: Path to dataset statistics JSON (for action un-normalization)
            t5_embeddings_path: Path to pre-computed T5 text embeddings
            text_embeddings_kind: Type of text embeddings — "t5" or "reason1"
            config_file: Model config file path (for Cosmos internal config system)
            config_name: Experiment config name
            use_wrist_image: Whether to use wrist camera images
            use_proprio: Whether to use proprioceptive state
            normalize_proprio: Whether to normalize proprioception
            unnormalize_actions: Whether to un-normalize predicted actions
            use_jpeg_compression: Apply JPEG compression (matches training)
            flip_images: Flip images vertically (matches LIBERO/RoboCasa convention)
            trained_with_image_aug: Apply test-time image augmentation
            action_dim: Action dimension (default: 7 for manipulation)
            server_url: URL for remote inference server (bypasses local model loading)
            guidance: Guidance scale for action-conditioned/world model inference
            num_steps: Number of denoising steps for video generation
        """
        self._model_id = model_id
        self._mode = mode
        self._suite = suite
        self._requested_device = device
        self._chunk_size = chunk_size
        self._num_denoising_steps = num_denoising_steps
        self._dataset_stats_path = dataset_stats_path
        self._t5_embeddings_path = t5_embeddings_path
        self._text_embeddings_kind = text_embeddings_kind
        self._config_file = (
            config_file
            or "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
        )
        self._config_name = config_name
        self._use_wrist_image = use_wrist_image
        self._use_proprio = use_proprio
        self._normalize_proprio = normalize_proprio
        self._unnormalize_actions = unnormalize_actions
        self._use_jpeg_compression = use_jpeg_compression
        self._flip_images = flip_images
        self._trained_with_image_aug = trained_with_image_aug
        self._action_dim = action_dim
        self._server_url = server_url
        self._guidance = guidance
        self._num_steps = num_steps
        self._extra_kwargs = kwargs
        self._robot_state_keys: List[str] = []

        # Model state (lazy-loaded)
        self._model = None
        self._config = None
        self._dataset_stats = None
        self._device = None
        self._loaded = False
        self._step = 0

        mode_str = f"server={server_url}" if server_url else f"local ({model_id})"
        logger.info(
            f"🌌 Cosmos Predict 2.5 policy: mode={mode}, suite={suite}, {mode_str}"
        )

    @property
    def provider_name(self) -> str:
        return "cosmos_predict"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        logger.info("🔧 Cosmos Predict robot state keys: %s", self._robot_state_keys)

    def _ensure_loaded(self):
        """Lazy-load model, dataset stats, and text embeddings."""
        if self._loaded:
            return

        if self._server_url:
            # Server mode — verify connectivity
            try:
                import requests

                requests.get(f"{self._server_url}/health", timeout=5)
                logger.info("🌌 Cosmos server connected: %s", self._server_url)
            except Exception as e:
                logger.warning(
                    f"Cosmos server not reachable at {self._server_url}: {e}"
                )
            self._loaded = True
            return

        # Local mode — load model
        logger.info("🌌 Loading Cosmos Predict 2.5 from %s...", self._model_id)
        start = time.time()

        self._device = detect_device(self._requested_device)

        if self._mode == "policy":
            self._load_policy_model()
        elif self._mode == "action_conditioned":
            self._load_action_conditioned_model()
        else:
            self._load_world_model()

        elapsed = time.time() - start
        logger.info("🌌 Cosmos loaded in %.1fs on %s", elapsed, self._device)
        self._loaded = True

    def _load_policy_model(self):
        """Load Cosmos Policy model for direct action prediction."""
        try:
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                get_model as cosmos_get_model,
            )
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                init_t5_text_embeddings_cache,
            )
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                load_dataset_stats as cosmos_load_dataset_stats,
            )

            # Build a config namespace for cosmos_get_model
            cfg = types.SimpleNamespace(
                ckpt_path=self._model_id,
                config=self._config_name or self._infer_config_name(),
                config_file=self._config_file,
            )

            self._model, self._config = cosmos_get_model(cfg)

            # Load dataset statistics for action un-normalization
            # Auto-resolve dataset_stats from HF checkpoint if not explicitly provided
            if self._dataset_stats_path:
                self._dataset_stats = cosmos_load_dataset_stats(
                    self._dataset_stats_path
                )
                logger.info("🌌 Dataset stats loaded from %s", self._dataset_stats_path)
            else:
                # Try to find dataset_statistics.json in the HF cache
                try:
                    from huggingface_hub import snapshot_download

                    ckpt_dir = snapshot_download(
                        self._model_id, allow_patterns=["*.json", "*.pkl"]
                    )
                    import os

                    for fname in [
                        "libero_dataset_statistics.json",
                        "robocasa_dataset_statistics.json",
                        "aloha_dataset_statistics.json",
                        "dataset_statistics.json",
                    ]:
                        stats_path = os.path.join(ckpt_dir, fname)
                        if os.path.exists(stats_path):
                            self._dataset_stats = cosmos_load_dataset_stats(stats_path)
                            self._dataset_stats_path = stats_path
                            logger.info("🌌 Dataset stats auto-resolved: %s", stats_path)
                            break
                    if not self._dataset_stats:
                        logger.warning(
                            "🌌 No dataset statistics found — actions will not be un-normalized"
                        )
                except Exception as e:
                    logger.warning("🌌 Could not auto-resolve dataset stats: %s", e)

            # Initialize text embeddings cache (auto-resolve from HF if needed)
            if not self._t5_embeddings_path and self._dataset_stats_path:
                import os

                ckpt_dir = os.path.dirname(self._dataset_stats_path)
                for fname in [
                    "libero_t5_embeddings.pkl",
                    "robocasa_t5_embeddings.pkl",
                    "aloha_t5_embeddings.pkl",
                    "t5_embeddings.pkl",
                ]:
                    t5_path = os.path.join(ckpt_dir, fname)
                    if os.path.exists(t5_path):
                        self._t5_embeddings_path = t5_path
                        logger.info("🌌 T5 embeddings auto-resolved: %s", t5_path)
                        break

            if self._t5_embeddings_path:
                init_t5_text_embeddings_cache(
                    self._t5_embeddings_path,
                    worker_id=0,
                    embeddings_kind=self._text_embeddings_kind,
                )
                logger.info("🌌 Text embeddings loaded (%s)", self._text_embeddings_kind)

        except ImportError as e:
            raise ImportError(
                f"Cosmos Predict 2.5 requires the cosmos-predict2 package and dependencies.\n"
                f"Install from source:\n"
                f"  git clone https://github.com/nvidia-cosmos/cosmos-predict2.5\n"
                f"  cd cosmos-predict2.5\n"
                f"  pip install -e packages/cosmos-oss -e packages/cosmos-cuda -e .\n"
                f"  pip install transformer-engine[pytorch] flash-attn omegaconf loguru iopath tyro h5py\n"
                f"\n"
                f"Note: Requires CUDA toolkit, cuDNN headers, and NVIDIA GPU.\n"
                f"The LIBERO checkpoint is: nvidia/Cosmos-Policy-LIBERO-Predict2-2B\n"
                f"Config name: cosmos_predict2_2b_480p_libero\n"
                f"Error: {e}"
            )

    def _load_action_conditioned_model(self):
        """Load action-conditioned video generation model."""
        try:
            from cosmos_predict2._src.predict2.inference.video2world import (
                Video2WorldInference,
            )
            from cosmos_predict2.config import MODEL_CHECKPOINTS

            # Determine checkpoint — MODEL_CHECKPOINTS is keyed by ModelKey objects,
            # not strings. Use MODEL_KEYS name→key lookup for string-based access.
            model_key_str = self._extra_kwargs.get("model_key", "robot/action-cond")
            checkpoint = None
            experiment = None
            checkpoint_path = None

            try:
                from cosmos_predict2.config import MODEL_KEYS

                if model_key_str in MODEL_KEYS:
                    key_obj = MODEL_KEYS[model_key_str]
                    checkpoint = MODEL_CHECKPOINTS.get(key_obj)
            except ImportError:
                pass

            if checkpoint is not None:
                experiment = checkpoint.experiment
                checkpoint_path = checkpoint.s3.uri
            else:
                experiment = self._config_name
                checkpoint_path = self._model_id

            ac_config = self._extra_kwargs.get(
                "ac_config_file",
                "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
            )

            self._video2world = Video2WorldInference(
                experiment_name=experiment,
                ckpt_path=checkpoint_path,
                s3_credential_path="",
                context_parallel_size=1,
                config_file=ac_config,
            )
            logger.info("🌌 Action-conditioned model loaded")

        except ImportError as e:
            raise ImportError(
                f"Action-conditioned mode requires cosmos-predict2.\n"
                f"Install: pip install cosmos-predict2\nError: {e}"
            )

    def _load_world_model(self):
        """Load base world model for text/image/video → future video."""
        try:
            from cosmos_predict2._src.predict2.inference.video2world import (
                Video2WorldInference,
            )
            from cosmos_predict2.config import MODEL_CHECKPOINTS

            size = "14B" if "14B" in self._model_id else "2B"
            model_key_str = f"base/post-trained/{size}"
            checkpoint = None
            experiment = None
            checkpoint_path = None

            # MODEL_CHECKPOINTS is keyed by ModelKey objects, not strings.
            try:
                from cosmos_predict2.config import MODEL_KEYS

                if model_key_str in MODEL_KEYS:
                    key_obj = MODEL_KEYS[model_key_str]
                    checkpoint = MODEL_CHECKPOINTS.get(key_obj)
            except ImportError:
                pass

            if checkpoint is not None:
                experiment = checkpoint.experiment
                checkpoint_path = checkpoint.s3.uri
            else:
                experiment = self._config_name
                checkpoint_path = self._model_id

            self._video2world = Video2WorldInference(
                experiment_name=experiment,
                ckpt_path=checkpoint_path,
                s3_credential_path="",
                context_parallel_size=1,
            )
            logger.info("🌌 World model loaded")

        except ImportError as e:
            raise ImportError(f"World model mode requires cosmos-predict2.\nError: {e}")

    def _infer_config_name(self) -> str:
        """Infer the Cosmos config name from the model ID."""
        model_lower = self._model_id.lower()

        if "libero" in model_lower:
            return "cosmos_predict2_2b_480p_libero"
        elif "robocasa" in model_lower:
            return "cosmos_predict2_2b_480p_robocasa"
        elif "aloha" in model_lower:
            return "cosmos_predict2_2b_480p_aloha"
        elif "14b" in model_lower:
            return "cosmos_predict2_14b_v2w"
        else:
            return "cosmos_predict2_2b_v2w"

    async def get_actions(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Get actions from Cosmos Predict 2.5.

        For **policy mode**: predicts an action chunk (16 steps × 7 DoF) via
        latent diffusion denoising. Optionally returns future image predictions
        and value estimates.

        For **action_conditioned mode**: generates future video frames conditioned
        on the provided action sequence. Returns predicted frames.

        For **world_model mode**: generates future video from text prompt + input
        image/video. Returns predicted frames.

        Args:
            observation_dict: Robot observation containing:
                - Camera images as numpy arrays (keys: "primary_image", "wrist_image", etc.)
                - Proprioceptive state: "proprio" or "observation.state"
                - For action_conditioned: "actions" numpy array (N, 7)
            instruction: Natural language task description
            **kwargs: Additional parameters (seed, num_denoising_steps, etc.)

        Returns:
            List of action dicts with keys matching robot_state_keys + "gripper"
        """
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction, **kwargs)

        if self._mode == "policy":
            return self._infer_policy(observation_dict, instruction, **kwargs)
        elif self._mode == "action_conditioned":
            return self._infer_action_conditioned(
                observation_dict, instruction, **kwargs
            )
        else:
            return self._infer_world_model(observation_dict, instruction, **kwargs)

    def _infer_policy(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Policy mode: predict action chunk via Cosmos latent diffusion."""
        try:
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                get_action as cosmos_get_action,
            )
        except ImportError:
            raise ImportError("Policy mode requires cosmos-predict2 package")

        # Build observation in Cosmos format
        obs = self._build_cosmos_observation(observation_dict)

        # Build config namespace with ALL required fields for get_action()
        # These defaults were verified working on NVIDIA L40S (EC2 g6e.4xlarge)
        suite_cfg = self.SUITE_CONFIGS.get(self._suite, self.SUITE_CONFIGS["libero"])

        cfg = types.SimpleNamespace(
            # Suite & camera config
            suite=self._suite,
            use_wrist_image=self._use_wrist_image,
            use_third_person_image=True,
            num_wrist_images=suite_cfg.get("num_wrist_images", 1),
            num_third_person_images=suite_cfg.get("num_third_person_images", 1),
            use_proprio=self._use_proprio,
            normalize_proprio=self._normalize_proprio,
            unnormalize_actions=self._unnormalize_actions,
            use_jpeg_compression=self._use_jpeg_compression,
            trained_with_image_aug=self._trained_with_image_aug,
            chunk_size=self._chunk_size,
            model_family="predict2",
            scale_multiplier=kwargs.get("scale_multiplier", 1.0),
            # Denoising & sampling
            num_denoising_steps_action=kwargs.get(
                "num_denoising_steps", self._num_denoising_steps
            ),
            seed=kwargs.get("seed", 1),
            randomize_seed=kwargs.get("randomize_seed", False),
            shift=kwargs.get("shift", 1.0),
            t=suite_cfg.get("state_t", 9),
            use_variance_scale=kwargs.get("use_variance_scale", False),
            # Future state & value prediction (disabled by default for speed)
            ar_future_prediction=kwargs.get("ar_future_prediction", False),
            ar_value_prediction=kwargs.get("ar_value_prediction", False),
            ar_qvalue_prediction=kwargs.get("ar_qvalue_prediction", False),
            use_ensemble_future_state_predictions=False,
            use_ensemble_value_predictions=False,
            num_future_state_predictions_in_ensemble=1,
            num_value_predictions_in_ensemble=1,
            future_state_ensemble_aggregation_scheme="mean",
            value_ensemble_aggregation_scheme="mean",
            mask_current_state_action_for_value_prediction=False,
            mask_future_state_for_qvalue_prediction=False,
            num_denoising_steps_future_state=5,
            num_denoising_steps_value=5,
            num_queries_best_of_n=kwargs.get("best_of_n", 1),
            parallel_timeout=30,
            search_depth=1,
            planning_model_ckpt_path=None,
            planning_model_config_name=None,
        )

        seed = kwargs.get("seed", 1)
        num_steps = kwargs.get("num_denoising_steps", self._num_denoising_steps)

        # Call Cosmos get_action
        result = cosmos_get_action(
            cfg=cfg,
            model=self._model,
            dataset_stats=self._dataset_stats or {},
            obs=obs,
            task_label_or_embedding=instruction,
            seed=seed,
            num_denoising_steps_action=num_steps,
            generate_future_state_and_value_in_parallel=True,
        )

        # Convert action arrays to action dicts
        actions = []
        raw_actions = result.get("actions", [])
        for action_vec in raw_actions:
            action_vec = np.asarray(action_vec, dtype=np.float32)
            action_dict = {}

            if self._robot_state_keys:
                for j, key in enumerate(self._robot_state_keys):
                    if j < len(action_vec) - 1:
                        action_dict[key] = float(action_vec[j])
                action_dict["gripper"] = (
                    float(action_vec[-1]) if len(action_vec) > 0 else 0.0
                )
            else:
                # Default 7-DoF mapping
                labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                for j, label in enumerate(labels):
                    if j < len(action_vec):
                        action_dict[label] = float(action_vec[j])

            actions.append(action_dict)

        # Attach metadata
        if actions:
            actions[0]["_cosmos_metadata"] = {
                "value_prediction": result.get("value_prediction"),
                "mode": "policy",
                "suite": self._suite,
                "chunk_size": self._chunk_size,
                "denoising_steps": num_steps,
            }

        self._step += 1
        logger.debug(
            f"🌌 Cosmos policy step {self._step}: {len(actions)} actions, "
            f"value={result.get('value_prediction', 'N/A')}"
        )
        return actions

    def _infer_action_conditioned(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Action-conditioned mode: generate future video from actions."""
        import torch

        # Extract actions from observation
        actions = observation_dict.get("actions")
        if actions is None:
            raise ValueError(
                "Action-conditioned mode requires 'actions' key in observation_dict "
                "(numpy array of shape (N, action_dim))"
            )
        actions = np.asarray(actions, dtype=np.float32)

        # Extract initial frame
        initial_frame = self._find_camera_image(observation_dict)
        if initial_frame is None:
            raise ValueError(
                "Action-conditioned mode requires at least one camera image"
            )

        # Run action-conditioned video generation
        import torchvision

        img_tensor = torchvision.transforms.functional.to_tensor(
            initial_frame
        ).unsqueeze(0)
        num_video_frames = min(actions.shape[0] + 1, self._chunk_size + 1)
        vid_input = torch.cat(
            [
                img_tensor,
                torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1),
            ],
            dim=0,
        )
        vid_input = (
            (vid_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
        )

        video = self._video2world.generate_vid2world(
            prompt=instruction or "",
            input_path=vid_input,
            action=torch.from_numpy(actions[: self._chunk_size]).float(),
            guidance=self._guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            seed=kwargs.get("seed", 0),
            num_steps=self._num_steps,
        )

        # Return frames as action dicts with video metadata
        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (
            (torch.clamp(video_normalized[0], 0, 1) * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )

        result_actions = []
        for i in range(video_clamped.shape[0]):
            action_dict = {"predicted_frame": video_clamped[i]}
            if i < actions.shape[0]:
                for j in range(min(self._action_dim, actions.shape[1])):
                    action_dict[f"action_{j}"] = float(actions[i, j])
            result_actions.append(action_dict)

        self._step += 1
        return result_actions

    def _infer_world_model(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """World model mode: text+image/video → future video."""
        import torch

        initial_frame = self._find_camera_image(observation_dict)
        if initial_frame is None:
            raise ValueError("World model mode requires at least one camera image")

        import torchvision

        img_tensor = torchvision.transforms.functional.to_tensor(
            initial_frame
        ).unsqueeze(0)
        num_frames = kwargs.get("num_video_frames", 17)
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_frames - 1, 1, 1, 1)],
            dim=0,
        )
        vid_input = (
            (vid_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
        )

        video = self._video2world.generate_vid2world(
            prompt=instruction,
            input_path=vid_input,
            guidance=self._guidance,
            num_video_frames=num_frames,
            num_latent_conditional_frames=1,
            seed=kwargs.get("seed", 0),
            num_steps=self._num_steps,
        )

        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (
            (torch.clamp(video_normalized[0], 0, 1) * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )

        return [
            {"predicted_frame": video_clamped[i]} for i in range(video_clamped.shape[0])
        ]

    async def _infer_server(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Inference via remote Cosmos policy server."""
        import io

        import requests

        # Build request payload
        payload = {"instruction": instruction, "mode": self._mode}

        # Send images
        for key, val in observation_dict.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                payload[key] = val.tolist()
            elif isinstance(val, np.ndarray):
                payload[key] = val.tolist()
            else:
                payload[key] = val

        endpoint = kwargs.get("endpoint", "/act")
        resp = requests.post(f"{self._server_url}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()

        actions = []
        for action_data in result.get("actions", []):
            if isinstance(action_data, list):
                action_dict = {}
                labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                for j, val in enumerate(action_data):
                    label = labels[j] if j < len(labels) else f"dim_{j}"
                    action_dict[label] = float(val)
                actions.append(action_dict)
            elif isinstance(action_data, dict):
                actions.append(action_data)

        return actions

    def _build_cosmos_observation(
        self, observation_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert strands-robots observation to Cosmos policy format.

        Cosmos expects specific key names depending on the suite:
        - libero: wrist_image, primary_image, proprio
        - robocasa: wrist_image, primary_image, secondary_image, proprio
        - aloha: left_wrist_image, right_wrist_image, primary_image, proprio
        """
        obs = {}

        # Try to find images by various key patterns
        camera_mappings = {
            "primary_image": [
                "primary",
                "camera_0",
                "cam_high",
                "front",
                "exterior",
                "third_person",
            ],
            "wrist_image": ["wrist", "hand", "cam_low", "gripper"],
            "secondary_image": ["secondary", "camera_1", "cam_side", "side"],
            "left_wrist_image": ["left_wrist", "cam_left_wrist"],
            "right_wrist_image": ["right_wrist", "cam_right_wrist"],
        }

        for cosmos_key, search_patterns in camera_mappings.items():
            # Direct match first
            if cosmos_key in observation_dict:
                val = observation_dict[cosmos_key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    obs[cosmos_key] = val.astype(np.uint8)
                    continue

            # Pattern-based search
            for pattern in search_patterns:
                for obs_key, val in observation_dict.items():
                    if (
                        pattern in obs_key.lower()
                        and isinstance(val, np.ndarray)
                        and val.ndim == 3
                    ):
                        obs[cosmos_key] = val[:, :, :3].astype(np.uint8)
                        break
                if cosmos_key in obs:
                    break

        # Map proprioceptive state
        proprio = None
        for key in ["proprio", "observation.state", "state", "joint_positions"]:
            if key in observation_dict:
                val = observation_dict[key]
                if isinstance(val, np.ndarray):
                    proprio = val.astype(np.float32)
                elif isinstance(val, (list, tuple)):
                    proprio = np.array(val, dtype=np.float32)
                break

        # Build from individual robot state keys
        if proprio is None and self._robot_state_keys:
            values = []
            for key in self._robot_state_keys:
                if key in observation_dict:
                    values.append(float(observation_dict[key]))
            if values:
                proprio = np.array(values, dtype=np.float32)

        if proprio is not None:
            obs["proprio"] = proprio

        return obs

    def _find_camera_image(self, observation_dict: Dict[str, Any]):
        """Find any camera image in the observation dict and return as PIL Image."""
        return extract_pil_image(
            observation_dict, fallback_size=(COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE)
        )

    def get_value_estimate(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> float:
        """Get value estimate V(s) for the current state.

        Cosmos Policy can jointly predict action chunks AND value estimates.
        This method extracts only the value prediction.

        Args:
            observation_dict: Robot observation
            instruction: Task instruction

        Returns:
            Value estimate in [0, 1] (probability of task success)
        """
        self._ensure_loaded()
        if self._mode != "policy":
            raise ValueError("Value estimation only available in policy mode")

        # Run full action prediction (which includes value)
        actions = self._infer_policy(observation_dict, instruction, **kwargs)
        if actions and "_cosmos_metadata" in actions[0]:
            value = actions[0]["_cosmos_metadata"].get("value_prediction")
            if value is not None:
                return float(value)
        return 0.0

    def reset(self):
        """Reset internal state."""
        self._step = 0
        logger.info("🌌 Cosmos Predict 2.5 policy reset")


__all__ = ["CosmosPredictPolicy"]
