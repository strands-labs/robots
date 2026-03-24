"""LeRobot Local Policy — Direct HuggingFace model inference (no server needed).

Uses LeRobot's own factory for auto-detection. Any model LeRobot supports,
this policy supports.

Architecture:
    Observation (dict)
        → ProcessorBridge.preprocess (normalize, device, crop, ...)
        → LeRobot PreTrainedPolicy.select_action / predict_action_chunk (RTC)
        → ProcessorBridge.postprocess (unnormalize, delta-action, ...)
        → Action dict
"""

import logging
import time
from collections import deque
from typing import Any

import numpy as np
import torch

from .. import Policy
from .processor import ProcessorBridge
from .resolution import resolve_policy_class_by_name, resolve_policy_class_from_hub

logger = logging.getLogger(__name__)


class LerobotLocalPolicy(Policy):
    """Policy that loads and runs LeRobot models directly (no server).

    Auto-detects policy type from HF config.json → delegates to LeRobot's
    own class registry.

    Optionally loads the model's processor pipeline (preprocessor.json /
    postprocessor.json) for automatic normalization, device transfer,
    observation formatting, and action unnormalization.

    Optionally supports Real-Time Chunking (RTC) for flow-matching policies,
    blending action chunks across inference calls to compensate for latency.

    Args:
        pretrained_name_or_path: HF model ID or local path. If empty, model
            is not loaded until first inference call.
        policy_type: Explicit LeRobot policy type (e.g. "act", "diffusion").
            Auto-detected from config.json if not provided.
        device: Target device (e.g. "cuda", "cpu"). Auto-detected if None.
        actions_per_step: Number of action steps to return per inference call.
        use_processor: Whether to load the model's processor pipeline.
        processor_overrides: Dict of overrides for processor pipeline steps.
        tokenizer_max_length: Max token length for VLA language tokenization.
        tokenizer_padding_side: Padding side for VLA tokenizer ("left" or "right").
        rtc_enabled: Enable Real-Time Chunking for flow-matching policies.
            Auto-detected from model config if None.
        rtc_execution_horizon: Number of timesteps from the prefix to use for
            guidance. Defaults to model config value or 10.
        rtc_max_guidance_weight: Maximum guidance weight for RTC correction.
            Defaults to model config value or 10.0.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "",
        policy_type: str | None = None,
        device: str | None = None,
        actions_per_step: int = 1,
        use_processor: bool = True,
        processor_overrides: dict | None = None,
        tokenizer_max_length: int = 48,
        tokenizer_padding_side: str = "right",
        rtc_enabled: bool | None = None,
        rtc_execution_horizon: int | None = None,
        rtc_max_guidance_weight: float | None = None,
        **kwargs,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.policy_type = policy_type
        self.requested_device = device
        self.actions_per_step = actions_per_step
        self.use_processor = use_processor
        self.processor_overrides = processor_overrides
        self.robot_state_keys: list[str] = []

        self._policy: Any | None = None
        self._device: torch.device | None = None
        self._input_features: dict[str, Any] = {}
        self._output_features: dict[str, Any] = {}
        self._loaded = False
        self._processor_bridge: ProcessorBridge | None = None
        self._tokenizer: Any = None
        self._tokenizer_max_length: int = tokenizer_max_length
        self._tokenizer_padding_side: str = tokenizer_padding_side

        # RTC state
        self._rtc_requested = rtc_enabled
        self._rtc_enabled = False
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._rtc_prev_chunk: torch.Tensor | None = None
        self._rtc_action_queue: deque = deque()
        self._rtc_latency_history: deque = deque(maxlen=100)
        self._rtc_last_inference_time: float = 0.0
        self._rtc_last_log_time: float = 0.0

        if pretrained_name_or_path:
            self._load_model()

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    def reset(self) -> None:
        """Reset policy state between episodes.

        **MUST** be called whenever the environment or task episode resets.
        LeRobot policies cache internal state such as
        action queues and temporal ensemble buffers. Without resetting, stale
        actions from the previous episode leak into the next one.

        Also clears RTC state (previous chunk leftover, action queue, latency
        history) to prevent cross-episode contamination.
        """
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
            logger.debug("Policy internal state reset")
        if self._processor_bridge is not None:
            self._processor_bridge.reset()
        # Clear RTC state
        self._rtc_prev_chunk = None
        self._rtc_action_queue.clear()
        self._rtc_latency_history.clear()
        self._rtc_last_inference_time = 0.0
        self._rtc_last_log_time = 0.0

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Set robot state keys for observation→tensor mapping.

        Args:
            robot_state_keys: List of joint/motor names. If empty, auto-detects
                from model output_features (action dim) or input_features (state dim).
                Auto-detected keys are generic (joint_0, joint_1, ...).

        Raises:
            ValueError: If keys are empty and no model features available for
                auto-detection.
        """
        if robot_state_keys:
            self.robot_state_keys = robot_state_keys
            logger.info(
                "LeRobot local state keys set: %d keys = %s%s",
                len(self.robot_state_keys),
                self.robot_state_keys[:5],
                "..." if len(self.robot_state_keys) > 5 else "",
            )
            return

        # Auto-detect from model's action output dimension
        if self._loaded and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-detected %d state keys from output_features.action.shape=%s. "
                    "For meaningful names, pass the robot's actual joint names.",
                    action_dim,
                    action_feat.shape,
                )
                return

        # Fallback: try input state dimension
        if self._loaded and self._input_features:
            state_feat = self._input_features.get("observation.state")
            if state_feat and hasattr(state_feat, "shape") and state_feat.shape:
                state_dim = state_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(state_dim)]
                logger.info(
                    "Auto-detected %d state keys from input_features.observation.state.shape=%s.",
                    state_dim,
                    state_feat.shape,
                )
                return

        raise ValueError(
            "robot_state_keys is empty and no model features available for auto-detection. "
            "Call set_robot_state_keys() with the robot's actual joint/motor names."
        )

    # ------------------------------------------------------------------
    # Tokenizer resolution (VLA language token injection)
    # ------------------------------------------------------------------

    def _resolve_tokenizer(self) -> Any | None:
        """Resolve and cache the tokenizer for VLA language token injection.

        Resolution order:
            1. Explicit ``tokenizer_name`` on policy config (e.g. xvla)
            2. ``vlm_model_name`` on policy config (maps to the VLM's tokenizer)
            3. Policy's own ``.processor.tokenizer`` (e.g. Paligemma-based)

        Returns:
            The tokenizer instance, or None if not available.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        if not self._loaded or not self._policy:
            return None

        config = getattr(self._policy, "config", None)
        if config is None:
            return None

        # Override defaults with model config if present
        self._tokenizer_max_length = getattr(config, "tokenizer_max_length", self._tokenizer_max_length)
        self._tokenizer_padding_side = getattr(config, "tokenizer_padding_side", self._tokenizer_padding_side)

        # 1. tokenizer_name (explicit config field)
        tokenizer_id = getattr(config, "tokenizer_name", None)

        # 2. vlm_model_name (VLA models)
        if not tokenizer_id:
            tokenizer_id = getattr(config, "vlm_model_name", None)

        if tokenizer_id:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                self._tokenizer.padding_side = self._tokenizer_padding_side  # type: ignore[attr-defined]
                logger.info("Auto-resolved tokenizer from '%s' (%s)", tokenizer_id, type(self._tokenizer).__name__)
                return self._tokenizer
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("Tokenizer '%s' unavailable (%s), trying next strategy...", tokenizer_id, exc)

        # 3. policy.processor.tokenizer (built-in)
        processor = getattr(self._policy, "processor", None)
        if processor and hasattr(processor, "tokenizer"):
            self._tokenizer = processor.tokenizer
            self._tokenizer.padding_side = self._tokenizer_padding_side  # type: ignore[attr-defined]
            logger.info("Using policy's built-in processor tokenizer (%s)", type(self._tokenizer).__name__)
            return self._tokenizer

        return None

    def _tokenize_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Tokenize an instruction into (input_ids, attention_mask) tensors.

        Args:
            instruction: Natural language instruction string.

        Returns:
            Tuple of (input_ids, attention_mask) tensors, or None if no tokenizer.
        """
        tokenizer = self._resolve_tokenizer()
        if tokenizer is None or not instruction:
            return None

        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        tokens = encoded["input_ids"].to(self._device)
        mask = encoded.get("attention_mask")
        if mask is not None:
            mask = mask.bool().to(self._device)
        return tokens, mask

    def _needs_language_tokens(self) -> bool:
        """Check whether this policy requires observation.language.tokens.

        Returns True if the model config indicates VLA language input is needed
        (tokenizer_name, vlm_model_name, or language-related input features).
        """
        config = getattr(self._policy, "config", None)
        if config is None:
            return False

        if getattr(config, "tokenizer_name", None):
            return True
        if getattr(config, "vlm_model_name", None):
            return True
        if any("language" in key for key in self._input_features):
            return True

        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the LeRobot model from pretrained path.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If model path is invalid or config cannot be parsed.
            RuntimeError: If model loading fails.
        """
        # XVLA compat: Florence2LanguageConfig.forced_bos_token_id missing
        # in transformers 5.x. Florence2 was originally built with an older
        # transformers that had this attribute. Without this patch, XVLA
        # models fail to load with AttributeError.
        try:
            from transformers.models.florence2.configuration_florence2 import (  # type: ignore[attr-defined,import-not-found]  # noqa: E501
                Florence2LanguageConfig,
            )

            if not hasattr(Florence2LanguageConfig, "forced_bos_token_id"):
                Florence2LanguageConfig.forced_bos_token_id = None
                logger.debug("Patched Florence2LanguageConfig.forced_bos_token_id for XVLA compat")
        except ImportError:
            pass

        logger.info("Loading %s...", self.pretrained_name_or_path)
        start = time.time()

        # Resolve the correct policy class
        if self.policy_type:
            PolicyClass = resolve_policy_class_by_name(self.policy_type)
        else:
            PolicyClass, self.policy_type = resolve_policy_class_from_hub(self.pretrained_name_or_path)

        self._policy = PolicyClass.from_pretrained(self.pretrained_name_or_path)
        assert self._policy is not None

        self._policy.eval()

        # Resolve device: prefer user-requested, then config.device, fallback to first param
        if self.requested_device:
            self._device = torch.device(self.requested_device)
        elif hasattr(self._policy, "config") and hasattr(self._policy.config, "device"):
            self._device = torch.device(self._policy.config.device)
        else:
            self._device = next(self._policy.parameters()).device

        if hasattr(self._policy, "config"):
            config = self._policy.config
            if hasattr(config, "input_features"):
                self._input_features = config.input_features
            if hasattr(config, "output_features"):
                self._output_features = config.output_features

        elapsed = time.time() - start
        logger.info(
            "Loaded %s (type='%s') in %.1fs on %s",
            type(self._policy).__name__,
            self.policy_type,
            elapsed,
            self._device,
        )
        self._loaded = True

        # Auto-detect robot_state_keys from model config if not set
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-generated %d generic state keys (joint_0..joint_%d). "
                    "Set explicit keys with set_robot_state_keys() for meaningful joint names.",
                    action_dim,
                    action_dim - 1,
                )

        # Load processor pipeline (preprocessor + postprocessor)
        if self.use_processor and self.pretrained_name_or_path:
            try:
                self._processor_bridge = ProcessorBridge.from_pretrained(
                    self.pretrained_name_or_path,
                    device=str(self._device) if self._device else None,
                    overrides=self.processor_overrides or {},
                )
                if self._processor_bridge.is_active:
                    logger.info("Processor bridge loaded: %s", self._processor_bridge)
                else:
                    self._processor_bridge = None
                    logger.debug("No processor configs found, using raw obs/action flow")
            except (FileNotFoundError, ValueError, ImportError) as exc:
                # Processor bridge is optional — models work without it via raw obs/action flow.
                # Fail-fast only if the user explicitly requested processor overrides.
                if self.processor_overrides:
                    raise RuntimeError(
                        f"Processor bridge failed to load but processor_overrides were specified: {exc}"
                    ) from exc
                logger.debug("Processor bridge not loaded: %s", exc)
                self._processor_bridge = None

        # Initialize RTC if supported by this policy
        self._init_rtc()

    # ------------------------------------------------------------------
    # Real-Time Chunking (RTC) support
    # ------------------------------------------------------------------

    def _init_rtc(self) -> None:
        """Initialize RTC if the loaded policy supports it.

        RTC is supported by flow-matching policies that implement
        ``predict_action_chunk(**kwargs)``. It requires the policy to have
        an ``rtc_config`` on its config.

        Auto-detection: if ``rtc_enabled=None`` (default), RTC is enabled
        when the model's config has ``rtc_config.enabled=True``.
        """
        if not self._loaded or self._policy is None:
            return

        # Check if policy supports predict_action_chunk (required for RTC)
        has_predict_chunk = hasattr(self._policy, "predict_action_chunk")
        if not has_predict_chunk:
            if self._rtc_requested is True:
                logger.warning(
                    "RTC requested but policy '%s' does not implement predict_action_chunk.",
                    type(self._policy).__name__,
                )
            self._rtc_enabled = False
            return

        # Auto-detect from model config.
        # RTC requires rtc_config on the model — not just predict_action_chunk().
        # In LeRobot 0.5+, predict_action_chunk() is a base class method that ALL
        # policies inherit (ACT, Diffusion, etc.), but only flow-matching policies
        # (Pi0, SmolVLA) have an rtc_config that parameterizes the denoiser for
        # cross-chunk temporal blending.  Without rtc_config, calling
        # predict_action_chunk() with RTC kwargs would either be ignored or crash.
        config = getattr(self._policy, "config", None)
        rtc_config = getattr(config, "rtc_config", None) if config else None

        if self._rtc_requested is None:
            # Auto-detect: use model's rtc_config.enabled
            self._rtc_enabled = rtc_config is not None and getattr(rtc_config, "enabled", False)
        elif self._rtc_requested is True:
            if rtc_config is None:
                # User explicitly asked for RTC, but this policy has no rtc_config.
                # This means it's not a flow-matching policy — warn and disable.
                logger.warning(
                    "RTC requested but policy '%s' has no rtc_config. "
                    "RTC is only supported by flow-matching policies (Pi0, SmolVLA). "
                    "Falling back to select_action().",
                    type(self._policy).__name__,
                )
                self._rtc_enabled = False
            else:
                self._rtc_enabled = True
        else:
            self._rtc_enabled = False

        if not self._rtc_enabled:
            logger.debug("RTC disabled for policy '%s'", type(self._policy).__name__)
            return

        # Read RTC parameters from config, with user overrides
        if rtc_config is not None:
            if self._rtc_execution_horizon is None:
                self._rtc_execution_horizon = getattr(rtc_config, "execution_horizon", 10)
            if self._rtc_max_guidance_weight is None:
                self._rtc_max_guidance_weight = getattr(rtc_config, "max_guidance_weight", 10.0)
        else:
            if self._rtc_execution_horizon is None:
                self._rtc_execution_horizon = 10
            if self._rtc_max_guidance_weight is None:
                self._rtc_max_guidance_weight = 10.0

        logger.info(
            "RTC enabled for '%s': execution_horizon=%d, max_guidance_weight=%.1f",
            type(self._policy).__name__,
            self._rtc_execution_horizon,
            self._rtc_max_guidance_weight,
        )

    def _estimate_inference_delay(self, fps: float = 30.0) -> int:
        """Estimate the number of action steps consumed during inference.

        Uses the p95 latency from recent inference calls to estimate how many
        action steps the robot executed while waiting for the new chunk.

        Args:
            fps: Robot control frequency in Hz. Defaults to 30.

        Returns:
            Estimated delay in action steps (minimum 0).
        """
        if not self._rtc_latency_history:
            return 0

        # Use p95 of recent latencies for robust delay estimation
        latencies = list(self._rtc_latency_history)
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_idx, len(latencies) - 1)]

        delay = int(p95_latency * fps)
        return max(0, delay)

    def _predict_with_rtc(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run inference using predict_action_chunk with RTC kwargs.

        This replaces select_action() for RTC-enabled policies. It:
        1. Calls predict_action_chunk with prev_chunk_left_over and execution_horizon
        2. Tracks inference latency for delay estimation
        3. Stores the new chunk's leftover for the next call

        Args:
            batch: Observation batch tensors ready for the policy.

        Returns:
            Action tensor — first action(s) from the chunk, accounting for
            inference delay.
        """
        inference_start = time.time()

        # Build RTC kwargs for flow-matching denoiser
        rtc_kwargs: dict[str, Any] = {}
        if self._rtc_prev_chunk is not None:
            rtc_kwargs["prev_chunk_left_over"] = self._rtc_prev_chunk
        if self._rtc_execution_horizon is not None:
            rtc_kwargs["execution_horizon"] = self._rtc_execution_horizon

        # predict_action_chunk returns (batch, chunk_size, action_dim)
        assert self._policy is not None, "Policy not loaded"
        action_chunk = self._policy.predict_action_chunk(batch, **rtc_kwargs)

        inference_elapsed = time.time() - inference_start
        self._rtc_latency_history.append(inference_elapsed)

        # Remove batch dim if present: (1, T, A) → (T, A)
        if action_chunk.dim() == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk.squeeze(0)

        # Estimate inference delay — how many steps were consumed while computing
        inference_delay = self._estimate_inference_delay()

        # Store leftover for next RTC call (unconsumed portion of this chunk)
        # The delay represents steps already consumed, so leftover starts after
        # the steps we'll actually return
        steps_to_consume = min(
            max(self.actions_per_step, inference_delay + self.actions_per_step),
            action_chunk.shape[0],
        )
        if steps_to_consume < action_chunk.shape[0]:
            self._rtc_prev_chunk = action_chunk[steps_to_consume:].detach()
        else:
            self._rtc_prev_chunk = None

        # Skip delay steps — they correspond to time spent during inference
        usable_start = min(inference_delay, action_chunk.shape[0] - 1)
        usable_actions = action_chunk[usable_start:]

        # Log RTC details at debug level — throttled to once every 2s regardless of Hz
        _now = time.monotonic()
        if _now - self._rtc_last_log_time >= 2.0:
            self._rtc_last_log_time = _now
            logger.debug(
                "RTC: chunk=%s, delay=%d, usable_start=%d, leftover=%s, avg_latency=%.3fs",
                action_chunk.shape,
                inference_delay,
                usable_start,
                self._rtc_prev_chunk.shape if self._rtc_prev_chunk is not None else None,
                sum(self._rtc_latency_history) / len(self._rtc_latency_history),
            )

        return usable_actions

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).
            instruction: Natural language instruction.

        Returns:
            List of action dicts for robot execution.

        Raises:
            RuntimeError: If model is not loaded and no path is set.
        """
        if not self._loaded:
            if self.pretrained_name_or_path:
                self._load_model()
            else:
                raise RuntimeError(
                    "No model loaded and no pretrained_name_or_path set. Create the policy with a model path."
                )

        observation = dict(observation_dict)
        if instruction and "task" not in observation:
            observation["task"] = instruction

        # When the processor bridge has a preprocessor, delegate normalization
        # and tokenization to it, then fix up any remaining raw arrays/tensors
        # that the pipeline did not convert (e.g. images left as HWC uint8
        # numpy, state tensors missing a batch dimension).
        if self._processor_bridge and self._processor_bridge.has_preprocessor:
            batch = self._processor_bridge.preprocess(observation, instruction=instruction)
            if not isinstance(batch, dict):
                batch = {"observation.state": batch}
            batch = self._fixup_preprocessed_batch(batch)
        else:
            batch = self._build_observation_batch(observation, instruction)

        with torch.inference_mode():
            assert self._policy is not None
            self._policy.eval()
            # RTC uses predict_action_chunk() directly with cross-chunk guidance;
            # non-RTC uses select_action() which manages temporal ensemble + action queue.
            if self._rtc_enabled:
                # RTC (Real-Time Chunking) path: calls predict_action_chunk() directly
                # with prev_chunk for temporal blending. Used only for flow-matching
                # policies that support rtc_config.
                action_tensor = self._predict_with_rtc(batch)
            elif self.actions_per_step > 1:
                # Multi-step path: call predict_action_chunk() directly to get the
                # full action horizon, then slice in _tensor_to_action_dicts().
                # select_action() uses an internal queue and returns only 1 action
                # at a time, so it can't return multiple steps per call.
                action_tensor = self._policy.predict_action_chunk(batch)
            else:
                # Default single-step path: delegates to LeRobot's select_action()
                # which handles temporal ensemble smoothing
                # (config.temporal_ensemble_coeff) and action queue management
                # (n_action_steps > 1) internally.
                # We intentionally use select_action() rather than predict_action_chunk()
                # here to preserve all upstream action scheduling logic.
                action_tensor = self._policy.select_action(batch)

        if self._processor_bridge and self._processor_bridge.has_postprocessor:
            action_tensor = self._processor_bridge.postprocess(action_tensor)

        return self._tensor_to_action_dicts(action_tensor)

    # ------------------------------------------------------------------
    # Observation batch building
    # ------------------------------------------------------------------

    def _fixup_preprocessed_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Fix up a preprocessor-produced batch so every value is a proper batched tensor.

        The LeRobot DataProcessorPipeline may leave some entries in their
        original format (e.g. images as HWC uint8 numpy arrays, state tensors
        without a leading batch dimension).  This method ensures every value
        is a ``torch.Tensor`` on the correct device with the shapes that
        ``policy.select_action()`` expects:

        - Images: ``(B, C, H, W)`` float32
        - State: ``(B, D)`` float32
        - Language tokens/mask: already batched by the tokenizer step

        Args:
            batch: Dict from ``ProcessorBridge.preprocess()``.

        Returns:
            Dict with all values as properly shaped device tensors.
        """
        import torch

        device = self._device or "cpu"
        fixed: dict[str, Any] = {}

        for key, val in batch.items():
            # --- numpy arrays → torch tensors ---
            if isinstance(val, np.ndarray):
                if "image" in key:
                    # HWC uint8 → CHW float32 → (1,C,H,W)
                    img = torch.from_numpy(val).float()
                    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                        img = img.permute(2, 0, 1)  # HWC → CHW
                    if img.ndim == 3:
                        img = img.unsqueeze(0)  # CHW → (1,C,H,W)
                    fixed[key] = img.to(device)
                else:
                    t = torch.from_numpy(val).float()
                    if t.ndim == 1:
                        t = t.unsqueeze(0)  # (D,) → (1,D)
                    fixed[key] = t.to(device)

            # --- torch tensors: ensure batch dim + device ---
            elif isinstance(val, torch.Tensor):
                # Auto-cast float64 → float32: ROS/dynamixel drivers often produce float64
                t = val.float() if val.dtype == torch.float64 else val
                if "image" in key:
                    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                        t = t.permute(2, 0, 1)  # HWC → CHW
                    if t.ndim == 3:
                        t = t.unsqueeze(0)  # CHW → (1,C,H,W)
                elif t.ndim == 1:
                    t = t.unsqueeze(0)  # (D,) → (1,D)
                fixed[key] = t.to(device)

            # --- pass through anything else (strings, etc.) ---
            else:
                fixed[key] = val

        return fixed

    def _build_observation_batch(self, observation_dict: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Convert observation dict to LeRobot-compatible batch tensors.

        Handles two observation formats:
        1. LeRobot native: keys prefixed with "observation." (e.g. "observation.state")
        2. strands-robots native: individual joint keys (e.g. "shoulder", "elbow")

        For VLA models, injects tokenized language instructions into the batch
        as "observation.language.tokens" and "observation.language.attention_mask".

        Args:
            observation_dict: Raw observation dict from robot/sim.
            instruction: Natural language instruction for VLA models.

        Returns:
            Dict of tensors ready for LeRobot policy.select_action().
        """
        batch: dict[str, Any] = {}

        has_lerobot_keys = any(key.startswith("observation.") for key in observation_dict)
        if has_lerobot_keys:
            batch = self._build_batch_from_lerobot_format(observation_dict, batch)
        else:
            batch = self._build_batch_from_strands_format(observation_dict, batch)

        # Inject tokenized language instruction for VLA models.
        # VLA models that use language tokenization expect language tokens as part
        # of the observation batch. We only inject if the model declares
        # language-related input features (tokenizer_name, vlm_model_name).
        if instruction and "observation.language.tokens" not in batch and self._needs_language_tokens():
            result = self._tokenize_instruction(instruction)
            if result is not None:
                tokens, mask = result
                batch["observation.language.tokens"] = tokens
                if mask is not None:
                    batch["observation.language.attention_mask"] = mask
                logger.debug("VLA tokenized instruction: '%s...' -> %s tokens", instruction[:50], tokens.shape)

        # Fill task key for models that read it directly from the batch
        # (e.g. some VLA models read "task" or "observation.task" from the
        # input dict rather than using tokenized language tokens)
        if instruction and has_lerobot_keys:
            needs_task = any("task" in key for key in self._input_features) and "task" not in batch
            if needs_task:
                for feat_name in self._input_features:
                    if "task" in feat_name and feat_name not in batch:
                        batch[feat_name] = instruction

        # Validate required image features are present. Missing images would
        # cause the model to produce garbage outputs silently.
        for feat_name in self._input_features:
            if feat_name not in batch and "image" in feat_name:
                raise ValueError(
                    f"Missing required image feature '{feat_name}' in observation. "
                    f"The model expects this camera input. Provide it in the observation dict "
                    f"or check your camera configuration."
                )

        return batch

    def _build_batch_from_lerobot_format(
        self, observation_dict: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Build batch from observation dict already in LeRobot format (observation.* keys).

        Converts each value to the appropriate tensor format:
        - Images (HWC uint8) → CHW float32 [0, 1] with batch dim
        - State vectors → float32 with batch dim
        - Scalars → float32 tensor with batch dim

        Non-numeric types (strings, pre-batched int64 tokens) are passed through
        unchanged — LeRobot expects these as-is for task descriptions and
        pre-tokenized inputs.

        Args:
            observation_dict: Dict with "observation.*" prefixed keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with tensors on target device.
        """
        for key, value in observation_dict.items():
            # Determine if this key represents an image based on key name
            # or shape metadata from the model's input_features config
            is_image = "image" in key or (
                key in self._input_features
                and hasattr(self._input_features.get(key), "shape")
                and len(getattr(self._input_features.get(key), "shape", ())) >= 2
            )

            if isinstance(value, torch.Tensor):
                # Auto-cast float64 → float32: common from ROS/dynamixel drivers
                tensor = value.float() if value.dtype == torch.float64 else value
                # Detect unlabeled images by shape: 3D tensor with channel-last layout
                if not is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    is_image = True
                # HWC → CHW: LeRobot expects channel-first image layout
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # Add batch dimension (required by policy.select_action)
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy()).float()
                if not is_image and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                    is_image = True
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # uint8 images are [0, 255] — normalize to [0, 1] for model input
                if is_image and value.dtype == np.uint8:
                    tensor = tensor / 255.0
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif value.ndim < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, (int, float)):
                batch[key] = torch.tensor([value], dtype=torch.float32).unsqueeze(0).to(self._device)

            elif isinstance(value, (list, tuple)):
                try:
                    array = np.array(value, dtype=np.float32)
                except (ValueError, TypeError):
                    # Non-numeric lists (e.g. string lists) — skip silently, they aren't tensor data
                    logger.debug("Skipping non-numeric list/tuple for key in observation batch")
                    continue
                tensor = torch.from_numpy(array).float()
                if array.ndim >= 2:
                    if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                        tensor = tensor.permute(2, 0, 1)
                    if is_image and array.dtype == np.uint8:
                        tensor = tensor / 255.0
                    if is_image and tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    batch[key] = tensor.to(self._device)
                else:
                    batch[key] = tensor.unsqueeze(0).to(self._device)

        return batch

    def _build_batch_from_strands_format(
        self, observation_dict: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Build batch from strands-robots native observation format.

        Maps individual joint keys (e.g. {"shoulder": 0.5, "elbow": -0.3}) to
        LeRobot's "observation.state" tensor using robot_state_keys ordering.
        Camera images are matched to the model's image input features by
        assigning each ndarray with ndim >= 2 to the first unoccupied image slot.

        Args:
            observation_dict: Dict with individual joint/image keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with "observation.state" and image tensors.

        Raises:
            ValueError: If robot_state_keys is empty (cannot map joints).
        """
        if not self.robot_state_keys:
            raise ValueError(
                "robot_state_keys is empty — cannot map observation to state tensor. "
                "Call set_robot_state_keys() with the robot's motor names."
            )

        # Collect state values in robot_state_keys order. Each key maps to a
        # single float value representing one joint/motor position.
        state_values = []
        for key in self.robot_state_keys:
            if key in observation_dict:
                value = observation_dict[key]
                if isinstance(value, (int, float)):
                    state_values.append(float(value))
                elif isinstance(value, (np.floating, np.integer)):
                    state_values.append(float(value))
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    state_values.append(float(value))

        if state_values:
            # Validate state dimension matches what the model expects.
            # Dimension mismatches cause shape errors in the forward pass.
            state_feature = self._input_features.get("observation.state")
            if state_feature:
                expected_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else len(state_values)
                if len(state_values) != expected_dim:
                    raise ValueError(
                        f"State dimension mismatch: got {len(state_values)} values from "
                        f"robot_state_keys but model expects {expected_dim}. "
                        f"Check that robot_state_keys matches your robot's actual joint count."
                    )
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Map camera images to model's image input features.
        # Non-state ndarray values with ndim >= 2 are assumed to be images.
        # Each image is matched to the first unoccupied image feature slot
        # from the model's input_features config.
        for key, value in observation_dict.items():
            if key in self.robot_state_keys:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                image_tensor = torch.from_numpy(value.copy()).float()
                # HWC → CHW: convert from camera output format to model input format
                if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3, 4):
                    image_tensor = image_tensor.permute(2, 0, 1)
                # uint8 [0, 255] → float32 [0, 1]
                if value.dtype == np.uint8:
                    image_tensor = image_tensor / 255.0
                # Assign to first available image feature slot
                for feat_name in self._input_features:
                    if "image" in feat_name and feat_name not in batch:
                        batch[feat_name] = image_tensor.unsqueeze(0).to(self._device)
                        break

        return batch

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _tensor_to_action_dicts(self, action_tensor: torch.Tensor) -> list[dict[str, Any]]:
        """Convert action tensor to list of robot action dicts.

        Maps tensor values to robot_state_keys by index. Handles:
        - 1D tensor: single action step (shape [action_dim])
        - 2D tensor: action sequence (shape [horizon, action_dim])
        - 3D tensor: batched sequence (shape [batch, horizon, action_dim])

        Args:
            action_tensor: Raw action tensor from policy.select_action().

        Returns:
            List of action dicts, length capped by actions_per_step.

        Raises:
            RuntimeError: If robot_state_keys is empty.
        """
        action_array = action_tensor.cpu().numpy()

        # Normalize tensor shape to a list of 1D action arrays
        if action_array.ndim == 1:
            actions_list = [action_array]
        elif action_array.ndim == 2:
            actions_list = [action_array[i] for i in range(min(len(action_array), self.actions_per_step))]
        elif action_array.ndim == 3:
            # Batched: take first batch element, then slice horizon
            actions_list = [action_array[0, i] for i in range(min(action_array.shape[1], self.actions_per_step))]
        else:
            actions_list = [action_array.flatten()]

        if not self.robot_state_keys:
            raise RuntimeError(
                "Cannot convert action tensor to dicts: robot_state_keys is empty. "
                "Call set_robot_state_keys() before inference."
            )

        result = []
        for action_values in actions_list:
            action_dict = {}
            for index, key in enumerate(self.robot_state_keys):
                action_dict[key] = float(action_values[index]) if index < len(action_values) else 0.0
            result.append(action_dict)

        return result
