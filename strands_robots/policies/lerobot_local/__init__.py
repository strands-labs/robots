"""LeRobot Local Policy — Direct HuggingFace model inference (no server needed).

Uses LeRobot's own factory (`get_policy_class`) for auto-detection.
No hardcoded policy classes — any model LeRobot supports, we support.

Usage:
    policy = create_policy(
        "lerobot_local",
        pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
    )
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .. import Policy

logger = logging.getLogger(__name__)


def _resolve_policy_class_from_hub(pretrained_name_or_path: str):
    """Resolve the LeRobot policy class from a pretrained path or HF repo.

    Zero hardcoding — uses PreTrainedPolicy.from_pretrained()
    which handles config resolution, class lookup, and weight loading
    internally via the draccus config registry.

    Flow:
        1. PreTrainedConfig.from_pretrained() loads config.json from HF/local
        2. Config's draccus type field resolves to the correct PolicyClass
        3. PreTrainedPolicy.from_pretrained() loads weights into that class

    Falls back to reading config.json manually + class name matching if
    the draccus path fails (e.g. third-party policies not in registry).
    """
    import json
    from pathlib import Path

    # Strategy 1: Use PreTrainedConfig draccus resolution
    try:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy_type = getattr(config, "type", type(config).__name__.replace("Config", "").lower())
        logger.info(f"Auto-resolved via PreTrainedConfig: '{pretrained_name_or_path}' -> type='{policy_type}'")
        # Return the config's associated policy class
        from lerobot.policies.pretrained import PreTrainedPolicy

        return PreTrainedPolicy, policy_type
    except Exception as e:
        logger.debug("PreTrainedConfig resolution failed, trying manual: %s", e)

    # Strategy 2: Manual config.json reading (fallback for custom/third-party)
    policy_type = None
    local_path = Path(pretrained_name_or_path)
    if local_path.is_dir() and (local_path / "config.json").exists():
        with open(local_path / "config.json") as f:
            config = json.load(f)
        policy_type = config.get("type")
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(pretrained_name_or_path, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            policy_type = config.get("type")
        except Exception as e:
            logger.warning("Could not download config.json: %s", e)

    if not policy_type:
        raise ValueError(
            f"Could not determine policy type from '{pretrained_name_or_path}'. "
            f"No 'type' field found in config.json. "
            f"Pass policy_type= explicitly."
        )

    PolicyClass = _resolve_policy_class_by_name(policy_type)
    logger.info(f"Auto-resolved: '{pretrained_name_or_path}' -> type='{policy_type}' -> {PolicyClass.__name__}")
    return PolicyClass, policy_type


def _resolve_policy_class_by_name(policy_type: str):
    """Resolve policy class from an explicit type string.

    Uses PreTrainedConfig.get_known_choices() + make_policy()
    or direct module import from lerobot.policies.{policy_type}.
    Falls back to legacy get_policy_class() for older lerobot versions.
    """
    # Strategy 1: Direct module import
    try:
        import importlib

        mod = importlib.import_module(f"lerobot.policies.{policy_type}")
        # Find the Policy class (convention: {Type}Policy)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("Policy")
                and attr_name != "PreTrainedPolicy"
                and hasattr(obj, "from_pretrained")
            ):
                return obj
    except ImportError:
        pass

    # Strategy 2: PreTrainedPolicy (generic loader)
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy

        return PreTrainedPolicy
    except ImportError:
        pass

    # Strategy 3: Legacy get_policy_class
    try:
        from lerobot.policies.factory import get_policy_class

        return get_policy_class(policy_type)
    except (ImportError, AttributeError, RuntimeError) as e:
        raise ImportError(
            f"Could not resolve LeRobot policy class for type '{policy_type}': {e}. Ensure lerobot is installed."
        ) from e


class LerobotLocalPolicy(Policy):
    """Policy that loads and runs LeRobot models directly (no server).

    Auto-detects policy type from HF config.json → delegates to LeRobot's
    own get_policy_class() factory. Zero hardcoded policy mappings.

    Optionally loads the model's processor pipeline (preprocessor.json /
    postprocessor.json) for automatic normalization, device transfer,
    observation formatting, and action unnormalization.

    Architecture:
        Observation (dict)
            → ProcessorBridge.preprocess (normalize, device, crop, ...)
            → LeRobot PreTrainedPolicy.select_action
            → ProcessorBridge.postprocess (unnormalize, delta-action, ...)
            → Action dict
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "",
        policy_type: str = None,
        device: str = None,
        actions_per_step: int = 1,
        use_processor: bool = True,
        processor_overrides: dict = None,
        **kwargs,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.policy_type = policy_type
        self.requested_device = device
        self.actions_per_step = actions_per_step
        self.use_processor = use_processor
        self.processor_overrides = processor_overrides
        self.robot_state_keys: List[str] = []

        self._policy = None
        self._device = None
        self._input_features = {}
        self._output_features = {}
        self._loaded = False
        self._processor_bridge = None
        self._tokenizer = None
        self._tokenizer_max_length: int = 48
        self._tokenizer_padding_side: str = "right"
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        if pretrained_name_or_path:
            self._load_model()

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys for observation→tensor mapping.

        If robot_state_keys is provided and non-empty, uses those directly.
        If empty, auto-detects from the model's output_features (action dimension).

        Auto-detection:
        1. Uses provided keys if non-empty
        2. Falls back to model output_features action dimension
        3. Falls back to model input_features state dimension
        4. Last resort: generic joint_0..joint_N keys
        """
        if robot_state_keys:
            self.robot_state_keys = robot_state_keys
            logger.info(
                f"LeRobot local state keys set: {len(self.robot_state_keys)} keys = {self.robot_state_keys[:5]}{'...' if len(self.robot_state_keys) > 5 else ''}"
            )
            return

        # Auto-detect from model config
        if self._loaded and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.warning(
                    f"robot_state_keys empty — auto-detected {action_dim} keys from "
                    f"model output_features.action.shape={action_feat.shape}. "
                    f"For meaningful names, pass the robot's actual joint names."
                )
                return

        if self._loaded and self._input_features:
            state_feat = self._input_features.get("observation.state")
            if state_feat and hasattr(state_feat, "shape") and state_feat.shape:
                state_dim = state_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(state_dim)]
                logger.warning(
                    f"robot_state_keys empty — auto-detected {state_dim} keys from "
                    f"model input_features.observation.state.shape={state_feat.shape}."
                )
                return

        logger.warning(
            "robot_state_keys empty and no model features available for auto-detection. "
            "Actions will be empty dicts. Call set_robot_state_keys() with actual joint names."
        )

    def _resolve_tokenizer(self):
        """Resolve and cache the tokenizer for VLA language token injection.

        Resolution order:
            1. Explicit ``tokenizer_name`` on policy config (e.g. xvla)
            2. ``vlm_model_name`` on policy config (e.g. SmolVLA → SmolVLM2)
            3. Policy's own ``.processor.tokenizer`` (e.g. Paligemma-based)

        Also reads ``tokenizer_max_length`` and ``tokenizer_padding_side``
        from the policy config when available.

        Returns the tokenizer or ``None``.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        if not self._loaded or not self._policy:
            return None

        cfg = getattr(self._policy, "config", None)
        if cfg is None:
            return None

        # Read sizing / padding from config (used by all paths)
        self._tokenizer_max_length = getattr(cfg, "tokenizer_max_length", 48)
        self._tokenizer_padding_side = getattr(cfg, "tokenizer_padding_side", "right")

        # 1. tokenizer_name (explicit)
        tokenizer_id = getattr(cfg, "tokenizer_name", None)

        # 2. vlm_model_name (SmolVLA, etc.)
        if not tokenizer_id:
            tokenizer_id = getattr(cfg, "vlm_model_name", None)

        if tokenizer_id:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                self._tokenizer.padding_side = self._tokenizer_padding_side
                logger.info(
                    "Auto-resolved tokenizer from '%s' (%s)",
                    tokenizer_id,
                    type(self._tokenizer).__name__,
                )
                return self._tokenizer
            except Exception as e:
                logger.warning("Failed to load tokenizer from '%s': %s", tokenizer_id, e)

        # 3. policy.processor.tokenizer
        proc = getattr(self._policy, "processor", None)
        if proc and hasattr(proc, "tokenizer"):
            self._tokenizer = proc.tokenizer
            self._tokenizer.padding_side = self._tokenizer_padding_side
            logger.info(
                "Using policy's built-in processor tokenizer (%s)",
                type(self._tokenizer).__name__,
            )
            return self._tokenizer

        return None

    def _tokenize_instruction(self, instruction: str):
        """Tokenize an instruction into ``(input_ids, attention_mask)`` tensors.

        Returns ``None`` if no tokenizer is available.
        """
        tokenizer = self._resolve_tokenizer()
        if tokenizer is None or not instruction:
            return None

        max_len = self._tokenizer_max_length
        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        tokens = encoded["input_ids"].to(self._device)
        mask = encoded.get("attention_mask")
        if mask is not None:
            mask = mask.bool().to(self._device)
        return tokens, mask

    def _needs_language_tokens(self) -> bool:
        """Check whether this policy requires observation.language.tokens."""
        cfg = getattr(self._policy, "config", None)
        if cfg is None:
            return False

        # Explicit tokenizer_name or vlm_model_name → VLA that needs tokens
        if getattr(cfg, "tokenizer_name", None):
            return True
        if getattr(cfg, "vlm_model_name", None):
            return True

        # Input features mention 'language'
        if any("language" in k for k in self._input_features):
            return True

        return False

    def _load_model(self):
        import warnings

        import torch

        warnings.filterwarnings("ignore", message=".*Device.*")

        # XVLA compat: Florence2LanguageConfig.forced_bos_token_id missing in transformers 5.x
        try:
            from transformers.models.florence2.configuration_florence2 import (
                Florence2LanguageConfig,
            )

            if not hasattr(Florence2LanguageConfig, "forced_bos_token_id"):
                Florence2LanguageConfig.forced_bos_token_id = None
                logger.debug("Patched Florence2LanguageConfig.forced_bos_token_id")
        except (ImportError, Exception):
            pass

        logger.info("Loading %s...", self.pretrained_name_or_path)
        start = time.time()

        # Resolve the correct policy class — zero hardcoding
        if self.policy_type:
            # Explicit type given — use LeRobot's factory directly
            PolicyClass = _resolve_policy_class_by_name(self.policy_type)
        else:
            # Auto-detect from HF config.json
            PolicyClass, self.policy_type = _resolve_policy_class_from_hub(self.pretrained_name_or_path)

        self._policy = PolicyClass.from_pretrained(self.pretrained_name_or_path)
        self._policy.eval()
        self._device = next(self._policy.parameters()).device

        if hasattr(self._policy, "config"):
            cfg = self._policy.config
            if hasattr(cfg, "input_features"):
                self._input_features = cfg.input_features
            if hasattr(cfg, "output_features"):
                self._output_features = cfg.output_features

        elapsed = time.time() - start
        logger.info(f"Loaded {PolicyClass.__name__} (type='{self.policy_type}') in {elapsed:.1f}s on {self._device}")
        self._loaded = True

        # Auto-detect robot_state_keys from model config if not set
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.warning(
                    f"robot_state_keys not set — auto-generated {action_dim} generic keys "
                    f"(joint_0..joint_{action_dim - 1}). Set explicit keys with "
                    f"set_robot_state_keys() for meaningful joint names."
                )

        # Try to load processor pipeline (preprocessor + postprocessor)
        if self.use_processor and self.pretrained_name_or_path:
            try:
                from strands_robots.processor import ProcessorBridge

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
            except Exception as e:
                logger.debug("Processor bridge not loaded: %s", e)
                self._processor_bridge = None

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        if not self._loaded:
            if self.pretrained_name_or_path:
                self._load_model()
            else:
                return self._zero_actions()

        import torch

        try:
            # Inject task/instruction into obs for VLA tokenizer preprocessing
            obs = dict(observation_dict)
            if instruction and "task" not in obs:
                obs["task"] = instruction

            # Apply preprocessor if available
            if self._processor_bridge and self._processor_bridge.has_preprocessor:
                obs = self._processor_bridge.preprocess(obs)

            batch = self._build_observation_batch(obs, instruction)
            with torch.no_grad():
                action_tensor = self._policy.select_action(batch)

            # Apply postprocessor if available
            if self._processor_bridge and self._processor_bridge.has_postprocessor:
                action_tensor = self._processor_bridge.postprocess(action_tensor)

            self._consecutive_failures = 0
            return self._tensor_to_action_dicts(action_tensor)
        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Inference error ({self._consecutive_failures}/{self._max_consecutive_failures}): {e}")
            if self._consecutive_failures >= self._max_consecutive_failures:
                raise RuntimeError(
                    f"LeRobot policy failed {self._consecutive_failures} consecutive times, last error: {e}"
                ) from e
            return self._zero_actions()

    def select_action_sync(self, observation_dict: Dict[str, Any]) -> np.ndarray:
        """Synchronous inference — returns raw action numpy array.

        Convenience for simulation loops that don't need async.
        Applies processor pipeline if available.

        For VLA models (xvla, smolvla, etc.) that require language tokens:
        Uses the processor pipeline's TokenizerProcessorStep to generate
        observation.language.tokens from a 'task' key in the observation dict.
        """
        import torch

        if not self._loaded:
            self._load_model()

        # Apply preprocessor if available
        obs = observation_dict
        if self._processor_bridge and self._processor_bridge.has_preprocessor:
            obs = self._processor_bridge.preprocess(obs)

        batch = self._build_observation_batch(obs, "")
        with torch.no_grad():
            action_tensor = self._policy.select_action(batch)

        # Apply postprocessor if available
        if self._processor_bridge and self._processor_bridge.has_postprocessor:
            action_tensor = self._processor_bridge.postprocess(action_tensor)

        if hasattr(action_tensor, "cpu"):
            return action_tensor.cpu().numpy()
        return np.asarray(action_tensor)

    def _build_observation_batch(self, observation_dict: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        import torch

        batch = {}

        # If already in LeRobot format, pass through
        has_lerobot_keys = any(k.startswith("observation.") for k in observation_dict)
        if has_lerobot_keys:
            for k, v in observation_dict.items():
                # Detect image-like data by shape or key name
                is_image = (
                    "image" in k
                    or k in self._input_features
                    and hasattr(self._input_features.get(k), "shape")
                    and len(getattr(self._input_features.get(k), "shape", ())) >= 2
                )

                if isinstance(v, torch.Tensor):
                    t = v
                    # Detect image by shape: 3D with channel-like last dim
                    if not is_image and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                        is_image = True
                    if is_image and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                        t = t.permute(2, 0, 1)
                    if is_image and t.dim() == 3:
                        t = t.unsqueeze(0)
                    elif t.dim() < 2 and not is_image:
                        t = t.unsqueeze(0)
                    batch[k] = t.to(self._device)
                elif isinstance(v, np.ndarray):
                    if v.ndim >= 2:
                        if not is_image and v.ndim == 3 and v.shape[-1] in (1, 3, 4):
                            is_image = True
                        t = torch.from_numpy(v.copy()).float()
                        if is_image and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                            t = t.permute(2, 0, 1)
                        if is_image and v.dtype == np.uint8:
                            t = t / 255.0
                        if is_image and t.dim() == 3:
                            t = t.unsqueeze(0)
                        elif t.dim() < 2:
                            t = t.unsqueeze(0)
                        batch[k] = t.to(self._device)
                    else:
                        batch[k] = torch.from_numpy(v.copy()).float().unsqueeze(0).to(self._device)
                elif isinstance(v, (int, float)):
                    batch[k] = torch.tensor([v], dtype=torch.float32).unsqueeze(0).to(self._device)
                elif isinstance(v, (list, tuple)):
                    # Simulation backends may return lists for observation.state — convert to tensors
                    # Convert list/tuple to tensor (handles 1D state vectors and nested lists)
                    try:
                        arr = np.array(v, dtype=np.float32)
                    except (ValueError, TypeError):
                        # Skip non-numeric lists (e.g. string lists)
                        continue
                    if arr.ndim >= 2:
                        t = torch.from_numpy(arr).float()
                        if is_image and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                            t = t.permute(2, 0, 1)
                        if is_image and arr.dtype == np.uint8:
                            t = t / 255.0
                        if is_image and t.dim() == 3:
                            t = t.unsqueeze(0)
                        batch[k] = t.to(self._device)
                    else:
                        batch[k] = torch.from_numpy(arr).float().unsqueeze(0).to(self._device)
                # Pass through non-tensor types (e.g. already-batched int64 tokens)
                # This handles observation.language.tokens from the preprocessor

            # VLA language token injection: if preprocessor generated language tokens, keep them
            # If not present but needed, auto-tokenize using _resolve_tokenizer()
            if "observation.language.tokens" not in batch and instruction:
                if self._needs_language_tokens():
                    try:
                        result = self._tokenize_instruction(instruction)
                        if result is not None:
                            tokens, mask = result
                            batch["observation.language.tokens"] = tokens
                            if mask is not None:
                                batch["observation.language.attention_mask"] = mask
                    except Exception as e:
                        logger.debug("VLA tokenization fallback failed: %s", e)

            return batch

        # Convert strands-robots format to LeRobot format
        if not self.robot_state_keys:
            logger.warning(
                "robot_state_keys is empty — observation.state will be skipped. "
                "Call set_robot_state_keys() with the robot's motor names for proper state handling."
            )

        state_values = []
        for key in self.robot_state_keys:
            if key in observation_dict:
                val = observation_dict[key]
                if isinstance(val, (int, float)):
                    state_values.append(float(val))
                elif isinstance(val, (np.floating, np.integer)):
                    state_values.append(float(val))
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    state_values.append(float(val))

        if state_values:
            state_feature = self._input_features.get("observation.state")
            if state_feature:
                expected_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else len(state_values)
                while len(state_values) < expected_dim:
                    state_values.append(0.0)
                state_values = state_values[:expected_dim]
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Camera images
        for key, value in observation_dict.items():
            if key in self.robot_state_keys:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                # use .copy() for memory safety, dtype for normalization
                img_tensor = torch.from_numpy(value.copy()).float()
                if img_tensor.dim() == 3 and img_tensor.shape[-1] in (1, 3, 4):
                    img_tensor = img_tensor.permute(2, 0, 1)
                if value.dtype == np.uint8:
                    img_tensor = img_tensor / 255.0
                for feat_name in self._input_features:
                    if "image" in feat_name and feat_name not in batch:
                        batch[feat_name] = img_tensor.unsqueeze(0).to(self._device)
                        break

        # VLA language token injection for models that need observation.language.tokens
        # (xvla, smolvla, etc.) — auto-resolved from tokenizer_name or vlm_model_name
        if instruction:
            needs_language = "observation.language.tokens" not in batch and self._needs_language_tokens()
            needs_task = any("task" in k for k in self._input_features) and "task" not in batch

            if needs_language:
                try:
                    result = self._tokenize_instruction(instruction)
                    if result is not None:
                        tokens, mask = result
                        batch["observation.language.tokens"] = tokens
                        if mask is not None:
                            batch["observation.language.attention_mask"] = mask
                        logger.debug(
                            "VLA tokenized instruction: '%s...' -> %s",
                            instruction[:50],
                            tokens.shape,
                        )
                except Exception as e:
                    logger.warning("VLA language token injection failed: %s", e)

            if needs_task:
                for feat_name in self._input_features:
                    if "task" in feat_name and feat_name not in batch:
                        batch[feat_name] = instruction

        # Fill missing image features with zeros
        for feat_name, feat_info in self._input_features.items():
            if feat_name not in batch and "image" in feat_name:
                shape = feat_info.shape if hasattr(feat_info, "shape") else (3, 480, 640)
                batch[feat_name] = torch.zeros(1, *shape, device=self._device)

        return batch

    def _tensor_to_action_dicts(self, action_tensor) -> List[Dict[str, Any]]:
        action_np = action_tensor.cpu().numpy()

        if action_np.ndim == 1:
            actions_list = [action_np]
        elif action_np.ndim == 2:
            actions_list = [action_np[i] for i in range(min(len(action_np), self.actions_per_step))]
        elif action_np.ndim == 3:
            actions_list = [action_np[0, i] for i in range(min(action_np.shape[1], self.actions_per_step))]
        else:
            actions_list = [action_np.flatten()]

        result = []
        for action_arr in actions_list:
            action_dict = {}
            for j, key in enumerate(self.robot_state_keys):
                if j < len(action_arr):
                    action_dict[key] = float(action_arr[j])
                else:
                    action_dict[key] = 0.0
            result.append(action_dict)

        return result if result else self._zero_actions()

    def _zero_actions(self) -> List[Dict[str, Any]]:
        return [{key: 0.0 for key in self.robot_state_keys}]

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "provider": "lerobot_local",
            "model_id": self.pretrained_name_or_path,
            "policy_type": self.policy_type,
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
        }
        if self._loaded:
            info["input_features"] = {
                k: str(v.shape) if hasattr(v, "shape") else str(v) for k, v in self._input_features.items()
            }
            info["output_features"] = {
                k: str(v.shape) if hasattr(v, "shape") else str(v) for k, v in self._output_features.items()
            }
            info["policy_class"] = type(self._policy).__name__
            info["n_parameters"] = sum(p.numel() for p in self._policy.parameters())
        if self._processor_bridge:
            info["processor"] = self._processor_bridge.get_info()
        return info


__all__ = ["LerobotLocalPolicy"]
