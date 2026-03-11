"""Processor Pipeline Bridge for strands-robots.

Integrates LeRobot's DataProcessorPipeline into the strands-robots policy flow.
Handles observation preprocessing and action postprocessing using the model's
own saved pipeline configs (preprocessor.json / postprocessor.json).

Bridges LeRobot DataProcessorPipeline for automatic observation and action normalization.

Architecture:
    Robot observation (dict)
        → ProcessorBridge.preprocess(obs)
            → LeRobot DataProcessorPipeline (normalize, device, batch, ...)
        → Policy.select_action(processed_obs)
        → ProcessorBridge.postprocess(action)
            → LeRobot DataProcessorPipeline (unnormalize, delta-action, ...)
        → Robot action (dict)

Usage:
    from strands_robots.processor import ProcessorBridge

    # Auto-load from pretrained model (reads preprocessor.json + postprocessor.json)
    bridge = ProcessorBridge.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

    # Or from a local checkpoint
    bridge = ProcessorBridge.from_pretrained("/path/to/checkpoint")

    # Use in policy loop
    processed_obs = bridge.preprocess(raw_observation)
    action_tensor = policy.select_action(processed_obs)
    robot_action = bridge.postprocess(action_tensor)

    # Or wrap an existing policy
    wrapped = bridge.wrap_policy(my_lerobot_local_policy)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

# Standard pipeline config filenames used by LeRobot
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"


def _try_import_processor():
    """Lazy import LeRobot processor module."""
    try:
        from lerobot.processor.converters import (
            batch_to_transition,
            observation_to_transition,
            transition_to_batch,
            transition_to_observation,
        )
        from lerobot.processor.core import EnvTransition, TransitionKey
        from lerobot.processor.pipeline import DataProcessorPipeline

        return {
            "DataProcessorPipeline": DataProcessorPipeline,
            "batch_to_transition": batch_to_transition,
            "transition_to_batch": transition_to_batch,
            "observation_to_transition": observation_to_transition,
            "transition_to_observation": transition_to_observation,
            "EnvTransition": EnvTransition,
            "TransitionKey": TransitionKey,
        }
    except ImportError:
        return None


class ProcessorBridge:
    """Bridge between strands-robots observation/action format and LeRobot's processor pipeline.

    Handles:
    - Loading preprocessor + postprocessor from pretrained model dirs / HF Hub
    - Converting strands-robots observation dicts to LeRobot EnvTransition format
    - Running the pipeline steps (normalize, device transfer, observation processing, etc.)
    - Converting processed output back to strands-robots format

    Thread-safe: each bridge instance holds its own pipeline state.
    """

    def __init__(
        self,
        preprocessor=None,
        postprocessor=None,
        device: Optional[str] = None,
    ):
        """Initialize with optional pre/post processor pipelines.

        Args:
            preprocessor: LeRobot DataProcessorPipeline for observation preprocessing
            postprocessor: LeRobot DataProcessorPipeline for action postprocessing
            device: Target device for tensor operations (auto-detected if None)
        """
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._modules = _try_import_processor()

        if self._modules is None:
            logger.warning(
                "LeRobot processor module not available. "
                "ProcessorBridge will pass data through unchanged. "
                "Install LeRobot >= 0.4.0 for full processor support."
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        device: Optional[str] = None,
        preprocessor_config: str = PREPROCESSOR_CONFIG,
        postprocessor_config: str = POSTPROCESSOR_CONFIG,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ProcessorBridge":
        """Load processor pipelines from a pretrained model.

        Tries to load preprocessor.json and postprocessor.json from the model
        directory or HuggingFace Hub. If either doesn't exist, that pipeline
        is skipped (passthrough).

        Args:
            pretrained_name_or_path: HF model ID or local path
            device: Target device (auto-detected if None)
            preprocessor_config: Filename for preprocessor config
            postprocessor_config: Filename for postprocessor config
            overrides: Dict of step overrides (passed to both pipelines)

        Returns:
            ProcessorBridge instance with loaded pipelines
        """
        modules = _try_import_processor()
        if modules is None:
            logger.info("LeRobot processor not available, creating passthrough bridge")
            return cls(device=device)

        DataProcessorPipeline = modules["DataProcessorPipeline"]

        preprocessor = None
        postprocessor = None

        # Load preprocessor
        try:
            preprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=preprocessor_config,
                overrides=overrides or {},
            )
            logger.info(
                f"Loaded preprocessor from {pretrained_name_or_path}: "
                f"{len(preprocessor)} steps"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.debug("No preprocessor found: %s", e)
        except Exception as e:
            logger.warning("Failed to load preprocessor: %s", e)

        # Load postprocessor
        try:
            postprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=postprocessor_config,
                overrides=overrides or {},
            )
            logger.info(
                f"Loaded postprocessor from {pretrained_name_or_path}: "
                f"{len(postprocessor)} steps"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.debug("No postprocessor found: %s", e)
        except Exception as e:
            logger.warning("Failed to load postprocessor: %s", e)

        return cls(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )

    @property
    def has_preprocessor(self) -> bool:
        """Whether a preprocessor pipeline is loaded."""
        return self._preprocessor is not None

    @property
    def has_postprocessor(self) -> bool:
        """Whether a postprocessor pipeline is loaded."""
        return self._postprocessor is not None

    @property
    def is_active(self) -> bool:
        """Whether any processing pipeline is active."""
        return self.has_preprocessor or self.has_postprocessor

    def preprocess(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a raw observation dict through the pipeline.

        If no preprocessor is loaded, returns observation unchanged.

        Args:
            observation: Raw observation dict from robot/sim
                         (camera images as numpy arrays, state as floats/arrays)

        Returns:
            Processed observation dict (tensors on target device, normalized, etc.)
        """
        if self._preprocessor is None or self._modules is None:
            return observation

        try:
            # Use the pipeline's process_observation convenience method
            # which wraps the obs in an EnvTransition, runs all steps,
            # and extracts the observation back out
            processed = self._preprocessor.process_observation(observation)
            return processed
        except Exception as e:
            logger.warning("Preprocessor failed, passing raw observation: %s", e)
            return observation

    def postprocess(self, action: Any) -> Any:
        """Postprocess a policy action through the pipeline.

        If no postprocessor is loaded, returns action unchanged.

        Args:
            action: Raw action from policy (tensor or dict)

        Returns:
            Processed action (unnormalized, converted to robot format, etc.)
        """
        if self._postprocessor is None or self._modules is None:
            return action

        try:
            processed = self._postprocessor.process_action(action)
            return processed
        except Exception as e:
            logger.warning("Postprocessor failed, passing raw action: %s", e)
            return action

    def process_full_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """Process a full EnvTransition dict through the preprocessor.

        For advanced use cases where you need to process observation + action + reward
        together (e.g., during training data preparation).

        Args:
            transition: Full transition dict with observation, action, reward, etc.

        Returns:
            Processed transition dict
        """
        if self._preprocessor is None or self._modules is None:
            return transition

        try:
            return self._preprocessor(transition)
        except Exception as e:
            logger.warning("Full transition processing failed: %s", e)
            return transition

    def reset(self):
        """Reset pipeline state (e.g., clear running stats in stateful steps)."""
        if self._preprocessor is not None:
            self._preprocessor.reset()
        if self._postprocessor is not None:
            self._postprocessor.reset()

    def wrap_policy(self, policy):
        """Wrap a strands-robots Policy with pre/post processing.

        Returns a new policy-like object that automatically applies
        preprocessing to observations and postprocessing to actions.

        Args:
            policy: A strands-robots Policy instance

        Returns:
            ProcessedPolicy wrapper
        """
        return ProcessedPolicy(policy, self)

    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded pipelines."""
        info = {
            "has_preprocessor": self.has_preprocessor,
            "has_postprocessor": self.has_postprocessor,
            "is_active": self.is_active,
            "device": self._device,
        }
        if self._preprocessor is not None:
            info["preprocessor_steps"] = len(self._preprocessor)
            info["preprocessor_step_names"] = [
                type(s).__name__ for s in self._preprocessor.steps
            ]
        if self._postprocessor is not None:
            info["postprocessor_steps"] = len(self._postprocessor)
            info["postprocessor_step_names"] = [
                type(s).__name__ for s in self._postprocessor.steps
            ]
        return info

    def __repr__(self) -> str:
        pre = (
            f"pre={len(self._preprocessor)}steps" if self._preprocessor else "pre=None"
        )
        post = (
            f"post={len(self._postprocessor)}steps"
            if self._postprocessor
            else "post=None"
        )
        return f"ProcessorBridge({pre}, {post})"


class ProcessedPolicy(Policy):
    """Wraps a strands-robots Policy with automatic pre/post processing.

    Transparently applies the ProcessorBridge to observations before inference
    and to actions after inference. The wrapped policy works identically to
    the original but with normalized/processed data.

    Inherits from Policy so ``isinstance(wrapped, Policy)`` returns True.
    """

    def __init__(self, policy, bridge: ProcessorBridge):
        self._policy = policy
        self._bridge = bridge

    @property
    def provider_name(self) -> str:
        return f"processed:{self._policy.provider_name}"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._policy.set_robot_state_keys(robot_state_keys)

    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get actions with automatic pre/post processing."""
        # Preprocess observation
        processed_obs = self._bridge.preprocess(observation_dict)

        # Run policy inference
        actions = await self._policy.get_actions(processed_obs, instruction, **kwargs)

        # Postprocess each action
        if self._bridge.has_postprocessor:
            processed_actions = []
            for action in actions:
                processed = self._bridge.postprocess(action)
                if isinstance(processed, dict):
                    processed_actions.append(processed)
                else:
                    # If postprocessor returns a tensor, convert back to dict
                    processed_actions.append(action)
            return processed_actions

        return actions

    def select_action_sync(self, observation_dict: Dict[str, Any]) -> np.ndarray:
        """Synchronous inference with pre/post processing."""
        processed_obs = self._bridge.preprocess(observation_dict)

        result = self._policy.select_action_sync(processed_obs)

        if self._bridge.has_postprocessor:
            import torch

            if isinstance(result, np.ndarray):
                tensor = torch.from_numpy(result)
            else:
                tensor = result
            processed = self._bridge.postprocess(tensor)
            if isinstance(processed, dict):
                return np.array(list(processed.values()))
            elif hasattr(processed, "numpy"):
                return processed.numpy()
            return processed

        return result

    def reset(self):
        """Reset both policy and processor state."""
        self._bridge.reset()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped policy."""
        return getattr(self._policy, name)


def create_processor_bridge(
    pretrained_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ProcessorBridge:
    """Factory function to create a ProcessorBridge.

    Convenience wrapper that handles common cases:
    1. From pretrained model (loads saved pipeline configs)
    2. Empty bridge (passthrough, no processing)
    3. With custom stats override

    Args:
        pretrained_name_or_path: HF model ID or local path (None for passthrough)
        device: Target device
        stats: Custom normalization stats to override model's saved stats
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        ProcessorBridge instance
    """
    if pretrained_name_or_path is None:
        return ProcessorBridge(device=device)

    overrides = kwargs.pop("overrides", {})

    # If custom stats provided, inject into normalizer overrides
    if stats:
        overrides.setdefault("normalizer_processor", {})["stats"] = stats
        overrides.setdefault("unnormalizer_processor", {})["stats"] = stats

    return ProcessorBridge.from_pretrained(
        pretrained_name_or_path,
        device=device,
        overrides=overrides,
        **kwargs,
    )


__all__ = [
    "ProcessorBridge",
    "ProcessedPolicy",
    "create_processor_bridge",
    "PREPROCESSOR_CONFIG",
    "POSTPROCESSOR_CONFIG",
]
