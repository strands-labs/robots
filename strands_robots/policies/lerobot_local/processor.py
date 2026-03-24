"""Processor Pipeline Bridge for LeRobot Local policy.

Integrates LeRobot's DataProcessorPipeline into the strands-robots policy flow.
Handles observation preprocessing and action postprocessing using the model's
own saved pipeline configs (preprocessor.json / postprocessor.json).

Architecture:
    Robot observation (dict)
        → ProcessorBridge.preprocess(obs)
            → LeRobot DataProcessorPipeline (normalize, device, batch, ...)
        → Policy.select_action(processed_obs)
        → ProcessorBridge.postprocess(action)
            → LeRobot DataProcessorPipeline (unnormalize, delta-action, ...)
        → Robot action (dict)
"""

import logging
from typing import Any

from ...utils import require_optional

logger = logging.getLogger(__name__)

# Standard pipeline config filenames used by LeRobot
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"


def _try_import_processor() -> Any | None:
    """Import LeRobot processor pipeline class.

    Uses require_optional for consistent dependency management. Returns
    the DataProcessorPipeline class directly, or None if lerobot < 0.5.

    Returns:
        DataProcessorPipeline class, or None if not available.
    """
    try:
        lerobot_pipeline = require_optional(
            "lerobot.processor.pipeline",
            pip_install="lerobot",
            extra="lerobot",
            purpose="processor pipeline support",
        )
        DataProcessorPipeline = getattr(lerobot_pipeline, "DataProcessorPipeline", None)
        if DataProcessorPipeline is None:
            logger.debug("lerobot.processor.pipeline has no DataProcessorPipeline")
            return None
        logger.debug("LeRobot DataProcessorPipeline loaded successfully")
        return DataProcessorPipeline
    except ImportError:
        logger.debug(
            "LeRobot processor module not available. "
            "ProcessorBridge will pass data through unchanged. "
            "Install lerobot >= 0.5.0 for full processor support."
        )
        return None


class ProcessorBridge:
    """Bridge between strands-robots observation/action format and LeRobot's processor pipeline.

    Handles:
    - Loading preprocessor + postprocessor from pretrained model dirs / HF Hub
    - Running the pipeline steps (normalize, device transfer, observation processing, etc.)
    - Converting processed output back to strands-robots format

    Thread-safe: each bridge instance holds its own pipeline state.
    """

    def __init__(
        self,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        device: str | None = None,
    ):
        """Initialize with optional pre/post processor pipelines.

        Args:
            preprocessor: LeRobot DataProcessorPipeline for observation preprocessing.
            postprocessor: LeRobot DataProcessorPipeline for action postprocessing.
            device: Target device for tensor operations (auto-detected if None).
        """
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        device: str | None = None,
        preprocessor_config: str = PREPROCESSOR_CONFIG,
        postprocessor_config: str = POSTPROCESSOR_CONFIG,
        overrides: dict[str, Any] | None = None,
    ) -> "ProcessorBridge":
        """Load processor pipelines from a pretrained model.

        Tries to load preprocessor.json and postprocessor.json from the model
        directory or HuggingFace Hub. If either doesn't exist, that pipeline
        is skipped (passthrough).

        Args:
            pretrained_name_or_path: HF model ID or local path.
            device: Target device (auto-detected if None).
            preprocessor_config: Filename for preprocessor config.
            postprocessor_config: Filename for postprocessor config.
            overrides: Dict of step overrides (passed to both pipelines).

        Returns:
            ProcessorBridge instance with loaded pipelines.
        """
        DataProcessorPipeline = _try_import_processor()
        if DataProcessorPipeline is None:
            logger.info("LeRobot processor not available, creating passthrough bridge")
            return cls(device=device)

        preprocessor = None
        postprocessor = None

        # Load preprocessor
        try:
            preprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=preprocessor_config,
                overrides=overrides or {},
            )
            logger.info("Loaded preprocessor from %s: %d steps", pretrained_name_or_path, len(preprocessor))
        except (FileNotFoundError, ValueError) as exc:
            # No config file found — model doesn't ship a preprocessor. This is normal.
            logger.debug("No preprocessor found: %s", exc)

        # Load postprocessor
        try:
            postprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=postprocessor_config,
                overrides=overrides or {},
            )
            logger.info("Loaded postprocessor from %s: %d steps", pretrained_name_or_path, len(postprocessor))
        except (FileNotFoundError, ValueError) as exc:
            # No config file found — model doesn't ship a postprocessor. This is normal.
            logger.debug("No postprocessor found: %s", exc)

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

    def preprocess(self, observation: dict[str, Any], instruction: str | None = None) -> dict[str, Any]:
        """Preprocess a raw observation dict through the pipeline.

        If no preprocessor is loaded, returns observation unchanged.

        For VLA models, the instruction is passed as complementary data so that
        LeRobot's TokenizerProcessorStep can access it via the ``task`` key.
        Using ``process_observation()`` alone would create a transition without
        complementary data, causing a ``KeyError: 'task'``.

        Args:
            observation: Raw observation dict from robot/sim.
            instruction: Natural language task instruction for VLA models.

        Returns:
            Processed observation dict (tensors on target device, normalized, etc.).

        Raises:
            RuntimeError: If the preprocessor pipeline fails.
        """
        if self._preprocessor is None:
            return observation

        try:
            # Build a full transition so complementary_data (containing the
            # task instruction) is available to all pipeline steps.
            from lerobot.processor.converters import create_transition
            from lerobot.processor.core import TransitionKey

            complementary: dict[str, Any] = {}
            if instruction:
                complementary["task"] = instruction

            transition = create_transition(
                observation=observation,
                complementary_data=complementary if complementary else None,
            )
            processed = self._preprocessor._forward(transition)
            return processed[TransitionKey.OBSERVATION]
        except Exception as exc:
            raise RuntimeError(f"Preprocessor pipeline failed: {exc}") from exc

    def postprocess(self, action: Any) -> Any:
        """Postprocess a policy action through the pipeline.

        If no postprocessor is loaded, returns action unchanged.

        Args:
            action: Raw action from policy (tensor or dict).

        Returns:
            Processed action (unnormalized, converted to robot format, etc.).

        Raises:
            RuntimeError: If the postprocessor pipeline fails.
        """
        if self._postprocessor is None:
            return action

        try:
            return self._postprocessor.process_action(action)
        except Exception as exc:
            raise RuntimeError(f"Postprocessor pipeline failed: {exc}") from exc

    def reset(self) -> None:
        """Reset pipeline state (e.g., clear running stats in stateful steps)."""
        if self._preprocessor is not None:
            self._preprocessor.reset()
        if self._postprocessor is not None:
            self._postprocessor.reset()

    def __repr__(self) -> str:
        pre = f"pre={len(self._preprocessor)}steps" if self._preprocessor else "pre=None"
        post = f"post={len(self._postprocessor)}steps" if self._postprocessor else "post=None"
        return f"ProcessorBridge({pre}, {post})"

    def get_info(self) -> dict[str, Any]:
        """Return a summary dict describing the processor bridge state.

        Useful for diagnostics and integration tests.
        """
        return {
            "has_preprocessor": self.has_preprocessor,
            "has_postprocessor": self.has_postprocessor,
            "is_active": self.is_active,
            "repr": repr(self),
        }


__all__ = [
    "ProcessorBridge",
    "PREPROCESSOR_CONFIG",
    "POSTPROCESSOR_CONFIG",
]
