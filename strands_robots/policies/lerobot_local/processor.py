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


def _register_policy_processor_steps(policy_type: str | None) -> None:
    """Import a policy's processor module so its custom pipeline steps register.

    LeRobot uses a lazy ``@ProcessorStepRegistry.register()`` pattern: a policy's
    bespoke processor steps (e.g. MolmoAct2's ``molmoact2_masked_normalizer`` /
    ``molmoact2_pack_inputs`` that tokenize images+language into ``model_inputs``)
    are ONLY added to the registry when
    ``lerobot.policies.<type>.processor_<type>`` is imported.

    ``DataProcessorPipeline.from_pretrained`` resolves each step by registry name,
    so loading a model whose ``policy_preprocessor.json`` references these steps
    fails with ``KeyError: Processor step '...' not found in registry`` unless the
    module was imported first. The pre-built ``act``/``smolvla`` etc. steps happen
    to be registered transitively, but VLA policies with custom packing steps
    (MolmoAct2) need their module imported explicitly.

    Best-effort: failures (e.g. heavy optional deps) are logged at DEBUG and the
    caller proceeds — the pipeline load will then raise a clear error itself.
    """
    if not policy_type:
        return
    import importlib

    for mod in (
        f"lerobot.policies.{policy_type}.processor_{policy_type}",
        f"lerobot.policies.{policy_type}",
    ):
        try:
            importlib.import_module(mod)
            logger.debug("Registered processor steps via %s", mod)
            return
        except Exception as exc:  # noqa: BLE001 - registration is best-effort
            logger.debug("Could not import %s for processor-step registration: %s", mod, exc)


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
        policy_type: str | None = None,
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

        # Register any policy-specific processor steps before loading the
        # pipeline. Without this, models whose preprocessor.json references
        # custom steps (e.g. MolmoAct2's pack_inputs) fail with a registry
        # KeyError and silently fall back to a no-op passthrough bridge,
        # leaving model_inputs empty at inference time. See B10.
        _register_policy_processor_steps(policy_type)

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
            # No config file found - model doesn't ship a preprocessor. This is normal.
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
            # No config file found - model doesn't ship a postprocessor. This is normal.
            logger.debug("No postprocessor found: %s", exc)

        return cls(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )

    def apply_embodiment(self, embodiment, input_features: dict | None = None) -> None:
        """Inject a declarative :class:`EmbodimentMap` into the loaded pipeline.

        This is the heart of the mapping:
        instead of remapping observations imperatively on the hot path, we
        configure LeRobot's OWN pipeline once at load time:

        1. Populate the existing ``RenameObservationsProcessorStep.rename_map``
           with ``embodiment.obs_rename`` (camera / direct key renames). If the
           pipeline has no rename step (unusual), one is inserted at the front.
        2. Insert a registered ``strands_pack_state`` step right AFTER the
           rename step to compose scalar joint keys into ``observation.state``
           in the embodiment's declared order, with an explicit dim policy.

        After this, ``preprocess(raw_obs)`` does the full transform with zero
        strands-side per-step remapping.

        Idempotent within a bridge instance: re-applying replaces the rename map
        and refreshes the single pack-state step.

        Args:
            embodiment: A resolved :class:`EmbodimentMap`.
            input_features: Model ``config.input_features`` (for state dim). When
                provided, the pack-state step's ``expected_dim`` is set from the
                model's declared ``observation.state`` shape.
        """
        if self._preprocessor is None:
            logger.debug("apply_embodiment: no preprocessor loaded, nothing to configure")
            return

        from .embodiment import register_pack_state_step

        steps = list(self._preprocessor.steps)

        # 1. Find (or create) the rename step and set its rename_map.
        rename_idx = None
        for i, step in enumerate(steps):
            if (
                type(step).__name__ == "RenameObservationsProcessorStep"
                or getattr(step, "_registry_name", None) == "rename_observations_processor"
            ):
                rename_idx = i
                break

        if rename_idx is not None:
            # Mutate the existing step's rename_map in place (preserves order).
            try:
                steps[rename_idx].rename_map = dict(embodiment.obs_rename)
            except Exception as exc:  # noqa: BLE001 - frozen/odd step, fall back to insert
                logger.debug("Could not set rename_map on existing step (%s); inserting new", exc)
                rename_idx = None

        if rename_idx is None:
            try:
                from lerobot.processor.rename_processor import RenameObservationsProcessorStep

                steps.insert(0, RenameObservationsProcessorStep(rename_map=dict(embodiment.obs_rename)))
            except ImportError:
                logger.warning("RenameObservationsProcessorStep unavailable; obs_rename not applied")

        # 2. Insert / refresh the pack-state step immediately after rename.
        PackState = register_pack_state_step()
        if PackState is not None and embodiment.state_keys:
            expected_dim = (
                embodiment.expected_state_dim(input_features) if input_features else len(embodiment.state_keys)
            )
            # Drop any prior pack-state step (idempotent re-apply).
            steps = [s for s in steps if getattr(s, "_registry_name", None) != "strands_pack_state"]
            # Recompute rename position after the filter.
            insert_at = 0
            for i, step in enumerate(steps):
                if getattr(step, "_registry_name", None) == "rename_observations_processor" or (
                    type(step).__name__ == "RenameObservationsProcessorStep"
                ):
                    insert_at = i + 1
                    break
            steps.insert(
                insert_at,
                PackState(
                    state_keys=list(embodiment.state_keys),
                    expected_dim=expected_dim,
                    dim_policy=embodiment.dim_policy,
                ),
            )

        # Pipelines are mutable dataclasses; reassign the steps sequence.
        self._preprocessor.steps = steps
        logger.info(
            "Embodiment '%s' applied: rename=%d keys, state_keys=%d, dim_policy=%s -> %d pipeline steps",
            embodiment.name,
            len(embodiment.obs_rename),
            len(embodiment.state_keys),
            embodiment.dim_policy,
            len(steps),
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
            # TransitionKey moved out of the (now-removed) lerobot.processor.core
            # submodule in LeRobot 0.5.2. It is re-exported from lerobot.processor
            # on 0.5.0/0.5.1/0.5.2 alike, so import from the package root for
            # version independence. (Canonical home is lerobot.types.)
            from lerobot.processor import TransitionKey
            from lerobot.processor.converters import create_transition

            complementary: dict[str, Any] = {}
            if instruction:
                complementary["task"] = instruction

            transition = create_transition(
                observation=observation,
                complementary_data=complementary if complementary else None,
            )
            processed = self._preprocessor._forward(transition)

            # Some VLA preprocessors (e.g. MolmoAct2's
            # ``pack_inputs`` step) emit the actual model-ready tensors
            # (``input_ids``, ``pixel_values``, ``image_grids``, ...) into the
            # transition's COMPLEMENTARY_DATA rather than OBSERVATION. The
            # policy's ``_model_inputs`` reads those keys from a FLAT batch, so
            # returning only OBSERVATION drops them and the model sees an empty
            # input dict (StopIteration on ``next(iter(model_inputs))``). Merge
            # complementary_data into the returned batch. OBSERVATION keys win
            # on conflict (they're the canonical normalized obs); complementary
            # keys (packed inputs, task, *_is_pad masks) fill in the rest.
            obs_out = processed.get(TransitionKey.OBSERVATION)
            comp_out = processed.get(TransitionKey.COMPLEMENTARY_DATA) or {}
            if isinstance(obs_out, dict):
                merged = dict(comp_out)
                merged.update(obs_out)
                return merged
            return obs_out
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
