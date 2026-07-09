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


def _load_checkpoint_state_dict(pretrained_name_or_path: str) -> dict[str, Any] | None:
    """Load a checkpoint's single-file ``model.safetensors`` state dict.

    Resolves a local directory first, then the HF Hub (cached). Returns ``None``
    when no single-file ``model.safetensors`` is present (e.g. a sharded VLA
    checkpoint) or safetensors is unavailable -- the OLD-FORMAT checkpoints this
    supports (ACT / diffusion / tdmpc / vqbet zoo) are all single-file, so this
    stays best-effort and never raises into the load path.

    Args:
        pretrained_name_or_path: HF model ID or local checkpoint path.

    Returns:
        The loaded state dict, or ``None``.
    """
    import os

    try:
        from safetensors import SafetensorError
        from safetensors.torch import load_file
    except ImportError:
        return None

    # A truncated / corrupt model.safetensors (e.g. an interrupted Hub download)
    # raises safetensors' own SafetensorError, which is NOT an OSError/ValueError.
    # Catch it too so the best-effort loader degrades to passthrough instead of
    # crashing the ACT/diffusion load path (see the module and function docstring).
    local = os.path.join(pretrained_name_or_path, "model.safetensors")
    if os.path.isfile(local):
        try:
            return load_file(local)
        except (OSError, ValueError, SafetensorError) as exc:
            logger.debug("Could not read %s: %s", local, exc)
            return None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        path = hf_hub_download(pretrained_name_or_path, "model.safetensors")
        return load_file(path)
    except ImportError:
        return None
    except (HfHubHTTPError, OSError, ValueError, SafetensorError) as exc:
        # No single-file weights on the Hub (sharded), offline, corrupt, or unreadable.
        logger.debug("No single-file model.safetensors for %s: %s", pretrained_name_or_path, exc)
        return None


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


def _missing_config_errors() -> tuple[type[BaseException], ...]:
    """Exception types meaning a standard processor config is absent.

    LeRobot 0.5.2 raises ``ProcessorMigrationError`` (not ``FileNotFoundError``)
    when a checkpoint ships no ``policy_preprocessor.json`` /
    ``policy_postprocessor.json`` and instead carries legacy/normalization
    stats. Treating that as "no standard config" lets the bridge fall back to
    the ``norm_stats.json`` path rather than crashing.
    """
    errors: tuple[type[BaseException], ...] = (FileNotFoundError, ValueError)
    try:
        from lerobot.processor.pipeline import ProcessorMigrationError

        errors = (*errors, ProcessorMigrationError)
    except ImportError:
        # Older lerobot lacks ProcessorMigrationError; FileNotFoundError/ValueError
        # already cover the missing-config case on those versions.
        pass
    return errors


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
    caller proceeds - the pipeline load will then raise a clear error itself.
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
        # The embodiment's obs_rename map ({runtime_key: model_feature}),
        # latched by apply_embodiment. Used to enrich the 'image_keys
        # missing' preprocessor failure with the expected camera source
        # keys and what the runtime observation actually provided.
        self._obs_rename: dict[str, str] = {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        device: str | None = None,
        preprocessor_config: str = PREPROCESSOR_CONFIG,
        postprocessor_config: str = POSTPROCESSOR_CONFIG,
        overrides: dict[str, Any] | None = None,
        policy_type: str | None = None,
        norm_tag: str | None = None,
        policy_config: Any | None = None,
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
            policy_type: Policy type name, used to register policy-specific
                processor steps before loading the standard pipeline configs.
            norm_tag: Embodiment tag selecting which stats to apply from a
                ``norm_stats.json`` fallback (auto-resolved when None).
            policy_config: The loaded policy's ``PreTrainedConfig``. Enables the
                in-model-normalization fallback (see Notes) for OLD-FORMAT
                checkpoints; when None that fallback is skipped.

        Returns:
            ProcessorBridge instance with loaded pipelines.

        Notes:
            When a checkpoint ships neither ``policy_preprocessor.json`` nor
            ``policy_postprocessor.json`` but DOES ship a recognized
            ``norm_stats.json`` (e.g. the MolmoAct2 SO-100/101 family), the
            bridge falls back to building quantile/min-max/mean-std normalizers
            from those stats instead of silently passing data through
            un-normalized. See :mod:`.norm_stats`.

            When a checkpoint ships no processor configs AND no
            ``norm_stats.json`` but DOES carry OLD-FORMAT in-model normalization
            buffers in its ``model.safetensors`` (``normalize_inputs.*`` /
            ``unnormalize_outputs.*`` -- the pre-processor-era lerobot format
            still used by the canonical zoo checkpoints, e.g.
            ``lerobot/act_aloha_sim_transfer_cube_human``), those buffers are
            silently DROPPED by ``PreTrainedPolicy.from_pretrained`` under
            current lerobot (the modules moved to the processor pipeline). The
            bridge then reconstructs the pre/post pipelines from those buffers
            via lerobot's own ``extract_normalization_stats`` +
            ``make_pre_post_processors`` (requires ``policy_config``). Without
            this, a MEAN_STD checkpoint runs UN-normalized -- raw z-scored
            actions applied as robot units make the arm flail. See
            :meth:`_load_in_model_normalization_fallback`.
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

        preprocessor = cls._load_pipeline(
            DataProcessorPipeline,
            pretrained_name_or_path,
            preprocessor_config,
            overrides or {},
            device,
            kind="preprocessor",
        )
        postprocessor = cls._load_pipeline(
            DataProcessorPipeline,
            pretrained_name_or_path,
            postprocessor_config,
            overrides or {},
            device,
            kind="postprocessor",
        )

        # Fallback: a checkpoint may ship NEITHER standard pipeline config but a
        # recognized norm_stats.json (e.g. MolmoAct2 SO-100/101). Without this,
        # both pipelines are None and the bridge silently passes data through
        # un-normalized -- the single biggest cause of off-policy arm motion on
        # such checkpoints. Build quantile/min-max/mean-std normalizers instead.
        if preprocessor is None and postprocessor is None:
            preprocessor, postprocessor = cls._load_norm_stats_fallback(pretrained_name_or_path, norm_tag=norm_tag)

        # Third fallback: an OLD-FORMAT checkpoint ships no processor configs and
        # no norm_stats.json, but carries in-model normalization buffers that
        # current lerobot drops on load (see the class-level Notes). Reconstruct
        # the pre/post pipelines from those buffers so the policy runs normalized
        # instead of flailing on raw MEAN_STD actions. Needs the policy config.
        if preprocessor is None and postprocessor is None and policy_config is not None:
            preprocessor, postprocessor = cls._load_in_model_normalization_fallback(
                pretrained_name_or_path, policy_config, device
            )

        return cls(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )

    @staticmethod
    def _load_pipeline(
        pipeline_cls: Any,
        pretrained_name_or_path: str,
        config_filename: str,
        overrides: dict[str, Any],
        device: str | None,
        kind: str,
    ) -> Any | None:
        """Load one processor pipeline, reconciling a device-pinned step.

        A checkpoint trained on GPU bakes ``device_processor.device = "cuda"``
        into its ``policy_preprocessor.json`` / ``policy_postprocessor.json``.
        On a host without that device, LeRobot's ``get_safe_torch_device``
        asserts the device is available and the ``device_processor`` step fails
        to instantiate, which ``DataProcessorPipeline.from_pretrained`` surfaces
        as a ``ValueError`` indistinguishable from "no config file present".
        Swallowing it drops normalization silently: observations reach the model
        un-normalized and predicted actions reach the motors un-unnormalized
        (the arm barely moves). Since the bridge already knows its resolved
        target ``device`` and moves every tensor there itself, reconcile the
        pinned step onto that device and retry once before giving up.

        Args:
            pipeline_cls: LeRobot ``DataProcessorPipeline`` class.
            pretrained_name_or_path: HF model ID or local checkpoint path.
            config_filename: Pipeline config filename to load.
            overrides: User-provided step overrides (never mutated here).
            device: Resolved target device, or None when unknown.
            kind: ``"preprocessor"`` or ``"postprocessor"`` (for log messages).

        Returns:
            The loaded pipeline, or ``None`` when the checkpoint genuinely ships
            no such config.
        """
        try:
            pipeline = pipeline_cls.from_pretrained(
                pretrained_name_or_path,
                config_filename=config_filename,
                overrides=overrides,
            )
            logger.info("Loaded %s from %s: %d steps", kind, pretrained_name_or_path, len(pipeline))
            return pipeline
        except _missing_config_errors() as exc:
            # Distinguish a device-pinned step that failed to instantiate from a
            # genuinely absent config. The former is a ValueError whose message
            # names the device_processor step (LeRobot wraps the underlying
            # device assertion via _instantiate_step); the latter is a
            # FileNotFoundError / ProcessorMigrationError. Only retry the former,
            # and only when we know our device and the user has not already
            # pinned device_processor themselves.
            if (
                device
                and "device_processor" not in overrides
                and isinstance(exc, ValueError)
                and "device_processor" in str(exc)
            ):
                retry_overrides = {**overrides, "device_processor": {"device": device}}
                try:
                    pipeline = pipeline_cls.from_pretrained(
                        pretrained_name_or_path,
                        config_filename=config_filename,
                        overrides=retry_overrides,
                    )
                    logger.warning(
                        "%s ships a device-pinned 'device_processor' step that is "
                        "unavailable on this host; reconciled it onto '%s' so "
                        "normalization is applied. Original error: %s",
                        kind,
                        device,
                        exc,
                    )
                    return pipeline
                except _missing_config_errors() as retry_exc:
                    logger.debug("%s device_processor retry on '%s' failed: %s", kind, device, retry_exc)
            # No config file found - model doesn't ship this pipeline. Normal.
            logger.debug("No %s found: %s", kind, exc)
            return None

    @staticmethod
    def _load_norm_stats_fallback(
        pretrained_name_or_path: str,
        norm_tag: str | None = None,
    ) -> tuple[Any | None, Any | None]:
        """Build pre/post pipelines from a ``norm_stats.json`` when present.

        Returns ``(None, None)`` if no recognized norm-stats file is found, so
        the bridge stays a passthrough only when there is genuinely nothing to
        apply.

        Args:
            pretrained_name_or_path: HF model ID or local checkpoint path.
            norm_tag: Explicit embodiment tag (auto-resolved when None).

        Returns:
            ``(preprocessor, postprocessor)`` pipelines or ``(None, None)``.
        """
        from . import norm_stats as _norm_stats

        payload = _norm_stats.load_norm_stats(pretrained_name_or_path)
        if not _norm_stats.is_norm_stats_payload(payload):
            return None, None
        assert payload is not None  # narrowed by is_norm_stats_payload
        logger.info(
            "No standard processor configs for %s; falling back to norm_stats.json",
            pretrained_name_or_path,
        )
        return _norm_stats.build_norm_stats_processors(payload, norm_tag=norm_tag)

    @staticmethod
    def _load_in_model_normalization_fallback(
        pretrained_name_or_path: str,
        policy_config: Any,
        device: str | None = None,
    ) -> tuple[Any | None, Any | None]:
        """Rebuild pre/post pipelines from OLD-FORMAT in-model normalization buffers.

        Pre-processor-era lerobot checkpoints (the canonical zoo: ``act_aloha_*``,
        ``diffusion_pusht``, tdmpc/vqbet entries) stored their Normalize modules
        *inside* the policy, so ``model.safetensors`` carries
        ``normalize_inputs.*`` / ``normalize_targets.*`` / ``unnormalize_outputs.*``
        buffers and ``config.json`` carries ``normalization_mapping``. Current
        lerobot's policy classes no longer register those modules, so
        ``_load_as_safetensor(strict=False)`` drops the buffers as "unexpected
        keys" (only a ``WARNING:root`` is logged) and no processor JSON exists to
        replace them -- the policy then runs with normalization dropped. For a
        MEAN_STD checkpoint that means raw z-scored actions applied as robot
        units, i.e. the arm FLAILS.

        This reconstructs the pipelines from those same buffers using lerobot's
        own recovery machinery -- ``extract_normalization_stats`` (the exact
        scanner ``migrate_policy_normalization`` uses) + the public
        ``make_pre_post_processors`` factory -- so the checkpoint runs normalized
        with zero user action, equivalent to running lerobot's migration script.

        Best-effort: returns ``(None, None)`` (leaving the bridge a passthrough,
        so the caller's missing-postprocessor diagnostic still fires) when the
        recovery API is unavailable, no single-file ``model.safetensors`` is
        present, or it carries no in-model normalization buffers. When buffers
        ARE present but reconstruction fails, warns with the exact manual
        migration command rather than failing silently.

        Args:
            pretrained_name_or_path: HF model ID or local checkpoint path.
            policy_config: The loaded policy's ``PreTrainedConfig`` (supplies
                ``input_features`` / ``output_features`` / ``normalization_mapping``
                to the factory).
            device: Resolved target device (unused directly -- the built pipeline
                inherits ``policy_config.device`` and the bridge moves tensors).

        Returns:
            ``(preprocessor, postprocessor)`` reconstructed pipelines, or
            ``(None, None)``.
        """
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.processor.migrate_policy_normalization import (
                extract_normalization_stats,
            )
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort
            # The helpers may be genuinely absent (older lerobot -> ImportError)
            # or unimportable because an unrelated sibling policy module fails at
            # definition time (e.g. a broken dataclass in lerobot.policies.groot
            # raises TypeError while importing the policies package). Either way
            # reconstruction is impossible, so degrade to passthrough instead of
            # letting an unrelated lerobot defect crash ACT/diffusion loads.
            logger.debug("In-model normalization recovery unavailable: %s", exc)
            return None, None

        state_dict = _load_checkpoint_state_dict(pretrained_name_or_path)
        if not state_dict:
            return None, None
        stats = extract_normalization_stats(state_dict)
        if not stats:
            # No in-model normalization buffers -> genuinely no normalization to
            # recover; let the passthrough diagnostic handle it.
            return None, None

        try:
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats)
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort
            logger.warning(
                "lerobot_local: %s ships OLD-FORMAT in-model normalization (%d buffers "
                "in model.safetensors: %s) that current lerobot drops on load, and "
                "automatic reconstruction failed (%s). Migrate it once with: "
                "python -m lerobot.processor.migrate_policy_normalization "
                "--pretrained-path %s --output-dir <dir>  then load <dir>. Until then "
                "the policy runs UN-normalized (raw MEAN_STD actions applied as robot "
                "units make the arm flail).",
                pretrained_name_or_path,
                len(stats),
                sorted(stats),
                exc,
                pretrained_name_or_path,
            )
            return None, None

        logger.info(
            "lerobot_local: %s ships no processor configs but carries OLD-FORMAT "
            "in-model normalization; reconstructed pre/post pipelines from its "
            "%d in-model buffer group(s) %s (equivalent to lerobot's "
            "migrate_policy_normalization). The policy runs normalized.",
            pretrained_name_or_path,
            len(stats),
            sorted(stats),
        )
        return preprocessor, postprocessor

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
                    state_units=embodiment.state_units,
                    gripper_index=embodiment.gripper_index,
                    gripper_joint_range=list(embodiment.gripper_joint_range),
                    joint_mids=list(embodiment.joint_mids),
                ),
            )

        # Pipelines are mutable dataclasses; reassign the steps sequence.
        self._preprocessor.steps = steps
        self._obs_rename = dict(embodiment.obs_rename)
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
    def preprocessor_steps(self) -> list[Any]:
        """Ordered preprocessor pipeline steps (empty when no preprocessor is loaded).

        Exposes the underlying LeRobot ``DataProcessorPipeline.steps`` so callers
        can introspect the pipeline for specific steps - e.g. an enabled
        ``RelativeActionsProcessorStep`` and the paired ``NormalizerProcessorStep``
        a Real-Time-Chunking consumer needs to re-anchor the leftover chunk prefix
        against the current robot state (LeRobot's ``reanchor_relative_rtc_prefix``).
        Returns a shallow copy so callers cannot reorder the live pipeline.
        """
        if self._preprocessor is None:
            return []
        return list(self._preprocessor.steps)

    @property
    def has_postprocessor(self) -> bool:
        """Whether a postprocessor pipeline is loaded."""
        return self._postprocessor is not None

    @property
    def is_active(self) -> bool:
        """Whether any processing pipeline is active."""
        return self.has_preprocessor or self.has_postprocessor

    def inert_normalization_features(self) -> list[str]:
        """Declared normalization features that will silently pass through.

        A LeRobot ``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep``
        declares a ``norm_map`` (e.g. ``STATE`` / ``ACTION`` -> ``MEAN_STD``) but
        applies it only when the looked-up stats key is present; otherwise it
        returns the tensor unchanged (``normalize_processor.py``:
        ``if norm_mode == IDENTITY or key not in self._tensor_stats: return
        tensor``). Pretraining *base* checkpoints - notably
        ``lerobot/smolvla_base`` - ship stats keyed by the training dataset
        (e.g. ``so100.buffer.action``) with no ``observation.state`` stats and
        no bare ``action`` key, so a present, active pipeline normalizes
        NOTHING: ``observation.state`` reaches the model raw and the predicted
        ``action`` reaches the robot without unnormalization. This is the same
        silent-passthrough hazard :mod:`.norm_stats` guards for the MolmoAct2
        ``norm_stats.json`` path, but it slips past the standard-pipeline path
        because the pipeline *is* present.

        Returns a list of ``"<key> (<type>/<mode>)"`` descriptors for every
        feature whose declared, non-IDENTITY normalization will be skipped.
        Empty when every declared normalization is backed by matching stats -
        the normal case for a fine-tuned checkpoint whose stats use the
        canonical ``action`` / ``observation.state`` keys.
        """
        try:
            from lerobot.configs.types import FeatureType, NormalizationMode
            from lerobot.utils.constants import ACTION
        except ImportError:
            return []

        inert: list[str] = []
        for pipeline in (self._preprocessor, self._postprocessor):
            if pipeline is None:
                continue
            for step in getattr(pipeline, "steps", []):
                class_name = type(step).__name__
                if class_name not in ("NormalizerProcessorStep", "UnnormalizerProcessorStep"):
                    continue
                features = getattr(step, "features", None) or {}
                norm_map = getattr(step, "norm_map", None) or {}
                stat_keys = set((getattr(step, "stats", None) or {}).keys())
                stat_keys |= set(getattr(step, "_tensor_stats", {}).keys())
                is_unnormalizer = class_name == "UnnormalizerProcessorStep"
                for key, feature in features.items():
                    ftype = getattr(feature, "type", None)
                    if ftype is None:
                        continue
                    mode = norm_map.get(ftype)
                    if mode is None or mode == NormalizationMode.IDENTITY:
                        continue
                    # A NormalizerProcessorStep applies only observation features
                    # (it skips ACTION); an UnnormalizerProcessorStep applies only
                    # the ACTION. Mirror that so a feature the step never touches
                    # is not falsely flagged.
                    if is_unnormalizer and ftype != FeatureType.ACTION:
                        continue
                    if not is_unnormalizer and ftype == FeatureType.ACTION:
                        continue
                    lookup = ACTION if ftype == FeatureType.ACTION else key
                    if lookup not in stat_keys:
                        descriptor = f"{key} ({ftype.value}/{mode.value})"
                        if descriptor not in inert:
                            inert.append(descriptor)
        return inert

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

            # Always include the "task" key so language-conditioned VLA
            # pipelines (LeRobot's TokenizerProcessorStep) can find it. An
            # empty or None instruction is a valid EMPTY task string (""),
            # which LeRobot tokenizes without complaint. Omitting the key
            # entirely (the old `if instruction:` guard) makes the tokenizer
            # step raise a cryptic `KeyError: 'task'` -- the exact failure
            # this transition-based path exists to avoid, per this method's
            # docstring. run_policy's own default is `instruction=""`, so the
            # guard broke the documented default for every language VLA.
            # Non-language pipelines (ACT, diffusion) have no task-consuming
            # step and simply ignore the extra complementary key.
            transition = create_transition(
                observation=observation,
                complementary_data={"task": instruction or ""},
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
            raise RuntimeError(f"Preprocessor pipeline failed: {exc}{self._camera_hint(exc, observation)}") from exc

    def _camera_hint(self, exc: Exception, observation: dict[str, Any]) -> str:
        """Actionable hint appended to an "image_keys missing" preprocessor error.

        VLA preprocessors (e.g. MolmoAct2's ``pack_inputs``) raise when a
        declared image feature is absent from the (already renamed) observation.
        That happens when the runtime camera names do not match any source key
        in the embodiment's ``obs_rename``, so the rename step never produces
        the model's ``observation.images.*`` features. The raw error names only
        the model feature keys, not the camera names to fix, so we append the
        expected source camera key(s) and what the observation actually
        provided. Returns an empty string for unrelated failures.

        Args:
            exc: The exception raised inside the pipeline.
            observation: The raw observation fed to the pipeline.

        Returns:
            A "\nHint: ..." suffix, or "" when the failure is not an
            image-keys mismatch or no obs_rename is known.
        """
        if "image_keys missing" not in str(exc) or not self._obs_rename:
            return ""
        expected = sorted(src for src, dst in self._obs_rename.items() if "image" in dst)
        if not expected:
            return ""
        got = sorted(observation) if isinstance(observation, dict) else []
        return (
            f"\nHint: this usually means the runtime camera names do not match the "
            f"embodiment's obs_rename. Expected camera source key(s): {expected}. "
            f"Observation provided: {got}. Rename your cameras to the expected key(s), "
            f"or pass policy_config={{'obs_rename_override': {{'<your_camera>': "
            f"'observation.images.<feature>'}}}}."
        )

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
