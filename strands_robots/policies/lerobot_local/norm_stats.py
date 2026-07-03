"""Norm-stats fallback for checkpoints that ship only ``norm_stats.json``.

Some LeRobot-compatible checkpoints (notably the MolmoAct2 SO-100/101 family,
``allenai/MolmoAct2-SO100_101``) do NOT ship the standard
``policy_preprocessor.json`` / ``policy_postprocessor.json`` pipeline configs.
Instead they carry a single ``norm_stats.json`` describing per-feature
quantile/min-max/mean-std statistics keyed by an embodiment *tag*.

Without those standard configs, :class:`~strands_robots.policies.lerobot_local
.processor.ProcessorBridge` would build empty pipelines and pass observations
and actions through untouched: state reaches the policy un-normalized and the
predicted actions reach the motors un-unnormalized. That silent passthrough is
the single biggest functional cause of off-policy arm motion on these
checkpoints.

This module loads ``norm_stats.json``, detects the
``molmoact2_norm_stats.v1`` schema, and builds two LeRobot ``ProcessorStep``
objects:

* :class:`NormStatsPreprocessorStep` normalizes ``observation.state``.
* :class:`NormStatsPostprocessorStep` unnormalizes ``action``.

The numeric transform is a bit-for-bit port of the reference normalizer in
``lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2._FeatureNormalizer``
(see the ``_RobotStats.normalize_state`` / ``.unnormalize_action`` norm-tag
dispatch in the same module). For ``q01_q99``::

    x_norm   = clip(2 * (x - q01) / max(q99 - q01, eps) - 1, -1, 1)
    x_unnorm = (clip(x_norm, -1, 1) + 1) * (q99 - q01) / 2 + q01

``mean_std``, ``min_max`` and ``q10_q90`` modes are also supported for
faithfulness; any other mode raises (no silent passthrough).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Schema identifier for the MolmoAct2 norm-stats file.
MOLMOACT2_NORM_STATS_FORMAT = "molmoact2_norm_stats.v1"
# Default embodiment tag for SO-100 / SO-101 checkpoints.
DEFAULT_SO_NORM_TAG = "so100_so101_molmoact2"
# Observation state key (matches LeRobot's lerobot.utils.constants).
OBS_STATE = "observation.state"

_EPS = 1e-6


def _to_array(value: Any) -> np.ndarray | None:
    """Coerce a stats value to a float32 ndarray (mirrors the reference helper).

    Tensors are detached, upcast from half/bfloat16 to float and moved to CPU.
    ``None`` passes through as ``None``.
    """
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return tensor.cpu().numpy().astype(np.float32, copy=False)
    except ImportError:
        # torch is optional; without it, fall through to the numpy coercion below.
        pass
    return np.asarray(value, dtype=np.float32)


def _to_mask(value: Any, fallback_like: np.ndarray | None) -> np.ndarray | None:
    """Coerce a mask value to a bool ndarray, broadcasting to ``fallback_like``."""
    if value is None:
        return None
    mask = np.asarray(value, dtype=np.bool_)
    if fallback_like is not None and mask.shape != fallback_like.shape:
        mask = np.broadcast_to(mask, fallback_like.shape)
    return mask


class FeatureNormalizer:
    """Per-feature normalizer ported from MolmoAct2's reference implementation.

    Supports ``none``, ``mean_std``, ``min_max``, ``q01_q99`` and ``q10_q90``
    normalization modes. ``normalize`` maps raw robot units into the model's
    normalized space; ``unnormalize`` inverts it. Both preserve the input
    container type (numpy in -> numpy out, tensor in -> tensor out on the same
    device/dtype).
    """

    def __init__(
        self,
        *,
        mode: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        min_val: np.ndarray | None = None,
        max_val: np.ndarray | None = None,
        q_low: np.ndarray | None = None,
        q_high: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        zero_mask: np.ndarray | None = None,
    ):
        self.mode = mode
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.q_low = q_low
        self.q_high = q_high
        self.mask = mask
        self.zero_mask = zero_mask

    @classmethod
    def from_stats(cls, stats: dict[str, Any] | None, mode: str) -> FeatureNormalizer | None:
        """Build a normalizer from a stats mapping for ``mode``.

        Raises:
            ValueError: If the stats required by ``mode`` are missing, or if
                ``mode`` is not a supported normalization mode.
        """
        if stats is None:
            return None
        raw_mask = stats.get("mask") if isinstance(stats, dict) else None
        if mode == "none":
            fallback = None
            for key in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask"):
                fallback = _to_array(stats.get(key))
                if fallback is not None:
                    break
            return cls(mode=mode, mask=_to_mask(raw_mask, fallback))
        if mode == "mean_std":
            mean = _to_array(stats.get("mean"))
            std = _to_array(stats.get("std"))
            if mean is None or std is None:
                raise ValueError("norm_mode='mean_std' requires mean and std stats.")
            return cls(mode=mode, mean=mean, std=std, mask=_to_mask(raw_mask, mean))
        if mode == "min_max":
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            if min_val is None or max_val is None:
                raise ValueError("norm_mode='min_max' requires min and max stats.")
            return cls(
                mode=mode,
                min_val=min_val,
                max_val=max_val,
                mask=_to_mask(raw_mask, min_val),
                zero_mask=(min_val == max_val),
            )
        if mode in {"q01_q99", "q10_q90"}:
            low_key, high_key = ("q01", "q99") if mode == "q01_q99" else ("q10", "q90")
            q_low = _to_array(stats.get(low_key))
            q_high = _to_array(stats.get(high_key))
            if q_low is None or q_high is None:
                raise ValueError(f"norm_mode={mode!r} requires {low_key} and {high_key} stats.")
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            fallback = min_val if min_val is not None else q_low
            zero_mask = None if min_val is None or max_val is None else (min_val == max_val)
            return cls(
                mode=mode,
                min_val=min_val,
                max_val=max_val,
                q_low=q_low,
                q_high=q_high,
                mask=_to_mask(raw_mask, fallback),
                zero_mask=zero_mask,
            )
        raise ValueError(f"Unsupported robot normalization mode {mode!r}.")

    def normalize(self, x: Any) -> Any:
        """Map raw values into the model's normalized space."""
        arr = _to_array(x)
        if arr is None:
            return None
        if self.mode == "none":
            normed = arr
        elif self.mode == "mean_std":
            assert self.mean is not None and self.std is not None
            normed = (arr - self.mean) / np.maximum(self.std, _EPS)
        elif self.mode == "min_max":
            assert self.min_val is not None and self.max_val is not None
            normed = 2.0 * (arr - self.min_val) / np.maximum(self.max_val - self.min_val, _EPS) - 1.0
        elif self.mode in {"q01_q99", "q10_q90"}:
            assert self.q_low is not None and self.q_high is not None
            normed = 2.0 * (arr - self.q_low) / np.maximum(self.q_high - self.q_low, _EPS) - 1.0
        else:
            normed = arr
        if self.mode in {"min_max", "q01_q99", "q10_q90"}:
            normed = np.clip(normed, -1.0, 1.0)
        if self.mask is not None:
            normed = np.where(self.mask, normed, arr)
        if self.zero_mask is not None:
            normed = np.where(self.zero_mask, 0.0, normed)
        return self._restore_like(normed, x)

    def unnormalize(self, x: Any) -> Any:
        """Invert :meth:`normalize`, mapping normalized values back to robot units."""
        arr = _to_array(x)
        if arr is None:
            return None
        if self.mode in {"min_max", "q01_q99", "q10_q90"}:
            arr = np.clip(arr, -1.0, 1.0)
        if self.mode == "none":
            out = arr
        elif self.mode == "mean_std":
            assert self.mean is not None and self.std is not None
            out = arr * self.std + self.mean
        elif self.mode == "min_max":
            assert self.min_val is not None and self.max_val is not None
            out = (arr + 1.0) * (self.max_val - self.min_val) / 2.0 + self.min_val
        elif self.mode in {"q01_q99", "q10_q90"}:
            assert self.q_low is not None and self.q_high is not None
            out = (arr + 1.0) * (self.q_high - self.q_low) / 2.0 + self.q_low
        else:
            out = arr
        if self.mask is not None:
            out = np.where(self.mask, out, arr)
        return self._restore_like(out, x)

    @staticmethod
    def _restore_like(result: np.ndarray, original: Any) -> Any:
        """Return ``result`` as a tensor matching ``original`` if it was one."""
        try:
            import torch

            if torch.is_tensor(original):
                return torch.as_tensor(result, device=original.device, dtype=original.dtype)
        except ImportError:
            # torch is optional; without it the original was already an ndarray, so
            # return the numpy result unchanged.
            pass
        return result


def load_norm_stats(
    pretrained_name_or_path: str,
    *,
    filename: str = "norm_stats.json",
) -> dict[str, Any] | None:
    """Load a ``norm_stats.json`` payload from a local dir or the HF Hub.

    Honors a ``norm_stats_filename`` override in the checkpoint's ``config.json``
    (matching the reference loader). Returns the parsed payload, or ``None`` if
    no norm-stats file can be found (network/repo errors are non-fatal here).

    Args:
        pretrained_name_or_path: HF repo id or local checkpoint directory.
        filename: Default norm-stats filename to look for.

    Returns:
        Parsed JSON payload dict, or ``None`` when unavailable.
    """
    if not pretrained_name_or_path:
        return None

    local = Path(pretrained_name_or_path)

    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with open(path) as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError) as exc:
            logger.debug("norm_stats: could not read %s: %s", path, exc)
            return None

    # Resolve a config.json override for the stats filename first.
    resolved_filename = filename
    if local.is_dir():
        cfg = _read_json(local / "config.json")
        if cfg and cfg.get("norm_stats_filename"):
            resolved_filename = str(cfg["norm_stats_filename"])
        stats_path = local / resolved_filename
        return _read_json(stats_path) if stats_path.exists() else None

    # Hub path: try the (possibly overridden) filename.
    try:
        from huggingface_hub import hf_hub_download

        try:
            cfg_path = hf_hub_download(pretrained_name_or_path, "config.json")
            cfg = _read_json(Path(cfg_path))
            if cfg and cfg.get("norm_stats_filename"):
                resolved_filename = str(cfg["norm_stats_filename"])
        except Exception as exc:  # noqa: BLE001 - config.json is optional
            logger.debug("norm_stats: no config.json on hub: %s", exc)

        downloaded = hf_hub_download(pretrained_name_or_path, resolved_filename)
        return _read_json(Path(downloaded))
    except Exception as exc:  # noqa: BLE001 - network/repo errors are non-fatal
        logger.debug("norm_stats: could not fetch %s from hub: %s", resolved_filename, exc)
        return None


def is_norm_stats_payload(payload: dict[str, Any] | None) -> bool:
    """Return True if ``payload`` is a recognized norm-stats schema.

    Currently recognizes the ``molmoact2_norm_stats.v1`` format. The check is on
    ``format`` plus a ``metadata_by_tag`` mapping so unrelated JSON files are
    rejected.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("format") != MOLMOACT2_NORM_STATS_FORMAT:
        return False
    return isinstance(payload.get("metadata_by_tag"), dict)


def select_norm_tag(
    payload: dict[str, Any],
    requested: str | None = None,
    *,
    default: str = DEFAULT_SO_NORM_TAG,
) -> str | None:
    """Choose the embodiment tag whose stats to apply.

    Priority: explicit ``requested`` > sole tag (single-embodiment checkpoint) >
    ``default`` when present. Returns ``None`` when the tag cannot be resolved
    unambiguously (multiple tags, none matching ``default``) so the caller does
    not guess wrong stats.

    Args:
        payload: Parsed norm-stats payload.
        requested: Explicit tag from the user (highest priority).
        default: Fallback tag preferred for SO-100/101 checkpoints.

    Returns:
        Resolved tag string, or ``None`` if undetermined.
    """
    tags = payload.get("metadata_by_tag")
    if not isinstance(tags, dict) or not tags:
        return None
    if requested:
        if requested in tags:
            return requested
        logger.warning(
            "norm_stats: requested norm_tag=%r not in %s; cannot apply.",
            requested,
            sorted(tags),
        )
        return None
    if len(tags) == 1:
        return next(iter(tags))
    if default in tags:
        logger.info("norm_stats: multiple tags %s; using default %r.", sorted(tags), default)
        return default
    logger.warning(
        "norm_stats: %d tags %s and no default %r; pass norm_tag= explicitly.",
        len(tags),
        sorted(tags),
        default,
    )
    return None


def _make_step_classes() -> tuple[type, type] | None:
    """Define the pre/post ProcessorStep subclasses against LeRobot's bases.

    Returns ``None`` if LeRobot's processor framework is unavailable. Built
    lazily so importing this module never hard-requires lerobot.
    """
    try:
        from lerobot.processor.pipeline import ActionProcessorStep, ObservationProcessorStep
    except ImportError:
        return None

    class NormStatsPreprocessorStep(ObservationProcessorStep):  # type: ignore[misc,valid-type]
        """Quantile/min-max/mean-std normalize ``observation.state`` in place."""

        def __init__(self, normalizer: FeatureNormalizer, state_key: str = OBS_STATE):
            self._normalizer = normalizer
            self._state_key = state_key

        def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
            if self._state_key in observation and observation[self._state_key] is not None:
                observation[self._state_key] = self._normalizer.normalize(observation[self._state_key])
            return observation

        def transform_features(self, features: Any) -> Any:
            return features

    class NormStatsPostprocessorStep(ActionProcessorStep):  # type: ignore[misc,valid-type]
        """Unnormalize the policy ``action`` back into robot units."""

        def __init__(self, normalizer: FeatureNormalizer):
            self._normalizer = normalizer

        def action(self, action: Any) -> Any:
            if action is None:
                return action
            return self._normalizer.unnormalize(action)

        def transform_features(self, features: Any) -> Any:
            return features

    return NormStatsPreprocessorStep, NormStatsPostprocessorStep


def build_norm_stats_processors(
    payload: dict[str, Any],
    norm_tag: str | None = None,
    *,
    pipeline_cls: Any = None,
) -> tuple[Any | None, Any | None]:
    """Build (preprocessor, postprocessor) pipelines from a norm-stats payload.

    Args:
        payload: Parsed, schema-validated norm-stats payload.
        norm_tag: Explicit embodiment tag (else auto-resolved via
            :func:`select_norm_tag`).
        pipeline_cls: ``DataProcessorPipeline`` class (injected for testing); the
            installed LeRobot class is imported when ``None``.

    Returns:
        ``(preprocessor, postprocessor)`` pipelines, or ``(None, None)`` when the
        tag is unresolved, the stats are unusable, or LeRobot is unavailable.
    """
    tag = select_norm_tag(payload, norm_tag)
    if tag is None:
        return None, None

    metadata = payload.get("metadata_by_tag", {}).get(tag)
    if not isinstance(metadata, dict):
        return None, None
    state_stats = metadata.get("state_stats")
    action_stats = metadata.get("action_stats")
    if not isinstance(state_stats, dict) or not isinstance(action_stats, dict):
        logger.warning("norm_stats: tag %r lacks state_stats/action_stats; skipping.", tag)
        return None, None

    mode = str(payload.get("norm_mode", "min_max"))
    state_norm = FeatureNormalizer.from_stats(state_stats, mode)
    action_norm = FeatureNormalizer.from_stats(action_stats, mode)
    if state_norm is None or action_norm is None:
        return None, None

    step_classes = _make_step_classes()
    if step_classes is None:
        return None, None
    pre_step_cls, post_step_cls = step_classes

    if pipeline_cls is None:
        try:
            from lerobot.processor.pipeline import DataProcessorPipeline as pipeline_cls  # type: ignore[no-redef]
        except ImportError:
            return None, None

    preprocessor = pipeline_cls(steps=[pre_step_cls(state_norm)])
    postprocessor = pipeline_cls(steps=[post_step_cls(action_norm)])
    logger.info("norm_stats: applied %r normalization for tag %r (state+action)", mode, tag)
    return preprocessor, postprocessor


__all__ = [
    "MOLMOACT2_NORM_STATS_FORMAT",
    "DEFAULT_SO_NORM_TAG",
    "FeatureNormalizer",
    "load_norm_stats",
    "is_norm_stats_payload",
    "select_norm_tag",
    "build_norm_stats_processors",
]
