"""MolmoAct2 load path for the LeRobot local policy.

MolmoAct2 SO-100/101 checkpoints (e.g. ``allenai/MolmoAct2-SO100_101``) are
**transformers-native** HuggingFace checkpoints, not lerobot-native. Their
``config.json`` has ``model_type: molmoact2`` and NO lerobot draccus ``type``
key, so the standard resolution path fails::

    PreTrainedConfig.from_pretrained(repo)
        -> ParsingError: Expected a dict with a 'type' key

and ``MolmoAct2Policy.from_pretrained(repo)`` cannot wrap it either (it routes
through the same ``PreTrainedConfig`` machinery).

The supported way to run these checkpoints is via lerobot's own public factory
API (``lerobot.policies.factory``):

* ``make_policy_config("molmoact2", checkpoint_path=<repo>, norm_tag=..., ...)``
  builds the ``MolmoAct2Config``.
* ``get_policy_class("molmoact2")(cfg)`` instantiates the policy. We construct it
  directly rather than via ``PreTrainedPolicy.from_pretrained`` because the
  SO-100/101 checkpoint is a *sharded* transformers-native ckpt with no
  single-file safetensors (``from_pretrained``'s ``_load_as_safetensor`` path
  fails). Direct construction is exactly what lerobot's ``make_policy()`` does in
  its from-scratch branch; ``MolmoAct2Policy.__init__`` loads the HF weights via
  ``checkpoint_path``.
* ``make_pre_post_processors(cfg)`` dispatches to
  ``make_molmoact2_pre_post_processors(cfg)`` internally and returns the
  pre/post pipelines. The repo ships no ``policy_preprocessor.json``, so the
  generic ``ProcessorBridge.from_pretrained`` would be a no-op passthrough —
  hence we build the processors through the factory instead.

We deliberately go through the factory (not the molmoact2 submodule classes
directly) so upstream lerobot changes to the config/processor signatures are
absorbed by lerobot's own dispatcher rather than breaking this wrapper.

This module encapsulates that path so ``LerobotLocalPolicy._load_model`` can
special-case ``policy_type == "molmoact2"`` cleanly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MOLMOACT2_TYPE = "molmoact2"

# Sensible default camera feature keys (match the ``so_real`` / ``so101``
# embodiments' obs_rename targets). Overridable via ``image_keys=``.
DEFAULT_IMAGE_KEYS = ["observation.images.image", "observation.images.wrist_image"]


def is_molmoact2(pretrained_name_or_path: str, policy_type: str | None) -> bool:
    """Return True if this checkpoint should use the MolmoAct2 wrapper path.

    Detection (cheap → expensive):
      1. Explicit ``policy_type == "molmoact2"``.
      2. ``config.json`` has ``model_type == "molmoact2"`` (transformers-native)
         AND no lerobot ``type`` key. Reads local file or HF Hub config.json.

    Args:
        pretrained_name_or_path: HF repo id or local dir.
        policy_type: Explicit policy type (if the caller passed one).

    Returns:
        True if the MolmoAct2 wrapper path applies.
    """
    if policy_type and policy_type.lower() == MOLMOACT2_TYPE:
        return True
    if not pretrained_name_or_path:
        return False

    config = _read_config_json(pretrained_name_or_path)
    if not config:
        return False
    # transformers-native molmoact2 checkpoint: model_type set, lerobot type absent.
    return config.get("model_type") == MOLMOACT2_TYPE and "type" not in config


def _read_config_json(pretrained_name_or_path: str) -> dict[str, Any] | None:
    """Read config.json from a local dir or the HF Hub. Returns None on failure."""
    from pathlib import Path

    local = Path(pretrained_name_or_path)
    if local.is_dir() and (local / "config.json").exists():
        try:
            with open(local / "config.json") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            logger.debug("molmoact2: could not read local config.json: %s", exc)
            return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(pretrained_name_or_path, "config.json")
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - network/repo errors are non-fatal here
        logger.debug("molmoact2: could not fetch config.json from hub: %s", exc)
        return None


def auto_norm_tag(pretrained_name_or_path: str, requested: str | None) -> str | None:
    """Resolve the normalization tag for the checkpoint.

    If ``requested`` is given, it wins. Otherwise we read ``norm_stats.json``'s
    ``metadata_by_tag`` and, if it contains exactly one tag, use it (the common
    case for single-embodiment checkpoints like ``MolmoAct2-SO100_101`` which
    ships only ``so100_so101_molmoact2``). Multiple tags → return None and let
    the caller/model decide (avoids guessing wrong stats).

    Args:
        pretrained_name_or_path: HF repo id or local dir.
        requested: Explicit norm_tag from the user (highest priority).

    Returns:
        The resolved norm tag string, or None if undetermined.
    """
    if requested:
        return requested

    try:
        from pathlib import Path

        local = Path(pretrained_name_or_path)
        if local.is_dir() and (local / "norm_stats.json").exists():
            norm_path = str(local / "norm_stats.json")
        else:
            from huggingface_hub import hf_hub_download

            norm_path = hf_hub_download(pretrained_name_or_path, "norm_stats.json")
        with open(norm_path) as fh:
            data = json.load(fh)
        tags = list((data.get("metadata_by_tag") or {}).keys())
        if len(tags) == 1:
            logger.info("molmoact2: auto-detected norm_tag=%r", tags[0])
            return tags[0]
        if len(tags) > 1:
            logger.warning(
                "molmoact2: norm_stats.json has %d tags %s; pass norm_tag= explicitly. Proceeding without one.",
                len(tags),
                tags,
            )
    except Exception as exc:  # noqa: BLE001 - best-effort tag discovery
        logger.debug("molmoact2: norm_tag auto-detect failed: %s", exc)
    return None


def derive_image_keys(image_keys: list[str] | None, embodiment_spec: Any | None) -> list[str]:
    """Pick the model VISUAL feature keys to declare on the config.

    MolmoAct2's ``validate_features`` requires at least one ``FeatureType.VISUAL``
    input feature to exist BEFORE instantiation. Priority:
      1. Explicit ``image_keys`` argument.
      2. The embodiment spec's ``obs_rename`` *targets* that look like image
         features (``observation.images.*``) — keeps config aligned with what
         the embodiment will actually feed.
      3. :data:`DEFAULT_IMAGE_KEYS`.

    Args:
        image_keys: Explicit list of model image feature keys (or None).
        embodiment_spec: Raw embodiment spec (name str / dict / EmbodimentMap).

    Returns:
        Non-empty list of ``observation.images.*`` feature keys.
    """
    if image_keys:
        return list(image_keys)

    targets = _embodiment_image_targets(embodiment_spec)
    if targets:
        return targets

    return list(DEFAULT_IMAGE_KEYS)


def _embodiment_image_targets(embodiment_spec: Any | None) -> list[str]:
    """Extract ``observation.images.*`` rename targets from an embodiment spec."""
    if embodiment_spec is None:
        return []
    try:
        from .embodiment import load_embodiment

        emb = load_embodiment(embodiment_spec)
        targets = [v for v in emb.obs_rename.values() if "image" in v]
        # Preserve declaration order, dedupe.
        seen: set[str] = set()
        ordered = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered
    except Exception as exc:  # noqa: BLE001 - spec may be a bare dict / unknown name
        logger.debug("molmoact2: could not derive image keys from embodiment: %s", exc)
        return []


def build_policy(
    pretrained_name_or_path: str,
    *,
    device: str | None,
    norm_tag: str | None,
    inference_action_mode: str,
    image_keys: list[str] | None,
    embodiment_spec: Any | None,
    state_dim: int = 6,
    action_dim: int = 6,
) -> tuple[Any, Any, Any, Any]:
    """Build a MolmoAct2 policy + its pre/post processor pipelines.

    Args:
        pretrained_name_or_path: HF repo id or local dir (the transformers ckpt).
        device: Target device string ("cuda"/"cpu"); None → cuda if available.
        norm_tag: Normalization tag; auto-detected from norm_stats.json if None.
        inference_action_mode: "continuous" or "discrete" (MolmoAct2 requires one).
        image_keys: Explicit model image feature keys (else derived).
        embodiment_spec: Embodiment spec used to derive image keys when not given.
        state_dim: observation.state dimensionality (SO arms = 6).
        action_dim: action dimensionality (SO arms = 6).

    Returns:
        Tuple ``(policy, preprocessor, postprocessor, config)``.
    """
    import torch

    try:
        from lerobot.configs import FeatureType, PolicyFeature
    except ImportError as exc:
        raise ImportError(
            "MolmoAct2 requires lerobot >= 0.5.2 (from source). The PyPI release "
            "(0.5.1) does not include MolmoAct2Policy. Install from source:\n"
            "  uv pip install 'lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git'\n"
            "On Jetson/aarch64, add --no-build-isolation if pyav fails to build."
        ) from exc

    # Use lerobot's PUBLIC factory API rather than importing the molmoact2
    # config/processor classes directly:
    #   * make_policy_config("molmoact2", ...)  -> MolmoAct2Config (blessed entry)
    #   * make_pre_post_processors(cfg)          -> dispatches to
    #       make_molmoact2_pre_post_processors(cfg, ...) internally
    # This rides lerobot's own contract so signature/step changes upstream are
    # absorbed by the factory rather than breaking this wrapper. get_policy_class
    # resolves the concrete MolmoAct2Policy. The policy is instantiated directly
    # (NOT via PreTrainedPolicy.from_pretrained): the SO-100/101 checkpoint is a
    # sharded transformers-native ckpt with no single-file safetensors, so
    # from_pretrained's _load_as_safetensor path fails -- direct construction is
    # exactly what lerobot's own make_policy() does in its from-scratch branch
    # (MolmoAct2Policy.__init__ loads the HF weights via checkpoint_path).
    try:
        from lerobot.policies.factory import (
            get_policy_class,
            make_policy_config,
            make_pre_post_processors,
        )
    except ImportError as exc:
        raise ImportError(
            "MolmoAct2 requires lerobot >= 0.5.2 (from source). The PyPI release "
            "(0.5.1) does not include MolmoAct2Policy. Install from source:\n"
            "  uv pip install 'lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git'\n"
            "On Jetson/aarch64, add --no-build-isolation if pyav fails to build."
        ) from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_tag = auto_norm_tag(pretrained_name_or_path, norm_tag)
    keys = derive_image_keys(image_keys, embodiment_spec)

    input_features: dict[str, Any] = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in keys}
    # observation.state / action are required by MolmoAct2Config.validate_features
    # (it raises without a VISUAL feature; state/action are auto-filled if absent
    # but we declare them explicitly to pin the SO-arm dims).
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    logger.info(
        "molmoact2: building config (checkpoint_path=%s, norm_tag=%r, mode=%s, image_keys=%s, "
        "state_dim=%d, action_dim=%d, device=%s)",
        pretrained_name_or_path,
        resolved_tag,
        inference_action_mode,
        keys,
        state_dim,
        action_dim,
        resolved_device,
    )
    cfg = make_policy_config(
        MOLMOACT2_TYPE,
        checkpoint_path=pretrained_name_or_path,
        norm_tag=resolved_tag,
        inference_action_mode=inference_action_mode,
        device=resolved_device,
        input_features=input_features,
        output_features=output_features,
    )

    policy_cls = get_policy_class(MOLMOACT2_TYPE)
    policy = policy_cls(cfg)
    policy.to(resolved_device)
    policy.eval()

    # Public dispatcher -> make_molmoact2_pre_post_processors(cfg) under the hood.
    preprocessor, postprocessor = make_pre_post_processors(cfg)
    logger.info("molmoact2: policy + processors ready on %s", resolved_device)

    return policy, preprocessor, postprocessor, cfg


__all__ = [
    "MOLMOACT2_TYPE",
    "is_molmoact2",
    "auto_norm_tag",
    "derive_image_keys",
    "build_policy",
]
