"""A real, tiny, randomly initialised LeRobot ACT checkpoint for an N-joint robot: no download, CPU, ~1 MB."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_tiny_act(
    path: Path,
    n: int,
    *,
    camera: str | None = "front",
    size: tuple[int, int] = (64, 48),
    span: tuple[list[float], list[float]] | None = None,
) -> Path:
    """``span`` = the per-joint (min, max) its "dataset" saved, in its recorded units (None: mean/std only)."""
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    w, h = size
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n,))}
    if camera:
        inputs[f"observation.images.{camera}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
    cfg = ACTConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(n,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        chunk_size=60,  # a 2 s horizon at 30 Hz: a slow first turnaround (~1 s) stays inside it
        n_action_steps=10,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    stats = {
        "observation.state": {"mean": np.zeros(n, np.float32), "std": np.full(n, 10.0, np.float32)},
        "action": {"mean": np.zeros(n, np.float32), "std": np.full(n, 5.0, np.float32)},
    }
    if span is not None:
        for feat in stats.values():
            feat["min"], feat["max"] = np.array(span[0], np.float32), np.array(span[1], np.float32)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return path
