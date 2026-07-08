"""Regression tests for the OLD-FORMAT in-model normalization fallback.

Covers :meth:`ProcessorBridge._load_in_model_normalization_fallback` and its
wiring into :meth:`ProcessorBridge.from_pretrained` via ``policy_config``.

The bug these guard against: pre-processor-era lerobot checkpoints (the
canonical zoo -- ``act_aloha_*``, ``diffusion_pusht``, tdmpc/vqbet entries)
store their Normalize modules *inside* the policy, so ``model.safetensors``
carries ``normalize_inputs.*`` / ``unnormalize_outputs.*`` buffers and
``config.json`` carries ``normalization_mapping``, with no processor JSON.
Current lerobot no longer registers those modules, so
``PreTrainedPolicy.from_pretrained`` drops the buffers as "unexpected keys"
(only a ``WARNING:root`` is logged) and the policy runs with normalization
dropped. For a MEAN_STD checkpoint this makes the arm FLAIL (raw z-scored
actions applied as robot units), not merely under-move.

The fix reconstructs the pre/post pipelines from those same buffers using
lerobot's own ``extract_normalization_stats`` + ``make_pre_post_processors``,
so the checkpoint runs normalized with zero user action. These tests build a
synthetic OLD-FORMAT checkpoint on disk (real safetensors buffers + a real
``ACTConfig``) and drive the real lerobot machinery -- no mocks.
"""

from __future__ import annotations

import builtins
import importlib

import pytest
import torch
from safetensors.torch import save_file

from strands_robots.policies.lerobot_local.processor import ProcessorBridge

# The reconstruction rides lerobot's migration helper + factory; skip cleanly on
# an install where they cannot be imported. ``pytest.importorskip`` only catches
# ImportError, but these modules can also fail at definition time -- e.g. a
# broken dataclass in an unrelated sibling policy module (lerobot.policies.groot)
# raises TypeError while importing the policies package. Skip on any import-time
# failure so an unrelated lerobot defect does not fail collection here (the
# production fallback degrades to passthrough for the same failure modes).
try:
    importlib.import_module("lerobot.policies")
    importlib.import_module("lerobot.processor.migrate_policy_normalization")
except Exception as exc:  # noqa: BLE001 - mirror the production best-effort guard
    pytest.skip(
        f"lerobot migration helpers unimportable: {exc}",
        allow_module_level=True,
    )


def _act_config():
    """Minimal real ACTConfig with STATE+ACTION MEAN_STD features."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device="cpu",
    )


def _write_old_format_checkpoint(path, *, with_norm_buffers: bool = True) -> None:
    """Write a synthetic single-file checkpoint (in-model norm buffers, no JSON)."""
    state_dict = {"model.core.weight": torch.zeros(2, 2)}
    if with_norm_buffers:
        state_dict.update(
            {
                "normalize_inputs.buffer_observation_state.mean": torch.arange(4.0),
                "normalize_inputs.buffer_observation_state.std": torch.ones(4),
                "normalize_targets.buffer_action.mean": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                "normalize_targets.buffer_action.std": torch.ones(4),
                "unnormalize_outputs.buffer_action.mean": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                "unnormalize_outputs.buffer_action.std": torch.ones(4),
            }
        )
    save_file(state_dict, str(path / "model.safetensors"))


def _action_unnorm_mean(postprocessor):
    """Return the postprocessor's ``action`` unnormalization mean, or None."""
    for step in postprocessor.steps:
        stats = getattr(step, "stats", None) or getattr(step, "_stats", None)
        if stats and "action" in stats and "mean" in stats["action"]:
            return stats["action"]["mean"]
    return None


def test_reconstructs_postprocessor_from_in_model_buffers(tmp_path):
    """An OLD-FORMAT checkpoint (no processor JSON) gets its pipelines rebuilt.

    Fails before the fix: ``from_pretrained`` had no ``policy_config`` parameter,
    so this call raised ``TypeError`` and no reconstruction happened.
    """
    _write_old_format_checkpoint(tmp_path, with_norm_buffers=True)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), policy_config=_act_config(), device="cpu")

    assert bridge.has_postprocessor, "postprocessor should be reconstructed"
    assert bridge.has_preprocessor, "preprocessor should be reconstructed"
    mean = _action_unnorm_mean(bridge._postprocessor)
    assert mean is not None, "reconstructed postprocessor must carry action stats"
    # The exact in-model unnormalize_outputs.buffer_action.mean must survive.
    assert torch.allclose(mean.cpu().float(), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    # And those stats cover the action key -> not reported as inert.
    assert "action" not in bridge.inert_normalization_features()


def test_reconstruction_requires_policy_config(tmp_path):
    """Without a policy config the fallback is skipped (stays a passthrough).

    Guards backward compatibility: callers that never pass ``policy_config``
    keep the old passthrough behaviour rather than reconstructing from a config
    the bridge does not have.
    """
    _write_old_format_checkpoint(tmp_path, with_norm_buffers=True)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), policy_config=None, device="cpu")

    assert not bridge.has_postprocessor
    assert not bridge.has_preprocessor


def test_no_buffers_stays_passthrough(tmp_path):
    """A checkpoint with no in-model norm buffers is not touched by the fallback.

    Ensures the reconstruction only fires for genuine OLD-FORMAT checkpoints and
    never fabricates a pipeline for a modern checkpoint that legitimately ships
    none (which must remain a passthrough so the caller's diagnostic fires).
    """
    _write_old_format_checkpoint(tmp_path, with_norm_buffers=False)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), policy_config=_act_config(), device="cpu")

    assert not bridge.has_postprocessor
    assert not bridge.has_preprocessor


def test_reconstruction_degrades_when_lerobot_import_raises_non_import_error(monkeypatch, tmp_path):
    """A definition-time failure while importing the lerobot helpers degrades to
    passthrough instead of crashing the load.

    The reconstruction imports ``lerobot.policies`` (for ``make_pre_post_processors``)
    and ``lerobot.processor.migrate_policy_normalization``. An unrelated broken
    sibling policy module -- e.g. a dataclass with a non-default field after a
    default one in ``lerobot.policies.groot`` -- raises ``TypeError`` (not
    ``ImportError``) while importing the policies package. The fallback must
    treat that as "recovery unavailable" and return ``(None, None)`` so an
    unrelated lerobot defect cannot take down an ACT/diffusion checkpoint load.
    """
    _write_old_format_checkpoint(tmp_path, with_norm_buffers=True)
    policy_config = _act_config()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        # Match the two imports the reconstruction performs:
        # ``from lerobot.policies.factory import make_pre_post_processors`` and
        # ``from lerobot.processor.migrate_policy_normalization import ...``.
        if name.startswith("lerobot.policies.factory") or name.startswith(
            "lerobot.processor.migrate_policy_normalization"
        ):
            raise TypeError("non-default argument 'backbone_cfg' follows default argument")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    pre, post = ProcessorBridge._load_in_model_normalization_fallback(
        str(tmp_path), policy_config=policy_config, device="cpu"
    )

    assert pre is None
    assert post is None


def test_missing_single_file_weights_stays_passthrough(monkeypatch, tmp_path):
    """A checkpoint with no single-file ``model.safetensors`` degrades cleanly.

    Sharded VLA checkpoints (and any path where the single-file weights cannot
    be read) make ``_load_checkpoint_state_dict`` return ``None``. The fallback
    must then return ``(None, None)`` without attempting reconstruction, leaving
    the caller's missing-postprocessor diagnostic to fire -- it must never
    fabricate a pipeline from stats it does not have.
    """
    import strands_robots.policies.lerobot_local.processor as proc

    monkeypatch.setattr(proc, "_load_checkpoint_state_dict", lambda _path: None)

    pre, post = ProcessorBridge._load_in_model_normalization_fallback(
        str(tmp_path), policy_config=_act_config(), device="cpu"
    )

    assert pre is None
    assert post is None


def test_reconstruction_failure_warns_with_migration_command(monkeypatch, caplog, tmp_path):
    """In-model buffers present but pipeline rebuild fails -> warn + passthrough.

    When ``extract_normalization_stats`` finds buffers but
    ``make_pre_post_processors`` raises (e.g. a config the factory rejects), the
    fallback must degrade to ``(None, None)`` rather than crash the load, and it
    must warn with the exact manual ``migrate_policy_normalization`` command so
    the operator can recover -- never silently drop normalization.
    """
    import logging

    import lerobot.policies.factory as factory

    _write_old_format_checkpoint(tmp_path, with_norm_buffers=True)

    def _raise_factory(*args, **kwargs):
        raise RuntimeError("factory rejected the reconstructed config")

    monkeypatch.setattr(factory, "make_pre_post_processors", _raise_factory)

    with caplog.at_level(logging.WARNING):
        pre, post = ProcessorBridge._load_in_model_normalization_fallback(
            str(tmp_path), policy_config=_act_config(), device="cpu"
        )

    assert pre is None
    assert post is None
    assert any("migrate_policy_normalization" in record.getMessage() for record in caplog.records), (
        "must warn with the manual migration command on reconstruction failure"
    )
