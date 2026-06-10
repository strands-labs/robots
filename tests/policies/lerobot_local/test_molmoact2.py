"""Tests for ``strands_robots.policies.lerobot_local.molmoact2`` -- the
transformers-native MolmoAct2 load-path helpers used by ``LerobotLocalPolicy``
to support ``allenai/MolmoAct2-*`` checkpoints that have no lerobot draccus
``type`` key.

These tests are dependency-light: they exercise detection, norm-tag discovery,
and image-key derivation by stubbing ``config.json`` / ``norm_stats.json``
reads. They do NOT download the 21GB checkpoint or import lerobot's heavy
modeling code (build_policy is covered by the hardware/e2e validation).
"""

from __future__ import annotations

import json

import pytest

from strands_robots.policies.lerobot_local import molmoact2


def test_is_molmoact2_explicit_type():
    """Explicit policy_type='molmoact2' short-circuits to True without any I/O."""
    assert molmoact2.is_molmoact2("anything/at-all", "molmoact2") is True
    assert molmoact2.is_molmoact2("anything/at-all", "MolmoAct2") is True  # case-insensitive


def test_is_molmoact2_empty_path_no_type():
    """No path and no type → not molmoact2 (avoids spurious hub calls)."""
    assert molmoact2.is_molmoact2("", None) is False


def test_is_molmoact2_from_config_transformers_native(monkeypatch):
    """A transformers-native ckpt (model_type=molmoact2, no lerobot type) → True."""
    monkeypatch.setattr(
        molmoact2,
        "_read_config_json",
        lambda _p: {"model_type": "molmoact2", "hidden_size": 4096},
    )
    assert molmoact2.is_molmoact2("allenai/MolmoAct2-SO100_101", None) is True


def test_is_molmoact2_lerobot_native_is_false(monkeypatch):
    """A lerobot-native molmoact2 (has draccus 'type') goes through the normal
    resolution path, NOT this wrapper → False."""
    monkeypatch.setattr(
        molmoact2,
        "_read_config_json",
        lambda _p: {"model_type": "molmoact2", "type": "molmoact2"},
    )
    assert molmoact2.is_molmoact2("some/lerobot-native-molmoact2", None) is False


def test_is_molmoact2_other_model_is_false(monkeypatch):
    """An ACT/Pi0/etc. checkpoint is not molmoact2."""
    monkeypatch.setattr(molmoact2, "_read_config_json", lambda _p: {"type": "act"})
    assert molmoact2.is_molmoact2("lerobot/act_aloha", None) is False


def test_auto_norm_tag_explicit_wins():
    """An explicitly requested norm_tag is returned verbatim (no I/O)."""
    assert molmoact2.auto_norm_tag("any/repo", "my_custom_tag") == "my_custom_tag"


def test_auto_norm_tag_single_tag(tmp_path):
    """A norm_stats.json with exactly one tag → that tag is auto-selected."""
    norm = {"metadata_by_tag": {"so100_so101_molmoact2": {"action_horizon": 30}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(norm))
    assert molmoact2.auto_norm_tag(str(tmp_path), None) == "so100_so101_molmoact2"


def test_auto_norm_tag_multiple_tags_returns_none(tmp_path):
    """Multiple tags → None (refuse to guess; caller must pass norm_tag=)."""
    norm = {"metadata_by_tag": {"tag_a": {}, "tag_b": {}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(norm))
    assert molmoact2.auto_norm_tag(str(tmp_path), None) is None


def test_auto_norm_tag_missing_file_returns_none(tmp_path):
    """No norm_stats.json locally and offline → None, not a crash."""
    assert molmoact2.auto_norm_tag(str(tmp_path), None) is None


def test_derive_image_keys_explicit_wins():
    """Explicit image_keys are returned unchanged."""
    keys = ["observation.images.top", "observation.images.side"]
    assert molmoact2.derive_image_keys(keys, "so_real") == keys


def test_derive_image_keys_default_when_none():
    """No keys and no embodiment → the documented default image keys."""
    assert molmoact2.derive_image_keys(None, None) == molmoact2.DEFAULT_IMAGE_KEYS


def test_derive_image_keys_from_embodiment():
    """Image rename targets are pulled from the embodiment's obs_rename."""
    pytest.importorskip("lerobot")
    # so_real renames front->observation.images.image, wrist->...wrist_image
    keys = molmoact2.derive_image_keys(None, "so_real")
    assert "observation.images.image" in keys
    assert all(k.startswith("observation.images.") for k in keys)
