"""LeRobot policy-type validation is discovered from lerobot's live registry.

``LerobotTrainer.validate`` guards ``extra['policy_type']`` two ways: it rejects
types that are not LeRobot-native, and it rejects ``relative_actions`` for types
whose config lacks ``use_relative_actions``. Both gates used to consult
hardcoded sets that drifted behind lerobot: the native-type set listed only a
subset of the policies lerobot actually ships (so newer types such as ``eo1`` /
``molmoact2`` / ``vla_jepa`` / ``wall_x`` were wrongly reported "not
LeRobot-native"), and the relative-action set omitted ``groot`` (which exposes
``use_relative_actions``), so a valid ``groot`` + relative-actions run was
wrongly rejected. Both are now discovered live from lerobot's
``PreTrainedConfig`` ChoiceRegistry - the same zero-maintenance discovery the
reward-model, robot, teleop, and camera surfaces already use.

These tests pin the invariant against whatever lerobot is installed (they read
its live registry rather than hardcoding type names), so they hold across
lerobot versions and fail on the pre-fix hardcoded gates whenever the registry
contains a type outside the stale sets.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.lerobot import (
    _LEROBOT_POLICY_TYPES_FALLBACK,
    _RELATIVE_ACTION_POLICY_TYPES_FALLBACK,
    LerobotTrainer,
    _lerobot_policy_types,
    _policy_registry,
    _policy_supports_relative_actions,
)


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


def _spec(dataset_root, tmp_path, **extra) -> TrainSpec:
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=10,
        extra=extra,
    )


class TestNativePolicyTypeDiscovery:
    def test_every_registry_type_is_accepted_as_native(self, dataset_root, tmp_path):
        """No policy type lerobot registers may be flagged "not LeRobot-native".

        Pre-fix the check consulted a hardcoded 10-name set; any registered type
        beyond it (eo1, molmoact2, vla_jepa, wall_x, ... - present even before
        the latest additions) was wrongly rejected. This iterates lerobot's live
        registry, so it is version-agnostic and fails on the stale set.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed; registry-driven check not applicable")
        assert reg, "lerobot registry unexpectedly empty"
        trainer = LerobotTrainer(device="cpu")
        for ptype in reg:
            problems = trainer.validate(_spec(dataset_root, tmp_path, policy_type=ptype))
            not_native = [p for p in problems if "not LeRobot-native" in p]
            assert not not_native, f"registry type {ptype!r} wrongly rejected: {not_native}"

    def test_lerobot_policy_types_matches_registry(self):
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        assert _lerobot_policy_types() == set(reg)

    def test_unknown_type_still_rejected(self, dataset_root, tmp_path):
        # A genuinely non-native name must still be caught (no over-permissiveness).
        problems = LerobotTrainer(device="cpu").validate(
            _spec(dataset_root, tmp_path, policy_type="definitely_not_a_policy")
        )
        assert any("not LeRobot-native" in p for p in problems)


class TestRelativeActionDiscovery:
    def test_gate_tracks_config_field_for_every_registry_type(self, dataset_root, tmp_path):
        """`relative_actions` is rejected iff the config class lacks the field.

        Pre-fix the gate used a hardcoded {pi0, pi05, pi0_fast} set; groot also
        exposes ``use_relative_actions`` on current lerobot, so a groot +
        relative-actions run was wrongly rejected. This derives the expectation
        from the actual dataclass field, so it fails whenever the hardcoded set
        diverges from lerobot's configs.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        trainer = LerobotTrainer(device="cpu")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "use_relative_actions" for f in dataclasses.fields(cfg_cls))
            problems = trainer.validate(_spec(dataset_root, tmp_path, policy_type=ptype, relative_actions=True))
            rejected = any("relative_actions is not supported" in p for p in problems)
            assert rejected == (not has_field), (
                f"{ptype!r}: config has use_relative_actions={has_field} but "
                f"validate {'rejected' if rejected else 'accepted'} relative_actions"
            )

    def test_helper_agrees_with_config_field(self):
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "use_relative_actions" for f in dataclasses.fields(cfg_cls))
            assert _policy_supports_relative_actions(ptype) == has_field


class TestOfflineFallback:
    """When lerobot's registry is unavailable, the static fallbacks drive the gate."""

    def test_native_types_fall_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        assert _lerobot_policy_types() == set(_LEROBOT_POLICY_TYPES_FALLBACK)

    def test_relative_actions_fall_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        for ptype in _RELATIVE_ACTION_POLICY_TYPES_FALLBACK:
            assert _policy_supports_relative_actions(ptype) is True
        assert _policy_supports_relative_actions("act") is False
        assert _policy_supports_relative_actions("definitely_not_a_policy") is False
