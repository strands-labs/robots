# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A boolean observation value must never be read as a joint reading.

``bool`` is a subclass of ``int`` in Python, so a naive ``isinstance(v, int)``
scalar check in :meth:`LerobotLocalPolicy._collect_state_values` would silently
accept ``True``/``False`` and emit ``1.0``/``0.0`` into ``observation.state`` at
that joint's model index. That is a plausible-looking but wrong joint value: a
gripper that reports open/closed as a bool (or any stray boolean flag keyed by a
configured joint name) would then feed garbage into the policy while the run
still reported success.

The contract these tests pin: a boolean value under a configured
``robot_state_keys`` name is treated as MISSING - its slot is zero-filled IN
PLACE (keeping every following joint index-aligned) and the missing-key
degradation is surfaced (``missing_state_keys_used`` set, warn-once), exactly
like an absent key. In particular ``True`` must NOT land in the state vector as
``1.0``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(*, strict_keys=False, keys=("j0", "gripper", "j2")):
    """A minimal ACT policy with the model load stubbed out.

    Only the state-vector assembly path is exercised, so no checkpoint is
    needed - patching ``_load_model`` keeps construction offline.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act", strict_keys=strict_keys)
    policy._input_features = {"observation.images.top": _visual(), "observation.state": _state(len(keys))}
    policy._device = torch.device("cpu")
    policy.robot_state_keys = list(keys)
    return policy


def _obs(gripper_value):
    return {
        "top": np.zeros((224, 224, 3), np.uint8),
        "j0": 0.5,
        "gripper": gripper_value,
        "j2": 0.7,
    }


class TestBooleanIsNotAJointReading:
    """A bool under a keyed joint is zero-filled in place, not read as 1.0/0.0."""

    def test_true_gripper_flag_is_zero_filled_not_one(self):
        policy = _policy()
        state = [round(float(v), 3) for v in policy._collect_state_values(_obs(True), policy.robot_state_keys)]
        # gripper=True must NOT become 1.0 - it is treated as a missing reading.
        assert state == [0.5, 0.0, 0.7]

    def test_bool_value_keeps_following_joints_index_aligned(self):
        policy = _policy()
        out = policy._to_lerobot_observation(_obs(True))
        state = [round(float(v), 3) for v in out["observation.state"]]
        # j2 (index 2) stays put; the bool slot at index 1 is zeroed in place.
        assert state == [0.5, 0.0, 0.7]

    def test_bool_value_surfaces_missing_degradation(self, caplog):
        policy = _policy()
        with caplog.at_level(logging.WARNING):
            policy._collect_state_values(_obs(True), policy.robot_state_keys)
        assert policy.missing_state_keys_used is True
        warnings = [r for r in caplog.records if "not present in the observation" in r.getMessage()]
        assert len(warnings) == 1
        assert "gripper" in warnings[0].getMessage()

    def test_false_flag_also_treated_as_missing(self):
        policy = _policy()
        policy._collect_state_values(_obs(False), policy.robot_state_keys)
        # False would coincidentally read as 0.0, but it must still be flagged
        # as a non-reading so the degradation is not silently masked.
        assert policy.missing_state_keys_used is True

    def test_strict_keys_raises_on_boolean_reading(self):
        policy = _policy(strict_keys=True)
        with pytest.raises(ValueError, match="strict_keys=True"):
            policy._collect_state_values(_obs(True), policy.robot_state_keys)
