# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Partial state-key mismatch must keep present joints index-aligned, not shift them.

``_resolve_state_order`` already makes the *all*-missing case loud (generic
``joint_0..N`` vs named joints). The *partial* case - SOME configured
``robot_state_keys`` present and some absent - was still silently wrong: both
state-build paths iterated the resolved order and appended only the keys found
in the observation, so an absent key was DROPPED, shifting every following
joint value up one index before the trailing zero-pad.

The canonical trigger is a mimic/tendon gripper whose actuator name
(``left/gripper`` / ``right/gripper`` on aloha) is not among the observation's
finger-joint names while the arm joints are: the right-arm joints slid into the
gripper slots, so the model received a garbage ``observation.state`` while the
run reported success.

These tests pin the corrected behaviour: a missing key is zero-filled IN PLACE
(present dims keep their model index), the degradation is surfaced
(``strict_keys=True`` raises; ``strict_keys=False`` warns once and sets
``missing_state_keys_used``), and a fully-present key set (so101) is unchanged.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# aloha-style 14 actuator keys: 6 arm + 1 gripper actuator per side, gripper
# actuators interspersed at indices 6 and 13.
ALOHA_ACTUATOR_KEYS = [
    "l0",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "left/gripper",
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "right/gripper",
]
LEFT_ARM = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
RIGHT_ARM = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25]


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(*, strict_keys=False, state_dim=14, keys=None):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act", strict_keys=strict_keys)
    policy._input_features = {"observation.images.top": _visual(), "observation.state": _state(state_dim)}
    policy._device = torch.device("cpu")
    policy.robot_state_keys = list(ALOHA_ACTUATOR_KEYS if keys is None else keys)
    return policy


def _aloha_obs():
    """Observation with the 12 arm joints + finger joints, but NOT the two
    ``*/gripper`` actuator names the policy is keyed by."""
    obs = {"top": np.zeros((224, 224, 3), np.uint8)}
    obs.update(dict(zip(["l0", "l1", "l2", "l3", "l4", "l5"], LEFT_ARM)))
    obs.update(dict(zip(["r0", "r1", "r2", "r3", "r4", "r5"], RIGHT_ARM)))
    # Finger joints exist but under names that don't match the actuator keys.
    obs.update({"left/left_finger": 0.9, "left/right_finger": 0.9, "right/left_finger": 0.9})
    return obs


def _assert_aligned(state):
    """The 14-dim vector must be arm-in-place with gripper slots zeroed."""
    state = [round(float(v), 3) for v in state]
    assert len(state) == 14
    assert state[0:6] == LEFT_ARM  # left arm, indices 0..5
    assert state[6] == 0.0  # left gripper slot (missing) zero-filled IN PLACE
    assert state[7:13] == RIGHT_ARM  # right arm, indices 7..12 - NOT shifted into idx 6
    assert state[13] == 0.0  # right gripper slot (missing) zero-filled IN PLACE


class TestPartialMismatchAlignment:
    """A missing interspersed key zero-fills in place instead of shifting."""

    def test_vla_path_keeps_arm_joints_aligned(self):
        policy = _policy()
        out = policy._to_lerobot_observation(_aloha_obs())
        _assert_aligned(out["observation.state"])

    def test_strands_native_path_keeps_arm_joints_aligned(self):
        policy = _policy()
        batch = policy._build_batch_from_strands_format(_aloha_obs(), {})
        _assert_aligned(batch["observation.state"][0].numpy())

    def test_missing_keys_set_telemetry_and_warn_once(self, caplog):
        policy = _policy()
        import logging

        with caplog.at_level(logging.WARNING):
            policy._to_lerobot_observation(_aloha_obs())
            policy._to_lerobot_observation(_aloha_obs())  # second step must NOT re-warn
        assert policy.missing_state_keys_used is True
        warnings = [r for r in caplog.records if "not present in the observation" in r.getMessage()]
        assert len(warnings) == 1  # warn-once
        assert "left/gripper" in warnings[0].getMessage()


class TestStrictKeysRaisesOnPartialMismatch:
    """strict_keys turns a partial mismatch into a clear, actionable error."""

    def test_vla_path_raises_naming_missing_keys(self):
        policy = _policy(strict_keys=True)
        with pytest.raises(ValueError) as exc:
            policy._to_lerobot_observation(_aloha_obs())
        msg = str(exc.value)
        assert "strict_keys=True" in msg
        assert "left/gripper" in msg and "right/gripper" in msg
        assert "set_robot_state_keys" in msg

    def test_strands_native_path_raises_naming_missing_keys(self):
        policy = _policy(strict_keys=True)
        with pytest.raises(ValueError) as exc:
            policy._build_batch_from_strands_format(_aloha_obs(), {})
        assert "right/gripper" in str(exc.value)


class TestAllPresentUnaffected:
    """so101-like fully-present key set is unchanged - no warn, no telemetry flag."""

    def test_no_missing_keys_no_warning(self, caplog):
        keys = ["j0", "j1", "j2", "j3", "j4", "j5"]
        policy = _policy(state_dim=6, keys=keys)
        obs = {"top": np.zeros((224, 224, 3), np.uint8)}
        obs.update(dict(zip(keys, [0.10, 0.11, 0.12, 0.13, 0.14, 0.15])))
        import logging

        with caplog.at_level(logging.WARNING):
            out = policy._to_lerobot_observation(obs)
        state = [round(float(v), 3) for v in out["observation.state"]]
        assert state == [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
        assert policy.missing_state_keys_used is False
        assert not [r for r in caplog.records if "not present in the observation" in r.getMessage()]
