# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Declarative-embodiment state build must zero-fill a missing state_key IN PLACE.

This is the embodiment-path sibling of the generic-path fix pinned in
``test_partial_state_key_alignment.py``. ``PackStateProcessorStep.observation()``
(the step a declarative embodiment installs on the preprocessor) previously
appended a value only for state_keys *found* in the observation and skipped the
absent ones, so a missing key DROPPED its slot and shifted every following joint
up one index before the trailing pad -- the model received a garbage
``observation.state`` while the run reported success.

The canonical trigger is the aloha bimanual embodiment: its 14 actuators follow
the gym-aloha / LeRobot ACT convention ``[6 arm + 1 gripper] x 2`` with the
gripper ACTUATORS ``left/gripper`` / ``right/gripper`` at indices 6 and 13, but
the sim observation exposes the finger JOINTS (``left/left_finger`` ...), never a
``*/gripper`` value. With the old skip logic the right-arm joints slid into the
left gripper slot; and because the shipped config declared the 16 finger-JOINT
names (not the 14 actuators), a canonical 14-D ACT crashed outright
(``observation.state dim 16 > model expected 14``).

These tests pin the corrected behaviour: a missing state_key is zero-filled IN
PLACE (present joints keep their model index), the degradation is warned once,
a fully-present key set is unchanged, an all-missing set is left untouched for a
clearer downstream error, and the aloha embodiment declares the 14 actuator keys.
"""

import numpy as np
import pytest

pytest.importorskip("lerobot")

import strands_robots.policies.lerobot_local.embodiment as E

# The aloha bimanual actuator convention: 6 arm + 1 gripper actuator per side,
# gripper actuators interspersed at indices 6 and 13.
ALOHA_14 = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/gripper",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/gripper",
]
LEFT_ARM = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
RIGHT_ARM = [108.0, 109.0, 110.0, 111.0, 112.0, 113.0]
ARM_KEYS_L = ALOHA_14[0:6]
ARM_KEYS_R = ALOHA_14[7:13]


def _sim_obs_no_gripper() -> dict[str, float]:
    """A sim observation exposing the arm joints + finger joints but NOT the
    gripper actuator keys ``left/gripper`` / ``right/gripper`` (mirrors what the
    real MuJoCo aloha's ``get_observation`` returns)."""
    obs: dict[str, float] = {}
    for k, v in zip(ARM_KEYS_L, LEFT_ARM):
        obs[k] = v
    for k, v in zip(ARM_KEYS_R, RIGHT_ARM):
        obs[k] = v
    # finger joints present but NOT declared as state_keys (so they are ignored):
    obs["left/left_finger"] = 0.02
    obs["left/right_finger"] = 0.02
    obs["right/left_finger"] = 0.02
    obs["right/right_finger"] = 0.02
    return obs


def _step(state_keys, expected_dim, **kw):
    Step = E.register_pack_state_step()
    assert Step is not None, "lerobot processor framework unavailable"
    return Step(state_keys=list(state_keys), expected_dim=expected_dim, dim_policy=kw.pop("dim_policy", "pad"), **kw)


class TestPackStateZeroFillInPlace:
    def test_missing_gripper_keys_zero_filled_in_place(self):
        """Arm joints keep their canonical model index; the two absent gripper
        actuator slots read 0.0. FAILS pre-fix: the right arm slid into the
        left-gripper slot and both zeros landed at the tail."""
        E._WARNED_MISSING_STATE_KEYS.clear()
        out = _step(ALOHA_14, 14).observation(_sim_obs_no_gripper())
        state = out["observation.state"].numpy()
        assert len(state) == 14
        np.testing.assert_allclose(state[0:6], LEFT_ARM, atol=1e-5)
        assert state[6] == 0.0  # left/gripper slot held in place
        np.testing.assert_allclose(state[7:13], RIGHT_ARM, atol=1e-5)  # <- shifted pre-fix
        assert state[13] == 0.0  # right/gripper slot held in place

    def test_missing_keys_warn_once(self, caplog):
        """The degradation is surfaced once (naming the absent keys), then
        deduplicated across the hot control loop."""
        E._WARNED_MISSING_STATE_KEYS.clear()
        step = _step(ALOHA_14, 14)
        import logging

        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.embodiment"):
            step.observation(_sim_obs_no_gripper())
            step.observation(_sim_obs_no_gripper())
        warns = [r for r in caplog.records if "absent from the observation" in r.getMessage()]
        assert len(warns) == 1
        assert "left/gripper" in warns[0].getMessage() and "right/gripper" in warns[0].getMessage()

    def test_all_present_unchanged(self):
        """A fully-present key set is packed verbatim with no zero-fill (the
        so101-style path); this must not regress."""
        E._WARNED_MISSING_STATE_KEYS.clear()
        keys = ["a", "b", "c"]
        out = _step(keys, 3).observation({"a": 1.0, "b": 2.0, "c": 3.0})
        state = out["observation.state"].numpy()
        np.testing.assert_allclose(state, [1.0, 2.0, 3.0], atol=1e-5)
        assert not E._WARNED_MISSING_STATE_KEYS  # no missing -> no warn recorded

    def test_all_missing_passthrough(self):
        """When NONE of the declared keys are present, leave the observation
        untouched so a clearer downstream error can fire (do not emit all-zero)."""
        E._WARNED_MISSING_STATE_KEYS.clear()
        obs = {"unrelated": 5.0}
        out = _step(["a", "b"], 2).observation(dict(obs))
        assert "observation.state" not in out
        assert out == obs


class TestAlohaEmbodimentActuatorConvention:
    def test_aloha_declares_14_actuator_keys(self):
        """The shipped aloha embodiment uses the 14 actuator convention (matching
        the model's actuators / robot_action_keys), not the 16 finger-JOINT
        names that crash/mis-align a canonical 14-D ACT."""
        emb = E.load_embodiment("aloha")
        assert emb.state_keys == ALOHA_14
        assert emb.action_keys == ALOHA_14
        # the old 16-finger-joint layout is gone
        assert "left/left_finger" not in emb.state_keys
        assert "left/left_finger" not in emb.action_keys

    def test_aloha_state_build_is_canonically_aligned(self):
        """End-to-end: build observation.state from a gripper-less sim obs through
        the real aloha embodiment config; arm stays aligned, grippers zero-filled."""
        E._WARNED_MISSING_STATE_KEYS.clear()
        emb = E.load_embodiment("aloha")
        step = _step(emb.state_keys, 14, dim_policy=emb.dim_policy)
        state = step.observation(_sim_obs_no_gripper())["observation.state"].numpy()
        expected = LEFT_ARM + [0.0] + RIGHT_ARM + [0.0]
        np.testing.assert_allclose(state, expected, atol=1e-5)
