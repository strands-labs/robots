#!/usr/bin/env python3
"""Comprehensive tests for PickAndPlaceReward — 4-phase curriculum reward.

Tests the full lifecycle: Reach → Grasp → Transport → Place, including:
- Phase transitions and auto-advance
- Dense reward shaping (distance-based)
- One-time bonus guards (grasp, lift, place)
- Drop detection and phase regression
- Stability bonus counting
- Energy and smoothness penalties
- Dict/array observation extraction
- Reset and diagnostic methods
- Edge cases (short state, None action, contact force)

Refs: Issue #124 (Marble 3D → Isaac Sim Pick-and-Place Pipeline)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strands_robots.rl_trainer import PickAndPlaceReward

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def reward():
    """Default reward function with standard SO-101-like observation layout."""
    return PickAndPlaceReward(
        object_pos_indices=(7, 10),
        ee_pos_indices=(0, 3),
        gripper_index=6,
        target_place_pos=np.array([0.3, 0.0, 0.75]),
    )


@pytest.fixture
def reward_with_contact():
    """Reward function with contact force detection enabled."""
    return PickAndPlaceReward(
        object_pos_indices=(7, 10),
        ee_pos_indices=(0, 3),
        gripper_index=6,
        contact_force_index=11,
        target_place_pos=np.array([0.3, 0.0, 0.75]),
        contact_force_threshold=0.1,
    )


def make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7), gripper=0.05, extra=None):
    """Helper: build a 12-element flat state vector.

    Layout: [ee_x, ee_y, ee_z, 0, 0, 0, gripper, obj_x, obj_y, obj_z, 0, 0]
    """
    state = np.zeros(12)
    state[0:3] = ee
    state[7:10] = obj
    state[6] = gripper
    if extra is not None:
        for idx, val in extra.items():
            state[idx] = val
    return state


# ── Initialization ────────────────────────────────────────────────


class TestInit:
    def test_default_target(self):
        r = PickAndPlaceReward()
        np.testing.assert_array_equal(r.target_place, [0.3, 0.0, 0.75])

    def test_custom_target(self):
        r = PickAndPlaceReward(target_place_pos=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(r.target_place, [1.0, 2.0, 3.0])

    def test_initial_phase_is_reach(self, reward):
        assert reward._phase == PickAndPlaceReward.PHASE_REACH
        assert reward.phase_name == "Reach"

    def test_phase_constants(self):
        assert PickAndPlaceReward.PHASE_REACH == 0
        assert PickAndPlaceReward.PHASE_GRASP == 1
        assert PickAndPlaceReward.PHASE_TRANSPORT == 2
        assert PickAndPlaceReward.PHASE_PLACE == 3

    def test_not_success_initially(self, reward):
        assert reward.is_success is False

    def test_repr(self, reward):
        rep = repr(reward)
        assert "Reach" in rep
        assert "success=False" in rep


# ── Phase 1: Reach ────────────────────────────────────────────────


class TestReachPhase:
    def test_far_from_object_low_reward(self, reward):
        state = make_state(ee=(1.0, 1.0, 1.0), obj=(0.0, 0.0, 0.7))
        r = reward(state, np.zeros(7))
        assert r < 0.5  # Far away → low reward
        assert reward._phase == PickAndPlaceReward.PHASE_REACH

    def test_close_to_object_high_reward(self, reward):
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        r = reward(state, np.zeros(7))
        assert r > 1.5  # Very close → near max (reach_scale=2.0)

    def test_reward_increases_as_distance_decreases(self, reward):
        r_far = reward(make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7)), np.zeros(7))
        reward.reset()
        r_close = reward(make_state(ee=(0.15, 0.0, 0.7), obj=(0.1, 0.0, 0.7)), np.zeros(7))
        assert r_close > r_far

    def test_phase_advances_when_close(self, reward):
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_GRASP

    def test_phase_stays_when_far(self, reward):
        state = make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_REACH


# ── Phase 2: Grasp ────────────────────────────────────────────────


class TestGraspPhase:
    def _advance_to_grasp(self, reward):
        """Helper: advance to grasp phase."""
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_GRASP

    def test_grasp_bonus_on_gripper_close(self, reward):
        self._advance_to_grasp(reward)
        # Close gripper near object
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        r = reward(state, np.zeros(7))
        # Should include grasp_bonus (5.0) + proximity reward
        assert r > 5.0
        assert reward._grasp_awarded is True

    def test_grasp_bonus_only_once(self, reward):
        self._advance_to_grasp(reward)
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        r1 = reward(state, np.zeros(7))
        r2 = reward(state, np.zeros(7))
        # Second call should NOT include grasp bonus again
        assert r1 > r2

    def test_no_grasp_with_open_gripper(self, reward):
        self._advance_to_grasp(reward)
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.05)
        r = reward(state, np.zeros(7))
        assert reward._grasp_awarded is False
        assert r < 5.0

    def test_lift_detection(self, reward):
        self._advance_to_grasp(reward)
        # Grasp
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        # Lift object above lift_height (0.10m default)
        state = make_state(ee=(0.1, 0.0, 0.85), obj=(0.1, 0.0, 0.85), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._lift_awarded is True
        assert reward._phase == PickAndPlaceReward.PHASE_TRANSPORT

    def test_partial_lift_reward(self, reward):
        self._advance_to_grasp(reward)
        # Grasp
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        # Partial lift (5cm, below lift_height of 10cm)
        state = make_state(ee=(0.1, 0.0, 0.75), obj=(0.1, 0.0, 0.75), gripper=0.01)
        r = reward(state, np.zeros(7))
        assert reward._lift_awarded is False
        assert reward._phase == PickAndPlaceReward.PHASE_GRASP
        # But should still get partial lift reward (lift_bonus * 0.5)
        assert r > 0


# ── Phase 3: Transport ────────────────────────────────────────────


class TestTransportPhase:
    def _advance_to_transport(self, reward):
        """Helper: advance to transport phase."""
        # Reach
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        # Grasp + Lift
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        state = make_state(ee=(0.1, 0.0, 0.85), obj=(0.1, 0.0, 0.85), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_TRANSPORT

    def test_reward_increases_near_target(self, reward):
        self._advance_to_transport(reward)
        r_far = reward(make_state(ee=(0.0, 0.5, 0.85), obj=(0.0, 0.5, 0.85), gripper=0.01), np.zeros(7))
        reward.reset()
        self._advance_to_transport(reward)
        r_close = reward(make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01), np.zeros(7))
        assert r_close > r_far

    def test_drop_detection_regresses_to_reach(self, reward):
        self._advance_to_transport(reward)
        # Drop: object falls back to table (Z back near initial)
        state = make_state(ee=(0.2, 0.0, 0.85), obj=(0.2, 0.0, 0.7), gripper=0.05)
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_REACH
        assert reward._grasp_awarded is False
        assert reward._lift_awarded is False

    def test_advances_to_place_at_target(self, reward):
        self._advance_to_transport(reward)
        # Object at target
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_PLACE


# ── Phase 4: Place ────────────────────────────────────────────────


class TestPlacePhase:
    def _advance_to_place(self, reward):
        """Helper: advance to place phase."""
        # Reach
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        # Grasp + Lift
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        state = make_state(ee=(0.1, 0.0, 0.85), obj=(0.1, 0.0, 0.85), gripper=0.01)
        reward(state, np.zeros(7))
        # Transport to target
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_PLACE

    def test_place_bonus_on_gripper_open(self, reward):
        self._advance_to_place(reward)
        # Open gripper at target
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.05)
        r = reward(state, np.zeros(7))
        assert r > 10.0  # place_bonus=10
        assert reward._place_awarded is True
        assert reward.is_success is True

    def test_place_bonus_only_once(self, reward):
        self._advance_to_place(reward)
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.05)
        r1 = reward(state, np.zeros(7))
        r2 = reward(state, np.zeros(7))
        assert r1 > r2

    def test_stability_bonus_after_10_steps(self, reward):
        self._advance_to_place(reward)
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.05)
        rewards = []
        for i in range(12):
            r = reward(state, np.zeros(7))
            rewards.append(r)
        # Step 10+ should include stability_bonus (5.0)
        assert rewards[10] > rewards[1]  # stability kicks in
        assert reward._steps_at_target >= 11

    def test_no_place_bonus_if_gripper_closed(self, reward):
        self._advance_to_place(reward)
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._place_awarded is False
        assert reward.is_success is False


# ── Contact Force Detection ───────────────────────────────────────


class TestContactForce:
    def test_grasp_requires_force_when_enabled(self, reward_with_contact):
        r = reward_with_contact
        # Advance to grasp phase
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        r(state, np.zeros(7))
        assert r._phase == PickAndPlaceReward.PHASE_GRASP

        # Close gripper but no contact force
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01, extra={11: 0.0})
        r(state, np.zeros(7))
        assert r._grasp_awarded is False

    def test_grasp_succeeds_with_force(self, reward_with_contact):
        r = reward_with_contact
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        r(state, np.zeros(7))

        # Close gripper WITH contact force
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01, extra={11: 0.5})
        r(state, np.zeros(7))
        assert r._grasp_awarded is True


# ── Penalties ─────────────────────────────────────────────────────


class TestPenalties:
    def test_energy_penalty(self, reward):
        state = make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7))
        r_zero = reward(state, np.zeros(7))
        reward.reset()
        r_high = reward(state, np.ones(7) * 10.0)
        # High action → more energy penalty
        assert r_high < r_zero

    def test_smoothness_penalty(self, reward):
        state = make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7))
        # First action (no prev)
        reward(state, np.zeros(7))
        # Smooth action
        r_smooth = reward(state, np.zeros(7))
        reward.reset()
        reward(state, np.zeros(7))
        # Jerky action
        r_jerky = reward(state, np.ones(7) * 10.0)
        assert r_smooth > r_jerky


# ── Observation Extraction ────────────────────────────────────────


class TestObsExtraction:
    def test_flat_array_obs(self, reward):
        state = np.zeros(12)
        state[0:3] = [0.1, 0.0, 0.7]
        state[7:10] = [0.1, 0.0, 0.7]
        r = reward(state, np.zeros(7))
        assert isinstance(r, float)

    def test_dict_obs_with_state_key(self, reward):
        state = np.zeros(12)
        state[0:3] = [0.1, 0.0, 0.7]
        state[7:10] = [0.1, 0.0, 0.7]
        obs = {"state": state}
        r = reward(obs, np.zeros(7))
        assert isinstance(r, float)

    def test_dict_obs_with_observation_state_key(self, reward):
        state = np.zeros(12)
        state[0:3] = [0.1, 0.0, 0.7]
        state[7:10] = [0.1, 0.0, 0.7]
        obs = {"observation.state": state}
        r = reward(obs, np.zeros(7))
        assert isinstance(r, float)

    def test_dict_obs_with_nested_dict(self, reward):
        """ManagerBasedRLEnv: dict of tensors → concatenated."""
        obs = {
            "state": {
                "joint_pos": np.zeros(3),
                "ee_pos": np.zeros(3),
                "gripper": np.array([0.05]),
                "obj_pos": np.zeros(3),
                "extra1": np.zeros(1),
                "extra2": np.zeros(1),
            }
        }
        r = reward(obs, np.zeros(7))
        assert isinstance(r, float)

    def test_none_action(self, reward):
        state = make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7))
        r = reward(state, None)
        assert isinstance(r, float)


# ── Reset ─────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_phase(self, reward):
        # Advance to grasp
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        assert reward._phase == PickAndPlaceReward.PHASE_GRASP

        reward.reset()
        assert reward._phase == PickAndPlaceReward.PHASE_REACH
        assert reward._prev_action is None
        assert reward._initial_obj_z is None
        assert reward._grasp_awarded is False
        assert reward._lift_awarded is False
        assert reward._place_awarded is False
        assert reward._steps_at_target == 0

    def test_reset_allows_re_grasp(self, reward):
        """After reset, grasp bonus can be awarded again."""
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        assert reward._grasp_awarded is True

        reward.reset()
        assert reward._grasp_awarded is False


# ── Diagnostics ───────────────────────────────────────────────────


class TestDiagnostics:
    def test_get_info_initial(self, reward):
        info = reward.get_info()
        assert info["phase"] == 0
        assert info["phase_name"] == "Reach"
        assert info["is_success"] is False
        assert info["grasp_awarded"] is False

    def test_get_info_after_grasp(self, reward):
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        reward(state, np.zeros(7))
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01)
        reward(state, np.zeros(7))
        info = reward.get_info()
        assert info["grasp_awarded"] is True
        assert info["phase"] == PickAndPlaceReward.PHASE_GRASP

    def test_phase_name_all_phases(self):
        r = PickAndPlaceReward()
        r._phase = 0
        assert r.phase_name == "Reach"
        r._phase = 1
        assert r.phase_name == "Grasp"
        r._phase = 2
        assert r.phase_name == "Transport"
        r._phase = 3
        assert r.phase_name == "Place"
        r._phase = 99
        assert r.phase_name == "Unknown"


# ── Auto-advance disabled ────────────────────────────────────────


class TestNoAutoAdvance:
    def test_phase_stays_when_auto_advance_off(self):
        r = PickAndPlaceReward(auto_advance=False)
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        r(state, np.zeros(7))
        assert r._phase == PickAndPlaceReward.PHASE_REACH  # Does NOT advance

    def test_manual_phase_set(self):
        r = PickAndPlaceReward(auto_advance=False)
        r._phase = PickAndPlaceReward.PHASE_TRANSPORT
        state = make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01)
        r(state, np.zeros(7))
        # Should NOT advance to Place even though near target
        assert r._phase == PickAndPlaceReward.PHASE_TRANSPORT


# ── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_short_state_vector(self):
        """State vector shorter than expected indices."""
        r = PickAndPlaceReward(object_pos_indices=(7, 10), ee_pos_indices=(0, 3))
        state = np.zeros(5)  # Too short
        val = r(state, np.zeros(3))
        assert isinstance(val, float)

    def test_gripper_index_out_of_range(self):
        r = PickAndPlaceReward(gripper_index=100)
        state = np.zeros(12)
        val = r(state, np.zeros(3))
        assert isinstance(val, float)

    def test_empty_dict_obs(self):
        r = PickAndPlaceReward()
        obs = {}
        val = r(obs, np.zeros(3))
        assert isinstance(val, float)

    def test_policy_key_obs(self):
        """ManagerBasedRLEnv sometimes uses 'policy' key."""
        r = PickAndPlaceReward()
        obs = {"policy": np.zeros(12)}
        val = r(obs, np.zeros(3))
        assert isinstance(val, float)

    def test_full_episode_lifecycle(self):
        """Full episode: reach → grasp → lift → transport → place → stable."""
        r = PickAndPlaceReward(
            object_pos_indices=(7, 10),
            ee_pos_indices=(0, 3),
            gripper_index=6,
            target_place_pos=np.array([0.3, 0.0, 0.75]),
        )
        action = np.zeros(7)

        # Phase 1: Reach
        r(make_state(ee=(0.5, 0.5, 0.5), obj=(0.1, 0.0, 0.7)), action)
        assert r.phase_name == "Reach"

        r(make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7)), action)
        assert r.phase_name == "Grasp"

        # Phase 2: Grasp + Lift
        r(make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7), gripper=0.01), action)
        assert r._grasp_awarded is True

        r(make_state(ee=(0.1, 0.0, 0.85), obj=(0.1, 0.0, 0.85), gripper=0.01), action)
        assert r.phase_name == "Transport"
        assert r._lift_awarded is True

        # Phase 3: Transport
        r(make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.01), action)
        assert r.phase_name == "Place"

        # Phase 4: Place
        r(make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.05), action)
        assert r.is_success is True
        assert r._place_awarded is True

        # Stability: 10 more steps
        for _ in range(10):
            r(make_state(ee=(0.3, 0.0, 0.75), obj=(0.3, 0.0, 0.75), gripper=0.05), action)
        assert r._steps_at_target >= 10

    def test_custom_reward_weights(self):
        r = PickAndPlaceReward(
            reach_scale=10.0,
            grasp_bonus=20.0,
            lift_bonus=15.0,
            transport_scale=8.0,
            place_bonus=50.0,
            stability_bonus=25.0,
        )
        state = make_state(ee=(0.1, 0.0, 0.7), obj=(0.1, 0.0, 0.7))
        val = r(state, np.zeros(7))
        # With reach_scale=10.0, reward should be much higher
        assert val > 5.0

    def test_list_input_converted_to_array(self):
        r = PickAndPlaceReward()
        obs = [0.1, 0.0, 0.7, 0, 0, 0, 0.05, 0.1, 0.0, 0.7, 0, 0]
        action = [0, 0, 0, 0, 0, 0, 0]
        val = r(obs, action)
        assert isinstance(val, float)


# ── Integration with existing RewardFunction ──────────────────────


class TestIntegration:
    def test_callable_interface(self, reward):
        """PickAndPlaceReward is callable like RewardFunction."""
        state = make_state()
        r = reward(state, np.zeros(7))
        assert isinstance(r, float)

    def test_importable(self):
        """Can be imported from rl_trainer."""
        from strands_robots.rl_trainer import PickAndPlaceReward as PPR

        assert PPR is not None
        assert hasattr(PPR, "PHASE_REACH")
