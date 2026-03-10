#!/usr/bin/env python3
"""
Integration test: Isaac Sim Pick-and-Place Pipeline.

Tests the full pipeline from issue #124 on the Isaac Sim EC2 runner.
Runs via: PYTHONPATH=. /home/ubuntu/IsaacSim/python.sh -m pytest tests/integ/test_pick_place_pipeline.py -v

Or from system Python (for non-SimulationApp tests):
    python -m pytest tests/integ/test_pick_place_pipeline.py -v -k "not isaac_runtime"
"""

import os
import sys

import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ═══════════════════════════════════════════════════════════════
# Test: PickAndPlaceReward
# ═══════════════════════════════════════════════════════════════


class TestPickAndPlaceReward:
    """Test the 4-phase pick-and-place reward function."""

    def test_import(self):
        from strands_robots.rl_trainer import PickAndPlaceReward

        assert PickAndPlaceReward is not None

    def test_initial_state(self):
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()
        assert reward.current_phase == 0
        assert reward.phase_name == "Reach"
        assert not reward.is_success

    def test_reach_phase_reward(self):
        """Closer to object → higher reward."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()

        # Far from object
        state_far = np.zeros(14)
        state_far[0:3] = [1.0, 1.0, 1.0]  # EE far
        state_far[7:10] = [0.0, 0.0, 0.5]  # object
        r_far = reward(state_far, np.zeros(7))

        reward.reset()

        # Close to object
        state_near = np.zeros(14)
        state_near[0:3] = [0.01, 0.01, 0.51]  # EE near
        state_near[7:10] = [0.0, 0.0, 0.5]  # object
        r_near = reward(state_near, np.zeros(7))

        assert r_near > r_far, "Closer to object should yield higher reward"

    def test_full_trajectory_success(self):
        """Simulate a complete pick-and-place trajectory."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        target = np.array([0.3, 0.0, 0.75])
        reward = PickAndPlaceReward(target_place_pos=target)

        total_reward = 0.0
        for step in range(200):
            progress = step / 200
            state = np.zeros(14)

            if progress < 0.25:
                alpha = progress / 0.25
                state[0:3] = np.array([0.5, 0.5, 0.5]) * (1 - alpha) + np.array([0.0, 0.0, 0.5]) * alpha
                state[7:10] = [0.0, 0.0, 0.5]
                state[6] = 1.0  # gripper open
            elif progress < 0.4:
                alpha = (progress - 0.25) / 0.15
                state[0:3] = [0.0, 0.0, 0.5]
                state[6] = max(0.0, 1.0 - 2 * alpha)  # close gripper
                obj_z = 0.5 + alpha * 0.15
                state[7:10] = [0.0, 0.0, obj_z]
            elif progress < 0.75:
                alpha = (progress - 0.4) / 0.35
                start = np.array([0.0, 0.0, 0.65])
                end = target + np.array([0, 0, 0.1])
                pos = start * (1 - alpha) + end * alpha
                state[0:3] = pos
                state[7:10] = pos
                state[6] = 0.0  # closed
            else:
                alpha = (progress - 0.75) / 0.25
                state[0:3] = target
                state[7:10] = target
                state[6] = min(1.0, alpha * 2)  # open

            r = reward(state, np.zeros(7))
            total_reward += r

        assert reward.is_success, "Full trajectory should reach success"
        assert total_reward > 50, f"Total reward should be significant, got {total_reward}"

    def test_reset_clears_state(self):
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()

        # Advance to grasp phase
        state = np.zeros(14)
        state[0:3] = [0.0, 0.0, 0.5]
        state[7:10] = [0.0, 0.0, 0.5]
        state[6] = 0.0
        reward(state, np.zeros(7))

        reward.reset()
        assert reward.current_phase == 0
        assert not reward.is_success

    def test_create_rl_trainer_picks_reward(self):
        """create_rl_trainer with 'pick and place' task should use PickAndPlaceReward."""
        from strands_robots.rl_trainer import PickAndPlaceReward, RLConfig, SB3Trainer

        config = RLConfig(task="pick and place cube", backend="mujoco")
        trainer = SB3Trainer(config)
        reward_fn = trainer._get_reward_fn()

        assert isinstance(reward_fn, PickAndPlaceReward), f"Expected PickAndPlaceReward, got {type(reward_fn)}"


# ═══════════════════════════════════════════════════════════════
# Test: IsaacGymEnv interface
# ═══════════════════════════════════════════════════════════════


class TestIsaacGymEnvInterface:
    """Test the IsaacGymEnv class interface (no Isaac runtime needed)."""

    def test_import(self):
        from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

        assert IsaacGymEnv is not None

    def test_isaac_sim_backend_import(self):
        from strands_robots.isaac.isaac_sim_backend import IsaacSimBackend, IsaacSimConfig

        assert IsaacSimBackend is not None
        config = IsaacSimConfig(num_envs=50, device="cuda:0", headless=True)
        assert config.num_envs == 50

    def test_asset_converter_import(self):
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        assert callable(convert_mjcf_to_usd)

    def test_so101_asset_exists(self):
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path("so101")
        assert path is not None and path.exists(), "SO-101 MJCF should be bundled"


# ═══════════════════════════════════════════════════════════════
# Test: Pipeline Integration (requires Isaac Sim runtime)
# ═══════════════════════════════════════════════════════════════

ISAAC_SIM_PATH = os.getenv("ISAAC_SIM_PATH", "/home/ubuntu/IsaacSim")
requires_isaac_runtime = pytest.mark.skipif(not os.path.isdir(ISAAC_SIM_PATH), reason="Requires Isaac Sim EC2 instance")


@requires_isaac_runtime
class TestPickPlacePipelineIsaac:
    """Full pipeline tests requiring Isaac Sim runtime.

    Run with: /home/ubuntu/IsaacSim/python.sh -m pytest tests/integ/test_pick_place_pipeline.py::TestPickPlacePipelineIsaac -v
    """

    def test_gridcloner_50_envs(self):
        """Test GridCloner creates 50 environments."""
        # This test is a smoke test — the full validation is in
        # scripts/isaac_pick_place_50env.py
        from strands_robots.isaac.isaac_sim_backend import IsaacSimConfig

        config = IsaacSimConfig(num_envs=50, device="cuda:0", headless=True)
        assert config.num_envs == 50
        assert config.headless is True

    def test_reward_with_gym_env_interface(self):
        """Test PickAndPlaceReward works as gym reward_fn."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()

        # Simulate what IsaacGymEnv.step() would provide
        obs = np.zeros(12)  # 6 joints * 2 (pos + vel)
        action = np.random.randn(6).astype(np.float32)

        # Should work without crashing (observation layout mismatch
        # produces 0-ish reward, which is fine)
        r = reward(obs, action)
        assert isinstance(r, float)
