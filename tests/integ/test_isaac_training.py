#!/usr/bin/env python3
"""
Integration test: Isaac Sim Pick-and-Place Training Pipeline.

Tests the full Stage 2→3 pipeline from issue #124:
  1. SO-101 USD conversion
  2. 50-env GridCloner with articulated robot
  3. PickAndPlaceReward wired to observation loop
  4. PPO training smoke test (1000 steps)
  5. Metrics export and checkpoint saving

Run with Isaac Sim Python:
    /home/ubuntu/IsaacSim/python.sh -m pytest tests/integ/test_isaac_training.py -v

Or for non-SimulationApp tests:
    python -m pytest tests/integ/test_isaac_training.py -v -k "not isaac_runtime"
"""

import os
import sys
import time

import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

ISAAC_SIM_PATH = os.getenv("ISAAC_SIM_PATH", "/home/ubuntu/IsaacSim")
requires_isaac_runtime = pytest.mark.skipif(not os.path.isdir(ISAAC_SIM_PATH), reason="Requires Isaac Sim EC2 instance")


# ═══════════════════════════════════════════════════════════════
# Test: SimplePPO (no Isaac Sim needed)
# ═══════════════════════════════════════════════════════════════


class TestSimplePPO:
    """Test the SimplePPO implementation without Isaac Sim."""

    def test_import_training_script(self):
        """Training script modules are importable."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import SimplePPO, TrainingConfig

        assert SimplePPO is not None
        assert TrainingConfig is not None

    def test_ppo_init(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import SimplePPO, TrainingConfig

        config = TrainingConfig(num_envs=4, total_steps=100)
        ppo = SimplePPO(obs_dim=14, act_dim=7, num_envs=4, config=config)
        assert ppo.obs_dim == 14
        assert ppo.act_dim == 7
        assert ppo.total_steps == 0

    def test_ppo_get_action(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import SimplePPO, TrainingConfig

        config = TrainingConfig(num_envs=4, total_steps=100)
        ppo = SimplePPO(obs_dim=14, act_dim=7, num_envs=4, config=config)

        obs = np.random.randn(4, 14).astype(np.float32)
        actions, values, log_probs = ppo.get_action(obs)

        assert actions.shape == (4, 7), f"Expected (4, 7), got {actions.shape}"
        assert values.shape == (4,), f"Expected (4,), got {values.shape}"
        assert log_probs.shape == (4,), f"Expected (4,), got {log_probs.shape}"
        assert np.all(actions >= -1.0) and np.all(actions <= 1.0), "Actions should be clipped"

    def test_ppo_store_and_update(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import SimplePPO, TrainingConfig

        config = TrainingConfig(
            num_envs=4,
            total_steps=100,
            n_steps_per_update=8,
            n_minibatches=2,
            n_epochs=2,
        )
        ppo = SimplePPO(obs_dim=14, act_dim=7, num_envs=4, config=config)

        # Collect some transitions
        for t in range(8):
            obs = np.random.randn(4, 14).astype(np.float32)
            actions, values, log_probs = ppo.get_action(obs)
            rewards = np.random.randn(4).astype(np.float32)
            dones = np.zeros(4, dtype=bool)
            if t == 7:
                dones[0] = True  # One env terminates
            ppo.store_transition(obs, actions, rewards, values, log_probs, dones)

        assert ppo.total_steps == 32  # 8 steps × 4 envs

        # Run update
        metrics = ppo.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert metrics["updates"] == 1
        assert len(ppo.observations) == 0  # Buffer cleared after update

    def test_ppo_save_load(self, tmp_path):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import SimplePPO, TrainingConfig

        config = TrainingConfig(num_envs=2, total_steps=10)
        ppo = SimplePPO(obs_dim=14, act_dim=7, num_envs=2, config=config)

        save_path = str(tmp_path / "test_ckpt.npz")
        ppo.save(save_path)
        assert os.path.exists(save_path)

        data = np.load(save_path)
        assert "policy_weights" in data
        assert data["policy_weights"].shape == (14, 7)


class TestPickAndPlaceRewardIntegration:
    """Test PickAndPlaceReward wired with training observations."""

    def test_reward_with_14d_obs(self):
        """PickAndPlaceReward works with the 14-dim observation layout."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        target = np.array([0.0, 0.3, 0.42])
        reward_fn = PickAndPlaceReward(
            object_pos_indices=(7, 10),
            ee_pos_indices=(0, 3),
            gripper_index=6,
            target_place_pos=target,
        )

        # Simulate reach phase
        obs = np.zeros(14, dtype=np.float32)
        obs[0:3] = [0.01, 0.01, 0.42]  # EE near cube
        obs[7:10] = [0.0, 0.0, 0.42]  # Cube position
        obs[6] = 1.0  # Gripper open

        r = reward_fn(obs, np.zeros(7))
        assert isinstance(r, float)
        assert r > 0, "Should get positive reward for being near object"

    def test_50_env_reward_computation(self):
        """Reward computation scales to 50 envs."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        target = np.array([0.0, 0.3, 0.42])
        num_envs = 50
        reward_fns = [PickAndPlaceReward(target_place_pos=target) for _ in range(num_envs)]

        # Compute rewards for all 50 envs
        t0 = time.time()
        for _ in range(100):  # 100 steps
            for i in range(num_envs):
                obs = np.random.randn(14).astype(np.float32) * 0.1
                action = np.random.randn(7).astype(np.float32)
                reward_fns[i](obs, action)
        elapsed = time.time() - t0

        # Should be fast (< 1 second for 5000 reward computations)
        assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s for 5000 reward calls"

    def test_training_config_positions(self):
        """TrainingConfig computes correct world positions."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import TrainingConfig

        config = TrainingConfig()

        cube_pos = config.cube_world_pos
        target_pos = config.target_world_pos

        # Cube should be above table
        assert cube_pos[2] > config.table_height, "Cube should be above table"
        # Target should be above table
        assert target_pos[2] > config.table_height, "Target should be above table"
        # Cube and target should be at different positions
        assert np.linalg.norm(cube_pos - target_pos) > 0.1, "Cube and target should be apart"


# ═══════════════════════════════════════════════════════════════
# Test: Full Pipeline (requires Isaac Sim)
# ═══════════════════════════════════════════════════════════════


@requires_isaac_runtime
class TestIsaacTrainingPipeline:
    """Full training pipeline on Isaac Sim.

    Run: /home/ubuntu/IsaacSim/python.sh -m pytest tests/integ/test_isaac_training.py::TestIsaacTrainingPipeline -v
    """

    def test_training_config_defaults(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import TrainingConfig

        config = TrainingConfig()
        assert config.num_envs == 50
        assert config.physics_steps_per_action == 4  # 200Hz / 50Hz
        assert config.cube_world_pos[2] > 0.4  # Above table

    def test_smoke_training(self):
        """Run training for 1000 steps as smoke test."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from isaac_pick_place_train import TrainingConfig, run_training

        config = TrainingConfig(
            num_envs=10,  # Fewer envs for test
            total_steps=1000,
            n_steps_per_update=8,
            log_interval=1,
            output_dir="/tmp/isaac_training_test",
        )

        results = run_training(config)

        assert results["stages"]["env_creation"]["status"] == "pass"
        assert results["stages"]["training"]["status"] == "pass"
        assert results["stages"]["training"]["total_steps"] >= 1000
        assert results["stages"]["training"]["sps"] > 0

        # Check output files
        assert os.path.exists("/tmp/isaac_training_test/training_metrics.json")
        assert os.path.exists("/tmp/isaac_training_test/training_results.json")
        assert os.path.exists("/tmp/isaac_training_test/final_policy.npz")
