#!/usr/bin/env python3
"""
End-to-End Training Integration Test

Validates the full pipeline: data collection → train → eval
for at least one policy (mock-based smoke test).

This test runs WITHOUT GPU dependencies — uses MuJoCo CPU simulation
and mock/lightweight components throughout.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check mujoco availability for tests that need simulation
try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# Check gymnasium availability for evaluation tests (StrandsSimEnv)
# Check for REAL gymnasium (not mock from other test files)
# Mock gymnasium has no __file__ attribute; real packages do
import sys as _sys

_gym_mod = _sys.modules.get("gymnasium")
if _gym_mod is not None:
    HAS_GYMNASIUM = hasattr(_gym_mod, "__file__") and _gym_mod.__file__ is not None
else:
    try:
        import gymnasium  # noqa: F401

        HAS_GYMNASIUM = True
    except (ImportError, ValueError):
        HAS_GYMNASIUM = False

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class TestE2EDataCollection:
    """Test end-to-end data collection pipeline."""

    def test_scripted_demo_generates_data(self):
        """Scripted demo should produce episode JSON files."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from scripted_pick_demo import ScriptedPickPolicy

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = ScriptedPickPolicy(num_joints=6)
            trajectory = policy.generate_trajectory(50)

            # Simulate recording
            episode_data = {"observations": [], "actions": []}
            for step in range(50):
                action = trajectory[step]
                episode_data["observations"].append(
                    {
                        "state": np.zeros(6).tolist(),
                        "step": step,
                    }
                )
                episode_data["actions"].append(action.tolist())

            ep_path = os.path.join(tmpdir, "episode_0000.json")
            with open(ep_path, "w") as f:
                json.dump(episode_data, f)

            # Validate
            assert os.path.exists(ep_path)
            with open(ep_path) as f:
                data = json.load(f)
            assert len(data["observations"]) == 50
            assert len(data["actions"]) == 50
            assert len(data["actions"][0]) == 7  # 6 joints + gripper

    def test_dataset_recorder_interface(self):
        """DatasetRecorder should be importable and have expected API."""
        from strands_robots.dataset_recorder import DatasetRecorder

        assert hasattr(DatasetRecorder, "create")
        assert hasattr(DatasetRecorder, "add_frame")
        assert hasattr(DatasetRecorder, "save_episode")
        assert hasattr(DatasetRecorder, "push_to_hub")


class TestE2ETrainerFactory:
    """Test that all trainers can be created and have expected interface."""

    TRAINER_CONFIGS = {
        "groot": {
            "base_model_path": "nvidia/GR00T-N1-2B",
            "dataset_path": "/tmp/test",
            "embodiment_tag": "so100",
            "data_config": "so100_dualcam",
            "max_steps": 10,
        },
        "lerobot": {
            "policy_type": "act",
            "dataset_repo_id": "test/data",
            "output_dir": "/tmp/test",
            "max_steps": 10,
        },
        "dreamgen_idm": {
            "dataset_path": "/tmp/test",
            "data_config": "so100",
        },
        "dreamgen_vla": {
            "base_model_path": "nvidia/GR00T-N1-2B",
            "dataset_path": "/tmp/test",
            "embodiment_tag": "so100",
        },
        "cosmos_predict": {
            "base_model_path": "nvidia/Cosmos-Predict2.5-2B",
            "dataset_path": "/tmp/test",
            "mode": "policy",
        },
    }

    @pytest.mark.parametrize("provider", TRAINER_CONFIGS.keys())
    def test_trainer_creation(self, provider):
        """Each trainer should be creatable with valid config."""
        from strands_robots.training import create_trainer

        kwargs = self.TRAINER_CONFIGS[provider]
        trainer = create_trainer(provider, **kwargs)
        assert trainer.provider_name == provider
        assert callable(trainer.train)
        assert callable(trainer.evaluate)

    def test_rl_trainer_creation(self):
        """RL trainer (PPO/SAC) should be creatable."""
        from strands_robots.rl_trainer import create_rl_trainer

        for algo in ["ppo", "sac"]:
            trainer = create_rl_trainer(
                algorithm=algo,
                env_config={"robot_name": "so100", "task": "pick cube"},
                total_timesteps=100,
            )
            assert trainer.config.algorithm == algo
            assert callable(trainer.train)


class TestE2EEvaluation:
    """Test evaluation pipeline end-to-end."""

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    def test_mock_policy_evaluation(self):
        """Mock policy should complete evaluation loop."""
        from strands_robots.policies import create_policy
        from strands_robots.training import evaluate

        policy = create_policy("mock")
        result = evaluate(
            policy=policy,
            task="pick up the red cube",
            robot_name="so100",
            num_episodes=2,
            max_steps_per_episode=10,
        )

        assert "success_rate" in result
        assert "mean_reward" in result
        assert "episodes" in result
        assert result["num_episodes"] == 2
        assert len(result["episodes"]) == 2
        assert result["policy_provider"] == "mock"

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    def test_evaluation_returns_per_episode_stats(self):
        """Each episode should have steps, reward, success."""
        from strands_robots.policies import create_policy
        from strands_robots.training import evaluate

        policy = create_policy("mock")
        result = evaluate(
            policy=policy,
            task="pick up the red cube",
            robot_name="so100",
            num_episodes=3,
            max_steps_per_episode=5,
        )

        for ep in result["episodes"]:
            assert "episode" in ep
            assert "steps" in ep
            assert "reward" in ep
            assert "success" in ep
            assert ep["steps"] > 0


class TestE2ERewardFunctions:
    """Test reward functions in realistic scenarios."""

    def test_locomotion_reward_batch(self):
        """Locomotion reward should handle batch of observations."""
        from strands_robots.rl_trainer import RewardFunction

        for _ in range(100):
            obs = np.random.randn(20)
            action = np.random.randn(6)
            reward = RewardFunction.locomotion_reward(obs, action)
            assert isinstance(reward, float)
            assert not np.isnan(reward)
            assert not np.isinf(reward)

    def test_manipulation_reward_batch(self):
        """Manipulation reward should handle batch of observations."""
        from strands_robots.rl_trainer import RewardFunction

        target = np.array([0.3, 0.0, 0.5])
        for _ in range(100):
            obs = np.random.randn(12)
            action = np.random.randn(6)
            reward = RewardFunction.manipulation_reward(obs, action, target_pos=target)
            assert isinstance(reward, float)
            assert not np.isnan(reward)


class TestE2EPolicyProviders:
    """Test that policy provider ecosystem is healthy."""

    def test_minimum_providers(self):
        """Should have at least 15 policy providers."""
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert len(providers) >= 15, f"Only {len(providers)} providers: {providers}"

    def test_mock_policy_works(self):
        """Mock policy should produce valid actions."""
        import asyncio

        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        obs = {"observation.state": np.zeros(6)}
        loop = asyncio.new_event_loop()
        try:
            actions = loop.run_until_complete(policy.get_actions(obs, "pick up the cube"))
            assert isinstance(actions, list)
            assert len(actions) > 0
        finally:
            loop.close()

    def test_policy_registry_consistency(self):
        """list_providers() should include all known providers."""
        from strands_robots.policies import list_providers

        providers = list_providers()
        expected = {"mock", "groot", "lerobot_local", "dreamgen", "dreamzero", "cosmos_predict"}
        for name in expected:
            assert name in providers, f"Missing provider: {name}"


class TestE2ESimulation:
    """Test simulation environment integration."""

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    def test_strands_sim_env_creation(self):
        """StrandsSimEnv should create with MuJoCo backend."""
        from strands_robots.envs import StrandsSimEnv

        env = StrandsSimEnv(
            robot_name="so100",
            task="test task",
            max_episode_steps=10,
        )
        assert env.action_space is not None
        assert env.observation_space is not None
        env.close()

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    @pytest.mark.skipif(not HAS_DISPLAY, reason="requires display for MuJoCo OpenGL rendering")
    def test_strands_sim_env_step_loop(self):
        """Full reset-step-render loop should work."""
        from strands_robots.envs import StrandsSimEnv

        env = StrandsSimEnv(
            robot_name="so100",
            task="test",
            max_episode_steps=5,
        )
        obs, info = env.reset()
        assert obs is not None

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)

        env.close()


class TestE2ENewtonConfig:
    """Test Newton backend configuration (no GPU required)."""

    def test_newton_config_all_solvers(self):
        """All 7 solver names should be valid."""
        from strands_robots.newton.newton_backend import SOLVER_MAP

        expected_solvers = {
            "mujoco",
            "featherstone",
            "semi_implicit",
            "xpbd",
            "vbd",
            "style3d",
            "implicit_mpm",
        }
        assert set(SOLVER_MAP.keys()) == expected_solvers

    def test_newton_config_creation(self):
        """NewtonConfig should accept all parameters."""
        from strands_robots.newton.newton_backend import NewtonConfig

        config = NewtonConfig(
            num_envs=4096,
            solver="featherstone",
            device="cuda:0",
            physics_dt=0.005,
        )
        assert config.num_envs == 4096
        assert config.solver == "featherstone"
