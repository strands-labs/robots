#!/usr/bin/env python3
"""Tests for training-related exports from strands_robots and parameter validation.

Validates:
1. Top-level package exports for RL trainer and Newton gym env
2. DreamgenIdmTrainer unknown parameter warnings
3. Full training matrix import paths from Issue #73
"""

import logging

# --- Tier 1: Top-level package exports ---


class TestTopLevelExports:
    """Verify all training-related symbols are importable from strands_robots."""

    def test_create_rl_trainer_in_all(self):
        import strands_robots

        assert "create_rl_trainer" in strands_robots.__all__

    def test_rl_config_in_all(self):
        import strands_robots

        assert "RLConfig" in strands_robots.__all__

    def test_sb3_trainer_in_all(self):
        import strands_robots

        assert "SB3Trainer" in strands_robots.__all__

    def test_reward_function_in_all(self):
        import strands_robots

        assert "RewardFunction" in strands_robots.__all__

    def test_newton_gym_env_in_all(self):
        import strands_robots

        assert "NewtonGymEnv" in strands_robots.__all__

    def test_evaluate_in_all(self):
        import strands_robots

        assert "evaluate" in strands_robots.__all__

    def test_create_trainer_in_all(self):
        import strands_robots

        assert "create_trainer" in strands_robots.__all__

    def test_cosmos_trainer_in_all(self):
        import strands_robots

        assert "CosmosTrainer" in strands_robots.__all__

    def test_strands_sim_env_in_all(self):
        import strands_robots

        assert "StrandsSimEnv" in strands_robots.__all__

    def test_direct_import_create_rl_trainer(self):
        from strands_robots import create_rl_trainer

        assert callable(create_rl_trainer)

    def test_direct_import_rl_config(self):
        from strands_robots import RLConfig

        config = RLConfig()
        assert config.algorithm == "ppo"

    def test_direct_import_newton_gym_env(self):
        from strands_robots import NewtonGymEnv

        assert NewtonGymEnv is not None

    def test_direct_import_evaluate(self):
        from strands_robots import evaluate

        assert callable(evaluate)

    def test_direct_import_create_trainer(self):
        from strands_robots import create_trainer

        assert callable(create_trainer)


# --- Tier 2: Import paths matching Issue #73 code examples ---


class TestIssue73ImportPaths:
    """Validate that every import path used in Issue #73 code examples works."""

    def test_create_trainer_from_training(self):
        """Issue #73: from strands_robots.training import create_trainer"""
        from strands_robots.training import create_trainer

        assert callable(create_trainer)

    def test_newton_config_from_newton(self):
        """Issue #73: from strands_robots.newton import NewtonConfig"""
        from strands_robots.newton import NewtonConfig

        config = NewtonConfig()
        assert hasattr(config, "num_envs")
        assert hasattr(config, "solver")

    def test_strands_sim_env_from_envs(self):
        """Issue #73: from strands_robots import StrandsSimEnv"""
        from strands_robots import StrandsSimEnv

        assert StrandsSimEnv is not None

    def test_create_policy_from_policies(self):
        """Issue #73: from strands_robots.policies import create_policy"""
        from strands_robots.policies import create_policy

        assert callable(create_policy)

    def test_newton_gym_env_from_newton(self):
        """Issue #73 RL section: from strands_robots.newton.newton_gym_env import NewtonGymEnv"""
        from strands_robots.newton.newton_gym_env import NewtonGymEnv

        assert NewtonGymEnv is not None

    def test_rl_trainer_from_rl_trainer(self):
        """Issue #73 RL section: from strands_robots.rl_trainer import create_rl_trainer"""
        from strands_robots.rl_trainer import create_rl_trainer

        assert callable(create_rl_trainer)

    def test_evaluate_from_training(self):
        """Issue #73 Eval section: from strands_robots.training import evaluate"""
        from strands_robots.training import evaluate

        assert callable(evaluate)

    def test_evaluate_signature(self):
        """Verify evaluate() accepts backend='newton' parameter."""
        import inspect

        from strands_robots.training import evaluate

        sig = inspect.signature(evaluate)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "mujoco"

    def test_create_trainer_all_providers(self):
        """Validate all 5 training providers are recognized by create_trainer."""
        from strands_robots.training import create_trainer

        providers = ["groot", "lerobot", "dreamgen_idm", "dreamgen_vla", "cosmos_predict"]
        for provider in providers:
            # Just test creation (will fail on train() without deps)
            trainer = create_trainer(provider, dataset_path="/tmp/test")
            assert trainer.provider_name == provider, f"Provider {provider} returned wrong name"


# --- Tier 3: DreamgenIdmTrainer parameter validation ---


class TestDreamgenIdmValidation:
    """Test that DreamgenIdmTrainer warns on commonly misused Issue #73 params."""

    def test_warns_on_idm_architecture(self, caplog):
        """Issue #73 incorrectly uses idm_architecture — should warn."""
        from strands_robots.training import create_trainer

        with caplog.at_level(logging.WARNING, logger="strands_robots.training"):
            create_trainer(
                "dreamgen_idm",
                dataset_path="/tmp/test",
                idm_architecture="siglip2_dit",
            )
        assert "idm_architecture" in caplog.text
        assert "NOT supported" in caplog.text

    def test_warns_on_action_dim(self, caplog):
        """Issue #73 incorrectly uses action_dim — should warn."""
        from strands_robots.training import create_trainer

        with caplog.at_level(logging.WARNING, logger="strands_robots.training"):
            create_trainer(
                "dreamgen_idm",
                dataset_path="/tmp/test",
                action_dim=7,
            )
        assert "action_dim" in caplog.text

    def test_warns_on_both_misused_params(self, caplog):
        """Both idm_architecture and action_dim together trigger warning."""
        from strands_robots.training import create_trainer

        with caplog.at_level(logging.WARNING, logger="strands_robots.training"):
            create_trainer(
                "dreamgen_idm",
                dataset_path="/tmp/test",
                idm_architecture="siglip2_dit",
                action_dim=7,
            )
        assert "idm_architecture" in caplog.text
        assert "action_dim" in caplog.text

    def test_no_warning_on_valid_params(self, caplog):
        """Valid params should NOT trigger the unknown parameter warning."""
        from strands_robots.training import create_trainer

        with caplog.at_level(logging.WARNING, logger="strands_robots.training"):
            create_trainer(
                "dreamgen_idm",
                dataset_path="/tmp/test",
                data_config="so100",
                embodiment_tag="so100",
                tune_action_head=True,
                video_backend="decord",
            )
        assert "NOT supported" not in caplog.text

    def test_still_stores_unknown_kwargs(self):
        """Unknown kwargs are stored in extra_kwargs (backwards compat)."""
        from strands_robots.training import create_trainer

        trainer = create_trainer(
            "dreamgen_idm",
            dataset_path="/tmp/test",
            idm_architecture="siglip2_dit",
            custom_param="foo",
        )
        assert trainer.extra_kwargs.get("idm_architecture") == "siglip2_dit"
        assert trainer.extra_kwargs.get("custom_param") == "foo"


# --- Tier 4: RL trainer configuration ---


class TestRLTrainerConfiguration:
    """Test RL trainer config matches Issue #73 use cases."""

    def test_ppo_g1_config(self):
        """Match Issue #73: PPO for G1 locomotion (4096 envs)."""
        from strands_robots.rl_trainer import create_rl_trainer

        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={
                "robot_name": "unitree_g1",
                "task": "walk forward at 1 m/s",
                "backend": "newton",
                "num_envs": 4096,
            },
            total_timesteps=10_000_000,
        )
        assert trainer.config.algorithm == "ppo"
        assert trainer.config.robot_name == "unitree_g1"
        assert trainer.config.backend == "newton"
        assert trainer.config.num_envs == 4096
        assert trainer.config.total_timesteps == 10_000_000

    def test_sac_so100_config(self):
        """Match Issue #73: SAC for manipulation (1024 envs)."""
        from strands_robots.rl_trainer import create_rl_trainer

        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={
                "robot_name": "so100",
                "task": "pick up the red cube",
                "backend": "newton",
                "num_envs": 1024,
                "newton_solver": "mujoco",
            },
            total_timesteps=2_000_000,
        )
        assert trainer.config.algorithm == "sac"
        assert trainer.config.robot_name == "so100"
        assert trainer.config.backend == "newton"
        assert trainer.config.num_envs == 1024
        assert trainer.config.newton_solver == "mujoco"

    def test_reward_function_locomotion(self):
        """Locomotion reward returns reasonable values."""
        import numpy as np

        from strands_robots.rl_trainer import RewardFunction

        # At target velocity
        obs = np.array([0.0, 0.0, 1.0, 0.0])  # qvel[0] = 1.0 (forward vel)
        action = np.zeros(2)
        reward = RewardFunction.locomotion_reward(obs, action, target_velocity=1.0)
        assert reward > 0, "Should be positive at target velocity"

    def test_reward_function_manipulation(self):
        """Manipulation reward is distance-based."""
        import numpy as np

        from strands_robots.rl_trainer import RewardFunction

        # Close to target
        obs = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # 3 qpos, 3 qvel
        target = np.array([0.0, 0.0, 0.1])
        reward_close = RewardFunction.manipulation_reward(obs, np.zeros(3), target)

        # Far from target
        obs_far = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        reward_far = RewardFunction.manipulation_reward(obs_far, np.zeros(3), target)

        assert reward_close > reward_far, "Closer to target should give higher reward"


# --- Tier 5: Eval harness script ---


class TestEvalHarness:
    """Test the eval_harness.py script configuration."""

    def test_eval_harness_importable(self):
        """The eval harness script defines correct policy configs."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("eval_harness", "scripts/eval_harness.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Verify all expected policies are configured
        assert "mock" in mod.POLICY_CONFIGS
        assert "groot_base" in mod.POLICY_CONFIGS
        assert "act_trained" in mod.POLICY_CONFIGS
        assert "pi0_trained" in mod.POLICY_CONFIGS
        assert "cosmos_posttrained" in mod.POLICY_CONFIGS

    def test_eval_harness_correct_param_names(self):
        """Eval harness uses pretrained_name_or_path (not model_path) for LeRobot."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("eval_harness", "scripts/eval_harness.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # LeRobot policies must use pretrained_name_or_path
        act_config = mod.POLICY_CONFIGS["act_trained"]
        assert "pretrained_name_or_path" in act_config, "ACT config should use pretrained_name_or_path, not model_path"
        pi0_config = mod.POLICY_CONFIGS["pi0_trained"]
        assert "pretrained_name_or_path" in pi0_config, "Pi0 config should use pretrained_name_or_path, not model_path"

    def test_eval_harness_tasks_defined(self):
        """All tasks from Issue #73 exist in the harness."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("eval_harness", "scripts/eval_harness.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert "pick_cube" in mod.TASKS
        assert "stack" in mod.TASKS
        assert "walk_forward" in mod.TASKS

        # Verify task robot assignments
        assert mod.TASKS["pick_cube"]["robot"] == "so100"
        assert mod.TASKS["walk_forward"]["robot"] == "unitree_g1"
