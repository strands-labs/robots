#!/usr/bin/env python3
"""Comprehensive tests for RL Trainer module (PPO/SAC for Newton/MuJoCo).

Extends existing tests to cover:
- SB3Trainer.train() (lines 274-371)
- SB3Trainer.evaluate() (lines 375-409)
- SB3Trainer.save() / load() (lines 418-428)
- _create_env / _create_mujoco_env / _create_newton_env (lines 227-259)
- Edge cases and error paths
"""

import builtins
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strands_robots.rl_trainer import (
    RewardFunction,
    RLConfig,
    RLTrainer,
    SB3Trainer,
    create_rl_trainer,
)

# ===========================================================================
# RLConfig tests (extended)
# ===========================================================================


class TestRLConfig:
    """Test RLConfig dataclass."""

    def test_defaults(self):
        config = RLConfig()
        assert config.algorithm == "ppo"
        assert config.robot_name == "so100"
        assert config.backend == "mujoco"
        assert config.num_envs == 1
        assert config.total_timesteps == 1_000_000
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99

    def test_ppo_config(self):
        config = RLConfig(
            algorithm="ppo",
            robot_name="unitree_g1",
            backend="newton",
            num_envs=4096,
            total_timesteps=10_000_000,
            n_steps=128,
            clip_range=0.2,
        )
        assert config.algorithm == "ppo"
        assert config.num_envs == 4096
        assert config.n_steps == 128

    def test_sac_config(self):
        config = RLConfig(
            algorithm="sac",
            robot_name="so100",
            total_timesteps=2_000_000,
            buffer_size=1_000_000,
            tau=0.005,
        )
        assert config.algorithm == "sac"
        assert config.buffer_size == 1_000_000

    def test_newton_config(self):
        config = RLConfig(
            backend="newton",
            newton_solver="featherstone",
            newton_dt=0.005,
            enable_cuda_graphs=True,
        )
        assert config.newton_solver == "featherstone"
        assert config.newton_dt == 0.005

    def test_output_defaults(self):
        config = RLConfig()
        assert config.output_dir == "./rl_outputs"
        assert config.save_freq == 50000
        assert config.eval_freq == 10000
        assert config.eval_episodes == 10
        assert config.seed == 42

    def test_reward_type_default(self):
        config = RLConfig()
        assert config.reward_type == "default"

    def test_misc_defaults(self):
        config = RLConfig()
        assert config.device == "auto"
        assert config.verbose == 1
        assert config.use_wandb is False

    def test_all_fields_settable(self):
        config = RLConfig(
            algorithm="sac",
            robot_name="panda",
            task="push block",
            backend="newton",
            num_envs=2048,
            newton_solver="xpbd",
            newton_dt=0.01,
            enable_cuda_graphs=False,
            total_timesteps=5_000_000,
            learning_rate=1e-3,
            batch_size=512,
            n_steps=64,
            gamma=0.98,
            gae_lambda=0.9,
            clip_range=0.3,
            ent_coef=0.01,
            vf_coef=0.25,
            buffer_size=2_000_000,
            tau=0.01,
            output_dir="/tmp/rl",
            save_freq=10000,
            eval_freq=5000,
            eval_episodes=5,
            log_interval=5,
            seed=123,
            reward_type="sparse",
            device="cuda",
            verbose=0,
            use_wandb=True,
        )
        assert config.ent_coef == 0.01
        assert config.device == "cuda"


# ===========================================================================
# create_rl_trainer tests (extended)
# ===========================================================================


class TestCreateRLTrainer:
    """Test create_rl_trainer() factory."""

    def test_ppo_creation(self):
        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={"robot_name": "unitree_g1", "task": "walk forward"},
            total_timesteps=100,
        )
        assert isinstance(trainer, SB3Trainer)
        assert trainer.config.algorithm == "ppo"
        assert trainer.config.robot_name == "unitree_g1"

    def test_sac_creation(self):
        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={"robot_name": "so100", "task": "pick up the red cube"},
            total_timesteps=100,
        )
        assert isinstance(trainer, SB3Trainer)
        assert trainer.config.algorithm == "sac"

    def test_newton_backend(self):
        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={
                "robot_name": "unitree_g1",
                "task": "walk",
                "backend": "newton",
                "num_envs": 4096,
                "newton_solver": "featherstone",
            },
        )
        assert trainer.config.backend == "newton"
        assert trainer.config.num_envs == 4096
        assert trainer.config.newton_solver == "featherstone"

    def test_default_env_config(self):
        trainer = create_rl_trainer()
        assert trainer.config.robot_name == "so100"
        assert trainer.config.backend == "mujoco"

    def test_extra_kwargs(self):
        trainer = create_rl_trainer(
            learning_rate=1e-4,
            gamma=0.95,
            seed=123,
        )
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.gamma == 0.95
        assert trainer.config.seed == 123

    def test_output_dir_passed(self):
        trainer = create_rl_trainer(output_dir="/tmp/test_rl")
        assert trainer.config.output_dir == "/tmp/test_rl"

    def test_total_timesteps_passed(self):
        trainer = create_rl_trainer(total_timesteps=42)
        assert trainer.config.total_timesteps == 42


# ===========================================================================
# RewardFunction tests (extended)
# ===========================================================================


class TestRewardFunction:
    """Test reward functions for common tasks."""

    def test_locomotion_reward_at_target(self):
        obs = np.zeros(20)
        obs[10] = 1.0
        action = np.zeros(6)
        reward = RewardFunction.locomotion_reward(obs, action, target_velocity=1.0)
        assert reward > 0.5

    def test_locomotion_reward_zero_vel(self):
        obs = np.zeros(20)
        action = np.zeros(6)
        reward_zero = RewardFunction.locomotion_reward(obs, action, target_velocity=1.0)
        obs_target = np.zeros(20)
        obs_target[10] = 1.0
        reward_target = RewardFunction.locomotion_reward(obs_target, action, target_velocity=1.0)
        assert reward_target > reward_zero

    def test_locomotion_energy_penalty(self):
        obs = np.zeros(20)
        obs[10] = 1.0
        action_low = np.zeros(6)
        action_high = np.ones(6) * 5.0
        reward_low = RewardFunction.locomotion_reward(obs, action_low)
        reward_high = RewardFunction.locomotion_reward(obs, action_high)
        assert reward_low > reward_high

    def test_locomotion_dict_obs(self):
        """Dict observation with 'state' key."""
        obs = {"state": np.zeros(20)}
        action = np.zeros(6)
        reward = RewardFunction.locomotion_reward(obs, action)
        assert isinstance(reward, float)

    def test_locomotion_dict_obs_lerobot_key(self):
        """Dict observation with 'observation.state' key."""
        obs = {"observation.state": np.zeros(20)}
        action = np.zeros(6)
        reward = RewardFunction.locomotion_reward(obs, action)
        assert isinstance(reward, float)

    def test_locomotion_single_element_state(self):
        """State with single element still works."""
        obs = np.array([0.5])
        action = np.zeros(1)
        reward = RewardFunction.locomotion_reward(obs, action)
        assert isinstance(reward, float)

    def test_locomotion_none_action(self):
        """None action → treated as zeros."""
        obs = np.zeros(20)
        reward = RewardFunction.locomotion_reward(obs, None)
        assert isinstance(reward, float)

    def test_locomotion_custom_penalties(self):
        """Custom energy_penalty and alive_bonus."""
        obs = np.zeros(20)
        action = np.ones(6)
        r1 = RewardFunction.locomotion_reward(obs, action, energy_penalty=0.0, alive_bonus=0.0)
        r2 = RewardFunction.locomotion_reward(obs, action, energy_penalty=1.0, alive_bonus=10.0)
        assert r2 != r1

    def test_manipulation_reward_closer_better(self):
        target = np.array([0.3, 0.0, 0.5])
        obs_close = np.zeros(12)
        obs_close[3] = 0.3
        obs_close[4] = 0.0
        obs_close[5] = 0.5
        obs_far = np.zeros(12)
        obs_far[3] = 1.0
        obs_far[4] = 1.0
        obs_far[5] = 1.0

        action = np.zeros(6)
        reward_close = RewardFunction.manipulation_reward(obs_close, action, target_pos=target)
        reward_far = RewardFunction.manipulation_reward(obs_far, action, target_pos=target)
        assert reward_close > reward_far

    def test_manipulation_reward_dict_obs(self):
        obs = {"state": np.zeros(12)}
        action = np.zeros(6)
        reward = RewardFunction.manipulation_reward(obs, action)
        assert isinstance(reward, float)

    def test_manipulation_reward_no_target(self):
        """No target_pos → reach_reward = 0."""
        obs = np.zeros(12)
        action = np.zeros(6)
        reward = RewardFunction.manipulation_reward(obs, action, target_pos=None)
        assert isinstance(reward, float)

    def test_manipulation_reward_short_state(self):
        """State shorter than 3 for ee_pos → reach_reward = 0."""
        obs = np.zeros(4)  # n_qpos=2, ee_pos has <3 elems
        action = np.zeros(2)
        target = np.array([0.5, 0.5, 0.5])
        reward = RewardFunction.manipulation_reward(obs, action, target_pos=target)
        assert isinstance(reward, float)

    def test_manipulation_none_action(self):
        """None action → zeros."""
        obs = np.zeros(12)
        reward = RewardFunction.manipulation_reward(obs, None)
        assert isinstance(reward, float)

    def test_sparse_reward(self):
        assert RewardFunction.sparse_success_reward(None, None, success=True) == 1.0
        assert RewardFunction.sparse_success_reward(None, None, success=False) == 0.0


# ===========================================================================
# RLTrainer ABC
# ===========================================================================


class TestRLTrainerAbstract:
    """Test RLTrainer ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RLTrainer()

    def test_sb3_trainer_is_concrete(self):
        config = RLConfig()
        trainer = SB3Trainer(config)
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")
        assert hasattr(trainer, "save")
        assert hasattr(trainer, "load")


# ===========================================================================
# SB3Trainer._get_reward_fn
# ===========================================================================


class TestSB3TrainerRewardSelection:
    """Test automatic reward function selection."""

    def test_locomotion_task_gets_locomotion_reward(self):
        trainer = create_rl_trainer(
            env_config={"robot_name": "unitree_g1", "task": "walk forward at 1 m/s"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is not None
        obs = np.zeros(20)
        result = reward_fn(obs, np.zeros(6))
        assert isinstance(result, float)

    def test_manipulation_task_gets_manipulation_reward(self):
        trainer = create_rl_trainer(
            env_config={"robot_name": "so100", "task": "pick up the red cube"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is not None

    def test_unknown_task_gets_none(self):
        trainer = create_rl_trainer(
            env_config={"robot_name": "so100", "task": "do something abstract"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is None

    def test_run_task(self):
        """'run' keyword triggers locomotion reward."""
        trainer = create_rl_trainer(
            env_config={"task": "run quickly"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is not None

    def test_grasp_task(self):
        """'grasp' keyword triggers manipulation reward."""
        trainer = create_rl_trainer(
            env_config={"task": "grasp the object"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is not None

    def test_cube_task(self):
        """'cube' keyword triggers manipulation reward."""
        trainer = create_rl_trainer(
            env_config={"task": "move the cube"},
        )
        reward_fn = trainer._get_reward_fn()
        assert reward_fn is not None


# ===========================================================================
# SB3Trainer._create_env
# ===========================================================================


class TestSB3TrainerCreateEnv:

    def test_create_env_mujoco(self):
        """MuJoCo backend calls _create_mujoco_env."""
        trainer = create_rl_trainer(env_config={"backend": "mujoco"})
        with patch.object(trainer, "_create_mujoco_env", return_value=MagicMock()) as mock:
            trainer._create_env()
            mock.assert_called_once()

    def test_create_env_newton(self):
        """Newton backend calls _create_newton_env."""
        trainer = create_rl_trainer(env_config={"backend": "newton"})
        with patch.object(trainer, "_create_newton_env", return_value=MagicMock()) as mock:
            trainer._create_env()
            mock.assert_called_once()

    def test_create_mujoco_env(self):
        """_create_mujoco_env instantiates StrandsSimEnv."""
        trainer = create_rl_trainer(env_config={"robot_name": "so100", "task": "test"})
        mock_env = MagicMock()
        mock_cls = MagicMock(return_value=mock_env)
        with patch("strands_robots.envs.StrandsSimEnv", mock_cls):
            env = trainer._create_mujoco_env()
        mock_cls.assert_called_once()
        assert env is mock_env

    def test_create_newton_env_fallback(self):
        """_create_newton_env falls back to mujoco when Newton unavailable.

        Uses builtins.__import__ interception to force ImportError even when
        Newton is actually installed (e.g. on Thor with GPU).
        """
        _real_import = builtins.__import__

        def _block_newton(name, *args, **kwargs):
            if "strands_robots.newton" in name:
                raise ImportError("mocked: newton not available")
            return _real_import(name, *args, **kwargs)

        trainer = create_rl_trainer(env_config={"backend": "newton"})
        builtins.__import__ = _block_newton
        try:
            with patch.object(trainer, "_create_mujoco_env", return_value=MagicMock()) as mock:
                trainer._create_newton_env()
                mock.assert_called_once()
        finally:
            builtins.__import__ = _real_import


# ===========================================================================
# SB3Trainer.train()
# ===========================================================================


class TestSB3TrainerTrain:

    def test_train_no_sb3_returns_error(self):
        """Without stable-baselines3, train returns error dict."""
        trainer = create_rl_trainer(total_timesteps=10)
        with patch.dict(sys.modules, {"stable_baselines3": None}):
            with patch("builtins.__import__", side_effect=ImportError("no sb3")):
                result = trainer.train()
        assert result["status"] == "error"
        assert "stable-baselines3" in result.get("error", "") or "stable-baselines3" in result.get("install", "")

    def test_train_ppo_full_flow(self, tmp_path):
        """Full PPO training flow with mocked SB3."""
        mock_ppo = MagicMock()
        mock_ppo_instance = MagicMock()
        mock_ppo.return_value = mock_ppo_instance

        mock_checkpoint = MagicMock()

        mock_sb3 = MagicMock()
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = MagicMock()
        mock_sb3.common.callbacks.CheckpointCallback = mock_checkpoint
        mock_sb3.common.callbacks.EvalCallback = MagicMock()

        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={"robot_name": "so100", "task": "walk forward"},
            total_timesteps=10,
            output_dir=str(tmp_path / "ppo_out"),
        )

        mock_env = MagicMock()

        with patch.object(trainer, "_create_env", return_value=mock_env):
            with patch.dict(
                sys.modules,
                {
                    "stable_baselines3": mock_sb3,
                    "stable_baselines3.common": mock_sb3.common,
                    "stable_baselines3.common.callbacks": mock_sb3.common.callbacks,
                },
            ):
                with patch("strands_robots.rl_trainer.PPO", mock_ppo, create=True):
                    with patch("strands_robots.rl_trainer.SAC", MagicMock(), create=True):
                        with patch("strands_robots.rl_trainer.CheckpointCallback", mock_checkpoint, create=True):
                            with patch("strands_robots.rl_trainer.EvalCallback", MagicMock(), create=True):
                                result = trainer.train()

        assert result["status"] == "success"
        assert result["algorithm"] == "ppo"
        assert result["total_timesteps"] == 10
        assert "model_path" in result
        assert "training_time_seconds" in result
        mock_ppo_instance.learn.assert_called_once()
        mock_ppo_instance.save.assert_called_once()

    def test_train_sac_full_flow(self, tmp_path):
        """Full SAC training flow with mocked SB3."""
        mock_sac = MagicMock()
        mock_sac_instance = MagicMock()
        mock_sac.return_value = mock_sac_instance

        mock_ppo = MagicMock()
        mock_checkpoint = MagicMock()
        mock_eval = MagicMock()

        mock_sb3 = MagicMock()
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = mock_sac
        mock_callbacks = MagicMock()
        mock_callbacks.CheckpointCallback = mock_checkpoint
        mock_callbacks.EvalCallback = mock_eval

        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={"robot_name": "so100", "task": "pick up cube"},
            total_timesteps=10,
            output_dir=str(tmp_path / "sac_out"),
        )

        mock_env = MagicMock()

        with patch.object(trainer, "_create_env", return_value=mock_env):
            with patch.dict(
                sys.modules,
                {
                    "stable_baselines3": mock_sb3,
                    "stable_baselines3.common": MagicMock(),
                    "stable_baselines3.common.callbacks": mock_callbacks,
                },
            ):
                result = trainer.train()

        assert result["status"] == "success"
        assert result["algorithm"] == "sac"

    def test_train_saves_config_json(self, tmp_path):
        """Training writes config.json to output dir."""
        mock_ppo = MagicMock()
        mock_ppo.return_value = MagicMock()

        mock_sb3 = MagicMock()
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = MagicMock()
        mock_callbacks = MagicMock()
        mock_callbacks.CheckpointCallback = MagicMock()
        mock_callbacks.EvalCallback = MagicMock()

        trainer = create_rl_trainer(
            algorithm="ppo",
            total_timesteps=10,
            output_dir=str(tmp_path / "out"),
        )

        with patch.object(trainer, "_create_env", return_value=MagicMock()):
            with patch.dict(
                sys.modules,
                {
                    "stable_baselines3": mock_sb3,
                    "stable_baselines3.common": MagicMock(),
                    "stable_baselines3.common.callbacks": mock_callbacks,
                },
            ):
                trainer.train()

        config_path = tmp_path / "out" / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            saved_config = json.load(f)
        assert saved_config["algorithm"] == "ppo"
        assert saved_config["total_timesteps"] == 10

    def test_train_fps_calculation(self, tmp_path):
        """FPS is calculated correctly."""
        mock_ppo = MagicMock()
        mock_ppo.return_value = MagicMock()

        mock_sb3 = MagicMock()
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = MagicMock()
        mock_callbacks = MagicMock()
        mock_callbacks.CheckpointCallback = MagicMock()
        mock_callbacks.EvalCallback = MagicMock()

        trainer = create_rl_trainer(
            total_timesteps=1000,
            output_dir=str(tmp_path / "fps"),
        )

        with patch.object(trainer, "_create_env", return_value=MagicMock()):
            with patch.dict(
                sys.modules,
                {
                    "stable_baselines3": mock_sb3,
                    "stable_baselines3.common": MagicMock(),
                    "stable_baselines3.common.callbacks": mock_callbacks,
                },
            ):
                result = trainer.train()

        assert "fps" in result
        assert result["fps"] > 0


# ===========================================================================
# SB3Trainer.evaluate()
# ===========================================================================


class TestSB3TrainerEvaluate:

    def test_evaluate_no_model(self):
        """Evaluate without model → error."""
        trainer = create_rl_trainer()
        result = trainer.evaluate()
        assert result["status"] == "error"
        assert "No model" in result["error"]

    def test_evaluate_with_model(self):
        """Evaluate runs episodes and returns stats."""
        trainer = create_rl_trainer()

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(6), None)
        trainer._model = mock_model

        # Mock env
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(12), {"task": "test"})
        # Simulate 3 steps then done
        mock_env.step.side_effect = [
            (np.zeros(12), 1.0, False, False, {"is_success": False}),
            (np.zeros(12), 1.0, False, False, {"is_success": False}),
            (np.zeros(12), 1.0, True, False, {"is_success": True}),
        ] * 5  # 5 episodes worth
        trainer._env = mock_env

        result = trainer.evaluate(num_episodes=5)
        assert "mean_reward" in result
        assert "success_rate" in result
        assert result["num_episodes"] == 5
        assert len(result["episodes"]) == 5

    def test_evaluate_truncation(self):
        """Episodes can end via truncation."""
        trainer = create_rl_trainer()

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(6), None)
        trainer._model = mock_model

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(12), {})
        mock_env.step.return_value = (np.zeros(12), 0.5, False, True, {"is_success": False})
        trainer._env = mock_env

        result = trainer.evaluate(num_episodes=2)
        assert result["success_rate"] == 0.0
        assert result["mean_reward"] == 0.5

    def test_evaluate_creates_env_if_needed(self):
        """If _env is None, evaluate creates one."""
        trainer = create_rl_trainer()

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(6), None)
        trainer._model = mock_model
        trainer._env = None

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(12), {})
        mock_env.step.return_value = (np.zeros(12), 1.0, True, False, {"is_success": True})

        with patch.object(trainer, "_create_env", return_value=mock_env):
            result = trainer.evaluate(num_episodes=1)

        assert result["success_rate"] == 100.0

    def test_evaluate_episode_details(self):
        """Each episode entry has correct fields."""
        trainer = create_rl_trainer()

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(6), None)
        trainer._model = mock_model

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(12), {})
        mock_env.step.return_value = (np.zeros(12), 2.5, True, False, {"is_success": True})
        trainer._env = mock_env

        result = trainer.evaluate(num_episodes=1)
        ep = result["episodes"][0]
        assert ep["episode"] == 0
        assert ep["reward"] == 2.5
        assert ep["steps"] == 1
        assert ep["success"] is True


# ===========================================================================
# SB3Trainer.save() / load()
# ===========================================================================


class TestSB3TrainerSaveLoad:

    def test_save_with_model(self, tmp_path):
        """save() calls model.save()."""
        trainer = create_rl_trainer()
        trainer._model = MagicMock()

        save_path = str(tmp_path / "model")
        trainer.save(save_path)
        trainer._model.save.assert_called_once_with(save_path)

    def test_save_without_model(self, tmp_path):
        """save() with no model does nothing."""
        trainer = create_rl_trainer()
        trainer._model = None
        trainer.save(str(tmp_path / "model"))  # Should not raise

    def test_load_ppo(self):
        """load() uses PPO.load for ppo algorithm."""
        trainer = create_rl_trainer(algorithm="ppo")
        mock_ppo = MagicMock()
        mock_sb3 = MagicMock()
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = MagicMock()

        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}):
            trainer.load("/path/to/model")

        mock_ppo.load.assert_called_once_with("/path/to/model")

    def test_load_sac(self):
        """load() uses SAC.load for sac algorithm."""
        trainer = create_rl_trainer(algorithm="sac")
        mock_sac = MagicMock()
        mock_sb3 = MagicMock()
        mock_sb3.PPO = MagicMock()
        mock_sb3.SAC = mock_sac

        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}):
            trainer.load("/path/to/model")

        mock_sac.load.assert_called_once_with("/path/to/model")

    def test_load_no_sb3_raises(self):
        """load() without sb3 raises ImportError."""
        trainer = create_rl_trainer()
        with patch.dict(sys.modules, {"stable_baselines3": None}):
            with patch("builtins.__import__", side_effect=ImportError("no sb3")):
                with pytest.raises(ImportError):
                    trainer.load("/path/to/model")


# ===========================================================================
# SB3Trainer attribute init
# ===========================================================================


class TestSB3TrainerInit:

    def test_initial_state(self):
        config = RLConfig()
        trainer = SB3Trainer(config)
        assert trainer.config is config
        assert trainer._model is None
        assert trainer._env is None
