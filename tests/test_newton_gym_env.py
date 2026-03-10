"""Tests for strands_robots/newton/newton_gym_env.py — Newton Gymnasium Wrapper.

Tests cover:
1. Module import and class existence
2. NewtonGymEnv init with default and custom configs
3. Observation/action space construction
4. reset() and step() API
5. Multi-env (batched) mode
6. Reward and success function integration
7. close() cleanup

All tests use mocked NewtonBackend to run on CPU/ubuntu without GPU.

IMPORTANT: The newton_gym_env module conditionally defines NewtonGymEnv only
when gymnasium is installed (HAS_GYM=True). On CI without gymnasium, we mock
the gymnasium package at module level and force-reimport newton_gym_env so
that the real class (not the stub) is defined. This uses the 4-step pattern:
  1. Create realistic mock gymnasium modules with types.ModuleType
  2. Inject into sys.modules["gymnasium"] and sys.modules["gymnasium.spaces"]
  3. Delete stale cache from BOTH sys.modules AND parent package __dict__
  4. Re-import the target module (forces top-level code re-execution)
"""

import ast
import builtins
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ───────────────────────── Mock gymnasium if absent ─────────────────────────

try:
    import gymnasium  # noqa: F401

    _REAL_GYM = True
except ImportError:
    _REAL_GYM = False


class _MockBox:
    """Minimal gymnasium.spaces.Box replacement for tests."""

    def __init__(self, low=-np.inf, high=np.inf, shape=None, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if shape else low
        self.high = np.full(shape, high, dtype=dtype) if shape else high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.zeros(self.shape, dtype=self.dtype)


class _MockEnv:
    """Minimal gymnasium.Env replacement for tests.

    Must be a real class (not MagicMock) so NewtonGymEnv can inherit from it
    without MRO errors.
    """

    metadata = {}
    observation_space = None
    action_space = None
    render_mode = None

    def __init__(self, **kwargs):
        pass

    def reset(self, seed=None, options=None):
        return None, {}

    def step(self, action):
        return None, 0.0, False, False, {}

    def render(self):
        return None

    def close(self):
        pass


def _install_mock_gymnasium():
    """Inject a mock gymnasium package into sys.modules if not installed."""
    if _REAL_GYM:
        return  # Real gymnasium available, no mocking needed

    # Create gymnasium module
    mock_gym = types.ModuleType("gymnasium")
    mock_gym.Env = _MockEnv

    # Create gymnasium.spaces submodule
    mock_spaces = types.ModuleType("gymnasium.spaces")
    mock_spaces.Box = _MockBox

    mock_gym.spaces = mock_spaces

    sys.modules["gymnasium"] = mock_gym
    sys.modules["gymnasium.spaces"] = mock_spaces


def _force_reimport_newton_gym_env():
    """Remove cached newton_gym_env from all import caches and reimport.

    Python caches submodules in two places:
      1. sys.modules["strands_robots.newton.newton_gym_env"]
      2. strands_robots.newton.__dict__["newton_gym_env"]  (parent package attribute)

    We must clean BOTH to force a fresh import with our mocked gymnasium.
    """
    mod_key = "strands_robots.newton.newton_gym_env"

    # Step 1: Remove from sys.modules
    if mod_key in sys.modules:
        del sys.modules[mod_key]

    # Step 2: Remove cached attribute from parent package
    parent_key = "strands_robots.newton"
    if parent_key in sys.modules:
        parent_pkg = sys.modules[parent_key]
        for attr in ("newton_gym_env", "NewtonGymEnv"):
            if hasattr(parent_pkg, attr):
                try:
                    delattr(parent_pkg, attr)
                except AttributeError:
                    pass

    # Step 3: Re-import (forces top-level code re-execution with mock gymnasium)
    from strands_robots.newton import newton_gym_env

    assert newton_gym_env.HAS_GYM is True, (
        "Mock gymnasium injection failed — HAS_GYM is still False. "
        "This means sys.modules caching prevented the reimport."
    )


# Execute mock injection at module load time (before pytest collects tests)
_install_mock_gymnasium()
_force_reimport_newton_gym_env()


# ───────────────────────────── Syntax & Import ─────────────────────────────


class TestNewtonGymEnvSyntax:
    """Verify the module parses and exports correctly."""

    def test_module_parses(self):
        """newton_gym_env.py is valid Python."""
        src = Path(__file__).resolve().parent.parent / "strands_robots" / "newton" / "newton_gym_env.py"
        source = src.read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_exports(self):
        """Module exports NewtonGymEnv."""
        from strands_robots.newton.newton_gym_env import __all__

        assert "NewtonGymEnv" in __all__

    def test_newton_init_exports(self):
        """Newton package __init__ exports NewtonGymEnv."""
        from strands_robots.newton import __all__

        assert "NewtonGymEnv" in __all__


# ───────────────────────────── Mock Helpers ─────────────────────────────


def _make_mock_backend(n_joints=6, num_envs=1):
    """Create a mock NewtonBackend with realistic behavior."""
    mock_backend = MagicMock()
    mock_backend.create_world.return_value = {"success": True}
    mock_backend.add_robot.return_value = {
        "success": True,
        "robot_info": {
            "name": "test_robot",
            "num_joints": n_joints,
            "num_bodies": n_joints + 1,
            "joint_offset": 0,
            "body_offset": 0,
        },
    }
    mock_backend.replicate.return_value = {
        "success": True,
        "env_info": {"num_envs": num_envs},
    }
    mock_backend.step.return_value = {
        "success": True,
        "sim_time": 0.005,
        "step_count": 1,
    }
    mock_backend.reset.return_value = {"success": True}

    jpos = np.zeros(n_joints * num_envs, dtype=np.float32)
    jvel = np.zeros(n_joints * num_envs, dtype=np.float32)
    mock_backend.get_observation.return_value = {
        "success": True,
        "observations": {
            "so100": {
                "joint_positions": jpos,
                "joint_velocities": jvel,
            }
        },
    }
    mock_backend.render.return_value = {"success": True, "image": None}
    mock_backend.destroy.return_value = {"success": True}
    mock_backend._finalize_model = MagicMock()
    return mock_backend


def _create_env(n_joints=6, num_envs=1, **kwargs):
    """Create a NewtonGymEnv with a fully mocked backend."""
    from strands_robots.newton.newton_gym_env import NewtonGymEnv

    mock_backend = _make_mock_backend(n_joints=n_joints, num_envs=num_envs)

    # Patch the imports inside _init_backend
    mock_config = MagicMock()
    mock_config.num_envs = num_envs
    mock_config.solver = "mujoco"
    mock_config.device = "cpu"
    mock_config.physics_dt = 0.005

    with patch("strands_robots.newton.NewtonBackend", return_value=mock_backend):
        with patch("strands_robots.newton.NewtonConfig", return_value=mock_config):
            env = NewtonGymEnv(
                robot_name="so100",
                config=mock_config,
                **kwargs,
            )

    # Store mock for test assertions
    env._mock_backend = mock_backend
    return env


def _make_import_raiser(*blocked_modules):
    """Create an import side-effect that raises ImportError for blocked modules.

    Works even on systems where the real module IS installed (e.g. Thor GPU)
    by intercepting builtins.__import__ and raising ImportError for the
    specified module names.
    """
    _real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        for blocked in blocked_modules:
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"Mocked: {name} not available")
        return _real_import(name, *args, **kwargs)

    return _fake_import


# ───────────────────────────── Single Env Tests ─────────────────────────────


class TestNewtonGymEnvSingleEnv:
    """Tests for single-env NewtonGymEnv (num_envs=1)."""

    def test_init_creates_backend(self):
        """NewtonGymEnv creates a backend and stores robot info."""
        env = _create_env(n_joints=6, num_envs=1)
        assert env._robot_name == "so100"
        assert env._num_envs == 1
        assert env._n_joints == 6

    def test_observation_space_shape(self):
        """Single env observation space is (obs_dim,)."""
        env = _create_env(n_joints=6, num_envs=1)
        # obs = joint_positions (6) + joint_velocities (6) = 12
        assert env.observation_space.shape == (12,)
        assert env.action_space.shape == (6,)

    def test_reset_returns_obs_and_info(self):
        """reset() returns (obs, info) tuple."""
        env = _create_env(n_joints=6, num_envs=1)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert "task" in info
        env._mock_backend.reset.assert_called_once()

    def test_step_returns_5_tuple(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = _create_env(n_joints=6, num_envs=1)
        env.reset()
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncation_at_max_steps(self):
        """step() returns truncated=True after max_episode_steps."""
        env = _create_env(n_joints=6, num_envs=1, max_episode_steps=5)
        env.reset()
        action = np.zeros(6, dtype=np.float32)

        for i in range(4):
            _, _, _, truncated, _ = env.step(action)
            assert not truncated, f"Should not be truncated at step {i + 1}"

        _, _, _, truncated, _ = env.step(action)
        assert truncated, "Should be truncated at step 5"

    def test_custom_reward_fn(self):
        """Custom reward function is called on each step."""
        reward_fn = MagicMock(return_value=42.0)
        env = _create_env(n_joints=6, num_envs=1, reward_fn=reward_fn)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert reward == 42.0
        reward_fn.assert_called_once()

    def test_custom_success_fn(self):
        """Custom success function triggers termination."""
        success_fn = MagicMock(return_value=True)
        env = _create_env(n_joints=6, num_envs=1, success_fn=success_fn)
        env.reset()
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is True
        assert info["is_success"] is True

    def test_close_destroys_backend(self):
        """close() calls backend.destroy()."""
        env = _create_env(n_joints=6, num_envs=1)
        mock = env._mock_backend
        env.close()
        mock.destroy.assert_called_once()
        assert env._backend is None

    def test_default_reward_is_zero(self):
        """Without reward_fn, reward is 0.0."""
        env = _create_env(n_joints=6, num_envs=1)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert reward == 0.0

    def test_default_terminated_is_false(self):
        """Without success_fn, terminated is False."""
        env = _create_env(n_joints=6, num_envs=1)
        env.reset()
        _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is False


# ───────────────────────────── Multi-Env Tests ─────────────────────────────


class TestNewtonGymEnvMultiEnv:
    """Tests for multi-env (batched) NewtonGymEnv."""

    def test_batched_observation_space(self):
        """Multi-env observation space is (num_envs, obs_dim)."""
        env = _create_env(n_joints=6, num_envs=4096)
        assert env.observation_space.shape == (4096, 12)
        assert env.action_space.shape == (4096, 6)
        assert env.num_envs == 4096

    def test_replicate_called_for_multi_env(self):
        """replicate() is called when num_envs > 1."""
        env = _create_env(n_joints=6, num_envs=16)
        env._mock_backend.replicate.assert_called_once_with(16)

    def test_batched_step_returns_arrays(self):
        """Multi-env step returns batched reward/terminated/truncated."""
        n_envs = 8
        env = _create_env(n_joints=6, num_envs=n_envs)
        env.reset()
        action = np.zeros((n_envs, 6), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (n_envs, 12)
        assert reward.shape == (n_envs,)
        assert terminated.shape == (n_envs,)
        assert truncated.shape == (n_envs,)

    def test_per_env_reset(self):
        """reset() supports per-env reset via options."""
        env = _create_env(n_joints=6, num_envs=4)
        env.reset(options={"env_ids": [0, 2]})
        env._mock_backend.reset.assert_called_with(env_ids=[0, 2])

    def test_single_env_does_not_call_replicate(self):
        """replicate() is NOT called when num_envs=1."""
        env = _create_env(n_joints=6, num_envs=1)
        env._mock_backend.replicate.assert_not_called()
        # Instead, _finalize_model should be called
        env._mock_backend._finalize_model.assert_called_once()


# ───────────────────────────── Integration with evaluate() ─────────────────


class TestEvaluateNewtonBackend:
    """Test that evaluate() now works with Newton backend."""

    def test_evaluate_newton_attempts_import(self):
        """evaluate(backend='newton') tries to import NewtonGymEnv."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.provider_name = "mock"

        # When Newton import fails, should return error with proper message.
        # We must block the real import even on systems where Newton IS installed
        # (e.g. Thor GPU), so we intercept builtins.__import__.
        fake_import = _make_import_raiser("strands_robots.newton")
        with patch("builtins.__import__", side_effect=fake_import):
            result = evaluate(mock_policy, "pick cube", "so100", backend="newton")

        assert "error" in result
        assert result["num_episodes"] == 0

    def test_evaluate_newton_with_mock_env(self):
        """evaluate(backend='newton') works with a mocked NewtonGymEnv."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.0, "j2": 0.0}])
        mock_policy.provider_name = "test"

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(12, dtype=np.float32), {})
        mock_env.step.return_value = (
            np.zeros(12, dtype=np.float32),
            1.0,
            True,
            False,
            {"is_success": True},
        )
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = np.full(6, -1.0)
        mock_env.action_space.high = np.full(6, 1.0)
        mock_env.action_space.sample.return_value = np.zeros(6)
        mock_env._config = MagicMock()
        mock_env._config.solver = "mujoco"

        mock_newton_gym = MagicMock(return_value=mock_env)
        mock_newton_config = MagicMock()

        with patch("strands_robots.newton.newton_gym_env.NewtonGymEnv", mock_newton_gym):
            with patch("strands_robots.newton.NewtonConfig", mock_newton_config):
                result = evaluate(
                    mock_policy,
                    "pick cube",
                    "so100",
                    num_episodes=2,
                    backend="newton",
                )

        assert result["success_rate"] == 100.0
        assert result["num_episodes"] == 2

    def test_evaluate_mujoco_still_works(self):
        """evaluate(backend='mujoco') continues to work as before."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.0}])
        mock_policy.provider_name = "mock"

        mock_env = MagicMock()
        mock_env.reset.return_value = ({}, {})
        mock_env.step.return_value = ({}, 0.0, True, False, {"is_success": False})
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env):
            result = evaluate(mock_policy, "pick", "so100", num_episodes=1, backend="mujoco")

        assert result["num_episodes"] == 1
        assert "error" not in result


# ───────────────────────────── CosmosTrainer Script Path ─────────────────


class TestCosmosTrainerScriptPath:
    """Test CosmosTrainer's new train_script_path resolution."""

    def test_default_train_script_path_eagerly_resolved(self):
        """By default, train_script_path is eagerly resolved from filesystem."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        # On systems with cosmos-predict2 installed, path is resolved;
        # on systems without it, path is None.
        # The key behavioral change: it is NOT always None anymore.
        if t.train_script_path is not None:
            import os

            assert os.path.isfile(t.train_script_path)

    def test_explicit_train_script_path(self):
        """Explicit train_script_path is resolved; nonexistent falls through."""
        import os
        import tempfile

        from strands_robots.training import CosmosTrainer

        # Create a real temp file to test explicit path
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test")
            real_path = f.name
        try:
            t = CosmosTrainer(train_script_path=real_path)
            assert t.train_script_path == real_path
        finally:
            os.unlink(real_path)

    def test_train_uses_script_path_when_file_exists(self):
        """When train_script_path file exists, it's used instead of -m."""
        import os
        import tempfile

        from strands_robots.training import CosmosTrainer

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            script_path = f.name
            f.write(b"# fake train script")

        try:
            t = CosmosTrainer(train_script_path=script_path)
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                t.train()

            cmd = mock_run.call_args[0][0]
            assert script_path in cmd
            assert "-m" not in cmd
        finally:
            os.unlink(script_path)

    def test_train_uses_module_when_no_script_found(self):
        """When no script path resolves, falls back to -m scripts.train."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        # Force train_script_path to None to simulate no script found
        t.train_script_path = None

        mock_result = MagicMock(returncode=0)
        with patch.object(CosmosTrainer, "_resolve_train_script", return_value=None):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                t.train()

        cmd = mock_run.call_args[0][0]
        assert "-m" in cmd
        assert "scripts.train" in cmd

    def test_train_auto_detects_cosmos_path(self):
        """Auto-detection finds ~/cosmos-predict2.5/scripts/train.py."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        t.train_script_path = None  # Reset for test

        mock_result = MagicMock(returncode=0)

        def mock_isfile(path):
            return "cosmos-predict2.5/scripts/train.py" in str(path)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.isfile", side_effect=mock_isfile):
                t.train()

        cmd = mock_run.call_args[0][0]
        assert any("cosmos-predict2.5/scripts/train.py" in str(c) for c in cmd)
        assert "-m" not in cmd

    def test_train_env_var_cosmos_path(self):
        """COSMOS_PREDICT2_PATH env var is checked first."""
        import os

        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        t.train_script_path = None  # Reset for test
        mock_result = MagicMock(returncode=0)

        def mock_isfile(path):
            return "/custom/cosmos/scripts/train.py" in str(path)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch.dict(os.environ, {"COSMOS_PREDICT2_PATH": "/custom/cosmos"}):
                with patch("os.path.isfile", side_effect=mock_isfile):
                    t.train()

        cmd = mock_run.call_args[0][0]
        assert any("/custom/cosmos/scripts/train.py" in str(c) for c in cmd)

    def test_multi_gpu_with_script_path(self):
        """Multi-GPU torchrun uses script path too."""
        import os
        import tempfile

        from strands_robots.training import CosmosTrainer, TrainConfig

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            script_path = f.name
            f.write(b"# fake")

        try:
            t = CosmosTrainer(
                config=TrainConfig(num_gpus=4),
                train_script_path=script_path,
            )
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                t.train()

            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "torchrun"
            assert "--nproc_per_node=4" in cmd
            assert script_path in cmd
            assert "-m" not in cmd
        finally:
            os.unlink(script_path)


# ───────────────────────────── RL Trainer Newton Integration ─────────────────


class TestRLTrainerNewton:
    """Test that rl_trainer.py uses NewtonGymEnv."""

    def test_create_newton_env_tries_import(self):
        """_create_newton_env() falls back to MuJoCo when Newton import fails.

        Bug fix: On systems where Newton IS installed (e.g. Thor GPU),
        we must intercept builtins.__import__ to simulate the ImportError.
        Simply patching the class attribute doesn't prevent the real
        ``from strands_robots.newton.newton_gym_env import NewtonGymEnv``
        from succeeding.
        """
        from strands_robots.rl_trainer import RLConfig, SB3Trainer

        config = RLConfig(backend="newton", num_envs=4)
        trainer = SB3Trainer(config)

        # Block the Newton import so _create_newton_env hits the
        # ``except ImportError`` branch and falls back to MuJoCo.
        fake_import = _make_import_raiser("strands_robots.newton")
        with patch("builtins.__import__", side_effect=fake_import):
            with patch("strands_robots.rl_trainer.SB3Trainer._create_mujoco_env") as mock_mujoco:
                mock_mujoco.return_value = MagicMock()
                trainer._create_newton_env()
                mock_mujoco.assert_called_once()


# ───────────────────────────── repr and properties ─────────────────────────────


class TestNewtonGymEnvProperties:
    """Test properties and repr."""

    def test_repr(self):
        """__repr__ contains useful info."""
        env = _create_env(n_joints=6, num_envs=1)
        repr_str = repr(env)
        assert "so100" in repr_str
        assert "num_envs=1" in repr_str

    def test_unwrapped_backend(self):
        """unwrapped_backend provides access to NewtonBackend."""
        env = _create_env(n_joints=6, num_envs=1)
        assert env.unwrapped_backend is env._mock_backend

    def test_render_rgb_array(self):
        """render() calls backend.render() when mode is rgb_array."""
        env = _create_env(n_joints=6, num_envs=1, render_mode="rgb_array")
        env._backend.render.return_value = {
            "success": True,
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        image = env.render()
        assert image is not None
        env._backend.render.assert_called_once()

    def test_render_none_mode(self):
        """render() returns None when mode is None."""
        env = _create_env(n_joints=6, num_envs=1, render_mode=None)
        assert env.render() is None
