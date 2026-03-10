"""Tests for IsaacGymEnv — Gymnasium wrapper for Isaac Sim GPU backend.

Tests the IsaacGymEnv class, which wraps IsaacSimBackend as a standard
gymnasium.Env for RL training integration (LeRobot, SB3, etc.).

Strategy: We mock the IsaacSimBackend since Isaac Sim runtime is not
available on all CI machines.  The tests validate:
  1. Constructor stores parameters correctly
  2. _get_obs returns correct shapes (single and batched)
  3. step() returns 5-tuple with correct types
  4. reset() clears state
  5. close() destroys backend
  6. Full episode lifecycle
  7. Batched multi-env operation

All tests use numpy arrays for mock data (no torch dependency for obs).
The production _get_obs() has `hasattr(jpos, 'cpu')` — numpy arrays
take the `np.asarray()` path, so this works correctly without torch.

Device note: Tests use device="cpu" to avoid requiring an NVIDIA GPU in CI.
The entire Isaac backend is mocked, so the device only affects the
torch.tensor() call in step() — using CPU exercises the same code path
without triggering CUDA driver initialization.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import gymnasium as gym

    # Ensure this is real gymnasium, not a mock injected by other test files
    # (e.g., test_envs.py injects a types.ModuleType mock at module level)
    HAS_GYM = hasattr(gym, "__version__")
except ImportError:
    HAS_GYM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use CPU device for tests — the backend is fully mocked so the device
# only affects the torch.tensor() call in step().  Using "cpu" avoids
# CUDA driver initialization which is unavailable in CI.
_TEST_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_robot(n_joints=6, num_envs=1):
    """Create a mock Isaac Lab Articulation robot.

    Uses numpy arrays instead of torch tensors. The production _get_obs()
    checks ``hasattr(jpos, 'cpu')`` — numpy lacks .cpu(), so it takes the
    ``np.asarray()`` path. This matches the test_newton_gym_env.py pattern.
    """
    robot = MagicMock()
    robot.num_joints = n_joints
    robot.joint_names = [f"joint_{i}" for i in range(n_joints)]

    # Mock robot data — plain numpy, no torch needed
    robot.data = MagicMock()
    robot.data.joint_pos = np.zeros((num_envs, n_joints), dtype=np.float32)
    robot.data.joint_vel = np.zeros((num_envs, n_joints), dtype=np.float32)
    robot.data.default_joint_pos = np.zeros((num_envs, n_joints), dtype=np.float32)
    robot.data.default_joint_vel = np.zeros((num_envs, n_joints), dtype=np.float32)
    robot.data.root_pos_w = np.zeros((num_envs, 3), dtype=np.float32)
    robot.data.root_quat_w = np.zeros((num_envs, 4), dtype=np.float32)
    robot.data.net_contact_forces = np.zeros((num_envs, n_joints, 3), dtype=np.float32)

    return robot


def _make_mock_backend(num_envs=1, n_joints=6, device=_TEST_DEVICE):
    """Create a mock IsaacSimBackend with realistic behavior."""
    backend = MagicMock()
    backend._robot = _make_mock_robot(n_joints, num_envs)
    backend._sim = MagicMock()
    backend._envs_created = True

    from strands_robots.isaac.isaac_sim_backend import IsaacSimConfig

    backend.config = IsaacSimConfig(num_envs=num_envs, device=device)

    backend.create_world.return_value = {"status": "success", "content": [{"text": "ok"}]}
    backend.add_robot.return_value = {"status": "success", "content": [{"text": "ok"}]}
    backend.step.return_value = {"status": "success", "content": [{"text": "ok"}]}
    backend.reset.return_value = {"status": "success", "content": [{"text": "ok"}]}
    backend.destroy.return_value = {"status": "success", "content": [{"text": "ok"}]}
    backend.render.return_value = {"status": "success", "content": [{"text": "ok"}]}

    return backend


def _make_env(num_envs=1, n_joints=6, reward_fn=None, success_fn=None, render_mode=None, max_steps=1000):
    """Create an IsaacGymEnv instance with mocked backend."""
    from strands_robots.isaac.isaac_gym_env import IsaacGymEnv
    from strands_robots.isaac.isaac_sim_backend import IsaacSimConfig

    env = IsaacGymEnv.__new__(IsaacGymEnv)
    env._robot_name = "so100"
    env._task = "test task"
    env._num_envs = num_envs
    env._device = _TEST_DEVICE
    env.render_mode = render_mode
    env.max_episode_steps = max_steps
    env._usd_path = None
    env.reward_fn = reward_fn
    env.success_fn = success_fn
    env._n_joints = n_joints
    env._step_count = 0
    env._config = IsaacSimConfig(num_envs=num_envs, device=_TEST_DEVICE)

    env._backend = _make_mock_backend(num_envs, n_joints)

    # Set up spaces
    obs_dim = n_joints * 2
    if num_envs > 1:
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, obs_dim), dtype=np.float32)
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_envs, n_joints), dtype=np.float32)
    else:
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)

    env.np_random = MagicMock()
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestIsaacGymEnvImport:
    """Test module-level imports."""

    def test_import_succeeds(self):
        """IsaacGymEnv should be importable."""
        from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

        assert IsaacGymEnv is not None

    def test_is_gym_env_subclass(self):
        """IsaacGymEnv should be a subclass of gymnasium.Env."""
        from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

        assert issubclass(IsaacGymEnv, gym.Env)

    def test_metadata(self):
        """Class metadata should include render modes and fps."""
        from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

        assert "rgb_array" in IsaacGymEnv.metadata["render_modes"]
        assert IsaacGymEnv.metadata["render_fps"] == 30


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestConstructor:
    """Test constructor attribute storage."""

    def test_default_attributes(self):
        env = _make_env()
        assert env._robot_name == "so100"
        assert env._task == "test task"
        assert env._num_envs == 1
        assert env._device == _TEST_DEVICE
        assert env.max_episode_steps == 1000
        assert env.reward_fn is None
        assert env.success_fn is None

    def test_custom_num_envs(self):
        env = _make_env(num_envs=16)
        assert env._num_envs == 16
        assert env.observation_space.shape == (16, 12)
        assert env.action_space.shape == (16, 6)

    def test_single_env_spaces(self):
        env = _make_env(num_envs=1, n_joints=8)
        assert env.observation_space.shape == (16,)  # 8*2
        assert env.action_space.shape == (8,)


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestGetObs:
    """Test observation retrieval."""

    def test_single_env_obs_shape(self):
        env = _make_env(num_envs=1, n_joints=6)
        obs = env._get_obs()
        assert obs.shape == (12,)
        assert obs.dtype == np.float32

    def test_batched_obs_shape(self):
        env = _make_env(num_envs=4, n_joints=6)
        obs = env._get_obs()
        assert obs.shape == (4, 12)
        assert obs.dtype == np.float32

    def test_obs_with_nonzero_values(self):
        """Verify _get_obs reads actual joint values correctly."""
        env = _make_env(num_envs=1, n_joints=3)
        env._backend._robot.data.joint_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        env._backend._robot.data.joint_vel = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        obs = env._get_obs()
        np.testing.assert_allclose(obs[:3], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(obs[3:], [0.1, 0.2, 0.3], atol=1e-6)

    def test_obs_no_backend(self):
        env = _make_env(num_envs=1, n_joints=6)
        env._backend = None
        obs = env._get_obs()
        assert obs.shape == (12,)
        assert np.allclose(obs, 0.0)


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestReset:
    """Test reset behavior."""

    def test_reset_returns_obs_and_info(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        assert info["task"] == "test task"

    def test_reset_clears_step_count(self):
        env = _make_env()
        env._step_count = 42
        env.reset()
        assert env._step_count == 0

    def test_reset_calls_backend_reset(self):
        env = _make_env()
        env.reset()
        env._backend.reset.assert_called_once()


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestStep:
    """Test step() behavior."""

    def test_step_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        result = env.step(np.zeros(6, dtype=np.float32))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_counter(self):
        env = _make_env()
        env.reset()
        env.step(np.zeros(6, dtype=np.float32))
        assert env._step_count == 1
        env.step(np.zeros(6, dtype=np.float32))
        assert env._step_count == 2

    def test_step_default_reward_zero(self):
        env = _make_env(reward_fn=None)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert reward == 0.0

    def test_step_custom_reward(self):
        env = _make_env(reward_fn=lambda obs, act: 42.5)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert reward == 42.5

    def test_step_truncation_at_max(self):
        env = _make_env(max_steps=3)
        env.reset()
        for i in range(3):
            _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert truncated is True

    def test_step_no_truncation_before_max(self):
        env = _make_env(max_steps=100)
        env.reset()
        _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert truncated is False

    def test_step_success_fn(self):
        env = _make_env(success_fn=lambda obs: True)
        env.reset()
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is True
        assert info["is_success"] is True

    def test_step_no_success_fn(self):
        env = _make_env(success_fn=None)
        env.reset()
        _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is False


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestBatchedStep:
    """Test batched multi-env step."""

    def test_batched_step_returns_arrays(self):
        env = _make_env(num_envs=4)
        env.reset()
        action = np.zeros((4, 6), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4, 12)
        assert reward.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

    def test_batched_reward(self):
        env = _make_env(num_envs=4, reward_fn=lambda obs, act: 1.0)
        env.reset()
        action = np.zeros((4, 6), dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        np.testing.assert_allclose(reward, [1.0, 1.0, 1.0, 1.0])


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestClose:
    """Test resource cleanup."""

    def test_close_destroys_backend(self):
        env = _make_env()
        backend_ref = env._backend  # save ref before close nullifies it
        env.close()
        backend_ref.destroy.assert_called_once()
        assert env._backend is None

    def test_close_with_none_backend(self):
        env = _make_env()
        env._backend = None
        env.close()  # should not raise


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestProperties:
    """Test properties."""

    def test_num_envs(self):
        env = _make_env(num_envs=16)
        assert env.num_envs == 16

    def test_unwrapped_backend(self):
        env = _make_env()
        assert env.unwrapped_backend is env._backend

    def test_repr(self):
        env = _make_env()
        r = repr(env)
        assert "IsaacGymEnv" in r
        assert "so100" in r


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestFullEpisode:
    """Test complete episode lifecycle."""

    def test_full_episode_to_truncation(self):
        env = _make_env(max_steps=5, reward_fn=lambda obs, act: 1.0)
        obs, info = env.reset()

        total_reward = 0.0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            total_reward += reward
            if truncated or terminated:
                break

        assert truncated is True
        assert env._step_count == 5
        assert total_reward == 5.0
        env.close()

    def test_multiple_episodes(self):
        env = _make_env(max_steps=3)
        for _ in range(3):
            env.reset()
            assert env._step_count == 0
            for _ in range(10):
                _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
                if truncated:
                    break
            assert env._step_count == 3
        env.close()
