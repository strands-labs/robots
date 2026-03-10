"""Tests for strands_robots.envs — Gymnasium environment wrapper.

Tests the StrandsSimEnv class, which wraps MuJoCo Simulation as a standard
gymnasium.Env for RL training integration (LeRobot, SB3, etc.).

Strategy: Neither gymnasium nor mujoco are guaranteed on CI. We mock both:
  1. gymnasium — mocked at module level using types.ModuleType (not MagicMock,
     which causes MRO errors when used as a base class for inheritance)
  2. mujoco — mocked per-test using unittest.mock.patch

Uses the 4-step sys.modules injection pattern from test_newton_gym_env.py:
  1. Create realistic mock gymnasium modules with types.ModuleType
  2. Inject into sys.modules["gymnasium"] and sys.modules["gymnasium.spaces"]
  3. Delete stale cache from BOTH sys.modules AND parent package __dict__
  4. Re-import the target module (forces top-level code re-execution)
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


class _MockDict:
    """Minimal gymnasium.spaces.Dict replacement for tests."""

    def __init__(self, spaces_dict=None):
        self.spaces = spaces_dict or {}

    def __getitem__(self, key):
        return self.spaces[key]


class _MockEnv:
    """Minimal gymnasium.Env replacement for tests.

    Must be a real class (not MagicMock) so StrandsSimEnv can inherit from it
    without MRO errors.
    """

    metadata = {}
    observation_space = None
    action_space = None
    render_mode = None
    np_random = None

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
    mock_spaces.Dict = _MockDict

    mock_gym.spaces = mock_spaces

    # gymnasium.register — no-op
    mock_gym.register = lambda **kwargs: None

    sys.modules["gymnasium"] = mock_gym
    sys.modules["gymnasium.spaces"] = mock_spaces


def _force_reimport_envs():
    """Remove cached envs module from all import caches and reimport.

    Python caches submodules in two places:
      1. sys.modules["strands_robots.envs"]
      2. strands_robots.__dict__["envs"] (parent package attribute)

    We must clean BOTH to force a fresh import with our mocked gymnasium.
    """
    mod_key = "strands_robots.envs"

    # Step 1: Remove from sys.modules
    if mod_key in sys.modules:
        del sys.modules[mod_key]

    # Step 2: Remove cached attribute from parent package
    parent_key = "strands_robots"
    if parent_key in sys.modules:
        parent_pkg = sys.modules[parent_key]
        for attr in ("envs", "StrandsSimEnv"):
            if hasattr(parent_pkg, attr):
                try:
                    delattr(parent_pkg, attr)
                except AttributeError:
                    pass

    # Step 3: Re-import (forces top-level code re-execution with mock gymnasium)
    from strands_robots import envs as envs_mod

    assert envs_mod.HAS_GYM is True, (
        "Mock gymnasium injection failed — HAS_GYM is still False. "
        "This means sys.modules caching prevented the reimport."
    )

    # Step 4: Restore StrandsSimEnv on parent package so that
    # 'from strands_robots import StrandsSimEnv' works in other test files.
    # The strands_robots.__init__.py try/except already ran before our reimport,
    # so we need to manually set the attribute.
    if parent_key in sys.modules:
        parent_pkg = sys.modules[parent_key]
        parent_pkg.StrandsSimEnv = envs_mod.StrandsSimEnv

    return envs_mod


# Execute mock injection at module load time (before pytest collects tests)
_install_mock_gymnasium()
_envs_mod = _force_reimport_envs()
StrandsSimEnv = _envs_mod.StrandsSimEnv


# ───────────────────────────── Check module state ─────────────────────────

try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mujoco_module():
    """Provide a mock mujoco module for patching into strands_robots.envs."""
    mock_mj = MagicMock()
    mock_mj.mj_step = MagicMock()

    # Mock Renderer
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_mj.Renderer = MagicMock(return_value=mock_renderer)

    # Mock viewer
    mock_viewer = MagicMock()
    mock_mj.viewer = MagicMock()
    mock_mj.viewer.launch_passive = MagicMock(return_value=mock_viewer)

    return mock_mj


@pytest.fixture
def mock_simulation():
    """Provide a mock Simulation with realistic model/data."""
    mock_model = MagicMock()
    mock_model.nq = 6
    mock_model.nv = 6
    mock_model.nu = 6
    mock_model.actuator_ctrllimited = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    mock_model.actuator_ctrlrange = np.array(
        [
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
        ],
        dtype=np.float64,
    )

    mock_data = MagicMock()
    mock_data.qpos = np.zeros(6, dtype=np.float64)
    mock_data.qvel = np.zeros(6, dtype=np.float64)
    mock_data.ctrl = np.zeros(6, dtype=np.float64)

    mock_world = MagicMock()
    mock_world._model = mock_model
    mock_world._data = mock_data

    mock_sim_instance = MagicMock()
    mock_sim_instance._world = mock_world
    mock_sim_instance._dispatch_action = MagicMock(return_value={"status": "success"})

    return mock_sim_instance


@pytest.fixture
def mock_sim_module(mock_simulation):
    """Create a mock strands_robots.simulation module for sys.modules injection.

    The _init_sim() method does 'from strands_robots.simulation import Simulation'.
    On CI without strands (the agent SDK), strands_robots.simulation can't be imported.
    This fixture creates a mock module in sys.modules so the import resolves.
    """
    mock_mod = types.ModuleType("strands_robots.simulation")
    mock_mod.Simulation = MagicMock(return_value=mock_simulation)
    return mock_mod


def _make_env(
    mock_simulation,
    include_pixels=False,
    render_mode="rgb_array",
    max_steps=1000,
    n_substeps=10,
    reward_fn=None,
    success_fn=None,
):
    """Create a StrandsSimEnv instance without calling __init__."""
    env = StrandsSimEnv.__new__(StrandsSimEnv)
    env._sim = mock_simulation
    env._include_pixels = include_pixels
    env._step_count = 0
    env._initialized = True
    env.robot_name = "so100"
    env.data_config = "so100"
    env.task = "test task"
    env.render_mode = render_mode
    env.render_width = 640
    env.render_height = 480
    env.max_episode_steps = max_steps
    env.physics_dt = 0.002
    env.control_dt = 0.02
    env.n_substeps = n_substeps
    env.objects_config = []
    env.cameras_config = []
    env.reward_fn = reward_fn
    env.success_fn = success_fn
    env.np_random = MagicMock()
    return env


# ---------------------------------------------------------------------------
# Module-level flag tests
# ---------------------------------------------------------------------------


class TestModuleFlags:
    """Test module-level HAS_GYM / HAS_MUJOCO detection."""

    def test_has_gym_flag_true(self):
        """HAS_GYM should be True (real or mocked gymnasium)."""
        assert _envs_mod.HAS_GYM is True

    def test_has_mujoco_flag(self):
        """HAS_MUJOCO should reflect mujoco availability."""
        assert _envs_mod.HAS_MUJOCO == HAS_MUJOCO

    def test_strands_sim_env_is_not_stub(self):
        """StrandsSimEnv should be the real class, not the ImportError stub."""
        # The stub raises ImportError on __init__, the real one has _init_sim
        assert hasattr(StrandsSimEnv, "_init_sim")

    def test_metadata_render_modes(self):
        """Class metadata should include both render modes."""
        assert "rgb_array" in StrandsSimEnv.metadata["render_modes"]
        assert "human" in StrandsSimEnv.metadata["render_modes"]

    def test_metadata_render_fps(self):
        """Class metadata should include render_fps=30."""
        assert StrandsSimEnv.metadata["render_fps"] == 30


# ---------------------------------------------------------------------------
# Stub tests (when mujoco is not available)
# ---------------------------------------------------------------------------


class TestStubBehavior:
    """Test that __init__ requires mujoco."""

    @pytest.mark.skipif(HAS_MUJOCO, reason="mujoco is installed")
    def test_init_asserts_mujoco(self):
        """__init__ should assert HAS_MUJOCO and fail without mujoco."""
        with pytest.raises(AssertionError, match="mujoco required"):
            StrandsSimEnv(robot_name="so100")


# ---------------------------------------------------------------------------
# Constructor attribute tests (via __new__)
# ---------------------------------------------------------------------------


class TestConstructorAttributes:
    """Test that constructor params are stored correctly."""

    def test_default_attributes(self, mock_simulation):
        """Default params should match expected values."""
        env = _make_env(mock_simulation)
        assert env.robot_name == "so100"
        assert env.task == "test task"
        assert env.render_mode == "rgb_array"
        assert env.render_width == 640
        assert env.render_height == 480
        assert env.max_episode_steps == 1000
        assert env.n_substeps == 10
        assert env.reward_fn is None
        assert env.success_fn is None

    def test_custom_attributes(self, mock_simulation):
        """Custom params should be stored correctly."""

        def reward_fn(obs, act):
            return 1.0

        def success_fn(obs):
            return False

        env = _make_env(
            mock_simulation,
            max_steps=500,
            n_substeps=5,
            reward_fn=reward_fn,
            success_fn=success_fn,
            render_mode="human",
        )
        assert env.max_episode_steps == 500
        assert env.n_substeps == 5
        assert env.reward_fn is reward_fn
        assert env.success_fn is success_fn
        assert env.render_mode == "human"

    def test_n_substeps_calculation(self):
        """n_substeps = max(1, int(control_dt / physics_dt))."""
        assert max(1, int(0.02 / 0.002)) == 10
        assert max(1, int(0.002 / 0.002)) == 1
        assert max(1, int(0.001 / 0.01)) == 1  # clamps to 1


# ---------------------------------------------------------------------------
# _init_sim tests
# ---------------------------------------------------------------------------


class TestInitSim:
    """Test _init_sim — world/robot/object/camera creation and space setup."""

    def test_init_sim_creates_world_and_robot(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """_init_sim should dispatch create_world and add_robot."""
        env = _make_env(mock_simulation)
        env._sim = None  # Force re-init

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        calls = [c[0] for c in mock_simulation._dispatch_action.call_args_list]
        assert ("create_world", {}) in calls
        assert any(c[0] == "add_robot" for c in calls)

    def test_init_sim_adds_objects(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """_init_sim should dispatch add_object for each object config."""
        env = _make_env(mock_simulation)
        env._sim = None
        env.objects_config = [{"name": "cube"}, {"name": "ball"}]

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        action_names = [c[0][0] for c in mock_simulation._dispatch_action.call_args_list]
        assert action_names.count("add_object") == 2

    def test_init_sim_adds_cameras(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """_init_sim should dispatch add_camera for each camera config."""
        env = _make_env(mock_simulation)
        env._sim = None
        env.cameras_config = [{"name": "front"}]

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        action_names = [c[0][0] for c in mock_simulation._dispatch_action.call_args_list]
        assert action_names.count("add_camera") == 1

    def test_init_sim_sets_action_space(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """_init_sim should create action_space from actuator limits."""
        env = _make_env(mock_simulation)
        env._sim = None

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env.action_space is not None

    def test_init_sim_sets_observation_space_with_pixels(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """With rgb_array, observation_space should be Dict (state + pixels)."""
        env = _make_env(mock_simulation, render_mode="rgb_array")
        env._sim = None

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env._include_pixels is True
        assert env.observation_space is not None

    def test_init_sim_sets_observation_space_without_pixels(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """With human mode + no cameras, observation_space should be Box."""
        env = _make_env(mock_simulation, render_mode="human")
        env._sim = None
        env.cameras_config = []

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env._include_pixels is False

    def test_init_sim_cameras_enable_pixels(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """Even in human mode, cameras_config should enable pixels."""
        env = _make_env(mock_simulation, render_mode="human")
        env._sim = None
        env.cameras_config = [{"name": "front"}]

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env._include_pixels is True

    def test_init_sim_no_ctrl_limits(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """Without actuator limits, action bounds default to [-1, 1]."""
        mock_simulation._world._model.actuator_ctrllimited = np.zeros(6, dtype=np.int32)
        env = _make_env(mock_simulation)
        env._sim = None

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env.action_space is not None

    def test_init_sim_sets_initialized(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """_init_sim should set _initialized to True."""
        env = _make_env(mock_simulation)
        env._sim = None
        env._initialized = False

        with (
            patch.dict(sys.modules, {"strands_robots.simulation": mock_sim_module}),
            patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True),
            patch.object(_envs_mod, "HAS_MUJOCO", True),
        ):
            env._init_sim()

        assert env._initialized is True


# ---------------------------------------------------------------------------
# _get_obs tests
# ---------------------------------------------------------------------------


class TestGetObs:
    """Test observation retrieval from simulation state."""

    def test_state_only_obs(self, mock_simulation):
        """Without pixels, obs = concatenated qpos + qvel as float32."""
        mock_simulation._world._data.qpos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mock_simulation._world._data.qvel = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        env = _make_env(mock_simulation, include_pixels=False)
        obs = env._get_obs()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert obs.dtype == np.float32
        np.testing.assert_allclose(obs[:6], [1, 2, 3, 4, 5, 6], atol=1e-6)
        np.testing.assert_allclose(obs[6:], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)

    def test_obs_with_pixels(self, mock_simulation, mock_mujoco_module):
        """With pixels, obs should be dict with 'state' and 'pixels' keys."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation, include_pixels=True)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            obs = env._get_obs()

        assert isinstance(obs, dict)
        assert "state" in obs
        assert "pixels" in obs
        assert obs["state"].shape == (12,)
        assert obs["pixels"].shape == (480, 640, 3)
        assert obs["pixels"].dtype == np.uint8

    def test_renderer_closed_after_use(self, mock_simulation, mock_mujoco_module):
        """Renderer should be garbage-collected (del) after each _get_obs call.

        mujoco 3.x Renderer has no close() — it's garbage-collected via del.
        We verify the renderer was created and used, then discarded.
        """
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco_module.Renderer.return_value = mock_renderer

        env = _make_env(mock_simulation, include_pixels=True)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            env._get_obs()

        # mujoco 3.x uses 'del renderer' not .close() — verify it was created and used
        mock_mujoco_module.Renderer.assert_called_once()
        mock_renderer.render.assert_called_once()

    def test_obs_float32_dtype(self, mock_simulation):
        """State observation should always be float32."""
        mock_simulation._world._data.qpos = np.zeros(6, dtype=np.float64)
        mock_simulation._world._data.qvel = np.zeros(6, dtype=np.float64)

        env = _make_env(mock_simulation, include_pixels=False)
        obs = env._get_obs()

        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# reset tests
# ---------------------------------------------------------------------------


class TestReset:
    """Test environment reset behavior."""

    def test_reset_returns_obs_and_info(self, mock_simulation):
        """reset() should return (obs, info) tuple."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(info, dict)
        assert info["task"] == "test task"

    def test_reset_clears_step_count(self, mock_simulation):
        """reset() should set _step_count to 0."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        env._step_count = 999
        env.reset()

        assert env._step_count == 0

    def test_reset_dispatches_reset_action(self, mock_simulation):
        """reset() should dispatch 'reset' to the simulation."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        env.reset()

        mock_simulation._dispatch_action.assert_called_with("reset", {})

    def test_reset_with_no_sim_calls_init(self, mock_simulation):
        """reset() with _sim=None should call _init_sim."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        env._sim = None

        with patch.object(env, "_init_sim") as mock_init:

            def set_sim():
                env._sim = mock_simulation

            mock_init.side_effect = set_sim
            env.reset()
            mock_init.assert_called_once()

    def test_reset_with_seed(self, mock_simulation):
        """reset(seed=42) should not raise."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        obs, info = env.reset(seed=42)
        assert obs is not None


# ---------------------------------------------------------------------------
# step tests
# ---------------------------------------------------------------------------


class TestStep:
    """Test step() — physics, reward, termination, truncation."""

    def test_step_returns_five_tuple(self, mock_simulation, mock_mujoco_module):
        """step() should return (obs, reward, terminated, truncated, info)."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            result = env.step(np.zeros(6, dtype=np.float32))

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_counter(self, mock_simulation, mock_mujoco_module):
        """Each step() increments _step_count."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            env.step(np.zeros(6, dtype=np.float32))
            assert env._step_count == 1
            env.step(np.zeros(6, dtype=np.float32))
            assert env._step_count == 2

    def test_step_applies_action_to_ctrl(self, mock_simulation, mock_mujoco_module):
        """step() copies action values into data.ctrl."""
        ctrl_array = np.zeros(6, dtype=np.float64)
        mock_simulation._world._data.ctrl = ctrl_array
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)

        env = _make_env(mock_simulation)
        action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            env.step(action)

        np.testing.assert_allclose(ctrl_array, action, atol=1e-6)

    def test_step_calls_mj_step_n_substeps_times(self, mock_simulation, mock_mujoco_module):
        """Physics should be stepped n_substeps times per step()."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, n_substeps=5)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            env.step(np.zeros(6, dtype=np.float32))

        assert mock_mujoco_module.mj_step.call_count == 5

    def test_step_default_reward_zero(self, mock_simulation, mock_mujoco_module):
        """Without reward_fn, reward=0.0."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, reward_fn=None)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert reward == 0.0

    def test_step_custom_reward_fn(self, mock_simulation, mock_mujoco_module):
        """Custom reward_fn return value should be used."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, reward_fn=lambda obs, act: 42.5)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert reward == 42.5

    def test_step_no_termination_without_success_fn(self, mock_simulation, mock_mujoco_module):
        """Without success_fn, terminated=False."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, success_fn=None)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert terminated is False

    def test_step_success_fn_true(self, mock_simulation, mock_mujoco_module):
        """success_fn returning True → terminated=True."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, success_fn=lambda obs: True)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert terminated is True

    def test_step_success_fn_false(self, mock_simulation, mock_mujoco_module):
        """success_fn returning False → terminated=False."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, success_fn=lambda obs: False)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert terminated is False

    def test_step_truncation_at_max_steps(self, mock_simulation, mock_mujoco_module):
        """truncated=True when _step_count reaches max_episode_steps."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, max_steps=3)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert truncated is False  # step 1

            _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert truncated is False  # step 2

            _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert truncated is True  # step 3 = max

    def test_step_info_keys(self, mock_simulation, mock_mujoco_module):
        """Info dict should have step, task, is_success."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))

        assert info["step"] == 1
        assert info["task"] == "test task"
        assert "is_success" in info

    def test_step_is_success_matches_terminated(self, mock_simulation, mock_mujoco_module):
        """info['is_success'] should equal terminated."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, success_fn=lambda obs: True)

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))

        assert info["is_success"] == terminated


# ---------------------------------------------------------------------------
# render tests
# ---------------------------------------------------------------------------


class TestRender:
    """Test render() for different modes."""

    def test_render_rgb_array(self, mock_simulation, mock_mujoco_module):
        """rgb_array render returns numpy array."""
        env = _make_env(mock_simulation, render_mode="rgb_array")

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco_module.Renderer.return_value = mock_renderer

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            frame = env.render()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        # mujoco 3.x uses 'del renderer' not .close() — verify it was created and used
        mock_mujoco_module.Renderer.assert_called_once()
        mock_renderer.render.assert_called_once()

    def test_render_human_creates_viewer(self, mock_simulation, mock_mujoco_module):
        """Human render creates passive viewer."""
        env = _make_env(mock_simulation, render_mode="human")

        mock_viewer = MagicMock()
        mock_mujoco_module.viewer.launch_passive.return_value = mock_viewer

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            result = env.render()

        assert result is None
        mock_mujoco_module.viewer.launch_passive.assert_called_once()
        mock_viewer.sync.assert_called_once()

    def test_render_human_reuses_viewer(self, mock_simulation, mock_mujoco_module):
        """Second human render reuses existing viewer."""
        env = _make_env(mock_simulation, render_mode="human")

        mock_viewer = MagicMock()
        mock_mujoco_module.viewer.launch_passive.return_value = mock_viewer

        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            env.render()
            env.render()

        assert mock_mujoco_module.viewer.launch_passive.call_count == 1
        assert mock_viewer.sync.call_count == 2


# ---------------------------------------------------------------------------
# close tests
# ---------------------------------------------------------------------------


class TestClose:
    """Test resource cleanup."""

    def test_close_destroys_sim(self, mock_simulation):
        """close() dispatches 'destroy' and nullifies _sim."""
        env = _make_env(mock_simulation)
        env.close()

        mock_simulation._dispatch_action.assert_called_with("destroy", {})
        assert env._sim is None

    def test_close_closes_viewer(self, mock_simulation):
        """close() closes viewer if it exists."""
        mock_viewer = MagicMock()

        env = _make_env(mock_simulation)
        env._viewer = mock_viewer
        env.close()

        mock_viewer.close.assert_called_once()
        assert env._viewer is None

    def test_close_without_viewer(self, mock_simulation):
        """close() without viewer does not raise."""
        env = _make_env(mock_simulation)
        # No _viewer attribute set
        env.close()
        assert env._sim is None

    def test_close_with_none_sim(self):
        """close() with _sim=None does not raise."""
        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._sim = None
        env.close()  # should not raise


# ---------------------------------------------------------------------------
# unwrapped_sim property
# ---------------------------------------------------------------------------


class TestUnwrappedSim:
    """Test the unwrapped_sim property."""

    def test_returns_sim(self, mock_simulation):
        """unwrapped_sim returns the underlying Simulation."""
        env = _make_env(mock_simulation)
        assert env.unwrapped_sim is mock_simulation

    def test_returns_none_before_init(self):
        """Before init, unwrapped_sim is None."""
        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._sim = None
        assert env.unwrapped_sim is None


# ---------------------------------------------------------------------------
# Integration: full episode loop
# ---------------------------------------------------------------------------


class TestEpisodeLoop:
    """Test complete episode lifecycle."""

    def test_full_episode_to_truncation(self, mock_simulation, mock_mujoco_module):
        """Run reset → step until truncation → close."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, max_steps=5, reward_fn=lambda obs, act: 1.0)

        obs, info = env.reset()
        assert env._step_count == 0

        total_reward = 0.0
        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
                total_reward += reward
                if truncated or terminated:
                    break

        assert truncated is True
        assert env._step_count == 5
        assert total_reward == 5.0

        env.close()
        assert env._sim is None

    def test_early_termination(self, mock_simulation, mock_mujoco_module):
        """Episode ends when success_fn returns True."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        call_count = [0]

        def success_after_3(obs):
            call_count[0] += 1
            return call_count[0] >= 3

        env = _make_env(mock_simulation, max_steps=100, n_substeps=1, success_fn=success_after_3)

        steps_taken = 0
        with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
            for _ in range(100):
                _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
                steps_taken += 1
                if terminated or truncated:
                    break

        assert terminated is True
        assert truncated is False
        assert steps_taken == 3
        assert info["is_success"] is True

    def test_multiple_episodes(self, mock_simulation, mock_mujoco_module):
        """Multiple reset→step cycles should work."""
        mock_simulation._world._data.qpos = np.zeros(6)
        mock_simulation._world._data.qvel = np.zeros(6)
        mock_simulation._world._data.ctrl = np.zeros(6)

        env = _make_env(mock_simulation, max_steps=3)

        for episode in range(3):
            env.reset()
            assert env._step_count == 0

            with patch.object(_envs_mod, "mujoco", mock_mujoco_module, create=True):
                for _ in range(10):
                    _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
                    if truncated:
                        break

            assert env._step_count == 3


# ---------------------------------------------------------------------------
# Backend delegation tests (isaac / newton / mujoco)
# ---------------------------------------------------------------------------


class TestBackendDelegation:
    """Test StrandsSimEnv backend delegation to Isaac and Newton gym envs."""

    def test_isaac_backend_delegates_to_isaac_gym_env(self):
        """backend='isaac' should create IsaacGymEnv delegate with correct args."""
        mock_isaac_env = MagicMock()
        mock_isaac_env.observation_space = MagicMock()
        mock_isaac_env.action_space = MagicMock()

        mock_isaac_mod = types.ModuleType("strands_robots.isaac.isaac_gym_env")
        mock_isaac_mod.IsaacGymEnv = MagicMock(return_value=mock_isaac_env)

        with patch.dict(
            sys.modules,
            {
                "strands_robots.isaac": types.ModuleType("strands_robots.isaac"),
                "strands_robots.isaac.isaac_gym_env": mock_isaac_mod,
            },
        ):
            env = StrandsSimEnv.__new__(StrandsSimEnv)
            env.__init__(
                robot_name="so100",
                task="test",
                backend="isaac",
                num_envs=16,
                device="cuda:1",
                render_mode="rgb_array",
                max_episode_steps=500,
                physics_dt=0.002,
            )

        # Verify IsaacGymEnv was called with num_envs and device
        mock_isaac_mod.IsaacGymEnv.assert_called_once()
        call_kwargs = mock_isaac_mod.IsaacGymEnv.call_args[1]
        assert call_kwargs["robot_name"] == "so100"
        assert call_kwargs["num_envs"] == 16
        assert call_kwargs["device"] == "cuda:1"
        assert call_kwargs["task"] == "test"
        assert call_kwargs["max_episode_steps"] == 500

        # Verify delegation
        assert env._delegate is mock_isaac_env
        assert env._backend == "isaac"
        assert env.observation_space is mock_isaac_env.observation_space
        assert env.action_space is mock_isaac_env.action_space

    def test_newton_backend_delegates_with_newton_config(self):
        """backend='newton' should create NewtonGymEnv with NewtonConfig containing num_envs/device."""
        mock_newton_env = MagicMock()
        mock_newton_env.observation_space = MagicMock()
        mock_newton_env.action_space = MagicMock()

        mock_newton_gym_mod = types.ModuleType("strands_robots.newton.newton_gym_env")
        mock_newton_gym_mod.NewtonGymEnv = MagicMock(return_value=mock_newton_env)

        # Create a real-ish NewtonConfig mock that captures init args
        captured_config = {}

        class FakeNewtonConfig:
            def __init__(self, **kwargs):
                captured_config.update(kwargs)
                self.num_envs = kwargs.get("num_envs", 1)
                self.device = kwargs.get("device", "cuda:0")
                self.solver = "mujoco"
                self.physics_dt = 1.0 / 200.0
                self.substeps = 1
                self.render_backend = "opengl"
                self.enable_cuda_graph = False
                self.enable_differentiable = False
                self.broad_phase = "explicit"
                self.soft_contact_margin = 0.5
                self.soft_contact_ke = 1.0e4
                self.soft_contact_kd = 1.0e1
                self.soft_contact_mu = 0.5
                self.soft_contact_restitution = 0.0

        mock_newton_backend_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_newton_backend_mod.NewtonConfig = FakeNewtonConfig

        mock_newton_init = types.ModuleType("strands_robots.newton")
        mock_newton_init.NewtonConfig = FakeNewtonConfig

        with patch.dict(
            sys.modules,
            {
                "strands_robots.newton": mock_newton_init,
                "strands_robots.newton.newton_gym_env": mock_newton_gym_mod,
                "strands_robots.newton.newton_backend": mock_newton_backend_mod,
            },
        ):
            env = StrandsSimEnv.__new__(StrandsSimEnv)
            env.__init__(
                robot_name="g1",
                task="locomotion",
                backend="newton",
                num_envs=4096,
                device="cuda:0",
                render_mode="rgb_array",
                max_episode_steps=1000,
            )

        # Verify NewtonGymEnv was called
        mock_newton_gym_mod.NewtonGymEnv.assert_called_once()
        call_kwargs = mock_newton_gym_mod.NewtonGymEnv.call_args[1]

        # Critical: num_envs and device must be in the NewtonConfig
        assert call_kwargs["robot_name"] == "g1"
        assert call_kwargs["task"] == "locomotion"
        assert call_kwargs["max_episode_steps"] == 1000
        assert "config" in call_kwargs
        config = call_kwargs["config"]
        assert config.num_envs == 4096, (
            f"Newton config num_envs={config.num_envs}, expected 4096. "
            "This is the bug from PR #106 — num_envs was silently dropped."
        )
        assert config.device == "cuda:0", (
            f"Newton config device={config.device}, expected cuda:0. "
            "This is the bug from PR #106 — device was silently dropped."
        )

        # Verify delegation
        assert env._delegate is mock_newton_env
        assert env._backend == "newton"
        assert env.observation_space is mock_newton_env.observation_space
        assert env.action_space is mock_newton_env.action_space

    def test_newton_default_num_envs_and_device(self):
        """Default num_envs=1 and device='cuda:0' should be passed to NewtonConfig."""
        mock_newton_env = MagicMock()
        mock_newton_env.observation_space = MagicMock()
        mock_newton_env.action_space = MagicMock()

        mock_newton_gym_mod = types.ModuleType("strands_robots.newton.newton_gym_env")
        mock_newton_gym_mod.NewtonGymEnv = MagicMock(return_value=mock_newton_env)

        captured_configs = []

        class FakeNewtonConfig:
            def __init__(self, **kwargs):
                self.num_envs = kwargs.get("num_envs", 1)
                self.device = kwargs.get("device", "cuda:0")
                captured_configs.append(self)

        mock_newton_backend_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_newton_backend_mod.NewtonConfig = FakeNewtonConfig

        mock_newton_init = types.ModuleType("strands_robots.newton")
        mock_newton_init.NewtonConfig = FakeNewtonConfig

        with patch.dict(
            sys.modules,
            {
                "strands_robots.newton": mock_newton_init,
                "strands_robots.newton.newton_gym_env": mock_newton_gym_mod,
                "strands_robots.newton.newton_backend": mock_newton_backend_mod,
            },
        ):
            env = StrandsSimEnv.__new__(StrandsSimEnv)
            # Use defaults (num_envs=1, device="cuda:0")
            env.__init__(robot_name="so100", backend="newton")

        assert len(captured_configs) == 1
        assert captured_configs[0].num_envs == 1
        assert captured_configs[0].device == "cuda:0"

    def test_mujoco_backend_no_delegate(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """backend='mujoco' (default) should NOT create a delegate."""
        env = _make_env(mock_simulation)
        assert env._sim is mock_simulation
        # _delegate should be absent or None for mujoco backend
        assert not hasattr(env, "_delegate") or env._delegate is None

    def test_delegate_step_forwarded(self):
        """step() on delegated env forwards to delegate.step()."""
        mock_delegate = MagicMock()
        mock_delegate.step.return_value = (np.zeros(12), 1.0, False, False, {"step": 1})

        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._delegate = mock_delegate

        action = np.zeros(6, dtype=np.float32)
        result = env.step(action)

        mock_delegate.step.assert_called_once_with(action)
        assert result[1] == 1.0  # reward

    def test_delegate_reset_forwarded(self):
        """reset() on delegated env forwards to delegate.reset()."""
        mock_delegate = MagicMock()
        mock_delegate.reset.return_value = (np.zeros(12), {"task": "test"})

        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._delegate = mock_delegate

        obs, info = env.reset(seed=42)

        mock_delegate.reset.assert_called_once_with(seed=42, options=None)

    def test_delegate_render_forwarded(self):
        """render() on delegated env forwards to delegate.render()."""
        mock_delegate = MagicMock()
        mock_delegate.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._delegate = mock_delegate

        frame = env.render()

        mock_delegate.render.assert_called_once()
        assert frame.shape == (480, 640, 3)

    def test_delegate_close_forwarded(self):
        """close() on delegated env forwards to delegate.close()."""
        mock_delegate = MagicMock()

        env = StrandsSimEnv.__new__(StrandsSimEnv)
        env._delegate = mock_delegate

        env.close()

        mock_delegate.close.assert_called_once()

    def test_invalid_backend_falls_through_to_mujoco(self, mock_simulation, mock_mujoco_module, mock_sim_module):
        """Unknown backend string falls through to MuJoCo path (no delegation)."""
        env = _make_env(mock_simulation)
        env._backend = "mujoco"
        # Env should function normally with MuJoCo
        assert env._sim is mock_simulation
