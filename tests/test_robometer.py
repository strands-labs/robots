"""Comprehensive tests for strands_robots/robometer.py.


Every heavy dependency (torch, robometer, gymnasium, scipy, transformers, peft)
is fully mocked so the suite runs in CI without GPU or any of those packages.

Strategy (same as test_envs.py / test_simulation_mock.py):
  1. Inject mock modules into sys.modules BEFORE import
  2. Force-reload the MUT (module under test)
  3. Use unittest.mock for all runtime interactions
  4. Pop mocks after MUT import; re-inject via autouse fixture to prevent
     cross-test pollution (e.g., test_record.py's pytest.importorskip("torch")
     finding our mock torch which lacks from_numpy).
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Module-level mocks – installed BEFORE importing the MUT
# ═══════════════════════════════════════════════════════════════════════════

# --- Mock torch ---
_torch = types.ModuleType("torch")
_torch.float32 = "torch.float32"
_torch.float16 = "torch.float16"
_torch.bfloat16 = "torch.bfloat16"
_torch.device = MagicMock
_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
_torch.Tensor = MagicMock
_torch.tensor = MagicMock
_torch.cuda = MagicMock()
_torch.cuda.is_available = MagicMock(return_value=False)
_torch_nn = types.ModuleType("torch.nn")
_torch.nn = _torch_nn
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", _torch_nn)

# --- Mock robometer (full hierarchy) ---
_robometer = types.ModuleType("robometer")
_robometer_utils = types.ModuleType("robometer.utils")
_robometer_utils_save = types.ModuleType("robometer.utils.save")
_robometer_utils_setup = types.ModuleType("robometer.utils.setup_utils")
_robometer_evals = types.ModuleType("robometer.evals")
_robometer_evals_utils = types.ModuleType("robometer.evals.eval_utils")
_robometer_evals_server = types.ModuleType("robometer.evals.eval_server")
_robometer_trainers = types.ModuleType("robometer.trainers")
_robometer_trainers_rbm = types.ModuleType("robometer.trainers.rbm_heads_trainer")
_robometer_data = types.ModuleType("robometer.data")
_robometer_data_types = types.ModuleType("robometer.data.dataset_types")
_robometer_data_datasets = types.ModuleType("robometer.data.datasets")
_robometer_data_datasets_helpers = types.ModuleType("robometer.data.datasets.helpers")
_robometer_configs = types.ModuleType("robometer.configs")
_robometer_configs_eval = types.ModuleType("robometer.configs.eval_configs")
_robometer_configs_exp = types.ModuleType("robometer.configs.experiment_configs")
_robometer_models = types.ModuleType("robometer.models")
_robometer_models_utils = types.ModuleType("robometer.models.utils")
_robometer_utils_logger = types.ModuleType("robometer.utils.logger")
_robometer_utils_config = types.ModuleType("robometer.utils.config_utils")

# Wire up key functions on their correct submodules
_robometer_utils_save.load_model_from_hf = MagicMock()
_robometer_utils_setup.setup_batch_collator = MagicMock(return_value=MagicMock())

_robometer_evals_utils.raw_dict_to_sample = MagicMock(return_value={"sample": "data"})
_robometer_evals_utils.extract_rewards_from_output = MagicMock(return_value=np.array([0.5]))
_robometer_evals_utils.extract_success_probs_from_output = MagicMock(return_value=np.array([0.3]))

_robometer_evals_server.process_batch_helper = MagicMock(return_value={"outputs_progress": {"progress_pred": [[0.5]]}})

_robometer_trainers_rbm.RBMHeadsTrainer = MagicMock()
_robometer_data_types.ProgressSample = MagicMock
_robometer_data_types.PreferenceSample = MagicMock
_robometer_data_types.Trajectory = MagicMock
_robometer_data_datasets_helpers.linspace_subsample_frames = MagicMock()
_robometer_data_datasets_helpers.pad_trajectory_to_max_frames_np = MagicMock()

# Parent-child links
_robometer.utils = _robometer_utils
_robometer.evals = _robometer_evals
_robometer.trainers = _robometer_trainers
_robometer.data = _robometer_data
_robometer.configs = _robometer_configs
_robometer.models = _robometer_models
_robometer_utils.save = _robometer_utils_save
_robometer_utils.setup_utils = _robometer_utils_setup
_robometer_utils.logger = _robometer_utils_logger
_robometer_utils.config_utils = _robometer_utils_config
_robometer_evals.eval_utils = _robometer_evals_utils
_robometer_evals.eval_server = _robometer_evals_server
_robometer_trainers.rbm_heads_trainer = _robometer_trainers_rbm
_robometer_data.dataset_types = _robometer_data_types
_robometer_data.datasets = _robometer_data_datasets
_robometer_data_datasets.helpers = _robometer_data_datasets_helpers
_robometer_configs.eval_configs = _robometer_configs_eval
_robometer_configs.experiment_configs = _robometer_configs_exp
_robometer_models.utils = _robometer_models_utils

_ROBOMETER_MODULES = {
    "robometer": _robometer,
    "robometer.utils": _robometer_utils,
    "robometer.utils.save": _robometer_utils_save,
    "robometer.utils.setup_utils": _robometer_utils_setup,
    "robometer.utils.logger": _robometer_utils_logger,
    "robometer.utils.config_utils": _robometer_utils_config,
    "robometer.evals": _robometer_evals,
    "robometer.evals.eval_utils": _robometer_evals_utils,
    "robometer.evals.eval_server": _robometer_evals_server,
    "robometer.trainers": _robometer_trainers,
    "robometer.trainers.rbm_heads_trainer": _robometer_trainers_rbm,
    "robometer.data": _robometer_data,
    "robometer.data.dataset_types": _robometer_data_types,
    "robometer.data.datasets": _robometer_data_datasets,
    "robometer.data.datasets.helpers": _robometer_data_datasets_helpers,
    "robometer.configs": _robometer_configs,
    "robometer.configs.eval_configs": _robometer_configs_eval,
    "robometer.configs.experiment_configs": _robometer_configs_exp,
    "robometer.models": _robometer_models,
    "robometer.models.utils": _robometer_models_utils,
}
for _k, _v in _ROBOMETER_MODULES.items():
    sys.modules.setdefault(_k, _v)

# --- Mock gymnasium ---
_gymnasium = types.ModuleType("gymnasium")
_gymnasium_spaces = types.ModuleType("gymnasium.spaces")


class _MockBox:
    def __init__(self, low=-np.inf, high=np.inf, shape=None, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if shape else low
        self.high = np.full(shape, high, dtype=dtype) if shape else high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.zeros(self.shape, dtype=self.dtype) if self.shape else 0.0


class _MockDict:
    def __init__(self, spaces_dict=None):
        self.spaces = spaces_dict or {}

    def __getitem__(self, key):
        return self.spaces[key]

    def __contains__(self, key):
        return key in self.spaces


class _MockEnv:
    metadata = {}
    observation_space = None
    action_space = None
    render_mode = None
    np_random = None

    def reset(self, seed=None, options=None):
        return None, {}

    def step(self, action):
        return None, 0.0, False, False, {}

    def render(self):
        return None

    def close(self):
        pass


_gymnasium_spaces.Box = _MockBox
_gymnasium_spaces.Dict = _MockDict
_gymnasium.spaces = _gymnasium_spaces
_gymnasium.Env = _MockEnv
_gymnasium.Wrapper = type("Wrapper", (_MockEnv,), {})
_gymnasium.register = lambda **kwargs: None

sys.modules.setdefault("gymnasium", _gymnasium)
sys.modules.setdefault("gymnasium.spaces", _gymnasium_spaces)

# --- Mock scipy ---
_scipy = types.ModuleType("scipy")
_scipy_stats = types.ModuleType("scipy.stats")

_SpearmanResult = MagicMock()
_SpearmanResult.statistic = 1.0
_SpearmanResult.pvalue = 0.0
_KendallResult = MagicMock()
_KendallResult.statistic = 1.0
_KendallResult.pvalue = 0.0

_scipy_stats.spearmanr = MagicMock(return_value=_SpearmanResult)
_scipy_stats.kendalltau = MagicMock(return_value=_KendallResult)
_scipy.stats = _scipy_stats

sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.stats", _scipy_stats)

# --- Mock transformers ---
_transformers = types.ModuleType("transformers")
_transformers.TrainingArguments = MagicMock
sys.modules.setdefault("transformers", _transformers)

# --- Mock peft ---
_peft = types.ModuleType("peft")
_peft.LoraConfig = MagicMock
_peft.get_peft_model = MagicMock(side_effect=lambda model, config: model)
sys.modules.setdefault("peft", _peft)

# ═══════════════════════════════════════════════════════════════════════════
# Force-reload the MUT to pick up our mocked sys.modules
# ═══════════════════════════════════════════════════════════════════════════

mod_key = "strands_robots.robometer"
if mod_key in sys.modules:
    del sys.modules[mod_key]

parent_key = "strands_robots"
if parent_key in sys.modules:
    parent_pkg = sys.modules[parent_key]
    if hasattr(parent_pkg, "robometer"):
        try:
            delattr(parent_pkg, "robometer")
        except AttributeError:
            pass

import strands_robots.robometer as robometer_mod  # noqa: E402

StrandsRobometerRewardWrapper = robometer_mod.StrandsRobometerRewardWrapper
robometer_reward_fn_factory = robometer_mod.robometer_reward_fn
RobometerTrainer = robometer_mod.RobometerTrainer
strands_robots_loader = robometer_mod.strands_robots_loader
RobometerBenchmark = robometer_mod.RobometerBenchmark

# ═══════════════════════════════════════════════════════════════════════════
# Cleanup: Pop mock entries that could pollute downstream tests.
# The MUT (robometer_mod) already captured its refs at import time.
# Without this, pytest.importorskip("torch") in test_record.py finds our
# mock torch (which lacks from_numpy, etc.) and fails.
# We keep robometer-specific mocks since no other test uses importorskip
# for those packages.
# ═══════════════════════════════════════════════════════════════════════════
_ALL_MOCK_MODULES = {
    "torch": _torch,
    "torch.nn": _torch_nn,
    "gymnasium": _gymnasium,
    "gymnasium.spaces": _gymnasium_spaces,
    "scipy": _scipy,
    "scipy.stats": _scipy_stats,
    "transformers": _transformers,
    "peft": _peft,
    **_ROBOMETER_MODULES,
}
for _mock_key, _mock_val in _ALL_MOCK_MODULES.items():
    if sys.modules.get(_mock_key) is _mock_val:
        sys.modules.pop(_mock_key, None)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reinject_mocks():
    """Re-inject mock modules for robometer tests, then clean up after.

    Since we popped all mocks from sys.modules after MUT import (to prevent
    cross-test pollution), we need to re-inject them for any robometer test
    that triggers lazy imports (e.g., RobometerTrainer.train() does
    ``from transformers import TrainingArguments``).
    ``patch.dict`` automatically restores the original state on exit.
    """
    with patch.dict(sys.modules, _ALL_MOCK_MODULES):
        yield


def _make_fake_env(
    pixels_key="pixels",
    image_shape=(64, 64, 3),
    task="pick up the red block",
):
    """Create a mock environment that behaves like StrandsSimEnv."""
    env = MagicMock()

    obs_dict = {}
    if pixels_key:
        obs_dict[pixels_key] = np.random.randint(0, 255, image_shape, dtype=np.uint8)
    obs_dict["state"] = np.zeros(12, dtype=np.float32)

    obs_space = _MockDict(obs_dict)
    env.observation_space = obs_space
    env.action_space = MagicMock()
    env.action_space.sample.return_value = np.zeros(6, dtype=np.float32)
    env.task = task
    env.unwrapped = env
    env.render_mode = "rgb_array"
    env.spec = None
    env.metadata = {"render_modes": ["rgb_array"]}
    env.reward_range = (-float("inf"), float("inf"))

    env.reset.return_value = (obs_dict, {"task": task})

    def _step_side_effect(action):
        new_obs = dict(obs_dict)
        new_obs[pixels_key] = np.random.randint(0, 255, image_shape, dtype=np.uint8)
        return (new_obs, 0.0, False, False, {"step": 1, "task": task})

    env.step.side_effect = _step_side_effect
    env.render.return_value = np.zeros(image_shape, dtype=np.uint8)
    env.close.return_value = None

    return env


@pytest.fixture
def mock_env():
    return _make_fake_env()


@pytest.fixture
def mock_robometer_model():
    config = MagicMock()
    config.model_type = "qwen2_5_vl"
    tokenizer = MagicMock()
    processor = MagicMock()
    model = MagicMock()
    model.eval.return_value = model
    model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=100), requires_grad=True)]
    return config, tokenizer, processor, model


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear the model cache between tests to prevent state leakage."""
    robometer_mod._MODEL_CACHE.clear()
    yield
    robometer_mod._MODEL_CACHE.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Module-level flags
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleLevelFlags:

    def test_has_gymnasium(self):
        assert robometer_mod.HAS_GYMNASIUM is True

    def test_has_torch(self):
        assert robometer_mod.HAS_TORCH is True

    def test_has_robometer(self):
        assert robometer_mod.HAS_ROBOMETER is True

    def test_has_scipy(self):
        assert robometer_mod.HAS_SCIPY is True

    def test_has_transformers(self):
        assert robometer_mod.HAS_TRANSFORMERS is True

    def test_module_loads(self):
        assert robometer_mod is not None

    def test_all_exports_defined(self):
        for name in robometer_mod.__all__:
            assert hasattr(robometer_mod, name), f"Missing export: {name}"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: StrandsRobometerRewardWrapper
# ═══════════════════════════════════════════════════════════════════════════


class TestStrandsRobometerRewardWrapper:

    def test_init_stores_config(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", device="cpu", task_instruction="test")
        assert w._model_path == "m/m"
        assert w._device == "cpu"
        assert w._max_frames == 16

    def test_init_resolves_task_from_env(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", device="cpu")
        assert w._task_instruction == "pick up the red block"

    def test_init_fallback_task(self):
        env = _make_fake_env()
        del env.task
        env.unwrapped = MagicMock(spec=[])
        w = StrandsRobometerRewardWrapper(env=env, model_path="m/m", device="cpu")
        assert w._task_instruction == "complete the task"

    def test_init_explicit_task(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", device="cpu", task_instruction="custom")
        assert w._task_instruction == "custom"

    def test_init_missing_pixels_key_raises(self, mock_env):
        with pytest.raises(ValueError, match="nonexistent"):
            StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", pixels_key="nonexistent")

    def test_init_proxied_attributes(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        assert w.observation_space is mock_env.observation_space
        assert w.action_space is mock_env.action_space
        assert w.render_mode == "rgb_array"

    # ── reset ─────────────────────────────────────────────────────────

    def test_reset_clears_buffers(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w._frame_buffer.append(np.zeros((64, 64, 3), dtype=np.uint8))
        w._prev_progress = 0.5
        w._success_votes.append(0.9)

        w.reset()

        assert w._prev_progress is None
        assert len(w._success_votes) == 0
        assert len(w._frame_buffer) == 1

    def test_reset_returns_obs_and_info(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        obs, info = w.reset()
        assert isinstance(info, dict)
        assert "robometer_progress" in info

    def test_reset_seeds_frame_buffer(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset()
        assert len(w._frame_buffer) == 1
        assert w._frame_buffer[0].shape == (64, 64, 3)
        assert w._frame_buffer[0].dtype == np.uint8

    def test_reset_passes_seed(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset(seed=42)
        mock_env.reset.assert_called_once_with(seed=42, options=None)

    # ── step ──────────────────────────────────────────────────────────

    def test_step_returns_five_tuple(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", device="cpu")
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.42, None)):
            result = w.step(np.zeros(6))
        assert len(result) == 5
        _, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_appends_frame(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset()
        assert len(w._frame_buffer) == 1
        with patch.object(w, "_infer_reward", return_value=(0.5, None)):
            w.step(np.zeros(6))
        assert len(w._frame_buffer) == 2

    def test_step_replaces_env_reward(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", add_estimated_reward=False)
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.75, None)):
            _, reward, _, _, _ = w.step(np.zeros(6))
        assert reward == 0.75

    def test_step_adds_to_env_reward(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", add_estimated_reward=True)
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.75, None)):
            _, reward, _, _, _ = w.step(np.zeros(6))
        assert reward == 0.75  # env_reward=0.0 + 0.75

    def test_step_reward_scale(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", reward_scale=2.0)
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.5, None)):
            _, reward, _, _, _ = w.step(np.zeros(6))
        assert reward == 1.0

    def test_step_info_keys(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.42, 0.8)):
            _, _, _, _, info = w.step(np.zeros(6))
        assert "robometer_reward_raw" in info
        assert "robometer_reward_scaled" in info
        assert "robometer_progress" in info
        assert "env_reward" in info

    def test_step_success_detection_terminates(self, mock_env):
        w = StrandsRobometerRewardWrapper(
            env=mock_env,
            model_path="m/m",
            use_success_detection=True,
            success_vote_window=3,
            success_threshold=0.5,
            success_vote_fraction=0.6,
        )
        w.reset()
        for _ in range(3):
            with patch.object(w, "_infer_reward", return_value=(0.9, 0.95)):
                _, _, terminated, _, info = w.step(np.zeros(6))
        assert terminated is True
        assert info["robometer_success"] is True

    def test_step_graceful_on_inference_failure(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset()
        with patch.object(w, "_infer_reward", side_effect=RuntimeError("CUDA OOM")):
            _, reward, terminated, _, _ = w.step(np.zeros(6))
        assert reward == 0.0
        assert isinstance(terminated, bool)

    # ── proxy ─────────────────────────────────────────────────────────

    def test_render_proxied(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.render()
        mock_env.render.assert_called_once()

    def test_close_proxied(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.close()
        mock_env.close.assert_called_once()

    def test_unwrapped(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        assert w.unwrapped is mock_env

    def test_getattr_delegates(self, mock_env):
        mock_env.custom_attr = "hello"
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        assert w.custom_attr == "hello"

    def test_str(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        assert "StrandsRobometerRewardWrapper" in str(w)

    def test_repr(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        assert "m/m" in repr(w)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: robometer_reward_fn
# ═══════════════════════════════════════════════════════════════════════════


class TestRobometerRewardFn:

    def test_returns_callable(self):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test")
        assert callable(fn)

    def test_has_reset_method(self):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test")
        assert hasattr(fn, "reset")
        assert callable(fn.reset)

    def test_reset_does_not_raise(self):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test")
        fn.reset()

    def test_callable_returns_float(self, mock_robometer_model):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test", device="cpu")

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
                obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
                reward = fn(obs, np.zeros(6))

        assert isinstance(reward, float)

    def test_missing_pixels_returns_zero(self, mock_robometer_model):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test", device="cpu")

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            obs = {"state": np.zeros(12)}
            reward = fn(obs, np.zeros(6))

        assert reward == 0.0

    def test_reward_scale(self, mock_robometer_model):
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test", device="cpu", reward_scale=3.0)

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
                obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
                reward = fn(obs, np.zeros(6))

        # 0.5 * 3.0 = 1.5
        assert reward == pytest.approx(1.5)

    def test_inference_failure_returns_zero(self, mock_robometer_model):
        """_reward() must not crash RL training loop on inference error (GPU OOM, tensor mismatch)."""
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test", device="cpu")

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            with patch.object(robometer_mod, "_infer_progress", side_effect=RuntimeError("CUDA OOM")):
                obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
                reward = fn(obs, np.zeros(6))

        assert reward == 0.0

    def test_inference_failure_does_not_corrupt_state(self, mock_robometer_model):
        """After an inference failure, subsequent successful calls should work normally."""
        fn = robometer_reward_fn_factory(model_path="m/m", task_instruction="test", device="cpu")

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            # First call: inference fails
            with patch.object(robometer_mod, "_infer_progress", side_effect=RuntimeError("OOM")):
                obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
                reward = fn(obs, np.zeros(6))
            assert reward == 0.0

            # Second call: inference succeeds
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.7, None)):
                obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
                reward = fn(obs, np.zeros(6))
            assert isinstance(reward, float)
            assert reward == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: RobometerTrainer
# ═══════════════════════════════════════════════════════════════════════════


class TestRobometerTrainer:

    def test_constructor(self):
        t = RobometerTrainer(model_path="m/m", device="cpu")
        assert t._model_path == "m/m"
        assert t.provider_name == "robometer"

    def test_train_returns_metrics(self, mock_robometer_model):
        t = RobometerTrainer(model_path="m/m", device="cpu")
        mock_dataset = MagicMock()
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})

        _robometer_utils_save.load_model_from_hf = MagicMock(return_value=mock_robometer_model)
        _robometer_utils_setup.setup_batch_collator = MagicMock(return_value=MagicMock())
        _robometer_trainers_rbm.RBMHeadsTrainer = MagicMock(return_value=mock_hf_trainer)

        result = t.train(train_dataset=mock_dataset)
        assert "loss" in result

    def test_evaluate_without_train_raises(self):
        t = RobometerTrainer(model_path="m/m", device="cpu")
        with pytest.raises(RuntimeError, match="No trainer"):
            t.evaluate()

    def test_save_without_train_raises(self):
        t = RobometerTrainer(model_path="m/m", device="cpu")
        with pytest.raises(RuntimeError, match="No trainer"):
            t.save_model()

    def test_lora_config(self):
        t = RobometerTrainer(model_path="m/m", device="cpu", use_lora=True, lora_rank=16, lora_alpha=32)
        assert t._use_lora is True
        assert t._lora_rank == 16

    def test_fsdp_config(self):
        t = RobometerTrainer(model_path="m/m", device="cpu", use_fsdp=True, fsdp_config={"key": "val"})
        assert t._use_fsdp is True
        assert "key" in t._fsdp_config


# ═══════════════════════════════════════════════════════════════════════════
# Tests: strands_robots_loader
# ═══════════════════════════════════════════════════════════════════════════


class TestStrandsRobotsLoader:

    def _frame(self, shape=(64, 64, 3)):
        return np.random.randint(0, 255, shape, dtype=np.uint8)

    def test_from_list(self):
        data = [
            {
                "observations": [{"pixels": self._frame(), "state": np.zeros(12)}, {"pixels": self._frame()}],
                "actions": [np.zeros(6), np.zeros(6)],
                "task": "pick up block",
            }
        ]
        result = strands_robots_loader(data)
        assert len(result) == 1
        assert result[0]["task_instruction"] == "pick up block"
        assert len(result[0]["frames"]) == 2

    def test_from_empty_list(self):
        assert len(strands_robots_loader([])) == 0

    def test_from_record_session(self):
        session = MagicMock()
        ep = MagicMock()
        ep.observations = [{"pixels": self._frame()}, {"pixels": self._frame()}]
        ep.actions = [np.zeros(6)]
        ep.task = "grasp"
        session.episodes = [ep]
        session.task = "fallback"

        result = strands_robots_loader(session)
        assert len(result) == 1
        assert result[0]["task_instruction"] == "grasp"

    def test_from_lerobot_path(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        info = {"task": {"instruction": "pick cube"}, "num_episodes": 0}
        (meta_dir / "info.json").write_text(json.dumps(info))

        result = strands_robots_loader(tmp_path)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_missing_meta_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            strands_robots_loader(tmp_path)

    def test_max_frames_subsampling(self):
        data = [{"observations": [{"pixels": self._frame()} for _ in range(100)], "task": "test"}]
        result = strands_robots_loader(data, max_frames=10)
        assert len(result[0]["frames"]) == 10

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            strands_robots_loader(12345)

    def test_task_instruction_override(self):
        data = [{"observations": [{"pixels": self._frame()}], "task": "original"}]
        result = strands_robots_loader(data, task_instruction="overridden")
        assert result[0]["task_instruction"] == "overridden"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: RobometerBenchmark
# ═══════════════════════════════════════════════════════════════════════════


class TestRobometerBenchmark:

    def test_constructor(self):
        b = RobometerBenchmark(model_path="m/m", device="cpu", task_instruction="test")
        assert b._num_episodes == 10

    def test_evaluate_with_ground_truth(self, mock_robometer_model):
        b = RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=2)

        env = _make_fake_env()
        call_count = [0]

        def terminating_step(action):
            call_count[0] += 1
            obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "state": np.zeros(12)}
            if call_count[0] >= 3:
                call_count[0] = 0
                return obs, 0.0, True, False, {}
            return obs, 0.0, False, False, {}

        env.step.side_effect = terminating_step

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, 0.3)):
                result = b.evaluate(
                    policies={"good": lambda obs: np.zeros(6), "bad": lambda obs: np.ones(6)},
                    env_factory=lambda: env,
                    ground_truth_ranking=["good", "bad"],
                )

        assert "policy_scores" in result
        assert "ranking" in result
        assert len(result["ranking"]) == 2
        assert "spearman_rho" in result
        assert "kendall_tau" in result

    def test_evaluate_no_ground_truth(self, mock_robometer_model):
        b = RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=1)

        env = _make_fake_env()
        env.step.side_effect = lambda action: (
            {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "state": np.zeros(12)},
            0.0,
            True,
            False,
            {},
        )

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_robometer_model):
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, 0.3)):
                result = b.evaluate(
                    policies={"p1": lambda obs: np.zeros(6)},
                    env_factory=lambda: env,
                )

        assert result["spearman_rho"] is None
        assert result["kendall_tau"] is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_step_without_reset(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        with patch.object(w, "_infer_reward", return_value=(0.5, None)):
            result = w.step(np.zeros(6))
        assert len(result) == 5

    def test_multiple_resets(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m")
        w.reset()
        w.reset()
        w.reset()
        assert mock_env.reset.call_count == 3
        assert len(w._frame_buffer) == 1

    def test_long_episode_caps_buffer(self, mock_env):
        w = StrandsRobometerRewardWrapper(env=mock_env, model_path="m/m", max_frames=8)
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.5, None)):
            for _ in range(50):
                w.step(np.zeros(6))
        assert len(w._frame_buffer) <= 8

    def test_success_vote_window_too_short(self, mock_env):
        w = StrandsRobometerRewardWrapper(
            env=mock_env, model_path="m/m", use_success_detection=True, success_vote_window=5
        )
        w.reset()
        assert w._vote_success(0.9) is False
        assert w._vote_success(0.9) is False

    def test_model_cache(self, mock_robometer_model):
        mock_load = MagicMock(return_value=mock_robometer_model)
        _robometer_utils_save.load_model_from_hf = mock_load

        r1 = robometer_mod._load_robometer_model("test/model", "cpu")
        r2 = robometer_mod._load_robometer_model("test/model", "cpu")

        mock_load.assert_called_once()
        assert r1 == r2


class TestWrapperImageShapes:

    @pytest.mark.parametrize(
        "shape",
        [
            (48, 48, 3),
            (64, 64, 3),
            (128, 128, 3),
            (224, 224, 3),
            (480, 640, 3),
        ],
    )
    def test_various_shapes(self, shape):
        env = _make_fake_env(image_shape=shape)
        w = StrandsRobometerRewardWrapper(env=env, model_path="m/m")
        w.reset()
        with patch.object(w, "_infer_reward", return_value=(0.5, None)):
            _, reward, _, _, _ = w.step(np.zeros(6))
        assert np.isfinite(reward)
        assert w._frame_buffer[0].shape == shape
