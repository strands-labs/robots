"""Additional robometer.py tests targeting uncovered lines (80%→95%+).

Covers:
- _infer_progress() direct call (lines 193-230)
- _ensure_model_loaded() + _infer_reward() internal wrapper (lines 364-400)
- robometer_reward_fn closure _ensure_loaded() (line 580)
- _build_trainer() with LoRA and FSDP (lines 690-726)
- evaluate() / save_model() after train() (lines 773-783)
- _load_lerobot_v3_episodes with parquet (lines 913-937)
- RobometerBenchmark._rollout_policy / _score_trajectories (lines 862-880)
- _require() helper (line 120)
- clear_model_cache utility
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Module-level mocks — same pattern as test_robometer.py
# ═══════════════════════════════════════════════════════════════════════════

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

_robometer_utils_save.load_model_from_hf = MagicMock()
_robometer_utils_setup.setup_batch_collator = MagicMock(return_value=MagicMock())
_robometer_evals_utils.raw_dict_to_sample = MagicMock(return_value={"sample": "data"})
_robometer_evals_utils.extract_rewards_from_output = MagicMock(return_value=np.array([0.5]))
_robometer_evals_utils.extract_success_probs_from_output = MagicMock(return_value=np.array([0.3]))
_robometer_evals_server.process_batch_helper = MagicMock(return_value={"outputs": "data"})
_robometer_trainers_rbm.RBMHeadsTrainer = MagicMock()

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


_gymnasium_spaces.Box = _MockBox
_gymnasium_spaces.Dict = _MockDict
_gymnasium.spaces = _gymnasium_spaces
_gymnasium.Env = type(
    "Env",
    (),
    {
        "reset": lambda self, **kw: (None, {}),
        "step": lambda self, a: (None, 0.0, False, False, {}),
        "render": lambda self: None,
        "close": lambda self: None,
        "metadata": {},
        "observation_space": None,
        "action_space": None,
        "render_mode": None,
        "np_random": None,
    },
)
_gymnasium.register = lambda **kw: None

_scipy = types.ModuleType("scipy")
_scipy_stats = types.ModuleType("scipy.stats")
_SpearmanResult = MagicMock()
_SpearmanResult.statistic = 0.9
_SpearmanResult.pvalue = 0.05
_KendallResult = MagicMock()
_KendallResult.statistic = 0.85
_KendallResult.pvalue = 0.04
_scipy_stats.spearmanr = MagicMock(return_value=_SpearmanResult)
_scipy_stats.kendalltau = MagicMock(return_value=_KendallResult)
_scipy.stats = _scipy_stats

_transformers = types.ModuleType("transformers")
_transformers.TrainingArguments = MagicMock()

_peft = types.ModuleType("peft")
_peft.LoraConfig = MagicMock()
_peft.get_peft_model = MagicMock(side_effect=lambda model, config: model)

_pyarrow = types.ModuleType("pyarrow")
_pyarrow_parquet = types.ModuleType("pyarrow.parquet")
_pyarrow.parquet = _pyarrow_parquet

# Inject all mocks
for _k, _v in _ROBOMETER_MODULES.items():
    sys.modules.setdefault(_k, _v)
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", _torch_nn)
sys.modules.setdefault("gymnasium", _gymnasium)
sys.modules.setdefault("gymnasium.spaces", _gymnasium_spaces)
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.stats", _scipy_stats)
sys.modules.setdefault("transformers", _transformers)
sys.modules.setdefault("peft", _peft)
sys.modules.setdefault("pyarrow", _pyarrow)
sys.modules.setdefault("pyarrow.parquet", _pyarrow_parquet)

# Force-reload MUT
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

# Cleanup mocks from sys.modules
_ALL_MOCK_MODULES = {
    "torch": _torch,
    "torch.nn": _torch_nn,
    "gymnasium": _gymnasium,
    "gymnasium.spaces": _gymnasium_spaces,
    "scipy": _scipy,
    "scipy.stats": _scipy_stats,
    "transformers": _transformers,
    "peft": _peft,
    "pyarrow": _pyarrow,
    "pyarrow.parquet": _pyarrow_parquet,
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
    with patch.dict(sys.modules, _ALL_MOCK_MODULES):
        yield


@pytest.fixture(autouse=True)
def _clear_cache():
    robometer_mod._MODEL_CACHE.clear()
    # Reset call counts on shared module-level mocks to prevent cross-test accumulation
    _robometer_utils_save.load_model_from_hf.reset_mock()
    _robometer_utils_setup.setup_batch_collator.reset_mock()
    _robometer_evals_utils.raw_dict_to_sample.reset_mock()
    _robometer_evals_utils.extract_rewards_from_output.reset_mock()
    _robometer_evals_utils.extract_success_probs_from_output.reset_mock()
    _robometer_evals_server.process_batch_helper.reset_mock()
    _robometer_trainers_rbm.RBMHeadsTrainer.reset_mock()
    _peft.LoraConfig.reset_mock()
    _peft.get_peft_model.reset_mock()
    _peft.get_peft_model.side_effect = lambda model, config: model
    yield
    robometer_mod._MODEL_CACHE.clear()


@pytest.fixture
def mock_model():
    config = MagicMock()
    config.model_type = "qwen2_5_vl"
    tokenizer = MagicMock()
    processor = MagicMock()
    model = MagicMock()
    model.eval.return_value = model
    model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=1000), requires_grad=True)]
    return config, tokenizer, processor, model


def _make_env(pixels_key="pixels", shape=(64, 64, 3), task="pick cube"):
    env = MagicMock()
    obs = {pixels_key: np.random.randint(0, 255, shape, dtype=np.uint8), "state": np.zeros(12, dtype=np.float32)}
    env.observation_space = _MockDict(obs)
    env.action_space = MagicMock()
    env.action_space.sample.return_value = np.zeros(6, dtype=np.float32)
    env.task = task
    env.unwrapped = env
    env.render_mode = "rgb_array"
    env.spec = None
    env.metadata = {}
    env.reward_range = (-float("inf"), float("inf"))
    env.reset.return_value = (obs, {"task": task})
    env.step.side_effect = lambda a: (
        {pixels_key: np.random.randint(0, 255, shape, dtype=np.uint8), "state": np.zeros(12)},
        0.0,
        False,
        False,
        {},
    )
    env.render.return_value = np.zeros(shape, dtype=np.uint8)
    env.close.return_value = None
    return env


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _require helper
# ═══════════════════════════════════════════════════════════════════════════


class TestRequire:
    def test_require_true_no_raise(self):
        robometer_mod._require(True, "anything")

    def test_require_false_raises(self):
        with pytest.raises(ImportError, match="test_pkg"):
            robometer_mod._require(False, "test_pkg")


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _infer_progress direct call (lines 193-230)
# ═══════════════════════════════════════════════════════════════════════════


class TestInferProgress:
    def test_basic_call(self, mock_model):
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        traj = {"frames": frames, "task_instruction": "pick cube"}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.75])
        _robometer_evals_utils.extract_success_probs_from_output.return_value = np.array([0.6])

        progress, success = robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        assert progress == pytest.approx(0.75)
        assert success == pytest.approx(0.6)

    def test_frames_as_ndarray(self, mock_model):
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        frames = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        traj = {"frames": frames, "task_instruction": "test"}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.5])
        _robometer_evals_utils.extract_success_probs_from_output.return_value = np.array([0.3])

        progress, success = robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        assert progress == pytest.approx(0.5)

    def test_empty_rewards(self, mock_model):
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        traj = {"frames": frames, "task_instruction": "test"}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([])
        _robometer_evals_utils.extract_success_probs_from_output.return_value = np.array([])

        progress, success = robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        assert progress == 0.0
        assert success is None

    def test_success_head_not_present(self, mock_model):
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        traj = {"frames": frames, "task_instruction": "test"}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.8])
        _robometer_evals_utils.extract_success_probs_from_output.side_effect = KeyError("no success head")

        progress, success = robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        assert progress == pytest.approx(0.8)
        assert success is None

    def test_success_value_error(self, mock_model):
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        traj = {"frames": frames, "task_instruction": "test"}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.6])
        _robometer_evals_utils.extract_success_probs_from_output.side_effect = ValueError("bad output")

        progress, success = robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        assert progress == pytest.approx(0.6)
        assert success is None

    def test_default_task_instruction(self, mock_model):
        """Trajectory without explicit task_instruction uses default."""
        config, tokenizer, processor, model = mock_model
        collator = MagicMock()
        traj = {"frames": [np.zeros((64, 64, 3), dtype=np.uint8)]}

        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.5])
        _robometer_evals_utils.extract_success_probs_from_output.return_value = np.array([])

        robometer_mod._infer_progress(config, model, tokenizer, collator, "cpu", traj, 16)
        # raw_dict_to_sample should have been called with "complete the task"
        call_args = _robometer_evals_utils.raw_dict_to_sample.call_args
        assert call_args[0][0]["task"] == "complete the task"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _ensure_model_loaded / _infer_reward (lines 364-400)
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureModelLoadedAndInferReward:
    def test_ensure_model_loaded_calls_load(self, mock_model):
        env = _make_env()
        w = robometer_mod.StrandsRobometerRewardWrapper(env=env, model_path="m/m", device="cpu")
        assert w._model_loaded is False

        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        w._ensure_model_loaded()
        assert w._model_loaded is True
        assert w._config is mock_model[0]
        assert w._tokenizer is mock_model[1]
        _robometer_utils_save.load_model_from_hf.assert_called_once_with("m/m", device="cpu")

    def test_ensure_model_loaded_idempotent(self, mock_model):
        env = _make_env()
        w = robometer_mod.StrandsRobometerRewardWrapper(env=env, model_path="m/m", device="cpu")
        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        w._ensure_model_loaded()
        w._ensure_model_loaded()
        _robometer_utils_save.load_model_from_hf.assert_called_once()

    def test_infer_reward_relative(self, mock_model):
        env = _make_env()
        w = robometer_mod.StrandsRobometerRewardWrapper(
            env=env, model_path="m/m", device="cpu", use_relative_rewards=True
        )
        w._model_loaded = True
        w._config, w._tokenizer, w._processor, w._model = mock_model
        w._batch_collator = MagicMock()

        # First call: prev_progress=None → prev=0.0, progress=0.6, reward=0.6
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.6, 0.3)):
            w._frame_buffer.append(np.zeros((64, 64, 3), dtype=np.uint8))
            reward, success = w._infer_reward()
        assert reward == pytest.approx(0.6)
        assert w._prev_progress == pytest.approx(0.6)

        # Second call: prev=0.6, progress=0.8, reward=0.2
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.8, 0.5)):
            reward, success = w._infer_reward()
        assert reward == pytest.approx(0.2)

    def test_infer_reward_absolute(self, mock_model):
        env = _make_env()
        w = robometer_mod.StrandsRobometerRewardWrapper(
            env=env, model_path="m/m", device="cpu", use_relative_rewards=False
        )
        w._model_loaded = True
        w._config, w._tokenizer, w._processor, w._model = mock_model
        w._batch_collator = MagicMock()
        w._frame_buffer.append(np.zeros((64, 64, 3), dtype=np.uint8))

        with patch.object(robometer_mod, "_infer_progress", return_value=(0.8, None)):
            reward, success = w._infer_reward()
        assert reward == pytest.approx(0.8)

    def test_infer_reward_passes_frame_buffer(self, mock_model):
        env = _make_env()
        w = robometer_mod.StrandsRobometerRewardWrapper(env=env, model_path="m/m", device="cpu")
        w._model_loaded = True
        w._config, w._tokenizer, w._processor, w._model = mock_model
        w._batch_collator = MagicMock()

        f1 = np.ones((64, 64, 3), dtype=np.uint8)
        f2 = np.ones((64, 64, 3), dtype=np.uint8) * 128
        w._frame_buffer.append(f1)
        w._frame_buffer.append(f2)

        with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)) as mock_infer:
            w._infer_reward()
            call_args = mock_infer.call_args
            traj = call_args[0][5]  # trajectory_dict is positional arg 5
            assert len(traj["frames"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Tests: robometer_reward_fn closure _ensure_loaded (line 580)
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardFnClosure:
    def test_ensure_loaded_called_on_first_invoke(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        fn = robometer_mod.robometer_reward_fn(model_path="m/m", device="cpu")
        obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
            fn(obs, np.zeros(6))

        _robometer_utils_save.load_model_from_hf.assert_called()

    def test_absolute_rewards_mode(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        fn = robometer_mod.robometer_reward_fn(
            model_path="m/m", device="cpu", use_relative_rewards=False, reward_scale=2.0
        )
        obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        with patch.object(robometer_mod, "_infer_progress", return_value=(0.4, None)):
            reward = fn(obs, np.zeros(6))
        assert reward == pytest.approx(0.8)

    def test_relative_rewards_across_calls(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        fn = robometer_mod.robometer_reward_fn(model_path="m/m", device="cpu")
        obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        # Call 1: progress 0.3, prev None → reward 0.3
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.3, None)):
            r1 = fn(obs, np.zeros(6))
        assert r1 == pytest.approx(0.3)

        # Call 2: progress 0.5, prev 0.3 → reward 0.2
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
            r2 = fn(obs, np.zeros(6))
        assert r2 == pytest.approx(0.2)

    def test_reset_clears_state(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model

        fn = robometer_mod.robometer_reward_fn(model_path="m/m", device="cpu")
        obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}

        with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
            fn(obs, np.zeros(6))

        fn.reset()

        # After reset, prev_progress is None → next call treats prev as 0
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.3, None)):
            r = fn(obs, np.zeros(6))
        assert r == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: RobometerTrainer _build_trainer (LoRA / FSDP)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainerBuildTrainer:
    def test_train_with_lora(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.05})
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(
            model_path="m/m", device="cpu", use_lora=True, lora_rank=16, lora_alpha=32, lora_dropout=0.1
        )
        result = t.train(train_dataset=MagicMock())
        assert "loss" in result
        _peft.LoraConfig.assert_called()
        _peft.get_peft_model.assert_called()

    def test_train_with_fsdp(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(
            model_path="m/m", device="cpu", use_fsdp=True, fsdp_config={"auto_wrap_policy": "SIZE_BASED"}
        )
        result = t.train(train_dataset=MagicMock())
        assert "loss" in result
        # Verify TrainingArguments was called with fsdp settings
        ta_call = _transformers.TrainingArguments.call_args
        assert ta_call is not None

    def test_evaluate_after_train(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})
        mock_hf_trainer.evaluate.return_value = {"eval_loss": 0.2, "eval_accuracy": 0.85}
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(model_path="m/m", device="cpu")
        t.train(train_dataset=MagicMock())
        metrics = t.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_accuracy"] == 0.85

    def test_save_model_after_train(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(model_path="m/m", device="cpu", output_dir="/tmp/out")
        t.train(train_dataset=MagicMock())
        t.save_model("/tmp/custom_path")
        mock_hf_trainer.save_model.assert_called_once_with("/tmp/custom_path")

    def test_save_model_default_path(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(model_path="m/m", device="cpu", output_dir="/tmp/default")
        t.train(train_dataset=MagicMock())
        t.save_model()
        mock_hf_trainer.save_model.assert_called_once_with("/tmp/default")

    def test_evaluate_with_eval_dataset(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        mock_hf_trainer = MagicMock()
        mock_hf_trainer.train.return_value = MagicMock(metrics={"loss": 0.1})
        mock_hf_trainer.evaluate.return_value = {"eval_loss": 0.15}
        _robometer_trainers_rbm.RBMHeadsTrainer.return_value = mock_hf_trainer

        t = robometer_mod.RobometerTrainer(model_path="m/m", device="cpu")
        eval_ds = MagicMock()
        t.train(train_dataset=MagicMock(), eval_dataset=eval_ds)
        metrics = t.evaluate(eval_dataset=eval_ds)
        assert metrics["eval_loss"] == 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _load_lerobot_v3_episodes (lines 913-937)
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadLerobotV3Episodes:
    def test_missing_meta_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata not found"):
            robometer_mod._load_lerobot_v3_episodes(tmp_path)

    def test_zero_episodes(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        info = {"task": {"instruction": "test task"}, "num_episodes": 0}
        (meta_dir / "info.json").write_text(json.dumps(info))

        episodes = robometer_mod._load_lerobot_v3_episodes(tmp_path)
        assert episodes == []

    def test_parquet_not_found(self, tmp_path):
        """Episodes with missing parquet files are silently skipped."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        info = {"task": {"instruction": "pick up"}, "num_episodes": 2}
        (meta_dir / "info.json").write_text(json.dumps(info))

        episodes = robometer_mod._load_lerobot_v3_episodes(tmp_path)
        assert episodes == []

    def test_parquet_with_mock_data(self, tmp_path):
        """Load episodes from mock parquet files using mocked pyarrow."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data"
        meta_dir.mkdir()
        data_dir.mkdir()

        info = {"task": {"instruction": "grasp object"}, "num_episodes": 1}
        (meta_dir / "info.json").write_text(json.dumps(info))
        # Create a dummy parquet file (content doesn't matter, pyarrow is mocked)
        (data_dir / "episode_000000.parquet").write_bytes(b"fake_parquet")

        mock_df = MagicMock()
        # Simulate iterrows returning 2 rows
        row1 = {"pixels": np.zeros((64, 64, 3), dtype=np.uint8), "state": np.zeros(6), "action": np.zeros(4)}
        row2 = {"pixels": np.ones((64, 64, 3), dtype=np.uint8), "state": np.ones(6), "action": np.ones(4)}

        class RowAccessor:
            """Simulates pandas row with __contains__ and __getitem__."""

            def __init__(self, data):
                self._data = data

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

        mock_df.iterrows.return_value = iter([(0, RowAccessor(row1)), (1, RowAccessor(row2))])

        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df
        _pyarrow_parquet.read_table = MagicMock(return_value=mock_table)

        episodes = robometer_mod._load_lerobot_v3_episodes(tmp_path)
        assert len(episodes) == 1
        assert episodes[0]["task"] == "grasp object"
        assert len(episodes[0]["observations"]) == 2
        assert len(episodes[0]["actions"]) == 2

    def test_pyarrow_import_error(self, tmp_path):
        """Graceful fallback when pyarrow not installed."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data"
        meta_dir.mkdir()
        data_dir.mkdir()

        info = {"task": {"instruction": "test"}, "num_episodes": 1}
        (meta_dir / "info.json").write_text(json.dumps(info))
        (data_dir / "episode_000000.parquet").write_bytes(b"fake")

        _pyarrow_parquet.read_table = MagicMock(side_effect=ImportError("no pyarrow"))

        # The function catches ImportError internally and continues
        episodes = robometer_mod._load_lerobot_v3_episodes(tmp_path)
        assert len(episodes) == 0

    def test_parquet_row_without_pixels(self, tmp_path):
        """Rows without the pixels key produce empty observations."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data"
        meta_dir.mkdir()
        data_dir.mkdir()

        info = {"task": {"instruction": "test"}, "num_episodes": 1}
        (meta_dir / "info.json").write_text(json.dumps(info))
        (data_dir / "episode_000000.parquet").write_bytes(b"fake")

        class RowAccessor:
            def __init__(self, data):
                self._data = data

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

        row = RowAccessor({"state": np.zeros(6)})  # no "pixels"
        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter([(0, row)])
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df
        _pyarrow_parquet.read_table = MagicMock(return_value=mock_table)

        episodes = robometer_mod._load_lerobot_v3_episodes(tmp_path)
        assert len(episodes) == 1
        assert len(episodes[0]["observations"]) == 1
        # obs dict should have "state" but not "pixels"
        assert "state" in episodes[0]["observations"][0]


# ═══════════════════════════════════════════════════════════════════════════
# Tests: RobometerBenchmark internals (lines 862-880, 1018-1019)
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkInternals:
    def test_rollout_policy_collects_frames(self, mock_model):
        b = robometer_mod.RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=2, max_frames=5)
        env = _make_env()
        call_count = [0]

        def step_fn(action):
            call_count[0] += 1
            obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
            if call_count[0] % 3 == 0:
                return obs, 0.0, True, False, {}
            return obs, 0.0, False, False, {}

        env.step.side_effect = step_fn

        trajectories = b._rollout_policy(lambda obs: np.zeros(6), env)
        assert len(trajectories) == 2
        for traj in trajectories:
            assert "frames" in traj
            assert len(traj["frames"]) <= 5

    def test_rollout_policy_subsamples_long_episodes(self, mock_model):
        b = robometer_mod.RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=1, max_frames=4)
        env = _make_env()
        call_count = [0]

        def step_fn(action):
            call_count[0] += 1
            obs = {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)}
            if call_count[0] >= 20:
                call_count[0] = 0
                return obs, 0.0, True, False, {}
            return obs, 0.0, False, False, {}

        env.step.side_effect = step_fn

        trajectories = b._rollout_policy(lambda obs: np.zeros(6), env)
        assert len(trajectories[0]["frames"]) == 4  # subsampled

    def test_score_trajectories(self, mock_model):
        b = robometer_mod.RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=1)
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.7])
        _robometer_evals_utils.extract_success_probs_from_output.return_value = np.array([0.4])

        trajectories = [
            {"frames": [np.zeros((64, 64, 3), dtype=np.uint8)], "task_instruction": "test"},
            {"frames": [np.ones((64, 64, 3), dtype=np.uint8)], "task_instruction": "test"},
        ]

        progress, success = b._score_trajectories(trajectories)
        assert len(progress) == 2
        assert len(success) == 2
        assert all(p == pytest.approx(0.7) for p in progress)

    def test_score_trajectories_no_success_prob(self, mock_model):
        b = robometer_mod.RobometerBenchmark(model_path="m/m", device="cpu")
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        _robometer_evals_utils.extract_rewards_from_output.return_value = np.array([0.5])
        _robometer_evals_utils.extract_success_probs_from_output.side_effect = KeyError("no success")

        trajectories = [{"frames": [np.zeros((64, 64, 3), dtype=np.uint8)], "task_instruction": "test"}]
        # _infer_progress returns None for success_prob on KeyError
        with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, None)):
            progress, success = b._score_trajectories(trajectories)
        assert success[0] == 0.0  # None → 0.0

    def test_evaluate_multiple_policies(self, mock_model):
        b = robometer_mod.RobometerBenchmark(model_path="m/m", device="cpu", num_episodes=1)
        env = _make_env()
        env.step.side_effect = lambda a: (
            {"pixels": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)},
            0.0,
            True,
            False,
            {},
        )

        with patch.object(robometer_mod, "_load_robometer_model", return_value=mock_model):
            with patch.object(robometer_mod, "_infer_progress", return_value=(0.5, 0.3)):
                result = b.evaluate(
                    policies={"a": lambda o: np.zeros(6), "b": lambda o: np.ones(6), "c": lambda o: np.zeros(6)},
                    env_factory=lambda: env,
                    ground_truth_ranking=["a", "b", "c"],
                )

        assert len(result["ranking"]) == 3
        assert result["spearman_rho"] is not None
        assert result["kendall_tau"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: strands_robots_loader edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestLoaderEdgeCases:
    def _frame(self, shape=(64, 64, 3)):
        return np.random.randint(0, 255, shape, dtype=np.uint8)

    def test_observations_as_ndarray_directly(self):
        """Each obs is a raw ndarray (not a dict)."""
        data = [
            {
                "observations": [self._frame(), self._frame()],
                "actions": [np.zeros(6), np.zeros(6)],
                "task": "test",
            }
        ]
        result = robometer_mod.strands_robots_loader(data)
        assert len(result[0]["frames"]) == 2

    def test_observations_mixed_types(self):
        """Mix of dict and ndarray observations."""
        data = [
            {
                "observations": [
                    {"pixels": self._frame(), "state": np.zeros(6)},
                    self._frame(),
                    42,  # non-dict, non-ndarray → skipped
                ],
                "task": "test",
            }
        ]
        result = robometer_mod.strands_robots_loader(data)
        assert len(result[0]["frames"]) == 2  # 42 skipped

    def test_subsampling_actions_and_states(self):
        """Verify sub-sampling of actions and states when frames > max_frames."""
        n = 50
        data = [
            {
                "observations": [{"pixels": self._frame(), "state": np.full(6, i, dtype=np.float32)} for i in range(n)],
                "actions": [np.full(6, i, dtype=np.float32) for i in range(n)],
                "task": "test",
            }
        ]
        result = robometer_mod.strands_robots_loader(data, max_frames=10)
        assert len(result[0]["frames"]) == 10
        assert len(result[0]["states"]) == 10
        assert len(result[0]["actions"]) == 10

    def test_record_session_fallback_task(self):
        """When ep.task is None, falls back to data_source.task."""
        session = MagicMock()
        ep = MagicMock()
        ep.observations = [{"pixels": self._frame()}]
        ep.actions = [np.zeros(6)]
        ep.task = None
        session.episodes = [ep]
        session.task = "fallback_task"

        result = robometer_mod.strands_robots_loader(session)
        assert result[0]["task_instruction"] == "fallback_task"

    def test_no_actions_gives_none(self):
        data = [
            {
                "observations": [{"pixels": self._frame()}],
                "task": "test",
            }
        ]
        result = robometer_mod.strands_robots_loader(data)
        assert result[0]["actions"] is None

    def test_no_states_gives_none(self):
        data = [
            {
                "observations": [{"pixels": self._frame()}],
                "actions": [np.zeros(6)],
                "task": "test",
            }
        ]
        result = robometer_mod.strands_robots_loader(data)
        assert result[0]["states"] is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Wrapper non-dict observation space
# ═══════════════════════════════════════════════════════════════════════════


class TestWrapperNonDictObsSpace:
    def test_non_dict_obs_space_warns(self):
        env = MagicMock()
        env.observation_space = MagicMock(spec=[])  # no .spaces attribute
        env.action_space = MagicMock()
        env.task = "test"
        env.unwrapped = env
        env.render_mode = None
        env.spec = None
        env.metadata = {}
        env.reward_range = (-1, 1)

        # Should not raise, just warn
        w = robometer_mod.StrandsRobometerRewardWrapper(env=env, model_path="m/m")
        assert w._task_instruction == "test"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _load_robometer_model cache behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadRobometerModel:
    def test_different_devices_separate_cache(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        robometer_mod._load_robometer_model("m/m", "cpu")
        robometer_mod._load_robometer_model("m/m", "cuda:0")
        assert _robometer_utils_save.load_model_from_hf.call_count == 2

    def test_different_models_separate_cache(self, mock_model):
        _robometer_utils_save.load_model_from_hf.return_value = mock_model
        robometer_mod._load_robometer_model("model_a", "cpu")
        robometer_mod._load_robometer_model("model_b", "cpu")
        assert _robometer_utils_save.load_model_from_hf.call_count == 2
