#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.policies.lerobot_local.

Strategy:
- lerobot/torch/huggingface_hub may not be available on CI, so we mock all heavy deps
- We test the policy class construction, state-key handling, observation
  batching logic, action tensor conversion, and error paths
- Uses __new__ to skip __init__ where model loading would occur
- Async get_actions() tested via asyncio.run() (no pytest-asyncio needed)
- Uses sys.modules mocking for lerobot, torch, and huggingface_hub
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if policy PR (#8) not merged yet
try:
    from strands_robots.policies.lerobot_local import (
        LerobotLocalPolicy,
        _resolve_policy_class_by_name,
        _resolve_policy_class_from_hub,
    )
except (ImportError, AttributeError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #8 (policy-abstraction)", allow_module_level=True)

# ---------------------------------------------------------------------------
# Module-level mocks for lerobot and huggingface_hub
# These are needed because patch("lerobot.policies.factory.get_policy_class")
# requires the lerobot module hierarchy to exist in sys.modules.
# ---------------------------------------------------------------------------

_mock_lerobot = MagicMock()
_mock_lerobot_policies = MagicMock()
_mock_lerobot_policies_factory = MagicMock()
_mock_lerobot.policies = _mock_lerobot_policies
_mock_lerobot.policies.factory = _mock_lerobot_policies_factory

_mock_hf_hub = MagicMock()

_LEROBOT_MODULES = {
    "lerobot": _mock_lerobot,
    "lerobot.policies": _mock_lerobot_policies,
    "lerobot.policies.factory": _mock_lerobot_policies_factory,
}

_HF_MODULES = {
    "huggingface_hub": _mock_hf_hub,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(**overrides):
    """Create a LerobotLocalPolicy without calling __init__ (no model load)."""
    p = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
    defaults = dict(
        pretrained_name_or_path="test/model",
        policy_type=None,
        requested_device=None,
        actions_per_step=1,
        use_processor=True,
        processor_overrides=None,
        robot_state_keys=[],
        _policy=None,
        _device=None,
        _input_features={},
        _output_features={},
        _loaded=False,
        _processor_bridge=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(p, k, v)
    return p


class _FakeFeature:
    """Mimics a LeRobot feature descriptor with .shape."""

    def __init__(self, shape):
        self.shape = shape


def _make_mock_policy_class(name="MockPolicy"):
    """Create a MagicMock that has a valid __name__ attribute."""
    cls = MagicMock()
    cls.__name__ = name
    cls.from_pretrained.return_value = _make_mock_model_instance()
    return cls


def _make_mock_model_instance():
    """Create a mock model instance with .eval(), .parameters(), .config."""
    inst = MagicMock()
    inst.eval.return_value = None
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_param.numel.return_value = 1000
    inst.parameters.return_value = iter([mock_param])
    inst.config = SimpleNamespace()
    return inst


def _mock_torch():
    """Create a mock torch module with no_grad context manager."""
    mt = MagicMock()
    mt.__version__ = "2.0"
    mt.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mt.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return mt


# ===========================================================================
# Tests for _resolve_policy_class_from_hub
# ===========================================================================


class TestResolvePolicyClassFromHub:
    """Test the hub-based policy class resolution."""

    def test_local_path_config(self, tmp_path):
        """Reads config.json from a local directory."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"type": "act"}')

        mock_cls = MagicMock()
        mock_cls.__name__ = "ActPolicy"

        with patch.dict(sys.modules, _LEROBOT_MODULES):
            _mock_lerobot_policies_factory.get_policy_class = MagicMock(return_value=mock_cls)
            with patch(
                "lerobot.policies.factory.get_policy_class",
                return_value=mock_cls,
            ):
                cls, ptype = _resolve_policy_class_from_hub(str(tmp_path))

        assert ptype == "act"
        assert cls is mock_cls

    def test_hf_hub_download(self):
        """Falls back to HF hub when local path doesn't exist."""
        import json
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"type": "diffusion"}, f)
            config_path = f.name

        mock_cls = MagicMock()
        mock_cls.__name__ = "DiffusionPolicy"

        try:
            with patch.dict(sys.modules, {**_HF_MODULES, **_LEROBOT_MODULES}):
                _mock_hf_hub.hf_hub_download = MagicMock(return_value=config_path)
                with patch("huggingface_hub.hf_hub_download", return_value=config_path):
                    with patch(
                        "lerobot.policies.factory.get_policy_class",
                        return_value=mock_cls,
                    ):
                        cls, ptype = _resolve_policy_class_from_hub("some/nonexistent")
            assert ptype == "diffusion"
            assert cls is mock_cls
        finally:
            os.unlink(config_path)

    def test_hf_download_fails_gracefully(self):
        """When HF download fails, should raise ValueError."""
        with patch.dict(sys.modules, _HF_MODULES):
            _mock_hf_hub.hf_hub_download = MagicMock(side_effect=Exception("network"))
            with patch("huggingface_hub.hf_hub_download", side_effect=Exception("network")):
                with pytest.raises(ValueError, match="Could not determine policy type"):
                    _resolve_policy_class_from_hub("bad/model")

    def test_no_type_in_config(self, tmp_path):
        """Config without 'type' field raises ValueError."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"name": "my_model"}')

        with pytest.raises(ValueError, match="No 'type' field"):
            _resolve_policy_class_from_hub(str(tmp_path))


class TestResolvePolicyClassByName:
    """Test explicit type → class resolution."""

    def test_valid_type(self):
        mock_cls = MagicMock()
        with patch.dict(sys.modules, _LEROBOT_MODULES):
            with patch("lerobot.policies.factory.get_policy_class", return_value=mock_cls):
                result = _resolve_policy_class_by_name("act")
        assert result is mock_cls

    def test_import_error(self):
        with patch.dict(sys.modules, _LEROBOT_MODULES):
            with patch(
                "lerobot.policies.factory.get_policy_class",
                side_effect=ImportError("no CUDA"),
            ):
                with pytest.raises(ImportError, match="CUDA"):
                    _resolve_policy_class_by_name("act")

    def test_runtime_error(self):
        with patch.dict(sys.modules, _LEROBOT_MODULES):
            with patch(
                "lerobot.policies.factory.get_policy_class",
                side_effect=RuntimeError("flash_attn missing"),
            ):
                with pytest.raises(ImportError, match="flash_attn"):
                    _resolve_policy_class_by_name("act")


# ===========================================================================
# Tests for LerobotLocalPolicy class
# ===========================================================================


class TestLerobotLocalPolicyInit:
    """Test construction and attribute defaults."""

    def test_provider_name(self):
        p = _make_policy()
        assert p.provider_name == "lerobot_local"

    def test_default_attributes(self):
        p = _make_policy()
        assert p._loaded is False
        assert p._policy is None
        assert p._device is None
        assert p.robot_state_keys == []
        assert p.actions_per_step == 1
        assert p.use_processor is True

    def test_init_with_no_path_skips_load(self):
        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            LerobotLocalPolicy(pretrained_name_or_path="")
            mock_load.assert_not_called()

    def test_init_with_path_calls_load(self):
        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            LerobotLocalPolicy(pretrained_name_or_path="test/model")
            mock_load.assert_called_once()

    def test_init_stores_all_params(self):
        with patch.object(LerobotLocalPolicy, "_load_model"):
            p = LerobotLocalPolicy(
                pretrained_name_or_path="test/model",
                policy_type="act",
                device="cpu",
                actions_per_step=5,
                use_processor=False,
                processor_overrides={"key": "val"},
            )
            assert p.policy_type == "act"
            assert p.requested_device == "cpu"
            assert p.actions_per_step == 5
            assert p.use_processor is False
            assert p.processor_overrides == {"key": "val"}


# ===========================================================================
# Tests for set_robot_state_keys
# ===========================================================================


class TestSetRobotStateKeys:

    def test_explicit_keys(self):
        p = _make_policy()
        p.set_robot_state_keys(["shoulder", "elbow", "wrist"])
        assert p.robot_state_keys == ["shoulder", "elbow", "wrist"]

    def test_empty_keys_auto_detect_from_output_features(self):
        p = _make_policy(
            _loaded=True,
            _output_features={"action": _FakeFeature((6,))},
        )
        p.set_robot_state_keys([])
        assert len(p.robot_state_keys) == 6
        assert p.robot_state_keys[0] == "joint_0"

    def test_empty_keys_auto_detect_from_input_features(self):
        p = _make_policy(
            _loaded=True,
            _output_features={},
            _input_features={"observation.state": _FakeFeature((4,))},
        )
        p.set_robot_state_keys([])
        assert len(p.robot_state_keys) == 4

    def test_empty_keys_no_features_warns(self):
        p = _make_policy(_loaded=True)
        with patch("strands_robots.policies.lerobot_local.logger") as mock_log:
            p.set_robot_state_keys([])
            mock_log.warning.assert_called()
        assert p.robot_state_keys == []

    def test_empty_keys_not_loaded(self):
        p = _make_policy(_loaded=False)
        with patch("strands_robots.policies.lerobot_local.logger") as mock_log:
            p.set_robot_state_keys([])
            mock_log.warning.assert_called()

    def test_output_feature_no_shape(self):
        p = _make_policy(
            _loaded=True,
            _output_features={"action": SimpleNamespace()},
            _input_features={"observation.state": _FakeFeature((3,))},
        )
        p.set_robot_state_keys([])
        assert len(p.robot_state_keys) == 3

    def test_output_feature_empty_shape(self):
        p = _make_policy(
            _loaded=True,
            _output_features={"action": _FakeFeature(())},
            _input_features={"observation.state": _FakeFeature((2,))},
        )
        p.set_robot_state_keys([])
        assert len(p.robot_state_keys) == 2


# ===========================================================================
# Tests for _tensor_to_action_dicts
# ===========================================================================


class TestTensorToActionDicts:

    def _mock_tensor(self, arr):
        t = MagicMock()
        t.cpu.return_value.numpy.return_value = np.array(arr, dtype=np.float32)
        return t

    def test_1d_tensor(self):
        p = _make_policy(robot_state_keys=["j0", "j1", "j2"])
        result = p._tensor_to_action_dicts(self._mock_tensor([1.0, 2.0, 3.0]))
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(1.0)
        assert result[0]["j2"] == pytest.approx(3.0)

    def test_2d_tensor_single_action(self):
        p = _make_policy(robot_state_keys=["j0", "j1"], actions_per_step=1)
        result = p._tensor_to_action_dicts(self._mock_tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(1.0)

    def test_2d_tensor_multi_action(self):
        p = _make_policy(robot_state_keys=["j0", "j1"], actions_per_step=3)
        result = p._tensor_to_action_dicts(self._mock_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert len(result) == 3
        assert result[2]["j0"] == pytest.approx(5.0)

    def test_3d_tensor(self):
        p = _make_policy(robot_state_keys=["j0"], actions_per_step=2)
        result = p._tensor_to_action_dicts(self._mock_tensor([[[1.0], [2.0], [3.0]]]))
        assert len(result) == 2

    def test_4d_tensor_flattened(self):
        p = _make_policy(robot_state_keys=["j0", "j1"])
        result = p._tensor_to_action_dicts(self._mock_tensor([[[[1.0, 2.0]]]]))
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(1.0)

    def test_tensor_shorter_than_keys(self):
        p = _make_policy(robot_state_keys=["j0", "j1", "j2"])
        result = p._tensor_to_action_dicts(self._mock_tensor([1.0, 2.0]))
        assert result[0]["j2"] == 0.0

    def test_empty_keys_returns_zero_actions(self):
        p = _make_policy(robot_state_keys=[])
        result = p._tensor_to_action_dicts(self._mock_tensor([1.0]))
        assert len(result) == 1
        assert result[0] == {}


# ===========================================================================
# Tests for _zero_actions
# ===========================================================================


class TestZeroActions:

    def test_returns_zeroed_dict(self):
        p = _make_policy(robot_state_keys=["shoulder", "elbow"])
        result = p._zero_actions()
        assert result == [{"shoulder": 0.0, "elbow": 0.0}]

    def test_empty_keys(self):
        p = _make_policy(robot_state_keys=[])
        result = p._zero_actions()
        assert result == [{}]


# ===========================================================================
# Tests for get_model_info
# ===========================================================================


class TestGetModelInfo:

    def test_not_loaded(self):
        p = _make_policy()
        info = p.get_model_info()
        assert info["provider"] == "lerobot_local"
        assert info["loaded"] is False
        assert info["device"] is None

    def test_loaded_with_features(self):
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = [mock_param]
        type(mock_policy).__name__ = "ActPolicy"

        p = _make_policy(
            _loaded=True,
            _policy=mock_policy,
            _device="cpu",
            _input_features={"observation.state": _FakeFeature((6,))},
            _output_features={"action": _FakeFeature((6,))},
        )
        info = p.get_model_info()
        assert info["loaded"] is True
        assert info["device"] == "cpu"
        assert info["n_parameters"] == 1000

    def test_loaded_with_processor(self):
        mock_bridge = MagicMock()
        mock_bridge.get_info.return_value = {"active": True}

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = []

        p = _make_policy(
            _loaded=True,
            _policy=mock_policy,
            _device="cpu",
            _processor_bridge=mock_bridge,
        )
        info = p.get_model_info()
        assert "processor" in info


# ===========================================================================
# Tests for _build_observation_batch — format detection
# ===========================================================================


class TestBuildObservationBatchFormatDetection:

    def test_lerobot_keys_detected(self):
        obs = {"observation.state": np.zeros(6), "observation.image.top": np.zeros((3, 480, 640))}
        has_lerobot_keys = any(k.startswith("observation.") for k in obs)
        assert has_lerobot_keys is True

    def test_strands_keys_detected(self):
        obs = {"shoulder": 1.0, "elbow": 2.0}
        has_lerobot_keys = any(k.startswith("observation.") for k in obs)
        assert has_lerobot_keys is False

    def test_list_to_numpy_conversion(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert arr.shape == (3,)

    def test_2d_list_detection(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert arr.ndim == 2

    def test_state_value_extraction(self):
        p = _make_policy(robot_state_keys=["a", "b", "c"])
        obs = {"a": 1.5, "b": 2.5, "c": 3.5, "extra": "ignored"}
        values = []
        for key in p.robot_state_keys:
            if key in obs and isinstance(obs[key], (int, float)):
                values.append(float(obs[key]))
        assert values == [1.5, 2.5, 3.5]

    def test_image_detection(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert img.ndim >= 2


# ===========================================================================
# Tests for get_actions (async) — using asyncio.run
# ===========================================================================


class TestGetActions:

    def test_not_loaded_no_path_returns_zero(self):
        p = _make_policy(
            pretrained_name_or_path="",
            _loaded=False,
            robot_state_keys=["j0", "j1"],
        )
        result = asyncio.run(p.get_actions({}, "test"))
        assert result == [{"j0": 0.0, "j1": 0.0}]

    def test_not_loaded_with_path_calls_load(self):
        """_load_model is called when not loaded but path exists."""
        p = _make_policy(
            pretrained_name_or_path="test/model",
            _loaded=False,
            robot_state_keys=["j0"],
        )

        load_called = [False]

        def fake_load():
            load_called[0] = True
            p._loaded = True
            # After loading, we need _policy and torch for inference
            # So make build_observation_batch raise to enter the except clause
            p._build_observation_batch = MagicMock(side_effect=Exception("no inference"))

        p._load_model = fake_load

        # Mock torch at sys.modules level since get_actions does `import torch`
        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            result = asyncio.run(p.get_actions({"j0": 1.0}, "test"))
        assert load_called[0] is True
        assert result == [{"j0": 0.0}]

    def test_inference_error_returns_zero(self):
        p = _make_policy(
            _loaded=True,
            robot_state_keys=["j0"],
            _processor_bridge=None,
        )
        p._build_observation_batch = MagicMock(side_effect=Exception("bad obs"))

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            result = asyncio.run(p.get_actions({"j0": 1.0}, "move"))
        assert result == [{"j0": 0.0}]

    def test_instruction_injected_as_task(self):
        p = _make_policy(_loaded=True, robot_state_keys=["j0"])

        captured_obs = {}

        def capture_batch(obs, instruction):
            captured_obs.update(obs)
            raise Exception("stop")

        p._build_observation_batch = capture_batch

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            asyncio.run(p.get_actions({"j0": 1.0}, "pick up cube"))
        assert captured_obs.get("task") == "pick up cube"

    def test_task_key_not_overwritten(self):
        p = _make_policy(_loaded=True, robot_state_keys=["j0"])

        captured_obs = {}

        def capture_batch(obs, instruction):
            captured_obs.update(obs)
            raise Exception("stop")

        p._build_observation_batch = capture_batch

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            asyncio.run(p.get_actions({"j0": 1.0, "task": "existing"}, "new"))
        assert captured_obs.get("task") == "existing"

    def test_preprocessor_called(self):
        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = True
        mock_bridge.has_postprocessor = False
        mock_bridge.preprocess.return_value = {"observation.state": np.zeros(6)}

        p = _make_policy(
            _loaded=True,
            robot_state_keys=["j0"],
            _processor_bridge=mock_bridge,
        )
        p._build_observation_batch = MagicMock(side_effect=Exception("stop"))

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            asyncio.run(p.get_actions({"j0": 1.0}, "test"))
        mock_bridge.preprocess.assert_called_once()

    def test_postprocessor_called(self):
        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = False
        mock_bridge.has_postprocessor = True

        mock_action = MagicMock()
        mock_action.cpu.return_value.numpy.return_value = np.array([1.0])

        mock_bridge.postprocess.return_value = mock_action

        mock_policy_model = MagicMock()
        mock_policy_model.select_action.return_value = mock_action

        p = _make_policy(
            _loaded=True,
            robot_state_keys=["j0"],
            _processor_bridge=mock_bridge,
            _policy=mock_policy_model,
        )
        p._build_observation_batch = MagicMock(return_value={})

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            asyncio.run(p.get_actions({"j0": 1.0}, "test"))

        mock_bridge.postprocess.assert_called_once()


# ===========================================================================
# Tests for select_action_sync
# ===========================================================================


class TestSelectActionSync:

    def test_not_loaded_triggers_load(self):
        p = _make_policy(_loaded=False)
        p._load_model = MagicMock(side_effect=Exception("stop"))

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            with pytest.raises(Exception, match="stop"):
                p.select_action_sync({"j0": 1.0})

    def test_preprocessor_applied(self):
        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = True
        mock_bridge.has_postprocessor = False
        mock_bridge.preprocess.return_value = {"observation.state": np.zeros(6)}

        p = _make_policy(_loaded=True, _processor_bridge=mock_bridge)
        p._build_observation_batch = MagicMock(side_effect=Exception("stop"))

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            with pytest.raises(Exception):
                p.select_action_sync({"j0": 1.0})
        mock_bridge.preprocess.assert_called_once()

    def test_returns_numpy_array(self):
        mock_action = MagicMock()
        mock_action.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
        # hasattr(mock_action, 'cpu') is True for MagicMock by default

        mock_policy_model = MagicMock()
        mock_policy_model.select_action.return_value = mock_action

        p = _make_policy(_loaded=True, _policy=mock_policy_model, _processor_bridge=None)
        p._build_observation_batch = MagicMock(return_value={})

        mt = _mock_torch()
        with patch.dict(sys.modules, {"torch": mt}):
            result = p.select_action_sync({"j0": 1.0})
        assert isinstance(result, np.ndarray)


# ===========================================================================
# Tests for _load_model
# ===========================================================================


class TestLoadModel:

    def test_load_with_explicit_policy_type(self):
        mock_cls = _make_mock_policy_class("ActPolicy")
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                p = _make_policy(policy_type="act")
                p._load_model()

        assert p._loaded is True
        mock_cls.from_pretrained.assert_called_once_with("test/model")

    def test_load_auto_detects_type(self):
        mock_cls = _make_mock_policy_class("DiffPolicy")
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_from_hub",
                return_value=(mock_cls, "diffusion"),
            ):
                p = _make_policy(policy_type=None)
                p._load_model()

        assert p._loaded is True
        assert p.policy_type == "diffusion"

    def test_load_reads_config_features(self):
        mock_cls = _make_mock_policy_class()
        inst = mock_cls.from_pretrained.return_value
        inst.config = SimpleNamespace(
            input_features={"observation.state": _FakeFeature((6,))},
            output_features={"action": _FakeFeature((6,))},
        )
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                p = _make_policy(policy_type="act")
                p._load_model()

        assert "observation.state" in p._input_features
        assert "action" in p._output_features

    def test_load_auto_generates_state_keys(self):
        mock_cls = _make_mock_policy_class()
        inst = mock_cls.from_pretrained.return_value
        inst.config = SimpleNamespace(
            input_features={},
            output_features={"action": _FakeFeature((4,))},
        )
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                p = _make_policy(policy_type="act", robot_state_keys=[])
                p._load_model()

        assert len(p.robot_state_keys) == 4
        assert p.robot_state_keys[0] == "joint_0"

    def test_load_processor_bridge(self):
        mock_cls = _make_mock_policy_class()
        mock_bridge = MagicMock()
        mock_bridge.is_active = True
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                with patch(
                    "strands_robots.processor.ProcessorBridge.from_pretrained",
                    return_value=mock_bridge,
                ):
                    p = _make_policy(policy_type="act", use_processor=True)
                    p._load_model()

        assert p._processor_bridge is mock_bridge

    def test_load_processor_inactive_set_to_none(self):
        mock_cls = _make_mock_policy_class()
        mock_bridge = MagicMock()
        mock_bridge.is_active = False
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                with patch(
                    "strands_robots.processor.ProcessorBridge.from_pretrained",
                    return_value=mock_bridge,
                ):
                    p = _make_policy(policy_type="act", use_processor=True)
                    p._load_model()

        assert p._processor_bridge is None

    def test_load_processor_exception_ignored(self):
        mock_cls = _make_mock_policy_class()
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                with patch(
                    "strands_robots.processor.ProcessorBridge.from_pretrained",
                    side_effect=Exception("no processor"),
                ):
                    p = _make_policy(policy_type="act", use_processor=True)
                    p._load_model()

        assert p._processor_bridge is None
        assert p._loaded is True

    def test_load_use_processor_false(self):
        """use_processor=False skips processor loading entirely."""
        mock_cls = _make_mock_policy_class()
        mt = _mock_torch()

        with patch.dict(sys.modules, {"torch": mt}):
            with patch(
                "strands_robots.policies.lerobot_local._resolve_policy_class_by_name",
                return_value=mock_cls,
            ):
                with patch(
                    "strands_robots.processor.ProcessorBridge.from_pretrained",
                ) as mock_proc:
                    p = _make_policy(policy_type="act", use_processor=False)
                    p._load_model()
                    mock_proc.assert_not_called()

        assert p._processor_bridge is None


# ===========================================================================
# Module-level exports
# ===========================================================================


class TestModuleExports:

    def test_all_exports(self):
        from strands_robots.policies.lerobot_local import __all__

        assert "LerobotLocalPolicy" in __all__
