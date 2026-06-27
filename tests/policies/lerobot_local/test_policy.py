"""Tests for strands_robots.policies.lerobot_local - LerobotLocalPolicy.

All tests run WITHOUT lerobot installed (pure mock/unit testing).
"""

import json
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies import create_policy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge
from strands_robots.policies.lerobot_local.resolution import (
    _read_policy_type_from_config,
    resolve_policy_class_by_name,
    resolve_policy_class_from_hub,
)
from strands_robots.registry import list_policy_providers

# (section)
# Helpers
# (section)


def _make_policy(**kwargs):
    """Create a LerobotLocalPolicy with model loading disabled."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(**kwargs)
    return policy


def _make_loaded_policy(action_dim=6, state_dim=6, device="cpu", include_images=True):
    """Create a LerobotLocalPolicy that appears loaded (mocked internals).

    Args:
        action_dim: Output action dimensions.
        state_dim: Input state dimensions.
        device: Torch device string.
        include_images: If True, include observation.images.top in input features.
            Set to False when tests don't provide image data to avoid the
            missing-image validation error.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device(device)

    action_feat = MagicMock()
    action_feat.shape = (action_dim,)
    policy._output_features = {"action": action_feat}

    state_feat = MagicMock()
    state_feat.shape = (state_dim,)
    input_features = {
        "observation.state": state_feat,
    }
    if include_images:
        input_features["observation.images.top"] = MagicMock(shape=(3, 480, 640))
    policy._input_features = input_features

    mock_lerobot_policy = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(1))
    mock_lerobot_policy.parameters.return_value = [mock_param]
    mock_lerobot_policy.select_action.return_value = torch.zeros(action_dim)
    policy._policy = mock_lerobot_policy

    return policy


# (section)
# Tests: Initialization
# (section)


class TestLerobotLocalInit:
    def test_init_without_path(self):
        """Creating without pretrained_name_or_path should not load model."""

        policy = LerobotLocalPolicy()
        assert policy._loaded is False
        assert policy.provider_name == "lerobot_local"
        assert policy.robot_state_keys == []

    def test_init_with_path_triggers_load(self):
        """Creating with pretrained_name_or_path should call _load_model."""

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            LerobotLocalPolicy(pretrained_name_or_path="lerobot/act_aloha_sim")
            mock_load.assert_called_once()

    def test_custom_actions_per_step(self):
        policy = _make_policy(actions_per_step=5)
        assert policy.actions_per_step == 5


class TestAutoDetectActionsPerStep:
    """`_auto_detect_actions_per_step` adopts the model's trained chunk size.

    Regression for the MolmoAct2 chunk-truncation bug: the SO-100/101
    checkpoints are trained for 30-step open-loop chunk replay
    (config.n_action_steps == 30), but the default actions_per_step=1 dropped
    29 of every 30 actions and re-queried vision out-of-distribution.
    """

    def _policy_with_config(self, n_action_steps):
        policy = _make_policy()
        mock_lerobot_policy = MagicMock()
        mock_lerobot_policy.config = types.SimpleNamespace(n_action_steps=n_action_steps)
        policy._policy = mock_lerobot_policy
        return policy

    def test_adopts_config_n_action_steps_when_default(self):
        """Default actions_per_step=1 + config.n_action_steps=30 -> 30."""
        policy = self._policy_with_config(30)
        assert policy.actions_per_step == 1  # default before detection
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 30

    def test_does_not_override_explicit_actions_per_step(self):
        """An explicit actions_per_step > 1 is never overridden."""
        policy = _make_policy(actions_per_step=4)
        policy._policy = MagicMock()
        policy._policy.config = types.SimpleNamespace(n_action_steps=30)
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 4

    def test_no_change_when_config_missing_n_action_steps(self):
        """A model without config.n_action_steps keeps the default 1."""
        policy = _make_policy()
        policy._policy = MagicMock()
        policy._policy.config = types.SimpleNamespace()  # no n_action_steps
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1

    def test_no_change_when_n_action_steps_is_one(self):
        """n_action_steps == 1 (closed-loop) leaves actions_per_step at 1."""
        policy = self._policy_with_config(1)
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1

    def test_no_policy_loaded_is_safe(self):
        """Calling with no loaded policy does not raise and keeps default."""
        policy = _make_policy()
        policy._policy = None
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1

    def _policy_with_ensemble(self, coeff, n_action_steps=100, actions_per_step=1):
        """A checkpoint that enables temporal ensembling (e.g. ACT folding).

        Mirrors LeRobot's ACTConfig: ``temporal_ensemble_coeff`` is set and
        ``n_action_steps`` is the full chunk (100). LeRobot forces
        ``n_action_steps == 1`` internally when ensembling, but the config still
        advertises the chunk size, so the detector must route on the coeff.
        """
        policy = _make_policy(actions_per_step=actions_per_step)
        mock_lerobot_policy = MagicMock()
        mock_lerobot_policy.config = types.SimpleNamespace(n_action_steps=n_action_steps, temporal_ensemble_coeff=coeff)
        policy._policy = mock_lerobot_policy
        return policy

    def test_ensembling_checkpoint_keeps_per_step_path(self):
        """temporal_ensemble_coeff set -> stay at actions_per_step=1.

        Regression: previously the detector bumped actions_per_step to
        config.n_action_steps (100 for ACT) for EVERY checkpoint, which routes
        get_actions() through predict_action_chunk() and silently bypasses the
        temporal ensembler. An ensembling checkpoint must stay per-step so
        select_action() runs the ensembler.
        """
        policy = self._policy_with_ensemble(0.01)
        assert policy.actions_per_step == 1
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1  # NOT auto-bumped to 100

    def test_ensembling_overrides_explicit_chunk_with_warning(self, caplog):
        """Explicit actions_per_step > 1 on an ensembling checkpoint is

        overridden to 1 with a loud warning - chunk replay and ensembling are
        mutually exclusive, and silently dropping the smoothing is the bug.
        """
        import logging

        policy = self._policy_with_ensemble(0.01, actions_per_step=8)
        with caplog.at_level(logging.WARNING):
            policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1
        assert any(
            "temporal ensembling" in r.message and "actions_per_step" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_non_ensembling_act_still_chunk_replays(self):
        """temporal_ensemble_coeff=None -> open-loop chunk replay (unchanged)."""
        policy = self._policy_with_ensemble(None, n_action_steps=100)
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 100


# (section)
# Tests: set_robot_state_keys
# (section)


class TestSetRobotStateKeys:
    def test_explicit_keys(self):
        policy = _make_policy()
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist"])
        assert policy.robot_state_keys == ["shoulder", "elbow", "wrist"]

    def test_empty_keys_auto_detect_from_output_features(self):
        policy = _make_loaded_policy(action_dim=7)
        policy.robot_state_keys = []
        policy.set_robot_state_keys([])
        assert len(policy.robot_state_keys) == 7
        assert policy.robot_state_keys[0] == "joint_0"

    def test_empty_keys_fallback_to_input_features(self):
        policy = _make_loaded_policy(state_dim=4)
        policy._output_features = {}
        policy.robot_state_keys = []
        policy.set_robot_state_keys([])
        assert len(policy.robot_state_keys) == 4

    def test_empty_keys_no_features_raises(self):
        """Empty keys with no model features should raise ValueError."""
        policy = _make_policy()
        policy._loaded = True
        policy._output_features = {}
        policy._input_features = {}
        with pytest.raises(ValueError, match="robot_state_keys is empty"):
            policy.set_robot_state_keys([])


# (section)
# Tests: Tokenizer resolution (VLA support)
# (section)


class TestResolveTokenizer:
    def test_tokenizer_from_tokenizer_name_falls_to_processor(self):
        """Strategy 1 (tokenizer_name) falls through when transformers missing, lands on Strategy 3."""
        policy = _make_loaded_policy()
        mock_tok = MagicMock()
        policy._policy.config = MagicMock(
            tokenizer_name="mock-tokenizer",
            vlm_model_name=None,
            tokenizer_max_length=64,
            tokenizer_padding_side="left",
        )
        policy._policy.processor = MagicMock()
        policy._policy.processor.tokenizer = mock_tok
        result = policy._resolve_tokenizer()
        assert result is mock_tok
        assert policy._tokenizer is mock_tok

    def test_tokenizer_from_processor_builtin(self):
        """Strategy 3: policy.processor.tokenizer."""
        policy = _make_loaded_policy()
        mock_tok = MagicMock()
        policy._policy.config = MagicMock(
            tokenizer_name=None,
            vlm_model_name=None,
            tokenizer_max_length=48,
            tokenizer_padding_side="right",
        )
        policy._policy.processor = MagicMock()
        policy._policy.processor.tokenizer = mock_tok
        result = policy._resolve_tokenizer()
        assert result is mock_tok
        assert mock_tok.padding_side == "right"

    def test_returns_none_when_no_tokenizer_available(self):
        """No tokenizer_name, no vlm_model_name, no processor.tokenizer."""
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(
            tokenizer_name=None,
            vlm_model_name=None,
            tokenizer_max_length=48,
            tokenizer_padding_side="right",
        )
        policy._policy.processor = None
        result = policy._resolve_tokenizer()
        assert result is None


class TestTokenizeInstruction:
    def test_returns_none_without_tokenizer(self):
        policy = _make_loaded_policy()
        policy._policy.config = None
        assert policy._tokenize_instruction("pick up") is None

    def test_returns_none_for_empty_instruction(self):
        policy = _make_loaded_policy()
        policy._tokenizer = MagicMock()
        assert policy._tokenize_instruction("") is None

    def test_tokenizes_and_transfers_to_device(self):
        policy = _make_loaded_policy()
        policy._device = torch.device("cpu")
        policy._tokenizer_max_length = 32

        mock_ids = MagicMock()
        mock_ids.to.return_value = mock_ids
        mock_mask = MagicMock()
        mock_mask.bool.return_value = mock_mask
        mock_mask.to.return_value = mock_mask

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": mock_ids, "attention_mask": mock_mask}
        policy._tokenizer = mock_tok

        result = policy._tokenize_instruction("pick up the cube")
        assert result is not None
        tokens, mask = result
        assert tokens is mock_ids
        assert mask is mock_mask
        mock_tok.assert_called_once_with(
            "pick up the cube",
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )

    def test_handles_missing_attention_mask(self):
        policy = _make_loaded_policy()
        policy._device = torch.device("cpu")
        mock_ids = MagicMock()
        mock_ids.to.return_value = mock_ids
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": mock_ids}
        policy._tokenizer = mock_tok

        tokens, mask = policy._tokenize_instruction("test")
        assert mask is None


class TestNeedsLanguageTokens:
    def test_tokenizer_name_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name="gpt2", vlm_model_name=None)
        assert policy._needs_language_tokens() is True

    def test_vlm_model_name_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name="smolvlm")
        assert policy._needs_language_tokens() is True

    def test_language_input_feature_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name=None)
        policy._input_features["observation.language.tokens"] = MagicMock()
        assert policy._needs_language_tokens() is True

    def test_no_language_indicators_returns_false(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name=None)
        assert policy._needs_language_tokens() is False


# (section)
# Tests: _load_model
# (section)


class TestLoadModel:
    def test_load_with_explicit_policy_type(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(
            input_features={"observation.state": MagicMock(shape=(6,))},
            output_features={"action": MagicMock(shape=(6,))},
            device="cpu",
        )
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy._loaded is True
        assert policy._device == torch.device("cpu")

    def test_load_without_policy_type_resolves_from_hub(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(spec=[])
        mock_inner.eval.return_value = None
        mock_inner.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_from_hub",
            return_value=(mock_policy_cls, "diffusion"),
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy.policy_type == "diffusion"
        assert policy._loaded is True

    def test_device_from_config(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(device="cpu", spec=["device"])
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy._device == torch.device("cpu")

    def test_auto_generates_state_keys_from_output(self):

        action_feat = MagicMock()
        action_feat.shape = (4,)
        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(
            device="cpu",
            input_features={},
            output_features={"action": action_feat},
        )
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy.robot_state_keys == ["joint_0", "joint_1", "joint_2", "joint_3"]


# (section)
# Tests: get_actions (async)
# (section)


class TestGetActions:
    def test_not_loaded_triggers_load(self):

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            policy = LerobotLocalPolicy()
            policy.pretrained_name_or_path = "test/model"

            def fake_load():
                policy._loaded = True
                policy._device = "cpu"
                mock_inner = MagicMock()
                mock_inner.select_action.return_value = torch.zeros(6)
                policy._policy = mock_inner
                policy._output_features = {}
                policy._input_features = {}
                policy.robot_state_keys = [f"j{i}" for i in range(6)]

            mock_load.side_effect = fake_load
            policy.get_actions_sync({}, "test")
            mock_load.assert_called()

    def test_returns_list_of_dicts(self):
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        actions = policy.get_actions_sync({}, "test")
        assert isinstance(actions, list)
        assert all(isinstance(action, dict) for action in actions)

    def test_action_keys_match_state_keys(self):
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        actions = policy.get_actions_sync({}, "pick up")
        assert set(actions[0].keys()) == {"shoulder", "elbow", "gripper"}

    def test_no_path_raises_runtime_error(self):

        policy = LerobotLocalPolicy()
        policy.robot_state_keys = ["a", "b"]
        with pytest.raises(RuntimeError, match="No model loaded"):
            policy.get_actions_sync({}, "test")

    def test_inference_error_raises(self):
        """Inference errors should propagate immediately."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy._policy.select_action.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            policy.get_actions_sync({}, "test")

    def test_multi_step_uses_predict_action_chunk(self):
        """actions_per_step > 1 should call predict_action_chunk for full chunk."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 4

        # predict_action_chunk returns (batch, horizon, action_dim)
        policy._policy.predict_action_chunk.return_value = torch.zeros(1, 10, 3)

        actions = policy.get_actions_sync({}, "test")

        # Should have called predict_action_chunk, NOT select_action
        policy._policy.predict_action_chunk.assert_called_once()
        policy._policy.select_action.assert_not_called()
        assert len(actions) == 4

    def test_single_step_uses_select_action(self):
        """actions_per_step=1 should use select_action for temporal ensemble."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 1

        actions = policy.get_actions_sync({}, "test")

        policy._policy.select_action.assert_called_once()
        assert len(actions) == 1

    def test_molmoact2_bypasses_select_action_at_single_step(self):
        """MolmoAct2 must use predict_action_chunk even at actions_per_step=1.

        Its select_action() raises AssertionError when the checkpoint's
        rtc_config is enabled, so routing single-step inference through it would
        crash. Detection is by the LeRobot policy ``name`` attribute.
        """
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 1
        # LeRobot sets PreTrainedPolicy.name = "molmoact2".
        policy._policy.name = "molmoact2"
        # If select_action were called it would crash, mirroring the real policy.
        policy._policy.select_action.side_effect = AssertionError(
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )
        policy._policy.predict_action_chunk.return_value = torch.zeros(1, 8, 3)

        actions = policy.get_actions_sync({}, "test")

        policy._policy.predict_action_chunk.assert_called_once()
        policy._policy.select_action.assert_not_called()
        assert len(actions) == 1

    def test_molmoact2_detected_by_class_name_fallback(self):
        """A stub without a ``name`` attr is detected by its class name."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 1

        class MolmoAct2Policy:
            def __init__(self):
                self.predict_action_chunk = MagicMock(return_value=torch.zeros(1, 8, 3))
                self.select_action = MagicMock(side_effect=AssertionError("RTC is not supported for select_action"))

            def eval(self):
                return self

        stub = MolmoAct2Policy()
        policy._policy = stub

        actions = policy.get_actions_sync({}, "test")

        stub.predict_action_chunk.assert_called_once()
        stub.select_action.assert_not_called()
        assert len(actions) == 1

    def test_requires_action_chunk_false_without_policy(self):
        """No loaded policy -> not chunk-required (no crash)."""
        policy = LerobotLocalPolicy()
        assert policy._requires_action_chunk() is False

    def test_non_molmoact2_single_step_still_uses_select_action(self):
        """A regular policy at actions_per_step=1 keeps the select_action path."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 1
        policy._policy.name = "act"

        actions = policy.get_actions_sync({}, "test")

        policy._policy.select_action.assert_called_once()
        policy._policy.predict_action_chunk.assert_not_called()
        assert len(actions) == 1

    def test_ensembling_act_routes_to_select_action_end_to_end(self):
        """ACT ensembling checkpoint: detection -> get_actions runs select_action.

        Regression for the silent ensembling-disable bug. The checkpoint
        advertises n_action_steps=100 and temporal_ensemble_coeff=0.01. Before
        the fix, auto-detect bumped actions_per_step to 100 and get_actions()
        called predict_action_chunk() (bypassing the ensembler). After the fix,
        actions_per_step stays 1 and get_actions() calls select_action() so the
        temporal ensembler runs.
        """
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy._policy.name = "act"
        policy._policy.config = types.SimpleNamespace(n_action_steps=100, temporal_ensemble_coeff=0.01)

        # Simulate the load-time regime selection.
        policy._auto_detect_actions_per_step()
        assert policy.actions_per_step == 1

        actions = policy.get_actions_sync({}, "fold the shirt")

        policy._policy.select_action.assert_called_once()
        policy._policy.predict_action_chunk.assert_not_called()
        assert len(actions) == 1

    def test_processor_bridge_preprocess_bypasses_batch_builder(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = True
        mock_bridge.has_postprocessor = False
        mock_bridge.preprocess.return_value = {
            "observation.state": torch.zeros(1, 3),
        }
        policy._processor_bridge = mock_bridge

        with patch.object(policy, "_build_observation_batch") as mock_build:
            policy.get_actions_sync({"state": [0, 0, 0]}, "test")
            mock_build.assert_not_called()

        mock_bridge.preprocess.assert_called_once()

    def test_processor_bridge_postprocess_applied(self):
        policy = _make_loaded_policy(action_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b"])

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = False
        mock_bridge.has_postprocessor = True
        mock_bridge.postprocess.return_value = torch.tensor([10.0, 20.0])
        policy._processor_bridge = mock_bridge

        actions = policy.get_actions_sync({}, "test")
        mock_bridge.postprocess.assert_called_once()
        assert actions[0]["a"] == 10.0
        assert actions[0]["b"] == 20.0


# (section)
# Tests: _build_observation_batch
# (section)


class TestBuildObservationBatch:
    def test_lerobot_format_passthrough(self):
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch

    def test_numpy_state_conversion(self):
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": np.array([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        assert isinstance(batch["observation.state"], torch.Tensor)

    def test_image_hwc_to_chw_conversion(self):
        policy = _make_loaded_policy()
        observation = {
            "observation.images.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)
        assert batch["observation.images.top"].max() <= 1.0

    def test_strands_format_state_mapping(self):
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        observation = {"shoulder": 0.5, "elbow": -0.3, "gripper": 1.0}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        state = batch["observation.state"]
        assert state.shape == (1, 3)
        assert abs(state[0, 0].item() - 0.5) < 1e-6

    def test_missing_image_features_filled_with_zeros(self):
        """Missing camera images should raise ValueError (fail-fast)."""
        policy = _make_loaded_policy()  # includes images in _input_features
        # Use lerobot-format keys so it goes through _build_batch_from_lerobot_format
        observation = {"observation.state": torch.zeros(6)}
        with pytest.raises(ValueError, match="Missing required image feature"):
            policy._build_observation_batch(observation, "test")

    def test_scalar_int_conversion(self):
        policy = _make_loaded_policy(include_images=False)
        observation = {"observation.gripper": 1}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.gripper" in batch
        assert batch["observation.gripper"].shape == (1, 1)

    def test_float64_tensor_auto_cast_to_float32(self):
        """float64 tensors from ROS/dynamixel drivers should be auto-cast to float32."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.state"].dtype == torch.float32

    def test_float64_numpy_auto_cast_to_float32(self):
        """float64 numpy arrays should be auto-cast to float32."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.state"].dtype == torch.float32


class TestBuildBatchFromLerobotFormat:
    """LeRobot-native (observation.*) batch building: image/list edge cases.

    These pin the conversion contract of ``_build_batch_from_lerobot_format``
    for inputs that arrive already prefixed with ``observation.`` but are not
    explicitly declared as image features - the builder must still detect
    channel-last images by shape, lay them out channel-first with a batch
    dimension, normalize uint8 to [0, 1], and route Python lists/tuples
    (numeric to tensors, non-numeric skipped) without raising.
    """

    def test_unlabeled_torch_image_detected_by_shape(self):
        """A 3D channel-last tensor on a non-image key is treated as an image."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        # Key has no "image" substring and is not a declared image feature.
        observation = {"observation.cam0": torch.zeros(48, 64, 3)}
        batch = policy._build_batch_from_lerobot_format(observation, {})
        # HWC -> CHW + batch dim.
        assert batch["observation.cam0"].shape == (1, 3, 48, 64)

    def test_unlabeled_numpy_uint8_image_normalized(self):
        """A uint8 channel-last numpy array is detected and scaled to [0, 1]."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        img = np.full((48, 64, 3), 255, dtype=np.uint8)
        observation = {"observation.cam1": img}
        batch = policy._build_batch_from_lerobot_format(observation, {})
        assert batch["observation.cam1"].shape == (1, 3, 48, 64)
        assert float(batch["observation.cam1"].max()) == 1.0

    def test_numeric_list_state_becomes_batched_tensor(self):
        """A 1D numeric list gains a batch dimension."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        observation = {"observation.state": [1.0, 2.0, 3.0]}
        batch = policy._build_batch_from_lerobot_format(observation, {})
        assert batch["observation.state"].shape == (1, 3)
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.0, 2.0, 3.0], atol=1e-6)

    def test_numeric_nested_list_image_laid_out_channel_first(self):
        """A nested numeric list shaped HWC is permuted to CHW with a batch dim."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        # Shape (2, 2, 3) channel-last -> (1, 3, 2, 2).
        observation = {"observation.images.x": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]}
        batch = policy._build_batch_from_lerobot_format(observation, {})
        assert batch["observation.images.x"].shape == (1, 3, 2, 2)

    def test_non_numeric_list_is_skipped_not_raised(self):
        """A list of strings (e.g. task names) is dropped, never crashes the build."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        observation = {
            "observation.state": torch.zeros(6),
            "observation.task_names": ["pick", "place"],
        }
        batch = policy._build_batch_from_lerobot_format(observation, {})
        assert "observation.task_names" not in batch
        assert "observation.state" in batch

    def test_instruction_fills_declared_task_feature(self):
        """When the model declares a 'task' input, the instruction is injected."""
        policy = _make_loaded_policy(state_dim=6, include_images=False)
        task_feat = MagicMock()
        policy._input_features["task"] = task_feat
        observation = {"observation.state": torch.zeros(6)}
        batch = policy._build_observation_batch(observation, "pick up the cube")
        assert batch["task"] == "pick up the cube"


# (section)
# Tests: _build_batch_from_strands_format
# (section)


class TestBuildBatchFromStrandsFormat:
    def test_numpy_floating_state(self):
        policy = _make_loaded_policy(state_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": np.float32(1.5), "b": np.float64(2.5)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.5, 2.5], atol=1e-5)

    def test_state_padded_to_expected_dim(self):
        """State dimension mismatch should auto-pad (not raise)."""
        policy = _make_loaded_policy(state_dim=4, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": 1.0, "b": 2.0}
        # After bug fix: auto-pads with zeros instead of raising
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"][0].numpy()
        assert len(state) == 4
        np.testing.assert_allclose(state[:2], [1.0, 2.0], atol=1e-5)
        np.testing.assert_allclose(state[2:], [0.0, 0.0], atol=1e-5)

    def test_state_truncated_to_expected_dim(self):
        """A robot exposing MORE joints than the model was trained on truncates
        the leading values to the model's expected state dim (not raise).

        Mirrors the documented aloha-16-joints / ACT-expects-14 case: the policy
        adapts the observation rather than crashing the rollout.
        """
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c", "d"])
        observation = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"][0].numpy()
        assert len(state) == 2
        np.testing.assert_allclose(state, [1.0, 2.0], atol=1e-5)

    def test_zero_dim_ndarray_state_value(self):
        """A joint value supplied as a 0-d numpy array is coerced to a scalar
        float and packed into the state tensor (not silently dropped)."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": np.array(1.5), "b": np.array(2.5)}
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"][0].numpy()
        assert len(state) == 2
        np.testing.assert_allclose(state, [1.5, 2.5], atol=1e-5)

    def test_empty_state_keys_raises(self):
        """Empty robot_state_keys should raise ValueError."""
        policy = _make_loaded_policy()
        policy.robot_state_keys = []
        with pytest.raises(ValueError, match="robot_state_keys is empty"):
            policy._build_batch_from_strands_format({"x": 1.0}, {})


# (section)
# Tests: camera-key routing (config.image_keys / camera_key_map)
# (section)


def _make_two_camera_policy():
    """Loaded policy whose model declares two image inputs (top, wrist)."""
    policy = _make_loaded_policy(state_dim=2, include_images=False)
    policy.set_robot_state_keys(["a", "b"])
    img = MagicMock(shape=(3, 480, 640))
    # Declaration order: top before wrist.
    policy._input_features["observation.images.top"] = img
    policy._input_features["observation.images.wrist"] = img
    return policy


class TestCameraKeyRouting:
    def test_exact_name_match_routes_by_name_not_position(self):
        """A 'wrist' cam must land in the wrist slot even when iterated last."""
        policy = _make_two_camera_policy()
        top = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Insertion order deliberately reversed vs declaration order.
        observation = {"a": 1.0, "b": 2.0, "wrist": wrist, "top": top}
        batch = policy._build_batch_from_strands_format(observation, {})
        # wrist (all-255 -> 1.0) must be in the wrist slot, top (zeros) in top.
        assert float(batch["observation.images.wrist"].max()) == 1.0
        assert float(batch["observation.images.top"].max()) == 0.0

    def test_config_image_keys_preferred_over_input_features_order(self):
        """config.image_keys ordering wins over _input_features declaration order."""
        policy = _make_two_camera_policy()
        # Reverse the model's declared ordering via config.image_keys.
        policy._policy.config.image_keys = [
            "observation.images.wrist",
            "observation.images.top",
        ]
        keys = policy._policy_image_keys()
        assert keys == ["observation.images.wrist", "observation.images.top"]

    def test_explicit_camera_key_map_overrides_naming(self):
        """camera_key_map binds a mismatched cam name to the intended slot."""
        policy = _make_two_camera_policy()
        policy.camera_key_map = {
            "front": "observation.images.top",
            "hand": "observation.images.wrist",
        }
        front = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = np.ones((480, 640, 3), dtype=np.uint8) * 255
        observation = {"a": 1.0, "b": 2.0, "front": front, "hand": hand}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert float(batch["observation.images.top"].max()) == 0.0
        assert float(batch["observation.images.wrist"].max()) == 1.0

    def test_positional_fallback_warns_on_name_mismatch(self, caplog):
        """Mismatched cam names fall back positionally but log a WARNING."""
        import logging

        policy = _make_two_camera_policy()
        side = np.zeros((480, 640, 3), dtype=np.uint8)
        other = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = {"a": 1.0, "b": 2.0, "side": side, "other": other}
        with caplog.at_level(logging.WARNING):
            batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.images.top" in batch
        assert "observation.images.wrist" in batch
        assert any("routing positionally" in r.message for r in caplog.records)

    def test_camera_key_map_to_undeclared_key_raises(self):
        policy = _make_two_camera_policy()
        policy.camera_key_map = {"top": "observation.images.nonexistent"}
        observation = {"a": 1.0, "b": 2.0, "top": np.zeros((480, 640, 3), dtype=np.uint8)}
        with pytest.raises(ValueError, match="does not declare it"):
            policy._build_batch_from_strands_format(observation, {})

    def test_fewer_cameras_than_policy_needs_raises(self):
        """One camera for a two-camera policy is a hard error, not silent."""
        policy = _make_two_camera_policy()
        observation = {"a": 1.0, "b": 2.0, "top": np.zeros((480, 640, 3), dtype=np.uint8)}
        with pytest.raises(ValueError, match="requires image input"):
            policy._build_batch_from_strands_format(observation, {})

    def test_extra_cameras_are_dropped(self):
        """A single-cam policy ignores extra cameras the robot provides."""
        policy = _make_loaded_policy(state_dim=2)  # declares observation.images.top
        policy.set_robot_state_keys(["a", "b"])
        observation = {
            "a": 1.0,
            "b": 2.0,
            "top": np.zeros((480, 640, 3), dtype=np.uint8),
            "extra": np.ones((480, 640, 3), dtype=np.uint8),
        }
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.images.top" in batch
        assert len([k for k in batch if "image" in k]) == 1


class TestStrictKeysCameraRouting:
    """strict_keys=True raises instead of routing cameras positionally.

    Default (strict_keys=False) keeps the positional-fallback-with-warning
    behavior unchanged; strict_keys=True turns an unresolved camera name into
    a hard ValueError naming the unmatched vs available keys.
    """

    def test_strict_raises_on_name_mismatch(self):
        policy = _make_two_camera_policy()
        policy.strict_keys = True
        side = np.zeros((480, 640, 3), dtype=np.uint8)
        other = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = {"a": 1.0, "b": 2.0, "side": side, "other": other}
        with pytest.raises(ValueError, match="strict_keys=True: cannot resolve camera keys"):
            policy._build_batch_from_strands_format(observation, {})

    def test_strict_error_lists_keys(self):
        policy = _make_two_camera_policy()
        policy.strict_keys = True
        observation = {
            "a": 1.0,
            "b": 2.0,
            "side": np.zeros((480, 640, 3), dtype=np.uint8),
            "other": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        with pytest.raises(ValueError) as exc:
            policy._build_batch_from_strands_format(observation, {})
        msg = str(exc.value)
        assert "side" in msg and "other" in msg  # unmatched robot keys
        assert "observation.images.top" in msg  # available model key

    def test_nonstrict_default_still_maps_positionally_and_warns(self, caplog):
        import logging

        policy = _make_two_camera_policy()
        assert policy.strict_keys is False  # default
        observation = {
            "a": 1.0,
            "b": 2.0,
            "side": np.zeros((480, 640, 3), dtype=np.uint8),
            "other": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        with caplog.at_level(logging.WARNING):
            batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.images.top" in batch
        assert "observation.images.wrist" in batch
        assert any("routing positionally" in r.message for r in caplog.records)

    def test_strict_exact_name_match_never_raises(self):
        policy = _make_two_camera_policy()
        policy.strict_keys = True
        top = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist = np.ones((480, 640, 3), dtype=np.uint8) * 255
        observation = {"a": 1.0, "b": 2.0, "top": top, "wrist": wrist}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert float(batch["observation.images.wrist"].max()) == 1.0
        assert float(batch["observation.images.top"].max()) == 0.0

    def test_strict_camera_key_map_satisfies_routing(self):
        policy = _make_two_camera_policy()
        policy.strict_keys = True
        policy.camera_key_map = {
            "front": "observation.images.top",
            "hand": "observation.images.wrist",
        }
        front = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = np.ones((480, 640, 3), dtype=np.uint8) * 255
        observation = {"a": 1.0, "b": 2.0, "front": front, "hand": hand}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert float(batch["observation.images.top"].max()) == 0.0
        assert float(batch["observation.images.wrist"].max()) == 1.0


# (section)
# Tests: _tensor_to_action_dicts
# (section)


class TestTensorToActionDicts:
    def test_1d_tensor(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        result = policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0, 3.0]))
        assert len(result) == 1
        assert result[0] == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_2d_tensor_respects_actions_per_step(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["x", "y"])
        policy.actions_per_step = 2
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 2
        assert result[0] == {"x": 1.0, "y": 2.0}
        assert result[1] == {"x": 3.0, "y": 4.0}

    def test_empty_state_keys_raises(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.robot_state_keys = []
        with pytest.raises(RuntimeError, match="robot_state_keys is empty"):
            policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0]))


# (section)
# Tests: reset
# (section)


class TestReset:
    def test_reset_delegates_to_inner_policy(self):
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock()
        policy._processor_bridge = None
        policy.reset()
        policy._policy.reset.assert_called_once()

    def test_reset_safe_when_not_loaded(self):
        policy = _make_policy()
        assert policy._policy is None
        policy.reset()  # Should not raise


# (section)
# Tests: Policy resolution helpers
# (section)


class TestPolicyResolution:
    def test_resolve_policy_class_by_name_raises_for_unknown(self):

        with pytest.raises((ImportError, ValueError)):
            resolve_policy_class_by_name("nonexistent_policy_type_xyz")

    def test_resolve_policy_class_by_name_degrades_when_factory_import_raises_typeerror(self, monkeypatch):
        """A broken optional lerobot policy config must not leak a TypeError.

        ``lerobot.policies.factory`` eagerly imports every optional policy
        config module. Under transformers 5.x (where ``PretrainedConfig`` is a
        dataclass) a config such as ``GR00TN15Config`` - which declares a
        ``field(init=False)`` without a default - fails to build at import
        time with ``TypeError: non-default argument 'backbone_cfg' follows
        default argument 'problem_type'``. ``resolve_policy_class_by_name``
        must treat that strategy as unavailable and surface a clean
        ``ImportError``, never the raw ``TypeError``.
        """
        import sys

        lerobot_stub = types.ModuleType("lerobot")
        lerobot_stub.__path__ = []  # type: ignore[attr-defined]
        policies_stub = types.ModuleType("lerobot.policies")
        policies_stub.__path__ = []  # type: ignore[attr-defined]

        class _BrokenFactory(types.ModuleType):
            def __getattr__(self, name):
                raise TypeError("non-default argument 'backbone_cfg' follows default argument 'problem_type'")

        monkeypatch.setitem(sys.modules, "lerobot", lerobot_stub)
        monkeypatch.setitem(sys.modules, "lerobot.policies", policies_stub)
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", _BrokenFactory("lerobot.policies.factory"))

        with pytest.raises(ImportError):
            resolve_policy_class_by_name("nonexistent_policy_type_xyz")

    def test_resolve_from_hub_raises_without_type(self):

        with pytest.raises((ValueError, ImportError, Exception)):
            resolve_policy_class_from_hub("completely/fake-model-path-that-does-not-exist")

    def test_resolve_by_name_modeling_submodule(self):

        mock_policy_class = type("ACTPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})
        mock_module = types.ModuleType("lerobot.policies.act.modeling_act")
        mock_module.ACTPolicy = mock_policy_class

        with patch("importlib.import_module", return_value=mock_module):
            result = resolve_policy_class_by_name("act")
            assert result is mock_policy_class

    def test_read_policy_type_from_local_config(self, tmp_path):

        config_dir = tmp_path / "model"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"type": "act"}))

        result = _read_policy_type_from_config(str(config_dir))
        assert result == "act"


# (section)
# Tests: Registry integration
# (section)


class TestRegistryIntegration:
    def test_lerobot_local_in_registry(self):

        providers = list_policy_providers()
        assert "lerobot_local" in providers

    def test_create_policy_lerobot_local_without_model(self, monkeypatch):
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
        policy = create_policy("lerobot_local")
        assert policy.provider_name == "lerobot_local"
        assert policy._loaded is False


# (section)
# Tests: ProcessorBridge
# (section)


class TestProcessorBridge:
    def test_preprocess_raises_on_pipeline_error(self):
        """preprocess() wraps pipeline exceptions in RuntimeError.

        The production code calls _preprocessor._forward(transition) after
        building a transition via create_transition(). We mock _forward to
        raise and patch the lerobot imports so the transition-building path
        is exercised regardless of whether lerobot is installed.
        """
        mock_pre = MagicMock()
        mock_pre._forward.side_effect = ValueError("bad data")
        bridge = ProcessorBridge(preprocessor=mock_pre)

        # Patch the lerobot imports used inside preprocess()
        mock_create_transition = MagicMock(return_value={"observation": {}})
        mock_transition_key = MagicMock()
        mock_transition_key.OBSERVATION = "observation"

        with patch(
            "strands_robots.policies.lerobot_local.processor.create_transition",
            mock_create_transition,
            create=True,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.processor.TransitionKey",
                mock_transition_key,
                create=True,
            ):
                with pytest.raises(RuntimeError, match="Preprocessor pipeline failed"):
                    bridge.preprocess({})

    def test_postprocess_raises_on_pipeline_error(self):

        mock_post = MagicMock()
        mock_post.process_action.side_effect = ValueError("bad action")
        bridge = ProcessorBridge(postprocessor=mock_post)

        with pytest.raises(RuntimeError, match="Postprocessor pipeline failed"):
            bridge.postprocess(torch.zeros(2))

    def test_from_pretrained_passthrough_when_no_lerobot(self):

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge.from_pretrained("test/model")
            assert not bridge.is_active


# ===========================================================================
# Tests: RTC (Real-Time Chunking)
# ===========================================================================


class TestRTCInit:
    """Tests for RTC initialization and auto-detection."""

    def test_rtc_default_disabled_when_no_predict_chunk(self):
        """RTC should be disabled when policy has no predict_action_chunk."""
        policy = _make_policy()
        mock_policy = MagicMock(spec=["reset", "eval", "parameters", "select_action"])
        mock_policy.config = MagicMock()
        mock_policy.config.device = "cpu"
        mock_policy.config.input_features = {}
        mock_policy.config.output_features = {}
        policy._policy = mock_policy
        policy._loaded = True
        policy._init_rtc()
        assert policy._rtc_enabled is False

    def test_rtc_auto_detect_from_config(self):
        """RTC should auto-detect from model config.rtc_config.enabled."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        rtc_cfg.execution_horizon = 15
        rtc_cfg.max_guidance_weight = 8.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = None  # auto-detect
        policy._init_rtc()
        assert policy._rtc_enabled is True
        assert policy._rtc_execution_horizon == 15
        assert policy._rtc_max_guidance_weight == 8.0

    def test_rtc_explicit_enable(self):
        """rtc_enabled=True should enable when policy has rtc_config."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = False  # config says disabled, but user forced True
        rtc_cfg.execution_horizon = 10
        rtc_cfg.max_guidance_weight = 10.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        policy._init_rtc()
        assert policy._rtc_enabled is True
        assert policy._rtc_execution_horizon == 10
        assert policy._rtc_max_guidance_weight == 10.0

    def test_rtc_explicit_enable_without_rtc_config_falls_back(self):
        """rtc_enabled=True without rtc_config should warn and disable."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        mock_policy.config = MagicMock(spec=[])  # no rtc_config attr
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        with patch("strands_robots.policies.lerobot_local.policy.logger") as mock_logger:
            policy._init_rtc()
            mock_logger.warning.assert_called_once()
            assert "no rtc_config" in mock_logger.warning.call_args[0][0]
        assert policy._rtc_enabled is False

    def test_rtc_explicit_disable(self):
        """rtc_enabled=False should disable even if config says enabled."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = False
        policy._init_rtc()
        assert policy._rtc_enabled is False

    def test_rtc_user_overrides_config_values(self):
        """User-provided execution_horizon/max_guidance_weight override config."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        rtc_cfg.execution_horizon = 15
        rtc_cfg.max_guidance_weight = 8.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        policy._rtc_execution_horizon = 20
        policy._rtc_max_guidance_weight = 5.0
        policy._init_rtc()
        assert policy._rtc_execution_horizon == 20
        assert policy._rtc_max_guidance_weight == 5.0

    def test_rtc_warning_when_requested_but_unsupported(self):
        """Should warn when RTC requested but policy lacks predict_action_chunk."""
        policy = _make_policy()
        mock_policy = MagicMock(spec=["reset", "eval", "parameters", "select_action"])
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        with patch("strands_robots.policies.lerobot_local.policy.logger") as mock_logger:
            policy._init_rtc()
            mock_logger.warning.assert_called_once()
        assert policy._rtc_enabled is False


class TestRTCInference:
    """Tests for RTC inference path."""

    def test_predict_with_rtc_first_call_no_prev_chunk(self):
        """First RTC call should have no prev_chunk_left_over."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        result = policy._predict_with_rtc({})

        call_kwargs = mock_policy.predict_action_chunk.call_args[1]
        assert "prev_chunk_left_over" not in call_kwargs
        assert call_kwargs["execution_horizon"] == 10
        assert result.dim() == 2  # (T, A) after squeeze

    def test_predict_with_rtc_passes_prev_chunk(self):
        """Subsequent RTC calls should pass prev_chunk_left_over."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = torch.randn(15, 6)
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        call_kwargs = mock_policy.predict_action_chunk.call_args[1]
        assert "prev_chunk_left_over" in call_kwargs
        assert call_kwargs["prev_chunk_left_over"].shape == (15, 6)

    def test_predict_with_rtc_stores_leftover(self):
        """After RTC inference, leftover should be stored for next call."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        assert policy._rtc_prev_chunk is not None
        assert policy._rtc_prev_chunk.dim() == 2

    def test_predict_with_rtc_leftover_keyed_on_execution_horizon(self):
        """Leftover is the chunk tail past execution_horizon, not actions_per_step.

        Regression: when the consumer drained the FULL trained chunk
        (actions_per_step == chunk length) the old bookkeeping set steps_to_consume
        to the whole chunk, so ``_rtc_prev_chunk`` was always ``None`` and the
        cross-chunk blend never received a previous-chunk tail. With the
        execution-horizon contract the consumer re-queries every
        execution_horizon steps, so the tail past that point must carry over.
        """
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 20  # full trained chunk == chunk length below

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        # delay ~= 0 on the first call -> leftover starts at execution_horizon.
        assert policy._rtc_prev_chunk is not None
        assert policy._rtc_prev_chunk.shape == (10, 6)

    def test_execution_horizon_prefers_rtc_over_actions_per_step(self):
        """execution_horizon is the RTC horizon while RTC is active."""
        policy = _make_policy()
        policy.actions_per_step = 50
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        assert policy.execution_horizon == 10

    def test_execution_horizon_falls_back_to_chunk_without_rtc(self):
        """Without RTC, execution_horizon is the trained chunk length."""
        policy = _make_policy()
        policy.actions_per_step = 50
        policy._rtc_enabled = False
        assert policy.execution_horizon == 50

    def test_predict_with_rtc_tracks_latency(self):
        """RTC should track inference latency for delay estimation."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        assert len(policy._rtc_latency_history) == 1
        assert policy._rtc_latency_history[0] > 0


class TestRTCDelayEstimation:
    """Tests for inference delay estimation."""

    def test_estimate_delay_empty_history(self):
        """No history should return delay=0."""
        policy = _make_policy()
        assert policy._estimate_inference_delay(fps=30.0) == 0

    def test_estimate_delay_single_sample(self):
        """Single latency sample should give correct delay."""
        policy = _make_policy()
        policy._rtc_latency_history.append(0.1)  # 100ms
        assert policy._estimate_inference_delay(fps=30.0) == 3

    def test_estimate_delay_uses_p95(self):
        """Should use p95 latency, not mean or max."""
        policy = _make_policy()
        for _ in range(99):
            policy._rtc_latency_history.append(0.033)
        policy._rtc_latency_history.append(1.0)  # outlier
        delay = policy._estimate_inference_delay(fps=30.0)
        assert delay < 5


class TestRTCDeterministicDelay:
    """RTC inference delay is the caller-supplied step count, not a wall-clock guess.

    Regression for the fixed-seed trajectory drift reported in multi-episode
    evals: when the inference delay was derived from wall-clock p95 latency it
    warmed up within an episode and varied run-to-run, so two otherwise-identical
    seeded episodes sliced the chunk seam differently. The runtime now supplies
    the exact number of control steps that elapse during inference
    (``set_rtc_observed_delay``); the policy must use that integer verbatim and
    ignore the latency history entirely.
    """

    @staticmethod
    def _chunk_policy():
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1
        mock_policy = MagicMock()
        policy._policy = mock_policy
        return policy, mock_policy

    def test_observed_delay_overrides_wallclock_estimate(self):
        """A supplied step count wins even when latency history implies a big delay."""
        policy, mock_policy = self._chunk_policy()
        # Deterministic chunk so we can assert exactly which row is returned.
        chunk = torch.arange(20 * 6, dtype=torch.float32).reshape(1, 20, 6)
        mock_policy.predict_action_chunk.return_value = chunk
        # Poison the wall-clock path: 5s latency * 50Hz -> ~250 steps (clamps to 19).
        policy.set_control_frequency(50.0)
        policy._rtc_latency_history.extend([5.0] * 10)
        # Runtime says exactly 2 control steps elapsed during inference.
        policy.set_rtc_observed_delay(2)

        result = policy._predict_with_rtc({})

        squeezed = chunk.squeeze(0)
        # usable_start == observed delay (2), NOT the wall-clock-implied 19.
        assert torch.equal(result[0], squeezed[2])

    def test_observed_delay_zero_executes_chunk_from_start(self):
        """Synchronous loop (delay 0): the chunk is used from row 0, no skip."""
        policy, mock_policy = self._chunk_policy()
        chunk = torch.arange(20 * 6, dtype=torch.float32).reshape(1, 20, 6)
        mock_policy.predict_action_chunk.return_value = chunk
        policy.set_control_frequency(50.0)
        policy._rtc_latency_history.extend([5.0] * 10)  # would imply a huge skip
        policy.set_rtc_observed_delay(0)

        result = policy._predict_with_rtc({})

        assert torch.equal(result[0], chunk.squeeze(0)[0])

    def test_reproducible_across_episodes_with_same_observed_delay(self):
        """Same observed delay -> identical first action despite differing latency.

        This is the fixed-seed drift the issue describes: only the latency
        history differs between the two episodes (warm-up bleed), yet with the
        deterministic delay the returned action is bit-identical.
        """
        chunk = torch.arange(20 * 6, dtype=torch.float32).reshape(1, 20, 6)

        def run_episode(latencies):
            policy, mock_policy = self._chunk_policy()
            mock_policy.predict_action_chunk.return_value = chunk.clone()
            policy.set_control_frequency(50.0)
            policy._rtc_latency_history.extend(latencies)
            policy.set_rtc_observed_delay(3)
            return policy._predict_with_rtc({})

        ep1 = run_episode([0.01] * 2)  # cold: short, sparse history
        ep2 = run_episode([0.4] * 100)  # warm: long, saturated history
        assert torch.equal(ep1[0], ep2[0])

    def test_falls_back_to_wallclock_when_no_observed_delay(self):
        """None observed delay preserves the wall-clock estimate (async hardware)."""
        policy, mock_policy = self._chunk_policy()
        chunk = torch.arange(20 * 6, dtype=torch.float32).reshape(1, 20, 6)
        mock_policy.predict_action_chunk.return_value = chunk
        policy.set_control_frequency(50.0)
        policy._rtc_latency_history.extend([0.04] * 10)  # 0.04 * 50 = 2 steps
        assert policy.rtc_observed_delay_steps is None

        result = policy._predict_with_rtc({})

        # Wall-clock estimate: p95(0.04) * 50 = 2 -> usable_start 2.
        assert torch.equal(result[0], chunk.squeeze(0)[2])

    def test_reset_clears_observed_delay(self):
        policy = _make_policy()
        policy._policy = MagicMock()
        policy._processor_bridge = None
        policy.set_rtc_observed_delay(4)
        policy.reset()
        assert policy.rtc_observed_delay_steps is None


class TestRTCReset:
    """Tests for RTC state clearing on reset."""

    def test_reset_clears_rtc_state(self):
        """reset() should clear all RTC state."""
        policy = _make_policy()
        policy._policy = MagicMock()
        policy._processor_bridge = None

        policy._rtc_prev_chunk = torch.randn(10, 6)
        policy._rtc_action_queue.extend([torch.randn(6) for _ in range(5)])
        policy._rtc_latency_history.extend([0.1, 0.2, 0.3])
        policy._rtc_last_inference_time = 1.5

        policy.reset()

        assert policy._rtc_prev_chunk is None
        assert len(policy._rtc_action_queue) == 0
        assert len(policy._rtc_latency_history) == 0
        assert policy._rtc_last_inference_time == 0.0


class TestRTCGetActionsIntegration:
    """Tests for RTC integration in get_actions flow."""

    def test_get_actions_uses_rtc_when_enabled(self):
        """get_actions should use predict_action_chunk when RTC enabled."""
        policy = _make_loaded_policy(action_dim=6, include_images=False)
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1
        policy._processor_bridge = None
        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])

        policy._policy.predict_action_chunk = MagicMock(return_value=torch.randn(1, 20, 6))

        actions = policy.get_actions_sync({}, "test")

        policy._policy.predict_action_chunk.assert_called_once()
        policy._policy.select_action.assert_not_called()
        assert isinstance(actions, list)
        assert len(actions) >= 1

    def test_get_actions_uses_select_action_when_rtc_disabled(self):
        """get_actions should use select_action when RTC disabled."""
        policy = _make_loaded_policy(action_dim=6, include_images=False)
        policy._rtc_enabled = False
        policy._processor_bridge = None
        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])

        result = policy.get_actions_sync({}, "test")

        policy._policy.select_action.assert_called_once()
        assert isinstance(result, list)


# (section)
# Tests: _load_model device + postprocessor regressions
# (section)


def _load_model_with_mocks(policy, *, param_device="cpu", has_postprocessor=True, bridge_active=True):
    """Drive the real LerobotLocalPolicy._load_model with the heavy load seams
    mocked, so the device-move and postprocessor-warning branches execute.

    Returns the mocked lerobot policy object (so tests can assert on .to(...)).
    """
    mock_lerobot_policy = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(1, device=param_device))
    # parameters() is consumed via next(...) at multiple points in _load_model,
    # so hand back a FRESH iterator on every call (a stored list breaks next()).
    mock_lerobot_policy.parameters.side_effect = lambda: iter([mock_param])
    mock_lerobot_policy.config = MagicMock(spec=[])  # no .device / .input_features

    PolicyClass = MagicMock()
    PolicyClass.from_pretrained.return_value = mock_lerobot_policy

    bridge = None
    if bridge_active:
        bridge = MagicMock()
        bridge.is_active = True
        bridge.has_postprocessor = has_postprocessor

    with (
        patch.object(
            LerobotLocalPolicy,
            "_load_model",
            LerobotLocalPolicy._load_model,
        ),
        patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=PolicyClass,
        ),
        patch.object(LerobotLocalPolicy, "_configure_embodiment", lambda self: None),
        patch.object(LerobotLocalPolicy, "_init_rtc", lambda self: None),
        patch.object(ProcessorBridge, "from_pretrained", return_value=bridge),
    ):
        policy._load_model()
    return mock_lerobot_policy


class TestLoadModelDeviceMove:
    """Regression: from_pretrained may place weights on a device that differs
    from the resolved self._device. _load_model must move the model so weights
    and inputs stay in lockstep (else the first conv2d raises 'input and weight
    must be on the same device')."""

    def test_moves_model_when_param_device_differs(self):
        # requested cpu, but the checkpoint params land on 'meta' (stand-in for
        # an mps/cuda baked into the checkpoint config that != requested).
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"
        policy.requested_device = "cpu"
        policy.use_processor = False
        policy.robot_state_keys = []
        policy._processor_bridge = None
        policy._output_features = {}
        policy._input_features = {}
        policy.processor_overrides = {}
        policy.actions_per_step = 1
        policy.cache_model = False

        mock_policy = _load_model_with_mocks(policy, param_device="meta", bridge_active=False)

        mock_policy.to.assert_called_once()
        assert policy._device == torch.device("cpu")

    def test_no_move_when_param_device_matches(self):
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"
        policy.requested_device = "cpu"
        policy.use_processor = False
        policy.robot_state_keys = []
        policy._processor_bridge = None
        policy._output_features = {}
        policy._input_features = {}
        policy.processor_overrides = {}
        policy.actions_per_step = 1
        policy.cache_model = False

        mock_policy = _load_model_with_mocks(policy, param_device="cpu", bridge_active=False)

        mock_policy.to.assert_not_called()


class TestLoadModelPostprocessorWarning:
    """Regression: a checkpoint without an action postprocessor emits RAW
    normalized actions -> micro-motion. _load_model must warn once at load."""

    def _base_policy(self):
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"
        policy.requested_device = "cpu"
        policy.use_processor = True
        policy.robot_state_keys = []
        policy._processor_bridge = None
        policy._output_features = {}
        policy._input_features = {}
        policy.processor_overrides = {}
        policy.actions_per_step = 1
        policy.cache_model = False
        return policy

    def test_warns_without_postprocessor(self, caplog):
        import logging

        policy = self._base_policy()
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy"):
            _load_model_with_mocks(policy, has_postprocessor=False)
        assert any("WITHOUT an action postprocessor" in r.message for r in caplog.records)

    def test_no_warn_with_postprocessor(self, caplog):
        import logging

        policy = self._base_policy()
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy"):
            _load_model_with_mocks(policy, has_postprocessor=True)
        assert not any("WITHOUT an action postprocessor" in r.message for r in caplog.records)


# (section)
# Tests: _to_lerobot_observation (strands-native obs -> LeRobot feature keys)
# (section)


class TestToLerobotObservation:
    """Remap of bare strands observations to the model's declared feature keys.

    Exercises the legacy heuristic bridge used when no embodiment map is
    declared: image short-name matching, positional image fill, scalar-state
    collection with dim adaptation, and the LeRobot-formatted passthrough.
    """

    def test_already_lerobot_formatted_passthrough(self):
        """An observation that already has observation.* keys is returned as-is."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        obs = {"observation.state": np.zeros(2, dtype=np.float32), "task": "pick"}
        out = policy._to_lerobot_observation(obs)
        assert out == obs
        # A copy is returned (mutating result must not touch the input).
        out["new"] = 1
        assert "new" not in obs

    def test_exact_camera_name_match(self):
        """A bare cam name matching a declared short name maps to that slot."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy._input_features["observation.images.top"] = MagicMock(shape=(3, 480, 640))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = policy._to_lerobot_observation({"top": img})
        assert "observation.images.top" in out
        assert out["observation.images.top"] is img

    def test_unmatched_camera_fills_declared_slot_positionally(self):
        """A cam whose name does not match fills a free declared image slot."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy._input_features["observation.images.top"] = MagicMock(shape=(3, 480, 640))
        img = np.ones((480, 640, 3), dtype=np.uint8)
        # 'front' has no exact match -> fills the only declared slot 'top'.
        out = policy._to_lerobot_observation({"front": img})
        assert "observation.images.top" in out
        assert out["observation.images.top"] is img

    def test_scalar_state_collected_in_robot_state_keys_order(self):
        """Scalar joints are packed into observation.state in declared order."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        # Insertion order deliberately differs from robot_state_keys order.
        out = policy._to_lerobot_observation({"c": 3.0, "a": 1.0, "b": 2.0})
        np.testing.assert_allclose(out["observation.state"], [1.0, 2.0, 3.0])
        assert out["observation.state"].dtype == np.float32

    def test_state_falls_back_to_observation_keys_when_names_mismatch(self):
        """If no robot_state_keys are present in obs, use the obs's own scalars."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        # Auto-filled generic names that the real obs does not use.
        policy.set_robot_state_keys(["joint_0", "joint_1", "joint_2"])
        out = policy._to_lerobot_observation({"shoulder": 1.0, "elbow": 2.0, "wrist": 3.0})
        np.testing.assert_allclose(sorted(out["observation.state"]), [1.0, 2.0, 3.0])

    def test_state_zero_padded_to_model_dim(self):
        """A short state vector is zero-padded to the model's declared dim."""
        policy = _make_loaded_policy(state_dim=4, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        out = policy._to_lerobot_observation({"a": 1.0, "b": 2.0})
        assert out["observation.state"].shape == (4,)
        np.testing.assert_allclose(out["observation.state"], [1.0, 2.0, 0.0, 0.0])

    def test_state_truncated_to_model_dim(self):
        """A long state vector is truncated to the model's declared dim."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        out = policy._to_lerobot_observation({"a": 1.0, "b": 2.0, "c": 3.0})
        assert out["observation.state"].shape == (2,)
        np.testing.assert_allclose(out["observation.state"], [1.0, 2.0])

    def test_zero_dim_ndarray_scalar_collected(self):
        """A 0-d numpy array counts as a scalar joint value."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        out = policy._to_lerobot_observation({"a": np.array(1.5), "b": 2.0})
        np.testing.assert_allclose(out["observation.state"], [1.5, 2.0])

    def test_task_passthrough(self):
        """A 'task' key is preserved verbatim and not packed into state."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        out = policy._to_lerobot_observation({"a": 1.0, "b": 2.0, "task": "stack blocks"})
        assert out["task"] == "stack blocks"
        assert out["observation.state"].shape == (2,)


# (section)
# Tests: _fixup_preprocessed_batch (raw arrays/tensors -> batched device tensors)
# (section)


class TestFixupPreprocessedBatch:
    """Shape/dtype normalization of entries the preprocessor left unconverted."""

    def test_numpy_image_hwc_to_bchw(self):
        """An HWC uint8 numpy image becomes a (1,C,H,W) float tensor."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        batch = {"observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8)}
        out = policy._fixup_preprocessed_batch(batch)
        t = out["observation.images.top"]
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 3, 480, 640)
        assert t.dtype == torch.float32

    def test_numpy_state_1d_gets_batch_dim(self):
        """A 1-D numpy state vector gains a leading batch dim."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        out = policy._fixup_preprocessed_batch({"observation.state": np.array([1.0, 2.0, 3.0])})
        t = out["observation.state"]
        assert t.shape == (1, 3)
        assert t.dtype == torch.float32

    def test_tensor_float64_state_autocast_and_batched(self):
        """A 1-D float64 tensor is cast to float32 and gains a batch dim."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        val = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out = policy._fixup_preprocessed_batch({"observation.state": val})
        t = out["observation.state"]
        assert t.shape == (1, 2)
        assert t.dtype == torch.float32

    def test_tensor_image_hwc_permuted_and_batched(self):
        """A 3-D HWC tensor image is permuted to CHW and batched."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        val = torch.zeros((480, 640, 3))
        out = policy._fixup_preprocessed_batch({"observation.images.top": val})
        assert out["observation.images.top"].shape == (1, 3, 480, 640)

    def test_non_array_values_pass_through(self):
        """Strings and other non-array values are passed through untouched."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        out = policy._fixup_preprocessed_batch({"task": "pick up the cube"})
        assert out["task"] == "pick up the cube"

    def test_already_batched_tensor_unchanged_shape(self):
        """An already-(B,D) tensor keeps its shape (no spurious batch dim)."""
        policy = _make_loaded_policy(state_dim=2, include_images=False)
        val = torch.zeros((1, 2), dtype=torch.float32)
        out = policy._fixup_preprocessed_batch({"observation.state": val})
        assert out["observation.state"].shape == (1, 2)


# (section)
# Tests: RTC inference-delay uses the loop's control frequency
# (section)


def _make_rtc_policy(chunk_steps=50, action_dim=6):
    """Loaded policy wired for the RTC path with a fixed-size action chunk."""
    policy = _make_loaded_policy(action_dim=action_dim, include_images=False)
    policy._rtc_enabled = True
    policy._rtc_execution_horizon = 10
    policy.actions_per_step = 1
    chunk = torch.zeros((1, chunk_steps, action_dim))
    policy._policy.predict_action_chunk = MagicMock(return_value=chunk)
    return policy


class TestRtcInferenceDelay:
    """RTC delay estimation must scale with the real control frequency.

    ``_estimate_inference_delay`` returns ``int(p95_latency * fps)``: the number
    of action steps consumed while the policy was computing the next chunk. If
    ``fps`` is not the loop's actual control rate, the chunk-seam slice is wrong
    at every other frequency (delay under-counted at >30Hz, jerky / oscillating
    output). The control rate is plumbed via ``Policy.set_control_frequency``.
    """

    def test_estimate_inference_delay_scales_linearly_with_fps(self):
        policy = _make_loaded_policy(include_images=False)
        policy._rtc_latency_history.extend([0.1] * 10)  # p95 == 0.1s
        assert policy._estimate_inference_delay(fps=30.0) == 3
        assert policy._estimate_inference_delay(fps=60.0) == 6
        assert policy._estimate_inference_delay(fps=120.0) == 12

    def test_estimate_inference_delay_empty_history_is_zero(self):
        policy = _make_loaded_policy(include_images=False)
        assert policy._estimate_inference_delay(fps=200.0) == 0

    def test_predict_with_rtc_uses_set_control_frequency(self):
        """Regression: at 90Hz the delay (and chunk slice) must reflect 90Hz,
        not the hardcoded 30Hz default. Pre-fix this returned the 30Hz slice
        regardless of control_frequency."""
        policy = _make_rtc_policy(chunk_steps=50)
        policy._rtc_latency_history.extend([0.1] * 20)  # p95 == 0.1s
        policy.set_control_frequency(90.0)

        usable = policy._predict_with_rtc({})

        # delay = int(0.1 * 90) = 9 -> usable_start = 9 -> 50 - 9 = 41 steps.
        # At the buggy 30Hz default delay would be 3 -> 47 steps.
        assert usable.shape[0] == 50 - 9

    def test_predict_with_rtc_30hz_matches_legacy_slice(self):
        """At 30Hz the result is unchanged from the historical behaviour."""
        policy = _make_rtc_policy(chunk_steps=50)
        policy._rtc_latency_history.extend([0.1] * 20)
        policy.set_control_frequency(30.0)
        usable = policy._predict_with_rtc({})
        assert usable.shape[0] == 50 - 3

    def test_predict_with_rtc_warns_once_when_frequency_unset(self, caplog):
        """RTC without a known control rate falls back to 30Hz but MUST warn
        loudly (once), never silently assume the rate."""
        policy = _make_rtc_policy(chunk_steps=50)
        policy._rtc_latency_history.extend([0.1] * 20)
        assert policy.control_frequency is None

        with caplog.at_level("WARNING"):
            policy._predict_with_rtc({})
            policy._predict_with_rtc({})

        warnings = [r for r in caplog.records if "control_frequency unknown" in r.message]
        assert len(warnings) == 1, "fallback must warn exactly once per policy"
        assert policy._rtc_freq_warned is True


class TestCanonicalizeObsImages:
    """_canonicalize_obs_images normalizes camera frames to CHW float32 [0,1]
    BEFORE the preprocessor's normalizer runs.

    Regression: direct get_actions() callers pass HWC uint8 frames (the natural
    OpenCV/renderer format). Feeding those raw made lerobot's normalizer broadcast
    a 3-vector against the 480 height ("size of tensor a (480) must match b (3)")
    or overflow uint8. The fix converts to CHW float32 up front; these pin every
    input layout the helper must accept.
    """

    def _policy(self):
        with patch.object(LerobotLocalPolicy, "_load_model"):
            return LerobotLocalPolicy(pretrained_name_or_path="test/model")

    def test_hwc_uint8_to_chw_float01(self):
        p = self._policy()
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        out = p._canonicalize_obs_images({"observation.images.top": img})["observation.images.top"]
        assert tuple(out.shape) == (3, 480, 640)  # HWC -> CHW
        assert out.dtype == torch.float32
        assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0  # uint8 [0,255] -> [0,1]

    def test_hwc_float_to_chw(self):
        p = self._policy()
        img = np.random.rand(480, 640, 3).astype(np.float32)
        out = p._canonicalize_obs_images({"observation.images.top": img})["observation.images.top"]
        assert tuple(out.shape) == (3, 480, 640)
        assert out.dtype == torch.float32

    def test_chw_float_passthrough_layout(self):
        p = self._policy()
        img = np.random.rand(3, 480, 640).astype(np.float32)
        out = p._canonicalize_obs_images({"observation.images.top": img})["observation.images.top"]
        assert tuple(out.shape) == (3, 480, 640)  # already CHW -> unchanged layout

    def test_chw_uint8_torch_normalized(self):
        p = self._policy()
        img = (torch.rand(3, 480, 640) * 255).to(torch.uint8)
        out = p._canonicalize_obs_images({"observation.images.top": img})["observation.images.top"]
        assert tuple(out.shape) == (3, 480, 640)
        assert out.dtype == torch.float32 and float(out.max()) <= 1.0

    def test_non_image_keys_pass_through(self):
        p = self._policy()
        obs = {"observation.state": [0.0] * 6, "task": "pick up the cube"}
        out = p._canonicalize_obs_images(obs)
        assert out["observation.state"] == [0.0] * 6
        assert out["task"] == "pick up the cube"
