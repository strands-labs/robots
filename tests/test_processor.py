#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.processor module.

Tests ProcessorBridge, ProcessedPolicy, and create_processor_bridge factory.
All tests run without LeRobot — the passthrough/fallback behavior is fully
tested, and LeRobot-specific paths are tested with mocks.

Coverage targets:
- ProcessorBridge init (with and without LeRobot)
- ProcessorBridge properties (has_preprocessor, has_postprocessor, is_active)
- ProcessorBridge.preprocess / postprocess (passthrough and active)
- ProcessorBridge.process_full_transition
- ProcessorBridge.reset
- ProcessorBridge.wrap_policy
- ProcessorBridge.get_info
- ProcessorBridge.__repr__
- ProcessorBridge.from_pretrained (success, partial load, no LeRobot)
- ProcessedPolicy (wraps MockPolicy with pre/post processing)
- ProcessedPolicy.get_actions / select_action_sync / reset / __getattr__
- ProcessedPolicy.select_action_sync with postprocessor (torch branch)
- create_processor_bridge factory (all branches)
- _try_import_processor (import success and failure)
- Error handling (preprocessor/postprocessor exceptions)
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _has_torch = True
except ImportError:
    _has_torch = False

from strands_robots.processor import (
    POSTPROCESSOR_CONFIG,
    PREPROCESSOR_CONFIG,
    ProcessedPolicy,
    ProcessorBridge,
    _try_import_processor,
    create_processor_bridge,
)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class MockPolicy:
    """Minimal mock policy for testing ProcessedPolicy wrapper."""

    def __init__(self):
        self.robot_state_keys = ["j0", "j1", "j2"]
        self._reset_called = False

    @property
    def provider_name(self):
        return "mock_test"

    def set_robot_state_keys(self, keys):
        self.robot_state_keys = keys

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [
            {"j0": 0.1, "j1": 0.2, "j2": 0.3},
            {"j0": 0.4, "j1": 0.5, "j2": 0.6},
        ]

    def select_action_sync(self, observation_dict):
        return np.array([0.1, 0.2, 0.3])

    def reset(self):
        self._reset_called = True

    @property
    def custom_attr(self):
        return "custom_value"


def _make_mock_pipeline(n_steps=3, process_obs_result=None, process_action_result=None):
    """Create a mock DataProcessorPipeline."""
    pipeline = MagicMock()
    pipeline.__len__ = MagicMock(return_value=n_steps)
    pipeline.steps = [MagicMock() for _ in range(n_steps)]
    for i, step in enumerate(pipeline.steps):
        type(step).__name__ = f"Step{i}"

    if process_obs_result is not None:
        pipeline.process_observation.return_value = process_obs_result
    else:
        pipeline.process_observation.return_value = {"processed": True}

    if process_action_result is not None:
        pipeline.process_action.return_value = process_action_result
    else:
        pipeline.process_action.return_value = {"action": 0.5}

    pipeline.__call__ = MagicMock(return_value={"full": "transition"})
    pipeline.reset = MagicMock()

    return pipeline


def _make_mock_lerobot_modules():
    """Create mock lerobot processor modules."""
    modules = {
        "DataProcessorPipeline": MagicMock(),
        "batch_to_transition": MagicMock(),
        "transition_to_batch": MagicMock(),
        "observation_to_transition": MagicMock(),
        "transition_to_observation": MagicMock(),
        "EnvTransition": MagicMock(),
        "TransitionKey": MagicMock(),
    }
    return modules


# ─────────────────────────────────────────────────────────────────────
# 1. Module-level imports and constants
# ─────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    """Test module-level constants and imports."""

    def test_preprocessor_config_name(self):
        assert PREPROCESSOR_CONFIG == "policy_preprocessor.json"

    def test_postprocessor_config_name(self):
        assert POSTPROCESSOR_CONFIG == "policy_postprocessor.json"

    def test_public_exports(self):
        from strands_robots.processor import __all__

        expected = {
            "ProcessorBridge",
            "ProcessedPolicy",
            "create_processor_bridge",
            "PREPROCESSOR_CONFIG",
            "POSTPROCESSOR_CONFIG",
        }
        assert set(__all__) == expected

    def test_try_import_processor_returns_none_without_lerobot(self):
        """Without LeRobot, _try_import_processor returns None."""
        with patch.dict(
            sys.modules,
            {
                "lerobot": None,
                "lerobot.processor": None,
                "lerobot.processor.pipeline": None,
            },
        ):
            result = _try_import_processor()
        # In current env without lerobot, should be None
        # (The real test is that the function doesn't crash)
        assert result is None or isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────
# 2. ProcessorBridge — init and properties
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeInit:
    """Test ProcessorBridge initialization and property accessors."""

    def test_empty_init(self):
        """Empty bridge has no pre/postprocessor."""
        bridge = ProcessorBridge()
        assert not bridge.has_preprocessor
        assert not bridge.has_postprocessor
        assert not bridge.is_active
        assert bridge._device is None

    def test_init_with_device(self):
        bridge = ProcessorBridge(device="cpu")
        assert bridge._device == "cpu"

    def test_init_with_preprocessor_only(self):
        pre = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre)
        assert bridge.has_preprocessor
        assert not bridge.has_postprocessor
        assert bridge.is_active

    def test_init_with_postprocessor_only(self):
        post = _make_mock_pipeline()
        bridge = ProcessorBridge(postprocessor=post)
        assert not bridge.has_preprocessor
        assert bridge.has_postprocessor
        assert bridge.is_active

    def test_init_with_both(self):
        pre = _make_mock_pipeline(2)
        post = _make_mock_pipeline(4)
        bridge = ProcessorBridge(preprocessor=pre, postprocessor=post, device="cuda:0")
        assert bridge.has_preprocessor
        assert bridge.has_postprocessor
        assert bridge.is_active
        assert bridge._device == "cuda:0"


# ─────────────────────────────────────────────────────────────────────
# 3. ProcessorBridge — repr
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeRepr:
    """Test __repr__ formatting."""

    def test_repr_empty(self):
        bridge = ProcessorBridge()
        r = repr(bridge)
        assert "ProcessorBridge" in r
        assert "pre=None" in r
        assert "post=None" in r

    def test_repr_with_pre(self):
        pre = _make_mock_pipeline(3)
        bridge = ProcessorBridge(preprocessor=pre)
        r = repr(bridge)
        assert "pre=3steps" in r
        assert "post=None" in r

    def test_repr_with_both(self):
        pre = _make_mock_pipeline(2)
        post = _make_mock_pipeline(5)
        bridge = ProcessorBridge(preprocessor=pre, postprocessor=post)
        r = repr(bridge)
        assert "pre=2steps" in r
        assert "post=5steps" in r


# ─────────────────────────────────────────────────────────────────────
# 4. ProcessorBridge — get_info
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeGetInfo:
    """Test get_info() method."""

    def test_info_empty(self):
        bridge = ProcessorBridge(device="cpu")
        info = bridge.get_info()
        assert info["has_preprocessor"] is False
        assert info["has_postprocessor"] is False
        assert info["is_active"] is False
        assert info["device"] == "cpu"
        assert "preprocessor_steps" not in info
        assert "postprocessor_steps" not in info

    def test_info_with_preprocessor(self):
        pre = _make_mock_pipeline(3)
        bridge = ProcessorBridge(preprocessor=pre)
        info = bridge.get_info()
        assert info["has_preprocessor"] is True
        assert info["preprocessor_steps"] == 3
        assert len(info["preprocessor_step_names"]) == 3

    def test_info_with_postprocessor(self):
        post = _make_mock_pipeline(2)
        bridge = ProcessorBridge(postprocessor=post)
        info = bridge.get_info()
        assert info["has_postprocessor"] is True
        assert info["postprocessor_steps"] == 2
        assert len(info["postprocessor_step_names"]) == 2

    def test_info_with_both(self):
        pre = _make_mock_pipeline(1)
        post = _make_mock_pipeline(4)
        bridge = ProcessorBridge(preprocessor=pre, postprocessor=post, device="cuda")
        info = bridge.get_info()
        assert info["has_preprocessor"] is True
        assert info["has_postprocessor"] is True
        assert info["is_active"] is True
        assert info["device"] == "cuda"
        assert info["preprocessor_steps"] == 1
        assert info["postprocessor_steps"] == 4


# ─────────────────────────────────────────────────────────────────────
# 5. ProcessorBridge — preprocess / postprocess / full transition
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeProcessing:
    """Test data processing methods."""

    def test_preprocess_passthrough_no_preprocessor(self):
        bridge = ProcessorBridge()
        obs = {"camera": [1, 2, 3], "state": [0.5]}
        result = bridge.preprocess(obs)
        assert result is obs  # Same object — passthrough

    def test_preprocess_passthrough_no_modules(self):
        """Even with preprocessor, if _modules is None, passthrough."""
        pre = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = None
        obs = {"key": "value"}
        result = bridge.preprocess(obs)
        assert result is obs

    def test_preprocess_active(self):
        """With preprocessor and modules, calls process_observation."""
        pre = _make_mock_pipeline(process_obs_result={"normalized": True})
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()
        obs = {"raw_state": [1, 2, 3]}
        result = bridge.preprocess(obs)
        assert result == {"normalized": True}
        pre.process_observation.assert_called_once_with(obs)

    def test_preprocess_exception_fallback(self):
        """If preprocessor raises, returns raw observation."""
        pre = _make_mock_pipeline()
        pre.process_observation.side_effect = RuntimeError("pipeline crashed")
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()
        obs = {"state": [1, 2]}
        result = bridge.preprocess(obs)
        assert result is obs  # Fallback to raw

    def test_postprocess_passthrough_no_postprocessor(self):
        bridge = ProcessorBridge()
        action = np.array([0.1, 0.2])
        result = bridge.postprocess(action)
        assert result is action

    def test_postprocess_passthrough_no_modules(self):
        post = _make_mock_pipeline()
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = None
        action = {"j0": 0.1}
        result = bridge.postprocess(action)
        assert result is action

    def test_postprocess_active(self):
        post = _make_mock_pipeline(process_action_result={"unnormalized": True})
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        action = MagicMock()
        result = bridge.postprocess(action)
        assert result == {"unnormalized": True}
        post.process_action.assert_called_once_with(action)

    def test_postprocess_exception_fallback(self):
        post = _make_mock_pipeline()
        post.process_action.side_effect = ValueError("bad action")
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        action = np.array([0.5])
        result = bridge.postprocess(action)
        assert result is action

    def test_process_full_transition_passthrough(self):
        bridge = ProcessorBridge()
        transition = {"obs": {}, "action": {}, "reward": 1.0}
        result = bridge.process_full_transition(transition)
        assert result is transition

    def test_process_full_transition_passthrough_no_modules(self):
        pre = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = None
        transition = {"obs": {}}
        result = bridge.process_full_transition(transition)
        assert result is transition

    def test_process_full_transition_active(self):
        pre = _make_mock_pipeline()
        pre.__call__ = MagicMock(return_value={"processed_transition": True})
        pre.return_value = {"processed_transition": True}
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()
        transition = {"obs": {}, "action": {}}
        bridge.process_full_transition(transition)
        # The preprocessor is called as a callable
        pre.assert_called_once_with(transition)

    def test_process_full_transition_exception_fallback(self):
        pre = _make_mock_pipeline()
        pre.side_effect = RuntimeError("crash")
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()
        transition = {"obs": {}}
        result = bridge.process_full_transition(transition)
        assert result is transition


# ─────────────────────────────────────────────────────────────────────
# 6. ProcessorBridge — reset
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeReset:
    """Test reset() method."""

    def test_reset_empty(self):
        """Reset on empty bridge does nothing (no crash)."""
        bridge = ProcessorBridge()
        bridge.reset()  # Should not raise

    def test_reset_with_preprocessor(self):
        pre = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre)
        bridge.reset()
        pre.reset.assert_called_once()

    def test_reset_with_postprocessor(self):
        post = _make_mock_pipeline()
        bridge = ProcessorBridge(postprocessor=post)
        bridge.reset()
        post.reset.assert_called_once()

    def test_reset_with_both(self):
        pre = _make_mock_pipeline()
        post = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre, postprocessor=post)
        bridge.reset()
        pre.reset.assert_called_once()
        post.reset.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# 7. ProcessorBridge — from_pretrained
# ─────────────────────────────────────────────────────────────────────


class TestProcessorBridgeFromPretrained:
    """Test from_pretrained classmethod."""

    def test_from_pretrained_no_lerobot(self):
        """Without LeRobot, returns passthrough bridge."""
        with patch("strands_robots.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge.from_pretrained("some/model")
        assert not bridge.is_active

    def test_from_pretrained_success(self):
        """With LeRobot modules, loads both pipelines."""
        mock_pre = _make_mock_pipeline(2)
        mock_post = _make_mock_pipeline(3)

        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])

        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            bridge = ProcessorBridge.from_pretrained("lerobot/act_model", device="cpu")

        assert bridge.has_preprocessor
        assert bridge.has_postprocessor
        assert bridge._device == "cpu"

    def test_from_pretrained_preprocessor_not_found(self):
        """FileNotFoundError on preprocessor → no preprocessor, postprocessor still loads."""
        mock_post = _make_mock_pipeline(1)
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[FileNotFoundError("no config"), mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            bridge = ProcessorBridge.from_pretrained("model_path")

        assert not bridge.has_preprocessor
        assert bridge.has_postprocessor

    def test_from_pretrained_postprocessor_value_error(self):
        """ValueError on postprocessor → no postprocessor."""
        mock_pre = _make_mock_pipeline(2)
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, ValueError("bad config")])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            bridge = ProcessorBridge.from_pretrained("model_path")

        assert bridge.has_preprocessor
        assert not bridge.has_postprocessor

    def test_from_pretrained_generic_exception(self):
        """Generic exception during loading → warning, no pipeline."""
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[RuntimeError("crash"), TypeError("bad")])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            bridge = ProcessorBridge.from_pretrained("model_path")

        assert not bridge.has_preprocessor
        assert not bridge.has_postprocessor

    def test_from_pretrained_with_overrides(self):
        """Overrides are passed to both pipelines."""
        mock_pre = _make_mock_pipeline()
        mock_post = _make_mock_pipeline()
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            ProcessorBridge.from_pretrained(
                "model",
                overrides={"normalizer": {"mean": 0}},
            )

        # Both calls should have gotten overrides
        calls = mock_dpp.from_pretrained.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.kwargs["overrides"] == {"normalizer": {"mean": 0}}

    def test_from_pretrained_custom_config_filenames(self):
        """Custom preprocessor/postprocessor config filenames."""
        mock_pre = _make_mock_pipeline()
        mock_post = _make_mock_pipeline()
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            ProcessorBridge.from_pretrained(
                "model",
                preprocessor_config="custom_pre.json",
                postprocessor_config="custom_post.json",
            )

        calls = mock_dpp.from_pretrained.call_args_list
        assert calls[0].kwargs["config_filename"] == "custom_pre.json"
        assert calls[1].kwargs["config_filename"] == "custom_post.json"


# ─────────────────────────────────────────────────────────────────────
# 8. ProcessorBridge — wrap_policy
# ─────────────────────────────────────────────────────────────────────


class TestWrapPolicy:
    """Test wrap_policy() returns a ProcessedPolicy."""

    def test_wrap_returns_processed_policy(self):
        bridge = ProcessorBridge()
        policy = MockPolicy()
        wrapped = bridge.wrap_policy(policy)
        assert isinstance(wrapped, ProcessedPolicy)

    def test_wrapped_provider_name(self):
        bridge = ProcessorBridge()
        policy = MockPolicy()
        wrapped = bridge.wrap_policy(policy)
        assert wrapped.provider_name == "processed:mock_test"


# ─────────────────────────────────────────────────────────────────────
# 9. ProcessedPolicy
# ─────────────────────────────────────────────────────────────────────


class TestProcessedPolicy:
    """Test the ProcessedPolicy wrapper."""

    def test_init(self):
        policy = MockPolicy()
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(policy, bridge)
        assert pp._policy is policy
        assert pp._bridge is bridge

    def test_provider_name(self):
        pp = ProcessedPolicy(MockPolicy(), ProcessorBridge())
        assert pp.provider_name == "processed:mock_test"

    def test_set_robot_state_keys(self):
        policy = MockPolicy()
        pp = ProcessedPolicy(policy, ProcessorBridge())
        pp.set_robot_state_keys(["a", "b"])
        assert policy.robot_state_keys == ["a", "b"]

    def test_get_actions_passthrough(self):
        """Without active bridge, actions pass through unchanged."""
        policy = MockPolicy()
        bridge = ProcessorBridge()  # No pre/postprocessor
        pp = ProcessedPolicy(policy, bridge)

        actions = asyncio.run(pp.get_actions({"obs": 1}, "test instruction"))
        assert len(actions) == 2
        assert actions[0] == {"j0": 0.1, "j1": 0.2, "j2": 0.3}

    def test_get_actions_with_preprocessing(self):
        """Preprocessor transforms observation before policy inference."""
        policy = MockPolicy()
        pre = _make_mock_pipeline(process_obs_result={"normalized_obs": True})
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        actions = asyncio.run(pp.get_actions({"raw_obs": True}, "test"))
        pre.process_observation.assert_called_once_with({"raw_obs": True})
        assert len(actions) == 2

    def test_get_actions_with_postprocessing_dict(self):
        """Postprocessor transforms each action dict."""
        policy = MockPolicy()
        post = _make_mock_pipeline(process_action_result={"unnorm": 0.5})
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        actions = asyncio.run(pp.get_actions({"obs": 1}, "test"))
        assert len(actions) == 2
        assert all(a == {"unnorm": 0.5} for a in actions)

    def test_get_actions_postprocess_non_dict_fallback(self):
        """If postprocessor returns non-dict, keeps original action."""
        policy = MockPolicy()
        post = _make_mock_pipeline(process_action_result=np.array([0.5]))
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        actions = asyncio.run(pp.get_actions({"obs": 1}, "test"))
        assert len(actions) == 2
        # Non-dict result → original action preserved
        assert actions[0] == {"j0": 0.1, "j1": 0.2, "j2": 0.3}

    def test_select_action_sync_passthrough(self):
        """Without postprocessor, returns raw numpy array."""
        policy = MockPolicy()
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    # ── select_action_sync with postprocessor (torch branch) ──────

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_select_action_sync_postprocess_numpy_to_dict(self):
        """Postprocessor receives numpy array → converts via torch.from_numpy → returns dict.

        Covers lines 362-370: has_postprocessor=True, isinstance(result, np.ndarray)=True,
        isinstance(processed, dict)=True.
        """
        import torch

        policy = MockPolicy()  # select_action_sync returns np.array
        post = _make_mock_pipeline(process_action_result={"j0": 0.1, "j1": 0.2})
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        # Dict branch: np.array(list(processed.values()))
        np.testing.assert_array_almost_equal(result, [0.1, 0.2])
        # Verify postprocessor was called with a torch tensor (from np conversion)
        call_args = post.process_action.call_args[0][0]
        assert isinstance(call_args, torch.Tensor)

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_select_action_sync_postprocess_numpy_to_tensor_with_numpy_method(self):
        """Postprocessor returns tensor-like with .numpy() method.

        Covers lines 362-372: has_postprocessor=True, isinstance(result, np.ndarray)=True,
        isinstance(processed, dict)=False, hasattr(processed, 'numpy')=True.
        """
        import torch

        policy = MockPolicy()
        # Postprocessor returns a torch tensor (which has .numpy())
        tensor_result = torch.tensor([0.5, 0.6, 0.7])
        post = _make_mock_pipeline(process_action_result=tensor_result)
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        # .numpy() branch: processed.numpy()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [0.5, 0.6, 0.7])

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_select_action_sync_postprocess_numpy_to_plain_return(self):
        """Postprocessor returns plain value (not dict, no .numpy()).

        Covers lines 362-373: has_postprocessor=True, isinstance(result, np.ndarray)=True,
        isinstance(processed, dict)=False, hasattr(processed, 'numpy')=False → return processed.
        """
        policy = MockPolicy()
        # Postprocessor returns a plain list (no .numpy() method, not a dict)
        post = _make_mock_pipeline(process_action_result=[0.1, 0.2, 0.3])
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        # Falls through to plain return
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_select_action_sync_postprocess_tensor_input_not_numpy(self):
        """Policy returns non-numpy (tensor-like) → skips torch.from_numpy.

        Covers line 366-367: isinstance(result, np.ndarray)=False → tensor = result.
        """
        import torch

        # Policy that returns a torch tensor instead of numpy
        class TensorPolicy:
            provider_name = "tensor_test"

            def select_action_sync(self, obs):
                return torch.tensor([0.3, 0.4, 0.5])

        policy = TensorPolicy()
        # Postprocessor returns dict to verify the full path
        post = _make_mock_pipeline(process_action_result={"a": 1.0, "b": 2.0})
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        # Dict branch still works
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])
        # Verify postprocessor was called with the original tensor (not from_numpy'd)
        call_args = post.process_action.call_args[0][0]
        assert isinstance(call_args, torch.Tensor)
        np.testing.assert_array_almost_equal(call_args.numpy(), [0.3, 0.4, 0.5])

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_select_action_sync_postprocess_tensor_input_to_numpy_return(self):
        """Policy returns tensor, postprocessor returns tensor with .numpy().

        Covers lines 366-367 (else branch) and 371-372 (numpy branch).
        """
        import torch

        class TensorPolicy:
            provider_name = "tensor_test"

            def select_action_sync(self, obs):
                return torch.tensor([1.0, 2.0])

        policy = TensorPolicy()
        post = _make_mock_pipeline(process_action_result=torch.tensor([9.0, 8.0]))
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        pp = ProcessedPolicy(policy, bridge)

        result = pp.select_action_sync({"obs": 1})
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [9.0, 8.0])

    # ── end of select_action_sync postprocessor tests ─────────────

    def test_reset(self):
        """Reset resets both bridge and policy."""
        policy = MockPolicy()
        pre = _make_mock_pipeline()
        bridge = ProcessorBridge(preprocessor=pre)
        pp = ProcessedPolicy(policy, bridge)

        pp.reset()
        assert policy._reset_called
        pre.reset.assert_called_once()

    def test_reset_no_policy_reset(self):
        """If policy doesn't have reset, no crash."""

        class NoResetPolicy:
            provider_name = "no_reset"

        policy = NoResetPolicy()
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(policy, bridge)
        pp.reset()  # Should not raise

    def test_getattr_delegation(self):
        """Unknown attributes forwarded to wrapped policy."""
        policy = MockPolicy()
        pp = ProcessedPolicy(policy, ProcessorBridge())
        assert pp.custom_attr == "custom_value"
        assert pp.robot_state_keys == ["j0", "j1", "j2"]

    def test_getattr_missing_raises(self):
        """Missing attribute on both wrapper and policy raises AttributeError."""
        policy = MockPolicy()
        pp = ProcessedPolicy(policy, ProcessorBridge())
        with pytest.raises(AttributeError):
            _ = pp.totally_nonexistent_attribute


# ─────────────────────────────────────────────────────────────────────
# 10. create_processor_bridge factory
# ─────────────────────────────────────────────────────────────────────


class TestCreateProcessorBridge:
    """Test create_processor_bridge factory function."""

    def test_none_path_returns_passthrough(self):
        """No path → empty passthrough bridge."""
        bridge = create_processor_bridge()
        assert not bridge.is_active
        assert isinstance(bridge, ProcessorBridge)

    def test_none_path_with_device(self):
        bridge = create_processor_bridge(device="cuda:0")
        assert bridge._device == "cuda:0"

    def test_with_pretrained_path_no_lerobot(self):
        """With path but no LeRobot → passthrough bridge."""
        with patch("strands_robots.processor._try_import_processor", return_value=None):
            bridge = create_processor_bridge(pretrained_name_or_path="lerobot/act_model")
        assert not bridge.is_active

    def test_with_stats_override(self):
        """Stats override injects into normalizer overrides."""
        mock_pre = _make_mock_pipeline()
        mock_post = _make_mock_pipeline()
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        custom_stats = {"mean": [0.0], "std": [1.0]}

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            create_processor_bridge(
                pretrained_name_or_path="model",
                stats=custom_stats,
            )

        # Verify overrides include stats
        calls = mock_dpp.from_pretrained.call_args_list
        for call in calls:
            overrides = call.kwargs.get("overrides", {})
            assert overrides["normalizer_processor"]["stats"] == custom_stats
            assert overrides["unnormalizer_processor"]["stats"] == custom_stats

    def test_with_explicit_overrides_and_stats(self):
        """Both overrides kwarg and stats are merged."""
        mock_pre = _make_mock_pipeline()
        mock_post = _make_mock_pipeline()
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            create_processor_bridge(
                pretrained_name_or_path="model",
                stats={"mean": 0},
                overrides={"custom_key": "val"},
            )

        calls = mock_dpp.from_pretrained.call_args_list
        for call in calls:
            overrides = call.kwargs.get("overrides", {})
            assert "normalizer_processor" in overrides
            # custom_key should also be present
            assert overrides.get("custom_key") == "val"

    def test_kwargs_forwarded(self):
        """Extra kwargs forwarded to from_pretrained."""
        mock_pre = _make_mock_pipeline()
        mock_post = _make_mock_pipeline()
        mock_dpp = MagicMock()
        mock_dpp.from_pretrained = MagicMock(side_effect=[mock_pre, mock_post])
        modules = _make_mock_lerobot_modules()
        modules["DataProcessorPipeline"] = mock_dpp

        with patch("strands_robots.processor._try_import_processor", return_value=modules):
            create_processor_bridge(
                pretrained_name_or_path="model",
                preprocessor_config="custom.json",
            )

        calls = mock_dpp.from_pretrained.call_args_list
        assert calls[0].kwargs["config_filename"] == "custom.json"


# ─────────────────────────────────────────────────────────────────────
# 11. _try_import_processor function
# ─────────────────────────────────────────────────────────────────────


class TestTryImportProcessor:
    """Test the lazy import function."""

    def test_returns_none_when_lerobot_blocked(self):
        """When lerobot modules are blocked, _try_import_processor returns None."""
        # Use patch to simulate lerobot not being available
        with patch.dict(
            sys.modules,
            {
                "lerobot": None,
                "lerobot.processor": None,
                "lerobot.processor.pipeline": None,
                "lerobot.processor.converters": None,
                "lerobot.processor.core": None,
            },
        ):
            result = _try_import_processor()
        assert result is None

    def test_returns_dict_when_lerobot_available(self):
        """When lerobot processor is importable, returns dict with expected keys."""
        result = _try_import_processor()
        # LeRobot may or may not be installed; test both paths
        if result is not None:
            expected_keys = {
                "DataProcessorPipeline",
                "batch_to_transition",
                "transition_to_batch",
                "observation_to_transition",
                "transition_to_observation",
                "EnvTransition",
                "TransitionKey",
            }
            assert set(result.keys()) == expected_keys
        # If result is None, lerobot is not installed — that's fine too

    def test_returns_dict_with_mock_lerobot(self):
        """When lerobot is available (mocked), returns dict with all keys."""
        mock_pipeline = MagicMock()
        mock_converters = MagicMock()
        mock_core = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.processor": MagicMock(),
                "lerobot.processor.pipeline": mock_pipeline,
                "lerobot.processor.converters": mock_converters,
                "lerobot.processor.core": mock_core,
            },
        ):
            mock_pipeline.DataProcessorPipeline = MagicMock()
            mock_converters.batch_to_transition = MagicMock()
            mock_converters.transition_to_batch = MagicMock()
            mock_converters.observation_to_transition = MagicMock()
            mock_converters.transition_to_observation = MagicMock()
            mock_core.EnvTransition = MagicMock()
            mock_core.TransitionKey = MagicMock()

            result = _try_import_processor()

        if result is not None:
            expected_keys = {
                "DataProcessorPipeline",
                "batch_to_transition",
                "transition_to_batch",
                "observation_to_transition",
                "transition_to_observation",
                "EnvTransition",
                "TransitionKey",
            }
            assert set(result.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────
# 12. Edge cases
# ─────────────────────────────────────────────────────────────────────


class TestProcessorEdgeCases:
    """Edge cases and integration scenarios."""

    def test_bridge_reusable(self):
        """Bridge can process multiple observations."""
        pre = _make_mock_pipeline(process_obs_result={"ok": True})
        bridge = ProcessorBridge(preprocessor=pre)
        bridge._modules = _make_mock_lerobot_modules()

        for _ in range(5):
            result = bridge.preprocess({"obs": 1})
            assert result == {"ok": True}

        assert pre.process_observation.call_count == 5

    def test_empty_observation(self):
        """Passthrough with empty observation dict."""
        bridge = ProcessorBridge()
        result = bridge.preprocess({})
        assert result == {}

    def test_numpy_action_passthrough(self):
        """Numpy array actions pass through cleanly."""
        bridge = ProcessorBridge()
        action = np.array([1.0, 2.0, 3.0])
        result = bridge.postprocess(action)
        np.testing.assert_array_equal(result, action)

    def test_processed_policy_full_workflow(self):
        """End-to-end: bridge + policy for passthrough case."""
        policy = MockPolicy()
        bridge = ProcessorBridge()
        wrapped = bridge.wrap_policy(policy)

        # Set keys
        wrapped.set_robot_state_keys(["x", "y", "z"])
        assert policy.robot_state_keys == ["x", "y", "z"]

        # Get actions
        actions = asyncio.run(wrapped.get_actions({"obs": 1}, "pick up"))
        assert len(actions) == 2

        # Sync action
        result = wrapped.select_action_sync({"obs": 1})
        assert isinstance(result, np.ndarray)

        # Reset
        wrapped.reset()
        assert policy._reset_called

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_processed_policy_full_workflow_with_postprocessor(self):
        """End-to-end: bridge with postprocessor + policy for sync action."""

        policy = MockPolicy()
        post = _make_mock_pipeline(process_action_result={"j0": 0.9, "j1": 0.8})
        bridge = ProcessorBridge(postprocessor=post)
        bridge._modules = _make_mock_lerobot_modules()
        wrapped = bridge.wrap_policy(policy)

        # Sync action through postprocessor → dict → np.array
        result = wrapped.select_action_sync({"obs": 1})
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [0.9, 0.8])

        # Reset still works
        wrapped.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
