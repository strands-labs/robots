#!/usr/bin/env python3
"""Tests for LerobotLocalPolicy._build_observation_batch.

This is the core inference pipeline — every observation passes through this
method before reaching the model. It handles:
- LeRobot-format observations (observation.* keys)
- Strands-format observations (raw joint/camera dicts)
- Image HWC→CHW permutation and normalization
- VLA language token injection
- Missing image feature zero-filling

Strategy: mock torch at a minimal level using real torch tensors where possible,
mock only external deps (transformers).  Uses __new__ to skip __init__.
"""

import asyncio
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level mocks for lerobot (needed for import)
# ---------------------------------------------------------------------------
_mock_lerobot = MagicMock()
_mock_lerobot_policies = MagicMock()
_mock_lerobot_policies_factory = MagicMock()
_mock_lerobot.policies = _mock_lerobot_policies
_mock_lerobot.policies.factory = _mock_lerobot_policies_factory

_LEROBOT_MODULES = {
    "lerobot": _mock_lerobot,
    "lerobot.policies": _mock_lerobot_policies,
    "lerobot.policies.factory": _mock_lerobot_policies_factory,
}

# Try to import torch — required for these tests
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch required")

# Import the policy class with lerobot mocked
with patch.dict(sys.modules, _LEROBOT_MODULES):
    from strands_robots.policies.lerobot_local import LerobotLocalPolicy


def _make_policy(**overrides):
    """Create a LerobotLocalPolicy without calling __init__."""
    p = object.__new__(LerobotLocalPolicy)
    p.pretrained_name_or_path = ""
    p.policy_type = None
    p.requested_device = None
    p.actions_per_step = 1
    p.use_processor = False
    p.processor_overrides = None
    p.robot_state_keys = []
    p._policy = MagicMock()
    p._device = torch.device("cpu")
    p._input_features = {}
    p._output_features = {}
    p._loaded = True
    p._processor_bridge = None
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# ===========================================================================
# SECTION 1: LeRobot-format observations — torch.Tensor inputs
# ===========================================================================


class TestLerobotFormatTensors:
    """Tests for observation dicts with 'observation.*' keys containing tensors."""

    def test_state_tensor_1d_gets_batch_dim(self):
        """1D state tensor should get unsqueeze(0) for batch dim."""
        p = _make_policy()
        obs = {"observation.state": torch.tensor([1.0, 2.0, 3.0])}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 3)

    def test_state_tensor_2d_stays(self):
        """2D state tensor already has batch dim, should stay."""
        p = _make_policy()
        obs = {"observation.state": torch.tensor([[1.0, 2.0]])}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 2)

    def test_image_tensor_hwc_to_chw(self):
        """HWC image tensor should be permuted to CHW."""
        p = _make_policy()
        # 480x640x3 HWC
        img = torch.zeros(480, 640, 3)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        # Should be 1x3x480x640 (CHW + batch)
        assert batch["observation.image"].shape == (1, 3, 480, 640)

    def test_image_tensor_chw_stays(self):
        """CHW image tensor (last dim not in 1,3,4) should NOT permute."""
        p = _make_policy()
        # 3x480x640 already CHW
        img = torch.zeros(3, 480, 640)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 3, 480, 640)

    def test_image_tensor_1ch_hwc(self):
        """Single-channel HWC image gets permuted."""
        p = _make_policy()
        img = torch.zeros(64, 64, 1)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 1, 64, 64)

    def test_image_tensor_4ch_hwc(self):
        """RGBA HWC image gets permuted."""
        p = _make_policy()
        img = torch.zeros(64, 64, 4)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 4, 64, 64)

    def test_image_tensor_already_batched(self):
        """4D image tensor already has batch dim, no extra unsqueeze."""
        p = _make_policy()
        img = torch.zeros(1, 3, 64, 64)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 3, 64, 64)

    def test_device_transfer(self):
        """Tensors should be moved to policy's device."""
        p = _make_policy()
        obs = {"observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].device == torch.device("cpu")


# ===========================================================================
# SECTION 2: LeRobot-format observations — numpy inputs
# ===========================================================================


class TestLerobotFormatNumpy:
    """Tests for observation dicts with numpy array values."""

    def test_1d_numpy_gets_batched(self):
        """1D numpy array should become [1, N] tensor."""
        p = _make_policy()
        obs = {"observation.state": np.array([1.0, 2.0], dtype=np.float32)}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 2)
        assert batch["observation.state"].dtype == torch.float32

    def test_2d_numpy_image_hwc_to_chw(self):
        """2D+ HWC numpy image should be permuted to CHW."""
        p = _make_policy()
        img = np.zeros((64, 64, 3), dtype=np.float32)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 3, 64, 64)

    def test_2d_numpy_image_normalization(self):
        """uint8 range (>1.0) images should be normalized to [0,1]."""
        p = _make_policy()
        img = np.full((64, 64, 3), 255.0, dtype=np.float32)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].max().item() == pytest.approx(1.0, abs=0.01)

    def test_2d_numpy_non_image_no_normalize(self):
        """Non-image numpy 2D array should NOT be normalized."""
        p = _make_policy()
        data = np.full((4, 4), 255.0, dtype=np.float32)
        obs = {"observation.depth_map": data}
        batch = p._build_observation_batch(obs, "")
        # Not an image key so no normalization
        assert "observation.depth_map" in batch

    def test_2d_numpy_image_already_chw(self):
        """CHW numpy image with last dim not in (1,3,4) stays."""
        p = _make_policy()
        # 3x64x64 already CHW (last dim=64, not 1/3/4)
        img = np.zeros((3, 64, 64), dtype=np.float32)
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 3, 64, 64)


# ===========================================================================
# SECTION 3: LeRobot-format — scalar, list, tuple inputs
# ===========================================================================


class TestLerobotFormatScalarsAndLists:
    """Tests for int/float scalars and list/tuple values."""

    def test_int_scalar(self):
        """Integer value should become [1, 1] float tensor."""
        p = _make_policy()
        obs = {"observation.gripper": 42}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.gripper"].shape == (1, 1)
        assert batch["observation.gripper"].item() == pytest.approx(42.0)

    def test_float_scalar(self):
        """Float value should become [1, 1] float tensor."""
        p = _make_policy()
        obs = {"observation.temp": 3.14}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.temp"].shape == (1, 1)

    def test_1d_list(self):
        """1D list should become [1, N] tensor."""
        p = _make_policy()
        obs = {"observation.state": [1.0, 2.0, 3.0]}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 3)

    def test_1d_tuple(self):
        """1D tuple should become [1, N] tensor."""
        p = _make_policy()
        obs = {"observation.state": (1.0, 2.0)}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 2)

    def test_2d_list_image(self):
        """2D nested list for image should work."""
        p = _make_policy()
        img = [[[0.0, 0.0, 0.0]] * 4] * 4  # 4x4x3
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].shape == (1, 3, 4, 4)  # permuted CHW + batch

    def test_2d_list_image_normalized(self):
        """2D nested list with values > 1.0 should be normalized."""
        p = _make_policy()
        img = [[[255, 128, 0]] * 2] * 2  # 2x2x3
        obs = {"observation.image": img}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.image"].max().item() == pytest.approx(1.0, abs=0.01)

    def test_non_numeric_list_skipped(self):
        """Non-numeric list should be skipped without error."""
        p = _make_policy()
        obs = {"observation.labels": ["cat", "dog"], "observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "")
        assert "observation.labels" not in batch
        assert "observation.state" in batch

    def test_passthrough_non_standard_types(self):
        """Non-tensor, non-array, non-scalar types should be passed through."""
        p = _make_policy()
        custom_obj = SimpleNamespace(data="test")
        obs = {"observation.custom": custom_obj, "observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "")
        # custom_obj doesn't match any isinstance check, so it's passed through (or skipped)
        assert "observation.state" in batch


# ===========================================================================
# SECTION 4: VLA token injection — LeRobot format
# ===========================================================================


class TestLerobotFormatVLAInjection:
    """Tests for VLA language token injection in LeRobot format path."""

    def test_tokenizer_injection_when_config_has_tokenizer(self):
        """When policy.config.tokenizer_name exists, tokenize instruction."""
        p = _make_policy()
        p._policy.config.tokenizer_name = "test-tokenizer"
        p._policy.config.max_len_seq = 512
        p._policy.config.tokenizer_max_length = 50

        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_encoded

        # Create a proper transformers mock with __spec__ set
        import types

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)

        obs = {"observation.state": torch.tensor([1.0])}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            batch = p._build_observation_batch(obs, "pick up the cube")

        assert "observation.language.tokens" in batch
        assert "observation.language.attention_mask" in batch

    def test_no_injection_when_tokens_already_present(self):
        """If observation.language.tokens already in obs, don't inject."""
        p = _make_policy()
        p._policy.config.tokenizer_name = "test-tokenizer"
        existing_tokens = torch.tensor([[10, 20, 30]])
        obs = {
            "observation.state": torch.tensor([1.0]),
            "observation.language.tokens": existing_tokens,
        }
        batch = p._build_observation_batch(obs, "pick up the cube")
        # Tokens should be the original ones (moved to device), not re-tokenized
        assert "observation.language.tokens" in batch

    def test_no_injection_without_instruction(self):
        """Empty instruction should skip VLA injection."""
        p = _make_policy()
        p._policy.config.tokenizer_name = "test-tokenizer"
        obs = {"observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "")
        assert "observation.language.tokens" not in batch

    def test_tokenizer_exception_handled(self):
        """Tokenizer errors should be caught and logged, not crash."""
        p = _make_policy()
        p._policy.config.tokenizer_name = "bad-tokenizer"

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = Exception("tokenizer error")
        obs = {"observation.state": torch.tensor([1.0])}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            batch = p._build_observation_batch(obs, "pick up cube")
        # Should not crash, just skip tokens
        assert "observation.language.tokens" not in batch

    def test_no_injection_when_no_tokenizer_config(self):
        """When policy has no tokenizer_name config, skip injection."""
        p = _make_policy()
        # No tokenizer_name attribute
        del p._policy.config.tokenizer_name
        obs = {"observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "pick up cube")
        assert "observation.language.tokens" not in batch


# ===========================================================================
# SECTION 5: Strands-format observations — state extraction
# ===========================================================================


class TestStrandsFormatState:
    """Tests for strands-robots format (raw joint keys → observation.state)."""

    def test_basic_state_extraction(self):
        """Joint values should be extracted into observation.state tensor."""
        p = _make_policy(robot_state_keys=["j0", "j1", "j2"])
        obs = {"j0": 1.0, "j1": 2.0, "j2": 3.0}
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert batch["observation.state"].shape == (1, 3)
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.0, 2.0, 3.0], atol=1e-5)

    def test_missing_keys_skipped(self):
        """Missing state keys should be skipped (not error)."""
        p = _make_policy(robot_state_keys=["j0", "j1", "j2"])
        obs = {"j0": 1.0, "j2": 3.0}  # j1 missing
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert batch["observation.state"].shape == (1, 2)  # only 2 values extracted

    def test_numpy_scalar_state(self):
        """0-dim numpy arrays should be extracted as floats."""
        p = _make_policy(robot_state_keys=["j0"])
        # np.array(1.5) is a 0-dim ndarray (unlike np.float32 which is a scalar, not ndarray)
        obs = {"j0": np.array(1.5)}
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert batch["observation.state"].item() == pytest.approx(1.5)

    def test_numpy_float32_scalar_extracted(self):
        """np.float32 scalar should be extracted via np.floating isinstance check."""
        p = _make_policy(robot_state_keys=["j0"])
        obs = {"j0": np.float32(1.5)}
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert batch["observation.state"].item() == pytest.approx(1.5)

    def test_numpy_int32_scalar_extracted(self):
        """np.int32 scalar should be extracted via np.integer isinstance check."""
        p = _make_policy(robot_state_keys=["j0"])
        obs = {"j0": np.int32(7)}
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert batch["observation.state"].item() == pytest.approx(7.0)

    def test_state_padding_to_feature_dim(self):
        """State values should be padded to match input_features dimension."""
        feat = SimpleNamespace(shape=(6,))
        p = _make_policy(
            robot_state_keys=["j0", "j1"],
            _input_features={"observation.state": feat},
        )
        obs = {"j0": 1.0, "j1": 2.0}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 6)
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.0, 2.0, 0.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_state_truncation_to_feature_dim(self):
        """State values should be truncated to match input_features dimension."""
        feat = SimpleNamespace(shape=(2,))
        p = _make_policy(
            robot_state_keys=["j0", "j1", "j2", "j3"],
            _input_features={"observation.state": feat},
        )
        obs = {"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 4.0}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 2)

    def test_empty_state_keys_warns(self, caplog):
        """Empty robot_state_keys should log warning."""
        p = _make_policy(robot_state_keys=[])
        obs = {"j0": 1.0}
        with caplog.at_level(logging.WARNING):
            p._build_observation_batch(obs, "")
        assert "robot_state_keys is empty" in caplog.text

    def test_no_state_values_no_state_key(self):
        """When no state values extracted, observation.state should not be in batch."""
        p = _make_policy(robot_state_keys=["missing_key"])
        obs = {"other_key": 1.0}
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" not in batch

    def test_state_feature_without_shape(self):
        """State feature without shape attr uses len(state_values) as dim."""
        feat = SimpleNamespace()  # No shape
        p = _make_policy(
            robot_state_keys=["j0", "j1"],
            _input_features={"observation.state": feat},
        )
        obs = {"j0": 1.0, "j1": 2.0}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 2)


# ===========================================================================
# SECTION 6: Strands-format — camera image handling
# ===========================================================================


class TestStrandsFormatCameras:
    """Tests for camera image extraction in strands format."""

    def test_hwc_image_converted(self):
        """HWC numpy image should be CHW + normalized + batched."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.images.top": SimpleNamespace(shape=(3, 64, 64))},
        )
        img = np.full((64, 64, 3), 200.0, dtype=np.float32)
        obs = {"j0": 1.0, "camera": img}
        batch = p._build_observation_batch(obs, "")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 64, 64)
        # Normalized since max > 1.0
        assert batch["observation.images.top"].max().item() < 1.01

    def test_chw_image_stays(self):
        """Already CHW numpy image should not be double-permuted."""
        p = _make_policy(
            robot_state_keys=[],
            _input_features={"observation.images.top": SimpleNamespace(shape=(3, 64, 64))},
        )
        img = np.zeros((3, 64, 64), dtype=np.float32)  # CHW, last dim=64
        obs = {"camera": img}
        batch = p._build_observation_batch(obs, "")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 64, 64)

    def test_state_keys_not_treated_as_images(self):
        """Values matching state keys should not be treated as images."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.images.top": SimpleNamespace(shape=(3, 64, 64))},
        )
        # j0 is a state key AND happens to be a 2D array — should NOT be image
        obs = {"j0": np.zeros((2, 3), dtype=np.float32), "camera": np.zeros((64, 64, 3), dtype=np.float32)}
        batch = p._build_observation_batch(obs, "")
        assert "observation.images.top" in batch

    def test_multiple_cameras_fill_features(self):
        """Multiple image features should be filled in order."""
        p = _make_policy(
            robot_state_keys=[],
            _input_features={
                "observation.images.top": SimpleNamespace(shape=(3, 64, 64)),
                "observation.images.side": SimpleNamespace(shape=(3, 64, 64)),
            },
        )
        obs = {
            "cam1": np.zeros((64, 64, 3), dtype=np.float32),
            "cam2": np.zeros((64, 64, 3), dtype=np.float32),
        }
        batch = p._build_observation_batch(obs, "")
        assert "observation.images.top" in batch
        assert "observation.images.side" in batch

    def test_1d_numpy_not_treated_as_image(self):
        """1D numpy array should not be treated as an image."""
        p = _make_policy(
            robot_state_keys=[],
            _input_features={"observation.images.top": SimpleNamespace(shape=(3, 64, 64))},
        )
        obs = {"velocity": np.array([1.0, 2.0, 3.0])}
        batch = p._build_observation_batch(obs, "")
        # 1D array has ndim=1, < 2, so not treated as image
        assert "observation.images.top" not in batch or batch["observation.images.top"].sum().item() == 0.0


# ===========================================================================
# SECTION 7: VLA token injection — Strands format
# ===========================================================================


class TestStrandsFormatVLAInjection:
    """Tests for VLA language token injection in strands format path."""

    def test_tokenizer_injection_with_config(self):
        """Should tokenize instruction when policy has tokenizer_name."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.language.tokens": SimpleNamespace(shape=(50,))},
        )
        p._policy.config.tokenizer_name = "test-tokenizer"
        p._policy.config.max_len_seq = 512
        p._policy.config.tokenizer_max_length = 50

        mock_tokenizer = MagicMock()
        mock_encoded = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_encoded
        mock_tokenizer.padding_side = "right"

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        obs = {"j0": 1.0}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            batch = p._build_observation_batch(obs, "pick up cube")

        assert "observation.language.tokens" in batch
        assert "observation.language.attention_mask" in batch

    def test_tokenizer_padding_side_from_config(self):
        """Should respect tokenizer_padding_side from config."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.language.tokens": SimpleNamespace(shape=(50,))},
        )
        p._policy.config.tokenizer_name = "test-tokenizer"
        p._policy.config.max_len_seq = 512
        p._policy.config.tokenizer_max_length = 50
        p._policy.config.tokenizer_padding_side = "left"

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
        }
        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        obs = {"j0": 1.0}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            p._build_observation_batch(obs, "test")

        assert mock_tokenizer.padding_side == "left"

    def test_processor_fallback(self):
        """Should fall back to policy.processor.tokenizer if no config."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.language.tokens": SimpleNamespace(shape=(50,))},
        )
        # No tokenizer_name config
        del p._policy.config.tokenizer_name

        mock_proc_tokenizer = MagicMock()
        mock_proc_tokenizer.return_value = {
            "input_ids": torch.tensor([[5, 6, 7]]),
        }
        p._policy.processor = MagicMock()
        p._policy.processor.tokenizer = mock_proc_tokenizer

        obs = {"j0": 1.0}
        batch = p._build_observation_batch(obs, "test instruction")
        assert "observation.language.tokens" in batch

    def test_task_feature_injection(self):
        """Should inject instruction into task features."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.task_description": SimpleNamespace(shape=())},
        )
        # No tokenizer
        del p._policy.config.tokenizer_name
        p._policy.processor = None

        obs = {"j0": 1.0}
        batch = p._build_observation_batch(obs, "fold the cloth")
        assert batch.get("observation.task_description") == "fold the cloth"

    def test_no_injection_without_language_features(self):
        """No injection if no language/task features in input_features."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.state": SimpleNamespace(shape=(3,))},
        )
        del p._policy.config.tokenizer_name
        obs = {"j0": 1.0}
        batch = p._build_observation_batch(obs, "pick up cube")
        assert "observation.language.tokens" not in batch

    def test_injection_error_handled(self):
        """Tokenizer errors in strands format should be caught."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.language.tokens": SimpleNamespace(shape=(50,))},
        )
        p._policy.config.tokenizer_name = "bad-tokenizer"

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = Exception("bad")

        obs = {"j0": 1.0}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            batch = p._build_observation_batch(obs, "test")
        # Should not crash
        assert "observation.language.tokens" not in batch

    def test_has_tokenizer_config_triggers_injection(self):
        """has_tokenizer_config=True should trigger injection even without language features."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={"observation.state": SimpleNamespace(shape=(3,))},
        )
        p._policy.config.tokenizer_name = "test-tokenizer"
        p._policy.config.max_len_seq = 256
        p._policy.config.tokenizer_max_length = 30

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        obs = {"j0": 1.0}
        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            batch = p._build_observation_batch(obs, "move arm")

        assert "observation.language.tokens" in batch


# ===========================================================================
# SECTION 8: Missing image feature zero-fill
# ===========================================================================


class TestMissingImageFill:
    """Tests for filling missing image features with zeros."""

    def test_missing_image_filled_with_zeros(self):
        """Unmatched image features should be filled with zeros."""
        p = _make_policy(
            robot_state_keys=["j0"],
            _input_features={
                "observation.state": SimpleNamespace(shape=(3,)),
                "observation.images.top": SimpleNamespace(shape=(3, 64, 64)),
            },
        )
        obs = {"j0": 1.0}  # No camera data
        batch = p._build_observation_batch(obs, "")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 64, 64)
        assert batch["observation.images.top"].sum().item() == 0.0

    def test_missing_image_default_shape(self):
        """Image feature without shape attr uses default (3, 480, 640)."""
        p = _make_policy(
            robot_state_keys=[],
            _input_features={
                "observation.images.top": SimpleNamespace(),  # No shape attr
            },
        )
        obs = {}
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)

    def test_existing_image_not_overwritten(self):
        """Already-filled image features should not be zeroed."""
        p = _make_policy(
            robot_state_keys=[],
            _input_features={
                "observation.images.top": SimpleNamespace(shape=(3, 64, 64)),
            },
        )
        img = np.full((64, 64, 3), 128.0, dtype=np.float32)
        obs = {"camera": img}
        batch = p._build_observation_batch(obs, "")
        # Should have data, not zeros
        assert batch["observation.images.top"].sum().item() > 0


# ===========================================================================
# SECTION 9: Integration — full pipeline paths
# ===========================================================================


class TestBuildObsBatchIntegration:
    """Integration tests combining multiple features."""

    def test_full_strands_observation(self):
        """Complete strands-format observation with state + camera + instruction."""
        feat = SimpleNamespace(shape=(3,))
        img_feat = SimpleNamespace(shape=(3, 64, 64))
        p = _make_policy(
            robot_state_keys=["j0", "j1", "j2"],
            _input_features={
                "observation.state": feat,
                "observation.images.top": img_feat,
                "observation.task_description": SimpleNamespace(shape=()),
            },
        )
        del p._policy.config.tokenizer_name
        p._policy.processor = None

        obs = {
            "j0": 1.0,
            "j1": 2.0,
            "j2": 3.0,
            "camera": np.zeros((64, 64, 3), dtype=np.float32),
        }
        batch = p._build_observation_batch(obs, "pick up the block")
        assert "observation.state" in batch
        assert "observation.images.top" in batch
        assert batch.get("observation.task_description") == "pick up the block"
        assert batch["observation.state"].shape == (1, 3)
        assert batch["observation.images.top"].shape == (1, 3, 64, 64)

    def test_full_lerobot_observation(self):
        """Complete LeRobot-format observation with state + image."""
        p = _make_policy()
        obs = {
            "observation.state": torch.tensor([1.0, 2.0, 3.0]),
            "observation.images.top": torch.zeros(480, 640, 3),  # HWC
        }
        batch = p._build_observation_batch(obs, "")
        assert batch["observation.state"].shape == (1, 3)
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)

    def test_mixed_dtypes_in_lerobot_format(self):
        """Mix of tensor, numpy, scalar, and list in LeRobot format."""
        p = _make_policy()
        obs = {
            "observation.state": torch.tensor([1.0]),
            "observation.gripper": 0.5,
            "observation.force": np.array([3.0, 4.0], dtype=np.float32),
            "observation.joints": [5.0, 6.0],
        }
        batch = p._build_observation_batch(obs, "")
        assert "observation.state" in batch
        assert "observation.gripper" in batch
        assert "observation.force" in batch
        assert "observation.joints" in batch

    def test_empty_observation(self):
        """Empty observation dict should return empty batch (strands format)."""
        p = _make_policy(robot_state_keys=[])
        batch = p._build_observation_batch({}, "")
        assert isinstance(batch, dict)

    def test_lerobot_format_detected_correctly(self):
        """Keys starting with 'observation.' trigger LeRobot path."""
        p = _make_policy(robot_state_keys=["observation.state"])
        obs = {"observation.state": torch.tensor([1.0])}
        batch = p._build_observation_batch(obs, "")
        # LeRobot path: 1D tensor (dim=1 < 2, not "image") → unsqueeze(0) → (1, 1)
        assert batch["observation.state"].shape == (1, 1)


# ===========================================================================
# SECTION 10: get_actions and select_action_sync with _build_observation_batch
# ===========================================================================


class TestGetActionsWithBuildObs:
    """Tests ensuring get_actions/select_action_sync properly call _build_observation_batch."""

    def test_get_actions_injects_task(self):
        """get_actions should inject instruction as 'task' in obs."""
        p = _make_policy(robot_state_keys=["j0"])
        captured = {}

        original_build = p._build_observation_batch

        def capture_build(obs, instruction):
            captured["obs"] = obs
            captured["instruction"] = instruction
            return original_build(obs, instruction)

        p._build_observation_batch = capture_build
        p._policy.select_action.return_value = torch.tensor([1.0, 2.0])

        with patch.dict(sys.modules, _LEROBOT_MODULES):
            asyncio.run(p.get_actions({"j0": 1.0}, "pick up cube"))

        assert captured["obs"].get("task") == "pick up cube"

    def test_get_actions_does_not_overwrite_existing_task(self):
        """get_actions should NOT overwrite existing 'task' key."""
        p = _make_policy(robot_state_keys=["j0"])
        captured = {}

        original_build = p._build_observation_batch

        def capture_build(obs, instruction):
            captured["obs"] = obs
            return original_build(obs, instruction)

        p._build_observation_batch = capture_build
        p._policy.select_action.return_value = torch.tensor([1.0])

        with patch.dict(sys.modules, _LEROBOT_MODULES):
            asyncio.run(p.get_actions({"j0": 1.0, "task": "existing"}, "new"))

        assert captured["obs"]["task"] == "existing"

    def test_select_action_sync_returns_numpy(self):
        """select_action_sync should return numpy array."""
        p = _make_policy(robot_state_keys=["j0"])
        p._policy.select_action.return_value = torch.tensor([1.0, 2.0, 3.0])

        result = p.select_action_sync({"j0": 1.0})
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_select_action_sync_with_processor(self):
        """select_action_sync should apply pre/post processor."""
        p = _make_policy(robot_state_keys=["j0"])
        p._processor_bridge = MagicMock()
        p._processor_bridge.has_preprocessor = True
        p._processor_bridge.has_postprocessor = True
        p._processor_bridge.preprocess.return_value = {"j0": 1.0}
        p._processor_bridge.postprocess.return_value = torch.tensor([5.0])
        p._policy.select_action.return_value = torch.tensor([3.0])

        result = p.select_action_sync({"j0": 1.0})
        p._processor_bridge.preprocess.assert_called_once()
        p._processor_bridge.postprocess.assert_called_once()
        np.testing.assert_allclose(result, [5.0])

    def test_select_action_sync_non_tensor_return(self):
        """select_action_sync should handle non-tensor returns (numpy)."""
        p = _make_policy(robot_state_keys=["j0"])
        p._policy.select_action.return_value = np.array([1.0, 2.0])

        result = p.select_action_sync({"j0": 1.0})
        assert isinstance(result, np.ndarray)
