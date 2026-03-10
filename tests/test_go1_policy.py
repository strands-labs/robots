#!/usr/bin/env python3
"""Comprehensive tests for the GO-1 (AgiBot World) policy provider.

Tests cover:
1. Policy ABC compliance and construction
2. Pure utility methods (no GPU/network): _extract_cameras, _extract_state,
   _actions_to_dicts, _generate_zero_actions, _to_numpy_image,
   _go1_key_to_server_key, _build_server_payload, _basic_image_preprocess
3. Server mode: _ensure_loaded, _infer_server with mocked requests
4. Local mode: _ensure_loaded with mocked torch/transformers
5. Policy registry: registration, aliases, auto-discovery
6. Policy resolver: HF model ID routing, shorthand, overrides
7. get_actions() with pre-loaded model (bypass loading)
8. Normalization / unnormalization
9. Edge cases and error paths
"""

import asyncio
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Async helper
def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════
# 1. Policy ABC compliance and construction
# ═══════════════════════════════════════════════════════════════════


class TestGo1PolicyConstruction:
    """Test Go1Policy instantiation and properties."""

    def test_default_construction(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        assert p.provider_name == "go1"
        assert p._model_id == "agibot-world/GO-1"
        assert p._server_url is None
        assert p._action_dim == 16
        assert p._action_chunk_size == 30
        assert p._ctrl_freq == 30.0
        assert p._loaded is False
        assert p._step == 0

    def test_server_mode_construction(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://localhost:9000")
        assert p._server_url == "http://localhost:9000"

    def test_custom_model_id(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(model_id="agibot-world/GO-1-Air")
        assert p._model_id == "agibot-world/GO-1-Air"

    def test_pretrained_name_or_path_alias(self):
        """pretrained_name_or_path overrides model_id (for policy_resolver compat)."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(pretrained_name_or_path="agibot-world/GO-1-Air")
        assert p._model_id == "agibot-world/GO-1-Air"

    def test_custom_action_dim(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(action_dim=7)
        assert p._action_dim == 7

    def test_custom_chunk_size(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(action_chunk_size=10)
        assert p._action_chunk_size == 10

    def test_custom_ctrl_freq(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(ctrl_freq=50.0)
        assert p._ctrl_freq == 50.0

    def test_normalize_flag(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(normalize=True)
        assert p._normalize is True

    def test_custom_camera_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        keys = ["cam1", "cam2"]
        p = Go1Policy(camera_keys=keys)
        assert p._camera_keys == keys

    def test_is_policy_subclass(self):
        from strands_robots.policies import Policy
        from strands_robots.policies.go1 import Go1Policy

        assert issubclass(Go1Policy, Policy)
        p = Go1Policy()
        assert isinstance(p, Policy)

    def test_provider_name_property(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        assert p.provider_name == "go1"

    def test_set_robot_state_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        keys = ["j0", "j1", "j2"]
        p.set_robot_state_keys(keys)
        assert p._robot_state_keys == keys

    def test_kwargs_ignored(self):
        """Extra kwargs are accepted and ignored."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(some_extra_param="value", another=42)
        assert p.provider_name == "go1"

    def test_lazy_loading(self):
        """Model should not be loaded at construction time."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        assert p._model is None
        assert p._loaded is False

    def test_reset(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p._step = 42
        p.reset()
        assert p._step == 0


# ═══════════════════════════════════════════════════════════════════
# 2. Pure utility methods
# ═══════════════════════════════════════════════════════════════════


class TestExtractCameras:
    """Test _extract_cameras() with various observation formats."""

    def test_direct_go1_camera_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {
            "cam_head_color": np.ones((100, 100, 3), dtype=np.uint8) * 128,
            "cam_hand_right_color": np.ones((100, 100, 3), dtype=np.uint8) * 64,
            "cam_hand_left_color": np.ones((100, 100, 3), dtype=np.uint8) * 32,
        }
        cameras = p._extract_cameras(obs)
        assert cameras["cam_head_color"] is not None
        assert cameras["cam_hand_right_color"] is not None
        assert cameras["cam_hand_left_color"] is not None
        assert cameras["cam_head_color"][0, 0, 0] == 128

    def test_alias_camera_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {
            "head": np.ones((100, 100, 3), dtype=np.uint8) * 10,
            "right_hand": np.ones((100, 100, 3), dtype=np.uint8) * 20,
            "left_hand": np.ones((100, 100, 3), dtype=np.uint8) * 30,
        }
        cameras = p._extract_cameras(obs)
        assert cameras["cam_head_color"] is not None
        assert cameras["cam_hand_right_color"] is not None
        assert cameras["cam_hand_left_color"] is not None
        assert cameras["cam_head_color"][0, 0, 0] == 10

    def test_fallback_aliases(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {
            "top": np.ones((100, 100, 3), dtype=np.uint8) * 50,
            "right": np.ones((80, 80, 3), dtype=np.uint8) * 60,
            "left": np.ones((80, 80, 3), dtype=np.uint8) * 70,
        }
        cameras = p._extract_cameras(obs)
        assert cameras["cam_head_color"] is not None
        assert cameras["cam_head_color"][0, 0, 0] == 50

    def test_fallback_any_image(self):
        """When no cameras match, fall back to any image-like array."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"random_cam": np.ones((100, 100, 3), dtype=np.uint8) * 99}
        cameras = p._extract_cameras(obs)
        assert cameras["cam_head_color"] is not None
        assert cameras["cam_head_color"][0, 0, 0] == 99

    def test_no_images_returns_none(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"scalar": 42.0, "vector": np.array([1.0, 2.0])}
        cameras = p._extract_cameras(obs)
        assert all(v is None for v in cameras.values())

    def test_rgba_stripped_to_rgb(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"head": np.ones((100, 100, 4), dtype=np.uint8) * 55}
        cameras = p._extract_cameras(obs)
        assert cameras["cam_head_color"] is not None
        assert cameras["cam_head_color"].shape == (100, 100, 3)

    def test_chw_format_converted(self):
        from strands_robots.policies.go1 import Go1Policy

        img = np.ones((3, 100, 100), dtype=np.uint8) * 77
        result = Go1Policy._to_numpy_image(img)
        assert result is not None
        assert result.shape == (100, 100, 3)
        assert result[0, 0, 0] == 77


class TestToNumpyImage:
    """Test _to_numpy_image static method."""

    def test_hwc_rgb(self):
        from strands_robots.policies.go1 import Go1Policy

        img = np.ones((50, 50, 3), dtype=np.uint8) * 42
        result = Go1Policy._to_numpy_image(img)
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_hwc_rgba(self):
        from strands_robots.policies.go1 import Go1Policy

        img = np.ones((50, 50, 4), dtype=np.uint8) * 42
        result = Go1Policy._to_numpy_image(img)
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_chw(self):
        from strands_robots.policies.go1 import Go1Policy

        img = np.ones((3, 50, 50), dtype=np.uint8) * 42
        result = Go1Policy._to_numpy_image(img)
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_non_image_returns_none(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy._to_numpy_image(np.array([1.0, 2.0])) is None
        assert Go1Policy._to_numpy_image(np.zeros((10, 10))) is None
        assert Go1Policy._to_numpy_image("not_an_image") is None

    def test_pil_image(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        result = Go1Policy._to_numpy_image(img)
        assert result is not None
        assert result.shape == (50, 50, 3)
        assert result[0, 0, 0] == 128


class TestExtractState:
    """Test _extract_state() with various observation formats."""

    def test_from_robot_state_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        obs = {"j0": 0.1, "j1": 0.2, "j2": 0.3}
        state = p._extract_state(obs)
        assert state.shape == (16,)  # Padded to action_dim
        np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[3:], 0.0)

    def test_from_observation_state_array(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"observation.state": np.array([0.5] * 16, dtype=np.float32)}
        state = p._extract_state(obs)
        assert state.shape == (16,)
        np.testing.assert_allclose(state, 0.5)

    def test_from_state_key(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"state": np.array([1.0, 2.0, 3.0])}
        state = p._extract_state(obs)
        assert state.shape == (16,)
        np.testing.assert_allclose(state[:3], [1.0, 2.0, 3.0])

    def test_from_qpos_key(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"qpos": np.array([0.1] * 20)}
        state = p._extract_state(obs)
        assert state.shape == (16,)  # Truncated

    def test_missing_state_returns_zeros(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        state = p._extract_state({"camera": np.zeros((100, 100, 3))})
        assert state.shape == (16,)
        np.testing.assert_allclose(state, 0.0)

    def test_state_keys_with_missing_values(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        obs = {"j0": 1.0}  # j1, j2 missing → 0.0
        state = p._extract_state(obs)
        assert state[0] == pytest.approx(1.0)
        assert state[1] == pytest.approx(0.0)


class TestActionsToDicts:
    """Test _actions_to_dicts() conversion."""

    def test_basic_conversion(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        actions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = p._actions_to_dicts(actions)
        assert len(result) == 2
        assert result[0] == {"j0": pytest.approx(0.1), "j1": pytest.approx(0.2), "j2": pytest.approx(0.3)}
        assert result[1] == {"j0": pytest.approx(0.4), "j1": pytest.approx(0.5), "j2": pytest.approx(0.6)}

    def test_1d_input(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1"])
        actions = np.array([0.1, 0.2])
        result = p._actions_to_dicts(actions)
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.1)

    def test_default_joint_keys(self):
        """Without robot_state_keys set, uses _DEFAULT_JOINT_KEYS."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        actions = np.zeros((2, 16))
        actions[0, 0] = 1.0
        actions[0, 7] = 0.5  # left gripper
        result = p._actions_to_dicts(actions)
        assert len(result) == 2
        assert "left_arm_joint_0" in result[0]
        assert "left_gripper" in result[0]
        assert result[0]["left_arm_joint_0"] == pytest.approx(1.0)
        assert result[0]["left_gripper"] == pytest.approx(0.5)

    def test_action_dim_mismatch_pads(self):
        """When action array is shorter than keys, remaining keys get 0.0."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1", "j2", "j3"])
        actions = np.array([[0.1, 0.2]])  # Only 2 values
        result = p._actions_to_dicts(actions)
        assert result[0]["j0"] == pytest.approx(0.1)
        assert result[0]["j1"] == pytest.approx(0.2)
        assert result[0]["j2"] == pytest.approx(0.0)
        assert result[0]["j3"] == pytest.approx(0.0)


class TestGenerateZeroActions:
    """Test _generate_zero_actions()."""

    def test_default_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        result = p._generate_zero_actions()
        assert len(result) == 30  # Default chunk size
        assert "left_arm_joint_0" in result[0]
        assert all(v == 0.0 for v in result[0].values())

    def test_custom_keys(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(action_chunk_size=5)
        p.set_robot_state_keys(["j0", "j1"])
        result = p._generate_zero_actions()
        assert len(result) == 5
        assert all(a == {"j0": 0.0, "j1": 0.0} for a in result)


class TestGo1KeyToServerKey:
    """Test _go1_key_to_server_key static method."""

    def test_head(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy._go1_key_to_server_key("cam_head_color") == "top"

    def test_right_hand(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy._go1_key_to_server_key("cam_hand_right_color") == "right"

    def test_left_hand(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy._go1_key_to_server_key("cam_hand_left_color") == "left"

    def test_unknown_passthrough(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy._go1_key_to_server_key("unknown_cam") == "unknown_cam"


class TestBuildServerPayload:
    """Test _build_server_payload()."""

    def test_basic_payload(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"head": np.ones((100, 100, 3), dtype=np.uint8) * 42}
        payload = p._build_server_payload(obs, "pick up the cup")
        assert payload["instruction"] == "pick up the cup"
        assert "top" in payload  # head → top
        assert "state" in payload
        assert "ctrl_freqs" in payload
        assert payload["ctrl_freqs"] == [30.0]

    def test_state_included(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p.set_robot_state_keys(["j0", "j1"])
        obs = {"j0": 0.5, "j1": 0.6}
        payload = p._build_server_payload(obs, "test")
        state = payload["state"]
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.6)

    def test_custom_ctrl_freq(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {}
        payload = p._build_server_payload(obs, "test", ctrl_freq=50.0)
        assert payload["ctrl_freqs"] == [50.0]


class TestBasicImagePreprocess:
    """Test _basic_image_preprocess fallback (requires torch)."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_produces_tensor(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        images = [Image.new("RGB", (100, 100))]
        result = Go1Policy._basic_image_preprocess(images)
        assert result.shape == (1, 3, 448, 448)

    def test_multiple_images(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (200, 200))]
        result = Go1Policy._basic_image_preprocess(images)
        assert result.shape == (2, 3, 448, 448)

    def test_normalized_range(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        images = [Image.new("RGB", (448, 448), color=(128, 128, 128))]
        result = Go1Policy._basic_image_preprocess(images)
        # Values should be roughly centered around 0 after ImageNet normalization
        assert result.mean().item() < 2.0  # Not raw pixel values


class TestNormalization:
    """Test _normalize_tensor and _unnormalize_tensor (requires torch)."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_normalize_unnormalize_roundtrip(self):
        import torch

        from strands_robots.policies.go1 import Go1Policy

        data = torch.tensor([1.0, 2.0, 3.0])
        stats = {
            "mean": torch.tensor([0.5, 1.0, 1.5]),
            "std": torch.tensor([0.5, 0.5, 0.5]),
        }
        normalized = Go1Policy._normalize_tensor(data, stats)
        recovered = Go1Policy._unnormalize_tensor(normalized, stats)
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_normalize_values(self):
        import torch

        from strands_robots.policies.go1 import Go1Policy

        data = torch.tensor([1.0, 2.0])
        stats = {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0, 1.0])}
        result = Go1Policy._normalize_tensor(data, stats)
        assert torch.allclose(result, data)  # With 0 mean, 1 std → identity


class TestExtractImage:
    """Test _extract_image (compatibility method)."""

    def test_returns_pil(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        obs = {"head": np.ones((100, 100, 3), dtype=np.uint8) * 55}
        img = p._extract_image(obs)
        assert isinstance(img, Image.Image)

    def test_no_cameras_returns_default(self):
        from PIL import Image

        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        img = p._extract_image({"scalar": 42.0})
        assert isinstance(img, Image.Image)
        assert img.size == (448, 448)


# ═══════════════════════════════════════════════════════════════════
# 3. Server mode: _ensure_loaded + _infer_server
# ═══════════════════════════════════════════════════════════════════


class TestEnsureLoadedServerMode:
    """Test _ensure_loaded() for server mode."""

    def test_server_connected(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            p._ensure_loaded()
        assert p._loaded is True

    def test_server_unreachable_warns(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        with patch("requests.get", side_effect=ConnectionError("refused")):
            p._ensure_loaded()
        assert p._loaded is True  # Still marks loaded (just warns)

    def test_idempotent(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True
        p._ensure_loaded()  # Should not call requests
        assert p._loaded is True


class TestInferServer:
    """Test _infer_server with mocked HTTP."""

    def test_basic_inference(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True
        p.set_robot_state_keys(["j0", "j1"])

        # Mock server returns 2 actions, 2 dims each
        mock_resp = MagicMock()
        mock_resp.json.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp.raise_for_status = MagicMock()

        obs = {"head": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "pick up"))

        assert len(actions) == 2
        assert actions[0]["j0"] == pytest.approx(0.1)
        assert actions[1]["j1"] == pytest.approx(0.4)
        assert p._step == 1

    def test_1d_response(self):
        """Server returns flat 1D array."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True
        p.set_robot_state_keys(["j0", "j1"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.5, 0.6]
        mock_resp.raise_for_status = MagicMock()

        obs = {}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "test"))

        assert len(actions) == 1
        assert actions[0]["j0"] == pytest.approx(0.5)

    def test_full_16dim_response(self):
        """Server returns full 30×16 action chunk."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True

        # 30 steps × 16 dims
        mock_resp = MagicMock()
        mock_resp.json.return_value = np.zeros((30, 16)).tolist()
        mock_resp.raise_for_status = MagicMock()

        obs = {"head": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "test"))

        assert len(actions) == 30
        assert "left_arm_joint_0" in actions[0]  # Default keys


# ═══════════════════════════════════════════════════════════════════
# 4. Local mode: _ensure_loaded with mocked torch/transformers
# ═══════════════════════════════════════════════════════════════════


class TestEnsureLoadedLocalMode:
    """Test _ensure_loaded() local mode with mocked imports."""

    def _mock_torch_and_transformers(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"

        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=2600000000))]
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoConfig.from_pretrained.return_value = MagicMock(
            force_image_size=448,
            vision_config=MagicMock(patch_size=14),
            downsample_ratio=0.5,
        )
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        return mock_torch, mock_transformers, mock_model

    def test_local_load_success(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()

        assert p._loaded is True
        assert p._model is mock_model
        assert p._device == "cpu"

    def test_local_load_failure_raises(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_tf = MagicMock()
        mock_tf.AutoConfig.from_pretrained.side_effect = RuntimeError("OOM")

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            with pytest.raises(ImportError, match="Failed to load GO-1"):
                p._ensure_loaded()


class TestInferLocal:
    """Test _infer_local with pre-loaded model."""

    def test_no_model_returns_zeros(self):
        """When model is None, _infer_local returns zero actions (needs torch import)."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p._loaded = True
        p._model = None
        p.set_robot_state_keys(["j0", "j1"])

        mock_torch = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p._infer_local({}, "test")
        assert len(result) == 30  # chunk size
        assert all(v == 0.0 for v in result[0].values())


# ═══════════════════════════════════════════════════════════════════
# 5. Policy registry
# ═══════════════════════════════════════════════════════════════════


class TestPolicyRegistry:
    """Test GO-1 registration in the global policy registry."""

    def test_go1_in_providers(self):
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert "go1" in providers

    def test_aliases_registered(self):
        from strands_robots.policies import list_providers

        providers = list_providers()
        for alias in ["go_1", "agibot_go1", "agibot_world", "go1_air"]:
            assert alias in providers, f"Alias '{alias}' not in providers"

    def test_create_policy_by_name(self):
        from strands_robots.policies import create_policy

        policy = create_policy("go1")
        assert policy.provider_name == "go1"

    def test_create_policy_by_alias(self):
        from strands_robots.policies import create_policy

        policy = create_policy("agibot_go1")
        assert policy.provider_name == "go1"

    def test_create_policy_with_kwargs(self):
        from strands_robots.policies import create_policy

        policy = create_policy("go1", server_url="http://test:9000")
        assert policy._server_url == "http://test:9000"

    def test_auto_discovery(self):
        """Go1Policy should also be auto-discoverable via convention."""
        from strands_robots.policies import PolicyRegistry

        reg = PolicyRegistry()
        cls = reg._auto_discover("go1")
        if cls is not None:
            assert cls.__name__ == "Go1Policy"


# ═══════════════════════════════════════════════════════════════════
# 6. Policy resolver
# ═══════════════════════════════════════════════════════════════════


class TestPolicyResolver:
    """Test GO-1 routing in the policy resolver."""

    def test_hf_org_routing(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("agibot-world/GO-1")
        assert provider == "go1"

    def test_hf_air_variant_routing(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("agibot-world/GO-1-Air")
        assert provider == "go1"

    def test_shorthand_routing(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("go1")
        assert provider == "go1"
        assert kwargs == {}

    def test_model_id_override(self):
        from strands_robots.policy_resolver import _MODEL_ID_OVERRIDES

        assert "agibot-world/go-1" in _MODEL_ID_OVERRIDES
        assert _MODEL_ID_OVERRIDES["agibot-world/go-1"] == "go1"

    def test_org_mapping(self):
        from strands_robots.policy_resolver import _HF_ORG_TO_PROVIDER

        assert "agibot-world" in _HF_ORG_TO_PROVIDER
        assert _HF_ORG_TO_PROVIDER["agibot-world"] == "go1"

    def test_shorthand_mapping(self):
        from strands_robots.policy_resolver import _SHORTHAND_TO_PROVIDER

        assert "go1" in _SHORTHAND_TO_PROVIDER
        assert _SHORTHAND_TO_PROVIDER["go1"] == "go1"

    def test_hf_routing_produces_model_id_kwarg(self):
        """When going through HF org routing, should produce model_id kwarg."""
        from strands_robots.policy_resolver import resolve_policy

        # Via org (not override)
        provider, kwargs = resolve_policy("agibot-world/some-new-go1-variant")
        assert provider == "go1"
        assert "model_id" in kwargs

    def test_extra_kwargs_merged(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("go1", action_dim=7)
        assert kwargs.get("action_dim") == 7


# ═══════════════════════════════════════════════════════════════════
# 7. get_actions() integration (bypass loading)
# ═══════════════════════════════════════════════════════════════════


class TestGetActionsIntegration:
    """Test full get_actions pipeline with pre-loaded mocks."""

    def test_server_mode_get_actions(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True
        p.set_robot_state_keys(["j0", "j1", "j2"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_resp.raise_for_status = MagicMock()

        obs = {
            "head": np.zeros((100, 100, 3), dtype=np.uint8),
            "j0": 0.5,
            "j1": 0.6,
            "j2": 0.7,
        }
        with patch("requests.post", return_value=mock_resp) as mock_post:
            actions = _run(p.get_actions(obs, "pick the cube"))

        # Verify the request was made correctly
        call_kwargs = mock_post.call_args
        assert "http://fake:9000/act" in str(call_kwargs)

        assert len(actions) == 2
        assert actions[0]["j0"] == pytest.approx(0.1)
        assert p._step == 1

    def test_successive_calls_increment_step(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True
        p.set_robot_state_keys(["j0"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = [[0.1]]
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            _run(p.get_actions({}, "test"))
            assert p._step == 1
            _run(p.get_actions({}, "test"))
            assert p._step == 2


# ═══════════════════════════════════════════════════════════════════
# 8. Data stats and normalization
# ═══════════════════════════════════════════════════════════════════


class TestDataStats:
    """Test dataset normalization stats handling."""

    def test_default_stats_shape(self):
        from strands_robots.policies.go1 import _DEFAULT_STATS

        assert _DEFAULT_STATS["state"]["mean"].shape == (16,)
        assert _DEFAULT_STATS["state"]["std"].shape == (16,)
        assert _DEFAULT_STATS["action"]["mean"].shape == (16,)
        assert _DEFAULT_STATS["action"]["std"].shape == (16,)

    def test_stats_values_match_hf_config(self):
        """Verify baked-in stats match the official dataset_stats.json."""
        from strands_robots.policies.go1 import _DEFAULT_STATS

        # First value of state mean
        assert _DEFAULT_STATS["state"]["mean"][0] == pytest.approx(-1.0137895, abs=1e-5)
        # Gripper std should be 1.0 (binary)
        assert _DEFAULT_STATS["state"]["std"][7] == pytest.approx(1.0)
        assert _DEFAULT_STATS["state"]["std"][15] == pytest.approx(1.0)

    def test_load_data_stats_from_file(self, tmp_path):
        """Test loading normalization stats from a JSON file (requires torch)."""
        from strands_robots.policies.go1 import Go1Policy

        stats_file = tmp_path / "dataset_stats.json"
        stats_file.write_text(
            json.dumps(
                {
                    "state": {"mean": [0.0] * 16, "std": [1.0] * 16},
                    "action": {"mean": [0.0] * 16, "std": [1.0] * 16},
                }
            )
        )

        p = Go1Policy(data_stats_path=str(stats_file))

        # Mock torch.from_numpy so _load_data_stats works without real torch
        mock_torch = MagicMock()
        mock_torch.from_numpy = lambda arr: MagicMock(shape=arr.shape)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            p._load_data_stats()

        assert p._data_stats is not None
        assert "state" in p._data_stats
        assert "action" in p._data_stats

    def test_normalize_with_default_stats(self):
        """Test normalization with default stats (requires torch)."""
        torch = pytest.importorskip("torch")
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(normalize=True)

        p._data_stats = {
            name: {
                "mean": torch.zeros(16),
                "std": torch.ones(16),
            }
            for name in ["state", "action"]
        }
        # With mean=0, std=1, normalize should be identity
        data = torch.ones(16)
        result = Go1Policy._normalize_tensor(data, p._data_stats["state"])
        assert torch.allclose(result, data)


# ═══════════════════════════════════════════════════════════════════
# 9. Edge cases and error paths
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_observation(self):
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True

        mock_resp = MagicMock()
        mock_resp.json.return_value = [[0.0] * 16]
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions({}, "do nothing"))
        assert len(actions) == 1

    def test_camera_aliases_exhaustive(self):
        """Test all camera aliases are reachable."""
        from strands_robots.policies.go1 import _CAMERA_ALIASES, Go1Policy

        p = Go1Policy()

        for go1_key, aliases in _CAMERA_ALIASES.items():
            for alias in aliases:
                obs = {alias: np.ones((50, 50, 3), dtype=np.uint8)}
                cameras = p._extract_cameras(obs)
                assert cameras[go1_key] is not None, f"Alias '{alias}' didn't map to '{go1_key}'"

    def test_default_joint_keys_count(self):
        from strands_robots.policies.go1 import _DEFAULT_JOINT_KEYS

        assert len(_DEFAULT_JOINT_KEYS) == 16
        assert _DEFAULT_JOINT_KEYS[7] == "left_gripper"
        assert _DEFAULT_JOINT_KEYS[15] == "right_gripper"

    def test_preprocess_inputs_no_cameras(self):
        """_preprocess_inputs handles empty camera input (requires torch)."""
        torch = pytest.importorskip("torch")
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy()
        p._config = MagicMock(
            force_image_size=448,
            vision_config=MagicMock(patch_size=14),
            downsample_ratio=0.5,
        )
        p._tokenizer = MagicMock()
        p._tokenizer.return_value = {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }
        p._img_transform = None

        raw_target = {"final_prompt": "test"}
        result = p._preprocess_inputs(raw_target)
        assert "pixel_values" in result
        assert "input_ids" in result

    def test_server_http_error_propagates(self):
        """HTTP errors from server should propagate."""
        from strands_robots.policies.go1 import Go1Policy

        p = Go1Policy(server_url="http://fake:9000")
        p._loaded = True

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Internal Server Error")

        obs = {}
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                _run(p.get_actions(obs, "test"))


# ═══════════════════════════════════════════════════════════════════
# 10. Module exports
# ═══════════════════════════════════════════════════════════════════


class TestModuleExports:
    """Test module structure and exports."""

    def test_all_export(self):
        from strands_robots.policies.go1 import __all__

        assert "Go1Policy" in __all__

    def test_import_from_package(self):
        from strands_robots.policies.go1 import Go1Policy

        assert Go1Policy is not None

    def test_provider_count_increased(self):
        """Adding GO-1 should increase the total provider count."""
        from strands_robots.policies import list_providers

        providers = list_providers()
        # Previous was 17 canonical + aliases, now 18+
        assert len(providers) > 30
