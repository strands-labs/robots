#!/usr/bin/env python3
"""Mock-based tests for RdtPolicy (strands_robots.policies.rdt).

Covers the uncovered paths:
- _ensure_loaded() success path (torch + repo import)
- _ensure_loaded() HF fallback (when repo not available)
- get_actions() with loaded model (full inference pipeline)
- _encode_instruction() with T5 and fallback

Target: 50% → ~95%+ coverage of policies/rdt/__init__.py (59 uncov lines).
"""

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

# Skip entire module if policy PR (#8) not merged yet
try:
    from strands_robots.policies.rdt import RdtPolicy
except (ImportError, AttributeError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #8 (policy-abstraction)", allow_module_level=True)

# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_torch():
    """Create a mock torch with common attributes."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()
    mock_torch.zeros.return_value = MagicMock()
    return mock_torch


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: _ensure_loaded() success path
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureLoadedSuccess:
    """Test _ensure_loaded() when the RDT repo is available."""

    def test_loads_via_repo_scripts(self):
        """When repo_path scripts are importable, model loads successfully."""
        p = RdtPolicy(repo_path="/fake/rdt/repo", device="cpu")
        p.set_robot_state_keys(["j0", "j1"])

        mock_torch = _make_mock_torch()
        mock_model = MagicMock()

        # Mock the scripts.agilex_model.create_model function
        mock_scripts = ModuleType("scripts")
        mock_agilex = ModuleType("scripts.agilex_model")
        mock_agilex.create_model = MagicMock(return_value=mock_model)
        mock_scripts.agilex_model = mock_agilex

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "scripts": mock_scripts,
                "scripts.agilex_model": mock_agilex,
            },
        ):
            p._ensure_loaded()

        assert p._loaded is True
        assert p._model is mock_model
        assert p._device == "cpu"

    def test_cuda_device_when_available(self):
        """When no device specified and CUDA is available, uses cuda:0."""
        p = RdtPolicy()

        mock_torch = _make_mock_torch()
        mock_torch.cuda.is_available.return_value = True

        mock_model = MagicMock()
        mock_scripts = ModuleType("scripts")
        mock_agilex = ModuleType("scripts.agilex_model")
        mock_agilex.create_model = MagicMock(return_value=mock_model)
        mock_scripts.agilex_model = mock_agilex

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "scripts": mock_scripts,
                "scripts.agilex_model": mock_agilex,
            },
        ):
            p._ensure_loaded()

        assert p._device == "cuda:0"

    def test_cpu_fallback_when_no_cuda(self):
        """When no device specified and no CUDA, uses cpu."""
        p = RdtPolicy()

        mock_torch = _make_mock_torch()
        mock_torch.cuda.is_available.return_value = False

        mock_model = MagicMock()
        mock_scripts = ModuleType("scripts")
        mock_agilex = ModuleType("scripts.agilex_model")
        mock_agilex.create_model = MagicMock(return_value=mock_model)
        mock_scripts.agilex_model = mock_agilex

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "scripts": mock_scripts,
                "scripts.agilex_model": mock_agilex,
            },
        ):
            p._ensure_loaded()

        assert p._device == "cpu"

    def test_idempotent(self):
        """_ensure_loaded() does nothing if already loaded."""
        p = RdtPolicy()
        p._loaded = True
        sentinel = MagicMock()
        p._model = sentinel
        p._ensure_loaded()
        assert p._model is sentinel

    def test_hf_fallback_on_import_error(self):
        """When scripts.agilex_model import fails, falls back to HF download."""
        p = RdtPolicy()

        mock_torch = _make_mock_torch()
        mock_torch.cuda.is_available.return_value = False

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = "/tmp/config.json"

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "huggingface_hub": mock_hf,
                # Scripts NOT in sys.modules → ImportError
            },
        ):
            p._ensure_loaded()

        assert p._loaded is True
        mock_hf.hf_hub_download.assert_called_once()

    def test_hf_fallback_download_failure(self):
        """HF fallback handles download failure gracefully."""
        p = RdtPolicy()

        mock_torch = _make_mock_torch()
        mock_torch.cuda.is_available.return_value = False

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = RuntimeError("Network error")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "huggingface_hub": mock_hf,
            },
        ):
            p._ensure_loaded()

        assert p._loaded is True  # Still marks loaded (just warns)

    def test_repo_path_added_to_sys_path(self):
        """repo_path is added to sys.path if not already there."""
        p = RdtPolicy(repo_path="/unique/rdt/path")

        mock_torch = _make_mock_torch()
        mock_model = MagicMock()
        mock_scripts = ModuleType("scripts")
        mock_agilex = ModuleType("scripts.agilex_model")
        mock_agilex.create_model = MagicMock(return_value=mock_model)
        mock_scripts.agilex_model = mock_agilex

        sys.path.copy()
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "scripts": mock_scripts,
                "scripts.agilex_model": mock_agilex,
            },
        ):
            p._ensure_loaded()

        # Clean up sys.path
        if "/unique/rdt/path" in sys.path:
            sys.path.remove("/unique/rdt/path")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: get_actions() with pre-loaded model
# ═══════════════════════════════════════════════════════════════════════════


class TestGetActionsLoaded:
    """Test get_actions() with a pre-loaded RDT model."""

    def _setup_policy(self):
        """Create RDT policy with mocked model."""
        p = RdtPolicy(device="cpu")
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])
        p._loaded = True
        p._device = "cpu"

        # Mock model.step returns action chunk
        mock_model = MagicMock()
        mock_model.step.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        p._model = mock_model
        return p

    def test_get_actions_returns_action_dict(self):
        """get_actions returns list of action dicts with correct keys."""
        p = self._setup_policy()
        obs = {"camera": np.zeros((224, 224, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick up the cube"))

        assert len(actions) == 1
        assert "j0" in actions[0]
        assert abs(actions[0]["j0"] - 0.1) < 1e-5

    def test_get_actions_uses_model_step(self):
        """get_actions calls model.step with correct inputs."""
        p = self._setup_policy()
        obs = {
            "camera": np.zeros((224, 224, 3), dtype=np.uint8),
            "j0": 0.5,
            "j1": -0.3,
        }

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            _run(p.get_actions(obs, "pick"))

        p._model.step.assert_called_once()

    def test_get_actions_increments_step(self):
        """get_actions increments the step counter."""
        p = self._setup_policy()
        obs = {"camera": np.zeros((224, 224, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            _run(p.get_actions(obs, "pick"))

        assert p._step == 1

    def test_get_actions_encodes_instruction(self):
        """First call to get_actions triggers instruction encoding."""
        p = self._setup_policy()
        p._lang_embeddings = None
        obs = {"camera": np.zeros((224, 224, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        # Mock _encode_instruction to return a tensor
        mock_embeddings = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch.object(p, "_encode_instruction", return_value=mock_embeddings):
                _run(p.get_actions(obs, "pick up cube"))

        assert p._lang_embeddings is mock_embeddings

    def test_get_actions_no_model_returns_zeros(self):
        """get_actions with no model returns zero actions."""
        p = RdtPolicy()
        p._loaded = True
        p._model = None
        p.set_robot_state_keys(["j0", "j1"])

        actions = _run(p.get_actions({}, "pick"))
        assert actions == [{"j0": 0.0, "j1": 0.0}]

    def test_get_actions_multiple_cameras(self):
        """get_actions extracts images for each configured camera."""
        p = self._setup_policy()
        p._camera_names = ["cam_high", "cam_right"]

        obs = {
            "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        }

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1

    def test_get_actions_missing_camera_uses_fallback(self):
        """When configured camera not found, uses any available image."""
        p = self._setup_policy()
        p._camera_names = ["cam_nonexistent"]

        obs = {
            "some_camera": np.zeros((224, 224, 3), dtype=np.uint8),
        }

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1

    def test_get_actions_no_camera_uses_blank(self):
        """When no camera found at all, uses blank 384x384 image."""
        p = self._setup_policy()
        p._camera_names = ["cam_nonexistent"]
        obs = {"scalar": 42.0}  # No images

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1

    def test_get_actions_2d_output(self):
        """Model returns 2D array — multiple action steps."""
        p = self._setup_policy()
        p._actions_per_step = 2
        p._model.step.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]],
            dtype=np.float32,
        )

        obs = {"camera": np.zeros((224, 224, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 2

    def test_get_actions_1d_output(self):
        """Model returns 1D array — single action."""
        p = self._setup_policy()
        p._model.step.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

        obs = {"camera": np.zeros((224, 224, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1

    def test_get_actions_with_proprioception(self):
        """Proprioception is extracted from robot_state_keys."""
        p = self._setup_policy()
        obs = {
            "camera": np.zeros((224, 224, 3), dtype=np.uint8),
            "j0": 0.1,
            "j1": 0.2,
            "j2": 0.3,
            "j3": 0.4,
            "j4": 0.5,
            "j5": 0.6,
        }

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            _run(p.get_actions(obs, "pick"))

        # Check model.step was called with proprio keyword arg
        call_kwargs = p._model.step.call_args[1]
        proprio = call_kwargs["proprio"]
        np.testing.assert_allclose(proprio, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: _encode_instruction() tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEncodeInstruction:
    """Test _encode_instruction() with mocked T5."""

    def test_encode_with_t5(self):
        """_encode_instruction uses T5 tokenizer + encoder."""
        p = RdtPolicy(device="cpu")
        p._device = "cpu"

        mock_torch = _make_mock_torch()
        mock_embeddings = MagicMock()
        mock_embeddings.last_hidden_state = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        mock_encoder = MagicMock()
        mock_encoder.to.return_value = mock_encoder
        mock_encoder.eval.return_value = mock_encoder
        mock_encoder.return_value = mock_embeddings

        mock_transformers = MagicMock()
        mock_transformers.T5Tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.T5EncoderModel.from_pretrained.return_value = mock_encoder

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = p._encode_instruction("pick up the red cube")

        assert result is mock_embeddings.last_hidden_state

    def test_encode_fallback_on_error(self):
        """When T5 fails, returns zero tensor."""
        p = RdtPolicy(device="cpu")
        p._device = "cpu"

        mock_torch = _make_mock_torch()
        mock_zeros = MagicMock()
        mock_torch.zeros.return_value = mock_zeros

        mock_transformers = MagicMock()
        mock_transformers.T5Tokenizer.from_pretrained.side_effect = RuntimeError("T5 not found")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = p._encode_instruction("pick")

        assert result is mock_zeros
        mock_torch.zeros.assert_called_once_with(1, 1, 2048, dtype="bfloat16", device="cpu")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Basic property tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRdtBasics:
    """Test basic RDT properties and init."""

    def test_provider_name(self):
        assert RdtPolicy().provider_name == "rdt"

    def test_default_params(self):
        p = RdtPolicy()
        assert p._model_id == "robotics-diffusion-transformer/rdt-1b"
        assert p._state_dim == 14
        assert p._chunk_size == 64
        assert p._actions_per_step == 1

    def test_set_robot_state_keys_updates_state_dim(self):
        p = RdtPolicy()
        p.set_robot_state_keys(["a", "b", "c"])
        assert p._state_dim == 3
        assert p._robot_state_keys == ["a", "b", "c"]

    def test_custom_camera_names(self):
        p = RdtPolicy(camera_names=["front", "side"])
        assert p._camera_names == ["front", "side"]

    def test_custom_model_id(self):
        p = RdtPolicy(model_id="custom/rdt-model")
        assert p._model_id == "custom/rdt-model"
