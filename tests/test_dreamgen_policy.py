#!/usr/bin/env python3
"""Mock-based tests for DreamgenPolicy (strands_robots.policies.dreamgen).

Covers the uncovered paths:
- _load_idm() with mocked torch + gr00t
- _load_vla() with mocked gr00t
- _get_idm_actions() inference pipeline
- _get_vla_actions() inference pipeline
- Error paths for both modes

Target: 75% → ~98% coverage of policies/dreamgen/__init__.py (39 uncov lines).
"""

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if policy PR (#8) not merged yet
try:
    from strands_robots.policies.dreamgen import DreamgenPolicy
except (ImportError, AttributeError):
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


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


def _make_policy(mode="idm", **kwargs):
    """Create a DreamgenPolicy with defaults."""
    defaults = {
        "model_path": "nvidia/test-idm",
        "mode": mode,
        "device": "cpu",
        "action_horizon": 4,
        "action_dim": 6,
    }
    defaults.update(kwargs)
    p = DreamgenPolicy(**defaults)
    p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])
    return p


def _make_mock_torch():
    """Create a mock torch module with inference_mode context manager."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.inference_mode.return_value.__enter__ = MagicMock()
    mock_torch.inference_mode.return_value.__exit__ = MagicMock()
    return mock_torch


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: _load_idm() tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadIdm:
    """Test _load_idm() with mocked torch and transformers."""

    def test_load_idm_success(self):
        """IDM loads successfully with mocked AutoModel."""
        p = _make_policy(mode="idm")
        mock_torch = _make_mock_torch()
        mock_model = MagicMock()
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoModel = mock_automodel
        mock_transformers.AutoConfig = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            p._load_idm()

        assert p._model is mock_model
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")

    def test_load_idm_skips_if_already_loaded(self):
        """_load_idm() is idempotent — does nothing if model already loaded."""
        p = _make_policy(mode="idm")
        sentinel = MagicMock()
        p._model = sentinel
        p._load_idm()
        assert p._model is sentinel  # Not replaced

    def test_load_idm_with_groot_available(self):
        """When gr00t.model.idm is importable, it's loaded before AutoModel."""
        p = _make_policy(mode="idm")
        mock_torch = _make_mock_torch()
        mock_model = MagicMock()
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoModel = mock_automodel
        mock_transformers.AutoConfig = MagicMock()

        # Mock gr00t.model.idm
        mock_gr00t = ModuleType("gr00t")
        mock_gr00t_model = ModuleType("gr00t.model")
        mock_gr00t_model_idm = MagicMock()
        mock_gr00t.model = mock_gr00t_model
        mock_gr00t_model.idm = mock_gr00t_model_idm

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "gr00t": mock_gr00t,
                "gr00t.model": mock_gr00t_model,
                "gr00t.model.idm": mock_gr00t_model_idm,
            },
        ):
            p._load_idm()

        assert p._model is mock_model
        mock_model.eval.assert_called_once()

    def test_load_idm_without_groot(self):
        """When gr00t is not available, falls back to AutoModel (still succeeds)."""
        p = _make_policy(mode="idm")
        mock_torch = _make_mock_torch()
        mock_model = MagicMock()
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoModel = mock_automodel
        mock_transformers.AutoConfig = MagicMock()

        # Explicitly block gr00t
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "gr00t": None,
                "gr00t.model": None,
                "gr00t.model.idm": None,
            },
        ):
            p._load_idm()

        assert p._model is mock_model

    def test_load_idm_failure_raises(self):
        """If AutoModel.from_pretrained fails, ImportError is raised."""
        p = _make_policy(mode="idm")
        mock_torch = _make_mock_torch()
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.side_effect = RuntimeError("model not found")

        mock_transformers = MagicMock()
        mock_transformers.AutoModel = mock_automodel
        mock_transformers.AutoConfig = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            with pytest.raises(ImportError, match="Failed to load DreamGen IDM"):
                p._load_idm()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: _load_vla() tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadVla:
    """Test _load_vla() with mocked GR00T-Dreams."""

    def test_load_vla_success(self):
        """VLA loads successfully with full config."""
        mock_dreams_policy = MagicMock()
        mock_gr00t_policy_mod = MagicMock()
        mock_gr00t_policy_mod.Gr00tPolicy = mock_dreams_policy

        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_config={"video.webcam": {"delta_indices": [0]}},
            modality_transform=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "gr00t": MagicMock(),
                "gr00t.model": MagicMock(),
                "gr00t.model.policy": mock_gr00t_policy_mod,
            },
        ):
            p._load_vla()

        assert p._policy is not None
        mock_dreams_policy.assert_called_once()

    def test_load_vla_skips_if_loaded(self):
        """_load_vla() is idempotent."""
        p = _make_policy(mode="vla")
        sentinel = MagicMock()
        p._policy = sentinel
        p._load_vla()
        assert p._policy is sentinel

    def test_load_vla_no_embodiment_tag(self):
        """VLA raises if embodiment_tag is missing."""
        mock_gr00t_policy_mod = MagicMock()

        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            modality_config={"video.webcam": {}},
            modality_transform=MagicMock(),
        )
        # embodiment_tag is None by default

        with patch.dict(
            sys.modules,
            {
                "gr00t": MagicMock(),
                "gr00t.model": MagicMock(),
                "gr00t.model.policy": mock_gr00t_policy_mod,
            },
        ):
            with pytest.raises(ValueError, match="embodiment_tag required"):
                p._load_vla()

    def test_load_vla_no_modality_config(self):
        """VLA raises if modality_config is missing."""
        mock_gr00t_policy_mod = MagicMock()

        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_transform=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "gr00t": MagicMock(),
                "gr00t.model": MagicMock(),
                "gr00t.model.policy": mock_gr00t_policy_mod,
            },
        ):
            with pytest.raises(ValueError, match="modality_config required"):
                p._load_vla()

    def test_load_vla_no_modality_transform(self):
        """VLA raises if modality_transform is missing."""
        mock_gr00t_policy_mod = MagicMock()

        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_config={"video.webcam": {}},
        )

        with patch.dict(
            sys.modules,
            {
                "gr00t": MagicMock(),
                "gr00t.model": MagicMock(),
                "gr00t.model.policy": mock_gr00t_policy_mod,
            },
        ):
            with pytest.raises(ValueError, match="modality_transform required"):
                p._load_vla()

    def test_load_vla_import_error(self):
        """VLA raises ImportError if gr00t.model.policy not available."""
        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_config={"video.webcam": {}},
            modality_transform=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "gr00t": None,
                "gr00t.model": None,
                "gr00t.model.policy": None,
            },
        ):
            with pytest.raises(ImportError, match="GR00T-Dreams not available"):
                p._load_vla()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: _get_idm_actions() tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetIdmActions:
    """Test the IDM inference pipeline via get_actions()."""

    def _setup_idm_policy(self):
        """Create IDM policy with mocked model."""
        p = _make_policy(mode="idm")
        mock_model = MagicMock()
        # IDM returns action_pred: (B, horizon, action_dim)
        mock_model.get_action.return_value = {
            "action_pred": np.random.randn(1, 4, 6).astype(np.float32),
        }
        p._model = mock_model
        return p

    def test_first_frame_returns_zeros(self):
        """First frame stores the frame and returns zero actions."""
        p = self._setup_idm_policy()
        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == p.action_horizon
        assert all(v == 0.0 for v in actions[0].values())
        assert p._previous_frame is not None

    def test_second_frame_runs_inference(self):
        """Second frame triggers IDM inference with frame pair."""
        p = self._setup_idm_policy()
        # Pre-set previous frame
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        obs = {"camera": np.ones((100, 100, 3), dtype=np.uint8) * 128}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) >= 1
        assert "j0" in actions[0]
        p._model.get_action.assert_called_once()

    def test_idm_updates_previous_frame(self):
        """After inference, _previous_frame is updated to current frame."""
        p = self._setup_idm_policy()
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        new_frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        obs = {"camera": new_frame}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            _run(p.get_actions(obs, "pick"))

        np.testing.assert_array_equal(p._previous_frame, new_frame)

    def test_idm_no_camera_returns_zeros(self):
        """If no camera frame in observation, returns zero actions."""
        p = self._setup_idm_policy()
        obs = {"scalar_key": 42.0}  # No image data

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == p.action_horizon
        assert all(v == 0.0 for v in actions[0].values())

    def test_idm_inference_error_returns_zeros(self):
        """If IDM model raises, returns zero actions gracefully."""
        p = _make_policy(mode="idm")
        mock_model = MagicMock()
        mock_model.get_action.side_effect = RuntimeError("CUDA OOM")
        p._model = mock_model
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        obs = {"camera": np.ones((100, 100, 3), dtype=np.uint8) * 128}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == p.action_horizon
        assert all(v == 0.0 for v in actions[0].values())

    def test_idm_action_pred_2d(self):
        """IDM returns 2D action_pred (horizon, action_dim) — no batch dim."""
        p = _make_policy(mode="idm")
        mock_model = MagicMock()
        # Return 2D (no batch dim) — code handles this via indexing
        action_pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        mock_model.get_action.return_value = {"action_pred": action_pred}
        p._model = mock_model
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        obs = {"camera": np.ones((100, 100, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1
        assert abs(actions[0]["j0"] - 0.1) < 1e-6

    def test_idm_chw_frame_extraction(self):
        """IDM handles CHW format camera frames (C, H, W) → transposes correctly."""
        p = self._setup_idm_policy()
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # CHW format: (3, H, W) — _extract_frame should handle this
        chw_frame = np.zeros((3, 100, 100), dtype=np.uint8)
        obs = {"camera": chw_frame}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        # Should succeed (transposed to HWC internally)
        assert len(actions) >= 1

    def test_idm_action_pred_with_cpu(self):
        """IDM handles action_pred as torch tensor with .cpu() method."""
        p = _make_policy(mode="idm")
        mock_model = MagicMock()
        # Simulate torch tensor with cpu().numpy() chain
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]], dtype=np.float32)
        mock_model.get_action.return_value = {"action_pred": mock_tensor}
        p._model = mock_model
        p._previous_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        obs = {"camera": np.ones((100, 100, 3), dtype=np.uint8)}

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) >= 1
        mock_tensor.cpu.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: _get_vla_actions() tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetVlaActions:
    """Test the VLA inference pipeline."""

    def _setup_vla_policy(self):
        """Create VLA policy with mocked Dreams policy."""
        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_config={"video.webcam": {}},
            modality_transform=MagicMock(),
        )
        mock_dreams = MagicMock()
        mock_dreams.get_action.return_value = {
            "action.joint_pos": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        }
        p._policy = mock_dreams
        return p

    def test_vla_get_actions(self):
        """VLA mode calls _policy.get_action and converts output."""
        p = self._setup_vla_policy()
        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}
        actions = _run(p.get_actions(obs, "pick up the red cube"))

        assert len(actions) >= 1
        assert "j0" in actions[0]
        p._policy.get_action.assert_called_once()

    def test_vla_builds_dreams_observation(self):
        """VLA passes correct observation format to Dreams policy."""
        p = self._setup_vla_policy()
        obs = {
            "camera": np.zeros((100, 100, 3), dtype=np.uint8),
            "j0": 0.5,
            "j1": -0.3,
        }
        _run(p.get_actions(obs, "pick"))

        call_args = p._policy.get_action.call_args[0][0]
        # Camera should be mapped to video.camera
        assert "video.camera" in call_args
        # Language instruction should be present
        assert "annotation.human.task_description" in call_args
        assert call_args["annotation.human.task_description"] == "pick"

    def test_vla_state_keys_mapped(self):
        """VLA maps robot state keys to state.* format."""
        p = self._setup_vla_policy()
        obs = {
            "j0": 1.0,
            "j1": 2.0,
        }
        _run(p.get_actions(obs, "move"))

        call_args = p._policy.get_action.call_args[0][0]
        assert "state.j0" in call_args
        assert "state.j1" in call_args

    def test_vla_inference_error_returns_zeros(self):
        """VLA returns zero actions if Dreams policy raises."""
        p = _make_policy(
            mode="vla",
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="test_embodiment",
            modality_config={"video.webcam": {}},
            modality_transform=MagicMock(),
        )
        mock_dreams = MagicMock()
        mock_dreams.get_action.side_effect = RuntimeError("CUDA OOM")
        p._policy = mock_dreams

        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}
        actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == p.action_horizon
        assert all(v == 0.0 for v in actions[0].values())

    def test_vla_empty_action_dict_returns_zeros(self):
        """VLA returns zeros if Dreams policy returns empty action dict."""
        p = self._setup_vla_policy()
        p._policy.get_action.return_value = {}  # No action.* keys

        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}
        actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == p.action_horizon
        assert all(v == 0.0 for v in actions[0].values())

    def test_vla_multiple_action_modalities(self):
        """VLA concatenates multiple action.* keys."""
        p = self._setup_vla_policy()
        p._policy.get_action.return_value = {
            "action.joint_pos": np.array([[0.1, 0.2, 0.3]]),
            "action.gripper": np.array([[0.4, 0.5, 0.6]]),
        }

        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}
        actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) >= 1
        # Actions should have 6 values (3 + 3 concatenated)
        assert len(actions[0]) == 6


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Utility method tests
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilityMethods:
    """Test utility methods of DreamgenPolicy."""

    def test_extract_frame_hwc(self):
        """_extract_frame extracts HWC format correctly."""
        p = _make_policy()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        obs = {"camera": frame}
        result = p._extract_frame(obs)
        assert result is not None
        assert result.shape == (100, 100, 3)
        np.testing.assert_array_equal(result, frame)

    def test_extract_frame_chw(self):
        """_extract_frame handles CHW format (C, H, W) → transposes to HWC."""
        p = _make_policy()
        chw = np.ones((3, 100, 100), dtype=np.uint8) * 42
        obs = {"camera": chw}
        result = p._extract_frame(obs)
        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_extract_frame_skips_state_keys(self):
        """_extract_frame skips keys in robot_state_keys."""
        p = _make_policy()
        obs = {
            "j0": np.zeros((100, 100, 3), dtype=np.uint8),  # matches state key
            "camera": np.ones((80, 80, 3), dtype=np.uint8),
        }
        result = p._extract_frame(obs)
        assert result is not None
        assert result.shape == (80, 80, 3)

    def test_extract_frame_no_image(self):
        """_extract_frame returns None if no suitable image found."""
        p = _make_policy()
        obs = {"scalar": 42.0}
        result = p._extract_frame(obs)
        assert result is None

    def test_build_dreams_observation(self):
        """_build_dreams_observation maps keys correctly."""
        p = _make_policy()
        obs = {
            "camera": np.zeros((100, 100, 3), dtype=np.uint8),
            "j0": 0.5,
            "j1": -0.3,
        }
        result = p._build_dreams_observation(obs, "pick up cube")
        assert "video.camera" in result
        assert "state.j0" in result
        assert "annotation.human.task_description" in result
        assert result["annotation.human.task_description"] == "pick up cube"

    def test_build_dreams_observation_preserves_video_prefix(self):
        """Keys already starting with video.* keep their prefix."""
        p = _make_policy()
        obs = {"video.webcam": np.zeros((100, 100, 3), dtype=np.uint8)}
        result = p._build_dreams_observation(obs, "test")
        assert "video.webcam" in result

    def test_convert_dreams_actions(self):
        """_convert_dreams_actions converts action dict to robot action list."""
        p = _make_policy()
        action_dict = {
            "action.joint_pos": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        }
        result = p._convert_dreams_actions(action_dict)
        assert len(result) == 1
        assert abs(result[0]["j0"] - 0.1) < 1e-6
        assert abs(result[0]["j5"] - 0.6) < 1e-6

    def test_convert_dreams_actions_1d(self):
        """_convert_dreams_actions handles 1D action arrays."""
        p = _make_policy()
        action_dict = {
            "action.joint_pos": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        }
        result = p._convert_dreams_actions(action_dict)
        assert len(result) == 1

    def test_convert_dreams_actions_empty(self):
        """_convert_dreams_actions returns zeros for empty dict."""
        p = _make_policy()
        result = p._convert_dreams_actions({})
        assert len(result) == p.action_horizon
        assert all(v == 0.0 for v in result[0].values())

    def test_actions_to_dicts(self):
        """_actions_to_dicts converts ndarray to list of dicts."""
        p = _make_policy()
        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        result = p._actions_to_dicts(actions)
        assert len(result) == 1
        assert "j0" in result[0]
        assert abs(result[0]["j0"] - 0.1) < 1e-6

    def test_actions_to_dicts_short_action(self):
        """_actions_to_dicts pads with 0.0 if action_dim < state_keys."""
        p = _make_policy()
        actions = np.array([[0.1, 0.2]], dtype=np.float32)
        result = p._actions_to_dicts(actions)
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.1)
        assert result[0]["j2"] == 0.0  # Padded

    def test_generate_zero_actions(self):
        """_generate_zero_actions returns list with one zero-action dict."""
        p = _make_policy()
        result = p._generate_zero_actions()
        assert len(result) == p.action_horizon
        assert all(v == 0.0 for v in result[0].values())
        assert set(result[0].keys()) == {"j0", "j1", "j2", "j3", "j4", "j5"}

    def test_unknown_mode_raises(self):
        """get_actions with unknown mode raises ValueError."""
        p = _make_policy(mode="invalid_mode")
        obs = {"camera": np.zeros((100, 100, 3), dtype=np.uint8)}
        with pytest.raises(ValueError, match="Unknown mode"):
            _run(p.get_actions(obs, "pick"))

    def test_reset(self):
        """reset() clears previous frame cache."""
        p = _make_policy()
        p._previous_frame = np.zeros((100, 100, 3))
        p._step = 5
        p.reset()
        assert p._previous_frame is None

    def test_provider_name(self):
        """provider_name returns 'dreamgen'."""
        p = _make_policy()
        assert p.provider_name == "dreamgen"

    def test_set_robot_state_keys(self):
        """set_robot_state_keys updates the keys list."""
        p = DreamgenPolicy(model_path="test", device="cpu")
        p.set_robot_state_keys(["a", "b", "c"])
        assert p.robot_state_keys == ["a", "b", "c"]
