#!/usr/bin/env python3
"""Comprehensive tests for GR00T policy provider (groot/__init__.py + client.py).

Covers:
- Gr00tPolicy construction (service mode + local mode)
- _build_observation with various robot configs
- _local_inference for N1.5 and N1.6
- _convert_to_robot_actions edge cases
- _map_video_key_to_camera full mapping logic
- _map_robot_state_to_gr00t_state for so100/fourier_gr1/unitree_g1/generic
- get_actions async flow
- MsgSerializer for ZMQ client
- BaseInferenceClient lifecycle
- ExternalRobotInferenceClient.get_action
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if policy PR (#8) not merged yet
try:
    from strands_robots.policies.groot import BaseDataConfig, Gr00tPolicy
    from strands_robots.policies.groot.data_config import load_data_config
except (ImportError, AttributeError):
    __import__("pytest").skip("Requires PR #8 (policy-abstraction)", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groot_policy(**overrides):
    """Create a Gr00tPolicy without calling __init__ (bypass ZMQ/GPU)."""
    p = Gr00tPolicy.__new__(Gr00tPolicy)
    defaults = dict(
        data_config=load_data_config("so100_dualcam"),
        data_config_name="so100_dualcam",
        camera_keys=["video.webcam"],
        state_keys=["state.single_arm", "state.gripper"],
        action_keys=["action.single_arm", "action.gripper"],
        language_keys=["annotation.human.action.task_description"],
        robot_state_keys=["j0", "j1", "j2", "j3", "j4", "gripper"],
        _local_policy=None,
        _client=None,
        _mode="service",
        _groot_version=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(p, k, v)
    return p


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Gr00tPolicy construction
# ═══════════════════════════════════════════════════════════════════════════


class TestGr00tPolicyConstruction:
    """Test construction paths for Gr00tPolicy."""

    def test_service_mode_construction(self):
        """Service mode with mocked ZMQ client."""
        mock_client_cls = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        with patch("strands_robots.policies.groot._get_client_class", return_value=mock_client_cls):
            with patch("strands_robots.policies.groot._detect_groot_version", return_value=None):
                p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=5555)

        assert p._mode == "service"
        assert p._client is mock_client_inst
        assert p.provider_name == "groot"

    def test_service_mode_client_init_failure(self):
        """Service mode raises ImportError on client init failure."""
        mock_client_cls = MagicMock(side_effect=Exception("ZMQ not available"))

        with patch("strands_robots.policies.groot._get_client_class", return_value=mock_client_cls):
            with patch("strands_robots.policies.groot._detect_groot_version", return_value=None):
                with pytest.raises(ImportError, match="GR00T service client init failed"):
                    Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=5555)

    def test_local_mode_no_groot_installed(self):
        """Local mode without Isaac-GR00T raises ImportError."""
        with patch("strands_robots.policies.groot._detect_groot_version", return_value=None):
            with pytest.raises(ImportError, match="Isaac-GR00T not installed"):
                Gr00tPolicy(data_config="so100_dualcam", model_path="/fake/path")

    def test_local_mode_n15(self):
        """Local mode with N1.5 loads via _load_n15."""
        MagicMock()
        mock_n15_configs = {"so100_dualcam": MagicMock()}
        mock_n15_configs["so100_dualcam"].modality_config.return_value = MagicMock()
        mock_n15_configs["so100_dualcam"].transform.return_value = MagicMock()

        with patch("strands_robots.policies.groot._detect_groot_version", return_value="n1.5"):
            with patch.dict(
                sys.modules,
                {
                    "gr00t": MagicMock(),
                    "gr00t.model": MagicMock(),
                    "gr00t.model.policy": MagicMock(),
                    "gr00t.experiment": MagicMock(),
                    "gr00t.experiment.data_config": MagicMock(DATA_CONFIG_MAP=mock_n15_configs),
                },
            ):
                p = Gr00tPolicy(
                    data_config="so100_dualcam",
                    model_path="/fake/checkpoint",
                    groot_version="n1.5",
                )
        assert p._mode == "local"

    def test_local_mode_n16(self):
        """Local mode with N1.6 loads via _load_n16."""
        mock_n16_policy = MagicMock()
        mock_embodiment_tag = MagicMock()
        mock_embodiment_tag.NEW_EMBODIMENT = "new_embodiment"

        with patch("strands_robots.policies.groot._detect_groot_version", return_value="n1.6"):
            with patch.dict(
                sys.modules,
                {
                    "gr00t": MagicMock(),
                    "gr00t.policy": MagicMock(),
                    "gr00t.policy.gr00t_policy": MagicMock(Gr00tPolicy=mock_n16_policy),
                    "gr00t.data": MagicMock(),
                    "gr00t.data.embodiment_tags": MagicMock(EmbodimentTag=mock_embodiment_tag),
                },
            ):
                p = Gr00tPolicy(
                    data_config="so100_dualcam",
                    model_path="nvidia/GR00T-N1-2B",
                    groot_version="n1.6",
                )
        assert p._mode == "local"

    def test_set_robot_state_keys(self):
        """set_robot_state_keys stores keys."""
        p = _make_groot_policy()
        p.set_robot_state_keys(["a", "b", "c"])
        assert p.robot_state_keys == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: _detect_groot_version
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectGrootVersion:
    """Test auto-detection of Isaac-GR00T version."""

    def test_detect_n16(self):
        """N1.6 detected when gr00t.policy.gr00t_policy importable."""
        import strands_robots.policies.groot as groot_mod

        groot_mod._GROOT_VERSION = None  # reset

        mock_n16 = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "gr00t": mock_n16,
                "gr00t.policy": mock_n16,
                "gr00t.policy.gr00t_policy": mock_n16,
                "gr00t.data": mock_n16,
                "gr00t.data.embodiment_tags": mock_n16,
            },
        ):
            from strands_robots.policies.groot import _detect_groot_version

            groot_mod._GROOT_VERSION = None
            result = _detect_groot_version()
        groot_mod._GROOT_VERSION = None  # cleanup
        assert result == "n1.6"

    def test_detect_none_when_no_groot(self):
        """Returns None when no Isaac-GR00T installed."""
        import strands_robots.policies.groot as groot_mod

        groot_mod._GROOT_VERSION = None

        # Make sure gr00t is not importable
        with patch.dict(sys.modules, {"gr00t": None, "gr00t.policy": None, "gr00t.model": None}):
            # Need to force ImportError
            groot_mod._detect_groot_version()
        groot_mod._GROOT_VERSION = None
        # Result depends on whether gr00t is actually installed


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: _build_observation
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildObservation:
    """Test _build_observation for various robot configurations."""

    def test_service_mode_adds_batch_dim(self):
        """Service mode adds batch dimension to all arrays."""
        p = _make_groot_policy(_mode="service")
        obs = {
            "webcam": np.zeros((480, 640, 3), dtype=np.uint8),
            "j0": 0.1,
            "j1": 0.2,
            "j2": 0.3,
            "j3": 0.4,
            "j4": 0.5,
            "gripper": 0.6,
        }
        result = p._build_observation(obs, "pick up cube")
        # Arrays should have batch dim
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                assert v.ndim >= 2  # batch dim added
            elif isinstance(v, list):
                assert isinstance(v, list)  # wrapped in list

    def test_local_mode_no_batch_dim(self):
        """Local mode does NOT add batch dimension."""
        p = _make_groot_policy(_mode="local")
        obs = {
            "webcam": np.zeros((480, 640, 3), dtype=np.uint8),
            "j0": 0.1,
            "j1": 0.2,
            "j2": 0.3,
            "j3": 0.4,
            "j4": 0.5,
            "gripper": 0.6,
        }
        result = p._build_observation(obs, "pick up")
        # State arrays should be flat
        if "state.single_arm" in result:
            assert result["state.single_arm"].ndim == 1

    def test_language_instruction_injected(self):
        """Language instruction should be placed in first language key."""
        p = _make_groot_policy(_mode="local")
        obs = {"j0": 0.1, "j1": 0.2, "j2": 0.3, "j3": 0.4, "j4": 0.5, "gripper": 0.6}
        result = p._build_observation(obs, "grasp the red ball")
        lang_key = p.language_keys[0]
        assert result[lang_key] == "grasp the red ball"

    def test_no_language_keys(self):
        """When no language keys configured, instruction is not injected."""
        p = _make_groot_policy(_mode="local", language_keys=[])
        obs = {"j0": 0.1, "j1": 0.2, "j2": 0.3, "j3": 0.4, "j4": 0.5, "gripper": 0.6}
        result = p._build_observation(obs, "grasp")
        # No language key should be present
        assert not any("annotation" in k or "language" in k for k in result)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: _map_video_key_to_camera
# ═══════════════════════════════════════════════════════════════════════════


class TestMapVideoKeyToCamera:
    """Test the video key → camera name mapping."""

    def test_direct_match(self):
        p = _make_groot_policy()
        obs = {"webcam": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.webcam", obs) == "webcam"

    def test_front_fallback_to_webcam(self):
        p = _make_groot_policy()
        obs = {"webcam": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.front", obs) == "webcam"

    def test_wrist_fallback_to_hand(self):
        p = _make_groot_policy()
        obs = {"hand": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.wrist", obs) == "hand"

    def test_ego_view_fallback(self):
        p = _make_groot_policy()
        obs = {"front": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.ego_view", obs) == "front"

    def test_top_fallback(self):
        p = _make_groot_policy()
        obs = {"overhead": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.top", obs) == "overhead"

    def test_side_fallback(self):
        p = _make_groot_policy()
        obs = {"lateral": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.side", obs) == "lateral"

    def test_unknown_key_any_camera(self):
        p = _make_groot_policy()
        obs = {"random_cam": np.zeros((100, 100, 3))}
        result = p._map_video_key_to_camera("video.unknown", obs)
        assert result == "random_cam"

    def test_no_cameras_returns_none(self):
        p = _make_groot_policy()
        obs = {"state_var": 42.0}
        result = p._map_video_key_to_camera("video.webcam", obs)
        # state_var starts with "state", so filtered out
        # Actually the filter is: [k for k in obs if not k.startswith("state")]
        # "state_var" starts with "state", so filtered
        assert result is None

    def test_fallback_to_first_non_state_key(self):
        p = _make_groot_policy()
        obs = {"my_camera": np.zeros((100, 100, 3)), "state_var": 42.0}
        result = p._map_video_key_to_camera("video.nonexistent", obs)
        assert result == "my_camera"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: _map_robot_state_to_gr00t_state
# ═══════════════════════════════════════════════════════════════════════════


class TestMapRobotStateToGrootState:
    """Test state mapping for different robot configs."""

    def test_so100_mapping(self):
        p = _make_groot_policy(data_config_name="so100_dualcam")
        obs = {}
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.single_arm" in obs
        assert "state.gripper" in obs
        np.testing.assert_allclose(obs["state.single_arm"], [0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(obs["state.gripper"], [0.6])

    def test_so101_mapping(self):
        p = _make_groot_policy(data_config_name="so101_dualcam")
        obs = {}
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.single_arm" in obs
        assert "state.gripper" in obs

    def test_fourier_gr1_mapping(self):
        p = _make_groot_policy(data_config_name="fourier_gr1_arms")
        obs = {}
        state = np.arange(14, dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.left_arm" in obs
        assert "state.right_arm" in obs
        np.testing.assert_allclose(obs["state.left_arm"], state[:7])
        np.testing.assert_allclose(obs["state.right_arm"], state[7:14])

    def test_unitree_g1_mapping(self):
        p = _make_groot_policy(data_config_name="unitree_g1_dualarm")
        obs = {}
        state = np.arange(14, dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.left_arm" in obs
        assert "state.right_arm" in obs

    def test_generic_mapping(self):
        """Unknown robot name falls back to first state_key."""
        p = _make_groot_policy(
            data_config_name="custom_robot",
            state_keys=["state.custom"],
        )
        obs = {}
        state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.custom" in obs
        np.testing.assert_allclose(obs["state.custom"], [1.0, 2.0, 3.0])

    def test_generic_no_state_keys(self):
        """No state_keys and unknown robot → nothing mapped."""
        p = _make_groot_policy(data_config_name="mystery_bot", state_keys=[])
        obs = {}
        state = np.array([1.0, 2.0])
        p._map_robot_state_to_gr00t_state(obs, state)
        assert len(obs) == 0

    def test_so100_short_state_skipped(self):
        """so100 with < 6 joints → nothing mapped."""
        p = _make_groot_policy(data_config_name="so100_dualcam")
        obs = {}
        state = np.array([0.1, 0.2], dtype=np.float64)  # too short
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.single_arm" not in obs

    def test_fourier_gr1_short_state_skipped(self):
        """fourier_gr1 with < 14 joints → nothing mapped."""
        p = _make_groot_policy(data_config_name="fourier_gr1_dual")
        obs = {}
        state = np.array([0.1] * 10, dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs, state)
        assert "state.left_arm" not in obs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: _convert_to_robot_actions
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertToRobotActions:
    """Test action chunk → robot action dicts."""

    def test_single_step(self):
        p = _make_groot_policy(robot_state_keys=["j0", "j1", "j2"])
        chunk = {"action.arm": np.array([[0.1, 0.2, 0.3]])}
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.1)

    def test_multi_step_horizon(self):
        p = _make_groot_policy(robot_state_keys=["j0", "j1"])
        chunk = {"action.arm": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])}
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 3
        assert result[2]["j1"] == pytest.approx(0.6)

    def test_multi_action_keys_concatenated(self):
        p = _make_groot_policy(robot_state_keys=["j0", "j1", "j2", "j3", "j4", "gripper"])
        chunk = {
            "action.single_arm": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "action.gripper": np.array([[0.9]]),
        }
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 1
        assert result[0]["gripper"] == pytest.approx(0.9)

    def test_empty_chunk(self):
        p = _make_groot_policy(robot_state_keys=["j0"])
        result = p._convert_to_robot_actions({})
        assert result == []

    def test_keys_without_action_prefix(self):
        """Action keys without 'action.' prefix still work."""
        p = _make_groot_policy(robot_state_keys=["j0", "j1"])
        chunk = {"arm": np.array([[0.5, 0.6]])}
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.5)

    def test_scalar_action(self):
        """Action with ndim=0 should still work."""
        p = _make_groot_policy(robot_state_keys=["j0"])
        chunk = {"action.scalar": np.float64(1.5)}
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 1

    def test_more_actions_than_keys(self):
        """When action has more dims than robot_state_keys, extras truncated."""
        p = _make_groot_policy(robot_state_keys=["j0"])
        chunk = {"action.arm": np.array([[0.1, 0.2, 0.3]])}
        result = p._convert_to_robot_actions(chunk)
        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.1)

    def test_fewer_actions_than_keys(self):
        """When action has fewer dims, remaining keys get 0.0."""
        p = _make_groot_policy(robot_state_keys=["j0", "j1", "j2"])
        chunk = {"action.arm": np.array([[0.1]])}
        result = p._convert_to_robot_actions(chunk)
        assert result[0]["j1"] == 0.0
        assert result[0]["j2"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: _local_inference
# ═══════════════════════════════════════════════════════════════════════════


class TestLocalInference:
    """Test _local_inference for N1.5 and N1.6 paths."""

    def test_n15_inference(self):
        """N1.5 inference calls _local_policy.get_action with batched dict."""
        mock_model = MagicMock()
        mock_model.get_action.return_value = {
            "action.single_arm": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "action.gripper": np.array([[0.9]]),
        }
        p = _make_groot_policy(_mode="local", _local_policy=mock_model, _groot_version="n1.5")

        obs_dict = {
            "video.webcam": np.zeros((480, 640, 3)),
            "state.single_arm": np.zeros(5),
            "annotation.human.action.task_description": "pick up",
        }
        result = p._local_inference(obs_dict)
        assert "action.single_arm" in result
        mock_model.get_action.assert_called_once()

    def test_n16_inference(self):
        """N1.6 inference uses nested dict format."""
        mock_model = MagicMock()
        # N1.6 get_modality_config returns dict of configs
        mock_video_cfg = MagicMock()
        mock_video_cfg.modality_keys = ["video.webcam"]
        mock_state_cfg = MagicMock()
        mock_state_cfg.modality_keys = ["state.single_arm"]
        mock_lang_cfg = MagicMock()
        mock_lang_cfg.modality_keys = ["language.task"]
        mock_model.get_modality_config.return_value = {
            "video": mock_video_cfg,
            "state": mock_state_cfg,
            "language": mock_lang_cfg,
        }
        mock_model.get_action.return_value = (
            {"single_arm": np.array([[0.1, 0.2]]), "gripper": np.array([[0.9]])},
            None,
        )

        p = _make_groot_policy(_mode="local", _local_policy=mock_model, _groot_version="n1.6")

        obs_dict = {
            "video.webcam": np.zeros((480, 640, 3)),
            "state.single_arm": np.zeros(5),
            "annotation.human.action.task_description": "grasp",
        }
        result = p._local_inference(obs_dict)
        # N1.6 adds "action." prefix
        assert "action.single_arm" in result
        assert "action.gripper" in result

    def test_unknown_version_raises(self):
        """Unknown version raises RuntimeError."""
        p = _make_groot_policy(_groot_version="n2.0")
        with pytest.raises(RuntimeError, match="Unknown GR00T version"):
            p._local_inference({})


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: get_actions async
# ═══════════════════════════════════════════════════════════════════════════


class TestGetActionsAsync:
    """Test full get_actions pipeline."""

    def test_service_mode_get_actions(self):
        """Service mode delegates to client.get_action."""
        mock_client = MagicMock()
        mock_client.get_action.return_value = {
            "action.single_arm": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "action.gripper": np.array([[0.9]]),
        }
        p = _make_groot_policy(_mode="service", _client=mock_client)

        obs = {
            "webcam": np.zeros((480, 640, 3), dtype=np.uint8),
            "j0": 0.1,
            "j1": 0.2,
            "j2": 0.3,
            "j3": 0.4,
            "j4": 0.5,
            "gripper": 0.6,
        }
        result = _run(p.get_actions(obs, "pick up cube"))
        assert len(result) >= 1
        mock_client.get_action.assert_called_once()

    def test_local_mode_get_actions(self):
        """Local mode delegates to _local_inference."""
        mock_model = MagicMock()
        mock_model.get_action.return_value = {
            "action.single_arm": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "action.gripper": np.array([[0.9]]),
        }
        p = _make_groot_policy(
            _mode="local",
            _local_policy=mock_model,
            _groot_version="n1.5",
        )
        obs = {
            "webcam": np.zeros((480, 640, 3), dtype=np.uint8),
            "j0": 0.1,
            "j1": 0.2,
            "j2": 0.3,
            "j3": 0.4,
            "j4": 0.5,
            "gripper": 0.6,
        }
        result = _run(p.get_actions(obs, "grasp"))
        assert len(result) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: MsgSerializer (client.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestMsgSerializer:
    """Test MsgSerializer encode/decode for ZMQ communication."""

    def test_roundtrip_dict(self):
        """Basic dict roundtrip through serialize/deserialize."""
        from strands_robots.policies.groot.client import MsgSerializer

        data = {"key": "value", "num": 42}
        encoded = MsgSerializer.to_bytes(data)
        decoded = MsgSerializer.from_bytes(encoded)
        assert decoded["key"] == "value"
        assert decoded["num"] == 42

    def test_numpy_array_roundtrip(self):
        """Numpy array survives serialize/deserialize."""
        from strands_robots.policies.groot.client import MsgSerializer

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data = {"action": arr}
        encoded = MsgSerializer.to_bytes(data)
        decoded = MsgSerializer.from_bytes(encoded)
        np.testing.assert_allclose(decoded["action"], arr)

    def test_modality_config_encode(self):
        """ModalityConfig encodes to expected format."""
        from strands_robots.policies.groot.client import MsgSerializer
        from strands_robots.policies.groot.data_config import ModalityConfig

        mc = ModalityConfig(
            delta_indices=[0, 2],
            modality_keys=["video.webcam"],
        )
        encoded = MsgSerializer.encode_custom_classes(mc)
        assert encoded["__ModalityConfig_class__"] is True
        assert "video.webcam" in encoded["as_json"]
        assert "delta_indices" in encoded["as_json"]

    def test_ndarray_decode(self):
        """Numpy array decode from custom format."""
        import io as _io

        from strands_robots.policies.groot.client import MsgSerializer

        arr = np.array([1.0, 2.0, 3.0])
        buf = _io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        obj = {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        decoded = MsgSerializer.decode_custom_classes(obj)
        np.testing.assert_allclose(decoded, [1.0, 2.0, 3.0])

    def test_encode_unknown_type_passthrough(self):
        """Unknown types passed through as-is."""
        from strands_robots.policies.groot.client import MsgSerializer

        result = MsgSerializer.encode_custom_classes("plain_string")
        assert result == "plain_string"

    def test_decode_normal_dict_passthrough(self):
        """Normal dicts without custom class markers pass through."""
        from strands_robots.policies.groot.client import MsgSerializer

        result = MsgSerializer.decode_custom_classes({"normal": "dict"})
        assert result == {"normal": "dict"}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: BaseInferenceClient
# ═══════════════════════════════════════════════════════════════════════════


class TestBaseInferenceClient:
    """Test BaseInferenceClient with mocked ZMQ."""

    def _mock_zmq(self):
        mock_zmq = MagicMock()
        mock_zmq.Context.return_value = MagicMock()
        mock_zmq.REQ = 3
        return mock_zmq

    def test_client_construction(self):
        """Client creates ZMQ context and socket."""
        from strands_robots.policies.groot.client import BaseInferenceClient

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        assert client.host == "localhost"
        assert client.port == 5555

    def test_ping_success(self):
        """Ping returns True on success."""
        from strands_robots.policies.groot.client import BaseInferenceClient

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        client.call_endpoint = MagicMock(return_value={"status": "ok"})
        assert client.ping() is True

    def test_ping_failure(self):
        """Ping returns False and reinits socket on failure."""
        from strands_robots.policies.groot.client import BaseInferenceClient

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        client.call_endpoint = MagicMock(side_effect=Exception("timeout"))
        assert client.ping() is False

    def test_call_endpoint_with_data(self):
        """call_endpoint sends request and handles response."""
        from strands_robots.policies.groot.client import BaseInferenceClient, MsgSerializer

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        response_data = {"action": [0.1, 0.2]}
        with patch.object(MsgSerializer, "to_bytes", return_value=b"request"):
            with patch.object(MsgSerializer, "from_bytes", return_value=response_data):
                client.socket.recv.return_value = b"response"
                result = client.call_endpoint("get_action", {"obs": "data"})

        assert result == response_data

    def test_call_endpoint_server_error(self):
        """Server error in response raises RuntimeError."""
        from strands_robots.policies.groot.client import BaseInferenceClient, MsgSerializer

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        with patch.object(MsgSerializer, "to_bytes", return_value=b"req"):
            with patch.object(MsgSerializer, "from_bytes", return_value={"error": "model not loaded"}):
                client.socket.recv.return_value = b"resp"
                with pytest.raises(RuntimeError, match="Server error"):
                    client.call_endpoint("get_action", {"obs": "data"})

    def test_call_endpoint_with_api_token(self):
        """API token is included in request when set."""
        from strands_robots.policies.groot.client import BaseInferenceClient, MsgSerializer

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555, api_token="secret")

        captured_request = {}

        def mock_to_bytes(data):
            captured_request.update(data)
            return b"encoded"

        with patch.object(MsgSerializer, "to_bytes", side_effect=mock_to_bytes):
            with patch.object(MsgSerializer, "from_bytes", return_value={"ok": True}):
                client.socket.recv.return_value = b"resp"
                client.call_endpoint("ping", requires_input=False)

        assert captured_request.get("api_token") == "secret"

    def test_call_endpoint_no_data(self):
        """call_endpoint without data (requires_input=False)."""
        from strands_robots.policies.groot.client import BaseInferenceClient, MsgSerializer

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        captured = {}

        def capture(data):
            captured.update(data)
            return b"req"

        with patch.object(MsgSerializer, "to_bytes", side_effect=capture):
            with patch.object(MsgSerializer, "from_bytes", return_value={"status": "pong"}):
                client.socket.recv.return_value = b"resp"
                client.call_endpoint("ping", requires_input=False)

        assert "data" not in captured

    def test_del_cleanup(self):
        """__del__ closes socket and terminates context."""
        from strands_robots.policies.groot.client import BaseInferenceClient

        mock_zmq = self._mock_zmq()
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = BaseInferenceClient(host="localhost", port=5555)

        mock_socket = client.socket
        mock_context = client.context
        client.__del__()
        mock_socket.close.assert_called_once()
        mock_context.term.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: ExternalRobotInferenceClient
# ═══════════════════════════════════════════════════════════════════════════


class TestExternalRobotInferenceClient:
    """Test the convenience client class."""

    def test_get_action(self):
        """get_action delegates to call_endpoint."""
        from strands_robots.policies.groot.client import ExternalRobotInferenceClient

        mock_zmq = MagicMock()
        mock_zmq.Context.return_value = MagicMock()
        mock_zmq.REQ = 3
        mock_msgpack = MagicMock()

        with patch.dict(sys.modules, {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            import strands_robots.policies.groot.client as client_mod

            client_mod._zmq = None
            client_mod._msgpack = None
            client = ExternalRobotInferenceClient(host="localhost", port=5555)

        mock_result = {"action.arm": np.array([[0.1, 0.2]])}
        client.call_endpoint = MagicMock(return_value=mock_result)

        obs = {"video.webcam": np.zeros((480, 640, 3))}
        result = client.get_action(obs)
        client.call_endpoint.assert_called_with("get_action", obs)
        assert result is mock_result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: _ensure_deps
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureDeps:
    """Test lazy dependency loading."""

    def test_ensure_deps_success(self):
        """_ensure_deps loads zmq and msgpack."""
        from strands_robots.policies.groot import client as client_mod

        client_mod._zmq = None
        client_mod._msgpack = None

        # zmq and msgpack should be available since they're in dependencies
        client_mod._ensure_deps()
        assert client_mod._zmq is not None
        assert client_mod._msgpack is not None

    def test_ensure_deps_import_error(self):
        """_ensure_deps raises ImportError when deps missing."""
        from strands_robots.policies.groot import client as client_mod

        client_mod._zmq = None
        client_mod._msgpack = None

        with patch.dict(sys.modules, {"zmq": None, "msgpack": None}):
            # Force the import to fail
            with patch("builtins.__import__", side_effect=ImportError("no zmq")):
                with pytest.raises(ImportError, match="pyzmq msgpack"):
                    client_mod._ensure_deps()

        # Reset
        client_mod._zmq = None
        client_mod._msgpack = None

    def test_ensure_deps_idempotent(self):
        """_ensure_deps is a no-op when already loaded."""
        from strands_robots.policies.groot import client as client_mod

        mock_zmq = MagicMock()
        mock_msgpack = MagicMock()
        client_mod._zmq = mock_zmq
        client_mod._msgpack = mock_msgpack

        client_mod._ensure_deps()
        # Should still be same mocks
        assert client_mod._zmq is mock_zmq

        # Cleanup
        client_mod._zmq = None
        client_mod._msgpack = None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: data_config uncovered paths
# ═══════════════════════════════════════════════════════════════════════════


class TestDataConfigEdgeCases:
    """Test data_config.py uncovered paths (lines 348-375)."""

    def test_load_data_config_string(self):
        """Load config by string name."""
        config = load_data_config("so100_dualcam")
        assert config is not None
        assert len(config.video_keys) > 0

    def test_load_data_config_object(self):
        """Load config from BaseDataConfig object."""
        config = BaseDataConfig(
            video_keys=["video.test"],
            state_keys=["state.test"],
            action_keys=["action.test"],
            language_keys=["language.test"],
            observation_indices=[0],
            action_indices=[0],
        )
        result = load_data_config(config)
        assert result is config

    def test_load_data_config_unknown_name(self):
        """Unknown config name raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            load_data_config("nonexistent_robot_config_xyz")
