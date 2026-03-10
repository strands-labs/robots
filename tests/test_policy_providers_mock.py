#!/usr/bin/env python3
"""Comprehensive mock-based tests for all policy providers with low coverage.

Strategy — modeled on the gold-standard test_dreamzero.py (99% coverage):

1. **Pure utility methods** — tested directly with no mocking:
   - _extract_image()  / _find_camera_image()
   - _array_to_dict()  / _parse_action_text() / _parse_action() / _parse_numbers()
   - _build_prompt()   / _build_action_dict()
   - _extract_frame()  / _actions_to_dicts() / _generate_zero_actions()
   - _extract_joints()  / _flatten_history() / _update_history()
   - reset()

2. **_ensure_loaded() with mocked imports** — tests both success and error paths:
   - Server-mode providers: mock `requests`
   - Local-mode providers: mock `torch` + `transformers`
   - ONNX provider (GEAR-SONIC): mock `onnxruntime`
   - gRPC provider (lerobot_async): mock `grpc` + `lerobot.transport`

3. **get_actions() with pre-loaded model** — bypass _ensure_loaded():
   - Set _loaded = True, _model = MagicMock()
   - Test the observation → action pipeline end-to-end

Coverage targets (from 56% overall):
   ~690 new lines covered → projected 60.4% overall
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Common test observation factory
# ---------------------------------------------------------------------------


def _make_obs(h=224, w=224, c=3, key="camera", include_state=False, state_keys=None):
    """Build a standard observation dict with a camera image and optional state."""
    obs = {key: np.random.randint(0, 255, (h, w, c), dtype=np.uint8)}
    if include_state and state_keys:
        for k in state_keys:
            obs[k] = np.random.uniform(-1, 1)
    return obs


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Pure utility method tests (zero mocking)
# ═══════════════════════════════════════════════════════════════════════


class TestExtractImage:
    """Test _extract_image() across providers that share the same pattern.

    All HF-based VLAs have an identical _extract_image:
      - scan sorted keys for 3D ndarray with shape[-1] in (3, 4)
      - return PIL Image; default to Image.new("RGB", (224, 224))
    """

    # Providers with standard _extract_image
    PROVIDERS = [
        ("strands_robots.policies.cogact", "CogactPolicy", {}),
        ("strands_robots.policies.internvla", "InternvlaPolicy", {}),
        ("strands_robots.policies.robobrain", "RobobrainPolicy", {}),
        ("strands_robots.policies.unifolm", "UnifolmPolicy", {}),
        ("strands_robots.policies.magma", "MagmaPolicy", {}),
    ]

    @pytest.fixture(params=PROVIDERS, ids=lambda p: p[1])
    def policy(self, request):
        mod_path, cls_name, kwargs = request.param
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        return cls(**kwargs)

    def test_extracts_rgb_image(self, policy):
        from PIL import Image

        obs = {"cam": np.ones((100, 100, 3), dtype=np.uint8) * 128}
        img = policy._extract_image(obs)
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)

    def test_extracts_rgba_image(self, policy):
        from PIL import Image

        obs = {"cam": np.ones((100, 100, 4), dtype=np.uint8) * 42}
        img = policy._extract_image(obs)
        assert isinstance(img, Image.Image)
        # RGBA → RGB, so should use first 3 channels
        assert img.size == (100, 100)

    def test_empty_obs_returns_default(self, policy):
        from PIL import Image

        img = policy._extract_image({})
        assert isinstance(img, Image.Image)
        assert img.size == (224, 224)

    def test_skips_non_image_arrays(self, policy):
        from PIL import Image

        obs = {
            "scalar": np.array([1.0, 2.0]),
            "matrix": np.zeros((10, 10)),
            "cam": np.ones((50, 50, 3), dtype=np.uint8) * 200,
        }
        img = policy._extract_image(obs)
        assert isinstance(img, Image.Image)
        assert img.size == (50, 50)

    def test_uses_sorted_key_order(self, policy):
        """Sorted keys ensure deterministic image selection."""
        obs = {
            "z_camera": np.ones((30, 30, 3), dtype=np.uint8) * 10,
            "a_camera": np.ones((50, 50, 3), dtype=np.uint8) * 200,
        }
        img = policy._extract_image(obs)
        assert img.size == (50, 50)  # "a_camera" comes first alphabetically


class TestAlpamayoFindCameraImage:
    """Test AlpamayoPolicy._find_camera_image() — more complex camera logic."""

    def test_direct_match(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        obs = {"front_wide": np.ones((100, 100, 3), dtype=np.uint8) * 55}
        from PIL import Image

        img = p._find_camera_image(obs, "front_wide")
        assert isinstance(img, Image.Image)

    def test_fallback_any_image(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        obs = {"random_cam": np.ones((80, 80, 3), dtype=np.uint8) * 77}
        from PIL import Image

        img = p._find_camera_image(obs, "front_wide")
        assert isinstance(img, Image.Image)

    def test_no_image_returns_none(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        obs = {"scalar_key": 42.0}
        assert p._find_camera_image(obs, "front_wide") is None


class TestAlpamayoExtractEgomotion:
    """Test AlpamayoPolicy._extract_egomotion()."""

    def test_from_egomotion_key(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        ego = np.array([[1, 2, 3, 4]], dtype=np.float32)
        obs = {"egomotion": ego}
        result = p._extract_egomotion(obs)
        np.testing.assert_allclose(result, ego)

    def test_from_individual_keys(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        obs = {"ego_x": 1.0, "ego_y": 2.0, "ego_z": 3.0, "ego_yaw": 0.5}
        result = p._extract_egomotion(obs)
        assert result is not None
        assert result.shape[-1] == 4

    def test_missing_ego_keys_returns_none(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        obs = {"some_key": 42}
        assert p._extract_egomotion(obs) is None

    def test_history_accumulates(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        for i in range(20):
            obs = {"ego_x": float(i), "ego_y": 0, "ego_z": 0, "ego_yaw": 0}
            p._extract_egomotion(obs)
        # History should be capped at 16
        assert len(p._ego_history) <= 16


class TestAlpamayoParseTrajectory:
    """Test AlpamayoPolicy._parse_trajectory_from_text()."""

    def test_parse_numbered_text(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        text = "1.0 2.0 3.0 0.5 4.0 5.0 6.0 0.1"
        traj = p._parse_trajectory_from_text(text)
        assert traj.shape == (64, 4)
        np.testing.assert_allclose(traj[0], [1.0, 2.0, 3.0, 0.5])
        np.testing.assert_allclose(traj[1], [4.0, 5.0, 6.0, 0.1])

    def test_empty_text_pads_zeros(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        traj = p._parse_trajectory_from_text("")
        assert traj.shape == (64, 4)
        assert traj.sum() == 0

    def test_partial_text_pads_last_waypoint(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        text = "1.0 2.0 3.0 4.0"  # Only 1 waypoint
        traj = p._parse_trajectory_from_text(text)
        assert traj.shape == (64, 4)
        # All 64 should be the same (first + 63 padded)
        for i in range(64):
            np.testing.assert_allclose(traj[i], [1.0, 2.0, 3.0, 4.0])


class TestAlpamayoBuildActionDict:
    """Test AlpamayoPolicy._build_action_dict()."""

    def test_basic_trajectory(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        traj = np.zeros((64, 4), dtype=np.float32)
        traj[0] = [1.0, 2.0, 0.5, 0.1]
        traj[1] = [1.5, 2.5, 0.6, 0.2]
        traj[10] = [5.0, 6.0, 1.0, 0.3]
        result = p._build_action_dict(traj, "reasoning text")
        assert "x" in result
        assert "y" in result
        assert "trajectory" in result
        assert "reasoning" in result
        assert result["reasoning"] == "reasoning text"
        assert result["x"] == pytest.approx(5.0)  # waypoint_select=10

    def test_empty_trajectory(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        traj = np.zeros((0, 4), dtype=np.float32)
        result = p._build_action_dict(traj)
        assert "x" in result
        assert result["linear_vel"] == 0.0

    def test_1d_trajectory_reshaped(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        traj = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = p._build_action_dict(traj)
        assert result["x"] == pytest.approx(1.0)

    def test_no_reasoning_omitted(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        traj = np.zeros((2, 4), dtype=np.float32)
        result = p._build_action_dict(traj, "")
        assert "reasoning" not in result

    def test_robot_state_keys_mapped(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        p.set_robot_state_keys(["x", "y", "z"])
        traj = np.array([[1.0, 2.0, 0.5, 0.1], [1.5, 2.5, 0.6, 0.2]], dtype=np.float32)
        result = p._build_action_dict(traj, "test")
        assert "x" in result
        assert "y" in result


class TestAlpamayoReset:
    def test_reset_clears_state(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        p._step = 10
        p._ego_history = [np.zeros(4) for _ in range(5)]
        p.reset()
        assert p._step == 0
        assert len(p._ego_history) == 0


class TestParseActionText:
    """Test _parse_action_text() / _parse_action() / _parse_numbers() across providers."""

    def test_cogact_parse(self):
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy(action_dim=7)
        result = p._parse_action_text("Out: 0.1 0.2 0.3 0.4 0.5 0.6 0.7")
        assert result.shape == (1, 7)
        np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def test_cogact_parse_short_pads(self):
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy(action_dim=7)
        result = p._parse_action_text("0.1 0.2")
        assert result.shape == (1, 7)
        assert result[0, 2] == 0.0  # padded

    def test_internvla_parse(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(action_dim=7)
        result = p._parse_action_text("Out: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 extra 8.0")
        # Takes values after "Out:", limited to action_dim
        assert len(result) == 7

    def test_internvla_parse_empty(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(action_dim=7)
        result = p._parse_action_text("no numbers here")
        assert len(result) == 7
        assert all(v == 0.0 for v in result)

    def test_robobrain_parse(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy(action_dim=7)
        result = p._parse_action("some text 0.1 0.2 0.3 0.4 0.5 0.6 0.7")
        assert len(result) == 7

    def test_robobrain_parse_pads(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy(action_dim=5)
        result = p._parse_action("only 0.1")
        assert len(result) == 5

    def test_magma_parse(self):
        from strands_robots.policies.magma import MagmaPolicy

        p = MagmaPolicy(action_dim=7)
        result = p._parse_action_text("action 0.1 0.2 0.3 0.4 0.5 0.6 0.7")
        assert len(result) == 7
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def test_unifolm_parse_numbers(self):
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy(action_dim=5)
        result = p._parse_numbers("1.0 2.0 3.0 4.0 5.0 6.0")
        assert len(result) == 5
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])


class TestArrayToDict:
    """Test _array_to_dict() for providers that have it."""

    def test_internvla_with_state_keys(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy()
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5", "gripper"])
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = p._array_to_dict(arr)
        assert result["j0"] == pytest.approx(0.1)
        assert result["gripper"] == pytest.approx(0.7)

    def test_internvla_default_eef_keys(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy()
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = p._array_to_dict(arr)
        assert "x" in result
        assert "gripper" in result

    def test_robobrain_array_to_dict(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()
        p.set_robot_state_keys(["a", "b", "c"])
        arr = np.array([1.0, 2.0, 3.0])
        result = p._array_to_dict(arr)
        assert result == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_unifolm_array_to_dict_default_keys(self):
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy()
        arr = np.array([0.1, 0.2, 0.3])
        result = p._array_to_dict(arr)
        assert "joint_0" in result
        assert "joint_2" in result


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: DreamGen-specific pure methods
# ═══════════════════════════════════════════════════════════════════════


class TestDreamgenPureMethods:
    """Test DreamgenPolicy utility methods without mocking."""

    def test_extract_frame_hwc(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        obs = {"cam": frame}
        result = p._extract_frame(obs)
        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_extract_frame_chw(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        frame = np.ones((3, 100, 100), dtype=np.uint8)
        obs = {"cam": frame}
        result = p._extract_frame(obs)
        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_extract_frame_skips_state_keys(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        p.set_robot_state_keys(["j0"])
        obs = {"j0": np.ones((100, 100, 3), dtype=np.uint8)}
        result = p._extract_frame(obs)
        assert result is None

    def test_extract_frame_no_images(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        obs = {"scalar": 42.0}
        result = p._extract_frame(obs)
        assert result is None

    def test_actions_to_dicts(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        p.set_robot_state_keys(["j0", "j1", "j2"])
        actions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = p._actions_to_dicts(actions)
        assert len(result) == 2
        assert result[0] == {"j0": pytest.approx(0.1), "j1": pytest.approx(0.2), "j2": pytest.approx(0.3)}

    def test_generate_zero_actions(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", action_horizon=4)
        p.set_robot_state_keys(["j0", "j1"])
        result = p._generate_zero_actions()
        assert len(result) == 4
        assert all(a == {"j0": 0.0, "j1": 0.0} for a in result)

    def test_reset_clears_previous_frame(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        p._previous_frame = np.zeros((100, 100, 3))
        p.reset()
        assert p._previous_frame is None

    def test_build_dreams_observation(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        p.set_robot_state_keys(["j0", "j1"])
        obs = {
            "cam": np.zeros((100, 100, 3), dtype=np.uint8),
            "j0": 0.5,
            "j1": 0.6,
        }
        result = p._build_dreams_observation(obs, "pick up")
        assert result["annotation.human.task_description"] == "pick up"
        assert "state.j0" in result or "state.j1" in result

    def test_convert_dreams_actions_empty(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy")
        p.set_robot_state_keys(["j0"])
        result = p._convert_dreams_actions({})
        assert len(result) > 0  # Should return zero actions


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: GEAR-SONIC-specific pure methods
# ═══════════════════════════════════════════════════════════════════════


class TestGearSonicPureMethods:
    """Test GearSonicPolicy utility methods without mocking."""

    @pytest.fixture
    def policy(self):
        with patch.object(
            __import__("strands_robots.policies.gear_sonic", fromlist=["GearSonicPolicy"]).GearSonicPolicy,
            "_resolve_model_dir",
            return_value="/mock/dir",
        ):
            from strands_robots.policies.gear_sonic import GearSonicPolicy

            return GearSonicPolicy(model_dir="/mock/dir")

    def test_extract_joints_from_state_keys(self, policy):
        policy.set_robot_state_keys(["j0", "j1", "j2"])
        obs = {"j0": 0.1, "j1": 0.2, "j2": 0.3}
        result = policy._extract_joints(obs)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    def test_extract_joints_from_array(self, policy):
        obs = {"joint_positions": np.array([0.5, 0.6, 0.7])}
        result = policy._extract_joints(obs)
        np.testing.assert_allclose(result, [0.5, 0.6, 0.7])

    def test_extract_joints_default_zeros(self, policy):
        result = policy._extract_joints({})
        assert result.shape == (29,)
        assert result.sum() == 0

    def test_flatten_history_empty(self, policy):
        result = policy._flatten_history([], 10)
        assert result.shape == (10,)
        assert result.sum() == 0

    def test_flatten_history_with_data(self, policy):
        history = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = policy._flatten_history(history, 10)
        assert result.shape == (10,)
        np.testing.assert_allclose(result[:6], [1, 2, 3, 4, 5, 6])
        assert result[6:].sum() == 0

    def test_update_history(self, policy):
        joint_pos = np.ones(5, dtype=np.float32)
        action = np.ones(5, dtype=np.float32) * 2
        policy._update_history(joint_pos, action)
        assert len(policy._joint_pos_history) == 1
        assert len(policy._action_history) == 1
        assert len(policy._gravity_history) == 1

    def test_update_history_computes_velocity(self, policy):
        policy._update_history(np.array([0, 0, 0]), np.array([0, 0, 0]))
        policy._update_history(np.array([1, 2, 3]), np.array([1, 1, 1]))
        assert len(policy._joint_vel_history) == 2
        np.testing.assert_allclose(policy._joint_vel_history[1], [1, 2, 3])

    def test_update_history_trims(self, policy):
        for i in range(25):
            policy._update_history(np.zeros(3), np.zeros(3))
        max_h = policy._history_len + 5
        assert len(policy._joint_pos_history) <= max_h

    def test_reset(self, policy):
        policy._step = 10
        policy._joint_pos_history = [np.zeros(3)] * 5
        policy.reset()
        assert policy._step == 0
        assert len(policy._joint_pos_history) == 0


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: GR00T-specific pure methods
# ═══════════════════════════════════════════════════════════════════════


class TestGr00tPureMethods:
    """Test Gr00tPolicy helpers that don't need network/GPU."""

    def test_convert_to_robot_actions(self):
        from strands_robots.policies.groot import Gr00tPolicy

        # Create minimal service-mode policy
        with patch.object(Gr00tPolicy, "__init__", lambda self, **kw: None):
            p = Gr00tPolicy.__new__(Gr00tPolicy)
            p.robot_state_keys = ["j0", "j1", "j2"]
            p.camera_keys = []
            p.state_keys = []
            p.action_keys = []
            p.language_keys = []

        action_chunk = {
            "action.arm": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        }
        result = p._convert_to_robot_actions(action_chunk)
        assert len(result) == 2
        assert result[0]["j0"] == pytest.approx(0.1)
        assert result[1]["j2"] == pytest.approx(0.6)

    def test_convert_empty_chunk(self):
        from strands_robots.policies.groot import Gr00tPolicy

        with patch.object(Gr00tPolicy, "__init__", lambda self, **kw: None):
            p = Gr00tPolicy.__new__(Gr00tPolicy)
            p.robot_state_keys = ["j0"]
        result = p._convert_to_robot_actions({})
        assert result == []

    def test_map_video_key_to_camera_direct(self):
        from strands_robots.policies.groot import Gr00tPolicy

        with patch.object(Gr00tPolicy, "__init__", lambda self, **kw: None):
            p = Gr00tPolicy.__new__(Gr00tPolicy)
        obs = {"webcam": np.zeros((100, 100, 3))}
        assert p._map_video_key_to_camera("video.webcam", obs) == "webcam"

    def test_map_video_key_fallback(self):
        from strands_robots.policies.groot import Gr00tPolicy

        with patch.object(Gr00tPolicy, "__init__", lambda self, **kw: None):
            p = Gr00tPolicy.__new__(Gr00tPolicy)
        obs = {"front": np.zeros((100, 100, 3))}
        result = p._map_video_key_to_camera("video.webcam", obs)
        assert result == "front"

    def test_map_robot_state_so100(self):
        from strands_robots.policies.groot import Gr00tPolicy

        with patch.object(Gr00tPolicy, "__init__", lambda self, **kw: None):
            p = Gr00tPolicy.__new__(Gr00tPolicy)
            p.data_config_name = "so100_dualcam"
            p.state_keys = []
        obs_dict = {}
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        p._map_robot_state_to_gr00t_state(obs_dict, state)
        assert "state.single_arm" in obs_dict
        assert "state.gripper" in obs_dict
        np.testing.assert_allclose(obs_dict["state.single_arm"], [0.1, 0.2, 0.3, 0.4, 0.5])


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: RoboBrain-specific pure methods
# ═══════════════════════════════════════════════════════════════════════


class TestRobobrainBuildPrompt:
    """Test RobobrainPolicy._build_prompt()."""

    def test_basic_prompt(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()
        prompt = p._build_prompt("pick up the cup", {})
        assert "pick up the cup" in prompt
        assert "<image>" in prompt

    def test_prompt_with_state_keys(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()
        p.set_robot_state_keys(["x", "y", "z"])
        obs = {"x": 0.1, "y": 0.2, "z": 0.3}
        prompt = p._build_prompt("grasp", obs)
        assert "Robot state:" in prompt
        assert "x=0.100" in prompt

    def test_prompt_with_scene_memory(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy(enable_scene_memory=True)
        p._scene_history = ["saw a cup", "robot moved left"]
        prompt = p._build_prompt("grasp", {})
        assert "Scene context:" in prompt

    def test_update_scene_memory(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy(enable_scene_memory=True)
        for i in range(15):
            p._update_scene_memory(f"observation {i}")
        assert len(p._scene_history) == 10  # capped at 10


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: LeRobot Async pure methods
# ═══════════════════════════════════════════════════════════════════════


class TestLerobotAsyncPureMethods:
    """Test LerobotAsyncPolicy utility methods."""

    def test_build_raw_observation(self):
        from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

        with patch("strands_robots.policies.lerobot_async._validate_policy_type"):
            p = LerobotAsyncPolicy(server_address="localhost:8080", policy_type="pi0")
        p.set_robot_state_keys(["j0", "j1"])
        obs = {
            "j0": 0.5,
            "j1": 0.6,
            "cam": np.zeros((100, 100, 3)),
        }
        raw = p._build_raw_observation(obs, "pick up")
        assert raw["j0"] == 0.5
        assert raw["task"] == "pick up"
        assert "cam" in raw

    def test_generate_zero_actions(self):
        from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

        with patch("strands_robots.policies.lerobot_async._validate_policy_type"):
            p = LerobotAsyncPolicy(actions_per_chunk=3)
        p.set_robot_state_keys(["j0", "j1"])
        result = p._generate_zero_actions()
        assert len(result) == 3
        assert all(a == {"j0": 0.0, "j1": 0.0} for a in result)

    def test_disconnect_noop_when_not_connected(self):
        from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

        with patch("strands_robots.policies.lerobot_async._validate_policy_type"):
            p = LerobotAsyncPolicy()
        p.disconnect()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: _ensure_loaded() with mocked imports — Server mode
# ═══════════════════════════════════════════════════════════════════════


class TestEnsureLoadedServerMode:
    """Test _ensure_loaded() for server-mode providers (mock requests)."""

    def test_internvla_server_mode(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(server_url="http://fake:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            p._ensure_loaded()
        assert p._loaded is True

    def test_internvla_server_unreachable(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(server_url="http://fake:8000")
        with patch("requests.get", side_effect=ConnectionError("refused")):
            p._ensure_loaded()
        assert p._loaded is True  # Still marks loaded (just warns)

    def test_alpamayo_server_mode(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy(server_url="http://fake:8000")
        with patch("requests.get", return_value=MagicMock()):
            p._ensure_loaded()
        assert p._loaded is True

    def test_unifolm_server_mode(self):
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy(server_url="http://fake:8000")
        p._ensure_loaded()
        assert p._loaded is True

    def test_noop_if_already_loaded(self):
        """_ensure_loaded() should be idempotent."""
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(server_url="http://fake:8000")
        p._loaded = True
        p._ensure_loaded()  # Should not call requests.get
        assert p._loaded is True


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: _ensure_loaded() with mocked torch/transformers — Local mode
# ═══════════════════════════════════════════════════════════════════════


class TestEnsureLoadedLocalMode:
    """Test _ensure_loaded() local mode with mocked torch + transformers."""

    def _mock_torch_and_transformers(self):
        """Create mock torch and transformers modules."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=1000))]
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoModelForVision2Seq.from_pretrained.return_value = mock_model
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoProcessor.from_pretrained.return_value = MagicMock()

        return mock_torch, mock_transformers, mock_model

    def test_cogact_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True
        assert p._model is mock_model

    def test_robobrain_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_magma_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.magma import MagmaPolicy

        p = MagmaPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_openvla_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.openvla import OpenvlaPolicy

        p = OpenvlaPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_internvla_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_unifolm_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_alpamayo_local_load(self):
        mock_torch, mock_tf, mock_model = self._mock_torch_and_transformers()
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            p._ensure_loaded()
        assert p._loaded is True

    def test_rdt_local_load_fallback(self):
        """RDT has a special loading pattern — test the HF fallback path."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = "config.json"

        from strands_robots.policies.rdt import RdtPolicy

        p = RdtPolicy()
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "huggingface_hub": mock_hf,
            },
        ):
            # The main import (scripts.agilex_model) will fail, triggering fallback
            p._ensure_loaded()
        assert p._loaded is True


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: GEAR-SONIC ONNX loading mock
# ═══════════════════════════════════════════════════════════════════════


class TestGearSonicLoadModels:
    """Test GearSonicPolicy._load_models() with mocked onnxruntime."""

    def test_load_onnx_models(self):
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input0", shape=[1, 64]),
            MagicMock(name="input1", shape=[1, 116]),
        ]
        mock_session.get_outputs.return_value = [MagicMock(name="output0")]
        mock_ort.InferenceSession.return_value = mock_session

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value="/mock"):
            p = GearSonicPolicy(model_dir="/mock")

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            p._load_models()

        assert p._encoder is mock_session
        assert p._decoder is mock_session


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: get_actions() with pre-loaded models (bypass loading)
# ═══════════════════════════════════════════════════════════════════════


class TestGetActionsPreloaded:
    """Test get_actions() with _loaded=True, _model=MagicMock().

    This tests the full observation → action pipeline without GPU dependencies.
    """

    def test_cogact_get_actions_with_predict_action(self):
        """CogACT with model.predict_action API."""
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy(action_dim=7)
        p._loaded = True
        p._device = "cpu"
        p._model = MagicMock()
        p._processor = MagicMock()
        p._model.predict_action.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5", "gripper"])

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick up the cube"))

        assert len(actions) >= 1
        assert "j0" in actions[0]
        assert p._step == 1

    def test_magma_get_actions_text_fallback(self):
        """Magma without predict_action → text generation fallback."""
        from strands_robots.policies.magma import MagmaPolicy

        p = MagmaPolicy(action_dim=7)
        p._loaded = True
        p._device = "cpu"
        p._model = MagicMock()
        del p._model.predict_action  # ensure hasattr returns False
        p._processor = MagicMock()

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        # model.generate returns tensor, processor.decode returns text with numbers
        p._model.generate.return_value = MagicMock()
        p._processor.decode.return_value = "action: 0.1 0.2 0.3 0.4 0.5 0.6 0.7"
        p._processor.return_value = MagicMock()
        p._processor.return_value.to.return_value = {}

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick up"))
        assert len(actions) == 1
        assert p._step == 1

    def test_internvla_server_get_actions(self):
        """InternVLA server mode get_actions with mocked requests."""
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy(server_url="http://fake:8000")
        p._loaded = True

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"actions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
        mock_resp.raise_for_status = MagicMock()

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 1
        assert "x" in actions[0]

    def test_alpamayo_server_get_actions(self):
        """Alpamayo server mode."""
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy(server_url="http://fake:8000")
        p._loaded = True

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "trajectory": np.zeros((64, 4)).tolist(),
            "reasoning": "test reasoning",
        }
        mock_resp.raise_for_status = MagicMock()

        obs = {"front_wide": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "turn left"))

        assert len(actions) == 1
        assert "trajectory" in actions[0]
        assert "reasoning" in actions[0]

    def test_unifolm_server_get_actions(self):
        """UnifoLM server mode."""
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy(server_url="http://fake:8000", action_dim=5)
        p._loaded = True
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4"])

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"actions": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_resp.raise_for_status = MagicMock()

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch("requests.post", return_value=mock_resp):
            actions = _run(p.get_actions(obs, "walk"))

        assert len(actions) == 1
        assert "j0" in actions[0]

    def test_unifolm_no_model_returns_zeros(self):
        """UnifoLM with no model loaded returns zeros."""
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy()
        p._loaded = True
        p._model = None
        p.set_robot_state_keys(["j0", "j1"])
        actions = _run(p.get_actions({}, "walk"))
        assert actions == [{"j0": 0.0, "j1": 0.0}]

    def test_rdt_no_model_returns_zeros(self):
        """RDT with no model returns zeros."""
        from strands_robots.policies.rdt import RdtPolicy

        p = RdtPolicy()
        p._loaded = True
        p._model = None
        p.set_robot_state_keys(["j0", "j1"])
        actions = _run(p.get_actions({}, "pick"))
        assert actions == [{"j0": 0.0, "j1": 0.0}]

    def test_dreamgen_idm_first_frame(self):
        """DreamGen IDM mode: first call stores frame, returns zeros."""
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", mode="idm")
        p._model = MagicMock()  # skip loading
        p.set_robot_state_keys(["j0", "j1"])

        obs = {"cam": np.ones((100, 100, 3), dtype=np.uint8)}

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        # First frame → previous_frame stored, zero actions returned
        assert p._previous_frame is not None
        assert all(a["j0"] == 0.0 for a in actions)

    def test_dreamgen_idm_second_frame(self):
        """DreamGen IDM mode: second call runs IDM inference."""
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", mode="idm")
        mock_model = MagicMock()
        mock_model.get_action.return_value = {"action_pred": np.array([[[0.1, 0.2], [0.3, 0.4]]])}
        p._model = mock_model
        p._previous_frame = np.ones((100, 100, 3), dtype=np.uint8)
        p.set_robot_state_keys(["j0", "j1"])

        obs = {"cam": np.ones((100, 100, 3), dtype=np.uint8) * 128}

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))

        assert len(actions) == 2
        assert actions[0]["j0"] == pytest.approx(0.1)

    def test_dreamgen_invalid_mode(self):
        """DreamGen with unknown mode raises ValueError."""
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", mode="invalid_mode")
        with pytest.raises(ValueError, match="Unknown mode"):
            _run(p.get_actions({}, "pick"))


class TestGetActionsGearSonic:
    """Test GEAR-SONIC get_actions with fully mocked ONNX sessions."""

    def test_get_actions_full_pipeline(self):
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value="/mock"):
            p = GearSonicPolicy(model_dir="/mock")

        p.set_robot_state_keys(["j0", "j1", "j2"])

        # Mock encoder session
        mock_encoder = MagicMock()
        mock_encoder.get_inputs.return_value = [
            MagicMock(name="input0", shape=[1, 64]),
        ]
        mock_encoder.run.return_value = [np.zeros((1, 64), dtype=np.float32)]

        # Mock decoder session
        mock_decoder = MagicMock()
        mock_decoder.get_inputs.return_value = [
            MagicMock(name="token_state", shape=[1, 64]),
            MagicMock(name="joint_position", shape=[1, 3]),
        ]
        mock_decoder.run.return_value = [np.array([[0.1, 0.2, 0.3]], dtype=np.float32)]

        p._encoder = mock_encoder
        p._decoder = mock_decoder
        p._enc_inputs = {i.name: i for i in mock_encoder.get_inputs()}
        p._dec_inputs = {i.name: i for i in mock_decoder.get_inputs()}
        p._enc_out_names = ["output0"]
        p._dec_out_names = ["output0"]

        obs = {"j0": 0.0, "j1": 0.0, "j2": 0.0}
        actions = _run(p.get_actions(obs, "walk"))

        assert len(actions) == 1
        assert "j0" in actions[0]
        assert actions[0]["j0"] == pytest.approx(0.1)
        assert p._step == 1


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: Provider properties and construction
# ═══════════════════════════════════════════════════════════════════════


class TestProviderProperties:
    """Verify provider_name and constructor defaults for all low-coverage providers."""

    PROVIDERS = [
        ("strands_robots.policies.cogact", "CogactPolicy", "cogact", {}),
        ("strands_robots.policies.internvla", "InternvlaPolicy", "internvla", {}),
        ("strands_robots.policies.robobrain", "RobobrainPolicy", "robobrain", {}),
        ("strands_robots.policies.unifolm", "UnifolmPolicy", "unifolm", {}),
        ("strands_robots.policies.magma", "MagmaPolicy", "magma", {}),
        ("strands_robots.policies.openvla", "OpenvlaPolicy", "openvla", {}),
        ("strands_robots.policies.rdt", "RdtPolicy", "rdt", {}),
        ("strands_robots.policies.alpamayo", "AlpamayoPolicy", "alpamayo", {}),
        ("strands_robots.policies.dreamgen", "DreamgenPolicy", "dreamgen", {"model_path": "x"}),
    ]

    @pytest.fixture(params=PROVIDERS, ids=lambda p: p[2])
    def provider_info(self, request):
        return request.param

    def test_provider_name(self, provider_info):
        mod_path, cls_name, expected_name, kwargs = provider_info
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        p = cls(**kwargs)
        assert p.provider_name == expected_name

    def test_set_robot_state_keys(self, provider_info):
        mod_path, cls_name, _, kwargs = provider_info
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        p = cls(**kwargs)
        keys = ["j0", "j1", "j2"]
        p.set_robot_state_keys(keys)
        # Most store in _robot_state_keys, dreamgen uses robot_state_keys
        stored = getattr(p, "_robot_state_keys", None) or getattr(p, "robot_state_keys", None)
        assert stored == keys

    def test_lazy_loading(self, provider_info):
        """Model should not be loaded at construction time."""
        mod_path, cls_name, _, kwargs = provider_info
        mod = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        p = cls(**kwargs)
        assert getattr(p, "_model", None) is None or getattr(p, "_loaded", None) is False


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12: Alpamayo _ensure_loaded() error path
# ═══════════════════════════════════════════════════════════════════════


class TestAlpamayoEnsureLoadedError:
    """Test that Alpamayo raises proper ImportError on model load failure."""

    def test_local_load_failure_raises(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_tf = MagicMock()
        mock_tf.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError("OOM")
        mock_tf.AutoProcessor.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            with pytest.raises(ImportError, match="Failed to load Alpamayo"):
                p._ensure_loaded()


class TestInternvlaEnsureLoadedError:
    """Test InternVLA local load error path."""

    def test_local_load_failure_raises(self):
        from strands_robots.policies.internvla import InternvlaPolicy

        p = InternvlaPolicy()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_tf = MagicMock()
        mock_tf.AutoModelForVision2Seq.from_pretrained.side_effect = RuntimeError("OOM")
        mock_tf.AutoProcessor.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            with pytest.raises(ImportError, match="Failed to load InternVLA"):
                p._ensure_loaded()


class TestRobobrainEnsureLoadedError:
    """Test RoboBrain local load error path."""

    def test_local_load_failure_raises(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_tf = MagicMock()
        mock_tf.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError("OOM")
        mock_tf.AutoProcessor.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            with pytest.raises(ImportError, match="Failed to load RoboBrain"):
                p._ensure_loaded()


class TestCogactEnsureLoadedError:
    """Test CogACT local load error path."""

    def test_local_load_failure_raises(self):
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_tf = MagicMock()
        mock_tf.AutoModelForVision2Seq.from_pretrained.side_effect = RuntimeError("OOM")
        mock_tf.AutoProcessor.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_tf}):
            with pytest.raises(ImportError, match="Failed to load CogACT"):
                p._ensure_loaded()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13: Alpamayo local inference with mocked model
# ═══════════════════════════════════════════════════════════════════════


class TestAlpamayoLocalInference:
    """Test AlpamayoPolicy._infer_local() with mocked model."""

    def test_predict_trajectory_api(self):
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = {"input_ids": MagicMock()}

        mock_model = MagicMock()
        mock_model.predict_trajectory.return_value = {
            "trajectory": np.zeros((64, 4)),
            "reasoning": "turning left because...",
        }
        p._model = mock_model

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.Tensor = type(MagicMock())

        obs = {"front_wide": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p._infer_local(obs, "turn left")

        assert len(result) == 1
        assert "trajectory" in result[0]

    def test_generate_fallback(self):
        """When model doesn't have predict_trajectory, fall back to generate."""
        from strands_robots.policies.alpamayo import AlpamayoPolicy

        p = AlpamayoPolicy()
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = {"input_ids": MagicMock()}
        p._processor.decode.return_value = "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0"

        mock_model = MagicMock()
        del mock_model.predict_trajectory  # ensure hasattr returns False
        mock_model.generate = MagicMock(return_value=[MagicMock()])
        p._model = mock_model

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.Tensor = type(MagicMock())

        obs = {"front_wide": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p._infer_local(obs, "go straight")

        assert len(result) == 1
        assert "trajectory" in result[0]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 14: RoboBrain bonus capabilities mock tests
# ═══════════════════════════════════════════════════════════════════════


class TestRobobrainBonusCapabilities:
    """Test RoboBrain's spatial_refer, predict_trajectory, reason_about_scene."""

    def _make_loaded_policy(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy()
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = MagicMock()
        p._processor.return_value.to.return_value = {}
        p._processor.decode.return_value = "coordinates: 0.5 0.3 0.8 0.6"
        p._model = MagicMock()
        p._model.generate.return_value = [MagicMock()]
        return p

    def test_spatial_refer(self):
        p = self._make_loaded_policy()
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p.spatial_refer(obs, "red cup")

        assert "text" in result
        assert "coordinates" in result
        assert len(result["coordinates"]) == 4

    def test_predict_trajectory(self):
        p = self._make_loaded_policy()
        p._processor.decode.return_value = "waypoints: 0.1 0.2 0.3 0.4 0.5 0.6"

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p.predict_trajectory(obs, "move to table")

        assert len(result) >= 1
        assert "x" in result[0]
        assert "y" in result[0]

    def test_reason_about_scene(self):
        p = self._make_loaded_policy()
        p._processor.decode.return_value = "I see a cup on the table."

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p.reason_about_scene(obs, "What objects are on the table?")

        assert "cup" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 15: Magma bonus capabilities
# ═══════════════════════════════════════════════════════════════════════


class TestMagmaBonusCapabilities:
    """Test Magma's reason_about_scene."""

    def test_reason_about_scene(self):
        from strands_robots.policies.magma import MagmaPolicy

        p = MagmaPolicy()
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = MagicMock()
        p._processor.return_value.to.return_value = {}
        p._processor.decode.return_value = "There is a red block on the table."
        p._model = MagicMock()
        p._model.generate.return_value = [MagicMock()]

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = p.reason_about_scene(obs, "What do you see?")

        assert "red block" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 16: Provider set_robot_state_keys side effects
# ═══════════════════════════════════════════════════════════════════════


class TestSetRobotStateKeysSideEffects:
    """Some providers update internal dims when set_robot_state_keys is called."""

    def test_cogact_updates_action_dim(self):
        from strands_robots.policies.cogact import CogactPolicy

        p = CogactPolicy(action_dim=7)
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8"])
        assert p._action_dim == 9  # max(9, 7)

    def test_robobrain_updates_action_dim(self):
        from strands_robots.policies.robobrain import RobobrainPolicy

        p = RobobrainPolicy(action_dim=7)
        p.set_robot_state_keys(["j0", "j1", "j2"])
        assert p._action_dim == 3

    def test_magma_updates_action_dim(self):
        from strands_robots.policies.magma import MagmaPolicy

        p = MagmaPolicy(action_dim=7)
        p.set_robot_state_keys(["j0", "j1", "j2", "j3"])
        assert p._action_dim == 4

    def test_unifolm_updates_action_dim(self):
        from strands_robots.policies.unifolm import UnifolmPolicy

        p = UnifolmPolicy(action_dim=12)
        p.set_robot_state_keys(["j0", "j1", "j2"])
        assert p._action_dim == 3

    def test_rdt_updates_state_dim(self):
        from strands_robots.policies.rdt import RdtPolicy

        p = RdtPolicy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        assert p._state_dim == 3


# ═══════════════════════════════════════════════════════════════════════
# SECTION 17: GR00T build_observation and build_encoder/decoder input
# ═══════════════════════════════════════════════════════════════════════


class TestGearSonicBuildInputs:
    """Test GEAR-SONIC _build_encoder_input and _build_decoder_input."""

    def test_build_encoder_input(self):
        from types import SimpleNamespace

        from strands_robots.policies.gear_sonic import GearSonicPolicy

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value="/mock"):
            p = GearSonicPolicy(model_dir="/mock", mode="teleop")

        mock_encoder = MagicMock()
        mock_encoder.get_inputs.return_value = [
            SimpleNamespace(name="input_mode", shape=[1, 3]),
            SimpleNamespace(name="other_input", shape=[1, 64]),
        ]
        p._encoder = mock_encoder

        result = p._build_encoder_input({})
        assert "input_mode" in result
        assert result["input_mode"].shape == (1, 3)

    def test_build_decoder_input_with_history(self):
        from types import SimpleNamespace

        from strands_robots.policies.gear_sonic import GearSonicPolicy

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value="/mock"):
            p = GearSonicPolicy(model_dir="/mock")

        p.set_robot_state_keys(["j0", "j1"])
        p._joint_pos_history = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        mock_decoder = MagicMock()
        mock_decoder.get_inputs.return_value = [
            SimpleNamespace(name="token_state", shape=[1, 64]),
            SimpleNamespace(name="his_body_joint_positions_10frame_step1", shape=[1, 20]),
        ]
        p._decoder = mock_decoder

        token = np.zeros(64, dtype=np.float32)
        result = p._build_decoder_input({"j0": 0.5, "j1": 0.6}, token)
        assert "token_state" in result


# ═══════════════════════════════════════════════════════════════════════
# SECTION 18: DreamGen VLA mode error handling
# ═══════════════════════════════════════════════════════════════════════


class TestDreamgenVLAMode:
    """Test DreamGen VLA mode paths."""

    def test_vla_get_actions_error_returns_zeros(self):
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", mode="vla")
        p._policy = MagicMock()
        p._policy.get_action.side_effect = RuntimeError("inference error")
        p.set_robot_state_keys(["j0", "j1"])
        actions = _run(p.get_actions({}, "pick"))
        assert all(a["j0"] == 0.0 for a in actions)

    def test_idm_inference_error_returns_zeros(self):
        """IDM inference error should return zeros, not crash."""
        from strands_robots.policies.dreamgen import DreamgenPolicy

        p = DreamgenPolicy(model_path="dummy", mode="idm")
        p._model = MagicMock()
        p._model.get_action.side_effect = RuntimeError("CUDA OOM")
        p._previous_frame = np.ones((100, 100, 3), dtype=np.uint8)
        p.set_robot_state_keys(["j0"])

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()

        obs = {"cam": np.ones((100, 100, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick"))
        assert all(a["j0"] == 0.0 for a in actions)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 19: LeRobot Async validate_policy_type
# ═══════════════════════════════════════════════════════════════════════


class TestLerobotAsyncValidation:
    """Test _validate_policy_type edge cases."""

    def test_validation_skipped_without_lerobot(self):
        """When lerobot is not installed, validation is skipped."""
        from strands_robots.policies.lerobot_async import _validate_policy_type

        with patch.dict(sys.modules, {"lerobot": None, "lerobot.policies": None, "lerobot.policies.factory": None}):
            _validate_policy_type("any_type")  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# SECTION 20: GEAR-SONIC _resolve_model_dir paths
# ═══════════════════════════════════════════════════════════════════════


class TestGearSonicResolveModelDir:
    """Test GEAR-SONIC model directory resolution."""

    def test_local_dir_exists(self, tmp_path):
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value=str(tmp_path)):
            p = GearSonicPolicy(model_dir=str(tmp_path))
        assert p._model_dir == str(tmp_path)

    def test_hf_download_fallback(self):
        """When local dir doesn't exist, try HF download."""
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        mock_hf = MagicMock()
        mock_hf.snapshot_download.return_value = "/cached/path"

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            p = GearSonicPolicy(model_dir="/nonexistent")
        # Should have attempted download
        assert p._model_dir is not None

    def test_no_dir_no_hf_raises(self):
        """When both local and HF fail, raises FileNotFoundError."""
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        mock_hf = MagicMock()
        mock_hf.snapshot_download.side_effect = Exception("network error")

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            with pytest.raises(FileNotFoundError):
                GearSonicPolicy()  # No model_dir, HF fails


# ═══════════════════════════════════════════════════════════════════════
# SECTION 21: GEAR-SONIC ONNX import error
# ═══════════════════════════════════════════════════════════════════════


class TestGearSonicOnnxImportError:
    """Test that _load_models raises ImportError without onnxruntime."""

    def test_import_error(self):
        from strands_robots.policies.gear_sonic import GearSonicPolicy

        with patch.object(GearSonicPolicy, "_resolve_model_dir", return_value="/mock"):
            p = GearSonicPolicy(model_dir="/mock")

        with patch.dict(sys.modules, {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                p._load_models()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 22: OpenVLA get_actions with mocked model
# ═══════════════════════════════════════════════════════════════════════


class TestOpenvlaGetActions:
    """Test OpenVLA get_actions with predict_action API."""

    def test_get_actions_predict_action(self):
        from strands_robots.policies.openvla import OpenvlaPolicy

        p = OpenvlaPolicy()
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = MagicMock()
        p._processor.return_value.to.return_value = {}

        mock_vla = MagicMock()
        mock_vla.predict_action.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        p._vla = mock_vla

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            actions = _run(p.get_actions(obs, "pick up"))

        assert len(actions) == 1
        assert "x" in actions[0]
        assert actions[0]["x"] == pytest.approx(0.1)
        assert p._step == 1

    def test_get_actions_with_custom_unnorm_key(self):
        from strands_robots.policies.openvla import OpenvlaPolicy

        p = OpenvlaPolicy(unnorm_key="bridge_orig")
        p._loaded = True
        p._device = "cpu"
        p._processor = MagicMock()
        p._processor.return_value = MagicMock()
        p._processor.return_value.to.return_value = {}
        mock_vla = MagicMock()
        mock_vla.predict_action.return_value = np.array([0.1] * 7)
        p._vla = mock_vla

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        obs = {"cam": np.zeros((224, 224, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"torch": mock_torch}):
            _run(p.get_actions(obs, "pick", unnorm_key="custom_key"))

        # Verify unnorm_key was passed through
        call_kwargs = mock_vla.predict_action.call_args
        assert "unnorm_key" in call_kwargs.kwargs or "unnorm_key" in str(call_kwargs)
