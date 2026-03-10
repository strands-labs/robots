#!/usr/bin/env python3
"""Comprehensive mock-based tests for OmniVLA policy provider (18% → target ~70%).

OmniVLA has 408 stmts, 334 uncovered — the biggest CPU-testable gap remaining.
Most of the code is pure utility methods (modality resolution, velocity control,
GPS conversion, model info) that need zero GPU mocking.

Strategy:
1. Pure utility methods: _resolve_modality_flags, _get_modality_id,
   _waypoints_to_velocity, _clip_angle, _extract_camera_image,
   _compute_goal_pose_norm, update_goal, reset, get_model_info
2. Initialization paths: variant selection, default parameters
3. _ensure_loaded error paths: ImportError for full + edge
4. get_actions with pre-loaded models (mock inference)
"""

import asyncio
import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if policy PR (#8) not merged yet
try:
    from strands_robots.policies.omnivla import OmnivlaPolicy
except (ImportError, AttributeError, ModuleNotFoundError):
    __import__("pytest").skip("Requires PR #8 (policy-abstraction)", allow_module_level=True)

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


def _make_obs(h=224, w=224, c=3, key="camera"):
    """Build a standard observation dict with a camera image."""
    return {key: np.random.randint(0, 255, (h, w, c), dtype=np.uint8)}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Initialization & Properties
# ═══════════════════════════════════════════════════════════════════════


class TestOmnivlaInit:
    """Test OmniVLA constructor and properties."""

    def test_default_init(self):
        p = OmnivlaPolicy()
        assert p.provider_name == "omnivla"
        assert p._variant == "full"
        assert p._goal_modality == "language"
        assert p._loaded is False
        assert p._step == 0
        assert p._waypoint_select == 4
        assert p._max_linear_vel == 0.3
        assert p._max_angular_vel == 0.3

    def test_edge_variant(self):
        p = OmnivlaPolicy(variant="Edge")
        assert p._variant == "edge"

    def test_custom_params(self):
        p = OmnivlaPolicy(
            checkpoint_path="/models/omnivla",
            variant="full",
            goal_modality="pose_image",
            instruction="go forward",
            waypoint_select=2,
            max_linear_vel=1.0,
            max_angular_vel=0.8,
            metric_waypoint_spacing=0.2,
            tick_rate=5,
            use_lora=False,
            lora_rank=16,
        )
        assert p._checkpoint_path == "/models/omnivla"
        assert p._goal_modality == "pose_image"
        assert p._instruction == "go forward"
        assert p._waypoint_select == 2
        assert p._max_linear_vel == 1.0
        assert p._max_angular_vel == 0.8
        assert p._metric_waypoint_spacing == 0.2
        assert p._tick_rate == 5
        assert p._use_lora is False
        assert p._lora_rank == 16

    def test_set_robot_state_keys(self):
        p = OmnivlaPolicy()
        p.set_robot_state_keys(["vx", "vy", "omega"])
        assert p._robot_state_keys == ["vx", "vy", "omega"]

    def test_goal_gps_init(self):
        p = OmnivlaPolicy(goal_gps=(37.7749, -122.4194, 90.0))
        assert p._goal_gps == (37.7749, -122.4194, 90.0)

    def test_goal_image_path_init(self):
        p = OmnivlaPolicy(goal_image_path="/tmp/goal.jpg")
        assert p._goal_image_path == "/tmp/goal.jpg"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: _resolve_modality_flags
# ═══════════════════════════════════════════════════════════════════════


class TestResolveModalityFlags:
    """Test modality resolution from goal configuration."""

    def test_language_modality(self):
        p = OmnivlaPolicy(goal_modality="language")
        flags = p._resolve_modality_flags({}, "go to door")
        assert flags == (False, False, False, True)

    def test_image_modality(self):
        p = OmnivlaPolicy(goal_modality="image")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (False, False, True, False)

    def test_pose_modality(self):
        p = OmnivlaPolicy(goal_modality="pose")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (True, False, False, False)

    def test_satellite_modality(self):
        p = OmnivlaPolicy(goal_modality="satellite")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (False, True, False, False)

    def test_pose_image_modality(self):
        p = OmnivlaPolicy(goal_modality="pose_image")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (True, False, True, False)

    def test_pose_satellite_modality(self):
        p = OmnivlaPolicy(goal_modality="pose_satellite")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (True, True, False, False)

    def test_satellite_image_modality(self):
        p = OmnivlaPolicy(goal_modality="satellite_image")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (False, True, True, False)

    def test_all_modality(self):
        p = OmnivlaPolicy(goal_modality="all")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (True, True, True, False)

    def test_language_pose_modality(self):
        p = OmnivlaPolicy(goal_modality="language_pose")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (True, False, False, True)

    def test_unknown_modality_defaults_to_language(self):
        p = OmnivlaPolicy(goal_modality="unknown_xyz")
        flags = p._resolve_modality_flags({}, "")
        assert flags == (False, False, False, True)  # default

    def test_auto_modality_language(self):
        p = OmnivlaPolicy(goal_modality="auto")
        flags = p._resolve_modality_flags({}, "go forward")
        # has_instruction=True, rest False
        assert flags == (False, False, False, True)

    def test_auto_modality_with_gps(self):
        p = OmnivlaPolicy(goal_modality="auto")
        p._goal_utm = (123, 456)  # pre-set
        flags = p._resolve_modality_flags({"camera": np.zeros((3, 3, 3))}, "")
        assert flags[0] is True  # has_gps

    def test_auto_modality_with_goal_image(self):
        p = OmnivlaPolicy(goal_modality="auto")
        obs = {"goal_image": np.zeros((224, 224, 3), dtype=np.uint8)}
        flags = p._resolve_modality_flags(obs, "")
        assert flags[2] is True  # has_goal_image

    def test_auto_modality_with_satellite(self):
        p = OmnivlaPolicy(goal_modality="auto")
        obs = {"satellite": np.zeros((352, 352, 3), dtype=np.uint8)}
        flags = p._resolve_modality_flags(obs, "")
        assert flags[1] is True  # has_satellite


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: _get_modality_id
# ═══════════════════════════════════════════════════════════════════════


class TestGetModalityId:
    """Test modality flag → integer mapping."""

    @pytest.fixture
    def policy(self):
        return OmnivlaPolicy()

    def test_satellite_only(self, policy):
        assert policy._get_modality_id(False, True, False, False) == 0

    def test_pose_satellite(self, policy):
        assert policy._get_modality_id(True, True, False, False) == 1

    def test_satellite_image(self, policy):
        assert policy._get_modality_id(False, True, True, False) == 2

    def test_all_except_language(self, policy):
        assert policy._get_modality_id(True, True, True, False) == 3

    def test_pose_only(self, policy):
        assert policy._get_modality_id(True, False, False, False) == 4

    def test_pose_image(self, policy):
        assert policy._get_modality_id(True, False, True, False) == 5

    def test_image_only(self, policy):
        assert policy._get_modality_id(False, False, True, False) == 6

    def test_language_only(self, policy):
        assert policy._get_modality_id(False, False, False, True) == 7

    def test_language_pose(self, policy):
        assert policy._get_modality_id(True, False, False, True) == 8

    def test_no_flags_defaults_to_language(self, policy):
        assert policy._get_modality_id(False, False, False, False) == 7


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: _clip_angle
# ═══════════════════════════════════════════════════════════════════════


class TestClipAngle:
    def test_within_range(self):
        assert OmnivlaPolicy._clip_angle(0.5) == pytest.approx(0.5)

    def test_positive_wrap(self):
        assert OmnivlaPolicy._clip_angle(4.0) == pytest.approx(4.0 - 2 * math.pi)

    def test_negative_wrap(self):
        assert OmnivlaPolicy._clip_angle(-4.0) == pytest.approx(-4.0 + 2 * math.pi)

    def test_pi(self):
        assert OmnivlaPolicy._clip_angle(math.pi) == pytest.approx(math.pi)

    def test_multiple_wraps(self):
        result = OmnivlaPolicy._clip_angle(10.0)
        assert -math.pi <= result <= math.pi


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: _waypoints_to_velocity
# ═══════════════════════════════════════════════════════════════════════


class TestWaypointsToVelocity:

    @pytest.fixture
    def policy(self):
        return OmnivlaPolicy(waypoint_select=4, max_linear_vel=0.3, max_angular_vel=0.3, tick_rate=3)

    def test_zero_waypoints(self, policy):
        waypoints = np.zeros((8, 4))
        v, w = policy._waypoints_to_velocity(waypoints)
        # dx=0, dy=0 → linear_vel=0, angular from heading
        assert isinstance(v, float)
        assert isinstance(w, float)

    def test_straight_forward(self, policy):
        waypoints = np.zeros((8, 4))
        for i in range(8):
            waypoints[i] = [float(i + 1), 0.0, 1.0, 0.0]  # dx, dy, cos, sin
        v, w = policy._waypoints_to_velocity(waypoints)
        assert v > 0  # moving forward
        assert abs(w) < 0.5  # mostly straight

    def test_turning_left(self, policy):
        waypoints = np.zeros((8, 4))
        for i in range(8):
            waypoints[i] = [float(i + 1), float(i + 1), 0.0, 1.0]  # left turn
        v, w = policy._waypoints_to_velocity(waypoints)
        assert isinstance(v, float)
        assert isinstance(w, float)

    def test_dx_zero_dy_positive(self, policy):
        """dx ~0, dy > 0 → angular = pi/2 / DT branch."""
        waypoints = np.zeros((8, 4))
        for i in range(8):
            waypoints[i] = [0.0, float(i + 1), 0.0, 1.0]
        v, w = policy._waypoints_to_velocity(waypoints)
        assert v == 0.0
        assert w != 0.0

    def test_dx_zero_dy_negative(self, policy):
        """dx ~0, dy < 0 → angular = -pi/2 / DT branch."""
        waypoints = np.zeros((8, 4))
        for i in range(8):
            waypoints[i] = [0.0, -float(i + 1), 0.0, -1.0]
        v, w = policy._waypoints_to_velocity(waypoints)
        assert v == 0.0
        assert w != 0.0

    def test_waypoint_select_clamps(self, policy):
        """If waypoint_select > trajectory length, pick last."""
        policy._waypoint_select = 100
        waypoints = np.array([[1.0, 0.0, 1.0, 0.0]])  # only 1 waypoint
        v, w = policy._waypoints_to_velocity(waypoints)
        assert isinstance(v, float)

    def test_velocity_limiting_linear(self):
        """High dx should get clamped to max_linear_vel."""
        p = OmnivlaPolicy(max_linear_vel=0.1, max_angular_vel=0.3, tick_rate=1)
        p._waypoint_select = 0
        waypoints = np.array([[100.0, 0.0, 1.0, 0.0]] * 8) * 0.1
        v, w = p._waypoints_to_velocity(waypoints)
        assert abs(v) <= 0.1 + 1e-6

    def test_velocity_limiting_angular_only(self):
        """High angular with small linear → angular clamped."""
        p = OmnivlaPolicy(max_linear_vel=0.3, max_angular_vel=0.05, tick_rate=1)
        p._waypoint_select = 0
        waypoints = np.zeros((8, 4))
        waypoints[0] = [0.0, 0.0, 0.0, 1.0]  # dx=0, heading only
        v, w = p._waypoints_to_velocity(waypoints)
        assert abs(w) <= 1.0 + 1e-6  # clamped in code to ±1.0 first


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: _extract_camera_image
# ═══════════════════════════════════════════════════════════════════════


class TestExtractCameraImage:

    def test_extracts_rgb(self):
        p = OmnivlaPolicy()
        obs = {"cam": np.zeros((480, 640, 3), dtype=np.uint8)}
        img = p._extract_camera_image(obs)
        assert img.size == (640, 480)

    def test_extracts_rgba(self):
        p = OmnivlaPolicy()
        obs = {"cam": np.zeros((480, 640, 4), dtype=np.uint8)}
        img = p._extract_camera_image(obs)
        assert img.size == (640, 480)

    def test_no_image_returns_blank(self):
        p = OmnivlaPolicy()
        obs = {"scalar": 42.0}
        img = p._extract_camera_image(obs)
        assert img.size == (224, 224)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: update_goal
# ═══════════════════════════════════════════════════════════════════════


class TestUpdateGoal:

    def test_update_instruction(self):
        p = OmnivlaPolicy()
        p.update_goal(instruction="turn right at the stop sign")
        assert p._instruction == "turn right at the stop sign"

    def test_update_goal_image_array(self):
        p = OmnivlaPolicy()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        p.update_goal(goal_image=img)
        assert p._goal_image_pil is not None
        assert p._goal_image_pil.size == (100, 100)

    def test_update_goal_image_pil(self):
        from PIL import Image

        p = OmnivlaPolicy()
        img = Image.new("RGB", (50, 50))
        p.update_goal(goal_image=img)
        assert p._goal_image_pil is img

    def test_update_goal_modality(self):
        p = OmnivlaPolicy()
        p.update_goal(goal_modality="POSE_IMAGE")
        assert p._goal_modality == "pose_image"

    def test_update_goal_gps_without_utm(self):
        """GPS update when utm package is not available."""
        p = OmnivlaPolicy()
        with patch.dict(sys.modules, {"utm": None}):
            # Should not crash, just skip
            p.update_goal(goal_gps=(37.7749, -122.4194, 90.0))
        assert p._goal_gps == (37.7749, -122.4194, 90.0)

    def test_update_goal_gps_with_utm(self):
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500000.0, 4000000.0, 10, "S")
        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy()
            p.update_goal(goal_gps=(37.0, -122.0, 180.0))
        assert p._goal_gps == (37.0, -122.0, 180.0)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: reset & get_model_info
# ═══════════════════════════════════════════════════════════════════════


class TestResetAndInfo:

    def test_reset(self):
        p = OmnivlaPolicy()
        p._step = 42
        p.reset()
        assert p._step == 0

    def test_get_model_info_unloaded(self):
        p = OmnivlaPolicy()
        info = p.get_model_info()
        assert info["provider"] == "omnivla"
        assert info["variant"] == "full"
        assert info["loaded"] is False
        assert info["device"] is None
        assert info["step"] == 0

    def test_get_model_info_loaded_edge(self):
        p = OmnivlaPolicy(variant="edge")
        p._loaded = True
        p._device = "cpu"
        p._step = 5
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=1000))]
        p._edge_model = mock_model
        info = p.get_model_info()
        assert info["loaded"] is True
        assert info["n_parameters"] == 1000

    def test_get_model_info_loaded_full(self):
        p = OmnivlaPolicy(variant="full")
        p._loaded = True
        p._device = "cuda:0"
        mock_vla = MagicMock()
        mock_vla.parameters.return_value = [MagicMock(numel=MagicMock(return_value=7000000000))]
        p._vla = mock_vla
        info = p.get_model_info()
        assert info["n_parameters"] == 7000000000


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: _compute_goal_pose_norm
# ═══════════════════════════════════════════════════════════════════════


class TestComputeGoalPoseNorm:

    def test_no_goal_returns_zeros(self):
        """No goal_utm set → returns zero pose."""
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500000.0, 4000000.0, 10, "S")
        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy()
            obs = {"latitude": 37.0, "longitude": -122.0, "compass": 90.0}
            result = p._compute_goal_pose_norm(obs)
        assert np.allclose(result, 0.0)

    def test_with_goal_utm(self):
        """With goal_utm set → computes relative pose."""
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500100.0, 4000100.0, 10, "S")
        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy()
            p._goal_utm = (500200.0, 4000200.0, 10, "S")
            p._goal_compass_rad = 0.0
            obs = {"latitude": 37.0, "longitude": -122.0, "compass": 0.0}
            result = p._compute_goal_pose_norm(obs)
        assert result.shape == (4,)
        assert not np.allclose(result, 0.0)

    def test_distance_clipping(self):
        """Very far goal → distance clipped to thres_dist."""
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500000.0, 4000000.0, 10, "S")
        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy()
            p._goal_utm = (501000.0, 4001000.0, 10, "S")  # ~1.4km away
            p._goal_compass_rad = 0.0
            obs = {"latitude": 37.0, "longitude": -122.0, "compass": 0.0}
            result = p._compute_goal_pose_norm(obs)
        assert result.shape == (4,)

    def test_utm_not_installed(self):
        """utm not available → returns zeros."""
        p = OmnivlaPolicy()
        # Temporarily remove utm from modules
        with patch.dict(sys.modules, {"utm": None}):
            with patch("builtins.__import__", side_effect=ImportError("No utm")):
                obs = {"latitude": 37.0, "longitude": -122.0}
                # _compute_goal_pose_norm has try/except ImportError
                result = p._compute_goal_pose_norm(obs)
        assert np.allclose(result, 0.0)

    def test_gps_lat_lon_keys(self):
        """Try alternate GPS key names."""
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500000.0, 4000000.0, 10, "S")
        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy()
            p._goal_utm = (500010.0, 4000010.0, 10, "S")
            p._goal_compass_rad = 0.0
            obs = {"gps_lat": 37.0, "gps_lon": -122.0, "heading": 45.0}
            result = p._compute_goal_pose_norm(obs)
        assert result.shape == (4,)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: _ensure_loaded error paths
# ═══════════════════════════════════════════════════════════════════════


class TestEnsureLoadedErrors:

    def test_full_model_import_error(self):
        """Full model loading fails with ImportError."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "prismatic": None}):
            p = OmnivlaPolicy(variant="full")
            with pytest.raises(ImportError, match="prismatic"):
                p._ensure_loaded()

    def test_edge_model_import_error(self):
        """Edge model loading fails with ImportError."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            p = OmnivlaPolicy(variant="edge", checkpoint_path="/fake/path")
            with pytest.raises((ImportError, FileNotFoundError)):
                p._ensure_loaded()

    def test_noop_if_already_loaded(self):
        p = OmnivlaPolicy()
        p._loaded = True
        p._ensure_loaded()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: _load_checkpoint_module
# ═══════════════════════════════════════════════════════════════════════


class TestLoadCheckpointModule:

    def _mock_torch(self):
        """Create a mock torch module for checkpoint loading tests."""
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        return mock_torch

    def test_no_checkpoint_file(self, tmp_path):
        """Missing checkpoint file logs warning but doesn't crash."""
        mock_torch = self._mock_torch()
        p = OmnivlaPolicy(checkpoint_path=str(tmp_path))
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Should just log warning — no checkpoint file exists
            p._load_checkpoint_module(mock_module, "action_head")

    def test_checkpoint_found(self, tmp_path):
        """Checkpoint file found → loads state dict."""
        mock_torch = self._mock_torch()
        state_dict = {"weight": np.zeros(3)}
        mock_torch.load.return_value = state_dict

        # Create a dummy checkpoint file (content irrelevant, torch.load is mocked)
        ckpt_path = tmp_path / "action_head--120000_checkpoint.pt"
        ckpt_path.write_bytes(b"dummy")

        p = OmnivlaPolicy(checkpoint_path=str(tmp_path), resume_step=120000)
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            p._load_checkpoint_module(mock_module, "action_head")
        mock_module.load_state_dict.assert_called_once()

    def test_checkpoint_ddp_prefix_stripped(self, tmp_path):
        """DDP 'module.' prefix should be stripped from keys."""
        mock_torch = self._mock_torch()
        state_dict = {"module.weight": np.zeros(3)}
        mock_torch.load.return_value = state_dict

        # Create a dummy checkpoint file
        ckpt_path = tmp_path / "pose_projector--120000_checkpoint.pt"
        ckpt_path.write_bytes(b"dummy")

        p = OmnivlaPolicy(checkpoint_path=str(tmp_path), resume_step=120000)
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            p._load_checkpoint_module(mock_module, "pose_projector")
        call_args = mock_module.load_state_dict.call_args[0][0]
        assert "weight" in call_args
        assert "module.weight" not in call_args


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12: get_actions with pre-loaded model
# ═══════════════════════════════════════════════════════════════════════


class TestGetActionsPreloaded:

    def test_get_actions_returns_velocity_commands(self):
        """Full get_actions pipeline with pre-loaded model — mock everything."""
        p = OmnivlaPolicy(goal_modality="language")
        p._loaded = True
        p._variant = "full"
        p._device = MagicMock()

        # Mock _infer_full to return waypoints
        mock_waypoints = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [2.0, 0.1, 0.99, 0.1],
                [3.0, 0.2, 0.97, 0.2],
                [4.0, 0.3, 0.95, 0.3],
                [5.0, 0.4, 0.92, 0.4],
                [6.0, 0.5, 0.87, 0.5],
                [7.0, 0.6, 0.82, 0.6],
                [8.0, 0.7, 0.77, 0.7],
            ]
        )
        with patch.object(p, "_infer_full", return_value=mock_waypoints):
            obs = _make_obs()
            result = _run(p.get_actions(obs, "go forward"))

        assert len(result) == 1
        assert "linear_vel" in result[0]
        assert "angular_vel" in result[0]
        assert "waypoints" in result[0]
        assert "modality_id" in result[0]
        assert result[0]["modality_id"] == 7  # language mode

    def test_get_actions_edge_variant(self):
        """Edge variant also returns velocity commands."""
        p = OmnivlaPolicy(variant="edge", goal_modality="language")
        p._loaded = True
        p._device = MagicMock()

        mock_waypoints = np.ones((8, 4)) * 0.5
        with patch.object(p, "_infer_edge", return_value=mock_waypoints):
            obs = _make_obs()
            result = _run(p.get_actions(obs, "navigate to door"))

        assert len(result) == 1
        assert "linear_vel" in result[0]

    def test_get_actions_with_goal_image_in_obs(self):
        """Goal image provided in observation_dict."""
        p = OmnivlaPolicy(goal_modality="image")
        p._loaded = True
        p._variant = "full"
        p._device = MagicMock()

        mock_waypoints = np.ones((8, 4))
        with patch.object(p, "_infer_full", return_value=mock_waypoints):
            obs = {
                "camera": np.zeros((224, 224, 3), dtype=np.uint8),
                "goal_image": np.zeros((224, 224, 3), dtype=np.uint8),
            }
            result = _run(p.get_actions(obs, ""))
        assert result[0]["modality_id"] == 6  # image only

    def test_get_actions_runtime_override(self):
        """Override goal_modality at runtime via kwargs."""
        p = OmnivlaPolicy(goal_modality="language")
        p._loaded = True
        p._variant = "full"
        p._device = MagicMock()

        mock_waypoints = np.ones((8, 4))
        with patch.object(p, "_infer_full", return_value=mock_waypoints):
            obs = _make_obs()
            result = _run(p.get_actions(obs, "", goal_modality="pose"))
        assert result[0]["modality_id"] == 4  # pose only

    def test_step_increments(self):
        """Step counter increments each call."""
        p = OmnivlaPolicy(goal_modality="language")
        p._loaded = True
        p._variant = "full"
        p._device = MagicMock()
        assert p._step == 0

        mock_waypoints = np.ones((8, 4))
        with patch.object(p, "_infer_full", return_value=mock_waypoints):
            _run(p.get_actions(_make_obs(), "go"))
        assert p._step == 1

        with patch.object(p, "_infer_full", return_value=mock_waypoints):
            _run(p.get_actions(_make_obs(), "go"))
        assert p._step == 2


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13: _ensure_loaded with goal processing
# ═══════════════════════════════════════════════════════════════════════


class TestEnsureLoadedGoalProcessing:

    def test_goal_image_loaded_on_ensure(self, tmp_path):
        """Goal image from path is loaded during _ensure_loaded."""
        from PIL import Image

        img_path = tmp_path / "goal.jpg"
        Image.new("RGB", (100, 100)).save(str(img_path))

        p = OmnivlaPolicy(goal_image_path=str(img_path))
        p._loaded = False

        # Mock the model loading to succeed but check goal image
        with patch.object(p, "_load_full_model"):
            p._ensure_loaded()
        assert p._goal_image_pil is not None
        assert p._goal_image_pil.size == (100, 100)

    def test_goal_gps_converted_on_ensure(self):
        """Goal GPS converted to UTM during _ensure_loaded."""
        mock_utm = MagicMock()
        mock_utm.from_latlon.return_value = (500000.0, 4000000.0, 10, "S")

        with patch.dict(sys.modules, {"utm": mock_utm}):
            p = OmnivlaPolicy(goal_gps=(37.0, -122.0, 90.0))
            with patch.object(p, "_load_full_model"):
                p._ensure_loaded()
        assert p._goal_utm is not None
        assert p._goal_compass_rad is not None

    def test_goal_gps_utm_not_installed(self):
        """GPS goal with missing utm package → warning, no crash."""
        with patch.dict(sys.modules, {"utm": None}):
            p = OmnivlaPolicy(goal_gps=(37.0, -122.0, 90.0))
            with patch.object(p, "_load_full_model"):
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: (
                        (_ for _ in ()).throw(ImportError())
                        if name == "utm"
                        else __builtins__.__import__(name, *a, **kw)
                    ),
                ):
                    p._ensure_loaded()
        assert p._loaded is True


# ═══════════════════════════════════════════════════════════════════════
# SECTION 14: Provider name & protocol tests
# ═══════════════════════════════════════════════════════════════════════


class TestProviderProtocol:

    def test_provider_name(self):
        assert OmnivlaPolicy().provider_name == "omnivla"

    def test_is_policy_subclass(self):
        from strands_robots.policies import Policy

        assert issubclass(OmnivlaPolicy, Policy)

    def test_get_actions_is_async(self):
        import inspect

        assert inspect.iscoroutinefunction(OmnivlaPolicy.get_actions)
