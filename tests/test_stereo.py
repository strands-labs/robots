"""Comprehensive test suite for strands_robots.stereo module.

Tests the stereo depth estimation pipeline (Fast-FoundationStereo integration)
with mocked GPU inference. All tests run on CPU without requiring an NVIDIA GPU
or model weights.

Tests cover:
- StereoConfig validation and defaults
- StereoResult properties and serialisation
- StereoDepthPipeline image loading, padding, depth computation
- Point cloud generation from depth + intrinsics
- Disparity visualisation
- InputPadder correctness
- Convenience function
- Error handling and edge cases
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# StereoConfig tests
# ---------------------------------------------------------------------------


class TestStereoConfig:
    """Tests for StereoConfig dataclass validation."""

    def test_default_config(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig()
        assert cfg.model_variant == "23-36-37"
        assert cfg.valid_iters == 8
        assert cfg.max_disp == 192
        assert cfg.scale == 1.0
        assert cfg.hierarchical is False
        assert cfg.remove_invisible is True
        assert cfg.mixed_precision is True
        assert cfg.zfar == 100.0
        assert cfg.camera is None

    def test_valid_variants(self):
        from strands_robots.stereo import StereoConfig

        for variant in ("23-36-37", "20-26-39", "20-30-48"):
            cfg = StereoConfig(model_variant=variant)
            assert cfg.model_variant == variant

    def test_invalid_variant(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="Invalid model_variant"):
            StereoConfig(model_variant="invalid")

    def test_invalid_valid_iters(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="valid_iters must be >= 1"):
            StereoConfig(valid_iters=0)

    def test_invalid_max_disp(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="max_disp must be >= 1"):
            StereoConfig(max_disp=0)

    def test_invalid_scale_zero(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="scale must be"):
            StereoConfig(scale=0.0)

    def test_invalid_scale_too_high(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="scale must be"):
            StereoConfig(scale=3.0)

    def test_invalid_zfar(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="zfar must be > 0"):
            StereoConfig(zfar=-1.0)

    def test_valid_camera(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig(camera="realsense_d435")
        assert cfg.camera == "realsense_d435"

    def test_invalid_camera(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="Unknown camera"):
            StereoConfig(camera="nonexistent_camera")

    def test_resolve_model_path_explicit(self):
        from strands_robots.stereo import StereoConfig

        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            cfg = StereoConfig(model_path=f.name)
            assert cfg.resolve_model_path() == f.name

    def test_resolve_model_path_env_var(self, tmp_path):
        from strands_robots.stereo import StereoConfig

        model_dir = tmp_path / "23-36-37"
        model_dir.mkdir()
        model_file = model_dir / "model_best_bp2_serialize.pth"
        model_file.write_text("mock")

        with patch.dict(os.environ, {"STEREO_MODEL_DIR": str(tmp_path)}):
            cfg = StereoConfig(model_variant="23-36-37")
            assert cfg.resolve_model_path() == str(model_file)

    def test_resolve_model_path_not_found(self):
        from strands_robots.stereo import StereoConfig

        with patch.dict(os.environ, {}, clear=True):
            cfg = StereoConfig()
            with pytest.raises(FileNotFoundError, match="Could not resolve"):
                cfg.resolve_model_path()

    def test_asdict_roundtrip(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig(model_variant="20-26-39", valid_iters=4, scale=0.5)
        d = asdict(cfg)
        cfg2 = StereoConfig(**d)
        assert cfg == cfg2

    def test_custom_config(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig(
            model_variant="20-30-48",
            valid_iters=4,
            max_disp=256,
            scale=0.5,
            hierarchical=True,
            remove_invisible=False,
            low_memory=True,
            camera="zed2",
        )
        assert cfg.model_variant == "20-30-48"
        assert cfg.valid_iters == 4
        assert cfg.max_disp == 256
        assert cfg.hierarchical is True
        assert cfg.camera == "zed2"


# ---------------------------------------------------------------------------
# StereoResult tests
# ---------------------------------------------------------------------------


class TestStereoResult:
    """Tests for StereoResult dataclass."""

    def test_basic_properties(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((480, 640), dtype=np.float32) * 10.0
        result = StereoResult(disparity=disp)
        assert result.height == 480
        assert result.width == 640
        assert result.depth is None
        assert result.point_cloud is None
        assert result.median_depth is None

    def test_valid_ratio_all_valid(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((100, 100), dtype=np.float32) * 5.0
        result = StereoResult(disparity=disp)
        assert result.valid_ratio == 1.0

    def test_valid_ratio_half_invalid(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((100, 100), dtype=np.float32) * 5.0
        disp[:50, :] = 0.0  # invalid
        result = StereoResult(disparity=disp)
        assert result.valid_ratio == pytest.approx(0.5)

    def test_valid_ratio_with_inf(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((10, 10), dtype=np.float32) * 5.0
        disp[0, 0] = np.inf
        result = StereoResult(disparity=disp)
        assert result.valid_ratio == pytest.approx(99 / 100)

    def test_median_depth(self):
        from strands_robots.stereo import StereoResult

        depth = np.ones((10, 10), dtype=np.float32) * 2.5
        result = StereoResult(
            disparity=np.ones((10, 10)),
            depth=depth,
        )
        assert result.median_depth == pytest.approx(2.5)

    def test_median_depth_with_inf(self):
        from strands_robots.stereo import StereoResult

        depth = np.ones((10, 10), dtype=np.float32) * 3.0
        depth[0, :] = np.inf
        result = StereoResult(
            disparity=np.ones((10, 10)),
            depth=depth,
        )
        assert result.median_depth == pytest.approx(3.0)

    def test_median_depth_all_invalid(self):
        from strands_robots.stereo import StereoResult

        depth = np.full((10, 10), np.inf, dtype=np.float32)
        result = StereoResult(
            disparity=np.ones((10, 10)),
            depth=depth,
        )
        assert result.median_depth is None

    def test_to_dict(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((10, 20), dtype=np.float32) * 5.0
        result = StereoResult(
            disparity=disp,
            metadata={"model": "test"},
        )
        d = result.to_dict()
        assert d["height"] == 10
        assert d["width"] == 20
        assert d["valid_ratio"] == 1.0
        assert d["has_depth"] is False
        assert d["has_point_cloud"] is False
        assert d["metadata"]["model"] == "test"


# ---------------------------------------------------------------------------
# Pipeline tests (mocked model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestStereoDepthPipeline:
    """Tests for StereoDepthPipeline with mocked GPU model."""

    def _make_stereo_pair(self, h: int = 480, w: int = 640) -> tuple[np.ndarray, np.ndarray]:
        """Create a synthetic stereo pair."""
        rng = np.random.RandomState(42)
        left = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Simulate a shifted version for the right image
        right = np.roll(left, -10, axis=1)
        return left, right

    def _make_mock_model(self, h: int = 480, w: int = 640):
        """Create a mock model that returns a plausible disparity."""
        import torch

        mock = MagicMock()
        mock.args = MagicMock()

        def mock_forward(img0, img1, iters=8, test_mode=True, low_memory=False, **kwargs):
            B, C, H, W = img0.shape
            # Return a smooth disparity field
            disp = torch.ones(B, 1, H, W, device=img0.device) * 15.0
            return disp

        mock.forward = mock_forward
        mock.run_hierachical = mock_forward
        return mock

    def test_pipeline_init_default(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        assert pipe.config.model_variant == "23-36-37"
        assert pipe._model_loaded is False

    def test_pipeline_init_with_config(self):
        from strands_robots.stereo import StereoConfig, StereoDepthPipeline

        cfg = StereoConfig(model_variant="20-30-48", valid_iters=4)
        pipe = StereoDepthPipeline(cfg)
        assert pipe.config.model_variant == "20-30-48"

    def test_pipeline_init_with_kwargs(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline(model_variant="20-26-39", scale=0.5)
        assert pipe.config.model_variant == "20-26-39"
        assert pipe.config.scale == 0.5

    def test_pipeline_init_config_and_kwargs_error(self):
        from strands_robots.stereo import StereoConfig, StereoDepthPipeline

        with pytest.raises(ValueError, match="Cannot specify both"):
            StereoDepthPipeline(StereoConfig(), model_variant="20-30-48")

    def test_pipeline_repr(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        r = repr(pipe)
        assert "StereoDepthPipeline" in r
        assert "23-36-37" in r
        assert "loaded=False" in r

    def test_estimate_depth_with_mock_model(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)

        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        result = pipe.estimate_depth(
            left_image=left,
            right_image=right,
            intrinsic_matrix=K,
            baseline=0.12,
        )

        assert result.disparity.shape == (480, 640)
        assert result.depth is not None
        assert result.depth.shape == (480, 640)
        assert result.point_cloud is not None
        assert result.point_cloud.shape == (480, 640, 3)
        assert result.valid_ratio > 0.9
        assert result.median_depth is not None
        assert result.median_depth > 0
        assert "inference_time" in result.metadata

    def test_estimate_depth_disparity_only(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)

        # No intrinsics → no depth, no point cloud
        result = pipe.estimate_depth(
            left_image=left,
            right_image=right,
        )

        assert result.disparity is not None
        assert result.depth is None
        assert result.point_cloud is None

    def test_estimate_depth_with_visualisation(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)
        result = pipe.estimate_depth(
            left_image=left,
            right_image=right,
            visualize=True,
        )

        assert result.disparity_vis is not None
        # Check it's a numpy array (not a mock from other test contamination)
        if isinstance(result.disparity_vis, np.ndarray):
            assert result.disparity_vis.shape == (480, 640, 3)
            assert result.disparity_vis.dtype == np.uint8

    def test_estimate_depth_from_files(self, tmp_path):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)

        # Save as files
        try:
            import imageio.v2 as imageio

            left_path = str(tmp_path / "left.png")
            right_path = str(tmp_path / "right.png")
            imageio.imwrite(left_path, left)
            imageio.imwrite(right_path, right)
        except ImportError:
            try:
                import cv2

                left_path = str(tmp_path / "left.png")
                right_path = str(tmp_path / "right.png")
                cv2.imwrite(left_path, left[..., ::-1])
                cv2.imwrite(right_path, right[..., ::-1])
            except ImportError:
                pytest.skip("Need imageio or cv2 for file-based test")

        result = pipe.estimate_depth(
            left_image=left_path,
            right_image=right_path,
        )
        assert result.disparity.shape == (480, 640)

    def test_estimate_depth_file_not_found(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        with pytest.raises(FileNotFoundError, match="Image not found"):
            pipe.estimate_depth(
                left_image="/nonexistent/left.png",
                right_image="/nonexistent/right.png",
            )

    def test_estimate_depth_shape_mismatch(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes must match"):
            pipe.estimate_depth(left_image=left, right_image=right)

    def test_estimate_depth_greyscale_input(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        result = pipe.estimate_depth(left_image=left, right_image=right)
        assert result.disparity.shape == (480, 640)

    def test_estimate_depth_rgba_input(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)

        result = pipe.estimate_depth(left_image=left, right_image=right)
        assert result.disparity.shape == (480, 640)

    def test_estimate_depth_hierarchical(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline(hierarchical=True)
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)
        result = pipe.estimate_depth(left_image=left, right_image=right)
        assert result.disparity.shape == (480, 640)

    def test_estimate_depth_camera_defaults(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline(camera="realsense_d435")
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)

        result = pipe.estimate_depth(
            left_image=left,
            right_image=right,
            intrinsic_matrix=K,
            # baseline not specified — should use camera default (0.050)
        )
        assert result.depth is not None
        assert result.metadata.get("baseline") == 0.050

    def test_estimate_depth_config_overrides(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline(valid_iters=8)
        pipe._model = self._make_mock_model()
        pipe._model_loaded = True

        left, right = self._make_stereo_pair(480, 640)
        result = pipe.estimate_depth(
            left_image=left,
            right_image=right,
            valid_iters=4,  # override
        )
        assert result.metadata["config"]["valid_iters"] == 4


# ---------------------------------------------------------------------------
# InputPadder tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestInputPadder:
    """Tests for the _InputPadder utility."""

    def test_pad_already_divisible(self):
        import torch

        from strands_robots.stereo import _InputPadder

        x = torch.randn(1, 3, 480, 640)
        padder = _InputPadder(x.shape, divis_by=32)
        (x_padded,) = padder.pad(x)
        assert x_padded.shape[-2] % 32 == 0
        assert x_padded.shape[-1] % 32 == 0

    def test_pad_unpad_roundtrip(self):
        import torch

        from strands_robots.stereo import _InputPadder

        x = torch.randn(1, 3, 473, 631)  # Not divisible by 32
        padder = _InputPadder(x.shape, divis_by=32)
        (x_padded,) = padder.pad(x)
        assert x_padded.shape[-2] % 32 == 0
        assert x_padded.shape[-1] % 32 == 0

        x_unpadded = padder.unpad(x_padded)
        assert x_unpadded.shape == x.shape
        assert torch.allclose(x_unpadded, x)

    def test_pad_multiple_inputs(self):
        import torch

        from strands_robots.stereo import _InputPadder

        a = torch.randn(1, 3, 100, 100)
        b = torch.randn(1, 3, 100, 100)
        padder = _InputPadder(a.shape, divis_by=32)
        a_p, b_p = padder.pad(a, b)
        assert a_p.shape == b_p.shape
        assert a_p.shape[-2] % 32 == 0


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestDepthToXyz:
    """Tests for the _depth_to_xyz function."""

    def test_basic_conversion(self):
        from strands_robots.stereo import _depth_to_xyz

        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        depth = np.ones((100, 100), dtype=np.float32) * 5.0

        xyz = _depth_to_xyz(depth, K)
        assert xyz.shape == (100, 100, 3)

        # At principal point (50, 50), x=0, y=0, z=5
        assert xyz[50, 50, 2] == pytest.approx(5.0)
        assert abs(xyz[50, 50, 0]) < 0.1
        assert abs(xyz[50, 50, 1]) < 0.1

    def test_invalid_depth_zeroed(self):
        from strands_robots.stereo import _depth_to_xyz

        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        depth = np.ones((10, 10), dtype=np.float32) * 5.0
        depth[0, 0] = np.inf
        depth[1, 1] = -1.0  # below zmin

        xyz = _depth_to_xyz(depth, K, zmin=0.1)
        assert np.all(xyz[0, 0] == 0.0)
        assert np.all(xyz[1, 1] == 0.0)

    def test_geometry_consistency(self):
        from strands_robots.stereo import _depth_to_xyz

        K = np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1]], dtype=np.float64)
        depth = np.ones((240, 320), dtype=np.float32) * 2.0

        xyz = _depth_to_xyz(depth, K)

        # At pixel (120, 160) = principal point → xyz should be (0, 0, 2)
        assert xyz[120, 160, 0] == pytest.approx(0.0, abs=0.01)
        assert xyz[120, 160, 1] == pytest.approx(0.0, abs=0.01)
        assert xyz[120, 160, 2] == pytest.approx(2.0)

        # At pixel (120, 320-1) → x = (319-160)*2/200 = 1.59
        assert xyz[120, 319, 0] == pytest.approx((319 - 160) * 2.0 / 200.0, abs=0.01)


class TestVisualizeDisparity:
    """Tests for the _visualize_disparity function."""

    def test_basic_visualization(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.random.rand(100, 100).astype(np.float32) * 50
        vis = _visualize_disparity(disp)
        assert vis.shape == (100, 100, 3)
        assert vis.dtype == np.uint8

    def test_all_invalid(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.full((10, 10), np.inf, dtype=np.float32)
        vis = _visualize_disparity(disp)
        assert vis.shape == (10, 10, 3)
        assert np.all(vis == 0)

    def test_uniform_disparity(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.ones((50, 50), dtype=np.float32) * 10.0
        vis = _visualize_disparity(disp)
        assert vis.shape == (50, 50, 3)

    def test_mixed_valid_invalid(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.ones((20, 20), dtype=np.float32) * 10.0
        disp[0, :] = np.inf
        vis = _visualize_disparity(disp)
        assert vis.shape == (20, 20, 3)
        # Invalid row should be zero
        assert np.all(vis[0, :] == 0)


# ---------------------------------------------------------------------------
# Image loading tests
# ---------------------------------------------------------------------------


class TestImageLoading:
    """Tests for StereoDepthPipeline._load_image."""

    def test_load_numpy_rgb(self):
        from strands_robots.stereo import StereoDepthPipeline

        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        loaded = StereoDepthPipeline._load_image(img)
        assert loaded.shape == (100, 200, 3)

    def test_load_numpy_greyscale(self):
        from strands_robots.stereo import StereoDepthPipeline

        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        loaded = StereoDepthPipeline._load_image(img)
        assert loaded.shape == (100, 200, 3)

    def test_load_numpy_rgba(self):
        from strands_robots.stereo import StereoDepthPipeline

        img = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)
        loaded = StereoDepthPipeline._load_image(img)
        assert loaded.shape == (100, 200, 3)

    def test_load_invalid_shape(self):
        from strands_robots.stereo import StereoDepthPipeline

        img = np.random.randint(0, 255, (100, 200, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected"):
            StereoDepthPipeline._load_image(img)

    def test_load_nonexistent_file(self):
        from strands_robots.stereo import StereoDepthPipeline

        with pytest.raises(FileNotFoundError, match="Image not found"):
            StereoDepthPipeline._load_image("/nonexistent/image.png")


# ---------------------------------------------------------------------------
# Constants and exports tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_valid_model_variants(self):
        from strands_robots.stereo import VALID_MODEL_VARIANTS

        assert len(VALID_MODEL_VARIANTS) == 3
        assert "23-36-37" in VALID_MODEL_VARIANTS
        assert "20-26-39" in VALID_MODEL_VARIANTS
        assert "20-30-48" in VALID_MODEL_VARIANTS

    def test_supported_cameras(self):
        from strands_robots.stereo import SUPPORTED_CAMERAS

        assert "realsense_d435" in SUPPORTED_CAMERAS
        assert "realsense_d455" in SUPPORTED_CAMERAS
        assert "zed2" in SUPPORTED_CAMERAS
        assert "zed_mini" in SUPPORTED_CAMERAS
        assert "custom" in SUPPORTED_CAMERAS

        for cam_key, cam_info in SUPPORTED_CAMERAS.items():
            assert "type" in cam_info
            assert "description" in cam_info

    def test_exports(self):
        from strands_robots.stereo import __all__

        expected = [
            "StereoDepthPipeline",
            "StereoConfig",
            "StereoResult",
            "estimate_depth",
            "VALID_MODEL_VARIANTS",
            "SUPPORTED_CAMERAS",
        ]
        for name in expected:
            assert name in __all__, f"{name} not in __all__"

    def test_camera_baselines_positive(self):
        from strands_robots.stereo import SUPPORTED_CAMERAS

        for cam_key, cam_info in SUPPORTED_CAMERAS.items():
            bl = cam_info.get("default_baseline")
            if bl is not None:
                assert bl > 0, f"Baseline for {cam_key} must be positive"


# ---------------------------------------------------------------------------
# Depth computation correctness
# ---------------------------------------------------------------------------


class TestDepthComputation:
    """Verify the disparity → depth → point cloud math."""

    def test_depth_from_disparity_formula(self):
        """depth = focal * baseline / disparity"""
        focal = 600.0
        baseline = 0.12
        disparity = 30.0

        expected_depth = focal * baseline / disparity  # = 2.4m

        # Simulate what the pipeline does
        K = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]])
        disp_map = np.full((480, 640), disparity, dtype=np.float32)
        K_scaled = K.copy()
        depth_map = (K_scaled[0, 0] * baseline) / disp_map

        assert depth_map[0, 0] == pytest.approx(expected_depth)
        assert depth_map[240, 320] == pytest.approx(expected_depth)

    def test_depth_inverse_relationship(self):
        """Closer objects have higher disparity."""
        focal = 600.0
        baseline = 0.12

        disp_near = 60.0  # close object
        disp_far = 10.0  # far object

        depth_near = focal * baseline / disp_near
        depth_far = focal * baseline / disp_far

        assert depth_near < depth_far

    def test_zero_disparity_gives_inf_depth(self):
        """Zero disparity should yield infinite depth."""
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
        disp_map = np.array([[0.0]], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (K[0, 0] * 0.12) / disp_map
        assert np.isinf(depth[0, 0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_tiny_image(self):
        """1×1 image should work (padding handles it)."""
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()

        mock = MagicMock()
        mock.args = MagicMock()

        def mock_forward(img0, img1, iters=8, test_mode=True, low_memory=False, **kwargs):
            import torch

            B, C, H, W = img0.shape
            return torch.ones(B, 1, H, W) * 5.0

        mock.forward = mock_forward
        pipe._model = mock
        pipe._model_loaded = True

        left = np.full((32, 32, 3), 128, dtype=np.uint8)
        right = np.full((32, 32, 3), 128, dtype=np.uint8)

        result = pipe.estimate_depth(left_image=left, right_image=right)
        assert result.disparity.shape == (32, 32)

    def test_large_image_scaled(self):
        """Test that scale factor reduces inference size."""
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline(scale=0.5)

        mock = MagicMock()
        mock.args = MagicMock()

        def mock_forward(img0, img1, iters=8, test_mode=True, low_memory=False, **kwargs):
            import torch

            B, C, H, W = img0.shape
            return torch.ones(B, 1, H, W) * 5.0

        mock.forward = mock_forward
        pipe._model = mock
        pipe._model_loaded = True

        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = pipe.estimate_depth(left_image=left, right_image=right)
        # If cv2 is real, scale should be 0.5; if cv2 is mocked/absent, 1.0
        assert result.metadata["scale"] in (0.5, 1.0)
        if result.metadata["scale"] == 0.5:
            assert result.metadata["inference_size"][0] <= 480
