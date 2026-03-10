#!/usr/bin/env python3
"""Tests for strands_robots.tools.stereo_depth tool.

Mocks the strands SDK and StereoDepthPipeline to test the tool's logic
(argument validation, config building, intrinsic matrix construction,
output file saving, error handling) without GPU or model weights.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Pre-mock strands before import
# ---------------------------------------------------------------------------

_mock_strands = types.ModuleType("strands")
_mock_strands.tool = lambda f: f  # type: ignore[attr-defined]

with patch.dict(sys.modules, {"strands": _mock_strands}):
    from strands_robots.tools.stereo_depth import stereo_depth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stereo_result(
    height: int = 480,
    width: int = 640,
    with_depth: bool = True,
    with_point_cloud: bool = True,
    with_vis: bool = True,
):
    """Build a mock StereoResult-like object."""
    from strands_robots.stereo import StereoResult

    disp = np.random.rand(height, width).astype(np.float32) * 100
    depth = np.random.rand(height, width).astype(np.float32) * 5 + 0.1 if with_depth else None
    pc = np.random.rand(height, width, 3).astype(np.float32) if with_point_cloud else None
    vis = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) if with_vis else None
    return StereoResult(
        disparity=disp,
        depth=depth,
        point_cloud=pc,
        disparity_vis=vis,
        metadata={"inference_time": 0.042, "total_time": 0.123},
    )


def _fake_images(tmp_path):
    """Create fake image files and return (left_path, right_path)."""
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    left.write_bytes(b"fake")
    right.write_bytes(b"fake")
    return str(left), str(right)


# The tool's lazy imports resolve StereoConfig and StereoDepthPipeline from
# strands_robots.stereo at call time. We patch them at that module path.
_STEREO_MOD = "strands_robots.stereo"


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test file-not-found and bad input handling."""

    def test_left_image_not_found(self, tmp_path):
        right = tmp_path / "right.png"
        right.write_bytes(b"fake")
        result = stereo_depth(
            left_image=str(tmp_path / "nonexistent.png"),
            right_image=str(right),
        )
        assert "error" in result
        assert "Left image not found" in result["error"]

    def test_right_image_not_found(self, tmp_path):
        left = tmp_path / "left.png"
        left.write_bytes(b"fake")
        result = stereo_depth(
            left_image=str(left),
            right_image=str(tmp_path / "nonexistent.png"),
        )
        assert "error" in result
        assert "Right image not found" in result["error"]

    def test_both_images_not_found(self, tmp_path):
        """Left image is checked first."""
        result = stereo_depth(
            left_image=str(tmp_path / "left.png"),
            right_image=str(tmp_path / "right.png"),
        )
        assert "error" in result
        assert "Left image not found" in result["error"]


# ---------------------------------------------------------------------------
# Tests: Intrinsic matrix construction
# ---------------------------------------------------------------------------


class TestIntrinsicMatrix:
    """Test intrinsic matrix building from focal_length/cx/cy."""

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_intrinsic_from_focal_length(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        MockPipeline._load_image = MagicMock(return_value=mock_img)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result()
        MockPipeline.return_value = mock_instance

        stereo_depth(left_image=left, right_image=right, focal_length=600.0)

        call_kwargs = mock_instance.estimate_depth.call_args
        K = call_kwargs.kwargs.get("intrinsic_matrix")
        assert K is not None
        assert K.shape == (3, 3)
        assert K[0, 0] == 600.0
        assert K[1, 1] == 600.0
        assert K[0, 2] == 320.0  # w/2
        assert K[1, 2] == 240.0  # h/2

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_intrinsic_with_custom_principal_point(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        MockPipeline._load_image = MagicMock(return_value=mock_img)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result()
        MockPipeline.return_value = mock_instance

        stereo_depth(
            left_image=left,
            right_image=right,
            focal_length=500.0,
            cx=300.0,
            cy=200.0,
        )

        K = mock_instance.estimate_depth.call_args.kwargs["intrinsic_matrix"]
        assert K[0, 2] == 300.0
        assert K[1, 2] == 200.0

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_no_intrinsic_without_focal_length(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(with_depth=False, with_point_cloud=False)
        MockPipeline.return_value = mock_instance

        stereo_depth(left_image=left, right_image=right)

        K = mock_instance.estimate_depth.call_args.kwargs.get("intrinsic_matrix")
        assert K is None


# ---------------------------------------------------------------------------
# Tests: Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Test the output dictionary structure."""

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_success_output_keys(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(with_vis=False, with_point_cloud=False)
        MockPipeline.return_value = mock_instance

        result = stereo_depth(left_image=left, right_image=right)

        assert result["status"] == "success"
        for key in (
            "height",
            "width",
            "median_depth",
            "valid_ratio",
            "inference_time",
            "total_time",
            "model_variant",
            "valid_iters",
            "scale",
        ):
            assert key in result, f"Missing key: {key}"

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_output_dimensions_correct(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(height=720, width=1280)
        MockPipeline.return_value = mock_instance

        result = stereo_depth(left_image=left, right_image=right)

        assert result["height"] == 720
        assert result["width"] == 1280


# ---------------------------------------------------------------------------
# Tests: Output file saving
# ---------------------------------------------------------------------------


class TestOutputSaving:
    """Test depth/vis/point_cloud file saving."""

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_depth_npy_saved(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "output"

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result()
        MockPipeline.return_value = mock_instance

        result = stereo_depth(
            left_image=left,
            right_image=right,
            output_dir=str(out_dir),
        )

        assert "depth_npy_path" in result
        assert os.path.isfile(result["depth_npy_path"])
        loaded = np.load(result["depth_npy_path"])
        assert loaded.shape == (480, 640)

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_point_cloud_npy_saved(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "output"

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result()
        MockPipeline.return_value = mock_instance

        result = stereo_depth(
            left_image=left,
            right_image=right,
            output_dir=str(out_dir),
        )

        assert "point_cloud_npy_path" in result
        assert os.path.isfile(result["point_cloud_npy_path"])
        loaded = np.load(result["point_cloud_npy_path"])
        assert loaded.shape == (480, 640, 3)

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_no_depth_no_file(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "output"

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_depth=False,
            with_point_cloud=False,
            with_vis=False,
        )
        MockPipeline.return_value = mock_instance

        result = stereo_depth(
            left_image=left,
            right_image=right,
            output_dir=str(out_dir),
        )

        assert "depth_npy_path" not in result
        assert "point_cloud_npy_path" not in result

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_vis_with_imageio(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "output"

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(with_vis=True)
        MockPipeline.return_value = mock_instance

        mock_imageio_v2 = MagicMock()
        mock_imageio_parent = MagicMock()
        mock_imageio_parent.v2 = mock_imageio_v2
        with patch.dict(
            sys.modules,
            {"imageio": mock_imageio_parent, "imageio.v2": mock_imageio_v2},
        ):
            result = stereo_depth(
                left_image=left,
                right_image=right,
                output_dir=str(out_dir),
                visualize=True,
            )

        assert "disparity_vis_path" in result
        mock_imageio_v2.imwrite.assert_called_once()

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_vis_no_libs_warning(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "output"

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(with_vis=True)
        MockPipeline.return_value = mock_instance

        with patch.dict(
            sys.modules,
            {"imageio": None, "imageio.v2": None, "cv2": None},
        ):
            result = stereo_depth(
                left_image=left,
                right_image=right,
                output_dir=str(out_dir),
                visualize=True,
            )

        assert result.get("disparity_vis_path") is None
        assert "warning" in result

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_output_dir_created_if_missing(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)
        out_dir = tmp_path / "deep" / "nested" / "output"
        assert not out_dir.exists()

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(with_vis=False)
        MockPipeline.return_value = mock_instance

        result = stereo_depth(
            left_image=left,
            right_image=right,
            output_dir=str(out_dir),
        )

        assert out_dir.exists()
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Tests: Default output_dir
# ---------------------------------------------------------------------------


class TestDefaultOutputDir:

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_default_output_dir_is_left_image_dir(
        self,
        MockConfig,
        MockPipeline,
        tmp_path,
    ):
        subdir = tmp_path / "images"
        subdir.mkdir()
        left = subdir / "left.png"
        right = subdir / "right.png"
        left.write_bytes(b"fake")
        right.write_bytes(b"fake")

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result()
        MockPipeline.return_value = mock_instance

        result = stereo_depth(left_image=str(left), right_image=str(right))

        assert result["status"] == "success"
        if "depth_npy_path" in result:
            assert str(subdir) in result["depth_npy_path"]


# ---------------------------------------------------------------------------
# Tests: Config passthrough
# ---------------------------------------------------------------------------


class TestConfigPassthrough:

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_model_variant_passed(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
            with_point_cloud=False,
            with_depth=False,
        )
        MockPipeline.return_value = mock_instance

        stereo_depth(
            left_image=left,
            right_image=right,
            model_variant="20-30-48",
            valid_iters=4,
            scale=0.5,
        )

        MockConfig.assert_called_once()
        call_str = str(MockConfig.call_args)
        assert "20-30-48" in call_str

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_camera_preset_passed(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
            with_point_cloud=False,
            with_depth=False,
        )
        MockPipeline.return_value = mock_instance

        stereo_depth(
            left_image=left,
            right_image=right,
            camera="realsense_d435",
        )

        MockConfig.assert_called_once()
        assert "realsense_d435" in str(MockConfig.call_args)

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_output_model_variant_in_result(
        self,
        MockConfig,
        MockPipeline,
        tmp_path,
    ):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
            with_point_cloud=False,
            with_depth=False,
        )
        MockPipeline.return_value = mock_instance

        result = stereo_depth(
            left_image=left,
            right_image=right,
            model_variant="20-26-39",
            valid_iters=12,
            scale=0.75,
        )

        assert result["model_variant"] == "20-26-39"
        assert result["valid_iters"] == 12
        assert result["scale"] == 0.75


# ---------------------------------------------------------------------------
# Tests: Baseline/visualize passthrough
# ---------------------------------------------------------------------------


class TestEstimateDepthParams:

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_baseline_passed_to_pipeline(
        self,
        MockConfig,
        MockPipeline,
        tmp_path,
    ):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
        )
        MockPipeline.return_value = mock_instance

        stereo_depth(left_image=left, right_image=right, baseline=0.12)

        call_kwargs = mock_instance.estimate_depth.call_args.kwargs
        assert call_kwargs.get("baseline") == 0.12

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_visualize_false_passed_to_pipeline(
        self,
        MockConfig,
        MockPipeline,
        tmp_path,
    ):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
        )
        MockPipeline.return_value = mock_instance

        stereo_depth(left_image=left, right_image=right, visualize=False)

        call_kwargs = mock_instance.estimate_depth.call_args.kwargs
        assert call_kwargs.get("visualize") is False


# ---------------------------------------------------------------------------
# Tests: Numeric output
# ---------------------------------------------------------------------------


class TestNumericOutput:

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_valid_ratio_rounded(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
        )
        MockPipeline.return_value = mock_instance

        result = stereo_depth(left_image=left, right_image=right)

        vr = result["valid_ratio"]
        assert isinstance(vr, float)
        assert 0.0 <= vr <= 1.0

    @patch(f"{_STEREO_MOD}.StereoDepthPipeline")
    @patch(f"{_STEREO_MOD}.StereoConfig")
    def test_inference_time_rounded(self, MockConfig, MockPipeline, tmp_path):
        left, right = _fake_images(tmp_path)

        mock_instance = MagicMock()
        mock_instance.estimate_depth.return_value = _make_stereo_result(
            with_vis=False,
        )
        MockPipeline.return_value = mock_instance

        result = stereo_depth(left_image=left, right_image=right)

        assert result["inference_time"] == 0.042
        assert result["total_time"] == 0.123
