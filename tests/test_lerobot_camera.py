"""Tests for strands_robots/tools/lerobot_camera.py — LeRobot-based camera tool.

Tests cover:
- Frame to image content conversion
- Camera discovery (OpenCV + RealSense)
- Camera listing and details
- Single image capture
- Batch capture (parallel)
- Video recording
- Live preview
- Performance testing
- Configuration save/load
- Error handling for all actions
- Unknown camera types
"""

import importlib

# --- Mock all heavy deps before importing the module ---
# Mock cv2
import importlib.machinery as _im_fix
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
_mock_cv2.__version__ = "4.10.0"
_mock_cv2.COLOR_RGB2BGR = 4
_mock_cv2.COLOR_BGR2RGB = 4
_mock_cv2.CAP_ANY = 0
_mock_cv2.CAP_V4L2 = 200
_mock_cv2.CAP_MSMF = 1400
_mock_cv2.CAP_AVFOUNDATION = 1200
_mock_cv2.VideoWriter_fourcc = MagicMock(return_value=0x7634706D)
_mock_cv2.FONT_HERSHEY_SIMPLEX = 0
_mock_cv2.cvtColor = MagicMock(side_effect=lambda frame, code: frame)  # passthrough
_mock_cv2.imencode = MagicMock(return_value=(True, MagicMock(tobytes=lambda: b"\xff\xd8JPEG")))
_mock_cv2.imwrite = MagicMock(return_value=True)

# Mock lerobot camera modules
_mock_camera = MagicMock()
_mock_opencv = MagicMock()
_mock_opencv_config = MagicMock()


# Create proper enums
class MockColorMode:
    RGB = "RGB"
    BGR = "BGR"


class MockCv2Rotation:
    NO_ROTATION = None
    ROTATE_90 = 0
    ROTATE_180 = 1
    ROTATE_270 = 2


_mock_opencv_config.ColorMode = MockColorMode
_mock_opencv_config.Cv2Rotation = MockCv2Rotation
_mock_opencv_config.OpenCVCameraConfig = MagicMock()

_mock_opencv.OpenCVCamera = MagicMock()
_mock_opencv.OpenCVCamera.find_cameras = MagicMock(return_value=[])

# Mock strands
_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f  # passthrough decorator

# Mock realsense (not available)
_mock_realsense_camera = MagicMock()
_mock_realsense_config = MagicMock()


@pytest.fixture(autouse=True)
def _mock_imports(monkeypatch):
    """Mock all heavy imports before loading the camera module."""
    monkeypatch.setitem(sys.modules, "cv2", _mock_cv2)
    monkeypatch.setitem(sys.modules, "lerobot", MagicMock())
    monkeypatch.setitem(sys.modules, "lerobot.cameras", MagicMock())
    monkeypatch.setitem(sys.modules, "lerobot.cameras.camera", _mock_camera)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.opencv", _mock_opencv)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.opencv.configuration_opencv", _mock_opencv_config)
    # Make RealSense import fail (common in CI)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.realsense", None)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.realsense.camera_realsense", None)
    monkeypatch.setitem(sys.modules, "lerobot.cameras.realsense.configuration_realsense", None)
    monkeypatch.setitem(sys.modules, "strands", _mock_strands)

    # Mock psutil (not always available in CI)
    monkeypatch.setitem(sys.modules, "psutil", MagicMock())

    # Clear the module to force re-import
    mod_name = "strands_robots.tools.lerobot_camera"
    if mod_name in sys.modules:
        del sys.modules[mod_name]


@pytest.fixture
def cam(_mock_imports):
    """Import the camera module fresh with mocked dependencies."""
    return importlib.import_module("strands_robots.tools.lerobot_camera")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frame to Image Content
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFrameToImageContent:
    def test_jpeg_encoding(self, cam):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = cam._frame_to_image_content(frame, "jpg")
        assert "image" in result
        assert result["image"]["format"] == "jpeg"

    def test_png_encoding(self, cam):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\x89PNG"))
        result = cam._frame_to_image_content(frame, "png")
        assert result["image"]["format"] == "png"

    def test_default_format(self, cam):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\xff\xd8"))
        result = cam._frame_to_image_content(frame, "bmp")  # unsupported → JPEG
        assert result["image"]["format"] == "jpeg"

    def test_encoding_failure(self, cam):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imencode.return_value = (False, None)
        result = cam._frame_to_image_content(frame, "jpg")
        assert "text" in result
        assert "Failed" in result["text"]

    def test_grayscale_frame(self, cam):
        frame = np.zeros((480, 640), dtype=np.uint8)  # 2D = grayscale
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\xff\xd8"))
        result = cam._frame_to_image_content(frame, "jpg")
        assert "image" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Action Dispatch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestActionDispatch:
    def test_unknown_action(self, cam):
        result = cam.lerobot_camera(action="fly")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_capture_requires_camera_id(self, cam):
        result = cam.lerobot_camera(action="capture")
        assert result["status"] == "error"
        assert "camera_id required" in result["content"][0]["text"]

    def test_record_requires_camera_id(self, cam):
        result = cam.lerobot_camera(action="record")
        assert result["status"] == "error"
        assert "camera_id required" in result["content"][0]["text"]

    def test_preview_requires_camera_id(self, cam):
        result = cam.lerobot_camera(action="preview")
        assert result["status"] == "error"
        assert "camera_id required" in result["content"][0]["text"]

    def test_test_requires_camera_id(self, cam):
        result = cam.lerobot_camera(action="test")
        assert result["status"] == "error"
        assert "camera_id required" in result["content"][0]["text"]

    def test_configure_requires_camera_id(self, cam):
        result = cam.lerobot_camera(action="configure")
        assert result["status"] == "error"
        assert "camera_id required" in result["content"][0]["text"]

    def test_exception_in_action(self, cam):
        with patch.object(cam, "_discover_cameras", side_effect=RuntimeError("boom")):
            result = cam.lerobot_camera(action="discover")
            assert result["status"] == "error"
            assert "boom" in result["content"][0]["text"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDiscoverCameras:
    def test_no_cameras(self, cam):
        cam.OpenCVCamera.find_cameras.return_value = []
        cam.REALSENSE_AVAILABLE = False
        result = cam._discover_cameras()
        assert result["status"] == "success"
        assert "No cameras detected" in result["content"][0]["text"]

    def test_opencv_cameras_found(self, cam):
        cam.OpenCVCamera.find_cameras.return_value = [
            {
                "name": "USB Camera",
                "id": 0,
                "backend_api": "V4L2",
                "default_stream_profile": {"width": 640, "height": 480, "fps": 30, "format": "MJPG"},
            }
        ]
        cam.REALSENSE_AVAILABLE = False
        result = cam._discover_cameras()
        assert result["status"] == "success"
        assert "USB Camera" in result["content"][0]["text"]
        assert "1 cameras found" in result["content"][0]["text"]

    def test_realsense_cameras_found(self, cam):
        cam.OpenCVCamera.find_cameras.return_value = []
        cam.REALSENSE_AVAILABLE = True
        mock_rs = MagicMock()
        mock_rs.find_cameras.return_value = [{"name": "Intel D435", "serial_number": "123456", "type": "D400"}]
        cam.RealSenseCamera = mock_rs
        result = cam._discover_cameras()
        assert result["status"] == "success"
        assert "Intel D435" in result["content"][0]["text"]

    def test_discovery_exception(self, cam):
        cam.OpenCVCamera.find_cameras.side_effect = RuntimeError("hw error")
        result = cam._discover_cameras()
        assert result["status"] == "error"

    def test_realsense_discovery_fails_gracefully(self, cam):
        cam.OpenCVCamera.find_cameras.side_effect = None  # Reset from prior test
        cam.OpenCVCamera.find_cameras.return_value = [
            {"name": "cam", "id": 0, "backend_api": "V4L2", "default_stream_profile": {}}
        ]
        cam.REALSENSE_AVAILABLE = True
        mock_rs = MagicMock()
        mock_rs.find_cameras.side_effect = RuntimeError("RS failure")
        cam.RealSenseCamera = mock_rs
        result = cam._discover_cameras()
        assert result["status"] == "success"  # Still succeeds with OpenCV cameras


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# List Camera Details
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestListCameraDetails:
    def test_opencv_details_no_id(self, cam):
        result = cam._list_camera_details("opencv")
        assert result["status"] == "success"
        assert "OpenCV Camera System" in result["content"][0]["text"]

    def test_opencv_details_with_id(self, cam):
        mock_camera = MagicMock()
        mock_camera.fps = 30
        mock_camera.width = 640
        mock_camera.height = 480
        mock_camera.color_mode.value = "RGB"
        cam.OpenCVCamera.return_value = mock_camera

        result = cam._list_camera_details("opencv", camera_id=0)
        assert result["status"] == "success"
        assert "Connection: ✅ Success" in result["content"][0]["text"]

    def test_opencv_details_connect_fails(self, cam):
        mock_camera = MagicMock()
        mock_camera.connect.side_effect = RuntimeError("No camera")
        cam.OpenCVCamera.return_value = mock_camera

        result = cam._list_camera_details("opencv", camera_id=99)
        assert result["status"] == "success"
        assert "Connection: ❌ Failed" in result["content"][0]["text"]

    def test_realsense_available(self, cam):
        cam.REALSENSE_AVAILABLE = True
        result = cam._list_camera_details("realsense")
        assert result["status"] == "success"
        assert "RealSense" in result["content"][0]["text"]

    def test_realsense_not_available(self, cam):
        cam.REALSENSE_AVAILABLE = False
        result = cam._list_camera_details("realsense")
        assert result["status"] == "success"
        assert "Not installed" in result["content"][0]["text"]

    def test_unknown_camera_type(self, cam):
        cam.REALSENSE_AVAILABLE = False
        result = cam._list_camera_details("kinect")
        assert result["status"] == "success"
        assert "Unknown camera type" in result["content"][0]["text"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Capture Single Image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCaptureSingleImage:
    def test_capture_success(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Reset imencode to return proper data
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\xff\xd8JPEG"))
        _mock_cv2.imwrite.return_value = True

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("os.path.getsize", return_value=12345):
                result = cam._capture_single_image(
                    "opencv", 0, str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
                )
                assert result["status"] == "success"
                assert "Capture Success" in result["content"][0]["text"]
                mock_camera.connect.assert_called_once()
                mock_camera.disconnect.assert_called_once()

    def test_capture_async(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.async_read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imwrite.return_value = True
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\xff\xd8"))

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("os.path.getsize", return_value=5000):
                result = cam._capture_single_image(
                    "opencv", 0, str(tmp_path), "test_frame", 640, 480, 30, "RGB", "NO_ROTATION", "jpg", True, 500, True
                )
                assert result["status"] == "success"
                mock_camera.async_read.assert_called_once()

    def test_capture_save_fails(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imwrite.return_value = False

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            result = cam._capture_single_image(
                "opencv", 0, str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
            )
            assert result["status"] == "error"
            assert "Failed to save" in result["content"][0]["text"]

    def test_capture_exception(self, cam, tmp_path):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("no device")):
            result = cam._capture_single_image(
                "opencv", 99, str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
            )
            assert result["status"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch Capture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBatchCapture:
    def test_batch_success(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imwrite.return_value = True
        _mock_cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"\xff\xd8"))

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("os.path.getsize", return_value=5000):
                result = cam._capture_batch_images(
                    "opencv", [0, 1], str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
                )
                assert result["status"] == "success"
                assert "2/2" in result["content"][0]["text"]

    def test_batch_partial_failure(self, cam, tmp_path):
        call_count = [0]

        def make_camera(*args, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                mock.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                mock.connect.side_effect = RuntimeError("cam2 fail")
            return mock

        _mock_cv2.imwrite.return_value = True

        with patch.object(cam, "_create_camera", side_effect=make_camera):
            with patch("os.path.getsize", return_value=3000):
                result = cam._capture_batch_images(
                    "opencv", [0, 1], str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
                )
                assert result["status"] == "success"  # At least one succeeded

    def test_batch_exception(self, cam, tmp_path):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("fail")):
            result = cam._capture_batch_images(
                "opencv", [0], str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", "jpg", False, 1000, True
            )
            # The error is caught per-camera, so we still get a result
            assert "status" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Record Video
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecordVideo:
    def test_record_success(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_writer = MagicMock()
        _mock_cv2.VideoWriter.return_value = mock_writer

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("os.path.getsize", return_value=50000):
                result = cam._record_video_sequence(
                    "opencv",
                    0,
                    str(tmp_path),
                    None,
                    640,
                    480,
                    30,
                    "RGB",
                    "NO_ROTATION",
                    0.1,
                    False,
                    True,  # 0.1s = 3 frames at 30fps
                )
                assert result["status"] == "success"
                assert "Recording Complete" in result["content"][0]["text"]
                mock_writer.release.assert_called_once()

    def test_record_exception(self, cam, tmp_path):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("no cam")):
            result = cam._record_video_sequence(
                "opencv", 0, str(tmp_path), None, 640, 480, 30, "RGB", "NO_ROTATION", 1.0, False, True
            )
            assert result["status"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Preview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPreview:
    def test_preview_success(self, cam):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.waitKey.return_value = ord("q")  # Quit immediately

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("time.sleep"):
                result = cam._preview_camera_live(
                    "opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION", 0.1, False, 1000, True
                )
                assert result["status"] == "success"
                assert "Preview Complete" in result["content"][0]["text"]

    def test_preview_exception(self, cam):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("fail")):
            result = cam._preview_camera_live("opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION", 1.0, False, 1000, True)
            assert result["status"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPerformance:
    def test_performance_sync(self, cam):
        mock_camera = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.read.return_value = frame
        mock_camera.fps = 30
        mock_camera.width = 640
        mock_camera.height = 480
        mock_camera.color_mode.value = "RGB"

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            result = cam._test_camera_performance("opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION", False, 1000, True)
            assert result["status"] == "success"
            assert "Performance Test" in result["content"][0]["text"]
            assert mock_camera.read.call_count == 10  # 10 test frames

    def test_performance_async(self, cam):
        mock_camera = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.read.return_value = frame
        mock_camera.async_read.return_value = frame
        mock_camera.fps = 30
        mock_camera.width = 640
        mock_camera.height = 480
        mock_camera.color_mode.value = "RGB"

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            result = cam._test_camera_performance("opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION", True, 1000, True)
            assert result["status"] == "success"
            assert "Async Capture" in result["content"][0]["text"]

    def test_performance_exception(self, cam):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("fail")):
            result = cam._test_camera_performance("opencv", 99, 640, 480, 30, "RGB", "NO_ROTATION", False, 1000, True)
            assert result["status"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfigure:
    def test_configure_no_save(self, cam):
        mock_camera = MagicMock()
        mock_camera.width = 640
        mock_camera.height = 480
        mock_camera.fps = 30
        mock_camera.color_mode.value = "RGB"
        mock_camera.rotation = None

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            result = cam._configure_camera_settings(
                "opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION", "/tmp/save", False, True
            )
            assert result["status"] == "success"
            assert "Configuration" in result["content"][0]["text"]

    def test_configure_with_save(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.width = 640
        mock_camera.height = 480
        mock_camera.fps = 30
        mock_camera.color_mode.value = "RGB"
        mock_camera.rotation = "ROTATE_90"

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            result = cam._configure_camera_settings(
                "opencv", 0, 640, 480, 30, "RGB", "ROTATE_90", str(tmp_path), True, True
            )
            assert result["status"] == "success"
            assert "Configuration Saved" in result["content"][0]["text"]
            # Verify JSON file was written
            json_files = list(tmp_path.glob("*.json"))
            assert len(json_files) == 1

    def test_configure_exception(self, cam):
        with patch.object(cam, "_create_camera", side_effect=RuntimeError("fail")):
            result = cam._configure_camera_settings(
                "opencv", 99, 640, 480, 30, "RGB", "NO_ROTATION", "/tmp", False, True
            )
            assert result["status"] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Create Camera
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreateCamera:
    def test_opencv_camera_rgb(self, cam):
        camera = cam._create_camera("opencv", 0, 640, 480, 30, "RGB", "NO_ROTATION")
        assert camera is not None

    def test_opencv_camera_bgr(self, cam):
        camera = cam._create_camera("opencv", 0, 640, 480, 30, "BGR", "NO_ROTATION")
        assert camera is not None

    def test_opencv_camera_rotate_90(self, cam):
        camera = cam._create_camera("opencv", 0, 640, 480, 30, "RGB", "ROTATE_90")
        assert camera is not None

    def test_opencv_camera_rotate_180(self, cam):
        camera = cam._create_camera("opencv", 0, 640, 480, 30, "RGB", "ROTATE_180")
        assert camera is not None

    def test_opencv_camera_rotate_270(self, cam):
        camera = cam._create_camera("opencv", 0, 640, 480, 30, "RGB", "ROTATE_270")
        assert camera is not None

    def test_realsense_camera(self, cam):
        cam.REALSENSE_AVAILABLE = True
        cam.RealSenseCameraConfig = MagicMock()
        cam.RealSenseCamera = MagicMock()
        camera = cam._create_camera("realsense", "123456", 640, 480, 30, "RGB", "NO_ROTATION")
        assert camera is not None

    def test_unsupported_type(self, cam):
        cam.REALSENSE_AVAILABLE = False
        with pytest.raises(ValueError, match="Unsupported camera type"):
            cam._create_camera("kinect", 0, 640, 480, 30, "RGB", "NO_ROTATION")

    def test_string_camera_id(self, cam):
        camera = cam._create_camera("opencv", "/dev/video0", 640, 480, 30, "RGB", "NO_ROTATION")
        assert camera is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenCV Backend Name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOpenCVBackendName:
    def test_backend_name(self, cam):
        name = cam._get_opencv_backend_name()
        assert isinstance(name, str)
        assert name in ["Auto", "V4L2", "MSMF", "AVFoundation", "Unknown"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration: Full Tool Call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntegration:
    def test_discover_via_tool(self, cam):
        cam.OpenCVCamera.find_cameras.side_effect = None  # Reset from prior test
        cam.OpenCVCamera.find_cameras.return_value = []
        cam.REALSENSE_AVAILABLE = False
        result = cam.lerobot_camera(action="discover")
        assert result["status"] == "success"

    def test_list_via_tool(self, cam):
        result = cam.lerobot_camera(action="list", camera_type="opencv")
        assert result["status"] == "success"

    def test_capture_batch_default_cameras(self, cam, tmp_path):
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        _mock_cv2.imwrite.return_value = True

        with patch.object(cam, "_create_camera", return_value=mock_camera):
            with patch("os.path.getsize", return_value=1000):
                result = cam.lerobot_camera(
                    action="capture_batch",
                    save_path=str(tmp_path),
                )
                assert "status" in result
