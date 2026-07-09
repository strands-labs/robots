"""Behavior tests for the ``lerobot_camera`` agent tool.

The camera tool wraps LeRobot's OpenCV/RealSense camera classes behind a single
agent-facing dispatcher. These tests exercise every action branch hardware-free
by substituting a fake camera, and pin two invariants the tool must uphold:

1. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
2. Boolean operating state (async read mode, connection warmup) is reported with
   meaningful on/off words, not rendered as an empty string.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.tools.lerobot_camera import lerobot_camera


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    """Fail if any character is outside the ASCII range."""
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


class FakeCamera:
    """Minimal stand-in for a LeRobot camera object.

    Records connect/disconnect calls and serves a fixed RGB frame for both the
    synchronous ``read`` and asynchronous ``async_read`` paths.
    """

    def __init__(self, width: int = 8, height: int = 6, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.color_mode = SimpleNamespace(value="RGB")
        self.rotation: Any = None
        self.connected = False
        self.disconnect_calls = 0

    def connect(self, warmup: bool = True) -> None:
        self.connected = True

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_calls += 1


@pytest.fixture
def fake_camera(monkeypatch: pytest.MonkeyPatch) -> FakeCamera:
    """Patch ``_create_camera`` so every action uses a hardware-free camera."""
    camera = FakeCamera()
    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: camera)
    return camera


# --- dispatcher routing + required-parameter validation -------------------


def test_unknown_action_returns_error() -> None:
    result = lerobot_camera(action="does_not_exist")
    assert result["status"] == "error"
    assert "Unknown action" in _texts(result)


@pytest.mark.parametrize("action", ["capture", "record", "preview", "test", "configure"])
def test_actions_requiring_camera_id_error_without_it(action: str) -> None:
    result = lerobot_camera(action=action)
    assert result["status"] == "error"
    body = _texts(result)
    assert "camera_id required" in body
    _assert_ascii(body)


# --- _frame_to_image_content (pure helper) --------------------------------


@pytest.mark.parametrize(
    "fmt,expected",
    [("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png"), ("bmp", "jpeg")],
)
def test_frame_to_image_content_formats(fmt: str, expected: str) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    content = cam_mod._frame_to_image_content(frame, fmt)
    assert content["image"]["format"] == expected
    assert isinstance(content["image"]["source"]["bytes"], bytes)
    assert content["image"]["source"]["bytes"]


def test_frame_to_image_content_handles_encode_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cam_mod.cv2, "imencode", lambda *a, **k: (False, None))
    content = cam_mod._frame_to_image_content(np.zeros((4, 4, 3), dtype=np.uint8), "jpg")
    assert "Failed to encode" in content["text"]


# --- _create_camera + backend helper --------------------------------------


def test_create_camera_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported camera type"):
        cam_mod._create_camera("nonsense", 0, 640, 480, 30, "RGB", "NO_ROTATION")


def test_get_opencv_backend_name_is_ascii() -> None:
    name = cam_mod._get_opencv_backend_name()
    assert name
    _assert_ascii(name)


# --- list + discover routing ----------------------------------------------


def test_list_opencv_details_ascii() -> None:
    result = lerobot_camera(action="list", camera_type="opencv")
    assert result["status"] == "success"
    body = _texts(result)
    assert "OpenCV Camera System" in body
    # rotations spelled out in ASCII, not degree symbols
    assert "0, 90, 180, 270 degrees" in body
    _assert_ascii(body)


def test_discover_uses_ascii_bullets(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cams = [
        {
            "name": "Cam0",
            "id": 0,
            "backend_api": "V4L2",
            "default_stream_profile": {"width": 640, "height": 480, "fps": 30, "format": "MJPG"},
        }
    ]
    monkeypatch.setattr(cam_mod.OpenCVCamera, "find_cameras", staticmethod(lambda: fake_cams))
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)
    result = lerobot_camera(action="discover")
    assert result["status"] == "success"
    body = _texts(result)
    assert "  - **Cam0**" in body  # ASCII hyphen bullet, not a unicode bullet
    assert "Total: 1 cameras found" in body
    _assert_ascii(body)


# --- capture / batch / record / preview / test / configure ----------------


@pytest.mark.parametrize("async_mode,expected", [(True, "Async mode: on"), (False, "Async mode: off")])
def test_capture_single_reports_async_state_ascii(
    fake_camera: FakeCamera, tmp_path, async_mode: bool, expected: str
) -> None:
    result = lerobot_camera(
        action="capture",
        camera_id=0,
        save_path=str(tmp_path),
        async_mode=async_mode,
    )
    assert result["status"] == "success"
    body = _texts(result)
    # Regression: pre-fix this rendered "Async mode: " with an empty value.
    assert expected in body
    _assert_ascii(body)
    assert fake_camera.disconnect_calls == 1
    # an image payload accompanies the text summary
    assert any("image" in item for item in result["content"])


def test_capture_batch_reports_async_state_ascii(fake_camera: FakeCamera, tmp_path) -> None:
    result = lerobot_camera(
        action="capture_batch",
        camera_ids=[0, 1],
        save_path=str(tmp_path),
        async_mode=True,
    )
    assert result["status"] == "success"
    body = _texts(result)
    assert "Async mode: on" in body
    assert "Success: 2/2 cameras" in body
    _assert_ascii(body)


def test_record_video_summary_ascii(fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = SimpleNamespace(write=lambda f: None, release=lambda: None)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)
    result = lerobot_camera(
        action="record",
        camera_id=0,
        save_path=str(tmp_path),
        fps=2,
        capture_duration=0.5,
        async_mode=False,
    )
    assert result["status"] == "success"
    body = _texts(result)
    # Regression: these lines previously carried an orphan U+FE0F variation selector.
    assert "Frames:" in body
    assert "Duration:" in body
    assert "Async mode: off" in body
    _assert_ascii(body)


def test_preview_summary_ascii(fake_camera: FakeCamera, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("imshow", "putText", "destroyAllWindows"):
        monkeypatch.setattr(cam_mod.cv2, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "waitKey", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.time, "sleep", lambda *a, **k: None)
    result = lerobot_camera(action="preview", camera_id=0, fps=2, preview_duration=0.01)
    assert result["status"] == "success"
    body = _texts(result)
    assert "Live Preview Complete" in body
    assert "Frames displayed:" in body
    _assert_ascii(body)


def test_performance_summary_uses_ascii_labels(fake_camera: FakeCamera) -> None:
    result = lerobot_camera(action="test", camera_id=0, async_mode=False)
    assert result["status"] == "success"
    body = _texts(result)
    # Regression: summary labels previously embedded leading-space + U+FE0F markers.
    assert "Connection:" in body
    assert ("Fast" in body) or ("Slow" in body)
    assert "Camera Configuration" in body
    _assert_ascii(body)


@pytest.mark.parametrize("warmup,expected", [(True, "Warmup: on"), (False, "Warmup: off")])
def test_configure_reports_warmup_state_ascii(fake_camera: FakeCamera, tmp_path, warmup: bool, expected: str) -> None:
    result = lerobot_camera(
        action="configure",
        camera_id=0,
        save_path=str(tmp_path),
        warmup=warmup,
        save_config=True,
    )
    assert result["status"] == "success"
    body = _texts(result)
    # Regression: pre-fix rendered "Warmup: " with an empty value.
    assert expected in body
    assert "Configuration Saved" in body
    _assert_ascii(body)
    # config JSON actually written
    assert list(tmp_path.glob("camera_config_*.json"))


def test_module_source_is_ascii_only() -> None:
    """The whole module must be free of non-ASCII characters (no-emoji rule)."""
    import inspect

    source = inspect.getsource(cam_mod)
    offenders = sorted({hex(ord(c)) for c in source if ord(c) > 127})
    assert not offenders, f"non-ASCII characters in module source: {offenders}"


# --- discovery aggregation (cameras present + RealSense failure tolerance) --


def test_discover_aggregates_opencv_and_realsense(monkeypatch: pytest.MonkeyPatch) -> None:
    """``discover`` formats found OpenCV + RealSense cameras and reports a total
    that sums both backends."""
    opencv_found = [
        {
            "name": "Logitech C920",
            "id": "/dev/video0",
            "backend_api": "V4L2",
            "default_stream_profile": {"width": 1280, "height": 720, "fps": 30, "format": "MJPG"},
        }
    ]
    realsense_found = [{"name": "Intel D435", "serial_number": "abc123", "type": "depth"}]
    monkeypatch.setattr(cam_mod.OpenCVCamera, "find_cameras", staticmethod(lambda: opencv_found))
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", True)
    monkeypatch.setattr(cam_mod, "RealSenseCamera", SimpleNamespace(find_cameras=staticmethod(lambda: realsense_found)))

    result = lerobot_camera(action="discover")
    assert result["status"] == "success"
    body = _texts(result)
    assert "Logitech C920" in body
    assert "1280x720" in body
    assert "Intel D435" in body
    assert "Total: 2 cameras found" in body
    _assert_ascii(body)


def test_discover_tolerates_realsense_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RealSense discovery error is swallowed (logged) and OpenCV results still
    surface - the tool never crashes on one backend failing."""

    def boom() -> list:
        raise RuntimeError("rs2 backend offline")

    monkeypatch.setattr(cam_mod.OpenCVCamera, "find_cameras", staticmethod(lambda: []))
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", True)
    monkeypatch.setattr(cam_mod, "RealSenseCamera", SimpleNamespace(find_cameras=staticmethod(boom)))

    result = lerobot_camera(action="discover")
    assert result["status"] == "success"
    body = _texts(result)
    assert "No cameras detected" in body
    _assert_ascii(body)


# --- list per-camera probe (success + failure branches) --------------------


def test_list_probes_specific_camera_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """``list`` with a camera_id connects to that camera and reports its actual
    resolution / fps / color mode."""
    probe = FakeCamera(width=1920, height=1080, fps=60)
    monkeypatch.setattr(cam_mod, "OpenCVCameraConfig", SimpleNamespace)
    monkeypatch.setattr(cam_mod, "OpenCVCamera", lambda config: probe)

    result = lerobot_camera(action="list", camera_id=2)
    assert result["status"] == "success"
    body = _texts(result)
    assert "Camera 2 Details" in body
    assert "Connection:  Success" in body
    assert "1920x1080" in body
    assert probe.disconnect_calls == 1
    _assert_ascii(body)


def test_list_probes_specific_camera_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera that fails to connect is reported as a failed probe, not a crash,
    and the overall tool call still succeeds."""

    class _DeadCamera:
        def __init__(self, config: Any) -> None:
            pass

        def connect(self, warmup: bool = True) -> None:
            raise OSError("device busy")

    monkeypatch.setattr(cam_mod, "OpenCVCameraConfig", SimpleNamespace)
    monkeypatch.setattr(cam_mod, "OpenCVCamera", _DeadCamera)

    result = lerobot_camera(action="list", camera_id=9)
    assert result["status"] == "success"
    body = _texts(result)
    assert "Connection:  Failed (device busy)" in body
    _assert_ascii(body)


def test_list_realsense_when_sdk_missing_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing a RealSense camera without the SDK reports the install hint
    instead of pretending the camera type is unknown."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)
    result = lerobot_camera(action="list", camera_type="realsense")
    assert result["status"] == "success"
    body = _texts(result)
    assert "Not installed" in body
    assert "pip install pyrealsense2" in body
    _assert_ascii(body)


def test_list_unknown_camera_type_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised camera_type produces an explicit 'Unknown camera type'
    line rather than silently succeeding with empty details."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", True)
    result = lerobot_camera(action="list", camera_type="thermal")
    assert result["status"] == "success"
    body = _texts(result)
    assert "Unknown camera type: thermal" in body
    _assert_ascii(body)


# --- capture / batch save-failure paths ------------------------------------


def test_capture_reports_save_failure(fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When cv2.imwrite returns False the capture is reported as an error with
    the target path, not a false success."""
    monkeypatch.setattr(cam_mod.cv2, "imwrite", lambda *a, **k: False)
    result = lerobot_camera(action="capture", camera_id=0, save_path=str(tmp_path))
    assert result["status"] == "error"
    body = _texts(result)
    assert "Failed to save image" in body
    # Camera is still released even on the save-failure path.
    assert fake_camera.disconnect_calls == 1


def test_capture_batch_all_fail_returns_error_status(
    fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every camera fails to save, the batch reports overall error status and
    a 0/N success summary."""
    monkeypatch.setattr(cam_mod.cv2, "imwrite", lambda *a, **k: False)
    result = lerobot_camera(action="capture_batch", camera_ids=[0, 1], save_path=str(tmp_path))
    assert result["status"] == "error"
    body = _texts(result)
    assert "Success: 0/2 cameras" in body
    _assert_ascii(body)


def test_capture_batch_defaults_camera_ids_when_omitted(
    fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting camera_ids falls back to the default robot camera set rather
    than failing."""
    captured: dict[str, Any] = {}
    real_batch = cam_mod._capture_batch_images

    def spy(camera_type, camera_ids, *args, **kwargs):
        captured["ids"] = camera_ids
        return real_batch(camera_type, camera_ids, *args, **kwargs)

    monkeypatch.setattr(cam_mod, "_capture_batch_images", spy)
    result = lerobot_camera(action="capture_batch", save_path=str(tmp_path))
    assert result["status"] == "success"
    assert captured["ids"] == [0, "/dev/video4"]


# --- performance test async branch -----------------------------------------


def test_performance_async_branch_reports_speedup(fake_camera: FakeCamera) -> None:
    """With async_mode the performance test also measures async capture and
    reports a sync/async speedup figure."""
    result = lerobot_camera(action="test", camera_id=0, async_mode=True)
    assert result["status"] == "success"
    body = _texts(result)
    assert "Async Capture (10 frames)" in body
    assert "Speedup:" in body
    _assert_ascii(body)


# --- _create_camera backend selection --------------------------------------


def test_create_camera_builds_realsense_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the RealSense SDK is present, _create_camera routes a 'realsense'
    request to the RealSense config + camera classes with the serial number."""
    seen: dict[str, Any] = {}

    def fake_config(serial_number, fps, width, height):
        seen["serial"] = serial_number
        return SimpleNamespace(serial_number=serial_number)

    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", True)
    monkeypatch.setattr(cam_mod, "RealSenseCameraConfig", fake_config)
    monkeypatch.setattr(cam_mod, "RealSenseCamera", lambda config: SimpleNamespace(config=config))

    cam = cam_mod._create_camera("realsense", "0123", 640, 480, 30, "RGB", "NO_ROTATION")
    assert seen["serial"] == "0123"
    assert cam.config.serial_number == "0123"


def test_create_camera_realsense_without_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting a RealSense camera without the SDK raises a clear
    unsupported-type error (no silent fallback to OpenCV)."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)
    with pytest.raises(ValueError, match="Unsupported camera type: realsense"):
        cam_mod._create_camera("realsense", "0", 640, 480, 30, "RGB", "NO_ROTATION")


# --- frame encoding failure -------------------------------------------------


def test_frame_to_image_content_unknown_format_defaults_to_jpeg() -> None:
    """An unrecognised format string falls back to JPEG encoding rather than
    erroring."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    content = cam_mod._frame_to_image_content(frame, "tiff")
    assert content["image"]["format"] == "jpeg"


def test_create_camera_opencv_maps_color_mode_and_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """_create_camera translates the string color_mode / rotation selectors into
    the LeRobot enum config it hands to OpenCVCamera."""
    seen: dict[str, Any] = {}

    def fake_config(index_or_path, fps, width, height, color_mode, rotation):
        seen.update(
            index_or_path=index_or_path,
            color_mode=color_mode,
            rotation=rotation,
        )
        return SimpleNamespace()

    monkeypatch.setattr(cam_mod, "OpenCVCameraConfig", fake_config)
    monkeypatch.setattr(cam_mod, "OpenCVCamera", lambda config: SimpleNamespace(config=config))

    cam_mod._create_camera("opencv", "/dev/video2", 640, 480, 30, "BGR", "ROTATE_180")
    assert seen["index_or_path"] == "/dev/video2"
    assert seen["color_mode"] == cam_mod.ColorMode.BGR
    assert seen["rotation"] == cam_mod.Cv2Rotation.ROTATE_180


def test_list_realsense_available_reports_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the RealSense SDK is present, listing a realsense camera advertises
    its depth + multi-stream capabilities."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", True)
    result = lerobot_camera(action="list", camera_type="realsense")
    assert result["status"] == "success"
    body = _texts(result)
    assert "Depth Support" in body
    assert "Color, Depth, Infrared" in body
    _assert_ascii(body)


# --- error-dict contract: hardware failures must degrade, never raise -------


class _ExplodingCamera:
    """A camera whose construction succeeds but whose ``connect`` fails.

    Models a device that vanishes between enumeration and use (cable yanked,
    busy handle, driver fault) - the failure mode the per-action error wrappers
    exist to absorb.
    """

    def connect(self, warmup: bool = True) -> None:
        raise RuntimeError("device disappeared")


@pytest.mark.parametrize(
    "action,expected_phrase",
    [
        ("capture", "Image capture failed"),
        ("record", "Video recording failed"),
        ("preview", "Preview failed"),
        ("test", "Performance test failed"),
        ("configure", "Configuration failed"),
    ],
)
def test_single_camera_actions_return_error_dict_on_hardware_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path, action: str, expected_phrase: str
) -> None:
    """Every single-camera action must catch a mid-operation hardware fault and
    return a structured ``{"status": "error"}`` dict - the agent-tool contract
    forbids raising past the dispatcher."""
    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: _ExplodingCamera())
    result = lerobot_camera(action=action, camera_id=0, save_path=str(tmp_path))
    assert result["status"] == "error"
    body = _texts(result)
    assert expected_phrase in body
    assert "device disappeared" in body
    _assert_ascii(body)


def test_capture_batch_aggregates_per_camera_failures(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When each camera in a batch fails, the per-camera errors are aggregated
    into a single error-status result rather than propagating an exception."""
    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: _ExplodingCamera())
    result = lerobot_camera(
        action="capture_batch",
        camera_ids=[0, 1],
        save_path=str(tmp_path),
    )
    assert result["status"] == "error"
    body = _texts(result)
    assert "Success: 0/2 cameras" in body
    _assert_ascii(body)


def test_capture_batch_rejects_path_traversal(fake_camera: FakeCamera) -> None:
    """A traversal ``save_path`` is rejected by validation and surfaces as a
    batch-level error dict, never an unhandled ValueError."""
    result = lerobot_camera(
        action="capture_batch",
        camera_ids=[0],
        save_path="../../etc/sneaky",
    )
    assert result["status"] == "error"
    body = _texts(result)
    assert "Batch capture failed" in body
    _assert_ascii(body)


# --- record / preview loop branches (async reads, periodic progress) --------


def test_record_async_mode_drives_progress_branch(
    fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording in async mode reads via ``async_read`` and emits a periodic
    progress line once a full second of frames is captured."""
    writer = SimpleNamespace(write=lambda f: None, release=lambda: None)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 4096)
    # fps=2, duration=1.0 -> target_frames=2 -> frame 2 hits the (n % fps == 0)
    # progress branch and async_read is exercised on every iteration.
    result = lerobot_camera(
        action="record",
        camera_id=0,
        save_path=str(tmp_path),
        fps=2,
        capture_duration=1.0,
        async_mode=True,
    )
    assert result["status"] == "success"
    body = _texts(result)
    assert "Frames: 2 @ 2 FPS" in body
    assert "Async mode: on" in body
    _assert_ascii(body)


def test_preview_async_fps_report_and_quit_key(fake_camera: FakeCamera, monkeypatch: pytest.MonkeyPatch) -> None:
    """Async preview reads via ``async_read``, reports a live-FPS line after a
    second elapses, and honours the 'q' quit key to break the loop early."""
    for name in ("imshow", "putText", "destroyAllWindows"):
        monkeypatch.setattr(cam_mod.cv2, name, lambda *a, **k: None, raising=False)
    # waitKey returns ord('q') so the loop breaks after the first frame.
    monkeypatch.setattr(cam_mod.cv2, "waitKey", lambda *a, **k: ord("q"), raising=False)
    monkeypatch.setattr(cam_mod.time, "sleep", lambda *a, **k: None)
    # Monotonic clock advancing 1s per call so the >=1.0s live-FPS branch fires
    # on the first iteration regardless of exact call count.
    ticks = iter(range(0, 10_000))
    monkeypatch.setattr(cam_mod.time, "time", lambda: float(next(ticks)))
    result = lerobot_camera(
        action="preview",
        camera_id=0,
        fps=2,
        preview_duration=100.0,
        async_mode=True,
    )
    assert result["status"] == "success"
    body = _texts(result)
    assert "Live Preview Complete" in body
    assert "Async mode: on" in body
    _assert_ascii(body)


# --- smaller branch coverage: batch filename, rotation config, grayscale -----


def test_capture_batch_uses_custom_filename_prefix(
    fake_camera: FakeCamera, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied ``filename`` becomes the per-camera file prefix instead
    of the default ``batch_<type>_...`` naming."""
    written: list[str] = []

    def _record_write(path: str, frame: Any) -> bool:
        written.append(path)
        return True

    monkeypatch.setattr(cam_mod.cv2, "imwrite", _record_write)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 256)
    result = lerobot_camera(
        action="capture_batch",
        camera_ids=[0],
        save_path=str(tmp_path),
        filename="mission",
    )
    assert result["status"] == "success"
    assert written and "mission_0_" in written[0]


def test_configure_emits_rotation_when_camera_exposes_it(fake_camera: FakeCamera, tmp_path) -> None:
    """A camera that reports a non-null ``rotation`` has that rotation surfaced
    in the configuration summary."""
    fake_camera.rotation = SimpleNamespace(value="ROTATE_90")
    result = lerobot_camera(
        action="configure",
        camera_id=0,
        save_path=str(tmp_path),
        rotation="ROTATE_90",
    )
    assert result["status"] == "success"
    body = _texts(result)
    assert "Rotation: ROTATE_90" in body
    _assert_ascii(body)


def test_frame_to_image_content_passes_through_non_rgb_frame() -> None:
    """A single-channel (grayscale) frame skips the RGB->BGR conversion and is
    still encoded to a valid image payload."""
    gray = np.zeros((6, 8), dtype=np.uint8)
    content = cam_mod._frame_to_image_content(gray, format="png")
    assert "image" in content
    assert content["image"]["format"] == "png"
    assert content["image"]["source"]["bytes"]


# --- dispatch catch-all: the tool never raises past dispatch ---------------
#
# Every action branch and helper is wrapped in a top-level ``try/except`` so an
# unexpected failure (a lerobot API drift, a driver crash) is turned into the
# ``{"status": "error"}`` tool-result contract rather than propagating out of
# the ``@tool`` and killing the agent turn. These pin that guarantee for each of
# the three catch-all wrappers.


def test_dispatch_catch_all_wraps_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected error inside an action branch becomes an error result.

    The ``discover`` branch delegates to ``_discover_cameras``; if that raises
    something the helper itself does not handle, the dispatcher's outer guard
    must still return the two-key error contract, never re-raise.
    """

    def boom() -> dict[str, Any]:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(cam_mod, "_discover_cameras", boom)

    result = lerobot_camera(action="discover")

    assert result["status"] == "error"
    assert set(result) == {"status", "content"}
    body = _texts(result)
    assert "Camera operation failed" in body
    assert "driver exploded" in body
    _assert_ascii(body)


def test_discover_catch_all_wraps_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing OpenCV probe surfaces as a discovery error, not an exception."""

    def boom() -> list[dict[str, Any]]:
        raise OSError("v4l2 enumeration failed")

    monkeypatch.setattr(cam_mod.OpenCVCamera, "find_cameras", staticmethod(boom))

    result = cam_mod._discover_cameras()

    assert result["status"] == "error"
    assert set(result) == {"status", "content"}
    body = _texts(result)
    assert "Camera discovery failed" in body
    assert "v4l2 enumeration failed" in body
    _assert_ascii(body)


def test_details_catch_all_wraps_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while assembling OpenCV details is returned as an error result."""

    def boom() -> str:
        raise RuntimeError("backend query failed")

    monkeypatch.setattr(cam_mod, "_get_opencv_backend_name", boom)

    result = cam_mod._list_camera_details("opencv")

    assert result["status"] == "error"
    assert set(result) == {"status", "content"}
    body = _texts(result)
    assert "Camera details failed" in body
    assert "backend query failed" in body
    _assert_ascii(body)
