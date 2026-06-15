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
        self.rotation = None
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
