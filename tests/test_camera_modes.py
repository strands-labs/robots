"""A camera spec chooses the format a device sends (OpenCV FOURCC) and the frames' format (AVFoundation), and a mode
the device lacks is refused naming the mode asked and the modes it has."""

from __future__ import annotations

import sys
import types

import pytest

from strands_robots.robot.drivers import devices
from strands_robots.robot.drivers.cameras import CameraSpec

SUPPORTED = [
    "Selected framerate (60.000000) is not supported by the device.",
    "Supported modes:",
    "  640x480@[1.000000 30.000000]fps",
    "  1280x720@[1.000000 10.000000]fps",
    "  1280x720@[1.000000 30.000000]fps",
]


def test_a_spec_takes_a_fourcc_and_a_pixel_format_and_refuses_ones_no_backend_has():
    assert CameraSpec.of("w", 1).fourcc == "MJPG"  # compressed by default: 720p @ 30 fits a USB 2.0 camera
    spec = CameraSpec.of("w", {"index": 1, "fourcc": "YUYV", "pixel_format": "nv12"})
    assert (spec.fourcc, spec.pixel_format) == ("YUYV", "nv12")
    assert CameraSpec.of("w", {"index": 1, "fourcc": None}).fourcc is None  # the device's own codec
    with pytest.raises(ValueError, match=r"camera 'w': fourcc must be 4 characters .* got 'MJPEG'"):
        CameraSpec.of("w", {"index": 1, "fourcc": "MJPEG"})
    with pytest.raises(ValueError, match=r"camera 'w': pixel_format 'mjpeg' is not one AVFoundation .* uyvy422, "):
        CameraSpec.of("w", {"name": "Logi", "pixel_format": "mjpeg"})
    with pytest.raises(TypeError, match=r"use index, name, width, height, fps, fourcc, pixel_format"):
        CameraSpec.of("w", {"name": "Logi", "format": "nv12"})


def test_avfoundation_opens_with_the_specs_pixel_format():
    assert devices.av_options(CameraSpec(name="Logi"))["pixel_format"] == "uyvy422"
    assert devices.av_options(CameraSpec(name="Logi", pixel_format="nv12"))["pixel_format"] == "nv12"


def test_opencv_asks_for_the_fourcc_before_the_size(monkeypatch):
    calls: list = []

    class Capture:
        def __init__(self, index):
            pass

        def set(self, prop, value):
            calls.append((prop, value))

    cv2 = types.SimpleNamespace(
        VideoCapture=Capture,
        VideoWriter_fourcc=lambda *c: "".join(c),
        CAP_PROP_FOURCC="fourcc",
        CAP_PROP_FRAME_WIDTH="w",
        CAP_PROP_FRAME_HEIGHT="h",
        CAP_PROP_FPS="fps",
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    devices._CVCapture(0, CameraSpec(0, 1280, 720, 30))
    assert calls == [("fourcc", "MJPG"), ("w", 1280), ("h", 720), ("fps", 30)]
    calls.clear()
    devices._CVCapture(0, CameraSpec(0, 640, 480, fourcc=None))
    assert calls == [("w", 640), ("h", 480)]


def test_a_mode_the_camera_lacks_is_refused_naming_it_and_the_cameras_modes(monkeypatch):
    levels: list = []

    class Capture(list):
        def __init__(self, local=True):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_(name, format, options):
        capture.extend(("libav.avfoundation", 16, line) for line in SUPPORTED)
        raise OSError(5, "Input/output error")

    capture = Capture()
    log = types.ModuleType("av.logging")
    log.ERROR, log.get_level, log.set_level = 16, lambda: None, levels.append
    log.Capture = lambda local=True: capture
    av = types.ModuleType("av")
    av.open, av.logging = open_, log
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.logging", log)
    with pytest.raises(OSError) as err:
        devices._AVCapture("USB2.0_CAM1", CameraSpec(name="USB2.0_CAM1", width=1280, height=720, fps=60))
    assert err.value.errno == 5
    assert str(err.value).endswith(
        "camera 'USB2.0_CAM1' refused 1280x720 @ 60 fps (uyvy422): Selected framerate (60.000000) is not supported "
        "by the device. Its modes: 640x480@30fps, 1280x720@10fps, 1280x720@30fps"
    )
    assert levels == [16, None]  # the indev's ERROR lines are captured, then PyAV's own level is put back
