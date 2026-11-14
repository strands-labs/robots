"""A camera spec chooses the format a device sends (OpenCV FOURCC) and the frames' format (AVFoundation), and a mode
the device lacks is refused naming the mode asked and the modes it has."""

from __future__ import annotations

import sys
import threading
import time
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


def _process_wide_av(monkeypatch):
    """PyAV's logging as it behaves: one process-wide level (a line at level L reaches the log when ``L <= level``;
    None drops every line) and a thread-local ``Capture``, so each open's lines go to the capture on its own thread.
    Listing devices takes ~60 ms; opening the first camera is refused in ~20 ms."""
    state: dict = {"level": None, "captures": {}}

    class Capture(list):
        def __init__(self, local=True):
            super().__init__()

        def __enter__(self):
            state["captures"][threading.get_ident()] = self
            return self

        def __exit__(self, *exc):
            state["captures"].pop(threading.get_ident(), None)
            return False

    def emit(level, line):
        cap = state["captures"].get(threading.get_ident())
        if state["level"] is not None and level <= state["level"] and cap is not None:
            cap.append(("libav.avfoundation", level, line))

    def open_(name, format, options):
        if options.get("list_devices"):
            for line in [
                "AVFoundation video devices:",
                "[0] Logi 4K",
                "[1] USB2.0_CAM1",
                "AVFoundation audio devices:",
            ]:
                time.sleep(0.015)
                emit(32, line)
            raise OSError(5, "Input/output error")
        time.sleep(0.02)
        for line in SUPPORTED:
            emit(16, line)
        raise OSError(5, "Input/output error")

    log = types.ModuleType("av.logging")
    log.ERROR, log.INFO = 16, 32
    log.get_level = lambda: state["level"]
    log.set_level = lambda v: state.__setitem__("level", v)
    log.Capture = Capture
    av = types.ModuleType("av")
    av.open, av.logging = open_, log
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.logging", log)
    return state


def test_a_refused_open_on_a_capture_thread_does_not_cut_the_next_cameras_name_lookup(monkeypatch):
    from strands_robots.robot.drivers import cameras

    state = _process_wide_av(monkeypatch)
    for _ in range(5):
        errors: list[OSError] = []

        def open_first(errors=errors):
            try:
                devices._AVCapture("Logi 4K", CameraSpec(name="Logi 4K", width=1280, height=720, fps=60))
            except OSError as exc:
                errors.append(exc)

        grabber = threading.Thread(target=open_first)  # Cameras.open: the first camera opens on its capture thread
        grabber.start()
        listed = cameras._avfoundation()  # ... while the caller resolves the next camera's name
        grabber.join()
        assert listed == [("Logi 4K", 0), ("USB2.0_CAM1", 1)]
        assert str(errors[0]).endswith("Its modes: 640x480@30fps, 1280x720@10fps, 1280x720@30fps")
        assert state["level"] is None and state["captures"] == {}  # PyAV left as it was found


def test_a_camera_stalled_in_av_open_does_not_block_the_next_cameras_name_lookup(monkeypatch):
    """The log lock is never held across ``av.open``: a camera stalled in ``av.open`` on its capture thread must not
    stop the caller from listing the next camera by name (which also raises the log level). Before the fix the lister
    blocked on the lock the stalled open held, so ``Cameras.open`` never reached ``grabber.wait(open_timeout_s)`` and
    hung forever instead of refusing within ``open_timeout_s``."""
    from strands_robots.robot.drivers import cameras

    state: dict = {"level": None, "captures": {}}
    entered = threading.Event()  # the first open has reached its stall, holding whatever av_log holds
    release = threading.Event()  # let the stalled open finish, so its daemon thread (and av_log) can unwind

    class Capture(list):
        def __init__(self, local=True):
            super().__init__()

        def __enter__(self):
            state["captures"][threading.get_ident()] = self
            return self

        def __exit__(self, *exc):
            state["captures"].pop(threading.get_ident(), None)
            return False

    def open_(name, format, options):
        if options.get("list_devices"):
            for line in ["AVFoundation video devices:", "[0] Logi 4K", "[1] USB2.0_CAM1"]:
                state["captures"].get(threading.get_ident(), []).append(("libav", 32, line))
            raise OSError(5, "Input/output error")
        entered.set()
        release.wait(5.0)  # stall inside av.open, as USB2.0_CAM1 does with no matching mode
        raise OSError(5, "Input/output error")

    log = types.ModuleType("av.logging")
    log.ERROR, log.INFO = 16, 32
    log.get_level = lambda: state["level"]
    log.set_level = lambda v: state.__setitem__("level", v)
    log.Capture = Capture
    av = types.ModuleType("av")
    av.open, av.logging = open_, log
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.logging", log)

    def open_first():
        try:
            devices._AVCapture("USB2.0_CAM1", CameraSpec(name="USB2.0_CAM1", width=1280, height=720, fps=60))
        except OSError:
            pass

    grabber = threading.Thread(target=open_first, daemon=True)
    grabber.start()
    assert entered.wait(2.0), "the first open never reached its stall"

    listed: list = []
    lister = threading.Thread(target=lambda: listed.append(cameras._avfoundation()), daemon=True)
    lister.start()
    lister.join(2.0)
    try:
        assert listed, "listing blocked behind the stalled open: the log lock is held across av.open"
        assert listed[0] == [("Logi 4K", 0), ("USB2.0_CAM1", 1)]
    finally:
        release.set()
        grabber.join(2.0)
