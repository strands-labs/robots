"""Hardware cameras by the device's own name: resolved against what the OS lists, sized by the policy."""

import numpy as np
import pytest

from strands_robots.robot.drivers import cameras as mod
from strands_robots.robot.drivers.cameras import Cameras, CameraSpec, resolve

DEVICES = [("USB2.0_CAM1", 0), ("Logi 4K Pro", 1), ("neon_net Camera", 2), ("neon_net Desk View Camera", 3)]


def test_a_string_is_a_name_unless_it_is_a_path_or_a_number():
    assert CameraSpec.of("front", "Logi 4K") == CameraSpec(name="Logi 4K")
    assert CameraSpec.of("front", "/dev/video2") == CameraSpec("/dev/video2")
    assert CameraSpec.of("front", "1") == CameraSpec("1") and CameraSpec.of("front", 1) == CameraSpec(1)
    assert CameraSpec.of("w", {"name": "USB2.0_CAM1", "width": 640, "height": 480}) == CameraSpec(
        None, 640, 480, None, "USB2.0_CAM1"
    )
    assert CameraSpec.of("w", {"index": 1, "width": 640, "height": 480, "fps": 30}) == CameraSpec(1, 640, 480, 30)
    with pytest.raises(TypeError, match="needs index= .* or name="):
        CameraSpec.of("x", {"width": 640})
    with pytest.raises(TypeError, match="use index, name, width, height, fps"):
        CameraSpec.of("x", {"name": "a", "w": 1})


def test_a_name_resolves_by_containment_and_refuses_none_or_several_naming_what_is_there():
    assert resolve("Logi 4K", DEVICES) == ("Logi 4K Pro", 1)
    assert resolve("usb2.0_cam1", DEVICES) == ("USB2.0_CAM1", 0)
    assert resolve("neon_net Camera", DEVICES) == ("neon_net Camera", 2)  # an exact name beats a longer one
    with pytest.raises(ValueError, match=r"no camera named 'Brio'; the OS reports: 'USB2.0_CAM1', 'Logi 4K Pro'"):
        resolve("Brio", DEVICES)
    with pytest.raises(ValueError, match=r"'neon_net' is ambiguous: 'neon_net Camera', 'neon_net Desk View Camera'"):
        resolve("neon_net", DEVICES)
    with pytest.raises(ValueError, match="the OS reports: none"):
        resolve("Logi", [])


class _Fake:
    opened: list = []

    def __init__(self, index, spec):
        self.index, self.spec, self.closed = index, spec, False
        _Fake.opened.append(self)

    def read(self):
        w, h = self.spec.width or 1280, self.spec.height or 720
        return np.zeros((h, w, 3), np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    _Fake.opened = []
    monkeypatch.setattr(mod, "list_cameras", lambda: DEVICES)
    monkeypatch.setattr(mod, "_CVCapture", _Fake)
    monkeypatch.setattr(mod, "_AVCapture", lambda name, spec: _Fake(name, spec))
    return _Fake


def test_named_cameras_open_at_the_policys_size_and_the_keys_are_the_policys_image_keys(fake):
    c = Cameras({"front": "Logi 4K", "wrist": "USB2.0_CAM1"})
    c.fit({"front": (640, 480), "wrist": (640, 480), "top": (64, 48)})  # a key the body has not is nobody's concern
    frames = c.read()
    assert {k: v.shape for k, v in frames.items()} == {"front": (480, 640, 3), "wrist": (480, 640, 3)}
    assert c.resolved == {"front": "Logi 4K Pro", "wrist": "USB2.0_CAM1"}
    assert c.specs["front"] == CameraSpec(None, 640, 480, None, "Logi 4K")


def test_a_declared_size_wins_and_a_fit_after_open_reopens_only_what_changed(fake):
    c = Cameras({"front": {"name": "Logi 4K", "width": 1280, "height": 720}, "wrist": "USB2.0_CAM1"})
    c.read()
    first = list(fake.opened)
    c.fit({"front": (640, 480), "wrist": (640, 480)})
    assert c.specs["front"].width == 1280 and c.specs["wrist"] == CameraSpec(None, 640, 480, None, "USB2.0_CAM1")
    assert [f.closed for f in first] == [False, True]  # the wrist reopened at the policy's size, the front kept
    c.read()
    assert len(fake.opened) == 3


def test_a_camera_that_delivers_another_size_is_refused_naming_both(fake, monkeypatch):
    monkeypatch.setattr(_Fake, "read", lambda self: np.zeros((360, 640, 3), np.uint8))
    c = Cameras({"wrist": {"name": "USB2.0_CAM1", "width": 640, "height": 480}})
    with pytest.raises(ValueError, match=r"camera 'wrist' delivers 640x360, asked for 640x480"):
        c.read()


def test_an_unknown_name_is_refused_at_open_listing_the_devices(fake):
    with pytest.raises(ValueError, match=r"no camera named 'Brio'"):
        Cameras({"front": "Brio"}).read()


def test_the_v4l2_list_skips_metadata_nodes(tmp_path):
    for n, (name, index) in enumerate([("Logi 4K Pro", "0"), ("Logi 4K Pro", "1"), ("USB2.0_CAM1", "0")]):
        d = tmp_path / f"video{n}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(index + "\n")
    assert mod._v4l2(tmp_path) == [("Logi 4K Pro", "/dev/video0"), ("USB2.0_CAM1", "/dev/video2")]


def test_avfoundation_always_gets_a_size_and_a_rate_and_a_declared_one_wins():
    # ffmpeg's avfoundation indev: no framerate -> OSError EIO on every open, a rate with no size -> the device's native
    # mode (a hang on USB2.0_CAM1, 4096x2160 on a Logi 4K Pro). Measured on the bench, 2026-10-28.
    assert mod.av_options(CameraSpec(name="Logi 4K")) == {
        "pixel_format": "uyvy422",
        "video_size": "640x480",
        "framerate": "30",
    }
    assert mod.av_options(CameraSpec(name="Logi 4K", width=1280, height=720, fps=15)) == {
        "pixel_format": "uyvy422",
        "video_size": "1280x720",
        "framerate": "15",
    }
    fitted = Cameras({"front": "Logi 4K"})
    fitted.fit({"front": (320, 240)})
    assert mod.av_options(fitted.specs["front"])["video_size"] == "320x240"
