"""The camera devices ``Cameras`` reads: OpenCV by index or path, AVFoundation by name (PyAV). Each ``read`` waits
for the device's next frame; ``capture.Grabber`` makes that wait a capture thread's, never the control thread's."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from .camera_spec import CameraSpec


class _CVCapture:
    def __init__(self, index: Any, s: CameraSpec) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("cameras need opencv: pip install opencv-python") from exc
        self.cap = cv2.VideoCapture(index)
        if s.fourcc is not None:  # first: V4L2 lists sizes and rates per codec (a no-op on macOS' AVFoundation)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc))
        for prop, v in ((cv2.CAP_PROP_FRAME_WIDTH, s.width), (cv2.CAP_PROP_FRAME_HEIGHT, s.height),
                        (cv2.CAP_PROP_FPS, s.fps)):  # fmt: skip
            if v is not None:
                self.cap.set(prop, v)

    def read(self) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        return np.ascontiguousarray(bgr[:, :, ::-1]) if ok else None

    def close(self) -> None:
        self.cap.release()


AV_DEFAULT_SIZE = (640, 480)  # what an undeclared, unfitted camera opens at; ``Cameras.fit`` overrides it
AV_DEFAULT_FPS = 30.0
AV_DEFAULT_PIXEL_FORMAT = "uyvy422"  # opened on both bench cameras (2026-10-28); a spec's ``pixel_format`` wins


def av_options(s: CameraSpec) -> dict[str, str]:
    """ffmpeg avfoundation options for a spec: ALWAYS a size and a rate.

    Measured on two USB cameras (Logi 4K Pro, USB2.0_CAM1, macOS 26): without ``framerate`` the indev fails every
    open with ``OSError(5, 'Input/output error')`` whatever the size; with a rate but no size it asks for the device's
    native mode, which hangs ``av.open`` (USB2.0_CAM1, 1920x1080 uyvy @30 is not a mode it has) or costs the full
    sensor (Logi: 4096x2160). A declared or fitted size wins; a camera that declared none opens at ``AV_DEFAULT_SIZE``.
    The indev picks the device's mode (and with it the USB codec, MJPEG or raw) by size and rate alone;
    ``pixel_format`` is only the frames' format in memory, so ``nv12`` (12 bits a pixel) copies less than ``uyvy422``.
    """
    w, h = (s.width, s.height) if s.width and s.height else AV_DEFAULT_SIZE
    fmt = s.pixel_format or AV_DEFAULT_PIXEL_FORMAT
    return {"pixel_format": fmt, "video_size": f"{w}x{h}", "framerate": f"{s.fps or AV_DEFAULT_FPS:g}"}


def refused(name: str, options: dict[str, str], exc: OSError, log: list[str]) -> OSError:
    """The open's error naming the mode asked and the modes the device has (the indev logs them, PyAV drops them)."""
    asked = f"{options['video_size']} @ {options['framerate']} fps ({options['pixel_format']})"
    modes = sorted({_mode(m) for m in log if _MODE.match(m)}, key=lambda m: tuple(map(float, re.findall(r"[\d.]+", m))))
    why = next((m.strip() for m in log if "not supported" in m), str(exc))
    have = ", ".join(modes) if modes else "none listed"
    return OSError(exc.errno, f"camera {name!r} refused {asked}: {why} Its modes: {have}")


_MODE = re.compile(r"\s*(\d+)x(\d+)@\[([\d.]+) ([\d.]+)\]fps")  # the indev's "  1280x720@[1.000000 30.000000]fps"


def _mode(line: str) -> str:
    w, h, _, top = _MODE.match(line).groups()  # type: ignore[union-attr]
    return f"{w}x{h}@{float(top):g}fps"


class _AVCapture:
    """AVFoundation by device name through PyAV; ffmpeg's indev keeps only the latest frame, so a read is fresh."""

    def __init__(self, name: str, s: CameraSpec) -> None:
        import av
        import av.logging

        options = av_options(s)
        level = av.logging.get_level()
        if level is None or level < av.logging.ERROR:  # the indev's "Supported modes" lines are ERROR lines
            av.logging.set_level(av.logging.ERROR)
        try:
            with av.logging.Capture(local=False) as logs:
                try:
                    self.container = av.open(name, format="avfoundation", options=options)
                except OSError as exc:
                    raise refused(name, options, exc, [msg for _, _, msg in logs]) from exc
        finally:
            av.logging.set_level(level)
        self.stream = self.container.streams.video[0]

    def read(self, wait_s: float = 0.5) -> np.ndarray | None:
        """The next frame; the indev answers EAGAIN until one is ready, so poll for up to ``wait_s``.

        A fresh ``demux`` generator per attempt: one that raised EAGAIN is finished and would yield nothing after."""
        import av

        deadline = time.monotonic() + wait_s
        while time.monotonic() <= deadline:
            try:
                for packet in self.container.demux(self.stream):
                    for frame in packet.decode():
                        if isinstance(frame, av.VideoFrame):
                            return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            except av.BlockingIOError:
                time.sleep(0.001)
        return None

    def close(self) -> None:
        self.container.close()
