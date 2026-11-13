"""The camera devices ``Cameras`` reads: OpenCV by index or path, AVFoundation by name (PyAV). Each ``read`` waits
for the device's next frame; ``capture.Grabber`` makes that wait a capture thread's, never the control thread's."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cameras import CameraSpec


class _CVCapture:
    def __init__(self, index: Any, s: CameraSpec) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("cameras need opencv: pip install opencv-python") from exc
        self.cap = cv2.VideoCapture(index)
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


def av_options(s: CameraSpec) -> dict[str, str]:
    """ffmpeg avfoundation options for a spec: ALWAYS a size and a rate.

    Measured on two USB cameras (Logi 4K Pro, USB2.0_CAM1, macOS 26): without ``framerate`` the indev fails every
    open with ``OSError(5, 'Input/output error')`` whatever the size; with a rate but no size it asks for the device's
    native mode, which hangs ``av.open`` (USB2.0_CAM1, 1920x1080 uyvy @30 is not a mode it has) or costs the full
    sensor (Logi: 4096x2160). A declared or fitted size wins; a camera that declared none opens at ``AV_DEFAULT_SIZE``.
    """
    w, h = (s.width, s.height) if s.width and s.height else AV_DEFAULT_SIZE
    return {"pixel_format": "uyvy422", "video_size": f"{w}x{h}", "framerate": f"{s.fps or AV_DEFAULT_FPS:g}"}


class _AVCapture:
    """AVFoundation by device name through PyAV; ffmpeg's indev keeps only the latest frame, so a read is fresh."""

    def __init__(self, name: str, s: CameraSpec) -> None:
        import av

        self.container = av.open(name, format="avfoundation", options=av_options(s))
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
