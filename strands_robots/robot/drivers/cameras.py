"""Hardware cameras, by NAME or by index: ``cameras={"front": "Logi 4K", "wrist": "USB2.0_CAM1"}``.

An index shifts whenever a phone or a screen capture joins the machine; the device's own name does not. A name is
resolved at open against what the OS reports (``list_cameras``: AVFoundation on macOS, V4L2 on Linux), case-insensitive,
by containment; two matches or none are refused naming what is there. A bare index/path and the ``{index, width,
height, fps}`` spec keep working. A camera with no declared size takes the policy's: ``fit`` adopts the sizes a
deployment's hello asks for before the first frame, so the dict's keys are the policy's image keys and nothing else
is written. A camera that delivers another size than asked is refused at the first frame, naming both sizes.
"""

from __future__ import annotations

import platform
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

_PATH = re.compile(r"^(/|\\\\|[A-Za-z]:\\)")  # a device path, never a name


@dataclass(frozen=True)
class CameraSpec:
    index: int | str | None = None  # a device number or path; None until a name is resolved
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    name: str | None = None  # the device's own name (or a unique part of it)

    @classmethod
    def of(cls, cam: str, v: Any) -> CameraSpec:
        if isinstance(v, CameraSpec):
            return v
        if isinstance(v, int):
            return cls(v)
        if isinstance(v, str):
            return cls(v) if v.isdigit() or _PATH.match(v) else cls(name=v)
        if isinstance(v, Mapping):
            unknown = set(v) - {"index", "index_or_path", "name", "width", "height", "fps"}
            if unknown:
                raise TypeError(
                    f"camera {cam!r}: unknown field(s) {sorted(unknown)}; use index, name, width, height, fps"
                )
            idx, name = v.get("index", v.get("index_or_path")), v.get("name")
            if idx is None and name is None:
                raise TypeError(f"camera {cam!r}: needs index= (a device number or path) or name= (the device's name)")
            return cls(idx, v.get("width"), v.get("height"), v.get("fps"), name)
        raise TypeError(f"camera {cam!r}: give a device name, an index, a path or {{index|name, width, height, fps}}")


def list_cameras() -> list[tuple[str, int | str]]:
    """``[(name, index_or_path), ...]`` as the OS reports its video capture devices, in the backend's own order."""
    if platform.system() == "Darwin":
        return _avfoundation()
    if platform.system() == "Linux":
        return _v4l2()
    return []


def _avfoundation() -> list[tuple[str, int | str]]:
    """AVFoundation's device list, in ffmpeg's (and PyAV's) index order: the order ``av.open(name)`` understands."""
    try:
        import av
        import av.logging
    except ImportError as exc:
        raise ImportError("cameras by name on macOS need PyAV: pip install av") from exc
    level = av.logging.get_level()
    if level is None or level < av.logging.INFO:  # PyAV ships with libav's log off; the list is an INFO line each
        av.logging.set_level(av.logging.INFO)
    try:
        with av.logging.Capture(local=False) as logs:
            try:
                av.open("", format="avfoundation", options={"list_devices": "true"})
            except Exception:  # noqa: BLE001  (listing always "fails" to open: the list is in the log)
                pass
    finally:
        av.logging.set_level(level)
    out: list[tuple[str, int | str]] = []
    video = False
    for _, _, msg in logs:
        if "video devices" in msg:
            video = True
        elif "audio devices" in msg:
            video = False
        elif video and (m := re.match(r"\[(\d+)\]\s+(.+)$", msg.strip())):
            out.append((m.group(2), int(m.group(1))))
    return out


def _v4l2(root: Path = Path("/sys/class/video4linux")) -> list[tuple[str, int | str]]:
    """Every V4L2 capture node (``index`` 0 of its device; the metadata node is skipped) with its reported name."""
    out: list[tuple[str, int | str]] = []
    for d in sorted(root.glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            if (d / "index").read_text().strip() not in ("", "0"):
                continue
            out.append(((d / "name").read_text().strip(), f"/dev/{d.name}"))
        except OSError:
            continue
    return out


def resolve(name: str, devices: list[tuple[str, int | str]]) -> tuple[str, int | str]:
    """The one device whose name contains ``name`` (case-insensitive); none or several are refused listing them."""
    hits = [(n, i) for n, i in devices if name.lower() in n.lower()]
    exact = [(n, i) for n, i in hits if n.lower() == name.lower()]
    hits = exact or hits
    have = ", ".join(repr(n) for n, _ in devices) or "none"
    if not hits:
        raise ValueError(f"no camera named {name!r}; the OS reports: {have}")
    if len(hits) > 1:
        raise ValueError(f"camera name {name!r} is ambiguous: {', '.join(repr(n) for n, _ in hits)}; name one of them")
    return hits[0]


class Cameras:
    """The cameras of one hardware body, opened lazily, read as HxWx3 uint8 RGB.

    A named camera on macOS is read through PyAV's AVFoundation input (the backend that takes names); everything else
    through OpenCV. Either way ``read`` returns the latest frame of each camera.
    """

    def __init__(self, cfg: Mapping[str, Any] | None) -> None:
        self.specs = {n: CameraSpec.of(n, v) for n, v in (cfg or {}).items()}
        self.resolved: dict[str, str] = {}  # camera -> the device name it resolved to
        self._caps: dict[str, Any] = {}

    def fit(self, sizes: Mapping[str, tuple[int, int] | None]) -> None:
        """Adopt a policy's camera sizes for every camera that declared none; an open camera reopens at them.

        The rate stays the device's own: a camera read at the policy's (slower) rate would make each read wait a
        whole camera period for a fresh frame."""
        for cam, size in sizes.items():
            s = self.specs.get(cam)
            if s is None or size is None:
                continue
            want = replace(s, width=s.width or size[0], height=s.height or size[1])
            if want != s:
                self.specs[cam] = want
                if (cap := self._caps.pop(cam, None)) is not None:
                    cap.close()

    def _open(self, cam: str) -> Any:
        s = self.specs[cam]
        cap: _AVCapture | _CVCapture
        if s.name is not None and s.index is None:
            name, index = resolve(s.name, list_cameras())
            self.resolved[cam] = name
            cap = _AVCapture(name, s) if platform.system() == "Darwin" else _CVCapture(index, s)
        else:
            cap = _CVCapture(s.index, s)
        self._caps[cam] = cap
        return cap

    def read(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for cam, s in self.specs.items():
            cap = self._caps.get(cam) or self._open(cam)
            rgb = cap.read()
            if rgb is None:
                continue
            h, w = rgb.shape[:2]
            if (s.width is not None and w != s.width) or (s.height is not None and h != s.height):
                raise ValueError(f"camera {cam!r} delivers {w}x{h}, asked for {s.width}x{s.height}")
            out[cam] = rgb
        return out

    def close(self) -> None:
        for cap in self._caps.values():
            cap.close()
        self._caps.clear()


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
