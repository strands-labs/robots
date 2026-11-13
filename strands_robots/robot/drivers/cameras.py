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
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .capture import Grabber, stop_all
from .devices import _AVCapture, _CVCapture
from .devices import av_options as av_options  # the spec -> AVFoundation options, where callers have read it

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
    """The cameras of one hardware body, read as HxWx3 uint8 RGB: one capture thread each (``capture.Grabber``).

    ``open`` starts every camera at once and waits for each first frame; ``Robot`` calls it at connect (``make_driver``)
    and ``read`` calls it for a body used bare. ``read`` then returns each camera's latest frame without waiting on a
    device; ``stamps`` holds when each returned frame was captured (``time.monotonic()``), and a camera whose latest
    frame is older than ``max_age_s`` (it stopped delivering) is left out, as a failed read is. A named camera on
    macOS is read through PyAV's AVFoundation input (the backend that takes names); everything else through OpenCV.
    """

    def __init__(self, cfg: Mapping[str, Any] | None, *, max_age_s: float = 1.0, open_timeout_s: float = 10.0) -> None:
        self.specs = {n: CameraSpec.of(n, v) for n, v in (cfg or {}).items()}
        self.resolved: dict[str, str] = {}  # camera -> the device name it resolved to
        self.stamps: dict[str, float] = {}  # camera -> capture time of the frame ``read`` last returned
        self.max_age_s, self.open_timeout_s = max_age_s, open_timeout_s
        self._caps: dict[str, Grabber] = {}
        self._lock = threading.RLock()  # one caller at a time starts, reads, refits or stops the capture threads
        weakref.finalize(self, stop_all, self._caps)  # a body dropped unclosed releases its devices

    def fit(self, sizes: Mapping[str, tuple[int, int] | None]) -> None:
        """Adopt a policy's camera sizes for every camera that declared none; an open camera reopens at them.

        The rate stays the device's own: the capture thread reads at it, and the policy takes the latest frame."""
        with self._lock:
            for cam, size in sizes.items():
                s = self.specs.get(cam)
                if s is None or size is None:
                    continue
                want = replace(s, width=s.width or size[0], height=s.height or size[1])
                if want != s:
                    self.specs[cam] = want
                    if (cap := self._caps.pop(cam, None)) is not None:
                        cap.stop()

    def _open(self, cam: str) -> Grabber:
        s = self.specs[cam]
        if s.name is not None and s.index is None:  # resolved here, on the caller's thread: a bad name fails at once
            name, index = resolve(s.name, list_cameras())
            self.resolved[cam] = name
            mac = platform.system() == "Darwin"
            grabber = Grabber(cam, lambda: _AVCapture(name, s) if mac else _CVCapture(index, s))
        else:
            grabber = Grabber(cam, lambda: _CVCapture(s.index, s))
        self._caps[cam] = grabber
        return grabber

    def open(self) -> None:
        """Start every camera not yet started (their opens run side by side), then wait for each one's first frame."""
        with self._lock:
            try:
                started = [self._open(cam) for cam in self.specs if cam not in self._caps]
                for grabber in started:
                    grabber.wait(self.open_timeout_s)
            except BaseException:
                self.close()
                raise

    def read(self) -> dict[str, np.ndarray]:
        with self._lock:
            self.open()
            out: dict[str, np.ndarray] = {}
            now = time.monotonic()
            for cam, s in self.specs.items():
                got = self._caps[cam].latest()
                if got is None or now - got[1] > self.max_age_s:
                    continue
                rgb, self.stamps[cam] = got
                h, w = rgb.shape[:2]
                if (s.width is not None and w != s.width) or (s.height is not None and h != s.height):
                    raise ValueError(f"camera {cam!r} delivers {w}x{h}, asked for {s.width}x{s.height}")
                out[cam] = rgb
            return out

    def ages(self) -> dict[str, float]:
        """Seconds since each frame ``read`` last returned was captured."""
        now = time.monotonic()
        return {cam: now - t for cam, t in self.stamps.items()}

    def close(self) -> None:
        with self._lock:
            for cap in self._caps.values():
                cap.stop()
            self._caps.clear()
