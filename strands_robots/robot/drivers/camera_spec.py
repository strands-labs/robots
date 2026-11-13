"""A hardware camera's spec, as ``Robot(cameras=...)`` gives it: a name, an index or path, and an optional mode."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

AV_PIXEL_FORMATS = (  # ffmpeg's avfoundation indev (avf_pixel_formats): what it can hand over, not the USB codec
    "uyvy422", "yuyv422", "nv12", "yuv420p", "bgr0", "0rgb", "rgb0", "0bgr", "rgb24", "bgr24", "gray", "monob",
    "rgb555be", "rgb555le", "rgb565be", "rgb565le", "bgr48be", "yuv444p", "yuva444p", "yuv422p10", "yuv444p10",
    "yuv422p16", "yuva444p16le",
)  # fmt: skip
FIELDS = ("name", "width", "height", "fps", "fourcc", "pixel_format")
_PATH = re.compile(r"^(/|\\\\|[A-Za-z]:\\)")  # a device path, never a name


@dataclass(frozen=True)
class CameraSpec:
    index: int | str | None = None  # a device number or path; None until a name is resolved
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    name: str | None = None  # the device's own name (or a unique part of it)
    fourcc: str | None = "MJPG"  # OpenCV: the codec the device sends (compressed fits USB 2.0); None: its own
    pixel_format: str | None = None  # AVFoundation: the frames' format in memory (``AV_PIXEL_FORMATS``)

    @classmethod
    def of(cls, cam: str, v: Any) -> CameraSpec:
        if isinstance(v, CameraSpec):
            return v
        if isinstance(v, int):
            return cls(v)
        if isinstance(v, str):
            return cls(v) if v.isdigit() or _PATH.match(v) else cls(name=v)
        if isinstance(v, Mapping):
            unknown = set(v) - {"index", "index_or_path", *FIELDS}
            if unknown:
                raise TypeError(f"camera {cam!r}: unknown field(s) {sorted(unknown)}; use index, {', '.join(FIELDS)}")
            idx, name = v.get("index", v.get("index_or_path")), v.get("name")
            if idx is None and name is None:
                raise TypeError(f"camera {cam!r}: needs index= (a device number or path) or name= (the device's name)")
            fmt = {k: v[k] for k in ("fourcc", "pixel_format") if k in v}
            return cls(idx, v.get("width"), v.get("height"), v.get("fps"), name, **fmt).checked(cam)
        raise TypeError(f"camera {cam!r}: give a device name, an index, a path or {{index|name, width, height, fps}}")

    def checked(self, cam: str) -> CameraSpec:
        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(f"camera {cam!r}: fourcc must be 4 characters (e.g. 'MJPG', 'YUYV'), got {self.fourcc!r}")
        if self.pixel_format is not None and self.pixel_format not in AV_PIXEL_FORMATS:
            raise ValueError(
                f"camera {cam!r}: pixel_format {self.pixel_format!r} is not one AVFoundation delivers; "
                f"use one of {', '.join(AV_PIXEL_FORMATS)}"
            )
        return self
