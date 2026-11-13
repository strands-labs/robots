"""A hardware camera's spec, as ``Robot(cameras=...)`` gives it: a name, an index or path, and an optional mode."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
