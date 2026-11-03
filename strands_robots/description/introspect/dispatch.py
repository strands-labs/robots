"""Joints, actuators, limits, keyframes of a model file: MuJoCo if available, else an XML walk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .xml import introspect_xml


def introspect(path: Path, fmt: str, scene: Path | None = None) -> dict[str, Any]:
    """Read a model file's facts. A file that only compiles inside its scene (LeKiwi's wheel
    contacts name the scene's floor) is read through the scene."""
    from ..._gl import ensure_gl

    ensure_gl()
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return introspect_xml(path, fmt)
    from .mujoco import introspect_mujoco

    try:
        return introspect_mujoco(path)
    except ValueError:
        if scene is None:
            raise
        return introspect_mujoco(scene)
