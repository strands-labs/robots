"""The MuJoCo backend (imports mujoco; loaded lazily by ``load_backend``)."""

from __future__ import annotations

from .backend import MuJoCoBackend

__all__ = ["MuJoCoBackend"]
