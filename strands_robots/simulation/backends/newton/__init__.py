"""MuJoCo-Warp as the solver of the MuJoCo world (``physics="newton"``); imports mujoco_warp at use."""

from .backend import NewtonBackend

__all__ = ["NewtonBackend"]
