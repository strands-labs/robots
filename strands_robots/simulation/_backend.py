"""MuJoCo lazy import and GL backend configuration."""

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)

_mujoco = None
_mujoco_viewer = None


def _is_headless() -> bool:
    """Detect if running in a headless environment (no display server).

    Returns True on Linux when no DISPLAY or WAYLAND_DISPLAY is set,
    which means GLFW-based rendering will fail.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def _configure_gl_backend() -> None:
    """Auto-configure MuJoCo's OpenGL backend for headless environments.

    MuJoCo reads MUJOCO_GL at import time to select the OpenGL backend:
    - "egl"    → EGL (GPU-accelerated offscreen, requires libEGL + NVIDIA driver)
    - "osmesa" → OSMesa (CPU software rendering, slower but always works)
    - "glfw"   → GLFW (default, requires X11/Wayland display server)

    This function MUST be called before `import mujoco`. Setting MUJOCO_GL
    after import has no effect — the backend is locked at import time.

    Never overrides a user-set MUJOCO_GL value.
    """
    if os.environ.get("MUJOCO_GL"):
        logger.debug(f"MUJOCO_GL already set to '{os.environ['MUJOCO_GL']}', respecting user config")
        return

    if not _is_headless():
        return

    # Headless Linux — probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        logger.info("Headless environment detected — using MUJOCO_GL=egl (GPU-accelerated offscreen)")
        return
    except OSError:
        pass

    try:
        ctypes.cdll.LoadLibrary("libOSMesa.so")
        os.environ["MUJOCO_GL"] = "osmesa"
        logger.info("Headless environment detected — using MUJOCO_GL=osmesa (CPU software rendering)")
        return
    except OSError:
        pass

    logger.warning(
        "Headless environment detected but neither EGL nor OSMesa found. "
        "MuJoCo rendering will likely fail. Install one of:\n"
        "  GPU: apt-get install libegl1-mesa-dev  (or NVIDIA driver provides libEGL)\n"
        "  CPU: apt-get install libosmesa6-dev\n"
        "Then set: export MUJOCO_GL=egl  (or osmesa)"
    )


def _ensure_mujoco():
    """Lazy import MuJoCo to avoid hard dependency.

    Auto-configures the OpenGL backend for headless environments before
    importing mujoco, since MUJOCO_GL must be set at import time.
    """
    global _mujoco, _mujoco_viewer
    if _mujoco is None:
        _configure_gl_backend()
        try:
            import mujoco

            _mujoco = mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for simulation. Install with:\n"
                "  pip install strands-robots[sim]\n"
                "Or: pip install mujoco"
            )
    if _mujoco_viewer is None and not _is_headless():
        try:
            import mujoco.viewer as viewer

            _mujoco_viewer = viewer
        except ImportError:
            pass
    return _mujoco
