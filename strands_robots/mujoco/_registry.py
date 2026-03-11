"""MuJoCo lazy loading, GL backend configuration, and URDF/model registry."""

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import mujoco — only needed when simulation is actually used
_mujoco = None
_mujoco_viewer = None


def _is_headless() -> bool:
    """Detect if running in a headless environment (no display server).

    Returns True on Linux when no DISPLAY or WAYLAND_DISPLAY is set,
    which means GLFW-based rendering will fail.
    """
    if sys.platform != "linux":
        return False  # macOS has CGL, Windows has WGL — always available
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
        logger.debug(
            "MUJOCO_GL already set to '%s', respecting user config",
            os.environ["MUJOCO_GL"],
        )
        return

    if not _is_headless():
        return  # Display available, GLFW will work fine

    # Headless Linux — probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)

    # Try EGL (fast, GPU-accelerated)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        logger.info(
            "Headless environment detected — using MUJOCO_GL=egl (GPU-accelerated offscreen)"
        )
        return
    except OSError:
        pass

    # Try OSMesa (CPU software rendering)
    try:
        ctypes.cdll.LoadLibrary("libOSMesa.so")
        os.environ["MUJOCO_GL"] = "osmesa"
        logger.info(
            "Headless environment detected — using MUJOCO_GL=osmesa (CPU software rendering)"
        )
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
        # CRITICAL: Configure GL backend BEFORE importing mujoco.
        # MuJoCo reads MUJOCO_GL at import time and locks the backend.
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
    if _mujoco_viewer is None:
        try:
            import mujoco.viewer as viewer

            _mujoco_viewer = viewer
        except ImportError:
            pass  # Viewer is optional — headless mode works fine
    return _mujoco


def get_mujoco_viewer():
    """Return the cached mujoco.viewer module, or None if unavailable."""
    _ensure_mujoco()
    return _mujoco_viewer


# ===================================================================
# URDF Registry — Maps data_config names to URDF paths
# ===================================================================

# Default URDF search paths (checked in order)
_URDF_SEARCH_PATHS = [
    Path.cwd() / "urdfs",
    Path.cwd() / "assets" / "urdfs",
    Path.cwd() / "robots",
    Path.home() / ".strands_robots" / "urdfs",
    Path("/opt/strands_robots/urdfs"),
]

# ─────────────────────────────────────────────────────────────────────
# Robot Model Resolution (MJCF + URDF) — delegates to unified registry
# ─────────────────────────────────────────────────────────────────────

try:
    from strands_robots.assets import (  # noqa: I001
        format_robot_table as _format_robot_table,
        list_available_robots as _list_menagerie_robots,
        resolve_model_path as _resolve_menagerie_model,
    )
    _HAS_ASSET_MANAGER = True
except ImportError:
    _HAS_ASSET_MANAGER = False
    _format_robot_table = None
    _list_menagerie_robots = None
    _resolve_menagerie_model = None

# Legacy URDF registry — now just a runtime cache for user-registered
# URDFs via register_urdf().  Builtin legacy_urdf paths are in robots.json.
_URDF_REGISTRY: Dict[str, str] = {}

# Allow overrides from environment
_URDF_DIR_OVERRIDE = os.getenv("STRANDS_URDF_DIR")
if _URDF_DIR_OVERRIDE:
    _URDF_SEARCH_PATHS.insert(0, Path(_URDF_DIR_OVERRIDE))


def register_urdf(data_config: str, urdf_path: str):
    """Register a URDF/MJCF file for a data_config name.

    Args:
        data_config: Data config name (e.g., "so100", "my_custom_robot")
        urdf_path: Absolute path or relative filename to the URDF/MJCF file
    """
    _URDF_REGISTRY[data_config] = urdf_path
    logger.info("📋 Registered model for '%s': %s", data_config, urdf_path)


def resolve_model(name: str, prefer_scene: bool = True) -> Optional[str]:
    """Resolve a robot name or data_config to an MJCF/URDF model path.

    Resolution order:
    1. Asset manager (32 bundled robots + 40 aliases)
    2. Legacy URDF registry (custom registrations)
    3. URDF search paths (STRANDS_URDF_DIR, ./urdfs, etc.)

    Args:
        name: Robot name, data_config, or alias (e.g. "so100", "panda", "unitree_g1")
        prefer_scene: If True, prefer scene.xml (includes ground plane, lighting, cameras)
                      over bare model.xml. Default True for simulation use.

    Returns:
        Absolute path to model file, or None if not found
    """
    # 1. Try asset manager (Menagerie MJCF models — preferred)
    if _HAS_ASSET_MANAGER:
        # Prefer scene.xml which includes ground plane, lighting, cameras
        path = _resolve_menagerie_model(name, prefer_scene=prefer_scene)
        if path and path.exists():
            return str(path)
        # Fallback: try without scene preference
        if prefer_scene:
            path = _resolve_menagerie_model(name, prefer_scene=False)
            if path and path.exists():
                return str(path)

    # 2. Try legacy URDF registry
    return resolve_urdf(name)


def resolve_urdf(data_config: str) -> Optional[str]:
    """Resolve a data_config name to a URDF file path (legacy).

    Checks runtime-registered URDFs first, then falls back to the
    ``legacy_urdf`` field in ``registry/robots.json``.

    Args:
        data_config: Data config name

    Returns:
        Absolute path to URDF file, or None if not found
    """
    # 1. Runtime-registered URDFs (via register_urdf())
    if data_config in _URDF_REGISTRY:
        urdf_rel = _URDF_REGISTRY[data_config]
        if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
            return urdf_rel
        for search_dir in _URDF_SEARCH_PATHS:
            candidate = search_dir / urdf_rel
            if candidate.exists():
                return str(candidate)

    # 2. Check registry robots.json for legacy_urdf
    try:
        from strands_robots.registry import get_robot, resolve_name
        canonical = resolve_name(data_config)
        info = get_robot(canonical)
        if info and "legacy_urdf" in info:
            urdf_rel = info["legacy_urdf"]
            if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
                return urdf_rel
            for search_dir in _URDF_SEARCH_PATHS:
                candidate = search_dir / urdf_rel
                if candidate.exists():
                    return str(candidate)
    except ImportError:
        pass

    logger.debug("URDF not found for '%s' in search paths", data_config)
    return None


def list_registered_urdfs() -> Dict[str, Optional[str]]:
    """List all registered URDF mappings and their resolved paths."""
    result = {}
    for config_name in _URDF_REGISTRY:
        result[config_name] = resolve_urdf(config_name)
    return result


def list_available_models() -> str:
    """List all available robot models (Menagerie + custom).

    Returns:
        Formatted table string of all robots
    """
    if _HAS_ASSET_MANAGER:
        return _format_robot_table()

    # Fallback: legacy registry only
    lines = ["Registered URDFs:"]
    for name, path in _URDF_REGISTRY.items():
        resolved = resolve_urdf(name)
        status = "✅" if resolved else "❌"
        lines.append(f"  {status} {name}: {path}")
    return "\n".join(lines)
