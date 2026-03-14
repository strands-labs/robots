"""Robot model resolution — URDF registry + Menagerie asset manager."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default URDF search paths (checked in order)
_URDF_SEARCH_PATHS = [
    Path.cwd() / "urdfs",
    Path.cwd() / "assets" / "urdfs",
    Path.cwd() / "robots",
    Path.home() / ".strands_robots" / "urdfs",
    Path("/opt/strands_robots/urdfs"),
]

try:
    from strands_robots.assets import (  # noqa: I001
        format_robot_table as _format_robot_table,
        resolve_model_path as _resolve_menagerie_model,
    )

    _HAS_ASSET_MANAGER = True
except ImportError:
    _HAS_ASSET_MANAGER = False

# Legacy URDF registry — runtime cache for user-registered URDFs
_URDF_REGISTRY: Dict[str, str] = {}

_URDF_DIR_OVERRIDE = os.getenv("STRANDS_URDF_DIR")
if _URDF_DIR_OVERRIDE:
    _URDF_SEARCH_PATHS.insert(0, Path(_URDF_DIR_OVERRIDE))


def register_urdf(data_config: str, urdf_path: str):
    """Register a URDF/MJCF file for a data_config name."""
    _URDF_REGISTRY[data_config] = urdf_path
    logger.info("📋 Registered model for '%s': %s", data_config, urdf_path)


def resolve_model(name: str, prefer_scene: bool = True) -> Optional[str]:
    """Resolve a robot name or data_config to an MJCF/URDF model path.

    Resolution order:
    1. Asset manager (32 bundled robots + 40 aliases)
    2. Legacy URDF registry (custom registrations)
    3. URDF search paths (STRANDS_URDF_DIR, ./urdfs, etc.)
    """
    if _HAS_ASSET_MANAGER:
        path = _resolve_menagerie_model(name, prefer_scene=prefer_scene)
        if path and path.exists():
            return str(path)
        if prefer_scene:
            path = _resolve_menagerie_model(name, prefer_scene=False)
            if path and path.exists():
                return str(path)

    return resolve_urdf(name)


def resolve_urdf(data_config: str) -> Optional[str]:
    """Resolve a data_config name to a URDF file path (legacy)."""
    if data_config in _URDF_REGISTRY:
        urdf_rel = _URDF_REGISTRY[data_config]
        if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
            return urdf_rel
        for search_dir in _URDF_SEARCH_PATHS:
            candidate = search_dir / urdf_rel
            if candidate.exists():
                return str(candidate)

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
    return {config_name: resolve_urdf(config_name) for config_name in _URDF_REGISTRY}


def list_available_models() -> str:
    """List all available robot models (Menagerie + custom)."""
    if _HAS_ASSET_MANAGER:
        return _format_robot_table()

    lines = ["Registered URDFs:"]
    for name, path in _URDF_REGISTRY.items():
        resolved = resolve_urdf(name)
        status = "✅" if resolved else "❌"
        lines.append(f"  {status} {name}: {path}")
    return "\n".join(lines)
