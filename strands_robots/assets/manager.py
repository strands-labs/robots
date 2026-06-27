"""Robot Asset Manager for Strands Robots Simulation.

Resolves robot model files (MJCF XML) from:
    1. ``STRANDS_ASSETS_DIR`` env var (user override)
    2. User cache (``~/.strands_robots/assets/``)
    3. ``robot_descriptions`` package (MuJoCo Menagerie)
    4. Project-local ``./assets/``
"""

import logging
import os
from pathlib import Path

from strands_robots.registry import (
    get_robot,
    list_robots,
)
from strands_robots.registry import (
    resolve_name as resolve_robot_name,
)
from strands_robots.utils import get_search_paths, safe_join

logger = logging.getLogger(__name__)

# Module-level conditional import - keeps manager.py importable in
# environments where the optional ``robot_descriptions`` package (and its
# transitive heavyweight deps like ``GitPython``) are not installed.
# When ``download`` is not available, auto-download simply returns False.
try:
    from .download import auto_download_robot as _auto_download_robot_impl
except ImportError:
    _auto_download_robot_impl = None  # type: ignore[assignment]


def _lookup(name: str) -> dict | None:
    """Resolve a registry entry for *name*, with ``robot_descriptions`` fallback.

    A curated ``robots.json`` entry always wins. When a name is unknown to the
    curated registry, fall back to :func:`strands_robots.registry.discovery`,
    which synthesizes an entry for any MJCF-capable ``robot_descriptions`` robot
    (the long tail: ``go2``, ``spot``, ``h1``, ...). Discovery imports the
    description module, which can trigger an asset download on first use, so
    this helper is used only by download-capable resolvers - never by the
    side-effect-free presence check.
    """
    info = get_robot(name)
    if info is not None:
        return info
    from strands_robots.registry.discovery import discover_robot

    return discover_robot(name)


#
# Model path resolution (delegates to registry)
#


def _auto_download_robot(name: str, info: dict) -> bool:
    """Delegate to :func:`strands_robots.assets.download.auto_download_robot`.

    Returns ``False`` immediately when the download module is unavailable
    (e.g. ``robot_descriptions`` not installed).
    """
    if _auto_download_robot_impl is None:
        logger.warning("Auto-download unavailable: install strands-robots[sim-mujoco] for automatic asset downloads")
        return False
    return _auto_download_robot_impl(name, info)


_MESH_EXTS = frozenset({".stl", ".obj", ".msh", ".ply"})

# Cache of (directory, mtime) -> has_meshes result. Avoids re-walking the tree
# when ``resolve_model_path`` checks multiple candidate locations for the same
# robot and when it re-checks after auto-download.
_MESH_CACHE: dict[tuple[str, float], bool] = {}


def _has_meshes(directory: Path) -> bool:
    """Check if a directory tree contains mesh files (cached, early-exit).

    Uses ``os.scandir`` with an early break on the first mesh found rather
    than ``rglob("*")``, which stats every file. Result is cached per
    (directory, mtime) so repeated calls are free.
    """
    if not directory.exists():
        return False
    try:
        cache_key = (str(directory), directory.stat().st_mtime)
    except OSError:
        cache_key = (str(directory), 0.0)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _walk(path: str) -> bool:
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in _MESH_EXTS:
                            return True
                    elif entry.is_dir(follow_symlinks=False) and _walk(entry.path):
                        return True
        except OSError:
            return False
        return False

    result = _walk(str(directory))
    _MESH_CACHE[cache_key] = result
    return result


def _resolve_candidates(asset_dir_name: str, xml_file: str, name: str) -> list[Path]:
    """Resolve candidate paths for a robot XML, with path-traversal protection.

    Uses ``safe_join`` to prevent ``../`` in registry-sourced ``asset_dir_name``
    or ``xml_file`` from escaping the search directories.
    """
    candidates: list[Path] = []
    for search_dir in get_search_paths():
        try:
            model_path = safe_join(search_dir, f"{asset_dir_name}/{xml_file}")
        except ValueError:
            logger.warning("Path traversal attempt blocked for robot: %s", name)
            return []
        if model_path.exists():
            candidates.append(model_path)
    return candidates


def is_robot_asset_present(name: str) -> bool:
    """Check whether a robot's model XML exists on disk without triggering downloads.

    Pure filesystem check - no auto-download, no mesh walk, no network.
    Use this for status queries (e.g. ``download_assets(action="status")``)
    where you need to quickly check presence without side effects.

    Args:
        name: Robot name (canonical or alias).

    Returns:
        True if the model XML file exists on at least one search path.
    """
    info = get_robot(name)
    if not info or "asset" not in info:
        return False

    asset = info["asset"]
    xml_file: str = str(asset["model_xml"])
    asset_dir_name: str = str(asset["dir"])

    # Check user-registered path first
    user_path = info.get("_user_asset_path")
    if user_path:
        try:
            user_model = safe_join(Path(user_path), xml_file)
            if user_model.exists():
                return True
        except ValueError:
            pass

    # Check standard search paths
    for search_dir in get_search_paths():
        try:
            model_path = safe_join(search_dir, f"{asset_dir_name}/{xml_file}")
            if model_path.exists():
                return True
        except ValueError:
            continue

    return False


def resolve_model_path(
    name: str,
    prefer_scene: bool = False,
) -> Path | None:
    """Resolve a robot name to its MJCF model XML path.

    Looks up the robot in ``registry/robots.json``, then searches
    the asset directories for the actual file.  If XML is found but
    mesh files are missing, automatically downloads them via
    ``robot_descriptions`` before returning.

    Args:
        name: Robot name (canonical or alias).
        prefer_scene: If True, return scene XML (with ground/lights)
                      instead of bare model XML.

    Returns:
        Path to the MJCF XML file, or None if not found.

    Examples::

        resolve_model_path("so100")             # → .../trs_so_arm100/so_arm100.xml
        resolve_model_path("so100", prefer_scene=True)  # → .../trs_so_arm100/scene.xml
        resolve_model_path("franka")            # → .../franka_emika_panda/panda.xml
    """
    info = _lookup(name)
    if not info or "asset" not in info:
        # DEBUG, not WARNING: resolve_model() probes several candidate names
        # (incl. suffix-stripped variants) and handles a None return cleanly,
        # so a miss here is normal control flow -- not something the user
        # needs to see on every add_robot.
        logger.debug("Unknown robot or no asset: %s", name)
        return None

    asset = info["asset"]
    # Explicit str() casts: dict subscript returns Any, but Path / Any → Any
    xml_file: str = str(asset["scene_xml"] if prefer_scene else asset["model_xml"])
    asset_dir_name: str = str(asset["dir"])

    candidates: list[Path] = []

    # Check user-registered asset path first (highest priority).
    # ``xml_file`` comes from user_robots.json, so we still gate it through
    # :func:`safe_join` to block path traversal even for user-authored entries
    # (defense in depth - protects against a compromised user_robots.json and
    # keeps the trust boundary identical to the built-in registry path).
    user_path = info.get("_user_asset_path")
    if user_path:
        try:
            user_model = safe_join(Path(user_path), xml_file)
        except ValueError:
            logger.warning(
                "Path traversal blocked in _user_asset_path for %s: %r",
                name,
                xml_file,
            )
            user_model = None
        if user_model is not None and user_model.exists():
            candidates.append(user_model)

    # Search standard paths with traversal protection
    candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates:
        # No XML found at all - try auto-download, then re-search
        logger.info("No XML found for %s, attempting auto-download...", name)
        if _auto_download_robot(name, info):
            candidates.extend(_resolve_candidates(asset_dir_name, xml_file, name))

    if not candidates:
        logger.warning("Robot model not found: %s → %s/%s", name, asset_dir_name, xml_file)
        return None

    # Prefer the candidate whose directory contains mesh files,
    # because an XML without meshes will fail to load in MuJoCo.
    for path in candidates:
        if _has_meshes(path.parent):
            logger.debug("Resolved %s → %s (has meshes)", name, path)
            return Path(path)

    # XML found but no meshes - auto-download and re-check
    logger.info("XML found for %s but no meshes, attempting auto-download...", name)
    if _auto_download_robot(name, info):
        # Re-scan after download (new symlinks may have appeared)
        refreshed = _resolve_candidates(asset_dir_name, xml_file, name)
        for path in refreshed:
            if _has_meshes(path.parent):
                logger.debug("Resolved %s → %s (auto-downloaded)", name, path)
                return Path(path)

    # Final fallback: return first candidate (some robots have no meshes)
    logger.debug("Resolved %s → %s (no meshes available)", name, candidates[0])
    return Path(candidates[0])


def resolve_model_dir(name: str) -> Path | None:
    """Resolve a robot name to its asset directory (containing XML + meshes).

    Args:
        name: Robot name (canonical or alias).

    Returns:
        Path to the robot's asset directory, or None if not found.
    """
    info = _lookup(name)
    if not info or "asset" not in info:
        return None

    asset_dir: str = str(info["asset"]["dir"])
    for search_dir in get_search_paths():
        try:
            dir_path = safe_join(search_dir, asset_dir)
        except ValueError:
            logger.warning("Path traversal attempt blocked in resolve_model_dir: %s", asset_dir)
            return None
        if dir_path.exists():
            return Path(dir_path)
    return None


def get_robot_info(name: str) -> dict | None:
    """Get information about a robot model.

    Args:
        name: Robot name (canonical or alias).

    Returns:
        Dict with description, category, joints, asset info, etc.
    """
    info = _lookup(name)
    if info is None:
        return None
    result = dict(info)
    result["canonical_name"] = resolve_robot_name(name)
    path = resolve_model_path(name)
    result["resolved_path"] = str(path) if path else None
    result["available"] = path is not None
    return result


def list_available_robots() -> list[dict]:
    """List all available robot models with their info.

    Uses :func:`is_robot_asset_present` for a fast filesystem-only check
    per robot instead of the heavier :func:`resolve_model_path` which can
    trigger auto-downloads and mesh cache walks.

    Returns:
        List of dicts with name, description, joints, category, available, path.
    """
    robots = []
    for r in list_robots(mode="sim"):
        name = r["name"]
        present = is_robot_asset_present(name)
        info = get_robot(name) or {}
        # Only resolve full path when asset is present - avoids download attempts
        path = resolve_model_path(name) if present else None
        robots.append(
            {
                "name": name,
                "description": r.get("description", ""),
                "joints": r.get("joints"),
                "category": r.get("category", ""),
                "dir": info.get("asset", {}).get("dir", ""),
                "available": present,
                "path": str(path) if path else None,
            }
        )
    return robots
