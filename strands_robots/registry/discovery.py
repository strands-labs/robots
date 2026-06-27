"""Auto-discovery of robots from the optional ``robot_descriptions`` package.

The curated registry (``robots.json``) deliberately carries only robots that
need project-specific metadata: hardware ports, custom joint counts, scene
tweaks, aliases, or local mesh overrides. The much larger long tail of standard
robots shipped by ``robot_descriptions`` (MuJoCo Menagerie and friends) does not
need a hand-written entry - this module resolves those on demand so
``Robot("go2", mode="sim")`` works without touching ``robots.json``.

Resolution rules:
    - A curated ``robots.json`` entry always wins. Discovery is consulted only
      for names unknown to the curated registry (see
      :func:`strands_robots.registry.get_robot`).
    - Only MJCF-capable descriptions are discoverable: the MuJoCo backend needs
      an ``.xml`` model, so URDF-only descriptions are skipped.
    - :func:`descriptions_module`, :func:`is_discoverable`, and
      :func:`list_discoverable` are cheap - a static dict lookup with no module
      import and no network. :func:`discover_robot` is heavy: importing a
      description module makes ``robot_descriptions`` clone the upstream asset
      repository on first use, so it is called only from asset-resolution paths
      that are already allowed to download.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Robot names are interpolated into ``importlib`` module paths, so restrict them
# to a conservative allowlist before any lookup - no dots, slashes, or
# whitespace can reach the import machinery.
_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# ``robot_descriptions`` names every MuJoCo (MJCF) description module with this
# suffix, e.g. ``go2_mj_description`` -> canonical robot name ``go2``.
_MJCF_SUFFIX = "_mj_description"

# Candidate scene files (ground plane + lights) a Menagerie description may ship
# alongside the bare robot model, in preference order.
_SCENE_CANDIDATES = ("scene.xml", "scene_mjx.xml")

# Cache of synthesized entries (and negative results) keyed by normalized name.
_DISCOVER_CACHE: dict[str, dict[str, Any] | None] = {}


def _normalize(name: str) -> str:
    """Lowercase, strip, and underscore-normalize a robot name."""
    return name.lower().strip().replace("-", "_")


@lru_cache(maxsize=1)
def _mjcf_modules() -> dict[str, str]:
    """Map canonical robot name -> MJCF description module name.

    Reads ``robot_descriptions``' static description table, which is a plain
    dict - no module import and no network. Returns an empty mapping when
    ``robot_descriptions`` is not installed.
    """
    try:
        from robot_descriptions._descriptions import (  # type: ignore[import-not-found]
            DESCRIPTIONS,
            Format,
        )
    except ImportError:
        return {}

    mapping: dict[str, str] = {}
    for module_name, desc in DESCRIPTIONS.items():
        if Format.MJCF not in desc.formats:
            continue
        if not module_name.endswith(_MJCF_SUFFIX):
            continue
        short = module_name[: -len(_MJCF_SUFFIX)]
        mapping[short] = module_name
    return mapping


def descriptions_module(name: str) -> str | None:
    """Return the MJCF ``robot_descriptions`` module for *name*, or ``None``.

    Cheap: a dict lookup against the static description table, with no module
    import and no network. Returns ``None`` when the robot is not an
    MJCF-capable ``robot_descriptions`` robot or when the package is missing.

    Examples::

        descriptions_module("go2")   # -> "go2_mj_description"
        descriptions_module("so100") # -> None (curated, not a description)
    """
    norm = _normalize(name)
    if not _NAME_RE.match(norm):
        return None
    return _mjcf_modules().get(norm)


def is_discoverable(name: str) -> bool:
    """Return ``True`` if *name* resolves from ``robot_descriptions`` (cheap)."""
    return descriptions_module(name) is not None


def list_discoverable() -> list[str]:
    """Return sorted canonical names resolvable from ``robot_descriptions``.

    This is the MJCF long tail - the standard robots that work in the MuJoCo
    backend without a curated ``robots.json`` entry (e.g. ``go2``, ``spot``,
    ``h1``, ``anymal_c``, ``cassie``). Cheap: no import, no network.
    """
    return sorted(_mjcf_modules())


def discover_robot(name: str) -> dict[str, Any] | None:
    """Synthesize a registry entry for *name* from ``robot_descriptions``.

    Heavy: imports the description module, which makes ``robot_descriptions``
    clone the upstream asset repository on first use. Call only from
    asset-resolution paths that are allowed to download. Results (including
    misses) are cached.

    Args:
        name: Robot name or alias (e.g. ``"go2"``).

    Returns:
        A registry-style entry with the same shape as a ``robots.json`` value -
        an ``asset`` block wired to the resolved ``robot_descriptions`` module -
        or ``None`` if the robot is not an MJCF-capable ``robot_descriptions``
        robot.
    """
    norm = _normalize(name)
    if norm in _DISCOVER_CACHE:
        return _DISCOVER_CACHE[norm]

    module_name = descriptions_module(norm)
    if module_name is None:
        _DISCOVER_CACHE[norm] = None
        return None

    try:
        mod = importlib.import_module(f"robot_descriptions.{module_name}")
    except ImportError as exc:
        logger.debug("Discovery import failed for %r (%s): %s", norm, module_name, exc)
        _DISCOVER_CACHE[norm] = None
        return None

    mjcf_path = getattr(mod, "MJCF_PATH", None)
    package_path = getattr(mod, "PACKAGE_PATH", None)
    if not mjcf_path or not package_path:
        logger.warning(
            "robot_descriptions module %r lacks MJCF_PATH/PACKAGE_PATH; cannot discover %r",
            module_name,
            norm,
        )
        _DISCOVER_CACHE[norm] = None
        return None

    package_dir = str(package_path)
    asset_dir = os.path.basename(os.path.normpath(package_dir))
    model_xml = os.path.basename(str(mjcf_path))

    # Prefer a scene file (ground + lights) when the description ships one;
    # otherwise fall back to the bare model so rendering still has something.
    scene_xml = model_xml
    for candidate in _SCENE_CANDIDATES:
        if os.path.exists(os.path.join(package_dir, candidate)):
            scene_xml = candidate
            break

    entry: dict[str, Any] = {
        "description": f"{norm} (discovered via robot_descriptions:{module_name})",
        "category": "discovered",
        "discovered": True,
        "asset": {
            "dir": asset_dir,
            "model_xml": model_xml,
            "scene_xml": scene_xml,
            "robot_descriptions_module": module_name,
        },
    }
    _DISCOVER_CACHE[norm] = entry
    return entry


def invalidate_cache() -> None:
    """Clear the discovery caches (synthesized entries + module table)."""
    _DISCOVER_CACHE.clear()
    # ``_mjcf_modules`` is normally an ``lru_cache``; guard ``cache_clear`` so a
    # test (or future refactor) that swaps in a plain callable cannot break this.
    clear = getattr(_mjcf_modules, "cache_clear", None)
    if clear is not None:
        clear()
