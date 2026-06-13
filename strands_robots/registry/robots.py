"""Robot registry - query, resolve, and list robot definitions.

All robot definitions live in robots.json.  This module provides
the public read API; the JSON file is the only thing you edit to add
or modify robots.
"""

import logging
from typing import Any

from .loader import _load

logger = logging.getLogger(__name__)


def _build_alias_map() -> dict[str, str]:
    """Build alias → canonical name mapping from robot entries.

    Each robot entry may have an "aliases" list.  This function
    inverts those into a flat lookup dict.
    """
    reg = _load("robots")
    alias_map: dict[str, str] = {}
    for name, info in reg.get("robots", {}).items():
        for alias in info.get("aliases", []):
            alias_map[alias] = name
    return alias_map


def resolve_name(name: str) -> str:
    """Resolve a robot name or alias to the canonical name.

    Args:
        name: Any robot name, alias, or data_config string.

    Returns:
        Canonical robot name (e.g. "so100", "panda", "unitree_g1").

    Examples::

        resolve_name("franka")        # → "panda"
        resolve_name("SO100_follower") # → "so100"
        resolve_name("g1")            # → "unitree_g1"
    """
    normalized = name.lower().strip().replace("-", "_")
    alias_map = _build_alias_map()
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized in alias_map.values():  # already canonical
        return normalized
    # Fallback: try with all underscores stripped (e.g. "so_100" -> "so100").
    # Only return the stripped form if it actually matches something we know.
    stripped = normalized.replace("_", "")
    if stripped in alias_map:
        return alias_map[stripped]
    if stripped in alias_map.values():
        return stripped
    return normalized


def get_robot(name: str) -> dict[str, Any] | None:
    """Get full robot definition by name or alias.

    Args:
        name: Robot name, alias, or data_config.

    Returns:
        Robot dict with keys like description, category, joints, asset,
        hardware - or None if not found.
    """
    reg = _load("robots")
    canonical = resolve_name(name)
    result: dict[str, Any] | None = reg.get("robots", {}).get(canonical)
    return result


# Discoverability alias (#405 / AX-2): autonomous agents reach for
# ``get_robot_info`` by analogy with ``get_robot_state`` and burn a call on
# ``ImportError: cannot import name 'get_robot_info'`` before finding
# ``get_robot``. Expose the intuitive name as a thin alias of ``get_robot`` so
# both spellings resolve to the same definition lookup. (Distinct from
# ``strands_robots.assets.get_robot_info``, which enriches the result with
# resolved asset paths -- the registry alias is the pure definition lookup.)
get_robot_info = get_robot


def has_sim(name: str) -> bool:
    """Check if a robot has simulation assets (MJCF/URDF)."""
    info = get_robot(name)
    return info is not None and "asset" in info


def has_hardware(name: str) -> bool:
    """Check if a robot has real hardware support (LeRobot type)."""
    info = get_robot(name)
    return info is not None and "hardware" in info


def get_hardware_type(name: str) -> str | None:
    """Get the LeRobot hardware type for a robot.

    Returns:
        LeRobot type string (e.g. "so100_follower"), or None.
    """
    info = get_robot(name)
    if info and "hardware" in info:
        hw_type: str | None = info["hardware"].get("lerobot_type")
        return hw_type
    return None


def list_robots(mode: str = "all") -> list[dict[str, Any]]:
    """List available robots, optionally filtered.

    Args:
        mode: Filter - "all", "sim", "real", or "both" (has sim AND real).

    Returns:
        List of dicts with name, description, has_sim, has_real.
    """
    reg = _load("robots")
    results = []
    for name, info in sorted(reg.get("robots", {}).items()):
        _has_sim = "asset" in info
        _has_real = "hardware" in info

        if mode == "sim" and not _has_sim:
            continue
        if mode == "real" and not _has_real:
            continue
        if mode == "both" and not (_has_sim and _has_real):
            continue

        results.append(
            {
                "name": name,
                "description": info.get("description", ""),
                "category": info.get("category", ""),
                "joints": info.get("joints"),
                "has_sim": _has_sim,
                "has_real": _has_real,
            }
        )
    return results


def list_robots_by_category() -> dict[str, list[dict[str, Any]]]:
    """List robots grouped by category (arm, humanoid, mobile, ...)."""
    categories: dict[str, list] = {}
    for robot in list_robots():
        cat = robot.get("category", "other")
        categories.setdefault(cat, []).append(robot)
    return categories


def list_aliases() -> dict[str, str]:
    """Return the full alias → canonical mapping."""
    return _build_alias_map()


_NAME_WIDTH = 20
_CAT_WIDTH = 15
_JOINTS_WIDTH = 8
_SIM_WIDTH = 5
_REAL_WIDTH = 5
# Width of the fixed prefix columns, including single-space separators.
_FIXED_PREFIX_WIDTH = _NAME_WIDTH + 1 + _CAT_WIDTH + 1 + _JOINTS_WIDTH + 1 + _SIM_WIDTH + 1 + _REAL_WIDTH + 1


def format_robot_table(max_width: int = 100) -> str:
    """Human-readable table of all robots for CLI/tool output.

    Args:
        max_width: Target terminal width. The ``Description`` column is
            truncated with an ellipsis to fit. Pass a large value (e.g.
            ``1000``) to disable truncation entirely. Default 100 is safe
            for a typical 100-column terminal.
    """
    desc_width = max(20, max_width - _FIXED_PREFIX_WIDTH)

    header = (
        f"{'Name':<{_NAME_WIDTH}} "
        f"{'Category':<{_CAT_WIDTH}} "
        f"{'Joints':<{_JOINTS_WIDTH}} "
        f"{'Sim':<{_SIM_WIDTH}} "
        f"{'Real':<{_REAL_WIDTH}} "
        f"Description"
    )
    rule_width = min(max(max_width, len(header)), _FIXED_PREFIX_WIDTH + desc_width)
    lines = [header, "─" * rule_width]

    for cat in ["arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip", "aerial"]:
        by_cat = list_robots_by_category()
        for r in by_cat.get(cat, []):
            sim = "✅" if r["has_sim"] else "  "
            real = "✅" if r["has_real"] else "  "
            joints = str(r["joints"]) if r["joints"] else "?"
            desc = r["description"] or ""
            if len(desc) > desc_width:
                desc = desc[: desc_width - 3].rstrip() + "..."
            lines.append(
                f"{r['name']:<{_NAME_WIDTH}} "
                f"{r['category']:<{_CAT_WIDTH}} "
                f"{joints:<{_JOINTS_WIDTH}} "
                f"{sim:<{_SIM_WIDTH}} "
                f"{real:<{_REAL_WIDTH}} "
                f"{desc}"
            )

    robots = list_robots()
    lines.append("")
    lines.append(f"Total: {len(robots)} robots | Aliases: {len(list_aliases())}")
    return "\n".join(lines)
