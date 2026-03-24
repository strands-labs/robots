"""Robot registry — query, resolve, and list robot definitions.

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
    return alias_map.get(normalized, normalized)


def get_robot(name: str) -> dict[str, Any] | None:
    """Get full robot definition by name or alias.

    Args:
        name: Robot name, alias, or data_config.

    Returns:
        Robot dict with keys like description, category, joints, asset,
        hardware — or None if not found.
    """
    reg = _load("robots")
    canonical = resolve_name(name)
    result: dict[str, Any] | None = reg.get("robots", {}).get(canonical)
    return result


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
        mode: Filter — "all", "sim", "real", or "both" (has sim AND real).

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


def format_robot_table() -> str:
    """Human-readable table of all robots for CLI/tool output."""
    lines = [
        f"{'Name':<20} {'Category':<15} {'Joints':<8} {'Sim':<5} {'Real':<5} Description",
        "─" * 100,
    ]
    for cat in ["arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip"]:
        by_cat = list_robots_by_category()
        for r in by_cat.get(cat, []):
            sim = "✅" if r["has_sim"] else "  "
            real = "✅" if r["has_real"] else "  "
            joints = str(r["joints"]) if r["joints"] else "?"
            lines.append(f"{r['name']:<20} {r['category']:<15} {joints:<8} {sim:<5} {real:<5} {r['description']}")

    robots = list_robots()
    lines.append("")
    lines.append(f"Total: {len(robots)} robots | Aliases: {len(list_aliases())}")
    return "\n".join(lines)
