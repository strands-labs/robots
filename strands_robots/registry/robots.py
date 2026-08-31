"""Robot registry - query, resolve, and list robot definitions.

All robot definitions live in robots.json.  This module provides
the public read API; the JSON file is the only thing you edit to add
or modify robots.
"""

import logging
from typing import Any

from .loader import _load

logger = logging.getLogger(__name__)

# Recognised filter values for ``list_robots(mode=...)``. Any other value is
# rejected with a ``ValueError`` rather than silently returning every robot,
# so a typo or an unsupported filter fails loudly instead of yielding a
# misleading unfiltered list (e.g. ``mode="hardware"`` returning sim-only arms).
LIST_ROBOTS_MODES = ("all", "sim", "real", "both")


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
    # Canonical names come straight from the registry keys. Using
    # ``alias_map.values()`` here was wrong: it only contains robots that
    # declare at least one alias, so the 16 alias-less robots (ur5e, reachy2,
    # ...) were treated as unknown and a normalized form like "reachy-2" ->
    # "reachy_2" never resolved to canonical "reachy2".
    canonical_names = set(_load("robots").get("robots", {}))
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized in canonical_names:  # already canonical
        return normalized
    # Fallback: try with all underscores stripped (e.g. "so_100" -> "so100").
    # Only return the stripped form if it actually matches something we know.
    stripped = normalized.replace("_", "")
    if stripped in alias_map:
        return alias_map[stripped]
    if stripped in canonical_names:
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


def has_sim(name: str) -> bool:
    """Check if a robot has simulation assets (MJCF/URDF)."""
    info = get_robot(name)
    return info is not None and "asset" in info


def has_hardware(name: str) -> bool:
    """Check if a robot declares a real-hardware backend.

    Reads the registry entry's ``hardware`` block, which has two independent
    fields: ``lerobot_type`` names a lerobot robot type, and ``driver`` names
    which driver builds the robot. Either alone is a hardware declaration --
    the Reachy Mini and the Microduck declare only a ``driver``, because lerobot
    has no robot type for them at all -- so this reads the block rather than one
    field of it.

    A robot may be drivable without declaring anything: a native driver
    registered through
    :func:`~strands_robots.drivers.register_native_driver` needs no registry
    declaration, and several servo-bus arms are in exactly that position.
    Declaration and registration are two different facts and this predicate
    reports only the first, because it is the one a caller can read without
    importing a driver package.
    :func:`~strands_robots.drivers.list_driver_coverage` joins both and is what
    answers "can this robot be driven for real" completely.

    Args:
        name: Robot name, alias, or data_config.

    Returns:
        True when the robot's registry entry carries a ``hardware`` block,
        False when it does not or the robot is not registered at all.
    """
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


def get_driver(name: str) -> str | None:
    """Get the driver a robot declares, verbatim, or ``None`` if it declares none.

    A pure reader, like its sibling :func:`get_hardware_type`: it reports what
    the registry says and applies no default. ``hardware.driver`` is optional and
    most robots declare none. Where it is declared it says which of two possible
    drivers wins: one declarer has no lerobot robot type at all, so the native
    driver is the only one that can build it; the other has a working lerobot
    type and prefers its native driver anyway.

    An absent declaration therefore means "no preference", not "no native
    driver" - a robot may have one registered and declare nothing, in which case
    the default still routes to lerobot. Deciding what an absent - or ``"auto"``
    - declaration means is
    :func:`~strands_robots.drivers.resolve_driver`'s job, which is also where a
    caller's explicit choice outranks the registry.

    Args:
        name: Robot name, alias, or data_config.

    Returns:
        The declared driver name, or ``None`` when the robot declares no driver
        (or is not registered at all).
    """
    info = get_robot(name)
    if info and "hardware" in info:
        declared: str | None = info["hardware"].get("driver")
        return declared
    return None


def list_robots(mode: str = "all") -> list[dict[str, Any]]:
    """List available robots, optionally filtered.

    Args:
        mode: Filter, one of :data:`LIST_ROBOTS_MODES`:

            - ``"all"``: every registered robot (no filter).
            - ``"sim"``: robots with a simulation asset (``has_sim``).
            - ``"real"``: robots with a hardware backend (``has_hardware``).
            - ``"both"``: robots that have BOTH sim and real.

    Returns:
        List of dicts with name, description, category, joints, has_sim, has_real.

    Raises:
        ValueError: If ``mode`` is not one of :data:`LIST_ROBOTS_MODES`. An
            unrecognized filter is rejected loudly instead of silently
            returning the full, unfiltered list.
    """
    if mode not in LIST_ROBOTS_MODES:
        raise ValueError(f"Unknown list_robots mode {mode!r}. Valid modes: {', '.join(LIST_ROBOTS_MODES)}.")
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
# Preferred display order for the category groups in ``format_robot_table``.
# It is only an ORDERING hint, not an allowlist: categories present in the
# registry but absent here (e.g. a user-registered robot with a custom
# category) are appended afterwards in sorted order so every robot still
# gets a row - the table body must never under-count the footer Total.
_CATEGORY_DISPLAY_ORDER = (
    "arm",
    "bimanual",
    "hand",
    "humanoid",
    "expressive",
    "mobile",
    "mobile_manip",
    "aerial",
)


def format_robot_table(max_width: int = 100) -> str:
    """Human-readable table of all robots for CLI/tool output.

    The ``Sim`` and ``Real`` columns hold the ASCII token ``"yes"`` when the
    robot supports that mode and are left blank otherwise. The output is
    pure ASCII so it aligns correctly in any monospace terminal and is safe
    to embed in logs and tool responses.

    Args:
        max_width: Target terminal width. The ``Description`` column is
            truncated with an ellipsis to fit. Pass a large value (e.g.
            ``1000``) to disable truncation entirely. Default 100 is safe
            for a typical 100-column terminal.

    Returns:
        Multi-line string: a header row, a rule, one row per robot grouped
        by category (common categories first, then any custom categories in
        sorted order), then a totals footer. Every registered robot gets
        exactly one row, so the body row count always matches the footer
        ``Total``.
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
    lines = [header, "-" * rule_width]

    by_cat = list_robots_by_category()
    # Preferred groups first, then any remaining categories in sorted order
    # so no robot is silently dropped from the body (see _CATEGORY_DISPLAY_ORDER).
    ordered_cats = [c for c in _CATEGORY_DISPLAY_ORDER if c in by_cat]
    ordered_cats += sorted(c for c in by_cat if c not in _CATEGORY_DISPLAY_ORDER)
    for cat in ordered_cats:
        for r in by_cat[cat]:
            sim = "yes" if r["has_sim"] else ""
            real = "yes" if r["has_real"] else ""
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
