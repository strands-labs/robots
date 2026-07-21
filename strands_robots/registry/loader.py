"""JSON registry loader with mtime-based hot-reload and validation.

Loads robots.json and policies.json from the registry directory,
re-reading only when the on-disk source changes.  Validates uniqueness of
aliases, shorthands, and URL patterns on every reload.

The ``robots`` registry is not a single file: its effective contents are the
package ``robots.json`` merged with the user-local overlay
(``$STRANDS_BASE_DIR/user_robots.json`` - see :func:`_merge_user_robots`).  The
hot-reload signature therefore tracks the mtimes of *both* files, so an edit to
the user overlay made outside this process (a second process, a manual edit, or
any writer that does not call :func:`invalidate_cache`) is picked up on the next
read - honoring the "re-read when the source changes" contract for the overlay
just as for the package file.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_DIR = Path(__file__).parent
_cache: dict[str, dict] = {}
# Cache-validity signature per registry (a tuple of mtimes).  For ``robots``
# the signature is (package_mtime, user_overlay_mtime_or_None); for every other
# registry it is (package_mtime,).
_mtimes: dict[str, tuple] = {}


def _user_registry_mtime() -> float | None:
    """Modification time of the user-local robot overlay, or None if absent.

    Kept in :mod:`user_registry` so the overlay path has a single source of
    truth; imported lazily to avoid an import cycle (``user_registry`` imports
    :func:`invalidate_cache` from this module).
    """
    try:
        from .user_registry import user_registry_mtime
    except ImportError:
        return None
    return user_registry_mtime()


def _registry_signature(name: str, pkg_mtime: float) -> tuple:
    """Cache-validity signature for a registry.

    The ``robots`` registry merges the user overlay on top of the package JSON,
    so its signature includes the overlay's mtime - otherwise an external edit
    to ``user_robots.json`` would never invalidate the cached merge.
    """
    if name != "robots":
        return (pkg_mtime,)
    return (pkg_mtime, _user_registry_mtime())


def _load(name: str) -> dict:
    """Load a JSON registry file, re-reading only when its source changes.

    Args:
        name: Base name without extension (e.g. "robots", "policies").

    Returns:
        Parsed JSON as a dict.
    """
    path = _REGISTRY_DIR / f"{name}.json"
    try:
        pkg_mtime = path.stat().st_mtime
    except FileNotFoundError:
        logger.error("Registry file not found: %s", path)
        return {}

    signature = _registry_signature(name, pkg_mtime)
    if name not in _cache or _mtimes.get(name) != signature:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Merge user-local robot registry (overlay on top of package JSON)
        if name == "robots":
            data = _merge_user_robots(data)

        _validate(name, data)
        _cache[name] = data
        _mtimes[name] = signature
        logger.debug("Loaded registry: %s (%d bytes)", path, path.stat().st_size)

    return _cache[name]


def _merge_user_robots(data: dict) -> dict:
    """Merge user-local robot registry on top of package robots.json.

    User entries override package entries on name collision.
    """
    try:
        from .user_registry import get_user_robots
    except ImportError:
        return data

    user_robots = get_user_robots()
    if not user_robots:
        return data

    merged = dict(data)
    merged_robots = dict(merged.get("robots", {}))
    merged_robots.update(user_robots)
    merged["robots"] = merged_robots

    logger.debug("Merged %d user-registered robot(s) into registry", len(user_robots))
    return merged


def _validate(name: str, data: dict) -> None:
    """Validate uniqueness constraints after loading a registry file.

    Raises:
        ValueError: On duplicate aliases, shorthands, or URL patterns.
    """
    if name == "robots":
        _validate_robots(data)
    elif name == "policies":
        _validate_policies(data)


def _validate_robots(data: dict) -> None:
    """Ensure no two robots share the same alias."""
    seen_aliases: dict[str, str] = {}
    for robot_name, info in data.get("robots", {}).items():
        for alias in info.get("aliases", []):
            if alias in seen_aliases:
                raise ValueError(
                    f"Duplicate robot alias '{alias}': claimed by both '{seen_aliases[alias]}' and '{robot_name}'"
                )
            if alias in data.get("robots", {}):
                raise ValueError(f"Robot alias '{alias}' in '{robot_name}' collides with a canonical robot name")
            seen_aliases[alias] = robot_name


def _validate_policies(data: dict) -> None:
    """Ensure no two providers share the same alias, shorthand, or URL pattern.

    Also rejects an alias or shorthand that collides with a *different*
    provider's canonical name. ``get_policy_provider`` resolves a lookup key
    through the alias/shorthand map before falling back to the canonical name
    (``alias_map.get(name, name)``), so such a collision silently shadows the
    real provider - its own name would resolve to the alias owner instead.
    A provider naming *itself* in its shorthands is allowed and idiomatic
    (every provider lists its canonical name as a shorthand so the bare name
    resolves), hence the ``!= provider_name`` guard.
    """
    seen_aliases: dict[str, str] = {}
    seen_url_patterns: dict[str, str] = {}
    providers = data.get("providers", {})

    for provider_name, info in providers.items():
        for alias in info.get("aliases", []):
            if alias in seen_aliases:
                raise ValueError(
                    f"Duplicate policy alias '{alias}': claimed by both '{seen_aliases[alias]}' and '{provider_name}'"
                )
            if alias != provider_name and alias in providers:
                raise ValueError(f"Policy alias '{alias}' in '{provider_name}' collides with a canonical provider name")
            seen_aliases[alias] = provider_name

        for shorthand in info.get("shorthands", []):
            if shorthand in seen_aliases:
                raise ValueError(
                    f"Duplicate policy shorthand '{shorthand}': claimed by both "
                    f"'{seen_aliases[shorthand]}' and '{provider_name}'"
                )
            if shorthand != provider_name and shorthand in providers:
                raise ValueError(
                    f"Policy shorthand '{shorthand}' in '{provider_name}' collides with a canonical provider name"
                )
            seen_aliases[shorthand] = provider_name

        for pattern in info.get("url_patterns", []):
            if pattern in seen_url_patterns:
                raise ValueError(
                    f"Duplicate URL pattern '{pattern}': claimed by both "
                    f"'{seen_url_patterns[pattern]}' and '{provider_name}'"
                )
            seen_url_patterns[pattern] = provider_name


def reload() -> None:
    """Force-reload all registry files (clears mtime cache)."""
    _cache.clear()
    _mtimes.clear()


def invalidate_cache(name: str | None = None) -> None:
    """Invalidate cached registry data, forcing a reload on next access.

    Args:
        name: Registry name to invalidate (e.g. "robots"). If None, clears all.
    """
    if name is None:
        _cache.clear()
        _mtimes.clear()
    else:
        _cache.pop(name, None)
        _mtimes.pop(name, None)
