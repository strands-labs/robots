"""JSON registry loader with mtime-based hot-reload and validation.

Loads robots.json and policies.json from the registry directory,
re-reading only when the file's mtime changes.  Validates uniqueness of
aliases, shorthands, and URL patterns on every reload.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_DIR = Path(__file__).parent
_cache: dict[str, dict] = {}
_mtimes: dict[str, float] = {}


def _load(name: str) -> dict:
    """Load a JSON registry file, re-reading only when mtime changes.

    Args:
        name: Base name without extension (e.g. "robots", "policies").

    Returns:
        Parsed JSON as a dict.
    """
    path = _REGISTRY_DIR / f"{name}.json"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        logger.error("Registry file not found: %s", path)
        return {}

    if name not in _cache or _mtimes.get(name) != mtime:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _validate(name, data)
        _cache[name] = data
        _mtimes[name] = mtime
        logger.debug("Loaded registry: %s (%d bytes)", path, path.stat().st_size)

    return _cache[name]


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
    """Ensure no two providers share the same alias, shorthand, or URL pattern."""
    seen_aliases: dict[str, str] = {}
    seen_url_patterns: dict[str, str] = {}

    for provider_name, info in data.get("providers", {}).items():
        for alias in info.get("aliases", []):
            if alias in seen_aliases:
                raise ValueError(
                    f"Duplicate policy alias '{alias}': claimed by both '{seen_aliases[alias]}' and '{provider_name}'"
                )
            seen_aliases[alias] = provider_name

        for shorthand in info.get("shorthands", []):
            if shorthand in seen_aliases:
                raise ValueError(
                    f"Duplicate policy shorthand '{shorthand}': claimed by both "
                    f"'{seen_aliases[shorthand]}' and '{provider_name}'"
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
