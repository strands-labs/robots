"""User-local robot registry - runtime registration without editing package JSON.

Provides ``register_robot()`` and ``unregister_robot()`` for adding custom
robots that persist across sessions via a ``user_robots.json`` file stored
alongside the asset cache.

File location (in priority order):
    1. ``$STRANDS_BASE_DIR/user_robots.json``
    2. ``~/.strands_robots/user_robots.json``

Note:
    ``STRANDS_ASSETS_DIR`` only controls where *assets* live, not the
    user registry. Use ``STRANDS_BASE_DIR`` to relocate user metadata.

At load time the user overlay is merged *on top of* the package
``robots.json`` - user entries win on name collision, so you can also
override built-in robots locally.

Usage::

    from strands_robots.registry import register_robot, unregister_robot

    # Register a custom robot with MJCF
    register_robot(
        name="my_arm",
        model_xml="my_arm.xml",
        description="My custom 6-DOF arm",
        category="arm",
        joints=6,
        asset_dir="my_arm",  # resolved relative to assets dir
    )

    # Now works everywhere:
    from strands_robots.simulation import create_simulation
    sim = create_simulation()
    sim.create_world()
    sim.add_robot("my_arm")   # auto-resolved

    # Remove it
    unregister_robot("my_arm")
"""

import json
import logging
from pathlib import Path
from typing import Any

from strands_robots.utils import get_base_dir, resolve_asset_path

from .loader import invalidate_cache

logger = logging.getLogger(__name__)


def _get_user_registry_path() -> Path:
    """Get path to the user-local robot registry file."""
    return get_base_dir() / "user_robots.json"


def user_registry_source() -> bytes | None:
    """Contents of the user-local robot registry file.

    Returns:
        The file's bytes, or ``None`` when the overlay does not exist yet or
        cannot be read. This is what the loader keys its hot-reload cache on, so
        an external edit to ``user_robots.json`` (a second process or a manual
        edit) invalidates the merged ``robots`` cache without requiring a manual
        ``invalidate_cache`` - including an edit that lands inside one
        filesystem timestamp tick, which a stat cannot distinguish.
    """
    try:
        return _get_user_registry_path().read_bytes()
    except OSError:
        return None


def parse_user_robots(source: bytes | None) -> dict[str, Any]:
    """Robot definitions held in raw user-overlay bytes.

    Args:
        source: Contents of ``user_robots.json``, or None when the overlay is
            absent. Taken as bytes so a caller that keys a cache on the file's
            contents parses exactly what it keyed on.

    Returns:
        Dict mapping robot names to their definitions; empty when the overlay
        is absent, unreadable, or does not declare ``"robots"``.
    """
    if source is None:
        return {}
    try:
        data = json.loads(source)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to load user registry %s: %s", _get_user_registry_path(), exc)
        return {}
    robots = data.get("robots") if isinstance(data, dict) else None
    return robots if isinstance(robots, dict) else {}


def _load_user_registry() -> dict[str, Any]:
    """Load the user-local robot registry file.

    Returns:
        Dict with ``"robots"`` key mapping names to robot definitions.
    """
    return {"robots": parse_user_robots(user_registry_source())}


def _save_user_registry(data: dict[str, Any]) -> None:
    """Save the user-local robot registry file."""
    path = _get_user_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    logger.info("Saved user registry: %s (%d robots)", path, len(data.get("robots", {})))


def get_user_robots() -> dict[str, Any]:
    """Get all user-registered robots.

    Returns:
        Dict mapping robot names to their definitions.
    """
    return parse_user_robots(user_registry_source())


def register_robot(
    name: str,
    *,
    model_xml: str,
    description: str = "",
    category: str = "arm",
    joints: int = 0,
    asset_dir: str | None = None,
    scene_xml: str | None = None,
    aliases: list[str] | None = None,
    robot_descriptions_module: str | None = None,
    hardware: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register a custom robot in the user-local registry.

    .. warning:: Security

        This function is a **library-only** API and must NOT be exposed
        as an agent @tool without additional safeguards.  A malicious
        agent could register a robot pointing to attacker-controlled MJCF
        that executes code via MuJoCo plugins.  If tool exposure is needed
        in the future, gate it behind STRANDS_TRUST_REMOTE_CODE and
        validate all paths with :func:`strands_robots.utils.safe_join`.

    The robot becomes immediately available in ``get_robot()``,
    ``list_robots()``, ``resolve_model_path()``, ``sim.add_robot()``, etc.

    Args:
        name: Canonical robot name (lowercase, underscores).
        model_xml: Path to MJCF/URDF model file, relative to ``asset_dir``.
        description: Human-readable description.
        category: Robot category (arm, humanoid, mobile, hand, aerial, bimanual, ...).
        joints: Number of actuated joints.
        asset_dir: Directory containing the model file and meshes.
            - Absolute path: used as-is (``~/`` expanded).
            - Relative path: resolved against the assets directory
              (``STRANDS_ASSETS_DIR`` or ``~/.strands_robots/assets/``).
            - None: defaults to ``<assets_dir>/<name>/``.
        scene_xml: Scene XML (with ground/lights). Defaults to ``model_xml``.
        aliases: Alternative names for this robot.
        robot_descriptions_module: Optional ``robot_descriptions`` module name.
        hardware: Optional hardware config dict (``lerobot_type``, etc.).
        overwrite: If False (default), raises ValueError if robot already exists.

    Returns:
        The registered robot definition dict.

    Raises:
        ValueError: If name already exists and ``overwrite`` is False, or if an
            ``aliases`` entry collides with an existing canonical robot name or
            another robot's alias (the same constraint the loader enforces at
            read time, checked here so the registration cannot brick lookups).
        FileNotFoundError: If ``model_xml`` doesn't exist at the resolved path.

    Example::

        register_robot(
            name="my_arm",
            model_xml="my_arm.xml",
            asset_dir="~/robots/my_arm_v2",
            description="Custom 6-DOF arm with gripper",
            category="arm",
            joints=7,
            aliases=["myarm", "custom_arm"],
        )
    """
    # Normalize name
    name = name.lower().strip().replace("-", "_")

    # Load existing
    data = _load_user_registry()

    # Check for existing (in user registry AND package registry)
    if not overwrite:
        if name in data.get("robots", {}):
            raise ValueError(f"Robot '{name}' already in user registry. Use overwrite=True to replace.")
        # Also check package registry
        try:
            from .robots import get_robot as _pkg_get_robot

            if _pkg_get_robot(name) is not None:
                logger.info(
                    "Robot '%s' exists in package registry - user registration will override it.",
                    name,
                )
        except ImportError:
            pass

    # Resolve asset_dir via shared utility (respects STRANDS_ASSETS_DIR)
    resolved_dir = resolve_asset_path(asset_dir, default_name=name)

    # Use the directory name as the asset "dir" key (relative to search paths)
    # This matches how resolve_model_path works: search_dir / asset["dir"] / xml
    dir_name = resolved_dir.name

    # Validate model_xml exists.  Previously we only checked when
    # ``resolved_dir`` existed - which silently accepted registrations for
    # dirs that didn't exist yet and surfaced a confusing error only at
    # ``add_robot()`` time.  Now we fail-closed on both conditions so the
    # user gets an immediate, actionable error at registration time.
    model_path = resolved_dir / model_xml
    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"Asset directory does not exist: {resolved_dir}\n"
            f"Create the directory and place '{model_xml}' inside it before registering."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model XML not found: {model_path}\nEnsure '{model_xml}' exists in '{resolved_dir}'")

    # Build entry
    entry: dict[str, Any] = {
        "description": description,
        "category": category,
        "joints": joints,
        "asset": {
            "dir": dir_name,
            "model_xml": model_xml,
            "scene_xml": scene_xml or model_xml,
        },
    }

    if robot_descriptions_module:
        entry["asset"]["robot_descriptions_module"] = robot_descriptions_module

    if aliases:
        entry["aliases"] = aliases

    if hardware:
        entry["hardware"] = hardware

    # Store the full resolved path so the asset manager can find it
    # even if the dir isn't in the standard search paths
    entry["_user_asset_path"] = str(resolved_dir)

    # Save
    data.setdefault("robots", {})[name] = entry

    # Fail-closed: refuse to persist an entry the loader would reject on the
    # next read (an alias that collides with a canonical name or another
    # robot's alias). Without this the registration "succeeds" but every
    # subsequent get_robot/resolve_name raises ValueError process-wide until
    # user_robots.json is hand-edited.
    _assert_registry_still_loads(data)

    _save_user_registry(data)

    # Invalidate loader cache so next get_robot() picks up the merge
    _invalidate_cache()

    logger.info("Registered robot '%s' -> %s/%s", name, dir_name, model_xml)
    return entry


def unregister_robot(name: str) -> bool:
    """Remove a robot from the user-local registry.

    Does not affect the package ``robots.json``. If the robot exists
    only in the package registry, this is a no-op.

    Args:
        name: Robot name to remove.

    Returns:
        True if the robot was removed, False if it wasn't in the user registry.
    """
    name = name.lower().strip().replace("-", "_")
    data = _load_user_registry()

    if name not in data.get("robots", {}):
        logger.info("Robot '%s' not in user registry - nothing to remove.", name)
        return False

    del data["robots"][name]
    _save_user_registry(data)
    _invalidate_cache()

    logger.info("Unregistered robot '%s'", name)
    return True


def list_user_robots() -> list[dict[str, Any]]:
    """List all user-registered robots.

    Returns:
        List of dicts with name, description, category, path info.
    """
    robots = get_user_robots()
    result = []
    for name, info in sorted(robots.items()):
        result.append(
            {
                "name": name,
                "description": info.get("description", ""),
                "category": info.get("category", ""),
                "joints": info.get("joints", 0),
                "asset_dir": info.get("_user_asset_path", ""),
                "model_xml": info.get("asset", {}).get("model_xml", ""),
            }
        )
    return result


def _assert_registry_still_loads(data: dict[str, Any]) -> None:
    """Reject a registration the loader would refuse to read.

    Runs the loader's own uniqueness validator on the prospective merged
    registry (package ``robots.json`` overlaid with the pending user entries),
    mirroring exactly what :func:`loader._merge_user_robots` +
    :func:`loader._validate_robots` do on every read. An invalid registration
    therefore fails loudly at write time instead of being persisted and then
    breaking every subsequent ``get_robot``/``resolve_name`` call.

    Args:
        data: The prospective user-registry dict (already containing the new
            entry) to validate against the merged registry.

    Raises:
        ValueError: If the merged registry would violate a uniqueness
            constraint (e.g. an alias colliding with a canonical name or
            another robot's alias).
    """
    from .loader import _REGISTRY_DIR, _validate_robots

    pkg_path = _REGISTRY_DIR / "robots.json"
    try:
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pkg = {}

    merged = dict(pkg.get("robots", {}))
    merged.update(data.get("robots", {}))
    _validate_robots({"robots": merged})


def _invalidate_cache() -> None:
    """Invalidate the loader cache so merged data is reloaded."""
    invalidate_cache("robots")
