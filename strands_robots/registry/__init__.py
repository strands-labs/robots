"""Unified Registry - single source of truth for robots and policies.

Loads robot definitions and policy provider configs from JSON files.

Features:
    - **One file to edit**: Add a robot -> edit robots.json, done.
    - **Hot-reload**: JSON is re-read when the file changes (mtime check).
    - **Self-contained entries**: Each robot/policy owns its aliases,
      shorthands, and URL patterns - no separate lookup tables.
    - **Validation**: Duplicate aliases, shorthands, and URL patterns
      are caught on load with clear error messages.

Usage::

    from strands_robots.registry import get_robot, resolve_name, list_robots
    from strands_robots.registry import get_policy_provider, resolve_policy

    info = get_robot("so100")
    name = resolve_name("franka")  # -> "panda"
    providers = list_policy_providers()

Architecture (this package re-exports the public API of its sibling modules):
    - :mod:`~strands_robots.registry.loader` - JSON loading + mtime hot-reload
      + validation.
    - :mod:`~strands_robots.registry.robots` - robot query/resolve/list functions.
    - :mod:`~strands_robots.registry.policies` - policy resolve/import/kwargs
      functions.
    - :mod:`~strands_robots.registry.discovery` - robot_descriptions
      auto-discovery of installed robot models.
    - :mod:`~strands_robots.registry.user_registry` - user-defined robot
      register/unregister persisted to disk.
    - ``robots.json`` - robot definitions (aliases inside each entry).
    - ``policies.json`` - policy providers (shorthands/urls inside each entry).
"""

from .discovery import (
    descriptions_module,
    discover_robot,
    is_discoverable,
    list_discoverable,
)
from .loader import invalidate_cache, reload
from .policies import (
    build_policy_kwargs,
    get_policy_provider,
    import_policy_class,
    list_policy_aliases,
    list_policy_providers,
    resolve_policy,
)
from .robots import (
    LIST_ROBOTS_MODES,
    format_robot_table,
    get_driver,
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    list_aliases,
    list_robots,
    list_robots_by_category,
    resolve_name,
)
from .user_registry import (
    list_user_robots,
    parse_user_robots,
    register_robot,
    unregister_robot,
    user_registry_source,
)

__all__ = [
    # Robot registry
    "resolve_name",
    "get_robot",
    "has_sim",
    "has_hardware",
    "get_driver",
    "get_hardware_type",
    "list_robots",
    "LIST_ROBOTS_MODES",
    "list_robots_by_category",
    "list_aliases",
    "format_robot_table",
    # robot_descriptions auto-discovery (long tail)
    "descriptions_module",
    "is_discoverable",
    "list_discoverable",
    "discover_robot",
    # Policy registry
    "get_policy_provider",
    "list_policy_providers",
    "list_policy_aliases",
    "resolve_policy",
    "import_policy_class",
    "build_policy_kwargs",
    # User-local registry
    "register_robot",
    "unregister_robot",
    "list_user_robots",
    "user_registry_source",
    "parse_user_robots",
    # Utilities
    "reload",
    "invalidate_cache",
]
