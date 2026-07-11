"""Simulation factory - create_simulation() and runtime backend registration.

Mirrors the policy factory pattern: JSON-driven defaults with runtime
override capability. Backends are lazy-loaded on first use.

Usage::

    from strands_robots.simulation import create_simulation

    # Default backend (MuJoCo)
    sim = create_simulation()

    # Explicit backend
    sim = create_simulation("mujoco", timestep=0.001)

    # GPU-native built-in backends
    sim = create_simulation("isaac", num_envs=1, headless=True)
    sim = create_simulation("newton")

    # Custom backend (runtime-registered)
    from strands_robots.simulation.factory import register_backend
    register_backend("my_sim", lambda: MySimBackend, aliases=["custom"])
    sim = create_simulation("custom")

Third-party packages may also register backends out-of-tree via the
``strands_robots.backends`` entry-point group (see ``create_simulation``).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

# Entry-point group third-party packages declare to register simulation
# backends without patching this module. See ``_load_plugin_backends``.
_ENTRY_POINT_GROUP = "strands_robots.backends"

# Built-in backend registry (lazy loaders - no imports at module load)

_BUILTIN_BACKENDS: dict[str, tuple[str, str]] = {
    "mujoco": (
        "strands_robots.simulation.mujoco.simulation",
        "MuJoCoSimEngine",
    ),
    "newton": (
        "strands_robots.simulation.newton.simulation",
        "NewtonSimEngine",
    ),
    "isaac": (
        "strands_robots.simulation.isaac.simulation",
        "IsaacSimulation",
    ),
}

_BUILTIN_ALIASES: dict[str, str] = {
    "mj": "mujoco",
    "mjc": "mujoco",
    "mjx": "mujoco",
    "nt": "newton",
    "isaac_sim": "isaac",
    "isaacsim": "isaac",
    "nvidia": "isaac",
}

DEFAULT_BACKEND = "mujoco"

# Suggested ``pip install`` hints surfaced in the "unknown backend" error so
# users discover that heavy GPU backends ship behind in-tree optional extras
# (``strands-robots[sim-newton]``). Keyed by the backend name (or common alias)
# a user is likely to reach for.
_PLUGIN_INSTALL_HINTS: dict[str, str] = {
    "newton": "pip install 'strands-robots[sim-newton]'",
    # The warp-lang based GPU-parallel MuJoCo path is the built-in ``newton``
    # backend; ``warp`` and ``mjwarp`` are common names users reach for, so map
    # both to the same actionable install hint instead of a bare "unknown
    # backend" dead end.
    "warp": "pip install 'strands-robots[sim-newton]'",
    "mjwarp": "pip install 'strands-robots[sim-newton]'",
}

# Plugin backends discovered via importlib.metadata entry points. Populated
# lazily on the first ``create_simulation`` / ``list_backends`` call (NOT at
# import time) so installing a plugin never slows cold ``import
# strands_robots.simulation``. ``None`` means "not yet discovered".
_PLUGIN_BACKENDS_CACHE: dict[str, type[SimEngine]] | None = None

# Runtime registration (for user-defined backends not in built-ins)

_runtime_registry: dict[str, Callable[[], type[SimEngine]]] = {}
_runtime_aliases: dict[str, str] = {}


def _load_plugin_backends() -> dict[str, type[SimEngine]]:
    """Discover third-party backends registered via entry points.

    Walks the ``strands_robots.backends`` entry-point group and loads each
    advertised class. Results are cached after the first call so the
    ``importlib.metadata`` scan and the plugin module imports happen at most
    once per process (one-time lazy cache - no hot-reload after install).

    A plugin that fails to import (missing heavy dependency, broken install)
    is logged and skipped rather than crashing the factory, so a single bad
    plugin can't take down ``create_simulation`` for the working backends.

    A plugin whose entry-point name collides with a **built-in** backend name
    or alias is skipped entirely (never loaded into the cache) so that:

    * the built-in always wins - a stale ``strands-robots-sim`` still on the
      ``PYTHONPATH`` after a backend has been absorbed in-tree (e.g. ``isaac``
      once it lands as a builtin) can neither shadow nor duplicate-register
      the builtin, and
    * the plugin's (often heavy - Isaac Sim, Newton) module import is avoided
      up front when the builtin will be used anyway.

    This skip is the backward-compat guarantee for the robots-sim -> in-tree
    migration: resolution prefers the builtin and no conflict is ever raised.

    Returns:
        Mapping of entry-point name -> backend class, excluding any name that
        collides with a built-in. Aliases are expressed as multiple
        entry-point names pointing at the same class (e.g. ``newton`` and
        ``warp`` both resolve to ``NewtonSimulation``); no dedup is applied
        among plugins so whichever requested name resolves cleanly.
    """
    global _PLUGIN_BACKENDS_CACHE
    if _PLUGIN_BACKENDS_CACHE is None:
        cache: dict[str, type[SimEngine]] = {}
        for ep in entry_points(group=_ENTRY_POINT_GROUP):
            # Built-ins win over plugins of the same name/alias. Skip the
            # plugin before ``ep.load()`` so a stale sibling package can never
            # shadow an absorbed-in-tree backend, and its heavy import is
            # avoided entirely.
            if ep.name in _BUILTIN_BACKENDS or ep.name in _BUILTIN_ALIASES:
                logger.debug(
                    "Skipping simulation backend plugin %r: shadowed by built-in of the same name.",
                    ep.name,
                )
                continue
            try:
                cache[ep.name] = ep.load()
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not break the factory
                logger.warning("Failed to load simulation backend plugin %r: %s", ep.name, exc)
        _PLUGIN_BACKENDS_CACHE = cache
    return _PLUGIN_BACKENDS_CACHE


def register_backend(
    name: str,
    loader: Callable[[], type[SimEngine]],
    aliases: list[str] | None = None,
    force: bool = False,
) -> None:
    """Register a custom simulation backend at runtime.

    Use this to add backends without editing source code.

    Args:
        name: Backend identifier (e.g., ``"my_physics"``).
        loader: Zero-arg callable that returns the backend **class**
            (not instance). Called lazily on first ``create_simulation()``.
        aliases: Optional short names that resolve to ``name``.
        force: If False (default), raises ValueError when ``name`` or
            an alias is already registered. Set True to overwrite.

    Raises:
        ValueError: If ``name`` or an alias conflicts with an existing
            registration and ``force`` is False.

    Example::

        from strands_robots.simulation.factory import register_backend

        register_backend(
            "bullet",
            lambda: BulletSimulation,
            aliases=["pybullet", "pb"],
        )
        sim = create_simulation("bullet")
    """
    if not force:
        # Check name against ALL existing identifiers (backends + aliases)
        if name in _runtime_registry or name in _BUILTIN_BACKENDS:
            raise ValueError(f"Backend {name!r} already registered. Use force=True to overwrite.")
        if name in _BUILTIN_ALIASES:
            raise ValueError(
                f"Name {name!r} conflicts with built-in alias (resolves to {_BUILTIN_ALIASES[name]!r}). Use force=True to overwrite."
            )
        if name in _runtime_aliases:
            raise ValueError(
                f"Name {name!r} conflicts with runtime alias (resolves to {_runtime_aliases[name]!r}). Use force=True to overwrite."
            )
        if aliases:
            for alias in aliases:
                if alias in _BUILTIN_BACKENDS or alias in _runtime_registry:
                    raise ValueError(
                        f"Alias {alias!r} conflicts with existing backend name. Use force=True to overwrite."
                    )
                if alias in _BUILTIN_ALIASES:
                    raise ValueError(f"Alias {alias!r} conflicts with built-in alias. Use force=True to overwrite.")
                if alias in _runtime_aliases:
                    raise ValueError(f"Alias {alias!r} already registered. Use force=True to overwrite.")

    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name
    logger.debug("Registered simulation backend: %s (aliases=%s)", name, aliases)


def list_backends() -> list[str]:
    """List all available backend names (built-in + plugin + runtime).

    Merges the built-in registry (and its aliases), entry-point plugin
    backends discovered via ``importlib.metadata`` (see
    ``create_simulation``), and any runtime-registered backends/aliases.
    Discovering plugins triggers a one-time lazy scan of the
    ``strands_robots.backends`` entry-point group.

    Returns:
        Sorted list of unique backend identifiers and aliases.

    Example::

        >>> list_backends()
        ['mj', 'mjc', 'mjx', 'mujoco']
    """
    names: set[str] = set()
    names.update(_BUILTIN_BACKENDS.keys())
    names.update(_BUILTIN_ALIASES.keys())
    names.update(_load_plugin_backends().keys())
    names.update(_runtime_registry.keys())
    names.update(_runtime_aliases.keys())
    return sorted(names)


def _resolve_name(backend: str) -> str:
    """Resolve aliases to canonical backend name."""
    # Runtime aliases first (user overrides win)
    if backend in _runtime_aliases:
        return _runtime_aliases[backend]
    # Built-in aliases
    if backend in _BUILTIN_ALIASES:
        return _BUILTIN_ALIASES[backend]
    return backend


def _import_backend_class(name: str) -> type[SimEngine]:
    """Import and return a backend class by canonical name."""
    # 1. Runtime registry (user-registered)
    if name in _runtime_registry:
        cls: type[SimEngine] = _runtime_registry[name]()
        logger.debug("Loaded runtime backend: %s → %s", name, cls.__name__)
        return cls

    # 2. Built-in registry
    if name in _BUILTIN_BACKENDS:
        module_path, class_name = _BUILTIN_BACKENDS[name]
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            # Map backend names to their pip extras (extras use "sim-" prefix)
            _BACKEND_EXTRAS = {"mujoco": "sim-mujoco", "newton": "sim-newton", "isaac": "sim-isaac"}
            extra = _BACKEND_EXTRAS.get(name, f"sim-{name}")
            raise ImportError(
                f"Simulation backend {name!r} is declared in the built-in registry "
                f"but its implementation module {module_path!r} is not available. "
                f"This usually means the backend has not been installed yet "
                f"(e.g. `pip install strands-robots[{extra}]`) or the backend "
                f"implementation has not landed in this release. "
                f"Register a custom backend via "
                f"`strands_robots.simulation.factory.register_backend()` to proceed."
            ) from exc
        backend_cls: type[SimEngine] = getattr(module, class_name)  # type: ignore[assignment]
        logger.debug("Loaded built-in backend: %s → %s.%s", name, module_path, class_name)
        return backend_cls

    # 3. Entry-point plugins (third-party packages, e.g. strands-robots-sim).
    #    Built-ins win over plugins of the same name: the built-in registry is
    #    checked above, and ``_load_plugin_backends`` additionally drops any
    #    plugin whose name/alias collides with a built-in, so a conflicting
    #    plugin can never shadow "mujoco" (or an absorbed-in-tree "isaac").
    plugins = _load_plugin_backends()
    if name in plugins:
        plugin_cls = plugins[name]
        logger.debug("Loaded plugin backend: %s → %s", name, plugin_cls.__name__)
        return plugin_cls

    available = ", ".join(list_backends())
    hint = _PLUGIN_INSTALL_HINTS.get(name)
    suffix = f" To install it: {hint}." if hint else ""
    raise ValueError(f"Unknown simulation backend: {name!r}. Available: {available}.{suffix}")


def create_simulation(
    backend: str = DEFAULT_BACKEND,
    **kwargs: Any,
) -> SimEngine:
    """Create a simulation backend instance.

    This is the primary entry point for creating simulations.
    Backend classes are lazy-loaded on first call.

    Resolution order for ``backend``:

    1. Runtime-registered backends (see ``register_backend``).
    2. Built-in backends (currently ``mujoco``, ``newton``, ``isaac``). Built-ins always
       win over entry-point plugins of the same name, so a third-party
       plugin can never accidentally shadow a built-in backend.
    3. Entry-point plugins. Third-party packages (e.g.
       `strands-robots-sim <https://github.com/strands-labs/robots-sim>`_)
       register heavy out-of-tree backends - Isaac Sim, Newton - by declaring
       them under the ``strands_robots.backends`` entry-point group in their
       ``pyproject.toml``::

           [project.entry-points."strands_robots.backends"]
           newton = "strands_robots_sim.newton.simulation:NewtonSimulation"
           warp = "strands_robots_sim.newton.simulation:NewtonSimulation"

       so they can be discovered on ``pip install`` without patching this
       package. A plugin may map several entry-point names to the same class
       (``newton`` and ``warp`` above) - whichever name is requested resolves
       cleanly. Plugins are discovered lazily on the first
       ``create_simulation`` / ``list_backends`` call (not at import time),
       and a plugin that fails to import is logged and skipped rather than
       crashing the factory. See the Python packaging spec for details:
       https://packaging.python.org/en/latest/specifications/entry-points/

    Args:
        backend: Backend name or alias. Defaults to ``"mujoco"``.
            Built-in: ``"mujoco"`` (aliases: ``"mj"``, ``"mjc"``, ``"mjx"``).
            May also be any entry-point plugin name (see above).
        **kwargs: Backend-specific keyword arguments passed to the
            constructor (e.g., ``tool_name``, ``timestep``).

    Returns:
        A ``SimEngine`` instance ready for ``create_world()``.

    Raises:
        ValueError: If the backend name is not recognized. The message lists
            all available backends (built-in + plugin) and, for known
            out-of-tree backends, a ``pip install`` hint.
        ImportError: If the backend's dependencies are missing
            (e.g., ``pip install mujoco``).

    Examples::

        # Default (MuJoCo)
        sim = create_simulation()
        sim.create_world()
        sim.add_robot("so100")

        # With alias
        sim = create_simulation("mj")

        # Pass kwargs to backend constructor
        sim = create_simulation("mujoco", tool_name="my_sim")

        # GPU-native built-in backend (requires strands-robots[sim-isaac])
        sim = create_simulation("isaac", num_envs=1, headless=True)
    """
    canonical = _resolve_name(backend)
    logger.info("Creating simulation: %s (resolved from %r)", canonical, backend)

    BackendClass = _import_backend_class(canonical)
    return BackendClass(**kwargs)
