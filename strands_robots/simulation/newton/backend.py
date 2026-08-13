"""Newton physics lazy import and solver registry.

Newton (newton-physics/newton) is a GPU-accelerated, NVIDIA-Warp-based physics
engine. It is an optional dependency installed via the ``[sim-newton]`` extra.
This module centralises the import so the rest of the backend can assume the
modules are present, and exposes the rigid-body solver registry consumed by
:class:`~strands_robots.simulation.newton.simulation.NewtonSimEngine`.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

# Memoized imports of the optional ``newton`` / ``warp`` modules. A single
# dict cache keeps the lazy-import state in one place (subscript read +
# write), so there are no module globals that are assigned but never read.
_modules: dict[str, Any] = {}


def ensure_newton() -> tuple[Any, Any]:
    """Import and return ``(newton, warp)``, raising a clear error if missing.

    Returns:
        Tuple of the imported ``newton`` and ``warp`` modules.

    Raises:
        ImportError: With an install hint pointing at the ``[sim-newton]``
            extra when Newton or Warp are not installed.
    """
    cached = _modules.get("newton"), _modules.get("warp")
    if all(m is not None for m in cached):
        return cached
    wp = require_optional(
        "warp",
        pip_install="warp-lang",
        extra="sim-newton",
        purpose="the Newton simulation backend",
    )
    nt = require_optional(
        "newton",
        extra="sim-newton",
        purpose="the Newton simulation backend",
    )
    _modules["newton"], _modules["warp"] = nt, wp
    return nt, wp


def solver_registry() -> dict[str, str]:
    """Map friendly solver names to ``newton.solvers`` class names.

    This is the factual name-to-class mapping and includes every solver Newton
    ships, whether or not it can drive an articulated robot. Only the subset
    :func:`articulated_solvers` returns integrates rigid articulated bodies;
    :func:`articulated_solver_error` reports the rest with the reason.

    Returns:
        Ordered mapping of solver alias to the ``newton.solvers`` attribute
        name implementing it.
    """
    return {
        "mujoco": "SolverMuJoCo",
        "featherstone": "SolverFeatherstone",
        "xpbd": "SolverXPBD",
        "semi_implicit": "SolverSemiImplicit",
        "vbd": "SolverVBD",
        "style3d": "SolverStyle3D",
        "mpm": "SolverImplicitMPM",
        "kamino": "SolverKamino",
    }


def resolve_solver_class(solver: str) -> Any:
    """Resolve a friendly solver name to its ``newton.solvers`` class.

    Args:
        solver: Friendly solver name (see :func:`solver_registry`).

    Returns:
        The solver class object from ``newton.solvers``.

    Raises:
        ValueError: If ``solver`` is not a known solver name.
    """
    nt, _ = ensure_newton()
    registry = solver_registry()
    key = solver.lower()
    if key not in registry:
        raise ValueError(f"Unknown Newton solver {solver!r}. Available: {sorted(registry)}")
    return getattr(nt.solvers, registry[key])


# Newton ships solvers for several physics families, and only some of them
# integrate rigid articulated bodies. The rest fail in two different ways when
# handed a robot, neither of which a caller can act on: ``vbd``, ``style3d`` and
# ``mpm`` raise from inside Newton naming a ``ModelBuilder`` the caller never
# touched, while ``xpbd`` and ``semi_implicit`` build and step without moving a
# joint, so every call reports success over a frozen world. Measured on newton
# 1.5.0 / warp 1.16.0 against a two-hinge arm: a commanded 0.9 rad target moved
# ``featherstone``, ``kamino`` and ``mujoco`` by 0.899 rad and ``xpbd`` /
# ``semi_implicit`` by 0.0 rad, and stepping under gravity alone moved them by
# 0.0 rad as well.
_NON_ARTICULATED_SOLVERS: dict[str, str] = {
    "vbd": (
        "it needs per-body colouring that finalize does not apply, and colouring the "
        "builder makes it step without moving a joint"
    ),
    "style3d": "its cloth custom attributes are absent from a rigid-body model",
    "mpm": "its constructor requires a config object this backend does not build",
    "xpbd": "it builds and steps without integrating rigid bodies, leaving the world frozen",
    "semi_implicit": "it builds and steps without integrating rigid bodies, leaving the world frozen",
}


def articulated_solvers() -> tuple[str, ...]:
    """Return the solver names that can drive an articulated robot.

    Returns:
        The subset of :func:`solver_registry` that integrates rigid articulated
        bodies, in registry order.
    """
    return tuple(name for name in solver_registry() if name not in _NON_ARTICULATED_SOLVERS)


def articulated_solver_error(solver: str) -> str | None:
    """Report a known solver that cannot drive an articulated robot.

    This backend builds rigid robots and rigid objects only, so a solver from
    another physics family has nothing it can integrate. Reporting the name here
    is what keeps the accepted domain equal to the solvers that work: the
    alternative outcomes are a Newton-internal error naming a ``ModelBuilder``
    the caller never touched, or -- for ``xpbd`` and ``semi_implicit`` -- a
    frozen world behind a successful ``add_robot`` / ``send_action`` / ``step``.

    Args:
        solver: Friendly solver name, already known to :func:`solver_registry`.

    Returns:
        A message naming the solver, why it cannot be honoured and the solvers
        that can, or ``None`` when the solver drives an articulated robot.
    """
    reason = _NON_ARTICULATED_SOLVERS.get(solver.lower())
    if reason is None:
        return None
    return (
        f"Newton solver {solver!r} cannot drive an articulated robot: {reason}. "
        f"Solvers that can: {list(articulated_solvers())}."
    )
