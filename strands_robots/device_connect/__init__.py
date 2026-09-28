"""Device Connect integration for strands-robots.

Provides DeviceDriver adapters that wrap Robot and Simulation instances,
exposing them to Device Connect's device registry, RPC routing, and event system.

Usage:
    from strands_robots.device_connect import init_device_connect

    robot = Robot("so100")
    runtime = await init_device_connect(robot, peer_id="so100-lab-1")

    # Now discoverable via Device Connect tools:
    #   discover_devices(device_type="strands_robot")
    #   invoke_device("so100-lab-1", "execute", {"instruction": "pick up cube"})

Module-load discipline
----------------------

Every symbol this package exports is behind ``__getattr__`` (:pep:`562`). The
package imports on a stock ``pip install strands-robots`` -- with no extras --
and only reaches ``device_connect_edge`` when a caller *uses* a name that needs
it. The four Device Connect drivers, ``DeviceRuntime`` and the two
``init_device_connect`` entry points all sit behind that gate.

This is required, not a stylistic choice. The sibling module
``strands_robots.device_connect.reachy_transport`` is stdlib-only and is
imported by the native Reachy driver (``strands_robots.drivers.reachy``, landing in #2762).
Importing that leaf executes this ``__init__``, so a package init that eagerly
imports ``device_connect_edge`` raises ``ModuleNotFoundError`` inside the
native driver's first daemon touch on any install without ``[device-connect]``
-- escaping the driver's own no-raise refusal contract and breaking three
:mod:`AGENTS.md` conventions at once (\"Return error dicts, never raise\",
``require_optional()`` for optional deps, the module-load discipline the
driver's docstring makes explicit).

The public names are unchanged: ``from strands_robots.device_connect import
init_device_connect`` still works, and only fails when the caller reaches for a
name whose implementation genuinely needs ``device_connect_edge``. Static tools
(mypy, IDE autocomplete) see the names through the ``TYPE_CHECKING`` guard.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only, never executed
    from device_connect_edge import DeviceRuntime  # noqa: F401

    from strands_robots.device_connect._impl import (  # noqa: F401
        init_device_connect,
        init_device_connect_sync,
        resolve_allow_insecure,
    )
    from strands_robots.device_connect.reachy_mini_driver import (  # noqa: F401
        ReachyMiniDriver,
    )
    from strands_robots.device_connect.robot_driver import (  # noqa: F401
        RobotDeviceDriver,
    )
    from strands_robots.device_connect.sim_driver import (  # noqa: F401
        SimulationDeviceDriver,
    )


__all__ = [
    "init_device_connect",
    "init_device_connect_sync",
    "resolve_allow_insecure",
    "RobotDeviceDriver",
    "SimulationDeviceDriver",
    "ReachyMiniDriver",
]


# The bring-up budget ``init_device_connect_sync`` bounds its background thread
# with. A float literal with no ``device_connect_edge`` dependency, so it is
# defined here rather than served from ``_impl``: routing it through that module
# would make reading the shipped budget import the extra, and a deferred import
# that still fires is not a deferral. ``init_device_connect_sync`` reads it back
# off this package, which is also what lets a test substitute a millisecond
# budget for the shipped 30 seconds. Expiry is a failed bring-up, not a slow one
# that later succeeds -- the wrapper has already returned by then.
_INIT_TIMEOUT_S: float = 30.0


# Frames inside the package are skipped when a removal warning is attributed,
# so it lands on the caller's line however many package layers it crossed
# (``strands_robots.init_device_connect`` resolves through two ``__getattr__``).
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _announce_removal(name: str) -> None:
    """Warn that ``name`` leaves with Device Connect in 0.7, naming the mesh."""
    warnings.warn(
        f"{name} is Device Connect, which is removed in 0.7; serve the robot on the mesh with "
        "Robot(..., mesh=True) and reach it with the robot_mesh tool",
        DeprecationWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


# How long ``init_device_connect_sync`` waits for the bring-up thread to return
# after a *failed* bring-up, before reporting that it did not. That thread owns
# the loop it created and closes it on its way out, so the wait is what makes
# the release observable to the caller that is about to be handed the failure.
# A budget rather than an unbounded wait: the thread is only as free to return
# as the coroutine it just ran, and a partially-built runtime whose teardown
# blocks would otherwise hold a failed ``init_device_connect_sync`` open for as
# long as that takes. Same family as ``mesh.input._INPUT_JOIN_TIMEOUT_S`` and
# ``drivers.reachy._LOOP_JOIN_TIMEOUT_S``, and defined here for the same reason
# as the budget above: it is a float literal that needs no extra to read.
_LOOP_JOIN_TIMEOUT_S: float = 5.0


# Map each name callers reach through the package to the module it lives in.
# The public :data:`__all__` names are the documented API; the additional
# private names below are module-level symbols ``_impl`` defines or re-exports
# (via ``from device_connect_edge import ...``) that existing tests and callers
# reach through this package. Serving them here keeps
# ``mock.patch("strands_robots.device_connect.DeviceRuntime", ...)`` working
# without eagerly importing ``device_connect_edge`` on package load -- the
# import is deferred until first access, same as every other lazy name.
#
# All three driver modules import ``device_connect_edge`` at module load, and
# the two ``init_device_connect*`` entry points construct a ``DeviceRuntime``
# from it. ``resolve_allow_insecure`` is stdlib-only, kept in the same private
# ``_impl`` module so its behavioural pin does not drag the extra in either.
_ATTR_TO_MODULE: dict[str, str] = {
    "init_device_connect": "strands_robots.device_connect._impl",
    "init_device_connect_sync": "strands_robots.device_connect._impl",
    "resolve_allow_insecure": "strands_robots.device_connect._impl",
    "RobotDeviceDriver": "strands_robots.device_connect.robot_driver",
    "SimulationDeviceDriver": "strands_robots.device_connect.sim_driver",
    "ReachyMiniDriver": "strands_robots.device_connect.reachy_mini_driver",
    # Names re-exported by ``_impl`` from ``device_connect_edge``. Not in
    # ``__all__`` (not part of the public API), but reachable through the
    # package because existing tests patch them here -- and because dropping
    # a name that used to live on the package silently is a worse contract
    # break than any lazy-import scheme is meant to prevent.
    "DeviceRuntime": "strands_robots.device_connect._impl",
    # Private helper the heartbeat regression suite imports through the
    # package (``from strands_robots.device_connect import _build_heartbeat``).
    # Leading underscore is a "not public API" signal, not a "not reachable"
    # signal; it lived here before this PR and existing suites depend on it.
    "_build_heartbeat": "strands_robots.device_connect._impl",
    # The opt-in vocabulary the insecure-setting refusal quotes. Read through
    # the package by the ``TestParsingTheArgumentWasRejectedForAMeasuredReason``
    # suite, which pins the refusal's contract by comparing its message against
    # this tuple's values (an ``on``/``enabled``/``y`` spelling outside the
    # vocabulary should read as secure, and the vocabulary itself is what the
    # refusal quotes).
    "_INSECURE_TRUE": "strands_robots.device_connect._impl",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name lazily.

    Follows :pep:`562`. Called by the Python attribute machinery only when
    ``name`` is not already in the module's namespace, so first access pays
    the import cost and every subsequent access is a plain attribute read.

    A name that needs ``device_connect_edge`` is not imported by this package
    on load. Reaching for one on an install without ``[device-connect]``
    raises the ``ImportError`` from the target module rather than during
    ``import strands_robots.device_connect``. That is what lets the stdlib-only
    leaf ``strands_robots.device_connect.reachy_transport`` be imported on a
    stock install: importing that leaf executes this ``__init__`` first, and
    with this contract that ``__init__`` succeeds.

    Warns:
        DeprecationWarning: On first resolve of a public name. The package is
            removed in 0.7 (#3818); ``Robot(..., mesh=True)`` replaces it.
    """
    # Public name from the export table -- resolve to a symbol its module carries.
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        if name in __all__:
            _announce_removal(f"{__name__}.{name}")
        module = importlib.import_module(module_name)
        value = getattr(module, name)
        # Cache on the package so subsequent lookups skip __getattr__ entirely
        # and ``getattr`` returns bit-identical references across calls.
        globals()[name] = value
        return value

    # Submodule lookup -- ``getattr(pkg, "reachy_transport")`` and
    # ``from strands_robots.device_connect import reachy_transport`` are the two
    # spellings a caller uses to reach the leaves this package ships, and only
    # the second one is served by Python's own import machinery before
    # ``__getattr__`` is consulted. Attempting the import here answers the first
    # spelling with the same module the second returns, without turning the
    # dispatcher into a wildcard: a non-existent submodule name still raises
    # ``AttributeError`` below, because ``import_module`` for a name that is
    # neither a module nor in the export table raises ``ModuleNotFoundError``
    # here rather than reaching the caller as an attribute miss.
    if not name.startswith("_"):
        try:
            submodule = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            pass
        else:
            globals()[name] = submodule
            return submodule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise the lazy names to ``dir()`` and IDE completion.

    Without this, ``dir(strands_robots.device_connect)`` reports only the
    module's own literals, hiding every public symbol from a reader listing
    the package's contents.
    """
    return sorted(set(__all__) | set(globals()))
