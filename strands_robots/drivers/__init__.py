"""Driver seam for ``Robot(..., mode="real")``.

:class:`~strands_robots.drivers.base.HardwareDriver` is the contract a real
robot satisfies; :mod:`strands_robots.drivers.registry` decides which
implementation a given robot gets. A driver package that is not lerobot-shaped
registers itself here::

    from strands_robots.drivers import register_native_driver

    register_native_driver("unitree_g1", G1Driver)

and ``Robot("unitree_g1", mode="real", driver="strands")`` then builds it. The
drivers shipped in this package register themselves from :data:`_SHIPPED_DRIVERS`
on import.
"""

import logging

from strands_robots.drivers.base import (
    DEFAULT_DRIVER,
    DRIVER_CHOICES,
    DRIVER_SURFACE,
    HardwareDriver,
    missing_driver_members,
)
from strands_robots.drivers.registry import (
    driver_choice_error,
    get_native_driver_class,
    list_driver_coverage,
    list_native_drivers,
    register_native_driver,
    resolve_driver,
)

#: The drivers this package ships, as ``(module, class name, robot names)``.
#: A table rather than a block per driver so a second driver cannot arrive with
#: a subtly different guard than the first - the import guard, the
#: already-registered tolerance and the alias handling are written once. Robot
#: names are the *canonical* names; :func:`register_native_driver` resolves
#: aliases, so listing an alias as well is harmless but redundant.
#:
#: The names are a literal tuple, or the name of an attribute on the driver's
#: own module holding them. A driver that supports a whole family declares that
#: family itself - restating it here would be a second list to keep in step
#: with the first, and the drift would show up as a robot that resolves to
#: ``lerobot`` for no reason a reader can see.
#:
#: Dynamixel is first because it registers the most robots and its codec has no
#: optional import that can fail: an ordering mistake then surfaces as the
#: smaller driver failing after the bigger one rather than the reverse, which
#: is the easier direction to read.
_SHIPPED_DRIVERS: tuple[tuple[str, str, tuple[str, ...] | str], ...] = (
    ("strands_robots.drivers.dynamixel.driver", "DynamixelDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.feetech.driver", "FeetechDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.franka.driver", "FrankaDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.g1", "G1Driver", ("g1", "unitree_g1")),
    ("strands_robots.drivers.go2", "Go2Driver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.reachy", "ReachyDriver", ("reachy_mini",)),
    ("strands_robots.drivers.microduck", "MicroduckDriver", ("microduck",)),
    ("strands_robots.drivers.robotiq.driver", "RobotiqDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.booster", "BoosterDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.ur", "URDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.crazyflie", "CrazyflieDriver", "SUPPORTED_ROBOTS"),
    ("strands_robots.drivers.earthrover", "EarthRoverDriver", "SUPPORTED_ROBOTS"),
)


def shipped_robot_names(module: object, names: tuple[str, ...] | str) -> tuple[str, ...]:
    """Resolve one table entry's robot names against the driver's own module.

    Shared with the test that grades every shipped driver against the seam, so
    the names it checks are the names actually registered rather than a second
    reading of the same table.

    Args:
        module: The driver's imported module.
        names: A literal tuple of canonical names, or the name of an attribute
            on *module* holding them.

    Returns:
        The canonical robot names for that entry.
    """
    if isinstance(names, str):
        return tuple(getattr(module, names))
    return names


logger = logging.getLogger(__name__)


def _register_shipped_drivers() -> None:
    """Register the drivers shipped with the package.

    A driver package outside this repo calls
    :func:`register_native_driver` itself; the drivers we ship register here
    so ``Robot("g1", mode="real", driver="strands")`` works without a second
    import. The registration is guarded per driver: an import that fails (a bad
    SDK install, a broken deps subset) skips *that* driver rather than breaking
    every ``from strands_robots.drivers import ...`` statement or costing the
    other drivers their registration. A driver package that overrides a shipped
    registration wins, because ``register_native_driver`` refuses
    double-registration by default and the refusal is tolerated here.
    """
    import importlib

    for module_path, class_name, robot_names in _SHIPPED_DRIVERS:
        try:
            module = importlib.import_module(module_path)
            driver_cls = getattr(module, class_name)
            canonical_names = shipped_robot_names(module, robot_names)
        except Exception:  # noqa: BLE001 - a broken driver must not break the seam
            logger.debug("Shipped driver %s.%s did not import; skipping", module_path, class_name)
            continue
        for canonical in canonical_names:
            try:
                register_native_driver(canonical, driver_cls)
            except ValueError:
                # Already registered - a caller registered it first, honour that.
                pass


_register_shipped_drivers()

__all__ = [
    "DEFAULT_DRIVER",
    "DRIVER_CHOICES",
    "DRIVER_SURFACE",
    "HardwareDriver",
    "driver_choice_error",
    "get_native_driver_class",
    "list_driver_coverage",
    "list_native_drivers",
    "missing_driver_members",
    "register_native_driver",
    "resolve_driver",
    "shipped_robot_names",
]
