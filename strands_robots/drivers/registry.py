"""Choose which driver builds a ``mode="real"`` robot, and find it.

Three questions, kept apart because they have different answers:

* *Which driver?* :func:`resolve_driver` - a name, from the caller's ``driver=``
  or the robot's registry entry, defaulting to
  :data:`~strands_robots.drivers.base.DEFAULT_DRIVER`.
* *Which class?* :func:`get_native_driver_class` - the class a native driver
  package registered for this robot, or ``None``.
* *Which drivers could?* :func:`list_driver_coverage` - for every registered
  robot, the driver names able to build it, joined from this table and the
  package registry's declared lerobot types. Resolution picks one of them; the
  join is the only surface that reports the robots for which there is nothing to
  pick.

The native table starts empty: every robot in the package registry is driven
through lerobot, so nothing is registered until a driver package calls
:func:`register_native_driver`. That is the seam - the point where an
implementation that is not lerobot-shaped can be reached by
:func:`~strands_robots.robot.Robot` without the factory knowing anything about
it.
"""

from __future__ import annotations

import logging

from strands_robots.drivers.base import DEFAULT_DRIVER, DRIVER_CHOICES, missing_driver_members
from strands_robots.registry import get_driver, get_hardware_type, list_robots, resolve_name

logger = logging.getLogger(__name__)

#: Canonical robot name -> the class that drives it natively. Written only by
#: :func:`register_native_driver`, so every entry has passed the surface check.
_NATIVE_DRIVERS: dict[str, type] = {}


def driver_choice_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a driver name, or ``None`` if it is one.

    Args:
        value: The candidate driver name.
        param: Parameter name to quote in the reason.
        context: Calling surface to quote in the reason.

    Returns:
        A reason naming the accepted values, or ``None`` when ``value`` is
        one of :data:`~strands_robots.drivers.base.DRIVER_CHOICES`.
    """
    if isinstance(value, str) and value in DRIVER_CHOICES:
        return None
    return f"{context}: {param} must be one of {', '.join(DRIVER_CHOICES)}, got {value!r}"


def resolve_driver(canonical: str, explicit: str | None = None) -> str:
    """Decide which driver name builds ``canonical``.

    Precedence, highest first: the caller's explicit choice, the robot's
    registry ``hardware.driver``, then
    :data:`~strands_robots.drivers.base.DEFAULT_DRIVER`. ``"auto"`` and ``None``
    both mean "no explicit choice", so they defer to the registry.

    Args:
        canonical: Canonical robot name (or any alias -
            :func:`~strands_robots.registry.resolve_name` is applied).
        explicit: The caller's ``driver=``, or ``None`` when unset.

    Returns:
        A concrete driver name - never ``"auto"``.

    Raises:
        ValueError: If ``explicit`` is not one of
            :data:`~strands_robots.drivers.base.DRIVER_CHOICES`.
    """
    reason = driver_choice_error(explicit, "driver", "resolve_driver") if explicit is not None else None
    if reason is not None:
        raise ValueError(reason)

    if explicit is not None and explicit != "auto":
        return explicit

    # The registry reports what a robot declares and applies no default, so the
    # two "no preference" spellings - an absent field and a declared "auto" -
    # both land here, on the same answer.
    declared = get_driver(canonical)
    if declared is None or declared == "auto":
        return DEFAULT_DRIVER
    return declared


def register_native_driver(canonical: str, driver_cls: type, overwrite: bool = False) -> None:
    """Register ``driver_cls`` as the native driver for ``canonical``.

    The surface is checked here rather than at build time, because here is where
    the mistake is made: a driver missing ``stream`` registers fine and then
    fails on the first agent call, one process and several minutes away from the
    line that is wrong.

    Args:
        canonical: Robot name the driver drives; resolved through
            :func:`~strands_robots.registry.resolve_name` so an alias and its
            canonical name cannot register two different drivers. The robot need
            not exist in the registry yet - a driver package may register before
            the entry it serves is merged, and refusing that would make
            registration order-dependent.
        driver_cls: The class to build. Must satisfy
            :class:`~strands_robots.drivers.base.HardwareDriver`.
        overwrite: Replace an existing registration instead of refusing it.

    Raises:
        TypeError: If ``driver_cls`` does not satisfy the driver surface. The
            missing members are named - a contract violation, so the same class
            of error :mod:`abc` raises for an unimplemented abstract method.
        ValueError: If ``canonical`` already has a driver and ``overwrite`` is
            ``False``.
    """
    missing = missing_driver_members(driver_cls)
    if missing:
        raise TypeError(
            f"register_native_driver: {driver_cls.__name__} cannot drive {canonical!r} - "
            f"it is missing {', '.join(missing)}. See strands_robots.drivers.HardwareDriver "
            "for what a driver must expose."
        )

    key = resolve_name(canonical)
    existing = _NATIVE_DRIVERS.get(key)
    if existing is not None and not overwrite:
        raise ValueError(
            f"register_native_driver: {key!r} is already driven by {existing.__name__}. "
            f"Pass overwrite=True to replace it with {driver_cls.__name__}."
        )

    _NATIVE_DRIVERS[key] = driver_cls
    logger.debug("Registered native driver %s for %r", driver_cls.__name__, key)


def get_native_driver_class(canonical: str) -> type | None:
    """Return the native driver class for ``canonical``, or ``None``.

    Args:
        canonical: Robot name or alias.

    Returns:
        The registered class, or ``None`` when no native driver serves this
        robot - which is every robot until a driver package registers one.
    """
    return _NATIVE_DRIVERS.get(resolve_name(canonical))


def list_native_drivers() -> dict[str, str]:
    """Report which robots have a native driver.

    The discovery surface behind the refusal
    :func:`~strands_robots.robot.Robot` raises for ``driver="strands"`` on a
    robot with no native driver: a caller who asked for one needs to see what
    is available, not only that their choice was not.

    Returns:
        Canonical robot name -> driver class name, sorted by robot name.
    """
    return {name: cls.__name__ for name, cls in sorted(_NATIVE_DRIVERS.items())}


def list_driver_coverage() -> dict[str, tuple[str, ...]]:
    """Report which drivers can build each registered robot, for every robot.

    :func:`list_native_drivers` answers half the question and
    :func:`~strands_robots.registry.get_hardware_type` the other half, and
    neither is a complete answer to "can I drive this robot for real". A robot
    may be reachable through lerobot only, through a native driver only, through
    both, or through neither -- and the last group is the one no existing surface
    reports at all, because it is defined by two absences. Assembling it by hand
    is how a coverage list goes stale: a robot that gained a native driver still
    reads as a gap until someone re-runs the join.

    Reports what the two registries *declare*, so it needs no optional
    dependency and no hardware:

    * ``"lerobot"`` when the robot declares a ``hardware.lerobot_type``. That is
      the type string :class:`~strands_robots.hardware_robot.Robot` hands to
      lerobot's ``RobotConfig``; whether lerobot is installed is a property of
      the environment, not of the robot, so it is deliberately not consulted.
    * ``"strands"`` when a native driver is registered for the robot -- by this
      package's own :data:`~strands_robots.drivers._SHIPPED_DRIVERS`, or by any
      driver package that has called :func:`register_native_driver`.

    Both names are :data:`~strands_robots.drivers.base.DRIVER_CHOICES` values, so
    an entry reads as the set of ``driver=`` arguments that can build that robot.
    Which one *wins* for a robot that has both is :func:`resolve_driver`'s
    answer, not this one's: coverage is about what exists, resolution about what
    is chosen.

    Returns:
        Canonical robot name -> the driver names that can build it, sorted by
        robot name. An empty tuple means the robot is simulation-only: neither
        registry can reach hardware for it, and ``mode="real"`` has nowhere to
        go until a driver is written or a lerobot type is declared.
    """
    coverage: dict[str, tuple[str, ...]] = {}
    for entry in list_robots("all"):
        name = str(entry["name"])
        drivers: list[str] = []
        if get_hardware_type(name) is not None:
            drivers.append(DEFAULT_DRIVER)
        if get_native_driver_class(name) is not None:
            # The literal rather than a constant, matching the one call site
            # that already reads it (``strands_robots.robot`` builds a native
            # driver when the resolved name is this).
            drivers.append("strands")
        coverage[name] = tuple(drivers)
    return coverage


def _native_driver_refusal(robot_type: str) -> str | None:
    """Name ``robot_type``'s native driver, or ``None`` when it has none.

    lerobot's ``RobotConfig`` registry and this package's native-driver registry
    are two answers to "what builds this robot", and a robot may be in the
    second and not the first: the Reachy Mini and the Trossen arms have no
    lerobot robot type at all, which is the gap their native drivers exist to
    close. When ``driver="lerobot"`` is chosen for such a robot -- and it is
    chosen by default, because :func:`resolve_driver` falls back to
    :data:`~strands_robots.drivers.base.DEFAULT_DRIVER` for a robot that
    declares nothing -- lerobot cannot build it, and answering with the names of
    the robots lerobot *does* know answers the wrong question. The driver that
    builds this robot ships in this package, one keyword away.

    The sibling of :func:`strands_robots.teleoperator._other_lerobot_kind_refusal`,
    which does the same for the other kind of wrong entry point: that one names a
    leader arm as a teleoperator, this one names a robot as natively driven. Both
    are consulted before the generic listing, and both return ``None`` to leave
    it in place -- for a name with no native driver the listing is the right
    answer, which is why this reports rather than raises.

    Args:
        robot_type: The device type string handed to lerobot. That is the
            robot's ``hardware.lerobot_type`` when it declares one, and its
            canonical name otherwise -- so a robot with no lerobot type is
            looked up here under the name a caller passed to
            :func:`~strands_robots.robot.Robot`.

    Returns:
        A refusal naming the native driver and the ``driver="strands"`` keyword
        that builds it, or ``None`` when no native driver serves ``robot_type``.
    """
    driver_cls = get_native_driver_class(robot_type)
    if driver_cls is None:
        return None
    return (
        f"Unsupported robot type: {robot_type!r}. lerobot has no robot type for it, but this "
        f"package ships a native driver for it ({driver_cls.__name__}): build it with "
        f"Robot({robot_type!r}, mode='real', driver='strands', ...). To make that the default "
        f"for this robot, declare hardware.driver='strands' on its registry entry."
    )
