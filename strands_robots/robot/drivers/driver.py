"""A Driver is a hardware Body: a servo bus, a DDS link, a vendor SDK.

Hardware bodies in 1.0: feetech (SO-100/101), unitree (G1/Go2/H1 over DDS), microduck (robotd), and
memory (no wire: any robot as a hardware body with no arm attached, for docs, notebooks and CI).
``make_driver`` is the only place that maps a description + port to a driver class; the driver is
read from ``description.hardware.driver`` (registry data), and drivers import their SDK lazily.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...description import Description
from ...types import JointState, Pose
from .calibration import Calibration

_DRIVERS: dict[str, str] = {
    "feetech": "strands_robots.robot.drivers.feetech:FeetechDriver",
    "unitree": "strands_robots.robot.drivers.unitree:UnitreeDriver",
    "microduck": "strands_robots.robot.drivers.microduck:MicroduckDriver",
    "memory": "strands_robots.robot.drivers.memory:MemoryDriver",  # no wire: docs, notebooks, CI
}


@runtime_checkable
class Driver(Protocol):
    """Hardware Body: what a bus must offer. ``names`` are the description's action_keys, in order.

    ``supports_position_hold``: does the bus keep its last goal under torque (registry
    ``hardware.position_hold`` wins over it). ``calibration`` + ``resolution`` (counts per turn), or
    None: with both, a LeRobot policy's degrees are converted exactly through the bus calibration.
    """

    names: tuple[str, ...]
    supports_position_hold: bool
    calibration: Calibration | None
    resolution: int | None

    def read(self) -> JointState: ...

    def write(self, targets: np.ndarray) -> None: ...

    def pose(self) -> Pose | None: ...

    def images(self) -> Mapping[str, np.ndarray]: ...

    def extras(self) -> Mapping[str, Any]: ...

    def hold(self) -> None: ...

    def estop(self) -> None: ...

    def calibrate(self, *, interactive: bool = True) -> Calibration: ...

    def close(self) -> None: ...


def driver_kind(description: Description) -> str | None:
    """Which driver family a registry entry declares (``hardware.driver``), or None for no 1.0 driver."""
    driver = description.hardware.driver
    return driver if driver in _DRIVERS else None


def make_driver(
    description: Description,
    *,
    port: str | None,
    driver: str | Driver | None,
    cameras: Mapping[str, Any] | None,
    calibration: Any = None,
    options: Mapping[str, Any] | None = None,
    id: str | None = None,
) -> Driver:
    """``id`` is the Robot's id; a driver that takes ``robot_id`` (LeRobot's ``--robot.id``, the calibration file's
    name) gets it unless the caller named one, so ``Robot("so101", port=…, id="blue_follower")`` reads
    ``so_follower/blue_follower.json`` instead of silently running uncalibrated."""
    if driver is not None and not isinstance(driver, str):
        if not isinstance(driver, Driver):
            raise TypeError(
                f"driver= must be a driver name or a Driver instance, got {type(driver).__name__} "
                f"(a Driver declares names, supports_position_hold, calibration, resolution and the Body verbs)"
            )
        if calibration is not None or options:
            raise TypeError("calibration= and driver options are for a driver Robot builds; pass them to your Driver")
        if tuple(driver.names) != tuple(description.action_keys):  # targets are written in action_keys order
            raise ValueError(
                f"{type(driver).__name__}.names {list(driver.names)} must be {description.name}'s action_keys "
                f"{list(description.action_keys)}, in order"
            )
        return driver
    kind = driver or driver_kind(description)
    if kind is None:
        raise ValueError(
            f"no driver known for {description.name!r}; "
            "pass driver='feetech'|'unitree'|'microduck'|'memory' or a Driver instance"
        )
    if kind not in _DRIVERS:
        raise ValueError(f"unknown driver {kind!r}; known drivers: {sorted(_DRIVERS)}")
    mod_name, cls_name = _DRIVERS[kind].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    kw = {**(options or {}), **({} if calibration is None else {"calibration": calibration})}
    accepted = driver_parameters(cls)
    if id is not None and id != description.name and "robot_id" in accepted:
        kw.setdefault("robot_id", id)
    unknown = sorted(set(kw) - set(accepted))
    if unknown:  # D3/D4: a misspelt option is an error naming what the driver takes, never silently dropped
        raise TypeError(f"the {kind} driver takes no {unknown}; it takes {sorted(accepted)}")
    return cls(description, port=port, cameras=cameras or {}, **kw)


def driver_parameters(cls: Any) -> tuple[str, ...]:
    """The keyword parameters a driver class accepts besides ``description`` (what ``Robot(**options)`` reaches)."""
    import inspect

    for klass in cls.__mro__:  # a subclass that forwards **kw (a wrapper, a test spy) takes what its base takes
        if "__init__" not in vars(klass):
            continue
        ps = inspect.signature(vars(klass)["__init__"]).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in ps.values()):
            return tuple(n for n, p in ps.items() if n not in ("self", "description") and p.kind is p.KEYWORD_ONLY)
    return ()


__all__ = ["Driver", "Calibration", "make_driver", "driver_kind", "driver_parameters", "DRIVERS"]

DRIVERS = _DRIVERS
