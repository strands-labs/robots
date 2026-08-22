"""One reader at a time on a robot's motor bus.

A serial motor bus is a single conversation: the host writes a sync-read packet
addressed to a set of servo ids, then reads the replies off the wire. Two
threads doing that at once interleave their packets, and the feetech/dynamixel
SDKs refuse outright::

    ConnectionError("Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4,
    5, 6] after 3 tries. [TxRxResult] Port is in use!")

Which is what a strands mesh peer did to itself. Four independent threads in one
child process reach for the same device -- the state probe that fills the fleet
snapshot, the hardware camera publisher (which calls ``get_observation()`` at
``STRANDS_MESH_CAMERA_HZ``, and lerobot's ``get_observation()`` reads the MOTORS
before it grabs any frame), the sensors probe, and the IoT camera offload -- and
teleop writes to it besides. Nobody was serialising them, so on real hardware
the reads collided continuously and every joint consumer drew nothing: no
positions in the fleet snapshot, no history traces, no motion detection. The
SDK's three retries did not help: all three landed inside the same collision.

The lock lives on the DEVICE, not on any one caller, because that is what is
actually being shared: the mesh modules, the teleop rail and any application on
top of them all hold different wrappers around the same lerobot robot. It is an ``RLock`` so a
caller that already holds it can read again without deadlocking (lerobot's own
``get_observation()`` is free to call back into a locked read).

This module deliberately knows nothing about lerobot or the mesh: it is import-
safe from anywhere, which is the only way every reader can be made to share one
lock without an import cycle.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Attribute the per-device lock is cached under. Named for us, so a driver's
#: own attributes can never be mistaken for it.
_LOCK_ATTR = "_strands_bus_lock"

#: Guards lock CREATION, so two threads racing to be first cannot end up with a
#: lock each -- which would serialise nothing at all.
_registry_guard = threading.Lock()

#: Used when a device refuses attribute assignment (``__slots__``, a frozen
#: dataclass, a proxy). Shared by all such devices: over-serialising two
#: unrelated robots is slow, and letting them collide is broken.
_fallback_lock = threading.RLock()


def _cached_lock(device: Any) -> threading.RLock | None:
    """The lock already cached on ``device``, or ``None`` on first use.

    One typed read of an untyped ``getattr``: the attribute is named for this
    module, so nothing but a lock created here is ever stored under it.

    Args:
        device: The object whose bus lock is being looked up.

    Returns:
        The cached lock, or ``None`` when the device has not been locked yet.
    """
    lock: threading.RLock | None = getattr(device, _LOCK_ATTR, None)
    return lock


def bus_lock(device: Any) -> threading.RLock:
    """Return the one lock guarding ``device``'s bus, creating it on first use.

    Args:
        device: The object whose bus is being shared -- normally a lerobot
            robot, or anything else with ``get_observation``/``send_action``.

    Returns:
        The device's lock. Every caller passing the same device gets the same
        lock object, including callers in other modules.
    """
    existing = _cached_lock(device)
    if existing is not None:
        return existing

    with _registry_guard:
        # Re-read inside the guard: another thread may have just created it.
        existing = _cached_lock(device)
        if existing is not None:
            return existing
        lock = threading.RLock()
        try:
            setattr(device, _LOCK_ATTR, lock)
        except Exception:  # noqa: BLE001 - __slots__, frozen, or a proxy
            logger.debug(
                "%s will not hold a bus lock; falling back to the shared one",
                type(device).__name__,
            )
            return _fallback_lock
        return lock


def read_observation(device: Any) -> Any:
    """Read one observation from ``device`` with exclusive use of its bus.

    The one call every reader should use. Blocking is the point: a probe that
    waits its turn produces a reading, while a probe that barges in produces a
    ``Port is in use!`` and no data for anyone.

    Args:
        device: The robot to read. Must expose ``get_observation()``.

    Returns:
        Whatever the driver's ``get_observation()`` returns.

    Raises:
        Exception: Anything the driver raises, unchanged -- callers already
            handle hardware errors, and the lock is released either way.
    """
    with bus_lock(device):
        return device.get_observation()


def write_action(device: Any, action: Any) -> Any:
    """Send one action to ``device`` with exclusive use of its bus.

    Takes the SAME lock as :func:`read_observation`, because a write that
    interleaves with a read corrupts both halves of the exchange -- teleop
    moving an arm while a probe reads its position is the common case.

    Args:
        device: The robot to command. Must expose ``send_action()``.
        action: The action, in whatever shape the driver accepts.

    Returns:
        Whatever the driver's ``send_action()`` returns.
    """
    with bus_lock(device):
        return device.send_action(action)


#: Register every SO-101-family bus reports positions from. Named here so the
#: joints-only read is one obvious call and not a string buried in a probe.
_POSITION_REGISTER = "Present_Position"


def _num_read_retries(device: Any) -> int | None:
    cfg = getattr(device, "config", None)
    n = getattr(cfg, "num_read_retries", None)
    return n if isinstance(n, int) else None


def read_joints(device: Any) -> Any:
    """Read ONLY the joint positions, so a broken camera cannot hide them.

    Measured on real hardware: an arm published ZERO joints for eleven hours
    while its mesh presence stayed healthy and non-stale. One line at startup
    explained it -- ``state probe 'hw_joints' failed ...
    RuntimeError('OpenCVCamera(1) read failed')`` -- and then the log went quiet.

    The cause is in lerobot's ``get_observation()``: it sync-reads the motors
    FIRST, then loops over the cameras calling ``read_latest()``. A camera that
    raises therefore throws away the joint positions ALREADY IN HAND. Every
    joints consumer went through that call -- the fleet snapshot, the joint
    history traces, motion detection, and teleop's 30Hz publisher -- so a single
    dead USB camera silently disarmed an entire arm.

    Joints and frames are independent facts about a robot and must fail
    independently. When the driver exposes its motor bus we read that directly
    (under the SAME lock as everything else, or this reintroduces the collision
    this module exists to prevent). When it does not, we fall back to the full
    observation: a driver whose only reader is ``get_observation`` cannot be
    asked for less, and pretending otherwise would return nothing at all.

    Args:
        device: The robot to read. ``device.bus.sync_read`` is used when present.

    Returns:
        A mapping of ``"<motor>.pos"`` -> position, shaped exactly like the joint
        half of ``get_observation()`` so callers need no new branch. On the
        fallback path, whatever ``get_observation()`` returns (frames included).
        A driver is taken to have no readable motor bus - and so takes that
        fallback - both when it exposes no ``bus.sync_read`` and when that call
        answers with something other than a mapping.

    Raises:
        Exception: Anything the driver raises, unchanged.
    """
    bus = getattr(device, "bus", None)
    if bus is None or not hasattr(bus, "sync_read"):
        return read_observation(device)
    retries = _num_read_retries(device)

    def _sync_read() -> Any:
        if retries is None:
            return bus.sync_read(_POSITION_REGISTER)
        try:
            return bus.sync_read(_POSITION_REGISTER, num_retry=retries)
        except TypeError:
            # A bus implementation without the retry keyword: the read still
            # matters more than the retry policy.
            return bus.sync_read(_POSITION_REGISTER)

    with bus_lock(device):
        raw = _sync_read()
    if not isinstance(raw, Mapping):
        # A ``bus`` that answers ``sync_read`` with something other than a
        # mapping is not a motor bus this function can read: a wrapper, a proxy,
        # or a driver using that attribute name for something else entirely.
        # Returning it would hand back a value this function documents as a
        # mapping and every joints consumer iterates, so the caller would fail
        # on ``.items()`` one frame later with nothing naming the cause. Fall
        # back to the full observation, which is already the answer given to a
        # driver that exposes no bus at all -- the bus is being detected here,
        # not trusted, and the same rule decides both halves of that detection.
        return read_observation(device)
    # ``.pos`` is lerobot's own suffix (see SOFollower.get_observation) and the
    # shape the rest of this codebase already parses.
    return {f"{motor}.pos": value for motor, value in raw.items()}
