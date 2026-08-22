"""One reader at a time on the motor bus: without it, joint telemetry is empty.

A real arm reported ``connected: true`` and streamed no joints at all, because
four threads in one process (state probe, camera publisher at 5Hz, sensors
probe, IoT offload) called ``get_observation()`` on the same serial bus and the
feetech SDK answered every collision with ``[TxRxResult] Port is in use!``.

These tests pin the contract that fixes it: every reader and writer of one
device shares ONE lock, that lock lives on the device, and it survives errors.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

from strands_robots.bus_access import bus_lock, read_observation, write_action


class RefusingBusRobot:
    """A robot that refuses overlapping access, exactly like the real SDK."""

    def __init__(self, *, read_seconds: float = 0.02) -> None:
        self.read_seconds = read_seconds
        self._in_use = False
        self.reads = 0
        self.writes = 0
        self.refusals = 0
        self.max_concurrent = 0
        self._concurrent = 0
        self._audit = threading.Lock()

    def _enter(self) -> None:
        with self._audit:
            if self._in_use:
                self.refusals += 1
                raise ConnectionError(
                    "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] "
                    "after 3 tries. [TxRxResult] Port is in use!"
                )
            self._in_use = True
            self._concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self._concurrent)

    def _exit(self) -> None:
        with self._audit:
            self._in_use = False
            self._concurrent -= 1

    def get_observation(self) -> dict[str, float]:
        self._enter()
        try:
            time.sleep(self.read_seconds)  # the wire takes real time
            with self._audit:
                self.reads += 1
            return {"shoulder_pan.pos": 1.0, "elbow_flex.pos": 2.0}
        finally:
            self._exit()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self._enter()
        try:
            time.sleep(self.read_seconds)
            with self._audit:
                self.writes += 1
            return action
        finally:
            self._exit()


def _run_all(fns: list) -> list[BaseException]:
    """Run callables in parallel threads, collecting whatever they raise.

    Delegated to ``ThreadPoolExecutor`` rather than marshalled by hand:
    ``Future.exception()`` already reports whatever the worker raised,
    ``BaseException`` included, so collecting them needs no ``except`` clause
    of its own. ``max_workers`` is the whole set because a callable left
    queued for a free worker never overlaps the others, and overlap is the
    condition under test.
    """
    with futures.ThreadPoolExecutor(max_workers=len(fns)) as pool:
        submitted = [pool.submit(fn) for fn in fns]
    # The context manager joined every worker, so no future is still pending.
    return [exc for fut in submitted if (exc := fut.exception()) is not None]


def test_the_live_bug_reproduces_without_the_lock():
    """Guard rail: if this ever stops failing, the test below proves nothing."""
    robot = RefusingBusRobot()

    errors = _run_all([robot.get_observation] * 6)

    assert errors, "the fake bus no longer refuses overlap; the fix is untested"
    assert "Port is in use" in str(errors[0])


def test_four_concurrent_probes_all_get_a_reading():
    """The four real readers: state probe, cameras, sensors, offload."""
    robot = RefusingBusRobot()

    errors = _run_all([lambda: read_observation(robot)] * 4)

    assert errors == []
    assert robot.refusals == 0
    assert robot.reads == 4  # every probe produced data, none was dropped
    assert robot.max_concurrent == 1  # one conversation at a time


def test_a_write_never_interleaves_with_a_read():
    """Teleop moving the arm while a probe reads its position."""
    robot = RefusingBusRobot()

    errors = _run_all([lambda: read_observation(robot), lambda: write_action(robot, {"elbow_flex.pos": 3.0})] * 3)

    assert errors == []
    assert robot.refusals == 0
    assert (robot.reads, robot.writes) == (3, 3)
    assert robot.max_concurrent == 1


def test_every_caller_of_one_device_shares_one_lock():
    """The point of hanging it on the device: separate modules, same lock."""
    robot = RefusingBusRobot()
    assert bus_lock(robot) is bus_lock(robot)


def test_two_robots_do_not_block_each_other():
    """Two arms are two buses -- serialising them would halve the fleet's rate."""
    left, right = RefusingBusRobot(), RefusingBusRobot()
    assert bus_lock(left) is not bus_lock(right)


def test_racing_first_callers_still_end_up_with_one_lock():
    """Lock creation is itself a race; two locks would serialise nothing."""
    robot = RefusingBusRobot()
    seen: list[object] = []
    lock = threading.Lock()

    def grab() -> None:
        got = bus_lock(robot)
        with lock:
            seen.append(got)

    assert _run_all([grab] * 8) == []
    assert len({id(x) for x in seen}) == 1


def test_a_failed_read_releases_the_bus():
    """A hardware error must not wedge every future probe."""

    class Boom(RefusingBusRobot):
        def get_observation(self):  # type: ignore[override]
            raise ConnectionError("motor 3 not responding")

    robot = Boom()
    for _ in range(3):
        try:
            read_observation(robot)
        except ConnectionError:
            # The point of the loop: the driver raises and the lock must still be
            # released, so the failure itself is what this test wants.
            pass

    # The lock is free: a plain acquire must succeed immediately.
    assert bus_lock(robot).acquire(blocking=False) is True
    bus_lock(robot).release()


def test_a_reader_holding_the_lock_can_read_again():
    """RLock, because a driver's get_observation() may call back into a read."""
    robot = RefusingBusRobot()

    with bus_lock(robot):
        # Same thread, nested: this deadlocks with a plain Lock. Hoisted out of
        # the assert so ``python -O`` cannot skip the read this test is about.
        nested = read_observation(robot)
    assert nested == {"shoulder_pan.pos": 1.0, "elbow_flex.pos": 2.0}


def test_a_device_that_refuses_attributes_still_gets_serialised():
    """__slots__/frozen/proxy devices fall back to a shared lock, never to none."""

    class Slotted:
        __slots__ = ()

        def get_observation(self) -> dict[str, float]:
            return {"a": 1.0}

    device = Slotted()
    first = bus_lock(device)
    assert isinstance(first, type(threading.RLock()))
    assert bus_lock(device) is first
    observed = read_observation(device)
    assert observed == {"a": 1.0}


def test_the_observation_is_returned_untouched():
    """The lock is the only thing added: no reshaping, no swallowing."""
    robot = RefusingBusRobot()
    observed = read_observation(robot)
    written = write_action(robot, {"elbow_flex.pos": 9.0})
    assert observed == {"shoulder_pan.pos": 1.0, "elbow_flex.pos": 2.0}
    assert written == {"elbow_flex.pos": 9.0}
