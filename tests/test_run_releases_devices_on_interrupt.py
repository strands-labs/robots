# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Ctrl+C on ``Robot(...).run()`` has to release what the robot holds.

``.run()`` is the documented way to bring a robot online as a server -- it
blocks in ``strands_robots.robot._run_device_connect_foreground`` until the
operator interrupts it. That handler ended with ``os._exit(0)``, which runs no
``atexit`` hook, no ``__del__`` and no ``finally`` block, so the instance's
``cleanup()`` never ran. Measured end to end on a real ``Robot("so101",
mode="sim")`` with a Device Connect runtime that genuinely came online, sending
the process a real ``SIGINT``:

    marker                     before   after
    cleanup() ran                no      yes
    an atexit hook ran           no       no
    printed "<peer> stopped."   yes      yes

``cleanup()`` is where a robot releases what it holds, and on hardware that is
the only path to the physical devices: it prefers the driver's own
``disconnect()``, which is where torque disable and gripper release live (the
contract :mod:`tests.test_hardware_cleanup_disconnects` pins). Nothing else
reaches them -- ``cleanup()`` is terminal, so no library entry point runs after
it, and lerobot's ``Robot.disconnect()`` is ``@check_if_not_connected`` and so
refuses a half-open robot by hand. So an operator's Ctrl+C left the arm
energised at its last commanded position, while the process printed
``"<peer> stopped."`` on the way out. The serial port itself is released --
the process dies, and the kernel closes every descriptor it held -- but the
torque state is not a descriptor: disabling it is a write that has to happen
*before* the port closes, and that write was skipped.

The exit stays abrupt, and that is not incidental. ``Robot.cleanup()`` drains
its task executor with ``shutdown(wait=True)``; measured, a wedged work item
keeps that call running indefinitely, and a ``ThreadPoolExecutor`` worker is not
a daemon thread, so the interpreter's own exit hook would join it as well.
Awaiting the teardown inline -- or returning normally and letting the
interpreter tear down -- therefore converts one Ctrl+C into a process that never
exits, and an operator who then reaches for ``SIGKILL`` gets no teardown at all,
which is the outcome this contract exists to produce. The teardown runs on a
daemon thread under a budget instead, on the reasoning
``MuJoCoSimEngine.cleanup`` already applies to a wedged policy worker: bound the
wait, report, proceed.

What these tests pin:

    - a connected arm's devices are released, through the driver's own
      ``disconnect()``, with torque disable requested;
    - a simulation's ``cleanup()`` runs too, and on the failed-bring-up branch
      as well -- the built-in mesh has already been stopped by then, so the
      instance's own resources are all that is left to release;
    - the shutdown line claims "stopped" only when the teardown completed, and
      names the reason when it did not;
    - Ctrl+C still ends the process: a teardown that never returns, and one
      that raises, both still reach the exit;
    - an instance exposing no ``cleanup`` is not an error.

No serial port and no camera device is opened: the driver, bus, camera and port
doubles are :mod:`tests.test_hardware_cleanup_disconnects`'s, which mirror
lerobot's connect ordering, its ``is_connected`` composition and its decorator
contracts, and model port exclusivity so the consequence is asserted rather than
the call.
"""

from __future__ import annotations

import threading
import time
import types
from typing import Any

from strands_robots import robot as robot_mod
from tests.test_hardware_cleanup_disconnects import _arm, _make_robot, _Port
from tests.test_robot_factory import _drive_foreground

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 10.0


class _Simulation:
    """Simulation-shaped instance: the ``cleanup()`` the foreground loop owns.

    ``MuJoCoSimEngine.cleanup`` takes an optional ``policy_stop_timeout``, so
    the double accepts one too -- the runner must not pass one, but a signature
    narrower than the real thing would pass for the wrong reason.
    """

    def __init__(self, *, blocks: threading.Event | None = None, raises: BaseException | None = None) -> None:
        self._peer_id = "sim-1"
        self._peer_type = "sim"
        self.mesh = None
        self.cleanup_calls = 0
        self.cleanup_threads: list[str] = []
        self._blocks = blocks
        self._raises = raises

    def cleanup(self, policy_stop_timeout: float | None = None) -> None:  # noqa: ARG002 - real signature
        self.cleanup_calls += 1
        self.cleanup_threads.append(threading.current_thread().name)
        if self._raises is not None:
            raise self._raises
        if self._blocks is not None:
            assert self._blocks.wait(timeout=DEADLINE), "test never released the blocked teardown"


def _connected_arm(port: _Port) -> tuple[Any, Any]:
    """A one-camera arm, connected, wired onto a ``Robot`` bound to ``.run()``.

    ``Any`` because ``_attach_device_connect`` sets ``_peer_id``/``_peer_type``
    on the instance at bind time rather than in ``Robot.__init__``, so they are
    not declared attributes -- the same reason the shipped runner reads them
    with ``getattr``.
    """
    driver = _arm(port)
    hw: Any = _make_robot(driver)
    hw._peer_id = "arm-1"
    hw._peer_type = "robot"
    driver.connect()
    assert driver.is_connected, "the arm has to be connected for teardown to have anything to release"
    return hw, driver


class TestCtrlCReleasesTheDevicesTheArmHolds:
    """The interrupt reaches the one path that disables torque."""

    def test_a_connected_arms_devices_are_released(self, monkeypatch, capsys) -> None:
        port = _Port()
        hw, driver = _connected_arm(port)

        _drive_foreground(monkeypatch, capsys, instance=hw)

        assert driver.disconnect_calls == 1, "the driver's own disconnect() never ran on Ctrl+C"
        assert not driver.is_connected
        assert port.held_by is None, "the arm's port was never closed through the driver"
        assert driver.cameras["wrist"].disconnect_calls == 1

    def test_the_torque_disable_is_what_reaches_the_bus(self, monkeypatch, capsys) -> None:
        """Torque disable is a write, and it has to precede the port close.

        ``bus.disconnect(disable_torque=True)`` is the driver-preferred path;
        the fallback in ``_close_open_devices`` deliberately passes ``False``,
        so recording only "a disconnect happened" would accept a teardown that
        left the motors energised.
        """
        port = _Port()
        hw, driver = _connected_arm(port)

        _drive_foreground(monkeypatch, capsys, instance=hw)

        assert driver.bus.disconnect_calls == [True], (
            f"expected one torque-disabling bus disconnect, got {driver.bus.disconnect_calls}"
        )

    def test_the_port_is_free_for_the_next_holder(self, monkeypatch, capsys) -> None:
        """A serial port is exclusive, so the next holder is the consequence."""
        port = _Port()
        hw, _driver = _connected_arm(port)

        _drive_foreground(monkeypatch, capsys, instance=hw)

        port.open("the next process")
        assert port.held_by == "the next process"


class TestCtrlCReleasesASimulationToo:
    """The runner is bound to sim instances as well, on both bring-up branches."""

    def test_a_simulations_cleanup_runs(self, monkeypatch, capsys) -> None:
        sim = _Simulation()

        _drive_foreground(monkeypatch, capsys, instance=sim)

        assert sim.cleanup_calls == 1, "cleanup() never ran on Ctrl+C"

    def test_a_failed_bring_up_still_releases_the_instance(self, monkeypatch, capsys) -> None:
        """The mesh is already stopped by then; the instance is what is left.

        A bring-up that failed keeps the process alive deliberately, so the
        operator's Ctrl+C is still the only teardown this path ever gets.
        """
        sim = _Simulation()

        out = _drive_foreground(monkeypatch, capsys, instance=sim, runtime=None)

        assert "NOT online" in out, "this test has to exercise the failed-bring-up branch"
        assert sim.cleanup_calls == 1, "a failed bring-up skipped the teardown entirely"


class TestTheShutdownLineClaimsOnlyWhatHappened:
    """ "stopped" is a claim about the teardown, so it has to follow one."""

    def test_a_completed_teardown_is_reported_stopped(self, monkeypatch, capsys) -> None:
        """The unchanged half: the ordinary path still reads the same."""
        sim = _Simulation()

        out = _drive_foreground(monkeypatch, capsys, instance=sim)

        assert "sim-1 stopped." in out
        assert "WITHOUT a completed shutdown" not in out

    def test_a_teardown_that_never_finishes_is_not_reported_stopped(self, monkeypatch, capsys) -> None:
        release = threading.Event()
        sim = _Simulation(blocks=release)
        monkeypatch.setattr(robot_mod, "_SHUTDOWN_TIMEOUT_S", 0.2, raising=False)
        try:
            out = _drive_foreground(monkeypatch, capsys, instance=sim)
        finally:
            release.set()

        assert "sim-1 stopped." not in out, "claimed the robot stopped while the teardown was still running"
        assert "WITHOUT a completed shutdown" in out
        assert "may not have been released" in out

    def test_a_teardown_that_raises_is_not_reported_stopped(self, monkeypatch, capsys) -> None:
        sim = _Simulation(raises=OSError("bus write failed"))

        out = _drive_foreground(monkeypatch, capsys, instance=sim)

        assert "sim-1 stopped." not in out
        assert "WITHOUT a completed shutdown" in out
        assert "bus write failed" in out

    def test_the_shutdown_report_is_ascii(self, monkeypatch, capsys) -> None:
        """Same rule the rest of ``run()``'s output follows."""
        sim = _Simulation(raises=OSError("bus write failed"))

        out = _drive_foreground(monkeypatch, capsys, instance=sim)

        out.encode("ascii")


class TestCtrlCStillEndsTheProcess:
    """A teardown that will not finish must not hold the operator hostage."""

    def test_a_teardown_that_never_returns_still_reaches_the_exit(self, monkeypatch, capsys) -> None:
        """``_drive_foreground`` returns only because ``os._exit`` was reached.

        This is what fails if the teardown is ever awaited on the calling
        thread or moved into a ``finally``: the executor drain a wedged rollout
        leaves running never returns, so Ctrl+C would stop ending the process.
        The blocked teardown gives up on its own after ``DEADLINE`` so a broken
        contract fails the test instead of hanging the suite.
        """
        release = threading.Event()
        sim = _Simulation(blocks=release)
        monkeypatch.setattr(robot_mod, "_SHUTDOWN_TIMEOUT_S", 0.2, raising=False)

        started = time.monotonic()
        try:
            out = _drive_foreground(monkeypatch, capsys, instance=sim)
        finally:
            release.set()
        elapsed = time.monotonic() - started

        assert "Shutting down sim-1" in out
        assert elapsed < DEADLINE / 2, f"the exit waited {elapsed:.1f}s on a teardown that never returns"

    def test_a_teardown_that_raises_still_reaches_the_exit(self, monkeypatch, capsys) -> None:
        sim = _Simulation(raises=RuntimeError("teardown exploded"))

        out = _drive_foreground(monkeypatch, capsys, instance=sim)

        assert "Shutting down sim-1" in out

    def test_a_second_ctrl_c_during_the_release_still_reaches_the_exit(self, monkeypatch, capsys) -> None:
        """The budget opened a window to interrupt; it must behave like the first.

        Unlike the rest of this module this pins the fix rather than the defect:
        before the budget existed there was no wait to interrupt, so there is no
        pre-fix behaviour for it to contradict.

        A ``KeyboardInterrupt`` that escaped the wait would skip ``os._exit``
        and unwind into interpreter shutdown, where ``concurrent.futures``' own
        exit hook joins the very executor drain the wait was covering -- so an
        impatient operator would get the hang the budget exists to prevent.
        """
        interrupted = threading.Event()

        class _InterruptedWait(threading.Event):
            def wait(self, timeout: float | None = None) -> bool:  # noqa: ARG002 - stands in for the budget
                interrupted.set()
                raise KeyboardInterrupt

        # Only the runner's view of ``threading`` is substituted; mutating the
        # real ``threading.Event`` would reach every other user of it.
        monkeypatch.setattr(
            robot_mod,
            "threading",
            types.SimpleNamespace(Event=_InterruptedWait, Thread=threading.Thread),
            raising=False,
        )
        sim = _Simulation()

        out = _drive_foreground(monkeypatch, capsys, instance=sim)

        assert interrupted.is_set(), "the release never reached the wait this test interrupts"
        assert "sim-1 stopped." not in out
        assert "interrupted again" in out

    def test_an_instance_without_a_cleanup_is_not_an_error(self, monkeypatch, capsys) -> None:
        """``_attach_device_connect`` binds ``.run()`` onto any instance."""
        instance = types.SimpleNamespace(_peer_id="bare-1", _peer_type="sim", mesh=None)

        out = _drive_foreground(monkeypatch, capsys, instance=instance)

        assert "bare-1 stopped." in out
        assert "WITHOUT a completed shutdown" not in out

    def test_a_non_callable_cleanup_attribute_is_not_an_error(self, monkeypatch, capsys) -> None:
        instance = types.SimpleNamespace(_peer_id="bare-2", _peer_type="sim", mesh=None, cleanup=None)

        out = _drive_foreground(monkeypatch, capsys, instance=instance)

        assert "bare-2 stopped." in out
