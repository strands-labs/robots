"""Accurate loop pacing that still stops the instant it is told to.

Every publish loop in the mesh paced itself with ``self._stop_event.wait(period)``
-- a shape chosen for a good reason: it sleeps for the period *and* returns early
when the loop is stopped, so shutdown never waits out a tick. What is wrong with
it is that a ``wait(period)`` is a **delay**, while a loop's rate needs a
**deadline**. The time a tick spends reading a bus or grabbing a frame is added to
the period instead of being subtracted from it, so the achieved rate is
``1 / (period + work)`` and every counter reports that as the rate the hardware
managed. Measured on Linux with an idle epoll timer, so this is the floor of the
problem rather than a platform quirk:

===========  ===========  ==================  ==========
requested    tick body    ``wait(period)``    this module
===========  ===========  ==================  ==========
10 Hz        0 ms         9.97 Hz             9.98 Hz
10 Hz        40 ms        7.12 Hz             10.00 Hz
30 Hz        10 ms        22.97 Hz            29.99 Hz
50 Hz        10 ms        31.92 Hz            49.96 Hz
50 Hz        15 ms        28.33 Hz            49.97 Hz
===========  ===========  ==================  ==========

A second, platform-specific cause stacks on top of that one. In a process tree
that inherits background-QoS timer coalescing -- any robot started by a daemon,
a launchd agent or a supervising process -- macOS inflates every
``nanosleep``-family wait by ~145ms, so a nominal 100ms ``Event.wait`` costs
~247ms while a kqueue timer costs ~102ms. There a 10Hz loop with an *empty* body
achieves 4.33Hz. The selector timer this module waits on carries neither cost:
it is the same primitive ``asyncio`` sleeps on, and it is a deadline.

Two properties are non-negotiable, because the loops that use it stream robot
state and teleop frames:

1. **The period is a deadline, not a delay.** :meth:`Ticker.wait` sleeps until
   the next multiple of the period from the loop's start, so time spent doing the
   work of a tick is subtracted rather than added, and a slow tick does not push
   every later tick late.
2. **A stop is honoured within a slice, not within a period.** The wait is broken
   into short selector waits (default 10ms) so a ``stop_event`` set by another
   thread is seen promptly, and :meth:`Ticker.wake` can interrupt it immediately
   from the thread doing the stopping. A 30Hz loop must not take a third of a
   second to notice that the robot is going down.

The doorbell :meth:`Ticker.wake` rings is a ``socket.socketpair()`` rather than an
``os.pipe()``, and that is a portability requirement rather than a preference. On
Windows ``selectors.DefaultSelector`` is ``SelectSelector``, whose WinSock
``select()`` accepts sockets ONLY: a pipe fd raises ``OSError`` (WSAENOTSOCK,
10038) on the first :meth:`Ticker.wait`. That call sits outside the ``try`` in
every paced loop, so each publish thread would die on its first tick while the
mesh itself looked up -- a robot that joins the fleet and streams nothing, which
is the hardest shape of this failure to attribute. It is also why the claim above
about sleeping on the same primitive as ``asyncio`` holds: ``asyncio``'s selector
loop builds its own self-pipe from a socketpair for exactly this reason.

Apart from the package's shared numeric-domain helper this module is
dependency-free, and it imports nothing from the mesh: it is pure timing,
unit-testable without a session, a robot or a network.
"""

from __future__ import annotations

import selectors
import socket
import threading
import time

from strands_robots.utils import positive_finite_number_error

__all__ = ["Ticker", "sleep_penalty_s"]

_DEFAULT_SLICE_S = 0.01


class Ticker:
    """Pace a loop at ``period`` seconds per tick using the selector timer.

    Args:
        period: Seconds per tick. Must be finite and > 0 -- a zero period would
            busy-spin the calling loop, which on these publish paths means a
            serial bus or a camera hammered as fast as the CPU allows, so it is
            refused here rather than discovered on hardware.
        stop_event: Optional event that ends the loop. When it is set,
            :meth:`wait` returns ``True`` promptly (within one slice) instead of
            waiting out the rest of the period.
        slice_s: Longest single selector wait. Smaller = faster stop, more
            wakeups. Must be > 0.

    Use it exactly where ``stop_event.wait(period)`` used to be::

        with Ticker(period, stop_event) as ticker:
            while running:
                do_one_tick()
                if ticker.wait():   # True == stopped, same sense as Event.wait
                    break

    The return value keeps ``Event.wait``'s sense on purpose (``True`` means
    "stop"), so converting a loop is a one-line change and cannot invert a
    shutdown test by accident.
    """

    def __init__(
        self,
        period: float,
        stop_event: threading.Event | None = None,
        slice_s: float = _DEFAULT_SLICE_S,
    ) -> None:
        # Both bounds go through the package's shared positive-finite domain rather
        # than a hand-rolled check, so a bool (an int subclass that would read as a
        # one-second period), a nan and an inf are refused here by the same rule
        # every other numeric knob is held to -- and refused BEFORE the float()
        # coercion, which would otherwise turn True into 1.0.
        for _name, _value in (("period", period), ("slice_s", slice_s)):
            _problem = positive_finite_number_error(_value, _name, "Ticker")
            if _problem is not None:
                raise ValueError(f"{_problem} A zero, negative or nan period busy-spins the loop that paces on it.")
        period = float(period)
        slice_s = float(slice_s)
        self.period = period
        self.slice_s = min(slice_s, period)
        self._stop_event = stop_event
        self._selector = selectors.DefaultSelector()
        # A self-socket gives two things: something registered (so the selector
        # always has an object to watch, whatever the platform's implementation)
        # and a way for the stopping thread to interrupt a wait immediately.
        # A socketpair rather than os.pipe() because Windows resolves
        # DefaultSelector to SelectSelector, and WinSock select() takes only
        # sockets -- a pipe fd there raises OSError(WSAENOTSOCK) out of the first
        # wait(), outside the try in every paced loop. asyncio's selector loop
        # builds its own self-pipe from a socketpair for the same reason.
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._selector.register(self._wake_r, selectors.EVENT_READ)
        self._closed = False
        self._deadline = time.perf_counter()

    # -- the loop side ---------------------------------------------------

    def wait(self) -> bool:
        """Sleep until the next tick is due. Return ``True`` if stopped.

        The deadline advances by exactly one period per call, so the achieved
        rate is the requested rate as long as the work of a tick fits inside it.
        When a tick overruns, the missed deadlines are dropped rather than
        chased: catching up would fire several ticks back to back, which on a
        publish loop means a burst of frames with near-identical timestamps.
        """
        if self._closed:
            raise RuntimeError("Ticker.wait() after close()")
        now = time.perf_counter()
        self._deadline += self.period
        if self._deadline < now:
            # Overran: resynchronise to the next whole period from now so we do
            # not accumulate a debt of ticks that will never be paid.
            missed = int((now - self._deadline) / self.period) + 1
            self._deadline += missed * self.period
        while True:
            if self._stopped():
                return True
            remaining = self._deadline - time.perf_counter()
            if remaining <= 0:
                return False
            events = self._selector.select(min(remaining, self.slice_s))
            if events:
                self._drain()
                if self._stopped():
                    return True
                # A spurious wake (someone called wake() without stopping) must
                # not shorten the tick: keep waiting out the deadline.

    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _drain(self) -> None:
        """Empty the doorbell so one wake cannot satisfy every later wait.

        Never raises. It is only a doorbell: whether it carried one byte or a
        hundred, the caller's next decision comes from the deadline and the stop
        event, so nothing here is worth propagating into a publish loop.
        """
        try:
            while self._wake_r.recv(4096):
                pass
        except BlockingIOError:
            # Nothing left to read on a non-blocking socket: it is drained,
            # which is exactly the post-condition this method is for.
            pass
        except OSError:
            # The socket is gone (a concurrent close). There is nothing to drain
            # and nothing a paced loop could do about it.
            pass

    # -- the stopper side ------------------------------------------------

    def wake(self) -> None:
        """Interrupt a wait in progress. Safe from any thread, and after close.

        Call it right after setting the stop event to make shutdown immediate
        rather than within-a-slice. Never raises: a shutdown path that itself
        fails is worse than a slice of latency.
        """
        if self._closed:
            return
        try:
            self._wake_w.send(b"\x01")
        except (OSError, ValueError):
            # The socket is closed or its buffer is full. Closed means the ticker
            # is already done, and full means a wake is ALREADY pending, so in
            # both cases the doorbell has been rung as far as the waiter is
            # concerned.
            pass

    def close(self) -> None:
        """Release the doorbell sockets and the selector. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._selector.unregister(self._wake_r)
        except (KeyError, OSError, ValueError):
            # Already unregistered, or the fd died under us. Either way the
            # selector no longer watches it, which is what this line is for.
            pass
        try:
            self._selector.close()
        except OSError:
            # Best effort: the epoll/kqueue fd is being released either way, and
            # close() is documented idempotent, so a shutdown must not raise here.
            pass
        for sock in (self._wake_r, self._wake_w):
            try:
                sock.close()
            except OSError:
                # Already closed; socket.close() is idempotent, so this only
                # guards a platform raising on a double close. Either way the
                # descriptor is released, which is the point.
                pass

    def __enter__(self) -> Ticker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def sleep_penalty_s(nominal: float = 0.05, samples: int = 3) -> float:
    """Measure how much this process tree inflates a plain ``time.sleep``.

    Returns the *extra* seconds a ``sleep(nominal)`` costs beyond ``nominal``
    (~0 in a terminal, ~0.145 under a daemon-started tree on macOS). Exposed so
    that a test asserting a wall-clock ceiling can calibrate against the machine
    it is running on instead of loosening its number until it passes -- and so a
    diagnostic can tell an operator that the rates their robot reports are being
    taxed by the environment it was started in.

    Takes the MEDIAN of several samples, and that is not defensive padding: on
    the machine this was written for, the first sleep after a burst of CPU-bound
    work came back 3.3ms late while every following one was ~145ms late
    (measured: ``[139.7, 148.7, 144.6, 148.7, 148.8, 142.5]`` ms for a 20ms
    sleep). A one-shot probe therefore reports "this machine is fine" often
    enough to make any test gated on it flaky in the direction that hides the
    bug.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples!r}")
    extras = []
    for _ in range(samples):
        start = time.perf_counter()
        time.sleep(nominal)
        extras.append(max(0.0, (time.perf_counter() - start) - nominal))
    extras.sort()
    return extras[len(extras) // 2]
