"""The world lock: reentrant like ``threading.RLock``, but handed to waiters in arrival order.

A plain ``RLock`` lets the thread that just released it take it straight back, so a reader that loops on
``with sim.lock:`` can starve ``Robot.act()`` for seconds on a host whose scheduler favours the running thread
(5 acts took 31.8 s beside such a reader on aarch64, #4814). Here a release wakes the oldest waiter and a
newcomer queues behind it, so a waiting ``act()`` gets the world at the next release.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class WorldLock:
    """A fair (FIFO) reentrant lock: the owner re-enters freely; everyone else takes a turn in arrival order."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.Lock())
        self._owner: int | None = None
        self._depth = 0
        self._queue: deque[int] = deque()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
                return True
            if self._owner is None and not self._queue:
                self._owner, self._depth = me, 1
                return True
            if not blocking:
                return False
            self._queue.append(me)
            turn = False
            try:  # an interrupt mid-wait (Ctrl-C) still leaves the queue, or it would wedge every waiter behind it
                turn = self._cv.wait_for(
                    lambda: self._owner is None and self._queue[0] == me, None if timeout < 0 else timeout
                )
            finally:
                self._queue.remove(me)
                if turn:
                    self._owner, self._depth = me, 1
                else:
                    self._cv.notify_all()  # the next in line may be free to go now that this waiter left
            return turn

    def release(self) -> None:
        with self._cv:
            if self._owner != threading.get_ident():
                raise RuntimeError("cannot release un-acquired lock")
            self._depth -= 1
            if not self._depth:
                self._owner = None
                self._cv.notify_all()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()
