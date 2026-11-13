"""Reopen a lost LeRobot session in the background, with backoff, for a bounded window.

Connecting blocks for up to the handshake timeout, so it never runs on the control loop's thread: the loop polls
``Redial.poll`` once a tick and holds while it returns None. A refusal that names a different deployment ends the
redial at once; any other failure (nobody answering yet) is retried until the window closes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..base import PolicyFault

FIRST_BACKOFF_S = 0.25
MAX_BACKOFF_S = 2.0


class Redial:
    """One background attempt loop: ``open_fn`` until it returns a session or ``window_s`` has passed."""

    def __init__(self, open_fn: Callable[[], Any], window_s: float, who: str) -> None:
        self.who, self.window_s = who, window_s
        self.lost_at = time.monotonic()
        self.attempts = 0
        self._open, self._stop = open_fn, threading.Event()
        self._lock = threading.Lock()
        self._result: Any = None
        self._error: BaseException | None = None  # the last attempt's, for the fault's message
        self._final: BaseException | None = None  # a refusal that no retry can change
        self._thread = threading.Thread(target=self._run, name="zenoh-redial", daemon=True)
        self._thread.start()

    @property
    def down_s(self) -> float:
        return time.monotonic() - self.lost_at

    def _run(self) -> None:
        backoff = FIRST_BACKOFF_S
        while not self._stop.is_set() and self.down_s < self.window_s:
            self.attempts += 1
            try:
                session = self._open()
            except PolicyFault as e:  # the server is back but is not the deployment this run was bound to
                with self._lock:
                    self._final = e
                return
            except Exception as e:  # noqa: BLE001  (nobody answering yet, or the session is still held)
                self._error = e
            else:
                with self._lock:
                    if self._stop.is_set():  # stopped while connecting: nobody will adopt it
                        session.close()
                    else:
                        self._result = session
                return
            self._stop.wait(min(backoff, max(0.0, self.window_s - self.down_s)))
            backoff = min(backoff * 2, MAX_BACKOFF_S)

    def poll(self) -> Any:
        """The reopened session, None while still trying, or ``PolicyFault`` once the window has closed."""
        with self._lock:
            result, final = self._result, self._final
        if result is not None:
            return result
        if final is not None:
            raise final
        if self._thread.is_alive():
            return None
        last = "" if self._error is None else f"; last attempt: {type(self._error).__name__}: {self._error}"
        raise PolicyFault(
            f"{self.who}: server presence lost; no server came back within reconnect_s={self.window_s:g} s "
            f"({self.attempts} attempts{last})"
        )

    def stop(self) -> None:
        """Stop trying; a session that opens after this is closed by the thread, never adopted."""
        with self._lock:
            self._stop.set()
            result, self._result = self._result, None
        if result is not None:
            result.close()
