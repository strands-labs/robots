"""A background run: a thread with ``stop()`` / ``status`` / ``result``."""

from __future__ import annotations

import threading
from collections.abc import Callable

from .result import RunResult


class Task:
    """A background ``Robot.run``: ``stop()``, ``status``, ``result``."""

    def __init__(self, target: Callable[[threading.Event], RunResult]) -> None:
        self._stop = threading.Event()
        self._result: RunResult | None = None
        self._error: BaseException | None = None
        self._raised = False
        self._thread = threading.Thread(target=self._go, args=(target,), daemon=True)
        self._thread.start()

    def _go(self, target: Callable[[threading.Event], RunResult]) -> None:
        try:
            self._result = target(self._stop)
        except BaseException as exc:  # noqa: BLE001 - surfaced via .result
            self._error = exc

    def stop(self, timeout: float | None = None) -> RunResult | None:
        """Stop and wait for the run to end, its recording saved; ``TimeoutError`` if it is still running."""
        self.halt(timeout)
        if self.running:
            raise TimeoutError(f"the run is still going after {timeout} s; its policy has not returned")
        return self.result

    def _wait(self) -> RunResult | None:
        """Wait for the run to end on its own (a limit, ``stop()`` from another thread, an e-stop): ``result``."""
        self._thread.join()
        return self.result

    def halt(self, timeout: float | None = 5.0) -> None:
        """Stop and join without reading ``result``, so teardown never re-raises the task's error."""
        self._stop.set()
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def status(self) -> str:
        if self.running:
            return "running"
        return "error" if self._error else "done"

    @property
    def result(self) -> RunResult | None:
        """The run's result; a run whose policy raised raises that error here once, then reads None."""
        if self._error is not None and not self._raised:
            self._raised = True
            raise self._error
        return self._result
