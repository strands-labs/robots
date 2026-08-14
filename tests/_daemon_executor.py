"""A single-worker executor for fixtures whose work item can be abandoned.

A fixture that hands :attr:`strands_robots.hardware_robot.Robot._executor` a
``ThreadPoolExecutor`` and then lets a test give up on the work item --
``future.result(timeout=...)`` raising ``TimeoutError`` -- leaves that work item
still running on a **non-daemon** worker. ``ThreadPoolExecutor`` registers an
interpreter-exit hook (``threading._register_atexit``) that joins every worker
it started, and ``shutdown(wait=False)`` does not detach one that is already
running. The process therefore cannot exit while the work item is wedged:
pytest prints the verdict and the job then hangs, so a red test is delivered as
a hung job rather than a failed one.

Measured on this shape -- a one-worker pool, a wedged work item, a wait that
gives up after 2s, and ``shutdown(wait=False)`` in teardown -- pytest reported
``1 failed in 2.03s`` and the process was still alive when a 45s wall clock
killed it (exit 124).

This executor keeps the semantics the fixtures rely on -- one worker,
submissions serialized in submit order, ``submit`` returning a real
:class:`concurrent.futures.Future`, ``shutdown(wait=True)`` draining -- while
running that worker as a **daemon** thread, which the interpreter does not join
at exit. An abandoned work item then costs the test its own verdict, reported at
the ``timeout=`` the test chose, and nothing more.

It does not hide the failure it makes survivable: the wait still raises
``TimeoutError`` and the test still fails. Only the hang goes away.

One deliberate difference from ``ThreadPoolExecutor``: a work item raising a
``BaseException`` that is not an ``Exception`` (``KeyboardInterrupt``,
``SystemExit``) is not captured into the future. ``ThreadPoolExecutor`` captures
it; here it is allowed to propagate and end the daemon worker, because turning
an interpreter-level interrupt into a future result is exactly the swallow that
makes an interrupted run look like a completed one.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

_SHUTDOWN = object()
"""Sentinel queued by :meth:`DaemonThreadExecutor.shutdown` to end the worker."""


class DaemonThreadExecutor(Executor):
    """``ThreadPoolExecutor(max_workers=1)`` with a daemon worker.

    Only the two members the hardware task path uses are implemented --
    ``submit`` (``Robot.start_task``) and ``shutdown`` (``Robot.cleanup``) --
    because those are the only two a fixture hands to production code.
    """

    def __init__(self, max_workers: int = 1, thread_name_prefix: str = "daemon_executor") -> None:
        """Build the executor.

        Args:
            max_workers: Must be ``1``. The queue below serializes work on a
                single worker, which is what the fixtures using this helper
                already asked ``ThreadPoolExecutor`` for; a larger pool would
                need a different implementation rather than a different number.
            thread_name_prefix: Prefix for the worker thread name, so a stuck
                worker is identifiable in a thread dump.

        Raises:
            ValueError: If ``max_workers`` is not ``1``.
        """
        if max_workers != 1:
            raise ValueError(
                f"DaemonThreadExecutor serializes on one worker; max_workers must be 1, got {max_workers!r}"
            )
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread_name_prefix = thread_name_prefix
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._shutdown = False

    @property
    def worker(self) -> threading.Thread | None:
        """The worker thread, or ``None`` before the first :meth:`submit`."""
        return self._worker

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        """Queue ``fn`` for the worker and return its future.

        Args:
            fn: Callable to run on the worker.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            A future completed by the worker.

        Raises:
            RuntimeError: If :meth:`shutdown` has already been called, matching
                ``ThreadPoolExecutor``.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._worker is None:
                self._worker = threading.Thread(target=self._serve, name=f"{self._thread_name_prefix}_0", daemon=True)
                self._worker.start()
            future: Future[_T] = Future()
            self._queue.put((future, fn, args, kwargs))
            return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stop accepting work and end the worker.

        Args:
            wait: Join the worker before returning. A wedged work item makes
                this block, exactly as ``ThreadPoolExecutor`` does -- inside the
                caller, where a test timeout can bound it, rather than at
                interpreter exit where nothing can.
            cancel_futures: Cancel work that is queued but not yet started.
        """
        with self._lock:
            first = not self._shutdown
            self._shutdown = True
            worker = self._worker
        if cancel_futures:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _SHUTDOWN:
                    item[0].cancel()
        if first:
            self._queue.put(_SHUTDOWN)
        if wait and worker is not None:
            worker.join()

    def _serve(self) -> None:
        """Run queued work items in order until the shutdown sentinel arrives."""
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - delivered to the caller via the future
                future.set_exception(exc)
            else:
                future.set_result(result)
