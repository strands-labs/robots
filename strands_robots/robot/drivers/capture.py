"""One capture thread per camera: it owns the device (open, read, close on that thread) and keeps the latest frame.

A device read waits for the device's next frame (OpenCV blocks, AVFoundation polls); on the control thread that wait
is the tick's, once per camera. Here the wait is the thread's, and ``latest`` hands the control thread whatever frame
arrived last, with the ``time.monotonic()`` it was captured at, without touching the device.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

IDLE_S = 0.005  # after a read that brought no frame: an unplugged OpenCV device answers at once, so never spin on it
JOIN_S = 2.0  # how long ``stop`` waits for a read in progress; a hung device's thread closes it when the read returns


class Grabber:
    """Opens ``open_device()`` on its own daemon thread and reads it until ``stop``.

    ``open_device`` returns an object with ``read() -> ndarray | None`` and ``close()``. An exception from opening or
    reading ends the thread and is re-raised by ``wait``/``latest`` on the caller's thread, naming the camera."""

    def __init__(self, camera: str, open_device: Callable[[], Any]) -> None:
        self.camera = camera
        self._open_device = open_device
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float] | None = None
        self._error: Exception | None = None
        self._ready = threading.Event()  # set by the first frame, or by the error that ends the thread
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"camera-{camera}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            device = self._open_device()
        except Exception as exc:  # noqa: BLE001  (kept and re-raised on the caller's thread)
            self._fail(exc)
            return
        try:
            while not self._stop.is_set():
                rgb = device.read()
                if rgb is None:
                    self._stop.wait(IDLE_S)
                    continue
                with self._lock:
                    self._frame = (rgb, time.monotonic())
                self._ready.set()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
        finally:
            device.close()

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            self._error = exc
        self._ready.set()

    def _raise(self) -> None:
        """The thread's exception, as its own type (an ImportError stays one) with the camera named first."""
        err = self._error
        if err is None:
            return
        try:
            named = type(err)(f"camera {self.camera!r}: {err}")
        except Exception:  # noqa: BLE001  (a type that takes other arguments)
            named = RuntimeError(f"camera {self.camera!r}: {err}")
        raise named from err

    def wait(self, timeout: float) -> None:
        """Until the first frame; a device that delivers none in ``timeout`` seconds is refused naming the camera."""
        if not self._ready.wait(timeout):
            raise TimeoutError(f"camera {self.camera!r} delivered no frame in {timeout:g} s")
        self._raise()

    def latest(self) -> tuple[np.ndarray, float] | None:
        """The newest ``(frame, captured_at)``, or None before the first; never waits on the device."""
        with self._lock:
            self._raise()
            return self._frame

    def stop(self, *, join: bool = True) -> None:
        self._stop.set()
        if join and self.thread is not threading.current_thread():
            self.thread.join(JOIN_S)


def stop_all(grabbers: dict[str, Grabber]) -> None:
    """What a ``Cameras`` nobody closed does when it is collected: every thread told to stop, none waited for."""
    for grabber in grabbers.values():
        grabber.stop(join=False)
