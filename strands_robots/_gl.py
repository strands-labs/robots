"""Pick MuJoCo's GL backend before the first ``import mujoco`` (it reads MUJOCO_GL once), and own the one GL thread.

Linux without a display → ``egl`` if libEGL loads, else ``osmesa``; macOS/Windows use the native
context MuJoCo picks itself. An explicit ``MUJOCO_GL`` is always respected.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import platform
import queue
import sys
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future
from functools import wraps
from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

_GL_THREAD = "strands-gl"
_gl: _GLThread | None = None
_gl_lock = threading.Lock()
_closers: list[weakref.WeakMethod[Callable[[], None]]] = []


def _loads(lib: str) -> bool:
    try:
        ctypes.CDLL(lib)
        return True
    except OSError:
        return False


def ensure_gl() -> str:
    """Set ``MUJOCO_GL`` for headless rendering when it is unset. Returns the value in effect."""
    cur = os.environ.get("MUJOCO_GL", "").strip().lower()
    if cur:
        return cur
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        for value, lib in (("egl", "libEGL.so.1"), ("osmesa", "libOSMesa.so")):
            if _loads(lib):
                os.environ["MUJOCO_GL"] = value
                return value
    return cur


def _glfw_on_macos() -> bool:
    """Whether MuJoCo renders through GLFW on macOS, which makes windows on the main thread only (else SIGTRAP)."""
    if platform.system() != "Darwin":
        return False
    import mujoco.gl_context  # only the MuJoCo backend gets here, so this import is already done

    return getattr(getattr(mujoco.gl_context, "GLContext", None), "__module__", "") == "mujoco.glfw"


def gl_in_effect() -> str:
    """Say which GL MuJoCo really uses: it reads MUJOCO_GL once, when ``mujoco`` is first imported."""
    want = ensure_gl()
    ctx = sys.modules.get("mujoco.rendering.classic.gl_context") or sys.modules.get("mujoco.gl_context")
    chosen = getattr(ctx, "_MUJOCO_GL", want)
    if chosen == want:
        return f"MUJOCO_GL={want or 'default'}"
    fix = f"set MUJOCO_GL={want} before the first import of mujoco"
    return f"mujoco was imported with MUJOCO_GL={chosen or 'unset'}; {fix}"


class _GLThread:
    """A daemon worker: it outlives ``threading._shutdown`` so contexts can still be freed on it at exit
    (a ``ThreadPoolExecutor`` is joined before ``atexit`` runs, which would leave them to the collector)."""

    def __init__(self) -> None:
        self.q: queue.SimpleQueue[tuple[Callable[..., Any], tuple[Any, ...], Future[Any]] | None] = queue.SimpleQueue()
        self.t = threading.Thread(target=self._loop, name=_GL_THREAD, daemon=True)
        self.t.start()

    def _loop(self) -> None:
        while (item := self.q.get()) is not None:
            fn, args, fut = item
            try:
                fut.set_result(fn(*args))
            except Exception as exc:  # noqa: BLE001 — a render's error belongs to the caller, through the future
                fut.set_exception(exc)
            except BaseException as exc:  # KeyboardInterrupt / SystemExit: the caller gets it, and the worker ends
                fut.set_exception(exc)
                raise

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        if not self.t.is_alive():
            raise RuntimeError("the GL thread has stopped")
        fut: Future[Any] = Future()
        self.q.put((fn, args, fut))
        return fut

    def stop(self) -> None:
        self.q.put(None)
        self.t.join(5.0)


def on_gl_thread(fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn`` on the one thread that owns every GL context in the process.

    A context is bound to the thread that made it: on macOS (CGL) a render from any other thread
    blocks forever, EGL refuses it (EGL_BAD_ACCESS), so a second ``run()`` from a background ``Task``
    would hang. Every render goes here, except with ``MUJOCO_GL=glfw`` on macOS: there ``fn`` runs on the
    thread that asks, the main thread.
    """
    global _gl
    if threading.current_thread().name == _GL_THREAD or _glfw_on_macos():
        return fn(*args)
    with _gl_lock:
        if _gl is None or not _gl.t.is_alive():  # first use, or a worker a KeyboardInterrupt ended
            _gl = _GLThread()
            atexit.register(_release_gl)
    return _gl.submit(fn, *args).result()


def gl_thread(fn: Callable[P, T]) -> Callable[P, T]:
    """Decorate a method that touches a GL context: it always runs on the one GL thread."""

    @wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        return on_gl_thread(lambda: fn(*args, **kwargs))

    return run


def close_on_gl_thread_at_exit(close: Callable[[], None]) -> None:
    """Run ``close`` (a bound method) on the GL thread before the interpreter exits: a context the collector frees
    from another thread at shutdown is a segfault on macOS."""
    _closers.append(weakref.WeakMethod(close))  # type: ignore[arg-type]


def release_before_gl_teardown() -> None:
    """Call after making a context: ``mujoco.egl`` registers ``eglTerminate`` at exit with its first one, and exit
    hooks run last-registered first, so the release moves behind it or the contexts die on a dead display."""
    atexit.unregister(_release_gl)
    atexit.register(_release_gl)


def _release_gl() -> None:
    global _gl
    if _gl is None:
        return
    for ref in reversed(_closers):
        fn = ref()
        if fn is not None:
            try:
                _gl.submit(fn).result()
            except Exception:  # best-effort teardown at exit: nobody left to report to
                pass
    _closers.clear()
    _gl.stop()
    _gl = None


ensure_gl()  # at `import strands_robots`: a headless Linux host renders even if the caller imports mujoco next
