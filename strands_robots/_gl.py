"""Pick MuJoCo's GL backend before the first ``import mujoco`` (it reads MUJOCO_GL once).

Linux without a display → ``egl`` if libEGL loads, else ``osmesa``; macOS/Windows use the native
context MuJoCo picks itself. An explicit ``MUJOCO_GL`` is always respected.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
_GL_THREAD = "strands-gl"
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


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


def on_gl_thread(fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn`` on the one thread that owns every GL context in the process.

    A context is bound to the thread that made it: on macOS (CGL) a render from any other thread
    blocks forever, so a second ``run()`` from a background ``Task`` would hang. Every render goes here,
    except with ``MUJOCO_GL=glfw`` on macOS: there ``fn`` runs on the thread that asks, the main thread.
    """
    global _pool
    if threading.current_thread().name.startswith(_GL_THREAD) or _glfw_on_macos():
        return fn(*args)
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(1, thread_name_prefix=_GL_THREAD)
    return _pool.submit(fn, *args).result()


def gl_thread(fn: Callable[P, T]) -> Callable[P, T]:
    """Decorate a method that touches a GL context: it always runs on the one GL thread."""

    @wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        return on_gl_thread(lambda: fn(*args, **kwargs))

    return run
