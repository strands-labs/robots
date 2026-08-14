"""Shared runtime OpenGL-availability probe for MuJoCo render tests.

MuJoCo render tests need a working offscreen GL context. Historically these
tests were gated behind a blunt ``skipif(CI == "true" and not
ROBOT_TEST_MUJOCO)`` opt-out, which skipped them on *every* CI runner - even
runners that do have a usable GL context (EGL/OSMesa). The result was that
GL-backed contracts (render sandboxing, camera-resolution behaviour, video
capture) went unverified anywhere ``CI`` was set unless a human remembered to
export ``ROBOT_TEST_MUJOCO=1``.

Instead, probe once whether a tiny offscreen render actually succeeds under the
ambient ``MUJOCO_GL`` backend and skip only when it genuinely fails. Setting
``ROBOT_TEST_MUJOCO=0`` forces the skip (e.g. to keep a known-bad runner from
attempting GL at all).

Constructing that probe renderer at most once per process is a **safety**
invariant, not a cost saving. On a host where the first attempt fails - the
headless case this probe exists for - the failure is graceful and reports
``False``, but a *second* construction in the same process does not raise: it
aborts the interpreter uncatchably, because the GL loader re-enters a
function-local static guard whose own initialisation already failed::

    libc++abi: __cxa_guard_acquire detected recursive initialization
    Aborted (core dumped)

``except Exception`` cannot see that, and the abort surfaces in whichever test
called :func:`gl_available` next rather than in the caller that re-armed the
probe. So the hardware answer is latched in a module-level sentinel that the
``functools.cache`` on the public entry point cannot reset. Clearing that cache
is a reasonable thing for a test to do - it is how the ``ROBOT_TEST_MUJOCO=0``
contract is exercised - and it re-reads the environment, but it can never
re-run the renderer construction.

Usage::

    from tests.simulation.mujoco._gl_probe import requires_gl

    @requires_gl
    def test_something_that_renders(): ...
"""

from __future__ import annotations

import functools
import os

import pytest

#: Latched answer from the one probe-renderer construction this process allows.
#: ``None`` means "not probed yet". Deliberately kept *outside* the
#: ``functools.cache`` on :func:`gl_available` so that clearing that cache
#: cannot re-arm the probe; a second construction on a host whose first attempt
#: failed aborts the interpreter (see the module docstring).
_HARDWARE_PROBE_RESULT: bool | None = None


def _probe_gl_once() -> bool:
    """Return whether an offscreen render works, constructing at most one renderer.

    The construction runs on the first call in the process and never again, even
    if :func:`gl_available` has had its cache cleared in between.
    """
    global _HARDWARE_PROBE_RESULT
    if _HARDWARE_PROBE_RESULT is not None:
        return _HARDWARE_PROBE_RESULT
    # Latch the negative *before* attempting the construction: a host whose
    # attempt fails gracefully must not be handed a second one, and the latch
    # has to already be in place when that retry would otherwise happen.
    _HARDWARE_PROBE_RESULT = False
    try:
        import mujoco as mj
    except ImportError:
        return _HARDWARE_PROBE_RESULT
    try:
        model = mj.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        renderer = mj.Renderer(model, height=1, width=1)
    except Exception:
        # Any failure (no EGL/OSMesa, no display, driver error) means the host
        # cannot render offscreen; the dependent tests must skip cleanly.
        return _HARDWARE_PROBE_RESULT
    else:
        del renderer
        _HARDWARE_PROBE_RESULT = True
        return _HARDWARE_PROBE_RESULT


@functools.cache
def gl_available() -> bool:
    """Return True when a minimal offscreen MuJoCo render context can be built.

    ``ROBOT_TEST_MUJOCO=0`` forces a negative result without probing. The
    hardware answer comes from :func:`_probe_gl_once`, which runs the underlying
    1x1 renderer construction at most once per process - including across a
    ``cache_clear()`` on this function.
    """
    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":
        return False
    return _probe_gl_once()


requires_gl = pytest.mark.skipif(
    not gl_available(),
    reason="no usable OpenGL context (headless without EGL/OSMesa); force-skip with ROBOT_TEST_MUJOCO=0",
)
