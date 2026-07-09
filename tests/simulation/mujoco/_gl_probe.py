"""Shared runtime OpenGL-availability probe for MuJoCo render tests.

MuJoCo render tests need a working offscreen GL context. Historically these
tests were gated behind a blunt ``skipif(CI == "true" and not
ROBOT_TEST_MUJOCO)`` opt-out, which skipped them on *every* CI runner - even
runners that do have a usable GL context (EGL/OSMesa). The result was that
GL-backed contracts (render sandboxing, camera-resolution behaviour, video
capture) went unverified anywhere ``CI`` was set unless a human remembered to
export ``ROBOT_TEST_MUJOCO=1``.

Instead, probe once whether a tiny offscreen render actually succeeds under the
ambient ``MUJOCO_GL`` backend and skip only when it genuinely fails. The probe
result is cached so the cost is a single 1x1 renderer construction per session.
Setting ``ROBOT_TEST_MUJOCO=0`` forces the skip (e.g. to keep a known-bad
runner from attempting GL at all).

Usage::

    from tests.simulation.mujoco._gl_probe import requires_gl

    @requires_gl
    def test_something_that_renders(): ...
"""

from __future__ import annotations

import functools
import os

import pytest


@functools.cache
def gl_available() -> bool:
    """Return True when a minimal offscreen MuJoCo render context can be built.

    Cached: the underlying 1x1 renderer construction runs at most once per test
    session. ``ROBOT_TEST_MUJOCO=0`` forces a negative result without probing.
    """
    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":
        return False
    try:
        import mujoco as mj
    except ImportError:
        return False
    try:
        model = mj.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        renderer = mj.Renderer(model, height=1, width=1)
    except Exception:
        # Any failure (no EGL/OSMesa, no display, driver error) means the host
        # cannot render offscreen; the dependent tests must skip cleanly.
        return False
    else:
        del renderer
        return True


requires_gl = pytest.mark.skipif(
    not gl_available(),
    reason="no usable OpenGL context (headless without EGL/OSMesa); force-skip with ROBOT_TEST_MUJOCO=0",
)
