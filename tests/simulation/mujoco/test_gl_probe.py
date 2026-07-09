"""Contract tests for the shared MuJoCo GL-availability probe.

These pin the behaviour that the render-test gating relies on: the probe
reports a boolean, honours the ``ROBOT_TEST_MUJOCO=0`` force-skip escape hatch,
and exposes a reusable ``requires_gl`` skip marker. They run without a GL
context (the force-skip and marker-shape assertions never touch a renderer).
"""

from __future__ import annotations

import pytest

from tests.simulation.mujoco import _gl_probe
from tests.simulation.mujoco._gl_probe import gl_available, requires_gl


def test_gl_available_returns_bool() -> None:
    """The probe result is a plain bool the skipif condition can consume."""
    assert isinstance(gl_available(), bool)


def test_robot_test_mujoco_zero_forces_no_gl(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_TEST_MUJOCO=0 forces a negative result without probing hardware."""
    monkeypatch.setenv("ROBOT_TEST_MUJOCO", "0")
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is False
    finally:
        # Do not leak the forced-negative result into other tests.
        _gl_probe.gl_available.cache_clear()


def test_requires_gl_is_a_skip_marker() -> None:
    """requires_gl is a usable skipif MarkDecorator (applies cleanly to tests)."""
    assert isinstance(requires_gl, pytest.MarkDecorator)
    assert requires_gl.name == "skipif"
