"""render_depth surfaces the GPU depth-precision warning correctly.

When the OpenGL driver emits the benign one-time ``ARB_clip_control`` notice
(older macOS GPUs, some headless drivers), :meth:`render_depth` must:

  1. surface a plain-language "precision degraded" warning in the response text
     so the calling agent hears about it,
  2. suppress the raw ``ARB_clip_control`` line from stderr (it is now carried in
     the response) while still forwarding any genuine stderr unchanged, and
  3. cache the warning so subsequent depth renders skip the stderr-capture path.

The warning fires only on hardware lacking ``ARB_clip_control``, so a real GPU
never exercises these branches; and under pytest the C-level fd-2 capture in
``capture_stderr_fd`` no-ops (pytest already owns fd 2), so a fake that merely
writes to ``sys.stderr`` does not reach them either. Injecting the captured
buffer directly is the only way to pin this contract deterministically on every
runner regardless of the local GL driver.
"""

from __future__ import annotations

import contextlib
import io
import sys

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import rendering as rendering_mod
from strands_robots.simulation.mujoco.simulation import Simulation

_ARB_LINE = "GL notice: ARB_clip_control unavailable, depth precision degraded\n"
_GENUINE_LINE = "libGL warning: unrelated genuine driver diagnostic\n"


@pytest.fixture
def sim():
    s = Simulation(tool_name="depth_warn_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class _FakeDepthRenderer:
    """Scripted offscreen renderer returning a fixed metric depth buffer.

    Mirrors the ``mujoco.Renderer`` depth contract (metric meters straight from
    ``render()``) so the test needs no real GL context.
    """

    def __init__(self, depth):
        self._depth = depth
        self.enable_calls = 0
        self.disable_calls = 0

    def update_scene(self, data, camera=None, scene_option=None):
        pass

    def enable_depth_rendering(self):
        self.enable_calls += 1

    def disable_depth_rendering(self):
        self.disable_calls += 1

    def render(self):
        return self._depth


def _fake_capture(captured_text: str):
    """Return a ``capture_stderr_fd`` stand-in yielding ``captured_text``.

    Mirrors the real context manager's contract: yields a single-element list
    whose element holds the captured stderr after the block exits.
    """

    @contextlib.contextmanager
    def _cm():
        box = [""]
        try:
            yield box
        finally:
            box[0] = captured_text

    return _cm


def _depth_buf():
    np = pytest.importorskip("numpy")
    return np.array([[0.75, 2.40], [1.50, 1.50]], dtype=np.float32)


def test_arb_clip_warning_surfaced_suppressed_and_cached(sim, monkeypatch):
    fake = _FakeDepthRenderer(_depth_buf())
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: fake)
    # A GPU without ARB_clip_control emits the benign notice mixed with any
    # genuine stderr. Inject that so the warning branch runs deterministically.
    monkeypatch.setattr(rendering_mod, "capture_stderr_fd", _fake_capture(_ARB_LINE + _GENUINE_LINE))
    # Capture the forwarded stderr without touching the real file descriptor.
    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "__stderr__", fake_stderr)

    # First depth render: no cached warning yet, so the capture path executes.
    assert getattr(sim, "_depth_warn_text", None) is None
    r1 = sim.render_depth(camera_name="default", width=2, height=2)
    assert r1["status"] == "success", r1

    # The degraded-precision warning is cached and surfaced in the response text.
    warn = sim._depth_warn_text
    assert "ARB_clip_control" in warn
    assert "approximate" in warn
    assert warn in r1["content"][0]["text"]

    # Genuine stderr is forwarded; the benign ARB line is dropped, not echoed.
    forwarded = fake_stderr.getvalue()
    assert "genuine driver diagnostic" in forwarded
    assert "ARB_clip_control" not in forwarded

    # A second render reuses the cached warning and skips the capture path
    # entirely (the already-warned branch): the capture stand-in below would
    # inject a bogus buffer if it were consulted, so the cache must win.
    monkeypatch.setattr(rendering_mod, "capture_stderr_fd", _fake_capture("must-not-be-read-again"))
    calls_before = fake.enable_calls
    r2 = sim.render_depth(camera_name="default", width=2, height=2)
    assert r2["status"] == "success"
    assert sim._depth_warn_text == warn
    assert warn in r2["content"][0]["text"]
    assert "must-not-be-read-again" not in r2["content"][0]["text"]
    # The already-warned branch still renders (no re-capture): the renderer was
    # driven again and the depth/disable pair ran.
    assert fake.enable_calls == calls_before + 1
    assert fake.disable_calls == fake.enable_calls


def test_no_arb_warning_leaves_depth_warn_text_empty(sim, monkeypatch):
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeDepthRenderer(_depth_buf()))
    # A capable GPU produces no ARB notice: the cached warning text is empty and
    # nothing is appended to the response summary.
    monkeypatch.setattr(rendering_mod, "capture_stderr_fd", _fake_capture(""))
    r = sim.render_depth(camera_name="default", width=2, height=2)
    assert r["status"] == "success", r
    assert sim._depth_warn_text == ""
    text = r["content"][0]["text"]
    assert "ARB_clip_control" not in text


def test_genuine_stderr_forward_is_best_effort_when_console_detached(sim, monkeypatch):
    # Forwarding genuine stderr is best-effort: if the original __stderr__ is
    # closed/detached (pytest teardown, Jupyter), the write raises and must be
    # swallowed so a depth render never fails on a logging side effect.
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeDepthRenderer(_depth_buf()))
    monkeypatch.setattr(rendering_mod, "capture_stderr_fd", _fake_capture(_ARB_LINE + _GENUINE_LINE))

    class _DetachedStderr:
        def write(self, _data):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "__stderr__", _DetachedStderr())

    r = sim.render_depth(camera_name="default", width=2, height=2)
    assert r["status"] == "success", r
    # The warning is still surfaced even though the stderr forward was dropped.
    assert "ARB_clip_control" in sim._depth_warn_text
