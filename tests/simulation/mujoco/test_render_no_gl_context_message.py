"""render / render_depth surface a clean error when no OpenGL context exists.

When the offscreen renderer cannot be built (no EGL/OSMesa GL context on the
host), :meth:`render` and :meth:`render_depth` short-circuit with a structured
``status=error`` agent-tool dict instead of raising. That message is read
verbatim by an LLM caller, so it must be a clean sentence: no stray leading or
trailing whitespace, and it must name the failure and the actionable fix
(install EGL/OSMesa).

The renderer-None branch is exercised deterministically by monkeypatching
``_get_renderer`` to return ``None``, so the test runs on every runner
regardless of the local GL driver.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="no_gl_context_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _error_text(result: dict) -> str:
    assert result["status"] == "error", result
    return next(block["text"] for block in result["content"] if "text" in block)


def test_render_reports_clean_message_when_no_gl_context(sim, monkeypatch):
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    text = _error_text(sim.render(camera_name="default", width=4, height=3))

    # No orphaned whitespace (a leading space here is a real cosmetic defect
    # left over from stripping a prefix glyph); actionable and self-describing.
    assert text == text.strip()
    assert "Rendering unavailable" in text
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


def test_render_depth_reports_clean_message_when_no_gl_context(sim, monkeypatch):
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    text = _error_text(sim.render_depth(camera_name="default", width=4, height=3))

    assert text == text.strip()
    assert "Depth rendering unavailable" in text
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)
