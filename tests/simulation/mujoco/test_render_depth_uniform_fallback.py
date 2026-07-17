"""render_depth degrades gracefully when a view has no depth variation.

Depth shading maps metric depth onto an 8-bit grayscale PNG by normalizing
against ``span = depth_max - depth_min``. When every visible pixel sits at the
same distance (an empty view where all rays reach the far-clip plane, or a
single flat surface filling the frame), ``span`` is zero. Dividing by it would
produce ``nan`` that collapses to an all-black frame - a misleading render that
looks like a failure. :meth:`render_depth` guards this case and emits a flat
mid-gray (128) frame instead, while still reporting the true metric bounds.

These tests inject a scripted uniform depth buffer so the zero-span branch runs
deterministically on every runner, independent of the local GL driver.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="depth_uniform_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class _FakeDepthRenderer:
    """Scripted offscreen renderer returning a fixed metric depth buffer.

    Mirrors the ``mujoco.Renderer`` depth contract (metric meters straight from
    ``render()``) so the test needs no real GL context.
    """

    def __init__(self, depth: Any) -> None:
        self._depth = depth

    def update_scene(self, data: Any, camera: Any = None, scene_option: Any = None) -> None:
        pass

    def enable_depth_rendering(self) -> None:
        pass

    def disable_depth_rendering(self) -> None:
        pass

    def render(self) -> Any:
        return self._depth


def _decode_png(result: dict) -> Any:
    """Extract the grayscale PNG from a render_depth response as a 2D array."""
    import numpy as np
    from PIL import Image

    png_bytes = next(block["image"]["source"]["bytes"] for block in result["content"] if "image" in block)
    return np.asarray(Image.open(io.BytesIO(png_bytes)))


def test_uniform_depth_view_renders_flat_gray_not_black(sim, monkeypatch):
    # Every pixel at the same distance -> span == 0. The fallback must fill the
    # frame with mid-gray (128), NOT divide by zero into an all-black frame.
    import numpy as np

    uniform = np.full((3, 4), 1.5, dtype=np.float32)
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeDepthRenderer(uniform))

    result = sim.render_depth(camera_name="default", width=4, height=3)

    assert result["status"] == "success", result
    gray = _decode_png(result)
    assert gray.shape == (3, 4)
    # Flat mid-gray everywhere - the divide-by-zero guard's whole point.
    assert np.all(gray == 128), np.unique(gray)


def test_uniform_depth_still_reports_true_metric_bounds(sim, monkeypatch):
    # The flat-gray fallback is a display convention only: the JSON block must
    # still carry the real (equal) metric min/max so downstream consumers see
    # the actual distance, not the 128 shading value.
    import numpy as np

    uniform = np.full((2, 2), 0.8, dtype=np.float32)
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeDepthRenderer(uniform))

    result = sim.render_depth(camera_name="default", width=2, height=2)

    assert result["status"] == "success", result
    json_block = next(block["json"] for block in result["content"] if "json" in block)
    assert json_block["depth_min"] == pytest.approx(0.8, abs=1e-4)
    assert json_block["depth_max"] == pytest.approx(0.8, abs=1e-4)
    assert json_block["depth_min"] == json_block["depth_max"]


def test_varied_depth_view_shades_near_bright_far_dark(sim, monkeypatch):
    # Contrast case (span > 0): the normal shading path must still produce a
    # non-flat frame where the nearest pixel is brightest and the farthest is
    # darkest, confirming the fallback is scoped to the zero-span case only.
    import numpy as np

    varied = np.array([[0.5, 2.5], [1.0, 2.0]], dtype=np.float32)
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeDepthRenderer(varied))

    result = sim.render_depth(camera_name="default", width=2, height=2)

    assert result["status"] == "success", result
    gray = _decode_png(result)
    assert gray.max() > gray.min()  # not flat
    # Nearest surface (0.5 m) is brightest; farthest (2.5 m) is darkest.
    assert gray[0, 0] == 255
    assert gray[0, 1] == 0
