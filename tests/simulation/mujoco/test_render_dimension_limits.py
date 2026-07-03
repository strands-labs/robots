"""Render dimension + payload-size safety caps.

``render()`` is an LLM-callable tool, so the width/height and the
``STRANDS_ROBOTS_RENDER_MAX_BYTES`` size cap are attacker-influenced. These
tests pin the out-of-memory / bad-input guardrails: non-integer dimensions, the
absolute 4096x4096 framebuffer ceiling, the per-model offscreen framebuffer cap,
and a non-positive byte budget. The dimension guards short-circuit before any
GL context is created, so they are deterministic even on a headless host.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import _max_render_bytes  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty world (no robot, no GL needed)."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


class TestRenderDimensionCaps:
    def test_non_integer_dimensions_rejected(self, sim_with_world):
        """A string width is refused with a type-explicit error, not a crash."""
        res = sim_with_world.render(camera_name="default", width="640", height=480)
        assert res["status"] == "error"
        assert "must be int" in res["content"][0]["text"]

    def test_dimensions_over_absolute_ceiling_rejected(self, sim_with_world):
        """Beyond the hard 4096 ceiling is refused regardless of model config."""
        res = sim_with_world.render(camera_name="default", width=8000, height=480)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "absolute maximum" in text
        assert "4096x4096" in text

    def test_dimensions_over_model_offscreen_cap_rejected(self, sim_with_world):
        """Within the absolute ceiling but past the model's offscreen framebuffer
        cap (default 1280x960) is refused with the actual cap surfaced."""
        cap_w = int(sim_with_world._world._model.vis.global_.offwidth)
        assert cap_w < 4096  # precondition: model cap is below the hard ceiling
        res = sim_with_world.render(camera_name="default", width=cap_w + 1, height=48)
        assert res["status"] == "error"
        assert "offscreen framebuffer cap" in res["content"][0]["text"]


class TestRenderMaxBytesCap:
    def test_non_positive_byte_budget_rejected(self, monkeypatch):
        """A non-positive size cap surfaces an error rather than disabling the cap."""
        monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "-5")
        with pytest.raises(ValueError, match="must be positive"):
            _max_render_bytes()

    def test_zero_byte_budget_rejected(self, monkeypatch):
        """Zero is treated as non-positive (would otherwise reject every render)."""
        monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "0")
        with pytest.raises(ValueError, match="must be positive"):
            _max_render_bytes()
