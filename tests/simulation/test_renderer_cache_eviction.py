"""The per-thread MuJoCo renderer cache is bounded and evicts oldest-first.

``RenderingMixin._get_renderer`` caches a ``mujoco.Renderer`` per
``(width, height)`` on a ``threading.local`` (each renderer binds a GL context
to its creating thread). Without a bound, a caller that renders at many
resolutions - a common pattern when sweeping preview / record / eval sizes -
would accumulate GL contexts for the lifetime of the ``Simulation`` and leak
GPU memory. The cache therefore holds at most four renderers per thread and
evicts the first-inserted (FIFO) one when a fifth distinct resolution arrives;
a GL driver that fails to free the evicted context must not break the render.

These tests pin that contract through the public ``render`` surface, counting
``mujoco.Renderer`` construction (the observable resource-allocation side
effect) rather than inspecting the private cache:

* distinct resolutions build one renderer each;
* re-rendering a cached resolution reuses it (no new build);
* exceeding the cap evicts the oldest first, so re-rendering the evicted
  resolution rebuilds it while a still-cached one does not (proves FIFO and
  that the cache stays bounded);
* a ``close()`` that raises while evicting is swallowed - the render succeeds.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (EGL/OSMesa required for offscreen rendering)",
)

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <camera name="side" pos="0.8 -0.8 0.4" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

_CAM = "arm1/side"
# Five distinct, small offscreen sizes (all well under any framebuffer cap).
# The cache bound is four, so _E forces an eviction of the oldest (_A).
_A, _B, _C, _D, _E = (32, 32), (48, 48), (64, 64), (80, 80), (96, 96)


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="renderer_cache", mesh=False)
    try:
        sim.create_world()
        r = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert r["status"] == "success", r
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def build_counter(monkeypatch):
    """Count ``mujoco.Renderer`` constructions (the cached GL resource)."""
    orig_init = mj.Renderer.__init__
    calls = {"n": 0}

    def counting_init(self, *args, **kwargs):
        calls["n"] += 1
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(mj.Renderer, "__init__", counting_init)
    return calls


def _render(sim, size):
    r = sim.render(camera_name=_CAM, width=size[0], height=size[1])
    assert r["status"] == "success", r
    return r


@requires_gl
class TestRendererCacheEviction:
    def test_distinct_resolutions_each_build_one_renderer(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        assert build_counter["n"] == 4

    def test_cached_resolution_is_reused_not_rebuilt(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        # Re-render the same four: all cached, so no renderer is constructed.
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        assert build_counter["n"] == 4

    def test_exceeding_cap_evicts_oldest_first_in_first_out(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        # Fifth distinct resolution: the cache is at the cap of four, so the
        # oldest (_A) is evicted rather than the cache growing to five.
        _render(sim_with_arm, _E)
        assert build_counter["n"] == 5

        # A still-cached resolution reuses its renderer (no rebuild)...
        _render(sim_with_arm, _D)
        assert build_counter["n"] == 5

        # ...while the evicted oldest (_A) must be rebuilt, proving it was the
        # one dropped (first-inserted-first-out), not a more-recent resolution.
        _render(sim_with_arm, _A)
        assert build_counter["n"] == 6

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_close_failure_during_eviction_is_fail_soft(self, sim_with_arm, monkeypatch):
        """A GL context that fails to free on eviction must not break rendering.

        Eviction best-effort-closes the evicted renderer; if the driver's
        ``close()`` raises, the exception is swallowed so the render that
        triggered the eviction still returns a frame.
        """
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)

        def _boom(self):
            raise RuntimeError("GL context free failed")

        monkeypatch.setattr(mj.Renderer, "close", _boom)

        # The fifth resolution evicts _A; its close() now raises but the
        # render must still succeed.
        result = sim_with_arm.render(camera_name=_CAM, width=_E[0], height=_E[1])
        assert result["status"] == "success", result
