"""T4: Renderer TLS cache hygiene - destroy and cleanup empty the cache; same
(w,h) reuses an existing renderer. Unit-level (no RSS measurement; see
tests_integ/test_resource_hygiene.py for the process-memory checks)."""

from __future__ import annotations

import pytest

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="renderer_hygiene_test", mesh=False)
    yield s
    s.cleanup()


@requires_gl
class TestRendererTLSCache:
    def test_destroy_empties_main_thread_renderer_cache(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        cached = getattr(sim._renderer_tls, "renderers", {})
        assert cached, "renderer should have been cached after render()"

        sim.destroy()
        cached_after = getattr(sim._renderer_tls, "renderers", {})
        assert not cached_after, "destroy() must empty the main-thread renderer cache"

    def test_render_reuses_renderer_for_identical_dims(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        first = sim._renderer_tls.renderers[(160, 120)]
        sim.render(width=160, height=120)
        second = sim._renderer_tls.renderers[(160, 120)]
        assert first is second

    def test_render_creates_new_renderer_for_different_dims(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        sim.render(width=320, height=240)
        keys = set(sim._renderer_tls.renderers.keys())
        assert (160, 120) in keys
        assert (320, 240) in keys

    def test_create_world_after_destroy_rebuilds_cache(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        sim.destroy()
        sim.create_world()
        sim.render(width=160, height=120)
        assert (160, 120) in sim._renderer_tls.renderers


@requires_gl
class TestRendererCacheBounding:
    """The per-thread renderer cache is bounded to a fixed number of distinct
    resolutions and evicts the oldest entry (closing its GL context) once the
    bound is exceeded. This prevents unbounded GL-context accumulation when a
    caller renders at many different resolutions on one thread."""

    MAX_RESOLUTIONS = 4

    def _distinct_dims(self, count: int) -> list[tuple[int, int]]:
        # Vary width only so each (w, h) is a distinct cache key; small sizes
        # keep the GL allocations cheap.
        return [(160 + i, 120) for i in range(count)]

    def test_cache_bounded_and_evicts_oldest(self, sim):
        sim.create_world()
        dims = self._distinct_dims(self.MAX_RESOLUTIONS + 2)
        for width, height in dims:
            sim.render(width=width, height=height)

        keys = list(sim._renderer_tls.renderers.keys())
        assert len(keys) == self.MAX_RESOLUTIONS, "cache must stay capped at the max resolution count"
        # The two oldest (first-inserted) keys are evicted; the most recent
        # MAX_RESOLUTIONS keys remain, preserving insertion order.
        assert keys == dims[-self.MAX_RESOLUTIONS :]
        assert dims[0] not in keys
        assert dims[1] not in keys

    def test_eviction_closes_evicted_renderer(self, sim):
        sim.create_world()
        dims = self._distinct_dims(self.MAX_RESOLUTIONS + 1)

        # Create the first (soon-to-be-evicted) renderer and wrap its close()
        # so we can assert the eviction path frees its GL context.
        first_w, first_h = dims[0]
        sim.render(width=first_w, height=first_h)
        evicted = sim._renderer_tls.renderers[(first_w, first_h)]
        close_calls: list[int] = []
        original_close = evicted.close
        evicted.close = lambda *a, **k: (close_calls.append(1), original_close())[1]

        # Fill past the cap so the first renderer is evicted.
        for width, height in dims[1:]:
            sim.render(width=width, height=height)

        assert (first_w, first_h) not in sim._renderer_tls.renderers
        assert close_calls == [1], "evicted renderer must have close() called exactly once"
