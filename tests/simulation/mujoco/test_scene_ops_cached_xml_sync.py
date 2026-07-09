"""Regression tests for the cached-XML refresh in scene mutations.

Scene mutations keep a legacy XML string in ``_backend_state["xml"]`` in sync
with the live ``MjSpec`` so the ``load_scene`` + ``add_robot`` round-trip can
read it. That refresh calls ``spec.to_xml()``, which can fail on specs MuJoCo
cannot serialise. Two of the four mutation paths (``replace_scene_mjcf`` and
``patch_scene_mjcf``) previously swallowed that failure with a bare
``except Exception: pass``, so a stale cache diverged from the live spec with no
diagnostic. All four paths now funnel through ``_sync_cached_xml`` which logs
the reason at debug and leaves the prior cache intact - never fatal, never
silent. These tests pin that contract.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    _sync_cached_xml,
    replace_scene_mjcf,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

REPLACEMENT_XML = """
<mujoco model="replacement_scene">
  <option timestep="0.002"/>
  <worldbody>
    <light name="l" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01"/>
    <body name="new_block" pos="0.5 0 0.1">
      <geom name="new_block_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation()
    try:
        yield s
    finally:
        s.cleanup()


def test_replace_scene_mjcf_refreshes_cache_on_success(sim: Simulation) -> None:
    """A successful replace refreshes the cached XML to the new scene."""
    sim.create_world()
    assert sim._world is not None

    assert replace_scene_mjcf(sim._world, REPLACEMENT_XML) is True

    cached = sim._world._backend_state.get("xml")
    assert cached is not None
    assert "replacement_scene" in cached
    assert "new_block" in cached


def test_replace_scene_mjcf_surfaces_xml_cache_refresh_failure(
    sim: Simulation, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A to_xml() failure during the cache refresh is logged, not swallowed.

    Pre-fix this path used ``except Exception: pass`` - no diagnostic - so a
    stale cache diverged silently. The mutation itself must still succeed
    (model/spec are already swapped) and the prior cache must be left intact.
    """
    sim.create_world()
    assert sim._world is not None
    sim._world._backend_state["xml"] = "<sentinel/>"

    def _boom(self: object, *args: object, **kwargs: object) -> str:
        raise RuntimeError("cannot serialise this spec")

    monkeypatch.setattr(mujoco.MjSpec, "to_xml", _boom)

    with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.mujoco.scene_ops"):
        result = replace_scene_mjcf(sim._world, REPLACEMENT_XML)

    assert result is True, "a cache-refresh failure must not fail the mutation"
    assert sim._world._backend_state["xml"] == "<sentinel/>", "prior cache must be preserved"
    assert any("cached XML left stale" in rec.getMessage() for rec in caplog.records), (
        "the to_xml() failure must be surfaced at debug, not silently swallowed"
    )


def test_sync_cached_xml_never_raises_and_logs_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    """The shared helper never raises, preserves the prior cache, and logs why."""
    world = SimpleNamespace(_backend_state={"xml": "<prev/>"})

    class _FailingSpec:
        def to_xml(self) -> str:
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.mujoco.scene_ops"):
        _sync_cached_xml(world, _FailingSpec())  # type: ignore[arg-type]

    assert world._backend_state["xml"] == "<prev/>"
    assert any("cached XML left stale" in rec.getMessage() for rec in caplog.records)


def test_sync_cached_xml_writes_serialised_xml_on_success() -> None:
    """On success the helper writes the serialised spec into the cache."""
    world = SimpleNamespace(_backend_state={})

    class _OkSpec:
        def to_xml(self) -> str:
            return "<mujoco model='ok'/>"

    _sync_cached_xml(world, _OkSpec())  # type: ignore[arg-type]
    assert world._backend_state["xml"] == "<mujoco model='ok'/>"
