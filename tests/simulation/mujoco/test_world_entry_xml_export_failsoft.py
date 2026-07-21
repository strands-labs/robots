"""The cached-XML export at world construction is best-effort, never fatal.

Both world-entry points compile a live ``MjSpec`` and then export it to
``_backend_state["xml"]`` for the legacy readers (the ``load_scene`` +
``add_robot`` round-trip) that still consume the raw MJCF string:

  * :meth:`Simulation._compile_world` - the fresh-world path used by
    ``create_world``.
  * :meth:`Simulation.load_scene` - the load-from-file path.

``spec.to_xml()`` can fail on specs MuJoCo cannot serialise. That export is a
tooling convenience, not a correctness invariant, so a failure must never
abort world construction: the compiled model stays valid and usable, the live
spec stays cached, and only the raw-XML convenience string is skipped (the
failure is logged at debug, never swallowed silently). These tests pin that
contract for both entry points.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.models import SimStatus  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

SCENE_XML = """
<mujoco model="loadable_scene">
  <option timestep="0.002"/>
  <worldbody>
    <light name="l" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01"/>
    <body name="block" pos="0.5 0 0.1">
      <geom name="block_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _raise_to_xml(self: object, *args: object, **kwargs: object) -> str:
    raise RuntimeError("synthetic: MuJoCo cannot serialise this spec")


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation()
    try:
        yield s
    finally:
        s.cleanup()


def test_create_world_survives_xml_export_failure(sim: Simulation, monkeypatch: pytest.MonkeyPatch) -> None:
    """create_world compiles a usable world even when to_xml() raises."""
    monkeypatch.setattr(mujoco.MjSpec, "to_xml", _raise_to_xml)

    result = sim.create_world()

    assert result["status"] == "success", "a to_xml() failure must not abort create_world"
    assert sim._world is not None
    assert sim._world.status is SimStatus.IDLE
    # The compiled model + live spec are the correctness invariants and must
    # survive; only the raw-XML convenience string is skipped.
    assert sim._world._model is not None
    assert sim._world._backend_state.get("spec") is not None
    assert "xml" not in sim._world._backend_state
    # The world is genuinely usable: physics steps without raising.
    step_result = sim.step(n_steps=1)
    assert step_result["status"] == "success"


def test_load_scene_survives_xml_export_failure(sim: Simulation, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_scene loads a usable scene even when to_xml() raises."""
    with tempfile.TemporaryDirectory() as tmp:
        scene_path = os.path.join(tmp, "scene.xml")
        with open(scene_path, "w", encoding="utf-8") as fh:
            fh.write(SCENE_XML)

        monkeypatch.setattr(mujoco.MjSpec, "to_xml", _raise_to_xml)

        result = sim.load_scene(scene_path)

        assert result["status"] == "success", "a to_xml() failure must not abort load_scene"
        assert sim._world is not None
        assert sim._world.status is SimStatus.IDLE
        assert sim._world._model is not None
        assert sim._world._backend_state.get("scene_loaded") is True
        assert "xml" not in sim._world._backend_state
        step_result = sim.step(n_steps=1)
        assert step_result["status"] == "success"
