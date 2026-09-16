"""A relative scene path names one place, whether it is written or read.

``export_xml`` anchors a relative ``output_path`` to the scenes directory
(``~/.strands_robots/scenes``, ``STRANDS_ROBOTS_SCENE_ROOT``) so an agent asked
to "save the scene" does not litter the process working directory. ``export_xml``
is documented as "reloadable via ``load_scene``", and the reload an agent writes
uses the name it just exported - so the reader has to look where the writer
wrote. Anchoring the writer alone left ``export_xml {"output_path":
"scene.xml"}`` followed by ``load_scene {"scene_path": "scene.xml"}`` refused
with ``Scene file not found: scene.xml``.

``scene_path`` is read AS GIVEN first, so any relative path that resolves
against the working directory keeps resolving there - the scenes directory is
consulted only when nothing is at the caller's path, which cannot change an
answer anything got before. A file in neither place is refused with both
directories named, because a refusal that quoted only the caller's spelling
said nothing about which of the two to fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _text(result) -> str:
    for block in result.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


def _bodies(sim: Simulation) -> int:
    """How many bodies the live model carries, a proxy for which scene is loaded."""
    world = sim._world
    assert world is not None and world._model is not None
    return int(world._model.nbody)


def _has_body(sim: Simulation, name: str) -> bool:
    """Whether the live model carries a body by that name."""
    world = sim._world
    assert world is not None and world._model is not None
    return bool(sim._mj.mj_name2id(world._model, sim._mj.mjtObj.mjOBJ_BODY, name) >= 0)


@pytest.fixture
def scenes(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "scenes"
    monkeypatch.setenv("STRANDS_ROBOTS_SCENE_ROOT", str(root))
    return root.resolve()


@pytest.fixture
def cwd(tmp_path, monkeypatch) -> Path:
    here = tmp_path / "checkout"
    here.mkdir()
    monkeypatch.chdir(here)
    return here


@pytest.fixture
def sim():
    s = Simulation(tool_name="scene_round_trip", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestTheRoundTrip:
    @pytest.mark.parametrize("name", ["scene.xml", "handoff/v2/scene.xml"])
    def test_a_relative_name_is_read_back_from_where_it_was_written(self, sim, scenes, cwd, name) -> None:
        assert sim.export_xml(output_path=name)["status"] == "success"
        assert (scenes / name).exists()
        assert not (cwd / name).exists()

        result = sim.load_scene(scene_path=name)

        assert result["status"] == "success", _text(result)
        assert "Scene loaded from scene.xml" in _text(result)

    def test_the_scene_that_comes_back_is_the_one_that_went_out(self, sim, scenes, cwd) -> None:
        sim.add_object(name="marker", shape="box", position=[0.4, 0.0, 0.1])
        exported = _bodies(sim)
        assert sim.export_xml(output_path="marked.xml")["status"] == "success"

        fresh = Simulation(tool_name="scene_round_trip_reader", mesh=False)
        try:
            fresh.create_world()
            assert _bodies(fresh) != exported
            assert fresh.load_scene(scene_path="marked.xml")["status"] == "success"
            assert _bodies(fresh) == exported
            assert _has_body(fresh, "marker")
        finally:
            fresh.cleanup()


class TestThePathReadAsGiven:
    def test_a_relative_path_in_the_working_directory_is_still_read_from_there(self, sim, scenes, cwd) -> None:
        assert sim.export_xml(output_path=str(cwd / "local.xml"))["status"] == "success"
        assert not scenes.exists()

        result = sim.load_scene(scene_path="local.xml")

        assert result["status"] == "success", _text(result)

    def test_the_working_directory_wins_when_both_hold_the_name(self, sim, scenes, cwd) -> None:
        sim.add_object(name="only_in_the_scenes_dir", shape="box")
        assert sim.export_xml(output_path="dup.xml")["status"] == "success"
        lean = Simulation(tool_name="scene_round_trip_lean", mesh=False)
        try:
            lean.create_world()
            assert lean.export_xml(output_path=str(cwd / "dup.xml"))["status"] == "success"
            local_bodies = _bodies(lean)
        finally:
            lean.cleanup()
        assert local_bodies != _bodies(sim)

        assert sim.load_scene(scene_path="dup.xml")["status"] == "success"

        assert _bodies(sim) == local_bodies
        assert not _has_body(sim, "only_in_the_scenes_dir")

    def test_an_absolute_source_is_read_as_given(self, sim, scenes, tmp_path) -> None:
        target = tmp_path / "elsewhere" / "scene.xml"
        assert sim.export_xml(output_path=str(target))["status"] == "success"

        assert sim.load_scene(scene_path=str(target))["status"] == "success"


class TestTheRefusal:
    def test_a_missing_relative_scene_names_both_directories(self, sim, scenes, cwd) -> None:
        result = sim.load_scene(scene_path="nowhere.xml")

        assert result["status"] == "error"
        text = _text(result)
        assert "Scene file not found: nowhere.xml" in text
        assert str(cwd) in text
        assert str(scenes) in text

    def test_a_missing_absolute_scene_names_only_the_path(self, sim, scenes, cwd, tmp_path) -> None:
        absent = tmp_path / "gone" / "scene.xml"

        result = sim.load_scene(scene_path=str(absent))

        assert result["status"] == "error"
        text = _text(result)
        assert text == f"Scene file not found: {absent}"
        assert str(scenes) not in text

    def test_a_blank_scene_path_is_quoted_without_a_search_report(self, scenes, cwd) -> None:
        from strands_robots.simulation.mujoco.physics import scene_not_found_error

        assert scene_not_found_error("   ") == "Scene file not found:    "


class TestTheDiscoverySurface:
    def test_the_tool_description_states_where_a_relative_scene_is_looked_for(self) -> None:
        import json

        import strands_robots.simulation.mujoco.simulation as simulation_module

        spec_path = Path(simulation_module.__file__).parent / "tool_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        described = spec["properties"]["scene_path"]["description"]
        assert "STRANDS_ROBOTS_SCENE_ROOT" in described
        assert "export_xml" in described
