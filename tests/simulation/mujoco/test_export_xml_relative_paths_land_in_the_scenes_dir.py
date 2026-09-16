"""A relative ``export_xml(output_path=...)`` lands in the scenes directory, not the CWD.

``export_xml`` is agent-callable, and the most natural call an agent makes when
asked to "save the scene" is ``export_xml {"output_path": "scene.xml"}``. That
resolved against the process working directory, so the file landed wherever the
process was started - in the observed replay, the user's git checkout - while
the same agent's ``render`` was confined to ``~/.strands_robots/renders``. Two
sinks, two rules for a model-supplied path.

The contract chosen keeps what scripts rely on and fixes what agents hit: an
absolute destination is written as given (the historic contract), a relative
one - bare or with directories - is anchored to ``~/.strands_robots/scenes``
(``STRANDS_ROBOTS_SCENE_ROOT``). Every guard still runs on the anchored path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import (  # noqa: E402
    anchor_relative_scene_path,
    scene_root,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _text(result) -> str:
    for block in result.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


@pytest.fixture
def scenes(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "scenes"
    monkeypatch.setenv("STRANDS_ROBOTS_SCENE_ROOT", str(root))
    return root


@pytest.fixture
def cwd(tmp_path, monkeypatch) -> Path:
    here = tmp_path / "checkout"
    here.mkdir()
    monkeypatch.chdir(here)
    return here


@pytest.fixture
def sim():
    s = Simulation(tool_name="export_scenes_dir", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestTheAnchor:
    def test_the_root_defaults_to_the_scenes_dir_beside_renders(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_ROBOTS_SCENE_ROOT", raising=False)
        assert scene_root() == (Path.home() / ".strands_robots" / "scenes").resolve()

    def test_the_env_var_moves_the_root(self, scenes) -> None:
        assert scene_root() == scenes.resolve()

    def test_a_bare_name_is_anchored(self, scenes) -> None:
        assert anchor_relative_scene_path("scene.xml") == str(scenes.resolve() / "scene.xml")

    def test_a_relative_path_with_directories_is_anchored_whole(self, scenes) -> None:
        assert anchor_relative_scene_path("handoff/v2/scene.xml") == str(
            scenes.resolve() / "handoff" / "v2" / "scene.xml"
        )

    def test_an_absolute_path_is_left_alone(self, scenes, tmp_path) -> None:
        target = str(tmp_path / "elsewhere" / "scene.xml")
        assert anchor_relative_scene_path(target) == target

    def test_a_home_relative_path_is_absolute_after_expansion(self, scenes) -> None:
        assert anchor_relative_scene_path("~/scene.xml") == "~/scene.xml"

    def test_an_empty_path_is_passed_through_for_the_guard_to_refuse(self, scenes) -> None:
        assert anchor_relative_scene_path("") == ""
        assert anchor_relative_scene_path("   ") == "   "


class TestTheSink:
    def test_a_bare_name_lands_in_the_scenes_dir_not_the_cwd(self, sim, scenes, cwd) -> None:
        result = sim.export_xml(output_path="scene.xml")

        assert result["status"] == "success", _text(result)
        written = scenes.resolve() / "scene.xml"
        assert written.exists(), "the export did not land in the scenes directory"
        assert not (cwd / "scene.xml").exists(), "the export littered the working directory"
        assert str(written) in _text(result)
        assert written.read_text(encoding="utf-8").lstrip().startswith("<mujoco")

    def test_a_relative_path_with_directories_lands_under_it_too(self, sim, scenes, cwd) -> None:
        result = sim.export_xml(output_path="handoff/scene.xml")

        assert result["status"] == "success", _text(result)
        assert (scenes.resolve() / "handoff" / "scene.xml").exists()
        assert not (cwd / "handoff").exists()

    def test_the_scenes_dir_is_created_on_first_use(self, sim, scenes, cwd) -> None:
        assert not scenes.exists()
        assert sim.export_xml(output_path="first.xml")["status"] == "success"
        assert (scenes.resolve() / "first.xml").exists()

    def test_an_absolute_destination_is_still_written_as_given(self, sim, scenes, tmp_path) -> None:
        target = tmp_path / "elsewhere" / "scene.xml"

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "success", _text(result)
        assert target.exists()
        assert not scenes.exists() or not any(scenes.rglob("scene.xml"))

    def test_traversal_cannot_climb_out_of_the_anchor(self, sim, scenes, cwd) -> None:
        result = sim.export_xml(output_path="../escape.xml")

        assert result["status"] == "error"
        assert "traversal" in _text(result)
        assert not (scenes.parent / "escape.xml").exists()
        assert not (cwd.parent / "escape.xml").exists()

    def test_a_symlink_planted_in_the_scenes_dir_is_not_followed(self, sim, scenes, tmp_path) -> None:
        scenes.mkdir(parents=True)
        victim = tmp_path / "victim.xml"
        victim.write_text("keep me", encoding="utf-8")
        os.symlink(victim, scenes / "scene.xml")

        result = sim.export_xml(output_path="scene.xml")

        assert result["status"] == "error"
        assert "symlink" in _text(result)
        assert victim.read_text(encoding="utf-8") == "keep me"

    def test_the_tool_description_states_the_rule(self) -> None:
        import json

        import strands_robots.simulation.mujoco.simulation as simulation_module

        spec_path = Path(simulation_module.__file__).parent / "tool_spec.json"
        text = json.dumps(json.loads(spec_path.read_text(encoding="utf-8")))
        assert "any absolute path accepted" not in text
        assert "STRANDS_ROBOTS_SCENE_ROOT" in text
        assert "never the process working directory" in text
