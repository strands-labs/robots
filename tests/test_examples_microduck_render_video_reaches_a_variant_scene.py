"""The Microduck render example can reach the scene a skill was trained in.

A shipped Pollen weight and the scene it was trained in are one pair.
``docs/policies/microduck.md`` carries the skill-to-scene table: five of the nine
weights run on the scene the registry entry declares, and four do not - ``roller``
and ``roller_crouch`` need the four passive ankle wheels only ``scene_rollers.xml``
carries, and ``ball_kick_left`` / ``ball_kick_right`` need the prop only
``scene_ball.xml`` places.

The page documents the by-path route in prose, and until this guard landed no
shipped example could take it. ``examples/microduck/render_video.py`` built
``Robot("microduck", mesh=False)`` and had no flag for anything else, which is
what ``changelog.d/2900-microduck-skill-scenes.md`` recorded when it fixed the
page: "``examples/microduck/render_video.py`` builds exactly that, which is why
the page said any shipped weight 'drops straight in'". The page was corrected;
the example it names was not.

Measured on ``de9762a9``, comparing the scene the example could build against the
ones those four skills need::

    scene.xml          njnt=15  nu=14  nq=21  wheel joints=0  ball bodies=0
    scene_rollers.xml  njnt=19  nu=14  nq=25  wheel joints=4  ball bodies=0
    scene_ball.xml     njnt=16  nu=14  nq=28  wheel joints=0  ball bodies=1

The four wheels ``scene_rollers.xml`` adds are ``passive_LF_wheel``,
``passive_LR_wheel``, ``passive_RF_wheel`` and ``passive_RR_wheel``. ``nu`` is 14
in all three, which is why nothing refuses the mismatch: a roller policy writes
exactly the same fourteen control targets whichever scene is loaded, and the
physics simply has nothing to roll on. The rollout reports success and the
rendered video shows a duck going nowhere with no indication why.

No rollout is measured here or in this guard - ``onnxruntime`` is absent on the
machine this was written on, so the shipped weights could not be stepped. The
claim above is about scene composition, which is what the flag addresses.

Why a refusal rather than a fallback: a misspelled ``--scene`` that quietly
resolved the entry's declared scene would render exactly that same
duck-going-nowhere and report success, which is the failure this flag exists to
remove. :func:`_resolve_scene` raises instead, naming the roots it searched.

Every cell here is hermetic - it points the asset root at a temporary directory
and writes the scene files it needs - so the guard runs on an install that has
downloaded no Microduck asset, has no ``mujoco`` and has no ``onnxruntime``.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO_ROOT / "examples" / "microduck" / "render_video.py"
_DOCS_PAGE = _REPO_ROOT / "docs" / "policies" / "microduck.md"

# The scene the registry entry declares. Every other scene the page names is a
# variant a weight needs, and is what --scene exists to reach.
_DECLARED_SCENE = "scene.xml"


def _load_example() -> ModuleType:
    """Import the example by path - it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("microduck_render_video", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documented_variant_scenes() -> set[str]:
    """The variant scenes ``docs/policies/microduck.md`` names, derived from the page.

    Read from the page rather than restated here, so a tenth weight that needs a
    new scene fails this guard until the flag's help names it too.
    """
    named = set(re.findall(r"`(scene[A-Za-z0-9_]*\.xml)`", _DOCS_PAGE.read_text(encoding="utf-8")))
    return named - {_DECLARED_SCENE}


def _asset_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *scenes: str) -> Path:
    """Point the asset search at ``tmp_path`` and write the named scene files."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    microduck = tmp_path / "microduck"
    microduck.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        (microduck / scene).write_text("<mujoco/>", encoding="utf-8")
    return microduck


def _args(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


class TestAVariantSceneIsReachable:
    """The flag resolves a scene by name and hands it to ``Robot`` as the asset."""

    def test_the_flag_resolves_a_scene_under_the_asset_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        microduck = _asset_root(monkeypatch, tmp_path, "scene_rollers.xml")
        module = _load_example()

        resolved = module._resolve_scene("scene_rollers.xml")

        assert Path(resolved) == microduck / "scene_rollers.xml"

    def test_the_resolved_scene_is_forwarded_as_the_robot_asset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        microduck = _asset_root(monkeypatch, tmp_path, "scene_ball.xml")
        module = _load_example()

        kwargs = module._sim_kwargs(_args(scene="scene_ball.xml"))

        assert kwargs == {"urdf_path": str(microduck / "scene_ball.xml")}

    def test_the_first_matching_asset_root_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Precedence is the shared search order, not whichever root is checked last.

        Both asset roots carry the scene, so only the order decides which one the
        example loads - and the answer has to be the one ``get_search_paths``
        returns first.
        """
        first = _asset_root(monkeypatch, tmp_path / "user", "scene_rollers.xml")
        project = tmp_path / "project"
        second = project / "assets" / "microduck"
        second.mkdir(parents=True)
        (second / "scene_rollers.xml").write_text("<mujoco/>", encoding="utf-8")
        monkeypatch.chdir(project)
        module = _load_example()
        from strands_robots.utils import get_search_paths

        roots = get_search_paths()
        assert [first.parent, second.parent] == roots, roots

        assert Path(module._resolve_scene("scene_rollers.xml")) == first / "scene_rollers.xml"

    @pytest.mark.parametrize("scene", sorted(_documented_variant_scenes()))
    def test_every_documented_variant_scene_is_reachable_by_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scene: str
    ) -> None:
        microduck = _asset_root(monkeypatch, tmp_path, scene)
        module = _load_example()

        assert Path(module._resolve_scene(scene)) == microduck / scene


class TestAMisspelledSceneIsRefused:
    """A scene no asset root carries is refused, not quietly swapped for the default."""

    def test_a_scene_no_root_carries_is_refused(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _asset_root(monkeypatch, tmp_path, "scene_rollers.xml")
        module = _load_example()

        with pytest.raises(SystemExit):
            module._resolve_scene("scene_rollrs.xml")

    def test_the_refusal_names_the_scene_that_was_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _asset_root(monkeypatch, tmp_path)
        module = _load_example()

        with pytest.raises(SystemExit) as excinfo:
            module._resolve_scene("scene_rollrs.xml")

        assert "scene_rollrs.xml" in str(excinfo.value), str(excinfo.value)

    def test_the_refusal_names_the_directories_it_searched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A caller who has not downloaded the asset needs to see where to look."""
        microduck = _asset_root(monkeypatch, tmp_path)
        module = _load_example()

        with pytest.raises(SystemExit) as excinfo:
            module._resolve_scene("scene_rollrs.xml")

        assert str(microduck) in str(excinfo.value), str(excinfo.value)

    def test_the_refusal_reaches_the_caller_through_the_robot_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The rollout asks for the kwargs before it builds anything, so it refuses first."""
        _asset_root(monkeypatch, tmp_path)
        module = _load_example()

        with pytest.raises(SystemExit):
            module._sim_kwargs(_args(scene="scene_rollrs.xml"))


class TestTheDeclaredSceneRouteIsUnchanged:
    """The flag does not disturb the route the five default-scene weights take.

    A caller who names no scene has to keep getting the registry entry resolved by
    name. The second cell holds before and after - the example still names the
    entry. The first cannot run before the fix, because it reads the helper the fix
    introduces; it is the over-reach guard that a no-scene run forwards no asset at
    all rather than a resolved path.
    """

    def test_omitting_the_flag_forwards_no_asset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _asset_root(monkeypatch, tmp_path, "scene_rollers.xml")
        module = _load_example()

        assert module._sim_kwargs(_args(scene=None)) == {}

    def test_the_example_still_resolves_the_registry_entry_by_name(self) -> None:
        source = _EXAMPLE.read_text(encoding="utf-8")

        assert 'Robot("microduck"' in source, "the example no longer names the registry entry"


class TestTheResolutionIsSingleSourced:
    """The scene is found through the shared search paths, and the flag is declared."""

    def test_the_resolver_consults_the_shared_search_paths(self) -> None:
        """Not a hardcoded ``~/.strands_robots`` path, so ``STRANDS_ASSETS_DIR`` is honored."""
        tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
        resolver = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_resolve_scene"
        )
        names = {node.id for node in ast.walk(resolver) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(resolver) if isinstance(node, ast.Attribute)}

        assert "get_search_paths" in names, sorted(names)

    def test_the_command_line_declares_the_scene_flag(self) -> None:
        tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
        flags = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }

        assert "--scene" in flags, sorted(flags)

    def test_the_example_names_every_documented_variant_scene(self) -> None:
        """A reader has to be able to discover which scene a weight needs from here.

        Graded over the whole file rather than the ``--scene`` help alone, because
        either surface answers the question - the docstring carries a runnable
        invocation per variant scene and the help lists them. A tenth weight that
        needs a new scene fails this cell until one of the two names it.
        """
        documented = _documented_variant_scenes()
        source = _EXAMPLE.read_text(encoding="utf-8")

        missing = sorted(scene for scene in documented if scene not in source)

        assert not missing, f"the example never names {missing}, which the page says a weight needs"


class TestTheRuleIsNotVacuous:
    """The derived scene set is read from the page and is not empty."""

    def test_the_page_names_at_least_two_variant_scenes(self) -> None:
        documented = _documented_variant_scenes()

        assert len(documented) >= 2, documented
        assert _DECLARED_SCENE not in documented, documented
