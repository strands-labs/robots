"""``convert_mjcf_to_usd`` caches by content and never returns a torn entry.

The vendor importer is a Kit extension with no published Python API contract, so
what is pinned here is everything around it: which inputs are refused, what the
cache key covers, and that a failed conversion leaves nothing a later call would
trust. Three of those were measured against the real importer on
``nvcr.io/nvidia/isaac-sim:6.0.1`` and are the reason this module exists in the
shape it does:

* ``usd_path`` is an output **directory root**, not a file name - the importer
  writes ``<usd_path>/<stem>/<stem>.usda`` and returns that path. So the return
  value is what a caller must reference, and deriving the path would be a guess
  about a layout that has changed across releases.
* given no ``usd_path`` it writes **beside the source**, which raised
  ``OSError: [Errno 30] Read-only file system`` for a description under a shared
  read-only ``robot_descriptions`` checkout. Every description this package
  resolves is such a file, so naming a destination is mandatory rather than tidy.
* it writes a USD *file* and adds nothing to the live stage, so the result still
  has to be referenced in - which is why this converts and stops.

The importer itself is stood in. A fake that writes the same layout is enough to
grade the caching and the failure handling, and the real one is exercised on GPU.
"""

from __future__ import annotations

import ast
import errno
import inspect
import os
import sys
import textwrap
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402
from strands_robots.simulation.isaac.mjcf_assets import (  # noqa: E402
    MJCF_EXTENSIONS,
    USD_EXTENSIONS,
    convert_mjcf_to_usd,
    robot_usd_cache_dir,
)

_MJCF = (
    '<mujoco model="probe"><worldbody><body name="b"><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
)


class _FakeImporter:
    """Writes the layout the real importer writes, and counts its own calls."""

    calls: list[dict[str, Any]] = []
    fail_with: BaseException | None = None
    write_nothing: bool = False

    def __init__(self, config: Any) -> None:
        self._config = config

    def import_mjcf(self) -> str | None:
        type(self).calls.append(
            {
                "mjcf_path": self._config.mjcf_path,
                "usd_path": self._config.usd_path,
                "fix_base": self._config.fix_base,
                "import_scene": self._config.import_scene,
            }
        )
        # Bound to a local before the check: ``type(self).X is not None`` does not
        # narrow the following ``raise type(self).X``.
        failure = type(self).fail_with
        if failure is not None:
            raise failure
        if type(self).write_nothing:
            return None
        stem = os.path.splitext(os.path.basename(self._config.mjcf_path))[0]
        out_dir = os.path.join(self._config.usd_path, stem)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{stem}.usda")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("#usda 1.0\n")
        return out


class _FakeConfig:
    def __init__(self) -> None:
        self.mjcf_path: str | None = None
        self.usd_path: str | None = None
        self.fix_base: bool | None = None
        self.import_scene: bool = True


@pytest.fixture
def importer(monkeypatch) -> type[_FakeImporter]:
    """Install a fake ``isaacsim.asset.importer.mjcf`` for the duration."""
    _FakeImporter.calls = []
    _FakeImporter.fail_with = None
    _FakeImporter.write_nothing = False

    for name in ("isaacsim", "isaacsim.asset", "isaacsim.asset.importer", "isaacsim.asset.importer.mjcf"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = sys.modules["isaacsim.asset.importer.mjcf"]
    module.MJCFImporter = _FakeImporter  # type: ignore[attr-defined]
    module.MJCFImporterConfig = _FakeConfig  # type: ignore[attr-defined]
    # Wire the dotted attributes so ``from ... import`` resolves.
    sys.modules["isaacsim"].asset = sys.modules["isaacsim.asset"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.asset"].importer = sys.modules["isaacsim.asset.importer"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.asset.importer"].mjcf = module  # type: ignore[attr-defined]
    return _FakeImporter


@pytest.fixture
def mjcf(tmp_path) -> str:
    path = tmp_path / "src" / "scene.xml"
    path.parent.mkdir(parents=True)
    path.write_text(_MJCF)
    return str(path)


class TestTheConversionIsCachedByContent:
    def test_a_second_call_reuses_the_first_result(self, importer, mjcf, tmp_path) -> None:
        cache = str(tmp_path / "cache")
        first = convert_mjcf_to_usd(mjcf, cache)
        second = convert_mjcf_to_usd(mjcf, cache)

        assert first == second
        assert os.path.isfile(first)
        assert len(importer.calls) == 1, "the importer ran twice for one description"

    def test_a_changed_dependency_invalidates_the_cache(self, importer, mjcf, tmp_path) -> None:
        """An MJCF is not self-contained.

        Menagerie's ``scene.xml`` is a handful of lines that ``<include>`` the
        robot body and reference a ``meshdir`` of STLs, and the geometry PhysX
        simulates lives in those. A cache keyed on the named file alone would hand
        back a stale USD after any change that did not touch it, which is most of
        them.
        """
        cache = str(tmp_path / "cache")
        mesh = os.path.join(os.path.dirname(mjcf), "arm.stl")
        with open(mesh, "w", encoding="utf-8") as fh:
            fh.write("solid a\nendsolid a\n")
        first = convert_mjcf_to_usd(mjcf, cache)

        with open(mesh, "w", encoding="utf-8") as fh:
            fh.write("solid b\nendsolid b\n")
        second = convert_mjcf_to_usd(mjcf, cache)

        assert first != second, "a changed sibling asset reused the cached USD"
        assert len(importer.calls) == 2

    def test_the_posture_options_are_part_of_the_key(self, importer, mjcf, tmp_path) -> None:
        """The same description with a welded base is a different robot."""
        cache = str(tmp_path / "cache")
        free = convert_mjcf_to_usd(mjcf, cache, fix_base=None)
        welded = convert_mjcf_to_usd(mjcf, cache, fix_base=True)
        scened = convert_mjcf_to_usd(mjcf, cache, import_scene=True)

        assert len({free, welded, scened}) == 3
        assert len(importer.calls) == 3

    def test_the_digest_does_not_depend_on_walk_order(self, importer, mjcf, tmp_path) -> None:
        """``os.walk`` promises no order, so an unsorted manifest would key the
        same bytes differently on two machines and miss every cache entry."""
        for name in ("z.stl", "a.stl", "m.stl"):
            with open(os.path.join(os.path.dirname(mjcf), name), "w", encoding="utf-8") as fh:
                fh.write(name)
        cache = str(tmp_path / "cache")
        first = convert_mjcf_to_usd(mjcf, cache)
        second = convert_mjcf_to_usd(mjcf, cache)
        assert first == second
        assert len(importer.calls) == 1


class TestTheImporterIsDrivenAsMeasured:
    def test_the_destination_is_always_named(self, importer, mjcf, tmp_path) -> None:
        """Left to itself the importer writes beside the source, which is a
        read-only checkout for every description this package resolves."""
        convert_mjcf_to_usd(mjcf, str(tmp_path / "cache"))

        assert len(importer.calls) == 1
        usd_path = importer.calls[0]["usd_path"]
        assert usd_path is not None
        assert not usd_path.startswith(os.path.dirname(mjcf))

    def test_the_source_is_absolute(self, importer, tmp_path, monkeypatch) -> None:
        """A relative path would resolve against the Kit process's cwd."""
        src = tmp_path / "rel"
        src.mkdir()
        (src / "scene.xml").write_text(_MJCF)
        monkeypatch.chdir(tmp_path)

        convert_mjcf_to_usd(os.path.join("rel", "scene.xml"), str(tmp_path / "cache"))

        assert os.path.isabs(importer.calls[0]["mjcf_path"])

    def test_the_defaults_honour_the_description(self, importer, mjcf, tmp_path) -> None:
        """``fix_base=None`` is what keeps a floating-base robot floating.

        Every shipped quadruped and humanoid declares its base with
        ``<freejoint>``; ``True`` would bolt such a robot to the ground and
        report success. ``import_scene=False`` because ``add_robot`` is asking
        for a robot, and the world it joins has its own ground and lights.
        """
        convert_mjcf_to_usd(mjcf, str(tmp_path / "cache"))

        assert importer.calls[0]["fix_base"] is None
        assert importer.calls[0]["import_scene"] is False


class TestAFailedConversionLeavesNothingBehind:
    def test_a_raising_importer_writes_no_cache_entry(self, importer, mjcf, tmp_path) -> None:
        cache = str(tmp_path / "cache")
        importer.fail_with = RuntimeError("kit died")

        with pytest.raises(RuntimeError, match="kit died"):
            convert_mjcf_to_usd(mjcf, cache)

        # No staging directory and no entry: a torn entry is one a later call
        # would trust.
        assert [name for name in os.listdir(cache) if not name.startswith(".")] == []
        importer.fail_with = None
        result = convert_mjcf_to_usd(mjcf, cache)
        assert os.path.isfile(result)

    def test_an_importer_that_writes_nothing_is_refused(self, importer, mjcf, tmp_path) -> None:
        """Success plus no file is the shape this whole change exists to end."""
        importer.write_nothing = True

        with pytest.raises(RuntimeError, match="wrote no USD file"):
            convert_mjcf_to_usd(mjcf, str(tmp_path / "cache"))

    def test_no_staging_directory_survives(self, importer, mjcf, tmp_path) -> None:
        cache = str(tmp_path / "cache")
        convert_mjcf_to_usd(mjcf, cache)
        assert [name for name in os.listdir(cache) if name.startswith(".")] == []


class TestTheInputDomain:
    @pytest.mark.parametrize("ext", USD_EXTENSIONS)
    def test_a_usd_input_is_returned_unchanged(self, importer, tmp_path, ext: str) -> None:
        path = tmp_path / f"robot{ext}"
        path.write_text("#usda 1.0\n")

        assert convert_mjcf_to_usd(str(path), str(tmp_path / "cache")) == str(path)
        assert importer.calls == [], "an already-referenceable asset was converted"

    def test_a_missing_usd_input_is_refused(self, importer, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="robot asset not found"):
            convert_mjcf_to_usd(str(tmp_path / "absent.usda"), str(tmp_path / "cache"))

    def test_a_missing_mjcf_is_refused(self, importer, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="MJCF description not found"):
            convert_mjcf_to_usd(str(tmp_path / "absent.xml"), str(tmp_path / "cache"))

    def test_an_unconvertible_extension_is_refused_by_name(self, importer, tmp_path) -> None:
        path = tmp_path / "robot.urdf"
        path.write_text("<robot name='x'/>")

        with pytest.raises(ValueError) as exc:
            convert_mjcf_to_usd(str(path), str(tmp_path / "cache"))

        # The refusal names what was given and what is accepted, because a URDF
        # has a loader of its own one branch over.
        text = str(exc.value)
        assert ".urdf" in text
        for ext in MJCF_EXTENSIONS:
            assert ext in text

    def test_the_refusals_precede_the_importer(self, importer, tmp_path) -> None:
        """A refused input must not need Isaac Sim to be refused."""
        path = tmp_path / "robot.urdf"
        path.write_text("<robot name='x'/>")
        with pytest.raises(ValueError):
            convert_mjcf_to_usd(str(path), str(tmp_path / "cache"))
        assert importer.calls == []


class TestAnAbsentImporterIsReportedByName:
    def test_the_import_error_names_the_module(self, mjcf, tmp_path, monkeypatch) -> None:
        """AGENTS.md: an absent optional dependency must be recoverable from the
        exception rather than only readable in prose."""
        monkeypatch.setitem(sys.modules, "isaacsim.asset.importer.mjcf", None)

        with pytest.raises(ImportError) as exc:
            convert_mjcf_to_usd(mjcf, str(tmp_path / "cache"))

        assert exc.value.name == "isaacsim.asset.importer.mjcf"
        # It is a Kit extension, so "install the wheel" is not the whole remedy.
        assert "Kit extension" in str(exc.value)


class TestTheCacheLocation:
    def test_it_is_a_sibling_of_the_mesh_cache(self) -> None:
        from strands_robots.simulation.isaac.mesh_assets import mesh_usd_cache_dir

        assert os.path.dirname(robot_usd_cache_dir()) == os.path.dirname(mesh_usd_cache_dir())
        assert os.path.basename(robot_usd_cache_dir()) == "usd_robots"

    def test_it_honours_the_base_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
        assert str(tmp_path) in robot_usd_cache_dir()


class TestTheToleratedUnlinkIsSafeBecauseTheRenameRefuses:
    """``_remove_tree`` swallows ``OSError`` from ``os.unlink``, and that is only
    sound while the one removal that is load-bearing stays in front of the rename.

    Three of its four callers pass ``staging``, where a failed unlink leaks a temp
    path. The fourth passes ``target_root`` - the real cache entry the conversion is
    about to move into place - and tolerating *that* failure is safe only because
    ``os.replace`` immediately after it refuses to rename onto an entry that is
    still there, which ``convert_mjcf_to_usd`` already states at the top of the
    staging block. These cells pin that reason so the tolerance cannot outlive it.

    Scoped to this handler rather than to the tree: the package holds many
    tolerated swallows and their reasons are not one idiom, so a tree-wide rule
    here would demand a comment it cannot state on any of them.
    """

    def _handler_of_remove_tree(self) -> tuple[ast.ExceptHandler, list[str]]:
        source = inspect.getsource(mjcf_assets._remove_tree)
        handlers = [
            node for node in ast.walk(ast.parse(textwrap.dedent(source))) if isinstance(node, ast.ExceptHandler)
        ]
        assert len(handlers) == 1, f"_remove_tree's shape changed: {len(handlers)} handlers"
        return handlers[0], source.splitlines()

    def test_the_handler_says_why_it_tolerates_the_failure(self) -> None:
        """``py/empty-except`` is deliberately not filtered in
        ``.github/codeql/codeql-config.yml``: a swallowed exception hides a bug
        unless the swallow is argued, so this one has to carry its argument."""
        handler, lines = self._handler_of_remove_tree()

        assert len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)
        window = lines[handler.lineno - 1 : handler.body[0].lineno]
        assert any("#" in line for line in window), (
            "the tolerated unlink in _remove_tree carries no reason, so a reader "
            "cannot tell a considered tolerance from a swallowed bug"
        )

    def test_the_load_bearing_removal_is_the_statement_before_the_rename(self) -> None:
        """The tolerance rests on this adjacency, so the adjacency is pinned."""
        module = ast.parse(inspect.getsource(mjcf_assets))
        convert = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "convert_mjcf_to_usd"
        )
        removals = [
            index
            for index, statement in enumerate(convert.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and getattr(statement.value.func, "id", None) == "_remove_tree"
            and statement.value.args
            and getattr(statement.value.args[0], "id", None) == "target_root"
        ]
        assert len(removals) == 1, f"expected one target_root removal, found {len(removals)}"

        following = convert.body[removals[0] + 1]
        assert isinstance(following, ast.Expr) and isinstance(following.value, ast.Call)
        assert ast.unparse(following.value).startswith("os.replace("), ast.unparse(following)

    def test_a_surviving_file_makes_the_rename_refuse(self, tmp_path) -> None:
        """If the unlink of a file at ``target_root`` failed, the rename reports it."""
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "probe.usda").write_text("#usda 1.0\n", encoding="utf-8")
        target = tmp_path / "entry"
        target.write_text("a file where the cache entry belongs", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            os.replace(str(staging), str(target))

        assert target.read_text(encoding="utf-8") == "a file where the cache entry belongs"

    def test_a_surviving_directory_makes_the_rename_refuse(self, tmp_path) -> None:
        """And if ``rmtree(ignore_errors=True)`` left anything behind, likewise."""
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "probe.usda").write_text("#usda 1.0\n", encoding="utf-8")
        target = tmp_path / "entry"
        target.mkdir()
        (target / "leftover").write_text("survived the rmtree", encoding="utf-8")

        with pytest.raises(OSError) as exc:
            os.replace(str(staging), str(target))

        assert exc.value.errno == errno.ENOTEMPTY
        assert (target / "leftover").read_text(encoding="utf-8") == "survived the rmtree"
