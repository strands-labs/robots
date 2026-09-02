"""Tests for the robot asset manager path resolution.

Covers ``strands_robots.assets.manager`` - the pure-filesystem layer that maps
a robot name to its MJCF model XML, directory, and availability status:

    - is_robot_asset_present: side-effect-free presence check
    - resolve_model_path: XML resolution, scene preference, mesh-aware selection
    - resolve_model_dir: directory resolution
    - get_robot_info: enriched metadata with resolved path
    - list_available_robots: presence-filtered listing
    - path-traversal protection on registry-sourced path components
    - _has_meshes: mesh detection, with an optional per-scan memo

These exercise observable behavior (returned paths, booleans, None) against a
temp asset tree wired through STRANDS_ASSETS_DIR + the user registry, with no
network and no auto-download dependency.
"""

import os
from pathlib import Path

import pytest

import strands_robots.assets.manager as manager
from strands_robots.registry.user_registry import (
    _invalidate_cache,
    register_robot,
)

#: This repository, located from this file: a relative literal resolves against
#: the working directory, so the example read below raised ``FileNotFoundError``
#: whenever the suite ran from anywhere but the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


@pytest.fixture(autouse=True)
def _isolate_assets(tmp_path, monkeypatch):
    """Point base + asset dirs at a temp tree and clear caches around each test."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))
    _invalidate_cache()
    yield
    _invalidate_cache()


def _register_bot(
    assets_dir: Path,
    name: str = "unitbot",
    xml_name: str = "unitbot.xml",
    *,
    scene_xml: str | None = None,
    meshes: tuple[str, ...] = (),
) -> Path:
    """Create a minimal MJCF asset dir and register it; return the dir path."""
    robot_dir = assets_dir / name
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / xml_name).write_text(_MINIMAL_MJCF)
    if scene_xml:
        (robot_dir / scene_xml).write_text(_MINIMAL_MJCF)
    for mesh in meshes:
        (robot_dir / mesh).write_bytes(b"meshbytes")
    register_robot(
        name=name,
        model_xml=xml_name,
        description="unit test robot",
        category="arm",
        joints=6,
        scene_xml=scene_xml,
        overwrite=True,
    )
    _invalidate_cache()
    return robot_dir


class TestIsRobotAssetPresent:
    def test_true_when_xml_exists(self, tmp_path):
        _register_bot(tmp_path / "assets")
        assert manager.is_robot_asset_present("unitbot") is True

    def test_false_for_unknown_robot(self):
        assert manager.is_robot_asset_present("no_such_robot_xyz") is False

    def test_false_when_xml_missing_on_disk(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets")
        (robot_dir / "unitbot.xml").unlink()
        _invalidate_cache()
        assert manager.is_robot_asset_present("unitbot") is False


class TestResolveModelPath:
    def test_resolves_registered_xml(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets")
        resolved = manager.resolve_model_path("unitbot")
        assert resolved == robot_dir / "unitbot.xml"

    def test_none_for_unknown_robot(self):
        assert manager.resolve_model_path("no_such_robot_xyz") is None

    def test_prefer_scene_returns_scene_xml(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets", scene_xml="scene.xml")
        assert manager.resolve_model_path("unitbot", prefer_scene=True) == robot_dir / "scene.xml"

    def test_prefers_candidate_dir_with_meshes(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets", meshes=("arm.stl",))
        # resolution should succeed and point at the mesh-bearing dir
        assert manager.resolve_model_path("unitbot") == robot_dir / "unitbot.xml"


class TestResolveModelDir:
    def test_resolves_directory(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets")
        assert manager.resolve_model_dir("unitbot") == robot_dir

    def test_none_for_unknown_robot(self):
        assert manager.resolve_model_dir("no_such_robot_xyz") is None


class TestGetRobotInfo:
    def test_enriches_with_resolved_path_and_availability(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets")
        info = manager.get_robot_info("unitbot")
        assert info is not None
        assert info["canonical_name"] == "unitbot"
        assert info["available"] is True
        assert info["resolved_path"] == str(robot_dir / "unitbot.xml")

    def test_none_for_unknown_robot(self):
        assert manager.get_robot_info("no_such_robot_xyz") is None


class TestListAvailableRobots:
    def test_includes_registered_present_robot(self, tmp_path):
        _register_bot(tmp_path / "assets")
        listed = {r["name"]: r for r in manager.list_available_robots()}
        assert "unitbot" in listed
        entry = listed["unitbot"]
        assert entry["available"] is True
        assert entry["joints"] == 6
        assert entry["category"] == "arm"

    def test_marks_missing_asset_unavailable(self, tmp_path):
        robot_dir = _register_bot(tmp_path / "assets")
        (robot_dir / "unitbot.xml").unlink()
        _invalidate_cache()
        listed = {r["name"]: r for r in manager.list_available_robots()}
        assert "unitbot" in listed
        assert listed["unitbot"]["available"] is False
        assert listed["unitbot"]["path"] is None


class TestListingInstalledAssetsNeverFetches:
    """A status listing reports what is here; it does not go and get the rest.

    :func:`list_available_robots` gated the resolver on
    :func:`is_robot_asset_present`, which stops it reaching for an absent XML but
    not for a cached XML whose meshes are missing - that passes the presence check
    and was fetched, once per such robot, every time the listing was built.
    """

    def test_a_mesh_less_cached_asset_is_listed_without_being_fetched(self, tmp_path, monkeypatch):
        """The row still reports the path; nothing is downloaded to produce it."""
        robot_dir = _register_bot(tmp_path / "assets")  # XML present, no meshes
        assert manager.is_robot_asset_present("unitbot") is True, "presence alone would not gate this"

        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", lambda n, _i: attempts.append(n) or True)

        rows = {r["name"]: r for r in manager.list_available_robots()}
        assert attempts == [], f"building the listing downloaded {attempts}"
        assert rows["unitbot"]["available"] is True
        assert rows["unitbot"]["path"] == str(robot_dir / "unitbot.xml")


class TestPathTraversalProtection:
    """Registry-sourced path components must never escape the search dirs."""

    @pytest.fixture
    def _evil_robot(self, monkeypatch):
        def fake_get_robot(_name):
            return {"asset": {"dir": "../../../etc", "model_xml": "passwd", "scene_xml": "passwd"}}

        monkeypatch.setattr(manager, "get_robot", fake_get_robot)

    def test_resolve_model_dir_blocks_traversal(self, _evil_robot):
        assert manager.resolve_model_dir("evil") is None

    def test_resolve_model_path_blocks_traversal(self, _evil_robot):
        assert manager.resolve_model_path("evil") is None

    def test_is_present_blocks_traversal(self, _evil_robot):
        assert manager.is_robot_asset_present("evil") is False


class TestHasMeshes:
    def test_false_for_missing_directory(self, tmp_path):
        assert manager._has_meshes(tmp_path / "does_not_exist") is False

    def test_false_when_no_mesh_files(self, tmp_path):
        d = tmp_path / "bare"
        d.mkdir()
        (d / "model.xml").write_text(_MINIMAL_MJCF)
        assert manager._has_meshes(d) is False

    def test_true_for_nested_mesh(self, tmp_path):
        d = tmp_path / "withmesh"
        (d / "meshes").mkdir(parents=True)
        (d / "meshes" / "link.obj").write_bytes(b"o")
        assert manager._has_meshes(d) is True

    def test_a_memo_answers_a_repeat_of_the_same_directory(self, tmp_path):
        """Within one scan the tree is walked once per directory.

        This is the saving the memo exists for: ``resolve_model_path`` ranks
        several candidate XMLs that can share a directory, and a mesh-less tree
        is the case the walk cannot early-exit out of.
        """
        d = tmp_path / "cachedir"
        d.mkdir()
        memo: dict[str, bool] = {}
        assert manager._has_meshes(d, memo) is False
        assert memo == {str(d): False}
        # A mesh arriving mid-scan is not re-walked: the memo is the answer.
        (d / "late.stl").write_bytes(b"m")
        assert manager._has_meshes(d, memo) is False

    def test_a_scan_that_brings_no_memo_reads_the_tree(self, tmp_path):
        """A mesh that appeared is observed even when no timestamp moved.

        A directory's own ``st_mtime`` is held stable here, which is what a
        write into one of its subdirectories does anyway. Mesh detection answers
        from the tree, so nothing outside a caller-owned memo can go stale.
        """
        d = tmp_path / "cachedir"
        d.mkdir()
        assert manager._has_meshes(d) is False
        before = d.stat()
        (d / "late.stl").write_bytes(b"m")
        os.utime(d, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert d.stat().st_mtime_ns == before.st_mtime_ns
        assert manager._has_meshes(d) is True

    def test_an_unstattable_directory_is_still_searched(self, tmp_path, monkeypatch):
        """A ``stat()`` failure after ``exists()`` must not crash mesh discovery.

        A directory can become un-stattable between the ``exists()`` guard and
        any later read of it (a TOCTOU removal, or its permissions stripped).
        Mesh detection asks the tree and never the clock, so there is no
        timestamp read left on this path for such a failure to escape from.
        """
        d = tmp_path / "withmesh"
        (d / "meshes").mkdir(parents=True)
        (d / "meshes" / "link.obj").write_bytes(b"o")

        def _raise_stat(self, *args, **kwargs):
            raise OSError("stat unavailable")

        # exists() stays truthy (the dir is really there); only the cache-key
        # stat() read fails, exactly as a mid-call permission strip would look.
        monkeypatch.setattr(Path, "exists", lambda self, *a, **k: True)
        monkeypatch.setattr(Path, "stat", _raise_stat)

        assert manager._has_meshes(d) is True


class TestAutoDownloadFallback:
    """When no XML is found on disk, resolution attempts an auto-download."""

    def test_auto_download_supplies_missing_xml(self, tmp_path, monkeypatch):
        # Known robot, but its XML is deleted so the first search finds nothing.
        robot_dir = _register_bot(tmp_path / "assets")
        xml = robot_dir / "unitbot.xml"
        xml.unlink()
        _invalidate_cache()

        def fake_download(_name, _info):
            xml.write_text(_MINIMAL_MJCF)  # simulate a successful download
            return True

        monkeypatch.setattr(manager, "_auto_download_robot", fake_download)
        assert manager.resolve_model_path("unitbot") == xml

    def test_returns_none_when_download_fails(self, tmp_path, monkeypatch):
        robot_dir = _register_bot(tmp_path / "assets")
        (robot_dir / "unitbot.xml").unlink()
        _invalidate_cache()
        monkeypatch.setattr(manager, "_auto_download_robot", lambda _n, _i: False)
        assert manager.resolve_model_path("unitbot") is None


class TestDownloadCanBeDeclined:
    """``allow_download=False`` resolves from disk and never reaches the network.

    The downloading default is right for a caller about to load the model: a path
    whose meshes are absent is useless to MuJoCo, so fetching them is the helpful
    thing. It is wrong for a caller that *reports* on assets rather than loading
    them - there it turns a read of what is on disk into a fetch of what is not,
    once per robot.

    Declining is defined as exactly a download that fails, so it cannot change
    the answer for an asset already present. Both triggers are covered, because
    guarding on :func:`is_robot_asset_present` alone stops only the first:
    a cached XML whose meshes are missing passes that check and still fetches.
    """

    @staticmethod
    def _recording_downloader(attempts: list[str]):
        """A downloader that reports success without touching the network."""

        def download(name: str, _info: dict) -> bool:
            attempts.append(name)
            return True

        return download

    def test_absent_xml_reports_a_miss_without_attempting(self, tmp_path, monkeypatch):
        """First trigger: no XML anywhere. Declining gives the same None."""
        robot_dir = _register_bot(tmp_path / "assets")
        (robot_dir / "unitbot.xml").unlink()
        _invalidate_cache()
        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", self._recording_downloader(attempts))

        assert manager.resolve_model_path("unitbot", allow_download=False) is None
        assert attempts == []

    def test_missing_meshes_return_the_xml_without_attempting(self, tmp_path, monkeypatch):
        """Second trigger: XML present, meshes absent. Declining keeps the XML."""
        robot_dir = _register_bot(tmp_path / "assets")  # registered with no meshes
        assert manager.is_robot_asset_present("unitbot") is True, "presence alone would not gate this"
        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", self._recording_downloader(attempts))

        assert manager.resolve_model_path("unitbot", allow_download=False) == robot_dir / "unitbot.xml"
        assert attempts == []

    def test_the_downloading_default_is_unchanged(self, tmp_path, monkeypatch):
        """Over-reach control: adding the knob must not stop the default fetching."""
        _register_bot(tmp_path / "assets")  # no meshes -> the default reaches for them
        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", self._recording_downloader(attempts))

        manager.resolve_model_path("unitbot")
        assert attempts == ["unitbot"], "the default must still attempt a download"

    @pytest.mark.parametrize("delete_xml", [True, False], ids=["absent-xml", "missing-meshes"])
    def test_declining_matches_a_download_that_fails(self, tmp_path, monkeypatch, delete_xml):
        """The documented equivalence, on both triggers.

        This is what makes the knob safe to add to a resolver 46 call sites
        already use: it introduces no third outcome.
        """
        robot_dir = _register_bot(tmp_path / "assets")
        if delete_xml:
            (robot_dir / "unitbot.xml").unlink()
            _invalidate_cache()

        monkeypatch.setattr(manager, "_auto_download_robot", lambda _n, _i: False)
        failed_download = manager.resolve_model_path("unitbot")

        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", self._recording_downloader(attempts))
        declined = manager.resolve_model_path("unitbot", allow_download=False)

        assert declined == failed_download
        assert attempts == []


class TestDecliningAlsoDeclinesDiscovery:
    """A declined resolve must not reach ``robot_descriptions`` discovery.

    ``allow_download=False`` promises "no network, no ``robot_descriptions``
    import", but the resolver's first statement is a registry lookup, and that
    lookup falls back to :func:`strands_robots.registry.discovery.discover_robot`
    for any name the curated registry does not know. For those names the promise
    was broken before ``allow_download`` was ever read.

    The import *is* the fetch. ``robot_descriptions`` calls ``clone_to_cache`` at
    module scope, so importing a description module clones the upstream asset
    repository on a cold cache - the exact side effect this keyword exists to
    decline. Both docstrings say so in advance: ``_lookup`` is "used only by
    download-capable resolvers" and ``discover_robot`` is "Heavy ... Call only
    from asset-resolution paths that are allowed to download."

    It costs the same on the miss path, which is what makes it more than a
    performance note: a name discovery *cannot* resolve still clones the
    repository and then returns ``None`` - the answer the caller would have had
    for free.

    The seam these cells drive is the fallback itself rather than
    ``robot_descriptions``, so they grade the contract on a core install too. The
    last cell re-asks the question of the real registry as an independent oracle.
    """

    @staticmethod
    def _recording_discovery(consulted: list[str], entry: dict | None = None):
        """Stand in for the discovery fallback, recording every consultation.

        ``_lookup`` imports ``discover_robot`` inside its body, so patching the
        module attribute is what the call site reads.
        """

        def discover(name: str) -> dict | None:
            consulted.append(name)
            return entry

        return discover

    def _patch_discovery(self, monkeypatch, consulted: list[str], entry: dict | None = None) -> None:
        from strands_robots.registry import discovery

        discovery.invalidate_cache()
        monkeypatch.setattr(discovery, "discover_robot", self._recording_discovery(consulted, entry))

    @staticmethod
    def _synthesizable_asset(assets_dir: Path, name: str = "synthbot") -> dict:
        """Put an asset on disk that only discovery can name, and return that entry.

        This is the shape the real registry has: ``gen3``'s XML sits in the asset
        cache under a directory only ``robot_descriptions`` knows the name of, so
        a resolve that consults discovery *succeeds* while
        :func:`is_robot_asset_present` - which reads ``get_robot`` alone - reports
        it absent. Without the file on disk a declined resolve returns ``None``
        because there was nothing to find, which is not the property under test.
        """
        robot_dir = assets_dir / name
        robot_dir.mkdir(parents=True, exist_ok=True)
        (robot_dir / f"{name}.xml").write_text(_MINIMAL_MJCF)
        return {"asset": {"dir": name, "model_xml": f"{name}.xml", "scene_xml": "scene.xml"}}

    # -- the declined path never consults discovery -----------------------

    def test_a_non_curated_name_never_reaches_discovery(self, monkeypatch):
        """The headline: declining the download declines the clone that finds it."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        manager.resolve_model_path("not-in-the-curated-registry", allow_download=False)

        assert consulted == [], (
            f"a declined resolve consulted discovery, whose import clones the asset repository: {consulted}"
        )

    def test_a_non_curated_name_reports_the_miss(self, tmp_path, monkeypatch):
        """Declining answers ``None`` rather than importing to find out.

        The asset is on disk under the name only discovery supplies, so the
        declined ``None`` is the decision - against a resolve that would
        otherwise have returned a real path.
        """
        entry = self._synthesizable_asset(tmp_path / "assets")
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted, entry=entry)
        monkeypatch.setattr(manager, "_auto_download_robot", lambda _n, _i: False)

        assert manager.resolve_model_path("synthbot", allow_download=False) is None
        assert consulted == []

    def test_the_miss_path_declines_too(self, monkeypatch):
        """Discovery that would answer ``None`` is still not worth a clone."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted, entry=None)

        assert manager.resolve_model_path("no-such-robot-anywhere", allow_download=False) is None
        assert consulted == [], "the clone was paid for an answer that is None either way"

    def test_it_answers_over_the_same_names_as_the_presence_check(self, tmp_path, monkeypatch):
        """The documented parity with :func:`is_robot_asset_present`.

        The docstring offers this as the *where* to that function's *whether*, on
        the same terms. That holds only if the two read the same names: the
        presence check consults ``get_robot`` alone, so a resolver that consults
        discovery answers about robots its stated companion cannot see.
        """
        entry = self._synthesizable_asset(tmp_path / "assets")
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted, entry=entry)
        monkeypatch.setattr(manager, "_auto_download_robot", lambda _n, _i: False)
        _register_bot(tmp_path / "assets")

        for name in ("unitbot", "synthbot"):
            present = manager.is_robot_asset_present(name)
            resolved = manager.resolve_model_path(name, allow_download=False)
            assert present is (resolved is not None), (
                f"{name!r}: presence says {present} and the declined resolve says "
                f"{resolved!r} - the two do not answer over the same names"
            )

    # -- structural: the decision is forwarded, not re-derived -------------

    def test_the_resolver_forwards_the_decision_to_the_lookup(self):
        """The gate belongs to the shared lookup, and the caller must pass it on.

        Reading the keyword's *value* rather than its presence: a forward of
        ``allow_discovery=True`` spells the same keyword and restores the clone.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(manager.resolve_model_path)))
        forwards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_lookup"
        ]
        assert len(forwards) == 1, f"expected one _lookup call, found {len(forwards)}"
        passed = {kw.arg: ast.unparse(kw.value) for kw in forwards[0].keywords if kw.arg is not None}
        assert passed.get("allow_discovery") == "allow_download", (
            f"resolve_model_path must hand its own allow_download to the lookup; it passes {passed!r}"
        )

    # -- over-reach controls: nothing else changes -------------------------

    def test_the_downloading_default_still_consults_discovery(self, monkeypatch):
        """The long tail still resolves for a caller about to load the model."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        manager.resolve_model_path("not-in-the-curated-registry")

        assert consulted == ["not-in-the-curated-registry"]

    @pytest.mark.parametrize("resolver", ["resolve_model_dir", "get_robot_info"])
    def test_the_sibling_resolvers_still_consult_discovery(self, monkeypatch, resolver):
        """The lookup's gate defaults open, so its other two callers are untouched."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        getattr(manager, resolver)("not-in-the-curated-registry")

        assert consulted == ["not-in-the-curated-registry"]

    def test_a_curated_name_is_unaffected(self, tmp_path, monkeypatch):
        """Declining changes no answer for a name the curated registry knows."""
        robot_dir = _register_bot(tmp_path / "assets")
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        assert manager.resolve_model_path("unitbot", allow_download=False) == robot_dir / "unitbot.xml"
        assert consulted == [], "a curated hit must not reach the fallback at all"

    # -- premise ----------------------------------------------------------

    def test_the_lookup_records_why_discovery_needs_permission(self):
        """The reason is the helper's own, so the gate is not a new policy."""
        doc = " ".join((manager._lookup.__doc__ or "").split())
        assert "download-capable resolvers" in doc
        assert "asset download" in doc

    # -- second layer: the same question, of the real registry ------------

    def test_a_really_discoverable_name_is_declined(self):
        """Independent oracle: real registry data, no stand-in.

        A name the curated registry does not carry but ``robot_descriptions``
        can synthesize is exactly the population the stand-in models. Asserting
        no description module is imported is the reviewer-suggested check, and
        it holds against whatever the installed ``robot_descriptions`` ships.
        """
        pytest.importorskip("robot_descriptions")
        import sys

        from strands_robots.registry import get_robot
        from strands_robots.registry.discovery import list_discoverable

        candidates = [name for name in sorted(list_discoverable()) if get_robot(name) is None]
        assert candidates, "no discoverable name is outside the curated registry to test with"

        name = candidates[0]
        before = {m for m in sys.modules if m.startswith("robot_descriptions")}
        assert manager.resolve_model_path(name, allow_download=False) is None
        new = {m for m in sys.modules if m.startswith("robot_descriptions")} - before
        assert new == set(), f"declining {name!r} imported description modules: {sorted(new)}"


class TestAutoDownloadUnavailable:
    """When the download module is absent, the delegate is a no-op returning False."""

    def test_delegate_returns_false_without_impl(self, monkeypatch):
        monkeypatch.setattr(manager, "_auto_download_robot_impl", None)
        assert manager._auto_download_robot("unitbot", {}) is False

    def test_delegate_calls_impl_when_present(self, monkeypatch):
        calls = []
        monkeypatch.setattr(manager, "_auto_download_robot_impl", lambda n, i: calls.append((n, i)) or True)
        assert manager._auto_download_robot("unitbot", {"k": 1}) is True
        assert calls == [("unitbot", {"k": 1})]


class TestHasMeshesScandirError:
    """A non-walkable path (not a directory / unreadable) degrades to False."""

    def test_false_when_scandir_raises(self, tmp_path):
        # A regular file passes ``Path.exists()`` and ``stat()`` but scandir
        # raises NotADirectoryError (an OSError subclass) during the walk.
        not_a_dir = tmp_path / "model.xml"
        not_a_dir.write_text(_MINIMAL_MJCF)
        assert manager._has_meshes(not_a_dir) is False


class TestIsRobotAssetPresentEdges:
    """Standard-search-path resolution and user-path traversal fall-through."""

    def test_present_via_standard_search_path(self, tmp_path, monkeypatch):
        # No ``_user_asset_path`` in the entry: presence must be detected via
        # the standard search paths (STRANDS_ASSETS_DIR) instead.
        assets_dir = tmp_path / "assets"
        (assets_dir / "bot").mkdir(parents=True)
        (assets_dir / "bot" / "bot.xml").write_text(_MINIMAL_MJCF)
        monkeypatch.setattr(
            manager,
            "get_robot",
            lambda _name: {"asset": {"dir": "bot", "model_xml": "bot.xml"}},
        )
        assert manager.is_robot_asset_present("bot") is True

    def test_user_path_traversal_falls_through_to_false(self, tmp_path, monkeypatch):
        # A traversal-bearing model_xml in the user-path branch is rejected by
        # safe_join; resolution falls through to the (also-blocked) standard
        # search and ultimately reports the asset as absent.
        monkeypatch.setattr(
            manager,
            "get_robot",
            lambda _name: {
                "asset": {"dir": "bot", "model_xml": "../../evil.xml"},
                "_user_asset_path": str(tmp_path),
            },
        )
        assert manager.is_robot_asset_present("bot") is False


class TestResolveModelPathEdges:
    """Mesh-aware auto-download retry and user-path traversal protection."""

    def test_user_path_traversal_blocked_returns_none(self, tmp_path, monkeypatch):
        # safe_join rejects the traversal in the user-path branch (logged, then
        # user_model=None); with no standard candidate and a failed download,
        # resolution returns None rather than escaping the asset root.
        monkeypatch.setattr(
            manager,
            "get_robot",
            lambda _name: {
                "asset": {
                    "dir": "bot",
                    "model_xml": "../../evil.xml",
                    "scene_xml": "../../evil.xml",
                },
                "_user_asset_path": str(tmp_path),
            },
        )
        monkeypatch.setattr(manager, "_auto_download_robot", lambda _n, _i: False)
        assert manager.resolve_model_path("bot") is None

    def test_auto_download_supplies_missing_meshes(self, tmp_path, monkeypatch):
        # XML is present but the directory has no meshes, so the first pass
        # finds no mesh-bearing candidate. A successful auto-download drops a
        # mesh in; the post-download re-scan then resolves to the XML.
        assets_dir = tmp_path / "assets"
        robot_dir = _register_bot(assets_dir)  # no meshes by default
        xml = robot_dir / "unitbot.xml"

        def fake_download(_name, _info):
            (robot_dir / "link.stl").write_bytes(b"meshbytes")
            return True

        monkeypatch.setattr(manager, "_auto_download_robot", fake_download)
        assert manager.resolve_model_path("unitbot") == xml


class TestDownloadIntoTheDeclaredMeshdir:
    """A download that lands meshes below the model directory is not discarded.

    ``<compiler meshdir="assets"/>`` puts meshes one level down, and
    :func:`strands_robots.assets.download._mjcf_mesh_subdir` exists to place
    them there. A POSIX directory's own ``st_mtime`` does not move when a file
    appears inside one of its subdirectories, so a reading of the model
    directory taken before the download cannot describe the tree after it.
    """

    def test_the_post_download_pass_prefers_the_dir_the_meshes_landed_in(self, tmp_path, monkeypatch):
        # Two candidate locations for one robot. The mesh-less one is ranked
        # first, so a stale mesh answer is visible in the resolved path rather
        # than in a log line alone.
        _register_bot(tmp_path / "assets")  # candidate 0, never gets meshes
        proj = tmp_path / "proj"  # CWD/assets is the second search path
        downloaded = proj / "assets" / "unitbot"
        downloaded.mkdir(parents=True)
        (downloaded / "unitbot.xml").write_text(_MINIMAL_MJCF)
        (downloaded / "assets").mkdir()  # declared meshdir: pre-exists, empty
        monkeypatch.chdir(proj)
        before = downloaded.stat().st_mtime_ns

        def fake_download(_name, _info):
            (downloaded / "assets" / "pelvis.stl").write_bytes(b"meshbytes")
            return True

        monkeypatch.setattr(manager, "_auto_download_robot", fake_download)
        resolved = manager.resolve_model_path("unitbot")

        # The write was invisible to the model directory's own timestamp, which
        # is what makes this deterministic rather than a clock race.
        assert downloaded.stat().st_mtime_ns == before
        assert resolved == downloaded / "unitbot.xml"


class TestResolveModelDirEdges:
    def test_none_when_asset_dir_absent_on_disk(self, monkeypatch):
        # Known robot whose asset directory does not exist on any search path.
        monkeypatch.setattr(
            manager,
            "get_robot",
            lambda _name: {"asset": {"dir": "ghostbot", "model_xml": "ghost.xml"}},
        )
        assert manager.resolve_model_dir("ghost") is None


class TestTheDirectoryResolverCanDeclineTheSameFetch:
    """The directory resolver holds the side effect without the capability.

    :func:`~strands_robots.assets.manager.resolve_model_dir` reads the
    filesystem: it returns a directory that already exists on a search path and
    never downloads the asset. Its first statement is the shared registry
    lookup, though, which falls back to ``robot_descriptions`` discovery for any
    name the curated registry does not know - and that import *is* the fetch,
    because ``robot_descriptions`` calls ``clone_to_cache`` at module scope.

    So the resolver that cannot download was the one whose caller could not
    decline a download. ``discover_robot`` is documented "Call only from
    asset-resolution paths that are allowed to download"; this path is not one.
    Its sibling :func:`resolve_model_path` has offered ``allow_download`` since
    the fetch was first made declinable, and it hands that keyword to the lookup.

    The default stays open, so ``test_the_sibling_resolvers_still_consult_discovery``
    above - the control that pinned these callers as untouched - keeps passing.
    """

    @staticmethod
    def _recording_discovery(consulted: list[str], entry: dict | None = None):
        """Stand in for the discovery fallback, recording every consultation."""

        def discover(name: str) -> dict | None:
            consulted.append(name)
            return entry

        return discover

    def _patch_discovery(self, monkeypatch, consulted: list[str], entry: dict | None = None) -> None:
        from strands_robots.registry import discovery

        discovery.invalidate_cache()
        monkeypatch.setattr(discovery, "discover_robot", self._recording_discovery(consulted, entry))

    # -- premise: the resolver has no capability to justify the side effect --

    def test_the_directory_resolver_downloads_nothing(self, tmp_path, monkeypatch):
        """It never reaches the downloader, while its sibling does.

        This is what makes the unavoidable clone a defect rather than a cost of
        doing business: the fetch it could not decline is one it cannot use.
        """
        robot_dir = _register_bot(tmp_path / "assets")
        (robot_dir / "unitbot.xml").unlink()  # absent asset: the downloading trigger
        _invalidate_cache()
        attempts: list[str] = []
        monkeypatch.setattr(manager, "_auto_download_robot", lambda n, _i: attempts.append(n) or False)

        # The directory is still here; only the model XML is gone. The resolver
        # answers from the filesystem and takes no side effect for the absence.
        assert manager.resolve_model_dir("unitbot") == robot_dir
        assert attempts == [], "resolve_model_dir must not download; it only reads the filesystem"

        manager.resolve_model_path("unitbot")
        assert attempts == ["unitbot"], "resolve_model_path is the download-capable sibling"

    # -- regression --------------------------------------------------------

    def test_declining_skips_the_discovery_fallback(self, monkeypatch):
        """``allow_download=False`` answers from the curated registry alone."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        assert manager.resolve_model_dir("not-in-the-curated-registry", allow_download=False) is None
        assert consulted == [], f"a declined directory resolve reached discovery: {consulted!r}"

    def test_the_resolver_hands_its_own_flag_to_the_lookup(self):
        """Read the keyword's value, not its presence.

        Forwarding ``allow_discovery=True`` spells the same keyword and restores
        the clone, so a scan for the name alone would pass on the defect.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(manager.resolve_model_dir)))
        forwards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_lookup"
        ]
        assert len(forwards) == 1, f"expected one _lookup call, found {len(forwards)}"
        passed = {kw.arg: ast.unparse(kw.value) for kw in forwards[0].keywords if kw.arg is not None}
        assert passed.get("allow_discovery") == "allow_download", (
            f"resolve_model_dir must hand its own allow_download to the lookup; it passes {passed!r}"
        )

    # -- over-reach controls: nothing else changes -------------------------

    def test_the_default_still_consults_discovery(self, monkeypatch):
        """The long tail still resolves for a caller about to load a model."""
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        manager.resolve_model_dir("not-in-the-curated-registry")

        assert consulted == ["not-in-the-curated-registry"]

    def test_a_curated_name_is_unaffected(self, tmp_path, monkeypatch):
        """Declining changes no answer for a name the curated registry knows."""
        robot_dir = _register_bot(tmp_path / "assets")
        consulted: list[str] = []
        self._patch_discovery(monkeypatch, consulted)

        assert manager.resolve_model_dir("unitbot", allow_download=False) == robot_dir
        assert consulted == [], "a curated hit must not reach the fallback at all"

    # -- second layer: the same question, of the real registry ------------

    def test_a_really_discoverable_name_is_declined(self):
        """Independent oracle: real registry data, no stand-in."""
        pytest.importorskip("robot_descriptions")
        import sys

        from strands_robots.registry import get_robot
        from strands_robots.registry.discovery import list_discoverable

        candidates = [name for name in sorted(list_discoverable()) if get_robot(name) is None]
        assert candidates, "no discoverable name is outside the curated registry to test with"

        name = candidates[0]
        before = {m for m in sys.modules if m.startswith("robot_descriptions")}
        assert manager.resolve_model_dir(name, allow_download=False) is None
        new = {m for m in sys.modules if m.startswith("robot_descriptions")} - before
        assert new == set(), f"declining {name!r} imported description modules: {sorted(new)}"

    # -- the consumer -----------------------------------------------------

    def test_the_curobo_helper_reaches_the_resolver_that_can_download(self):
        """The example's docstring promises a download; only one resolver does one.

        It reported that this helper "triggers the same auto-download the MuJoCo
        path uses" while calling the resolver measured above to download nothing,
        so on a cold cache it returned no URDF and sent the reader back to the
        manual flag its own docstring says it removed.
        """
        import ast

        source = (_REPO_ROOT / "examples/so101_curobo/planner.py").read_text()
        helper = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_so101_cache_urdf"
        )
        called = {
            node.func.id for node in ast.walk(helper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "resolve_model_path" in called, (
            "the helper documents an auto-download, so it must reach the resolver that performs one"
        )
        declined = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_model_dir"
            and any(kw.arg == "allow_download" and kw.value.value is False for kw in node.keywords)
        ]
        assert declined, "the best-effort probe must decline the network rather than clone a corpus"

    def test_the_xml_parent_is_not_the_asset_directory(self, tmp_path):
        """Why the helper re-resolves instead of taking the model XML's parent.

        A robot may nest its model XML in a subdirectory, and several shipped
        ones do, so the parent of a resolved XML is not the asset directory for
        them - a shortcut that would have looked right on the robots that do not.
        """
        assets = tmp_path / "assets"
        (assets / "nestbot" / "xml").mkdir(parents=True, exist_ok=True)
        robot_dir = _register_bot(assets, name="nestbot", xml_name="xml/nestbot.xml")

        path = manager.resolve_model_path("nestbot", allow_download=False)
        directory = manager.resolve_model_dir("nestbot", allow_download=False)

        assert path == robot_dir / "xml" / "nestbot.xml"
        assert directory == robot_dir
        assert path.parent != directory, "a nested XML's parent is not the asset directory"
