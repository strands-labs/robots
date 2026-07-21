"""Tests for the robot asset manager path resolution.

Covers ``strands_robots.assets.manager`` - the pure-filesystem layer that maps
a robot name to its MJCF model XML, directory, and availability status:

    - is_robot_asset_present: side-effect-free presence check
    - resolve_model_path: XML resolution, scene preference, mesh-aware selection
    - resolve_model_dir: directory resolution
    - get_robot_info: enriched metadata with resolved path
    - list_available_robots: presence-filtered listing
    - path-traversal protection on registry-sourced path components
    - _has_meshes: mesh detection with per-(dir, mtime) caching

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

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


@pytest.fixture(autouse=True)
def _isolate_assets(tmp_path, monkeypatch):
    """Point base + asset dirs at a temp tree and clear caches around each test."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))
    _invalidate_cache()
    manager._MESH_CACHE.clear()
    yield
    _invalidate_cache()
    manager._MESH_CACHE.clear()


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

    def test_result_is_cached_per_directory(self, tmp_path):
        d = tmp_path / "cachedir"
        d.mkdir()
        assert manager._has_meshes(d) is False
        key = (str(d), d.stat().st_mtime)
        assert manager._MESH_CACHE.get(key) is False
        # Adding a mesh without busting the cache still returns the cached value
        (d / "late.stl").write_bytes(b"m")
        os.utime(d, (d.stat().st_atime, key[1]))  # keep mtime stable
        assert manager._has_meshes(d) is False

    def test_stat_oserror_falls_back_to_stable_cache_key(self, tmp_path, monkeypatch):
        """A ``stat()`` failure after ``exists()`` must not crash mesh discovery.

        There is a window between the ``directory.exists()`` guard and the
        ``directory.stat().st_mtime`` cache-key read where the directory can
        become un-stattable (a TOCTOU removal, or its permissions stripped).
        Mesh detection degrades gracefully: it falls back to a stable
        ``(path, 0.0)`` cache key and still walks the tree, rather than letting
        the ``OSError`` escape and abort robot-asset resolution. Removing the
        fallback would let the ``stat()`` raise propagate and break this.
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
        # The fallback key (mtime pinned to 0.0) was used, so the walk result
        # is memoised there and served from cache on the next call.
        assert manager._MESH_CACHE.get((str(d), 0.0)) is True


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
            # A real download writes new files; invalidate the stale mtime-keyed
            # mesh cache so the post-download re-scan observes them.
            manager._MESH_CACHE.clear()
            return True

        monkeypatch.setattr(manager, "_auto_download_robot", fake_download)
        assert manager.resolve_model_path("unitbot") == xml


class TestResolveModelDirEdges:
    def test_none_when_asset_dir_absent_on_disk(self, monkeypatch):
        # Known robot whose asset directory does not exist on any search path.
        monkeypatch.setattr(
            manager,
            "get_robot",
            lambda _name: {"asset": {"dir": "ghostbot", "model_xml": "ghost.xml"}},
        )
        assert manager.resolve_model_dir("ghost") is None
