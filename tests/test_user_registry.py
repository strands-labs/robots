"""Tests for user-local robot registry and shared path utilities.

Covers:
    - strands_robots.registry.user_registry (register, unregister, list, persistence)
    - strands_robots.registry.loader._merge_user_robots (user overlay merge)
    - strands_robots.utils (get_base_dir, get_assets_dir, resolve_asset_path)
"""

import json
import logging
import os
from pathlib import Path
from unittest import mock

import pytest

from strands_robots.registry import get_robot, list_robots, resolve_name
from strands_robots.registry.user_registry import (
    _get_user_registry_path,
    _invalidate_cache,
    _load_user_registry,
    get_user_robots,
    list_user_robots,
    register_robot,
    unregister_robot,
)
from strands_robots.utils import get_assets_dir, get_base_dir, resolve_asset_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Point STRANDS_BASE_DIR + STRANDS_ASSETS_DIR to temp dirs for every test.

    ``STRANDS_BASE_DIR`` controls where ``user_robots.json`` lives.
    ``STRANDS_ASSETS_DIR`` controls where robot asset directories live.
    The two are independent - the base dir is not derived from the assets dir.
    """
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))
    _invalidate_cache()
    yield
    _invalidate_cache()


def _make_robot(parent: Path, name: str = "test_bot", xml_name: str = "bot.xml") -> Path:
    """Create a minimal MJCF robot directory and return its path."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / xml_name).write_text(_MINIMAL_MJCF)
    return d


# ===========================================================================
# Registration
# ===========================================================================


class TestRegisterRobot:
    """register_robot() stores metadata and makes the robot discoverable."""

    def test_stores_description_category_and_joints(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        entry = register_robot(
            name="test_bot",
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            description="A test bot",
            category="arm",
            joints=3,
        )
        assert entry["description"] == "A test bot"
        assert entry["category"] == "arm"
        assert entry["joints"] == 3
        assert entry["asset"]["model_xml"] == "bot.xml"

    def test_robot_visible_via_get_robot(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))
        assert get_robot("test_bot") is not None

    def test_robot_visible_in_list_robots(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))
        assert "test_bot" in [r["name"] for r in list_robots()]

    def test_aliases_resolve_to_canonical_name(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(
            name="test_bot",
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            aliases=["my_bot", "tb"],
        )
        assert resolve_name("my_bot") == "test_bot"
        assert resolve_name("tb") == "test_bot"

    def test_stores_robot_descriptions_module(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        entry = register_robot(
            name="test_bot",
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            robot_descriptions_module="my_pkg.test_bot",
        )
        assert entry["asset"]["robot_descriptions_module"] == "my_pkg.test_bot"

    def test_stores_hardware_config(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        hw = {"lerobot_type": "so100_follower", "cameras": {"top": 0}}
        entry = register_robot(
            name="test_bot",
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            hardware=hw,
        )
        assert entry["hardware"] == hw


class TestRegisterRobotNameNormalization:
    """Names are lower-cased, stripped, and hyphens become underscores."""

    def test_normalizes_whitespace_hyphens_and_case(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="  My-Bot  ", model_xml="bot.xml", asset_dir=str(robot_dir))
        assert get_robot("my_bot") is not None


class TestRegisterRobotDuplicates:
    """Duplicate handling: raise by default, allow with overwrite=True."""

    def test_duplicate_raises_by_default(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))
        with pytest.raises(ValueError, match="already in user registry"):
            register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))

    def test_overwrite_replaces_existing(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir), description="v1")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir), description="v2", overwrite=True)
        assert get_robot("test_bot")["description"] == "v2"

    def test_overriding_package_robot_logs_info(self, tmp_path, caplog):
        """Registering a name that exists in the package registry emits an info log."""
        panda_dir = _make_robot(tmp_path / "assets", name="panda", xml_name="panda.xml")
        with caplog.at_level(logging.INFO, logger="strands_robots.registry.user_registry"):
            register_robot(
                name="panda",
                model_xml="panda.xml",
                asset_dir=str(panda_dir),
                description="Custom panda",
            )
        assert any("exists in package registry" in m for m in caplog.messages)
        assert get_robot("panda")["description"] == "Custom panda"
        unregister_robot("panda")


class TestRegisterRobotValidation:
    """register_robot rejects invalid inputs."""

    def test_missing_model_xml_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "assets" / "empty"
        empty_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Model XML not found"):
            register_robot(name="empty", model_xml="nope.xml", asset_dir=str(empty_dir))

    def test_missing_asset_dir_raises_file_not_found(self, tmp_path):
        """A non-existent asset_dir fails closed at registration with an actionable message.

        Registration must not silently accept a dir that does not exist yet and
        defer the failure to add_robot() time.
        """
        ghost_dir = tmp_path / "assets" / "ghost"  # never created
        with pytest.raises(FileNotFoundError, match="Asset directory does not exist") as exc:
            register_robot(name="ghost", model_xml="g.xml", asset_dir=str(ghost_dir))
        # Message names the missing dir and the xml the user should place inside it.
        assert str(ghost_dir) in str(exc.value)
        assert "g.xml" in str(exc.value)
        # Nothing should have been persisted.
        assert get_user_robots().get("ghost") is None


class TestRegisterRobotAliasCollision:
    """register_robot fails closed when an alias would collide at read time.

    The loader RAISES on an alias that shadows a canonical name or another
    robot's alias on every subsequent read, so register_robot must reject such
    an alias at write time rather than persist it. Otherwise a "successful"
    registration bricks every get_robot/resolve_name call process-wide until
    user_robots.json is hand-edited.
    """

    def test_alias_shadowing_package_canonical_name_raises(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets", name="botA", xml_name="a.xml")
        with pytest.raises(ValueError, match="so100"):
            register_robot(name="botA", model_xml="a.xml", asset_dir=str(robot_dir), aliases=["so100"])
        # Nothing persisted, and the registry still loads.
        assert get_user_robots().get("bota") is None
        assert get_robot("so100") is not None

    def test_alias_already_used_by_another_user_robot_raises(self, tmp_path):
        first_dir = _make_robot(tmp_path / "assets", name="botB", xml_name="b.xml")
        second_dir = _make_robot(tmp_path / "assets", name="botC", xml_name="c.xml")
        register_robot(name="botB", model_xml="b.xml", asset_dir=str(first_dir), aliases=["grabber"])
        with pytest.raises(ValueError, match="grabber"):
            register_robot(name="botC", model_xml="c.xml", asset_dir=str(second_dir), aliases=["grabber"])
        # The second robot is not persisted; the first is still resolvable.
        assert get_user_robots().get("botc") is None
        assert get_robot("grabber") is not None

    def test_unique_alias_registers_successfully(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets", name="botD", xml_name="d.xml")
        register_robot(name="botD", model_xml="d.xml", asset_dir=str(robot_dir), aliases=["unique_grip_xyz"])
        assert get_user_robots().get("botd") is not None
        assert get_robot("unique_grip_xyz") is not None


class TestRegisterRobotAssetDirResolution:
    """asset_dir is resolved relative to STRANDS_ASSETS_DIR."""

    def test_none_defaults_to_assets_subdir(self, tmp_path):
        default_dir = _make_robot(tmp_path / "assets", name="auto_bot", xml_name="auto.xml")
        entry = register_robot(name="auto_bot", model_xml="auto.xml")
        assert entry["_user_asset_path"] == str(default_dir)

    def test_relative_path_resolved_against_assets(self, tmp_path):
        rel_dir = tmp_path / "assets" / "sub" / "bot"
        rel_dir.mkdir(parents=True)
        (rel_dir / "r.xml").write_text(_MINIMAL_MJCF)
        entry = register_robot(name="rel_bot", model_xml="r.xml", asset_dir="sub/bot")
        assert entry["_user_asset_path"] == str(rel_dir)

    def test_absolute_path_used_as_is(self, tmp_path):
        abs_dir = tmp_path / "elsewhere" / "bot"
        abs_dir.mkdir(parents=True)
        (abs_dir / "b.xml").write_text(_MINIMAL_MJCF)
        entry = register_robot(name="abs_bot", model_xml="b.xml", asset_dir=str(abs_dir))
        assert entry["_user_asset_path"] == str(abs_dir)


# ===========================================================================
# Unregistration
# ===========================================================================


class TestUnregisterRobot:
    """unregister_robot() removes from user registry only."""

    def test_removes_registered_robot(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))
        assert unregister_robot("test_bot") is True
        assert get_user_robots().get("test_bot") is None

    def test_returns_false_for_nonexistent(self):
        assert unregister_robot("nonexistent") is False

    def test_does_not_affect_package_robots(self):
        assert get_robot("panda") is not None
        assert unregister_robot("panda") is False
        assert get_robot("panda") is not None


# ===========================================================================
# Listing
# ===========================================================================


class TestListUserRobots:
    """list_user_robots() returns user-registered robots only."""

    def test_empty_when_nothing_registered(self):
        assert list_user_robots() == []

    def test_returns_registered_robot_metadata(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(
            name="test_bot",
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            description="Desc",
            joints=5,
        )
        result = list_user_robots()
        assert len(result) == 1
        assert result[0]["name"] == "test_bot"
        assert result[0]["description"] == "Desc"
        assert result[0]["joints"] == 5
        assert result[0]["model_xml"] == "bot.xml"


# ===========================================================================
# Persistence
# ===========================================================================


class TestPersistence:
    """User registry persists to a JSON file and survives corruption."""

    def test_writes_json_file(self, tmp_path):
        robot_dir = _make_robot(tmp_path / "assets")
        register_robot(name="test_bot", model_xml="bot.xml", asset_dir=str(robot_dir))
        path = _get_user_registry_path()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "test_bot" in data["robots"]

    def test_corrupted_json_returns_empty(self):
        path = _get_user_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT JSON!!!")
        assert _load_user_registry() == {"robots": {}}

    def test_valid_json_without_robots_key_returns_empty(self):
        path = _get_user_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version": 1}')
        assert _load_user_registry() == {"robots": {}}


# ===========================================================================
# Loader merge
# ===========================================================================


class TestLoaderMerge:
    """_merge_user_robots gracefully handles missing user_registry module."""

    def test_import_error_returns_data_unchanged(self):
        from strands_robots.registry.loader import _merge_user_robots

        data = {"robots": {"fake": {"description": "test"}}}
        with mock.patch.dict("sys.modules", {"strands_robots.registry.user_registry": None}):
            result = _merge_user_robots(data, None)
        assert "fake" in result["robots"]


# ===========================================================================
# STRANDS_BASE_DIR integration
# ===========================================================================


class TestStrandsBaseDirIntegration:
    """Registry file location respects STRANDS_BASE_DIR env var.

    STRANDS_ASSETS_DIR intentionally does NOT move the registry - it only
    controls where asset directories live. See utils.get_base_dir() docstring.
    """

    def test_registry_file_lives_in_base_dir(self, tmp_path):
        custom = tmp_path / "custom_base"
        custom.mkdir()
        with mock.patch.dict(os.environ, {"STRANDS_BASE_DIR": str(custom)}, clear=False):
            assert _get_user_registry_path().parent == custom

    def test_assets_dir_does_not_move_registry(self, tmp_path, monkeypatch):
        """Setting only STRANDS_ASSETS_DIR must not change the registry location."""
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        custom_assets = tmp_path / "custom_assets"
        custom_assets.mkdir()
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(custom_assets))
        # Registry should land under the default base, not the assets dir.
        assert ".strands_robots" in str(_get_user_registry_path())

    def test_defaults_to_dot_strands_robots(self, monkeypatch):
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        assert ".strands_robots" in str(_get_user_registry_path())


# ===========================================================================
# Path utilities (strands_robots.utils)
# ===========================================================================


class TestGetAssetsDir:
    """get_assets_dir() returns STRANDS_ASSETS_DIR or ~/.strands_robots/assets/."""

    def test_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        result = get_assets_dir()
        assert str(result).endswith("assets")
        assert ".strands_robots" in str(result)

    def test_custom(self, tmp_path, monkeypatch):
        custom = tmp_path / "my_assets"
        custom.mkdir()
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(custom))
        assert get_assets_dir() == custom


class TestGetBaseDir:
    """get_base_dir() returns STRANDS_BASE_DIR or ~/.strands_robots/.

    It is independent of STRANDS_ASSETS_DIR by design - the base dir holds
    user metadata (user_robots.json) and should not move just because the
    user repoints the asset cache.
    """

    def test_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        assert str(get_base_dir()).endswith(".strands_robots")

    def test_custom(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_base"
        custom.mkdir()
        monkeypatch.setenv("STRANDS_BASE_DIR", str(custom))
        assert get_base_dir() == custom

    def test_assets_dir_does_not_move_base(self, tmp_path, monkeypatch):
        """STRANDS_ASSETS_DIR must not affect get_base_dir()."""
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path / "assets"))
        assert str(get_base_dir()).endswith(".strands_robots")


class TestResolveAssetPath:
    """resolve_asset_path() resolves None, relative, absolute, and ~ paths."""

    def test_none_returns_assets_dir_plus_default_name(self, tmp_path, monkeypatch):
        assets = tmp_path / "a"
        assets.mkdir()
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets))
        assert resolve_asset_path(None, "robot") == assets / "robot"

    def test_relative_resolved_against_assets_dir(self, tmp_path, monkeypatch):
        assets = tmp_path / "a"
        assets.mkdir()
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets))
        assert resolve_asset_path("sub/dir") == assets / "sub" / "dir"

    def test_absolute_path_unchanged(self):
        assert resolve_asset_path("/absolute/path") == Path("/absolute/path")

    def test_tilde_expanded(self):
        result = resolve_asset_path("~/robots")
        assert str(result).startswith(str(Path.home()))
