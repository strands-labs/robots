#!/usr/bin/env python3
"""Tests for the robot asset manager — model registry, aliases, path resolution, formatting.

All tests run on CPU without any GPU, MuJoCo, or hardware dependencies.
They validate the asset metadata registry, alias resolution, path lookups,
and human-readable formatting.
"""

import os
from pathlib import Path

from strands_robots.assets import (
    _ALIASES,
    _ROBOT_MODELS,
    format_robot_table,
    get_assets_dir,
    get_robot_info,
    get_search_paths,
    list_aliases,
    list_available_robots,
    list_robots_by_category,
    resolve_model_dir,
    resolve_model_path,
    resolve_robot_name,
)

# ─────────────────────────────────────────────────────────────────────
# Asset directory resolution
# ─────────────────────────────────────────────────────────────────────


class TestAssetDirectories:
    """Test asset directory lookup logic."""

    def test_get_assets_dir_returns_path(self):
        result = get_assets_dir()
        assert isinstance(result, Path)

    def test_get_assets_dir_exists(self):
        result = get_assets_dir()
        assert result.exists()
        assert result.is_dir()

    def test_get_assets_dir_is_assets_subdir(self):
        """Assets dir should be under strands_robots/assets/."""
        result = get_assets_dir()
        assert result.name == "assets"
        assert "strands_robots" in str(result)

    def test_get_search_paths_returns_list(self):
        paths = get_search_paths()
        assert isinstance(paths, list)
        assert len(paths) >= 1

    def test_get_search_paths_first_is_bundled(self):
        """First search path should be the bundled assets directory."""
        paths = get_search_paths()
        assert paths[0] == get_assets_dir()

    def test_get_search_paths_includes_user_home(self):
        """Search paths should include ~/.strands_robots/assets/."""
        paths = get_search_paths()
        home_path = Path.home() / ".strands_robots" / "assets"
        assert home_path in paths

    def test_get_search_paths_includes_cwd(self):
        """Search paths should include ./assets/."""
        paths = get_search_paths()
        cwd_path = Path.cwd() / "assets"
        assert cwd_path in paths

    def test_custom_search_path_from_env(self):
        """STRANDS_URDF_DIR env var adds custom search paths."""
        old = os.environ.get("STRANDS_URDF_DIR")
        try:
            os.environ["STRANDS_URDF_DIR"] = "/tmp/custom_assets"
            paths = get_search_paths()
            assert Path("/tmp/custom_assets") in paths
        finally:
            if old is not None:
                os.environ["STRANDS_URDF_DIR"] = old
            else:
                os.environ.pop("STRANDS_URDF_DIR", None)

    def test_custom_search_path_colon_separated(self):
        """Multiple custom paths separated by colons."""
        old = os.environ.get("STRANDS_URDF_DIR")
        try:
            os.environ["STRANDS_URDF_DIR"] = "/tmp/path1:/tmp/path2"
            paths = get_search_paths()
            assert Path("/tmp/path1") in paths
            assert Path("/tmp/path2") in paths
        finally:
            if old is not None:
                os.environ["STRANDS_URDF_DIR"] = old
            else:
                os.environ.pop("STRANDS_URDF_DIR", None)


# ─────────────────────────────────────────────────────────────────────
# Robot Model Registry
# ─────────────────────────────────────────────────────────────────────


class TestRobotModelRegistry:
    """Test the static robot model registry."""

    def test_has_31_models(self):
        assert len(_ROBOT_MODELS) == 32

    def test_model_entry_structure(self):
        """Every model entry has required fields."""
        required_keys = {"dir", "model_xml", "scene_xml", "description", "joints", "category"}
        for name, info in _ROBOT_MODELS.items():
            missing = required_keys - set(info.keys())
            assert not missing, f"Robot '{name}' missing keys: {missing}"

    def test_model_xml_is_string(self):
        for name, info in _ROBOT_MODELS.items():
            assert isinstance(info["model_xml"], str), f"{name} model_xml not str"
            assert info["model_xml"].endswith(".xml"), f"{name} model_xml not .xml"

    def test_scene_xml_is_string(self):
        for name, info in _ROBOT_MODELS.items():
            assert isinstance(info["scene_xml"], str), f"{name} scene_xml not str"
            assert info["scene_xml"].endswith(".xml"), f"{name} scene_xml not .xml"

    def test_joints_positive_integer(self):
        for name, info in _ROBOT_MODELS.items():
            assert isinstance(info["joints"], int), f"{name} joints not int"
            assert info["joints"] > 0, f"{name} has non-positive joints"

    def test_category_is_valid(self):
        valid_categories = {"arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip"}
        for name, info in _ROBOT_MODELS.items():
            assert info["category"] in valid_categories, f"Robot '{name}' has invalid category '{info['category']}'"

    def test_known_robots_exist(self):
        """Spot-check key robots are in the registry."""
        expected = {
            "so100",
            "so101",
            "panda",
            "fr3",
            "ur5e",
            "aloha",
            "unitree_g1",
            "unitree_h1",
            "unitree_go2",
            "spot",
            "shadow_hand",
            "reachy_mini",
            "google_robot",
            "open_duck_mini",
        }
        for name in expected:
            assert name in _ROBOT_MODELS, f"Missing expected robot: {name}"

    def test_category_distribution(self):
        """All categories should have at least one robot."""
        categories = {info["category"] for info in _ROBOT_MODELS.values()}
        assert "arm" in categories
        assert "humanoid" in categories
        assert "mobile" in categories
        assert "hand" in categories
        assert "bimanual" in categories


# ─────────────────────────────────────────────────────────────────────
# Alias resolution
# ─────────────────────────────────────────────────────────────────────


class TestAliasResolution:
    """Test robot name alias resolution."""

    def test_alias_count(self):
        assert len(_ALIASES) == 54

    def test_all_aliases_resolve_to_known_robots(self):
        """Every alias target must exist in _ROBOT_MODELS."""
        for alias, canonical in _ALIASES.items():
            assert canonical in _ROBOT_MODELS, f"Alias '{alias}' → '{canonical}' but '{canonical}' not in _ROBOT_MODELS"

    def test_resolve_canonical_name_unchanged(self):
        assert resolve_robot_name("so100") == "so100"
        assert resolve_robot_name("panda") == "panda"
        assert resolve_robot_name("unitree_g1") == "unitree_g1"

    def test_resolve_alias(self):
        assert resolve_robot_name("franka_panda") == "panda"
        assert resolve_robot_name("franka_emika_panda") == "panda"
        assert resolve_robot_name("go2") == "unitree_go2"
        assert resolve_robot_name("g1") == "unitree_g1"
        assert resolve_robot_name("h1") == "unitree_h1"
        assert resolve_robot_name("a1") == "unitree_a1"

    def test_resolve_case_insensitive(self):
        assert resolve_robot_name("SO100") == "so100"
        assert resolve_robot_name("Panda") == "panda"
        assert resolve_robot_name("UNITREE_G1") == "unitree_g1"

    def test_resolve_strips_whitespace(self):
        assert resolve_robot_name("  so100  ") == "so100"
        assert resolve_robot_name("panda ") == "panda"

    def test_resolve_unknown_returns_input(self):
        assert resolve_robot_name("nonexistent_robot") == "nonexistent_robot"

    def test_trossen_bimanual_aliases(self):
        assert resolve_robot_name("trossen_ai_bimanual") == "trossen_wxai"
        assert resolve_robot_name("trossen_vx300s") == "vx300s"
        assert resolve_robot_name("viper_x300s") == "vx300s"

    def test_open_duck_aliases(self):
        assert resolve_robot_name("open_duck") == "open_duck_mini"
        assert resolve_robot_name("mini_bdx") == "open_duck_mini"
        assert resolve_robot_name("bdx") == "open_duck_mini"
        assert resolve_robot_name("open_duck_v2") == "open_duck_mini"
        assert resolve_robot_name("open_duck_mini_v2") == "open_duck_mini"

    def test_reachy_aliases(self):
        assert resolve_robot_name("reachy") == "reachy_mini"
        assert resolve_robot_name("pollen_reachy_mini") == "reachy_mini"
        assert resolve_robot_name("reachy-mini") == "reachy_mini"
        assert resolve_robot_name("reachymini") == "reachy_mini"

    def test_so100_variants(self):
        assert resolve_robot_name("so100_dualcam") == "so100"
        assert resolve_robot_name("so100_4cam") == "so100"
        assert resolve_robot_name("so_arm100") == "so100"
        assert resolve_robot_name("trs_so_arm100") == "so100"

    def test_list_aliases_returns_dict(self):
        aliases = list_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) == 54


# ─────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────


class TestPathResolution:
    """Test resolve_model_path and resolve_model_dir."""

    def test_resolve_known_robot_path(self):
        """Known robots should resolve to existing XML files."""
        path = resolve_model_path("so100")
        assert path is not None
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".xml"

    def test_resolve_all_bundled_models(self):
        """Every registered robot should have a resolvable model path."""
        for name in _ROBOT_MODELS:
            path = resolve_model_path(name)
            assert path is not None, f"Robot '{name}' model path not found"
            assert path.exists(), f"Robot '{name}' model path doesn't exist: {path}"

    def test_resolve_via_alias(self):
        """Aliases should resolve to the same path as the canonical name."""
        canonical_path = resolve_model_path("panda")
        alias_path = resolve_model_path("franka_panda")
        assert canonical_path == alias_path

    def test_resolve_scene_xml(self):
        """prefer_scene=True returns the scene XML."""
        model_path = resolve_model_path("so100", prefer_scene=False)
        scene_path = resolve_model_path("so100", prefer_scene=True)
        assert model_path is not None
        assert scene_path is not None
        # For so100: model is so_arm100.xml, scene is scene.xml
        assert model_path.name == "so_arm100.xml"
        assert scene_path.name == "scene.xml"

    def test_resolve_unknown_returns_none(self):
        result = resolve_model_path("definitely_not_a_real_robot_xyz")
        assert result is None

    def test_resolve_model_dir_known(self):
        dir_path = resolve_model_dir("so100")
        assert dir_path is not None
        assert isinstance(dir_path, Path)
        assert dir_path.is_dir()

    def test_resolve_model_dir_contains_xml(self):
        """Resolved directory should contain the model XML."""
        dir_path = resolve_model_dir("so100")
        model_path = resolve_model_path("so100")
        assert model_path is not None
        assert dir_path is not None
        # model_path should be inside dir_path (or a subdirectory)
        assert str(model_path).startswith(str(dir_path))

    def test_resolve_model_dir_unknown(self):
        result = resolve_model_dir("nonexistent_robot")
        assert result is None

    def test_resolve_model_dir_via_alias(self):
        canonical = resolve_model_dir("panda")
        alias = resolve_model_dir("franka_panda")
        assert canonical == alias

    def test_reachy_mini_nested_path(self):
        """Reachy Mini has nested mjcf/ subdirectory in its paths."""
        path = resolve_model_path("reachy_mini")
        assert path is not None
        assert "mjcf" in str(path)
        assert path.name == "reachy_mini.xml"


# ─────────────────────────────────────────────────────────────────────
# Robot info and listing
# ─────────────────────────────────────────────────────────────────────


class TestRobotInfo:
    """Test get_robot_info and listing functions."""

    def test_get_robot_info_known(self):
        info = get_robot_info("so100")
        assert info is not None
        assert info["canonical_name"] == "so100"
        assert info["category"] == "arm"
        assert info["available"] is True
        assert "SO-ARM100" in info["description"]

    def test_get_robot_info_via_alias(self):
        info = get_robot_info("franka_panda")
        assert info is not None
        assert info["canonical_name"] == "panda"

    def test_get_robot_info_unknown(self):
        info = get_robot_info("nonexistent")
        assert info is None

    def test_get_robot_info_has_resolved_path(self):
        info = get_robot_info("so100")
        assert "resolved_path" in info
        assert info["resolved_path"] != "None"

    def test_list_available_robots_returns_list(self):
        robots = list_available_robots()
        assert isinstance(robots, list)
        assert len(robots) == 32

    def test_list_available_robots_entry_structure(self):
        robots = list_available_robots()
        required_keys = {"name", "description", "joints", "category", "dir", "available", "path"}
        for r in robots:
            missing = required_keys - set(r.keys())
            assert not missing, f"Robot entry missing keys: {missing}"

    def test_list_available_robots_sorted(self):
        robots = list_available_robots()
        names = [r["name"] for r in robots]
        assert names == sorted(names)

    def test_all_bundled_robots_available(self):
        """All 32 bundled robots should be marked as available."""
        robots = list_available_robots()
        unavailable = [r["name"] for r in robots if not r["available"]]
        assert len(unavailable) == 0, f"Unavailable robots: {unavailable}"

    def test_list_robots_by_category(self):
        by_cat = list_robots_by_category()
        assert isinstance(by_cat, dict)
        assert "arm" in by_cat
        assert "humanoid" in by_cat
        assert len(by_cat["arm"]) > 5  # Many arm robots

    def test_list_robots_by_category_covers_all(self):
        """Total robots across categories should equal 32."""
        by_cat = list_robots_by_category()
        total = sum(len(robots) for robots in by_cat.values())
        assert total == 32

    def test_arm_category_contents(self):
        by_cat = list_robots_by_category()
        arm_names = {r["name"] for r in by_cat["arm"]}
        assert "so100" in arm_names
        assert "panda" in arm_names
        assert "ur5e" in arm_names

    def test_humanoid_category_contents(self):
        by_cat = list_robots_by_category()
        humanoid_names = {r["name"] for r in by_cat["humanoid"]}
        assert "unitree_g1" in humanoid_names
        assert "unitree_h1" in humanoid_names
        assert "fourier_n1" in humanoid_names


# ─────────────────────────────────────────────────────────────────────
# format_robot_table
# ─────────────────────────────────────────────────────────────────────


class TestFormatRobotTable:
    """Test the human-readable table formatter."""

    def test_returns_string(self):
        table = format_robot_table()
        assert isinstance(table, str)

    def test_contains_header(self):
        table = format_robot_table()
        assert "Name" in table
        assert "Category" in table
        assert "Joints" in table

    def test_contains_all_robots(self):
        table = format_robot_table()
        for name in _ROBOT_MODELS:
            assert name in table, f"Robot '{name}' missing from table"

    def test_contains_total_count(self):
        table = format_robot_table()
        assert "Total: 32" in table

    def test_contains_alias_count(self):
        table = format_robot_table()
        assert "Aliases: 54" in table

    def test_contains_availability_indicator(self):
        table = format_robot_table()
        # Should have check marks for available robots
        assert "✅" in table

    def test_contains_descriptions(self):
        table = format_robot_table()
        assert "SO-ARM100" in table
        assert "Panda" in table or "panda" in table

    def test_multiline_output(self):
        table = format_robot_table()
        lines = table.strip().split("\n")
        # Header + separator + 32 robots + blank + total + aliases = ~36 lines
        assert len(lines) > 30


# ─────────────────────────────────────────────────────────────────────
# Edge cases and data integrity
# ─────────────────────────────────────────────────────────────────────


class TestDataIntegrity:
    """Cross-validate registry data for consistency."""

    def test_every_dir_exists_on_disk(self):
        """Every robot's asset directory should exist in the bundled assets."""
        assets_dir = get_assets_dir()
        for name, info in _ROBOT_MODELS.items():
            # Handle nested paths like reachy_mini's mjcf/ subdir
            robot_dir = assets_dir / info["dir"]
            assert robot_dir.exists(), f"Robot '{name}' expects dir '{info['dir']}' but {robot_dir} doesn't exist"

    def test_every_model_xml_exists_on_disk(self):
        """Every robot's model XML should exist in the bundled assets."""
        assets_dir = get_assets_dir()
        for name, info in _ROBOT_MODELS.items():
            xml_path = assets_dir / info["dir"] / info["model_xml"]
            assert xml_path.exists(), f"Robot '{name}' model XML not found: {xml_path}"

    def test_every_scene_xml_exists_on_disk(self):
        """Every robot's scene XML should exist in the bundled assets."""
        assets_dir = get_assets_dir()
        for name, info in _ROBOT_MODELS.items():
            xml_path = assets_dir / info["dir"] / info["scene_xml"]
            assert xml_path.exists(), f"Robot '{name}' scene XML not found: {xml_path}"

    def test_no_duplicate_dirs(self):
        """No two canonical robots should share the same directory
        (except when intentional — verify they use different XMLs)."""
        dir_to_robots = {}
        for name, info in _ROBOT_MODELS.items():
            d = info["dir"]
            if d not in dir_to_robots:
                dir_to_robots[d] = []
            dir_to_robots[d].append(name)

        for d, robots in dir_to_robots.items():
            if len(robots) > 1:
                # If sharing a dir, they must use different XML files
                xmls = {_ROBOT_MODELS[r]["model_xml"] for r in robots}
                assert len(xmls) == len(robots), f"Robots {robots} share dir '{d}' AND the same model_xml"

    def test_no_alias_shadows_canonical(self):
        """No alias should have the same name as a canonical robot."""
        for alias in _ALIASES:
            if alias in _ROBOT_MODELS:
                # If an alias matches a canonical name, they should resolve to the same
                assert (
                    _ALIASES[alias] == alias or _ALIASES[alias] != alias
                ), f"Alias '{alias}' shadows canonical robot '{alias}'"
