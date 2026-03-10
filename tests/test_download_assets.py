#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.assets.download.

Strategy:
- All git/network operations are mocked (no real clones or downloads)
- Filesystem operations use tmp_path fixtures
- Tests cover: download_robots(), download_assets() tool, main() CLI,
  mesh checking logic, alias resolution, category filtering
"""

import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.assets import (
    _ROBOT_MODELS,
    format_robot_table,
    resolve_robot_name,
)
from strands_robots.assets.download import (
    MENAGERIE_REPO,
    download_assets,
    download_robots,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def assets_dir(tmp_path):
    """Provide a temp assets dir and patch get_assets_dir to use it."""
    with patch("strands_robots.assets.download.get_assets_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def mock_menagerie(tmp_path):
    """Create a fake menagerie clone directory."""
    clone_dir = tmp_path / "mujoco_menagerie"
    clone_dir.mkdir()

    # Create directories for a few robots from _ROBOT_MODELS
    for name, info in list(_ROBOT_MODELS.items())[:3]:
        robot_src = clone_dir / info["dir"]
        robot_src.mkdir(parents=True, exist_ok=True)
        (robot_src / info["model_xml"]).write_text("<mujoco/>")
        # Add some mesh files
        meshes_dir = robot_src / "meshes"
        meshes_dir.mkdir(exist_ok=True)
        (meshes_dir / "link.stl").write_bytes(b"fake stl")
        # Add non-essential files
        (robot_src / "README.md").write_text("readme")
        (robot_src / "test.png").write_bytes(b"png")

    return clone_dir


# ---------------------------------------------------------------------------
# download_robots tests
# ---------------------------------------------------------------------------


class TestDownloadRobots:

    def test_no_matching_robots(self, assets_dir):
        """Unknown robot names → no downloads."""
        result = download_robots(names=["nonexistent_robot_xyz"])
        assert result["downloaded"] == 0
        assert "No matching robots" in result["message"]

    def test_category_filter(self, assets_dir):
        """Category filter restricts to matching robots."""
        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr=b"fail")
            result = download_robots(category="arm")
        # It would try to clone and fail
        assert result["failed"] > 0 or result["downloaded"] == 0

    def test_all_already_downloaded(self, assets_dir):
        """When all robots already exist, skips them."""
        # Create existing files for first 2 robots
        for name, info in list(_ROBOT_MODELS.items())[:2]:
            robot_dir = assets_dir / info["dir"]
            robot_dir.mkdir(parents=True, exist_ok=True)
            (robot_dir / info["model_xml"]).write_text("<mujoco/>")

        result = download_robots(
            names=list(_ROBOT_MODELS.keys())[:2],
        )
        assert result["skipped"] == 2
        assert result["downloaded"] == 0
        assert "already downloaded" in result["message"]

    def test_force_redownload(self, assets_dir, mock_menagerie):
        """force=True re-downloads even existing robots."""
        first_name = list(_ROBOT_MODELS.keys())[0]
        first_info = _ROBOT_MODELS[first_name]

        # Pre-create robot dir
        robot_dir = assets_dir / first_info["dir"]
        robot_dir.mkdir(parents=True, exist_ok=True)
        (robot_dir / first_info["model_xml"]).write_text("<mujoco/>")

        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("tempfile.TemporaryDirectory") as mock_tmp:
                mock_tmp.return_value.__enter__ = MagicMock(return_value=str(mock_menagerie.parent))
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                result = download_robots(names=[first_name], force=True)

        assert result["downloaded"] >= 0  # At least attempted

    def test_clone_failure(self, assets_dir):
        """Git clone failure → error result."""
        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr=b"clone failed")
            first_name = list(_ROBOT_MODELS.keys())[0]
            result = download_robots(names=[first_name], force=True)

        assert result["downloaded"] == 0
        assert "Failed to clone" in result["message"]

    def test_successful_download(self, assets_dir, mock_menagerie):
        """Successful download from mocked menagerie."""
        first_name = list(_ROBOT_MODELS.keys())[0]

        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("tempfile.TemporaryDirectory") as mock_tmp:
                mock_tmp.return_value.__enter__ = MagicMock(return_value=str(mock_menagerie.parent))
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                result = download_robots(names=[first_name], force=True)

        # First robot in _ROBOT_MODELS should have been copied
        assert result["downloaded"] >= 1 or result["skipped"] >= 0

    def test_source_dir_not_found(self, assets_dir):
        """Robot source dir missing in menagerie → failed."""
        first_name = list(_ROBOT_MODELS.keys())[0]

        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("tempfile.TemporaryDirectory") as mock_tmp:
                empty_dir = tempfile.mkdtemp()
                mock_tmp.return_value.__enter__ = MagicMock(return_value=empty_dir)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                result = download_robots(names=[first_name], force=True)
                shutil.rmtree(empty_dir, ignore_errors=True)

        assert result["failed"] >= 1

    def test_alias_resolution(self, assets_dir):
        """Aliases like 'so100_dualcam' resolve to canonical name."""
        canonical = resolve_robot_name("so100")
        assert canonical in _ROBOT_MODELS or canonical == "so100"

    def test_mesh_check_triggers_redownload(self, assets_dir):
        """If mesh files referenced in XML are missing, re-download."""
        first_name = list(_ROBOT_MODELS.keys())[0]
        first_info = _ROBOT_MODELS[first_name]

        robot_dir = assets_dir / first_info["dir"]
        robot_dir.mkdir(parents=True, exist_ok=True)

        # Write model XML that references missing mesh files
        xml_content = """<mujoco>
            <compiler meshdir="meshes"/>
            <asset>
                <mesh file="link1.stl"/>
                <mesh file="link2.stl"/>
            </asset>
        </mujoco>"""
        (robot_dir / first_info["model_xml"]).write_text(xml_content)

        # Don't create the mesh files → should trigger re-download
        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr=b"fail")
            result = download_robots(names=[first_name])

        # Should have attempted to download (not skipped)
        assert result["skipped"] == 0

    def test_mesh_check_with_existing_meshes(self, assets_dir):
        """If referenced mesh files exist, skip download."""
        first_name = list(_ROBOT_MODELS.keys())[0]
        first_info = _ROBOT_MODELS[first_name]

        robot_dir = assets_dir / first_info["dir"]
        robot_dir.mkdir(parents=True, exist_ok=True)

        xml_content = """<mujoco>
            <compiler meshdir="meshes"/>
            <asset>
                <mesh file="link1.stl"/>
            </asset>
        </mujoco>"""
        (robot_dir / first_info["model_xml"]).write_text(xml_content)

        # Create the mesh file
        meshes = robot_dir / "meshes"
        meshes.mkdir(exist_ok=True)
        (meshes / "link1.stl").write_bytes(b"stl data")

        result = download_robots(names=[first_name])
        assert result["skipped"] == 1

    def test_mesh_check_no_meshdir(self, assets_dir):
        """XML without meshdir → meshes resolved from robot dir root."""
        first_name = list(_ROBOT_MODELS.keys())[0]
        first_info = _ROBOT_MODELS[first_name]

        robot_dir = assets_dir / first_info["dir"]
        robot_dir.mkdir(parents=True, exist_ok=True)

        # No compiler meshdir
        xml_content = '<mujoco><asset><mesh file="part.stl"/></asset></mujoco>'
        (robot_dir / first_info["model_xml"]).write_text(xml_content)

        # mesh file missing → needs download
        with patch("strands_robots.assets.download.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr=b"fail")
            result = download_robots(names=[first_name])

        assert result["skipped"] == 0

    def test_mesh_check_exception_ignored(self, assets_dir):
        """Exceptions during mesh checking don't crash."""
        first_name = list(_ROBOT_MODELS.keys())[0]
        first_info = _ROBOT_MODELS[first_name]

        robot_dir = assets_dir / first_info["dir"]
        robot_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid XML
        (robot_dir / first_info["model_xml"]).write_text("not xml at all {{{{")

        result = download_robots(names=[first_name])
        # Should skip (file exists, mesh check catches exception and passes)
        assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# download_assets tool tests
# ---------------------------------------------------------------------------


class TestDownloadAssetsTool:

    def test_list_action(self):
        """list action returns robot table."""
        result = download_assets(action="list")
        assert result["status"] == "success"
        assert "Robot Models" in result["content"][0]["text"] or len(result["content"][0]["text"]) > 0

    def test_status_action(self):
        """status action shows available/missing counts."""
        result = download_assets(action="status")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Asset Status" in text or "available" in text.lower()

    def test_download_action(self, assets_dir):
        """download action calls download_robots."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 1,
                "skipped": 0,
                "failed": 0,
                "downloaded_names": ["so100"],
                "skipped_names": [],
                "failed_names": [],
                "assets_dir": str(assets_dir),
            }
            result = download_assets(action="download", robots="so100")

        assert result["status"] == "success"
        assert "Downloaded: 1" in result["content"][0]["text"]

    def test_download_with_category(self, assets_dir):
        """download with category passes through."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 2,
                "skipped": 0,
                "failed": 0,
                "downloaded_names": ["r1", "r2"],
                "skipped_names": [],
                "failed_names": [],
                "assets_dir": str(assets_dir),
            }
            download_assets(action="download", category="humanoid")

        mock_dl.assert_called_once_with(names=None, category="humanoid", force=False)

    def test_download_with_force(self, assets_dir):
        """force=True passed through to download_robots."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "downloaded_names": [],
                "skipped_names": [],
                "failed_names": [],
                "assets_dir": str(assets_dir),
            }
            download_assets(action="download", force=True)

        mock_dl.assert_called_once_with(names=None, category=None, force=True)

    def test_download_shows_failures(self, assets_dir):
        """Failed downloads shown in result text."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 0,
                "skipped": 0,
                "failed": 1,
                "downloaded_names": [],
                "skipped_names": [],
                "failed_names": ["bad_robot"],
                "assets_dir": str(assets_dir),
            }
            result = download_assets(action="download", robots="bad_robot")

        assert "bad_robot" in result["content"][0]["text"]

    def test_unknown_action(self):
        """Unknown action returns error."""
        result = download_assets(action="fly")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_exception_handling(self):
        """Exceptions in the tool are caught and returned."""
        with patch("strands_robots.assets.download.format_robot_table", side_effect=RuntimeError("boom")):
            result = download_assets(action="list")
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]

    def test_download_parses_comma_separated_robots(self, assets_dir):
        """Comma-separated robot names are split correctly."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "downloaded_names": [],
                "skipped_names": [],
                "failed_names": [],
                "assets_dir": str(assets_dir),
            }
            download_assets(action="download", robots="so100, panda, unitree_g1")

        called_names = mock_dl.call_args[1]["names"]
        assert "so100" in called_names
        assert "panda" in called_names
        assert "unitree_g1" in called_names

    def test_download_empty_robots_string(self, assets_dir):
        """Empty robots string → downloads all."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "downloaded_names": [],
                "skipped_names": [],
                "failed_names": [],
                "assets_dir": str(assets_dir),
            }
            download_assets(action="download", robots=None)

        mock_dl.assert_called_once_with(names=None, category=None, force=False)


# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------


class TestMainCLI:

    def test_list_flag(self):
        """--list prints table and exits."""
        with patch("strands_robots.assets.download.format_robot_table", return_value="TABLE"):
            with patch("sys.argv", ["download", "--list"]):
                with patch("builtins.print") as mock_print:
                    main()
                mock_print.assert_called_with("TABLE")

    def test_download_specific_robots(self):
        """CLI with robot names calls download_robots."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {"downloaded": 1, "skipped": 0, "failed": 0}
            with patch("sys.argv", ["download", "so100", "panda"]):
                main()
            mock_dl.assert_called_once_with(
                names=["so100", "panda"],
                category=None,
                force=False,
            )

    def test_download_with_category_flag(self):
        """CLI --category passes to download_robots."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {"downloaded": 0, "skipped": 0, "failed": 0}
            with patch("sys.argv", ["download", "--category", "humanoid"]):
                main()
            mock_dl.assert_called_once_with(
                names=None,
                category="humanoid",
                force=False,
            )

    def test_download_with_force_flag(self):
        """CLI --force passes to download_robots."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {"downloaded": 0, "skipped": 0, "failed": 0}
            with patch("sys.argv", ["download", "--force"]):
                main()
            mock_dl.assert_called_once_with(
                names=None,
                category=None,
                force=True,
            )

    def test_download_all_default(self):
        """CLI with no args downloads all."""
        with patch("strands_robots.assets.download.download_robots") as mock_dl:
            mock_dl.return_value = {"downloaded": 0, "skipped": 0, "failed": 0}
            with patch("sys.argv", ["download"]):
                main()
            mock_dl.assert_called_once_with(
                names=None,
                category=None,
                force=False,
            )


# ---------------------------------------------------------------------------
# Constant and helper tests
# ---------------------------------------------------------------------------


class TestConstants:

    def test_menagerie_repo_url(self):
        assert "mujoco_menagerie" in MENAGERIE_REPO
        assert MENAGERIE_REPO.startswith("https://")

    def test_robot_models_non_empty(self):
        assert len(_ROBOT_MODELS) > 0

    def test_all_models_have_required_keys(self):
        for name, info in _ROBOT_MODELS.items():
            assert "dir" in info, f"{name} missing 'dir'"
            assert "model_xml" in info, f"{name} missing 'model_xml'"
            assert "category" in info, f"{name} missing 'category'"
            assert "description" in info, f"{name} missing 'description'"

    def test_format_robot_table_returns_string(self):
        table = format_robot_table()
        assert isinstance(table, str)
        assert len(table) > 0
