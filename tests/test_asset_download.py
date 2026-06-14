"""Behavior tests for :mod:`strands_robots.assets.download`.

Exercises the asset-download strategies end to end without network or
hardware: the ``robot_descriptions`` import path, the Menagerie ``git clone``
fallback, custom GitHub sources, and the :func:`download_robots` orchestrator
that partitions, downloads, and reports. ``subprocess``/``importlib`` are
mocked so clones become local directory fixtures, letting the tests assert on
observable outcomes (returned status dicts, files copied into the cache,
symlinks created) rather than implementation details.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.assets import download as dl

_MOD = "strands_robots.assets.download"


def _entry(asset_dir: str, xml: str = "model.xml", **asset_extra: object) -> dict[str, Any]:
    """Build a minimal registry entry with an ``asset`` block."""
    asset: dict[str, Any] = {"dir": asset_dir, "model_xml": xml}
    asset.update(asset_extra)
    return {"asset": asset, "category": "arm"}


# _robot_descriptions_available


def test_robot_descriptions_available_true_when_importable() -> None:
    with patch(f"{_MOD}.importlib.import_module"):  # not used; import is direct
        # Directly simulate a successful import of the package.
        with patch.dict("sys.modules", {"robot_descriptions": SimpleNamespace()}):
            assert dl._robot_descriptions_available() is True


def test_robot_descriptions_available_false_when_missing() -> None:
    real_import = __import__

    def _no_rd(name: str, *a: object, **k: object) -> object:
        if name == "robot_descriptions":
            raise ImportError("no robot_descriptions")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    with patch("builtins.__import__", side_effect=_no_rd):
        assert dl._robot_descriptions_available() is False


# _resolve_robot_descriptions_module


def test_resolve_module_opt_out_returns_none() -> None:
    info = _entry("panda", auto_download=False)
    assert dl._resolve_robot_descriptions_module("panda", info) is None


def test_resolve_module_prefers_explicit_registry_field() -> None:
    info = _entry("panda", robot_descriptions_module="panda_mj_description")
    assert dl._resolve_robot_descriptions_module("panda", info) == "panda_mj_description"


def test_resolve_module_naming_heuristic_finds_candidate() -> None:
    info = _entry("panda")  # no explicit module -> heuristic

    def _import(modpath: str) -> object:
        # First candidate "panda_mj_description" resolves.
        if modpath == "robot_descriptions.panda_mj_description":
            return SimpleNamespace()
        raise ImportError(modpath)

    with patch(f"{_MOD}.importlib.import_module", side_effect=_import):
        assert dl._resolve_robot_descriptions_module("panda", info) == "panda_mj_description"


def test_resolve_module_returns_none_when_no_candidate_imports() -> None:
    info = _entry("weird")
    with patch(f"{_MOD}.importlib.import_module", side_effect=ImportError):
        assert dl._resolve_robot_descriptions_module("weird", info) is None


# _needs_download


def test_needs_download_false_for_none_or_no_asset() -> None:
    assert dl._needs_download("x", None) is False
    assert dl._needs_download("x", {"asset": {}}) is False


def test_needs_download_true_when_xml_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_MOD}.get_search_paths", lambda: [tmp_path])
    assert dl._needs_download("so100", _entry("so100")) is True


def test_needs_download_false_when_xml_has_no_meshes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot_dir = tmp_path / "so100"
    robot_dir.mkdir()
    (robot_dir / "model.xml").write_text("<mujoco><worldbody/></mujoco>")
    monkeypatch.setattr(f"{_MOD}.get_search_paths", lambda: [tmp_path])
    assert dl._needs_download("so100", _entry("so100")) is False


def test_needs_download_true_when_mesh_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot_dir = tmp_path / "so100"
    robot_dir.mkdir()
    (robot_dir / "model.xml").write_text('<mujoco><asset><mesh file="base.stl"/></asset></mujoco>')
    monkeypatch.setattr(f"{_MOD}.get_search_paths", lambda: [tmp_path])
    assert dl._needs_download("so100", _entry("so100")) is True


def test_needs_download_respects_force_when_meshes_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot_dir = tmp_path / "so100"
    (robot_dir / "assets").mkdir(parents=True)
    (robot_dir / "model.xml").write_text(
        '<mujoco><compiler meshdir="assets"/><asset><mesh file="base.stl"/></asset></mujoco>'
    )
    (robot_dir / "assets" / "base.stl").write_text("solid")
    monkeypatch.setattr(f"{_MOD}.get_search_paths", lambda: [tmp_path])
    info = _entry("so100")
    assert dl._needs_download("so100", info, force=False) is False
    assert dl._needs_download("so100", info, force=True) is True


# _get_source


def test_get_source_defaults_to_menagerie() -> None:
    assert dl._get_source(None) == {"type": "menagerie"}
    assert dl._get_source(_entry("x")) == {"type": "menagerie"}


def test_get_source_returns_custom_source() -> None:
    info = _entry("x", source={"type": "github", "repo": "o/r"})
    assert dl._get_source(info)["type"] == "github"


# _shallow_clone


def test_shallow_clone_rejects_non_github_url() -> None:
    with pytest.raises(ValueError, match="Blocked clone URL"):
        dl._shallow_clone("ssh://github.com/o/r.git", "/tmp/x")


def test_shallow_clone_invokes_git_for_valid_url() -> None:
    with patch(f"{_MOD}.subprocess.run") as run:
        dl._shallow_clone("https://github.com/o/r.git", "/tmp/dest")
    args = run.call_args[0][0]
    assert args[:3] == ["git", "clone", "--depth"]
    assert args[-2:] == ["https://github.com/o/r.git", "/tmp/dest"]


# _copy_and_clean


def test_copy_and_clean_skips_docs_and_images(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.xml").write_text("x")
    (src / "README.md").write_text("doc")
    (src / "preview.png").write_text("img")
    (src / "mesh.stl").write_text("m")
    dst = tmp_path / "dst"
    dl._copy_and_clean(src, dst)
    assert (dst / "model.xml").exists()
    assert (dst / "mesh.stl").exists()
    assert not (dst / "README.md").exists()
    assert not (dst / "preview.png").exists()


# _download_via_robot_descriptions


def test_rd_download_symlinks_package_path(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "model.xml").write_text("<mujoco/>")
    dest = tmp_path / "cache"
    dest.mkdir()
    info = _entry("so100", robot_descriptions_module="so100_mj_description")

    with patch(f"{_MOD}.importlib.import_module", return_value=SimpleNamespace(PACKAGE_PATH=str(pkg))):
        results = dl._download_via_robot_descriptions({"so100": info}, dest)

    assert results["so100"] == "downloaded"
    linked = dest / "so100"
    assert linked.is_symlink()
    assert (linked / "model.xml").exists()


def test_rd_download_skips_when_no_module() -> None:
    info = _entry("panda", auto_download=False)
    results = dl._download_via_robot_descriptions({"panda": info}, Path("/tmp"))
    assert results["panda"].startswith("skipped")


def test_rd_download_reports_xml_mismatch(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()  # PACKAGE_PATH exists but lacks the expected XML
    dest = tmp_path / "cache"
    dest.mkdir()
    info = _entry("so100", robot_descriptions_module="so100_mj_description")
    with patch(f"{_MOD}.importlib.import_module", return_value=SimpleNamespace(PACKAGE_PATH=str(pkg))):
        results = dl._download_via_robot_descriptions({"so100": info}, dest)
    assert results["so100"].startswith("failed: XML mismatch")
    assert not (dest / "so100").exists()


# _download_via_git


def test_git_download_copies_matching_robot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "cache"
    dest.mkdir()
    info = _entry("panda")

    def _fake_clone(repo_url: str, clone_dir: str, **kw: object) -> None:
        robot_src = Path(clone_dir) / "panda"
        robot_src.mkdir(parents=True)
        (robot_src / "model.xml").write_text("<mujoco/>")

    with patch(f"{_MOD}._shallow_clone", side_effect=_fake_clone):
        results = dl._download_via_git({"panda": info}, dest)

    assert results["panda"] == "downloaded"
    assert (dest / "panda" / "model.xml").exists()


def test_git_download_reports_clone_failure(tmp_path: Path) -> None:
    info = _entry("panda")
    with patch(f"{_MOD}._shallow_clone", side_effect=subprocess.TimeoutExpired("git", 120)):
        results = dl._download_via_git({"panda": info}, tmp_path)
    assert results["panda"].startswith("failed: git clone timeout")


def test_git_download_reports_missing_dir(tmp_path: Path) -> None:
    info = _entry("ghost")
    with patch(f"{_MOD}._shallow_clone"):  # clone leaves an empty tree
        results = dl._download_via_git({"ghost": info}, tmp_path)
    assert "not in menagerie" in results["ghost"]


# _download_from_github


def test_github_download_rejects_bad_repo(tmp_path: Path) -> None:
    info = _entry("x", source={"type": "github", "repo": "not a repo!"})
    assert dl._download_from_github("x", info, tmp_path).startswith("failed: invalid repo")


def test_github_download_copies_subdir(tmp_path: Path) -> None:
    dest = tmp_path / "cache"
    dest.mkdir()
    info = _entry("myrobot", source={"type": "github", "repo": "owner/repo", "subdir": "robots/myrobot"})

    def _fake_clone(repo_url: str, clone_dir: str, **kw: object) -> None:
        sub = Path(clone_dir) / "robots" / "myrobot"
        sub.mkdir(parents=True)
        (sub / "model.xml").write_text("<mujoco/>")

    with patch(f"{_MOD}._shallow_clone", side_effect=_fake_clone):
        result = dl._download_from_github("myrobot", info, dest)

    assert result == "downloaded"
    assert (dest / "myrobot" / "model.xml").exists()


def test_github_download_reports_missing_subdir(tmp_path: Path) -> None:
    info = _entry("x", source={"type": "github", "repo": "owner/repo", "subdir": "absent"})
    with patch(f"{_MOD}._shallow_clone"):
        assert "not found in owner/repo" in dl._download_from_github("x", info, tmp_path)


# auto_download_robot


def test_auto_download_prefers_robot_descriptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_MOD}.get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    info = _entry("so100")
    with (
        patch(f"{_MOD}._robot_descriptions_available", return_value=True),
        patch(f"{_MOD}._download_via_robot_descriptions", return_value={"so100": "downloaded"}),
    ):
        assert dl.auto_download_robot("so100", info) is True


def test_auto_download_falls_back_to_github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_MOD}.get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    info = _entry("custom", source={"type": "github", "repo": "o/r"})
    with (
        patch(f"{_MOD}._robot_descriptions_available", return_value=False),
        patch(f"{_MOD}._download_from_github", return_value="downloaded"),
    ):
        assert dl.auto_download_robot("custom", info) is True


def test_auto_download_returns_false_when_all_strategies_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_MOD}.get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    info = _entry("custom")  # menagerie only, no github source
    with (
        patch(f"{_MOD}._robot_descriptions_available", return_value=True),
        patch(f"{_MOD}._download_via_robot_descriptions", return_value={"custom": "failed: x"}),
    ):
        assert dl.auto_download_robot("custom", info) is False


# download_robots orchestrator


def test_download_robots_unknown_name_returns_no_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: None)
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    result = dl.download_robots(names=["does-not-exist"])
    assert result["downloaded"] == 0
    assert "No matching robots" in result["message"]


def test_download_robots_all_present_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": "so100"}])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: _entry("so100"))
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: False)
    result = dl.download_robots()
    assert result["skipped"] == 1
    assert "already have assets" in result["message"]


def test_download_robots_partitions_and_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    registry = {
        "so100": _entry("so100"),  # menagerie
        "custom": _entry("custom", source={"type": "github", "repo": "o/r"}),  # github
    }
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": n} for n in registry])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: registry.get(n))
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: True)
    monkeypatch.setattr(f"{_MOD}._robot_descriptions_available", lambda: False)

    with (
        patch(f"{_MOD}._download_via_git", return_value={"so100": "downloaded"}) as git,
        patch(f"{_MOD}._download_from_github", return_value="downloaded") as gh,
    ):
        result = dl.download_robots()

    git.assert_called_once()
    gh.assert_called_once()
    assert result["downloaded"] == 2
    assert set(result["downloaded_names"]) == {"so100", "custom"}
    assert result["method"] == "git clone"


def test_download_robots_filters_by_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    registry = {
        "arm1": {"asset": {"dir": "arm1", "model_xml": "m.xml"}, "category": "arm"},
        "hum1": {"asset": {"dir": "hum1", "model_xml": "m.xml"}, "category": "humanoid"},
    }
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": n} for n in registry])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: registry.get(n))
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: False)
    result = dl.download_robots(category="humanoid")
    # Only the humanoid is considered; arm is filtered out before partition.
    assert result["skipped"] == 1
    assert result.get("skipped_names") == ["hum1"]


# Additional edge-case branches


def test_rd_download_reports_missing_package_path(tmp_path: Path) -> None:
    """When the module's PACKAGE_PATH does not exist, report failure."""
    info = _entry("so100", robot_descriptions_module="so100_mj_description")
    missing = tmp_path / "absent"
    with patch(f"{_MOD}.importlib.import_module", return_value=SimpleNamespace(PACKAGE_PATH=str(missing))):
        results = dl._download_via_robot_descriptions({"so100": info}, tmp_path)
    assert results["so100"].startswith("failed: PACKAGE_PATH missing")


def test_rd_download_rejects_invalid_module_name(tmp_path: Path) -> None:
    """A resolved module name with illegal characters is skipped, not imported."""
    info = _entry("so100", robot_descriptions_module="bad/name")
    results = dl._download_via_robot_descriptions({"so100": info}, tmp_path)
    assert results["so100"].startswith("skipped: invalid module name")


def test_rd_download_reuses_valid_existing_symlink(tmp_path: Path) -> None:
    """A symlink already pointing at PACKAGE_PATH with the XML is reused as-is."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "model.xml").write_text("<mujoco/>")
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "so100").symlink_to(pkg)
    info = _entry("so100", robot_descriptions_module="so100_mj_description")
    with patch(f"{_MOD}.importlib.import_module", return_value=SimpleNamespace(PACKAGE_PATH=str(pkg))):
        results = dl._download_via_robot_descriptions({"so100": info}, dest)
    assert results["so100"] == "downloaded"
    assert (dest / "so100").is_symlink()


def test_download_robots_retries_rd_failures_with_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Menagerie robots that fail via robot_descriptions are retried with git clone."""
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    registry = {"panda": _entry("panda")}
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": n} for n in registry])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: registry.get(n))
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: True)
    monkeypatch.setattr(f"{_MOD}._robot_descriptions_available", lambda: True)

    with (
        patch(f"{_MOD}._download_via_robot_descriptions", return_value={"panda": "failed: no module"}),
        patch(f"{_MOD}._download_via_git", return_value={"panda": "downloaded"}) as git,
    ):
        result = dl.download_robots()

    git.assert_called_once()
    assert result["downloaded"] == 1
    assert result["method"] == "robot_descriptions"
