"""Tests for ``robot_descriptions`` auto-discovery of the standard robot long tail.

The curated ``robots.json`` carries only robots that need project-specific
metadata. Standard MJCF robots shipped by ``robot_descriptions`` (MuJoCo
Menagerie) are resolved on demand by ``strands_robots.registry.discovery`` so
``Robot("iiwa14", mode="sim")`` works without a hand-written registry entry.

These tests exercise observable behavior:
    - the cheap name -> module lookup (``descriptions_module`` / ``is_discoverable``
      / ``list_discoverable``) against the real description table,
    - the heavy entry synthesis (``discover_robot``) with a stubbed import so the
      synthesis logic is verified without any network clone,
    - that a curated entry always wins over discovery in the asset resolver,
    - that the ``Robot`` factory accepts a discoverable name.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import strands_robots.assets.manager as manager
from strands_robots.registry import discovery
from strands_robots.registry.user_registry import register_robot

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    """Reset discovery caches around every test so synthesized entries don't leak."""
    discovery.invalidate_cache()
    yield
    discovery.invalidate_cache()


# Cheap lookup surface (real description table)


def test_descriptions_module_resolves_long_tail_robot() -> None:
    pytest.importorskip("robot_descriptions")
    # iiwa14 is an MJCF Menagerie robot that is NOT in the curated robots.json.
    assert discovery.descriptions_module("iiwa14") == "iiwa14_mj_description"


def test_descriptions_module_normalizes_dashes_and_case() -> None:
    pytest.importorskip("robot_descriptions")
    assert discovery.descriptions_module("IIWA14") == "iiwa14_mj_description"


def test_descriptions_module_rejects_traversal_names() -> None:
    # Names with path/dot/slash chars must never reach importlib.
    assert discovery.descriptions_module("../evil") is None
    assert discovery.descriptions_module("robot_descriptions.os") is None


def test_descriptions_module_unknown_returns_none() -> None:
    assert discovery.descriptions_module("definitely_not_a_robot_xyz") is None


def test_is_discoverable_true_for_menagerie_robot() -> None:
    pytest.importorskip("robot_descriptions")
    assert discovery.is_discoverable("iiwa14") is True


def test_is_discoverable_false_for_unknown() -> None:
    assert discovery.is_discoverable("definitely_not_a_robot_xyz") is False


def test_list_discoverable_is_sorted_and_contains_known_robots() -> None:
    pytest.importorskip("robot_descriptions")
    names = discovery.list_discoverable()
    assert names == sorted(names)
    # A representative sample of MJCF Menagerie robots.
    for expected in ("iiwa14", "go2", "h1", "cassie"):
        assert expected in names


# Entry synthesis (stubbed import - no network)


def _install_stub_description(monkeypatch, tmp_path, *, with_scene: bool) -> Path:
    """Wire discovery to a fake MJCF description rooted at *tmp_path*.

    Returns the package directory containing the synthesized model files.
    """
    pkg_dir = tmp_path / "fakebot_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "fakebot.xml").write_text(_MINIMAL_MJCF)
    if with_scene:
        (pkg_dir / "scene.xml").write_text(_MINIMAL_MJCF)

    monkeypatch.setattr(
        discovery,
        "_mjcf_modules",
        lambda: {"fakebot": "fakebot_mj_description"},
    )

    stub = SimpleNamespace(
        MJCF_PATH=str(pkg_dir / "fakebot.xml"),
        PACKAGE_PATH=str(pkg_dir),
    )

    def _fake_import(modpath: str):
        if modpath == "robot_descriptions.fakebot_mj_description":
            return stub
        raise ImportError(modpath)

    monkeypatch.setattr(discovery.importlib, "import_module", _fake_import)
    return pkg_dir


def test_discover_robot_synthesizes_entry_with_scene(monkeypatch, tmp_path) -> None:
    _install_stub_description(monkeypatch, tmp_path, with_scene=True)
    entry = discovery.discover_robot("fakebot")
    assert entry is not None
    assert entry["discovered"] is True
    assert entry["category"] == "discovered"
    asset = entry["asset"]
    assert asset["dir"] == "fakebot_pkg"
    assert asset["model_xml"] == "fakebot.xml"
    assert asset["scene_xml"] == "scene.xml"
    assert asset["robot_descriptions_module"] == "fakebot_mj_description"


def test_discover_robot_falls_back_to_model_xml_without_scene(monkeypatch, tmp_path) -> None:
    _install_stub_description(monkeypatch, tmp_path, with_scene=False)
    entry = discovery.discover_robot("fakebot")
    assert entry is not None
    # No scene.xml shipped -> scene_xml falls back to the bare model.
    assert entry["asset"]["scene_xml"] == "fakebot.xml"


def test_discover_robot_returns_none_for_non_description(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "_mjcf_modules", dict)

    def _explode(modpath: str):
        raise AssertionError(f"import_module must not be called for {modpath!r}")

    monkeypatch.setattr(discovery.importlib, "import_module", _explode)
    assert discovery.discover_robot("definitely_not_a_robot_xyz") is None


def test_discover_robot_handles_module_without_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(discovery, "_mjcf_modules", lambda: {"fakebot": "fakebot_mj_description"})
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda modpath: SimpleNamespace(),  # no MJCF_PATH / PACKAGE_PATH
    )
    assert discovery.discover_robot("fakebot") is None


# Resolver integration: curated wins, discovery fills the gap


def test_curated_entry_wins_over_discovery(monkeypatch, tmp_path) -> None:
    """A robots.json / user-registered entry must take precedence over discovery."""
    assets_dir = tmp_path / "assets"
    robot_dir = assets_dir / "curated_dir"
    robot_dir.mkdir(parents=True)
    (robot_dir / "curated.xml").write_text(_MINIMAL_MJCF)
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))

    register_robot(
        name="dualbot",
        model_xml="curated.xml",
        description="curated dualbot",
        category="arm",
        joints=6,
        asset_dir="curated_dir",
        overwrite=True,
    )

    def _discovery_must_not_run(name: str):
        raise AssertionError("discover_robot must not be consulted when curated entry exists")

    monkeypatch.setattr("strands_robots.registry.discovery.discover_robot", _discovery_must_not_run)

    resolved = manager.resolve_model_path("dualbot")
    assert resolved is not None
    assert resolved.name == "curated.xml"


def test_resolve_model_path_uses_discovery_for_uncurated_robot(monkeypatch, tmp_path) -> None:
    """When a name is unknown to the curated registry, the discovered asset resolves."""
    assets_dir = tmp_path / "assets"
    disc_dir = assets_dir / "discbot_pkg"
    disc_dir.mkdir(parents=True)
    (disc_dir / "discbot.xml").write_text(_MINIMAL_MJCF)
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))
    manager._MESH_CACHE.clear()

    synth = {
        "description": "discbot (discovered)",
        "category": "discovered",
        "discovered": True,
        "asset": {
            "dir": "discbot_pkg",
            "model_xml": "discbot.xml",
            "scene_xml": "discbot.xml",
            "robot_descriptions_module": "discbot_mj_description",
        },
    }
    monkeypatch.setattr(
        "strands_robots.registry.discovery.discover_robot",
        lambda name: synth if name == "discbot" else None,
    )

    resolved = manager.resolve_model_path("discbot")
    assert resolved is not None
    assert resolved.name == "discbot.xml"


# Robot factory validation


def test_robot_factory_accepts_discoverable_name(monkeypatch) -> None:
    """``_validate_known_robot`` must not reject a robot_descriptions robot."""
    from strands_robots.robot import _validate_known_robot

    monkeypatch.setattr("strands_robots.robot.get_robot", lambda name: None)
    monkeypatch.setattr("strands_robots.robot.has_sim", lambda name: False)
    monkeypatch.setattr("strands_robots.robot.has_hardware", lambda name: False)
    monkeypatch.setattr("strands_robots.robot.is_discoverable", lambda name: True)

    # Should not raise.
    _validate_known_robot("iiwa14", "iiwa14", None)


def test_robot_factory_rejects_truly_unknown_name(monkeypatch) -> None:
    from strands_robots.robot import _validate_known_robot

    monkeypatch.setattr("strands_robots.robot.get_robot", lambda name: None)
    monkeypatch.setattr("strands_robots.robot.has_sim", lambda name: False)
    monkeypatch.setattr("strands_robots.robot.has_hardware", lambda name: False)
    monkeypatch.setattr("strands_robots.robot.is_discoverable", lambda name: False)

    with pytest.raises(ValueError, match="Unknown robot"):
        _validate_known_robot("bogus", "bogus", None)
