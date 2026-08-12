"""Registry integrity tests - catch silent regressions in robots.json.

These tests enforce invariants on the robot registry that prevent classes
of bugs like the one flagged by @awsarron on PR #84 review (2026-04-21):
entries where ``robot_descriptions_module`` was accidentally dropped during
the 38→68 robot expansion, silently breaking auto-download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).parent.parent / "strands_robots" / "registry" / "robots.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    """Load the robot registry once per module."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    return data.get("robots", data)


def test_registry_loads(registry: dict) -> None:
    """Registry file parses as valid JSON with robot entries."""
    assert len(registry) > 0


def test_every_robot_declares_auto_download_strategy(registry: dict) -> None:
    """Every robot with an ``asset`` block must declare HOW it gets auto-downloaded.

    Valid options (exactly one required):
        1. ``asset.robot_descriptions_module`` - the robot_descriptions pip module name.
        2. ``asset.source`` with ``type: "github"`` - custom GitHub source block.
        3. ``asset.auto_download: false`` - explicit opt-out (user must supply assets).

    Without one of these, auto-download silently falls through to the
    naming-convention heuristic, which fails for most robots and only
    logs a warning. This was the trossen_wxai + google_robot regression.
    """
    offenders = []
    for name, info in registry.items():
        asset = info.get("asset")
        if not asset:
            continue  # No asset block - nothing to auto-download.

        has_rd = "robot_descriptions_module" in asset
        has_source = isinstance(asset.get("source"), dict) and asset["source"].get("type") == "github"
        opts_out = asset.get("auto_download") is False

        if not (has_rd or has_source or opts_out):
            offenders.append(name)

    assert not offenders, (
        "Robots missing auto-download strategy (add `robot_descriptions_module`, "
        "`source: {type: github, ...}`, or `auto_download: false`): " + ", ".join(offenders)
    )


def test_asset_dirs_are_unique(registry: dict) -> None:
    """No two robots should share the same asset directory name."""
    dir_counts: dict[str, list[str]] = {}
    for name, info in registry.items():
        asset_dir = info.get("asset", {}).get("dir")
        if asset_dir:
            dir_counts.setdefault(asset_dir, []).append(name)

    duplicates = {d: names for d, names in dir_counts.items() if len(names) > 1}
    assert not duplicates, f"Duplicate asset dirs: {duplicates}"


def test_no_path_traversal_in_asset_paths(registry: dict) -> None:
    """Registry-sourced paths must not contain ``..`` (path-traversal defense in depth)."""
    for name, info in registry.items():
        asset = info.get("asset", {})
        for key in ("dir", "model_xml", "scene_xml"):
            value = asset.get(key, "")
            assert ".." not in str(value).split("/"), f"{name}.asset.{key} contains '..': {value!r}"


def test_auto_download_false_is_bool_not_string(registry: dict) -> None:
    """``auto_download`` must be a proper JSON boolean, not the string ``"false"``."""
    for name, info in registry.items():
        ad = info.get("asset", {}).get("auto_download")
        if ad is not None:
            assert isinstance(ad, bool), f"{name}.asset.auto_download must be bool, got {type(ad).__name__}: {ad!r}"


def _all_canonical_names(registry: dict) -> set[str]:
    return set(registry.keys())


def _collect_aliases(registry: dict) -> dict[str, str]:
    """Return mapping of alias → owning robot name."""
    out: dict[str, str] = {}
    for name, info in registry.items():
        for alias in info.get("aliases", []) or []:
            out.setdefault(alias, name)
    return out


def test_aliases_unique_across_registry(registry: dict) -> None:
    """No two robots may declare the same alias - last-loaded would silently win."""
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for name, info in registry.items():
        for alias in info.get("aliases", []) or []:
            if alias in seen and seen[alias] != name:
                collisions.append(f"{alias!r} used by {seen[alias]} AND {name}")
            seen[alias] = name
    assert not collisions, "Alias collisions:\n  " + "\n  ".join(collisions)


def test_no_alias_shadows_canonical_name(registry: dict) -> None:
    """An alias must not equal the canonical name of another robot.

    Shadowing causes resolution order to silently determine the winner, which
    is fragile - a future reorder of robots.json could flip which robot a
    name resolves to.
    """
    canonical = _all_canonical_names(registry)
    shadows: list[str] = []
    for name, info in registry.items():
        for alias in info.get("aliases", []) or []:
            if alias in canonical and alias != name:
                shadows.append(f"{name}.aliases contains {alias!r} which is a canonical robot name")
    assert not shadows, "Alias shadows canonical:\n  " + "\n  ".join(shadows)


def test_hardware_only_robots_declare_lerobot_type(registry: dict) -> None:
    """Robots without an ``asset`` block must still declare a LeRobot hardware type.

    Prevents silent typos in ``hardware.lerobot_type`` - catches a misspelled
    type during registry expansion rather than at teleop time.
    """
    offenders: list[str] = []
    for name, info in registry.items():
        if "asset" in info:
            continue
        hw = info.get("hardware") or {}
        lerobot_type = hw.get("lerobot_type")
        if not isinstance(lerobot_type, str) or not lerobot_type.strip():
            offenders.append(name)
    assert not offenders, "Hardware-only robots missing 'hardware.lerobot_type': " + ", ".join(offenders)


def test_robot_descriptions_module_names_are_import_safe(registry: dict) -> None:
    """Every ``robot_descriptions_module`` must match the import-safe pattern
    the downloader enforces (``^[a-z0-9_+]+$``).

    Regression: the downloader's validation regex originally was
    ``^[a-z0-9_]+$``, which rejected the legitimate upstream module
    ``tiago++_mj_description`` ("skipped: invalid module name") - so tiago_dual
    never linked its assets. '+' is now allowed; this guards both the registry
    entries and the regex staying in sync.
    """
    import re

    pattern = re.compile(r"^[a-z0-9_+]+$")
    offenders = []
    for name, info in registry.items():
        mod = info.get("asset", {}).get("robot_descriptions_module")
        if mod and not pattern.match(mod):
            offenders.append((name, mod))
    assert not offenders, f"robot_descriptions_module names not import-safe: {offenders}"


def test_rebot_b601_family_is_drivable_real(registry: dict) -> None:
    """The Seeed reBot B601-DM family (single + bimanual) must be reachable via
    ``Robot(name, mode="real")``.

    LeRobot registers ``rebot_b601_follower`` and ``bi_rebot_b601_follower``
    (the latter only when the optional ``motorbridge`` SDK is present). Without
    a strands registry entry mapping a canonical name to those LeRobot types,
    ``Robot("rebot_b601", mode="real")`` raises ``ValueError: Unsupported robot
    type`` even though the policy embodiment configs already ship for it. This
    pins the registry mapping so the hardware stays reachable. Deterministic:
    it reads only robots.json, so it guards CI hosts without LeRobot installed.
    """
    from strands_robots.registry.robots import get_hardware_type, resolve_name

    expected = {
        "rebot_b601": "rebot_b601_follower",
        "bi_rebot_b601": "bi_rebot_b601_follower",
    }
    for canonical, lerobot_type in expected.items():
        assert canonical in registry, f"{canonical!r} missing from registry"
        assert registry[canonical]["hardware"]["lerobot_type"] == lerobot_type
        # The lerobot_type itself and the canonical name both resolve home.
        assert resolve_name(canonical) == canonical
        assert resolve_name(lerobot_type) == canonical
        assert get_hardware_type(canonical) == lerobot_type


# Valid values for gripper.closed / gripper.open: which END of the gripper's
# set-point range the state maps to. Kept in sync with
# strands_robots/simulation/motion_primitives_base.py::_CTRLRANGE_ENDS.
_GRIPPER_ENDS = {"low", "high"}


def test_gripper_metadata_shape(registry: dict) -> None:
    """Optional ``gripper`` blocks are shape-checked when present (GH #1658).

    The motion primitives treat this metadata as AUTHORITATIVE over the
    gripper name heuristic, so a malformed block would either brick
    ``set_gripper``/``move_to`` for that robot or silently misclassify an
    arm DOF as a gripper. Shape contract::

        "gripper": {
            "actuators": ["<actuator short name>", ...],   # non-empty
            "closed": "low" | "high",                       # ctrlrange end
            "open":   "low" | "high"                        # must differ
        }

    Actuator names are the namespace-stripped names in the robot's SHIPPED
    sim MJCF (``asset.model_xml``), matched case-insensitively at runtime.
    """
    problems: list[str] = []
    for name, info in registry.items():
        gripper = info.get("gripper")
        if gripper is None:
            continue
        if not isinstance(gripper, dict):
            problems.append(f"{name}.gripper must be a dict, got {type(gripper).__name__}")
            continue
        unknown = set(gripper) - {"actuators", "closed", "open"}
        if unknown:
            problems.append(f"{name}.gripper has unknown keys: {sorted(unknown)}")
        actuators = gripper.get("actuators")
        if not (isinstance(actuators, list) and actuators and all(isinstance(a, str) and a.strip() for a in actuators)):
            problems.append(f"{name}.gripper.actuators must be a non-empty list of non-empty strings: {actuators!r}")
        closed = gripper.get("closed", "low")
        opened = gripper.get("open", "high")
        if closed not in _GRIPPER_ENDS:
            problems.append(f"{name}.gripper.closed must be one of {sorted(_GRIPPER_ENDS)}: {closed!r}")
        if opened not in _GRIPPER_ENDS:
            problems.append(f"{name}.gripper.open must be one of {sorted(_GRIPPER_ENDS)}: {opened!r}")
        if closed in _GRIPPER_ENDS and opened in _GRIPPER_ENDS and closed == opened:
            problems.append(f"{name}.gripper: 'closed' and 'open' must map to different ctrlrange ends")
    assert not problems, "Malformed gripper metadata:\n  " + "\n  ".join(problems)


def test_shipped_gripper_metadata_entries(registry: dict) -> None:
    """Pin the gripper metadata for the robots we ship policy configs for.

    The actuator names were verified against each robot's shipped sim model
    (``asset.model_xml``); losing an entry silently demotes that robot to the
    name heuristic, which FAILS for so101 (actuators named ``1``..``6``) and
    panda (tendon-driven ``actuator8``) - their grippers would become
    undrivable through ``set_gripper``.
    """
    expected = {
        "so100": ["Jaw"],  # trs_so_arm100/so_arm100.xml
        "so101": ["6"],  # robotstudio so101_new_calib.xml (actuators named 1..6)
        "panda": ["actuator8"],  # franka_emika_panda/panda.xml (tendon 'split', 0=closed 255=open)
    }
    for name, actuators in expected.items():
        gripper = registry[name].get("gripper")
        assert gripper is not None, f"{name} lost its registry gripper metadata"
        assert gripper["actuators"] == actuators, f"{name}.gripper.actuators changed: {gripper['actuators']}"
        assert gripper["closed"] == "low" and gripper["open"] == "high", name
