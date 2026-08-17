"""Every GR00T ``data_config`` name must resolve to a registry robot.

``data_config`` is not only a GR00T inference concept: it is an accepted robot
identifier. ``add_robot(data_config=...)`` resolves it through
``resolve_name`` -> ``get_robot`` to find the model to load, ``resolve_model``
resolves the MJCF/URDF the Isaac IK solve runs on, and
``MotionPrimitivesCore._registry_gripper_metadata`` reads the registry
``gripper`` block for it so ``move_to`` can hold a grasp instead of guessing
the gripper heuristically.

The registry encodes that by declaring each embodiment's ``data_config``
spellings in the ``aliases`` list of the robot they name - ``panda`` claims
``libero_panda`` and ``oxe_droid``, ``so101`` claims ``so101_dualcam`` and
``so101_tricam``. A config name that no entry claims resolves to nothing, so
naming it refuses a robot the catalog advertises, and the two failures are not
symmetric: ``add_robot`` reports an error, while the gripper metadata lookup
returns "no metadata" and silently falls back to the heuristic it exists to
replace.

The GR00T catalog is the vocabulary owner and grows independently of the
registry, so these tests are keyed on the live ``data_configs.json`` rather
than a copied list: a config added there fails here until the registry offers
a robot for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry import get_robot, resolve_name

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_CONFIGS_PATH = _REPO_ROOT / "strands_robots" / "policies" / "groot" / "data_configs.json"

# A floor, not the exact count: the catalog is expected to grow. It guards
# against an extractor that silently reaches nothing, which would make every
# assertion below pass vacuously.
_MINIMUM_CATALOG_SIZE = 25

_REMEDY = (
    "Add each name to the 'aliases' list of the robot entry for its embodiment in "
    "strands_robots/registry/robots.json (the spelling the GR00T catalog uses, "
    "e.g. 'panda' claims 'oxe_droid'), so naming that data_config resolves the same "
    "robot as the rest of the embodiment's configs."
)


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    """The shipped GR00T data_config catalog."""
    return dict(json.loads(_DATA_CONFIGS_PATH.read_text()))


@pytest.fixture(scope="module")
def config_names(catalog: dict[str, Any]) -> list[str]:
    """Every name a caller may pass as a ``data_config`` - configs and aliases."""
    return sorted(set(catalog["configs"]) | set(catalog["aliases"]))


def test_the_catalog_is_read_and_not_empty(config_names: list[str]) -> None:
    """Premise: a clean sweep below must mean the registry agrees, not that nothing was read."""
    assert len(config_names) >= _MINIMUM_CATALOG_SIZE, (
        f"Only {len(config_names)} data_config names were read from {_DATA_CONFIGS_PATH.name}; "
        f"expected at least {_MINIMUM_CATALOG_SIZE}. The other tests in this module would pass "
        "vacuously, so the extractor is treated as broken rather than the catalog as clean."
    )


def test_every_data_config_name_resolves_to_a_registry_robot(config_names: list[str]) -> None:
    """No advertised ``data_config`` may resolve to nothing.

    An unclaimed name makes ``add_robot(data_config=...)`` refuse with "No model
    found" for an embodiment whose other configs load, and makes the registry
    gripper metadata silently unavailable.
    """
    unresolved = [name for name in config_names if get_robot(name) is None]
    assert not unresolved, (
        "These GR00T data_config names resolve to no registry robot: "
        + ", ".join(f"{name!r} -> {resolve_name(name)!r}" for name in unresolved)
        + ". "
        + _REMEDY
    )


def test_a_config_that_extends_another_resolves_to_the_same_robot(catalog: dict[str, Any]) -> None:
    """``_extends`` declares the parent embodiment, so both must name one robot.

    A config that inherits another's keys describes the same hardware with a
    different action representation. Resolving the two to different robots (or
    the child to none) would load different models for one embodiment.
    """
    configs = catalog["configs"]
    mismatches: list[str] = []
    checked = 0
    for name, spec in configs.items():
        parent = spec.get("_extends")
        if not parent:
            continue
        checked += 1
        child_robot = resolve_name(name)
        parent_robot = resolve_name(parent)
        if get_robot(name) is None or child_robot != parent_robot:
            mismatches.append(f"{name!r} -> {child_robot!r} but its _extends parent {parent!r} -> {parent_robot!r}")
    assert checked, "No config declares '_extends', so this test proves nothing about inheritance."
    assert not mismatches, "A config resolves to a different robot than the one it extends:\n  " + "\n  ".join(
        mismatches
    )


def test_a_catalog_alias_resolves_to_the_same_robot_as_its_target(catalog: dict[str, Any]) -> None:
    """A catalog alias and its target name one robot.

    Mirrors the GR00T-side guarantee that an alias resolves to the same
    ``Gr00tDataConfig`` as its target: the registry must agree, or the two
    spellings of one config would load different robots.
    """
    mismatches: list[str] = []
    for alias, target in catalog["aliases"].items():
        alias_robot, target_robot = resolve_name(alias), resolve_name(target)
        if get_robot(alias) is None or get_robot(target) is None or alias_robot != target_robot:
            mismatches.append(f"{alias!r} -> {alias_robot!r} but target {target!r} -> {target_robot!r}")
    assert catalog["aliases"], "The catalog declares no aliases, so this test proves nothing."
    assert not mismatches, "A catalog alias and its target resolve to different robots:\n  " + "\n  ".join(mismatches)


def test_a_resolved_data_config_lands_on_an_entry_that_declares_a_simulation_asset(
    config_names: list[str],
) -> None:
    """Resolving is not enough - ``add_robot(data_config=...)`` needs a model to load.

    Scoped to the names that do resolve, so this stays independent of whether
    any name resolves at all: it pins that a config name is only ever pointed
    at an embodiment carrying an ``asset`` block. Pointing one at a
    hardware-only entry would resolve cleanly and still refuse, which reads as
    the same "No model found" as claiming nothing.
    """
    resolved = [(name, entry) for name in config_names if (entry := get_robot(name)) is not None]
    assert resolved, "No data_config name resolves at all, so this test proves nothing about what they resolve to."
    assetless = [name for name, entry in resolved if not entry.get("asset")]
    assert not assetless, (
        "These data_config names resolve to a registry entry with no 'asset' block, so "
        "add_robot(data_config=...) has no model to load: "
        + ", ".join(f"{name!r} -> {resolve_name(name)!r}" for name in assetless)
    )
