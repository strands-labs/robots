# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The declared ``robot_descriptions`` floor must ship every module the registry names.

``strands_robots/registry/robots.json`` gives 57 robots an
``asset.robot_descriptions_module`` -- the ``robot_descriptions`` submodule whose
import fetches that robot's MJCF and meshes. ``robot_descriptions`` grows one
module per newly packaged robot, so the field is a version claim: a release older
than the robot simply does not contain the module.

The floor previously read ``>=1.11.0``, twelve releases below the capability.
Measured by resolving each declared module against the released wheels (module
existence only, via :func:`importlib.util.find_spec`, so nothing is downloaded):

===========  ====================================
release      registry modules it does not provide
===========  ====================================
1.11.0       27
1.12.0       25
1.13.0       23
1.14.0       22
1.15.0       20
1.16.0       17
1.17.0       14
1.18.0       10
1.19.0       8
1.20.0       7
1.21.0       5
1.22.0       1
1.23.0       0
===========  ====================================

So of the thirteen releases ``>=1.11.0,<2.0.0`` admitted, twelve were missing at
least one registry robot; ``openarm``'s ``openarm_v1_mj_description`` is the last
to arrive, in 1.23.0.

The consequence is not a slow path but a missing robot. None of the 27 robots
absent at 1.11.0 declares an ``asset.source`` GitHub fallback, and
:func:`strands_robots.assets.download.auto_download_robot` only tries
``robot_descriptions`` and then that GitHub source -- the Menagerie ``git clone``
fallback lives in the bulk ``download_robots`` path, not the auto one. So on a
resolve anywhere in the old range, ``Robot("so101", mode="sim")`` raised
``Robot 'so101' is registered but its model file is not on disk ... Fetch it with
the download_assets tool``, naming a remedy that cannot supply the module either.
``so100`` and ``so101`` -- the arms the README leads with -- were among the 27.

:data:`_FIRST_RELEASE_WITH_EVERY_REGISTRY_MODULE` is the single owner of the
number; the pyproject comment points back here rather than restating the
measurement. ``tests/test_registry_integrity.py`` already pins that every
declared module name is import-*safe* (matches the allowed character pattern);
this file pins that it is import-*able* on the oldest install packaging admits.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import strands_robots

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = Path(strands_robots.__file__).resolve().parent / "registry" / "robots.json"

#: Oldest ``robot_descriptions`` release providing a module for every robot the
#: built-in registry names. Raise this only after re-measuring the table above.
_FIRST_RELEASE_WITH_EVERY_REGISTRY_MODULE = "1.23.0"

#: The registry declared this many modules when the table above was measured. A
#: guard that graded far fewer would pass while checking almost nothing.
_MINIMUM_DECLARED_MODULES = 50


def _declared_registry_modules() -> dict[str, str]:
    """Map robot name -> declared ``robot_descriptions`` module for every robot with one."""
    robots = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))["robots"]
    declared: dict[str, str] = {}
    for name, entry in robots.items():
        asset = entry.get("asset")
        if not isinstance(asset, dict):
            continue
        module = asset.get("robot_descriptions_module")
        if module:
            declared[name] = str(module)
    return declared


def _declared_specifiers() -> dict[str, Requirement]:
    """Every declared ``robot_descriptions`` requirement, keyed by where it lives."""
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    out: dict[str, Requirement] = {}
    for raw in project["dependencies"]:
        req = Requirement(raw)
        if req.name.replace("-", "_") == "robot_descriptions":
            out["project.dependencies"] = req
    for extra, entries in project.get("optional-dependencies", {}).items():
        for raw in entries:
            req = Requirement(raw)
            if req.name.replace("-", "_") == "robot_descriptions":
                out[f"optional-dependencies.{extra}"] = req
    return out


def _provides(module: str) -> bool:
    """Whether the installed distribution contains ``robot_descriptions.<module>``.

    Uses :func:`importlib.util.find_spec`, which locates the submodule without
    executing it -- importing one of these modules clones the upstream asset
    repository, which a test must never do.
    """
    try:
        return importlib.util.find_spec(f"robot_descriptions.{module}") is not None
    except (ImportError, ValueError):
        return False


class TestTheDeclaredFloorCoversEveryRegistryRobot:
    """Packaging must not admit a release missing a robot the registry ships."""

    def test_every_declared_specifier_floors_at_the_capability(self) -> None:
        required = Version(_FIRST_RELEASE_WITH_EVERY_REGISTRY_MODULE)
        specifiers = _declared_specifiers()
        assert specifiers, "expected at least one declared robot_descriptions requirement"

        too_low = {}
        for where, req in specifiers.items():
            lower = [s for s in req.specifier if s.operator in (">=", "==", "~=")]
            assert lower, f"{where}: {req} declares no lower bound"
            floor = min(Version(s.version) for s in lower)
            if floor < required:
                too_low[where] = str(req)
        assert not too_low, (
            f"robot_descriptions floor must be >= {required}: the registry declares "
            f"{len(_declared_registry_modules())} description modules and no earlier release "
            "provides all of them, so an older resolve leaves registered robots (so100 and "
            "so101 among them) with no way to fetch a model. These specifiers admit older "
            f"releases: {too_low}"
        )

    def test_the_required_floor_is_admitted_by_every_specifier(self) -> None:
        # A floor above the cap would make the range unsatisfiable rather than fixed.
        required = Version(_FIRST_RELEASE_WITH_EVERY_REGISTRY_MODULE)
        for where, req in _declared_specifiers().items():
            assert required in req.specifier, (
                f"{where}: {req} excludes {required}, the oldest release that provides every "
                "registry description module"
            )

    def test_the_upper_bound_stays_inside_the_audited_major(self) -> None:
        # The table was measured against 1.x wheels only; reaching into 2.0 would
        # claim a major nobody probed.
        for where, req in _declared_specifiers().items():
            assert any(s.operator == "<" and Version(s.version) <= Version("2.0.0") for s in req.specifier), (
                f"{where}: {req} should cap below 2.0.0 until the 2.x module set is audited"
            )


class TestTheFloorIsSelfMaintaining:
    """A robot added later must not quietly outrun the declared floor."""

    def test_the_registry_declares_the_modules_this_guard_grades(self) -> None:
        declared = _declared_registry_modules()
        assert len(declared) >= _MINIMUM_DECLARED_MODULES, (
            f"only {len(declared)} robots declare a robot_descriptions module; the checks below "
            "would pass while grading almost nothing. Re-measure the floor if the registry "
            "genuinely shrank this far."
        )

    def test_the_installed_distribution_provides_every_registry_module(self) -> None:
        pytest.importorskip("robot_descriptions")
        missing = {name: module for name, module in _declared_registry_modules().items() if not _provides(module)}
        assert not missing, (
            "the installed robot_descriptions does not provide these registry modules, so those "
            f"robots cannot fetch a model: {missing}. Either the install is below the declared "
            f"floor of {_FIRST_RELEASE_WITH_EVERY_REGISTRY_MODULE}, or a newly registered robot "
            "needs a newer release than the floor admits - raise the floor (and the cap if the "
            "module only exists in a later major)."
        )
