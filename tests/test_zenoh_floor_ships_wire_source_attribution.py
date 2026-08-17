# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The declared eclipse-zenoh floor must ship the wire source attribution the mesh uses.

The mesh safety handlers (``strands/safety/estop`` and ``strands/safety/resume``)
authenticate a publisher at the wire level, below the JSON body an attacker can
mutate. Two zenoh names carry that:

* ``zenoh.SourceInfo`` -- the publisher half.
  :meth:`strands_robots.mesh.core.Mesh._publish_safety_envelope` attaches
  ``SourceInfo(publisher.id, sn)`` to the sample.
* ``Sample.source_info`` -- the receiver half.
  :func:`strands_robots.mesh.core._extract_sample_source_zid` reads
  ``sample.source_info.source_id.zid`` and the handlers compare it against the
  body's ``source_zid``.

Both first ship in **eclipse-zenoh 1.6.1**. Measured against the released wheels:

=================  ==================  =====================
release            ``zenoh.SourceInfo``  ``Sample.source_info``
=================  ==================  =====================
1.0.0 .. 1.5.1     absent              absent
1.6.1 and later    present             present
=================  ==================  =====================

The floor previously read ``>=1.0.0``, so ten of the audited releases satisfied
packaging while exposing neither name. Nothing raises there: the publisher side
is guarded by ``hasattr(zenoh, "SourceInfo")`` and the receiver side by
``getattr(sample, "source_info", None)``, both of which answer "no attribution
available" and continue. So the observable is not an error but two silent
outcomes:

1. On an all-old fleet every safety envelope goes out unattributed. The
   cross-session forgery defence the handlers document is entirely off, with no
   log line and no way for an operator to tell.
2. In a mixed fleet the safety path *fails closed on the wrong peer*. A
   publisher on 1.6.1 attaches both the wire zid and the body ``source_zid``; a
   receiver on 1.5.1 sees ``wire_zid is None`` with ``body_zid`` present, takes
   the "publisher misconfigured or attacker stripped SourceInfo" branch of
   :meth:`~strands_robots.mesh.core.Mesh._on_safety_estop`, and refuses -- so a
   fleet-wide emergency stop does not stop that robot, and the warning blames
   the publisher. Both peers satisfied the old declared range.

:data:`_ZENOH_ATTRIBUTE_FLOORS` and :data:`_SAMPLE_SOURCE_INFO_FLOOR` are the
single owners of the measurement; the pyproject comment points back here rather
than restating it. :class:`TestTheFloorIsSelfMaintaining` fails the moment the
mesh reaches for a zenoh name with no recorded release, so a newer API cannot
silently leave the floor behind.
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import strands_robots

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

#: The releases the table below was measured against, oldest first. A recorded
#: floor that is not one of these was read from a changelog rather than from a
#: wheel, which is exactly the guesswork this file exists to remove.
_PROBED_RELEASES: tuple[str, ...] = (
    "1.0.0",
    "1.0.4",
    "1.1.0",
    "1.1.1",
    "1.2.0",
    "1.2.1",
    "1.3.0",
    "1.4.0",
    "1.5.0",
    "1.5.1",
    "1.6.1",
    "1.6.2",
    "1.7.2",
    "1.8.0",
    "1.9.0",
    "1.10.0",
)

#: ``zenoh`` module attribute -> first release that provides it. A name that
#: predates the oldest probed release is recorded there: the required floor is a
#: maximum, so an older true origin cannot change the answer, and the entry can
#: never understate what the mesh needs.
_ZENOH_ATTRIBUTE_FLOORS: dict[str, str] = {
    "Config": "1.0.0",
    "open": "1.0.0",
    "ZError": "1.0.0",
    "SourceInfo": "1.6.1",
}

#: The receiver half of the same capability. ``Sample`` exists on every probed
#: release, but only 1.6.1 and later give it a ``source_info`` descriptor, so an
#: AST sweep of module attributes cannot see this one -- it is read off a sample
#: object at runtime and has to be recorded by hand.
_SAMPLE_SOURCE_INFO_ATTRIBUTE = "source_info"
_SAMPLE_SOURCE_INFO_FLOOR = "1.6.1"

#: A refactor that stops reaching for zenoh entirely must not turn this file
#: into a vacuous pass.
_MINIMUM_ATTRIBUTES_REACHED = 4


def _required_floor() -> Version:
    """The highest first-shipped release among the zenoh names the mesh uses."""
    return max(Version(v) for v in (*_ZENOH_ATTRIBUTE_FLOORS.values(), _SAMPLE_SOURCE_INFO_FLOOR))


def _zenoh_attributes_reached() -> dict[str, list[str]]:
    """Map every ``zenoh.<attr>`` the shipped sources reach for to its files.

    Covers both spellings the mesh uses: a direct attribute access
    (``zenoh.Config()``) and a capability probe that names the attribute as a
    string (``hasattr(zenoh, "SourceInfo")``, ``getattr(zenoh, "ZError", None)``).
    Parses the sources rather than reading ``sys.modules`` so the audit does not
    depend on zenoh being installed.
    """
    found: dict[str, list[str]] = {}
    probes = {"hasattr", "getattr"}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "zenoh" or alias.name.startswith("zenoh."):
                        bound.add(alias.asname or alias.name.split(".")[0])
        if not bound:
            continue
        rel = str(path.relative_to(_PACKAGE_ROOT.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound:
                found.setdefault(node.attr, []).append(rel)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in probes
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in bound
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                found.setdefault(node.args[1].value, []).append(rel)
    return found


def _declared_zenoh_specifiers() -> dict[str, Requirement]:
    """Every declared ``eclipse-zenoh`` requirement, keyed by where it lives."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    out: dict[str, Requirement] = {}
    for raw in project["dependencies"]:
        req = Requirement(raw)
        if req.name == "eclipse-zenoh":
            out["project.dependencies"] = req
    for extra, entries in project.get("optional-dependencies", {}).items():
        for raw in entries:
            req = Requirement(raw)
            if req.name == "eclipse-zenoh":
                out[f"optional-dependencies.{extra}"] = req
    return out


class TestTheDeclaredFloorCoversTheWireAttributionPath:
    """Packaging must not admit a zenoh that cannot attribute a safety envelope."""

    def test_every_declared_specifier_floors_at_the_capability(self) -> None:
        required = _required_floor()
        specifiers = _declared_zenoh_specifiers()
        assert specifiers, "expected at least one declared eclipse-zenoh requirement"

        too_low = {}
        for where, req in specifiers.items():
            lower = [s for s in req.specifier if s.operator in (">=", "==", "~=")]
            assert lower, f"{where}: {req} declares no lower bound"
            floor = min(Version(s.version) for s in lower)
            if floor < required:
                too_low[where] = str(req)
        assert not too_low, (
            f"eclipse-zenoh floor must be >= {required} because the mesh needs "
            f"zenoh.SourceInfo and Sample.{_SAMPLE_SOURCE_INFO_ATTRIBUTE} for the safety "
            "handlers' wire source attribution; these specifiers admit releases that "
            f"expose neither: {too_low}"
        )

    def test_the_required_floor_is_inside_every_declared_range(self) -> None:
        # A floor above the declared cap would be unsatisfiable rather than fixed.
        required = _required_floor()
        for where, req in _declared_zenoh_specifiers().items():
            assert req.specifier.contains(str(required)), (
                f"{where}: {req} does not admit {required}, the release that first ships the wire attribution names"
            )

    def test_the_upper_bound_stays_inside_the_audited_major(self) -> None:
        # The table was measured against 1.x wheels only.
        for where, req in _declared_zenoh_specifiers().items():
            assert any(s.operator == "<" and Version(s.version) <= Version("2.0.0") for s in req.specifier), (
                f"{where}: {req} should cap below 2.0.0 until the 2.x API is audited"
            )


class TestTheFloorIsSelfMaintaining:
    """The table must stay in step with the zenoh names the sources reach for."""

    def test_every_zenoh_attribute_reached_has_a_recorded_floor(self) -> None:
        reached = _zenoh_attributes_reached()
        assert len(reached) >= _MINIMUM_ATTRIBUTES_REACHED, (
            f"only {len(reached)} zenoh attributes found in the shipped sources "
            f"({sorted(reached)}); the sweep is no longer reaching the mesh, so a clean "
            "result here would prove nothing"
        )
        unrecorded = {attr: files for attr, files in reached.items() if attr not in _ZENOH_ATTRIBUTE_FLOORS}
        assert not unrecorded, (
            "these zenoh attributes are used with no recorded first-shipped release, so "
            f"nothing checks the packaging floor against them: {unrecorded}. Add each to "
            "_ZENOH_ATTRIBUTE_FLOORS with the release that first ships it (and to "
            "_PROBED_RELEASES if that release is not listed), then raise the pyproject "
            "floor if it is higher."
        )

    def test_the_table_records_nothing_the_mesh_stopped_using(self) -> None:
        # A stale entry could hold the floor above what the code needs.
        reached = set(_zenoh_attributes_reached())
        stale = sorted(attr for attr in _ZENOH_ATTRIBUTE_FLOORS if attr not in reached)
        assert not stale, f"_ZENOH_ATTRIBUTE_FLOORS records zenoh attributes the mesh no longer uses: {stale}"

    def test_the_receiver_still_reads_the_recorded_sample_attribute(self) -> None:
        # _SAMPLE_SOURCE_INFO_FLOOR is hand-recorded because the attribute is read
        # off a sample object, which no AST sweep can resolve. Pin the read so the
        # constant cannot outlive it.
        from strands_robots.mesh.core import _extract_sample_source_zid

        source = inspect.getsource(_extract_sample_source_zid)
        assert f'"{_SAMPLE_SOURCE_INFO_ATTRIBUTE}"' in source, (
            f"_extract_sample_source_zid no longer reads Sample.{_SAMPLE_SOURCE_INFO_ATTRIBUTE}, "
            "so _SAMPLE_SOURCE_INFO_FLOOR is holding the packaging floor up for a capability "
            "nothing uses"
        )

    def test_every_recorded_release_was_probed(self) -> None:
        probed = set(_PROBED_RELEASES)
        recorded = {*_ZENOH_ATTRIBUTE_FLOORS.values(), _SAMPLE_SOURCE_INFO_FLOOR}
        unprobed = sorted(recorded - probed)
        assert not unprobed, (
            f"these releases are recorded as first shipping a name but are not in "
            f"_PROBED_RELEASES: {unprobed}. Install that wheel and check the name is "
            "there rather than taking a changelog on trust."
        )


class TestTheInstalledZenohProvidesTheRecordedNames:
    """Guard the table against zenoh removing or renaming one of these."""

    @pytest.mark.parametrize("attribute", sorted(_ZENOH_ATTRIBUTE_FLOORS))
    def test_module_attribute_is_present(self, attribute: str) -> None:
        zenoh = pytest.importorskip("zenoh")
        assert hasattr(zenoh, attribute), (
            f"zenoh.{attribute} is recorded in _ZENOH_ATTRIBUTE_FLOORS and used by the "
            "mesh, but the installed eclipse-zenoh does not provide it"
        )

    def test_sample_exposes_the_wire_source_info_attribute(self) -> None:
        zenoh = pytest.importorskip("zenoh")
        assert hasattr(zenoh.Sample, _SAMPLE_SOURCE_INFO_ATTRIBUTE), (
            f"zenoh.Sample has no {_SAMPLE_SOURCE_INFO_ATTRIBUTE} descriptor, so "
            "_extract_sample_source_zid can never recover a publisher zid and every "
            "safety envelope arrives unattributed on this install"
        )
