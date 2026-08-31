"""Driver coverage is derived from both registries, so a gap list cannot go stale.

Two registries answer "what can build this robot for real": a registry entry's
``hardware.lerobot_type`` names a lerobot robot type, and the native-driver
registry holds the classes driver packages register. ``list_native_drivers()``
reports the second, ``get_hardware_type()`` the first, and neither reports the
group defined by *both* absences -- the simulation-only robots, which is the one
group a driver-coverage gap list is made of.

So that list was assembled by hand, and a hand-assembled join goes stale in one
direction only: a robot that gains a driver keeps reading as a gap. Measured on
the shipped registry, a hand-written gap list of sixteen robots named five that
were already reachable -- ``open_duck_mini`` through ``FeetechDriver``, and
``openarm``, ``bi_openarm``, ``rebot_b601`` and ``bi_rebot_b601`` through their
declared lerobot types.

:func:`~strands_robots.drivers.list_driver_coverage` is that join, derived on
every call. The tests below pin the two agreements it rests on, that it reads the
registry rather than a snapshot of today's drivers, and that coverage is not
resolution -- a robot both drivers can build is reported as both even though
:func:`~strands_robots.drivers.resolve_driver` picks one.
"""

from __future__ import annotations

import pytest

import strands_robots.drivers.registry as drivers_registry_mod
from strands_robots.drivers import (
    DEFAULT_DRIVER,
    DRIVER_CHOICES,
    get_native_driver_class,
    list_driver_coverage,
    resolve_driver,
)
from strands_robots.registry import get_hardware_type, list_robots


def test_every_registered_robot_is_reported_exactly_once() -> None:
    """The population is the registry's, so no robot can be silently skipped."""
    coverage = list_driver_coverage()
    assert set(coverage) == {entry["name"] for entry in list_robots("all")}


def test_the_four_groups_are_all_populated() -> None:
    """Non-vacuity, and the reason the join is worth having.

    A function that reported an empty tuple for every robot would satisfy every
    agreement below. The group that matters is the last one: fifty robots reach
    hardware through neither registry, and nothing else reports them.
    """
    coverage = list_driver_coverage()
    assert [name for name, d in coverage.items() if d == (DEFAULT_DRIVER,)]
    assert [name for name, d in coverage.items() if d == ("strands",)]
    assert [name for name, d in coverage.items() if len(d) == 2]
    assert [name for name, d in coverage.items() if not d]


def test_the_strands_entry_agrees_with_the_native_driver_registry() -> None:
    """One half of the join: ``"strands"`` iff a native driver is registered."""
    for name, drivers in list_driver_coverage().items():
        assert ("strands" in drivers) is (get_native_driver_class(name) is not None), name


def test_the_lerobot_entry_agrees_with_the_declared_robot_type() -> None:
    """The other half: ``"lerobot"`` iff the entry declares a lerobot robot type.

    The declaration, not the installation. Whether lerobot is importable is a
    property of the environment, so consulting it would make coverage depend on
    which extras a caller installed.
    """
    for name, drivers in list_driver_coverage().items():
        assert (DEFAULT_DRIVER in drivers) is (get_hardware_type(name) is not None), name


def test_a_driver_registered_later_leaves_the_gap_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived, not a snapshot: this is what a hand-written list cannot do.

    The robot planted against is read out of the gap rather than named, because a
    named one is a hostage to the next driver: every robot in this group is one
    someone is writing a driver for, and pinning the sentinel would make this
    cell fail on the PR that closes its gap for real.
    """

    class _PlantedDriver:
        pass

    sim_only = next(name for name, drivers in sorted(list_driver_coverage().items()) if not drivers)
    monkeypatch.setitem(drivers_registry_mod._NATIVE_DRIVERS, sim_only, _PlantedDriver)
    assert list_driver_coverage()[sim_only] == ("strands",)


def test_coverage_is_not_resolution() -> None:
    """A robot both drivers can build is reported as both; one of them wins.

    ``resolve_driver`` answers which driver a call gets, and never reports
    ``"auto"``. Coverage answers which ``driver=`` values could work, so it must
    keep naming the one resolution did not pick - and every name it reports has
    to be a value a caller may actually pass.
    """
    coverage = list_driver_coverage()
    both = [name for name, drivers in coverage.items() if len(drivers) == 2]
    assert both, "expected at least one robot both drivers can build"
    for name in both:
        assert resolve_driver(name) in coverage[name]
    for name, drivers in coverage.items():
        assert set(drivers) <= set(DRIVER_CHOICES) - {"auto"}, name
