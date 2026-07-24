"""``list_robots`` mode filters must match the public predicates they document.

``list_robots(mode=...)`` documents each filter by naming the query predicate a
caller would use to reproduce it: ``mode="sim"`` mirrors :func:`has_sim` and
``mode="real"`` mirrors :func:`has_hardware`. If the docstring names a predicate
that does not exist in the registry API (it once referenced a phantom
``has_real`` that was never a function -- the real predicate is
``has_hardware``), a reader who trusts the docstring reaches for an attribute
that raises ``AttributeError``.

These tests pin two things so the documentation cannot drift ahead of the
implementation again:

1. Every ``has_*`` predicate the ``list_robots`` docstring cites resolves to a
   real public callable in ``strands_robots.registry``.
2. The mode filters actually correspond to those predicates row-for-row:
   ``sim`` <-> :func:`has_sim`, ``real`` <-> :func:`has_hardware`, ``both`` is
   their intersection, and ``all`` is every registered robot.
"""

from __future__ import annotations

import re

import pytest

import strands_robots.registry as registry_pkg
from strands_robots.registry.robots import (
    LIST_ROBOTS_MODES,
    has_hardware,
    has_sim,
    list_robots,
)

# Backtick-quoted ``has_<name>`` tokens cited in the ``list_robots`` docstring.
_PREDICATE_TOKEN_RE = re.compile(r"``(has_[a-z_]+)``")


def _cited_predicates() -> set[str]:
    doc = list_robots.__doc__ or ""
    # Only inspect the Args block (the mode bullets); the Returns block lists
    # output-dict KEY names (has_sim, has_real), which are payload, not APIs.
    args_block = doc.split("Returns:", 1)[0]
    return set(_PREDICATE_TOKEN_RE.findall(args_block))


def test_docstring_cites_at_least_the_two_predicates() -> None:
    """Guard the parser: the docstring must cite the sim + hardware predicates."""
    cited = _cited_predicates()
    assert "has_sim" in cited
    assert "has_hardware" in cited


def test_cited_predicates_are_real_registry_callables() -> None:
    """Every ``has_*`` predicate the docstring names must be a real public API.

    Fails on the pre-fix docstring, which cited a phantom ``has_real``.
    """
    for name in sorted(_cited_predicates()):
        obj = getattr(registry_pkg, name, None)
        assert callable(obj), f"list_robots docstring cites ``{name}`` but strands_robots.registry has no such callable"


def test_real_mode_matches_has_hardware() -> None:
    """``mode='real'`` selects exactly the robots for which has_hardware() is True."""
    real_names = {r["name"] for r in list_robots(mode="real")}
    expected = {r["name"] for r in list_robots(mode="all") if has_hardware(r["name"])}
    assert real_names == expected
    assert real_names, "expected at least one hardware-backed robot in the registry"


def test_sim_mode_matches_has_sim() -> None:
    """``mode='sim'`` selects exactly the robots for which has_sim() is True."""
    sim_names = {r["name"] for r in list_robots(mode="sim")}
    expected = {r["name"] for r in list_robots(mode="all") if has_sim(r["name"])}
    assert sim_names == expected
    assert sim_names, "expected at least one sim-capable robot in the registry"


def test_both_mode_is_the_intersection() -> None:
    """``mode='both'`` selects robots that satisfy has_sim() AND has_hardware()."""
    both_names = {r["name"] for r in list_robots(mode="both")}
    expected = {r["name"] for r in list_robots(mode="all") if has_sim(r["name"]) and has_hardware(r["name"])}
    assert both_names == expected


def test_all_mode_is_every_registered_robot() -> None:
    """``mode='all'`` is the unfiltered set (a superset of sim, real, and both)."""
    all_names = {r["name"] for r in list_robots(mode="all")}
    for sub in ("sim", "real", "both"):
        assert {r["name"] for r in list_robots(mode=sub)} <= all_names


@pytest.mark.parametrize("bad_mode", ["hardware", "SIM", "", "any", "simulation"])
def test_unknown_mode_raises_valueerror(bad_mode: str) -> None:
    """An unrecognized mode fails loudly rather than returning an unfiltered list."""
    assert bad_mode not in LIST_ROBOTS_MODES
    with pytest.raises(ValueError, match="Unknown list_robots mode"):
        list_robots(mode=bad_mode)
