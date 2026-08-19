"""Which drive guarantees are fleet-wide, and which belong to one bridge only.

Three mobile-base bridges expose the same ``drive(linear, angular, duration,
count)`` call over three transports, and an operator or agent that learns the
contract from one drives the others with it. Two of the guarantees really are
shared - the numeric domains every value is checked against, and the single-shot
latch - while the velocity clamp, the ``max_duration`` ceiling and the trailing
zero Twist are carried by ``RosbridgeRobot`` alone.

That split matters most for the trailing zero, because it is the guarantee that
a *timed* drive cannot leave a live velocity behind. Presenting it as fleet-wide
tells a reader that ``RtpsRobot("rover", "/cmd_vel").drive(linear=1.0,
duration=5.0)`` self-stops; it publishes fifty Twists at 1.0 m/s and then stops
publishing, leaving the last one latched in the robot's controller.

Every check here therefore *measures* each guarantee on all three bridges and
grades the prose against the measurement, rather than pinning either half to a
hardcoded expectation. A bridge that later gains the trailing stop makes the
scope assertion fail, which is the intended signal: the guarantee has become
fleet-wide and the two places that scope it have to say so.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import strands_robots.mesh.ros_bridge as ros_bridge_mod
import strands_robots.mesh.rosbridge_robot as rosbridge_mod
import strands_robots.mesh.rtps_robot as rtps_mod


class _Recorder:
    """Records the kwargs of each forwarded transport call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


#: (label, module, forwarded transport symbol, robot factory). One publish rate
#: for all three so a duration hold derives the same message count everywhere.
_BRIDGES: list[tuple[str, Any, str, Callable[[], Any]]] = [
    (
        "RosBridgedRobot",
        ros_bridge_mod,
        "use_ros",
        lambda: ros_bridge_mod.RosBridgedRobot("rover", "/cmd_vel", "/odom", publish_rate=10.0),
    ),
    ("RtpsRobot", rtps_mod, "use_rtps", lambda: rtps_mod.RtpsRobot("rover", "/cmd_vel", publish_rate=10.0)),
    (
        "RosbridgeRobot",
        rosbridge_mod,
        "use_rosbridge",
        lambda: rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", publish_rate=10.0),
    ),
]

_ZERO_TWIST = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def _drive(
    monkeypatch: pytest.MonkeyPatch, bridge: tuple[str, Any, str, Callable[[], Any]], **kwargs: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drive one bridge with its transport replaced by a recorder."""
    _, module, symbol, factory = bridge
    recorder = _Recorder()
    monkeypatch.setattr(module, symbol, recorder)
    return factory().drive(**kwargs), recorder.calls


# The measurable form of each documented guarantee. Each probe returns True when
# the bridge exhibits it, so the prose can be graded against three real bridges
# instead of against a list someone kept by hand.


def _validates_before_publishing(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=float("nan"))
    return result["status"] == "error" and calls == []


def _latches_a_single_shot(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=1.0, count=1)
    return result["status"] == "success" and len(calls) == 1 and calls[0]["fields"] != _ZERO_TWIST


def _clamps_velocity(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    _, calls = _drive(monkeypatch, bridge, linear=99.0, count=1)
    return bool(calls) and calls[0]["fields"]["linear"]["x"] < 99.0


def _refuses_a_hold_past_a_ceiling(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=1.0, duration=3600.0)
    return result["status"] == "error" and calls == []


def _appends_a_trailing_zero(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    _, calls = _drive(monkeypatch, bridge, linear=1.0, duration=5.0)
    return bool(calls) and calls[-1]["fields"] == _ZERO_TWIST


#: guarantee -> (probe, docstring phrase, docs bullet label).
_GUARANTEES: dict[str, tuple[Callable[..., bool], str, str]] = {
    "validated inputs": (_validates_before_publishing, "validated", "Finite-input guards"),
    "single-shot latch": (_latches_a_single_shot, "latches", "Single-shot latch"),
    "velocity clamp": (_clamps_velocity, "clamped", "Velocity clamps"),
    "duration ceiling": (_refuses_a_hold_past_a_ceiling, "max_duration", "Loud duration rejection"),
    "trailing zero Twist": (_appends_a_trailing_zero, "zero Twist", "Timed-command trailing zero"),
}

_FLEET_MARKER = "Fleet-standard across all three mobile-base bridges:"
_BRIDGE_MARKER = "Specific to this bridge:"


def _measure(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The bridges that exhibit each guarantee, measured through ``drive``."""
    return {name: [b[0] for b in _BRIDGES if probe(monkeypatch, b)] for name, (probe, _, _) in _GUARANTEES.items()}


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _scoped_paragraph(marker: str) -> str:
    """The ``drive`` docstring paragraph introduced by ``marker``."""
    doc = inspect.getdoc(rosbridge_mod.RosbridgeRobot.drive) or ""
    matches = [p for p in _paragraphs(doc) if p.startswith(marker)]
    assert len(matches) == 1, (
        f"RosbridgeRobot.drive should carry exactly one paragraph opening {marker!r}, found {len(matches)}. "
        "Two of its guarantees hold on all three bridges and three on this one alone, so the docstring has to "
        "label which is which - an unlabelled list reads as fleet-wide."
    )
    return matches[0]


def test_the_three_bridges_are_the_ones_under_comparison() -> None:
    """Premise: a shrunken bridge list would make every scope check vacuous."""
    assert [b[0] for b in _BRIDGES] == ["RosBridgedRobot", "RtpsRobot", "RosbridgeRobot"]


def test_each_guarantee_is_exhibited_by_at_least_one_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise: a probe that measures nothing would let any prose pass."""
    for name, bridges in _measure(monkeypatch).items():
        assert bridges, f"the probe for {name!r} found no bridge that exhibits it - the probe is broken"


def test_the_drive_guarantees_split_into_fleet_wide_and_bridge_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured split the two prose surfaces have to describe.

    Fails if a bridge gains or loses one of these, which is the point: the
    guarantee's scope changed and the prose that scopes it is now stale.
    """
    measured = _measure(monkeypatch)
    every = {b[0] for b in _BRIDGES}

    fleet_wide = sorted(name for name, bridges in measured.items() if set(bridges) == every)
    bridge_only = sorted(name for name, bridges in measured.items() if bridges == ["RosbridgeRobot"])

    assert fleet_wide == ["single-shot latch", "validated inputs"], measured
    assert bridge_only == ["duration ceiling", "trailing zero Twist", "velocity clamp"], measured
    assert sorted(fleet_wide + bridge_only) == sorted(_GUARANTEES), (
        f"every guarantee must be either fleet-wide or this bridge's alone, got {measured}"
    )


def test_a_timed_drive_self_stops_on_one_bridge_and_latches_on_the_other_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence the trailing-zero scope is about, read off the wire."""
    final_velocity: dict[str, float] = {}
    for bridge in _BRIDGES:
        _, calls = _drive(monkeypatch, bridge, linear=1.0, duration=5.0)
        assert [c["count"] for c in calls][0] == 50, "round(5.0 * 10.0) messages"
        final_velocity[bridge[0]] = calls[-1]["fields"]["linear"]["x"]

    assert final_velocity == {"RosBridgedRobot": 1.0, "RtpsRobot": 1.0, "RosbridgeRobot": 0.0}


#: Any wording that asserts this bridge's contract is the fleet's. Matched
#: rather than pinned so the check grades whatever phrasing is in use.
_SHARING_CLAIM = re.compile(r"shared with|[Ff]leet-standard")


def test_no_paragraph_claiming_a_shared_contract_states_a_bridge_specific_guarantee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect this file exists for, graded against whatever prose is there.

    A guarantee carried by one bridge, written into a sentence group that says
    the contract is shared with the other two, is read as theirs as well. The
    trailing zero is the costly one: it is the guarantee that a timed drive
    cannot leave a live velocity behind, and only this bridge has it.
    """
    measured = _measure(monkeypatch)
    bridge_only = {_GUARANTEES[name][1]: name for name, bridges in measured.items() if bridges == ["RosbridgeRobot"]}
    assert bridge_only, f"premise: some guarantee must be this bridge's alone, got {measured}"

    doc = inspect.getdoc(rosbridge_mod.RosbridgeRobot.drive) or ""
    claiming = [p for p in _paragraphs(doc) if _SHARING_CLAIM.search(p)]
    assert claiming, "premise: the docstring should say which guarantees the three bridges share"

    for paragraph in claiming:
        overreach = sorted(f"{name!r} ({phrase!r})" for phrase, name in bridge_only.items() if phrase in paragraph)
        assert not overreach, (
            f"this paragraph of RosbridgeRobot.drive claims a shared contract and then states "
            f"{', '.join(overreach)}, which no other bridge carries: {paragraph}"
        )


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docstring_states_each_guarantee_in_the_scope_it_holds_in(
    monkeypatch: pytest.MonkeyPatch, guarantee: str
) -> None:
    """A guarantee three bridges keep and one they do not cannot share a scope."""
    measured = _measure(monkeypatch)[guarantee]
    is_fleet_wide = set(measured) == {b[0] for b in _BRIDGES}
    phrase = _GUARANTEES[guarantee][1]
    fleet, specific = _scoped_paragraph(_FLEET_MARKER), _scoped_paragraph(_BRIDGE_MARKER)
    holder, other = (fleet, specific) if is_fleet_wide else (specific, fleet)
    scope = "fleet-wide" if is_fleet_wide else f"carried only by {', '.join(measured)}"

    assert phrase in holder, (
        f"{guarantee!r} is {scope}, so {phrase!r} belongs in the "
        f"{'fleet-standard' if is_fleet_wide else 'bridge-specific'} paragraph of RosbridgeRobot.drive"
    )
    assert phrase not in other, f"{guarantee!r} is {scope}, so {phrase!r} must not appear in the other scope"


def test_the_docstring_names_what_the_other_two_bridges_do_instead() -> None:
    """A reader told a guarantee is local still needs to know the alternative."""
    specific = _scoped_paragraph(_BRIDGE_MARKER)
    for sibling in ("RosBridgedRobot.drive", "RtpsRobot.drive"):
        assert sibling in specific, f"the bridge-specific paragraph should name {sibling}"
    assert "latched" in specific, "it should say a timed drive on the other two leaves the velocity latched"


# The docs page carries the same claim, so it is graded the same way ----------


def _drive_contract_bullets() -> dict[str, str]:
    """``label -> bullet text`` for the safety-semantics list on the docs page."""
    page = Path(inspect.getfile(rosbridge_mod)).parents[2] / "docs" / "rosbridge-integration.md"
    body = page.read_text(encoding="utf-8")
    section = re.search(r"\n### Drive contract\n(.*?)(?=\n### )", body, re.DOTALL)
    assert section is not None, (
        "docs/rosbridge-integration.md should carry a '### Drive contract' section. It documented this "
        "bridge's clamp, ceiling and trailing zero under a 'Fleet drive contract' heading, which reads as "
        "though the ROS 2 and RTPS bridges carry them too."
    )
    bullets = re.findall(r"\n- \*\*(.+?)\*\*(.*?)(?=\n- \*\*|\n\n|\Z)", section.group(1), re.DOTALL)
    return {label: " ".join((label + tail).split()) for label, tail in bullets}


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docs_page_marks_each_bridge_specific_guarantee(monkeypatch: pytest.MonkeyPatch, guarantee: str) -> None:
    """The page's safety list is read as the fleet's unless it says otherwise."""
    measured = _measure(monkeypatch)[guarantee]
    is_fleet_wide = set(measured) == {b[0] for b in _BRIDGES}
    label = _GUARANTEES[guarantee][2]
    bullets = _drive_contract_bullets()

    assert label in bullets, f"the drive-contract list should still document {label!r}, got {sorted(bullets)}"
    marked = "this bridge only" in bullets[label]
    if is_fleet_wide:
        assert not marked, f"{guarantee!r} holds on all three bridges, so {label!r} must not be scoped to one"
    else:
        assert marked, (
            f"{guarantee!r} is carried only by {', '.join(measured)}, so the {label!r} bullet has to say so - "
            "unmarked, it reads as a guarantee of every mobile-base bridge"
        )
