"""A documented ``joint_limits`` key must be the key the bridge matches against.

``joint_limits`` is the inbound-command safety bound on the one surface that
drives a physical arm: an out-of-range joint rejects the WHOLE command. The
bound is applied by looking the commanded joint's name up in the mapping
(:meth:`~strands_robots.ros_telemetry.RosTelemetryBase._command_action`), so the
mapping's keys have to be spelled the way the joint names arrive on the wire -
the ``<motor>.pos`` names both hardware bridges publish in ``joint_states``.

A key spelled any other way is not an error. ``dict.get`` returns ``None``, the
joint is treated as having no declared bound, and the command is forwarded to
``send_action`` unchanged: identical behavior to passing no limits at all, with
no exception and no log line. For a safety bound that failure mode is silent in
the worst direction, so the spelling the documentation tells a reader to write
has to be the spelling the lookup uses.

This module grades that. Every documented declaration of the parameter - the
placeholder mappings in the guides and the package docstrings, and the runnable
example's concrete keys - is resolved for a real motor and driven through the
real command parser against wire names taken from the real publish path. A
spelling that does not clamp is reported with the site that documents it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import strands_robots
from strands_robots.hardware_robot import Robot
from strands_robots.ros_telemetry import RosTelemetryBase

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

# A concrete motor to resolve the documented placeholder against, and the motor
# name every generated command carries alongside it so "this joint is bounded"
# and "that joint is not" are both exercised.
_MOTOR = "shoulder_pan"
_OTHER_MOTOR = "wrist_roll"

# The placeholder form every prose declaration uses, e.g. ``{"<motor>.pos":
# (min, max)}``. Matched only where the surrounding text names the parameter, so
# an unrelated ``(min, max)`` pair (MotionBricks' ``speed_scale``) is not graded.
_PLACEHOLDER_MAPPING = re.compile(r"\{\s*([^{}:]{1,48}?)\s*:\s*\(min, max\)\s*\}")

# A runnable example's real mapping, e.g. ``joint_limits={"elbow.pos": (-1, 1)}``.
_CONCRETE_MAPPING = re.compile(r"joint_limits=(\{[^{}]*\})")

# Enough sites that a reformat which stops the sweep reaching the guides fails
# loudly instead of reporting a clean vocabulary.
_MINIMUM_DOCUMENTED_SITES = 8


def _graded_files() -> list[Path]:
    """Docs pages, the README and the package sources - everywhere it is declared."""
    files = sorted(_REPO_ROOT.glob("docs/**/*.md"))
    files.append(_REPO_ROOT / "README.md")
    files.extend(sorted((_REPO_ROOT / "strands_robots").rglob("*.py")))
    return [f for f in files if f.exists()]


def _documented_key_spellings() -> list[tuple[str, str]]:
    """``(site, key spelling)`` for every documented ``joint_limits`` mapping."""
    found: list[tuple[str, str]] = []
    for path in _graded_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for match in _PLACEHOLDER_MAPPING.finditer(text):
            if "joint_limits" not in text[max(0, match.start() - 400) : match.end() + 200]:
                continue
            found.append((f"{rel}:{text.count(chr(10), 0, match.start()) + 1}", match.group(1)))
        for match in _CONCRETE_MAPPING.finditer(text):
            try:
                mapping = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(mapping, dict):
                continue
            site = f"{rel}:{text.count(chr(10), 0, match.start()) + 1}"
            found.extend((site, str(key)) for key in mapping)
    return found


def _resolve(spelling: str, motor: str) -> str:
    """Instantiate a documented spelling for a concrete motor."""
    bare = spelling.strip().strip('"').strip("'")
    return re.sub(r"<motor>|\bmotor\b", motor, bare)


class _RecordingBridge:
    """Bridge stand-in that records what the publish path puts on the wire."""

    def __init__(self) -> None:
        self.joint_states: list[tuple[list[str], list[float]]] = []

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        self.joint_states.append((names, positions))

    def publish_image(self, robot: str, camera: str, image: Any) -> None:  # pragma: no cover - no images here
        raise AssertionError("this observation carries no camera frames")


def _published_wire_names(motors: tuple[str, ...]) -> list[str]:
    """The ``joint_states`` names the real publish path emits for ``motors``.

    Goes through :meth:`Robot._publish_ros_telemetry` rather than restating the
    naming rule, so the vocabulary this module grades against is the one the
    bridges actually advertise.
    """
    bridge = _RecordingBridge()
    hw = SimpleNamespace(
        _ros_bridge=bridge,
        robot=SimpleNamespace(name="so101"),
        tool_name_str="arm",
    )
    observation = {f"{motor}.pos": 0.0 for motor in motors}
    Robot._publish_ros_telemetry(hw, observation)  # type: ignore[arg-type]
    assert bridge.joint_states, "premise: the publish path advertised no joint names"
    names, _ = bridge.joint_states[0]
    return list(names)


def _command_for(motor: str, position: float) -> tuple[SimpleNamespace, list[str]]:
    """A ``joint_command`` driving ``motor`` to ``position``, others at rest."""
    names = _published_wire_names((motor, _OTHER_MOTOR))
    target = next(name for name in names if name.split(".")[0] == motor)
    positions = [position if name == target else 0.0 for name in names]
    return SimpleNamespace(name=list(names), position=positions), names


def _clamps(key: str) -> bool:
    """True when a limit under ``key`` rejects an out-of-range command for its joint."""
    motor = key.split(".")[0]
    msg, _ = _command_for(motor, 99.0)
    return RosTelemetryBase()._command_action(msg, joint_limits={key: (-1.0, 1.0)}) is None


def test_every_documented_key_spelling_clamps_the_joint_it_names() -> None:
    """A reader who copies the documented key spelling gets the bound.

    The headline contract: whatever spelling the docs and docstrings tell a
    caller to key ``joint_limits`` on, that key must reject a command that
    drives its joint out of range. A spelling the lookup never matches leaves
    the arm as unbounded as passing no limits.
    """
    spellings = _documented_key_spellings()
    assert len(spellings) >= _MINIMUM_DOCUMENTED_SITES, (
        f"premise: only {len(spellings)} documented joint_limits mappings were found; "
        "the sweep is no longer reaching the guides, so a clean result proves nothing"
    )
    offenders = [
        f"{site}: documented key {spelling!r} resolves to {_resolve(spelling, _MOTOR)!r}, "
        f"which does not bound {_MOTOR} - the command is forwarded unchanged"
        for site, spelling in spellings
        if not _clamps(_resolve(spelling, _MOTOR))
    ]
    assert not offenders, "documented joint_limits keys that silently constrain nothing:\n" + "\n".join(offenders)


def test_a_documented_key_spelling_is_a_name_the_bridge_publishes() -> None:
    """The documented vocabulary is the published one, not merely a working one.

    ``_command_action`` matches against the inbound message's ``name`` entries,
    and a controller sources those by echoing the bridge's own ``joint_states``.
    Grading the documented spelling against the published names pins the two
    halves of the round trip to one vocabulary.
    """
    published = _published_wire_names((_MOTOR, _OTHER_MOTOR))
    offenders = [
        f"{site}: {spelling!r} -> {_resolve(spelling, _MOTOR)!r}, not among the published names {published}"
        for site, spelling in _documented_key_spellings()
        if _resolve(spelling, _MOTOR).split(".")[0] == _MOTOR and _resolve(spelling, _MOTOR) not in published
    ]
    assert not offenders, "documented joint_limits keys outside the published joint-name vocabulary:\n" + "\n".join(
        offenders
    )


def test_a_limit_keyed_on_a_published_name_rejects_the_whole_command() -> None:
    """Round trip: publish the names, echo them back, and the bound applies."""
    published = _published_wire_names((_MOTOR, _OTHER_MOTOR))
    bounded = next(name for name in published if name.split(".")[0] == _MOTOR)
    msg, _ = _command_for(_MOTOR, 99.0)
    assert RosTelemetryBase()._command_action(msg, joint_limits={bounded: (-1.0, 1.0)}) is None


def test_a_key_that_names_no_commanded_joint_constrains_nothing() -> None:
    """The documented consequence of a mis-keyed bound, pinned as behavior.

    This is why a wrong spelling is silent rather than loud, and it is the
    contract the guides now state: a bound for a joint the command does not
    carry is not an error, it simply does not apply.
    """
    msg, published = _command_for(_MOTOR, 99.0)
    action = RosTelemetryBase()._command_action(msg, joint_limits={"no_such_joint.pos": (-1.0, 1.0)})
    assert action is not None
    assert set(action) == set(published)
    assert action[next(name for name in published if name.split(".")[0] == _MOTOR)] == pytest.approx(99.0)


def test_a_joint_without_a_declared_bound_is_unconstrained() -> None:
    """Bounding one joint must not implicitly bound the others."""
    published = _published_wire_names((_MOTOR, _OTHER_MOTOR))
    bounded = next(name for name in published if name.split(".")[0] == _MOTOR)
    unbounded = next(name for name in published if name.split(".")[0] == _OTHER_MOTOR)
    msg = SimpleNamespace(name=list(published), position=[0.5 if n == bounded else 99.0 for n in published])
    action = RosTelemetryBase()._command_action(msg, joint_limits={bounded: (-1.0, 1.0)})
    assert action is not None
    assert action[unbounded] == pytest.approx(99.0)


def test_the_published_names_carry_the_motor_they_report() -> None:
    """Non-vacuity for the vocabulary helper itself.

    Every assertion above is expressed against ``_published_wire_names``; if it
    ever returned bare motor names the sweep would grade the wrong vocabulary
    and pass for the wrong reason.
    """
    published = _published_wire_names((_MOTOR, _OTHER_MOTOR))
    assert sorted(name.split(".")[0] for name in published) == sorted((_MOTOR, _OTHER_MOTOR))
    assert all(name != name.split(".")[0] for name in published), (
        f"published joint names carry no per-quantity suffix: {published}"
    )
