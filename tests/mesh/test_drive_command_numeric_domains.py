"""The numeric domains a mobile-base bridge's drive command must share.

Four bridges expose the same fleet-standard ``drive(linear, angular, duration,
count)`` contract over four transports. A velocity command is the one call on
any of them that physically moves the robot, so each knob it carries is checked
before anything reaches the wire - and the accepted domain has to be the *same*
on all of them, because an agent that learns the contract from one bridge drives
the others with it. The wire message differs (a ``Twist`` for the three
differential-drive bases, a normalized servo pair for the Ackermann car); the
domain of every value the caller supplies does not.

Every check here therefore compares the bridges against each other rather than
against a hardcoded expectation, and asserts the *reason* a value was refused
rather than only that it was: a parity assertion on ``status == "error"`` alone
passes when two bridges refuse one value for two unrelated causes.

The structural guard at the end pins the mechanism: a bridge that re-implements
a finiteness test inline instead of calling the shared domain helpers agrees
with nothing, so no behavioral test fails when it drifts.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import strands_robots.drivers.ros.ackermann_robot as ackermann_mod
import strands_robots.drivers.ros.ros_bridge as ros_bridge_mod
import strands_robots.drivers.ros.rosbridge_robot as rosbridge_mod
import strands_robots.drivers.ros.rtps_robot as rtps_mod
from strands_robots.utils import (
    finite_number_error,
    positive_finite_number_error,
    positive_whole_number_error,
)
from tests._package_ast import parse_file, walk_tree
from tests.mesh._transport_stand_in import stands_in_for

#: The wire fields a bridge publishes for the usable probe command below
#: (``linear=0.5``, ``angular=-0.25``). A differential-drive base sends the
#: velocities straight through as a ``Twist``.
_TWIST_FIELDS: dict[str, Any] = {"linear": {"x": 0.5}, "angular": {"z": -0.25}}

#: The Ackermann car sends a normalized servo pair instead: the documented
#: bicycle model ``atan(wheelbase * angular / linear)`` normalized by the
#: steering limit, and the speed normalized by ``max_speed``. Re-derived here
#: from the platform defaults rather than by calling the converter under test.
_SERVO_FIELDS: dict[str, Any] = {
    "angle": pytest.approx(-0.15625849618727),
    "throttle": pytest.approx(0.3333333333333333),
}

#: (label, module, forwarded transport symbol, keyword-taking constructor,
#: published wire fields). The constructor takes the limit keywords so a
#: construction-time check cannot silently target the wrong bridge, and every
#: bridge is built with the same publish rate so a duration hold derives the
#: same message count.
_TRANSPORTS: list[tuple[str, Any, str, Callable[..., Any], dict[str, Any]]] = [
    (
        "ros_bridge",
        ros_bridge_mod,
        "ros_action",
        lambda **kw: ros_bridge_mod.RosBridgedRobot("rover", "/cmd_vel", "/odom", **kw),
        _TWIST_FIELDS,
    ),
    ("rtps", rtps_mod, "rtps_action", lambda **kw: rtps_mod.RtpsRobot("rover", "/cmd_vel", **kw), _TWIST_FIELDS),
    (
        "rosbridge",
        rosbridge_mod,
        "rosbridge_action",
        lambda **kw: rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", **kw),
        _TWIST_FIELDS,
    ),
    (
        "ackermann",
        ackermann_mod,
        "ros_action",
        lambda **kw: ackermann_mod.AckermannRosRobot("car", "/servo", **kw),
        _SERVO_FIELDS,
    ),
]

#: Values no signed velocity component can carry. ``nan``/``inf`` serialize into
#: a Twist as valid float64s, so the transport accepts them and the receiving
#: controller integrates them; the rest cannot be coerced at all.
UNUSABLE_VELOCITIES: list[Any] = ["0.5", None, [0.5], True, False, float("nan"), float("inf")]

#: Values no hold can express. ``None`` is absent, not invalid, so it is excluded.
UNUSABLE_DURATIONS: list[Any] = ["2", [2.0], True, 0, 0.0, -1.5, float("nan"), float("inf")]

#: Values no message count can express - a fractional or non-finite count cannot
#: index a publish loop, and a non-positive one publishes nothing at all.
UNUSABLE_COUNTS: list[Any] = [0, -5, 2.7, float("nan"), float("inf"), "3", True, None, [3]]


def _drive(monkeypatch: pytest.MonkeyPatch, transport: tuple[Any, ...], **kwargs: Any) -> Any:
    """Drive one bridge with the transport recorded, returning (result, recorder)."""
    _label, module, symbol, ctor, _fields = transport
    rec = stands_in_for(monkeypatch, module, symbol)
    return ctor(publish_rate=10.0).drive(**kwargs), rec


def _reason(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


# Cross-bridge parity ---------------------------------------------------------

#: Each command knob, the values it cannot carry, the shared domain that words
#: its refusal, and the rest of a usable command. ``bool`` is an ``int``
#: subclass, so ``True`` in the velocity rows would command 1.0 m/s in silence.
_COMMAND_DOMAINS: dict[str, tuple[list[Any], Callable[..., str | None], dict[str, Any]]] = {
    "linear": (UNUSABLE_VELOCITIES, finite_number_error, {}),
    "angular": (UNUSABLE_VELOCITIES, finite_number_error, {}),
    "duration": (UNUSABLE_DURATIONS, positive_finite_number_error, {"linear": 0.5}),
    "count": (UNUSABLE_COUNTS, positive_whole_number_error, {"linear": 0.5}),
}


@pytest.mark.parametrize("param", list(_COMMAND_DOMAINS))
def test_every_bridge_refuses_an_unusable_command_value_for_the_same_reason(
    monkeypatch: pytest.MonkeyPatch, param: str
) -> None:
    """A value no bridge can honor is refused identically by all of them.

    Asserting the shared helper's exact text (not merely ``status == "error"``)
    is what makes this a parity check: a bridge whose own inline test happens to
    reject the same value for a different reason still fails here. A count
    refusal names ``drive`` rather than the transport: the caller never invoked
    the transport, and its own count check reports a raw loop error.
    """
    values, domain, rest = _COMMAND_DOMAINS[param]
    for value in values:
        expected = domain(value, param, "drive")
        assert expected is not None, f"probe {param}={value!r} must be outside the domain"
        for transport in _TRANSPORTS:
            result, rec = _drive(monkeypatch, transport, **rest, **{param: value})
            assert result["status"] == "error", f"{transport[0]} accepted {param}={value!r}"
            assert _reason(result) == expected, f"{transport[0]} gave a different reason for {param}={value!r}"
            assert rec.calls == [], f"{transport[0]} reached the wire for {param}={value!r}"


_UNUSABLE_LIMITS: list[Any] = [True, "2", None, [2.0], 0, -1.0, float("nan"), float("inf")]


@pytest.mark.parametrize("limit", ["max_linear", "max_angular", "max_duration", "publish_rate"])
def test_an_unusable_velocity_or_horizon_limit_is_refused_at_construction(limit: str) -> None:
    """Each limit bounds every later command, so an unusable one is fatal here.

    ``max_linear=True`` would otherwise install a silent 1.0 m/s clamp on a
    rover configured for 2.0, quietly halving every command that exceeds it.
    ``publish_rate`` is the one limit every bridge shares - it converts a hold
    into a message count - so it is read on every bridge, each built through
    its tuple's own constructor rather than a dispatch on its label.
    """
    ctors = (
        [t[3] for t in _TRANSPORTS]
        if limit == "publish_rate"
        else [next(t[3] for t in _TRANSPORTS if t[0] == "rosbridge")]
    )
    for value in _UNUSABLE_LIMITS:
        assert positive_finite_number_error(value, limit, "") is not None, value
        for ctor in ctors:
            with pytest.raises(ValueError, match=f"{limit} must be a positive finite number"):
                ctor(**{limit: value})


def test_every_bridge_still_publishes_a_usable_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The premise: the domains above refuse only what cannot be honored.

    Without this, deleting the guards entirely would satisfy every check above.
    """
    for transport in _TRANSPORTS:
        result, rec = _drive(monkeypatch, transport, linear=0.5, angular=-0.25, count=3)
        assert result["status"] == "success", f"{transport[0]} refused a usable command"
        assert rec.calls, f"{transport[0]} published nothing"
        assert rec.calls[0]["count"] == 3
        assert rec.calls[0]["fields"] == transport[4], f"{transport[0]} published other than its wire contract"
        held, rec_held = _drive(monkeypatch, transport, linear=0.5, duration=1.5)
        assert held["status"] == "success"
        assert rec_held.calls[0]["count"] == 15, "round(1.5 * 10.0) messages"


# rosbridge-specific consequences --------------------------------------------


def test_a_non_numeric_command_value_does_not_escape_the_bound_agent_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value the caller cannot coerce is reported, not raised past dispatch.

    ``drive`` is bound as an agent tool, so an exception leaving it escapes the
    structured tool-result contract entirely - the agent sees a traceback where
    a result dict was promised, and cannot read which parameter was at fault.
    An omitted (``None``) duration means no hold, which is valid.
    """
    rec = stands_in_for(monkeypatch, rosbridge_mod, "rosbridge_action")
    rover = rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom")
    drive_tool: Any = next(t for t in rover.tools if t.tool_name == "drive_rover")
    for param in ("linear", "angular", "duration"):
        for value in ("0.5", None, [0.5]):
            if param == "duration" and value is None:
                continue
            result = drive_tool(**{param: value})
            assert result["status"] == "error", f"{param}={value!r}"
            assert param in _reason(result)
            assert rec.calls == [], "nothing may reach the wire for a refused command"


def test_a_refused_command_leaves_the_trailing_stop_rule_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed drive still self-stops, and a refused one publishes nothing at all.

    The zero Twist that follows a held command runs from a ``finally``, so a
    guard placed wrongly could either skip it for a valid command or fire it
    for a refused one.
    """
    rec = stands_in_for(monkeypatch, rosbridge_mod, "rosbridge_action")
    rover = rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", publish_rate=10.0)

    assert rover.drive(linear=0.5, duration=1.0)["status"] == "success"
    assert [c["count"] for c in rec.calls] == [10, 1]
    assert rec.calls[-1]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}

    rec.calls.clear()
    assert rover.drive(linear=0.5, count=0)["status"] == "error"
    assert rec.calls == [], "a refused command must not publish even a stop"


# Structural guard: the shared domain is the only finiteness authority --------

_SHARED_DOMAINS = ("finite_number_error", "positive_finite_number_error", "positive_whole_number_error")


def _mesh_modules() -> list[Path]:
    """Every module of the mesh package, located from a symbol it defines."""
    package = Path(inspect.getfile(rosbridge_mod.RosbridgeRobot)).parent
    return sorted(p for p in package.glob("*.py") if p.name != "__init__.py")


def _bridge_classes_defining_drive(tree: ast.Module) -> list[ast.ClassDef]:
    """Top-level classes that define a ``drive`` method of their own."""
    found: list[ast.ClassDef] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(isinstance(c, ast.FunctionDef) and c.name == "drive" for c in ast.iter_child_nodes(node)):
            found.append(node)
    return found


def _called_plain_names(node: ast.AST) -> set[str]:
    return {c.func.id for c in walk_tree(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def _inline_finiteness_calls(node: ast.AST) -> list[str]:
    return [
        f"{c.func.value.id}.{c.func.attr}"
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.attr in {"isfinite", "isnan", "isinf"}
    ]


def _drive_owning_modules() -> list[tuple[Path, ast.Module, ast.ClassDef]]:
    owners: list[tuple[Path, ast.Module, ast.ClassDef]] = []
    for path in _mesh_modules():
        tree = parse_file(path)
        owners.extend((path, tree, cls) for cls in _bridge_classes_defining_drive(tree))
    return owners


#: Bridges that inherit the drive contract instead of owning one. Their absence
#: from the scan is the consolidation; a name reappearing here means a subclass
#: has re-grown a hand-rolled ``drive`` that the guards below cannot vouch for.
_INHERIT_THE_CONTRACT = {"RosBridgedRobot", "RtpsRobot"}


def test_the_drive_owning_bridges_are_the_expected_ones() -> None:
    """Non-vacuity: the scan finds every owner, so the guards below apply.

    ``MobileBaseRobot`` owns the contract for the transports layered on it, so
    a bridge is checked here either as that base or as its own owner. Asserting
    that the layered bridges are *absent* pins the consolidation itself: the
    shared domains reach them by inheritance, and a subclass that reintroduces
    its own ``drive`` silently opts out of every guard below while still
    answering the fleet-standard signature.

    A further transport is picked up automatically - which is the point: it is
    checked the moment it lands rather than after it has drifted.
    """
    names = {cls.name for _path, _tree, cls in _drive_owning_modules()}
    assert {"AckermannRosRobot", "MobileBaseRobot", "RosbridgeRobot"} <= names, names
    redefined = _INHERIT_THE_CONTRACT & names
    assert not redefined, f"{sorted(redefined)} must inherit drive from MobileBaseRobot, not redefine it"


def test_every_drive_owning_bridge_calls_the_shared_numeric_domains() -> None:
    """A bridge that validates its own command values agrees with nothing.

    The three domains live in :mod:`strands_robots.utils` precisely so the
    bridges cannot diverge on what a velocity, a hold or a count may be.
    """
    for path, tree, cls in _drive_owning_modules():
        called = _called_plain_names(tree)
        missing = [name for name in _SHARED_DOMAINS if name not in called]
        assert not missing, f"{path.name}:{cls.name} never calls {missing}"


def test_no_bridge_command_path_tests_finiteness_inline() -> None:
    """The command path must defer to the shared domain, not re-derive it.

    An inline finiteness test accepts ``bool`` as a silent 1 and raises a bare
    coercion error for a numeric string, so it is both looser and less safe
    than the helper it stands in for.
    """
    for path, _tree, cls in _drive_owning_modules():
        for member in ast.iter_child_nodes(cls):
            if not isinstance(member, ast.FunctionDef) or member.name not in {"drive", "__init__"}:
                continue
            inline = _inline_finiteness_calls(member)
            assert not inline, f"{path.name}:{cls.name}.{member.name} tests finiteness inline via {inline}"


def test_the_structural_guard_detects_a_hand_rolled_bridge() -> None:
    """Meta: a scanner that silently matched nothing would look like a clean tree."""
    planted = ast.parse(
        "import math\n"
        "class PlantedRobot:\n"
        "    def drive(self, linear=0.0):\n"
        "        if not math.isfinite(linear):\n"
        "            return {'status': 'error'}\n"
        "        return {'status': 'success'}\n"
    )
    classes = _bridge_classes_defining_drive(planted)
    assert [c.name for c in classes] == ["PlantedRobot"]
    assert [name for name in _SHARED_DOMAINS if name not in _called_plain_names(planted)] == list(_SHARED_DOMAINS)
    assert _inline_finiteness_calls(classes[0]) == ["math.isfinite"]
