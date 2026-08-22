"""The documented dispatch ladders name the entry points, in the order probed.

``CuroboPolicy._plan_and_cache`` probes several planner entry points per goal
type and takes the first one the planner exposes, because cuRobo renamed them
between the legacy ``MotionGen`` surface and the ``main`` ``MotionPlanner``
surface. Its docstring publishes that ladder as a numbered list, and a reader
sizing a stub planner - or a maintainer adding a fourth entry point - works
from the list rather than from the ``getattr`` chain.

Two properties keep the list usable, and both are graded against the code
rather than against other prose:

* **Same rungs, same order.** The numbered list is the probe order, so a rung
  the code tries is a rung the list numbers.
* **One rung, one entry point.** A rung that also names the entry points below
  it collapses the ladder into itself: the same name then reads as both this
  rung's probe and the next rung's subject, and the count of rungs stops
  matching the count of things tried.

The joint-space ladder failed both: it numbered two rungs for a ladder of
three, and its first rung named all three in a parenthetical while its second
rung named the middle one again. The Cartesian ladder in the same docstring
numbered exactly the two entry points it probes, which is why the pair is
graded together - the passing half shows the rule is the file's own convention.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import textwrap
from typing import Any

import pytest

from strands_robots.policies.curobo import policy as curobo_policy
from strands_robots.policies.curobo.policy import CuroboPolicy

from .test_policy import _StubMotionGen, _StubMotionGenResult

#: Heading of each documented ladder, and the branch of the goal-type ``if``
#: whose probes it describes. ``"body"`` is the Cartesian branch (the ``if``
#: suite), ``"orelse"`` the joint-space one.
_LADDERS: tuple[tuple[str, str], ...] = (
    ("Dispatch order for Cartesian goals:", "body"),
    ("Dispatch order for joint-space goals:", "orelse"),
)

#: ``_LADDERS`` as parametrize cases, named for the goal type they describe.
_LADDER_CASES = [
    pytest.param(heading, branch, id="cartesian" if branch == "body" else "joint-space") for heading, branch in _LADDERS
]

#: A ladder names at least this many rungs. Without a floor, a docstring
#: reflow that hides the numbered lists would satisfy every rule below by
#: grading nothing.
_MINIMUM_RUNGS = 2


def _plan_and_cache_ast() -> ast.FunctionDef:
    """Return the parsed ``_plan_and_cache`` definition."""
    src = textwrap.dedent(inspect.getsource(CuroboPolicy._plan_and_cache))
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _goal_type_branch(which: str) -> list[ast.stmt]:
    """Return the statements of one branch of the goal-type dispatch ``if``."""
    for node in ast.walk(_plan_and_cache_ast()):
        if isinstance(node, ast.If) and "target_pose is not None" in ast.unparse(node.test):
            return node.body if which == "body" else node.orelse
    raise AssertionError("the goal-type dispatch 'if' is no longer recognisable")


def _probe_order(statements: list[ast.stmt]) -> list[str]:
    """Planner entry points *statements* reaches for, in source order.

    Both spellings count: the ``getattr(self._motion_planner, "name", None)``
    probe that asks whether an entry point exists, and the
    ``self._motion_planner.name(...)`` call that reaches for one directly.
    """
    found: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            name: str | None = None
            # ``ast.walk`` yields ``ast.AST``; the position is read off the
            # narrowed expression so the type checker can see it carries one.
            located: ast.expr | None = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                probed = node.args[1] if len(node.args) >= 2 else None
                if (
                    isinstance(probed, ast.Constant)
                    and isinstance(probed.value, str)
                    and "_motion_planner" in ast.unparse(node.args[0])
                ):
                    name = probed.value
                    located = node
            elif isinstance(node, ast.Attribute) and "_motion_planner" in ast.unparse(node.value):
                name = node.attr
                located = node
            if name is not None and located is not None and name not in seen:
                seen.add(name)
                found.append((located.lineno, located.col_offset, name))
    return [name for _line, _col, name in sorted(found)]


def _rung_texts(heading: str) -> list[str]:
    """Return each numbered rung under *heading* as one whitespace-normalised line."""
    doc = inspect.getdoc(CuroboPolicy._plan_and_cache) or ""
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise AssertionError(f"the docstring no longer carries the heading {heading!r}") from None

    rungs: list[list[str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.\s", stripped):
            rungs.append([stripped])
        elif rungs and line[:1].isspace():
            # An indented line continues the rung above it.
            rungs[-1].append(stripped)
        else:
            # A new heading, an ``Args:`` block, or unindented prose: the
            # numbered list is over.
            break
    return [" ".join(parts) for parts in rungs]


def _entry_points_named(rung: str) -> list[str]:
    """Entry-point names a rung's text carries, in order, deduplicated.

    A rung writes its subject as a double-backticked reference, optionally
    qualified (``MotionPlanner.plan_js``) and optionally with the signature
    (``plan_js(JointState, JointState)``); only the callable name matters here.
    """
    names: list[str] = []
    for raw in re.findall(r"``([A-Za-z_][\w.]*)(?:\([^`]*\))?``", rung):
        name = raw.split(".")[-1]
        if name.startswith("plan_") and name not in names:
            names.append(name)
    return names


def _documented_order(heading: str) -> list[str]:
    """The entry point each numbered rung under *heading* names, in rung order."""
    order: list[str] = []
    for rung in _rung_texts(heading):
        named = _entry_points_named(rung)
        if named:
            order.append(named[0])
    return order


class TestTheLadderIsDocumentedInProbeOrder:
    @pytest.mark.parametrize(("heading", "branch"), _LADDER_CASES)
    def test_the_numbered_rungs_are_the_entry_points_the_code_probes(self, heading: str, branch: str) -> None:
        """Every entry point the branch probes is numbered, in probe order."""
        probed = _probe_order(_goal_type_branch(branch))
        documented = _documented_order(heading)
        assert probed, f"premise: no planner entry point found for {heading!r}"
        assert documented == probed, (
            f"{heading!r} numbers {len(documented)} rung(s) {documented} for a ladder of "
            f"{len(probed)}: the code probes {probed}. A reader sizing a stub planner from "
            "the numbered list is told the entry points it does not name are unsupported."
        )

    @pytest.mark.parametrize(("heading", "branch"), _LADDER_CASES)
    def test_a_rung_names_one_entry_point(self, heading: str, branch: str) -> None:
        """No rung describes the probes belonging to the rungs below it."""
        offenders = {rung: named for rung in _rung_texts(heading) if len(named := _entry_points_named(rung)) > 1}
        assert not offenders, (
            f"{heading!r} has a rung naming several entry points, so the same name reads as "
            f"both this rung's probe and another rung's subject: {offenders}"
        )


class TestEveryDocumentedRungPlansAJointSpaceGoal:
    """Drive each rung of the joint-space ladder through the real dispatch.

    Rungs one and two are pinned elsewhere by the new-API and legacy stubs;
    the last resort - a planner exposing neither joint-space entry point -
    had no test, which is how it came to be absent from the numbered list.
    """

    @staticmethod
    def _plan(planner: Any) -> str:
        policy = CuroboPolicy(motion_gen=planner, action_horizon=4)
        asyncio.run(
            policy.get_actions(
                {"observation.state": [0.0, 0.0, 0.0]},
                "",
                target_joints={"j0": 0.5, "j1": -0.3, "j2": 0.2},
            )
        )
        assert planner.plan_calls, "the planner was never reached"
        return str(planner.plan_calls[0][0])

    def test_a_planner_exposing_plan_js_uses_it(self) -> None:
        class _WithPlanJs(_StubMotionGen):
            def plan_js(self, start_state: object, goal: object) -> _StubMotionGenResult:
                self.plan_calls.append(("plan_js", start_state, goal))
                return _StubMotionGenResult(ndof=self.ndof, horizon=self.horizon, success=True, status="ok")

        landed = self._plan(_WithPlanJs(ndof=3, horizon=4))
        assert landed == "plan_js"

    def test_a_planner_exposing_only_plan_single_js_uses_it(self) -> None:
        landed = self._plan(_StubMotionGen(ndof=3, horizon=4))
        assert landed == "plan_single_js"

    def test_a_planner_exposing_neither_joint_space_entry_point_uses_plan_single(self) -> None:
        """The last resort: the Cartesian-shaped legacy entry point still plans."""

        class _OnlyPlanSingle(_StubMotionGen):
            plan_single_js = None  # type: ignore[assignment]

        landed = self._plan(_OnlyPlanSingle(ndof=3, horizon=4))
        assert landed == "plan_single"


class TestTheScanIsNotVacuous:
    def test_every_ladder_names_rungs(self) -> None:
        for heading, _branch in _LADDERS:
            rungs = _rung_texts(heading)
            assert len(rungs) >= _MINIMUM_RUNGS, f"{heading!r} yielded {len(rungs)} rung(s): {rungs}"

    def test_every_ladder_probes_entry_points(self) -> None:
        for _heading, branch in _LADDERS:
            probed = _probe_order(_goal_type_branch(branch))
            assert len(probed) >= _MINIMUM_RUNGS, f"branch {branch!r} probes {probed}"

    def test_the_policy_module_is_the_one_under_test(self) -> None:
        assert curobo_policy.CuroboPolicy is CuroboPolicy
