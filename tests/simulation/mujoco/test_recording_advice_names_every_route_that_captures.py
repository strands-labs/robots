"""``start_recording``'s reply names every route that fills the recording, and no other.

A caller who opens a recording reads exactly one sentence about how frames get
into it, and acts on that sentence for the rest of the session. It therefore has
to name every route that captures - and must not deny one that does, because a
caller told that their route cannot record goes looking for a defect that is not
there, or fills the episode with something else.

Two kinds of route feed the recorder on the MuJoCo backend:

* A policy rollout, through its per-step ``on_frame`` hook. ``policy_running``
  is the flag that marks one in flight, and its raisers are the rollouts:
  ``_announce_rollout`` (shared by ``run_policy`` and ``start_policy``) and
  ``run_multi_policy``, whose synchronized loop calls ``add_frame`` itself.
* ``step``, while a recording is open - one frame per ``1/fps`` seconds of sim
  time - which makes ``set_joint_positions(hold=True)`` + ``step`` a scripted
  demonstration.

``teleoperate`` and ``replay_episode`` take no hook and cannot record however
they are called, so they are the only loops the denial clause may name. The
roster is derived from the code rather than spelled here, so a rollout added
later fails this until the advice is updated.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco import recording  # noqa: E402

# Rollout launchers a caller can dial, plus ``step``: every route the advice
# names has to be reachable through the tool's own action enum, since advice a
# caller cannot dial is worse than no advice. ``run_multi_policy`` is advertised
# by ``describe()`` rather than the enum, so it is graded on naming alone.
DIALABLE_ROUTES = frozenset({"run_policy", "start_policy", "step"})
ROLLOUTS = frozenset({"run_policy", "start_policy", "run_multi_policy"})
# The loops that take no on_frame hook, so nothing a caller does makes them record.
CANNOT_RECORD = ("teleoperate", "replay_episode")

_DENIAL_MARKERS = ("cannot feed", "cannot fill", "do not feed", "does not feed", "never feed", "no such hook")


def _advice() -> str:
    """The success text ``start_recording`` replies with, read from its source.

    Located by the marker it opens with rather than by joining every literal in
    the function: the body also holds refusal texts and dict keys, and joining
    those made a single-word check pass on prose that says nothing about
    capture. Interpolations (``{fps}``) drop out, so assertions here avoid
    spanning one.
    """
    tree = ast.parse(inspect.getsource(recording))
    start_recording = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "start_recording"
    )
    for node in ast.walk(start_recording):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if "Recording to LeRobotDataset" in text:
                return text
    raise AssertionError("start_recording has no success text naming the dataset it opened")


def _names(text: str, name: str) -> bool:
    """Whether ``text`` names ``name`` as a whole token."""
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text) is not None


def _denial_clause(text: str) -> str:
    """The sentence that says which loops cannot feed the recorder.

    Located by what it *says* rather than one phrasing, so any wording that
    denies a loop the ability to record is graded; keyed to a single spelling
    this would pass judgement only on the sentence it was written against.
    """
    for part in re.split(r"(?<=[.])\s+|\n", text):
        if any(marker in part.lower() for marker in _DENIAL_MARKERS):
            return part
    raise AssertionError(f"the advice names no clause denying a loop the ability to record: {text!r}")


class TestTheAdviceNamesEveryRouteThatCaptures:
    def test_every_rollout_that_raises_the_flag_the_recorder_guard_reads_is_named(self) -> None:
        """``policy_running`` defines "a rollout is in flight", so its raisers are
        the rollouts that own the recorder - and the advice has to name each.

        Pinned as the whole set rather than one name per case: the advice named
        ``run_policy`` alone while three sites raise the flag, and a fourth
        raiser added later would quietly make the advertised rule false again.
        This fails when the set changes, which is the moment to decide what the
        advice says.
        """
        from strands_robots.simulation.mujoco import simulation as mujoco_sim

        raisers: set[str] = set()
        stack: list[str] = []

        class Walk(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "policy_running"
                        and getattr(node.value, "value", None) is True
                    ):
                        raisers.add(stack[-1])
                self.generic_visit(node)

        Walk().visit(ast.parse(inspect.getsource(mujoco_sim)))
        # ``_make_run_policy_hook`` is run_policy's own hook, not a separate
        # entry point; ``_announce_rollout`` is shared by run_policy and
        # start_policy.
        assert raisers == {"_announce_rollout", "_make_run_policy_hook", "run_multi_policy"}, raisers

        advice = _advice()
        for rollout in sorted(ROLLOUTS):
            assert _names(advice, rollout), f"{rollout} launches a rollout that feeds the recorder"

    def test_step_is_named_as_a_capture_route_with_its_rate(self) -> None:
        """``step`` fills a recording, so the advice has to offer it.

        This is the half a scripted demonstration depends on: an agent told only
        about rollouts either gives up on scripting the motion or interleaves a
        policy to make frames appear.
        """
        advice = _advice()
        assert _names(advice, "step"), advice
        assert "records one frame per" in advice, advice
        assert "of sim time" in advice, advice
        assert _names(advice, "set_joint_positions"), "the pairing that makes a scripted demonstration"

    def test_the_denial_clause_names_no_route_that_captures(self) -> None:
        """The half that goes wrong: denying a route that does record.

        ``step`` and ``set_joint_positions`` were listed as loops that do not
        feed the recorder. They do now, together, and a caller who believes
        otherwise records nothing on purpose.
        """
        clause = _denial_clause(_advice())
        for loop in CANNOT_RECORD:
            assert _names(clause, loop), f"{loop} genuinely cannot record and is worth naming: {clause!r}"
        for route in ("step", "set_joint_positions", *sorted(ROLLOUTS)):
            assert not _names(clause, route), f"{route} fills a recording but is denied: {clause!r}"

    def test_every_route_the_advice_names_is_a_dialable_action(self) -> None:
        sim = Simulation(tool_name="recording_advice_test", mesh=False)
        try:
            actions = set(sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        finally:
            sim.cleanup()
        assert DIALABLE_ROUTES <= actions, f"not dialable: {sorted(DIALABLE_ROUTES - actions)}"
